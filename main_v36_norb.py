# main_v36_norb.py
import os
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler

# ================== ENV ==================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

SYMBOLS = (
    os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
    .replace(" ", "")
    .split(",")
)

TIMEFRAME = os.getenv("TIMEFRAME", "1m")       # 1m,3m,5m,15m,30m,1h
ENTRYMODE = os.getenv("ENTRY_MODE", "close")   # close|tick
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "100"))

# Mock-avgifter (per sida). 0.001 = 0.1%
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))

# Trend/RSI-parametrar
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
RSI_LEN  = int(os.getenv("RSI_LEN",  "14"))
RSI_LONG_TH  = float(os.getenv("RSI_LONG_TH",  "55"))
RSI_SHORT_TH = float(os.getenv("RSI_SHORT_TH", "45"))

# Pollning: milda rate limits
ENGINE_SLEEP_SEC = int(os.getenv("ENGINE_SLEEP_SEC", "3"))
MIN_FETCH_GAP_SEC = int(os.getenv("MIN_FETCH_GAP_SEC", "10"))

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ================== STATE ==================
@dataclass
class Position:
    side: str           # "LONG" eller "SHORT"
    entry: float
    stop: float
    qty: float
    fees_paid: float = 0.0

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_gross: float = 0.0
    realized_fees: float = 0.0
    trades: List[Tuple[str, float, float]] = field(default_factory=list)  # (label, gross, fees)
    last_fetch_ts: float = 0.0

@dataclass
class EngineState:
    engine_on: bool = False
    entry_mode: str = ENTRYMODE
    timeframe: str = TIMEFRAME
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    allow_shorts: bool = True
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ================== UI (endast nedre knappar) ==================
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/th_up"), KeyboardButton("/th_down")],   # justerar RSI-trösklar
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ================== DATA ==================
async def get_klines(symbol: str, tf: str, limit: int = 200) -> pd.DataFrame:
    """
    Hämtar candels (nyast först från KuCoin), returnerar DataFrame i stigande tid.
    Kolumner: time_ms, open, high, low, close
    """
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        raw = r.json()["data"]  # newest first
    if not raw:
        return pd.DataFrame(columns=["time_ms", "open", "high", "low", "close"])
    rows = []
    for row in raw[:limit]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        rows.append((t_ms, o, h, l, c))
    df = pd.DataFrame(rows, columns=["time_ms", "open", "high", "low", "close"])
    df = df.iloc[::-1].reset_index(drop=True)  # äldst -> nyast
    return df

def ta_ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def ta_rsi(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

# ================== STRATEGI (ingen ORB) ==================
def detect_signals(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Returnerar dict med signalflaggor på SISTA stängda candle:
      long_close, short_close, long_tick, short_tick
    """
    if len(df) < max(EMA_SLOW, RSI_LEN) + 2:
        return {"long_close": False, "short_close": False, "long_tick": False, "short_tick": False}

    ema_fast = ta_ema(df["close"], EMA_FAST)
    ema_slow = ta_ema(df["close"], EMA_SLOW)
    rsi = ta_rsi(df["close"], RSI_LEN)

    # Stängd candle = näst sista
    i_prev = len(df) - 2
    i_curr = len(df) - 1

    prev_close = df.loc[i_prev, "close"]
    curr_close = df.loc[i_curr, "close"]
    prev_high  = df.loc[i_prev, "high"]
    prev_low   = df.loc[i_prev, "low"]

    prev_ef, prev_es = ema_fast.loc[i_prev], ema_slow.loc[i_prev]
    curr_ef, curr_es = ema_fast.loc[i_curr], ema_slow.loc[i_curr]
    prev_rsi = rsi.loc[i_prev]

    # CROSS på stängd candle
    long_close  = (prev_close <= prev_es and curr_close > curr_es and prev_rsi >= RSI_LONG_TH)
    short_close = (prev_close >= prev_es and curr_close < curr_es and prev_rsi <= RSI_SHORT_TH)

    # TICK-variant (om hög/låg penetrerar EMA-slow på senaste stängda candlen)
    long_tick  = (prev_low <= prev_es and prev_high >= prev_es and prev_close > prev_es and prev_rsi >= RSI_LONG_TH)
    short_tick = (prev_high >= prev_es and prev_low <= prev_es and prev_close < prev_es and prev_rsi <= RSI_SHORT_TH)

    return {
        "long_close": long_close,
        "short_close": short_close,
        "long_tick": long_tick,
        "short_tick": short_tick,
    }

def size_from_price(price: float) -> float:
    return POSITION_SIZE_USDT / price if price > 0 else 0.0

def trail_stop_long(cur_stop: float, prev_closed_low: float) -> float:
    return max(cur_stop, prev_closed_low)

def trail_stop_short(cur_stop: float, prev_closed_high: float) -> float:
    return min(cur_stop, prev_closed_high)

# ================== ENGINE ==================
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                now = asyncio.get_event_loop().time()
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    # rate-limiter per symbol
                    if now - st.last_fetch_ts < MIN_FETCH_GAP_SEC:
                        continue
                    st.last_fetch_ts = now

                    try:
                        df = await get_klines(sym, STATE.timeframe, limit=200)

                        # Behöver minst 2 candles för SL-trailing
                        if len(df) < 3:
                            continue

                        sig = detect_signals(df)
                        last_closed = df.iloc[-2]  # senaste stängda
                        prev_closed = df.iloc[-3]

                        # ENTRY
                        if st.pos is None:
                            if STATE.entry_mode == "close":
                                if sig["long_close"]:
                                    entry = float(last_closed["close"])
                                    qty = size_from_price(entry)
                                    fees = entry * qty * FEE_RATE
                                    st.pos = Position("LONG", entry=entry, stop=float(prev_closed["low"]), qty=qty, fees_paid=fees)
                                    if STATE.chat_id:
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"🟢 ENTRY LONG (close) {sym} @ {entry:.6f}\n"
                                            f"SL={st.pos.stop:.6f} | QTY={qty:.6f} | Fees: {fees:.6f}",
                                            reply_markup=reply_kb()
                                        )
                                elif STATE.allow_shorts and sig["short_close"]:
                                    entry = float(last_closed["close"])
                                    qty = size_from_price(entry)
                                    fees = entry * qty * FEE_RATE
                                    st.pos = Position("SHORT", entry=entry, stop=float(prev_closed["high"]), qty=qty, fees_paid=fees)
                                    if STATE.chat_id:
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"🔻 ENTRY SHORT (close) {sym} @ {entry:.6f}\n"
                                            f"SL={st.pos.stop:.6f} | QTY={qty:.6f} | Fees: {fees:.6f}",
                                            reply_markup=reply_kb()
                                        )
                            else:  # tick
                                if sig["long_tick"]:
                                    entry = float(last_closed["high"])  # approx intrabar
                                    qty = size_from_price(entry)
                                    fees = entry * qty * FEE_RATE
                                    st.pos = Position("LONG", entry=entry, stop=float(prev_closed["low"]), qty=qty, fees_paid=fees)
                                    if STATE.chat_id:
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"🟢 ENTRY LONG (tick) {sym} @ {entry:.6f}\n"
                                            f"SL={st.pos.stop:.6f} | QTY={qty:.6f} | Fees: {fees:.6f}",
                                            reply_markup=reply_kb()
                                        )
                                elif STATE.allow_shorts and sig["short_tick"]:
                                    entry = float(last_closed["low"])
                                    qty = size_from_price(entry)
                                    fees = entry * qty * FEE_RATE
                                    st.pos = Position("SHORT", entry=entry, stop=float(prev_closed["high"]), qty=qty, fees_paid=fees)
                                    if STATE.chat_id:
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"🔻 ENTRY SHORT (tick) {sym} @ {entry:.6f}\n"
                                            f"SL={st.pos.stop:.6f} | QTY={qty:.6f} | Fees: {fees:.6f}",
                                            reply_markup=reply_kb()
                                        )

                        # TRAIL/EXIT
                        if st.pos:
                            pos = st.pos
                            # flytta SL mot förra stängda
                            if pos.side == "LONG":
                                new_sl = trail_stop_long(pos.stop, float(prev_closed["low"]))
                                if new_sl > pos.stop:
                                    pos.stop = new_sl
                                    if STATE.chat_id:
                                        await app.bot.send_message(STATE.chat_id, f"🔧 SL flyttad {sym} -> {pos.stop:.6f}")
                                # stop-hit?
                                if float(last_closed["close"]) < pos.stop:
                                    exit_px = float(last_closed["close"]) if STATE.entry_mode == "close" else pos.stop
                                    gross = (exit_px - pos.entry) * pos.qty
                                    fees_exit = exit_px * pos.qty * FEE_RATE
                                    st.realized_gross += gross
                                    st.realized_fees  += (pos.fees_paid + fees_exit)
                                    st.trades.append((f"{sym} LONG", gross, pos.fees_paid + fees_exit))
                                    if STATE.chat_id:
                                        net = gross - (pos.fees_paid + fees_exit)
                                        emoji = "✅" if net >= 0 else "❌"
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"🔴 EXIT {sym} @ {exit_px:.6f}\n"
                                            f"Gross: {gross:+.4f}  Fees: {(pos.fees_paid + fees_exit):.4f}  Net: {net:+.4f} USDT {emoji}",
                                            reply_markup=reply_kb()
                                        )
                                    st.pos = None

                            else:  # SHORT
                                new_sl = trail_stop_short(pos.stop, float(prev_closed["high"]))
                                if new_sl < pos.stop:
                                    pos.stop = new_sl
                                    if STATE.chat_id:
                                        await app.bot.send_message(STATE.chat_id, f"🔧 SL flyttad {sym} -> {pos.stop:.6f}")
                                if float(last_closed["close"]) > pos.stop:
                                    exit_px = float(last_closed["close"]) if STATE.entry_mode == "close" else pos.stop
                                    gross = (pos.entry - exit_px) * pos.qty
                                    fees_exit = exit_px * pos.qty * FEE_RATE
                                    st.realized_gross += gross
                                    st.realized_fees  += (pos.fees_paid + fees_exit)
                                    st.trades.append((f"{sym} SHORT", gross, pos.fees_paid + fees_exit))
                                    if STATE.chat_id:
                                        net = gross - (pos.fees_paid + fees_exit)
                                        emoji = "✅" if net >= 0 else "❌"
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"🟢 COVER {sym} @ {exit_px:.6f}\n"
                                            f"Gross: {gross:+.4f}  Fees: {(pos.fees_paid + fees_exit):.4f}  Net: {net:+.4f} USDT {emoji}",
                                            reply_markup=reply_kb()
                                        )
                                    st.pos = None

                    except httpx.HTTPStatusError as he:
                        # Skicka kort fel + bilagd json med senaste candles om det var 429/5xx
                        if STATE.chat_id:
                            try:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"⚠️ Engine-fel: {he}"
                                )
                            except:
                                pass
                    except Exception as e:
                        if STATE.chat_id:
                            try:
                                await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                            except:
                                pass
            await asyncio.sleep(ENGINE_SLEEP_SEC)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ================== TELEGRAM ==================
tg_app = Application.builder().token(BOT_TOKEN).build()

def total_net_pnl() -> float:
    gross = sum(s.realized_gross for s in STATE.per_sym.values())
    fees  = sum(s.realized_fees  for s in STATE.per_sym.values())
    return gross - fees

async def send_status(chat_id: int):
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            p = st.pos
            pos_lines.append(f"{s}: {p.side} @ {p.entry:.6f} | SL {p.stop:.6f} | QTY {p.qty:.6f}")
    sym_pnls = ", ".join([
        f"{s}: Net {STATE.per_sym[s].realized_gross - STATE.per_sym[s].realized_fees:+.4f}"
        for s in STATE.symbols
    ])
    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Entry mode: {STATE.entry_mode}    # tick eller close",
        f"Timeframe: {STATE.timeframe}      # 1m eller 3m",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Avgiftssats (mock): {FEE_RATE*100:.4f}%",
        f"PnL total: {total_net_pnl():+.4f} USDT",
        f"PnL per symbol: {sym_pnls if sym_pnls else '-'}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    lines = [f"📈 Total Net PnL: {total_net_pnl():+.4f} USDT"]
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        net = st.realized_gross - st.realized_fees
        lines.append(f"• {s}: Net {net:+.4f} (Gross {st.realized_gross:+.4f}, Fees {st.realized_fees:.4f})")
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! AI-trendbot (utan ORB) redo ✅", reply_markup=reply_kb())
    await send_status(STATE.chat_id)

async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_status(STATE.chat_id)

async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(STATE.chat_id, "Engine: ON ✅", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

async def cmd_entrymode(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.entry_mode = "tick" if STATE.entry_mode == "close" else "close"
    await tg_app.bot.send_message(STATE.chat_id, f"Entry mode satt till: {STATE.entry_mode}", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    order = ["1m", "3m", "5m", "15m", "30m", "1h"]
    i = order.index(STATE.timeframe) if STATE.timeframe in order else 0
    STATE.timeframe = order[(i + 1) % len(order)]
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframe satt till: {STATE.timeframe}", reply_markup=reply_kb())

async def cmd_th_up(update: Update, _):
    """Höj känsligheten: sänk long-tröskel, höj short-tröskel."""
    global RSI_LONG_TH, RSI_SHORT_TH
    STATE.chat_id = update.effective_chat.id
    RSI_LONG_TH = max(50.0, RSI_LONG_TH - 1.0)
    RSI_SHORT_TH = min(50.0, RSI_SHORT_TH + 1.0)
    await tg_app.bot.send_message(STATE.chat_id,
        f"Nya RSI-trösklar → long ≥ {RSI_LONG_TH:.1f}, short ≤ {RSI_SHORT_TH:.1f}",
        reply_markup=reply_kb())

async def cmd_th_down(update: Update, _):
    """Minska känsligheten: höj long-tröskel, sänk short-tröskel."""
    global RSI_LONG_TH, RSI_SHORT_TH
    STATE.chat_id = update.effective_chat.id
    RSI_LONG_TH += 1.0
    RSI_SHORT_TH -= 1.0
    await tg_app.bot.send_message(STATE.chat_id,
        f"Nya RSI-trösklar → long ≥ {RSI_LONG_TH:.1f}, short ≤ {RSI_SHORT_TH:.1f}",
        reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            p = st.pos
            exit_px = p.stop
            if p.side == "LONG":
                gross = (exit_px - p.entry) * p.qty
            else:
                gross = (p.entry - exit_px) * p.qty
            fees_exit = exit_px * p.qty * FEE_RATE
            st.realized_gross += gross
            st.realized_fees  += (p.fees_paid + fees_exit)
            st.trades.append((f"{s} {p.side} (panic)", gross, p.fees_paid + fees_exit))
            closed.append(f"{s} Net {(gross - (p.fees_paid + fees_exit)):+.4f}")
            st.pos = None
    msg = " | ".join(closed) if closed else "Inga positioner öppna."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_gross = 0.0
        STATE.per_sym[s].realized_fees = 0.0
        STATE.per_sym[s].trades.clear()
        STATE.per_sym[s].pos = None
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

# Registrera handlers (OBS: inga ORB-kommandon)
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("th_up", cmd_th_up))
tg_app.add_handler(CommandHandler("th_down", cmd_th_down))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))

# ================== FASTAPI WEBHOOK ==================
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tf": STATE.timeframe,
        "mode": STATE.entry_mode,
        "fee_rate": FEE_RATE,
        "pnl_total_net": round(total_net_pnl(), 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
