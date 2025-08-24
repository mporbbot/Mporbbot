# main.py  (v36 + korrekt ORB/entry + EMA/Volym-filter, inga inline-knappar)
import os
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler

# ========== ENV ==========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")          # 1m,3m,5m,15m,30m,1h
ENTRYMODE = os.getenv("ENTRY_MODE", "close")      # close|tick
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "100"))  # mock-storlek

# Filterkonfig (kan styras med kommandon)
EMA_LEN = int(os.getenv("EMA_LEN", "200"))
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.2"))
TREND_FILTER_ON = True
VOLUME_FILTER_ON = True

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ========== STATE ==========
@dataclass
class Position:
    side: str           # "LONG"
    entry: float
    stop: float
    qty: float          # beräknas från POSITION_SIZE_USDT/entry
    locked: bool = False

@dataclass
class SymState:
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_candle_time: Optional[int] = None
    pos: Optional[Position] = None
    realized_pnl: float = 0.0      # USDT
    trades: List[Tuple[str, float]] = field(default_factory=list)  # ("symbol LONG", pnl)

@dataclass
class EngineState:
    engine_on: bool = False
    entry_mode: str = ENTRYMODE
    timeframe: str = TIMEFRAME
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    orb_on: bool = True
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ========== UI (reply keyboard, inga inline) ==========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/trend"), KeyboardButton("/volym")],
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ========== UTIL ==========
async def get_klines(symbol: str, tf: str, limit: int = 3) -> List[Tuple[int,float,float,float,float]]:
    """Returnerar lista av (startMs, open, high, low, close) – nyast först."""
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # newest first
    out = []
    for row in data[:limit]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

async def get_klines_full(symbol: str, tf: str, limit: int = 300):
    """Som ovan men med volym (nyast först): (t_ms, o,h,l,c,v)"""
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]
    out = []
    for row in data[:limit]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4]); v = float(row[5])
        out.append((t_ms, o, h, l, c, v))
    return out

def ema(values: List[float], length: int) -> Optional[float]:
    if len(values) < length:
        return None
    k = 2 / (length + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def is_green(o,h,l,c): return c > o
def is_red(o,h,l,c):   return c < o

# ========== ORB LOGIK ==========
def detect_orb_from_closed(prev_closed: Tuple[int,float,float,float,float],
                           last_closed: Tuple[int,float,float,float,float]) -> Optional[Tuple[float,float,int]]:
    """
    ORB = första STÄNGDA gröna candle efter en STÄNGD röd candle.
    Byggs ALLTID på två stängda candles: prev_closed (äldre) och last_closed (nyare).
    Returnerar (ORB_high, ORB_low, time_ms) om last_closed kvalar.
    """
    _, po, ph, pl, pc = prev_closed
    _, co, ch, cl, cc = last_closed
    if is_red(po,ph,pl,pc) and is_green(co,ch,cl,cc):
        return (ch, cl, last_closed[0])
    return None

def entry_close_rule(orb_high: float, closed_candle: Tuple[int,float,float,float,float]) -> bool:
    """Stängd candle MÅSTE stänga över ORB-high (då är den också grön)."""
    _, o, h, l, c = closed_candle
    return c > orb_high

def entry_tick_rule(orb_high: float, current_candle: Tuple[int,float,float,float,float]) -> bool:
    """Pågående candle’s wick (high) måste nå över ORB-high."""
    _, o, h, l, c = current_candle
    return h >= orb_high

def stop_hit_close(stop: float, closed_candle: Tuple[int,float,float,float,float]) -> bool:
    _, o,h,l,c = closed_candle
    return c < stop

def stop_hit_tick(stop: float, candle: Tuple[int,float,float,float,float]) -> bool:
    _, o,h,l,c = candle
    return l <= stop

def next_candle_stop(cur_stop: float, prev_closed_candle: Tuple[int,float,float,float,float]) -> float:
    """Flytta SL till föregående stängda candlens low (aldrig nedåt)."""
    _, o,h,l,c = prev_closed_candle
    return max(cur_stop, l)

# ========== ENGINE ==========
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    global TREND_FILTER_ON, VOLUME_FILTER_ON
    while True:
        try:
            if STATE.engine_on and STATE.orb_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]

                    # --- HÄMTA LÅNG SERIE FÖR FILTER ---
                    kl_full = await get_klines_full(sym, STATE.timeframe, limit=max(EMA_LEN+5, VOL_LOOKBACK+5))
                    if len(kl_full) < max(EMA_LEN+5, VOL_LOOKBACK+5):
                        # ändå fortsätt med kort logik om möjligt
                        pass

                    # Gör tidsordning äldst->nyast
                    closed_full = list(reversed(kl_full))  # (t,o,h,l,c,v)
                    closes = [c[4] for c in closed_full]
                    vols   = [c[5] for c in closed_full]

                    # EMA på stängda (exkludera allra sista om den vore pågående – men här är alla stängda från API)
                    ema_val = ema(closes, EMA_LEN) if len(closes) >= EMA_LEN else None

                    # Volymfilter: jämför SENASTE STÄNGDA (closed_full[-1]) mot snitt av föregående N
                    if len(vols) >= VOL_LOOKBACK + 1:
                        last_closed_vol = vols[-1]
                        avg_vol = sum(vols[-(VOL_LOOKBACK+1):-1]) / VOL_LOOKBACK
                    else:
                        last_closed_vol = None
                        avg_vol = None

                    trend_ok = (not TREND_FILTER_ON) or (ema_val is not None and closes[-1] > ema_val)
                    vol_ok   = (not VOLUME_FILTER_ON) or (avg_vol is not None and last_closed_vol is not None and last_closed_vol > VOL_MULT * avg_vol)

                    # --- HÄMTA 3 CANDLES FÖR ORB/ENTRY/SL ---
                    kl = await get_klines(sym, STATE.timeframe, limit=3)
                    if len(kl) < 3:
                        continue

                    current      = kl[0]  # pågående
                    last_closed  = kl[1]  # senaste stängda
                    prev_closed  = kl[2]  # föregående stängda

                    # 1) ORB (alltid på två STÄNGDA candles)
                    orb = detect_orb_from_closed(prev_closed, last_closed)
                    if orb:
                        st.orb_high, st.orb_low, st.orb_candle_time = orb

                    # 2) ENTRY (med filter)
                    if st.pos is None and st.orb_high and st.orb_low and trend_ok and vol_ok:
                        if STATE.entry_mode == "close" and entry_close_rule(st.orb_high, last_closed):
                            entry_px = last_closed[4]
                            qty = POSITION_SIZE_USDT / entry_px if entry_px > 0 else 0.0
                            st.pos = Position("LONG", entry=entry_px, stop=st.orb_low, qty=qty)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY LONG (close) {sym} @ {entry_px:.4f}\n"
                                    f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | "
                                    f"SL={st.pos.stop:.4f} | QTY={qty:.6f}\n"
                                    f"[Filter] Trend:{'OK' if trend_ok else 'NEJ'} EMA{EMA_LEN}={ema_val:.4f if ema_val else float('nan')}, "
                                    f"Vol:{'OK' if vol_ok else 'NEJ'} (last={last_closed_vol:.0f if last_closed_vol else 0}, avg={avg_vol:.0f if avg_vol else 0})"
                                )
                        elif STATE.entry_mode == "tick" and entry_tick_rule(st.orb_high, current):
                            entry_px = st.orb_high
                            qty = POSITION_SIZE_USDT / entry_px if entry_px > 0 else 0.0
                            st.pos = Position("LONG", entry=entry_px, stop=st.orb_low, qty=qty)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY LONG (tick) {sym} @ {entry_px:.4f}\n"
                                    f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | "
                                    f"SL={st.pos.stop:.4f} | QTY={qty:.6f}\n"
                                    f"[Filter] Trend:{'OK' if trend_ok else 'NEJ'} EMA{EMA_LEN}={ema_val:.4f if ema_val else float('nan')}, "
                                    f"Vol:{'OK' if vol_ok else 'NEJ'} (last={last_closed_vol:.0f if last_closed_vol else 0}, avg={avg_vol:.0f if avg_vol else 0})"
                                )

                    # 3) TRAILING SL + EXIT
                    if st.pos:
                        # flytta SL till förra stängda candlens low
                        new_sl = next_candle_stop(st.pos.stop, last_closed)
                        if new_sl > st.pos.stop:
                            st.pos.stop = new_sl
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔧 SL flyttad {sym} -> {st.pos.stop:.4f}")

                        # exit beslut
                        exited = False
                        if STATE.entry_mode == "close" and stop_hit_close(st.pos.stop, last_closed):
                            exit_px = last_closed[4]
                            exited = True
                        elif STATE.entry_mode == "tick" and stop_hit_tick(st.pos.stop, current):
                            exit_px = st.pos.stop
                            exited = True

                        if exited:
                            pnl = (exit_px - st.pos.entry) * st.pos.qty
                            st.realized_pnl += pnl
                            st.trades.append((f"{sym} LONG", pnl))
                            if len(st.trades) > 50:
                                st.trades = st.trades[-50:]
                            if STATE.chat_id:
                                sign = "✅" if pnl >= 0 else "❌"
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔴 EXIT {sym} @ {exit_px:.4f} | PnL: {pnl:+.4f} USDT {sign}"
                                )
                            st.pos = None

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ========== TELEGRAM ==========
tg_app = Application.builder().token(BOT_TOKEN).build()

def pnl_summary() -> str:
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    lines = [f"📈 PnL total: {total_pnl:+.4f} USDT"]
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        lines.append(f"• {s}: {ss.realized_pnl:+.4f} USDT")
    # senaste trades
    last = []
    for s in STATE.symbols:
        for lbl, p in STATE.per_sym[s].trades[-5:]:
            last.append((lbl, p))
    if last:
        lines.append("\nSenaste affärer:")
        for lbl, p in last[-10:]:
            lines.append(f"  - {lbl}: {p:+.4f} USDT")
    return "\n".join(lines)

async def send_status(chat_id: int):
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(f"{s}: LONG @ {st.pos.entry:.4f} | SL {st.pos.stop:.4f} | QTY {st.pos.qty:.6f}")
    sym_pnls = ", ".join([f"{s}:{STATE.per_sym[s].realized_pnl:+.2f}" for s in STATE.symbols])

    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Entry mode: {STATE.entry_mode}",
        f"Timeframe: {STATE.timeframe}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"ORB: {'PÅ' if STATE.orb_on else 'AV'} (första STÄNGDA gröna efter STÄNGD röd)",
        f"Filter – Trend(EMA{EMA_LEN}): {'PÅ' if TREND_FILTER_ON else 'AV'}, Volym: {'PÅ' if VOLUME_FILTER_ON else 'AV'} (>{VOL_MULT}× snitt {VOL_LOOKBACK})",
        f"PnL total: {total_pnl:+.2f} USDT",
        f"PnL per symbol: {sym_pnls if sym_pnls else '-'}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    await tg_app.bot.send_message(chat_id, pnl_summary(), reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! ORB-bot v36+filter redo ✅", reply_markup=reply_kb())
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

async def cmd_orb_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.orb_on = True
    await tg_app.bot.send_message(STATE.chat_id, "ORB: PÅ ✅", reply_markup=reply_kb())

async def cmd_orb_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.orb_on = False
    await tg_app.bot.send_message(STATE.chat_id, "ORB: AV ⛔️", reply_markup=reply_kb())

async def cmd_entrymode(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.entry_mode = "tick" if STATE.entry_mode == "close" else "close"
    await tg_app.bot.send_message(STATE.chat_id, f"Entry mode satt till: {STATE.entry_mode}", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    order = ["1m","3m","5m","15m","30m","1h"]
    i = order.index(STATE.timeframe) if STATE.timeframe in order else 0
    STATE.timeframe = order[(i+1) % len(order)]
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframe satt till: {STATE.timeframe}", reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            exit_px = st.pos.stop
            pnl = (exit_px - st.pos.entry) * st.pos.qty
            st.realized_pnl += pnl
            st.trades.append((f"{s} LONG (panic)", pnl))
            closed.append(f"{s} pnl {pnl:+.4f}")
            st.pos = None
    msg = " | ".join(closed) if closed else "Inga positioner öppna."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl = 0.0
        STATE.per_sym[s].trades.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

async def cmd_trend(update: Update, _):
    global TREND_FILTER_ON
    STATE.chat_id = update.effective_chat.id
    TREND_FILTER_ON = not TREND_FILTER_ON
    await tg_app.bot.send_message(STATE.chat_id, f"Trendfilter (EMA{EMA_LEN}): {'PÅ' if TREND_FILTER_ON else 'AV'}", reply_markup=reply_kb())

async def cmd_volym(update: Update, _):
    global VOLUME_FILTER_ON
    STATE.chat_id = update.effective_chat.id
    VOLUME_FILTER_ON = not VOLUME_FILTER_ON
    await tg_app.bot.send_message(STATE.chat_id, f"Volymfilter: {'PÅ' if VOLUME_FILTER_ON else 'AV'} (>{VOL_MULT}× snitt {VOL_LOOKBACK})", reply_markup=reply_kb())

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("trend", cmd_trend))
tg_app.add_handler(CommandHandler("volym", cmd_volym))

# ========== FASTAPI WEBHOOK ==========
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
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tf": STATE.timeframe,
        "mode": STATE.entry_mode,
        "pnl_total": round(total_pnl, 6),
        "filters": {"trend": TREND_FILTER_ON, "volym": VOLUME_FILTER_ON},
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
