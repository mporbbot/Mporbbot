# main_v40_pro.py
# ------------------------------------------------------
# MP Grid AI v40 Pro – robust grid/DCA med MTF-konsensus,
# avgifter (mock), riskkontroller, rate-limit/caching och live-tuning.
# Byggd ovanpå v36-stommen (FastAPI + Telegram reply-knappar).
# ------------------------------------------------------
import os
import io
import csv
import math
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import httpx
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler

# ========= ENV =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")

DEFAULT_TFS = ["3m", "5m", "15m"]  # 2 av 3 röster krävs

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))  # 0.10% per sida mock

# Grid/DCA (går att justera live)
GRID_MAX_SAFETY      = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_ATR_MULT   = float(os.getenv("GRID_STEP_ATR_MULT", "0.6"))
GRID_STEP_MIN_PCT    = float(os.getenv("GRID_STEP_MIN_PCT", "0.3"))
GRID_SIZE_MULT       = float(os.getenv("GRID_SIZE_MULT", "1.5"))
GRID_TP_PCT          = float(os.getenv("GRID_TP_PCT", "0.25"))   # min-TP (utan avgifter)
TP_EXTRA_PER_SAFETY  = float(os.getenv("TP_EXTRA_PER_SAFETY", "0.05"))  # +0.05% per safety

# Risk
MAX_DRAWDOWN_PCT     = float(os.getenv("MAX_DRAWDOWN_PCT", "3.0"))
MIN_ATR_PCT          = float(os.getenv("MIN_ATR_PCT", "0.08"))    # handla inte om för låg vol
MAX_ATR_PCT          = float(os.getenv("MAX_ATR_PCT", "3.0"))     # handla inte om för hög vol
MAX_CONCURRENT_POS   = int(os.getenv("MAX_CONCURRENT_POS", "3"))  # max samtidiga symboler

# Engine
ENGINE_SLEEP_SEC = int(os.getenv("ENGINE_SLEEP_SEC", "5"))
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1hour"}

# Rate-limit/caching
MIN_FETCH_INTERVAL_SEC = 30  # minsta hämtintervall per (symbol, tf)

# ========= STATE =========
@dataclass
class TradeRec:
    time: datetime
    symbol: str
    side: str       # LONG/SHORT
    action: str     # ENTRY/SAFETY/EXIT:TP/EXIT:DD/EXIT:PANIC
    price: float
    qty: float
    gross: float
    fee_in: float
    fee_out: float
    net: float

@dataclass
class Position:
    side: str
    entries: List[Tuple[float,float]] = field(default_factory=list) # (price, qty)
    fee_in_acc: float = 0.0
    first_entry_price: float = 0.0
    last_entry_price: float = 0.0
    qty_total: float = 0.0
    avg_price: float = 0.0
    safety_count: int = 0
    next_step_price: Optional[float] = None
    tp_price: Optional[float] = None

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_gross: float = 0.0
    realized_fees: float = 0.0
    realized_net: float = 0.0
    cooldown_until: Optional[datetime] = None
    last_signal: Optional[str] = None  # LONG/SHORT/NONE
    last_fetch: Dict[str, datetime] = field(default_factory=dict) # tf -> ts

@dataclass
class EngineState:
    engine_on: bool = False
    tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    allow_shorts: bool = True
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

TRADE_LOG: List[TradeRec] = []

# Enkel DF-cache = {(symbol, tf): (ts, df)}
_CANDLE_CACHE: Dict[Tuple[str,str], Tuple[datetime, pd.DataFrame]] = {}

# ========= UI =========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/export_csv")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ========= DATA =========
async def fetch_klines(symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
    """Hämtar candles, återanvänder cache om < MIN_FETCH_INTERVAL_SEC."""
    key = (symbol, tf)
    now = datetime.now(timezone.utc)
    cached = _CANDLE_CACHE.get(key)
    if cached:
        ts, df = cached
        if (now - ts).total_seconds() < MIN_FETCH_INTERVAL_SEC:
            return df

    params = {"symbol": symbol, "type": TF_MAP.get(tf, "3min")}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # NYAST först

    data = list(reversed(data))[:limit]
    rows = []
    for row in data:
        # [time, open, close, high, low, volume, turnover]
        t = pd.to_datetime(int(row[0]), unit="s", utc=True)
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        v = float(row[5])
        rows.append((t,o,h,l,c,v))
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    _CANDLE_CACHE[key] = (now, df)
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"]-df["low"],
        (df["high"]-prev_close).abs(),
        (df["low"]-prev_close).abs()
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def atr_pct(last_atr: float, price: float) -> float:
    return 0.0 if price <= 0 else (last_atr / price) * 100.0

async def mtf_signal(symbol: str, tfs: List[str]) -> Tuple[str, float, float, float, float]:
    """
    (signal, last_price, last_atr, atr_pct_val, vol_score)
    Bull = close>EMA200 & EMA50>EMA200 & slope(EMA50)>0
    Bear = close<EMA200 & EMA50<EMA200 & slope(EMA50)<0
    Kräver 2 av 3 röster.
    vol_score = volMA(20)/volMA(50) på minsta TF.
    """
    votes_long = 0
    votes_short = 0
    smallest_tf = tfs[0]
    last_price = 0.0
    last_atr = 0.0
    vol_score = 1.0

    for tf in tfs:
        df = await fetch_klines(symbol, tf, limit=300)
        if len(df) < 210:
            continue
        ema50 = ema(df["close"], 50)
        ema200 = ema(df["close"], 200)
        slope50 = ema50.iloc[-1] - ema50.iloc[-2]
        c = float(df["close"].iloc[-1])

        bull = (c > float(ema200.iloc[-1])) and (float(ema50.iloc[-1]) > float(ema200.iloc[-1])) and (slope50 > 0)
        bear = (c < float(ema200.iloc[-1])) and (float(ema50.iloc[-1]) < float(ema200.iloc[-1])) and (slope50 < 0)

        if bull: votes_long += 1
        elif bear: votes_short += 1

        if tf == smallest_tf:
            last_price = c
            last_atr = atr(df)
            v20 = float(df["volume"].rolling(20).mean().iloc[-1])
            v50 = float(df["volume"].rolling(50).mean().iloc[-1]) or 1.0
            vol_score = v20 / v50

    if votes_long >= 2:
        return "LONG", last_price, last_atr, atr_pct(last_atr, last_price), vol_score
    if votes_short >= 2:
        return "SHORT", last_price, last_atr, atr_pct(last_atr, last_price), vol_score
    return "NONE", last_price, last_atr, atr_pct(last_atr, last_price), vol_score

# ========= GRID =========
def fee_amount(notional: float) -> float:
    return notional * FEE_PER_SIDE

def min_tp_pct_for_fees() -> float:
    # 2*fee + liten buffert
    return (2*FEE_PER_SIDE + 0.0001) * 100.0

def dynamic_tp_pct(safety_count: int) -> float:
    # större position → kräv lite större procentuell TP
    return max(GRID_TP_PCT + safety_count * TP_EXTRA_PER_SAFETY,
               min_tp_pct_for_fees()*100.0)  # värden i %

def _qty_from_usdt(price: float, usdt: float) -> float:
    return 0.0 if price <= 0 else usdt / price

def recalc(pos: Position):
    if pos.qty_total > 0:
        pos.avg_price = sum(p*q for p,q in pos.entries) / pos.qty_total
    else:
        pos.avg_price = 0.0

def set_targets(pos: Position, last_atr: float):
    step_abs = max(last_atr*GRID_STEP_ATR_MULT, pos.avg_price*GRID_STEP_MIN_PCT/100.0)
    tp_pct = dynamic_tp_pct(pos.safety_count) / 100.0
    if pos.side=="LONG":
        pos.next_step_price = pos.last_entry_price - step_abs
        pos.tp_price = pos.avg_price * (1.0 + tp_pct)
    else:
        pos.next_step_price = pos.last_entry_price + step_abs
        pos.tp_price = pos.avg_price * (1.0 - tp_pct)

def open_initial(symbol: str, side: str, price: float, last_atr: float) -> Position:
    qty = _qty_from_usdt(price, POSITION_SIZE_USDT)
    fee_in = fee_amount(qty*price)
    pos = Position(side=side)
    pos.entries.append((price, qty))
    pos.qty_total += qty
    pos.first_entry_price = price
    pos.last_entry_price = price
    pos.fee_in_acc += fee_in
    recalc(pos); set_targets(pos, last_atr)
    TRADE_LOG.append(TradeRec(datetime.now(timezone.utc), symbol, side, "ENTRY",
                              price, qty, 0.0, fee_in, 0.0, -fee_in))
    return pos

def add_safety(pos: Position, price: float, last_atr: float):
    k = pos.safety_count + 1
    usdt = POSITION_SIZE_USDT * (GRID_SIZE_MULT ** k)
    qty = _qty_from_usdt(price, usdt)
    fee_in = fee_amount(qty*price)
    pos.entries.append((price, qty))
    pos.qty_total += qty
    pos.last_entry_price = price
    pos.safety_count += 1
    pos.fee_in_acc += fee_in
    recalc(pos); set_targets(pos, last_atr)
    return qty, fee_in

def close_position(symbol: str, st: SymState, price: float, reason: str):
    pos = st.pos
    if not pos or pos.qty_total <= 0:
        return
    notional = pos.qty_total * price
    fee_out = fee_amount(notional)
    gross = (price - pos.avg_price)*pos.qty_total if pos.side=="LONG" else (pos.avg_price - price)*pos.qty_total
    net = gross - pos.fee_in_acc - fee_out

    st.realized_gross += gross
    st.realized_fees  += pos.fee_in_acc + fee_out
    st.realized_net   += net

    TRADE_LOG.append(TradeRec(datetime.now(timezone.utc), symbol, pos.side, f"EXIT:{reason}",
                              price, pos.qty_total, gross, pos.fee_in_acc, fee_out, net))
    st.pos = None
    st.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=2)

# ========= ENGINE =========
def open_positions_count() -> int:
    return sum(1 for s in STATE.symbols if STATE.per_sym[s].pos)

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    if open_positions_count() >= MAX_CONCURRENT_POS and not STATE.per_sym[sym].pos:
                        continue  # begränsa samtidiga marknadsrisker

                    st = STATE.per_sym[sym]
                    now = datetime.now(timezone.utc)
                    if st.cooldown_until and now < st.cooldown_until:
                        continue

                    try:
                        signal, last_price, last_atr, atrp, vol_score = await mtf_signal(sym, STATE.tfs)
                    except Exception:
                        continue

                    st.last_signal = signal

                    # Volatilitets- & volymfilter
                    if not (MIN_ATR_PCT <= atrp <= MAX_ATR_PCT):
                        continue
                    if vol_score < 0.8:  # svag volymtrend → hoppa
                        continue

                    # Öppna
                    if st.pos is None:
                        if signal == "LONG":
                            st.pos = open_initial(sym, "LONG", last_price, last_atr)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {last_price:.6f} "
                                    f"| qty {st.pos.qty_total:.6f} | avgFee≈{fee_amount(st.pos.qty_total*last_price):.4f}",
                                    reply_markup=reply_kb()
                                )
                        elif signal == "SHORT" and STATE.allow_shorts:
                            st.pos = open_initial(sym, "SHORT", last_price, last_atr)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔻 ENTRY {sym} SHORT @ {last_price:.6f} "
                                    f"| qty {st.pos.qty_total:.6f} | avgFee≈{fee_amount(st.pos.qty_total*last_price):.4f}",
                                    reply_markup=reply_kb()
                                )
                        continue

                    # Hantera öppen position
                    pos = st.pos

                    # Trendflip-skydd: om motsatt signal i 2/3 TF → ta hem (eller minska – vi stänger helt)
                    if (pos.side=="LONG" and signal=="SHORT") or (pos.side=="SHORT" and signal=="LONG"):
                        close_position(sym, st, last_price, reason="FLIP")
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id,
                                f"🔁 EXIT {sym} pga trendflip @ {last_price:.6f}", reply_markup=reply_kb())
                        continue

                    # Safety
                    if pos.safety_count < GRID_MAX_SAFETY:
                        if pos.side=="LONG" and last_price <= (pos.next_step_price or -1e9):
                            qty, fee_in = add_safety(pos, last_price, last_atr)
                            TRADE_LOG.append(TradeRec(datetime.now(timezone.utc), sym, "LONG",
                                                      "SAFETY", last_price, qty, 0.0, fee_in, 0.0, -fee_in))
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"➕ SAFETY {sym} LONG @ {last_price:.6f} | qty {qty:.6f} | avg {pos.avg_price:.6f} "
                                    f"| saf {pos.safety_count}/{GRID_MAX_SAFETY}",
                                    reply_markup=reply_kb()
                                )

                        if pos.side=="SHORT" and last_price >= (pos.next_step_price or 1e9):
                            qty, fee_in = add_safety(pos, last_price, last_atr)
                            TRADE_LOG.append(TradeRec(datetime.now(timezone.utc), sym, "SHORT",
                                                      "SAFETY", last_price, qty, 0.0, fee_in, 0.0, -fee_in))
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"➕ SAFETY {sym} SHORT @ {last_price:.6f} | qty {qty:.6f} | avg {pos.avg_price:.6f} "
                                    f"| saf {pos.safety_count}/{GRID_MAX_SAFETY}",
                                    reply_markup=reply_kb()
                                )

                    # TP
                    if pos.side=="LONG" and pos.tp_price and last_price >= pos.tp_price:
                        close_position(sym, st, last_price, reason="TP")
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id,
                                f"✅ EXIT {sym} @ {last_price:.6f} | Net PnL {st.realized_net:+.4f} USDT",
                                reply_markup=reply_kb())
                    elif pos.side=="SHORT" and pos.tp_price and last_price <= pos.tp_price:
                        close_position(sym, st, last_price, reason="TP")
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id,
                                f"✅ EXIT {sym} @ {last_price:.6f} | Net PnL {st.realized_net:+.4f} USDT",
                                reply_markup=reply_kb())

                    # Max-drawdown från första entry
                    if st.pos:
                        dd_level = pos.first_entry_price*(1 - MAX_DRAWDOWN_PCT/100.0) if pos.side=="LONG" \
                                   else pos.first_entry_price*(1 + MAX_DRAWDOWN_PCT/100.0)
                        hit = (pos.side=="LONG" and last_price <= dd_level) or (pos.side=="SHORT" and last_price >= dd_level)
                        if hit:
                            close_position(sym, st, last_price, reason="DD")
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🛑 EXIT {sym} (Drawdown) @ {last_price:.6f}", reply_markup=reply_kb())

            await asyncio.sleep(ENGINE_SLEEP_SEC)
        except Exception:
            await asyncio.sleep(ENGINE_SLEEP_SEC)

# ========= TELEGRAM =========
tg_app = Application.builder().token(BOT_TOKEN).build()

def fmt(x: float) -> str: return f"{x:+.4f}"

async def send_status(chat_id: int):
    total_net = sum(st.realized_net for st in STATE.per_sym.values())
    open_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            p = st.pos
            open_lines.append(f"{s} {p.side} avg:{p.avg_price:.6f} qty:{p.qty_total:.6f} tp:{(p.tp_price or 0):.6f} "
                              f"saf:{p.safety_count}/{GRID_MAX_SAFETY}")
    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {POSITION_SIZE_USDT:.1f} USDT | Fee/side: {FEE_PER_SIDE*100:.4f}%",
        f"Grid: max_safety={GRID_MAX_SAFETY} step_mult={GRID_STEP_ATR_MULT} step_min={GRID_STEP_MIN_PCT}% "
        f"size_mult={GRID_SIZE_MULT} tp={GRID_TP_PCT}% (+{TP_EXTRA_PER_SAFETY}%/safety)",
        f"Risk: dd={MAX_DRAWDOWN_PCT}% | atr% {MIN_ATR_PCT}–{MAX_ATR_PCT} | max_pos={MAX_CONCURRENT_POS} | shorts={'ON' if STATE.allow_shorts else 'OFF'}",
        f"PnL total (NET): {fmt(total_net)} USDT",
        "PnL per symbol: " + ", ".join([f"{s}:{fmt(STATE.per_sym[s].realized_net)}" for s in STATE.symbols]),
        "Positioner: " + (", ".join(open_lines) if open_lines else "inga")
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# --- Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! MP Grid v40 Pro redo ✅", reply_markup=reply_kb())
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

async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts)==2:
        tfs = [p.strip() for p in parts[1].split(",") if p.strip()]
        if all(p in TF_MAP for p in tfs) and len(tfs)>=2:
            STATE.tfs = tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes: {', '.join(STATE.tfs)}", reply_markup=reply_kb())
            return
        await tg_app.bot.send_message(STATE.chat_id, "Format: /timeframe 3m,5m,15m", reply_markup=reply_kb())
        return
    # toggle två presets
    STATE.tfs = ["1m","3m","5m"] if STATE.tfs!=["1m","3m","5m"] else ["3m","5m","15m"]
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframes: {', '.join(STATE.tfs)}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    total_net = sum(st.realized_net for st in STATE.per_sym.values())
    total_fees = sum(st.realized_fees for st in STATE.per_sym.values())
    total_gross = sum(st.realized_gross for st in STATE.per_sym.values())
    lines = [f"📈 TOTAL Net {fmt(total_net)} | Gross {fmt(total_gross)} | Fees {total_fees:.4f}"]
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        lines.append(f"• {s}: Net {fmt(st.realized_net)} | Gross {fmt(st.realized_gross)} | Fees {st.realized_fees:.4f}")
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_export(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    if not TRADE_LOG:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades att exportera.", reply_markup=reply_kb())
        return
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time","symbol","side","action","price","qty","gross","fee_in","fee_out","net"])
    for t in TRADE_LOG:
        w.writerow([t.time.isoformat(), t.symbol, t.side, t.action,
                    f"{t.price:.8f}", f"{t.qty:.8f}",
                    f"{t.gross:.8f}", f"{t.fee_in:.8f}", f"{t.fee_out:.8f}", f"{t.net:.8f}"])
    buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode()),
                                   filename="trades_v40_pro.csv", caption="Export (mock) – v40 Pro",
                                   reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = False
    for s, st in STATE.per_sym.items():
        if st.pos:
            price = st.pos.avg_price
            close_position(s, st, price, reason="PANIC")
            closed = True
    await tg_app.bot.send_message(STATE.chat_id, "Panic close: " + ("klart" if closed else "inga positioner"),
                                  reply_markup=reply_kb())

async def cmd_reset(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    TRADE_LOG.clear()
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        ss.realized_net = ss.realized_gross = ss.realized_fees = 0.0
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

# --- Live-tuning: /grid och /risk
def _float(v:str)->Optional[float]:
    try: return float(v)
    except: return None
def _int(v:str)->Optional[int]:
    try: return int(v)
    except: return None

async def cmd_grid(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = (update.message.text or "").strip()
    parts = txt.split()
    global GRID_MAX_SAFETY, GRID_STEP_ATR_MULT, GRID_STEP_MIN_PCT, GRID_SIZE_MULT, GRID_TP_PCT, TP_EXTRA_PER_SAFETY
    if len(parts)>=3 and parts[0]=="/grid" and parts[1]=="set":
        key, val = parts[2], (parts[3] if len(parts)>=4 else None)
        ok=False
        if key=="max" and (n:=_int(val)) is not None: GRID_MAX_SAFETY=n; ok=True
        elif key=="step_mult" and (f:=_float(val)) is not None: GRID_STEP_ATR_MULT=f; ok=True
        elif key=="step_min" and (f:=_float(val)) is not None: GRID_STEP_MIN_PCT=f; ok=True
        elif key=="size_mult" and (f:=_float(val)) is not None: GRID_SIZE_MULT=f; ok=True
        elif key=="tp" and (f:=_float(val)) is not None: GRID_TP_PCT=f; ok=True
        elif key=="tp_extra" and (f:=_float(val)) is not None: TP_EXTRA_PER_SAFETY=f; ok=True
        await tg_app.bot.send_message(STATE.chat_id,
            ("Grid uppdaterad." if ok else "Fel. /grid set <max|step_mult|step_min|size_mult|tp|tp_extra> <värde>"),
            reply_markup=reply_kb())
        return
    # show
    await tg_app.bot.send_message(STATE.chat_id,
        f"Grid:\n"
        f"  max_safety={GRID_MAX_SAFETY}\n"
        f"  step_mult={GRID_STEP_ATR_MULT}\n"
        f"  step_min%={GRID_STEP_MIN_PCT}\n"
        f"  size_mult={GRID_SIZE_MULT}\n"
        f"  tp%={GRID_TP_PCT} (+{TP_EXTRA_PER_SAFETY}%/safety)\n"
        f"Ex: /grid set step_mult 0.7",
        reply_markup=reply_kb())

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = (update.message.text or "").strip()
    parts = txt.split()
    global MAX_DRAWDOWN_PCT, MIN_ATR_PCT, MAX_ATR_PCT, MAX_CONCURRENT_POS
    if len(parts)>=3 and parts[0]=="/risk" and parts[1]=="set":
        key, val = parts[2], (parts[3] if len(parts)>=4 else None)
        ok=False
        if key=="dd" and (f:=_float(val)) is not None: MAX_DRAWDOWN_PCT=f; ok=True
        elif key=="min_atr" and (f:=_float(val)) is not None: MIN_ATR_PCT=f; ok=True
        elif key=="max_atr" and (f:=_float(val)) is not None: MAX_ATR_PCT=f; ok=True
        elif key=="max_pos" and (n:=_int(val)) is not None: MAX_CONCURRENT_POS=n; ok=True
        await tg_app.bot.send_message(STATE.chat_id,
            ("Risk uppdaterad." if ok else "Fel. /risk set <dd|min_atr|max_atr|max_pos> <värde>"),
            reply_markup=reply_kb())
        return
    await tg_app.bot.send_message(STATE.chat_id,
        f"Risk:\n"
        f"  drawdown%={MAX_DRAWDOWN_PCT}\n"
        f"  atr% min/max = {MIN_ATR_PCT} – {MAX_ATR_PCT}\n"
        f"  max_concurrent_pos={MAX_CONCURRENT_POS}\n"
        f"Ex: /risk set dd 2.5",
        reply_markup=reply_kb())

# Registrera handlers
tg_app.add_handler(CommandHandler("start",      cmd_start))
tg_app.add_handler(CommandHandler("status",     cmd_status))
tg_app.add_handler(CommandHandler("engine_on",  cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe",  cmd_timeframe))
tg_app.add_handler(CommandHandler("pnl",        cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export))
tg_app.add_handler(CommandHandler("panic",      cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl",  cmd_reset))
tg_app.add_handler(CommandHandler("grid",       cmd_grid))
tg_app.add_handler(CommandHandler("risk",       cmd_risk))

# ========= FASTAPI WEBHOOK =========
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
    total_net = sum(s.realized_net for s in STATE.per_sym.values())
    return {"ok": True, "engine_on": STATE.engine_on, "tfs": STATE.tfs, "pnl_net": round(total_net, 6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
