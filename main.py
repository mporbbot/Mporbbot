# main_v43.py
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
TIMEFRAME = os.getenv("TIMEFRAME", "3m")          # 1m,3m,5m,15m,30m,1h
ENTRYMODE = os.getenv("ENTRY_MODE", "close")      # close|tick
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "100"))
EMA_ENABLED = os.getenv("EMA_ENABLED", "true").lower() in ("1","true","yes","on")
EMA_PERIOD = int(os.getenv("EMA_PERIOD", "200"))

# Handelsavgift i baspunkter per sida (entry OCH exit). 8 = 0.08%
FEE_BPS = float(os.getenv("FEE_BPS", "8"))
FEE_RATE = FEE_BPS / 10_000.0  # t.ex. 8 bp -> 0.0008 (0.08%)

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ========== STATE ==========
@dataclass
class Position:
    side: str           # "LONG" eller "SHORT"
    entry: float
    stop: float
    qty: float          # beräknas från POSITION_SIZE_USDT/entry

@dataclass
class SymState:
    # LONG-ORB (första gröna efter röd)
    long_orb_high: Optional[float] = None
    long_orb_low: Optional[float] = None
    long_orb_time: Optional[int] = None

    # SHORT-ORB (första röda efter grön)
    short_orb_high: Optional[float] = None
    short_orb_low: Optional[float] = None
    short_orb_time: Optional[int] = None

    pos: Optional[Position] = None
    realized_pnl: float = 0.0
    trades: List[Tuple[str, float]] = field(default_factory=list)

@dataclass
class EngineState:
    engine_on: bool = False
    entry_mode: str = ENTRYMODE
    timeframe: str = TIMEFRAME
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    orb_on: bool = True
    ema_on: bool = EMA_ENABLED
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ========== UTIL ==========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/ema_on"), KeyboardButton("/ema_off")],
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

async def get_klines(symbol: str, tf: str, limit: int = 210) -> List[Tuple[int,float,float,float,float]]:
    """
    Returnerar lista av (startMs, open, high, low, close) – nyast först.
    Vi hämtar många candles så EMA200 kan räknas stabilt.
    KuCoin data: [time, open, close, high, low, volume, turnover]
    """
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=12) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # newest first
    out = []
    for row in data[:limit]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

def is_green(o,h,l,c): return c > o
def is_red(o,h,l,c):   return c < o

def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

# ========== ORB & ENTRY ==========
def detect_orbs(prev_closed: Tuple[int,float,float,float,float],
                last_closed: Tuple[int,float,float,float,float]) -> Tuple[Optional[Tuple[float,float,int]],
                                                                          Optional[Tuple[float,float,int]]]:
    """
    ORB byggs på FÖRSTA stängda candlen som byter färg:
      - LONG-ORB: prev röd & last grön  → ORB = last(h,l,time)
      - SHORT-ORB: prev grön & last röd → ORB = last(h,l,time)
    Returnerar (long_orb, short_orb) där varje är (H,L,time_ms) eller None.
    """
    _, po, ph, pl, pc = prev_closed
    _, co, ch, cl, cc = last_closed

    long_orb = None
    short_orb = None
    if is_red(po,ph,pl,pc) and is_green(co,ch,cl,cc):
        long_orb = (ch, cl, last_closed[0])
    elif is_green(po,ph,pl,pc) and is_red(co,ch,cl,cc):
        short_orb = (ch, cl, last_closed[0])
    return long_orb, short_orb

def long_entry_close_rule(orb_high: float, orb_time: int,
                          last_closed: Tuple[int,float,float,float,float]) -> bool:
    t,o,h,l,c = last_closed
    return (t > orb_time) and (c > orb_high)

def long_entry_tick_rule(orb_high: float, orb_time: int,
                         current: Tuple[int,float,float,float,float]) -> bool:
    t,o,h,l,c = current
    return (t > orb_time) and (h >= orb_high)

def short_entry_close_rule(orb_low: float, orb_time: int,
                           last_closed: Tuple[int,float,float,float,float]) -> bool:
    t,o,h,l,c = last_closed
    return (t > orb_time) and (c < orb_low)

def short_entry_tick_rule(orb_low: float, orb_time: int,
                          current: Tuple[int,float,float,float,float]) -> bool:
    t,o,h,l,c = current
    return (t > orb_time) and (l <= orb_low)

def trail_stop_long(cur_stop: float, prev_closed: Tuple[int,float,float,float,float]) -> float:
    _, o,h,l,c = prev_closed
    return max(cur_stop, l)

def trail_stop_short(cur_stop: float, prev_closed: Tuple[int,float,float,float,float]) -> float:
    _, o,h,l,c = prev_closed
    return min(cur_stop, h)

def stop_hit_long_close(stop: float, last_closed: Tuple[int,float,float,float,float]) -> bool:
    _, o,h,l,c = last_closed
    return c < stop

def stop_hit_long_tick(stop: float, current: Tuple[int,float,float,float,float]) -> bool:
    _, o,h,l,c = current
    return l <= stop

def stop_hit_short_close(stop: float, last_closed: Tuple[int,float,float,float,float]) -> bool:
    _, o,h,l,c = last_closed
    return c > stop

def stop_hit_short_tick(stop: float, current: Tuple[int,float,float,float,float]) -> bool:
    _, o,h,l,c = current
    return h >= stop

# ========== ENGINE ==========
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on and STATE.orb_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]

                    kl = await get_klines(sym, STATE.timeframe, limit=max(EMA_PERIOD+3, 210))
                    if len(kl) < 3:
                        continue
                    current, last_closed, prev_closed = kl[0], kl[1], kl[2]

                    # EMA200-filter (på stängda candles)
                    ema_ok_long = True
                    ema_ok_short = True
                    if STATE.ema_on:
                        closes = [c[4] for c in reversed(kl)]  # äldst -> nyast
                        e = ema(closes, EMA_PERIOD)
                        if e is None:
                            ema_ok_long = ema_ok_short = False
                        else:
                            last_c_close = last_closed[4]
                            ema_ok_long = last_c_close > e
                            ema_ok_short = last_c_close < e

                    # 1) ORB-UPPDATERING
                    long_orb, short_orb = detect_orbs(prev_closed, last_closed)
                    if long_orb:
                        st.long_orb_high, st.long_orb_low, st.long_orb_time = long_orb
                    if short_orb:
                        st.short_orb_high, st.short_orb_low, st.short_orb_time = short_orb

                    # 2) ENTRY (bara om ingen position)
                    if st.pos is None:
                        # LONG
                        if st.long_orb_high and st.long_orb_low:
                            cond = (long_entry_close_rule(st.long_orb_high, st.long_orb_time, last_closed)
                                    if STATE.entry_mode == "close"
                                    else long_entry_tick_rule(st.long_orb_high, st.long_orb_time, current))
                            if cond and ema_ok_long:
                                entry_px = last_closed[4] if STATE.entry_mode == "close" else st.long_orb_high
                                qty = POSITION_SIZE_USDT / entry_px if entry_px > 0 else 0.0
                                st.pos = Position("LONG", entry=entry_px, stop=st.long_orb_low, qty=qty)
                                if STATE.chat_id:
                                    fee_est = (entry_px * qty) * FEE_RATE
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        (f"🟢 ENTRY LONG {sym} @ {entry_px:.6f}\n"
                                         f"ORB H:{st.long_orb_high:.6f} L:{st.long_orb_low:.6f} | "
                                         f"SL {st.pos.stop:.6f} | QTY {qty:.6f} | Fee≈{fee_est:.4f} USDT")
                                    )

                        # SHORT
                        if st.pos is None and st.short_orb_low and st.short_orb_high:
                            cond = (short_entry_close_rule(st.short_orb_low, st.short_orb_time, last_closed)
                                    if STATE.entry_mode == "close"
                                    else short_entry_tick_rule(st.short_orb_low, st.short_orb_time, current))
                            if cond and ema_ok_short:
                                entry_px = last_closed[4] if STATE.entry_mode == "close" else st.short_orb_low
                                qty = POSITION_SIZE_USDT / entry_px if entry_px > 0 else 0.0
                                st.pos = Position("SHORT", entry=entry_px, stop=st.short_orb_high, qty=qty)
                                if STATE.chat_id:
                                    fee_est = (entry_px * qty) * FEE_RATE
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        (f"🔻 ENTRY SHORT {sym} @ {entry_px:.6f}\n"
                                         f"ORB H:{st.short_orb_high:.6f} L:{st.short_orb_low:.6f} | "
                                         f"SL {st.pos.stop:.6f} | QTY {qty:.6f} | Fee≈{fee_est:.4f} USDT")
                                    )

                    # 3) TRAILING SL + EXIT
                    if st.pos:
                        exited = False
                        if st.pos.side == "LONG":
                            new_sl = trail_stop_long(st.pos.stop, last_closed)
                            if new_sl > st.pos.stop:
                                st.pos.stop = new_sl
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id,
                                        f"🔧 SL LONG {sym} -> {st.pos.stop:.6f}")
                            if STATE.entry_mode == "close" and stop_hit_long_close(st.pos.stop, last_closed):
                                exit_px = last_closed[4]; exited = True
                            elif STATE.entry_mode == "tick" and stop_hit_long_tick(st.pos.stop, current):
                                exit_px = st.pos.stop; exited = True
                            if exited:
                                gross = (exit_px - st.pos.entry) * st.pos.qty
                                fee = (st.pos.entry * st.pos.qty) * FEE_RATE + (exit_px * st.pos.qty) * FEE_RATE
                                pnl = gross - fee
                        else:  # SHORT
                            new_sl = trail_stop_short(st.pos.stop, last_closed)
                            if new_sl < st.pos.stop:
                                st.pos.stop = new_sl
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id,
                                        f"🔧 SL SHORT {sym} -> {st.pos.stop:.6f}")
                            if STATE.entry_mode == "close" and stop_hit_short_close(st.pos.stop, last_closed):
                                exit_px = last_closed[4]; exited = True
                            elif STATE.entry_mode == "tick" and stop_hit_short_tick(st.pos.stop, current):
                                exit_px = st.pos.stop; exited = True
                            if exited:
                                gross = (st.pos.entry - exit_px) * st.pos.qty
                                fee = (st.pos.entry * st.pos.qty) * FEE_RATE + (exit_px * st.pos.qty) * FEE_RATE
                                pnl = gross - fee

                        if exited:
                            st.realized_pnl += pnl
                            st.trades.append((f"{sym} {st.pos.side}", pnl))
                            if len(st.trades) > 50:
                                st.trades = st.trades[-50:]
                            if STATE.chat_id:
                                sign = "✅" if pnl >= 0 else "❌"
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    (f"🔴 EXIT {sym} @ {exit_px:.6f} | "
                                     f"Gross: {gross:+.4f} | Fee: {fee:.4f} | "
                                     f"PnL NET: {pnl:+.4f} USDT {sign}")
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

def fmt_orb(st: SymState) -> str:
    a = (f"LONG-ORB(H:{st.long_orb_high:.6f} L:{st.long_orb_low:.6f})" 
         if st.long_orb_high and st.long_orb_low else "LONG-ORB: -")
    b = (f"SHORT-ORB(H:{st.short_orb_high:.6f} L:{st.short_orb_low:.6f})"
         if st.short_orb_high and st.short_orb_low else "SHORT-ORB: -")
    return f"{a} | {b}"

async def send_status(chat_id: int):
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} @ {st.pos.entry:.6f} | SL {st.pos.stop:.6f} | QTY {st.pos.qty:.6f}")
    sym_pnls = ", ".join([f"{s}:{STATE.per_sym[s].realized_pnl:+.2f}" for s in STATE.symbols])

    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Entry mode: {STATE.entry_mode}",
        f"Timeframe: {STATE.timeframe}",
        f"EMA200 filter: {'PÅ' if STATE.ema_on else 'AV'} (period {EMA_PERIOD})",
        f"Fee: {FEE_BPS:.2f} bp per sida ({FEE_RATE*100:.4f}%)",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"ORB: {'PÅ' if STATE.orb_on else 'AV'} (1 candle – första grön efter röd / första röd efter grön)",
        f"PnL total (NET): {total_pnl:+.2f} USDT",
        f"PnL per symbol: {sym_pnls if sym_pnls else '-'}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())
    for s in STATE.symbols:
        await tg_app.bot.send_message(chat_id, f"{s} | {fmt_orb(STATE.per_sym[s])}")

async def send_pnl(chat_id: int):
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    lines = [f"📈 PnL total (NET): {total_pnl:+.4f} USDT  |  Fee per sida: {FEE_BPS:.2f} bp"]
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
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! ORB-bot v43 (fee i mock-PnL) ✅", reply_markup=reply_kb())
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

async def cmd_ema_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.ema_on = True
    await tg_app.bot.send_message(STATE.chat_id, "EMA200-filter: PÅ ✅", reply_markup=reply_kb())

async def cmd_ema_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.ema_on = False
    await tg_app.bot.send_message(STATE.chat_id, "EMA200-filter: AV ⛔️", reply_markup=reply_kb())

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
            # panic på SL-priset
            exit_px = st.pos.stop
            if st.pos.side == "LONG":
                gross = (exit_px - st.pos.entry) * st.pos.qty
            else:
                gross = (st.pos.entry - exit_px) * st.pos.qty
            fee = (st.pos.entry * st.pos.qty) * FEE_RATE + (exit_px * st.pos.qty) * FEE_RATE
            pnl = gross - fee
            st.realized_pnl += pnl
            st.trades.append((f"{s} {st.pos.side} (panic)", pnl))
            closed.append(f"{s} pnl {pnl:+.4f} (gross {gross:+.4f} - fee {fee:.4f})")
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

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
tg_app.add_handler(CommandHandler("ema_on", cmd_ema_on))
tg_app.add_handler(CommandHandler("ema_off", cmd_ema_off))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))

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
        "ema200": STATE.ema_on,
        "fee_bps": FEE_BPS,
        "pnl_total_net": round(total_pnl, 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
