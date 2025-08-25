# main_v36_both.py
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

# ========= ENV =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")          # 1m,3m,5m,15m,30m,1h
ENTRYMODE = os.getenv("ENTRY_MODE", "close")      # close|tick
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "100"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.001"))    # 0.001 = 0.10% per sida (mock)

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ========= STATE =========
@dataclass
class Position:
    side: str           # "LONG" eller "SHORT"
    entry: float
    stop: float
    qty: float          # USDT-storlek/entry
    locked: bool = False

@dataclass
class SymState:
    # Long-ORB
    long_orb_h: Optional[float] = None
    long_orb_l: Optional[float] = None
    long_orb_t: Optional[int]   = None  # ms tid för ORB-candle

    # Short-ORB
    short_orb_h: Optional[float] = None
    short_orb_l: Optional[float] = None
    short_orb_t: Optional[int]   = None

    pos: Optional[Position] = None
    realized_gross: float = 0.0
    realized_fees: float = 0.0
    trades: List[Tuple[str, float, float]] = field(default_factory=list)  # (label, gross, fee)

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

# ========= UI =========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ========= DATA =========
async def get_klines(symbol: str, tf: str, limit: int = 3) -> List[Tuple[int,float,float,float,float]]:
    """
    Returnerar lista av (startMs, open, high, low, close) – nyast först (KuCoin).
    Vi hämtar få candles för att undvika 429.
    """
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

def is_green(o,h,l,c): return c > o
def is_red(o,h,l,c):   return c < o

# ========= ORB =========
def detect_long_orb(prev, cur) -> Optional[Tuple[float,float,int]]:
    """Första GRÖNA efter RÖD → (high, low, time_ms)."""
    _, po, ph, pl, pc = prev
    _, co, ch, cl, cc = cur
    if is_red(po,ph,pl,pc) and is_green(co,ch,cl,cc):
        return (ch, cl, cur[0])
    return None

def detect_short_orb(prev, cur) -> Optional[Tuple[float,float,int]]:
    """Första RÖDA efter GRÖN → (high, low, time_ms)."""
    _, po, ph, pl, pc = prev
    _, co, ch, cl, cc = cur
    if is_green(po,ph,pl,pc) and is_red(co,ch,cl,cc):
        return (ch, cl, cur[0])
    return None

# Entryregler
def entry_long_close(orb_high: float, closed):
    _, o,h,l,c = closed
    return c > orb_high

def entry_long_tick(orb_high: float, cur):
    _, o,h,l,c = cur
    return h >= orb_high

def entry_short_close(orb_low: float, closed):
    _, o,h,l,c = closed
    return c < orb_low

def entry_short_tick(orb_low: float, cur):
    _, o,h,l,c = cur
    return l <= orb_low

# Stop/exit-regler
def next_stop_long(cur_stop: float, prev_closed) -> float:
    _, o,h,l,c = prev_closed
    return max(cur_stop, l)

def next_stop_short(cur_stop: float, prev_closed) -> float:
    _, o,h,l,c = prev_closed
    return min(cur_stop, h)

def stop_hit_long_close(stop: float, closed) -> bool:
    _, o,h,l,c = closed
    return c < stop

def stop_hit_long_tick(stop: float, cur) -> bool:
    _, o,h,l,c = cur
    return l <= stop

def stop_hit_short_close(stop: float, closed) -> bool:
    _, o,h,l,c = closed
    return c > stop

def stop_hit_short_tick(stop: float, cur) -> bool:
    _, o,h,l,c = cur
    return h >= stop

# ========= ENGINE =========
async def engine_loop(app: Application):
    # lugnare polling så vi slipper 429
    SLEEP_SEC = 10
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on and STATE.orb_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    kl = await get_klines(sym, STATE.timeframe, limit=3)
                    if len(kl) < 2:
                        continue
                    last_closed, current = kl[1], kl[0]

                    # 1) Bygg ORB (endast stängda candles)
                    long_orb = detect_long_orb(last_closed, current)
                    if long_orb:
                        st.long_orb_h, st.long_orb_l, st.long_orb_t = long_orb

                    short_orb = detect_short_orb(last_closed, current)
                    if short_orb:
                        st.short_orb_h, st.short_orb_l, st.short_orb_t = short_orb

                    # 2) ENTRY (om ingen position)
                    if st.pos is None:
                        # LONG?
                        if st.long_orb_h and st.long_orb_l:
                            if (STATE.entry_mode == "close" and entry_long_close(st.long_orb_h, last_closed)) or \
                               (STATE.entry_mode == "tick"  and entry_long_tick(st.long_orb_h, current)):
                                entry_px = last_closed[4] if STATE.entry_mode == "close" else st.long_orb_h
                                qty = POSITION_SIZE_USDT / entry_px if entry_px > 0 else 0.0
                                st.pos = Position("LONG", entry=entry_px, stop=st.long_orb_l, qty=qty)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 ENTRY LONG ({STATE.entry_mode}) {sym} @ {entry_px:.4f}\n"
                                        f"ORB(H:{st.long_orb_h:.4f} L:{st.long_orb_l:.4f}) | "
                                        f"SL={st.pos.stop:.4f} | QTY={qty:.6f}"
                                    )

                        # SHORT? (bara om fortfarande ingen pos)
                        if st.pos is None and st.short_orb_h and st.short_orb_l:
                            if (STATE.entry_mode == "close" and entry_short_close(st.short_orb_l, last_closed)) or \
                               (STATE.entry_mode == "tick"  and entry_short_tick(st.short_orb_l, current)):
                                entry_px = last_closed[4] if STATE.entry_mode == "close" else st.short_orb_l
                                qty = POSITION_SIZE_USDT / entry_px if entry_px > 0 else 0.0
                                st.pos = Position("SHORT", entry=entry_px, stop=st.short_orb_h, qty=qty)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 ENTRY SHORT ({STATE.entry_mode}) {sym} @ {entry_px:.4f}\n"
                                        f"ORB(H:{st.short_orb_h:.4f} L:{st.short_orb_l:.4f}) | "
                                        f"SL={st.pos.stop:.4f} | QTY={qty:.6f}"
                                    )

                    # 3) TRAIL & EXIT
                    if st.pos:
                        if st.pos.side == "LONG":
                            # trail
                            new_sl = next_stop_long(st.pos.stop, last_closed)
                            if new_sl > st.pos.stop:
                                st.pos.stop = new_sl
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🔧 SL flyttad {sym} -> {st.pos.stop:.4f}")
                            # exit?
                            hit = stop_hit_long_close(st.pos.stop, last_closed) if STATE.entry_mode == "close" \
                                  else stop_hit_long_tick(st.pos.stop, current)
                            if hit:
                                exit_px = last_closed[4] if STATE.entry_mode == "close" else st.pos.stop
                                gross = (exit_px - st.pos.entry) * st.pos.qty
                                fee = (abs(st.pos.entry) + abs(exit_px)) * st.pos.qty * FEE_PCT
                                st.realized_gross += gross
                                st.realized_fees += fee
                                st.trades.append((f"{sym} LONG", gross, fee))
                                if STATE.chat_id:
                                    sign = "✅" if (gross - fee) >= 0 else "❌"
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔴 EXIT {sym} @ {exit_px:.4f} | "
                                        f"Gross: {gross:+.4f}  Fees: {fee:.4f}  Net: {(gross-fee):+.4f} USDT {sign}"
                                    )
                                st.pos = None

                        elif st.pos.side == "SHORT":
                            # trail
                            new_sl = next_stop_short(st.pos.stop, last_closed)
                            if new_sl < st.pos.stop:
                                st.pos.stop = new_sl
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🔧 SL flyttad {sym} -> {st.pos.stop:.4f}")
                            # exit?
                            hit = stop_hit_short_close(st.pos.stop, last_closed) if STATE.entry_mode == "close" \
                                  else stop_hit_short_tick(st.pos.stop, current)
                            if hit:
                                exit_px = last_closed[4] if STATE.entry_mode == "close" else st.pos.stop
                                gross = (st.pos.entry - exit_px) * st.pos.qty
                                fee = (abs(st.pos.entry) + abs(exit_px)) * st.pos.qty * FEE_PCT
                                st.realized_gross += gross
                                st.realized_fees += fee
                                st.trades.append((f"{sym} SHORT", gross, fee))
                                if STATE.chat_id:
                                    sign = "✅" if (gross - fee) >= 0 else "❌"
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔴 EXIT {sym} @ {exit_px:.4f} | "
                                        f"Gross: {gross:+.4f}  Fees: {fee:.4f}  Net: {(gross-fee):+.4f} USDT {sign}"
                                    )
                                st.pos = None

            await asyncio.sleep(SLEEP_SEC)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ========= TELEGRAM =========
tg_app = Application.builder().token(BOT_TOKEN).build()

async def _totals():
    gross = sum(s.realized_gross for s in STATE.per_sym.values())
    fees  = sum(s.realized_fees for s in STATE.per_sym.values())
    return gross, fees, gross - fees

async def send_status(chat_id: int):
    gross, fees, net = await _totals()
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} @ {st.pos.entry:.4f} | SL {st.pos.stop:.4f} | QTY {st.pos.qty:.6f}")
    sym_pnls = ", ".join([f"{s}: Net {STATE.per_sym[s].realized_gross-STATE.per_sym[s].realized_fees:+.4f}"
                          f" (Gross {STATE.per_sym[s].realized_gross:+.4f}, Fees {STATE.per_sym[s].realized_fees:.4f})"
                          for s in STATE.symbols])

    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Entry mode: {STATE.entry_mode}",
        f"Timeframe: {STATE.timeframe}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"ORB: {'PÅ' if STATE.orb_on else 'AV'} (1:a gröna efter röd = LONG, 1:a röda efter grön = SHORT)",
        f"PnL (TOTAL)  Net: {net:+.4f}   Gross: {gross:+.4f}   Fees: {fees:.4f}",
        f"PnL per symbol: {sym_pnls if sym_pnls else '-'}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
        f"Avgiftssats (mock): {FEE_PCT*100:.4f}%",
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    gross, fees, net = await _totals()
    lines = [f"📈 PnL TOTAL  Net: {net:+.4f}  (Gross {gross:+.4f}, Fees {fees:.4f})"]
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        lines.append(f"• {s}: Net {ss.realized_gross-ss.realized_fees:+.4f} "
                     f"(Gross {ss.realized_gross:+.4f}, Fees {ss.realized_fees:.4f})")
    if any(STATE.per_sym[s].trades for s in STATE.symbols):
        lines.append("\nSenaste affärer:")
        recent = []
        for s in STATE.symbols:
            recent.extend(STATE.per_sym[s].trades[-5:])
        for lbl, g, f in recent[-10:]:
            lines.append(f"  - {lbl}: Net {(g-f):+.4f} (Gross {g:+.4f}, Fees {f:.4f})")
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! ORB-bot (båda håll) redo ✅", reply_markup=reply_kb())
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
            if st.pos.side == "LONG":
                gross = (exit_px - st.pos.entry) * st.pos.qty
            else:
                gross = (st.pos.entry - exit_px) * st.pos.qty
            fee = (abs(st.pos.entry) + abs(exit_px)) * st.pos.qty * FEE_PCT
            st.realized_gross += gross
            st.realized_fees += fee
            st.trades.append((f"{s} {st.pos.side} (panic)", gross, fee))
            closed.append(f"{s} Net {(gross-fee):+.4f}")
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
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

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

# ========= FASTAPI / WEBHOOK =========
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
    gross, fees, net = await _totals()
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tf": STATE.timeframe,
        "mode": STATE.entry_mode,
        "gross_total": round(gross, 6),
        "fees_total": round(fees, 6),
        "net_total": round(net, 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
