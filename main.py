# main_v36_fee.py
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

SYMBOLS = (
    os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
    .replace(" ", "")
).split()

# tillåt både kommatecken och mellanslag
if len(SYMBOLS) == 1 and "," in SYMBOLS[0]:
    SYMBOLS = SYMBOLS[0].split(",")

TIMEFRAME = os.getenv("TIMEFRAME", "1m")          # 1m,3m,5m,15m,30m,1h
ENTRYMODE = os.getenv("ENTRY_MODE", "close")      # close|tick
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "100"))  # mock-positionstorlek

# Handelsavgift (mock): t.ex. 0.001 = 0.1%
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))

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
class TradeRec:
    label: str
    gross: float
    fees: float
    net: float

@dataclass
class SymState:
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_candle_time: Optional[int] = None
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0     # USDT (efter avgifter)
    realized_pnl_gross: float = 0.0   # USDT (före avgifter)
    realized_fees: float = 0.0        # USDT (summa avgifter)
    trades: List[TradeRec] = field(default_factory=list)

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

# ========== UTIL ==========
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
        # KuCoin time är i sekunder -> ms
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

def is_green(o,h,l,c): return c > o
def is_red(o,h,l,c):   return c < o

# ======= FEE HELPERS =======
def fee_notional(price: float, qty: float) -> float:
    """Avgift per affärsben (maker/taker oräknat) = FEE_RATE * (price * qty)"""
    return FEE_RATE * price * qty

def trade_pnl_long(entry: float, exit_px: float, qty: float) -> Tuple[float, float, float]:
    """
    LONG:
      gross = (exit - entry) * qty
      fees = fee(entry*qty) + fee(exit*qty)
      net = gross - fees
    """
    gross = (exit_px - entry) * qty
    fees = fee_notional(entry, qty) + fee_notional(exit_px, qty)
    net = gross - fees
    return gross, fees, net

# ========== ORB LOGIK ==========
def detect_orb(prev: Tuple[int,float,float,float,float],
               cur: Tuple[int,float,float,float,float]) -> Optional[Tuple[float,float,int]]:
    """
    ORB = första gröna candle efter en röd.
    Returnerar (H,L,time_ms) om cur kvalar.
    """
    _, po, ph, pl, pc = prev
    _, co, ch, cl, cc = cur
    if is_red(po,ph,pl,pc) and is_green(co,ch,cl,cc):
        return (ch, cl, cur[0])
    return None

def entry_close_rule(orb_high: float, closed_candle: Tuple[int,float,float,float,float]) -> bool:
    # Stänger över ORB-high
    _, o, h, l, c = closed_candle
    return c > orb_high

def entry_tick_rule(orb_high: float, candle: Tuple[int,float,float,float,float]) -> bool:
    # Touch/över ORB-high intra
    _, o, h, l, c = candle
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
    while True:
        try:
            if STATE.engine_on and STATE.orb_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    kl = await get_klines(sym, STATE.timeframe, limit=3)
                    if len(kl) < 2:
                        continue
                    last_closed, current = kl[1], kl[0]

                    # 1) ORB
                    orb = detect_orb(last_closed, current)
                    if orb:
                        st.orb_high, st.orb_low, st.orb_candle_time = orb

                    # 2) ENTRY
                    if st.pos is None and st.orb_high and st.orb_low:
                        if STATE.entry_mode == "close" and entry_close_rule(st.orb_high, last_closed):
                            entry_px = last_closed[4]
                            qty = POSITION_SIZE_USDT / entry_px if entry_px > 0 else 0.0
                            st.pos = Position("LONG", entry=entry_px, stop=st.orb_low, qty=qty)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY LONG (close) {sym} @ {entry_px:.4f}\n"
                                    f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | "
                                    f"SL={st.pos.stop:.4f} | QTY={qty:.6f}"
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
                                    f"SL={st.pos.stop:.4f} | QTY={qty:.6f}"
                                )

                    # 3) TRAILING SL + EXIT
                    if st.pos:
                        # flytta SL till förra stängda candlens low
                        new_sl = next_candle_stop(st.pos.stop, last_closed)
                        if new_sl > st.pos.stop:
                            st.pos.stop = new_sl
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id, f"🔧 SL flyttad {sym} -> {st.pos.stop:.4f}"
                                )

                        # exit beslut
                        exited = False
                        exit_px = None
                        reason = ""
                        if STATE.entry_mode == "close" and stop_hit_close(st.pos.stop, last_closed):
                            exit_px = last_closed[4]
                            exited = True
                            reason = "STOP (close)"
                        elif STATE.entry_mode == "tick" and stop_hit_tick(st.pos.stop, current):
                            exit_px = st.pos.stop
                            exited = True
                            reason = "STOP (tick)"

                        if exited and exit_px is not None:
                            gross, fees, net = trade_pnl_long(st.pos.entry, exit_px, st.pos.qty)
                            st.realized_pnl_gross += gross
                            st.realized_fees += fees
                            st.realized_pnl_net += net
                            st.trades.append(TradeRec(f"{sym} LONG • {reason}", gross, fees, net))
                            if len(st.trades) > 50:
                                st.trades = st.trades[-50:]
                            if STATE.chat_id:
                                sign = "✅" if net >= 0 else "❌"
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔴 EXIT {sym} @ {exit_px:.4f}\n"
                                    f"Gross: {gross:+.4f}  Fees: {fees:.4f}  Net: {net:+.4f} USDT {sign}"
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

def fmt_money(x: float) -> str:
    return f"{x:+.4f}"

async def send_status(chat_id: int):
    total_net = sum(s.realized_pnl_net for s in STATE.per_sym.values())
    total_gross = sum(s.realized_pnl_gross for s in STATE.per_sym.values())
    total_fees = sum(s.realized_fees for s in STATE.per_sym.values())

    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(
                f"{s}: LONG @ {st.pos.entry:.4f} | SL {st.pos.stop:.4f} | QTY {st.pos.qty:.6f}"
            )
    sym_pnls = ", ".join([
        f"{s}: Net {fmt_money(STATE.per_sym[s].realized_pnl_net)} "
        f"(Gross {fmt_money(STATE.per_sym[s].realized_pnl_gross)}, Fees {STATE.per_sym[s].realized_fees:.4f})"
        for s in STATE.symbols
    ])

    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Entry mode: {STATE.entry_mode}",
        f"Timeframe: {STATE.timeframe}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"ORB: {'PÅ' if STATE.orb_on else 'AV'} (första gröna efter röd)",
        f"PnL (TOTAL)  Net: {fmt_money(total_net)}   Gross: {fmt_money(total_gross)}   Fees: {total_fees:.4f}",
        f"PnL per symbol: {sym_pnls if sym_pnls else '-'}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
        f"Avgiftssats (mock): {FEE_RATE:.4%}",
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    total_net = sum(s.realized_pnl_net for s in STATE.per_sym.values())
    total_gross = sum(s.realized_pnl_gross for s in STATE.per_sym.values())
    total_fees = sum(s.realized_fees for s in STATE.per_sym.values())

    lines = [
        f"📈 PnL TOTAL",
        f"• Net:   {fmt_money(total_net)} USDT",
        f"• Gross: {fmt_money(total_gross)} USDT",
        f"• Fees:  {total_fees:.4f} USDT",
        "",
        "Per symbol:"
    ]
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        lines.append(
            f"• {s}: Net {fmt_money(ss.realized_pnl_net)} | Gross {fmt_money(ss.realized_pnl_gross)} | Fees {ss.realized_fees:.4f}"
        )

    # senaste trades (net visas)
    last: List[TradeRec] = []
    for s in STATE.symbols:
        last.extend(STATE.per_sym[s].trades[-5:])
    if last:
        lines.append("\nSenaste affärer:")
        for tr in last[-10:]:
            lines.append(
                f"  - {tr.label}: Net {fmt_money(tr.net)}  (Gross {fmt_money(tr.gross)}, Fees {tr.fees:.4f})"
            )

    lines.append(f"\nAvgiftssats (mock): {FEE_RATE:.4%}")
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! ORB-bot v36 + fees redo ✅", reply_markup=reply_kb())
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
            gross, fees, net = trade_pnl_long(st.pos.entry, exit_px, st.pos.qty)
            st.realized_pnl_gross += gross
            st.realized_fees += fees
            st.realized_pnl_net += net
            st.trades.append(TradeRec(f"{s} LONG (panic)", gross, fees, net))
            closed.append(f"{s} Net {net:+.4f}")
            st.pos = None
    msg = " | ".join(closed) if closed else "Inga positioner öppna."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl_net = 0.0
        STATE.per_sym[s].realized_pnl_gross = 0.0
        STATE.per_sym[s].realized_fees = 0.0
        STATE.per_sym[s].trades.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

# Registrera handlers (inga inline-knappar – endast reply-keyboard)
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
    total_net = sum(s.realized_pnl_net for s in STATE.per_sym.values())
    total_gross = sum(s.realized_pnl_gross for s in STATE.per_sym.values())
    total_fees = sum(s.realized_fees for s in STATE.per_sym.values())
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tf": STATE.timeframe,
        "mode": STATE.entry_mode,
        "fee_rate": FEE_RATE,
        "pnl_total_net": round(total_net, 6),
        "pnl_total_gross": round(total_gross, 6),
        "fees_total": round(total_fees, 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
