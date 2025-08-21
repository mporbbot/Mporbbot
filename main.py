# main.py
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

# ====== ENV ======
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")  # 1m,3m,5m,15m,30m,1h
ENTRYMODE = os.getenv("ENTRY_MODE", "close")  # close|tick

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"

TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ====== STATE ======
@dataclass
class Position:
    side: str  # "LONG"
    entry: float
    stop: float
    locked: bool = False

@dataclass
class SymState:
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_candle_time: Optional[int] = None  # epoch ms for ORB candle start
    pos: Optional[Position] = None

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

# ====== UTIL ======
def pretty_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

async def get_klines(symbol: str, tf: str, limit: int = 3) -> List[Tuple[int,float,float,float,float]]:
    """
    Returnerar lista av tuples: (startTimeMs, open, high, low, close) – senast först.
    """
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # KuCoin ger newest first
    out = []
    for row in data[:limit]:
        # KuCoin format: [time, open, close, high, low, volume, turnover]
        t_s = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_s, o, h, l, c))
    return out

def is_green(o,h,l,c): return c > o
def is_red(o,h,l,c):   return c < o

# ====== CORE ORB LOGIK ======
def detect_orb(prev: Tuple[int,float,float,float,float],
               cur: Tuple[int,float,float,float,float]) -> Optional[Tuple[float,float,int]]:
    """
    ORB = första gröna candle efter en röd.
    Returnerar (orb_high, orb_low, time_ms) om cur är den första gröna efter en röd.
    """
    _, po, ph, pl, pc = prev
    _, co, ch, cl, cc = cur
    if is_red(po,ph,pl,pc) and is_green(co,ch,cl,cc):
        return (ch, cl, cur[0])
    return None

def entry_close_rule(orb_high: float, closed_candle: Tuple[int,float,float,float,float]) -> bool:
    # stänger över ORB-high
    _, o, h, l, c = closed_candle
    return c > orb_high

def entry_tick_rule(orb_high: float, candle: Tuple[int,float,float,float,float]) -> bool:
    # når över ORB-high
    _, o, h, l, c = candle
    return h >= orb_high

def stop_hit_close(stop: float, closed_candle: Tuple[int,float,float,float,float]) -> bool:
    _, o,h,l,c = closed_candle
    return c < stop

def stop_hit_tick(stop: float, candle: Tuple[int,float,float,float,float]) -> bool:
    _, o,h,l,c = candle
    return l <= stop

def next_candle_stop(cur_stop: float, prev_closed_candle: Tuple[int,float,float,float,float]) -> float:
    """
    Flytta SL till föregående stängda candlens low, men aldrig neråt.
    """
    _, o,h,l,c = prev_closed_candle
    return max(cur_stop, l)

# ====== ENGINE LOOP ======
async def engine_loop(app: Application):
    await asyncio.sleep(2)  # liten fördröjning tills webhook är klar
    while True:
        try:
            if STATE.engine_on and STATE.orb_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    kl = await get_klines(sym, STATE.timeframe, limit=3)
                    if len(kl) < 2:
                        continue
                    last_closed, current = kl[1], kl[0]  # [0] nyaste (kan vara under bildning), [1] är stängd
                    # 1) Upptäck ORB
                    orb = detect_orb(last_closed, current)  # OBS: current kan bildas nu; ORB definieras när den blir grön efter röd
                    if orb:
                        st.orb_high, st.orb_low, st.orb_candle_time = orb

                    # 2) Entry
                    if st.pos is None and st.orb_high and st.orb_low:
                        if STATE.entry_mode == "close" and entry_close_rule(st.orb_high, last_closed):
                            st.pos = Position("LONG", entry=last_closed[4], stop=st.orb_low)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🟢 ENTRY LONG (close) {sym} @ {st.pos.entry:.4f}\n"
                                    f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | SL={st.pos.stop:.4f}")
                        elif STATE.entry_mode == "tick" and entry_tick_rule(st.orb_high, current):
                            # approx: fyll på ORB-high
                            st.pos = Position("LONG", entry=st.orb_high, stop=st.orb_low)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🟢 ENTRY LONG (tick) {sym} @ {st.pos.entry:.4f}\n"
                                    f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | SL={st.pos.stop:.4f}")

                    # 3) Hantera stop/exit + flytta SL till föregående candles low
                    if st.pos:
                        # flytta SL upp till förra stängda candlens low
                        new_sl = next_candle_stop(st.pos.stop, last_closed)
                        if new_sl > st.pos.stop:
                            st.pos.stop = new_sl
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔧 SL flyttad {sym} -> {st.pos.stop:.4f}")

                        # exit-regler
                        if STATE.entry_mode == "close" and stop_hit_close(st.pos.stop, last_closed):
                            exit_px = last_closed[4]
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔴 EXIT (close) {sym} @ {exit_px:.4f} | SL {st.pos.stop:.4f}")
                            st.pos = None
                            # behåll ORB tills en ny definieras
                        elif STATE.entry_mode == "tick" and stop_hit_tick(st.pos.stop, current):
                            exit_px = st.pos.stop
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔴 EXIT (tick) {sym} @ {exit_px:.4f} | SL {st.pos.stop:.4f}")
                            st.pos = None
            await asyncio.sleep(2)  # polling-tempo
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ====== TELEGRAM ======
tg_app = Application.builder().token(BOT_TOKEN).build()

async def send_status(chat_id: int):
    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Entry mode: {STATE.entry_mode}",
        f"Timeframe: {STATE.timeframe}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"ORB: {'PÅ' if STATE.orb_on else 'AV'} (din ORB: första gröna efter röd)",
    ]
    # positioner
    poss = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            poss.append(f"{s}: LONG @ {st.pos.entry:.4f} | SL {st.pos.stop:.4f}")
    lines.append("Positioner: " + (", ".join(poss) if poss else "inga"))
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! ORB-bot redo ✅", reply_markup=reply_kb())
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
            closed.append(f"{s} @ {st.pos.stop:.4f}")
            st.pos = None
    txt = "Panic close utfört." if closed else "Inga positioner öppna."
    await tg_app.bot.send_message(STATE.chat_id, txt, reply_markup=reply_kb())

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("panic", cmd_panic))

# ====== FASTAPI (webhook) ======
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None  # vi bryr oss inte – PTB hanterar

@app.on_event("startup")
async def on_startup():
    # webhook
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    # starta PTB + motor
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    return {"ok": True, "engine_on": STATE.engine_on, "tf": STATE.timeframe, "mode": STATE.entry_mode}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
