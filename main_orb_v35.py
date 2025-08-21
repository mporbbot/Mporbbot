# main_orb_v35.py
# Global kontrollpanel (reply-keyboard), ORB-trading (mock, KuCoin livepriser)
import os, asyncio, time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler

# ---------- ENV ----------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
OWNER_CHAT_ID = int((os.getenv("OWNER_CHAT_ID", "0") or "0").strip() or 0)
LIVE = int((os.getenv("LIVE", "0") or "0").strip() or 0)

SYMBOLS_ENV = os.getenv("SYMBOLS", "BTCUSDT")
SYMBOLS = [s.strip().upper() for s in SYMBOLS_ENV.split(",") if s.strip()]

if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN (TELEGRAM_BOT_TOKEN).")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (t.ex. https://din-app.onrender.com)")

# ---------- FastAPI & Telegram ----------
app = FastAPI(title="mporbbot ORB v35")
tg_app = Application.builder().token(BOT_TOKEN).build()

class _TGUpdate(BaseModel):
    update_id: Optional[int] = None
    message: Optional[dict] = None
    edited_message: Optional[dict] = None
    callback_query: Optional[dict] = None

# ---------- Hjälp ----------
KU_TF_MAP = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min"}

def to_kucoin_symbol(sym: str) -> str:
    s = sym.upper()
    if "-" in s: return s
    if s.endswith("USDT"): return s[:-4] + "-USDT"
    if s.endswith("USD"):  return s[:-3] + "-USD"
    if len(s) > 3: return s[:-3] + "-" + s[-3:]
    return s

async def ku_level1_price(session: httpx.AsyncClient, symbol: str) -> Optional[float]:
    try:
        r = await session.get("https://api.kucoin.com/api/v1/market/orderbook/level1",
                              params={"symbol": symbol}, timeout=10)
        d = r.json()
        if d.get("code") == "200000":
            return float(d["data"]["price"])
    except Exception:
        return None
    return None

async def ku_recent_candles(session: httpx.AsyncClient, symbol: str, tf: str, limit: int = 6) -> List[Tuple[int,float,float,float,float]]:
    t = KU_TF_MAP.get(tf, "3min")
    try:
        r = await session.get("https://api.kucoin.com/api/v1/market/candles",
                              params={"symbol": symbol, "type": t}, timeout=10)
        d = r.json()
        if d.get("code") == "200000":
            raw = d["data"][:limit]  # nyast först
            out = []
            for row in raw:
                ts = int(row[0]); o=float(row[1]); c=float(row[2]); h=float(row[3]); l=float(row[4])
                out.append((ts,o,h,l,c))
            out.sort(key=lambda x: x[0])  # äldst->nyast
            return out
    except Exception:
        pass
    return []

def is_green(c): return c[4] > c[1]  # close > open
def is_red(c):   return c[4] < c[1]

# ---------- State ----------
@dataclass
class Position:
    side: str
    entry: float
    size: float
    stop: float
    max_fav_price: float
    locked: bool = False

@dataclass
class SymState:
    symbol: str
    orb_high: Optional[float]=None
    orb_low: Optional[float]=None
    orb_ts: Optional[int]=None
    pos: Optional[Position]=None
    last_closed_ts: Optional[int]=None

@dataclass
class GlobalState:
    symbols: Dict[str, SymState] = field(default_factory=dict)
    engine_on: bool = False
    orb_enabled: bool = True
    entry_mode: str = "close"   # "close" / "tick"
    timeframe: str = "3m"
    lock_trigger_bp: float = 0.001  # 0.1 %
    qty_usd: float = 50.0
    live: bool = bool(LIVE)

STATE = GlobalState()
for s in SYMBOLS:
    STATE.symbols[s] = SymState(symbol=s)

# ---------- UI ----------
REPLY_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/panic")],
    ],
    resize_keyboard=True
)

def tf_markup():
    rows = [
        [InlineKeyboardButton("1m", callback_data="TF:1m"),
         InlineKeyboardButton("3m ✅" if STATE.timeframe=="3m" else "3m", callback_data="TF:3m"),
         InlineKeyboardButton("5m", callback_data="TF:5m")],
        [InlineKeyboardButton("15m", callback_data="TF:15m"),
         InlineKeyboardButton("30m", callback_data="TF:30m")]
    ]
    return InlineKeyboardMarkup(rows)

def status_text() -> str:
    lines = [f"📊 Status",
             f"• Engine: {'AKTIV' if STATE.engine_on else 'AV'}",
             f"• Entry mode: {STATE.entry_mode}",
             f"• Timeframe: {STATE.timeframe}",
             f"• ORB: {'PÅ' if STATE.orb_enabled else 'AV'}",
             f"• Symbols: {', '.join(STATE.symbols.keys())}"]
    for st in STATE.symbols.values():
        pos = "Ingen"
        if st.pos:
            pos = f"{st.pos.side.upper()} @ {st.pos.entry:.4f} | stop {st.pos.stop:.4f} | locked={st.pos.locked}"
        orb = "saknas" if st.orb_high is None else f"H:{st.orb_high:.4f} L:{st.orb_low:.4f}"
        lines.append(f"\n[{st.symbol}] ORB {orb} | Pos {pos}")
    return "\n".join(lines)

async def safe_send(text: str):
    if OWNER_CHAT_ID:
        try:
            await tg_app.bot.send_message(OWNER_CHAT_ID, text, reply_markup=REPLY_KB)
        except Exception:
            pass

# ---------- Telegram handlers ----------
async def cmd_start(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    await update.message.reply_text("Hej! ORB v35 📈\n\nAnvänd knapparna nedan.",
                                    reply_markup=REPLY_KB)

async def cmd_status(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    await update.message.reply_text(status_text(), reply_markup=REPLY_KB)

async def cmd_engine_on(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    STATE.engine_on = True
    await update.message.reply_text("✅ Engine är nu AKTIV.", reply_markup=REPLY_KB)

async def cmd_engine_off(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    STATE.engine_on = False
    # stäng alla mock-positioner
    for st in STATE.symbols.values(): st.pos = None
    await update.message.reply_text("🛑 Engine stoppad. Alla mock-positioner stängda.", reply_markup=REPLY_KB)

async def cmd_entrymode(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    STATE.entry_mode = "tick" if STATE.entry_mode == "close" else "close"
    await update.message.reply_text(f"Entry mode satt till: {STATE.entry_mode}", reply_markup=REPLY_KB)

async def cmd_timeframe(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    await update.message.reply_text("Välj timeframe:", reply_markup=tf_markup())

async def cmd_orb_on(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    STATE.orb_enabled = True
    await update.message.reply_text("🟩 ORB: PÅ.", reply_markup=REPLY_KB)

async def cmd_orb_off(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    STATE.orb_enabled = False
    await update.message.reply_text("⬛️ ORB: AV.", reply_markup=REPLY_KB)

async def cmd_panic(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    for st in STATE.symbols.values(): st.pos = None
    STATE.engine_on = False
    await update.message.reply_text("🆘 PANIC: allt stängt och engine OFF.", reply_markup=REPLY_KB)

async def cbk(update: Update, ctx):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID: return
    q = update.callback_query
    if not q or not q.data: return
    if q.data.startswith("TF:"):
        tf = q.data.split(":")[1]
        STATE.timeframe = tf
        # nollställ ORB per symbol när TF byts
        for st in STATE.symbols.values():
            st.orb_high = st.orb_low = st.orb_ts = st.last_closed_ts = None
        await q.answer(f"TF={tf}")
        await q.message.reply_text(f"⏱ Timeframe satt till {tf}.", reply_markup=REPLY_KB)

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CallbackQueryHandler(cbk))

# ---------- ORB/Trading ----------
async def process_symbol(session: httpx.AsyncClient, st: SymState):
    if not STATE.engine_on or not STATE.orb_enabled:
        return
    sym_ku = to_kucoin_symbol(st.symbol)
    candles = await ku_recent_candles(session, sym_ku, STATE.timeframe, limit=6)
    if len(candles) < 3:
        return

    closed_prev = candles[-2]
    closed_last = candles[-1]

    # Sätt ORB om saknas
    if st.orb_high is None:
        for i in range(1, len(candles)):
            a,b = candles[i-1], candles[i]
            if is_red(a) and is_green(b):
                st.orb_high = b[2]; st.orb_low = b[3]; st.orb_ts = b[0]
                await safe_send(f"🧭 ORB {st.symbol} H:{st.orb_high:.4f} L:{st.orb_low:.4f}")
                break

    # Trailing för long
    if st.pos:
        price = await ku_level1_price(session, sym_ku)
        if price:
            st.pos.max_fav_price = max(st.pos.max_fav_price, price)
            up = st.pos.max_fav_price / st.pos.entry - 1
            if (not st.pos.locked) and up >= STATE.lock_trigger_bp:
                st.pos.locked = True
        # flytta vid ny candle
        if st.pos.locked and (st.last_closed_ts is None or closed_last[0] != st.last_closed_ts):
            prev_low = closed_prev[3]
            if prev_low > st.pos.stop:
                st.pos.stop = prev_low
                await safe_send(f"🔒 Trail stop {st.symbol} → {st.pos.stop:.4f}")
            st.last_closed_ts = closed_last[0]

        # Exit på tick mot stop
        if price and price <= st.pos.stop:
            pnl = (price - st.pos.entry) / st.pos.entry * 100
            await safe_send(f"🔻 EXIT {st.symbol} @ {price:.4f} | PnL {pnl:.2f}%")
            st.pos = None
            return

    # Entry
    if st.pos is None and st.orb_high is not None:
        if STATE.entry_mode == "close":
            if closed_last[4] > st.orb_high:
                entry = closed_last[4]
                size = max(STATE.qty_usd/entry, 0.0001)
                st.pos = Position("long", entry, size, st.orb_low, entry)
                st.last_closed_ts = closed_last[0]
                await safe_send(f"🟢 ENTRY (close) {st.symbol} @ {entry:.4f} | stop {st.pos.stop:.4f}")
        else:
            price = await ku_level1_price(session, sym_ku)
            if price and price > st.orb_high:
                entry = price
                size = max(STATE.qty_usd/entry, 0.0001)
                st.pos = Position("long", entry, size, st.orb_low, entry)
                st.last_closed_ts = closed_last[0]
                await safe_send(f"🟢 ENTRY (tick) {st.symbol} @ {entry:.4f} | stop {st.pos.stop:.4f}")

# ---------- Loop ----------
async def trading_loop():
    await asyncio.sleep(1)
    async with httpx.AsyncClient(timeout=10) as session:
        while True:
            try:
                await asyncio.gather(*(process_symbol(session, st) for st in STATE.symbols.values()))
            except Exception:
                pass
            await asyncio.sleep(2.5)

# ---------- FastAPI ----------
@app.get("/")
async def root(): return {"ok": True, "service": "ORB v35"}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != BOT_TOKEN: return Response(status_code=403)
    data = await request.json()
    await tg_app.process_update(Update.de_json(data, tg_app.bot))
    return Response(status_code=200)

# ---------- Lifecycle ----------
async def set_bot_commands():
    try:
        await tg_app.bot.set_my_commands([
            ("start","Visa knappar"),
            ("status","Visa status"),
            ("engine_on","Starta motorn"),
            ("engine_off","Stoppa motorn"),
            ("entrymode","Växla entry close/tick"),
            ("timeframe","Välj timeframe"),
            ("orb_on","Slå på ORB"),
            ("orb_off","Stäng av ORB"),
            ("panic","Stäng allt + motor av"),
        ])
    except Exception:
        pass

@app.on_event("startup")
async def _startup():
    await tg_app.initialize(); await tg_app.start(); await set_bot_commands()
    await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    await safe_send("🔧 ORB v35 startad. Använd knapparna. Engine är OFF.")
    asyncio.create_task(trading_loop())

@app.on_event("shutdown")
async def _shutdown():
    try: await tg_app.bot.delete_webhook()
    except Exception: pass
    await tg_app.stop(); await tg_app.shutdown()
