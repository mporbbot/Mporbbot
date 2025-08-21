import os
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from starlette.middleware.cors import CORSMiddleware

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes,
)

# ========= Miljövariabler =========
BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN".upper())
if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN (eller TELEGRAM_BOT_TOKEN) i environment.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")  # 1m,3m,5m,15m,30m,1h osv (KuCoin: 1min,3min,5min,15min,30min,1hour)
ENTRY_DEFAULT = os.getenv("ENTRY_DEFAULT", "close").lower()  # 'close' (standard) eller 'tick'
MODE_DEFAULT = os.getenv("MODE", "MOCK").upper()  # MOCK eller LIVE

# ORB-inställningar
ORB_PERIOD = int(os.getenv("ORB_PERIOD", "20"))  # antal candles som bygger lådan innan breakouts tillåts
TRAIL_TRIGGER_PCT = float(os.getenv("TRAIL_TRIGGER_PCT", "0.30"))  # låsning startar efter X% i vinst
TRAIL_MIN_LOCK_PCT = float(os.getenv("TRAIL_MIN_LOCK_PCT", "0.10"))  # minsta låst vinst efter trigger
TRAIL_OFFSET_PCT = float(os.getenv("TRAIL_OFFSET_PCT", "0.15"))  # hur långt efter priset stoppen får släpa

# ========= Hjälp: KuCoin TF mapping =========
def ku_tf(tf: str) -> str:
    tf = tf.lower().strip()
    return {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "2h": "2hour",
        "4h": "4hour",
        "8h": "8hour",
        "1d": "1day",
    }.get(tf, "1min")

# ========= State =========
class Position:
    def __init__(self, side: str, entry: float, stop: float):
        self.side = side  # "LONG" eller "SHORT"
        self.entry = entry
        self.stop = stop
        self.locked = False  # om vinsten är låst (trailing lås)

class SymState:
    def __init__(self):
        self.orb_on: bool = True
        self.entry_mode: str = ENTRY_DEFAULT  # 'close' eller 'tick' (vi använder 'close' här)
        self.timeframe: str = TIMEFRAME
        self.engine_on: bool = False
        self.mode: str = MODE_DEFAULT  # MOCK / LIVE

        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.orb_ready: bool = False
        self.candle_count: int = 0

        self.position: Optional[Position] = None
        self.last_closed_ts: Optional[int] = None  # unix ms sista stängda candlen

# Globalt
symbols: List[str] = [s.strip().upper().replace("_", "-") for s in SYMBOLS.split(",") if s.strip()]
state: Dict[str, SymState] = {s: SymState() for s in symbols}
chat_ids: set[int] = set()  # för push-meddelanden

# ========= Telegram setup =========
tg_app: Application = ApplicationBuilder().token(BOT_TOKEN).build()

def kb_main() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("Status", callback_data="cmd:status")],
        [
            InlineKeyboardButton("Engine ON", callback_data="cmd:engine_on"),
            InlineKeyboardButton("Engine OFF", callback_data="cmd:engine_off"),
        ],
        [
            InlineKeyboardButton("Entry: close", callback_data="cmd:entry_close"),
            InlineKeyboardButton("Entry: tick", callback_data="cmd:entry_tick"),
        ],
        [
            InlineKeyboardButton("TF: 1m", callback_data="cmd:tf:1m"),
            InlineKeyboardButton("TF: 3m", callback_data="cmd:tf:3m"),
            InlineKeyboardButton("TF: 5m", callback_data="cmd:tf:5m"),
        ],
        [
            InlineKeyboardButton("TF: 15m", callback_data="cmd:tf:15m"),
            InlineKeyboardButton("TF: 30m", callback_data="cmd:tf:30m"),
            InlineKeyboardButton("TF: 1h", callback_data="cmd:tf:1h"),
        ],
        [
            InlineKeyboardButton("ORB ON", callback_data="cmd:orb_on"),
            InlineKeyboardButton("ORB OFF", callback_data="cmd:orb_off"),
        ],
        [InlineKeyboardButton("PANIC (stäng allt)", callback_data="cmd:panic")],
    ]
    return InlineKeyboardMarkup(rows)

async def send(chat_id: int, text: str) -> None:
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
    except Exception:
        pass

def fmt_status() -> str:
    lines = ["📊 Status"]
    # Delad setup – vi visar gemensamma settings (entrymode/tf/engine) om de är likadana för alla
    em = {st.entry_mode for st in state.values()}
    tf = {st.timeframe for st in state.values()}
    en = {st.engine_on for st in state.values()}
    md = {st.mode for st in state.values()}
    orb = {st.orb_on for st in state.values()}

    lines.append(f"• Engine: {'AKTIV' if True in en else 'AV'}")
    lines.append(f"• Entry mode: {list(em)[0] if len(em)==1 else 'blandat'}")
    lines.append(f"• Timeframe: {list(tf)[0] if len(tf)==1 else 'blandat'}")
    lines.append(f"• Mode: {list(md)[0] if len(md)==1 else 'blandat'}")
    lines.append(f"• ORB: {'PÅ' if True in orb else 'AV'}")
    lines.append("• Symbols: " + ", ".join(symbols))
    return "\n".join(lines)

# ========= KuCoin data =========
async def ku_get_klines(symbol: str, tf: str, limit: int = 100) -> List[Tuple[int, float, float, float, float]]:
    """
    Returnerar lista med (ts_ms, open, high, low, close) – stängda candles (nyaste sist).
    KuCoin svarar i ordning nyast->äldst; vi vänder till äldst->nyast.
    """
    ktype = ku_tf(tf)
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"symbol": symbol, "type": ktype}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # lista av [time, open, close, high, low, volume, turnover]
    out = []
    for row in reversed(data[-limit:]):
        ts_sec = int(row[0])
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        ts_ms = ts_sec * 1000
        out.append((ts_ms, o, h, l, c))
    return out

# ========= ORB & tradinglogik =========
def _pcent(a: float, b: float) -> float:
    return (a - b) / b * 100.0

async def maybe_enter(symbol: str, st: SymState, last: Tuple[int, float, float, float, float], chat_id: Optional[int]):
    ts, o, h, l, c = last
    if not st.orb_ready or not st.orb_on or not st.engine_on or st.position:
        return
    if st.entry_mode == "tick":
        # (Vi använder close ändå – men entry_mode finns kvar för framtida utökning)
        pass

    # Breakout vid candle close
    if c > (st.orb_high or 0):
        # LONG
        st.position = Position("LONG", entry=c, stop=st.orb_low)
        if chat_id:
            await send(chat_id, f"🟢 ENTRY LONG (close) {symbol} @ {c:.4f}\n"
                                f"ORB: H={st.orb_high:.4f} L={st.orb_low:.4f} | stop {st.orb_low:.4f}")
    elif c < (st.orb_low or 0):
        # SHORT (om du vill – men du sa att vi börjar köra på uppgång, så hoppa över short)
        pass

async def maybe_exit_or_trail(symbol: str, st: SymState, last: Tuple[int, float, float, float, float], chat_id: Optional[int]):
    if not st.position:
        return
    ts, o, h, l, c = last
    pos = st.position

    # Stop hit?
    if pos.side == "LONG":
        if c <= pos.stop:
            # EXIT
            pnl = _pcent(c, pos.entry)
            st.position = None
            if chat_id:
                await send(chat_id, f"🛑 EXIT {symbol} @ {c:.4f} (Stop hit)\nPNL: {pnl:+.2f}%")
            return

        # Trailing – starta låsning efter trigger
        gain_pct = _pcent(c, pos.entry)
        if gain_pct >= TRAIL_TRIGGER_PCT * 100:
            # lås vinsten till minst TRAIL_MIN_LOCK_PCT
            lock_price = pos.entry * (1.0 + TRAIL_MIN_LOCK_PCT)
            trail_stop = c * (1.0 - TRAIL_OFFSET_PCT)
            new_stop = max(pos.stop, lock_price, trail_stop)
            if new_stop > pos.stop:
                pos.stop = new_stop
                pos.locked = True
                if chat_id:
                    await send(chat_id, f"🔧 {symbol} Trail stop → {pos.stop:.4f}")

async def build_orb_if_needed(symbol: str, st: SymState, candles: List[Tuple[int, float, float, float, float]]):
    if st.orb_ready:
        return
    # Bygg lådan av de första ORB_PERIOD stängda candles
    if len(candles) >= ORB_PERIOD:
        box = candles[-ORB_PERIOD:]
        st.orb_high = max(x[2] for x in box)  # high
        st.orb_low = min(x[3] for x in box)   # low
        st.orb_ready = True

async def engine_loop():
    await tg_app.wait_until_ready()
    # välj första kända chat_id (om något finns)
    chat_id = next(iter(chat_ids), None)

    while True:
        # loopa snällt
        await asyncio.sleep(5)
        for sym in symbols:
            st = state[sym]
            if not st.engine_on or not st.orb_on:
                continue
            try:
                kl = await ku_get_klines(sym, st.timeframe, limit=max(ORB_PERIOD + 2, 50))
                if not kl:
                    continue
                last_closed = kl[-1]  # vi behandlar senaste stängda
                ts = last_closed[0]
                if st.last_closed_ts == ts:
                    continue  # redan hanterad
                st.last_closed_ts = ts

                # Bygg ORB första gången
                await build_orb_if_needed(sym, st, kl)
                # Uppdatera trail/exit – körs alltid om position finns
                await maybe_exit_or_trail(sym, st, last_closed, chat_id)
                # Försök entry på close
                await maybe_enter(sym, st, last_closed, chat_id)

            except Exception as e:
                if chat_id:
                    await send(chat_id, f"[{sym}] Fel vid klines (KuCoin): {e}")

# ========= Telegram kommandon =========
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        chat_ids.add(update.effective_chat.id)
    text = (f"✅ Startad på {WEBHOOK_BASE}\n"
            f"Mode: {MODE_DEFAULT}\n"
            f"TF: {TIMEFRAME}\n"
            f"Entry: {ENTRY_DEFAULT}\n"
            f"Symbols: {', '.join(symbols)}")
    await update.message.reply_text(text, reply_markup=kb_main())

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        chat_ids.add(update.effective_chat.id)
    await update.message.reply_text(fmt_status(), reply_markup=kb_main())

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for st in state.values():
        st.engine_on = True
    await update.message.reply_text("✅ Engine är nu AKTIV.", reply_markup=kb_main())

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for st in state.values():
        st.engine_on = False
    await update.message.reply_text("⛔️ Engine stoppad.", reply_markup=kb_main())

async def cmd_entrymode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for st in state.values():
        st.entry_mode = "close" if st.entry_mode != "close" else "tick"
    await update.message.reply_text(f"Entry mode satt till: {list({s.entry_mode for s in state.values()})}",
                                    reply_markup=kb_main())

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # enkel cykel 1m->3m->5m->15m->30m->1h->1m
    order = ["1m", "3m", "5m", "15m", "30m", "1h"]
    current = list({s.timeframe for s in state.values()})[0]
    nxt = order[(order.index(current) + 1) % len(order)] if current in order else "1m"
    for st in state.values():
        st.timeframe = nxt
        st.orb_ready = False
        st.orb_high = st.orb_low = None
        st.last_closed_ts = None
    await update.message.reply_text(f"⏱ Timeframe satt till: {nxt} (ORB nollställd)", reply_markup=kb_main())

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for st in state.values():
        st.orb_on = True
    await update.message.reply_text("🟩 ORB: PÅ.", reply_markup=kb_main())

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for st in state.values():
        st.orb_on = False
    await update.message.reply_text("🟥 ORB: AV.", reply_markup=kb_main())

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # stäng lokala positioner (mock). I live skulle vi lägga marknadsorder här.
    for st in state.values():
        st.position = None
    await update.message.reply_text("⚠️ PANIC: Alla mock-positioner rensade.", reply_markup=kb_main())

# knapp-callbacks (enkelt: mappa till kommandon)
from telegram.ext import CallbackQueryHandler

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    data = q.data or ""
    await q.answer()
    if data == "cmd:status":
        await q.edit_message_text(fmt_status(), reply_markup=kb_main())
    elif data == "cmd:engine_on":
        for st in state.values():
            st.engine_on = True
        await q.edit_message_text("✅ Engine är nu AKTIV.", reply_markup=kb_main())
    elif data == "cmd:engine_off":
        for st in state.values():
            st.engine_on = False
        await q.edit_message_text("⛔️ Engine stoppad.", reply_markup=kb_main())
    elif data == "cmd:entry_close":
        for st in state.values():
            st.entry_mode = "close"
        await q.edit_message_text("Entry mode satt till: close", reply_markup=kb_main())
    elif data == "cmd:entry_tick":
        for st in state.values():
            st.entry_mode = "tick"
        await q.edit_message_text("Entry mode satt till: tick", reply_markup=kb_main())
    elif data.startswith("cmd:tf:"):
        tf = data.split(":", 2)[2]
        for st in state.values():
            st.timeframe = tf
            st.orb_ready = False
            st.orb_high = st.orb_low = None
            st.last_closed_ts = None
        await q.edit_message_text(f"⏱ Timeframe satt till: {tf} (ORB nollställd)", reply_markup=kb_main())
    elif data == "cmd:orb_on":
        for st in state.values():
            st.orb_on = True
        await q.edit_message_text("🟩 ORB: PÅ.", reply_markup=kb_main())
    elif data == "cmd:orb_off":
        for st in state.values():
            st.orb_on = False
        await q.edit_message_text("🟥 ORB: AV.", reply_markup=kb_main())
    elif data == "cmd:panic":
        for st in state.values():
            st.position = None
        await q.edit_message_text("⚠️ PANIC: Alla mock-positioner rensade.", reply_markup=kb_main())

# ========= FastAPI (webhook) =========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
async def on_startup():
    # Telegram handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
    tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
    tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
    tg_app.add_handler(CommandHandler("panic", cmd_panic))
    tg_app.add_handler(CallbackQueryHandler(on_button))

    await tg_app.initialize()
    await tg_app.bot.set_webhook(f"{WEBHOOK_BASE.rstrip('/')}/webhook/{BOT_TOKEN}")
    asyncio.create_task(tg_app.start())
    # starta motor
    asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.bot.delete_webhook()
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return "OK"
