# main_orb_v34.py
import os, asyncio, time, json, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from telegram import (
    Update, BotCommand, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup,
    InlineKeyboardButton
)
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
)

# -------------------- ENV --------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "5397586616") or "5397586616")

SYMBOLS = [s.strip().replace("-", "").upper() for s in os.getenv(
    "SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "1m").lower()   # bara 1m används praktiskt nu
MOCK_REALTIME = int(os.getenv("MOCK_REALTIME", "1") or "1")

FEE_PCT = float(os.getenv("FEE_PCT", "0.001") or "0.001")          # 0.1 %
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0002") or "0.0002") # 0.02 %
TRAIL_MIN_LOCK_PCT = float(os.getenv("TRAIL_MIN_LOCK_PCT", "0.0015") or "0.0015")
TRAIL_BUFFER_PCT   = float(os.getenv("TRAIL_BUFFER_PCT",   "0.0008") or "0.0008")

# -------------------- GLOBAL STATE --------------------
ENGINE_ON = False
ENTRY_MODE = "close"   # "tick" / "close"
ORB_ENABLED = True     # separat switch om du vill stänga av box-logik
LIVE_MODE = False      # mock = False live = True (orders skickas ej i denna fil)

KUCOIN_REST = "https://api.kucoin.com"

@dataclass
class Candle:
    ts: int
    o: float
    h: float
    l: float
    c: float
    closed: bool = True

@dataclass
class Position:
    symbol: str
    side: str          # long/short
    qty: float
    entry_price: float
    stop_price: float
    trail_high: float
    pnl_usdt: float = 0.0
    opened_ts: int = field(default_factory=lambda: int(time.time()*1000))

# Pris-cache: last/bid/ask/ts
TICKS: Dict[str, Dict[str, float]] = {s: {"last": None, "bid": None, "ask": None, "ts": 0} for s in SYMBOLS}
# Senaste ”aktiva” candle per symbol (1m)
CANDLES: Dict[str, Candle] = {}
# ORB-box per symbol (beräknad från första minuten efter engine_on)
ORB_BOX: Dict[str, Dict[str, float]] = {}  # {symbol: {"high": x, "low": y, "ready": bool}}
# Öppen position per symbol (max 1 i denna enkla version)
POSITIONS: Dict[str, Optional[Position]] = {s: None for s in SYMBOLS}
# Ackumulerad PnL mock
PNL_ACCUM: float = 0.0

# -------------------- UTIL --------------------
def fmt_usd(x: float) -> str:
    return f"{x:.2f} USDT"

def now_ms() -> int:
    return int(time.time()*1000)

def trail_update(side: str, entry: float, prev_stop: float, c: Candle,
                 min_lock: float, buffer_pct: float) -> Tuple[float, bool, bool]:
    lock_hit = False
    moved = False
    new_stop = prev_stop

    if side == "long":
        peak = max(c.h, entry)
        wanted = max(entry*(1+min_lock), peak*(1-buffer_pct))
        if wanted > prev_stop:
            new_stop, moved = wanted, True
        if new_stop >= entry*(1+min_lock) - 1e-12:
            lock_hit = True
    else:
        trough = min(c.l, entry)
        wanted = min(entry*(1-min_lock), trough*(1+buffer_pct))
        if wanted < prev_stop:
            new_stop, moved = wanted, True
        if new_stop <= entry*(1-min_lock) + 1e-12:
            lock_hit = True
    return new_stop, moved, lock_hit

def stop_hit(side: str, price: float, stop: float) -> bool:
    return (side == "long" and price <= stop) or (side == "short" and price >= stop)

def mock_fill(side: str, tick: Dict[str, float]) -> Tuple[float, float]:
    # realistisk fyll på ask/bid + slippage + fee
    last = tick["last"]
    bid = tick["bid"] or last
    ask = tick["ask"] or last
    px = ask if side == "long" else bid
    px *= (1 + SLIPPAGE_PCT) if side == "long" else (1 - SLIPPAGE_PCT)
    fee = px * FEE_PCT
    return px, fee

def symbol_to_kucoin(s: str) -> str:
    # BTCUSDT -> BTC-USDT
    if "-" in s: return s
    base = s[:-4]
    quote = s[-4:]
    return f"{base}-{quote}"

# -------------------- PRICE POLLER --------------------
async def poll_prices():
    # Enkel, robust polling var ~1s
    async with httpx.AsyncClient(timeout=8.0) as client:
        while True:
            try:
                for s in SYMBOLS:
                    ks = symbol_to_kucoin(s)
                    r = await client.get(f"{KUCOIN_REST}/api/v1/market/orderbook/level1", params={"symbol": ks})
                    d = r.json()["data"]
                    TICKS[s]["last"] = float(d["price"])
                    TICKS[s]["bid"]  = float(d["bestBid"])
                    TICKS[s]["ask"]  = float(d["bestAsk"])
                    TICKS[s]["ts"]   = int(d["time"])
            except Exception:
                pass
            await asyncio.sleep(1.0)

# -------------------- KLINE FETCH --------------------
async def fetch_last_closed_1m_candle(client: httpx.AsyncClient, s: str) -> Optional[Candle]:
    ks = symbol_to_kucoin(s)
    # KuCoin klines: /api/v1/market/candles?symbol=BTC-USDT&type=1min
    try:
        r = await client.get(f"{KUCOIN_REST}/api/v1/market/candles", params={"symbol": ks, "type": "1min", "limit": 2})
        arr = r.json()["data"]
        # data kommer som [ [time, open, close, high, low, volume, turnover], ...] i nyast-först
        latest = arr[0]
        ts = int(float(latest[0]))*1000
        o = float(latest[1]); c = float(latest[2]); h = float(latest[3]); l = float(latest[4])
        return Candle(ts=ts, o=o, h=h, l=l, c=c, closed=True)
    except Exception:
        return None

# -------------------- TRADING LOOP (MOCK) --------------------
async def trading_loop(app: Application):
    global PNL_ACCUM
    async with httpx.AsyncClient(timeout=8.0) as client:
        # initial candle load
        for s in SYMBOLS:
            c = await fetch_last_closed_1m_candle(client, s)
            if c: CANDLES[s] = c

        last_minute = None
        while True:
            await asyncio.sleep(1.0)
            if not ENGINE_ON:
                continue

            # kolla om ny candle stängt
            for s in SYMBOLS:
                c = await fetch_last_closed_1m_candle(client, s)
                if not c: 
                    continue
                prev = CANDLES.get(s)
                if not prev or c.ts != prev.ts:
                    # ny stängd candle
                    CANDLES[s] = c
                    # uppdatera ORB-box om aktiv
                    if ORB_ENABLED:
                        box = ORB_BOX.get(s)
                        if not box:
                            # första candle efter start => initiera box: använd denna candle som “box”
                            ORB_BOX[s] = {"high": c.h, "low": c.l, "ready": True}
                            await safe_send(app, OWNER_CHAT_ID,
                                f"🟩 {s} ORB init: High={c.h:.4f} Low={c.l:.4f}")
                        else:
                            # låt high/low expandera under dagen om du vill (enkel variant: lås första)
                            pass

                    # entry på CLOSE?
                    if ENTRY_MODE == "close" and ORB_ENABLED:
                        tick = TICKS[s]
                        if tick["last"] is None: 
                            continue
                        box = ORB_BOX.get(s)
                        if not box or not box["ready"]: 
                            continue
                        # brytningar med stängd candle:
                        if c.c > box["high"] and POSITIONS[s] is None:
                            # LONG
                            px, fee = mock_fill("long", tick)
                            stop = box["low"]
                            POSITIONS[s] = Position(symbol=s, side="long", qty=1.0, entry_price=px,
                                                    stop_price=stop, trail_high=c.h)
                            await safe_send(app, OWNER_CHAT_ID,
                                f"✅ {s} ENTRY (close) LONG @ {px:.4f} | stop start = {stop:.4f}")
                        elif c.c < box["low"] and POSITIONS[s] is None:
                            # SHORT
                            px, fee = mock_fill("short", tick)
                            stop = box["high"]
                            POSITIONS[s] = Position(symbol=s, side="short", qty=1.0, entry_price=px,
                                                    stop_price=stop, trail_high=c.l)
                            await safe_send(app, OWNER_CHAT_ID,
                                f"✅ {s} ENTRY (close) SHORT @ {px:.4f} | stop start = {stop:.4f}")

                # trailing + tick-stop varje sekund
                pos = POSITIONS.get(s)
                if pos and TICKS[s]["last"] is not None and s in CANDLES:
                    price = TICKS[s]["last"]
                    cnow = CANDLES[s]
                    new_stop, moved, locked = trail_update(
                        side=pos.side, entry=pos.entry_price, prev_stop=pos.stop_price,
                        c=cnow, min_lock=TRAIL_MIN_LOCK_PCT, buffer_pct=TRAIL_BUFFER_PCT
                    )
                    if moved:
                        pos.stop_price = new_stop
                        await safe_send(app, OWNER_CHAT_ID,
                            f"🔧 {s} Trail stop → {new_stop:.4f}" + (" (lock)" if locked else ""))

                    if stop_hit(pos.side, price, pos.stop_price):
                        pnl = (price - pos.entry_price) if pos.side=="long" else (pos.entry_price - price)
                        PNL_ACCUM += pnl
                        await safe_send(app, OWNER_CHAT_ID,
                            f"🛑 {s} EXIT @ {price:.4f} (Stop hit)\nPNL: {fmt_usd(pnl)}")
                        POSITIONS[s] = None

                    # entry på TICK?
                    if ENTRY_MODE == "tick" and ORB_ENABLED and POSITIONS[s] is None and s in ORB_BOX:
                        box = ORB_BOX[s]
                        if price > box["high"]:
                            px, fee = mock_fill("long", TICKS[s])
                            stop = box["low"]
                            POSITIONS[s] = Position(symbol=s, side="long", qty=1.0, entry_price=px,
                                                    stop_price=stop, trail_high=cnow.h)
                            await safe_send(app, OWNER_CHAT_ID,
                                f"✅ {s} ENTRY (tick) LONG @ {px:.4f} | stop start = {stop:.4f}")
                        elif price < box["low"]:
                            px, fee = mock_fill("short", TICKS[s])
                            stop = box["high"]
                            POSITIONS[s] = Position(symbol=s, side="short", qty=1.0, entry_price=px,
                                                    stop_price=stop, trail_high=cnow.l)
                            await safe_send(app, OWNER_CHAT_ID,
                                f"✅ {s} ENTRY (tick) SHORT @ {px:.4f} | stop start = {stop:.4f}")

# -------------------- TELEGRAM --------------------
tg_app: Application = ApplicationBuilder().token(BOT_TOKEN).build()

def menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/mock"), KeyboardButton("/live")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

async def safe_send(app: Application, chat_id: int, text: str):
    try:
        await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception:
        pass

async def set_bot_commands():
    cmds = [
        BotCommand("status", "Visa status"),
        BotCommand("engine_on", "Starta motorn"),
        BotCommand("engine_off", "Stoppa motorn"),
        BotCommand("mock", "Mock-läge"),
        BotCommand("live", "Live-läge (placeholder)"),
        BotCommand("entrymode", "Växla entry: tick/close"),
        BotCommand("timeframe", "Visa timeframe"),
        BotCommand("pnl", "Visa PnL"),
        BotCommand("reset_pnl", "Nollställ PnL"),
        BotCommand("orb_on", "Sätt ORB på"),
        BotCommand("orb_off", "Sätt ORB av"),
        BotCommand("panic", "Stäng alla positioner"),
    ]
    await tg_app.bot.set_my_commands(cmds)

# ---- Handlers ----
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔧 ORB v34 startad.\nAnvänd /engine_on för att börja.",
                                    reply_markup=menu_keyboard())

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sym = ", ".join(SYMBOLS)
    txt = (
        "📊 Status\n"
        f"• Engine: {'AKTIV' if ENGINE_ON else 'AV'}\n"
        f"• Entry mode: {ENTRY_MODE}\n"
        f"• Timeframe: {TIMEFRAME}\n"
        f"• Symbols: {sym}"
    )
    await update.message.reply_text(txt, reply_markup=menu_keyboard())

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON, ORB_BOX
    ENGINE_ON = True
    ORB_BOX = {}  # initiera om boxar
    await update.message.reply_text("✅ Engine är nu AKTIV.", reply_markup=menu_keyboard())

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON
    ENGINE_ON = False
    await update.message.reply_text("⛔️ Engine stoppad.", reply_markup=menu_keyboard())

async def cmd_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LIVE_MODE
    LIVE_MODE = False
    await update.message.reply_text("🧪 Mock-läge aktivt (realpriser, inga riktiga orders).")

async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LIVE_MODE
    LIVE_MODE = True
    await update.message.reply_text("⚠️ Live-läge flaggat (i denna fil skickas fortfarande inga riktiga orders).")

async def cmd_entrymode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENTRY_MODE
    # Inline-knappar för tick/close
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("tick", callback_data="entry_tick"),
         InlineKeyboardButton("close", callback_data="entry_close")]
    ])
    await update.message.reply_text("Välj entry-läge:", reply_markup=kb)

async def cb_entrymode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENTRY_MODE
    q = update.callback_query
    await q.answer()
    if q.data == "entry_tick":
        ENTRY_MODE = "tick"
    elif q.data == "entry_close":
        ENTRY_MODE = "close"
    await q.edit_message_text(f"Entry mode satt till: {ENTRY_MODE}")

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Timeframe: {TIMEFRAME}")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ORB_ENABLED
    ORB_ENABLED = True
    await update.message.reply_text("🟩 ORB: PÅ (mock).")

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ORB_ENABLED
    ORB_ENABLED = False
    await update.message.reply_text("🟥 ORB: AV.")

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"PNL (mock): {fmt_usd(PNL_ACCUM)}")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global PNL_ACCUM
    PNL_ACCUM = 0.0
    await update.message.reply_text("PNL nollställd.")

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # stäng alla mock-positioner på last
    global POSITIONS, PNL_ACCUM
    closed = 0
    for s, pos in list(POSITIONS.items()):
        if pos and TICKS[s]["last"] is not None:
            px = TICKS[s]["last"]
            pnl = (px - pos.entry_price) if pos.side=="long" else (pos.entry_price - px)
            PNL_ACCUM += pnl
            POSITIONS[s] = None
            closed += 1
            await update.message.reply_text(f"🛑 {s} PANIC EXIT @ {px:.4f}\nPNL: {fmt_usd(pnl)}")
    if closed == 0:
        await update.message.reply_text("Inga öppna positioner.")

# Register handlers
tg_app.add_handler(CommandHandler("start", on_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("mock", cmd_mock))
tg_app.add_handler(CommandHandler("live", cmd_live))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CallbackQueryHandler(cb_entrymode, pattern="^entry_"))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("panic", cmd_panic))

# -------------------- FASTAPI (webhook) --------------------
app = FastAPI()

@app.on_event("startup")
async def _startup():
    await set_bot_commands()
    # sätt webhook
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    await tg_app.bot.set_webhook(url)
    # starta pris-poller + trading-loop
    asyncio.create_task(poll_prices())
    asyncio.create_task(trading_loop(tg_app))
    # ping ägaren
    await safe_send(tg_app, OWNER_CHAT_ID, "🔧 ORB v34 startad. Använd /engine_on för att börja.")

@app.post(f"/webhook/{BOT_TOKEN}")
async def tg_webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return PlainTextResponse("OK")

@app.get("/")
async def root():
    return PlainTextResponse("OK")
