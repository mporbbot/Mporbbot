# ==============================
# main_v37.py  (MOCK + PTB v20)
# ==============================
import os
import asyncio
import random
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

# -------- Miljövariabler / konfig --------
BOT_TOKEN     = os.environ.get("BOT_TOKEN", "")
WEBHOOK_BASE  = os.environ.get("WEBHOOK_BASE", "").rstrip("/")
PORT          = int(os.environ.get("PORT", "10000"))

SYMBOLS       = os.environ.get(
    "SYMBOLS",
    "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT"
).split(",")

TIMEFRAME     = os.environ.get("TIMEFRAME", "1m")  # kosmetisk i mock

# ORB & entry-regler
MIN_GREEN_BODY_PCT = float(os.environ.get("MIN_GREEN_BODY_PCT", "0.05"))  # 5%
USE_ORB            = os.environ.get("USE_ORB", "on").lower() == "on"

CAPITAL_PER_TRADE  = float(os.environ.get("CAPITAL_PER_TRADE", "100"))

# -------- Telegram webhook-URL --------
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL  = f"{WEBHOOK_BASE}{WEBHOOK_PATH}" if WEBHOOK_BASE and BOT_TOKEN else ""

# -------- FastAPI-app --------
app = FastAPI()

# -------- Telegram-app + chat-id --------
tg_app: Optional[Application] = None
CHAT_ID: Optional[int] = None  # sparas vid /start

# -------- Reply-keyboard (bara knapparna där nere) --------
reply_kb = ReplyKeyboardMarkup(
    [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/panic")],
    ],
    resize_keyboard=True
)

# -------- Global state --------
engine_running: bool = False
entry_mode: str = "close"
timeframe: str = TIMEFRAME
use_orb: bool = USE_ORB

class OrbState:
    def __init__(self):
        self.active: bool = False
        self.anchor_index: Optional[int] = None
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None

    def reset(self):
        self.active = False
        self.anchor_index = None
        self.orb_high = None
        self.orb_low = None

orb_state: Dict[str, OrbState] = {s: OrbState() for s in SYMBOLS}

class Position:
    def __init__(self, symbol: str, qty: float, entry: float, sl: float):
        self.symbol = symbol
        self.qty = qty
        self.entry = entry
        self.sl = sl

positions: Dict[str, Position] = {}
pnl_total: float = 0.0

# -------- MOCK-kandeldata (ingen Binance) --------
def mock_next_price(prev: float) -> float:
    step = random.uniform(-0.003, 0.003)  # +-0.3% per “tick”
    out = prev * (1.0 + step)
    return max(0.0001, out)

candles: Dict[str, List[Tuple[float, float, float, float]]] = {s: [] for s in SYMBOLS}
last_price: Dict[str, float] = {s: random.uniform(10, 30000) for s in SYMBOLS}

def push_candle(symbol: str):
    o = last_price[symbol]
    h = o
    l = o
    c = o
    # bygg en liten “interna ticks”-sväng för att få rimliga OHLC
    for _ in range(6):
        c = mock_next_price(c)
        h = max(h, c)
        l = min(l, c)
    last_price[symbol] = c
    arr = candles[symbol]
    arr.append((o, h, l, c))
    if len(arr) > 400:
        arr.pop(0)

def is_green(c): o, h, l, cl = c; return cl > o
def is_red(c):   o, h, l, cl = c; return cl < o
def green_body_pct(c): o, h, l, cl = c; return max(0.0, (cl - o) / o) if o > 0 else 0.0

# -------- ORB-logik (NY enligt dina regler) --------
def update_orb(symbol: str):
    """
    Sätt ORB när: candle[-2] röd, candle[-1] grön med kropp >= MIN_GREEN_BODY_PCT.
    ORB-high/low tas från denna ‘första gröna’.
    """
    arr = candles[symbol]
    st = orb_state[symbol]
    if len(arr) < 2:
        return

    if not st.active:
        prev, cur = arr[-2], arr[-1]
        if is_red(prev) and is_green(cur) and green_body_pct(cur) >= MIN_GREEN_BODY_PCT:
            st.active = True
            st.anchor_index = len(arr) - 1
            st.orb_high = cur[1]
            st.orb_low = cur[2]
    else:
        # Om marknaden gör en ny “första grön efter röd” långt senare kan man välja
        # att reseta ORB. Vi låter aktiv ORB ligga tills en entry sker och stänger.
        pass

def should_enter_long(symbol: str) -> Optional[float]:
    """
    Entry: TIDIGAST candle #2 efter ‘första gröna’, som dessutom
    MÅSTE stänga över ORB-high (och vara grön).
    Candle #3, #4, ... får också trigga när de uppfyller kravet.
    """
    if not use_orb:
        return None
    arr = candles[symbol]
    st = orb_state[symbol]
    if not (st.active and st.orb_high is not None and st.anchor_index is not None):
        return None
    if len(arr) < st.anchor_index + 2:
        return None  # behöver åtminstone candle #2

    i = len(arr) - 1  # aktuell candle
    if i < st.anchor_index + 1:
        return None  # inte #2 ännu

    o, h, l, c = arr[i]
    if c > o and c > st.orb_high:
        return c
    return None

# -------- Trailing SL (följ förra candlens low uppåt) --------
def trail_sl_up(symbol: str):
    if symbol not in positions:
        return
    arr = candles[symbol]
    if len(arr) < 2:
        return
    prev_low = arr[-2][2]
    p = positions[symbol]
    if prev_low > p.sl:
        p.sl = prev_low

# -------- Telegram utils --------
async def send_text(text: str):
    if CHAT_ID and tg_app and tg_app.bot:
        await tg_app.bot.send_message(chat_id=CHAT_ID, text=text, reply_markup=reply_kb)

# -------- Trade helpers --------
def qty_for(capital: float, price: float) -> float:
    if price <= 0:
        return 0.0
    return capital / price

async def open_long(symbol: str, price: float):
    if symbol in positions:
        return
    st = orb_state[symbol]
    sl = st.orb_low if st.orb_low else price * 0.995  # fallback
    q = qty_for(CAPITAL_PER_TRADE, price)
    positions[symbol] = Position(symbol, q, price, sl)
    await send_text(
        f"🟢 ENTRY LONG (close) {symbol} @ {price:.4f}\n"
        f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | SL={sl:.4f} | QTY={q:.6f}"
    )

async def close_long(symbol: str, price: float):
    global pnl_total
    if symbol not in positions:
        return
    p = positions.pop(symbol)
    pnl = (price - p.entry) * p.qty
    pnl_total += pnl
    await send_text(f"🔴 EXIT {symbol} @ {price:.4f} | PnL: {pnl:+.4f} USDT ❌")
    # Efter exit kan vi välja att nollställa ORB så en ny sekvens krävs
    orb_state[symbol].reset()

async def close_all():
    for s in list(positions.keys()):
        arr = candles[s]
        price = arr[-1][3] if arr else last_price[s]
        await close_long(s, price)

# -------- Engine loop --------
async def engine_loop():
    while True:
        await asyncio.sleep(1.0)  # 1 sekund ~ 1 “minut” i mocken
        if not engine_running:
            continue

        # Skapa nya candles
        for s in SYMBOLS:
            push_candle(s)
            update_orb(s)

        # Traila SL + stäng om träff
        for s in list(positions.keys()):
            trail_sl_up(s)
            c = candles[s][-1][3]
            if c <= positions[s].sl:
                await close_long(s, positions[s].sl)

        # Nya entries
        for s in SYMBOLS:
            if s in positions:
                continue
            ep = should_enter_long(s)
            if ep is not None:
                await open_long(s, ep)

# -------- Commands --------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CHAT_ID
    CHAT_ID = update.effective_chat.id
    await update.message.reply_text(
        "Startad ✅\n"
        "Mode: MOCK\n"
        f"TF: {timeframe}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"ORB: PÅ (första gröna efter röd, entry när en senare grön stänger över ORB-high)\n"
        f"PnL total: {pnl_total:.2f} USDT",
        reply_markup=reply_kb
    )

async def engine_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_running, CHAT_ID
    CHAT_ID = update.effective_chat.id
    engine_running = True
    await update.message.reply_text("Engine: ON ✅", reply_markup=reply_kb)

async def engine_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_running
    engine_running = False
    await update.message.reply_text("Engine: OFF ⏹️", reply_markup=reply_kb)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s_pos = [f"{sym} qty={p.qty:.4f} entry={p.entry:.4f} SL={p.sl:.4f}" for sym, p in positions.items()]
    pos_str = "\n".join(s_pos) if s_pos else "inga"
    await update.message.reply_text(
        "Engine: " + ("ON" if engine_running else "OFF") + "\n"
        f"Entry mode: {entry_mode}\n"
        f"Timeframe: {timeframe}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"ORB: PÅ (första gröna efter röd)\n"
        f"Positioner:\n{pos_str}\n"
        f"PnL total: {pnl_total:.2f} USDT",
        reply_markup=reply_kb
    )

async def entrymode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Entry: close", reply_markup=reply_kb)

async def timeframe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Timeframe satt till {timeframe}", reply_markup=reply_kb)

async def orb_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global use_orb
    use_orb = True
    await update.message.reply_text("ORB: PÅ (första gröna efter röd)", reply_markup=reply_kb)

async def orb_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global use_orb
    use_orb = False
    await update.message.reply_text("ORB: AV", reply_markup=reply_kb)

async def panic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await close_all()
    await update.message.reply_text("PANIC: Allt stängt.", reply_markup=reply_kb)

# Fånga övriga /kommandon och visa status
async def fallback_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await status_cmd(update, context)

# -------- Startup / Shutdown --------
async def on_startup():
    global tg_app
    if not BOT_TOKEN:
        raise RuntimeError("Saknar BOT_TOKEN")
    if not WEBHOOK_URL:
        raise RuntimeError("WEBHOOK_BASE saknas")

    tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Knappar/kommandon – samma layout som v36 (endast reply-kb)
    tg_app.add_handler(CommandHandler("start", start_cmd))
    tg_app.add_handler(CommandHandler("engine_on", engine_on_cmd))
    tg_app.add_handler(CommandHandler("engine_off", engine_off_cmd))
    tg_app.add_handler(CommandHandler("status", status_cmd))
    tg_app.add_handler(CommandHandler("entrymode", entrymode_cmd))
    tg_app.add_handler(CommandHandler("timeframe", timeframe_cmd))
    tg_app.add_handler(CommandHandler("orb_on", orb_on_cmd))
    tg_app.add_handler(CommandHandler("orb_off", orb_off_cmd))
    tg_app.add_handler(CommandHandler("panic", panic_cmd))
    tg_app.add_handler(MessageHandler(filters.COMMAND, fallback_cmd))  # fallback

    await tg_app.initialize()
    # Rensa ev gamla och sätt nytt webhook
    await tg_app.bot.delete_webhook(drop_pending_updates=True)
    await tg_app.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=["message"])
    await tg_app.start()

    # Starta motorn
    asyncio.create_task(engine_loop())

async def on_shutdown():
    if tg_app:
        await tg_app.stop()

# -------- FastAPI endpoints --------
@app.on_event("startup")
async def _startup_event():
    await on_startup()

@app.on_event("shutdown")
async def _shutdown_event():
    await on_shutdown()

@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if tg_app is None or tg_app.bot is None:
        raise HTTPException(status_code=503, detail="Bot not ready")
    if request.headers.get("content-type") != "application/json":
        raise HTTPException(status_code=415, detail="Content type must be application/json")
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return JSONResponse({"ok": True})
