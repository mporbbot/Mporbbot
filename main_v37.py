# ==============================
# main_v37.py
# ==============================
import os
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    MessageHandler, ContextTypes, filters,
)

# --------------------
# Konfiguration
# --------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
WEBHOOK_BASE = os.environ.get("WEBHOOK_BASE", "").rstrip("/")
PORT = int(os.environ.get("PORT", "10000"))
MODE = os.environ.get("MODE", "MOCK").upper()  # håll MOCK
SYMBOLS = os.environ.get("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT").split(",")
TIMEFRAME = os.environ.get("TIMEFRAME", "1m")

# ORB & entry
MIN_GREEN_BODY_PCT = float(os.environ.get("MIN_GREEN_BODY_PCT", "0.05"))  # 5%
USE_ORB = os.environ.get("USE_ORB", "on").lower() == "on"

# Risk & kapital per trade i MOCK
CAPITAL_PER_TRADE = float(os.environ.get("CAPITAL_PER_TRADE", "100"))

# --------------------
# Telegram-app
# --------------------
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_BASE}{WEBHOOK_PATH}" if WEBHOOK_BASE and BOT_TOKEN else ""

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

app = FastAPI()
tg_app: Application

# --------------------
# Intern state
# --------------------
engine_running: bool = False
entry_mode: str = "close"          # "close" (vi använder close-entries)
timeframe: str = TIMEFRAME
use_orb: bool = USE_ORB

# ORB state per symbol
class OrbState:
    def __init__(self):
        self.active: bool = False
        self.anchor_index: Optional[int] = None   # index för gröna candlen
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None

orb_state: Dict[str, OrbState] = {s: OrbState() for s in SYMBOLS}

# Position & PnL
class Position:
    def __init__(self, symbol: str, qty: float, entry: float, sl: float):
        self.symbol = symbol
        self.qty = qty
        self.entry = entry
        self.sl = sl

positions: Dict[str, Position] = {}
pnl_total: float = 0.0

# --------------------
# MOCK: candle-feed
# --------------------
# Vi simmar priser lite “rimligt” (ingen extern börs).
import random
def mock_next_price(prev: float) -> float:
    # litet random steg
    step = random.uniform(-0.003, 0.003)  # +-0.3%
    return max(0.0001, prev * (1.0 + step))

# Håller senaste 3 candles per symbol (o,h,l,c)
candles: Dict[str, List[Tuple[float,float,float,float]]] = {s: [] for s in SYMBOLS}
last_price: Dict[str, float] = {s: random.uniform(10, 30000) for s in SYMBOLS}

def push_candle(symbol: str):
    global last_price
    o = last_price[symbol]
    # skapar en candle
    h = o
    l = o
    c = o
    # låt den "handla" 6 ticks
    for _ in range(6):
        c = mock_next_price(c)
        h = max(h, c); l = min(l, c)
    last_price[symbol] = c
    arr = candles[symbol]
    arr.append((o,h,l,c))
    if len(arr) > 400:
        arr.pop(0)

def body_pct_green(cndl: Tuple[float,float,float,float]) -> float:
    o,h,l,c = cndl
    if c <= o:
        return 0.0
    return (c - o) / o

def is_green(cndl: Tuple[float,float,float,float]) -> bool:
    o,h,l,c = cndl
    return c > o

def is_red(cndl: Tuple[float,float,float,float]) -> bool:
    o,h,l,c = cndl
    return c < o

# --------------------
# ORB-logik (NY)
# --------------------
def update_orb(symbol: str):
    """
    Sätter ORB när vi ser första GRÖNA candle DIREKT efter en RÖD.
    ORB-high/low = high/low på den gröna.
    Kräver minst MIN_GREEN_BODY_PCT i green-body.
    """
    arr = candles[symbol]
    st = orb_state[symbol]
    if len(arr) < 2:
        return

    # Om ingen aktiv ORB: leta efter mönstret [röd, grön]
    if not st.active:
        prev = arr[-2]
        cur = arr[-1]
        if is_red(prev) and is_green(cur) and body_pct_green(cur) >= MIN_GREEN_BODY_PCT:
            # sätt ORB
            st.active = True
            st.anchor_index = len(arr) - 1
            st.orb_high = cur[1]  # high
            st.orb_low = cur[2]   # low
    else:
        # ORB reset var valfri i v36 — vi behåller aktiv tills trade tas, eller man stänger av ORB
        pass

def should_enter_long(symbol: str) -> Optional[float]:
    """
    Entry tidigast på candle #2 efter ankar-gröna, och endast om stänger över ORB-high.
    Om #2 inte triggar så kan #3,#4,... trigga.
    Returnerar entrypris (close) om vi ska köpa nu, annars None.
    """
    arr = candles[symbol]
    st = orb_state[symbol]
    if not use_orb:
        return None
    if not st.active or st.orb_high is None or st.anchor_index is None:
        return None
    if len(arr) < st.anchor_index + 2:
        return None
    # index för första candlen som får köpa
    first_ok_i = st.anchor_index + 1  # “candle #2” i mänsklig beskrivning
    # titta på SISTA candlen (close-entry)
    i = len(arr) - 1
    if i < first_ok_i:
        return None
    o,h,l,c = arr[i]
    # kräver grön stängning ÖVER ORB-high
    if c > o and c > st.orb_high:
        return c
    return None

# --------------------
# SL-följning (som i v36 – flytta till senaste candle low)
# --------------------
def trail_sl_up(symbol: str):
    if symbol not in positions:
        return
    pos = positions[symbol]
    arr = candles[symbol]
    if len(arr) < 2:
        return
    # sätt SL till föregående candles low om det är högre än nuvarande SL
    prev_low = arr[-2][2]
    if prev_low > pos.sl:
        pos.sl = prev_low

# --------------------
# Telegram helpers
# --------------------
async def send_msg(context: ContextTypes.DEFAULT_TYPE, text: str):
    # Skicka till ägarchatten – i MOCK kör vi helt enkelt till första som pingade oss.
    # För enkelhet: spara chat_id efter första kommandot.
    chat_id = context.bot_data.get("chat_id")
    if chat_id:
        await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_kb)

# --------------------
# Kommandon
# --------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.bot_data["chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        "Startad ✅\nMode: MOCK\n"
        f"TF: {timeframe}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"ORB: {'PÅ' if use_orb else 'AV'} (första gröna efter röd)\n"
        f"PnL total: {pnl_total:.2f} USDT",
        reply_markup=reply_kb
    )

async def engine_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_running
    context.bot_data["chat_id"] = update.effective_chat.id
    engine_running = True
    await update.message.reply_text("Engine: ON ✅", reply_markup=reply_kb)

async def engine_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_running
    engine_running = False
    await update.message.reply_text("Engine: OFF ⏹️", reply_markup=reply_kb)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s_pos = []
    for sym, p in positions.items():
        s_pos.append(f"{sym} qty={p.qty:.4f} entry={p.entry:.4f} SL={p.sl:.4f}")
    pos_str = "\n".join(s_pos) if s_pos else "inga"
    await update.message.reply_text(
        "Engine: " + ("ON" if engine_running else "OFF") + "\n"
        f"Entry mode: {entry_mode}\n"
        f"Timeframe: {timeframe}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"ORB: {'PÅ (första gröna efter röd)'}\n"
        f"Positioner: {pos_str}\n"
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
    # stäng alla pos
    await close_all(context)
    await update.message.reply_text("PANIC: Allt stängt.", reply_markup=reply_kb)

# --------------------
# Trading helpers
# --------------------
def qty_for(capital: float, price: float) -> float:
    if price <= 0:
        return 0.0
    return capital / price

async def open_long(symbol: str, price: float, context: ContextTypes.DEFAULT_TYPE):
    global positions
    if symbol in positions:
        return
    q = qty_for(CAPITAL_PER_TRADE, price)
    # SL i start = ORB-low (om finns), annars lite under entry
    st = orb_state[symbol]
    sl = st.orb_low if st.orb_low else price * 0.995
    positions[symbol] = Position(symbol=symbol, qty=q, entry=price, sl=sl)
    await send_msg(context,
        f"🟢 ENTRY LONG (close) {symbol} @ {price:.4f}\n"
        f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | SL={sl:.4f} | QTY={q:.6f}"
    )

async def close_long(symbol: str, price: float, context: ContextTypes.DEFAULT_TYPE):
    global positions, pnl_total
    if symbol not in positions:
        return
    pos = positions.pop(symbol)
    pnl = (price - pos.entry) * pos.qty
    pnl_total += pnl
    await send_msg(context, f"🔴 EXIT {symbol} @ {price:.4f} | PnL: {pnl:+.4f} USDT ❌")

async def close_all(context: ContextTypes.DEFAULT_TYPE):
    for sym in list(positions.keys()):
        # stänger på senaste close
        arr = candles[sym]
        price = arr[-1][3] if arr else last_price[sym]
        await close_long(sym, price, context)

# --------------------
# Motor
# --------------------
async def engine_loop():
    # kör hela tiden
    while True:
        await asyncio.sleep(1.0)  # 1s = fusk-tick motsvarar ~1m candle i MOCK
        if not engine_running:
            continue

        # skapa en ny candle per symbol, uppdatera ORB och utvärdera
        for sym in SYMBOLS:
            push_candle(sym)
            update_orb(sym)

        # trail SL & exits
        for sym, pos in list(positions.items()):
            trail_sl_up(sym)
            # SL-trigger på ny close
            c = candles[sym][-1][3]
            if c <= pos.sl:
                # stäng på SL
                # (om du vill ha exakt SL-nivå i meddelandet kan du använda pos.sl)
                await close_long(sym, pos.sl, tg_app.application_context)

        # nya entries
        for sym in SYMBOLS:
            if sym in positions:
                continue
            entry_price = should_enter_long(sym)
            if entry_price is not None:
                await open_long(sym, entry_price, tg_app.application_context)

# --------------------
# Telegram wiring
# --------------------
async def on_startup():
    global tg_app
    if not BOT_TOKEN:
        raise RuntimeError("Saknar BOT_TOKEN")
    tg_app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .build()
    )

    # handlers
    tg_app.add_handler(CommandHandler("start", start_cmd))
    tg_app.add_handler(CommandHandler("engine_on", engine_on_cmd))
    tg_app.add_handler(CommandHandler("engine_off", engine_off_cmd))
    tg_app.add_handler(CommandHandler("status", status_cmd))
    tg_app.add_handler(CommandHandler("entrymode", entrymode_cmd))
    tg_app.add_handler(CommandHandler("timeframe", timeframe_cmd))
    tg_app.add_handler(CommandHandler("orb_on", orb_on_cmd))
    tg_app.add_handler(CommandHandler("orb_off", orb_off_cmd))
    tg_app.add_handler(CommandHandler("panic", panic_cmd))
    # även /ping för test
    async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.bot_data["chat_id"] = update.effective_chat.id
        await update.message.reply_text("Pong ✅", reply_markup=reply_kb)
    tg_app.add_handler(CommandHandler("ping", ping_cmd))

    # webhook
    if WEBHOOK_URL:
        await tg_app.bot.delete_webhook(drop_pending_updates=True)
        await tg_app.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=["message"])
    else:
        raise RuntimeError("WEBHOOK_BASE saknas eller BOT_TOKEN saknas")

    # spara en context att använda från motorn
    tg_app.application_context = (await tg_app.initialize()).bot._application  # liten hack för att ha context

    # starta motorn
    asyncio.create_task(engine_loop())

    # starta telegram application (pollar inte; webhook används)
    await tg_app.start()

async def on_shutdown():
    await tg_app.stop()

# --------------------
# FastAPI endpoints
# --------------------
@app.on_event("startup")
async def _startup():
    await on_startup()

@app.on_event("shutdown")
async def _shutdown():
    await on_shutdown()

@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if request.headers.get("content-type") != "application/json":
        raise HTTPException(status_code=415, detail="Content type must be application/json")
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return JSONResponse({"ok": True})
