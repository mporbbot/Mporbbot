# main_orb_v34.py
import os
import logging
from typing import Optional

from fastapi import FastAPI, Request, Response
from telegram import (
    Update, ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton
)
from telegram.ext import (
    Application, ContextTypes, CommandHandler, CallbackQueryHandler
)

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("orb_v34")

# ===== ENV =====
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
WEBHOOK_BASE = (os.getenv("WEBHOOK_BASE") or "").rstrip("/")
OWNER_CHAT_ID = (os.getenv("OWNER_CHAT_ID") or "").strip()

ENTRY_MODE = (os.getenv("ENTRY_MODE", "close") or "close").lower()  # default close
TIMEFRAME = (os.getenv("TIMEFRAME", "5m") or "5m")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT").strip() or
           "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT").split(",")

if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

# ===== STATE =====
ENGINE_RUNNING = False
ALLOWED_TFS = ["1m", "3m", "5m", "15m", "30m", "1h"]

# ===== Telegram Application =====
tg_app: Application = Application.builder().token(BOT_TOKEN).build()

def main_keyboard() -> ReplyKeyboardMarkup:
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

async def send_menu(update: Update, text: str):
    if update.message:
        await update.message.reply_text(text, reply_markup=main_keyboard())
    elif update.callback_query:
        await update.callback_query.message.reply_text(text, reply_markup=main_keyboard())

# ===== Handlers (functions) =====
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "🔧 ORB v34 startad. Använd /engine_on för att börja.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "📊 Status\n"
        f"• Engine: {'AKTIV' if ENGINE_RUNNING else 'STOPPAD'}\n"
        f"• Entry mode: {ENTRY_MODE}\n"
        f"• Timeframe: {TIMEFRAME}\n"
        f"• Symbols: {', '.join(s.strip() for s in SYMBOLS)}"
    )
    await send_menu(update, txt)

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_RUNNING
    ENGINE_RUNNING = True
    await send_menu(update, "✅ Engine är nu AKTIV.")

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_RUNNING
    ENGINE_RUNNING = False
    await send_menu(update, "🛑 Engine stoppad.")

async def cmd_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "🧪 Mock-läge valt (ingen riktig handel).")

async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "💼 Live-läge markerat (KuCoin).")

async def cmd_entrymode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENTRY_MODE
    ENTRY_MODE = "tick" if ENTRY_MODE == "close" else "close"
    await send_menu(update, f"🔁 Entry mode satt till: {ENTRY_MODE}")

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup.from_row(
        [InlineKeyboardButton(tf, callback_data=f"tf:{tf}") for tf in ALLOWED_TFS]
    )
    await update.message.reply_text(f"⏱ Välj timeframe (nu: {TIMEFRAME})", reply_markup=kb)

async def on_cbq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TIMEFRAME
    if not update.callback_query or not update.callback_query.data:
        return
    data = update.callback_query.data
    if data.startswith("tf:"):
        tf = data.split(":", 1)[1]
        if tf in ALLOWED_TFS:
            TIMEFRAME = tf
            await update.callback_query.answer(f"Timeframe: {tf}", show_alert=False)
            try:
                await update.callback_query.edit_message_text(f"⏱ Timeframe satt till {tf}")
            except Exception:
                # Om edit inte går (t.ex. pga samma text)
                pass
            await send_menu(update, f"✅ Timeframe uppdaterad till {tf}")
        else:
            await update.callback_query.answer("Ogiltigt timeframe", show_alert=True)

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_RUNNING
    ENGINE_RUNNING = False
    await send_menu(update, "🛑 PANIC: Engine stoppad och mock-positioner anses stängda.")

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "📈 PnL: 0.00 USDT (mock).")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "♻️ PnL återställt (mock).")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "🟩 ORB: PÅ (mock).")

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "🟥 ORB: AV (mock).")

# Registrera handlers (v20-stil, utan dekoratorer)
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("mock", cmd_mock))
tg_app.add_handler(CommandHandler("live", cmd_live))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CallbackQueryHandler(on_cbq))

# ===== FastAPI + Webhook =====
app = FastAPI()

@app.on_event("startup")
async def _startup():
    log.info("Initierar Telegram Application…")
    await tg_app.initialize()
    await tg_app.start()
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    await tg_app.bot.set_webhook(url)
    log.info(f"Webhook satt: {url}")

@app.on_event("shutdown")
async def _shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    return {
        "ok": True,
        "engine": ENGINE_RUNNING,
        "entry": ENTRY_MODE,
        "timeframe": TIMEFRAME,
        "symbols": [s.strip() for s in SYMBOLS],
    }

@app.post(f"/webhook/{{token}}")
async def webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return Response(status_code=403)
    try:
        body = await request.json()
    except Exception:
        return Response(status_code=400)
    update = Update.de_json(body, tg_app.bot)
    await tg_app.process_update(update)
    return Response(status_code=200)
