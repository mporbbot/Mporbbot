import os
import logging
import asyncio
from fastapi import FastAPI, Request
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler
)

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Miljövariabler ===
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "")
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN i miljövariablerna")

if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

# === Bot state ===
ENGINE_ON = False
MOCK_MODE = True
ENTRY_MODE = "close"   # default
TIMEFRAME = "3m"
ORB_ON = True
PNL = 0.0

# === FastAPI & Telegram App ===
app = FastAPI()
tg_app = Application.builder().token(BOT_TOKEN).build()

# === Hjälp ===
async def cmd_help(update: Update, context):
    msg = (
        "/status – Visa status\n"
        "/engine_start – Starta motorn\n"
        "/engine_stop – Stoppa motorn\n"
        "/start_mock – Mock-läge\n"
        "/start_live – Live-läge (KuCoin)\n"
        "/entry_mode – Växla entry (tick/close)\n"
        "/timeframe – Välj timeframe\n"
        "/pnl – Visa PnL\n"
        "/reset_pnl – Nollställ PnL\n"
        "/orb_on – Slå på ORB-filter\n"
        "/orb_off – Stäng av ORB-filter\n"
        "/panic – Stäng alla positioner\n"
    )
    await update.message.reply_text(msg)

# === Status ===
async def cmd_status(update: Update, context):
    status = (
        f"Engine: {'ON' if ENGINE_ON else 'OFF'}\n"
        f"Mode: {'MOCK' if MOCK_MODE else 'LIVE'}\n"
        f"Entry: {ENTRY_MODE}\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"ORB: {'ON' if ORB_ON else 'OFF'}\n"
        f"PnL: {PNL:.2f} USDT"
    )
    await update.message.reply_text(status)

# === Engine ON/OFF ===
async def cmd_engine_on(update: Update, context):
    global ENGINE_ON
    ENGINE_ON = True
    await update.message.reply_text("✅ Motorn STARTAD")

async def cmd_engine_off(update: Update, context):
    global ENGINE_ON
    ENGINE_ON = False
    await update.message.reply_text("🛑 Motorn STOPPAD")

# === Mock / Live ===
async def cmd_start_mock(update: Update, context):
    global MOCK_MODE
    MOCK_MODE = True
    await update.message.reply_text("🤖 Mock-läge aktiverat")

async def cmd_start_live(update: Update, context):
    global MOCK_MODE
    MOCK_MODE = False
    await update.message.reply_text("💰 Live-läge (KuCoin) aktiverat")

# === Entry mode ===
async def cmd_entry_mode(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("Close", callback_data="entry_close"),
         InlineKeyboardButton("Tick", callback_data="entry_tick")]
    ]
    await update.message.reply_text("Välj entry-läge:", reply_markup=InlineKeyboardMarkup(keyboard))

# === Timeframe ===
async def cmd_timeframe(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("1m", callback_data="tf_1m"),
         InlineKeyboardButton("3m", callback_data="tf_3m"),
         InlineKeyboardButton("5m", callback_data="tf_5m")],
        [InlineKeyboardButton("15m", callback_data="tf_15m"),
         InlineKeyboardButton("1h", callback_data="tf_1h"),
         InlineKeyboardButton("1d", callback_data="tf_1d")]
    ]
    await update.message.reply_text("Välj timeframe:", reply_markup=InlineKeyboardMarkup(keyboard))

# === PnL ===
async def cmd_pnl(update: Update, context):
    await update.message.reply_text(f"PnL: {PNL:.2f} USDT")

async def cmd_reset_pnl(update: Update, context):
    global PNL
    PNL = 0.0
    await update.message.reply_text("PnL nollställd ✅")

# === ORB ON/OFF ===
async def cmd_orb_on(update: Update, context):
    global ORB_ON
    ORB_ON = True
    await update.message.reply_text("🔵 ORB är PÅ")

async def cmd_orb_off(update: Update, context):
    global ORB_ON
    ORB_ON = False
    await update.message.reply_text("⚪ ORB är AV")

# === PANIC ===
async def cmd_panic(update: Update, context):
    await update.message.reply_text("🚨 PANIC – alla positioner stängda!")

# === Callback Buttons ===
async def cb_buttons(update: Update, context):
    global ENTRY_MODE, TIMEFRAME
    query = update.callback_query
    await query.answer()

    if query.data == "entry_close":
        ENTRY_MODE = "close"
        await query.edit_message_text("✅ Entry mode satt till CLOSE")
    elif query.data == "entry_tick":
        ENTRY_MODE = "tick"
        await query.edit_message_text("✅ Entry mode satt till TICK")

    elif query.data.startswith("tf_"):
        TIMEFRAME = query.data.replace("tf_", "")
        await query.edit_message_text(f"✅ Timeframe satt till {TIMEFRAME}")

# === Webhook Routes ===
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    # Kommandon för Telegram-meny
    commands = [
        ("status", "Visa status (sparar chat_id)"),
        ("engine_start", "Starta motorn"),
        ("engine_stop", "Stoppa motorn"),
        ("start_mock", "Mock-läge"),
        ("start_live", "Live-läge (KuCoin)"),
        ("entry_mode", "Växla entry: tick/close"),
        ("timeframe", "Välj timeframe"),
        ("pnl", "Visa PnL"),
        ("reset_pnl", "Nollställ PnL"),
        ("orb_on", "Slå på ORB-filter"),
        ("orb_off", "Stäng av ORB-filter"),
        ("panic", "Stäng alla positioner"),
        ("help", "Hjälp"),
    ]
    await tg_app.bot.set_my_commands([BotCommand(c, d) for c, d in commands])

    # Handlers
    tg_app.add_handler(CommandHandler(["help", "start"], cmd_help))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(CommandHandler(["engine_on", "engine_start"], cmd_engine_on))
    tg_app.add_handler(CommandHandler(["engine_off", "engine_stop"], cmd_engine_off))
    tg_app.add_handler(CommandHandler(["start_mock", "mock"], cmd_start_mock))
    tg_app.add_handler(CommandHandler(["start_live", "live"], cmd_start_live))
    tg_app.add_handler(CommandHandler(["entry_mode", "entrymode"], cmd_entry_mode))
    tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
    tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    tg_app.add_handler(CommandHandler("trailing", cmd_pnl))  # placeholder
    tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
    tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
    tg_app.add_handler(CommandHandler("panic", cmd_panic))
    tg_app.add_handler(CallbackQueryHandler(cb_buttons))

    # Sätt webhook
    url = f"{WEBHOOK_BASE}/webhook"
    await tg_app.bot.set_webhook(url)
    logger.info(f"Webhook satt: {url}")

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.bot.delete_webhook()
