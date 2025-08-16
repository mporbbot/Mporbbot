import os
import logging
import pandas as pd
import numpy as np
import requests
import schedule
import time
import threading
from fastapi import FastAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# FastAPI app (for Render health check)
# ------------------------------------------------------------
app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Mp ORBbot running"}

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))

# Mock trading config
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))

# ------------------------------------------------------------
# Trading state
# ------------------------------------------------------------
ai_mode = "neutral"  # default
bot_running = False
mock_trades = []
real_trades = []

# ------------------------------------------------------------
# Telegram Bot setup
# ------------------------------------------------------------
updater = Updater(TELEGRAM_TOKEN, use_context=True)
dispatcher = updater.dispatcher

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def log_trade(trade_type, symbol, entry, exit_price, profit, is_mock=True):
    """Log trades to CSV for Skatteverket"""
    filename = "mock_trade_log.csv" if is_mock else "real_trade_log.csv"
    df = pd.DataFrame([{
        "type": trade_type,
        "symbol": symbol,
        "entry": entry,
        "exit": exit_price,
        "profit": profit
    }])
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, index=False)

def restricted(func):
    def wrapper(update: Update, context: CallbackContext, *args, **kwargs):
        if update.effective_user.id != AUTHORIZED_USER_ID:
            update.message.reply_text("Unauthorized")
            return
        return func(update, context, *args, **kwargs)
    return wrapper

# ------------------------------------------------------------
# Telegram commands
# ------------------------------------------------------------
@restricted
def start(update: Update, context: CallbackContext):
    global bot_running
    keyboard = [[InlineKeyboardButton("JA", callback_data="start_mock")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text("Vill du starta mock trading?", reply_markup=reply_markup)

def button(update: Update, context: CallbackContext):
    global bot_running
    query = update.callback_query
    query.answer()
    if query.data == "start_mock":
        bot_running = True
        query.edit_message_text("✅ Mock trading startad")

@restricted
def stop(update: Update, context: CallbackContext):
    global bot_running
    bot_running = False
    update.message.reply_text("⏹ Bot stoppad")

@restricted
def status(update: Update, context: CallbackContext):
    update.message.reply_text(f"🤖 Status:\nAI-läge: {ai_mode}\nBot running: {bot_running}")

@restricted
def set_ai(update: Update, context: CallbackContext):
    global ai_mode
    if not context.args:
        update.message.reply_text("Usage: /set_ai [neutral|aggressiv|försiktig]")
        return
    ai_mode = context.args[0]
    update.message.reply_text(f"AI-läge ändrat till: {ai_mode}")

# ------------------------------------------------------------
# Register handlers
# ------------------------------------------------------------
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CallbackQueryHandler(button))
dispatcher.add_handler(CommandHandler("stop", stop))
dispatcher.add_handler(CommandHandler("status", status))
dispatcher.add_handler(CommandHandler("set_ai", set_ai))

# ------------------------------------------------------------
# Background thread for mock trading loop
# ------------------------------------------------------------
def trading_loop():
    while True:
        if bot_running:
            # TODO: implement ORB logic + KuCoin fetch
            logger.info("Checking market (placeholder)")
        time.sleep(10)

threading.Thread(target=trading_loop, daemon=True).start()

# ------------------------------------------------------------
# Start Telegram bot
# ------------------------------------------------------------
def run_telegram():
    updater.start_polling()
    updater.idle()

threading.Thread(target=run_telegram, daemon=True).start()
