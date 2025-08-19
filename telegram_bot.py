import os
import asyncio
import logging
import threading
import queue
from typing import Optional

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

LOG = logging.getLogger("tg_bot")
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# === Shared state inside this module ===
_CHAT_ID: Optional[int] = None
_OUTBOX: "queue.Queue[dict]" = queue.Queue()
_APP: Optional[Application] = None

STATE = {
    "orb_on": True,
    "entry_mode": "both",  # long | short | both
    "timeframe": "1m",     # 1m | 3m | 5m | 15m
    "trailing": True,
}

def build_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("TF 1m", callback_data="tf:1m"),
            InlineKeyboardButton("TF 3m", callback_data="tf:3m"),
            InlineKeyboardButton("TF 5m", callback_data="tf:5m"),
            InlineKeyboardButton("TF 15m", callback_data="tf:15m"),
        ],
        [
            InlineKeyboardButton("Entry LONG", callback_data="entry:long"),
            InlineKeyboardButton("Entry SHORT", callback_data="entry:short"),
            InlineKeyboardButton("Entry BOTH", callback_data="entry:both"),
        ],
        [
            InlineKeyboardButton("ORB ON" if STATE["orb_on"] else "ORB OFF", callback_data="orb:toggle"),
            InlineKeyboardButton("Trailing ON" if STATE["trailing"] else "Trailing OFF", callback_data="trail:toggle"),
        ],
    ])

# === Public API for other modules ===
def send_text(text: str, parse_mode: Optional[str] = "HTML"):
    """
    Thread-safe: queue a message for Telegram.
    Requires the user to /start at least once to set _CHAT_ID.
    """
    try:
        _OUTBOX.put_nowait({"type": "text", "text": text, "parse_mode": parse_mode})
    except Exception as e:
        LOG.warning(f"[TG] Outbox put failed: {e}")

def start_in_thread(token: str):
    """
    Start Telegram bot in its own thread with its own event loop.
    """
    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_amain(token))

    t = threading.Thread(target=_runner, name="telegram-bot", daemon=True)
    t.start()
    LOG.info("[TG] thread started")

# === Handlers ===
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _CHAT_ID
    _CHAT_ID = update.effective_chat.id
    await update.message.reply_text(
        "Hej! Jag är igång.\nAnvänd knapparna nedan för att styra ORB.",
        reply_markup=build_menu(),
    )

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Meny:", reply_markup=build_menu())

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = (q.data or "")
    if data.startswith("tf:"):
        STATE["timeframe"] = data.split(":")[1]
        await q.edit_message_reply_markup(build_menu())
        await q.message.reply_text(f"TF = {STATE['timeframe']}")
        return
    if data.startswith("entry:"):
        STATE["entry_mode"] = data.split(":")[1]
        await q.edit_message_reply_markup(build_menu())
        await q.message.reply_text(f"Entry mode = {STATE['entry_mode'].upper()}")
        return
    if data == "orb:toggle":
        STATE["orb_on"] = not STATE["orb_on"]
        await q.edit_message_reply_markup(build_menu())
        await q.message.reply_text(f"ORB = {'ON' if STATE['orb_on'] else 'OFF'}")
        return
    if data == "trail:toggle":
        STATE["trailing"] = not STATE["trailing"]
        await q.edit_message_reply_markup(build_menu())
        await q.message.reply_text(f"Trailing = {'ON' if STATE['trailing'] else 'OFF'}")
        return

# === Background pumper ===
async def _outbox_pumper():
    """
    Periodically drains the outbox queue and sends to the last known _CHAT_ID.
    """
    while True:
        try:
            item = _OUTBOX.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.3)
            continue

        try:
            if item["type"] == "text" and _CHAT_ID and _APP:
                await _APP.bot.send_message(
                    chat_id=_CHAT_ID,
                    text=item["text"],
                    parse_mode=item.get("parse_mode"),
                    disable_web_page_preview=True,
                )
        except Exception as e:
            LOG.warning(f"[TG] send failed: {e}")

async def _amain(token: str):
    global _APP
    if not token:
        LOG.error("[TG] Missing TELEGRAM_BOT_TOKEN, Telegram will not start.")
        return

    _APP = ApplicationBuilder().token(token).build()
    _APP.add_handler(CommandHandler("start", cmd_start))
    _APP.add_handler(CommandHandler("menu", cmd_menu))
    _APP.add_handler(CallbackQueryHandler(on_button))

    # start pumper
    _APP.create_task(_outbox_pumper())

    # IMPORTANT: stop_signals=None avoids signal-handling in a non-main thread
    try:
        await _APP.run_polling(drop_pending_updates=True, stop_signals=None)
    except Exception as e:
        LOG.error(f"[TG] run_polling error: {e}")
