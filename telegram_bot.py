# telegram_bot.py
import os
import asyncio
import logging
from contextlib import suppress
from typing import Optional, Set, Dict, Any

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

LOG = logging.getLogger("orb_v26")

# Miljövariabler
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0")) if os.getenv("OWNER_CHAT_ID") else None

# Delad outbox-kö injiceras från main via set_outbox(...)
_OUTBOX = None  # ska vara asyncio.Queue
_APP: Optional[Application] = None

# Lagra kända chats + en “senast använd”
_KNOWN_CHAT_IDS: Set[int] = set()
_LAST_CHAT_ID: Optional[int] = OWNER_CHAT_ID

# ===== Public API för main =====

def set_outbox(queue_obj):
    """Main anropar detta för att ge TG-modulen outbox-kön."""
    global _OUTBOX
    _OUTBOX = queue_obj

async def start_in_thread():
    """
    Starta Telegram-boten i den redan skapade event-loopen i en separat tråd.
    (Main tråd skapar ny event loop och kör _amain(token) där.)
    """
    if not TELEGRAM_BOT_TOKEN:
        LOG.error("[TG] Missing TELEGRAM_BOT_TOKEN, Telegram will not start.")
        return

    try:
        await _amain(TELEGRAM_BOT_TOKEN)
    except Exception as e:
        LOG.error(f"[TG] run error: {e}")

# ===== TG Handlers =====

def _menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Meny", callback_data="noop")],
    ])

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _LAST_CHAT_ID
    chat_id = update.effective_chat.id
    _KNOWN_CHAT_IDS.add(chat_id)
    _LAST_CHAT_ID = chat_id
    await update.message.reply_text(
        "Hej! Du är nu kopplad till ORB-boten.\n"
        "Jag skickar signaler till denna chatt.\n\n"
        "Kommandon: /menu /id",
        reply_markup=_menu_kb()
    )
    LOG.info(f"[TG] Registered chat_id {chat_id}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("OK.", reply_markup=_menu_kb())

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"Din chat_id: <code>{chat_id}</code>", parse_mode="HTML")

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer("👍")

# ===== Outbox pumper =====

async def _resolve_target_chat_id(item: Dict[str, Any]) -> Optional[int]:
    # Prioritet: item.chat_id -> OWNER_CHAT_ID -> _LAST_CHAT_ID -> valfri känd
    if "chat_id" in item and item["chat_id"]:
        return int(item["chat_id"])
    if OWNER_CHAT_ID:
        return OWNER_CHAT_ID
    if _LAST_CHAT_ID:
        return _LAST_CHAT_ID
    if _KNOWN_CHAT_IDS:
        return next(iter(_KNOWN_CHAT_IDS))
    return None

async def _outbox_pumper():
    if _OUTBOX is None:
        LOG.warning("[TG] Outbox not set; pumper idle.")
        return
    LOG.info("[TG] Outbox pumper running")
    while True:
        try:
            item = await _OUTBOX.get()
        except Exception:
            await asyncio.sleep(0.2)
            continue
        try:
            chat_id = await _resolve_target_chat_id(item)
            if not chat_id:
                LOG.debug("[TG] No chat_id yet; dropping message.")
                continue

            if item.get("type") == "text":
                await _APP.bot.send_message(
                    chat_id=chat_id,
                    text=item.get("text", ""),
                    parse_mode=item.get("parse_mode"),
                    reply_markup=item.get("reply_markup"),
                    disable_web_page_preview=True,
                )
        except Exception as e:
            LOG.warning(f"[TG] send fail: {e}")

# ===== Lifecycle =====

async def _amain(token: str):
    """
    Kör PTB v20 korrekt utan .idle() och utan att ta över huvudloop.
    """
    global _APP
    _APP = ApplicationBuilder().token(token).build()

    _APP.add_handler(CommandHandler("start", cmd_start))
    _APP.add_handler(CommandHandler("menu", cmd_menu))
    _APP.add_handler(CommandHandler("id", cmd_id))
    _APP.add_handler(CallbackQueryHandler(on_button))

    try:
        await _APP.initialize()
        await _APP.start()
        # run_polling styr själv event loop – undvik. Använd updater.start_polling
        await _APP.updater.start_polling(drop_pending_updates=True)

        # Starta outbox-pumparen inne i samma loop
        _APP.create_task(_outbox_pumper())

        LOG.info("[TG] Bot started & polling. Waiting forever…")
        # “idle” utan .idle(): vänta på ett Event som aldrig sätts
        forever = asyncio.Event()
        await forever.wait()

    except Exception as e:
        LOG.error(f"[TG] startup error: {e}")
    finally:
        LOG.info("[TG] shutting down…")
        with suppress(Exception):
            await _APP.updater.stop()
        with suppress(Exception):
            await _APP.stop()
        with suppress(Exception):
            await _APP.shutdown()
