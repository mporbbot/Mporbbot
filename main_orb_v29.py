"""
Mp ORBbot – webhook + ägarskydd (endast Magnus får styra)
- FastAPI för webhook
- Telegram Application (python-telegram-bot v20)
- Enkel kommandomeny
- Åtkomstspärr: endast OWNER_CHAT_ID får köra kommandon
"""

from __future__ import annotations
import os
import logging
from typing import Callable, Awaitable, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    ContextTypes
)

# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("orb_v29_webhook")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN saknas i env.")

# Endast denne användare får styra (din ID nedan). Kan överskridas via env OWNER_CHAT_ID
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "5397586616"))

# Render sätter PORT i env
PORT = int(os.getenv("PORT", "10000"))

# Bas-URL till tjänsten (för att sätta webhook). Fylls för Render automatiskt.
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", os.getenv("PING_URL", "")).strip()
# Ex: https://mporbbot.onrender.com
# Om den är tom, räknar vi med att Invoke/Deploy sätter den innan start.
# Vi kan fortfarande ta emot råa POST:ar om Telegram redan har rätt webhook.

WEBHOOK_PATH = f"/webhook/{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL = (RENDER_URL + WEBHOOK_PATH) if RENDER_URL else ""

# -----------------------------------------------------------------------------
# Access control – dekorator
# -----------------------------------------------------------------------------

def only_owner(func: Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[Any]]):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        uid = user.id if user else None
        if uid != OWNER_CHAT_ID:
            try:
                # Svara artigt men inte för informativt
                if update.message:
                    await update.message.reply_text("❌ Åtkomst nekad.")
                elif update.callback_query:
                    await update.callback_query.answer("Åtkomst nekad.", show_alert=True)
            finally:
                log.warning("Access denied for user_id=%s", uid)
            return
        return await func(update, context)
    return wrapper

# -----------------------------------------------------------------------------
# Telegram Application (byggs i startup)
# -----------------------------------------------------------------------------

app = FastAPI(title="Mp ORBbot (webhook, owner-locked)")

tg_app: Application | None = None  # fylls i startup


def main_menu_markup() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_start"), KeyboardButton("/engine_stop")],
        [KeyboardButton("/start_mock"), KeyboardButton("/start_live")],
        [KeyboardButton("/entry_mode"), KeyboardButton("/trailing")],
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# -------------------- Kommandon --------------------

@only_owner
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Hej! 👋 Mp ORBbot är online (webhook).\n\n"
        "/help – hjälp\n"
        "/ping – pingtest\n"
        "/status – enkel status\n"
    )
    await update.message.reply_text(txt, reply_markup=main_menu_markup())

@only_owner
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Tillgängliga kommandon:\n"
        "/start – starta\n"
        "/help – hjälp\n"
        "/ping – pingtest\n"
        "/status – enkel status\n"
        "/engine_start, /engine_stop\n"
        "/start_mock, /start_live\n"
        "/entry_mode, /trailing\n"
        "/pnl, /reset_pnl\n"
        "/orb_on, /orb_off\n"
        "/panic"
    )

@only_owner
async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong 🏓")

@only_owner
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Status: online (webhook) ✅", reply_markup=main_menu_markup())

# Följande kommandon är “stubbar” – de svarar och är spärrade till ägaren.
# Koppla dem till din trading-motor i samma handler om du vill.

@only_owner
async def cmd_engine_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Engine: START (stub)")

@only_owner
async def cmd_engine_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Engine: STOP (stub)")

@only_owner
async def cmd_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Mock-läge startat (stub)")

@only_owner
async def cmd_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Live-läge startat (stub)")

@only_owner
async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Entry mode (stub) – här kopplar du din ORB-logik.")

@only_owner
async def cmd_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Trailing-inställning (stub).")

@only_owner
async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PnL (stub).")

@only_owner
async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PnL reset (stub).")

@only_owner
async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ORB: ON (stub)")

@only_owner
async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ORB: OFF (stub)")

@only_owner
async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PANIC (stub). Stäng alla! (koppla mot din motor)")

# -----------------------------------------------------------------------------
# FastAPI routes
# -----------------------------------------------------------------------------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Mp ORBbot up (webhook)."

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    """Mottar uppdateringar från Telegram."""
    if tg_app is None:
        return Response(status_code=500)
    data = await request.json()
    update: Update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return Response(status_code=200)

# -----------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global tg_app
    log.info("Startup: bygger Telegram Application…")

    tg_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("help", cmd_help))
    tg_app.add_handler(CommandHandler("ping", cmd_ping))
    tg_app.add_handler(CommandHandler("status", cmd_status))

    tg_app.add_handler(CommandHandler("engine_start", cmd_engine_start))
    tg_app.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    tg_app.add_handler(CommandHandler("start_mock", cmd_start_mock))
    tg_app.add_handler(CommandHandler("start_live", cmd_start_live))
    tg_app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    tg_app.add_handler(CommandHandler("trailing", cmd_trailing))
    tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
    tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
    tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
    tg_app.add_handler(CommandHandler("panic", cmd_panic))

    # Sätt webhook om vi har extern URL
    if WEBHOOK_URL:
        await tg_app.bot.set_webhook(WEBHOOK_URL, allowed_updates=["message"])
        log.info("Webhook satt: %s", WEBHOOK_URL)
    else:
        log.warning("RENDER_EXTERNAL_URL/PING_URL saknas – webhook sätts inte nu.")

    log.info("Application started (owner=%s).", OWNER_CHAT_ID)

@app.on_event("shutdown")
async def on_shutdown():
    if tg_app:
        await tg_app.bot.delete_webhook(drop_pending_updates=False)
        await tg_app.shutdown()
        await tg_app.stop()
    log.info("Shutdown klart.")
