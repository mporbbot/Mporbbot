# main_orb_v32.py
import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - v32 - %(levelname)s - %(message)s"
)
log = logging.getLogger("v32")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

WEBHOOK_SECRET_PATH = f"/webhook/{BOT_TOKEN}"  # unikt för just denna bot
app = FastAPI()

# --- enkel “motor”-state i minnet ---
STATE = {
    "engine_on": False,
    "mode": "mock",   # bara etikett
}

def owner_guard(update: Update) -> bool:
    chat_id = update.effective_chat.id if update.effective_chat else 0
    if OWNER_CHAT_ID and chat_id != OWNER_CHAT_ID:
        log.info(f"Blockerad användare: {chat_id} (OWNER={OWNER_CHAT_ID})")
        return False
    return True

# --- handlers ---
async def cmd_start(update: Update, _):
    if not owner_guard(update):
        await update.effective_chat.send_message("❌ Du är inte behörig.")
        return
    await update.effective_chat.send_message(
        "🔧 ORB v32 startad.\n"
        "Använd /engine_on för att börja.\n"
        "Prova även /help, /status, /ping.",
    )

async def cmd_help(update: Update, _):
    if not owner_guard(update):
        await update.effective_chat.send_message("❌ Du är inte behörig.")
        return
    text = (
        "🧭 **Kommandon**\n"
        "/start – starta boten\n"
        "/help – denna hjälp\n"
        "/engine_on – aktivera engine\n"
        "/engine_off – stäng av engine\n"
        "/status – visa status\n"
        "/ping – pingtest\n"
        "/panic – (dummy) stänger av engine\n"
    )
    await update.effective_chat.send_message(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_engine_on(update: Update, _):
    if not owner_guard(update):
        await update.effective_chat.send_message("❌ Du är inte behörig.")
        return
    STATE["engine_on"] = True
    await update.effective_chat.send_message("✅ Engine är nu **AKTIV** (mock-läge).", parse_mode=ParseMode.MARKDOWN)

async def cmd_engine_off(update: Update, _):
    if not owner_guard(update):
        await update.effective_chat.send_message("❌ Du är inte behörig.")
        return
    STATE["engine_on"] = False
    await update.effective_chat.send_message("⛔️ Engine är nu **AV**.", parse_mode=ParseMode.MARKDOWN)

async def cmd_status(update: Update, _):
    if not owner_guard(update):
        await update.effective_chat.send_message("❌ Du är inte behörig.")
        return
    await update.effective_chat.send_message(
        f"Status: engine_on={STATE['engine_on']} | mode={STATE['mode']}"
    )

async def cmd_ping(update: Update, _):
    if not owner_guard(update):
        await update.effective_chat.send_message("❌ Du är inte behörig.")
        return
    await update.effective_chat.send_message("pong ✅")

async def cmd_panic(update: Update, _):
    if not owner_guard(update):
        await update.effective_chat.send_message("❌ Du är inte behörig.")
        return
    STATE["engine_on"] = False
    await update.effective_chat.send_message("🛑 PANIC utförd. Engine OFF.")

async def echo_any(update: Update, _):
    # Fallback för att bekräfta att webhooken fungerar
    chat_id = update.effective_chat.id if update.effective_chat else 0
    log.info(f"Echo från {chat_id}: {update.effective_message.text!r}")
    if not owner_guard(update):
        await update.effective_chat.send_message("❌ Du är inte behörig.")
        return
    await update.effective_chat.send_message("Jag är online ✅ (skriv /help).")

# --- bygg Telegram Application ---
def build_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN saknas.")
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("engine_on", cmd_engine_on))
    application.add_handler(CommandHandler("engine_off", cmd_engine_off))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("ping", cmd_ping))
    application.add_handler(CommandHandler("panic", cmd_panic))
    # fallback på all text
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_any))
    return application

tg_app: Application = build_app()

# --- FastAPI endpoints ---
@app.on_event("startup")
async def on_startup():
    log.info("Initierar Telegram Application…")
    await tg_app.initialize()
    await tg_app.bot.set_webhook(url=f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'mporbbot.onrender.com')}{WEBHOOK_SECRET_PATH}")
    await tg_app.start()
    log.info(f"Webhook satt: https://{os.getenv('RENDER_EXTERNAL_HOSTNAME','mporbbot.onrender.com')}{WEBHOOK_SECRET_PATH}")

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    return PlainTextResponse("Mporbbot v32 – OK")

@app.post(WEBHOOK_SECRET_PATH)
async def telegram_webhook(request: Request):
    payload = await request.json()
    log.info(f"Update in: {payload}")
    update = Update.de_json(data=payload, bot=tg_app.bot)
    await tg_app.process_update(update)
    return PlainTextResponse("OK")
