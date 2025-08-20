import os
import logging
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler

# ================== KONFIG ==================
BOT_TOKEN = os.getenv("BOT_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")

# Endast denne användare får styra (din ID nedan).
owner_env = os.getenv("OWNER_CHAT_ID", "").strip()
OWNER_CHAT_ID = int(owner_env) if owner_env else 5397586616

# ============================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = FastAPI()
telegram_app: Application = None  # Skapas vid startup


# ================== TELEGRAM KOMMANDON ==================
async def start(update: Update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        await update.message.reply_text("❌ Du har inte behörighet.")
        return
    await update.message.reply_text("🤖 Mp ORBbot startad! Använd /help för kommandon.")


async def help_command(update: Update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    text = (
        "/start – starta\n"
        "/help – hjälp\n"
        "/ping – pingtest\n"
        "/status – enkel status\n"
    )
    await update.message.reply_text(text)


async def ping(update: Update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    await update.message.reply_text("🏓 Pong!")


async def status(update: Update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    await update.message.reply_text("Status: online (webhook) ✅")


# ================== FASTAPI WEBHOOK ==================
@app.on_event("startup")
async def startup_event():
    global telegram_app
    telegram_app = (
        Application.builder()
        .token(BOT_TOKEN)
        .updater(None)  # Webhook-läge
        .build()
    )

    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("ping", ping))
    telegram_app.add_handler(CommandHandler("status", status))

    # Sätt webhook
    url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/webhook/{BOT_TOKEN}"
    await telegram_app.bot.set_webhook(url)
    logger.info(f"Sätter webhook: {url}")

    await telegram_app.initialize()
    await telegram_app.start()
    logger.info("Startup klar. Webhook-läge aktivt.")


@app.post("/webhook/{token}")
async def webhook(request: Request, token: str):
    if token != BOT_TOKEN:
        return {"error": "Invalid token"}
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}


@app.get("/")
async def root():
    return {"status": "Mp ORBbot körs!"}
