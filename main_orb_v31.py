import os
import logging
from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import Application, CommandHandler

# === Logging ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("orb_v31")

# === Miljövariabler ===
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "5397586616"))

# Render host (för webhook)
RENDER_EXTERNAL_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME", "mporbbot.onrender.com")
WEBHOOK_URL = f"https://{RENDER_EXTERNAL_HOSTNAME}/webhook/{TOKEN}"

# === Telegram Application ===
tg_app = Application.builder().token(TOKEN).build()

# === FastAPI ===
app = FastAPI()


# === Kommandon ===
async def start(update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        await update.message.reply_text("⛔ Du har inte behörighet att styra denna bot.")
        return
    await update.message.reply_text("🔧 ORB v31 startad. Använd /engine_on för att börja.")


async def help_cmd(update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    await update.message.reply_text(
        "📋 Kommandon:\n"
        "/start – starta boten\n"
        "/engine_on – aktivera trading engine\n"
        "/engine_off – stäng av engine\n"
        "/help – denna hjälp"
    )


async def engine_on(update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    await update.message.reply_text("✅ Engine är nu AKTIV (mock-läge).")


async def engine_off(update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    await update.message.reply_text("🛑 Engine stoppad.")


# === Registrera handlers ===
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("help", help_cmd))
tg_app.add_handler(CommandHandler("engine_on", engine_on))
tg_app.add_handler(CommandHandler("engine_off", engine_off))


# === Startup & Shutdown ===
@app.on_event("startup")
async def on_startup():
    logger.info("Initierar Telegram Application...")
    await tg_app.initialize()
    await tg_app.start()
    await tg_app.bot.set_webhook(WEBHOOK_URL)
    logger.info(f"Webhook satt: {WEBHOOK_URL}")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Stänger ned Telegram Application...")
    try:
        await tg_app.bot.delete_webhook()
    except Exception as e:
        logger.warning(f"Kunde inte ta bort webhook: {e}")
    await tg_app.stop()
    await tg_app.shutdown()


# === Webhook endpoint ===
@app.post(f"/webhook/{TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return Response(status_code=200)
