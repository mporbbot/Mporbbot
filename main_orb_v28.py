# main_orb_v28.py
import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("orb_v28_webhook")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
PUBLIC_BASE_URL    = os.getenv("PUBLIC_BASE_URL", "https://mporbbot.onrender.com").rstrip("/")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing env var TELEGRAM_BOT_TOKEN")

app = FastAPI()
tg_app: Application | None = None

# ── Telegram handlers ──────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Hej! 👋 Jag är online. Prova /help eller /ping."
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "/start – starta\n"
        "/help – hjälp\n"
        "/ping – pingtest\n"
        "/status – enkel status"
    )

async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("pong ✅")

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Status: online (webhook) ✅")

async def log_all(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Logga ALLT som kommer in så vi ser att handlers nås
    chat = update.effective_chat.id if update.effective_chat else "unknown"
    text = (update.effective_message.text if update.effective_message else None)
    log.info(f"Update från chat {chat} | text={text!r}")
    # (Valfritt) auto-svar för vanliga textmeddelanden:
    if text and not text.startswith("/"):
        await update.effective_message.reply_text("Jag är här 👋 (skriv /help)")

# ── FastAPI lifecycle ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    global tg_app
    log.info("Startup: bygger Telegram Application…")

    tg_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)  # robust i webhookläge
        .build()
    )

    # Registrera handlers
    tg_app.add_handler(CommandHandler("start",  cmd_start))
    tg_app.add_handler(CommandHandler("help",   cmd_help))
    tg_app.add_handler(CommandHandler("ping",   cmd_ping))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(MessageHandler(filters.ALL, log_all))

    # Sätt Webhook
    webhook_url = f"{PUBLIC_BASE_URL}/webhook/{TELEGRAM_BOT_TOKEN}"
    await tg_app.bot.set_webhook(
        url=webhook_url,
        drop_pending_updates=True,   # rensa kö om det fanns gamla
        allowed_updates=["message", "edited_message", "callback_query"]
    )
    await tg_app.initialize()
    await tg_app.start()
    log.info(f"Webhook satt: {webhook_url}")
    log.info("Startup klar. Webhook-läge aktivt.")

@app.on_event("shutdown")
async def on_shutdown():
    if tg_app:
        await tg_app.stop()
        await tg_app.shutdown()
        log.info("Telegram Application stoppad.")

# ── Hälso-endpoints ───────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("healthy")

# ── Telegram webhook endpoint (VIKTIG) ────────────────────────────────────────
@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token in path")
    if tg_app is None:
        raise HTTPException(status_code=503, detail="Telegram app not ready")

    data = await request.json()
    update = Update.de_json(data, tg_app.bot)

    # OBS: det är *denna* raden som gör att handlers körs
    await tg_app.process_update(update)

    return PlainTextResponse("OK")
