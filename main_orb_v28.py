# main_orb_v28.py  — webhook-version (FastAPI + python-telegram-bot v20)
import os
import logging
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

# ──────────────────────────────────────────────────────────────────────────────
# Loggning
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("orb_v28_webhook")

# ──────────────────────────────────────────────────────────────────────────────
# Konfiguration (ENV)
# ──────────────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://mporbbot.onrender.com").rstrip("/")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing env TELEGRAM_BOT_TOKEN")

WEBHOOK_PATH = f"/webhook/{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL = f"{PUBLIC_URL}{WEBHOOK_PATH}"

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI-app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Mporbbot ORB v28 (webhook)")

# Global PTB Application (en instans!)
tg_app: Optional[Application] = None


# ──────────────────────────────────────────────────────────────────────────────
# Telegram handlers (exempel – lägg till dina egna här)
# ──────────────────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, _context):
    await update.effective_message.reply_text("👋 Botten kör i webhook-läge och är online.")

async def echo(update: Update, _context):
    # Minimal fallback så du ser att webhook fungerar
    if update.effective_message:
        await update.effective_message.reply_text(f"Du sa: {update.effective_message.text[:200]}")


def register_handlers(app_: Application) -> None:
    app_.add_handler(CommandHandler("start", cmd_start))
    # Lägg gärna till dina riktiga kommandon/handlers här
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan: initiera PTB, sätt webhook, starta/stoppa snyggt
# ──────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    global tg_app
    log.info("Startup: bygger Telegram Application…")

    tg_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    register_handlers(tg_app)

    # Initiera PTB-delar
    await tg_app.initialize()

    # Sätt webhook på Telegram (droppa ev. köade uppdateringar)
    log.info(f"Sätter webhook: {WEBHOOK_URL}")
    await tg_app.bot.set_webhook(
        url=WEBHOOK_URL,
        drop_pending_updates=True,
        allowed_updates=None,  # ta emot allt
    )

    # Starta bakgrundsdelar (job queues etc). INTE polling.
    await tg_app.start()
    log.info("Startup klar. Webhook-läge aktivt.")


@app.on_event("shutdown")
async def on_shutdown():
    global tg_app
    if tg_app is None:
        return
    log.info("Shutdown: stoppar Telegram Application…")
    # Stäng ner prydligt
    await tg_app.stop()
    await tg_app.shutdown()
    log.info("Shutdown klar.")


# ──────────────────────────────────────────────────────────────────────────────
# HTTP routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Mporbbot ORB v28 webhook OK"

@app.get("/status")
async def status():
    return {"webhook_url": WEBHOOK_URL, "running": tg_app is not None}

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    """
    Telegram skickar uppdateringar hit. Vi matar dem direkt till PTB Application.
    """
    global tg_app
    if tg_app is None:
        log.error("Webhook kallades innan tg_app var initierad.")
        return Response(status_code=503)

    data = await request.json()
    update = Update.de_json(data, tg_app.bot)

    # Processera EN uppdatering
    await tg_app.process_update(update)
    return JSONResponse({"ok": True})
