# main.py
import os
import asyncio
import logging
from typing import Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# -------------------------
# Konfig / Environment
# -------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN")

OWNER_CHAT_ID_ENV = os.getenv("OWNER_CHAT_ID", "").strip()
try:
    OWNER_CHAT_ID = int(OWNER_CHAT_ID_ENV) if OWNER_CHAT_ID_ENV else None
except ValueError:
    OWNER_CHAT_ID = None

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")  # t.ex. https://mporbbot.onrender.com
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

# 0 = mock, 1 = live (bara som flagga – själva orderlogik ligger utanför denna main)
LIVE = 1 if os.getenv("LIVE", "0").strip() in ("1", "true", "True") else 0

# Default-symboler och entry-mode
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").replace(" ", "")
EntryMode = Literal["CLOSE", "TICK"]
ENTRY_MODE_DEFAULT: EntryMode = "CLOSE"  # <-- Startar i CLOSE som du önskade

# -------------------------
# Globalt läge / state
# -------------------------
ENGINE_ON: bool = False          # starta alltid i OFF för säkerhet
ENTRY_MODE: EntryMode = ENTRY_MODE_DEFAULT

# (Valfritt) Något statusfält för framtida tradeloop
RUNNING: bool = False

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("orb-main")

# -------------------------
# FastAPI & Telegram App
# -------------------------
app = FastAPI()
tg_app: Optional[Application] = None

def only_owner(func):
    """Decorator: tillåt endast OWNER_CHAT_ID om satt."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if OWNER_CHAT_ID is not None:
            chat_id = update.effective_chat.id if update.effective_chat else None
            if chat_id != OWNER_CHAT_ID:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⛔️ Otillåten. Denna bot är låst till ägaren."
                )
                return
        return await func(update, context)
    return wrapper

# -------------------------
# Kommandon
# -------------------------
@only_owner
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON, ENTRY_MODE, RUNNING
    await update.message.reply_text(
        "\n".join([
            "🤖 MporbBot v33 (minimal main för kontroll)",
            f"Engine: {'ON' if ENGINE_ON else 'OFF'}",
            f"Entry-mode: {ENTRY_MODE}",
            f"Run-state: {'RUNNING' if RUNNING else 'IDLE'}",
            f"Symbols: {SYMBOLS}",
            f"Live-flagga: {LIVE}",
            "",
            "Kommandon:",
            "/engine_on – aktivera trading",
            "/engine_off – stoppa trading",
            "/set_entry close|tick – byt entry-läge",
            "/status – visa läge",
            "/stop – nödstoppa (Engine OFF)",
        ])
    )

@only_owner
async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON
    ENGINE_ON = True
    await update.message.reply_text("✅ Engine ON – trading kan köras.")

@only_owner
async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON, RUNNING
    ENGINE_ON = False
    RUNNING = False
    await update.message.reply_text("🛑 Engine OFF – all trading stoppad.")

@only_owner
async def cmd_set_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENTRY_MODE
    args = [a.strip().upper() for a in context.args] if context.args else []
    if not args or args[0] not in ("CLOSE", "TICK"):
        await update.message.reply_text("Använd: /set_entry close eller /set_entry tick")
        return
    ENTRY_MODE = "CLOSE" if args[0] == "CLOSE" else "TICK"
    await update.message.reply_text(f"🔁 Entry-mode satt till: {ENTRY_MODE}")

@only_owner
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        f"📊 Status\n"
        f"Engine: {'ON' if ENGINE_ON else 'OFF'}\n"
        f"Entry-mode: {ENTRY_MODE}\n"
        f"Running: {'RUNNING' if RUNNING else 'IDLE'}\n"
        f"Symbols: {SYMBOLS}\n"
        f"Live: {LIVE}\n"
    )
    await update.message.reply_text(txt)

@only_owner
async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Nödstop – stäng av Engine direkt."""
    global ENGINE_ON, RUNNING
    ENGINE_ON = False
    RUNNING = False
    await update.message.reply_text("⛔️ STOP utförd: Engine OFF och körning avbruten.")

# -------------------------
# FastAPI endpoints
# -------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.get("/healthz", response_class=JSONResponse)
async def healthz():
    return {
        "ok": True,
        "engine_on": ENGINE_ON,
        "entry_mode": ENTRY_MODE,
        "live": LIVE,
    }

@app.post("/webhook/{token}")
async def webhook(request: Request, token: str):
    if token != BOT_TOKEN:
        return PlainTextResponse("Bad token", status_code=403)
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    # Viktigt i webhookläge: Application måste vara initialiserad
    await tg_app.process_update(update)
    return PlainTextResponse("OK")

# -------------------------
# Startup / Shutdown
# -------------------------
@app.on_event("startup")
async def on_startup():
    global tg_app

    # Bygg Telegram Application
    tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Lägg på handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    tg_app.add_handler(CommandHandler("set_entry", cmd_set_entry))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(CommandHandler("stop", cmd_stop))

    # Initiera appen (krävs för webhook + process_update)
    await tg_app.initialize()

    # Sätt webhook till denna tjänst
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    await tg_app.bot.set_webhook(url)
    log.info(f"Webhook satt: {url}")

@app.on_event("shutdown")
async def on_shutdown():
    if tg_app:
        await tg_app.shutdown()
        await tg_app.stop()
