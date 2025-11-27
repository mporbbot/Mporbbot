import os
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mporbbot")

# ---------------------------------------------------------
# TOKEN
# ---------------------------------------------------------
TELEGRAM_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("BOT_TOKEN")
)

if not TELEGRAM_TOKEN:
    raise RuntimeError("❌ Ingen TELEGRAM_BOT_TOKEN hittades i Environment Variables!")

# ---------------------------------------------------------
# BOTTENS STATE
# ---------------------------------------------------------
class BotState:
    def __init__(self):
        self.ai_mode = "neutral"
        self.trading_running = False
        self.mock_mode = True
        self.last_status = "Bot startad – ingen trading ännu."

    def status_text(self):
        mode = "MOCK" if self.mock_mode else "LIVE"
        active = "🔵 Aktiv" if self.trading_running else "⏸️ Stoppad"
        return (
            f"🤖 *Mp ORBbot*\n"
            f"• Trading: {active}\n"
            f"• Läge: {mode}\n"
            f"• AI-mode: `{self.ai_mode}`\n"
            f"• Info: {self.last_status}"
        )

state = BotState()

# ---------------------------------------------------------
# KOMMANDON
# ---------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.trading_running = True
    await update.message.reply_text("🚀 Boten är startad (MOCK-läge).")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.trading_running = False
    await update.message.reply_text("⏹ Trading stoppad.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_markdown(state.status_text())

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_markdown(
        "📜 *Kommandon:*\n"
        "/start – starta trading\n"
        "/stop – stoppa trading\n"
        "/status – visa status\n"
        "/set_ai <läge>\n"
        "/backtest SYMBOL TID\n"
        "/mock_trade SYMBOL\n"
        "/export_csv\n"
        "/help"
    )

async def cmd_set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Använd: /set_ai neutral|aggressiv|försiktig")
    mode = context.args[0].lower()
    if mode not in ["neutral", "aggressiv", "försiktig"]:
        return await update.message.reply_text("Ogiltigt AI-läge.")
    state.ai_mode = mode
    await update.message.reply_text(f"AI-läge ändrat till {mode}")

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Backtest placeholder.")

async def cmd_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Mock trade placeholder.")

async def cmd_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("CSV export placeholder.")

# ---------------------------------------------------------
# TELEGRAM START (EXTREMT VIKTIG FÖR DIGITALOCEAN)
# ---------------------------------------------------------
telegram_app: Optional[Application] = None
telegram_task: Optional[asyncio.Task] = None

async def telegram_runner():
    global telegram_app

    telegram_app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    telegram_app.add_handler(CommandHandler("start", cmd_start))
    telegram_app.add_handler(CommandHandler("stop", cmd_stop))
    telegram_app.add_handler(CommandHandler("help", cmd_help))
    telegram_app.add_handler(CommandHandler("status", cmd_status))
    telegram_app.add_handler(CommandHandler("set_ai", cmd_set_ai))
    telegram_app.add_handler(CommandHandler("backtest", cmd_backtest))
    telegram_app.add_handler(CommandHandler("mock_trade", cmd_mock))
    telegram_app.add_handler(CommandHandler("export_csv", cmd_csv))

    logger.info("📡 Startar Telegram polling…")
    await telegram_app.run_polling(close_loop=False)

# ---------------------------------------------------------
# FASTAPI (krav för DO)
# ---------------------------------------------------------
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Mp ORBbot up and running!"

@app.on_event("startup")
async def startup_event():
    global telegram_task
    loop = asyncio.get_event_loop()
    telegram_task = loop.create_task(telegram_runner())
    logger.info("Startup klart → Telegram-task skapad.")

@app.get("/health")
async def health():
    return "OK"

@app.on_event("shutdown")
async def shutdown_event():
    global telegram_app, telegram_task

    if telegram_task:
        telegram_task.cancel()
        try:
            await telegram_task
        except Exception:
            pass

    if telegram_app:
        try:
            await telegram_app.stop()
        except:
            pass
