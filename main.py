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

# --------------------------------------------------------------------
# LOGGNING
# --------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("mporbbot")

# --------------------------------------------------------------------
# TELEGRAM-TOKEN (HÅRDKODAD)
# --------------------------------------------------------------------

BOT_TOKEN = "8265069090:AAF8W7l3-MNwyV8_DBRiLxUeEJHtGevURUg"

if not BOT_TOKEN:
    raise RuntimeError("Du måste sätta BOT_TOKEN i main.py till din riktiga Telegram-token!")

# --------------------------------------------------------------------
# Enkel “state” för bottens läge (mock/live, AI-mode osv.)
# --------------------------------------------------------------------


class BotState:
    def __init__(self) -> None:
        # AI-läge: 'neutral', 'aggressiv', 'försiktig'
        self.ai_mode: str = "neutral"

        # om trading är aktiv (mock eller live)
        self.trading_running: bool = False

        # mock-läge som standard
        self.mock_mode: bool = True

        # senaste info (status, PnL osv – kan fyllas på)
        self.last_status: str = "Bot startad, men ingen trading igång ännu."

    def status_text(self) -> str:
        mode_text = "MOCK" if self.mock_mode else "LIVE"
        running = "✅ På" if self.trading_running else "⏸ Stoppad"
        return (
            f"🤖 *Mp ORBbot*\n"
            f"• Trading: {running}\n"
            f"• Läge: {mode_text}\n"
            f"• AI-mode: `{self.ai_mode}`\n"
            f"• Info: {self.last_status}"
        )


bot_state = BotState()

# --------------------------------------------------------------------
# Telegram-application (python-telegram-bot v20+)
# --------------------------------------------------------------------

telegram_app: Optional[Application] = None
telegram_task: Optional[asyncio.Task] = None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Starta trading (mock-läge)."""
    user = update.effective_user
    logger.info("Mottog /start från %s (%s)", user.id if user else "okänd", user.username if user else "-")

    bot_state.trading_running = True
    text = (
        "🚀 Mp ORBbot är *startad*.\n\n"
        "Just nu kör vi i MOCK-läge (inga riktiga orders).\n"
        "Använd /status för att se läget och /stop för att stoppa."
    )
    if update.message:
        await update.message.reply_markdown(text)


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stoppa tradingen (men boten svarar fortfarande på kommandon)."""
    user = update.effective_user
    logger.info("Mottog /stop från %s (%s)", user.id if user else "okänd", user.username if user else "-")

    bot_state.trading_running = False
    if update.message:
        await update.message.reply_text("⏹ Trading stoppad. Boten lyssnar fortfarande på kommandon.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Visa status."""
    user = update.effective_user
    logger.info("Mottog /status från %s (%s)", user.id if user else "okänd", user.username if user else "-")

    if update.message:
        await update.message.reply_markdown(bot_state.status_text())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Lista kommandon."""
    user = update.effective_user
    logger.info("Mottog /help från %s (%s)", user.id if user else "okänd", user.username if user else "-")

    text = (
        "📜 *Kommandon*:\n"
        "/start – starta trading (mock-läge som standard)\n"
        "/stop – stoppa trading\n"
        "/status – visa status & läge\n"
        "/set_ai <neutral|aggressiv|försiktig> – ändra AI-läge\n"
        "/backtest SYMBOL TID [avgift] – kör backtest (ex: `/backtest btcusdt 3d 0.001`)\n"
        "/mock_trade SYMBOL – kör en manuell mock-trade\n"
        "/export_csv – exportera senaste backtest/mocktrades till CSV\n"
        "/help – denna hjälptext"
    )
    if update.message:
        await update.message.reply_markdown(text)


async def cmd_set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sätt AI-mode."""
    user = update.effective_user
    logger.info("Mottog /set_ai från %s (%s) args=%s",
                user.id if user else "okänd", user.username if user else "-", context.args)

    if not context.args:
        if update.message:
            await update.message.reply_text("Använd: /set_ai neutral|aggressiv|försiktig")
        return

    mode = context.args[0].lower()
    if mode not in {"neutral", "aggressiv", "försiktig"}:
        if update.message:
            await update.message.reply_text("Ogiltigt läge. Välj neutral, aggressiv eller försiktig.")
        return

    bot_state.ai_mode = mode
    if update.message:
        await update.message.reply_text(f"AI-läge ändrat till: {mode}")


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Kör backtest (placeholder)."""
    user = update.effective_user
    logger.info("Mottog /backtest från %s (%s) args=%s",
                user.id if user else "okänd", user.username if user else "-", context.args)

    if len(context.args) < 2:
        if update.message:
            await update.message.reply_text(
                "Använd: /backtest SYMBOL TID [avgift]\nEx: /backtest btcusdt 3d 0.001"
            )
        return

    symbol = context.args[0].upper()
    period = context.args[1]
    fee = float(context.args[2]) if len(context.args) >= 3 else 0.001

    logger.info("Backtest requested: %s %s fee=%s", symbol, period, fee)
    if update.message:
        await update.message.reply_text(
            f"🔧 (Dummy) kör backtest på {symbol} för {period} med avgift {fee:.4f}.\n"
            "Koppla in din riktiga backtest-logik i main.py senare."
        )


async def cmd_mock_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Kör en manuell mock-trade (placeholder)."""
    user = update.effective_user
    logger.info("Mottog /mock_trade från %s (%s) args=%s",
                user.id if user else "okänd", user.username if user else "-", context.args)

    if not context.args:
        if update.message:
            await update.message.reply_text("Använd: /mock_trade SYMBOL\nEx: /mock_trade BTCUSDT")
        return

    symbol = context.args[0].upper()
    logger.info("Mock trade requested: %s", symbol)
    if update.message:
        await update.message.reply_text(f"📈 (Dummy) mock-trade genomförd i {symbol} (ingen riktig order).")


async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Exportera CSV (placeholder)."""
    user = update.effective_user
    logger.info("Mottog /export_csv från %s (%s)",
                user.id if user else "okänd", user.username if user else "-")

    if update.message:
        await update.message.reply_text("📂 (Dummy) CSV-export kommer senare.")


# --------------------------------------------------------------------
# Telegram-runner
# --------------------------------------------------------------------


async def build_telegram_app() -> Application:
    """Skapar telegram-application och registrerar handlers."""
    logger.info("Bygger Telegram-application med token som slutar på ...%s", BOT_TOKEN[-6:])

    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("set_ai", cmd_set_ai))
    application.add_handler(CommandHandler("backtest", cmd_backtest))
    application.add_handler(CommandHandler("mock_trade", cmd_mock_trade))
    application.add_handler(CommandHandler("export_csv", cmd_export_csv))

    return application


async def telegram_main() -> None:
    """Kör telegram-boten tills processen stoppas."""
    global telegram_app
    telegram_app = await build_telegram_app()
    logger.info("Startar Telegram-bot (polling)…")
    # run_polling blockar, därför körs den i egen task från FastAPI-startup
    await telegram_app.run_polling(close_loop=False)


# --------------------------------------------------------------------
# FastAPI-app (DigitalOcean Web Service)
# --------------------------------------------------------------------

app = FastAPI(title="Mp ORBbot", version="1.0.0")


@app.get("/", response_class=PlainTextResponse)
async def root() -> str:
    """Enkel healthcheck-route."""
    return "Mp ORBbot up and running"


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "OK"


@app.on_event("startup")
async def on_startup() -> None:
    """Startas av uvicorn när containern bootar."""
    global telegram_task
    loop = asyncio.get_event_loop()
    telegram_task = loop.create_task(telegram_main())
    logger.info("Startup: Telegram-task skapad.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Försök stoppa telegram-boten snyggt."""
    global telegram_task, telegram_app

    logger.info("Shutdown påkallad, försöker stoppa Telegram-bot…")

    if telegram_task is not None:
        telegram_task.cancel()
        try:
            await telegram_task
        except asyncio.CancelledError:
            pass

    if telegram_app is not None:
        try:
            await telegram_app.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fel vid stop av telegram_app: %s", exc)


# --------------------------------------------------------------------
# Lokal körning (inte använd på DigitalOcean – där kör vi uvicorn via Run Command)
# --------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
