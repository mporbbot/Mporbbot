from notifier import save_chat_id
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

from config import (
    SYMBOLS,
    MAX_OPEN_TRADES,
    TRADE_SIZE_USDT,
)
from logger import get_statistics


class TelegramBot:

    def __init__(self, strategy, trade_manager):

        self.strategy = strategy
        self.trade_manager = trade_manager

        self.engine_running = False

    # -----------------------------
    # ENGINE
    # -----------------------------

    async def engine_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        self.engine_running = True

        await update.message.reply_text(
            "✅ Engine ON"
        )

    async def engine_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        self.engine_running = False

        await update.message.reply_text(
            "⛔ Engine OFF"
        )

    # -----------------------------
    # STATUS
    # -----------------------------

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        stats = get_statistics()

        text = (
            "📊 Mp ORBBot\n\n"
            f"Engine: {'ON' if self.engine_running else 'OFF'}\n"
            f"Coins: {len(SYMBOLS)}\n"
            f"Open trades: {len(self.trade_manager.open_positions)}/{MAX_OPEN_TRADES}\n"
            f"Stake: {TRADE_SIZE_USDT} USDT\n\n"
            f"Trades: {stats['trades']}\n"
            f"Wins: {stats['wins']}\n"
            f"Winrate: {stats['winrate']}%\n"
            f"PnL: {stats['pnl']} USDT"
        )

        await update.message.reply_text(text)

    # -----------------------------
    # OPEN TRADES
    # -----------------------------

    async def open(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        await update.message.reply_text(
            self.trade_manager.open_summary()
        )

    # -----------------------------
    # COINS
    # -----------------------------

    async def coins(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        msg = "🪙 Coins\n\n"

        for coin in SYMBOLS:
            msg += f"{coin}\n"

        await update.message.reply_text(msg)

    # -----------------------------
    # DEBUG
    # -----------------------------

    async def debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        targets = SYMBOLS

        if context.args:

            coin = context.args[0].upper().replace("/", "")

            targets = [
                coin.replace("USDT", "/USDT")
            ]

        text = "🔍 Debug\n\n"

        for symbol in targets:

            try:

                self.strategy.check(symbol)

                d = self.strategy.debug(symbol)

                text += (
                    f"{symbol}\n"
                    f"Reason: {d['reason']}\n"
                    f"15m Breakout: {d['breakout']}\n"
                    f"5m Retest: {d['retest']}\n\n"
                )

            except Exception as e:

                text += f"{symbol}: {e}\n\n"

        await update.message.reply_text(text)

    # -----------------------------
    # HELP
    # -----------------------------

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        await update.message.reply_text(
            "🤖 Mp ORBBot\n\n"
            "/engine_on\n"
            "/engine_off\n"
            "/status\n"
            "/open\n"
            "/coins\n"
            "/debug\n"
            "/debug BTC\n"
            "/help"
        )

    # -----------------------------
    # REGISTER
    # -----------------------------

    def register(self, app):

        app.add_handler(
            CommandHandler(
                "engine_on",
                self.engine_on
            )
        )

        app.add_handler(
            CommandHandler(
                "engine_off",
                self.engine_off
            )
        )

        app.add_handler(
            CommandHandler(
                "status",
                self.status
            )
        )

        app.add_handler(
            CommandHandler(
                "open",
                self.open
            )
        )

        app.add_handler(
            CommandHandler(
                "coins",
                self.coins
            )
        )

        app.add_handler(
            CommandHandler(
                "debug",
                self.debug
            )
        )

        app.add_handler(
            CommandHandler(
                "help",
                self.help
            )
        )
