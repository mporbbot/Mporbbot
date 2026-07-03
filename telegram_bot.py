from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

from config import SYMBOLS
from logger import get_stats, get_history
from storage import save_chat_id


class TelegramBot:
    def __init__(self, strategy, trade_manager):
        self.strategy = strategy
        self.trade_manager = trade_manager
        self.engine_running = False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        save_chat_id(chat_id)
        await update.message.reply_text(
            f"✅ Chat ID sparat\n\n"
            f"ID: {chat_id}\n\n"
            f"Kör /engine_on för att starta boten."
        )

    async def engine_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.engine_running = True
        await update.message.reply_text("✅ Engine ON")

    async def engine_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.engine_running = False
        await update.message.reply_text("⛔ Engine OFF")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = get_stats()

        await update.message.reply_text(
            f"📊 MP ORB Bot\n\n"
            f"Engine: {'ON' if self.engine_running else 'OFF'}\n"
            f"Coins: {len(SYMBOLS)}\n"
            f"Open trades: {len(self.trade_manager.open_positions)}\n\n"
            f"Trades: {stats['trades']}\n"
            f"Wins: {stats['wins']}\n"
            f"Winrate: {stats['winrate']}%\n"
            f"PnL: {stats['pnl']} USDT"
        )

    async def open(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            self.trade_manager.open_summary()
        )

    async def coins(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🪙 Coins\n\n" + "\n".join(SYMBOLS)
        )

    async def pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = get_stats()

        await update.message.reply_text(
            f"📈 PnL\n\n"
            f"Trades: {stats['trades']}\n"
            f"Wins: {stats['wins']}\n"
            f"Winrate: {stats['winrate']}%\n"
            f"PnL: {stats['pnl']} USDT"
        )

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        rows = get_history(20)

        if not rows:
            await update.message.reply_text("Ingen tradehistorik ännu.")
            return

        msg = "📜 Senaste trades\n\n"

        for r in rows:
            msg += (
                f"{r['time']} {r['symbol']}\n"
                f"{r['exit_reason']} | {r['entry_type']}\n"
                f"PnL: {r['pnl_usdt']} USDT\n\n"
            )

        await update.message.reply_text(msg)

    async def debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        targets = SYMBOLS

        if context.args:
            raw = context.args[0].upper().replace("/", "")
            targets = [raw.replace("USDT", "/USDT")]

        msg = "🔍 Debug\n\n"

        for symbol in targets:
            try:
                self.strategy.check(symbol)
                d = self.strategy.debug(symbol)

                msg += (
                    f"{symbol}\n"
                    f"Reason: {d['reason']}\n"
                    f"Breakout: {d['breakout_seen']}\n"
                    f"Retest: {d['retest_seen']}\n"
                    f"ORB High: {d['orb_high']}\n"
                    f"ORB Low: {d['orb_low']}\n\n"
                )

            except Exception as e:
                msg += f"{symbol}: ERROR {e}\n\n"

        await update.message.reply_text(msg)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 MP ORB Bot\n\n"
            "/start\n"
            "/engine_on\n"
            "/engine_off\n"
            "/status\n"
            "/open\n"
            "/pnl\n"
            "/history\n"
            "/coins\n"
            "/debug\n"
            "/debug ETH\n"
            "/help"
        )

    def register(self, app):
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("engine_on", self.engine_on))
        app.add_handler(CommandHandler("engine_off", self.engine_off))
        app.add_handler(CommandHandler("status", self.status))
        app.add_handler(CommandHandler("open", self.open))
        app.add_handler(CommandHandler("pnl", self.pnl))
        app.add_handler(CommandHandler("history", self.history))
        app.add_handler(CommandHandler("coins", self.coins))
        app.add_handler(CommandHandler("debug", self.debug))
        app.add_handler(CommandHandler("help", self.help))
