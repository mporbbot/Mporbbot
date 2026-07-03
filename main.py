import asyncio

from telegram.ext import ApplicationBuilder

from config import (
    TELEGRAM_TOKEN,
    SYMBOLS,
    MAX_OPEN_TRADES,
    POLL_SECONDS,
)

from exchange_client import ExchangeClient
from strategy import ORBStrategy
from trade_manager import TradeManager
from telegram_bot import TelegramBot
from notifier import Notifier
from logger import ensure_logs


exchange = ExchangeClient()
strategy = ORBStrategy(exchange)
trade_manager = TradeManager(exchange)
telegram_bot = TelegramBot(strategy, trade_manager)


async def trading_loop(app):
    notifier = Notifier(app)

    while True:
        try:
            if telegram_bot.engine_running:

                closed_trades = trade_manager.manage_trades()

                for trade in closed_trades:
                    await notifier.exit(trade)

                for symbol in SYMBOLS:
                    if not trade_manager.can_open(symbol, MAX_OPEN_TRADES):
                        continue

                    setup = strategy.check(symbol)

                    if setup is None:
                        continue

                    position = trade_manager.open_trade(symbol, setup)

                    await notifier.entry(symbol, position)

        except Exception as e:
            print("Trading loop error:", e)
            try:
                notifier = Notifier(app)
                await notifier.error(str(e))
            except:
                pass

        await asyncio.sleep(POLL_SECONDS)


async def post_init(app):
    app.create_task(trading_loop(app))


def main():
    ensure_logs()

    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN saknas")

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    telegram_bot.register(app)

    print("MP ORB Bot running...")

    app.run_polling()


if __name__ == "__main__":
    main()
