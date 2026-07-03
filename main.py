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
from logger import create_logs


# ==========================================
# Init
# ==========================================

exchange = ExchangeClient()

strategy = ORBStrategy(exchange)

trade_manager = TradeManager(exchange)

telegram_bot = TelegramBot(
    strategy,
    trade_manager
)


# ==========================================
# Trading loop
# ==========================================

async def trading_loop(app):

    notifier = Notifier(app)

    while True:

        try:

            if telegram_bot.engine_running:

                # ---------------------------------
                # Hantera öppna trades
                # ---------------------------------

                closed = trade_manager.manage_trades()

                for trade in closed:

                    await notifier.exit(trade)

                # ---------------------------------
                # Leta nya setups
                # ---------------------------------

                for symbol in SYMBOLS:

                    if not trade_manager.can_open(
                        symbol,
                        MAX_OPEN_TRADES
                    ):
                        continue

                    setup = strategy.check(symbol)

                    if setup is None:
                        continue

                    position = trade_manager.open_trade(
                        symbol,
                        setup
                    )

                    await notifier.entry(
                        symbol,
                        position
                    )

        except Exception as e:

            print("Trading loop:", e)

        await asyncio.sleep(POLL_SECONDS)


# ==========================================
# Telegram startup
# ==========================================

async def post_init(app):

    app.create_task(
        trading_loop(app)
    )


# ==========================================
# Main
# ==========================================

def main():

    create_logs()

    if not TELEGRAM_TOKEN:

        raise ValueError(
            "TELEGRAM_TOKEN saknas i .env"
        )

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    telegram_bot.register(app)

    print("")
    print("===========================")
    print(" MP ORB BOT STARTAD")
    print("===========================")
    print("Exchange : KuCoin")
    print("Mode     : Mock")
    print(f"Coins    : {len(SYMBOLS)}")
    print("===========================")
    print("")

    app.run_polling()


if __name__ == "__main__":
    main()
