import os
import csv
import time
import asyncio
from datetime import datetime, timezone

import ccxt
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT"]
ACTIVE_SYMBOL = "BTC/USDT"

ORB_TIMEFRAME = "15m"
RETEST_TIMEFRAME = "5m"

MODE = "mock"
BOT_RUNNING = False
AI_MODE = "neutral"

TRADE_SIZE_USDT = 30.0
FEE_RATE = 0.001

POLL_SECONDS = 30

exchange = ccxt.kucoin({
    "enableRateLimit": True
})

open_position = None
last_trade_time = None

MOCK_LOG = "mock_trade_log.csv"


def ensure_logs():
    if not os.path.exists(MOCK_LOG):
        with open(MOCK_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "symbol",
                "side",
                "entry",
                "exit",
                "size_usdt",
                "fee_entry",
                "fee_exit",
                "pnl_usdt",
                "reason",
                "mode"
            ])


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fetch_candles(symbol, timeframe, limit=100):
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return [
        {
            "time": c[0],
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        }
        for c in candles
    ]


def get_orb(candles_15m):
    recent = candles_15m[-8:]
    orb = recent[0]

    return {
        "high": orb["high"],
        "low": orb["low"],
        "time": orb["time"]
    }


def ai_filter(symbol, breakout_strength):
    if AI_MODE == "aggressiv":
        return breakout_strength > 0.0005
    if AI_MODE == "försiktig":
        return breakout_strength > 0.002
    return breakout_strength > 0.001


def check_setup(symbol):
    candles_15m = fetch_candles(symbol, ORB_TIMEFRAME, 50)
    candles_5m = fetch_candles(symbol, RETEST_TIMEFRAME, 50)

    orb = get_orb(candles_15m)

    latest_5m = candles_5m[-1]
    prev_5m = candles_5m[-2]

    orb_high = orb["high"]
    orb_low = orb["low"]

    breakout = prev_5m["close"] > orb_high
    retest = latest_5m["low"] <= orb_high and latest_5m["close"] > orb_high
    bullish_close = latest_5m["close"] > latest_5m["open"]

    breakout_strength = (latest_5m["close"] - orb_high) / orb_high

    ai_ok = ai_filter(symbol, breakout_strength)

    if breakout and retest and bullish_close and ai_ok:
        return {
            "entry": latest_5m["close"],
            "stop": orb_low,
            "take_profit": latest_5m["close"] * 1.006,
            "orb_high": orb_high,
            "orb_low": orb_low
        }

    return None


def log_mock_trade(symbol, entry, exit_price, reason):
    global open_position

    size = TRADE_SIZE_USDT
    fee_entry = size * FEE_RATE
    gross_position = size - fee_entry

    qty = gross_position / entry
    exit_value = qty * exit_price
    fee_exit = exit_value * FEE_RATE

    pnl = exit_value - fee_exit - size

    with open(MOCK_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now(),
            symbol,
            "LONG",
            round(entry, 8),
            round(exit_price, 8),
            size,
            round(fee_entry, 6),
            round(fee_exit, 6),
            round(pnl, 6),
            reason,
            MODE
        ])

    return pnl


async def trading_loop(app):
    global open_position, last_trade_time

    ensure_logs()

    while True:
        try:
            if BOT_RUNNING and MODE == "mock":
                symbol = ACTIVE_SYMBOL

                if open_position is None:
                    setup = check_setup(symbol)

                    if setup:
                        open_position = {
                            "symbol": symbol,
                            "entry": setup["entry"],
                            "stop": setup["stop"],
                            "take_profit": setup["take_profit"],
                            "opened": now()
                        }

                        last_trade_time = now()

                else:
                    ticker = exchange.fetch_ticker(open_position["symbol"])
                    price = float(ticker["last"])

                    if price <= open_position["stop"]:
                        pnl = log_mock_trade(
                            open_position["symbol"],
                            open_position["entry"],
                            price,
                            "STOP_LOSS"
                        )
                        open_position = None

                    elif price >= open_position["take_profit"]:
                        pnl = log_mock_trade(
                            open_position["symbol"],
                            open_position["entry"],
                            price,
                            "TAKE_PROFIT"
                        )
                        open_position = None

        except Exception as e:
            print("Trading loop error:", e)

        await asyncio.sleep(POLL_SECONDS)


async def start_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING
    BOT_RUNNING = True
    await update.message.reply_text("✅ Mp ORBbot startad i mockläge.")


async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING
    BOT_RUNNING = False
    await update.message.reply_text("⛔ Mp ORBbot stoppad.")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pos = "Ingen öppen position"

    if open_position:
        pos = (
            f"Öppen LONG\n"
            f"Symbol: {open_position['symbol']}\n"
            f"Entry: {open_position['entry']}\n"
            f"Stop: {open_position['stop']}\n"
            f"TP: {open_position['take_profit']}"
        )

    text = (
        f"📊 Mp ORBbot Status\n\n"
        f"Running: {BOT_RUNNING}\n"
        f"Mode: {MODE}\n"
        f"Symbol: {ACTIVE_SYMBOL}\n"
        f"AI-läge: {AI_MODE}\n"
        f"ORB: {ORB_TIMEFRAME}\n"
        f"Retest: {RETEST_TIMEFRAME}\n"
        f"Trade size: {TRADE_SIZE_USDT} USDT\n"
        f"Fee: {FEE_RATE * 100:.2f}% per order\n"
        f"Senaste trade: {last_trade_time}\n\n"
        f"{pos}"
    )

    await update.message.reply_text(text)


async def set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_MODE

    if not context.args:
        await update.message.reply_text("Använd: /set_ai aggressiv | neutral | försiktig")
        return

    mode = context.args[0].lower()

    if mode not in ["aggressiv", "neutral", "försiktig"]:
        await update.message.reply_text("Fel AI-läge. Välj aggressiv, neutral eller försiktig.")
        return

    AI_MODE = mode
    await update.message.reply_text(f"✅ AI-läge satt till: {AI_MODE}")


async def set_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ACTIVE_SYMBOL

    if not context.args:
        await update.message.reply_text("Använd: /symbol BTCUSDT")
        return

    raw = context.args[0].upper().replace("/", "")

    formatted = raw.replace("USDT", "/USDT")

    if formatted not in SYMBOLS:
        await update.message.reply_text(f"Symbol stöds inte. Välj: {', '.join(SYMBOLS)}")
        return

    ACTIVE_SYMBOL = formatted
    await update.message.reply_text(f"✅ Aktiv symbol: {ACTIVE_SYMBOL}")


async def set_fee(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global FEE_RATE

    if not context.args:
        await update.message.reply_text("Använd: /set_fee 0.001")
        return

    try:
        FEE_RATE = float(context.args[0])
        await update.message.reply_text(f"✅ Avgift satt till {FEE_RATE * 100:.3f}% per order")
    except:
        await update.message.reply_text("Fel format. Exempel: /set_fee 0.001")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 Mp ORBbot kommandon

/start_bot - Starta mocktrading
/stop_bot - Stoppa boten
/status - Visa status
/symbol BTCUSDT - Byt coin
/set_ai aggressiv - AI-läge
/set_ai neutral
/set_ai försiktig
/set_fee 0.001 - Sätt avgift

Strategi:
15m ORB breakout
5m retest
Entry efter bullish close över ORB-high
"""
    await update.message.reply_text(text)


async def post_init(app):
    app.create_task(trading_loop(app))


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN saknas i .env")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start_bot", start_bot))
    app.add_handler(CommandHandler("stop_bot", stop_bot))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("set_ai", set_ai))
    app.add_handler(CommandHandler("symbol", set_symbol))
    app.add_handler(CommandHandler("set_fee", set_fee))
    app.add_handler(CommandHandler("help", help_cmd))

    print("Mp ORBbot körs...")
    app.run_polling()


if __name__ == "__main__":
    main()
