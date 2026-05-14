import os
import csv
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
RISK_PERCENT = 1.0

BREAKOUT_BUFFER = 0.001
TRAIL_PERCENT = 0.003
COOLDOWN_MINUTES = 5
SL_ATR_MULTIPLIER = 1.5

USE_TREND_FILTER = True
USE_SWEEP_FILTER = False
USE_RECLAIM_FILTER = True

POLL_SECONDS = 30

MOCK_LOG = "mock_trade_log.csv"

exchange = ccxt.kucoin({"enableRateLimit": True})

open_position = None
last_trade_time = None


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def ensure_logs():
    if not os.path.exists(MOCK_LOG):
        with open(MOCK_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "symbol", "side", "entry", "exit",
                "size_usdt", "fee_entry", "fee_exit",
                "pnl_usdt", "reason", "mode"
            ])


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


def ai_filter(breakout_strength):
    if AI_MODE == "aggressiv":
        return breakout_strength > 0.0005
    if AI_MODE == "försiktig":
        return breakout_strength > 0.002
    return breakout_strength > 0.001


def trend_ok(candles):
    if not USE_TREND_FILTER:
        return True
    closes = [c["close"] for c in candles[-20:]]
    return closes[-1] > sum(closes) / len(closes)


def check_setup(symbol):
    candles_15m = fetch_candles(symbol, ORB_TIMEFRAME, 50)
    candles_5m = fetch_candles(symbol, RETEST_TIMEFRAME, 50)

    if len(candles_15m) < 10 or len(candles_5m) < 10:
        return None

    orb = get_orb(candles_15m)

    latest = candles_5m[-1]
    prev = candles_5m[-2]

    orb_high = orb["high"]
    orb_low = orb["low"]

    breakout_level = orb_high * (1 + BREAKOUT_BUFFER)

    breakout = prev["close"] > breakout_level
    retest = latest["low"] <= orb_high and latest["close"] > orb_high
    bullish_close = latest["close"] > latest["open"]

    breakout_strength = (latest["close"] - orb_high) / orb_high

    if not trend_ok(candles_5m):
        return None

    if breakout and retest and bullish_close and ai_filter(breakout_strength):
        entry = latest["close"]
        stop = orb_low
        take_profit = entry * 1.006

        return {
            "entry": entry,
            "stop": stop,
            "take_profit": take_profit,
            "orb_high": orb_high,
            "orb_low": orb_low
        }

    return None


def log_mock_trade(symbol, entry, exit_price, reason):
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
            now(), symbol, "LONG",
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
                            "trail_stop": setup["entry"] * (1 - TRAIL_PERCENT),
                            "opened": now()
                        }

                        last_trade_time = now()

                else:
                    ticker = exchange.fetch_ticker(open_position["symbol"])
                    price = float(ticker["last"])

                    new_trail = price * (1 - TRAIL_PERCENT)
                    if new_trail > open_position["trail_stop"]:
                        open_position["trail_stop"] = new_trail

                    if price <= open_position["stop"]:
                        log_mock_trade(open_position["symbol"], open_position["entry"], price, "STOP_LOSS")
                        open_position = None

                    elif price <= open_position["trail_stop"]:
                        log_mock_trade(open_position["symbol"], open_position["entry"], price, "TRAIL_STOP")
                        open_position = None

                    elif price >= open_position["take_profit"]:
                        log_mock_trade(open_position["symbol"], open_position["entry"], price, "TAKE_PROFIT")
                        open_position = None

        except Exception as e:
            print("Trading loop error:", e)

        await asyncio.sleep(POLL_SECONDS)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pos = "Ingen öppen position"

    if open_position:
        pos = (
            f"Öppen LONG\n"
            f"Symbol: {open_position['symbol']}\n"
            f"Entry: {open_position['entry']}\n"
            f"Stop: {open_position['stop']}\n"
            f"Trail: {open_position['trail_stop']}\n"
            f"TP: {open_position['take_profit']}"
        )

    await update.message.reply_text(
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


async def engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING
    BOT_RUNNING = True
    await update.message.reply_text("✅ /engine_on aktiverad. Mocktrading körs.")


async def engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING
    BOT_RUNNING = False
    await update.message.reply_text("⛔ /engine_off aktiverad. Trading stoppad.")


async def pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_logs()
    total_pnl = 0.0
    trades = 0

    with open(MOCK_LOG, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_pnl += float(row["pnl_usdt"])
            trades += 1

    await update.message.reply_text(
        f"📈 PnL\n\n"
        f"Trades: {trades}\n"
        f"Total PnL: {round(total_pnl, 4)} USDT"
    )


async def coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🪙 Coins:\n" + "\n".join(SYMBOLS))


async def stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_SIZE_USDT

    if context.args:
        try:
            TRADE_SIZE_USDT = float(context.args[0])
            await update.message.reply_text(f"✅ Stake ändrad till {TRADE_SIZE_USDT} USDT")
            return
        except:
            pass

    await update.message.reply_text(f"💰 Stake: {TRADE_SIZE_USDT} USDT\nÄndra med: /stake 30")


async def risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RISK_PERCENT

    if context.args:
        try:
            RISK_PERCENT = float(context.args[0])
            await update.message.reply_text(f"✅ Risk ändrad till {RISK_PERCENT}%")
            return
        except:
            pass

    await update.message.reply_text(f"⚠️ Risk: {RISK_PERCENT}%\nÄndra med: /risk 1")


async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global USE_TREND_FILTER

    if context.args and context.args[0].lower() in ["on", "off"]:
        USE_TREND_FILTER = context.args[0].lower() == "on"

    await update.message.reply_text(f"📉 Trendfilter: {'ON' if USE_TREND_FILTER else 'OFF'}")


async def sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global USE_SWEEP_FILTER

    if context.args and context.args[0].lower() in ["on", "off"]:
        USE_SWEEP_FILTER = context.args[0].lower() == "on"

    await update.message.reply_text(f"🧹 Sweepfilter: {'ON' if USE_SWEEP_FILTER else 'OFF'}")


async def reclaim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global USE_RECLAIM_FILTER

    if context.args and context.args[0].lower() in ["on", "off"]:
        USE_RECLAIM_FILTER = context.args[0].lower() == "on"

    await update.message.reply_text(f"♻️ Reclaimfilter: {'ON' if USE_RECLAIM_FILTER else 'OFF'}")


async def breakbuf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BREAKOUT_BUFFER

    if context.args:
        try:
            BREAKOUT_BUFFER = float(context.args[0])
            await update.message.reply_text(f"✅ Breakout buffer ändrad till {BREAKOUT_BUFFER}")
            return
        except:
            pass

    await update.message.reply_text(f"📏 Breakout buffer: {BREAKOUT_BUFFER}\nÄndra med: /breakbuf 0.001")


async def sl_atr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SL_ATR_MULTIPLIER

    if context.args:
        try:
            SL_ATR_MULTIPLIER = float(context.args[0])
            await update.message.reply_text(f"✅ SL ATR multiplier ändrad till {SL_ATR_MULTIPLIER}")
            return
        except:
            pass

    await update.message.reply_text(f"🛑 SL ATR multiplier: {SL_ATR_MULTIPLIER}\nÄndra med: /sl_atr 1.5")


async def trail(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRAIL_PERCENT

    if context.args:
        try:
            TRAIL_PERCENT = float(context.args[0])
            await update.message.reply_text(f"✅ Trail ändrad till {TRAIL_PERCENT * 100:.2f}%")
            return
        except:
            pass

    await update.message.reply_text(f"🔁 Trail: {TRAIL_PERCENT * 100:.2f}%\nÄndra med: /trail 0.003")


async def cooldown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global COOLDOWN_MINUTES

    if context.args:
        try:
            COOLDOWN_MINUTES = int(context.args[0])
            await update.message.reply_text(f"✅ Cooldown ändrad till {COOLDOWN_MINUTES} min")
            return
        except:
            pass

    await update.message.reply_text(f"⏱ Cooldown: {COOLDOWN_MINUTES} min\nÄndra med: /cooldown 5")


async def set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_MODE

    if not context.args:
        await update.message.reply_text("Använd: /set_ai aggressiv | neutral | försiktig")
        return

    mode = context.args[0].lower()

    if mode not in ["aggressiv", "neutral", "försiktig"]:
        await update.message.reply_text("Fel AI-läge.")
        return

    AI_MODE = mode
    await update.message.reply_text(f"✅ AI-läge: {AI_MODE}")


async def symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ACTIVE_SYMBOL

    if not context.args:
        await update.message.reply_text("Använd: /symbol BTCUSDT")
        return

    raw = context.args[0].upper().replace("/", "")
    formatted = raw.replace("USDT", "/USDT")

    if formatted not in SYMBOLS:
        await update.message.reply_text("Symbol stöds inte.")
        return

    ACTIVE_SYMBOL = formatted
    await update.message.reply_text(f"✅ Aktiv symbol: {ACTIVE_SYMBOL}")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Mp ORBbot kommandon\n\n"
        "/status\n"
        "/pnl\n"
        "/engine_on\n"
        "/engine_off\n"
        "/coins\n"
        "/stake 30\n"
        "/risk 1\n"
        "/trend on/off\n"
        "/sweep on/off\n"
        "/reclaim on/off\n"
        "/breakbuf 0.001\n"
        "/sl_atr 1.5\n"
        "/trail 0.003\n"
        "/cooldown 5\n"
        "/symbol BTCUSDT\n"
        "/set_ai neutral"
    )


async def post_init(app):
    app.create_task(trading_loop(app))


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN saknas i .env")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("pnl", pnl))
    app.add_handler(CommandHandler("engine_on", engine_on))
    app.add_handler(CommandHandler("engine_off", engine_off))
    app.add_handler(CommandHandler("coins", coins))
    app.add_handler(CommandHandler("stake", stake))
    app.add_handler(CommandHandler("risk", risk))
    app.add_handler(CommandHandler("trend", trend))
    app.add_handler(CommandHandler("sweep", sweep))
    app.add_handler(CommandHandler("reclaim", reclaim))
    app.add_handler(CommandHandler("breakbuf", breakbuf))
    app.add_handler(CommandHandler("sl_atr", sl_atr))
    app.add_handler(CommandHandler("trail", trail))
    app.add_handler(CommandHandler("cooldown", cooldown))
    app.add_handler(CommandHandler("symbol", symbol))
    app.add_handler(CommandHandler("set_ai", set_ai))
    app.add_handler(CommandHandler("help", help_cmd))

    print("Mp ORBbot körs...")
    app.run_polling()


if __name__ == "__main__":
    main()
