import os
import csv
import asyncio
from datetime import datetime, timezone, timedelta

import ccxt
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# =========================
# LOAD ENV
# =========================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# =========================
# CONFIG
# =========================

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "LINK/USDT"
]

MODE = "mock"

BOT_RUNNING = False

AI_MODE = "neutral"

MAX_OPEN_TRADES = 5

TRADE_SIZE_USDT = 30.0

FEE_RATE = 0.001

POLL_SECONDS = 20

# =========================
# ORB CONFIG
# =========================

ORB_TIMEFRAME = "15m"

ENTRY_TIMEFRAME = "5m"

# 08:00 UTC = 10:00 svensk sommartid
ORB_HOUR_UTC = 8

ORB_MINUTE_UTC = 0

# =========================
# FILTERS
# =========================

USE_VOLUME_FILTER = True

USE_TREND_FILTER = True

USE_RECLAIM_FILTER = True

USE_TRAILING_STOP = True

BREAKOUT_BUFFER = 0.001

TRAIL_PERCENT = 0.004

COOLDOWN_MINUTES = 20

MIN_ORB_PERCENT = 0.002
MAX_ORB_PERCENT = 0.02

# =========================
# FILES
# =========================

MOCK_LOG = "mock_trade_log.csv"

# =========================
# EXCHANGE
# =========================

exchange = ccxt.kucoin({
    "enableRateLimit": True
})

# =========================
# STATE
# =========================

open_positions = {}

cooldowns = {}

last_trade_time = None

# =========================
# UTILS
# =========================

def now():
    return datetime.now(timezone.utc)

def now_str():
    return now().strftime("%Y-%m-%d %H:%M:%S")

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

def fetch_candles(symbol, timeframe, limit=200):
    candles = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=limit
    )

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

# =========================
# ORB LOGIC
# =========================

def get_daily_orb(symbol):

    candles = fetch_candles(symbol, ORB_TIMEFRAME, 100)

    today = now().date()

    for candle in candles:

        candle_time = datetime.fromtimestamp(
            candle["time"] / 1000,
            tz=timezone.utc
        )

        if (
            candle_time.date() == today
            and candle_time.hour == ORB_HOUR_UTC
            and candle_time.minute == ORB_MINUTE_UTC
        ):

            orb_size = (
                candle["high"] - candle["low"]
            ) / candle["low"]

            if orb_size < MIN_ORB_PERCENT:
                return None

            if orb_size > MAX_ORB_PERCENT:
                return None

            return {
                "high": candle["high"],
                "low": candle["low"],
                "time": candle_time
            }

    return None

# =========================
# FILTERS
# =========================

def trend_filter(candles):

    if not USE_TREND_FILTER:
        return True

    closes = [c["close"] for c in candles[-50:]]

    ema = sum(closes) / len(closes)

    return closes[-1] > ema

def volume_filter(candles):

    if not USE_VOLUME_FILTER:
        return True

    latest_volume = candles[-1]["volume"]

    avg_volume = sum(
        c["volume"] for c in candles[-21:-1]
    ) / 20

    return latest_volume > avg_volume

def ai_filter(strength):

    if AI_MODE == "aggressiv":
        return strength > 0.0005

    if AI_MODE == "försiktig":
        return strength > 0.002

    return strength > 0.001

# =========================
# SETUP DETECTION
# =========================

def check_setup(symbol):

    if symbol in cooldowns:

        if now() < cooldowns[symbol]:
            return None

    orb = get_daily_orb(symbol)

    if not orb:
        return None

    candles_5m = fetch_candles(
        symbol,
        ENTRY_TIMEFRAME,
        100
    )

    candles_1h = fetch_candles(
        symbol,
        "1h",
        100
    )

    latest = candles_5m[-1]

    prev = candles_5m[-2]

    orb_high = orb["high"]

    orb_low = orb["low"]

    breakout_level = orb_high * (
        1 + BREAKOUT_BUFFER
    )

    breakout = (
        prev["close"] > breakout_level
    )

    reclaim = (
        latest["low"] <= orb_high
        and latest["close"] > orb_high
    )

    bullish_close = (
        latest["close"] > latest["open"]
    )

    breakout_strength = (
        latest["close"] - orb_high
    ) / orb_high

    if not trend_filter(candles_1h):
        return None

    if not volume_filter(candles_5m):
        return None

    if not ai_filter(breakout_strength):
        return None

    if USE_RECLAIM_FILTER and not reclaim:
        return None

    if (
        breakout
        and bullish_close
    ):

        entry = latest["close"]

        stop = orb_low

        risk = entry - stop

        take_profit = entry + (risk * 2)

        return {
            "entry": entry,
            "stop": stop,
            "take_profit": take_profit,
            "orb_high": orb_high,
            "orb_low": orb_low
        }

    return None

# =========================
# LOGGING
# =========================

def log_trade(
    symbol,
    entry,
    exit_price,
    reason
):

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
            now_str(),
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

# =========================
# TELEGRAM SEND
# =========================

async def notify(app, text):

    try:
        await app.bot.send_message(
            chat_id=os.getenv("CHAT_ID"),
            text=text
        )
    except:
        pass

# =========================
# TRADING LOOP
# =========================

async def trading_loop(app):

    global last_trade_time

    ensure_logs()

    while True:

        try:

            if BOT_RUNNING:

                # =========================
                # OPEN NEW POSITIONS
                # =========================

                if len(open_positions) < MAX_OPEN_TRADES:

                    for symbol in SYMBOLS:

                        if symbol in open_positions:
                            continue

                        if len(open_positions) >= MAX_OPEN_TRADES:
                            break

                        setup = check_setup(symbol)

                        if setup:

                            open_positions[symbol] = {

                                "entry": setup["entry"],

                                "stop": setup["stop"],

                                "take_profit": setup["take_profit"],

                                "trail_stop": (
                                    setup["entry"]
                                    * (1 - TRAIL_PERCENT)
                                ),

                                "opened": now_str()
                            }

                            last_trade_time = now_str()

                            await notify(
                                app,
                                f"🚀 NEW MOCK LONG\n\n"
                                f"{symbol}\n"
                                f"Entry: {setup['entry']}\n"
                                f"SL: {setup['stop']}\n"
                                f"TP: {setup['take_profit']}"
                            )

                # =========================
                # MANAGE POSITIONS
                # =========================

                symbols_to_close = []

                for symbol, pos in open_positions.items():

                    ticker = exchange.fetch_ticker(symbol)

                    price = float(ticker["last"])

                    if USE_TRAILING_STOP:

                        new_trail = (
                            price * (1 - TRAIL_PERCENT)
                        )

                        if new_trail > pos["trail_stop"]:

                            pos["trail_stop"] = new_trail

                    # STOP LOSS

                    if price <= pos["stop"]:

                        pnl = log_trade(
                            symbol,
                            pos["entry"],
                            price,
                            "STOP_LOSS"
                        )

                        cooldowns[symbol] = (
                            now()
                            + timedelta(
                                minutes=COOLDOWN_MINUTES
                            )
                        )

                        symbols_to_close.append(symbol)

                        await notify(
                            app,
                            f"❌ STOP LOSS\n\n"
                            f"{symbol}\n"
                            f"PnL: {round(pnl, 4)} USDT"
                        )

                    # TRAILING STOP

                    elif (
                        USE_TRAILING_STOP
                        and price <= pos["trail_stop"]
                    ):

                        pnl = log_trade(
                            symbol,
                            pos["entry"],
                            price,
                            "TRAIL_STOP"
                        )

                        symbols_to_close.append(symbol)

                        await notify(
                            app,
                            f"🔁 TRAIL EXIT\n\n"
                            f"{symbol}\n"
                            f"PnL: {round(pnl, 4)} USDT"
                        )

                    # TAKE PROFIT

                    elif price >= pos["take_profit"]:

                        pnl = log_trade(
                            symbol,
                            pos["entry"],
                            price,
                            "TAKE_PROFIT"
                        )

                        symbols_to_close.append(symbol)

                        await notify(
                            app,
                            f"✅ TAKE PROFIT\n\n"
                            f"{symbol}\n"
                            f"PnL: {round(pnl, 4)} USDT"
                        )

                for symbol in symbols_to_close:

                    del open_positions[symbol]

        except Exception as e:

            print("Trading loop error:", e)

        await asyncio.sleep(POLL_SECONDS)

# =========================
# TELEGRAM COMMANDS
# =========================

async def engine_on(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    global BOT_RUNNING

    BOT_RUNNING = True

    await update.message.reply_text(
        "✅ Engine ON"
    )

async def engine_off(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    global BOT_RUNNING

    BOT_RUNNING = False

    await update.message.reply_text(
        "⛔ Engine OFF"
    )

async def status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        f"📊 Mp ORBbot\n\n"
        f"Running: {BOT_RUNNING}\n"
        f"Mode: {MODE}\n"
        f"AI: {AI_MODE}\n"
        f"Open trades: {len(open_positions)} / {MAX_OPEN_TRADES}\n"
        f"Trade Size: {TRADE_SIZE_USDT} USDT\n"
        f"Fee: {FEE_RATE*100:.2f}%\n"
        f"Trail: {TRAIL_PERCENT*100:.2f}%\n"
        f"Cooldown: {COOLDOWN_MINUTES}m\n"
        f"Last Trade: {last_trade_time}\n"
    )

    if open_positions:

        text += "\n📌 Open Positions:\n"

        for s, p in open_positions.items():

            text += (
                f"\n{s}\n"
                f"Entry: {round(p['entry'],4)}\n"
                f"SL: {round(p['stop'],4)}"
            )

    await update.message.reply_text(text)

async def pnl(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    ensure_logs()

    total_pnl = 0

    trades = 0

    wins = 0

    try:

        with open(MOCK_LOG, "r") as f:

            reader = csv.DictReader(f)

            for row in reader:

                pnl_val = float(row["pnl_usdt"])

                total_pnl += pnl_val

                trades += 1

                if pnl_val > 0:
                    wins += 1

    except:
        pass

    winrate = 0

    if trades > 0:
        winrate = (wins / trades) * 100

    await update.message.reply_text(
        f"📈 PnL\n\n"
        f"Trades: {trades}\n"
        f"Wins: {wins}\n"
        f"Winrate: {round(winrate,2)}%\n"
        f"PnL: {round(total_pnl,4)} USDT"
    )

async def maxtrades(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    global MAX_OPEN_TRADES

    if context.args:

        try:

            MAX_OPEN_TRADES = int(context.args[0])

            await update.message.reply_text(
                f"✅ Max trades = {MAX_OPEN_TRADES}"
            )

            return

        except:
            pass

    await update.message.reply_text(
        f"Current max trades: {MAX_OPEN_TRADES}"
    )

async def stake(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    global TRADE_SIZE_USDT

    if context.args:

        try:

            TRADE_SIZE_USDT = float(context.args[0])

            await update.message.reply_text(
                f"✅ Stake = {TRADE_SIZE_USDT}"
            )

            return

        except:
            pass

    await update.message.reply_text(
        f"Stake = {TRADE_SIZE_USDT}"
    )

async def set_ai(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    global AI_MODE

    if not context.args:

        await update.message.reply_text(
            "Usage: /set_ai aggressiv|neutral|försiktig"
        )

        return

    mode = context.args[0].lower()

    if mode not in [
        "aggressiv",
        "neutral",
        "försiktig"
    ]:

        await update.message.reply_text(
            "Invalid mode"
        )

        return

    AI_MODE = mode

    await update.message.reply_text(
        f"✅ AI mode = {AI_MODE}"
    )

async def help_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🤖 Mp ORBbot\n\n"
        "/engine_on\n"
        "/engine_off\n"
        "/status\n"
        "/pnl\n"
        "/stake 30\n"
        "/maxtrades 5\n"
        "/set_ai neutral"
    )

# =========================
# APP
# =========================

async def post_init(app):

    app.create_task(
        trading_loop(app)
    )

def main():

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

    app.add_handler(
        CommandHandler(
            "engine_on",
            engine_on
        )
    )

    app.add_handler(
        CommandHandler(
            "engine_off",
            engine_off
        )
    )

    app.add_handler(
        CommandHandler(
            "status",
            status
        )
    )

    app.add_handler(
        CommandHandler(
            "pnl",
            pnl
        )
    )

    app.add_handler(
        CommandHandler(
            "stake",
            stake
        )
    )

    app.add_handler(
        CommandHandler(
            "maxtrades",
            maxtrades
        )
    )

    app.add_handler(
        CommandHandler(
            "set_ai",
            set_ai
        )
    )

    app.add_handler(
        CommandHandler(
            "help",
            help_cmd
        )
    )

    print("Mp ORBbot running...")

    app.run_polling()

if __name__ == "__main__":
    main()
