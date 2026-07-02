import os
import csv
import asyncio
from datetime import datetime, timezone, timedelta

import ccxt
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

MODE = "mock"
BOT_RUNNING = False

SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT"]

MAX_OPEN_TRADES = 1
TRADE_SIZE_USDT = 30.0
FEE_RATE = 0.001

POLL_SECONDS = 15

ORB_TIMEFRAME = "15m"
ENTRY_TIMEFRAME = "5m"

# 07:00 UTC = 09:00 svensk sommartid
ORB_HOUR_UTC = 7
ORB_MINUTE_UTC = 0

BREAKOUT_BUFFER = 0.0003
TRAIL_PERCENT = 0.006
COOLDOWN_MINUTES = 20

MOCK_LOG = "mock_trade_log.csv"

exchange = ccxt.kucoin({"enableRateLimit": True})

open_positions = {}
cooldowns = {}
symbol_state = {}

last_trade_time = None


def now():
    return datetime.now(timezone.utc)


def now_str():
    return now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_logs():
    if not os.path.exists(MOCK_LOG):
        with open(MOCK_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "symbol", "side", "entry", "exit",
                "size_usdt", "fee_entry", "fee_exit",
                "pnl_usdt", "reason", "entry_reason", "mode"
            ])


def fetch_candles(symbol, timeframe, limit=200):
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


def get_orb(symbol):
    candles = fetch_candles(symbol, ORB_TIMEFRAME, 100)
    today = now().date()

    for c in candles:
        t = datetime.fromtimestamp(c["time"] / 1000, tz=timezone.utc)

        if (
            t.date() == today
            and t.hour == ORB_HOUR_UTC
            and t.minute == ORB_MINUTE_UTC
        ):
            return {
                "high": c["high"],
                "low": c["low"],
                "time": t,
            }

    return None


def get_state(symbol):
    if symbol not in symbol_state:
        symbol_state[symbol] = {
            "breakout_seen": False,
            "retest_seen": False,
            "last_reason": "Ingen analys ännu",
            "orb_high": None,
            "orb_low": None,
        }
    return symbol_state[symbol]


def analyze_symbol(symbol):
    state = get_state(symbol)

    if symbol in cooldowns and now() < cooldowns[symbol]:
        state["last_reason"] = "Cooldown aktiv"
        return None

    orb = get_orb(symbol)

    if not orb:
        state["last_reason"] = "Ingen ORB hittad"
        return None

    orb_high = orb["high"]
    orb_low = orb["low"]

    state["orb_high"] = orb_high
    state["orb_low"] = orb_low

    candles_15m = fetch_candles(symbol, "15m", 30)
    candles_5m = fetch_candles(symbol, "5m", 60)

    latest_15m = candles_15m[-1]
    latest_5m = candles_5m[-1]

    breakout_level = orb_high * (1 + BREAKOUT_BUFFER)

    # Steg 1: 15m breakout
    if latest_15m["close"] > breakout_level:
        state["breakout_seen"] = True

    if not state["breakout_seen"]:
        state["last_reason"] = "Väntar på 15m breakout"
        return None

    # Steg 2: 5m retest
    if latest_5m["low"] <= orb_high:
        state["retest_seen"] = True

    if not state["retest_seen"]:
        state["last_reason"] = "15m breakout klar, väntar på 5m retest"
        return None

    # Steg 3: 5m reclaim/entry
    reclaim = latest_5m["close"] > orb_high
    bullish = latest_5m["close"] > latest_5m["open"]

    if not reclaim:
        state["last_reason"] = "Retest klar, väntar på close över ORB high"
        return None

    if not bullish:
        state["last_reason"] = "Close över ORB men inte bullish candle"
        return None

    entry = latest_5m["close"]
    stop = orb_low
    risk = entry - stop

    if risk <= 0:
        state["last_reason"] = "Ogiltig risk"
        return None

    take_profit = entry + risk * 2

    state["last_reason"] = "TRADE GODKÄND"

    return {
        "entry": entry,
        "stop": stop,
        "take_profit": take_profit,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "entry_reason": "15m breakout ✅ | 5m retest ✅ | 5m reclaim ✅"
    }


def log_trade(symbol, entry, exit_price, reason, entry_reason):
    size = TRADE_SIZE_USDT
    fee_entry = size * FEE_RATE
    qty = (size - fee_entry) / entry
    exit_value = qty * exit_price
    fee_exit = exit_value * FEE_RATE
    pnl = exit_value - fee_exit - size

    with open(MOCK_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now_str(), symbol, "LONG", round(entry, 8), round(exit_price, 8),
            size, round(fee_entry, 6), round(fee_exit, 6),
            round(pnl, 6), reason, entry_reason, MODE
        ])

    return pnl


async def notify(app, text):
    if not CHAT_ID:
        return
    try:
        await app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        print("Telegram error:", e)


async def trading_loop(app):
    global last_trade_time

    ensure_logs()

    while True:
        try:
            if BOT_RUNNING:

                if len(open_positions) < MAX_OPEN_TRADES:
                    for symbol in SYMBOLS:
                        if symbol in open_positions:
                            continue

                        if len(open_positions) >= MAX_OPEN_TRADES:
                            break

                        setup = analyze_symbol(symbol)

                        if setup:
                            open_positions[symbol] = {
                                "entry": setup["entry"],
                                "stop": setup["stop"],
                                "take_profit": setup["take_profit"],
                                "trail_stop": setup["entry"] * (1 - TRAIL_PERCENT),
                                "entry_reason": setup["entry_reason"],
                                "opened": now_str()
                            }

                            last_trade_time = now_str()

                            await notify(
                                app,
                                f"🟢 NEW MOCK LONG\n\n"
                                f"{symbol}\n"
                                f"Entry: {setup['entry']}\n"
                                f"SL: {setup['stop']}\n"
                                f"TP: {setup['take_profit']}\n\n"
                                f"{setup['entry_reason']}"
                            )

                to_close = []

                for symbol, pos in open_positions.items():
                    price = float(exchange.fetch_ticker(symbol)["last"])

                    new_trail = price * (1 - TRAIL_PERCENT)
                    if new_trail > pos["trail_stop"]:
                        pos["trail_stop"] = new_trail

                    reason = None

                    if price <= pos["stop"]:
                        reason = "STOP_LOSS"
                        cooldowns[symbol] = now() + timedelta(minutes=COOLDOWN_MINUTES)

                    elif price <= pos["trail_stop"]:
                        reason = "TRAIL_STOP"

                    elif price >= pos["take_profit"]:
                        reason = "TAKE_PROFIT"

                    if reason:
                        pnl = log_trade(
                            symbol,
                            pos["entry"],
                            price,
                            reason,
                            pos["entry_reason"]
                        )

                        await notify(
                            app,
                            f"🔴 EXIT {reason}\n\n"
                            f"{symbol}\n"
                            f"Entry: {pos['entry']}\n"
                            f"Exit: {price}\n"
                            f"PnL: {round(pnl, 4)} USDT"
                        )

                        to_close.append(symbol)

                for s in to_close:
                    del open_positions[s]

        except Exception as e:
            print("Trading loop error:", e)

        await asyncio.sleep(POLL_SECONDS)


async def engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING
    BOT_RUNNING = True
    await update.message.reply_text("✅ Engine ON")


async def engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING
    BOT_RUNNING = False
    await update.message.reply_text("⛔ Engine OFF")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 Mp ORBbot\n\n"
        f"Running: {BOT_RUNNING}\n"
        f"Coins: {len(SYMBOLS)}\n"
        f"Open: {len(open_positions)} / {MAX_OPEN_TRADES}\n"
        f"Stake: {TRADE_SIZE_USDT} USDT\n"
        f"Fee: {FEE_RATE * 100:.2f}%\n"
        f"ORB UTC: {ORB_HOUR_UTC:02d}:{ORB_MINUTE_UTC:02d}\n"
        f"Trail: {TRAIL_PERCENT * 100:.2f}%\n"
        f"Last trade: {last_trade_time}"
    )


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    targets = SYMBOLS

    if context.args:
        raw = context.args[0].upper().replace("/", "")
        targets = [raw.replace("USDT", "/USDT")]

    msg = "🔍 Debug\n\n"

    for symbol in targets:
        analyze_symbol(symbol)
        s = get_state(symbol)

        msg += (
            f"{symbol}\n"
            f"Reason: {s['last_reason']}\n"
            f"ORB High: {s['orb_high']}\n"
            f"ORB Low: {s['orb_low']}\n"
            f"15m Breakout: {s['breakout_seen']}\n"
            f"5m Retest: {s['retest_seen']}\n\n"
        )

    await update.message.reply_text(msg)


async def pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_logs()

    trades = 0
    wins = 0
    total = 0.0

    with open(MOCK_LOG, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val = float(r["pnl_usdt"])
            trades += 1
            total += val
            if val > 0:
                wins += 1

    winrate = wins / trades * 100 if trades else 0

    await update.message.reply_text(
        f"📈 PnL\n\n"
        f"Trades: {trades}\n"
        f"Wins: {wins}\n"
        f"Winrate: {round(winrate, 2)}%\n"
        f"PnL: {round(total, 4)} USDT"
    )


async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not open_positions:
        await update.message.reply_text("Inga öppna trades.")
        return

    msg = "📌 Öppna trades\n\n"

    for s, p in open_positions.items():
        msg += (
            f"{s}\n"
            f"Entry: {p['entry']}\n"
            f"SL: {p['stop']}\n"
            f"Trail: {p['trail_stop']}\n"
            f"TP: {p['take_profit']}\n\n"
        )

    await update.message.reply_text(msg)


async def coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🪙 Coins:\n\n" + "\n".join(SYMBOLS))


async def maxtrades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MAX_OPEN_TRADES
    if context.args:
        MAX_OPEN_TRADES = int(context.args[0])
    await update.message.reply_text(f"Max trades = {MAX_OPEN_TRADES}")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/engine_on\n"
        "/engine_off\n"
        "/status\n"
        "/debug\n"
        "/debug ETH\n"
        "/open\n"
        "/pnl\n"
        "/coins\n"
        "/maxtrades 1"
    )


async def post_init(app):
    app.create_task(trading_loop(app))


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN saknas i .env")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("engine_on", engine_on))
    app.add_handler(CommandHandler("engine_off", engine_off))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("debug", debug))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("pnl", pnl))
    app.add_handler(CommandHandler("coins", coins))
    app.add_handler(CommandHandler("maxtrades", maxtrades))
    app.add_handler(CommandHandler("help", help_cmd))

    print("Mp ORBbot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
