import os
import csv
import asyncio
from datetime import datetime, timezone, timedelta

import ccxt
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

MODE = "mock"
BOT_RUNNING = False
AI_MODE = "neutral"

SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT"]

MAX_OPEN_TRADES = 1
TRADE_SIZE_USDT = 30.0
FEE_RATE = 0.001
POLL_SECONDS = 20

ORB_TIMEFRAME = "15m"
ENTRY_TIMEFRAME = "5m"
ORB_HOUR_UTC = 8
ORB_MINUTE_UTC = 0

USE_VOLUME_FILTER = True
USE_TREND_FILTER = True
USE_RECLAIM_FILTER = True

BREAKOUT_BUFFER = 0.001
TRAIL_PERCENT = 0.006
COOLDOWN_MINUTES = 30

MIN_ORB_PERCENT = 0.001
MAX_ORB_PERCENT = 0.04

MOCK_LOG = "mock_trade_log.csv"

exchange = ccxt.kucoin({"enableRateLimit": True})

open_positions = {}
cooldowns = {}
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
                "size_usdt", "fee_entry", "fee_exit", "pnl_usdt",
                "reason", "entry_reason", "mode"
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


def get_daily_orb(symbol):
    candles = fetch_candles(symbol, ORB_TIMEFRAME, 100)
    today = now().date()

    for candle in candles:
        candle_time = datetime.fromtimestamp(candle["time"] / 1000, tz=timezone.utc)

        if (
            candle_time.date() == today
            and candle_time.hour == ORB_HOUR_UTC
            and candle_time.minute == ORB_MINUTE_UTC
        ):
            orb_size = (candle["high"] - candle["low"]) / candle["low"]

            if orb_size < MIN_ORB_PERCENT:
                return None, "ORB för liten"

            if orb_size > MAX_ORB_PERCENT:
                return None, "ORB för stor"

            return {
                "high": candle["high"],
                "low": candle["low"],
                "time": candle_time,
                "size": orb_size
            }, "OK"

    return None, "Ingen ORB hittad"


def trend_filter(candles):
    if not USE_TREND_FILTER:
        return True
    closes = [c["close"] for c in candles[-50:]]
    avg = sum(closes) / len(closes)
    return closes[-1] > avg


def volume_filter(candles):
    if not USE_VOLUME_FILTER:
        return True
    latest_volume = candles[-1]["volume"]
    avg_volume = sum(c["volume"] for c in candles[-21:-1]) / 20
    return latest_volume > avg_volume


def ai_filter(strength):
    if AI_MODE == "aggressiv":
        return strength > 0.0005
    if AI_MODE == "försiktig":
        return strength > 0.002
    return strength > 0.001


def analyze_symbol(symbol):
    if symbol in cooldowns and now() < cooldowns[symbol]:
        return None, {"reason": "Cooldown aktiv"}

    orb, orb_reason = get_daily_orb(symbol)

    if not orb:
        return None, {"reason": orb_reason}

    candles_5m = fetch_candles(symbol, ENTRY_TIMEFRAME, 100)
    candles_1h = fetch_candles(symbol, "1h", 100)

    latest = candles_5m[-1]
    prev = candles_5m[-2]

    orb_high = orb["high"]
    orb_low = orb["low"]
    breakout_level = orb_high * (1 + BREAKOUT_BUFFER)

    breakout = prev["close"] > breakout_level
    reclaim = latest["low"] <= orb_high and latest["close"] > orb_high
    bullish = latest["close"] > latest["open"]
    trend_ok = trend_filter(candles_1h)
    volume_ok = volume_filter(candles_5m)

    strength = (latest["close"] - orb_high) / orb_high
    ai_ok = ai_filter(strength)

    debug_data = {
        "reason": "Analyserad",
        "orb_high": orb_high,
        "orb_low": orb_low,
        "price": latest["close"],
        "breakout": breakout,
        "reclaim": reclaim,
        "bullish": bullish,
        "trend_ok": trend_ok,
        "volume_ok": volume_ok,
        "ai_ok": ai_ok,
        "strength": strength,
    }

    if not breakout:
        debug_data["reason"] = "Ingen breakout"
        return None, debug_data

    if USE_RECLAIM_FILTER and not reclaim:
        debug_data["reason"] = "Ingen retest/reclaim"
        return None, debug_data

    if not bullish:
        debug_data["reason"] = "Inte bullish candle"
        return None, debug_data

    if not trend_ok:
        debug_data["reason"] = "Trendfilter nekar"
        return None, debug_data

    if not volume_ok:
        debug_data["reason"] = "Volymfilter nekar"
        return None, debug_data

    if not ai_ok:
        debug_data["reason"] = "AI-filter nekar"
        return None, debug_data

    entry = latest["close"]
    stop = orb_low
    risk = entry - stop
    take_profit = entry + risk * 2

    setup = {
        "entry": entry,
        "stop": stop,
        "take_profit": take_profit,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "entry_reason": (
            f"Breakout ✅ | Retest ✅ | Bullish ✅ | "
            f"Trend ✅ | Volym ✅ | AI ✅"
        )
    }

    debug_data["reason"] = "TRADE GODKÄND"

    return setup, debug_data


def create_chart(symbol, entry=None, exit_price=None, stop=None, take_profit=None, orb_high=None, orb_low=None, title="Trade"):
    candles = fetch_candles(symbol, ENTRY_TIMEFRAME, 80)
    closes = [c["close"] for c in candles]

    file_name = f"chart_{symbol.replace('/', '')}_{int(datetime.now().timestamp())}.png"

    plt.figure(figsize=(10, 5))
    plt.plot(closes)

    if orb_high:
        plt.axhline(orb_high, linestyle="--", label="ORB High")

    if orb_low:
        plt.axhline(orb_low, linestyle="--", label="ORB Low")

    if entry:
        plt.axhline(entry, linestyle="-", label="Entry")

    if exit_price:
        plt.axhline(exit_price, linestyle="-", label="Exit")

    if stop:
        plt.axhline(stop, linestyle=":", label="SL")

    if take_profit:
        plt.axhline(take_profit, linestyle=":", label="TP")

    plt.title(f"{title} - {symbol}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    return file_name


def log_trade(symbol, entry, exit_price, reason, entry_reason):
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
            now_str(), symbol, "LONG", round(entry, 8), round(exit_price, 8),
            size, round(fee_entry, 6), round(fee_exit, 6), round(pnl, 6),
            reason, entry_reason, MODE
        ])

    return pnl


async def notify_text(app, text):
    if not CHAT_ID:
        return
    try:
        await app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        print("Telegram text error:", e)


async def notify_photo(app, photo_path, caption):
    if not CHAT_ID:
        return
    try:
        with open(photo_path, "rb") as img:
            await app.bot.send_photo(chat_id=CHAT_ID, photo=img, caption=caption)
    except Exception as e:
        print("Telegram photo error:", e)


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

                        setup, debug_data = analyze_symbol(symbol)

                        if setup:
                            open_positions[symbol] = {
                                "entry": setup["entry"],
                                "stop": setup["stop"],
                                "take_profit": setup["take_profit"],
                                "trail_stop": setup["entry"] * (1 - TRAIL_PERCENT),
                                "orb_high": setup["orb_high"],
                                "orb_low": setup["orb_low"],
                                "entry_reason": setup["entry_reason"],
                                "opened": now_str()
                            }

                            last_trade_time = now_str()

                            chart = create_chart(
                                symbol,
                                entry=setup["entry"],
                                stop=setup["stop"],
                                take_profit=setup["take_profit"],
                                orb_high=setup["orb_high"],
                                orb_low=setup["orb_low"],
                                title="ENTRY"
                            )

                            await notify_photo(
                                app,
                                chart,
                                f"🟢 NEW MOCK LONG\n\n"
                                f"{symbol}\n"
                                f"Entry: {setup['entry']}\n"
                                f"SL: {setup['stop']}\n"
                                f"TP: {setup['take_profit']}\n\n"
                                f"{setup['entry_reason']}"
                            )

                close_symbols = []

                for symbol, pos in open_positions.items():
                    ticker = exchange.fetch_ticker(symbol)
                    price = float(ticker["last"])

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

                        chart = create_chart(
                            symbol,
                            entry=pos["entry"],
                            exit_price=price,
                            stop=pos["stop"],
                            take_profit=pos["take_profit"],
                            orb_high=pos["orb_high"],
                            orb_low=pos["orb_low"],
                            title="EXIT"
                        )

                        await notify_photo(
                            app,
                            chart,
                            f"🔴 EXIT {reason}\n\n"
                            f"{symbol}\n"
                            f"Entry: {pos['entry']}\n"
                            f"Exit: {price}\n"
                            f"PnL: {round(pnl, 4)} USDT"
                        )

                        close_symbols.append(symbol)

                for symbol in close_symbols:
                    del open_positions[symbol]

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
        f"Mode: {MODE}\n"
        f"AI: {AI_MODE}\n"
        f"Coins: {len(SYMBOLS)}\n"
        f"Open: {len(open_positions)} / {MAX_OPEN_TRADES}\n"
        f"Stake: {TRADE_SIZE_USDT} USDT\n"
        f"Fee: {FEE_RATE * 100:.2f}%\n"
        f"Trail: {TRAIL_PERCENT * 100:.2f}%\n"
        f"Cooldown: {COOLDOWN_MINUTES} min\n"
        f"Last trade: {last_trade_time}"
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


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_logs()

    rows = []

    with open(MOCK_LOG, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[-20:]

    if not rows:
        await update.message.reply_text("Ingen historik ännu.")
        return

    msg = "📜 Senaste trades\n\n"

    for r in rows:
        msg += (
            f"{r['time']} {r['symbol']}\n"
            f"{r['reason']} | PnL: {r['pnl_usdt']} USDT\n\n"
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
            total += val
            trades += 1
            if val > 0:
                wins += 1

    winrate = (wins / trades * 100) if trades else 0

    await update.message.reply_text(
        f"📈 PnL\n\n"
        f"Trades: {trades}\n"
        f"Wins: {wins}\n"
        f"Winrate: {round(winrate, 2)}%\n"
        f"PnL: {round(total, 4)} USDT"
    )


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    targets = SYMBOLS

    if context.args:
        raw = context.args[0].upper().replace("/", "")
        formatted = raw.replace("USDT", "/USDT")
        targets = [formatted]

    msg = "🔍 Debug\n\n"

    for symbol in targets:
        try:
            setup, d = analyze_symbol(symbol)

            msg += (
                f"{symbol}\n"
                f"Reason: {d.get('reason')}\n"
                f"Price: {round(d.get('price', 0), 4)}\n"
                f"Breakout: {d.get('breakout')}\n"
                f"Reclaim: {d.get('reclaim')}\n"
                f"Bullish: {d.get('bullish')}\n"
                f"Trend: {d.get('trend_ok')}\n"
                f"Volume: {d.get('volume_ok')}\n"
                f"AI: {d.get('ai_ok')}\n\n"
            )

        except Exception as e:
            msg += f"{symbol}: ERROR {e}\n\n"

    await update.message.reply_text(msg)


async def coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🪙 Coins:\n\n" + "\n".join(SYMBOLS))


async def maxtrades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MAX_OPEN_TRADES

    if context.args:
        MAX_OPEN_TRADES = int(context.args[0])
        await update.message.reply_text(f"✅ Max trades = {MAX_OPEN_TRADES}")
        return

    await update.message.reply_text(f"Max trades = {MAX_OPEN_TRADES}")


async def stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADE_SIZE_USDT

    if context.args:
        TRADE_SIZE_USDT = float(context.args[0])
        await update.message.reply_text(f"✅ Stake = {TRADE_SIZE_USDT} USDT")
        return

    await update.message.reply_text(f"Stake = {TRADE_SIZE_USDT} USDT")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Mp ORBbot\n\n"
        "/engine_on\n"
        "/engine_off\n"
        "/status\n"
        "/open\n"
        "/history\n"
        "/pnl\n"
        "/debug\n"
        "/debug BTC\n"
        "/coins\n"
        "/stake 30\n"
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
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("pnl", pnl))
    app.add_handler(CommandHandler("debug", debug))
    app.add_handler(CommandHandler("coins", coins))
    app.add_handler(CommandHandler("stake", stake))
    app.add_handler(CommandHandler("maxtrades", maxtrades))
    app.add_handler(CommandHandler("help", help_cmd))

    print("Mp ORBbot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
