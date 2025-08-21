import os
import asyncio
import logging
import pandas as pd
from fastapi import FastAPI
from kucoin.client import Market
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==============================
# ENVIRONMENT VARS
# ==============================
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")  # Render miljövariabel
if not TELEGRAM_TOKEN:
    raise Exception("BOT_TOKEN måste vara satt i Render environment")

# ==============================
# FASTAPI för Render
# ==============================
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Mp ORBbot is running"}

# ==============================
# BOT STATE
# ==============================
ENGINE_ON = False
AI_MODE = "neutral"  # default
TRADE_MODE = "mock"  # mock eller live
ACTIVE_TRADES = {}
TRADE_LOG_FILE = "real_trade_log.csv"
MOCK_LOG_FILE = "mock_trade_log.csv"
TRADE_SIZE = 30  # USDT per trade i mock

# ==============================
# KUCOIN MARKET CLIENT
# ==============================
market_client = Market(url="https://api.kucoin.com")

# ==============================
# ORB LOGIK
# ==============================
SYMBOLS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]
ENTRY_MODE = "CLOSE"  # eller TICK

async def fetch_candles(symbol, interval="3min", limit=50):
    try:
        data = market_client.get_kline(symbol, interval, limit=limit)
        df = pd.DataFrame(data, columns=["time", "open", "close", "high", "low", "volume", "amount"])
        df["open"] = df["open"].astype(float)
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        return df.sort_values("time")
    except Exception as e:
        logger.error(f"Kunde inte hämta candles för {symbol}: {e}")
        return None

def detect_orb(df):
    """Ny ORB = första gröna candle efter en röd"""
    for i in range(1, len(df)):
        if df.iloc[i-1]["close"] < df.iloc[i-1]["open"] and df.iloc[i]["close"] > df.iloc[i]["open"]:
            orb_high = df.iloc[i]["high"]
            orb_low = df.iloc[i]["low"]
            return orb_high, orb_low, i
    return None, None, None

# ==============================
# TRADE FUNKTIONER
# ==============================
def log_trade(symbol, side, price, qty, mode="mock"):
    file = MOCK_LOG_FILE if mode == "mock" else TRADE_LOG_FILE
    exists = os.path.isfile(file)
    df = pd.DataFrame([{
        "symbol": symbol,
        "side": side,
        "price": price,
        "qty": qty,
        "mode": mode
    }])
    df.to_csv(file, mode="a", header=not exists, index=False)

async def try_trade(symbol):
    global ACTIVE_TRADES
    df = await fetch_candles(symbol)
    if df is None or len(df) < 5:
        return

    orb_high, orb_low, idx = detect_orb(df)
    if orb_high is None:
        return

    last = df.iloc[-1]
    if ENTRY_MODE == "CLOSE" and last["close"] > orb_high and symbol not in ACTIVE_TRADES:
        entry_price = last["close"]
        qty = round(TRADE_SIZE / entry_price, 5)
        ACTIVE_TRADES[symbol] = {
            "entry": entry_price,
            "stop": orb_low,
            "qty": qty,
            "mode": TRADE_MODE
        }
        log_trade(symbol, "BUY", entry_price, qty, TRADE_MODE)
        return f"ENTRY {symbol} @ {entry_price}"

    if symbol in ACTIVE_TRADES:
        trade = ACTIVE_TRADES[symbol]
        stop = trade["stop"]
        qty = trade["qty"]
        price = last["close"]

        # Trailing stop: flytta upp stop när vi går +0.1%
        if price > trade["entry"] * 1.001 and price > stop:
            ACTIVE_TRADES[symbol]["stop"] = price * 0.999

        # Exit
        if price <= ACTIVE_TRADES[symbol]["stop"]:
            log_trade(symbol, "SELL", price, qty, TRADE_MODE)
            del ACTIVE_TRADES[symbol]
            return f"EXIT {symbol} @ {price}"

    return None

# ==============================
# ENGINE LOOP
# ==============================
async def engine_loop(app):
    global ENGINE_ON
    while True:
        if ENGINE_ON:
            for sym in SYMBOLS:
                msg = await try_trade(sym)
                if msg:
                    await app.bot.send_message(chat_id=ADMIN_CHAT_ID, text=msg)
        await asyncio.sleep(10)

# ==============================
# TELEGRAM COMMANDS
# ==============================
ADMIN_CHAT_ID = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON, ADMIN_CHAT_ID
    ENGINE_ON = True
    ADMIN_CHAT_ID = update.effective_chat.id
    await update.message.reply_text("✅ Engine STARTAD")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON
    ENGINE_ON = False
    await update.message.reply_text("🛑 Engine STOPPAD")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = f"Engine: {'ON' if ENGINE_ON else 'OFF'}\nAI: {AI_MODE}\nMode: {TRADE_MODE}\nTrades: {len(ACTIVE_TRADES)}"
    await update.message.reply_text(status_msg)

async def set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_MODE
    if context.args:
        AI_MODE = context.args[0].lower()
        await update.message.reply_text(f"AI-läge satt till {AI_MODE}")
    else:
        await update.message.reply_text("Använd: /set_ai neutral|aggressiv|försiktig")

async def mock_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /mock_trade SYMBOL")
        return
    symbol = context.args[0].upper()
    price = 100  # placeholder
    qty = round(TRADE_SIZE / price, 5)
    log_trade(symbol, "BUY", price, qty, "mock")
    await update.message.reply_text(f"Mock trade ENTRY {symbol} @ {price}")

async def export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    files = []
    if os.path.isfile(MOCK_LOG_FILE):
        await update.message.reply_document(document=open(MOCK_LOG_FILE, "rb"))
    if os.path.isfile(TRADE_LOG_FILE):
        await update.message.reply_document(document=open(TRADE_LOG_FILE, "rb"))
    if not files:
        await update.message.reply_text("Inga loggar ännu.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start – starta engine\n"
        "/stop – stoppa engine\n"
        "/status – visa status\n"
        "/set_ai MODE – byt AI-läge\n"
        "/mock_trade SYMBOL – simulera trade\n"
        "/backtest SYMBOL TID [avgift] – kör backtest\n"
        "/export_csv – exportera loggar\n"
        "/help – denna hjälp"
    )

# ==============================
# TELEGRAM APP
# ==============================
tg_app = Application.builder().token(TELEGRAM_TOKEN).build()
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("stop", stop))
tg_app.add_handler(CommandHandler("status", status))
tg_app.add_handler(CommandHandler("set_ai", set_ai))
tg_app.add_handler(CommandHandler("mock_trade", mock_trade))
tg_app.add_handler(CommandHandler("export_csv", export_csv))
tg_app.add_handler(CommandHandler("help", help_cmd))

# ==============================
# MAIN
# ==============================
async def main():
    asyncio.create_task(engine_loop(tg_app))
    await tg_app.initialize()
    await tg_app.start()
    await tg_app.updater.start_polling()
    await tg_app.updater.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
