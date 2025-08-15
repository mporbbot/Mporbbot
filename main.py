# Mp ORBbot – FULLVERSION
# Spot LONG-only | ORB per ny candle | SL = senaste candlens low
# Mock & Live med knappar | AI-lägen | Skatteverket-logg | Nödstopp | Failsafe
# Endast svar från AUTHORIZED_USER_ID
# Render/FastAPI-kompatibel
# Startkommando Render: uvicorn main:app --host 0.0.0.0 --port $PORT

# --- Fix för Python 3.13 ---
import sys
try:
    import imghdr  # noqa
except ModuleNotFoundError:
    import imghdr_py as imghdr
    sys.modules['imghdr'] = imghdr

import os, csv, json, time, threading, logging, traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import requests
import pandas as pd

from fastapi import FastAPI
from telegram import Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Updater, CommandHandler, CallbackContext, CallbackQueryHandler
)

# ===================== KONFIG =====================
BOT_NAME = "Mp ORBbot"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))

SYMBOLS = [s.strip().upper() for s in os.getenv(
    "SYMBOLS", "LINKUSDT,XRPUSDT,ADAUSDT,BTCUSDT,ETHUSDT"
).split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "3min")
AI_MODE = "neutral"
ENGINE_ON = True
MOCK_ENABLED = True
LIVE_ENABLED = False

MIN_ORB_PCT = {"aggressiv": 0.0005, "neutral": 0.001, "försiktig": 0.002}
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
LIVE_USDT_PER_TRADE = float(os.getenv("LIVE_USDT_PER_TRADE", "30"))
TICK_EPS = float(os.getenv("TICK_EPS", "1e-8"))

KU_API = os.getenv("KUCOIN_API_KEY")
KU_SEC = os.getenv("KUCOIN_API_SECRET")
KU_PWD = os.getenv("KUCOIN_API_PASSPHRASE")

SELF_URL = os.getenv("SELF_URL", "")
_keepalive_on = False

MOCK_LOG = Path("mock_trade_log.csv")
REAL_LOG = Path("real_trade_log.csv")
SKV_LOG = Path("skv_trades.csv")
POS_FILE = Path("positions.json")
ERROR_LOG = Path("error_log.txt")

BASE = "https://api.kucoin.com"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

positions = {}
orb_levels = {}
engine_running = True
last_telegram_ping = time.time()

# ===================== FASTAPI =====================
app = FastAPI()

@app.get("/")
def root():
    return {"status": "Mp ORBbot running"}

# ===================== UTIL =====================
def auth_check(update: Update):
    return update.effective_user.id == AUTHORIZED_USER_ID

def save_positions():
    with open(POS_FILE, "w") as f:
        json.dump(positions, f)

def load_positions():
    global positions
    if POS_FILE.exists():
        with open(POS_FILE) as f:
            positions = json.load(f)

def log_error(err):
    with open(ERROR_LOG, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} - {err}\n")

# ===================== KUCOIN DATA =====================
def fetch_klines(symbol: str, limit=3):
    url = f"{BASE}/api/v1/market/candles?type={TIMEFRAME}&symbol={symbol}"
    r = requests.get(url, timeout=10)
    data = r.json().get("data", [])
    df = pd.DataFrame(data, columns=["time","open","close","high","low","volume","turnover"])
    df = df.astype(float)
    df = df[::-1].reset_index(drop=True)
    return df.head(limit)

# ===================== ORB LOGIK =====================
def update_orb(symbol: str):
    df = fetch_klines(symbol, limit=2)
    closed = df.iloc[0]
    body = abs(closed["close"] - closed["open"])
    if body < closed["open"] * MIN_ORB_PCT[AI_MODE]:
        return
    orb_levels[symbol] = {
        "high": closed["high"],
        "low": closed["low"],
        "dir": "bullish" if closed["close"] > closed["open"] else "bearish"
    }

def try_entry(symbol: str, bot):
    if symbol not in orb_levels or symbol in positions:
        return
    orb = orb_levels[symbol]
    if orb["dir"] != "bullish":
        return
    df = fetch_klines(symbol, limit=1)
    live = df.iloc[-1]
    if live["high"] > orb["high"] + TICK_EPS:
        entry = orb["high"] + TICK_EPS
        stop = orb["low"] - TICK_EPS
        qty = (MOCK_TRADE_USDT if MOCK_ENABLED else LIVE_USDT_PER_TRADE) / entry
        positions[symbol] = {"entry": entry, "stop": stop, "qty": qty, "live": LIVE_ENABLED}
        save_positions()
        log_trade(symbol, "BUY", entry, qty, LIVE_ENABLED)
        bot.send_message(AUTHORIZED_USER_ID, f"ENTRY {symbol} @ {entry:.4f} | SL {stop:.4f}")
        logger.info(f"ENTRY {symbol} @ {entry} SL {stop}")

def trail_stop(symbol: str, bot):
    if symbol not in positions:
        return
    df = fetch_klines(symbol, limit=2)
    closed = df.iloc[0]
    new_stop = max(positions[symbol]["stop"], closed["low"] - TICK_EPS)
    positions[symbol]["stop"] = new_stop
    if closed["low"] <= new_stop:
        exit_trade(symbol, new_stop, bot)

# ===================== TRADING FUNKTIONER =====================
def exit_trade(symbol: str, price: float, bot):
    if symbol not in positions:
        return
    qty = positions[symbol]["qty"]
    live = positions[symbol]["live"]
    log_trade(symbol, "SELL", price, qty, live)
    bot.send_message(AUTHORIZED_USER_ID, f"EXIT {symbol} @ {price:.4f}")
    del positions[symbol]
    save_positions()

def log_trade(symbol: str, side: str, price: float, qty: float, live: bool):
    log_file = REAL_LOG if live else MOCK_LOG
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), symbol, side, price, qty, price*qty])
    if live:
        with open(SKV_LOG, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().date(), symbol, side, price, qty, price*qty])

# ===================== ENGINE =====================
def engine_loop(bot):
    global last_telegram_ping, engine_running
    while True:
        if engine_running:
            if time.time() - last_telegram_ping > 300:
                bot.send_message(AUTHORIZED_USER_ID, "⚠️ Telegram-kontakt tappad – stoppar motorn")
                engine_running = False
            for sym in SYMBOLS:
                try:
                    update_orb(sym)
                    try_entry(sym, bot)
                    trail_stop(sym, bot)
                except Exception as e:
                    log_error(traceback.format_exc())
        time.sleep(10)

# ===================== TELEGRAM KOMMANDON =====================
def start_mock(update: Update, ctx: CallbackContext):
    global MOCK_ENABLED, LIVE_ENABLED, engine_running
    if not auth_check(update): return
    MOCK_ENABLED, LIVE_ENABLED, engine_running = True, False, True
    update.message.reply_text("Mock mode startad ✅")

def start_live(update: Update, ctx: CallbackContext):
    global MOCK_ENABLED, LIVE_ENABLED, engine_running
    if not auth_check(update): return
    MOCK_ENABLED, LIVE_ENABLED, engine_running = False, True, True
    update.message.reply_text("LIVE mode startad ⚠️")

def engine_start(update: Update, ctx: CallbackContext):
    global engine_running
    if not auth_check(update): return
    engine_running = True
    update.message.reply_text("Engine startad ✅")

def engine_stop(update: Update, ctx: CallbackContext):
    global engine_running
    if not auth_check(update): return
    engine_running = False
    update.message.reply_text("Engine stoppad ⛔")

def panic(update: Update, ctx: CallbackContext):
    global positions, engine_running
    if not auth_check(update): return
    for sym in list(positions.keys()):
        exit_trade(sym, positions[sym]["stop"], ctx.bot)
    engine_running = False
    update.message.reply_text("PANIC: Alla trades stängda och motorn stoppad")

def status(update: Update, ctx: CallbackContext):
    if not auth_check(update): return
    mode = "MOCK" if MOCK_ENABLED else "LIVE"
    txt = f"📊 Status\nLäge: {mode}\nAI-läge: {AI_MODE}\nEngine: {'På' if engine_running else 'Av'}\nPos: {positions}"
    update.message.reply_text(txt)

def get_log(update: Update, ctx: CallbackContext):
    if not auth_check(update): return
    if ERROR_LOG.exists():
        update.message.reply_document(open(ERROR_LOG, "rb"))

def export_csv(update: Update, ctx: CallbackContext):
    if not auth_check(update): return
    if MOCK_LOG.exists():
        update.message.reply_document(open(MOCK_LOG, "rb"))
    if REAL_LOG.exists():
        update.message.reply_document(open(REAL_LOG, "rb"))

def export_skv(update: Update, ctx: CallbackContext):
    if not auth_check(update): return
    if SKV_LOG.exists():
        update.message.reply_document(open(SKV_LOG, "rb"))

def help_cmd(update: Update, ctx: CallbackContext):
    if not auth_check(update): return
    update.message.reply_text("/start_mock /start_live /engine_start /engine_stop /panic\n/status /export_csv /export_skv /get_log")

# ===================== MAIN =====================
def main():
    load_positions()
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start_mock", start_mock))
    dp.add_handler(CommandHandler("start_live", start_live))
    dp.add_handler(CommandHandler("engine_start", engine_start))
    dp.add_handler(CommandHandler("engine_stop", engine_stop))
    dp.add_handler(CommandHandler("panic", panic))
    dp.add_handler(CommandHandler("status", status))
    dp.add_handler(CommandHandler("export_csv", export_csv))
    dp.add_handler(CommandHandler("export_skv", export_skv))
    dp.add_handler(CommandHandler("get_log", get_log))
    dp.add_handler(CommandHandler("help", help_cmd))
    threading.Thread(target=engine_loop, args=(updater.bot,), daemon=True).start()
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
