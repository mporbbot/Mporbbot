import sys
try:
    import imghdr
except ModuleNotFoundError:
    import imghdr_py as imghdr
    sys.modules['imghdr'] = imghdr

import logging
import pandas as pd
import numpy as np
import requests
from kucoin.client import Market, Trade
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
from datetime import datetime

# ===== SETTINGS =====
API_KEY = os.getenv("KUCOIN_API_KEY")
API_SECRET = os.getenv("KUCOIN_API_SECRET")
API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

SYMBOLS = ["ADAUSDT", "LINKUSDT", "XRPUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAME = "3min"
MOCK_TRADE_SIZE = 30  # USDT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client_market = Market(url='https://api.kucoin.com')
client_trade = Trade(key=API_KEY, secret=API_SECRET, passphrase=API_PASSPHRASE, is_sandbox=False)

engine_running = False
mock_mode = True
positions = {}

# ====== FUNCTIONS ======

def fetch_klines(symbol, limit=20):
    data = client_market.get_kline(symbol, TIMEFRAME, limit=limit)
    df = pd.DataFrame(data, columns=['time','open','close','high','low','volume','turnover'])
    df = df.astype(float)
    return df[::-1].reset_index(drop=True)

def check_orb_entry(symbol):
    df = fetch_klines(symbol, limit=5)
    first_candle = df.iloc[0]
    last_candle = df.iloc[-1]
    orb_high = first_candle['high']
    orb_low = first_candle['low']

    # Endast long-entry om senaste candle stänger över ORB-high
    if last_candle['close'] > orb_high:
        sl = last_candle['low']  # Stop loss = senaste candles botten
        return True, sl
    return False, None

def place_mock_trade(symbol, sl):
    global positions
    positions[symbol] = {"sl": sl, "entry": True}
    logger.info(f"[MOCK] Köpt {symbol}, SL = {sl}")

def update_stop_loss(symbol):
    global positions
    df = fetch_klines(symbol, limit=1)
    new_low = df.iloc[-1]['low']
    if new_low > positions[symbol]['sl']:
        positions[symbol]['sl'] = new_low
        logger.info(f"[MOCK] Ny SL för {symbol}: {new_low}")

def engine_job():
    if not engine_running:
        return
    for symbol in SYMBOLS:
        if symbol not in positions:
            entry, sl = check_orb_entry(symbol)
            if entry:
                if mock_mode:
                    place_mock_trade(symbol, sl)
        else:
            update_stop_loss(symbol)

# ===== TELEGRAM COMMANDS =====
def start_mock(update, context):
    global mock_mode, engine_running
    mock_mode = True
    update.message.reply_text("MOCK-trading startad.")
    engine_running = True

def start_live(update, context):
    global mock_mode, engine_running
    mock_mode = False
    update.message.reply_text("LIVE-trading startad.")
    engine_running = True

def stop_engine(update, context):
    global engine_running
    engine_running = False
    update.message.reply_text("Motorn stoppad.")

def status(update, context):
    mode = "MOCK" if mock_mode else "LIVE"
    update.message.reply_text(f"Läge: {mode}\nEngine: {'På' if engine_running else 'Av'}\nPositions: {positions}")

def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start_mock", start_mock))
    dp.add_handler(CommandHandler("start_live", start_live))
    dp.add_handler(CommandHandler("stop", stop_engine))
    dp.add_handler(CommandHandler("status", status))

    updater.start_polling()

    import time
    while True:
        engine_job()
        time.sleep(10)

if __name__ == '__main__':
    main()
