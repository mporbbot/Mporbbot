#########################################################
# Mp HybridBot – Regime Switching PRO
# Hybrid EMA Momentum + Mean Reversion
# Automatisk Regime Switching (EMA20/50 + ATR + BB)
# Mock + Live trading
# KuCoin Spot CSV-export (A-format, exakt 1:1)
# Telegram kontrollpanel + knappar + stats + testbuy
#########################################################

import pandas as pd
import numpy as np
import asyncio
import datetime
import csv
import os

from aiogram import Bot, Dispatcher, executor, types
from kucoin.client import Market

#########################################################
# KONFIG
#########################################################

TELEGRAM_TOKEN = "DIN_TELEGRAM_TOKEN_HÄR"

KUCOIN_API = "DIN_API_KEY"
KUCOIN_SECRET = "DIN_SECRET"
KUCOIN_PASSPHRASE = "DIN_PASSPHRASE"

SYMBOLS = ["ADA-USDT", "BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT"]

# ----- Trading-inställningar -----
TRADE_AMOUNT = 30.0          # mock-belopp
STOP_LOSS_PCT = 1.0          # styr via /sl 1.0
THRESHOLD = 25               # momentum score gräns via /risk

RUNNING = False
MOCK_MODE = True

CURRENT_REGIME = "TREND"
LAST_STATUS = "Ingen signal ännu."

# Stats
signal_history = []
trade_count = 0
win_count = 0
loss_count = 0
open_positions = {}          # per symbol

# CSV-loggar
MOCK_LOG = "mock_trade_log.csv"
LIVE_LOG = "live_trade_log.csv"

#########################################################
# INIT
#########################################################

bot = Bot(TELEGRAM_TOKEN)
dp = Dispatcher(bot)

client = Market(url="https://api.kucoin.com")

#########################################################
# CSV – KuCoin Spot-format (A)
#########################################################

def write_csv(spottype="mock", row=None):
    """
    spottype = "mock" eller "live"
    row = {
        "Time": ..., "Symbol": ..., "Side": ...,
        "Order Type": "market",
        "Price": float,
        "Size": float,
        "Fee": float,
        "Fee Coin": "USDT",
        "Amount": float,
        "Remark": str
    }
    """

    filename = MOCK_LOG if spottype == "mock" else LIVE_LOG
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Time","Symbol","Side","Order Type",
            "Price","Size","Fee","Fee Coin","Amount","Remark"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

#########################################################
# Hjälpfunktioner
#########################################################

def now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def parse_float(x):
    try: return float(x)
    except: return 0.0

#########################################################
# Tekniska indikatorer
#########################################################

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def true_range(df):
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).rolling(period).mean()

def bollinger(df, period=20, std=2):
    ma = df["close"].rolling(period).mean()
    dev = df["close"].rolling(period).std()
    return ma + dev*std, ma - dev*std

#########################################################
# Regime detection
#########################################################

def detect_regime(df):
    global CURRENT_REGIME

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr14"] = atr(df, 14)
    df["bb_up"], df["bb_lo"] = bollinger(df, 20, 2)

    ema_slope = df["ema20"].diff().iloc[-1]
    atr_slope = df["atr14"].diff().iloc[-1]
    bb_width = (df["bb_up"].iloc[-1] - df["bb_lo"].iloc[-1]) / df["close"].iloc[-1]

    trend_cond = (
        df["ema20"].iloc[-1] > df["ema50"].iloc[-1] and
        ema_slope > 0 and
        atr_slope > 0 and
        bb_width > 0.02
    )

    CURRENT_REGIME = "TREND" if trend_cond else "RANGE"
    return CURRENT_REGIME

#########################################################
# Momentum-score
#########################################################

def momentum_score(df):
    last = df.iloc[-1]

    candle_strength = (last["close"] - last["open"]) / (last["high"] - last["low"] + 1e-9)
    range_factor = (last["high"] - last["low"]) / last["close"]
    score = candle_strength * 10 + range_factor * 20 + 10

    return max(0, round(score, 2))

#########################################################
# ENTRY: Momentum
#########################################################

def momentum_entry(df):
    score = momentum_score(df)
    if score >= THRESHOLD:
        return "LONG", score
    return None, score

#########################################################
# ENTRY: Mean Reversion
#########################################################

def mean_reversion_entry(df):
    last = df.iloc[-1]
    if last["close"] < last["bb_lo"]:
        return "LONG", "MR"
    return None, None

#########################################################
# Stop Loss
#########################################################

def compute_sl(entry_price):
    return entry_price * (1 - STOP_LOSS_PCT / 100)

#########################################################
# Hämta 3m-data
#########################################################

def get_data(symbol, limit=200):
    raw = client.get_kline(symbol, "3min")
    df = pd.DataFrame(raw, columns=["time","open","close","high","low","volume"])
    df = df.astype(float)
    df = df[::-1].reset_index(drop=True)
    return df.head(limit)

#########################################################
# Handel per coin
#########################################################

async def run_symbol(symbol):
    global LAST_STATUS
    global trade_count, win_count, loss_count

    while RUNNING:
        df = get_data(symbol)
        regime = detect_regime(df)
        last_close = df["close"].iloc[-1]

        if regime == "TREND":
            side, score = momentum_entry(df)
        else:
            side, score = mean_reversion_entry(df)

        # ----- ENTRY -----
        if side == "LONG" and symbol not in open_positions:
            entry = last_close
            sl = compute_sl(entry)

            open_positions[symbol] = {
                "entry": entry,
                "sl": sl,
                "time": now(),
                "regime": regime,
                "strategy": "MOMENTUM" if regime=="TREND" else "MEAN-REV"
            }

            # CSV BUY
            write_csv("mock", {
                "Time": now(),
                "Symbol": symbol,
                "Side": "BUY",
                "Order Type": "market",
                "Price": entry,
                "Size": TRADE_AMOUNT,
                "Fee": round(entry * TRADE_AMOUNT * 0.001, 4),
                "Fee Coin": "USDT",
                "Amount": round(entry * TRADE_AMOUNT, 4),
                "Remark": "mock entry"
            })

            LAST_STATUS = f"{symbol} | {regime} | ENTRY {entry} | SL {sl} | Score {score}"

        # ----- EXIT (SL) -----
        if symbol in open_positions:
            pos = open_positions[symbol]

            if last_close <= pos["sl"]:  # SL träffad
                entry = pos["entry"]
                pnl = (last_close - entry) * TRADE_AMOUNT

                # CSV SELL
                write_csv("mock", {
                    "Time": now(),
                    "Symbol": symbol,
                    "Side": "SELL",
                    "Order Type": "market",
                    "Price": last_close,
                    "Size": TRADE_AMOUNT,
                    "Fee": round(last_close * TRADE_AMOUNT * 0.001, 4),
                    "Fee Coin": "USDT",
                    "Amount": round(last_close * TRADE_AMOUNT, 4),
                    "Remark": f"SL exit PnL={round(pnl,4)}"
                })

                trade_count += 1
                if pnl >= 0: win_count += 1
                else: loss_count += 1

                LAST_STATUS = f"{symbol} | SL EXIT {last_close} | PnL {round(pnl,4)}"
                del open_positions[symbol]

        await asyncio.sleep(3)

#########################################################
# Telegram-knappar
#########################################################

def keyboard():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row("/start", "/stop")
    kb.row("/status", "/stats")
    kb.row("/testbuy ADA-USDT")
    kb.row("/risk 25", "/sl 1.0")
    kb.row("/export_mock")
    return kb

#########################################################
# Telegram-kommandon
#########################################################

@dp.message_handler(commands=["start"])
async def start_cmd(msg: types.Message):
    global RUNNING
    RUNNING = True
    await msg.answer("Bot **STARTAD**.", reply_markup=keyboard())

    for s in SYMBOLS:
        asyncio.create_task(run_symbol(s))

@dp.message_handler(commands=["stop"])
async def stop_cmd(msg: types.Message):
    global RUNNING
    RUNNING = False
    await msg.answer("Bot **STOPPAD**.")

@dp.message_handler(commands=["status"])
async def status_cmd(msg: types.Message):
    await msg.answer(LAST_STATUS)

@dp.message_handler(commands=["sl"])
async def sl_cmd(msg: types.Message):
    global STOP_LOSS_PCT
    parts = msg.text.split()
    if len(parts) == 2:
        STOP_LOSS_PCT = float(parts[1])
        await msg.answer(f"Stoploss satt till {STOP_LOSS_PCT} %")
    else:
        await msg.answer(f"Aktuell SL: {STOP_LOSS_PCT} %")

@dp.message_handler(commands=["risk"])
async def risk_cmd(msg: types.Message):
    global THRESHOLD
    parts = msg.text.split()
    if len(parts) == 2:
        THRESHOLD = int(parts[1])
        await msg.answer(f"Risk/threshold satt till {THRESHOLD}")
    else:
        await msg.answer(f"Aktuell threshold: {THRESHOLD}")

@dp.message_handler(commands=["stats"])
async def stats_cmd(msg: types.Message):
    wr = 0.0
    if trade_count > 0:
        wr = round(win_count / trade_count * 100, 2)

    text = (
        f"📊 **STATISTIK**\n"
        f"---------------\n"
        f"Trades: {trade_count}\n"
        f"Vinster: {win_count}\n"
        f"Förluster: {loss_count}\n"
        f"Winrate: {wr} %\n"
        f"Öppna positioner: {len(open_positions)}\n"
        f"Regime: {CURRENT_REGIME}\n"
        f"Threshold: {THRESHOLD}\n"
        f"SL: {STOP_LOSS_PCT}%\n"
    )
    await msg.answer(text)

@dp.message_handler(commands=["testbuy"])
async def testbuy_cmd(msg: types.Message):
    symbol = msg.text.split()[1]
    price = get_data(symbol).iloc[-1]["close"]

    write_csv("mock", {
        "Time": now(),
        "Symbol": symbol,
        "Side": "BUY",
        "Order Type": "market",
        "Price": price,
        "Size": TRADE_AMOUNT,
        "Fee": round(price * TRADE_AMOUNT * 0.001, 4),
        "Fee Coin": "USDT",
        "Amount": round(price * TRADE_AMOUNT, 4),
        "Remark": "TESTBUY"
    })

    await msg.answer(f"TestBuy loggad för {symbol} @ {price}")

@dp.message_handler(commands=["export_mock"])
async def export_mock_cmd(msg: types.Message):
    if os.path.isfile(MOCK_LOG):
        await msg.answer_document(open(MOCK_LOG, "rb"))
    else:
        await msg.answer("Ingen mock-log finnes ännu.")

#########################################################
# MAIN
#########################################################

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
