###############################################
# Mp ORBbot – Regime Switching Edition
# Hybrid Momentum + Mean Reversion Trading Bot
###############################################

import pandas as pd
import numpy as np
import asyncio
import datetime
import math
from kucoin.client import Market
from aiogram import Bot, Dispatcher, executor, types

API_KEY = "DIN_API"
API_SECRET = "DIN_SECRET"
API_PASSPHRASE = "DIN_PASSPHRASE"

TELEGRAM_TOKEN = "DIN_TELEGRAM_TOKEN"

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

###############################################
# GLOBALA INSTÄLLNINGAR
###############################################

TRADE_AMOUNT = 30        # Mock-läge
STOP_LOSS_PCT = 1.0      # Standard 1 % SL
THRESHOLD = 25           # Momentum-score threshold
RUNNING = False
MOCK_MODE = True

CURRENT_REGIME = "TREND"     # TREND eller RANGE
LAST_STATUS = ""

client = Market(url='https://api.kucoin.com')

###############################################
# TEKNISKA INDIKATORER
###############################################

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def true_range(df):
    tr1 = abs(df["high"] - df["low"])
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    tr = true_range(df)
    return tr.rolling(period).mean()

def bollinger(df, period=20, std=2):
    ma = df["close"].rolling(period).mean()
    dev = df["close"].rolling(period).std()
    upper = ma + dev * std
    lower = ma - dev * std
    return upper, lower

###############################################
# REGIME DETECTION
###############################################

def detect_regime(df):
    global CURRENT_REGIME

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)

    df["atr14"] = atr(df, 14)
    df["bb_up"], df["bb_lo"] = bollinger(df, 20, 2)

    ema_slope = df["ema20"].diff().iloc[-1]
    atr_slope = df["atr14"].diff().iloc[-1]
    bb_width = (df["bb_up"].iloc[-1] - df["bb_lo"].iloc[-1]) / df["close"].iloc[-1]

    # TREND-logik
    trend_cond = (
        df["ema20"].iloc[-1] > df["ema50"].iloc[-1] and
        ema_slope > 0 and
        atr_slope > 0 and
        bb_width > 0.02
    )

    if trend_cond:
        CURRENT_REGIME = "TREND"
    else:
        CURRENT_REGIME = "RANGE"

    return CURRENT_REGIME

###############################################
# MOMENTUM ENTRY
###############################################

def momentum_score(df):
    last = df.iloc[-1]

    candle_strength = (last["close"] - last["open"]) / (last["high"] - last["low"] + 1e-9)
    range_factor = (last["high"] - last["low"]) / df["close"].median()
    volume_factor = 1.0  # placeholder

    score = candle_strength * 10 + range_factor * 20 + volume_factor * 10
    return max(0, round(score, 2))

def check_momentum_entry(df):
    score = momentum_score(df)
    if score >= THRESHOLD:
        return ("LONG", score)
    return (None, score)

###############################################
# MEAN REVERSION ENTRY
###############################################

def check_mean_reversion_entry(df):
    last = df.iloc[-1]

    if last["close"] < last["bb_lo"]:
        return ("LONG", "MR-LONG")

    return (None, None)

###############################################
# STOP LOSS
###############################################

def compute_sl(entry, sl_pct=1.0):
    return entry * (1 - sl_pct / 100)

###############################################
# HÄMTA DATA
###############################################

def get_data(symbol, lookback=200):
    raw = client.get_kline(symbol, "3min")
    df = pd.DataFrame(raw, columns=["time", "open", "close", "high", "low", "volume"])
    df = df.astype(float)
    df = df[::-1].reset_index(drop=True)
    return df.head(lookback)

###############################################
# HANDLE EN TRADINGLOOP
###############################################

async def run_symbol(symbol):
    global LAST_STATUS

    while RUNNING:
        df = get_data(symbol)

        # 1: Regime detection
        regime = detect_regime(df)

        # 2: Entry beroende på regime
        if regime == "TREND":
            side, score = check_momentum_entry(df)
        else:
            side, score = check_mean_reversion_entry(df)

        # 3: Om trade triggar
        if side == "LONG":
            entry = df["close"].iloc[-1]
            sl = compute_sl(entry, STOP_LOSS_PCT)

            LAST_STATUS = f"{symbol} | {regime} | Entry {entry} | SL {sl} | Score {score}"

        await asyncio.sleep(3)

###############################################
# TELEGRAM
###############################################

@dp.message_handler(commands=["start"])
async def start_cmd(msg: types.Message):
    global RUNNING
    RUNNING = True
    await msg.answer("Bot STARTAD (Regime Switching aktiv).")

@dp.message_handler(commands=["stop"])
async def stop_cmd(msg: types.Message):
    global RUNNING
    RUNNING = False
    await msg.answer("Bot STOPPAD.")

@dp.message_handler(commands=["sl"])
async def sl_cmd(msg: types.Message):
    global STOP_LOSS_PCT
    parts = msg.text.split()
    if len(parts) == 1:
        await msg.answer(f"Aktuellt SL: {STOP_LOSS_PCT}%")
        return

    try:
        val = float(parts[1])
        STOP_LOSS_PCT = val
        await msg.answer(f"Nytt SL: {val}%")
    except:
        await msg.answer("Fel format. Ex: /sl 1.0")

@dp.message_handler(commands=["risk"])
async def risk_cmd(msg: types.Message):
    global THRESHOLD
    try:
        val = int(msg.text.split()[1])
        THRESHOLD = val
        await msg.answer(f"Ny momentum-threshold: {val}")
    except:
        await msg.answer("Ex: /risk 25")

@dp.message_handler(commands=["status"])
async def status_cmd(msg: types.Message):
    await msg.answer(LAST_STATUS if LAST_STATUS else "Ingen trade än.")

###############################################
# MAIN
###############################################

async def main_loop():
    coins = ["ADA-USDT", "BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT"]

    tasks = [run_symbol(c) for c in coins]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main_loop())
    executor.start_polling(dp, skip_updates=True)
