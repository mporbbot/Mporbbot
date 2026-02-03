import os
import csv
import time
import json
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mp_orbbot_mr_momentum")

STRATEGY_NAME = "MR Entry + Momentum Exit"
STRATEGY_CODE = "MR_MOMENTUM_EXIT"
EXCHANGE_NAME = "KuCoin"
KUCOIN_BASE = "https://api.kucoin.com"

DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

TF_TREND = "5min"
TF_ENTRY = "1min"
TREND_CANDLES = 210
ENTRY_CANDLES = 210

ENGINE_LOOP_SEC = 3
COOLDOWN_PER_COIN_SEC = 30
MAX_POSITIONS = 5

DEFAULT_FEE_RATE_PER_SIDE = 0.0010
DEFAULT_SLIPPAGE_RATE_PER_SIDE = 0.0002
DEFAULT_STAKE_USDT = 30.0

DEFAULT_MR_BAND_PCT = 1.2
DEFAULT_MR_RSI_MAX = 30
DEFAULT_MR_VWAP_LOOKBACK_MIN = 120
DEFAULT_MR_TREND_FILTER = False

DEFAULT_TP_PCT = 0.8
DEFAULT_SL_PCT = 1.2
DEFAULT_TRAIL_ACTIVATE_PCT = 0.6
DEFAULT_TRAIL_DIST_PCT = 0.4

MOCK_LOG_PATH = "mock_trade_log.csv"
STATE_PATH = "bot_state.json"

def utc_now():
    return datetime.now(timezone.utc)

def ts():
    return int(time.time())

def ema(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def vwap(candles):
    pv = vol = 0.0
    for c in candles:
        h, l, cl, v = float(c[3]), float(c[4]), float(c[2]), float(c[5])
        tp = (h + l + cl) / 3
        pv += tp * v
        vol += v
    return pv / vol if vol > 0 else None

class KuCoinPublic:
    def get_candles(self, symbol, tf, limit):
        sec = {"1min":60,"5min":300}[tf]
        end = ts()
        start = end - sec * limit
        r = requests.get(
            f"{KUCOIN_BASE}/api/v1/market/candles",
            params={"symbol":symbol,"type":tf,"startAt":start,"endAt":end},
            timeout=10
        )
        data = r.json()["data"]
        return list(reversed(data))[-limit:]

    def get_price(self, symbol):
        r = requests.get(f"{KUCOIN_BASE}/api/v1/market/orderbook/level1",
                         params={"symbol":symbol}, timeout=10)
        return float(r.json()["data"]["price"])

@dataclass
class Position:
    symbol: str
    qty: float
    entry: float
    sl: float
    tp_arm: float
    trail_on: bool
    trail_high: float
    trail_stop: float

@dataclass
class Settings:
    engine_on: bool = False
    stake: float = DEFAULT_STAKE_USDT
    mr_band: float = DEFAULT_MR_BAND_PCT
    mr_rsi: int = DEFAULT_MR_RSI_MAX
    tp: float = DEFAULT_TP_PCT
    sl: float = DEFAULT_SL_PCT
    trail_activate: float = DEFAULT_TRAIL_ACTIVATE_PCT
    trail_dist: float = DEFAULT_TRAIL_DIST_PCT
class Engine:
    def __init__(self):
        self.k = KuCoinPublic()
        self.settings = Settings()
        self.positions: Dict[str, Position] = {}

    def step(self):
        if not self.settings.engine_on:
            return []

        msgs = []
        for sym in DEFAULT_COINS:
            if sym in self.positions:
                msgs += self.manage(sym)
            else:
                m = self.try_entry(sym)
                if m:
                    msgs.append(m)
        return msgs

    def try_entry(self, symbol):
        candles = self.k.get_candles(symbol, "1min", ENTRY_CANDLES)
        closes = [float(c[2]) for c in candles]
        vw = vwap(candles[-self.settings.mr_band:])
        r = rsi(closes)
        price = closes[-1]

        if vw is None or r is None:
            return None

        dev = (price - vw) / vw * 100
        if dev <= -self.settings.mr_band and r <= self.settings.mr_rsi:
            qty = self.settings.stake / price
            self.positions[symbol] = Position(
                symbol, qty, price,
                price * (1 - self.settings.sl/100),
                price * (1 + self.settings.tp/100),
                False, price, price
            )
            return f"ENTRY {symbol} @ {price:.4f}"
        return None

    def manage(self, symbol):
        pos = self.positions[symbol]
        price = self.k.get_price(symbol)
        msgs = []

        if price <= pos.sl:
            del self.positions[symbol]
            return [f"SL EXIT {symbol}"]

        if not pos.trail_on and price >= pos.tp_arm:
            pos.trail_on = True
            pos.trail_high = price
            pos.trail_stop = price * (1 - self.settings.trail_dist/100)
            msgs.append(f"TRAIL ARMED {symbol}")

        if pos.trail_on:
            if price > pos.trail_high:
                pos.trail_high = price
                pos.trail_stop = price * (1 - self.settings.trail_dist/100)
            if price <= pos.trail_stop:
                del self.positions[symbol]
                msgs.append(f"TRAIL EXIT {symbol}")

        return msgs

ENGINE = Engine()

async def loop(app):
    while True:
        msgs = await asyncio.get_event_loop().run_in_executor(None, ENGINE.step)
        if msgs:
            chat = app.bot_data.get("chat")
            if chat:
                for m in msgs:
                    await app.bot.send_message(chat, m)
        await asyncio.sleep(ENGINE_LOOP_SEC)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data["chat"] = update.effective_chat.id
    await update.message.reply_text("Bot started")

async def engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.engine_on = True
    await update.message.reply_text("ENGINE ON")

async def engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.engine_on = False
    await update.message.reply_text("ENGINE OFF")

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("engine_on", engine_on))
    app.add_handler(CommandHandler("engine_off", engine_off))
    app.create_task(loop(app))
    app.run_polling()

if __name__ == "__main__":
    main()
