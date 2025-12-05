#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################
#   MP ORBbot – KuCoin Spot – Mock + Live – Telegram UI       #
###############################################################

import os
import time
import json
import math
import hmac
import base64
import hashlib
import threading
import requests
from datetime import datetime
from collections import deque

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

###############################################################
#  ENVIRONMENT VARIABLES (INGEN HÅRDKODNING!)
###############################################################

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

if not TELEGRAM_TOKEN:
    raise Exception("🔴 FEL: TELEGRAM_TOKEN saknas i miljövariablerna!")

BASE_URL = "https://api.kucoin.com"

###############################################################
#  KONFIGURATION
###############################################################

ACTIVE_COINS = ["BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT", "ADA-USDT"]

bot_running = False
bot_mode = "mock"
ai_mode = "neutral"
entry_mode = "retest"

MOCK_SPREAD = 0.0002     # 0.02 %
MOCK_SLIPPAGE = 0.0001   # 0.01 %

TRADE_SIZE_FILE = "trade_size.json"
DEFAULT_TRADE_SIZE = 30


###############################################################
#  TRADE SIZE FILHANTERING
###############################################################

def load_trade_size() -> int:
    if not os.path.exists(TRADE_SIZE_FILE):
        save_trade_size(DEFAULT_TRADE_SIZE)
        return DEFAULT_TRADE_SIZE

    try:
        with open(TRADE_SIZE_FILE, "r") as f:
            data = json.load(f)
            return int(data.get("trade_size", DEFAULT_TRADE_SIZE))
    except:
        return DEFAULT_TRADE_SIZE


def save_trade_size(size: int):
    with open(TRADE_SIZE_FILE, "w") as f:
        json.dump({"trade_size": int(size)}, f)


trade_size_usdt = load_trade_size()


###############################################################
#  KUCOIN HJÄLPMETODER
###############################################################

def kucoin_headers(method: str, endpoint: str, body: dict | None = None):
    now = int(time.time() * 1000)
    payload = json.dumps(body) if body else ""

    msg = f"{now}{method}{endpoint}{payload}"

    signature = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode(), msg.encode(), hashlib.sha256
        ).digest()
    ).decode()

    passphrase = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode(), KUCOIN_API_PASSPHRASE.encode(), hashlib.sha256
        ).digest()
    ).decode()

    return {
        "KC-API-KEY": KUCOIN_API_KEY,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }


def get_kucoin_candles(symbol: str, interval="3min", limit=100):
    url = f"{BASE_URL}/api/v1/market/candles?type={interval}&symbol={symbol}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "data" not in data:
            return []
        candles = []
        for c in data["data"]:
            candles.append({
                "timestamp": int(c[0]),
                "open": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "volume": float(c[5]),
            })
        candles.reverse()
        return candles[:limit]
    except:
        return []


###############################################################
#  ORB STATE & LOGIK
###############################################################

class ORBState:
    def __init__(self):
        self.current_orb = None
        self.entry_price = None
        self.stop_price = None
        self.in_position = False
        self.trailing_active = False


orb_states = {s: ORBState() for s in ACTIVE_COINS}


def detect_new_orb(c1, c2):
    return (c1["close"] < c1["open"]) and (c2["close"] > c2["open"])


def create_orb(candle):
    return {
        "high": candle["high"],
        "low": candle["low"],
        "timestamp": candle["timestamp"],
    }


def should_enter(candle, orb):
    return candle["high"] >= orb["high"]


def ai_filter_pass(candle, prev_list, mode):
    body = abs(candle["close"] - candle["open"])
    wick = candle["high"] - max(candle["close"], candle["open"])

    if mode in ["neutral", "forsiktig"]:
        if body < wick:
            return False

    if mode == "forsiktig":
        if len(prev_list) > 0 and candle["close"] <= prev_list[-1]["close"]:
            return False

    return True


def mock_entry(price):
    return price * (1 + MOCK_SPREAD + MOCK_SLIPPAGE)


def mock_exit(price):
    return price * (1 - MOCK_SPREAD - MOCK_SLIPPAGE)


###############################################################
#  ENTRY & EXIT
###############################################################

def handle_entry(symbol, candle, state):
    global bot_mode, trade_size_usdt

    entry = state.current_orb["high"]

    if entry_mode == "retest":
        if candle["low"] > state.current_orb["low"] * 1.001:
            return False

    if bot_mode == "mock":
        adj = mock_entry(entry)
        state.entry_price = adj
        state.stop_price = state.current_orb["low"]
        state.in_position = True
        state.trailing_active = False
        print(f"[MOCK ENTRY] {symbol} @ {adj}")
        return True

    if not KUCOIN_API_KEY:
        print("[LIVE] API-nycklar saknas, mockar.")
        return False

    return True


def handle_exit(symbol, candle, state):
    global bot_mode

    exit_price = candle["low"]

    if bot_mode == "mock":
        adj = mock_exit(exit_price)
        print(f"[MOCK EXIT] {symbol} @ {adj}")
        state.in_position = False
        state.trailing_active = False
        return adj

    state.in_position = False
    state.trailing_active = False
    return exit_price


###############################################################
#  ORB LOOP PER SYMBOL
###############################################################

def run_orb(symbol):
    global bot_running, ai_mode

    state = orb_states[symbol]
    print(f"[ORB] Startar {symbol}")

    while True:
        try:
            if not bot_running:
                time.sleep(2)
                continue

            candles = get_kucoin_candles(symbol)
            if len(candles) < 10:
                time.sleep(1)
                continue

            last_c = candles[-1]
            prev_c = candles[-2]

            if detect_new_orb(prev_c, last_c):
                state.current_orb = create_orb(last_c)
                state.in_position = False
                state.trailing_active = False
                print(f"[ORB] Ny ORB i {symbol}: {state.current_orb}")

            if not state.current_orb:
                time.sleep(1)
                continue

            if not state.in_position and should_enter(last_c, state.current_orb):
                if ai_filter_pass(last_c, candles[-6:-1], ai_mode):
                    handle_entry(symbol, last_c, state)

            if state.in_position and last_c["low"] <= state.stop_price:
                handle_exit(symbol, last_c, state)

            time.sleep(1)

        except Exception as e:
            print(f"[ERROR {symbol}]", e)
            time.sleep(3)


def start_orb_threads():
    for symbol in ACTIVE_COINS:
        t = threading.Thread(target=run_orb, args=(symbol,), daemon=True)
        t.start()


###############################################################
#  TELEGRAM UI
###############################################################

def main_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶ Mock Start", callback_data="start_mock"),
            InlineKeyboardButton("🔥 Live Start", callback_data="start_live"),
        ],
        [InlineKeyboardButton("⛔ Stop", callback_data="stop")],
        [
            InlineKeyboardButton("🤖 AI", callback_data="ai_menu"),
            InlineKeyboardButton("🎯 Entry", callback_data="entry_menu"),
        ],
        [InlineKeyboardButton("💰 Trade Size", callback_data="size_menu")],
        [InlineKeyboardButton("📊 Status", callback_data="status")],
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("MP ORBbot – välj:", reply_markup=main_menu())


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode, ai_mode, entry_mode, trade_size_usdt

    q = update.callback_query
    await q.answer()
    cmd = q.data

    if cmd == "start_mock":
        bot_mode = "mock"
        bot_running = True
        await q.edit_message_text("Mock-läge startat.", reply_markup=main_menu())
        return

    if cmd == "start_live":
        bot_mode = "live"
        bot_running = True
        await q.edit_message_text("Live-läge aktiverat.", reply_markup=main_menu())
        return

    if cmd == "stop":
        bot_running = False
        await q.edit_message_text("Stoppad.", reply_markup=main_menu())
        return

    if cmd == "status":
        await q.edit_message_text(
            f"Mode: {bot_mode}\n"
            f"AI: {ai_mode}\n"
            f"Entry: {entry_mode}\n"
            f"Trade size: {trade_size_usdt}",
            reply_markup=main_menu()
        )
        return

    if cmd == "ai_menu":
        kb = [
            [InlineKeyboardButton("Aggressiv", callback_data="ai_aggr")],
            [InlineKeyboardButton("Neutral", callback_data="ai_neut")],
            [InlineKeyboardButton("Försiktig", callback_data="ai_safe")],
        ]
        await q.edit_message_text("Välj AI:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("ai_"):
        if cmd == "ai_aggr":
            ai_mode = "aggressiv"
        elif cmd == "ai_neut":
            ai_mode = "neutral"
        elif cmd == "ai_safe":
            ai_mode = "forsiktig"

        await q.edit_message_text(f"AI: {ai_mode}", reply_markup=main_menu())
        return

    if cmd == "entry_menu":
        kb = [
            [InlineKeyboardButton("Close", callback_data="entry_close")],
            [InlineKeyboardButton("Tick", callback_data="entry_tick")],
            [InlineKeyboardButton("Retest", callback_data="entry_retest")],
        ]
        await q.edit_message_text("Entry-mode:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("entry_"):
        if cmd == "entry_close":
            entry_mode = "close"
        elif cmd == "entry_tick":
            entry_mode = "tick"
        elif cmd == "entry_retest":
            entry_mode = "retest"

        await q.edit_message_text(f"Entry: {entry_mode}", reply_markup=main_menu())
        return

    if cmd == "size_menu":
        kb = [
            [InlineKeyboardButton("10 USDT", callback_data="size_10"),
             InlineKeyboardButton("20 USDT", callback_data="size_20")],
            [InlineKeyboardButton("30 USDT", callback_data="size_30"),
             InlineKeyboardButton("50 USDT", callback_data="size_50")],
        ]
        await q.edit_message_text("Trade size:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("size_"):
        val = int(cmd.replace("size_", ""))
        trade_size_usdt = val
        save_trade_size(val)
        await q.edit_message_text(f"Trade size satt till {val}.", reply_markup=main_menu())
        return


###############################################################
#  MAIN
###############################################################

def main():
    print("ORBbot startar...")
    start_orb_threads()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))

    print("Telegram startar...")
    app.run_polling()


if __name__ == "__main__":
    main()
