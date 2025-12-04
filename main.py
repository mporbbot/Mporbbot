#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################
#   MP ORBbot – KuCoin Spot – Mock + Live – Full Telegram UI  #
#   DEL 1/3                                                    #
###############################################################

import requests
import time
import json
import hmac
import base64
import hashlib
import threading
import traceback
import datetime
import math
import os
from urllib.parse import urlencode
from collections import deque

from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    CallbackQueryHandler, ContextTypes
)

###############################################################
#  CONFIG
###############################################################

TELEGRAM_TOKEN = "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"

# KuCoin Spot API keys (lägg in egna nycklar när du vill)
KUCOIN_API_KEY = ""
KUCOIN_API_SECRET = ""
KUCOIN_API_PASSPHRASE = ""

BASE_URL = "https://api.kucoin.com"

# Default Trade Size (USDT) – ändras via /set_size
TRADE_SIZE_FILE = "trade_size.json"
DEFAULT_TRADE_SIZE = 30

# Coins boten handlar
ACTIVE_COINS = ["BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT", "ADA-USDT"]

# Botläge
bot_running = False
bot_mode = "mock"   # mock / live
ai_mode = "neutral" # aggressiv / neutral / försiktig
entry_mode = "retest"  # close / tick / retest

###############################################################
#  LOAD & SAVE TRADE SIZE
###############################################################

def load_trade_size():
    if not os.path.exists(TRADE_SIZE_FILE):
        save_trade_size(DEFAULT_TRADE_SIZE)
        return DEFAULT_TRADE_SIZE
    try:
        with open(TRADE_SIZE_FILE, "r") as f:
            size = json.load(f)
            return size.get("trade_size", DEFAULT_TRADE_SIZE)
    except:
        return DEFAULT_TRADE_SIZE

def save_trade_size(size):
    with open(TRADE_SIZE_FILE, "w") as f:
        json.dump({"trade_size": size}, f)

trade_size_usdt = load_trade_size()

###############################################################
#  KUCOIN SIGNING FUNCTION
###############################################################

def kucoin_headers(method: str, endpoint: str, body: dict = None):
    """Return headers for KuCoin authenticated request."""
    now = int(time.time() * 1000)

    if body:
        encoded_body = json.dumps(body)
    else:
        encoded_body = ""

    str_to_sign = f"{now}{method}{endpoint}{encoded_body}"
    signature = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode("utf-8"),
            str_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode()

    passphrase = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode("utf-8"),
            KUCOIN_API_PASSPHRASE.encode("utf-8"),
            hashlib.sha256,
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

###############################################################
#  KUCOIN HELPERS – GET CANDLES
###############################################################

def get_kucoin_candles(symbol: str, interval="3min", limit=150):
    """Hämtar 3-minutercandles från KuCoin Spot API."""
    url = f"{BASE_URL}/api/v1/market/candles?type={interval}&symbol={symbol}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "data" not in data:
            return []
        # KuCoin returnerar: [time, open, close, high, low, volume, turnover]
        candles = []
        for c in data["data"][:limit]:
            candles.append({
                "timestamp": int(c[0]),
                "open": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "volume": float(c[5])
            })
        candles.reverse()
        return candles
    except Exception as e:
        print("Error candles:", e)
        return []

###############################################################
#  ORB ENGINE – LOGIK
###############################################################

class ORBState:
    def __init__(self):
        self.current_orb = None
        self.in_position = False
        self.entry_price = None
        self.stop_price = None
        self.take_profit = None
        self.trailing_active = False
        self.position_side = "long"

orb_states = {coin: ORBState() for coin in ACTIVE_COINS}

def detect_new_orb(c1, c2):
    """Return True om en röd → grön sekvens är upptäckt."""
    return (c1["close"] < c1["open"]) and (c2["close"] > c2["open"])

def create_orb(candle):
    return {
        "high": candle["high"],
        "low": candle["low"],
        "timestamp": candle["timestamp"]
    }

###############################################################
#  ENTRY LOGIK
###############################################################

def should_enter(candle, orb):
    """Return True om pris bryter ORB-high."""
    return candle["high"] >= orb["high"]

def should_exit_trailing(candle, orb_state: ORBState):
    """Trailing stop logik."""
    if not orb_state.trailing_active:
        return False
    if candle["low"] <= orb_state.stop_price:
        return True
    return False

###############################################################
#  MOCK ENGINE – IDENTISK MED LIVE
###############################################################

def mock_fill_limit_entry(price, orb_state: ORBState):
    orb_state.in_position = True
    orb_state.entry_price = price
    # Stop = ORB-low
    orb_state.stop_price = orb_state.current_orb["low"]
    orb_state.trailing_active = False

def mock_fill_exit(price, orb_state: ORBState):
    orb_state.in_position = False
    orb_state.trailing_active = False
    return price

###############################################################
#  TELEGRAM KNAPP-MENY
###############################################################

def main_menu():
    keyboard = [
        [InlineKeyboardButton("▶ START MOCK", callback_data="start_mock"),
         InlineKeyboardButton("🔥 START LIVE", callback_data="start_live")],
        [InlineKeyboardButton("⛔ STOP", callback_data="stop")],
        [InlineKeyboardButton("🤖 AI-Läge", callback_data="ai_menu"),
         InlineKeyboardButton("🎯 Entry Mode", callback_data="entry_menu")],
        [InlineKeyboardButton("💰 Trade Size", callback_data="size_menu")],
        [InlineKeyboardButton("📊 Status", callback_data="status")],
    ]
    return InlineKeyboardMarkup(keyboard)

###############################################################
#  TELEGRAM KOMMANDON – START
###############################################################

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "MP ORBbot – Spot Edition\nVälj ett alternativ:",
        reply_markup=main_menu()
    )

###############################################################
#  CALLBACKS FÖR KNAPPAR
###############################################################

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode, ai_mode, entry_mode, trade_size_usdt

    query = update.callback_query
    await query.answer()
    cmd = query.data

    if cmd == "start_mock":
        bot_mode = "mock"
        bot_running = True
        await query.edit_message_text("Mock-läge aktiverat!", reply_markup=main_menu())
        return

    if cmd == "start_live":
        bot_mode = "live"
        bot_running = True
        await query.edit_message_text("Live-läge aktiverat!", reply_markup=main_menu())
        return

    if cmd == "stop":
        bot_running = False
        await query.edit_message_text("Boten stoppad.", reply_markup=main_menu())
        return

    if cmd == "status":
        await query.edit_message_text(
            f"📊 Status:\n"
            f"Mode: {bot_mode}\n"
            f"AI: {ai_mode}\n"
            f"Entry mode: {entry_mode}\n"
            f"Trade size: {trade_size_usdt} USDT\n"
            f"Coins: {', '.join(ACTIVE_COINS)}",
            reply_markup=main_menu()
        )
        return

    if cmd == "ai_menu":
        kb = [
            [InlineKeyboardButton("Aggressiv", callback_data="ai_aggr")],
            [InlineKeyboardButton("Neutral", callback_data="ai_neut")],
            [InlineKeyboardButton("Försiktig", callback_data="ai_safe")],
        ]
        await query.edit_message_text("Välj AI-läge:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("ai_"):
    if cmd == "ai_aggr": ai_mode = "aggressiv"
    if cmd == "ai_neut": ai_mode = "neutral"
    if cmd == "ai_safe": ai_mode = "forsiktig"
    await query.edit_message_text(f"AI-läge satt till: {ai_mode}", reply_markup=main_menu())
    return

    if cmd == "entry_menu":
        kb = [
            [InlineKeyboardButton("Close", callback_data="entry_close")],
            [InlineKeyboardButton("Tick", callback_data="entry_tick")],
            [InlineKeyboardButton("Retest (standard)", callback_data="entry_retest")],
        ]
        await query.edit_message_text("Välj entry-mode:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("entry_"):
        global entry_mode
        entry_mode = cmd.replace("entry_", "")
        await query.edit_message_text(f"Entry mode satt till: {entry_mode}", reply_markup=main_menu())
        return

    if cmd == "size_menu":
        kb = [
            [InlineKeyboardButton("10 USDT", callback_data="size_10"),
             InlineKeyboardButton("20 USDT", callback_data="size_20")],
            [InlineKeyboardButton("30 USDT", callback_data="size_30"),
             InlineKeyboardButton("50 USDT", callback_data="size_50")],
            [InlineKeyboardButton("100 USDT", callback_data="size_100"),
             InlineKeyboardButton("200 USDT", callback_data="size_200")],
        ]
        await query.edit_message_text("Välj trade size:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("size_"):
        global trade_size_usdt
        trade_size_usdt = int(cmd.replace("size_", ""))
        save_trade_size(trade_size_usdt)
        await query.edit_message_text(
            f"Trade size ändrad till {trade_size_usdt} USDT.",
            reply_markup=main_menu()
        )
        return
###############################################################
#  KOMMANDO: /set_size manuellt
###############################################################

async def set_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global trade_size_usdt
    try:
        num = int(context.args[0])
        trade_size_usdt = num
        save_trade_size(num)
        await update.message.reply_text(f"Trade size satt till {num} USDT.")
    except:
        await update.message.reply_text("Använd: /set_size 50")


###############################################################
#  HJÄLPMETODER FÖR KUCOIN LIVE-ORDERS
###############################################################

def kucoin_place_limit_buy(symbol, price, usdt_amount):
    """Place maker-limit BUY."""
    endpoint = "/api/v1/orders"
    qty = round(usdt_amount / price, 8)

    body = {
        "clientOid": str(time.time()),
        "side": "buy",
        "symbol": symbol,
        "type": "limit",
        "price": str(price),
        "size": str(qty),
        "timeInForce": "GTC"
    }

    headers = kucoin_headers("POST", endpoint, body)
    r = requests.post(BASE_URL + endpoint, headers=headers, json=body)
    return r.json()


def kucoin_place_limit_sell(symbol, price, coin_amount):
    """Place maker-limit SELL."""
    endpoint = "/api/v1/orders"
    body = {
        "clientOid": str(time.time()),
        "side": "sell",
        "symbol": symbol,
        "type": "limit",
        "price": str(price),
        "size": str(coin_amount),
        "timeInForce": "GTC"
    }
    headers = kucoin_headers("POST", endpoint, body)
    r = requests.post(BASE_URL + endpoint, headers=headers, json=body)
    return r.json()


def kucoin_place_stoploss(symbol, stop_price, coin_amount):
    """STOP-LIMIT stop-loss."""
    endpoint = "/api/v1/stop-order"

    # Vi lägger limit 0.2% under stop för garanterad fill
    limit_price = stop_price * 0.998

    body = {
        "clientOid": str(time.time()),
        "symbol": symbol,
        "side": "sell",
        "type": "limit",
        "price": str(round(limit_price, 2)),
        "size": str(coin_amount),
        "stop": "loss",
        "stopPrice": str(round(stop_price, 2)),
        "timeInForce": "GTC"
    }

    headers = kucoin_headers("POST", endpoint, body)
    r = requests.post(BASE_URL + endpoint, headers=headers, json=body)
    return r.json()


###############################################################
#  MOCK ENGINE – SPREAD & SLIPPAGE-SIMULERING
###############################################################

MOCK_SPREAD = 0.0002   # 0.02%
MOCK_SLIPPAGE = 0.0001 # 0.01%

def mock_adjust_entry(price):
    return price * (1 + MOCK_SPREAD/2 + MOCK_SLIPPAGE)

def mock_adjust_exit(price):
    return price * (1 - MOCK_SPREAD/2 - MOCK_SLIPPAGE)


###############################################################
#  ORB HANDEL – ENTRY & EXIT
###############################################################

def handle_entry(symbol, candle, orb_state: ORBState):
    """ENTRY-LOGIK för både mock & live."""
    global bot_mode, trade_size_usdt

    entry_price = orb_state.current_orb["high"]

    # RETEST-mode: bara ge entry om low <= orb_low + liten buffert
    if entry_mode == "retest":
        if candle["low"] > orb_state.current_orb["high"] * 0.999:
            return False

    if bot_mode == "mock":
        adj = mock_adjust_entry(entry_price)
        mock_fill_limit_entry(adj, orb_state)
        return True

    # LIVE – PLACE LIMIT ORDER
    r = kucoin_place_limit_buy(symbol, entry_price, trade_size_usdt)
    print("LIVE BUY RESPONSE:", r)
    if "orderId" in str(r):
        orb_state.in_position = True
        orb_state.entry_price = entry_price
        orb_state.stop_price = orb_state.current_orb["low"]
        return True

    return False


def handle_exit(symbol, candle, orb_state: ORBState):
    """EXIT-logik – både mock & live."""
    global bot_mode

    exit_price = candle["low"]

    if bot_mode == "mock":
        adj = mock_adjust_exit(exit_price)
        final = mock_fill_exit(adj, orb_state)
        return final

    # LIVE – PLACE LIMIT SELL
    coin_amount = round(trade_size_usdt / orb_state.entry_price, 8)
    r = kucoin_place_limit_sell(symbol, exit_price, coin_amount)
    print("LIVE SELL RESPONSE:", r)
    orb_state.in_position = False
    return exit_price


###############################################################
#  TRAILING STOP
###############################################################

def update_trailing_stop(candle, orb_state: ORBState):
    """Update trailing-stop baserat på candle-high."""
    if not orb_state.in_position:
        return

    # Aktivera trailing stop vid +0.25%
    if not orb_state.trailing_active:
        if candle["high"] >= orb_state.entry_price * 1.0025:
            orb_state.trailing_active = True
            orb_state.stop_price = candle["close"] * 0.998  # 0.2% ned
            return

    # Vid aktiv trailing – följ priset
    if orb_state.trailing_active:
        new_stop = candle["close"] * 0.998
        if new_stop > orb_state.stop_price:
            orb_state.stop_price = new_stop


###############################################################
#  AI-FILTER
###############################################################

def ai_filter_pass(candle, prev_candles, ai_level):
    """Return om breakouten ska tas baserat på AI-modell."""

    body = abs(candle["close"] - candle["open"])
    wick = candle["high"] - max(candle["close"], candle["open"])

    # DOJI-filter (försiktig & neutral)
    if ai_level in ["neutral", "forsiktig"]:
        if body < wick:  # klassisk doji/weak candle
            return False

    # Momentumfilter – aggressiv tar fler
    if ai_level == "forsiktig":
        if candle["close"] <= prev_candles[-1]["close"]:
            return False

    return True


###############################################################
#  ORB MAIN LOOP PER COIN
###############################################################

def run_orb_for_symbol(symbol):
    global bot_running

    state = orb_states[symbol]

    while True:
        if not bot_running:
            time.sleep(1)
            continue

        candles = get_kucoin_candles(symbol)
        if len(candles) < 10:
            time.sleep(3)
            continue

        last_c = candles[-1]
        prev_c = candles[-2]

        # Detektera ny ORB
        if detect_new_orb(prev_c, last_c):
            state.current_orb = create_orb(last_c)
            state.in_position = False
            state.trailing_active = False

        # Hoppas om ingen ORB aktiv
        if not state.current_orb:
            time.sleep(1)
            continue

        # ENTRY
        if (not state.in_position) and should_enter(last_c, state.current_orb):
            if ai_filter_pass(last_c, candles[-6:-1], ai_mode):
                handle_entry(symbol, last_c, state)

        # Trailing
        if state.in_position:
            update_trailing_stop(last_c, state)

        # EXIT via trailing stop
        if state.in_position and should_exit_trailing(last_c, state):
            handle_exit(symbol, last_c, state)

        time.sleep(1)


###############################################################
#  MULTI-THREAD START
###############################################################

def start_all_orb_threads():
    for symbol in ACTIVE_COINS:
        t = threading.Thread(
            target=run_orb_for_symbol,
            args=(symbol,),
            daemon=True
        )
        t.start()
###############################################################
#  TELEGRAM: /status, /start_all, /stop
###############################################################

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 Status:\n"
        f"Bot läge: {bot_mode}\n"
        f"AI-läge: {ai_mode}\n"
        f"Entry mode: {entry_mode}\n"
        f"Trade size: {trade_size_usdt} USDT\n"
        f"Aktiva coins: {', '.join(ACTIVE_COINS)}"
    )


###############################################################
#  /backtest – simplifierad ORB-backtest direkt i boten
###############################################################

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper()
    except:
        await update.message.reply_text("Användning: /backtest BTC-USDT")
        return

    candles = get_kucoin_candles(symbol, limit=500)
    if len(candles) < 10:
        await update.message.reply_text("Kunde inte hämta historik.")
        return

    wins = 0
    losses = 0

    orb = None
    in_pos = False
    entry = 0
    stop = 0

    for i in range(2, len(candles)):
        c1 = candles[i-1]
        c2 = candles[i]

        # Ny ORB
        if detect_new_orb(c1, c2):
            orb = create_orb(c2)
            in_pos = False

        if not orb:
            continue

        # ENTRY
        if (not in_pos) and (c2["high"] >= orb["high"]):
            in_pos = True
            entry = orb["high"]
            stop = orb["low"]

        # EXIT pga SL
        if in_pos and c2["low"] <= stop:
            in_pos = False
            losses += 1

        # EXIT pga quick TP (0.3%)
        if in_pos and c2["high"] >= entry * 1.003:
            in_pos = False
            wins += 1

    total = wins + losses
    if total == 0:
        await update.message.reply_text("Inga trades hittades.")
        return

    wr = round(wins / total * 100, 2)
    msg = f"Backtest resultat {symbol}:\nTrades: {total}\nWins: {wins}\nLosses: {losses}\nWinrate: {wr}%"
    await update.message.reply_text(msg)


###############################################################
#  INIT & START AV TELEGRAM-BOTEN
###############################################################

async def start_bot():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("set_size", set_size))
    app.add_handler(CommandHandler("backtest", backtest))

    print("Bot är igång. Väntar på Telegram-kommandon...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await app.updater.idle()


###############################################################
#  MAIN PROGRAM
###############################################################

if __name__ == "__main__":
    print("MP ORBbot – startar ORB-trådar...")
    start_all_orb_threads()

    import asyncio
    asyncio.run(start_bot())
