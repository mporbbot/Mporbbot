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
from collections import deque
from datetime import datetime

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

###############################################################
#  KONFIGURATION
###############################################################

# Telegram
TELEGRAM_TOKEN = "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"

# KuCoin Spot (fyll i när du vill köra live)
KUCOIN_API_KEY = ""
KUCOIN_API_SECRET = ""
KUCOIN_API_PASSPHRASE = ""
BASE_URL = "https://api.kucoin.com"

# Aktiva coins (KuCoin-notation)
ACTIVE_COINS = ["BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT", "ADA-USDT"]

# Trade size
TRADE_SIZE_FILE = "trade_size.json"
DEFAULT_TRADE_SIZE = 30

# Bot-lägen
bot_running = False          # True = kör ORB-trådar
bot_mode = "mock"            # "mock" eller "live"
ai_mode = "neutral"          # "aggressiv", "neutral", "forsiktig"
entry_mode = "retest"        # "close", "tick", "retest"

# Mock-simulering
MOCK_SPREAD = 0.0002   # 0.02 %
MOCK_SLIPPAGE = 0.0001 # 0.01 %


###############################################################
#  TRADE SIZE – LADDA & SPARA
###############################################################

def load_trade_size() -> int:
    if not os.path.exists(TRADE_SIZE_FILE):
        save_trade_size(DEFAULT_TRADE_SIZE)
        return DEFAULT_TRADE_SIZE
    try:
        with open(TRADE_SIZE_FILE, "r") as f:
            data = json.load(f)
            return int(data.get("trade_size", DEFAULT_TRADE_SIZE))
    except Exception:
        return DEFAULT_TRADE_SIZE


def save_trade_size(size: int) -> None:
    with open(TRADE_SIZE_FILE, "w") as f:
        json.dump({"trade_size": int(size)}, f)


trade_size_usdt = load_trade_size()


###############################################################
#  KUCOIN SIGN & API-HJÄLPARE
###############################################################

def kucoin_headers(method: str, endpoint: str, body: dict | None = None):
    """Skapar signatur-header för KuCoin Spot API."""
    now = int(time.time() * 1000)
    encoded_body = json.dumps(body) if body else ""
    str_to_sign = f"{now}{method}{endpoint}{encoded_body}"

    signature = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode("utf-8"),
            str_to_sign.encode("utf-8"),
            hashlib.sha256
        ).digest()
    ).decode()

    passphrase = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode("utf-8"),
            KUCOIN_API_PASSPHRASE.encode("utf-8"),
            hashlib.sha256
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


def get_kucoin_candles(symbol: str, interval: str = "3min", limit: int = 150):
    """
    Hämtar candles från KuCoin Spot.
    Returnerar lista av dicts: {timestamp, open, high, low, close, volume}
    """
    url = f"{BASE_URL}/api/v1/market/candles?type={interval}&symbol={symbol}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "data" not in data:
            return []
        # c = [time, open, close, high, low, volume, turnover]
        candles = []
        for c in data["data"][:limit]:
            candles.append({
                "timestamp": int(c[0]),
                "open": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "volume": float(c[5]),
            })
        candles.reverse()
        return candles
    except Exception as e:
        print("Error get_kucoin_candles:", e)
        return []


def kucoin_place_limit_buy(symbol: str, price: float, usdt_amount: float):
    """Skicka limit BUY till KuCoin. Om API-nycklar saknas – mocka."""
    if not KUCOIN_API_KEY or not KUCOIN_API_SECRET or not KUCOIN_API_PASSPHRASE:
        print("[LIVE] Skulle lagt BUY-order, men API-nycklar saknas. Mockar.")
        return {"mock": True}

    endpoint = "/api/v1/orders"
    qty = round(usdt_amount / price, 8)
    body = {
        "clientOid": str(time.time()),
        "side": "buy",
        "symbol": symbol,
        "type": "limit",
        "price": str(price),
        "size": str(qty),
        "timeInForce": "GTC",
    }
    headers = kucoin_headers("POST", endpoint, body)
    r = requests.post(BASE_URL + endpoint, headers=headers, json=body, timeout=10)
    return r.json()


def kucoin_place_limit_sell(symbol: str, price: float, coin_amount: float):
    """Skicka limit SELL till KuCoin. Mockar om nycklar saknas."""
    if not KUCOIN_API_KEY or not KUCOIN_API_SECRET or not KUCOIN_API_PASSPHRASE:
        print("[LIVE] Skulle lagt SELL-order, men API-nycklar saknas. Mockar.")
        return {"mock": True}

    endpoint = "/api/v1/orders"
    body = {
        "clientOid": str(time.time()),
        "side": "sell",
        "symbol": symbol,
        "type": "limit",
        "price": str(price),
        "size": str(coin_amount),
        "timeInForce": "GTC",
    }
    headers = kucoin_headers("POST", endpoint, body)
    r = requests.post(BASE_URL + endpoint, headers=headers, json=body, timeout=10)
    return r.json()


###############################################################
#  ORB – STATE & LOGIK
###############################################################

class ORBState:
    def __init__(self):
        self.current_orb: dict | None = None
        self.in_position: bool = False
        self.entry_price: float | None = None
        self.stop_price: float | None = None
        self.trailing_active: bool = False
        self.position_side: str = "long"


orb_states: dict[str, ORBState] = {s: ORBState() for s in ACTIVE_COINS}


def detect_new_orb(c1: dict, c2: dict) -> bool:
    """Ny ORB: röd candle följt av grön candle."""
    return (c1["close"] < c1["open"]) and (c2["close"] > c2["open"])


def create_orb(candle: dict) -> dict:
    return {
        "high": candle["high"],
        "low": candle["low"],
        "timestamp": candle["timestamp"],
    }


def should_enter(candle: dict, orb: dict) -> bool:
    """Pris bryter ORB-high."""
    return candle["high"] >= orb["high"]


def ai_filter_pass(candle: dict, prev_candles: list[dict], ai_level: str) -> bool:
    """Enkel AI-filterlogik."""
    body = abs(candle["close"] - candle["open"])
    wick = candle["high"] - max(candle["close"], candle["open"])

    # DOJI-filter
    if ai_level in ["neutral", "forsiktig"]:
        if body < wick:
            return False

    # Momentumfilter (försiktig kräver högre close än föregående)
    if ai_level == "forsiktig":
        if len(prev_candles) > 0 and candle["close"] <= prev_candles[-1]["close"]:
            return False

    return True


def mock_adjust_entry(price: float) -> float:
    return price * (1 + MOCK_SPREAD / 2 + MOCK_SLIPPAGE)


def mock_adjust_exit(price: float) -> float:
    return price * (1 - MOCK_SPREAD / 2 - MOCK_SLIPPAGE)


def update_trailing_stop(candle: dict, state: ORBState) -> None:
    """Trailing stop baserat på candle-close."""
    if not state.in_position:
        return

    # Aktivera trailing vid +0.25 %
    if not state.trailing_active:
        if candle["high"] >= state.entry_price * 1.0025:
            state.trailing_active = True
            state.stop_price = candle["close"] * 0.998
            return

    if state.trailing_active:
        new_stop = candle["close"] * 0.998
        if new_stop > state.stop_price:
            state.stop_price = new_stop


def should_exit_trailing(candle: dict, state: ORBState) -> bool:
    if not state.trailing_active:
        return False
    if candle["low"] <= state.stop_price:
        return True
    return False


def handle_entry(symbol: str, candle: dict, state: ORBState):
    """ENTRY – funkar för både mock & live."""
    global bot_mode, trade_size_usdt, entry_mode

    entry_price = state.current_orb["high"]

    # RETEST: kräver att priset varit nere nära ORB-låga efter break
    if entry_mode == "retest":
        if candle["low"] > state.current_orb["low"] * 1.001:
            return False

    if bot_mode == "mock":
        adj = mock_adjust_entry(entry_price)
        state.in_position = True
        state.entry_price = adj
        state.stop_price = state.current_orb["low"]
        state.trailing_active = False
        print(f"[MOCK] ENTRY {symbol} @ {adj:.4f}")
        return True

    # LIVE
    r = kucoin_place_limit_buy(symbol, entry_price, trade_size_usdt)
    print("[LIVE BUY RESPONSE]", r)
    state.in_position = True
    state.entry_price = entry_price
    state.stop_price = state.current_orb["low"]
    state.trailing_active = False
    return True


def handle_exit(symbol: str, candle: dict, state: ORBState):
    """EXIT – både mock & live."""
    global bot_mode, trade_size_usdt

    exit_price = candle["low"]

    if bot_mode == "mock":
        adj = mock_adjust_exit(exit_price)
        print(f"[MOCK] EXIT {symbol} @ {adj:.4f}")
        state.in_position = False
        state.trailing_active = False
        return adj

    coin_amount = round(trade_size_usdt / state.entry_price, 8)
    r = kucoin_place_limit_sell(symbol, exit_price, coin_amount)
    print("[LIVE SELL RESPONSE]", r)
    state.in_position = False
    state.trailing_active = False
    return exit_price


###############################################################
#  ORB-LOOP PER SYMBOL (TRÅD)
###############################################################

def run_orb_for_symbol(symbol: str):
    global bot_running, ai_mode

    state = orb_states[symbol]
    print(f"[ORB] Startar loop för {symbol}")

    while True:
        try:
            if not bot_running:
                time.sleep(2)
                continue

            candles = get_kucoin_candles(symbol)
            if len(candles) < 10:
                time.sleep(3)
                continue

            last_c = candles[-1]
            prev_c = candles[-2]

            # Ny ORB
            if detect_new_orb(prev_c, last_c):
                state.current_orb = create_orb(last_c)
                state.in_position = False
                state.trailing_active = False
                print(f"[ORB] Ny ORB {symbol} H={state.current_orb['high']} L={state.current_orb['low']}")

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

            # EXIT
            if state.in_position and should_exit_trailing(last_c, state):
                handle_exit(symbol, last_c, state)

            time.sleep(1)

        except Exception as e:
            print(f"[ORB ERROR] {symbol}:", e)
            time.sleep(5)


def start_all_orb_threads():
    for symbol in ACTIVE_COINS:
        t = threading.Thread(target=run_orb_for_symbol, args=(symbol,), daemon=True)
        t.start()


###############################################################
#  TELEGRAM – MENY & HANDLERS
###############################################################

def main_menu():
    keyboard = [
        [
            InlineKeyboardButton("▶ START MOCK", callback_data="start_mock"),
            InlineKeyboardButton("🔥 START LIVE", callback_data="start_live"),
        ],
        [InlineKeyboardButton("⛔ STOP", callback_data="stop")],
        [
            InlineKeyboardButton("🤖 AI-läge", callback_data="ai_menu"),
            InlineKeyboardButton("🎯 Entry-mode", callback_data="entry_menu"),
        ],
        [InlineKeyboardButton("💰 Trade size", callback_data="size_menu")],
        [InlineKeyboardButton("📊 Status", callback_data="status")],
    ]
    return InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "MP ORBbot – KuCoin Spot\nVälj ett alternativ:",
        reply_markup=main_menu(),
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_mode, ai_mode, entry_mode, trade_size_usdt
    await update.message.reply_text(
        f"📊 Status:\n"
        f"Läge: {bot_mode}\n"
        f"AI-läge: {ai_mode}\n"
        f"Entry-mode: {entry_mode}\n"
        f"Trade size: {trade_size_usdt} USDT\n"
        f"Coins: {', '.join(ACTIVE_COINS)}"
    )


async def set_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global trade_size_usdt
    try:
        value = int(context.args[0])
        trade_size_usdt = value
        save_trade_size(value)
        await update.message.reply_text(f"Trade size satt till {value} USDT.")
    except Exception:
        await update.message.reply_text("Använd: /set_size 50")


async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper()
    except Exception:
        await update.message.reply_text("Användning: /backtest BTC-USDT")
        return

    candles = get_kucoin_candles(symbol, limit=500)
    if len(candles) < 10:
        await update.message.reply_text("Kunde inte hämta historik.")
        return

    wins, losses = 0, 0
    orb = None
    in_pos = False
    entry = 0.0
    stop = 0.0

    for i in range(2, len(candles)):
        c1 = candles[i - 1]
        c2 = candles[i]

        if detect_new_orb(c1, c2):
            orb = create_orb(c2)
            in_pos = False

        if not orb:
            continue

        if (not in_pos) and (c2["high"] >= orb["high"]):
            in_pos = True
            entry = orb["high"]
            stop = orb["low"]

        if in_pos and c2["low"] <= stop:
            in_pos = False
            losses += 1

        if in_pos and c2["high"] >= entry * 1.003:
            in_pos = False
            wins += 1

    total = wins + losses
    if total == 0:
        await update.message.reply_text("Inga trades hittades.")
        return

    wr = round(wins / total * 100, 2)
    await update.message.reply_text(
        f"Backtest {symbol} (enkel ORB):\n"
        f"Trades: {total}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Winrate: {wr}%"
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode, ai_mode, entry_mode, trade_size_usdt

    query = update.callback_query
    await query.answer()
    cmd = query.data

    # Start / stop
    if cmd == "start_mock":
        bot_mode = "mock"
        bot_running = True
        await query.edit_message_text("Mock-läge aktiverat ✅", reply_markup=main_menu())
        return

    if cmd == "start_live":
        bot_mode = "live"
        bot_running = True
        await query.edit_message_text("Live-läge aktiverat ⚠️ (API-nycklar krävs)", reply_markup=main_menu())
        return

    if cmd == "stop":
        bot_running = False
        await query.edit_message_text("Boten stoppad ⛔", reply_markup=main_menu())
        return

    if cmd == "status":
        await query.edit_message_text(
            f"📊 Status:\n"
            f"Läge: {bot_mode}\n"
            f"AI-läge: {ai_mode}\n"
            f"Entry-mode: {entry_mode}\n"
            f"Trade size: {trade_size_usdt} USDT\n"
            f"Coins: {', '.join(ACTIVE_COINS)}",
            reply_markup=main_menu(),
        )
        return

    # AI-meny
    if cmd == "ai_menu":
        kb = [
            [InlineKeyboardButton("Aggressiv", callback_data="ai_aggr")],
            [InlineKeyboardButton("Neutral", callback_data="ai_neut")],
            [InlineKeyboardButton("Försiktig", callback_data="ai_safe")],
        ]
        await query.edit_message_text("Välj AI-läge:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("ai_"):
        if cmd == "ai_aggr":
            ai_mode = "aggressiv"
        elif cmd == "ai_neut":
            ai_mode = "neutral"
        elif cmd == "ai_safe":
            ai_mode = "forsiktig"

        await query.edit_message_text(
            f"AI-läge satt till: {ai_mode}",
            reply_markup=main_menu(),
        )
        return

    # Entry-mode meny
    if cmd == "entry_menu":
        kb = [
            [InlineKeyboardButton("Close", callback_data="entry_close")],
            [InlineKeyboardButton("Tick", callback_data="entry_tick")],
            [InlineKeyboardButton("Retest (standard)", callback_data="entry_retest")],
        ]
        await query.edit_message_text("Välj entry-mode:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("entry_"):
        if cmd == "entry_close":
            entry_mode = "close"
        elif cmd == "entry_tick":
            entry_mode = "tick"
        elif cmd == "entry_retest":
            entry_mode = "retest"

        await query.edit_message_text(
            f"Entry-mode satt till: {entry_mode}",
            reply_markup=main_menu(),
        )
        return

    # Trade size meny
    if cmd == "size_menu":
        kb = [
            [
                InlineKeyboardButton("10 USDT", callback_data="size_10"),
                InlineKeyboardButton("20 USDT", callback_data="size_20"),
            ],
            [
                InlineKeyboardButton("30 USDT", callback_data="size_30"),
                InlineKeyboardButton("50 USDT", callback_data="size_50"),
            ],
            [
                InlineKeyboardButton("100 USDT", callback_data="size_100"),
                InlineKeyboardButton("200 USDT", callback_data="size_200"),
            ],
        ]
        await query.edit_message_text("Välj trade size:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("size_"):
        value = int(cmd.replace("size_", ""))
        trade_size_usdt = value
        save_trade_size(value)
        await query.edit_message_text(
            f"Trade size ändrad till {value} USDT.",
            reply_markup=main_menu(),
        )
        return


###############################################################
#  START TELEGRAM & ORB
###############################################################

def main():
    print("MP ORBbot – startar ORB-trådar...")
    start_all_orb_threads()

    print("Startar Telegram-bot...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("set_size", set_size))
    app.add_handler(CommandHandler("backtest", backtest))
    app.add_handler(CallbackQueryHandler(button_handler))

    # Blockerar och kör polling
    app.run_polling()


if __name__ == "__main__":
    main()
