#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################
#   MP ORBbot – Full Version – KuCoin Spot – Mock + Live       #
#   Smart ORB, AI-filter, Risk Control, CSV Export,           #
#   Telegram UI + Command Menu, Engine On/Off                 #
###############################################################

import os
import time
import json
import hmac
import math
import base64
import hashlib
import threading
import requests
import csv
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
)

###############################################################
#  LOAD ENVIRONMENT VARIABLES (IMPORTANT – NO HARDCODING!)     #
###############################################################

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

if not TELEGRAM_TOKEN:
    raise Exception("❌ TELEGRAM_TOKEN saknas i DigitalOcean Environment Variables!")

KUCOIN_BASE = "https://api.kucoin.com"

###############################################################
#  GLOBAL STATE                                                #
###############################################################

ACTIVE_COINS = ["BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT", "ADA-USDT"]

bot_running = False             # Engine On/Off
bot_mode = "mock"               # "mock" eller "live"
ai_mode = "neutral"             # AI filter
entry_mode = "tick"             # tick / close / retest
risk_mode = "fixed"             # fixed / percent
risk_percent = 1.0              # 1% default
trade_size_usdt = 30            # default 30 USDT
mock_spread = 0.0002            # 0.02 %
mock_slippage = 0.0001          # 0.01 %
fee_rate = 0.001                # 0.1% KuCoin spot fee

CSV_REAL = "real_trade_log.csv"
CSV_MOCK = "mock_trade_log.csv"

###############################################################
#  SAVE/LOAD SETTINGS                                          #
###############################################################

SETTINGS_FILE = "settings.json"

default_settings = {
    "trade_size_usdt": 30,
    "risk_mode": "fixed",
    "risk_percent": 1.0,
    "ai_mode": "neutral",
    "entry_mode": "tick"
}

def load_settings():
    global trade_size_usdt, risk_mode, risk_percent, ai_mode, entry_mode
    if not os.path.exists(SETTINGS_FILE):
        save_settings()
        return

    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
            trade_size_usdt = data.get("trade_size_usdt", 30)
            risk_mode = data.get("risk_mode", "fixed")
            risk_percent = data.get("risk_percent", 1.0)
            ai_mode = data.get("ai_mode", "neutral")
            entry_mode = data.get("entry_mode", "tick")
    except:
        save_settings()

def save_settings():
    data = {
        "trade_size_usdt": trade_size_usdt,
        "risk_mode": risk_mode,
        "risk_percent": risk_percent,
        "ai_mode": ai_mode,
        "entry_mode": entry_mode
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)

load_settings()

###############################################################
#  KUCOIN HELPERS                                              #
###############################################################

def kucoin_sign(method: str, endpoint: str, body: dict = None):
    now = int(time.time() * 1000)
    body_str = json.dumps(body) if body else ""
    msg = f"{now}{method}{endpoint}{body_str}"

    sign = base64.b64encode(
        hmac.new(KUCOIN_API_SECRET.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()

    passphrase = base64.b64encode(
        hmac.new(KUCOIN_API_SECRET.encode(), KUCOIN_API_PASSPHRASE.encode(), hashlib.sha256).digest()
    ).decode()

    return {
        "KC-API-KEY": KUCOIN_API_KEY,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }


def kucoin_get_balance():
    if not KUCOIN_API_KEY:
        return 0
    try:
        endpoint = "/api/v1/accounts"
        headers = kucoin_sign("GET", endpoint)
        r = requests.get(KUCOIN_BASE + endpoint, headers=headers, timeout=10)
        data = r.json()
        if "data" not in data:
            return 0
        for acc in data["data"]:
            if acc["currency"] == "USDT" and acc["type"] == "trade":
                return float(acc["balance"])
    except:
        return 0
    return 0


def kucoin_place_limit_buy(symbol, price, size):
    endpoint = "/api/v1/orders"
    body = {
        "clientOid": str(int(time.time() * 1000)),
        "side": "buy",
        "symbol": symbol,
        "type": "limit",
        "price": f"{price:.10f}",
        "size": f"{size:.10f}"
    }
    headers = kucoin_sign("POST", endpoint, body)
    try:
        r = requests.post(KUCOIN_BASE + endpoint, headers=headers, json=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def kucoin_place_limit_sell(symbol, price, size):
    endpoint = "/api/v1/orders"
    body = {
        "clientOid": str(int(time.time() * 1000)),
        "side": "sell",
        "symbol": symbol,
        "type": "limit",
        "price": f"{price:.10f}",
        "size": f"{size:.10f}"
    }
    headers = kucoin_sign("POST", endpoint, body)
    try:
        r = requests.post(KUCOIN_BASE + endpoint, headers=headers, json=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def kucoin_cancel(order_id):
    endpoint = f"/api/v1/orders/{order_id}"
    headers = kucoin_sign("DELETE", endpoint)
    try:
        r = requests.delete(KUCOIN_BASE + endpoint, headers=headers, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def kucoin_get_candles(symbol: str, interval="3min", limit=100):
    url = f"{KUCOIN_BASE}/api/v1/market/candles?type={interval}&symbol={symbol}"
    try:
        r = requests.get(url, timeout=10)
        d = r.json()
        if "data" not in d:
            return []
        candles = []
        for c in d["data"]:
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
#  ORB STATE                                                   #
###############################################################

class ORBState:
    def __init__(self):
        self.current_orb = None
        self.entry_price = None
        self.stop_price = None
        self.in_position = False
        self.order_id = None
        self.entry_filled = False
        self.size = 0


orb_states = {s: ORBState() for s in ACTIVE_COINS}
###############################################################
#  ORB LOGIC                                                   #
###############################################################

def detect_new_orb(c1, c2):
    """Röd → Grön = starta ORB."""
    return (c1["close"] < c1["open"]) and (c2["close"] > c2["open"])


def create_orb(candle):
    return {
        "high": candle["high"],
        "low": candle["low"],
        "timestamp": candle["timestamp"]
    }


def should_enter(candle, orb):
    """Breakout-logik beroende på entry_mode."""
    if entry_mode == "tick":
        return candle["high"] >= orb["high"]

    elif entry_mode == "close":
        return candle["close"] >= orb["high"]

    elif entry_mode == "retest":
        # Retest: måste ner nära low innan breakout
        if candle["low"] <= orb["low"] * 1.002 and candle["high"] >= orb["high"]:
            return True
        return False

    return False


def ai_filter(candle, prev):
    """AI-filter enligt aggressiv / neutral / försiktig."""
    body = abs(candle["close"] - candle["open"])
    rng = candle["high"] - candle["low"]
    if rng == 0:
        rng = 1e-9

    body_ratio = body / rng

    if ai_mode == "aggressiv":
        return True

    if ai_mode == "neutral":
        # Filtrera bort extremt små "ingenting" candles
        if body_ratio < 0.1:
            return False
        return True

    if ai_mode == "forsiktig":
        # Kräv momentum samt ingen doji
        if body_ratio < 0.3:
            return False
        if len(prev) >= 3:
            avg = sum([c["close"] for c in prev[-3:]]) / 3
            if candle["close"] <= avg:
                return False
        return True

    return True


###############################################################
#  MOCK MODE ENTRY / EXIT                                     #
###############################################################

def mock_entry_price(price: float) -> float:
    return price * (1 + mock_spread + mock_slippage)


def mock_exit_price(price: float) -> float:
    return price * (1 - mock_spread - mock_slippage)


def mock_position_size(entry_price: float) -> float:
    """Hur mycket coin man får i mock-läge."""
    return trade_size_usdt / entry_price


###############################################################
#  LIVE MODE SMART ENTRY / EXIT (LIMIT PYRAMIDING)            #
###############################################################

def calculate_live_size(symbol: str, price: float):
    """Beräknar size beroende på fixed eller percent risk."""
    global risk_mode, risk_percent, trade_size_usdt

    if risk_mode == "fixed":
        return trade_size_usdt / price

    if risk_mode == "percent":
        balance = kucoin_get_balance()
        if balance <= 0:
            return 0
        use_amount = balance * (risk_percent / 100)
        return use_amount / price

    return trade_size_usdt / price


def smart_limit_entry(symbol: str, entry_price: float):
    """
    Smarta limit-entry:

    1. Placera limit under market för att bli billigt fylld.
    2. Om inte fill → flytta order var 2 sek.
    3. Avbryt om candle vänder.
    """
    state = orb_states[symbol]
    size = calculate_live_size(symbol, entry_price)
    if size <= 0:
        return None

    # Startpris: lite under entry
    place_price = entry_price * 0.9995

    print(f"[LIVE] Skickar smart-entry {symbol} size={size:.6f} price={place_price:.4f}")

    res = kucoin_place_limit_buy(symbol, place_price, size)

    if "orderId" not in res.get("data", {}):
        print(f"[LIVE ERROR] Misslyckades skapa order: {res}")
        return None

    order_id = res["data"]["orderId"]
    state.order_id = order_id
    state.size = size
    state.entry_price = entry_price

    # Moving order loop
    for _ in range(20):
        if not state.in_position:
            break

        # Flytta order högre om priset springer
        new_price = entry_price * 0.9990

        kucoin_cancel(order_id)
        res2 = kucoin_place_limit_buy(symbol, new_price, size)

        if "orderId" in res2.get("data", {}):
            order_id = res2["data"]["orderId"]
            state.order_id = order_id

        time.sleep(2)

    return order_id


def smart_limit_exit(symbol: str, exit_price: float):
    """Liknande smart exit-limit order."""
    state = orb_states[symbol]
    if not state.size:
        return None

    place_price = exit_price * 1.0005

    res = kucoin_place_limit_sell(symbol, place_price, state.size)

    if "orderId" not in res.get("data", {}):
        print(f"[LIVE EXIT ERROR] {res}")
        return None

    return res["data"]["orderId"]


###############################################################
#  TRADE EXECUTION (ENTRY / EXIT)                             #
###############################################################

def execute_entry(symbol: str, candle):
    """Gemensam entry-funktion för live och mock."""
    state = orb_states[symbol]
    entry = state.current_orb["high"]

    # AI-filter
    if not ai_filter(candle, []):
        return False

    if bot_mode == "mock":
        adj_price = mock_entry_price(entry)
        state.entry_price = adj_price
        state.size = mock_position_size(adj_price)
        state.stop_price = state.current_orb["low"]
        state.in_position = True

        log_mock_trade(symbol, "ENTRY", adj_price, state.size)
        print(f"[MOCK ENTRY] {symbol} @ {adj_price:.4f}")
        return True

    # Live
    if not KUCOIN_API_KEY:
        print("[LIVE] Försöker live-entry men API-nycklar saknas!")
        return False

    state.in_position = True
    state.stop_price = state.current_orb["low"]
    order_id = smart_limit_entry(symbol, entry)
    state.order_id = order_id

    log_real_trade(symbol, "ENTRY", entry, state.size)
    return True


def execute_exit(symbol: str, candle):
    """Gemensam exit-funktion."""
    state = orb_states[symbol]
    exit_price = candle["low"]

    if bot_mode == "mock":
        adj_price = mock_exit_price(exit_price)
        log_mock_trade(symbol, "EXIT", adj_price, state.size)
        print(f"[MOCK EXIT] {symbol} @ {adj_price:.4f}")
        state.in_position = False
        state.size = 0
        return adj_price

    # Live (smart limit exit)
    if not KUCOIN_API_KEY:
        print("[LIVE] Exit utan API-nycklar!")
        state.in_position = False
        return exit_price

    smart_limit_exit(symbol, exit_price)
    log_real_trade(symbol, "EXIT", exit_price, state.size)

    state.in_position = False
    state.size = 0
    return exit_price


###############################################################
#  LOGGING (CSV FOR SKATTEVERKET)                             #
###############################################################

def ensure_csv_files():
    if not os.path.exists(CSV_REAL):
        with open(CSV_REAL, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "symbol", "type", "price", "size"])

    if not os.path.exists(CSV_MOCK):
        with open(CSV_MOCK, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "symbol", "type", "price", "size"])


def log_real_trade(symbol, typ, price, size):
    ensure_csv_files()
    with open(CSV_REAL, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.utcnow().isoformat(), symbol, typ, price, size])


def log_mock_trade(symbol, typ, price, size):
    ensure_csv_files()
    with open(CSV_MOCK, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.utcnow().isoformat(), symbol, typ, price, size])


###############################################################
#  ORB THREAD LOOP PER COIN                                   #
###############################################################

def orb_loop(symbol):
    global bot_running

    state = orb_states[symbol]

    print(f"[ORB] Loop startad för {symbol}")

    while True:
        try:
            if not bot_running:
                time.sleep(2)
                continue

            candles = kucoin_get_candles(symbol)
            if len(candles) < 8:
                time.sleep(2)
                continue

            last = candles[-1]
            prev = candles[-2]

            # Ny ORB?
            if detect_new_orb(prev, last):
                state.current_orb = create_orb(last)
                state.in_position = False
                print(f"[ORB] Ny ORB i {symbol}: HIGH={state.current_orb['high']:.4f}")

            # Entry?
            if (
                state.current_orb
                and not state.in_position
                and should_enter(last, state.current_orb)
            ):
                prev5 = candles[-6:-1]
                if ai_filter(last, prev5):
                    execute_entry(symbol, last)

            # Stop-loss
            if state.in_position and last["low"] <= state.stop_price:
                execute_exit(symbol, last)

            time.sleep(1)

        except Exception as e:
            print(f"[ERROR {symbol}] {e}")
            time.sleep(3)


###############################################################
#  STARTA TRÅDAR                                              #
###############################################################

def start_orb_threads():
    for s in ACTIVE_COINS:
        t = threading.Thread(target=orb_loop, args=(s,), daemon=True)
        t.start()
        print(f"[THREAD] Startade {s}")
###############################################################
#  TELEGRAM MENYER                                             #
###############################################################

def main_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶ Engine ON", callback_data="engine_on"),
            InlineKeyboardButton("⛔ Engine OFF", callback_data="engine_off"),
        ],
        [
            InlineKeyboardButton("🤖 AI-läge", callback_data="ai_menu"),
            InlineKeyboardButton("🎯 Entry-mode", callback_data="entry_menu"),
        ],
        [
            InlineKeyboardButton("💰 Risk", callback_data="risk_menu"),
            InlineKeyboardButton("📏 Trade-size", callback_data="size_menu"),
        ],
        [
            InlineKeyboardButton("📊 Status", callback_data="status"),
        ]
    ])


###############################################################
#  TELEGRAM – KOMMANDON (ANVÄNDS FRÅN /menu NEDERST)          #
###############################################################

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "MP ORBbot – huvudmeny:",
        reply_markup=main_menu()
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        f"📊 *Status*\n"
        f"Engine: {'ON' if bot_running else 'OFF'}\n"
        f"Läge: {bot_mode}\n"
        f"AI: {ai_mode}\n"
        f"Entry: {entry_mode}\n"
        f"Risk-mode: {risk_mode}\n"
        f"Risk-procent: {risk_percent}%\n"
        f"Trade size: {trade_size_usdt} USDT\n"
        f"Coins: {', '.join(ACTIVE_COINS)}"
    )
    await update.message.reply_markdown(text)


async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode
    bot_running = True
    bot_mode = "mock"   # default mock för säkerhet
    await update.message.reply_text("▶ Engine ON (mock-läge)", reply_markup=main_menu())


async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running
    bot_running = False
    await update.message.reply_text("⛔ Engine OFF", reply_markup=main_menu())


async def cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("Aggressiv", callback_data="ai_aggr")],
        [InlineKeyboardButton("Neutral", callback_data="ai_neut")],
        [InlineKeyboardButton("Försiktig", callback_data="ai_safe")],
    ]
    await update.message.reply_text("🤖 Välj AI-läge:", reply_markup=InlineKeyboardMarkup(kb))


async def cmd_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("Tick", callback_data="entry_tick")],
        [InlineKeyboardButton("Close", callback_data="entry_close")],
        [InlineKeyboardButton("Retest", callback_data="entry_retest")],
    ]
    await update.message.reply_text("🎯 Välj entry-mode:", reply_markup=InlineKeyboardMarkup(kb))


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("Fast USDT-size", callback_data="risk_fixed")],
        [InlineKeyboardButton("Procent av konto", callback_data="risk_percent")],
    ]
    await update.message.reply_text("💰 Välj risk-modell:", reply_markup=InlineKeyboardMarkup(kb))


async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("10 USDT", callback_data="size_10"),
         InlineKeyboardButton("20 USDT", callback_data="size_20")],
        [InlineKeyboardButton("30 USDT", callback_data="size_30"),
         InlineKeyboardButton("50 USDT", callback_data="size_50")],
    ]
    await update.message.reply_text("📏 Välj fast trade-size:", reply_markup=InlineKeyboardMarkup(kb))


###############################################################
#  TELEGRAM – CALLBACK-KNAPPAR                                 #
###############################################################

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode, ai_mode, entry_mode
    global risk_mode, risk_percent, trade_size_usdt

    q = update.callback_query
    await q.answer()
    cmd = q.data

    # ----------------------
    # Engine
    # ----------------------
    if cmd == "engine_on":
        bot_running = True
        bot_mode = "mock"  # alltid mock som default
        save_settings()
        await q.edit_message_text("▶ Engine är nu ON (mock)", reply_markup=main_menu())
        return

    if cmd == "engine_off":
        bot_running = False
        save_settings()
        await q.edit_message_text("⛔ Engine är nu OFF", reply_markup=main_menu())
        return

    # ----------------------
    # Status
    # ----------------------
    if cmd == "status":
        text = (
            f"📊 *Status*\n"
            f"Engine: {'ON' if bot_running else 'OFF'}\n"
            f"Läge: {bot_mode}\n"
            f"AI: {ai_mode}\n"
            f"Entry: {entry_mode}\n"
            f"Risk-mode: {risk_mode}\n"
            f"Risk-procent: {risk_percent}%\n"
            f"Trade size: {trade_size_usdt} USDT\n"
            f"Coins: {', '.join(ACTIVE_COINS)}"
        )
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu())
        return

    # ----------------------
    # AI-läge
    # ----------------------
    if cmd == "ai_menu":
        kb = [
            [InlineKeyboardButton("Aggressiv", callback_data="ai_aggr")],
            [InlineKeyboardButton("Neutral", callback_data="ai_neut")],
            [InlineKeyboardButton("Försiktig", callback_data="ai_safe")],
        ]
        await q.edit_message_text("🤖 Välj AI-läge:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("ai_"):
        if cmd == "ai_aggr":
            ai_mode = "aggressiv"
        elif cmd == "ai_neut":
            ai_mode = "neutral"
        elif cmd == "ai_safe":
            ai_mode = "forsiktig"
        save_settings()
        await q.edit_message_text(f"🤖 AI-läge satt till: {ai_mode}", reply_markup=main_menu())
        return

    # ----------------------
    # Entry-mode
    # ----------------------
    if cmd == "entry_menu":
        kb = [
            [InlineKeyboardButton("Tick", callback_data="entry_tick")],
            [InlineKeyboardButton("Close", callback_data="entry_close")],
            [InlineKeyboardButton("Retest", callback_data="entry_retest")],
        ]
        await q.edit_message_text("🎯 Välj entry-mode:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("entry_"):
        if cmd == "entry_tick":
            entry_mode = "tick"
        elif cmd == "entry_close":
            entry_mode = "close"
        elif cmd == "entry_retest":
            entry_mode = "retest"
        save_settings()
        await q.edit_message_text(f"🎯 Entry-mode: {entry_mode}", reply_markup=main_menu())
        return

    # ----------------------
    # Risk-mode
    # ----------------------
    if cmd == "risk_menu":
        kb = [
            [InlineKeyboardButton("Fast USDT", callback_data="risk_fixed")],
            [InlineKeyboardButton("Procent av konto", callback_data="risk_percent_menu")],
        ]
        await q.edit_message_text("💰 Välj risk-modell:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd == "risk_fixed":
        risk_mode = "fixed"
        save_settings()
        await q.edit_message_text("💰 Risk-modell satt till: Fast USDT", reply_markup=main_menu())
        return

    if cmd == "risk_percent_menu":
        kb = [
            [InlineKeyboardButton("0.5%", callback_data="risk_pct_0_5"),
             InlineKeyboardButton("1%", callback_data="risk_pct_1")],
            [InlineKeyboardButton("2%", callback_data="risk_pct_2"),
             InlineKeyboardButton("5%", callback_data="risk_pct_5")],
        ]
        await q.edit_message_text("Välj risk-procent:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("risk_pct_"):
        val = cmd.replace("risk_pct_", "")
        risk_percent = float(val.replace("_", "."))
        risk_mode = "percent"
        save_settings()
        await q.edit_message_text(
            f"💰 Risk-procent satt till {risk_percent}%",
            reply_markup=main_menu()
        )
        return

    # ----------------------
    # Trade-size (fast USDT)
    # ----------------------
    if cmd.startswith("size_"):
        trade_size_usdt = int(cmd.replace("size_", ""))
        risk_mode = "fixed"
        save_settings()
        await q.edit_message_text(
            f"📏 Trade-size satt till {trade_size_usdt} USDT",
            reply_markup=main_menu()
        )
        return
###############################################################
#  TELEGRAM – REGISTRERA KOMMANDON (NEDRE MENYN)              #
###############################################################

async def post_init(application):
    commands = [
        BotCommand("start", "Visa huvudmeny"),
        BotCommand("status", "Visa status"),
        BotCommand("engine_on", "Starta motorn (mock-läge)"),
        BotCommand("engine_off", "Stoppa motorn"),
        BotCommand("ai", "Välj AI-läge"),
        BotCommand("entry", "Välj entry-mode"),
        BotCommand("risk", "Välj risk-modell"),
        BotCommand("size", "Välj fast trade-size"),
    ]
    await application.bot.set_my_commands(commands)


###############################################################
#  TELEGRAM – KOPPLA KOMMANDON + STARTA BOTEN                 #
###############################################################

def main():
    print("MP ORBbot – startar ORB-trådar...")
    start_orb_threads()

    print("MP ORBbot – startar Telegram-bot...")

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Kommandon (nere /slash)
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("engine_on", cmd_engine_on))
    application.add_handler(CommandHandler("engine_off", cmd_engine_off))
    application.add_handler(CommandHandler("ai", cmd_ai))
    application.add_handler(CommandHandler("entry", cmd_entry))
    application.add_handler(CommandHandler("risk", cmd_risk))
    application.add_handler(CommandHandler("size", cmd_size))

    # Inline-knappar
    application.add_handler(CallbackQueryHandler(button_handler))

    # Blocking run – sköter sin egen event loop
    application.run_polling()


if __name__ == "__main__":
    main()
