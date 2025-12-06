#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################
#   MP ORBbot – KuCoin Spot – Mock + Live – Telegram UI       #
###############################################################

import os
import time
import json
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
    BotCommand,
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

# Global state
bot_running = False          # om motorn är på
bot_mode = "mock"            # "mock" eller "live"
ai_mode = "neutral"          # "aggressiv" / "neutral" / "forsiktig"
entry_mode = "tick"          # "tick" / "close" / "retest"

# Mock-marknad
MOCK_SPREAD = 0.0002         # 0.02 %
MOCK_SLIPPAGE = 0.0001       # 0.01 %

# Trade size (USDT)
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
#  KUCOIN HJÄLPMETODER (just nu bara för candles)
###############################################################

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
    except Exception as e:
        print(f"[KUCOIN ERROR] {symbol} {e}")
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
    """
    Ny ORB varje gång vi får röd -> grön.
    """
    return (c1["close"] < c1["open"]) and (c2["close"] > c2["open"])


def create_orb(candle):
    return {
        "high": candle["high"],
        "low": candle["low"],
        "timestamp": candle["timestamp"],
    }


def should_enter(candle, orb):
    """
    Kollar själva break över ORB-high beroende på entry_mode.
    """
    if entry_mode == "tick":
        # bara att högsta touchar/bryter high
        return candle["high"] >= orb["high"]
    elif entry_mode == "close":
        # close måste vara över high
        return candle["close"] >= orb["high"]
    elif entry_mode == "retest":
        # retest-läge: priset ska ha varit nere nära low innan
        if candle["high"] >= orb["high"] and candle["low"] <= orb["low"] * 1.002:
            return True
        return False
    return False


def ai_filter_pass(candle, previous_candles, mode):
    """
    Lite snällare filter nu för mer trejds:
      - Aggressiv: alltid OK
      - Neutral: filtrera bara bort super-dojis
      - Försiktig: doji-filter + kräver momentum uppåt
    """
    body = abs(candle["close"] - candle["open"])
    full_range = candle["high"] - candle["low"]
    if full_range == 0:
        full_range = 1e-9

    body_ratio = body / full_range  # hur stor del av candle som är kropp

    if mode == "aggressiv":
        return True

    if mode == "neutral":
        # släng bara bort extremt små "ingenting"-candles
        if body_ratio < 0.1:
            return False
        return True

    if mode == "forsiktig":
        # doji-bort + kräver att close > genomsnitt av senaste 3 closes
        if body_ratio < 0.3:
            return False
        if len(previous_candles) >= 3:
            avg_close = sum(c["close"] for c in previous_candles[-3:]) / 3
            if candle["close"] <= avg_close:
                return False
        return True

    return True


def mock_entry(price):
    return price * (1 + MOCK_SPREAD + MOCK_SLIPPAGE)


def mock_exit(price):
    return price * (1 - MOCK_SPREAD - MOCK_SLIPPAGE)


###############################################################
#  ENTRY & EXIT
###############################################################

def handle_entry(symbol, candle, state):
    """
    Hanterar entry beroende på mock/live. Just nu är live bara "planned".
    """
    global bot_mode, trade_size_usdt

    entry = state.current_orb["high"]

    if bot_mode == "mock":
        adj = mock_entry(entry)
        state.entry_price = adj
        state.stop_price = state.current_orb["low"]
        state.in_position = True
        state.trailing_active = False
        print(f"[MOCK ENTRY] {symbol} @ {adj:.4f} (ORB high {entry:.4f})")
        return True

    # Live-mode: plats för riktig KuCoin-orderläggning senare
    if not KUCOIN_API_KEY:
        print("[LIVE] Försökte entry utan API-nycklar – ignoreras.")
        return False

    # TODO: implementera riktig order här
    print(f"[LIVE ENTRY PLANERAD] {symbol} @ {entry:.4f}")
    state.entry_price = entry
    state.stop_price = state.current_orb["low"]
    state.in_position = True
    return True


def handle_exit(symbol, candle, state):
    global bot_mode

    exit_price = candle["low"]

    if bot_mode == "mock":
        adj = mock_exit(exit_price)
        print(f"[MOCK EXIT] {symbol} @ {adj:.4f} (candle low {exit_price:.4f})")
        state.in_position = False
        state.trailing_active = False
        return adj

    if not KUCOIN_API_KEY:
        print("[LIVE] Försökte exit utan API-nycklar – ignoreras.")
        state.in_position = False
        state.trailing_active = False
        return exit_price

    # TODO: implementera riktig sell-order här
    print(f"[LIVE EXIT PLANERAD] {symbol} @ {exit_price:.4f}")
    state.in_position = False
    state.trailing_active = False
    return exit_price


###############################################################
#  ORB LOOP PER SYMBOL
###############################################################

def run_orb(symbol):
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
                time.sleep(2)
                continue

            last_c = candles[-1]
            prev_c = candles[-2]

            # Ny ORB?
            if detect_new_orb(prev_c, last_c):
                state.current_orb = create_orb(last_c)
                state.in_position = False
                state.trailing_active = False
                print(f"[ORB] Ny ORB i {symbol}: H={state.current_orb['high']:.4f} "
                      f"L={state.current_orb['low']:.4f}")

            if not state.current_orb:
                time.sleep(1)
                continue

            # Entry?
            if (not state.in_position) and should_enter(last_c, state.current_orb):
                if ai_filter_pass(last_c, candles[-6:-1], ai_mode):
                    handle_entry(symbol, last_c, state)

            # Enkel SL
            if state.in_position and last_c["low"] <= state.stop_price:
                handle_exit(symbol, last_c, state)

            time.sleep(2)

        except Exception as e:
            print(f"[ERROR {symbol}] {e}")
            time.sleep(3)


def start_orb_threads():
    for symbol in ACTIVE_COINS:
        t = threading.Thread(target=run_orb, args=(symbol,), daemon=True)
        t.start()


###############################################################
#  TELEGRAM UI – MENY + KOMMANDON
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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("MP ORBbot – välj ett alternativ:", reply_markup=main_menu())


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode, ai_mode, entry_mode, trade_size_usdt

    text = (
        "📊 *Status*\n"
        f"Motor: {'ON' if bot_running else 'OFF'}\n"
        f"Läge: {bot_mode}\n"
        f"AI: {ai_mode}\n"
        f"Entry: {entry_mode}\n"
        f"Trade size: {trade_size_usdt} USDT\n"
        f"Coins: {', '.join(ACTIVE_COINS)}"
    )
    await update.message.reply_markdown(text)


async def cmd_mock_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode
    bot_mode = "mock"
    bot_running = True
    await update.message.reply_text("▶ Mock-läge startat.", reply_markup=main_menu())


async def cmd_live_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode
    bot_mode = "live"
    bot_running = True
    await update.message.reply_text("🔥 Live-läge (planerat) startat.", reply_markup=main_menu())


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running
    bot_running = False
    await update.message.reply_text("⛔ Motorn stoppad.", reply_markup=main_menu())


async def cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("Aggressiv", callback_data="ai_aggr")],
        [InlineKeyboardButton("Neutral", callback_data="ai_neut")],
        [InlineKeyboardButton("Försiktig", callback_data="ai_safe")],
    ]
    await update.message.reply_text("Välj AI-läge:", reply_markup=InlineKeyboardMarkup(kb))


async def cmd_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("Close", callback_data="entry_close")],
        [InlineKeyboardButton("Tick", callback_data="entry_tick")],
        [InlineKeyboardButton("Retest", callback_data="entry_retest")],
    ]
    await update.message.reply_text("Välj entry-mode:", reply_markup=InlineKeyboardMarkup(kb))


async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("10 USDT", callback_data="size_10"),
         InlineKeyboardButton("20 USDT", callback_data="size_20")],
        [InlineKeyboardButton("30 USDT", callback_data="size_30"),
         InlineKeyboardButton("50 USDT", callback_data="size_50")],
    ]
    await update.message.reply_text("Välj trade size:", reply_markup=InlineKeyboardMarkup(kb))


###############################################################
#  CALLBACKS FÖR INLINE-KNAPPAR
###############################################################

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode, ai_mode, entry_mode, trade_size_usdt

    q = update.callback_query
    await q.answer()
    cmd = q.data

    # Huvudkontroller
    if cmd == "start_mock":
        bot_mode = "mock"
        bot_running = True
        await q.edit_message_text("▶ Mock-läge startat.", reply_markup=main_menu())
        return

    if cmd == "start_live":
        bot_mode = "live"
        bot_running = True
        await q.edit_message_text("🔥 Live-läge (planerat) startat.", reply_markup=main_menu())
        return

    if cmd == "stop":
        bot_running = False
        await q.edit_message_text("⛔ Motorn stoppad.", reply_markup=main_menu())
        return

    if cmd == "status":
        text = (
            "📊 *Status*\n"
            f"Motor: {'ON' if bot_running else 'OFF'}\n"
            f"Läge: {bot_mode}\n"
            f"AI: {ai_mode}\n"
            f"Entry: {entry_mode}\n"
            f"Trade size: {trade_size_usdt} USDT\n"
            f"Coins: {', '.join(ACTIVE_COINS)}"
        )
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu())
        return

    # AI-meny
    if cmd == "ai_menu":
        kb = [
            [InlineKeyboardButton("Aggressiv", callback_data="ai_aggr")],
            [InlineKeyboardButton("Neutral", callback_data="ai_neut")],
            [InlineKeyboardButton("Försiktig", callback_data="ai_safe")],
        ]
        await q.edit_message_text("Välj AI-läge:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("ai_"):
        if cmd == "ai_aggr":
            ai_mode = "aggressiv"
        elif cmd == "ai_neut":
            ai_mode = "neutral"
        elif cmd == "ai_safe":
            ai_mode = "forsiktig"

        await q.edit_message_text(f"🤖 AI-läge satt till: {ai_mode}", reply_markup=main_menu())
        return

    # Entry-meny
    if cmd == "entry_menu":
        kb = [
            [InlineKeyboardButton("Close", callback_data="entry_close")],
            [InlineKeyboardButton("Tick", callback_data="entry_tick")],
            [InlineKeyboardButton("Retest", callback_data="entry_retest")],
        ]
        await q.edit_message_text("Välj entry-mode:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("entry_"):
        if cmd == "entry_close":
            entry_mode = "close"
        elif cmd == "entry_tick":
            entry_mode = "tick"
        elif cmd == "entry_retest":
            entry_mode = "retest"

        await q.edit_message_text(f"🎯 Entry-mode: {entry_mode}", reply_markup=main_menu())
        return

    # Trade size-meny
    if cmd == "size_menu":
        kb = [
            [InlineKeyboardButton("10 USDT", callback_data="size_10"),
             InlineKeyboardButton("20 USDT", callback_data="size_20")],
            [InlineKeyboardButton("30 USDT", callback_data="size_30"),
             InlineKeyboardButton("50 USDT", callback_data="size_50")],
        ]
        await q.edit_message_text("Välj trade size:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("size_"):
        val = int(cmd.replace("size_", ""))
        trade_size_usdt = val
        save_trade_size(val)
        await q.edit_message_text(
            f"💰 Trade size satt till {val} USDT.",
            reply_markup=main_menu()
        )
        return


###############################################################
#  REGISTRERA TELEGRAM-KOMMANDON (NEDRE RADEN)
###############################################################

async def set_bot_commands(application):
    commands = [
        BotCommand("start", "Visa huvudmenyn"),
        BotCommand("status", "Visa status"),
        BotCommand("mock_start", "Starta mock-läge"),
        BotCommand("live_start", "Starta live-läge (planerat)"),
        BotCommand("stop", "Stoppa motorn"),
        BotCommand("ai", "Välj AI-läge"),
        BotCommand("entry", "Välj entry-mode"),
        BotCommand("size", "Välj trade size"),
    ]
    await application.bot.set_my_commands(commands)


###############################################################
#  MAIN
###############################################################

async def main_async():
    print("MP ORBbot startar ORB-trådar...")
    start_orb_threads()

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Kommandon
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("mock_start", cmd_mock_start))
    application.add_handler(CommandHandler("live_start", cmd_live_start))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(CommandHandler("ai", cmd_ai))
    application.add_handler(CommandHandler("entry", cmd_entry))
    application.add_handler(CommandHandler("size", cmd_size))

    # Knappar
    application.add_handler(CallbackQueryHandler(button_handler))

    # Sätt kommandomenyn (knapparna längst nere)
    await set_bot_commands(application)

    print("Telegram-bot kör run_polling()...")
    await application.run_polling()


def main():
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
