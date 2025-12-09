#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################
#   MP ORBbot – KuCoin Spot – Mock + (framtida) Live          #
#   ORB på 3-min, multi-coin, Telegram-styrning               #
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
from typing import Dict, List, Tuple, Set

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
    Bot,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
)

###############################################################
#  ENVIRONMENT / KONFIG                                       #
###############################################################

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN", "")

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

if not TELEGRAM_TOKEN:
    raise Exception("❌ TELEGRAM_TOKEN eller BOT_TOKEN saknas i Environment Variables!")

KUCOIN_BASE = "https://api.kucoin.com"

# De coins vi kör ORB på
ACTIVE_COINS = ["BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT", "ADA-USDT"]

# Global state
bot_running = False          # Engine ON/OFF
bot_mode = "mock"            # "mock" eller "live" (standard mock)
ai_mode = "neutral"          # aggressiv / neutral / forsiktig
entry_mode = "tick"          # tick / close / retest

risk_mode = "fixed"          # fixed / percent
risk_percent = 1.0           # om percent
trade_size_usdt = 30         # fast USDT-size för fixed

# Mock-marknadsparametrar
mock_spread = 0.0002         # 0.02 %
mock_slippage = 0.0001       # 0.01 %
fee_rate = 0.001             # 0.1 % fee per sida (modell)

CSV_REAL = "real_trade_log.csv"
CSV_MOCK = "mock_trade_log.csv"

SETTINGS_FILE = "settings.json"

# Chattar som får trade-notiser
subscribed_chats: Set[int] = set()

###############################################################
#  SETTINGS (SPARAS MELLAN RESTARTS)                          #
###############################################################

def load_settings():
    global trade_size_usdt, risk_mode, risk_percent, ai_mode, entry_mode, bot_mode
    if not os.path.exists(SETTINGS_FILE):
        save_settings()
        return
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        trade_size_usdt = data.get("trade_size_usdt", trade_size_usdt)
        risk_mode = data.get("risk_mode", risk_mode)
        risk_percent = data.get("risk_percent", risk_percent)
        ai_mode = data.get("ai_mode", ai_mode)
        entry_mode = data.get("entry_mode", entry_mode)
        bot_mode = data.get("bot_mode", bot_mode)
    except Exception as e:
        print(f"[SETTINGS] Kunde inte läsa settings: {e}")
        save_settings()


def save_settings():
    data = {
        "trade_size_usdt": trade_size_usdt,
        "risk_mode": risk_mode,
        "risk_percent": risk_percent,
        "ai_mode": ai_mode,
        "entry_mode": entry_mode,
        "bot_mode": bot_mode,
    }
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[SETTINGS] Kunde inte spara settings: {e}")


load_settings()

###############################################################
#  KUCOIN-HELPERS (FÖR FRAMTIDA LIVE)                         #
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
    """Hämta USDT-trade-balans. Används för risk_mode=percent i framtiden."""
    if not KUCOIN_API_KEY:
        return 0.0
    try:
        endpoint = "/api/v1/accounts"
        headers = kucoin_sign("GET", endpoint)
        r = requests.get(KUCOIN_BASE + endpoint, headers=headers, timeout=10)
        data = r.json()
        if "data" not in data:
            return 0.0
        for acc in data["data"]:
            if acc["currency"] == "USDT" and acc["type"] == "trade":
                return float(acc["balance"])
    except Exception as e:
        print(f"[KUCOIN BALANCE ERROR] {e}")
    return 0.0


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
        candles.reverse()  # äldst först
        return candles[:limit]
    except Exception as e:
        print(f"[CANDLES ERROR {symbol}] {e}")
        return []


###############################################################
#  ORB-STATE PER COIN                                         #
###############################################################

class ORBState:
    def __init__(self):
        self.current_orb = None   # {"high": .., "low": .., "timestamp": ..}
        self.entry_price = None
        self.stop_price = None
        self.in_position = False
        self.size = 0.0
        self.last_close = None    # för /close_all


orb_states: Dict[str, ORBState] = {s: ORBState() for s in ACTIVE_COINS}

###############################################################
#  TELEGRAM-NOTISER                                           #
###############################################################

def notify_trades(text: str):
    """Skicka trade-notis till alla chattar som kört /start."""
    if not subscribed_chats:
        return
    try:
        bot = Bot(TELEGRAM_TOKEN)
        for cid in subscribed_chats:
            try:
                bot.send_message(chat_id=cid, text=text, parse_mode="Markdown")
            except Exception as e:
                print(f"[NOTIFY ERROR chat {cid}] {e}")
    except Exception as e:
        print(f"[NOTIFY ERROR] {e}")

###############################################################
#  ORB-LOGIK                                                  #
###############################################################

def detect_new_orb(c1, c2) -> bool:
    """Ny ORB när röd → grön (din regel)."""
    return (c1["close"] < c1["open"]) and (c2["close"] > c2["open"])


def create_orb(candle):
    return {
        "high": candle["high"],
        "low": candle["low"],
        "timestamp": candle["timestamp"],
    }


def should_enter(candle, orb) -> bool:
    """Breakout av ORB-high beroende på entry_mode."""
    if entry_mode == "tick":
        return candle["high"] >= orb["high"]
    elif entry_mode == "close":
        return candle["close"] >= orb["high"]
    elif entry_mode == "retest":
        # kräver retest av low innan break
        if candle["low"] <= orb["low"] * 1.002 and candle["high"] >= orb["high"]:
            return True
        return False
    return False


def ai_filter(candle, prev_candles: List[dict]) -> bool:
    """Enkel AI-filter. Aggressiv = stäng av filter."""
    body = abs(candle["close"] - candle["open"])
    rng = candle["high"] - candle["low"]
    if rng <= 0:
        rng = 1e-9
    body_ratio = body / rng

    # Aggressiv = inga filter
    if ai_mode == "aggressiv":
        return True

    # Neutral – bara ta bort micro-dojis
    if ai_mode == "neutral":
        if body_ratio < 0.05:
            return False
        return True

    # Försiktig – kräv tydlig kropp + lite momentum
    if ai_mode == "forsiktig":
        if body_ratio < 0.3:
            return False
        if len(prev_candles) >= 3:
            avg = sum(c["close"] for c in prev_candles[-3:]) / 3
            if candle["close"] <= avg:
                return False
        return True

    return True


###############################################################
#  MOCK-TRADING                                               #
###############################################################

def mock_entry_price(price: float) -> float:
    return price * (1 + mock_spread + mock_slippage + fee_rate)


def mock_exit_price(price: float) -> float:
    return price * (1 - mock_spread - mock_slippage - fee_rate)


def mock_position_size(entry_price: float) -> float:
    return trade_size_usdt / entry_price


###############################################################
#  RISK / STORLEK (för framtida live)                         #
###############################################################

def calculate_live_size(price: float) -> float:
    global risk_mode, risk_percent, trade_size_usdt
    if risk_mode == "fixed":
        return trade_size_usdt / price
    if risk_mode == "percent":
        bal = kucoin_get_balance()
        if bal <= 0:
            return 0.0
        use_amount = bal * (risk_percent / 100.0)
        return use_amount / price
    return trade_size_usdt / price


###############################################################
#  CSV-LOGGNING (SKATTEVERKET)                                #
###############################################################

def ensure_csv_files():
    if not os.path.exists(CSV_REAL):
        with open(CSV_REAL, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "symbol", "side", "price", "size", "mode"])

    if not os.path.exists(CSV_MOCK):
        with open(CSV_MOCK, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "symbol", "side", "price", "size", "mode"])


def log_real_trade(symbol: str, side: str, price: float, size: float):
    ensure_csv_files()
    with open(CSV_REAL, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.utcnow().isoformat(), symbol, side, price, size, "live"])


def log_mock_trade(symbol: str, side: str, price: float, size: float):
    ensure_csv_files()
    with open(CSV_MOCK, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.utcnow().isoformat(), symbol, side, price, size, "mock"])


###############################################################
#  PnL-BERÄKNING FRÅN CSV                                     #
###############################################################

def compute_mock_pnl_from_csv() -> Tuple[float, int]:
    """
    Läser mock_trade_log.csv och räknar:
    - total PnL (USDT)
    - antal avslutade trades (ENTRY + EXIT)
    Antagande: max 1 öppen position per symbol åt gången.
    """
    if not os.path.exists(CSV_MOCK):
        return 0.0, 0

    total_pnl = 0.0
    trades = 0
    open_pos = {}  # symbol -> (entry_price, size)

    try:
        with open(CSV_MOCK, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) < 6:
                    continue
                ts, symbol, side, price_str, size_str, mode = row
                price = float(price_str)
                size = float(size_str)

                if side == "ENTRY":
                    open_pos[symbol] = (price, size)
                elif side == "EXIT":
                    if symbol in open_pos:
                        ep, sz = open_pos.pop(symbol)
                        pnl = (price - ep) * sz
                        total_pnl += pnl
                        trades += 1
    except Exception as e:
        print(f"[PNL] Fel vid läsning av CSV: {e}")
        return 0.0, 0

    return total_pnl, trades


###############################################################
#  ENTRY / EXIT-GEMENSAMT                                     #
###############################################################

def execute_entry(symbol: str, candle: dict) -> bool:
    """Gemensam entry-funktion för mock (och framtida live)."""
    state = orb_states[symbol]
    orb = state.current_orb
    if not orb:
        return False

    entry_price_raw = orb["high"]

    if bot_mode == "mock":
        adj_price = mock_entry_price(entry_price_raw)
        size = mock_position_size(adj_price)
        state.entry_price = adj_price
        state.stop_price = orb["low"]
        state.size = size
        state.in_position = True

        log_mock_trade(symbol, "ENTRY", adj_price, size)
        print(f"[MOCK ENTRY] {symbol} @ {adj_price:.4f} size={size:.4f}")
        notify_trades(
            f"🟢 *ENTRY* ({bot_mode})\n"
            f"Symbol: `{symbol}`\n"
            f"Pris: {adj_price:.4f}\n"
            f"Size: {size:.4f}"
        )
        return True

    # Live-handeln kan läggas till senare
    print("[LIVE] Live-mode ej fullt implementerat än – kör mock istället.")
    return False


def execute_exit(symbol: str, candle: dict, reason: str = "EXIT") -> float:
    """Gemensam exit-funktion."""
    state = orb_states[symbol]
    if not state.in_position:
        return 0.0

    exit_price_raw = candle["low"]

    if bot_mode == "mock":
        adj_price = mock_exit_price(exit_price_raw)
        log_mock_trade(symbol, "EXIT", adj_price, state.size)
        print(f"[MOCK EXIT] {symbol} @ {adj_price:.4f} size={state.size:.4f}")
        notify_trades(
            f"🔴 *{reason}* ({bot_mode})\n"
            f"Symbol: `{symbol}`\n"
            f"Pris: {adj_price:.4f}\n"
            f"Size: {state.size:.4f}"
        )
        state.in_position = False
        state.size = 0.0
        state.entry_price = None
        state.stop_price = None
        return adj_price

    # Live-handeln kan läggas till senare
    print("[LIVE] Exit i live-mode ej implementerad ännu.")
    state.in_position = False
    state.size = 0.0
    return exit_price_raw


###############################################################
#  ORB-LOOP PER COIN                                          #
###############################################################

def orb_loop(symbol: str):
    global bot_running

    state = orb_states[symbol]
    print(f"[ORB] Loop startad för {symbol}")

    last_debug = 0

    while True:
        try:
            if not bot_running:
                time.sleep(2)
                continue

            candles = kucoin_get_candles(symbol, limit=60)
            if len(candles) < 5:
                time.sleep(2)
                continue

            last = candles[-1]
            prev = candles[-2]
            state.last_close = last["close"]

            # debug var 60:e sekund
            now = int(time.time())
            if now - last_debug > 60:
                last_debug = now
                orb_high = state.current_orb["high"] if state.current_orb else 0.0
                print(
                    f"[{symbol}] close={last['close']:.4f} "
                    f"ORB_high={orb_high:.4f} "
                    f"in_position={state.in_position}"
                )

            # Ny ORB? (röd → grön)
            if detect_new_orb(prev, last):
                state.current_orb = create_orb(last)
                state.in_position = False
                state.entry_price = None
                state.stop_price = None
                state.size = 0.0
                print(
                    f"[ORB] Ny ORB {symbol} "
                    f"HIGH={state.current_orb['high']:.4f} "
                    f"LOW={state.current_orb['low']:.4f}"
                )

            # Har vi ORB men ingen position → kolla entry
            if state.current_orb and not state.in_position:
                if should_enter(last, state.current_orb):
                    prev5 = candles[-6:-1]
                    if ai_filter(last, prev5):
                        print(f"[TRIGGER] ORB-breakout i {symbol}, försöker ENTRY...")
                        execute_entry(symbol, last)

            # Stop-loss – break under ORB-low
            if state.in_position and state.stop_price is not None:
                if last["low"] <= state.stop_price:
                    print(f"[STOP] Stop-loss i {symbol}")
                    execute_exit(symbol, last, reason="STOP-LOSS")

            time.sleep(3)

        except Exception as e:
            print(f"[ERROR {symbol}] {e}")
            time.sleep(5)


###############################################################
#  STARTA ORB-TRÅDAR                                           #
###############################################################

def start_orb_threads():
    for s in ACTIVE_COINS:
        t = threading.Thread(target=orb_loop, args=(s,), daemon=True)
        t.start()
        print(f"[THREAD] Startade ORB-loop för {s}")


###############################################################
#  TELEGRAM – MENYER                                           #
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
        ],
    ])


###############################################################
#  TELEGRAM – KOMMANDON                                        #
###############################################################

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Registrera chatten för notiser + visa meny."""
    chat_id = update.effective_chat.id
    subscribed_chats.add(chat_id)
    await update.message.reply_text(
        "MP ORBbot – du är nu registrerad för trade-notiser.\n\nHuvudmeny:",
        reply_markup=main_menu(),
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
    bot_mode = "mock"
    save_settings()
    await update.message.reply_text("▶ Engine ON (mock-läge)", reply_markup=main_menu())


async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running
    bot_running = False
    save_settings()
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
        [InlineKeyboardButton("Procent av konto", callback_data="risk_percent_menu")],
    ]
    await update.message.reply_text("💰 Välj risk-modell:", reply_markup=InlineKeyboardMarkup(kb))


async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [
            InlineKeyboardButton("10 USDT", callback_data="size_10"),
            InlineKeyboardButton("20 USDT", callback_data="size_20"),
        ],
        [
            InlineKeyboardButton("30 USDT", callback_data="size_30"),
            InlineKeyboardButton("50 USDT", callback_data="size_50"),
        ],
    ]
    await update.message.reply_text("📏 Välj fast trade-size:", reply_markup=InlineKeyboardMarkup(kb))


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Visa total mock-PnL baserat på mock_trade_log.csv."""
    total_pnl, trades = compute_mock_pnl_from_csv()
    text = (
        f"📈 *Mock-PnL*\n"
        f"Avslutade trades: {trades}\n"
        f"Total PnL: {total_pnl:.4f} USDT"
    )
    await update.message.reply_markdown(text)


async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏱ ORB kör alltid på 3-minuters timeframe i denna version.")


async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "🎛 *Threshold / AI-läge*\n"
        "- Aggressiv = tar nästan alla breakouts\n"
        "- Neutral   = filtrerar bort små dojis\n"
        "- Försiktig = kräver större kropp + momentum\n\n"
        "Ändra via /ai eller knappen 'AI-läge'."
    )
    await update.message.reply_markdown(txt)


async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Skicka mock_trade_log.csv till Telegram."""
    ensure_csv_files()
    chat_id = update.message.chat_id
    if not os.path.exists(CSV_MOCK):
        await update.message.reply_text("Ingen mock_trade_log.csv hittades ännu.")
        return
    try:
        await context.bot.send_document(chat_id=chat_id, document=open(CSV_MOCK, "rb"))
    except Exception as e:
        await update.message.reply_text(f"Kunde inte skicka CSV: {e}")


async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stäng alla öppna mock-positioner direkt (använder senaste close)."""
    closed = 0
    for symbol, state in orb_states.items():
        if state.in_position and state.last_close is not None:
            candle = {"low": state.last_close}
            execute_exit(symbol, candle, reason="CLOSE_ALL")
            closed += 1
    await update.message.reply_text(f"🔚 Stängde {closed} öppna position(er).")


async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Nollställ mock-PnL (töm CSV och skriv bara header)."""
    with open(CSV_MOCK, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "symbol", "side", "price", "size", "mode"])
    await update.message.reply_text("🧹 mock_trade_log.csv är nollställd.")


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Växla mellan mock och live (live är förberedd men inte klar)."""
    global bot_mode
    if bot_mode == "mock":
        bot_mode = "live"
    else:
        bot_mode = "mock"
    save_settings()
    await update.message.reply_text(
        f"🔁 Bot-läge växlat till: {bot_mode}\n"
        "Obs: live-handeln är ännu *inte* fullt implementerad – använd mock för riktiga tester.",
        parse_mode="Markdown",
    )


async def cmd_test_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Gör en snabb test-trade i BTC-USDT:
    - Hämtar senaste pris
    - Loggar ENTRY + EXIT direkt
    Så du kan testa CSV + /pnl utan att vänta på marknaden.
    """
    symbol = "BTC-USDT"
    candles = kucoin_get_candles(symbol, limit=1)
    if not candles:
        await update.message.reply_text("Kunde inte hämta pris för test_buy.")
        return
    price = candles[-1]["close"]
    size = trade_size_usdt / price
    ensure_csv_files()
    log_mock_trade(symbol, "ENTRY", price, size)
    log_mock_trade(symbol, "EXIT", price * 1.001, size)
    notify_trades(
        f"🧪 *TEST-TRADE* loggad i `{symbol}`\n"
        f"Entry: {price:.4f} → Exit: {price*1.001:.4f}\n"
        f"Size: {size:.4f}"
    )
    await update.message.reply_text(
        f"✅ Test-buy loggad i {symbol} (ENTRY + EXIT). Kolla /pnl eller /export_csv."
    )


###############################################################
#  TELEGRAM – INLINE-KNAPPAR                                   #
###############################################################

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running, bot_mode, ai_mode, entry_mode
    global risk_mode, risk_percent, trade_size_usdt

    q = update.callback_query
    await q.answer()
    cmd = q.data

    # Engine ON/OFF
    if cmd == "engine_on":
        bot_running = True
        bot_mode = "mock"
        save_settings()
        await q.edit_message_text("▶ Engine är nu ON (mock)", reply_markup=main_menu())
        return

    if cmd == "engine_off":
        bot_running = False
        save_settings()
        await q.edit_message_text("⛔ Engine är nu OFF", reply_markup=main_menu())
        return

    # Status
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

    # AI-läge
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

    # Entry-mode
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

    # Risk-mode
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
        await q.edit_message_text("💰 Risk-modell: Fast USDT", reply_markup=main_menu())
        return

    if cmd == "risk_percent_menu":
        kb = [
            [
                InlineKeyboardButton("0.5%", callback_data="risk_pct_0_5"),
                InlineKeyboardButton("1%", callback_data="risk_pct_1"),
            ],
            [
                InlineKeyboardButton("2%", callback_data="risk_pct_2"),
                InlineKeyboardButton("5%", callback_data="risk_pct_5"),
            ],
        ]
        await q.edit_message_text("Välj risk-procent:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if cmd.startswith("risk_pct_"):
        val = cmd.replace("risk_pct_", "")
        risk_percent = float(val.replace("_", "."))
        risk_mode = "percent"
        save_settings()
        await q.edit_message_text(f"💰 Risk-procent: {risk_percent}%", reply_markup=main_menu())
        return

    # Trade size
    if cmd.startswith("size_"):
        trade_size_usdt = int(cmd.replace("size_", ""))
        risk_mode = "fixed"
        save_settings()
        await q.edit_message_text(f"📏 Trade-size: {trade_size_usdt} USDT", reply_markup=main_menu())
        return


###############################################################
#  TELEGRAM – KOMMANDOLISTA (NEDRE MENU)                      #
###############################################################

async def post_init(application):
    cmds = [
        BotCommand("status", "Visa status"),
        BotCommand("pnl", "Visa mock-PnL"),
        BotCommand("engine_on", "Starta motorn (mock)"),
        BotCommand("engine_off", "Stoppa motorn"),
        BotCommand("timeframe", "Visa timeframe"),
        BotCommand("threshold", "AI-threshold / filter-info"),
        BotCommand("risk", "Välj risk-modell"),
        BotCommand("export_csv", "Exportera mock_trade_log.csv"),
        BotCommand("close_all", "Stäng alla öppna positioner"),
        BotCommand("reset_pnl", "Nollställ mock-PnL"),
        BotCommand("mode", "Växla mock/live-läge"),
        BotCommand("test_buy", "Gör en test-trade"),
    ]
    await application.bot.set_my_commands(cmds)


###############################################################
#  MAIN                                                        #
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

    # Kommandon (för nedre knappar)
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("pnl", cmd_pnl))
    application.add_handler(CommandHandler("engine_on", cmd_engine_on))
    application.add_handler(CommandHandler("engine_off", cmd_engine_off))
    application.add_handler(CommandHandler("timeframe", cmd_timeframe))
    application.add_handler(CommandHandler("threshold", cmd_threshold))
    application.add_handler(CommandHandler("risk", cmd_risk))
    application.add_handler(CommandHandler("export_csv", cmd_export_csv))
    application.add_handler(CommandHandler("close_all", cmd_close_all))
    application.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    application.add_handler(CommandHandler("mode", cmd_mode))
    application.add_handler(CommandHandler("test_buy", cmd_test_buy))

    # Extra (via inline-knappar)
    application.add_handler(CommandHandler("ai", cmd_ai))
    application.add_handler(CommandHandler("entry", cmd_entry))
    application.add_handler(CommandHandler("size", cmd_size))
    application.add_handler(CallbackQueryHandler(button_handler))

    application.run_polling()


if __name__ == "__main__":
    main()
