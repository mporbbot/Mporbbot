# main.py
# Mp ORBbot – Render Web Service + Telegram (PTB 13.15)

import os
import csv
import time
import json
import hmac
import math
import queue
import base64
import logging
import hashlib
import threading
from datetime import datetime, timedelta, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# === Telegram v13.15 ===
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ParseMode
from telegram.ext import (
    Updater,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    Filters,
)

# -------------------------
# Konfig & Globals
# -------------------------
TZ = timezone(timedelta(hours=0))  # logga i UTC
LOG_DIR = "logs"
CSV_PATH = os.path.join(LOG_DIR, "mporbbot_trades.csv")
os.makedirs(LOG_DIR, exist_ok=True)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "XRPUSDT", "LINKUSDT"]
TIMEFRAME_MIN = 3  # 1 / 3 / 5 min stöds
FEE_PER_SIDE = 0.0010  # 0.10% per sida
MOCK_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))

# Motor/AI-state
STATE = {
    "mode": "MOCK",  # MOCK eller LIVE
    "engine_on": False,
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe_min": TIMEFRAME_MIN,
    "ai": "neutral",  # bara text-info
    "pnl_day_mock": 0.0,
    "pnl_day_live": 0.0,
}

# ORB working state per symbol
# Vi använder "candle_ref" = senaste **stängda** candlen (high/low)
# Entry: brytning över ref_high. Stop följer med till varje ny candles low.
SYMBOL_STATE = {}
SYMBOL_LOCK = threading.Lock()

# Telegram
updater = None
dispatcher = None

# Motortrådar
WORKER_THREAD = None
WORKER_STOP = threading.Event()

# Keep-alive ping (för att Render inte ska somna om du vill)
KEEPALIVE = {"on": False, "last_ping": 0}

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -------------------------
# Hjälp-funktioner
# -------------------------
def now_utc() -> datetime:
    return datetime.now(tz=TZ)


def ts() -> str:
    return now_utc().strftime("%Y-%m-%d %H:%M:%S")


def write_csv_row(row: dict):
    header = [
        "ts_utc",
        "mode",
        "symbol",
        "side",
        "entry",
        "exit",
        "qty",
        "fee_entry",
        "fee_exit",
        "pnl_usdt",
        "comment",
    ]
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def ku_public_kline(symbol: str, interval_min: int, limit: int = 3):
    """
    Hämtar senaste klines via KuCoin public REST.
    Returnerar lista med candles (close_time, open, high, low, close) – float.
    KuCoin endpoint: https://api.kucoin.com/api/v1/market/candles?type=1min&symbol=BTC-USDT
    """
    pair = symbol.replace("USDT", "-USDT")
    type_map = {1: "1min", 3: "3min", 5: "5min"}
    typ = type_map.get(interval_min, "3min")
    url = f"https://api.kucoin.com/api/v1/market/candles?type={typ}&symbol={pair}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        # KuCoin returnerar: [time, open, close, high, low, volume, turnover]
        # Vi mappas om till: (close_time, open, high, low, close)
        out = []
        for item in data[:limit]:
            # time är slutet av intervallet som string: "1700834580"
            close_time = int(item[0])
            open_p = float(item[1])
            close_p = float(item[2])
            high_p = float(item[3])
            low_p = float(item[4])
            out.append((close_time, open_p, high_p, low_p, close_p))
        # KuCoin ger senaste först – vi vill ha stigande tid
        out.sort(key=lambda x: x[0])
        return out
    except Exception as e:
        logging.warning(f"kline error {symbol}: {e}")
        return []


def ku_public_price(symbol: str) -> float:
    pair = symbol.replace("USDT", "-USDT")
    url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={pair}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return float(r.json()["data"]["price"])
    except Exception as e:
        logging.warning(f"price error {symbol}: {e}")
        return 0.0


def calc_qty_usdt(symbol: str, usdt_amount: float, price: float) -> float:
    if price <= 0:
        return 0.0
    qty = usdt_amount / price
    # förenklad precision
    return float(f"{qty:.6f}")


# -------------------------
# ORB-Motor (endast LONG)
# -------------------------
def ensure_symbol_state(symbol: str):
    with SYMBOL_LOCK:
        if symbol not in SYMBOL_STATE:
            SYMBOL_STATE[symbol] = {
                "ref_ts": 0,          # candle close time vi refererar till
                "ref_high": None,
                "ref_low": None,
                "pos_qty": 0.0,
                "entry": None,
                "stop": None,
                "trades": 0,
                "side": None,
            }


def update_orb_reference(symbol: str, candles: list):
    """
    Använd senast STÄNGDA candle som referens.
    """
    ensure_symbol_state(symbol)
    if len(candles) < 2:
        return
    # näst sista är senast stängda
    ref = candles[-2]
    ref_close_time, _o, ref_high, ref_low, _c = ref
    st = SYMBOL_STATE[symbol]
    if st["ref_ts"] != ref_close_time:
        st["ref_ts"] = ref_close_time
        st["ref_high"] = ref_high
        st["ref_low"] = ref_low
        # Flytta stop upp till senaste candle low om position finns
        if st["pos_qty"] > 0 and st["stop"] is not None:
            st["stop"] = max(st["stop"], ref_low)


def engine_step_symbol(symbol: str, tf_min: int):
    """
    Ett steg av strategin för ett symbol: hämtar candles+price och agerar.
    """
    candles = ku_public_kline(symbol, tf_min, limit=3)
    if not candles:
        return
    update_orb_reference(symbol, candles)

    price = ku_public_price(symbol)
    if price <= 0:
        return

    st = SYMBOL_STATE[symbol]
    ref_high = st["ref_high"]
    ref_low = st["ref_low"]

    # Entry: bryter över ref_high, endast om ingen position
    if st["pos_qty"] == 0 and ref_high and price > ref_high:
        qty = calc_qty_usdt(symbol, MOCK_USDT if STATE["mode"] == "MOCK" else MOCK_USDT, price)
        if qty > 0:
            st["pos_qty"] = qty
            st["entry"] = price
            st["stop"] = ref_low  # initial stop under referens-candlens low
            st["trades"] += 1
            st["side"] = "LONG"
            msg = f"✅ {STATE['mode']}: BUY {symbol} @ {price:.6f} x {qty}"
            bot_send(msg)

    # Exit: om stop träffas
    if st["pos_qty"] > 0 and st["stop"] is not None and price <= st["stop"]:
        exit_price = price
        qty = st["pos_qty"]
        entry = st["entry"] or price
        fee_in = entry * qty * FEE_PER_SIDE
        fee_out = exit_price * qty * FEE_PER_SIDE
        pnl = (exit_price - entry) * qty - fee_in - fee_out

        # uppdatera PnL-dag
        if STATE["mode"] == "MOCK":
            STATE["pnl_day_mock"] += pnl
        else:
            STATE["pnl_day_live"] += pnl

        write_csv_row(
            {
                "ts_utc": ts(),
                "mode": STATE["mode"],
                "symbol": symbol,
                "side": "LONG",
                "entry": f"{entry:.8f}",
                "exit": f"{exit_price:.8f}",
                "qty": f"{qty:.6f}",
                "fee_entry": f"{fee_in:.6f}",
                "fee_exit": f"{fee_out:.6f}",
                "pnl_usdt": f"{pnl:.6f}",
                "comment": "stop hit",
            }
        )
        bot_send(
            f"📉 Stängd LONG {symbol}\n"
            f"Entry: {entry:.6f}  Exit: {exit_price:.6f}\n"
            f"Qty: {qty:.6f}\n"
            f"PnL idag ({STATE['mode']}): {STATE['pnl_day_mock'] if STATE['mode']=='MOCK' else STATE['pnl_day_live']:.6f}"
        )

        # nollställ
        st["pos_qty"] = 0.0
        st["entry"] = None
        st["stop"] = None
        st["side"] = None


def engine_loop():
    logging.info("Engine loop started")
    while not WORKER_STOP.is_set():
        if STATE["engine_on"]:
            for sym in STATE["symbols"]:
                try:
                    engine_step_symbol(sym, STATE["timeframe_min"])
                except Exception as e:
                    logging.warning(f"engine step error {sym}: {e}")
        # Keep-alive ping logik (valfritt)
        if KEEPALIVE["on"]:
            now = time.time()
            if now - KEEPALIVE["last_ping"] > 60:
                KEEPALIVE["last_ping"] = now
                try:
                    requests.get("https://www.google.com", timeout=5)
                except Exception:
                    pass
        time.sleep(3)
    logging.info("Engine loop stopped")


def start_engine_thread():
    global WORKER_THREAD
    if WORKER_THREAD and WORKER_THREAD.is_alive():
        return
    WORKER_STOP.clear()
    WORKER_THREAD = threading.Thread(target=engine_loop, daemon=True)
    WORKER_THREAD.start()


def stop_engine_thread():
    WORKER_STOP.set()
    if WORKER_THREAD:
        WORKER_THREAD.join(timeout=2)


# -------------------------
# Telegram-hjälp
# -------------------------
def bot_send(text: str):
    try:
        if updater:
            updater.bot.send_message(chat_id=ADMIN_CHAT_ID(), text=text)
    except Exception as e:
        logging.warning(f"send err: {e}")


def ADMIN_CHAT_ID() -> int:
    # Om du vill låsa till din egen chat – lägg ett env-värde ADMIN_CHAT_ID
    val = os.getenv("ADMIN_CHAT_ID")
    try:
        return int(val) if val else None
    except:
        return None


def require_admin(update: Update) -> bool:
    admin = ADMIN_CHAT_ID()
    if admin is None:
        return True  # fritt
    return update.effective_user and update.effective_user.id == admin


# -------------------------
# Telegram handlers
# -------------------------
def cmd_help(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    txt = (
        "/status – visa status\n"
        "/set_ai <neutral|aggressiv|försiktig>\n"
        "/start_mock – starta MOCK (svara JA)\n"
        "/start_live – starta LIVE (svara JA)\n"
        "/engine_start – starta motor\n"
        "/engine_stop – stoppa motor\n"
        "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
        "/timeframe 1min|3min|5min – byt tidsram\n"
        "/pnl – visa dagens PnL\n"
        "/reset_pnl – nollställ PnL idag\n"
        "/keepalive_on – håll Render vaken\n"
        "/keepalive_off – stäng keepalive\n"
    )
    update.message.reply_text(txt)


def cmd_status(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    tf = STATE["timeframe_min"]
    pnl_m = STATE["pnl_day_mock"]
    pnl_l = STATE["pnl_day_live"]
    # sammanställ symbolstate
    lines = []
    for s in STATE["symbols"]:
        st = SYMBOL_STATE.get(s) or {}
        pos = "Y" if (st.get("pos_qty") or 0) > 0 else "N"
        side = st.get("side")
        trades = st.get("trades", 0)
        rh = st.get("ref_high")
        rl = st.get("ref_low")
        lines.append(
            f"{s}: pos={pos} side={side} refH={fmt(rh)} refL={fmt(rl)} trades={trades}"
        )
    header = (
        f"Bot: Mp ORBbot\n"
        f"Läge: {STATE['mode']}\n"
        f"Motor: {'ON' if STATE['engine_on'] else 'OFF'}\n"
        f"Timeframe: {tf}min\n"
        f"Symbols: {','.join(STATE['symbols'])}\n"
        f"PnL idag → MOCK: {pnl_m:.6f} | LIVE: {pnl_l:.6f}\n\n"
    )
    update.message.reply_text(header + "\n".join(lines) if lines else header + "(ingen symbolinfo)")


def fmt(x):
    return f"{x:.6f}" if isinstance(x, (int, float)) and not math.isnan(x) else "None"


def cmd_set_ai(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    if not ctx.args:
        update.message.reply_text("Använd: /set_ai neutral|aggressiv|försiktig")
        return
    val = ctx.args[0].strip().lower()
    if val not in ["neutral", "aggressiv", "försiktig"]:
        update.message.reply_text("Ogiltigt värde.")
        return
    STATE["ai"] = val
    update.message.reply_text(f"AI-läge satt till: {val}")


def cmd_symbols(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    if not ctx.args:
        update.message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,...")
        return
    raw = " ".join(ctx.args).replace(" ", "")
    syms = [s for s in raw.split(",") if s]
    if not syms:
        update.message.reply_text("Inga symbols hittades.")
        return
    STATE["symbols"] = syms
    for s in syms:
        ensure_symbol_state(s)
    update.message.reply_text("Symbols uppdaterade.")


def cmd_timeframe(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    if not ctx.args:
        update.message.reply_text("Använd: /timeframe 1min|3min|5min")
        return
    val = ctx.args[0].strip().lower()
    m = {"1min": 1, "3min": 3, "5min": 5}.get(val)
    if not m:
        update.message.reply_text("Endast 1min, 3min eller 5min stöds.")
        return
    STATE["timeframe_min"] = m
    update.message.reply_text(f"Tidsram satt till {m}min.")


def cmd_engine_start(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    STATE["engine_on"] = True
    start_engine_thread()
    update.message.reply_text("Motor: ON")


def cmd_engine_stop(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    STATE["engine_on"] = False
    update.message.reply_text("Motor: OFF")


def cmd_pnl(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    update.message.reply_text(
        f"Dagens PnL\nMOCK: {STATE['pnl_day_mock']:.6f} USDT\nLIVE: {STATE['pnl_day_live']:.6f} USDT"
    )


def cmd_reset_pnl(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    STATE["pnl_day_mock"] = 0.0
    STATE["pnl_day_live"] = 0.0
    update.message.reply_text("PnL idag nollställd.")


def cmd_keepalive_on(update: Update, ctx: CallbackContext):
    KEEPALIVE["on"] = True
    update.message.reply_text("Keepalive: ON")


def cmd_keepalive_off(update: Update, ctx: CallbackContext):
    KEEPALIVE["on"] = False
    update.message.reply_text("Keepalive: OFF")


# Start MOCK/LIVE med "JA" bekräftelse (enkelt UI likt din bild)
PENDING_CONFIRM = {"type": None}  # "MOCK" eller "LIVE"


def cmd_start_mock(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    PENDING_CONFIRM["type"] = "MOCK"
    update.message.reply_text("Vill du starta MOCK-trading? Svara JA.")


def cmd_start_live(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    if not (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE):
        update.message.reply_text("LIVE kräver KuCoin API-nycklar i miljövariablerna.")
        return
    PENDING_CONFIRM["type"] = "LIVE"
    update.message.reply_text("Vill du starta LIVE-trading? Svara JA.")


def msg_text(update: Update, ctx: CallbackContext):
    if not require_admin(update):
        return
    txt = (update.message.text or "").strip().lower()
    if txt == "ja" and PENDING_CONFIRM["type"]:
        STATE["mode"] = PENDING_CONFIRM["type"]
        PENDING_CONFIRM["type"] = None
        update.message.reply_text(f"Läge satt: {STATE['mode']}.")
        return


# -------------------------
# Starta Telegram-botten en gång
# -------------------------
_started = False
def start_bot_once():
    global updater, dispatcher, _started
    if _started:
        return
    if not TELEGRAM_TOKEN:
        logging.warning("Ingen TELEGRAM_TOKEN satt – boten startas inte.")
        return
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("help", cmd_help))
    dispatcher.add_handler(CommandHandler("status", cmd_status))
    dispatcher.add_handler(CommandHandler("set_ai", cmd_set_ai, pass_args=True))
    dispatcher.add_handler(CommandHandler("symbols", cmd_symbols, pass_args=True))
    dispatcher.add_handler(CommandHandler("timeframe", cmd_timeframe, pass_args=True))
    dispatcher.add_handler(CommandHandler("engine_start", cmd_engine_start))
    dispatcher.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dispatcher.add_handler(CommandHandler("pnl", cmd_pnl))
    dispatcher.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dispatcher.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dispatcher.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dispatcher.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dispatcher.add_handler(CommandHandler("start_live", cmd_start_live))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, msg_text))

    updater.start_polling(drop_pending_updates=True)
    _started = True
    logging.info("Telegram bot started")


# -------------------------
# FastAPI (Render Web Service)
# -------------------------
app = FastAPI()


@app.get("/", response_class=PlainTextResponse)
def root():
    return "Mp ORBbot up"


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


# starta bot och motor när modulen laddas (Render/uvicorn importerar 'app')
start_bot_once()
start_engine_thread()

if __name__ == "__main__":
    # Lokal körning (valfritt): python main.py
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
