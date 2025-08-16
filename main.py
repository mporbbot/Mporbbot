import os
import time
import hmac
import hashlib
import base64
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI
from telegram import (
    Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
)
from telegram.ext import (
    Updater, CommandHandler, CallbackQueryHandler, CallbackContext
)

# =============================
# Environment & Globals
# =============================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
KUCOIN_API_KEY = os.environ.get("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.environ.get("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.environ.get("KUCOIN_API_PASSPHRASE", "")

DEFAULT_SYMBOLS = os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT").split(",")
DEFAULT_TF = os.environ.get("TIMEFRAME", "1m")  # "1m","3m","5m"
MOCK_TRADE_USDT = float(os.environ.get("MOCK_TRADE_USDT", "30"))

# Trailing: OFF by default, but configurable in Telegram
TRIGGER_PCT = float(os.environ.get("TRAIL_TRIGGER_PCT", "0.9"))   # starta trailing när +0.9%
OFFSET_PCT  = float(os.environ.get("TRAIL_OFFSET_PCT",  "0.2"))   # ligg 0.2% från Högsta
MIN_LOCK_PCT = float(os.environ.get("TRAIL_MIN_LOCK_PCT", "0.0")) # min vinst som låses

KEEPALIVE_EVERY_SEC = 120

# State
STATE = {
    "mode": "mock",            # "mock" eller "live"
    "engine": False,           # start/stop
    "symbols": DEFAULT_SYMBOLS,
    "tf": DEFAULT_TF,
    "keepalive": True,
    "trailing_on": False,      # Off som standard (kan slås på)
    "trail_cfg": {"trigger": TRIGGER_PCT, "offset": OFFSET_PCT, "min_lock": MIN_LOCK_PCT},
    "chat_id": None,
}

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
TRADES_CSV = os.path.join(DATA_DIR, "trades.csv")   # alla affärer (mock/live)
K4_CSV     = os.path.join(DATA_DIR, "k4.csv")

# =============================
# Helpers
# =============================
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def log(msg: str):
    print(f"[{now_iso()}] {msg}", flush=True)

def notify(context: Optional[CallbackContext], text: str):
    chat_id = STATE.get("chat_id")
    if context and chat_id:
        try:
            context.bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            log(f"Notify error: {e}")

def kucoin_headers(endpoint: str, method: str, body: str = "") -> Dict[str, str]:
    """
    KuCoin v2 sign (spot). Keep it minimal for market orders and klines.
    """
    now_ms = str(int(time.time() * 1000))
    str_to_sign = now_ms + method.upper() + endpoint + body
    signature = base64.b64encode(
        hmac.new(KUCOIN_API_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).digest()
    ).decode()
    passphrase = base64.b64encode(
        hmac.new(KUCOIN_API_SECRET.encode(), KUCOIN_API_PASSPHRASE.encode(), hashlib.sha256).digest()
    ).decode()
    return {
        "KC-API-KEY": KUCOIN_API_KEY,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": now_ms,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }

def kucoin_get(path: str, params: Dict = None):
    url = "https://api.kucoin.com" + path
    qs = ""
    if params:
        qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    endpoint = path + (qs if qs else "")
    headers = kucoin_headers(endpoint, "GET")
    r = requests.get(url + (qs if qs else ""), headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def kucoin_post(path: str, payload: Dict):
    url = "https://api.kucoin.com" + path
    body = json.dumps(payload)
    headers = kucoin_headers(path, "POST", body)
    r = requests.post(url, headers=headers, data=body, timeout=10)
    r.raise_for_status()
    return r.json()

# =============================
# Market Data (klines)
# =============================
TF_TO_MIN = {"1m": 1, "3m": 3, "5m": 5}
def fetch_last_n_candles(symbol: str, tf: str, limit: int = 3) -> List[Dict]:
    """
    Hämtar de senaste N candlarna. Returnerar lista med dict: open, high, low, close, ts
    KuCoin endpoint: /api/v1/market/candles?type=1min&symbol=BTC-USDT
    """
    ktype = {"1m": "1min", "3m": "3min", "5m": "5min"}[tf]
    sym = symbol.replace("USDT", "-USDT")
    try:
        data = kucoin_get("/api/v1/market/candles", {"type": ktype, "symbol": sym})
        # KuCoin returns: [time, open, close, high, low, volume, turnover] newest first
        rows = data.get("data", [])[:limit]
        out = []
        for row in reversed(rows):  # oldest -> newest
            ts = int(row[0])  # unix milli (string sometimes)
            o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
            out.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
        return out
    except Exception as e:
        log(f"fetch_last_n_candles error {symbol}: {e}")
        return []

def fetch_price(symbol: str) -> Optional[float]:
    """
    Hämtar senaste pris (last trade) från KuCoin.
    """
    try:
        sym = symbol.replace("USDT", "-USDT")
        data = kucoin_get(f"/api/v1/market/orderbook/level1?symbol={sym}")
        p = float(data["data"]["price"])
        return p
    except Exception as e:
        log(f"fetch_price error {symbol}: {e}")
        return None

# =============================
# ORB v2 State per symbol
# =============================
class OrbState:
    def __init__(self):
        self.active = False
        self.trigger_high = 0.0
        self.trigger_low = 0.0
        self.entry_done = False
        self.sl = 0.0
        self.position_qty = 0.0
        self.avg_price = 0.0
        self.highest_since_entry = 0.0  # för frivillig 0.9/0.2 trailing

ORB: Dict[str, OrbState] = {s: OrbState() for s in DEFAULT_SYMBOLS}

def is_green(c): return c["close"] > c["open"]
def is_red(c):   return c["close"] < c["open"]

# =============================
# Trading (mock & live)
# =============================
def calc_qty_usdt(symbol: str, price: float) -> float:
    # enkel qty (spot): USDT-belopp / pris
    if price <= 0:
        return 0.0
    return round(MOCK_TRADE_USDT / price, 6)

def append_trade_csv(mode: str, symbol: str, side: str, qty: float, price: float, reason: str):
    is_new = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a") as f:
        if is_new:
            f.write("time_utc,mode,symbol,side,qty,price,reason\n")
        f.write(f"{now_iso()},{mode},{symbol},{side},{qty},{price},{reason}\n")

def place_market_buy(symbol: str, qty: float, context: Optional[CallbackContext]=None) -> bool:
    if STATE["mode"] == "mock":
        append_trade_csv("mock", symbol, "BUY", qty, fetch_price(symbol) or 0.0, "mock")
        return True
    # LIVE
    try:
        sym = symbol.replace("USDT", "-USDT")
        payload = {"clientOid": str(int(time.time()*1000)), "side":"buy", "symbol": sym, "type":"market", "funds": str(MOCK_TRADE_USDT)}
        kucoin_post("/api/v1/orders", payload)
        append_trade_csv("live", symbol, "BUY", qty, fetch_price(symbol) or 0.0, "live")
        return True
    except Exception as e:
        log(f"BUY live error {symbol}: {e}")
        notify(context, f"Live BUY error {symbol}: {e}")
        return False

def place_market_sell(symbol: str, qty: float, context: Optional[CallbackContext]=None) -> bool:
    if STATE["mode"] == "mock":
        append_trade_csv("mock", symbol, "SELL", qty, fetch_price(symbol) or 0.0, "mock")
        return True
    # LIVE
    try:
        sym = symbol.replace("USDT", "-USDT")
        payload = {"clientOid": str(int(time.time()*1000)), "side":"sell", "symbol": sym, "type":"market", "size": str(qty)}
        kucoin_post("/api/v1/orders", payload)
        append_trade_csv("live", symbol, "SELL", qty, fetch_price(symbol) or 0.0, "live")
        return True
    except Exception as e:
        log(f"SELL live error {symbol}: {e}")
        notify(context, f"Live SELL error {symbol}: {e}")
        return False

# =============================
# Strategy Core (per symbol)
# =============================
def maybe_new_orb(symbol: str, candles: List[Dict]):
    """
    Om senaste par (prev, curr) = röd->grön, sätt ny ORB på GRÖNA candlen.
    """
    if len(candles) < 2:
        return
    prev, curr = candles[-2], candles[-1]
    st = ORB[symbol]
    if is_red(prev) and is_green(curr) and not st.position_qty:
        st.active = True
        st.entry_done = False
        st.trigger_high = curr["high"]
        st.trigger_low = curr["low"]
        st.sl = st.trigger_low
        log(f"{symbol} ORB trigger: HIGH={st.trigger_high:.6f}, LOW={st.trigger_low:.6f}")

def maybe_enter(symbol: str, price: float, context: Optional[CallbackContext]):
    st = ORB[symbol]
    if st.active and not st.entry_done and price is not None and price > st.trigger_high:
        qty = calc_qty_usdt(symbol, price)
        if qty <= 0: 
            return
        ok = place_market_buy(symbol, qty, context)
        if ok:
            st.entry_done = True
            st.position_qty = qty
            st.avg_price = price
            st.highest_since_entry = price
            log(f"{symbol} BUY @ {price:.6f} qty {qty}")
            notify(context, f"BUY {symbol} @ {price:.6f} | SL {st.sl:.6f}")

def candle_trail_sl(symbol: str, last_closed: Dict, context: Optional[CallbackContext]):
    st = ORB[symbol]
    if st.position_qty > 0:
        new_sl = max(st.sl, float(last_closed["low"]))
        if new_sl > st.sl:
            st.sl = new_sl
            notify(context, f"{symbol} Trail SL -> {st.sl:.6f}")

def dynamic_trailing(symbol: str, price: float, context: Optional[CallbackContext]):
    """
    Frivillig 0.9/0.2% trailing: lås in vinst när vi nått +trigger%.
    """
    if not STATE["trailing_on"]:
        return
    st = ORB[symbol]
    if st.position_qty <= 0 or price is None:
        return
    # högsta sedan entry
    if price > st.highest_since_entry:
        st.highest_since_entry = price
    # procent mot entry
    gain_pct = (st.highest_since_entry / st.avg_price - 1.0) * 100.0
    if gain_pct >= STATE["trail_cfg"]["trigger"]:
        # nytt SL = highest * (1 - offset%)
        desired = st.highest_since_entry * (1.0 - STATE["trail_cfg"]["offset"]/100.0)
        # se till att minst min_lock uppnås
        min_price = st.avg_price * (1.0 + STATE["trail_cfg"]["min_lock"]/100.0)
        desired = max(desired, min_price)
        if desired > st.sl:
            st.sl = desired
            notify(context, f"{symbol} DynTrail SL -> {st.sl:.6f} (gain {gain_pct:.2f}%)")

def maybe_stop(symbol: str, price: float, context: Optional[CallbackContext]):
    st = ORB[symbol]
    if st.position_qty > 0 and price is not None and price <= st.sl:
        qty = st.position_qty
        ok = place_market_sell(symbol, qty, context)
        if ok:
            notify(context, f"SELL {symbol} @ {price:.6f} (SL hit {st.sl:.6f})")
        # Reset state (ny ORB kan fångas igen)
        ORB[symbol] = OrbState()

# =============================
# Engine Loop
# =============================
def engine_tick(context: Optional[CallbackContext]=None):
    if not STATE["engine"]:
        return
    for symbol in STATE["symbols"]:
        candles = fetch_last_n_candles(symbol, STATE["tf"], limit=3)
        if not candles:
            continue
        maybe_new_orb(symbol, candles)              # kolla röd->grön
        last_closed = candles[-1]
        candle_trail_sl(symbol, last_closed, context)  # trail SL till candle-low

        price = fetch_price(symbol)
        maybe_enter(symbol, price, context)         # entry när high bryts
        dynamic_trailing(symbol, price, context)    # valfri 0.9/0.2 trail
        maybe_stop(symbol, price, context)          # SL-exit

def engine_loop(updater: Updater):
    # kör var 2–4 sek beroende på TF (lätt vikt)
    while True:
        try:
            engine_tick(updater.dispatcher)
        except Exception as e:
            log(f"engine_loop error: {e}")
        time.sleep(2)

# =============================
# CSV Export (K4)
# =============================
def export_csv_common(chat_id: int, bot: Bot):
    if os.path.exists(TRADES_CSV):
        bot.send_document(chat_id=chat_id, document=open(TRADES_CSV, "rb"), filename="trades.csv")
    else:
        bot.send_message(chat_id=chat_id, text="Ingen trades.csv ännu.")

def export_k4_csv(chat_id: int, bot: Bot):
    """
    Enkel K4-rad per stängd affär (mock/live). Här antar vi roundtrip sell avslutar en position.
    För robust matchning krävs positionsjournal – detta är en enkel version.
    """
    if not os.path.exists(TRADES_CSV):
        bot.send_message(chat_id=chat_id, text="Inga affärer ännu.")
        return
    # Bygg enkla par (BUY->SELL) per symbol i tidsordning
    rows = []
    with open(TRADES_CSV) as f:
        next(f)  # skip header
        for line in f:
            t, mode, sym, side, qty, price, reason = line.strip().split(",")
            rows.append({"t": t, "mode": mode, "sym": sym, "side": side, "qty": float(qty), "price": float(price)})

    positions: Dict[str, List[Dict]] = {}
    realized = []
    for r in rows:
        if r["side"] == "BUY":
            positions.setdefault(r["sym"], []).append(r)
        elif r["side"] == "SELL":
            # matcha mot FIFO
            buys = positions.get(r["sym"], [])
            remain = r["qty"]
            while remain > 1e-9 and buys:
                b = buys[0]
                take = min(remain, b["qty"])
                buy_cost = take * b["price"]
                sell_val = take * r["price"]
                pnl = sell_val - buy_cost
                realized.append({
                    "date": r["t"],
                    "symbol": r["sym"],
                    "qty": round(take, 6),
                    "buy_price": b["price"],
                    "sell_price": r["price"],
                    "profit": pnl
                })
                b["qty"] -= take
                remain -= take
                if b["qty"] <= 1e-9:
                    buys.pop(0)
            positions[r["sym"]] = buys

    # Skriv K4-lik csv (mycket förenklad)
    with open(K4_CSV, "w") as f:
        f.write("date,symbol,amount,buy_price,sell_price,profit_sek\n")
        for rr in realized:
            f.write(f"{rr['date']},{rr['symbol']},{rr['qty']},{rr['buy_price']},{rr['sell_price']},{rr['profit']}\n")
    bot.send_document(chat_id=chat_id, document=open(K4_CSV, "rb"), filename="k4.csv")

# =============================
# Telegram Bot
# =============================
def kb_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Start MOCK", callback_data="start_mock"),
         InlineKeyboardButton("Start LIVE", callback_data="start_live")],
        [InlineKeyboardButton("Timeframe", callback_data="tf_menu"),
         InlineKeyboardButton("Trail on/off", callback_data="trail_toggle")],
        [InlineKeyboardButton("Keepalive ON", callback_data="keep_on"),
         InlineKeyboardButton("Keepalive OFF", callback_data="keep_off")],
        [InlineKeyboardButton("Export CSV", callback_data="export_csv"),
         InlineKeyboardButton("Export K4", callback_data="export_k4")],
        [InlineKeyboardButton("Engine START", callback_data="engine_start"),
         InlineKeyboardButton("Engine STOP", callback_data="engine_stop")]
    ])

def kb_tf():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1m", callback_data="tf_1m"),
         InlineKeyboardButton("3m", callback_data="tf_3m"),
         InlineKeyboardButton("5m", callback_data="tf_5m")]
    ])

def start(update: Update, context: CallbackContext):
    STATE["chat_id"] = update.effective_chat.id
    update.message.reply_text("Välkommen! Välj med knapparna nedan.", reply_markup=kB)

def help_cmd(update: Update, context: CallbackContext):
    text = (
        "/status – visa status\n"
        "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
        "/export_csv – skicka trades.csv\n"
        "/export_k4 – skicka k4.csv\n"
        "/keepalive_on | /keepalive_off\n"
        "/trail_on | /trail_off – frivillig 0.9/0.2 trailing\n"
    )
    update.message.reply_text(text, reply_markup=kb_menu())

def status_cmd(update: Update, context: CallbackContext):
    lines = [
        f"Mode: *{STATE['mode']}*    Engine: *{'ON' if STATE['engine'] else 'OFF'}*",
        f"TF: *{STATE['tf']}*",
        f"Symbols: {', '.join(STATE['symbols'])}",
        f"Trail: *{'ON' if STATE['trailing_on'] else 'OFF'}* "
        f"| trig {STATE['trail_cfg']['trigger']}% avst {STATE['trail_cfg']['offset']}% min {STATE['trail_cfg']['min_lock']}%",
        f"Keepalive: *{'ON' if STATE['keepalive'] else 'OFF'}*",
    ]
    for s in STATE["symbols"]:
        st = ORB[s]
        pos = "✅" if st.position_qty>0 else "❌"
        lines.append(f"{s}: pos={pos} trHigh={st.trigger_high or '-'} SL={f'{st.sl:.6f}' if st.sl else '-'}")
    update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

def symbols_cmd(update: Update, context: CallbackContext):
    if context.args:
        STATE["symbols"] = [x.strip().upper() for x in " ".join(context.args).split(",") if x.strip()]
        for s in STATE["symbols"]:
            ORB.setdefault(s, OrbState())
        update.message.reply_text(f"Nya symbols: {', '.join(STATE['symbols'])}")
    else:
        update.message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,ADAUSDT")

def export_csv_cmd(update: Update, context: CallbackContext):
    export_csv_common(update.effective_chat.id, context.bot)

def export_k4_cmd(update: Update, context: CallbackContext):
    export_k4_csv(update.effective_chat.id, context.bot)

def keepalive_on_cmd(update: Update, context: CallbackContext):
    STATE["keepalive"] = True
    update.message.reply_text("Keepalive: ON")

def keepalive_off_cmd(update: Update, context: CallbackContext):
    STATE["keepalive"] = False
    update.message.reply_text("Keepalive: OFF")

def trail_on_cmd(update: Update, context: CallbackContext):
    STATE["trailing_on"] = True
    update.message.reply_text("Dynamic trailing: ON")

def trail_off_cmd(update: Update, context: CallbackContext):
    STATE["trailing_on"] = False
    update.message.reply_text("Dynamic trailing: OFF")

def button_cb(update: Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data

    if data == "start_mock":
        STATE["mode"] = "mock"
        q.edit_message_text("Mode: MOCK", reply_markup=kb_menu())
    elif data == "start_live":
        STATE["mode"] = "live"
        q.edit_message_text("Mode: LIVE", reply_markup=kb_menu())
    elif data == "tf_menu":
        q.edit_message_text("Välj timeframe:", reply_markup=kb_tf())
    elif data.startswith("tf_"):
        STATE["tf"] = data.split("_",1)[1]
        q.edit_message_text(f"TF satt till {STATE['tf']}", reply_markup=kb_menu())
    elif data == "trail_toggle":
        STATE["trailing_on"] = not STATE["trailing_on"]
        q.edit_message_text(f"Trail: {'ON' if STATE['trailing_on'] else 'OFF'}", reply_markup=kb_menu())
    elif data == "keep_on":
        STATE["keepalive"] = True
        q.edit_message_text("Keepalive: ON", reply_markup=kb_menu())
    elif data == "keep_off":
        STATE["keepalive"] = False
        q.edit_message_text("Keepalive: OFF", reply_markup=kb_menu())
    elif data == "export_csv":
        export_csv_common(update.effective_chat.id, context.bot)
    elif data == "export_k4":
        export_k4_csv(update.effective_chat.id, context.bot)
    elif data == "engine_start":
        STATE["engine"] = True
        q.edit_message_text("Engine: ON", reply_markup=kb_menu())
    elif data == "engine_stop":
        STATE["engine"] = False
        q.edit_message_text("Engine: OFF", reply_markup=kb_menu())

# =============================
# Keepalive (ping Render)
# =============================
def keepalive_loop():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    while True:
        try:
            if STATE["keepalive"]:
                requests.get(url + "/health", timeout=5)
        except Exception:
            pass
        time.sleep(KEEPALIVE_EVERY_SEC)

# =============================
# FastAPI App
# =============================
app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "msg": "Mporbbot up"}

@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}

# =============================
# Bootstrap
# =============================
kB = kb_menu()

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN saknas.")
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # commands
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(CommandHandler("status", status_cmd))
    dp.add_handler(CommandHandler("symbols", symbols_cmd))
    dp.add_handler(CommandHandler("export_csv", export_csv_cmd))
    dp.add_handler(CommandHandler("export_k4", export_k4_cmd))
    dp.add_handler(CommandHandler("keepalive_on", keepalive_on_cmd))
    dp.add_handler(CommandHandler("keepalive_off", keepalive_off_cmd))
    dp.add_handler(CommandHandler("trail_on", trail_on_cmd))
    dp.add_handler(CommandHandler("trail_off", trail_off_cmd))

    # buttons
    dp.add_handler(CallbackQueryHandler(button_cb))

    # start threads
    threading.Thread(target=engine_loop, args=(updater,), daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()

    updater.start_polling(drop_pending_updates=True)
    log("Telegram bot started (polling).")

# Start bot automatically when uvicorn imports main:app
bot_thread_started = False
if not bot_thread_started:
    try:
        threading.Thread(target=main, daemon=True).start()
        bot_thread_started = True
    except Exception as e:
        log(f"Failed to start Telegram thread: {e}")
