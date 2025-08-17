import os
import time
import json
import math
import threading
from datetime import datetime, timezone, date
from typing import Dict, List, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# === Telegram (python-telegram-bot 13.15) ===
from telegram import Update, ParseMode
from telegram.ext import (
    Updater, CallbackContext, CommandHandler, Filters,
)

# -------------------------------
# FastAPI app for Render health
# -------------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "ts": time.time()}

@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}

# -------------------------------
# Global configuration & state
# -------------------------------
BINANCE_BASE = "https://api.binance.com"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
VALID_TF = {"1m": "1m", "3m": "3m", "5m": "5m"}
DEFAULT_TF = "1m"

KEEPALIVE_URL = os.getenv("KEEPALIVE_URL", "").strip()
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "120"))

state = {
    "mode": "mock",                 # "mock" / "live"
    "engine_on": True,              # motor kör
    "symbols": DEFAULT_SYMBOLS.copy(),
    "timeframe": DEFAULT_TF,        # 1m/3m/5m
    "entry_mode": "close",          # "tick" / "close" (vi kör på close)
    # Trailing-konfig
    "trail_enabled": True,
    "trail_trigger": 0.009,         # 0.90%
    "trail_distance": 0.002,        # 0.20%
    "trail_min_lock": 0.007,        # 0.70%
    # ORB master + vilka symboler som har ORB aktiv
    "orb_master": True,             # auto-ON
    "orb_symbols": set(DEFAULT_SYMBOLS),
    # Keepalive till Render
    "keepalive": True,
    # PnL & positions
    "day_pnl": 0.0,                 # summerad dags-PnL i USDT (mock)
}

# Per-symbol data
positions: Dict[str, Dict] = {}        # symbol -> {"entry": float, "qty": float, "sl": float, "trail_on": bool, "max_up": float}
orb_state: Dict[str, Dict] = {}         # symbol -> {"armed": bool, "high": float, "low": float, "last_candle_ts": int}
last_prices: Dict[str, float] = {}      # symbol -> float
pnl_log: List[Dict] = []                # för export_csv/k4

# Lås för trådar
lock = threading.RLock()

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# -------------------------------
# Helpers: exchange data (Binance)
# -------------------------------
def binance_klines(symbol: str, interval: str, limit: int = 3):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def get_latest_candles(symbol: str, interval: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Returnera [(open_time, open, high, low, close), ...] (floatar)
    """
    raw = binance_klines(symbol, interval, limit=3)
    out = []
    for k in raw:
        open_time = int(k[0])  # ms
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
        out.append((open_time, o, h, l, c))
    return out

def get_last_price(symbol: str) -> float:
    # hämtar senaste pris (ticker price)
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    r = requests.get(url, params={"symbol": symbol}, timeout=8)
    r.raise_for_status()
    price = float(r.json()["price"])
    last_prices[symbol] = price
    return price

# -------------------------------
# ORB logic
# -------------------------------
def update_orb(symbol: str, interval: str):
    """
    Hålla koll på första gröna efter röd:
      - när vi ser en röd candle följd av en grön -> ORB 'armed' och range sätts till den gröna candlens high/low
      - entry triggas när close > orb_high
      - initial SL = orb_low
      - SL följer med upp candle-för-candle (senaste gröna candlens low, men aldrig nedåt)
    """
    try:
        candles = get_latest_candles(symbol, interval)
    except Exception:
        return

    if len(candles) < 2:
        return

    # Senaste två färdiga candles (den sista i listan kan ibland vara ofärdig)
    # Vi tar näst sista som "senaste färdiga close"
    (t1, o1, h1, l1, c1) = candles[-2]
    (t2, o2, h2, l2, c2) = candles[-1]

    prev_red = c1 < o1
    last_green = c2 > o2

    with lock:
        s = orb_state.setdefault(symbol, {"armed": False, "high": None, "low": None, "last_candle_ts": 0})

        # Ny färdig candle?
        if t2 != s.get("last_candle_ts"):
            # Om vi får mönstret röd -> grön => beväpna ORB med den gröna candlens high/low
            if prev_red and last_green:
                s["armed"] = True
                s["high"] = h2
                s["low"] = l2
            # Om sista candle är grön och ORB redan är beväpnad => uppdatera SL-range uppåt (följa med uppåt)
            if last_green and s.get("armed") and s.get("low") is not None:
                # följ med uppåt, aldrig nedåt
                s["low"] = max(s["low"], l2)
                s["high"] = max(s["high"], h2)
            s["last_candle_ts"] = t2

        # Entry-villkor: close (senaste färdiga = c2) bryter upp över ORB-high
        if s.get("armed") and s.get("high") is not None:
            if c2 > s["high"]:
                # Starta/addera position endast om vi inte redan är inne
                if symbol not in positions:
                    enter_long(symbol, entry_price=c2, initial_sl=s["low"])
                # Efter entry, nollställ/beväpna om du vill undvika fler entry innan ny setup
                s["armed"] = False

def enter_long(symbol: str, entry_price: float, initial_sl: float):
    qty = 1.0  # mock - 1 kontrakt/enhet
    positions[symbol] = {
        "entry": entry_price,
        "qty": qty,
        "sl": initial_sl,
        "trail_on": False,
        "max_up": entry_price
    }
    log_trade(symbol, "BUY", entry_price, pnl=0.0)
    notify(f"{symbol}: ORB breakout BUY @ {fmt(entry_price)} | SL {fmt(initial_sl)}")

def fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.4f}"

def log_trade(symbol: str, side: str, price: float, pnl: float):
    pnl_log.append({
        "ts": now_iso(),
        "symbol": symbol,
        "side": side,
        "price": price,
        "pnl": pnl,
        "mode": state["mode"]
    })

def close_position(symbol: str, price: float, reason: str):
    pos = positions.get(symbol)
    if not pos:
        return
    entry = pos["entry"]; qty = pos["qty"]
    pnl = (price - entry) * qty
    state["day_pnl"] += pnl
    log_trade(symbol, "SELL", price, pnl)
    notify(f"{symbol}: EXIT @ {fmt(price)} | PnL {pnl:+.4f} USDT | reason: {reason} | day {state['day_pnl']:+.4f}")
    del positions[symbol]

# -------------------------------
# Trailing stop / SL uppdatering
# -------------------------------
def manage_stops(symbol: str):
    """
    - SL följer med upp när vi får nya högre lows från uppföljande gröna candles (hanteras i update_orb via orb_state.low)
    - Trailing trigger: +0.90% från entry => slå på trailing, håll avstånd 0.20% från senaste high/max_up
      med minimum-lås 0.70% vinst.
    """
    pos = positions.get(symbol)
    if not pos:
        return

    try:
        price = get_last_price(symbol)
    except Exception:
        return

    entry = pos["entry"]
    # uppdatera max_up
    if price > pos["max_up"]:
        pos["max_up"] = price

    # trail-setup?
    if state["trail_enabled"]:
        gain = (price / entry) - 1.0
        # slå på trailing om +0.90% uppnåtts
        if (not pos["trail_on"]) and gain >= state["trail_trigger"]:
            pos["trail_on"] = True

        if pos["trail_on"]:
            # nytt teoretiskt SL från trail
            trail_sl = pos["max_up"] * (1.0 - state["trail_distance"])
            # säkerställ minst 0.70% lås på SL när den väl aktiverats (om möjligt)
            min_lock_sl = entry * (1.0 + state["trail_min_lock"])
            new_sl = max(trail_sl, min_lock_sl)
            # kombinera med ORB-SL (orb_state låg följer uppåt)
            orb_low = orb_state.get(symbol, {}).get("low")
            if orb_low is not None:
                new_sl = max(new_sl, orb_low)
            # uppdatera SL bara uppåt
            if new_sl > pos["sl"]:
                pos["sl"] = new_sl

    # SL-utlösning?
    if pos["sl"] is not None and price <= pos["sl"]:
        close_position(symbol, price, reason="SL")

# -------------------------------
# Engine loop (per symbol)
# -------------------------------
def engine_symbol(symbol: str):
    while True:
        try:
            with lock:
                running = state["engine_on"]
                mode = state["mode"]
                tf = state["timeframe"]
                use_orb = state["orb_master"] and (symbol in state["orb_symbols"])
            if not running:
                time.sleep(1.0)
                continue

            # ORB endast när aktiverad
            if use_orb:
                update_orb(symbol, tf)

            # Hantera stops / trailing (om position finns)
            manage_stops(symbol)

        except Exception as e:
            # bara logga tyst
            pass
        finally:
            # 1s cykel i 1m-läge är lagom – ORB triggar på candle close
            time.sleep(1.0)

# -------------------------------
# Keepalive (Render)
# -------------------------------
def keepalive_loop():
    while True:
        try:
            with lock:
                if not state["keepalive"]:
                    time.sleep(KEEPALIVE_INTERVAL)
                    continue
                url = KEEPALIVE_URL
            if url:
                requests.get(url, timeout=5)
        except Exception:
            pass
        finally:
            time.sleep(KEEPALIVE_INTERVAL)

# -------------------------------
# Telegram helpers
# -------------------------------
def notify(text: str):
    # skickas av Updater-botten – fylls i init (global)
    try:
        if updater is not None:
            updater.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception:
        pass

def render_status() -> str:
    with lock:
        lines = []
        lines.append(f"Mode: {state['mode']}   Engine: {'ON' if state['engine_on'] else 'OFF'}")
        lines.append(f"TF: {state['timeframe']}   Symbols: {','.join(state['symbols'])}")
        lines.append(f"Entry: {state['entry_mode']}   Trail: {'ON' if state['trail_enabled'] else 'OFF'} ({state['trail_trigger']*100:.2f}%/{state['trail_distance']*100:.2f}% min {state['trail_min_lock']*100:.2f}%)")
        lines.append(f"Keepalive: {'ON' if state['keepalive'] else 'OFF'}   DayPnL: {state['day_pnl']:.4f} USDT")
        lines.append(f"ORB master: {'ON' if state['orb_master'] else 'OFF'}")
        for sym in state["symbols"]:
            pos = "✅" if sym in positions else "❌"
            stop = positions[sym]["sl"] if sym in positions else None
            orbflag = "ON" if sym in state["orb_symbols"] else "OFF"
            lines.append(f"{sym}: pos={pos} stop={fmt(stop)} | ORB: {orbflag}")
        return "\n".join(lines)

# -------------------------------
# Telegram command handlers
# -------------------------------
def require_chat(update: Update) -> int:
    # return the chat id; first private chat wins
    return update.effective_chat.id

def cmd_help(update: Update, context: CallbackContext):
    txt = (
        "/status\n"
        "/engine_start /engine_stop\n"
        "/start_mock /start_live\n"
        "/symbols BTCUSDT/ETHUSDT/ADAUSDT\n"
        "/timeframe\n"
        "/entry_mode tick|close\n"
        "/trailing\n"
        "/pnl /reset_pnl /export_csv /export_k4\n"
        "/keepalive_on /keepalive_off\n"
        "/panic\n"
    )
    update.message.reply_text(txt)

def cmd_status(update: Update, context: CallbackContext):
    update.message.reply_text(render_status())

def cmd_engine_start(update: Update, context: CallbackContext):
    with lock:
        state["engine_on"] = True
        # synka ORB till alla valda symboler om master är ON
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("Engine: ON")

def cmd_engine_stop(update: Update, context: CallbackContext):
    with lock:
        state["engine_on"] = False
    update.message.reply_text("Engine: OFF")

def cmd_start_mock(update: Update, context: CallbackContext):
    with lock:
        state["mode"] = "mock"
        state["engine_on"] = True
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("Mode: MOCK")

def cmd_start_live(update: Update, context: CallbackContext):
    # placeholder – behåll samma motor men markera "live"
    with lock:
        state["mode"] = "live"
        state["engine_on"] = True
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("Mode: LIVE (demo)")

def cmd_symbols(update: Update, context: CallbackContext):
    # /symbols BTCUSDT/ETHUSDT/ADAUSDT
    try:
        text = update.message.text.strip()
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            update.message.reply_text(f"Aktuella symboler: {','.join(state['symbols'])}")
            return
        raw = parts[1].replace(",", "/").upper()
        syms = [s for s in raw.split("/") if s]
        if not syms:
            raise ValueError
        with lock:
            state["symbols"] = syms
            if state["orb_master"]:
                state["orb_symbols"] = set(state["symbols"])
        update.message.reply_text(f"Symboler satta till: {','.join(syms)}")
    except Exception:
        update.message.reply_text("Använd: /symbols BTCUSDT/ETHUSDT/ADAUSDT")

def cmd_timeframe(update: Update, context: CallbackContext):
    # /timeframe 1m|3m|5m  (utan arg visar nuvarande)
    parts = update.message.text.strip().split()
    if len(parts) == 1:
        update.message.reply_text(f"Tidsram satt till {state['timeframe']}")
        return
    tf = parts[1].lower()
    if tf not in VALID_TF:
        update.message.reply_text("Använd: /timeframe 1m|3m|5m")
        return
    with lock:
        state["timeframe"] = tf
    update.message.reply_text(f"Tidsram satt till {tf}")

def cmd_entry_mode(update: Update, context: CallbackContext):
    parts = update.message.text.strip().split()
    if len(parts) == 1:
        update.message.reply_text(f"Entry mode: {state['entry_mode']}")
        return
    mode = parts[1].lower()
    if mode not in ("tick", "close"):
        update.message.reply_text("Använd: /entry_mode tick|close")
        return
    with lock:
        state["entry_mode"] = mode
    update.message.reply_text(f"Entry mode: {mode}")

def cmd_trailing(update: Update, context: CallbackContext):
    # /trailing 0.9 0.2 0.7 (procent) – valfritt, utan args visar nuvarande
    parts = update.message.text.strip().split()
    if len(parts) == 1:
        with lock:
            msg = f"Trail: {'ON' if state['trail_enabled'] else 'OFF'} ({state['trail_trigger']*100:.2f}%/{state['trail_distance']*100:.2f}% min {state['trail_min_lock']*100:.2f}%)"
        update.message.reply_text(msg)
        return
    try:
        trig = float(parts[1]) / 100.0
        dist = float(parts[2]) / 100.0
        minl = float(parts[3]) / 100.0
        with lock:
            state["trail_enabled"] = True
            state["trail_trigger"] = trig
            state["trail_distance"] = dist
            state["trail_min_lock"] = minl
        update.message.reply_text(f"Trail uppdaterad: trig {parts[1]}% / avst {parts[2]}% / min {parts[3]}%")
    except Exception:
        update.message.reply_text("Använd: /trailing <trig%> <avst%> <min%>  t.ex. /trailing 0.9 0.2 0.7")

def cmd_pnl(update: Update, context: CallbackContext):
    with lock:
        update.message.reply_text(f"Dagens PnL: {state['day_pnl']:.4f} USDT")

def cmd_reset_pnl(update: Update, context: CallbackContext):
    with lock:
        state["day_pnl"] = 0.0
    update.message.reply_text("Dagens PnL nollställd.")

def _export_csv_rows() -> List[str]:
    rows = ["ts,symbol,side,price,pnl,mode"]
    for r in pnl_log:
        rows.append(f"{r['ts']},{r['symbol']},{r['side']},{r['price']:.6f},{r['pnl']:.6f},{r['mode']}")
    return rows

def cmd_export_csv(update: Update, context: CallbackContext):
    rows = _export_csv_rows()
    text = "\n".join(rows)
    update.message.reply_document(document=bytes(text, "utf-8"), filename=f"trades_{date.today().isoformat()}.csv")

def cmd_export_k4(update: Update, context: CallbackContext):
    """
    Enkel K4-liknande export (ej officiellt format, men rad-för-rad: datum,symbol,köp/sälj,belopp)
    """
    rows = ["datum,symbol,typ,belopp"]
    for r in pnl_log:
        typ = "Försäljning" if r["side"] == "SELL" else "Köp"
        belopp = f"{r['price']:.6f}"
        datum = r["ts"][:10]
        rows.append(f"{datum},{r['symbol']},{typ},{belopp}")
    text = "\n".join(rows)
    update.message.reply_document(document=bytes(text, "utf-8"), filename=f"k4_{date.today().isoformat()}.csv")

def cmd_keepalive_on(update: Update, context: CallbackContext):
    with lock:
        state["keepalive"] = True
    update.message.reply_text("Keepalive: ON")

def cmd_keepalive_off(update: Update, context: CallbackContext):
    with lock:
        state["keepalive"] = False
    update.message.reply_text("Keepalive: OFF")

def cmd_panic(update: Update, context: CallbackContext):
    # stäng alla positioner på senaste pris
    with lock:
        syms = list(positions.keys())
    for sym in syms:
        try:
            price = get_last_price(sym)
        except Exception:
            continue
        close_position(sym, price, "PANIC")
    update.message.reply_text("Panic: alla positioner stängda.")

# -------------------------------
# ORB kommandon (master)
# -------------------------------
def cmd_orb_on(update: Update, context: CallbackContext):
    with lock:
        state["orb_master"] = True
        state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("ORB: ON (för alla valda symboler)")

def cmd_orb_off(update: Update, context: CallbackContext):
    with lock:
        state["orb_master"] = False
        state["orb_symbols"].clear()
    update.message.reply_text("ORB: OFF (för alla)")

# -------------------------------
# Boot / Telegram init
# -------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Sätt TELEGRAM_TOKEN i miljövariablerna.")

updater: Optional[Updater] = None
CHAT_ID: Optional[int] = None   # fylls när första /status kommer i privat chat

def set_chat_id(update: Update):
    global CHAT_ID
    cid = require_chat(update)
    if CHAT_ID is None:
        CHAT_ID = cid

def start_telegram():
    global updater
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("status", lambda u, c: (set_chat_id(u), cmd_status(u, c))))
    dp.add_handler(CommandHandler("engine_start", cmd_engine_start))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(CommandHandler("symbols", cmd_symbols))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe))
    dp.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    dp.add_handler(CommandHandler("trailing", cmd_trailing))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("export_csv", cmd_export_csv))
    dp.add_handler(CommandHandler("export_k4", cmd_export_k4))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dp.add_handler(CommandHandler("panic", cmd_panic))
    # ORB master
    dp.add_handler(CommandHandler("orb_on", cmd_orb_on))
    dp.add_handler(CommandHandler("orb_off", cmd_orb_off))

    updater.start_polling(drop_pending_updates=True)

# -------------------------------
# Threads start
# -------------------------------
def start_threads():
    # motortrådar per symbol
    for sym in state["symbols"]:
        th = threading.Thread(target=engine_symbol, args=(sym,), daemon=True)
        th.start()
    # keepalive
    thk = threading.Thread(target=keepalive_loop, daemon=True)
    thk.start()

# -------------------------------
# Uvicorn entry for Render
# -------------------------------
def bootstrap_once():
    # ORB auto-on på start
    with lock:
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])

    # starta Telegram + motortrådar
    start_telegram()
    start_threads()

# Kör bootstrap när processen startar
bootstrap_once()
