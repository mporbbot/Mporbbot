# -*- coding: utf-8 -*-
import os
import io
import csv
import time
import threading
from datetime import datetime, timezone, date
from typing import Dict, List, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# Telegram (python-telegram-bot 13.15)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater, CallbackContext, CommandHandler, CallbackQueryHandler,
)
from telegram.error import Conflict, NetworkError

# =========================
# FastAPI (Render health)
# =========================
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "mporbbot OK"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "healthy"

# =========================
# Konfiguration & State
# =========================
BINANCE_BASE = "https://api.binance.com"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]
VALID_TF = {"1m", "3m", "5m"}
DEFAULT_TF = "1m"

KEEPALIVE_URL = os.getenv("KEEPALIVE_URL", "").strip()
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "120"))
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "10"))

state = {
    "mode": "mock",                 # "mock" | "live"
    "engine_on": True,              # motor går
    "symbols": DEFAULT_SYMBOLS.copy(),
    "timeframe": DEFAULT_TF,        # 1m/3m/5m
    "entry_mode": "close",          # "close" (default) | "tick"
    # trailing
    "trail_enabled": True,
    "trail_trigger": 0.009,         # +0.90%
    "trail_distance": 0.002,        # 0.20%
    "trail_min_lock": 0.007,        # minst +0.70%
    # ORB
    "orb_master": True,             # auto-ON
    "orb_symbols": set(DEFAULT_SYMBOLS),
    # Keepalive
    "keepalive": True,
    # PnL
    "day_pnl": 0.0,
}

# Per-symbol containers
positions: Dict[str, Dict] = {}        # symbol -> {"entry","qty","sl","trail_on","max_up"}
orb_state: Dict[str, Dict] = {}         # symbol -> {"locked","high","low","last_candle_ts","entry_used"}
last_prices: Dict[str, float] = {}      # symbol -> last price
pnl_log: List[Dict] = []                # för export

lock = threading.RLock()

# Telegram globals
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Sätt TELEGRAM_TOKEN i miljövariablerna.")
updater: Optional[Updater] = None
CHAT_ID: Optional[int] = None

# =========================
# Hjälpare
# =========================
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.6f}"

def set_chat_id(update: Update):
    global CHAT_ID
    cid = update.effective_chat.id
    if CHAT_ID is None:
        CHAT_ID = cid

def notify(text: str):
    try:
        if updater is not None and CHAT_ID is not None:
            updater.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception:
        pass

def log_trade(symbol: str, side: str, price: float, pnl: float):
    pnl_log.append({
        "ts": now_iso(),
        "symbol": symbol,
        "side": side,
        "price": price,
        "pnl": pnl,
        "mode": state["mode"],
    })

# =========================
# Binance helpers
# =========================
def binance_klines(symbol: str, interval: str, limit: int = 3):
    url = f"{BINANCE_BASE}/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json()

def get_latest_candles(symbol: str, interval: str) -> List[Tuple[int, float, float, float, float]]:
    raw = binance_klines(symbol, interval, limit=3)
    out = []
    for k in raw:
        open_time = int(k[0])
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
        out.append((open_time, o, h, l, c))
    return out

def get_last_price(symbol: str) -> Optional[float]:
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    try:
        r = requests.get(url, params={"symbol": symbol}, timeout=8)
        r.raise_for_status()
        price = float(r.json()["price"])
        last_prices[symbol] = price
        return price
    except Exception:
        return None

# =========================
# ORB & Trading
# =========================
def enter_long(symbol: str, entry_price: float, initial_sl: float):
    if symbol in positions:
        return
    qty = 0.0
    if entry_price > 0:
        qty = round(MOCK_TRADE_USDT / entry_price, 6)
    positions[symbol] = {
        "entry": entry_price,
        "qty": qty,
        "sl": initial_sl,
        "trail_on": False,
        "max_up": entry_price,
        "last_low_ts": 0,
    }
    log_trade(symbol, "BUY", entry_price, 0.0)
    notify(f"{symbol}: ORB breakout BUY @ {fmt(entry_price)} | SL {fmt(initial_sl)} (qty {qty})")

def close_position(symbol: str, price: float, reason: str):
    pos = positions.get(symbol)
    if not pos:
        return
    entry = pos["entry"]; qty = pos["qty"]
    pnl = (price - entry) * qty
    with lock:
        state["day_pnl"] += pnl
    log_trade(symbol, "SELL", price, pnl)
    notify(f"{symbol}: EXIT @ {fmt(price)} | PnL {pnl:+.4f} USDT | reason: {reason} | day {state['day_pnl']:+.4f}")
    del positions[symbol]

def lock_orb(symbol: str, candles: List[Tuple[int,float,float,float,float]]):
    """
    Lås ORB på första GRÖNA stängda efter en RÖD stängd.
    ORB ändras inte förrän en RÖD candle stänger igen.
    """
    if len(candles) < 3:
        return
    (t0, o0, h0, l0, c0) = candles[-3]  # föregående stängd
    (t1, o1, h1, l1, c1) = candles[-2]  # senaste stängda
    (t2, o2, h2, l2, c2) = candles[-1]  # pågående

    s = orb_state.setdefault(symbol, {"locked": False, "high": None, "low": None, "last_candle_ts": 0, "entry_used": False})

    # Ny stängd candle?
    if t1 != s["last_candle_ts"]:
        # Om ny röd stängde → lås upp för nästa gröna
        if c1 < o1:
            s["locked"] = False
            s["high"] = None
            s["low"] = None
            s["entry_used"] = False

        # Om mönster röd -> grön → lås ORB på den gröna
        if c0 < o0 and c1 > o1:
            s["locked"] = True
            s["high"] = h1
            s["low"] = l1
            s["entry_used"] = False

        s["last_candle_ts"] = t1

def try_entry(symbol: str, candles: List[Tuple[int,float,float,float,float]]):
    s = orb_state.get(symbol, {})
    if not s or not s.get("locked") or s.get("high") is None or s.get("entry_used"):
        return
    orb_high = s["high"]; orb_low = s["low"]

    if state["entry_mode"] == "close":
        # köp på senaste STÄNGDA candle som stängt över ORB-high
        if len(candles) < 2:
            return
        (_, o1, h1, l1, c1) = candles[-2]
        if c1 > orb_high:
            enter_long(symbol, c1, orb_low)
            s["entry_used"] = True
    else:
        # tick: köp om live bryter ORB-high (eller current candle's high)
        tick = get_last_price(symbol)
        (_, o_cur, h_cur, l_cur, c_cur) = candles[-1]
        if (tick and tick > orb_high) or (h_cur > orb_high):
            entry_px = tick if (tick and tick > orb_high) else max(orb_high, c_cur)
            enter_long(symbol, entry_px, orb_low)
            s["entry_used"] = True

def step_trailing(symbol: str, candles: List[Tuple[int,float,float,float,float]]):
    pos = positions.get(symbol)
    if not pos:
        return

    # Höj SL till senaste stängda candle's low om högre
    if len(candles) >= 2:
        (t1, o1, h1, l1, c1) = candles[-2]
        if t1 != pos["last_low_ts"]:
            if pos["sl"] is None or l1 > pos["sl"]:
                pos["sl"] = l1
            pos["last_low_ts"] = t1

    # Extra trailing efter +0.9% (0.2% avstånd, min +0.7%)
    tick = get_last_price(symbol)
    if tick is None:
        return

    if tick > pos["max_up"]:
        pos["max_up"] = tick

    if state["trail_enabled"]:
        entry = pos["entry"]
        gain = (tick / entry) - 1.0
        if (not pos["trail_on"]) and gain >= state["trail_trigger"]:
            pos["trail_on"] = True
        if pos["trail_on"]:
            trail_sl = pos["max_up"] * (1.0 - state["trail_distance"])
            min_lock_sl = entry * (1.0 + state["trail_min_lock"])
            new_sl = max(pos["sl"] or -1e9, trail_sl, min_lock_sl)
            pos["sl"] = new_sl

    # SL-exit?
    if pos["sl"] is not None and tick <= pos["sl"]:
        close_position(symbol, tick, reason="SL")

# =========================
# Engine loops
# =========================
def engine_symbol(symbol: str):
    while True:
        try:
            with lock:
                running = state["engine_on"]
                tf = state["timeframe"]
                use_orb = state["orb_master"] and (symbol in state["orb_symbols"])
            if not running:
                time.sleep(1.0); continue

            candles = get_latest_candles(symbol, tf)

            if use_orb:
                lock_orb(symbol, candles)
                try_entry(symbol, candles)

            step_trailing(symbol, candles)

        except Exception:
            pass
        finally:
            time.sleep(1.0)

def keepalive_loop():
    while True:
        try:
            with lock:
                if not state["keepalive"]:
                    time.sleep(KEEPALIVE_INTERVAL); continue
                url = KEEPALIVE_URL
            if url:
                requests.get(url, timeout=5)
        except Exception:
            pass
        finally:
            time.sleep(KEEPALIVE_INTERVAL)

# =========================
# Telegram handlers
# =========================
def render_status() -> str:
    with lock:
        lines = []
        lines.append(f"Mode: {state['mode']}   Engine: {'ON' if state['engine_on'] else 'OFF'}")
        lines.append(f"TF: {state['timeframe']}   Symbols: {','.join(state['symbols'])}")
        lines.append(
            f"Entry: {state['entry_mode'].upper()}   "
            f"Trail: {'ON' if state['trail_enabled'] else 'OFF'} "
            f"({state['trail_trigger']*100:.2f}%/{state['trail_distance']*100:.2f}% min {state['trail_min_lock']*100:.2f}%)"
        )
        lines.append(f"Keepalive: {'ON' if state['keepalive'] else 'OFF'}   DayPnL: {state['day_pnl']:.4f} USDT")
        lines.append(f"ORB master: {'ON' if state['orb_master'] else 'OFF'}")
        for sym in state["symbols"]:
            pos = "✅" if sym in positions else "❌"
            stop = positions[sym]["sl"] if sym in positions else None
            orbflag = "ON" if sym in state["orb_symbols"] else "OFF"
            lines.append(f"{sym}: pos={pos} stop={fmt(stop)} | ORB: {orbflag}")
        return "\n".join(lines)

def kb_entry_mode():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📉 Close", callback_data="entry_close"),
        InlineKeyboardButton("⚡ Tick", callback_data="entry_tick"),
    ]])

def kb_timeframe():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("TF 1m", callback_data="tf_1m"),
        InlineKeyboardButton("TF 3m", callback_data="tf_3m"),
        InlineKeyboardButton("TF 5m", callback_data="tf_5m"),
    ]])

def cmd_help(update: Update, _: CallbackContext):
    set_chat_id(update)
    txt = (
        "/status\n"
        "/engine_start /engine_stop\n"
        "/start_mock /start_live\n"
        "/symbols BTCUSDT/ETHUSDT/ADAUSDT/LINKUSDT/XRPUSDT\n"
        "/timeframe  (öppnar knappar)\n"
        "/entry_mode (öppnar knappar)\n"
        "/trailing <trig%> <avst%> <min%>  ex /trailing 0.9 0.2 0.7\n"
        "/pnl /reset_pnl /export_csv /export_k4\n"
        "/keepalive_on /keepalive_off\n"
        "/orb_on /orb_off\n"
        "/panic\n"
    )
    update.message.reply_text(txt)

def cmd_status(update: Update, _: CallbackContext):
    set_chat_id(update)
    update.message.reply_text(render_status())

def cmd_engine_start(update: Update, _: CallbackContext):
    with lock:
        state["engine_on"] = True
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("Engine: ON")

def cmd_engine_stop(update: Update, _: CallbackContext):
    with lock:
        state["engine_on"] = False
    update.message.reply_text("Engine: OFF")

def cmd_start_mock(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        state["mode"] = "mock"
        state["engine_on"] = True
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("Mode: MOCK")

def cmd_start_live(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        state["mode"] = "live"
        state["engine_on"] = True
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("Mode: LIVE (demo)")

def cmd_symbols(update: Update, _: CallbackContext):
    set_chat_id(update)
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
        update.message.reply_text("Använd: /symbols BTCUSDT/ETHUSDT/ADAUSDT/LINKUSDT/XRPUSDT")

def cmd_timeframe(update: Update, _: CallbackContext):
    set_chat_id(update)
    update.message.reply_text("Välj timeframe:", reply_markup=kb_timeframe())

def cmd_entry_mode(update: Update, _: CallbackContext):
    set_chat_id(update)
    update.message.reply_text("Välj entry-mode:", reply_markup=kb_entry_mode())

def on_button(update: Update, _: CallbackContext):
    q = update.callback_query
    if not q:
        return
    data = q.data
    if data == "entry_close":
        with lock:
            state["entry_mode"] = "close"
        q.edit_message_text("✅ Entry mode set to: CLOSE")
    elif data == "entry_tick":
        with lock:
            state["entry_mode"] = "tick"
        q.edit_message_text("✅ Entry mode set to: TICK")
    elif data == "tf_1m":
        with lock:
            state["timeframe"] = "1m"
        q.edit_message_text("Timeframe satt till 1m")
    elif data == "tf_3m":
        with lock:
            state["timeframe"] = "3m"
        q.edit_message_text("Timeframe satt till 3m")
    elif data == "tf_5m":
        with lock:
            state["timeframe"] = "5m"
        q.edit_message_text("Timeframe satt till 5m")
    else:
        q.answer()

def cmd_trailing(update: Update, _: CallbackContext):
    set_chat_id(update)
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

def cmd_pnl(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        update.message.reply_text(f"Dagens PnL: {state['day_pnl']:.4f} USDT")

def cmd_reset_pnl(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        state["day_pnl"] = 0.0
    update.message.reply_text("Dagens PnL nollställd.")

def _export_csv_rows() -> List[str]:
    rows = ["ts,symbol,side,price,pnl,mode"]
    for r in pnl_log:
        rows.append(f"{r['ts']},{r['symbol']},{r['side']},{r['price']:.6f},{r['pnl']:.6f},{r['mode']}")
    return rows

def cmd_export_csv(update: Update, _: CallbackContext):
    set_chat_id(update)
    rows = _export_csv_rows()
    text = "\n".join(rows)
    update.message.reply_document(document=bytes(text, "utf-8"), filename=f"trades_{date.today().isoformat()}.csv")

def cmd_export_k4(update: Update, _: CallbackContext):
    set_chat_id(update)
    rows = ["Datum;Typ;Värdepapper;Antal;Försäljningspris;Omkostnadsbelopp;Vinst/Förlust"]
    # (mock) qty kommer från MOCK_TRADE_USDT/entry
    for r in pnl_log:
        # Vi fyller bara sälj-rader (EXIT) för K4
        if r["side"] != "SELL":
            continue
        # Hitta entry för symboln närmast före denna sell
        # (enkel modell: anta senaste BUY före denna SELL)
        # Här har vi inte qty i loggen – vi härleder inte exakt; K4 mockas minimalt.
        datum = r["ts"][:10]
        typ = "Köp/Sälj (krypto)"
        vp = r["symbol"]
        antal = ""  # lämnas tomt i denna minimala export
        förs = f"{r['price']:.6f}"
        omk = ""    # kräver spårning av entry->exit qty/price par
        vinst = f"{r['pnl']:.6f}"
        rows.append(";".join([datum, typ, vp, antal, förs, omk, vinst]))
    text = "\n".join(rows)
    update.message.reply_document(document=bytes(text, "utf-8"), filename=f"k4_{date.today().isoformat()}.csv")

def cmd_keepalive_on(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        state["keepalive"] = True
    update.message.reply_text("Keepalive: ON")

def cmd_keepalive_off(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        state["keepalive"] = False
    update.message.reply_text("Keepalive: OFF")

def cmd_panic(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        syms = list(positions.keys())
    for sym in syms:
        try:
            price = get_last_price(sym)
        except Exception:
            price = None
        if price is not None:
            close_position(sym, price, "PANIC")
    update.message.reply_text("Panic: alla positioner stängda.")

def cmd_orb_on(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        state["orb_master"] = True
        state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("ORB: ON (för alla valda symboler)")

def cmd_orb_off(update: Update, _: CallbackContext):
    set_chat_id(update)
    with lock:
        state["orb_master"] = False
        state["orb_symbols"].clear()
    update.message.reply_text("ORB: OFF (för alla)")

# =========================
# Telegram bootstrap + guard
# =========================
def start_polling_with_guard(upd: Updater):
    backoff = 15  # sek
    while True:
        try:
            upd.start_polling(clean=True)
            upd.idle()
            break  # normal shutdown
        except Conflict as e:
            # En annan instans kör – försök igen senare i stället för att dö.
            print(f"[WARN] Telegram Conflict (annan instans kör). Försöker igen om {backoff}s. {e}")
            time.sleep(backoff)
            continue
        except NetworkError as e:
            print(f"[WARN] Telegram nätverksfel. Försöker igen om {backoff}s. {e}")
            time.sleep(backoff)
            continue
        except Exception as e:
            print(f"[ERROR] Oväntat fel i polling: {e}. Försöker igen om {backoff}s.")
            time.sleep(backoff)

def start_telegram():
    global updater
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("engine_start", cmd_engine_start))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(CommandHandler("symbols", cmd_symbols))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe))
    dp.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    dp.add_handler(CallbackQueryHandler(on_button))
    dp.add_handler(CommandHandler("trailing", cmd_trailing))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("export_csv", cmd_export_csv))
    dp.add_handler(CommandHandler("export_k4", cmd_export_k4))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dp.add_handler(CommandHandler("panic", cmd_panic))
    dp.add_handler(CommandHandler("orb_on", cmd_orb_on))
    dp.add_handler(CommandHandler("orb_off", cmd_orb_off))

    # Kör med guard som hanterar Conflict
    t = threading.Thread(target=start_polling_with_guard, args=(updater,), daemon=True)
    t.start()

# =========================
# Threads
# =========================
def start_threads():
    # ORB auto-ON vid start
    with lock:
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    # en motortråd per symbol
    for sym in state["symbols"]:
        th = threading.Thread(target=engine_symbol, args=(sym,), daemon=True)
        th.start()
    # keepalive
    thk = threading.Thread(target=keepalive_loop, daemon=True)
    thk.start()

# =========================
# Uvicorn entry (Render)
# =========================
@app.on_event("startup")
def _startup():
    start_telegram()
    start_threads()
