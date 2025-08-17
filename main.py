import os
import time
import math
import threading
from datetime import datetime, timezone, date
from typing import Dict, List, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# --- Telegram (python-telegram-bot 13.15) ---
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater, CallbackContext, CommandHandler, CallbackQueryHandler,
)

# =========================
# FastAPI (Render health)
# =========================
app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "ts": time.time()}

@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}

# =========================
# Konfiguration & state
# =========================
BINANCE_BASE = "https://api.binance.com"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]
VALID_TF = {"1m": "1m", "3m": "3m", "5m": "5m"}
DEFAULT_TF = "1m"

KEEPALIVE_URL = os.getenv("KEEPALIVE_URL", "").strip()
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "120"))

state = {
    "mode": "mock",                 # "mock" / "live"
    "engine_on": True,              # motor kör
    "symbols": DEFAULT_SYMBOLS.copy(),
    "timeframe": DEFAULT_TF,        # 1m/3m/5m
    "entry_mode": "close",          # "tick" / "close"
    # Trailing-konfig
    "trail_enabled": True,
    "trail_trigger": 0.009,         # +0.90%
    "trail_distance": 0.002,        # 0.20%
    "trail_min_lock": 0.007,        # minst +0.70% lås
    # ORB master + aktiva symboler
    "orb_master": True,             # auto ON
    "orb_symbols": set(DEFAULT_SYMBOLS),
    # Keepalive
    "keepalive": True,
    # Mock-PnL
    "day_pnl": 0.0,
}

# Per-symbol runtime
positions: Dict[str, Dict] = {}      # symbol -> {"entry","qty","sl","trail_on","max_up"}
orb_state: Dict[str, Dict] = {}       # symbol -> {"locked":bool,"high":float,"low":float,"arm_ts":int,"last_candle_ts":int,"entry_used":bool}
pnl_log: List[Dict] = []              # trade-rader för export

# Lås för trådar
lock = threading.RLock()

# Telegram globals
updater: Optional[Updater] = None
CHAT_ID: Optional[int] = None

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.4f}"

# =========================
# Datahämtning (Binance)
# =========================
def binance_klines(symbol: str, interval: str, limit: int = 3):
    url = f"{BINANCE_BASE}/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json()

def get_latest_candles(symbol: str, interval: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Returnerar [(open_time, open, high, low, close), ...] (floatar).
    Sista elementet är ofta 'pågående'; näst sista är säkert stängd.
    """
    raw = binance_klines(symbol, interval, limit=3)
    out = []
    for k in raw:
        open_time = int(k[0])  # ms
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
        out.append((open_time, o, h, l, c))
    return out

def get_last_price(symbol: str) -> Optional[float]:
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    try:
        r = requests.get(url, params={"symbol": symbol}, timeout=8)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None

# =========================
# Notifier / logg
# =========================
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
# ORB & entry/exit logik
# =========================
def lock_new_orb_if_red_to_green(symbol: str, candles: List[Tuple[int,float,float,float,float]]):
    """
    Låser ORB (high/low) på första GRÖNA som stänger efter en RÖD.
    ORB ändras inte förrän en RÖD candle stänger igen.
    """
    # vi använder två senaste stängda candles: [-3], [-2] säkra stängda; [-1] kan vara pågående.
    if len(candles) < 2:
        return
    # ta de två sista elementen; [-2] = senast stängda, [-1] = troligen pågående
    (t_prev, o_prev, h_prev, l_prev, c_prev) = candles[-2]
    # försök även få föregående till den, dvs [-3]
    if len(candles) >= 3:
        (t_prev2, o_prev2, h_prev2, l_prev2, c_prev2) = candles[-3]
    else:
        t_prev2 = None; o_prev2 = h_prev2 = l_prev2 = c_prev2 = None

    s = orb_state.setdefault(symbol, {"locked": False, "high": None, "low": None, "arm_ts": 0, "last_candle_ts": 0, "entry_used": False})

    # Ny färdig candle?
    if t_prev != s["last_candle_ts"]:
        # Om senast stängda är RÖD → nollställ "entry_used" (ny potentiell setup när grön kommer)
        if c_prev < o_prev:
            # När en röd stänger: lås upp systemet så nästa GRÖNA kan låsa ORB
            s["locked"] = False
            s["high"] = None
            s["low"] = None
            s["entry_used"] = False

        # Om vi har en röd föregående ([-3]) och sen en grön ([-2]) → lås ORB på den gröna
        if (c_prev2 is not None) and (c_prev2 < o_prev2) and (c_prev > o_prev):
            s["locked"] = True
            s["high"] = h_prev
            s["low"] = l_prev
            s["arm_ts"] = t_prev

        s["last_candle_ts"] = t_prev

def try_entry(symbol: str, candles: List[Tuple[int,float,float,float,float]]):
    """
    Entry:
      - close-mode: köp när SENASTE STÄNGDA candle stänger > ORB-high.
      - tick-mode: köp så fort 'pågående' candle (eller live-price) bryter ORB-high.
    Endast 1 köp per låst ORB (entry_used=False -> True).
    """
    s = orb_state.get(symbol, {})
    if not s or not s.get("locked") or s.get("high") is None or s.get("entry_used"):
        return

    orb_high = s["high"]
    tf = state["timeframe"]

    if state["entry_mode"] == "close":
        # näst sista är senast stängda
        if len(candles) < 2:
            return
        (_, o_cl, h_cl, l_cl, c_cl) = candles[-2]
        if c_cl > orb_high:
            enter_long(symbol, entry_price=c_cl, initial_sl=s["low"])
            s["entry_used"] = True

    else:  # tick
        # kolla pågående candles high eller live price
        last_px = get_last_price(symbol)
        (_, o_cur, h_cur, l_cur, c_cur) = candles[-1]
        crossed = False
        if last_px is not None and last_px > orb_high:
            crossed = True
        elif h_cur > orb_high:
            crossed = True
        if crossed:
            # använd orb_high som entry eller aktuell close — vi väljer orb_high för konservativ fill
            entry_px = max(orb_high, c_cur)
            enter_long(symbol, entry_price=entry_px, initial_sl=s["low"])
            s["entry_used"] = True

def enter_long(symbol: str, entry_price: float, initial_sl: float):
    if symbol in positions:
        return
    qty = 1.0  # mockstorlek
    positions[symbol] = {
        "entry": entry_price,
        "qty": qty,
        "sl": initial_sl,
        "trail_on": False,
        "max_up": entry_price,
        "last_closed_low": initial_sl,   # för "stegvis" SL
    }
    log_trade(symbol, "BUY", entry_price, 0.0)
    notify(f"{symbol}: ORB breakout BUY @ {fmt(entry_price)} | SL {fmt(initial_sl)}")

def step_trail_sl(symbol: str, candles: List[Tuple[int,float,float,float,float]]):
    """
    Efter entry: varje NY stängd candle flyttar SL till föregående candle's low, om högre än nuvarande SL.
    (stegvis uppåt – aldrig nedåt)
    """
    pos = positions.get(symbol)
    if not pos:
        return
    s = orb_state.get(symbol, {})
    if len(candles) < 2:
        return
    (t_prev, o_prev, h_prev, l_prev, c_prev) = candles[-2]  # senast stängda candle
    # höj SL om l_prev > nuvarande SL
    if pos["sl"] is None or l_prev > pos["sl"]:
        pos["sl"] = l_prev
    pos["last_closed_low"] = l_prev
    # uppdatera ORB-låg uppåt (speglat, men ORB i sig ändras inte förrän röd close)
    # inget att göra på s["low"] här – vi använder bara den som initial

def manage_peak_trailing(symbol: str):
    """
    Extra trailing:
      triggas vid +0.90% från entry → trail 0.20% under högsta priset (max_up).
      garanterar minst +0.70% om möjligt.
    """
    pos = positions.get(symbol)
    if not pos:
        return
    price = get_last_price(symbol)
    if price is None:
        return
    entry = pos["entry"]
    # max-up
    if price > pos["max_up"]:
        pos["max_up"] = price
    # trigger?
    if state["trail_enabled"]:
        gain = (price / entry) - 1.0
        if (not pos["trail_on"]) and gain >= state["trail_trigger"]:
            pos["trail_on"] = True
        if pos["trail_on"]:
            trail_sl = pos["max_up"] * (1.0 - state["trail_distance"])
            min_lock_sl = entry * (1.0 + state["trail_min_lock"])
            new_sl = max(trail_sl, min_lock_sl)
            # även jämför med "stegvis" SL (pos["sl"]) – ta det högsta
            if pos["sl"] is not None:
                new_sl = max(new_sl, pos["sl"])
            pos["sl"] = new_sl

def stop_hit_check(symbol: str):
    """
    Exit om priset når SL (tick-baserat).
    """
    pos = positions.get(symbol)
    if not pos:
        return
    price = get_last_price(symbol)
    if price is None:
        return
    if pos["sl"] is not None and price <= pos["sl"]:
        close_position(symbol, price, reason="SL")

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
    # OBS: ORB förblir låst tills en röd candle stänger → då låses upp för nästa setup i lock_new_orb_if_red_to_green

# =========================
# Engine loop per symbol
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

            # Hämta candles
            candles = get_latest_candles(symbol, tf)

            # Lås ORB när röd→grön uppstår, och håll låst tills ny röd stänger
            if use_orb:
                lock_new_orb_if_red_to_green(symbol, candles)
                # Entry enligt mode (close/tick)
                try_entry(symbol, candles)

            # Stegvis SL efter varje stängd candle (om inne)
            step_trail_sl(symbol, candles)

            # Extra trailing vid +0.90% / 0.20% (min 0.70%)
            manage_peak_trailing(symbol)

            # SL-träff tick-baserad
            stop_hit_check(symbol)

        except Exception:
            pass
        finally:
            time.sleep(1.0)

# =========================
# Keepalive (Render ping)
# =========================
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
# Telegram UI / Kommandon
# =========================
def set_chat_id(update: Update):
    global CHAT_ID
    cid = update.effective_chat.id
    if CHAT_ID is None:
        CHAT_ID = cid

def render_status() -> str:
    with lock:
        lines = []
        lines.append(f"Mode: {state['mode']}   Engine: {'ON' if state['engine_on'] else 'OFF'}")
        lines.append(f"TF: {state['timeframe']}   Symbols: {','.join(state['symbols'])}")
        lines.append(f"Entry mode: {state['entry_mode'].upper()}")
        lines.append(f"Trail: {'ON' if state['trail_enabled'] else 'OFF'} ({state['trail_trigger']*100:.2f}%/{state['trail_distance']*100:.2f}% min {state['trail_min_lock']*100:.2f}%)")
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

def cmd_help(update: Update, _: CallbackContext):
    txt = (
        "/status\n"
        "/engine_start /engine_stop\n"
        "/start_mock /start_live\n"
        "/symbols BTCUSDT/ETHUSDT/ADAUSDT/LINKUSDT/XRPUSDT\n"
        "/timeframe 1m|3m|5m\n"
        "/entry_mode   (visar knappar)\n"
        "/trailing <trig%> <avst%> <min%>  ex /trailing 0.9 0.2 0.7\n"
        "/pnl /reset_pnl /export_csv /export_k4\n"
        "/keepalive_on /keepalive_off\n"
        "/panic\n"
    )
    update.message.reply_text(txt)

def cmd_status(update: Update, context: CallbackContext):
    set_chat_id(update)
    update.message.reply_text(render_status())

def cmd_engine_start(update: Update, context: CallbackContext):
    with lock:
        state["engine_on"] = True
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
    with lock:
        state["mode"] = "live"
        state["engine_on"] = True
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text("Mode: LIVE (demo)")

def cmd_symbols(update: Update, context: CallbackContext):
    text = update.message.text.strip()
    parts = text.split(maxsplit=1)
    if len(parts) == 1:
        update.message.reply_text(f"Aktuella symboler: {','.join(state['symbols'])}")
        return
    raw = parts[1].replace(",", "/").upper()
    syms = [s for s in raw.split("/") if s]
    if not syms:
        update.message.reply_text("Använd: /symbols BTCUSDT/ETHUSDT/ADAUSDT/LINKUSDT/XRPUSDT"); return
    with lock:
        state["symbols"] = syms
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    update.message.reply_text(f"Symboler satta till: {','.join(syms)}")

def cmd_timeframe(update: Update, context: CallbackContext):
    parts = update.message.text.strip().split()
    if len(parts) == 1:
        update.message.reply_text(f"Tidsram: {state['timeframe']}"); return
    tf = parts[1].lower()
    if tf not in VALID_TF:
        update.message.reply_text("Använd: /timeframe 1m|3m|5m"); return
    with lock:
        state["timeframe"] = tf
    update.message.reply_text(f"Tidsram satt till {tf}")

def cmd_entry_mode(update: Update, context: CallbackContext):
    update.message.reply_text("Välj entry-mode:", reply_markup=kb_entry_mode())

def on_button(update: Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    if q.data == "entry_close":
        with lock:
            state["entry_mode"] = "close"
        q.edit_message_text("✅ Entry mode set to: CLOSE")
    elif q.data == "entry_tick":
        with lock:
            state["entry_mode"] = "tick"
        q.edit_message_text("✅ Entry mode set to: TICK")

def cmd_trailing(update: Update, context: CallbackContext):
    parts = update.message.text.strip().split()
    if len(parts) == 1:
        with lock:
            msg = f"Trail: {'ON' if state['trail_enabled'] else 'OFF'} ({state['trail_trigger']*100:.2f}%/{state['trail_distance']*100:.2f}% min {state['trail_min_lock']*100:.2f}%)"
        update.message.reply_text(msg); return
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
        update.message.reply_text("Använd: /trailing <trig%> <avst%> <min%>  ex /trailing 0.9 0.2 0.7")

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
    text = "\n".join(_export_csv_rows())
    update.message.reply_document(document=bytes(text, "utf-8"),
                                  filename=f"trades_{date.today().isoformat()}.csv")

def cmd_export_k4(update: Update, context: CallbackContext):
    rows = ["datum,symbol,typ,belopp"]
    for r in pnl_log:
        typ = "Försäljning" if r["side"] == "SELL" else "Köp"
        belopp = f"{r['price']:.6f}"
        datum = r["ts"][:10]
        rows.append(f"{datum},{r['symbol']},{typ},{belopp}")
    text = "\n".join(rows)
    update.message.reply_document(document=bytes(text, "utf-8"),
                                  filename=f"k4_{date.today().isoformat()}.csv")

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
        price = get_last_price(sym)
        if price is not None:
            close_position(sym, price, "PANIC")
    update.message.reply_text("Panic: alla positioner stängda.")

# =========================
# Telegram init
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Sätt TELEGRAM_TOKEN i miljövariablerna.")

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

    updater.start_polling(drop_pending_updates=True)

# =========================
# Trådar
# =========================
def start_threads():
    # motortråd per symbol
    for sym in state["symbols"]:
        th = threading.Thread(target=engine_symbol, args=(sym,), daemon=True)
        th.start()
    # keepalive
    thk = threading.Thread(target=keepalive_loop, daemon=True)
    thk.start()

# =========================
# Uvicorn entry (Render)
# =========================
def bootstrap_once():
    # ORB auto-ON för alla valda symboler vid start
    with lock:
        if state["orb_master"]:
            state["orb_symbols"] = set(state["symbols"])
    start_telegram()
    start_threads()

bootstrap_once()
