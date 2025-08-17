# ---------- main_v9.py ----------
import os
import time
import csv
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI
from telegram import (
    Bot, Update, InlineKeyboardMarkup, InlineKeyboardButton, ParseMode
)
from telegram.ext import (
    Updater, CommandHandler, CallbackQueryHandler, CallbackContext
)

# =========================
# Config & Globals
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
PING_URL = os.getenv("PING_URL", os.getenv("RENDER_EXTERNAL_URL", ""))

ENGINE_ON = False            # startar avstängd; slå på med /engine_start
MODE = "mock"                # mock | live (orders för live är inte aktiva i denna fil)
TIMEFRAME = "1m"             # 1m / 3m / 5m
ENTRY_MODE = "close"         # close | tick
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

# trailing (trigger 0.9%, trail 0.2%, min lock 0.7%)
TRIG_PCT = 0.009
TRAIL_PCT = 0.002
MIN_LOCK_PCT = 0.007
TRAIL_ENABLED = True

KEEPALIVE = True
RENDER_PING_EVERY_SEC = 120

day_pnl = 0.0
lock = threading.Lock()
trade_log: List[dict] = []

# ORB master switch
ORB_ACTIVE_GLOBAL = True

# =========================
# FastAPI (health)
# =========================
app = FastAPI()
APP_START = datetime.now(timezone.utc)

@app.get("/")
def root():
    return {
        "ok": True,
        "uptime_sec": int((datetime.now(timezone.utc)-APP_START).total_seconds()),
        "engine": ENGINE_ON,
        "mode": MODE,
        "tf": TIMEFRAME
    }

# =========================
# Hjälp-funktioner
# =========================
TF_MAP = {"1m":"1min", "3m":"3min", "5m":"5min"}

def ku_klines(symbol: str, tf: str, limit:int=3) -> List[Tuple[int,float,float,float,float]]:
    """
    Returnerar [(ts, open, high, low, close), ...] (äldst -> nyast)
    """
    t = TF_MAP.get(tf, "1min")
    sym = symbol.replace("USDT", "-USDT")
    url = "https://api.kucoin.com/api/v1/market/candles"
    r = requests.get(url, params={"type": t, "symbol": sym}, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    out = []
    # KuCoin returnerar nyast först: [time, open, close, high, low, volume, turnover]
    for row in data[:limit][::-1]:
        ts = int(row[0]); op=float(row[1]); cl=float(row[2]); hi=float(row[3]); lo=float(row[4])
        out.append((ts, op, hi, lo, cl))
    return out

def tick_price(symbol: str) -> Optional[float]:
    try:
        sym = symbol.replace("USDT", "-USDT")
        r = requests.get("https://api.kucoin.com/api/v1/market/orderbook/level1",
                         params={"symbol": sym}, timeout=10)
        r.raise_for_status()
        return float(r.json()["data"]["price"])
    except Exception:
        return None

def qty_for_mock(price: float) -> float:
    if price <= 0: return 0.0
    return round(MOCK_TRADE_USDT / price, 6)

def fmt(n: float, d: int = 6) -> str:
    if n is None: return "-"
    return f"{n:.{d}f}"

def record_trade(symbol: str, side: str, price: float, pnl: float = 0.0):
    global day_pnl
    with lock:
        ts = datetime.now(timezone.utc).isoformat()
        trade_log.append({"t": ts, "symbol": symbol, "side": side, "price": price, "pnl": pnl, "mode": MODE})
        day_pnl += pnl

# =========================
# Position & ORB-state
# =========================
class Pos:
    def __init__(self):
        self.in_pos = False
        self.entry: Optional[float] = None
        self.size: float = 0.0
        self.stop: Optional[float] = None
        self.max_up: float = 0.0

        # ORB
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.orb_on: bool = False
        self.last_orb_candle_id: Optional[int] = None

positions: Dict[str, Pos] = {s: Pos() for s in SYMBOLS}

def is_green(op: float, cl: float) -> bool:
    return cl > op

# =========================
# ORB-logik (första grön efter röd)
# =========================
def update_orb_from_closed_candle(symbol: str):
    """
    Ny ORB när vi får första gröna candle efter en röd.
    ORB = high/low på den gröna candlen. ORB blir aktiv (ON).
    """
    if not ORB_ACTIVE_GLOBAL:  # master switch
        return
    pos = positions[symbol]
    kl = ku_klines(symbol, TIMEFRAME, limit=3)
    if len(kl) < 2:
        return
    prev = kl[-2]  # (ts, op, hi, lo, cl)
    cur  = kl[-1]

    if pos.last_orb_candle_id == cur[0]:
        return  # redan processad

    pos.last_orb_candle_id = cur[0]

    # Röd -> Grön => armera ny ORB
    if (prev[4] < prev[1]) and (cur[4] > cur[1]):  # prev red, cur green
        pos.orb_high = cur[2]
        pos.orb_low  = cur[3]
        pos.orb_on   = True
        # om i position: höj stop till minst den nya candlens low
        if pos.in_pos and pos.stop is not None:
            pos.stop = max(pos.stop, pos.orb_low)
    else:
        # om ny röd: disarma ORB (vänta på nästa röd->grön)
        if cur[4] < cur[1]:
            pos.orb_on = False

def try_enter_long(symbol: str, bot: Bot, chat_id: Optional[int]):
    """
    Entry på close (default) eller tick när priset bryter ORB-high.
    1 entry per ORB (pos.orb_on sätts av).
    """
    pos = positions[symbol]
    if pos.in_pos or not pos.orb_on or pos.orb_high is None or pos.orb_low is None:
        return

    price_now = tick_price(symbol)
    if price_now is None:
        return

    # bestäm trigg beroende på ENTRY_MODE
    trigger = False
    if ENTRY_MODE == "close":
        # kräver att SENASTE STÄNGDA candle stängde över ORB-high
        kl = ku_klines(symbol, TIMEFRAME, limit=1)
        if not kl: return
        last = kl[-1]
        trigger = last[4] > pos.orb_high
        entry_px = last[4]
    else:  # tick
        trigger = price_now > pos.orb_high
        entry_px = price_now

    if trigger:
        pos.in_pos = True
        pos.entry = entry_px
        pos.size = qty_for_mock(entry_px)
        pos.stop = pos.orb_low
        pos.max_up = 0.0
        pos.orb_on = False  # 1 köp per ORB
        record_trade(symbol, "BUY", entry_px, 0.0)
        if chat_id:
            bot.send_message(chat_id, f"{symbol}: ORB breakout BUY @ {fmt(entry_px)} | SL {fmt(pos.stop)}")

def trail_and_exit(symbol: str, bot: Bot, chat_id: Optional[int]):
    """
    - Stop-loss på pris <= stop
    - Ratchet stop till senaste stängda candlens low (aldrig nedåt)
    - Trailing: när upp ≥ 0.9 %, trail 0.2 % och lås minst +0.7 %
    """
    pos = positions[symbol]
    if not pos.in_pos or pos.entry is None:
        return

    price_now = tick_price(symbol)
    if price_now is None:
        return

    # SL träffad?
    if pos.stop is not None and price_now <= pos.stop:
        pnl = (price_now - pos.entry) * pos.size
        record_trade(symbol, "EXIT", price_now, pnl)
        if chat_id:
            bot.send_message(chat_id, f"{symbol}: EXIT @ {fmt(price_now)} | PnL {pnl:+.4f} USDT")
        # nolla position
        pos.in_pos = False
        pos.entry = None
        pos.size = 0.0
        pos.stop = None
        pos.max_up = 0.0
        return

    # följ stoppet till senaste STÄNGDA candlens low
    kl = ku_klines(symbol, TIMEFRAME, limit=2)
    if len(kl) >= 1:
        last_closed = kl[-1]
        last_low = last_closed[3]
        if pos.stop is None:
            pos.stop = last_low
        else:
            pos.stop = max(pos.stop, last_low)

    # trailing på vinst
    up = (price_now - pos.entry) / pos.entry
    if up > pos.max_up:
        pos.max_up = up

    if TRAIL_ENABLED and up >= TRIG_PCT:
        lock_level = pos.entry * (1.0 + MIN_LOCK_PCT)           # +0.7%
        trail_level = pos.entry * (1.0 + (pos.max_up - TRAIL_PCT))  # 0.2% under peak
        new_stop = max(pos.stop or 0.0, lock_level, trail_level)
        if new_stop > (pos.stop or 0.0):
            pos.stop = new_stop

# =========================
# Trading-loop
# =========================
def trading_loop(bot: Bot, chat_id: Optional[int]):
    while True:
        try:
            if ENGINE_ON and SYMBOLS:
                for s in SYMBOLS:
                    try:
                        update_orb_from_closed_candle(s)
                        try_enter_long(s, bot, chat_id)
                        trail_and_exit(s, bot, chat_id)
                    except Exception:
                        pass
            time.sleep(5 if ENTRY_MODE=="close" else 1)
        except Exception:
            time.sleep(2)

# =========================
# Telegram bot
# =========================
updater: Optional[Updater] = None
ADMIN_CHAT_ID: Optional[int] = None

def kb_timeframes():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1m", callback_data="tf|1m"),
         InlineKeyboardButton("3m", callback_data="tf|3m"),
         InlineKeyboardButton("5m", callback_data="tf|5m")]
    ])

def kb_entry_mode():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Entry: CLOSE", callback_data="emode|close")],
        [InlineKeyboardButton("Entry: TICK",  callback_data="emode|tick")],
    ])

def help_text() -> str:
    return (
        "/status\n"
        "/engine_start  /engine_stop\n"
        "/start_mock  /start_live\n"
        "/symbols BTCUSDT/ETHUSDT/ADAUSDT\n"
        "/timeframe  (knappar)\n"
        "/entry_mode  (knappar)\n"
        "/orb_on  /orb_off\n"
        "/pnl  /reset_pnl  /export_csv  /export_k4\n"
        "/keepalive_on  /keepalive_off\n"
        "/panic"
    )

def cmd_help(update: Update, ctx: CallbackContext):
    update.message.reply_text(help_text())

def cmd_status(update: Update, ctx: CallbackContext):
    global ADMIN_CHAT_ID
    ADMIN_CHAT_ID = update.effective_chat.id
    lines = [
        f"Mode: {MODE}   Engine: {'ON' if ENGINE_ON else 'OFF'}",
        f"TF: {TIMEFRAME}   Symbols: {','.join(SYMBOLS)}",
        f"Entry: {ENTRY_MODE}   Trail: {'ON' if TRAIL_ENABLED else 'OFF'} "
        f"({TRIG_PCT*100:.1f}%/{TRAIL_PCT*100:.1f}% min {MIN_LOCK_PCT*100:.1f}%)",
        f"Keepalive: {'ON' if KEEPALIVE else 'OFF'}   DayPnL: {day_pnl:.4f} USDT",
        f"ORB master: {'ON' if ORB_ACTIVE_GLOBAL else 'OFF'}",
    ]
    for s in SYMBOLS:
        p = positions[s]
        posflag = "✅" if p.in_pos else "❌"
        stop_s = fmt(p.stop) if p.stop else "-"
        orb_s = "ON" if p.orb_on else "OFF"
        orb_rng = f"[{fmt(p.orb_low)},{fmt(p.orb_high)}]" if (p.orb_low and p.orb_high) else "[-]"
        lines.append(f"{s}: pos={posflag} stop={stop_s} | ORB: {orb_s} {orb_rng}")
    update.message.reply_text("\n".join(lines))

def cmd_engine_start(update: Update, ctx: CallbackContext):
    global ENGINE_ON
    ENGINE_ON = True
    update.message.reply_text("Engine: ON")

def cmd_engine_stop(update: Update, ctx: CallbackContext):
    global ENGINE_ON
    ENGINE_ON = False
    update.message.reply_text("Engine: OFF")

def cmd_start_mock(update: Update, ctx: CallbackContext):
    global MODE
    MODE = "mock"
    update.message.reply_text("Mode: MOCK")

def cmd_start_live(update: Update, ctx: CallbackContext):
    global MODE
    MODE = "live"
    update.message.reply_text("Mode: LIVE (obs: placeholder – denna fil skickar ej riktiga orders)")

def cmd_symbols(update: Update, ctx: CallbackContext):
    global SYMBOLS, positions
    if ctx.args:
        raw = " ".join(ctx.args).replace(",", "/").upper()
        parts = [p.strip() for p in raw.split("/") if p.strip()]
        with lock:
            SYMBOLS = parts[:8]
            for s in SYMBOLS:
                if s not in positions:
                    positions[s] = Pos()
        update.message.reply_text(f"Symbols: {','.join(SYMBOLS)}")
    else:
        update.message.reply_text("Använd t.ex. /symbols BTCUSDT/ETHUSDT/ADAUSDT")

def cmd_timeframe(update: Update, ctx: CallbackContext):
    update.message.reply_text("Välj timeframe:", reply_markup=kb_timeframes())

def cmd_entry_mode(update: Update, ctx: CallbackContext):
    update.message.reply_text(f"Aktuellt: {ENTRY_MODE}\nVälj nytt:", reply_markup=kb_entry_mode())

def cmd_trailing(update: Update, ctx: CallbackContext):
    global TRAIL_ENABLED
    if ctx.args and ctx.args[0].lower() in ("on","off"):
        TRAIL_ENABLED = (ctx.args[0].lower() == "on")
        update.message.reply_text(f"Trailing: {'ON' if TRAIL_ENABLED else 'OFF'}")
    else:
        update.message.reply_text("Använd: /trailing on|off")

def cmd_pnl(update: Update, ctx: CallbackContext):
    update.message.reply_text(f"Dagens PnL: {day_pnl:.4f} USDT")

def cmd_reset_pnl(update: Update, ctx: CallbackContext):
    global day_pnl
    day_pnl = 0.0
    update.message.reply_text("Day PnL reset.")

def cmd_export_csv(update: Update, ctx: CallbackContext):
    if not trade_log:
        update.message.reply_text("Ingen logg ännu.")
        return
    header = "time,symbol,side,price,pnl,mode"
    rows = [header] + [f"{r['t']},{r['symbol']},{r['side']},{r['price']},{r['pnl']},{r['mode']}" for r in trade_log]
    txt = "```\n" + "\n".join(rows) + "\n```"
    update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

def cmd_export_k4(update: Update, ctx: CallbackContext):
    # enkel K4-liknande csv i chatten
    if not trade_log:
        update.message.reply_text("Ingen logg ännu.")
        return
    lines = ["datum,typ,vardepapper,antal,belopp"]
    for r in trade_log:
        typ = "Försäljning" if r["side"] == "EXIT" else "Köp"
        antal = "" if r["side"] == "EXIT" else "1"
        belopp = f"{float(r['pnl']):.6f}" if r["side"] == "EXIT" else f"{float(r['price']):.6f}"
        lines.append(f"{r['t'][:10]},{typ},{r['symbol']},{antal},{belopp}")
    txt = "```\n" + "\n".join(lines) + "\n```"
    update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

def cmd_keepalive_on(update: Update, ctx: CallbackContext):
    global KEEPALIVE
    KEEPALIVE = True
    update.message.reply_text("Keepalive: ON")

def cmd_keepalive_off(update: Update, ctx: CallbackContext):
    global KEEPALIVE
    KEEPALIVE = False
    update.message.reply_text("Keepalive: OFF")

def cmd_panic(update: Update, ctx: CallbackContext):
    # stäng mock-positioner om några fanns
    for s in SYMBOLS:
        p = positions[s]
        if p.in_pos and p.entry:
            px = tick_price(s) or p.entry
            pnl = (px - p.entry) * p.size
            record_trade(s, "EXIT", px, pnl)
        positions[s] = Pos()
    update.message.reply_text("ALLA positioner stängda och motor stoppad.")
    global ENGINE_ON
    ENGINE_ON = False

def cmd_orb_on(update: Update, ctx: CallbackContext):
    global ORB_ACTIVE_GLOBAL
    ORB_ACTIVE_GLOBAL = True
    update.message.reply_text("ORB: ON (första gröna efter röd; entry på close > ORB-high; SL=ORB-low; SL följer varje candle).")

def cmd_orb_off(update: Update, ctx: CallbackContext):
    global ORB_ACTIVE_GLOBAL
    ORB_ACTIVE_GLOBAL = False
    update.message.reply_text("ORB: OFF")

def on_cb(update: Update, ctx: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data or ""
    global TIMEFRAME, ENTRY_MODE
    if data.startswith("tf|"):
        TIMEFRAME = data.split("|",1)[1]
        q.edit_message_text(f"Timeframe satt till {TIMEFRAME}")
        return
    if data.startswith("emode|"):
        ENTRY_MODE = data.split("|",1)[1]
        q.edit_message_text(f"Entry mode: {ENTRY_MODE}")
        return

# =========================
# Keepalive ping
# =========================
def keepalive_loop(url: str):
    if not url:
        return
    while True:
        try:
            if KEEPALIVE:
                requests.get(url, timeout=10)
        except Exception:
            pass
        time.sleep(RENDER_PING_EVERY_SEC)

# =========================
# Start all
# =========================
def start_all():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Missing TELEGRAM_TOKEN")

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    bot = updater.bot

    # Viktigt: nollställ ev. webhook så polling funkar på Render
    try:
        bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass

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
    dp.add_handler(CallbackQueryHandler(on_cb))

    updater.start_polling(drop_pending_updates=True)

    # Trader
    threading.Thread(target=trading_loop, args=(bot, None), daemon=True).start()

    # Keepalive
    threading.Thread(target=keepalive_loop, args=(PING_URL,), daemon=True).start()

# autostart för Render/uvicorn import
start_all()
# ---------- end main_v9.py ----------
