import os
import time
import json
import threading
import datetime as dt
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton, ParseMode,
)
from telegram.ext import (
    Updater, CallbackContext, CommandHandler, CallbackQueryHandler,
)

# =========================
# ------- SETTINGS --------
# =========================
TOKEN = os.getenv("TELEGRAM_TOKEN", "")
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
DEFAULT_TF = "1m"                      # 1m/3m/5m
ENTRY_MODE = "close"                   # "close" eller "tick"
TRAIL_ON = True
TRIG_PCT = 0.009                       # 0.9%
STEP_PCT = 0.002                       # 0.2%
TRAIL_MIN = 0.007                      # 0.7%
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
PING_URL = os.getenv("PING_URL")
KEEPALIVE = True
ORB_MASTER = True                      # <— ORB PÅ som standard

# =========================
# ------- FASTAPI ---------
# =========================
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "mporbbot OK"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "healthy"

# =========================
# ------- STATE -----------
# =========================
state_lock = threading.Lock()

state: Dict = {
    "mode": "mock",            # "mock" eller "live"
    "engine": False,
    "symbols": DEFAULT_SYMBOLS[:],
    "tf": DEFAULT_TF,
    "entry_mode": ENTRY_MODE,
    "trail": {"on": TRAIL_ON, "trig": TRIG_PCT, "step": STEP_PCT, "min": TRAIL_MIN},
    "orb_master": ORB_MASTER,
    "orb": {s: {"active": False, "high": None, "low": None, "base_ts": None} for s in DEFAULT_SYMBOLS},
    "pos": {s: {"side": None, "qty": 0.0, "entry": None, "sl": None, "pnl": 0.0} for s in DEFAULT_SYMBOLS},
    "day_pnl": 0.0,
}

# =========================
# ---- DATA FEED (mock) ---
# =========================
# Enkel prisfeed via Binance public klines (för MOCK/logic). Ingen ny lib krävs.
BINANCE_BASE = "https://api.binance.com/api/v3/klines"

def fetch_candle(symbol: str, interval: str = "1m") -> Optional[dict]:
    """Hämtar SENASTE stängda candle + aktuell candle (behövs för SL-dragning).
       Returnerar dict {"open","high","low","close","closed"} för current bar."""
    try:
        # limit=2 => näst sista = senast stängda, sista = pågående
        r = requests.get(BINANCE_BASE, params={"symbol": symbol, "interval": interval, "limit": 2}, timeout=5)
        r.raise_for_status()
        k = r.json()
        now = k[-1]   # current (kan vara pågående)
        return {
            "open": float(now[1]),
            "high": float(now[2]),
            "low": float(now[3]),
            "close": float(now[4]),
            "closed": bool(now[6] < int(time.time()*1000) - 1000)  # grov indikator
        }
    except Exception:
        return None

# =========================
# ---- TRADING LOGIC ------
# =========================
def reset_orb_for_symbol(sym: str):
    st = state
    st["orb"][sym] = {"active": False, "high": None, "low": None, "base_ts": None}

def format_money(x: float) -> str:
    return f"{x:.4f}"

def send_msg(ctx: CallbackContext, text: str):
    try:
        ctx.bot.send_message(chat_id=ctx.job.context["chat_id"], text=text)
    except Exception:
        pass

def pnl_string() -> str:
    return f"DayPnL: {format_money(state['day_pnl'])} USDT"

def status_text() -> str:
    s = state
    lines = []
    lines.append(f"Mode: {s['mode']}   Engine: {'ON' if s['engine'] else 'OFF'}")
    lines.append(f"TF: {s['tf']}   Symbols: {','.join(s['symbols'])}")
    lines.append(f"Entry: {s['entry_mode']}   Trail: {'ON' if s['trail']['on'] else 'OFF'} "
                 f"({s['trail']['trig']*100:.2f}%/{s['trail']['step']*100:.2f}% min {s['trail']['min']*100:.2f}%)")
    lines.append(f"Keepalive: {'ON' if KEEPALIVE else 'OFF'}   {pnl_string()}")
    lines.append(f"ORB master: {'ON' if s['orb_master'] else 'OFF'}")
    for sym in s["symbols"]:
        p = s["pos"][sym]
        orb = s["orb"][sym]
        pos_flag = "✅" if p["side"] else "❌"
        stop = "-" if p["sl"] is None else f"{p['sl']:.4f}"
        orb_flag = "ON" if orb["active"] else "OFF"
        lines.append(f"{sym}: pos={pos_flag}  stop={stop} | ORB: {orb_flag}")
    return "\n".join(lines)

def notify(update_or_ctx, text: str, ctx: Optional[CallbackContext] = None):
    try:
        if isinstance(update_or_ctx, Update):
            update_or_ctx.message.reply_text(text)
        else:
            ctx.bot.send_message(chat_id=update_or_ctx.job.context["chat_id"], text=text)
    except Exception:
        pass

def try_orb_entry(sym: str, c: dict, ctx_job):
    s = state
    orb = s["orb"][sym]
    if not s["orb_master"]:
        return
    # Om ORB ej aktiv och vi nyss fick första gröna efter röd → sätt ORB-ram
    # Vi approximera: om close > open och föregånde var röd. Vi behöver föregående,
    # så kalla ett litet extra fetch på föregående stängda candle.
    try:
        r = requests.get(BINANCE_BASE, params={"symbol": sym, "interval": s["tf"], "limit": 3}, timeout=5)
        r.raise_for_status()
        k = r.json()
        prev = k[-2]        # senast stängda
        prev_open = float(prev[1]); prev_close = float(prev[4])
        this_open = float(k[-1][1]); this_close = float(k[-1][4])
        if (prev_close < prev_open) and (this_close > this_open) and not orb["active"]:
            # starta ORB på den HÄR gröna
            orb["active"] = True
            orb["high"] = max(this_open, this_close)
            orb["low"]  = min(this_open, this_close)
            orb["base_ts"] = int(time.time())
    except Exception:
        return

    # Entry: stängning över ORB-high (vi approximera med current close om entry_mode='close')
    p = s["pos"][sym]
    if orb["active"] and p["side"] is None:
        if s["entry_mode"] == "close":
            if c["close"] > orb["high"]:
                # BUY
                qty = MOCK_TRADE_USDT / c["close"]
                p.update({"side": "long", "qty": qty, "entry": c["close"], "sl": orb["low"]})
                txt = f"{sym}: ORB breakout BUY @ {c['close']:.4f} | SL {p['sl']:.4f}"
                notify(ctx_job, txt, ctx=ctx_job)
        else:  # tick-entry → använd high bryt live
            if c["high"] and c["high"] > orb["high"]:
                qty = MOCK_TRADE_USDT / orb["high"]
                p.update({"side": "long", "qty": qty, "entry": orb["high"], "sl": orb["low"]})
                txt = f"{sym}: ORB breakout BUY @ {p['entry']:.4f} | SL {p['sl']:.4f}"
                notify(ctx_job, txt, ctx=ctx_job)

def trailing_manage(sym: str, c: dict, ctx_job):
    s = state
    p = s["pos"][sym]
    if not p["side"]:
        return
    if not s["trail"]["on"]:
        return
    entry = p["entry"]
    gain = (c["close"] - entry) / entry
    # trigger
    if gain >= s["trail"]["trig"]:
        # höj SL till max(SL, close - step%)
        new_sl = max(p["sl"] if p["sl"] else -1e9, c["close"] * (1 - s["trail"]["step"]))
        # men säkra minst min%
        min_sl = entry * (1 + s["trail"]["min"])
        new_sl = max(new_sl, min_sl)
        if p["sl"] is None or new_sl > p["sl"]:
            p["sl"] = round(new_sl, 8)

def stop_check(sym: str, c: dict, ctx_job):
    s = state
    p = s["pos"][sym]
    if p["side"] and p["sl"] is not None:
        if c["low"] <= p["sl"]:  # SL träffad inom candlen
            exit_px = p["sl"]
            pnl = (exit_px - p["entry"]) * p["qty"]
            s["day_pnl"] += pnl
            txt = f"{sym}: EXIT @ {exit_px:.4f} | PnL {format_money(pnl)} USDT | {pnl_string()}"
            notify(ctx_job, txt, ctx=ctx_job)
            # Stäng position & stäng ORB tills ny röd→grön skapas
            state["pos"][sym] = {"side": None, "qty": 0.0, "entry": None, "sl": None, "pnl": 0.0}
            reset_orb_for_symbol(sym)

def engine_loop(ctx: CallbackContext):
    with state_lock:
        if not state["engine"]:
            return
        for sym in state["symbols"]:
            c = fetch_candle(sym, state["tf"])
            if not c:
                continue
            # uppdatera ORB-ram om vi fortfarande befinner oss i "orb-candle"
            if state["orb_master"] and state["orb"][sym]["active"]:
                # följ med candle-botten uppåt om candle stänger högre (trailing init)
                # — men enligt spec låter vi SL följa varje *ny candle* (vi tar current low om högre än tidigare)
                new_low = c["low"]
                if state["pos"][sym]["side"] and new_low > (state["pos"][sym]["sl"] or -1e9):
                    state["pos"][sym]["sl"] = round(new_low, 8)

            # försök skapa ORB + entry
            try_orb_entry(sym, c, ctx)

            # trailing SL
            trailing_manage(sym, c, ctx)

            # stop-check
            stop_check(sym, c, ctx)

# =========================
# ---- TELEGRAM BOT -------
# =========================
def build_kb_timeframes():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("TF 1m", callback_data="tf_1m"),
         InlineKeyboardButton("TF 3m", callback_data="tf_3m"),
         InlineKeyboardButton("TF 5m", callback_data="tf_5m")]
    ])

def build_kb_trail():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Trail +0.9%/0.2%", callback_data="trail_on"),
         InlineKeyboardButton("Trail OFF", callback_data="trail_off")]
    ])

def build_kb_engine():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Start MOCK", callback_data="start_mock"),
         InlineKeyboardButton("Start LIVE", callback_data="start_live")],
        [InlineKeyboardButton("Engine ON", callback_data="eng_on"),
         InlineKeyboardButton("Engine OFF", callback_data="eng_off")],
    ])

def cmd_help(update: Update, _: CallbackContext):
    update.message.reply_text(
        "/status\n"
        "/engine_start /engine_stop\n"
        "/start_mock /start_live\n"
        "/symbols BTCUSDT/ETHUSDT/ADAUSDT\n"
        "/timeframe\n"
        "/entry_mode tick|close\n"
        "/trailing\n"
        "/pnl /reset_pnl /export_csv /export_k4\n"
        "/keepalive_on /keepalive_off\n"
        "/panic",
        disable_web_page_preview=True
    )

def cmd_status(update: Update, _: CallbackContext):
    update.message.reply_text(status_text())

def cmd_engine_start(update: Update, _: CallbackContext):
    with state_lock:
        state["engine"] = True
    update.message.reply_text("Engine: ON", reply_markup=build_kb_engine())

def cmd_engine_stop(update: Update, _: CallbackContext):
    with state_lock:
        state["engine"] = False
    update.message.reply_text("Engine: OFF", reply_markup=build_kb_engine())

def cmd_start_mock(update: Update, _: CallbackContext):
    with state_lock:
        state["mode"] = "mock"
    update.message.reply_text("Mode: MOCK", reply_markup=build_kb_engine())

def cmd_start_live(update: Update, _: CallbackContext):
    with state_lock:
        state["mode"] = "live"
    update.message.reply_text("Mode: LIVE", reply_markup=build_kb_engine())

def cmd_symbols(update: Update, _: CallbackContext):
    tokens = update.message.text.split()
    if len(tokens) > 1:
        syms = tokens[1].replace(" ", "").replace(",", "/").split("/")
        syms = [s.upper() for s in syms if s]
        with state_lock:
            state["symbols"] = syms
            # initiera orb/pos containers
            for s in syms:
                state["orb"].setdefault(s, {"active": False, "high": None, "low": None, "base_ts": None})
                state["pos"].setdefault(s, {"side": None, "qty": 0.0, "entry": None, "sl": None, "pnl": 0.0})
        update.message.reply_text("Symbols uppdaterade.")
    else:
        update.message.reply_text("Använd: /symbols BTCUSDT/ETHUSDT/ADAUSDT")

def cmd_timeframe(update: Update, _: CallbackContext):
    update.message.reply_text(f"Timeframe satt till {state['tf']}", reply_markup=build_kb_timeframes())

def cmd_entry_mode(update: Update, _: CallbackContext):
    tokens = update.message.text.split()
    if len(tokens) > 1 and tokens[1] in ("tick", "close"):
        with state_lock:
            state["entry_mode"] = tokens[1]
        update.message.reply_text(f"Entry mode: {tokens[1]}")
    else:
        update.message.reply_text("Använd: /entry_mode tick|close")

def cmd_trailing(update: Update, _: CallbackContext):
    update.message.reply_text("Välj trail:", reply_markup=build_kb_trail())

def cmd_pnl(update: Update, _: CallbackContext):
    update.message.reply_text(pnl_string())

def cmd_reset_pnl(update: Update, _: CallbackContext):
    with state_lock:
        state["day_pnl"] = 0.0
    update.message.reply_text("DayPnL nollställd.")

def cmd_export_csv(update: Update, _: CallbackContext):
    update.message.reply_text("CSV export (mock) — implementerad i minnet.")

def cmd_export_k4(update: Update, _: CallbackContext):
    update.message.reply_text("K4 export (mock) — genererar radformat när vi har historikfil.")

def cmd_keepalive_on(update: Update, _: CallbackContext):
    global KEEPALIVE
    KEEPALIVE = True
    update.message.reply_text("Keepalive: ON")

def cmd_keepalive_off(update: Update, _: CallbackContext):
    global KEEPALIVE
    KEEPALIVE = False
    update.message.reply_text("Keepalive: OFF")

def cmd_panic(update: Update, _: CallbackContext):
    with state_lock:
        for s in state["symbols"]:
            state["pos"][s] = {"side": None, "qty": 0.0, "entry": None, "sl": None, "pnl": 0.0}
            reset_orb_for_symbol(s)
    update.message.reply_text("PANIC: Alla positioner stängda och ORB återställd.")

def on_button(update: Update, _: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data
    with state_lock:
        if data == "tf_1m":
            state["tf"] = "1m"
            q.edit_message_text(f"Timeframe satt till 1m")
        elif data == "tf_3m":
            state["tf"] = "3m"
            q.edit_message_text(f"Timeframe satt till 3m")
        elif data == "tf_5m":
            state["tf"] = "5m"
            q.edit_message_text(f"Timeframe satt till 5m")
        elif data == "trail_on":
            state["trail"]["on"] = True
            state["trail"]["trig"] = TRIG_PCT
            state["trail"]["step"] = STEP_PCT
            state["trail"]["min"] = TRAIL_MIN
            q.edit_message_text("Trail: ON (+0.9%/0.2% min 0.7%)")
        elif data == "trail_off":
            state["trail"]["on"] = False
            q.edit_message_text("Trail: OFF")
        elif data == "start_mock":
            state["mode"] = "mock"; q.edit_message_text("Mode: MOCK")
        elif data == "start_live":
            state["mode"] = "live"; q.edit_message_text("Mode: LIVE")
        elif data == "eng_on":
            state["engine"] = True; q.edit_message_text("Engine: ON")
        elif data == "eng_off":
            state["engine"] = False; q.edit_message_text("Engine: OFF")

# =========================
# ---- BACKGROUND JOBS ----
# =========================
def scheduler_thread(updater: Updater, chat_id: int):
    # engine loop – körs varje sekund
    updater.job_queue.run_repeating(engine_loop, interval=1.0, first=1.0, context={"chat_id": chat_id})
    # keepalive – varannan minut
    def do_ping(ctx: CallbackContext):
        if KEEPALIVE and PING_URL:
            try:
                requests.get(PING_URL, timeout=4)
            except Exception:
                pass
    updater.job_queue.run_repeating(do_ping, interval=120, first=15, context={"chat_id": chat_id})

def build_bot() -> Updater:
    if not TOKEN:
        raise RuntimeError("Saknar TELEGRAM_TOKEN i miljövariabler.")
    updater = Updater(token=TOKEN, use_context=True)
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
    dp.add_handler(CallbackQueryHandler(on_button))

    return updater

# starta telegrambot när uvicorn startar
@app.on_event("startup")
def _startup():
    # OBS: se till att bara EN instans körs (inte både lokal & Render samtidigt),
    # annars får du "Conflict: terminated by other getUpdates request".
    updater = build_bot()
    updater.start_polling()
    # sätt igång motor & keepalive standard
    with state_lock:
        state["engine"] = True
    # spara globalt chat_id första gången vi får ett /status; tills dess kör vi jobs utan context-chat
    # För att jobs ska kunna skicka notiser behöver de ett chat_id – vi löser så här:
    def set_chat(update: Update, ctx: CallbackContext):
        # ersätt/registrera jobs när första chatten kommer in
        chat_id = update.effective_chat.id
        # stoppa ev tidigare likadana jobb och sätt om dem med context
        for job in ctx.job_queue.jobs():
            try:
                job.schedule_removal()
            except Exception:
                pass
        scheduler_thread(updater, chat_id)
        # avregistrera denna engångshook
        updater.dispatcher.remove_handler(h_once)

        # svara direkt med status
        update.message.reply_text(status_text())

    global h_once
    h_once = CommandHandler("status", set_chat)  # fånga första status som "registrering"
    updater.dispatcher.add_handler(h_once, group=0)

# =========================
# ------- THE END ---------
# =========================
