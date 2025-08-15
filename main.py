# main.py
# Mp ORBbot – Candle-ORB (color flip), 1 trade/ORB, trailing SL per candle, long & short,
# Telegram-kommandon enligt spec, PnL, CSV-export, symbols/timeframe.
# Körs på Render via FastAPI (uvicorn) + Telegrambot i bakgrundstråd.
# Python 3.10+, python-telegram-bot >= 20

import os
import time
import csv
import json
import threading
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

# --- Web/HTTP for data ---
import requests
from requests.adapters import HTTPAdapter, Retry

# --- FastAPI för Render keep-alive (uvicorn main:app) ---
from fastapi import FastAPI

# --- Telegram bot ---
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
)

# ===========================
#      KONFIG & GLOBALS
# ===========================
BOT_NAME = "Mp ORBbot"

# Standard-symboler & timeframe (kan ändras med /symbols, /timeframe)
DEFAULT_SYMBOLS = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]
SUPPORTED_INTERVALS = {"1min", "3min", "5min", "15min", "30min", "1hour", "4hour"}
INTERVAL = "3min"

# Trading
DEFAULT_FEE = 0.001          # 0.1% per köp/sälj
TRADE_SIZE_USDT = 30.0       # mock trade storlek
ONLY_ONE_ENTRY_PER_ORB = True
TRAIL_ON_EACH_CANDLE = True
STOP_OFFSET = 0.0            # buffert (long: -, short: +)

# AI-lägen -> minsta breakout-buffert
MIN_ORB_PCT_NEUTRAL = 0.001
MIN_ORB_PCT_CAUTIOUS = 0.0015
MIN_ORB_PCT_AGGRESSIVE = 0.0006

# Heartbeat / failsafe
HEARTBEAT_SEC = 30
TELEGRAM_FAILSAFE_MIN = 5  # minuter

# Loggar & PnL
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"
PNL_ANCHOR_FILE = "pnl_anchor.json"

# API-nycklar
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")
KUCOIN_PUBLIC = "https://api.kucoin.com"

UTC = timezone.utc
app = FastAPI(title="Mp ORBbot")  # <— för Render/uvicorn

def now_ts() -> int:
    return int(time.time())

def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M:%S")

def today_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")

def ensure_csv_header(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","symbol","side","entry_price","exit_price","qty",
                "fee","pnl","mode","ai_mode","orb_id","note"
            ])

def http_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

SESSION = http_session()

# ===========================
#        DATA & AI-HELPERS
# ===========================
def ku_symbol(symbol: str) -> str:
    if "-" in symbol: return symbol
    base = symbol[:-4]; quote = symbol[-4:]
    return f"{base}-{quote}"

def fetch_kucoin_candles(symbol: str, start_ts: int, end_ts: int, interval: str) -> List[dict]:
    sym = ku_symbol(symbol.upper())
    url = f"{KUCOIN_PUBLIC}/api/v1/market/candles"
    params = {"type": interval, "symbol": sym, "startAt": start_ts, "endAt": end_ts}
    r = SESSION.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    candles = []
    # KuCoin: [ time, open, close, high, low, volume, turnover ] (reverse chrono)
    for row in reversed(data):
        ts = int(row[0])
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        candles.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
    return candles

def parse_symbols_arg(arg: str) -> List[str]:
    parts = [p.strip().upper().replace("-", "") for p in arg.split(",") if p.strip()]
    return [p if p.endswith("USDT") else p+"USDT" for p in parts]

def is_doji(c) -> bool:
    body = abs(c["close"] - c["open"])
    rng = max(1e-12, c["high"] - c["low"])
    return (body / rng) < 0.1

def bullish(c) -> bool:
    return c["close"] > c["open"]

def bearish(c) -> bool:
    return c["close"] < c["open"]

def candle_color(c) -> str:
    if bullish(c): return "green"
    if bearish(c): return "red"
    return "doji"

def min_orb_pct_for_mode(mode: str) -> float:
    m = mode.lower()
    if m == "försiktig": return MIN_ORB_PCT_CAUTIOUS
    if m == "aggressiv": return MIN_ORB_PCT_AGGRESSIVE
    return MIN_ORB_PCT_NEUTRAL

# ===========================
#            ORB
# ===========================
class ORBWindow:
    def __init__(self, start_ts: int, high: float, low: float, direction: str):
        self.start_ts = start_ts
        self.high = high
        self.low = low
        self.direction = direction  # "bullish" (red->green) / "bearish" (green->red)
        self.id = f"{start_ts}"

def orb_on_color_flip(prev_c: dict, c: dict) -> Optional[ORBWindow]:
    prev_col = candle_color(prev_c)
    col = candle_color(c)
    if prev_col in ("red","green") and col in ("red","green") and col != prev_col:
        direction = "bullish" if col == "green" else "bearish"
        return ORBWindow(c["ts"], c["high"], c["low"], direction)
    return None

# ===========================
#           STATE
# ===========================
class SymbolState:
    def __init__(self):
        self.in_position = False
        self.side = None             # "long" | "short"
        self.entry_price = None
        self.qty = 0.0
        self.stop = None
        self.orb: Optional[ORBWindow] = None
        self.has_entered_this_orb = False
        self.trades_today = 0

def build_state(symbols: List[str]) -> Dict[str, SymbolState]:
    return {s: SymbolState() for s in symbols}

SYMBOLS = DEFAULT_SYMBOLS.copy()
STATE: Dict[str, SymbolState] = build_state(SYMBOLS)

CONFIG = {
    "ai_mode": "neutral",
    "mock_mode": True,          # mock som standard
    "is_trading": False,        # motor på/av
    "min_orb_pct": min_orb_pct_for_mode("neutral"),
    "last_heartbeat": now_ts(),
    "chat_id": None,
}

# ===========================
#        LOGGAR & PNL
# ===========================
def log_trade_row(path: str, row: List):
    ensure_csv_header(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def log_trade(side: str, symbol: str, entry_price: float, exit_price: float, qty: float,
              fee: float, pnl: float, note: str):
    path = MOCK_LOG if CONFIG["mock_mode"] else REAL_LOG
    row = [
        fmt_ts(now_ts()), symbol, side,
        f"{entry_price:.8f}", f"{exit_price:.8f}",
        f"{qty:.8f}", f"{fee:.6f}", f"{pnl:.6f}",
        "mock" if CONFIG["mock_mode"] else "live",
        CONFIG["ai_mode"],
        STATE[symbol].orb.id if STATE[symbol].orb else "",
        note
    ]
    log_trade_row(path, row)

def load_pnl_anchor() -> Dict[str,float]:
    if not os.path.exists(PNL_ANCHOR_FILE):
        return {"mock":0.0, "live":0.0, "date": today_str()}
    try:
        with open(PNL_ANCHOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") != today_str():
            return {"mock":0.0, "live":0.0, "date": today_str()}
        return data
    except Exception:
        return {"mock":0.0, "live":0.0, "date": today_str()}

def save_pnl_anchor(mock_total: float, live_total: float):
    with open(PNL_ANCHOR_FILE, "w", encoding="utf-8") as f:
        json.dump({"mock":mock_total, "live":live_total, "date": today_str()}, f)

def read_pnl_for_today(paths: List[str]) -> Dict[str, float]:
    totals = {"mock": 0.0, "live": 0.0}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if (row.get("timestamp","")[:10]) == today_str():
                    try:
                        mode = row.get("mode","mock").strip().lower()
                        pnl = float(row.get("pnl","0") or 0)
                        totals[mode] += pnl
                    except Exception:
                        pass
    anchor = load_pnl_anchor()
    totals["mock"] -= anchor.get("mock",0.0)
    totals["live"] -= anchor.get("live",0.0)
    return totals

# ===========================
#       TRADINGLOGIK
# ===========================
def update_trailing_stop(symbol: str, prev_candle: dict):
    st = STATE[symbol]
    if not st.in_position or not TRAIL_ON_EACH_CANDLE:
        return
    if st.side == "long":
        new_stop = prev_candle["low"] - STOP_OFFSET
        st.stop = new_stop if st.stop is None else max(st.stop, new_stop)
    elif st.side == "short":
        new_stop = prev_candle["high"] + STOP_OFFSET
        st.stop = new_stop if st.stop is None else min(st.stop, new_stop)

def compute_qty(price: float) -> float:
    if price <= 0: return 0.0
    return round(TRADE_SIZE_USDT / price, 6)

def ai_allows_entry(c: dict, direction: str) -> bool:
    mode = CONFIG["ai_mode"]
    if mode == "försiktig":
        if is_doji(c): return False
        if direction == "bullish" and not bullish(c): return False
        if direction == "bearish" and not bearish(c): return False
    return True

def maybe_exit_by_stop(symbol: str, last_price: float, fee_rate: float=DEFAULT_FEE):
    st = STATE[symbol]
    if not st.in_position or st.stop is None:
        return
    hit = (st.side == "long" and last_price <= st.stop) or (st.side == "short" and last_price >= st.stop)
    if not hit: return

    entry = st.entry_price
    exitp = st.stop
    qty = st.qty
    fee = (entry * qty * fee_rate) + (abs(exitp) * qty * fee_rate)
    pnl = (exitp - entry) * qty - fee if st.side == "long" else (entry - exitp) * qty - fee
    note = "stop hit"
    side_label = "LONG->SL" if st.side == "long" else "SHORT->SL"

    st.in_position = False
    st.entry_price = None
    st.qty = 0.0
    st.stop = None
    st.trades_today += 1
    st.side = None

    log_trade(side_label, symbol, entry, exitp, qty, fee, pnl, note)

def try_breakout_entry(symbol: str, candles: List[dict], min_orb_pct: float) -> Optional[Tuple[str,float,float]]:
    st = STATE[symbol]
    if not st.orb or st.in_position or (ONLY_ONE_ENTRY_PER_ORB and st.has_entered_this_orb):
        return None
    c = candles[-1]
    if st.orb.direction == "bullish":
        if c["close"] > st.orb.high * (1.0 + min_orb_pct) and ai_allows_entry(c, "bullish"):
            return ("long", c["close"], st.orb.low - STOP_OFFSET)
    else:
        if c["close"] < st.orb.low * (1.0 - min_orb_pct) and ai_allows_entry(c, "bearish"):
            return ("short", c["close"], st.orb.high + STOP_OFFSET)
    return None

def send_bot_message(context: ContextTypes.DEFAULT_TYPE, text: str):
    try:
        if CONFIG["chat_id"] is not None:
            context.bot.send_message(chat_id=CONFIG["chat_id"], text=text)
            CONFIG["last_heartbeat"] = now_ts()
    except Exception:
        # failsafe: tappa Telegram > 5 min => stoppa motor
        if now_ts() - CONFIG["last_heartbeat"] > TELEGRAM_FAILSAFE_MIN * 60:
            CONFIG["is_trading"] = False

# ===========================
#         MOTOR / TICK
# ===========================
def trading_tick(interval: str):
    notes = []
    if not SYMBOLS: return notes
    for symbol in SYMBOLS:
        try:
            end_ts = now_ts()
            start_ts = end_ts - 3 * 3600
            candles = fetch_kucoin_candles(symbol, start_ts, end_ts, interval)
            if len(candles) < 6:  # kräver minst två senaste + lite historik
                continue
            st = STATE[symbol]

            prev = candles[-2]
            last = candles[-1]

            # 1) ORB vid färgskifte
            new_orb = orb_on_color_flip(prev, last)
            if new_orb:
                st.orb = new_orb
                st.has_entered_this_orb = False

            # 2) Trailing
            update_trailing_stop(symbol, prev)

            # 3) Stop?
            maybe_exit_by_stop(symbol, last["close"], DEFAULT_FEE)

            # 4) Entry (om motor på)
            if CONFIG["is_trading"]:
                br = try_breakout_entry(symbol, candles, CONFIG["min_orb_pct"])
                if br:
                    side, price, stop_init = br
                    qty = compute_qty(price)
                    if qty <= 0: 
                        continue
                    st.in_position = True
                    st.side = side
                    st.entry_price = price
                    st.qty = qty
                    st.stop = stop_init
                    st.has_entered_this_orb = True
                    notes.append(f"✅ {'MOCK' if CONFIG['mock_mode'] else 'LIVE'}: {side.upper()} {symbol} @ {price:.6f} x {qty} | SL={stop_init:.6f}")

        except Exception as e:
            print("trading_tick error:", e)
            traceback.print_exc()
    return notes

def run_trading_job():
    async def job(context: ContextTypes.DEFAULT_TYPE):
        notes = trading_tick(INTERVAL)
        for n in notes:
            send_bot_message(context, n)
    return job

# ===========================
#         TELEGRAM
# ===========================
HELP_TXT = (
    "Kommandon:\n"
    "/status – visa status\n"
    "/set_ai <neutral|aggressiv|försiktig>\n"
    "/start_mock – starta mock (svara JA)\n"
    "/start_live – starta LIVE (svara JA)\n"
    "/engine_start – starta motor\n"
    "/engine_stop – stoppa motor\n"
    "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
    "/timeframe 1min – byt tidsram (1min,3min,5min,15min,30min,1hour,4hour)\n"
    "/export_csv – skicka loggar\n"
    "/pnl – visa dagens PnL\n"
    "/reset_pnl – nollställ PnL\n"
    "/help – denna hjälp\n"
)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TXT)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["chat_id"] = update.effective_chat.id
    totals = read_pnl_for_today([MOCK_LOG, REAL_LOG])
    lines = [
        f"Bot: {BOT_NAME}",
        f"Läge: {'MOCK' if CONFIG['mock_mode'] else 'LIVE'}",
        f"AI: {CONFIG['ai_mode']} (min ORB {CONFIG['min_orb_pct']*100:.3f}%)",
        f"Motor: {'ON' if CONFIG['is_trading'] else 'OFF'}",
        f"Timeframe: {INTERVAL}",
        f"Symbols: {', '.join(SYMBOLS)}",
        f"PnL idag → MOCK: {totals['mock']:.4f}  |  LIVE: {totals['live']:.4f}",
    ]
    for s in SYMBOLS:
        st = STATE[s]
        lines.append(
            f"{s}: pos={'Y' if st.in_position else 'N'} side={st.side} "
            f"entry={st.entry_price} stop={st.stop} trades idag={st.trades_today} "
            f"ORB={'Y' if st.orb else 'N'} {('('+st.orb.direction+')' if st.orb else '')}"
        )
    await update.message.reply_text("\n".join(lines))

async def set_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /set_ai neutral|aggressiv|försiktig")
        return
    mode = context.args[0].lower()
    if mode not in ["neutral","aggressiv","försiktig"]:
        await update.message.reply_text("Fel läge. Välj: neutral, aggressiv, försiktig.")
        return
    CONFIG["ai_mode"] = mode
    CONFIG["min_orb_pct"] = min_orb_pct_for_mode(mode)
    await update.message.reply_text(f"AI-läge satt till {mode}. Min ORB: {CONFIG['min_orb_pct']*100:.3f}%")

async def start_mock_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["chat_id"] = update.effective_chat.id
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("JA – Starta MOCK", callback_data="confirm_start_mock")],
        [InlineKeyboardButton("AVBRYT", callback_data="cancel_start_mock")]
    ])
    await update.message.reply_text("Starta MOCK-läge?", reply_markup=kb)

async def start_live_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["chat_id"] = update.effective_chat.id
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("JA – Starta LIVE", callback_data="confirm_start_live")],
        [InlineKeyboardButton("AVBRYT", callback_data="cancel_start_live")]
    ])
    await update.message.reply_text("Starta LIVE-läge?", reply_markup=kb)

async def confirm_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "confirm_start_mock":
        CONFIG["mock_mode"] = True
        await q.edit_message_text("MOCK-läge aktiverat ✅")
    elif q.data == "cancel_start_mock":
        await q.edit_message_text("Mock-start avbruten.")
    elif q.data == "confirm_start_live":
        CONFIG["mock_mode"] = False
        await q.edit_message_text("LIVE-läge aktiverat ⚠️")
    elif q.data == "cancel_start_live":
        await q.edit_message_text("Live-start avbruten.")

async def engine_start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["chat_id"] = update.effective_chat.id
    CONFIG["is_trading"] = True
    await update.message.reply_text("Motor startad ✅ (trading ON).")

async def engine_stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["is_trading"] = False
    await update.message.reply_text("Motor stoppad ⛔ (trading OFF).")

async def symbols_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,...")
        return
    new_list = parse_symbols_arg(" ".join(context.args))
    if not new_list:
        await update.message.reply_text("Kunde inte tolka symbol-listan.")
        return
    global SYMBOLS, STATE
    SYMBOLS = new_list
    STATE = build_state(SYMBOLS)
    await update.message.reply_text(f"Symbols uppdaterade: {', '.join(SYMBOLS)}")

async def timeframe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /timeframe 1min (t.ex. 1min,3min,5min,15min,30min,1hour,4hour)")
        return
    tf = context.args[0].lower()
    if tf not in SUPPORTED_INTERVALS:
        await update.message.reply_text(f"Ogiltig timeframe. Tillåtna: {', '.join(sorted(SUPPORTED_INTERVALS))}")
        return
    global INTERVAL
    INTERVAL = tf
    await update.message.reply_text(f"Timeframe satt till {INTERVAL}")

async def export_csv_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sent = False
    if os.path.exists(MOCK_LOG):
        await update.message.reply_document(open(MOCK_LOG, "rb"), filename=MOCK_LOG, caption="Mock-logg")
        sent = True
    if os.path.exists(REAL_LOG):
        await update.message.reply_document(open(REAL_LOG, "rb"), filename=REAL_LOG, caption="Live-logg")
        sent = True
    if not sent:
        await update.message.reply_text("Inga loggar hittades ännu.")

async def pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    totals = read_pnl_for_today([MOCK_LOG, REAL_LOG])
    await update.message.reply_text(
        f"Dagens PnL\nMOCK: {totals['mock']:.4f} USDT\nLIVE: {totals['live']:.4f} USDT"
    )

async def reset_pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Sätt ankare = nuvarande totals (råa, utan tidigare ankare)
    mock_raw = live_raw = 0.0
    for path in [MOCK_LOG, REAL_LOG]:
        if not os.path.exists(path): continue
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if (row.get("timestamp","")[:10]) == today_str():
                    pnl = float(row.get("pnl","0") or 0)
                    if row.get("mode","mock").strip().lower() == "mock":
                        mock_raw += pnl
                    else:
                        live_raw += pnl
    save_pnl_anchor(mock_raw, live_raw)
    await update.message.reply_text("PnL nollställt för idag (ankare satt).")

async def heartbeat_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        if CONFIG["chat_id"] and CONFIG["is_trading"]:
            await context.bot.send_chat_action(chat_id=CONFIG["chat_id"], action="typing")
            CONFIG["last_heartbeat"] = now_ts()
    except Exception:
        if now_ts() - CONFIG["last_heartbeat"] > TELEGRAM_FAILSAFE_MIN * 60:
            CONFIG["is_trading"] = False

# ===========================
#   STARTA TELEGRAM-BOTEN
# ===========================
def build_telegram_app():
    app_tg = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app_tg.add_handler(CommandHandler("help", help_cmd))
    app_tg.add_handler(CommandHandler("status", status_cmd))
    app_tg.add_handler(CommandHandler("set_ai", set_ai_cmd))

    app_tg.add_handler(CommandHandler("start_mock", start_mock_cmd))
    app_tg.add_handler(CommandHandler("start_live", start_live_cmd))
    app_tg.add_handler(CallbackQueryHandler(
        confirm_cb,
        pattern="^(confirm_start_mock|cancel_start_mock|confirm_start_live|cancel_start_live)$"
    ))

    app_tg.add_handler(CommandHandler("engine_start", engine_start_cmd))
    app_tg.add_handler(CommandHandler("engine_stop", engine_stop_cmd))

    app_tg.add_handler(CommandHandler("symbols", symbols_cmd))
    app_tg.add_handler(CommandHandler("timeframe", timeframe_cmd))

    app_tg.add_handler(CommandHandler("export_csv", export_csv_cmd))
    app_tg.add_handler(CommandHandler("pnl", pnl_cmd))
    app_tg.add_handler(CommandHandler("reset_pnl", reset_pnl_cmd))

    # Repeating jobs
    app_tg.job_queue.run_repeating(run_trading_job(), interval=10, first=5)
    app_tg.job_queue.run_repeating(heartbeat_job, interval=HEARTBEAT_SEC, first=5)

    return app_tg

def start_bot_thread():
    # Skapa loggfiler om de saknas
    ensure_csv_header(MOCK_LOG)
    ensure_csv_header(REAL_LOG)
    # init PnL-ankare för dagens datum vid behov
    anc = load_pnl_anchor()
    if anc.get("date") != today_str():
        save_pnl_anchor(0.0, 0.0)

    app_tg = build_telegram_app()
    print(f"{BOT_NAME} startar Telegram-motor...")
    app_tg.run_polling(close_loop=False)

# ===========================
#    FASTAPI (Render)
# ===========================
@app.get("/")
def root():
    return {"status": "Mp ORBbot running", "timeframe": INTERVAL, "symbols": SYMBOLS, "mode": "mock" if CONFIG["mock_mode"] else "live"}

@app.on_event("startup")
def on_startup():
    # Starta Telegram-boten i en bakgrundstråd när uvicorn startar
    threading.Thread(target=start_bot_thread, daemon=True).start()
