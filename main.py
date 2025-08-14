# main.py
# Mp ORBbot – Candle-ORB, 1 trade per ORB, trailing SL per candle, long & short, Telegram, backtest & CSV
# Python 3.10+, python-telegram-bot >= 20

import os
import time
import csv
import math
import threading
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from http.server import BaseHTTPRequestHandler, HTTPServer  # Render keep-alive

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
)

# ========== Konfig ==========
BOT_NAME = "Mp ORBbot"
SYMBOLS = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]
INTERVAL = "3min"  # KuCoin candle-typ

# Tradingparametrar
DEFAULT_FEE = 0.001           # 0.1 % per köp/sälj
TRADE_SIZE_USDT = 30.0        # mocktrade storlek
ONLY_ONE_ENTRY_PER_ORB = True
TRAIL_ON_EACH_CANDLE = True
STOP_OFFSET = 0.0             # ev. buffert: long -> -offset, short -> +offset

# AI-lägen (påverkar minsta breakout-buffert + filter)
MIN_ORB_PCT_NEUTRAL = 0.001      # 0.10 %
MIN_ORB_PCT_CAUTIOUS = 0.0015    # 0.15 %
MIN_ORB_PCT_AGGRESSIVE = 0.0006  # 0.06 %

# Heartbeat / failsafe
HEARTBEAT_SEC = 30
TELEGRAM_FAILSAFE_MIN = 5

# Filnamn
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"
BACKTEST_EXPORT = "backtest_export.csv"

# Tokens & nycklar
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")
KUCOIN_PUBLIC = "https://api.kucoin.com"

# ========== Hjälpfunktioner ==========
UTC = timezone.utc

def now_ts() -> int:
    return int(time.time())

def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M:%S")

def ensure_csv_header(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp","symbol","side","entry_price","exit_price","qty",
                "fee","pnl","mode","ai_mode","orb_id","note"
            ])

def http_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

SESSION = http_session()

def ku_symbol(symbol: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    if "-" in symbol: return symbol
    base = symbol[:-4]; quote = symbol[-4:]
    return f"{base}-{quote}"

def fetch_kucoin_candles(symbol: str, start_ts: int, end_ts: int, interval: str = INTERVAL) -> List[dict]:
    sym = ku_symbol(symbol.upper())
    url = f"{KUCOIN_PUBLIC}/api/v1/market/candles"
    params = {"type": interval, "symbol": sym, "startAt": start_ts, "endAt": end_ts}
    r = SESSION.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    candles = []
    # KuCoin ger reverse-chronological: [ time, open, close, high, low, volume, turnover ]
    for row in reversed(data):
        ts = int(row[0])
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        candles.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
    return candles

def parse_period(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("d"): return int(s[:-1]) * 86400
    if s.endswith("h"): return int(s[:-1]) * 3600
    if s.endswith("m"): return int(s[:-1]) * 60
    return int(s)  # sekunder

def safe_float(x, default=None):
    try: return float(x)
    except Exception: return default

# ========== Candlestick-helpers & AI-filter ==========
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

# ========== ORB (candle-färgskifte) ==========
class ORBWindow:
    def __init__(self, start_ts: int, high: float, low: float, direction: str):
        self.start_ts = start_ts
        self.high = high
        self.low = low
        self.direction = direction  # "bullish" (röd->grön) eller "bearish" (grön->röd)
        self.id = f"{start_ts}"

def orb_on_color_flip(prev_c: dict, c: dict) -> Optional[ORBWindow]:
    prev_col = candle_color(prev_c)
    col = candle_color(c)
    if prev_col in ("red","green") and col in ("red","green") and col != prev_col:
        direction = "bullish" if col == "green" else "bearish"
        # ORB definieras av SJÄLVA flip-candlen (dvs 'c')
        return ORBWindow(c["ts"], c["high"], c["low"], direction)
    return None

# ========== State ==========
class SymbolState:
    def __init__(self):
        self.in_position = False
        self.side = None             # "long" eller "short"
        self.entry_price = None
        self.qty = 0.0
        self.stop = None
        self.orb: Optional[ORBWindow] = None
        self.has_entered_this_orb = False
        self.trades_today = 0

STATE: Dict[str, SymbolState] = {s: SymbolState() for s in SYMBOLS}

CONFIG = {
    "ai_mode": "neutral",          # 'aggressiv' | 'neutral' | 'försiktig'
    "mock_mode": True,             # default mock
    "is_trading": False,           # kräver /start + bekräftelse
    "min_orb_pct": MIN_ORB_PCT_NEUTRAL,
    "last_heartbeat": now_ts(),
    "chat_id": None,
}

# ========== Logging ==========
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

# ========== Tradinglogik ==========
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
        if is_doji(c):
            return False
        if direction == "bullish" and not bullish(c):
            return False
        if direction == "bearish" and not bearish(c):
            return False
    # 'neutral' och 'aggressiv' släpper igenom (aggressiv har bara lägre min_orb_pct)
    return True

def maybe_exit_by_stop(symbol: str, last_price: float, fee_rate: float=DEFAULT_FEE):
    st = STATE[symbol]
    if not st.in_position or st.stop is None:
        return
    hit = False
    if st.side == "long" and last_price <= st.stop:
        hit = True
    elif st.side == "short" and last_price >= st.stop:
        hit = True
    if hit:
        entry = st.entry_price
        exitp = st.stop
        qty = st.qty
        fee = (entry * qty * fee_rate) + (abs(exitp) * qty * fee_rate)
        pnl = (exitp - entry) * qty - fee if st.side == "long" else (entry - exitp) * qty - fee
        note = "stop hit"
        st.in_position = False
        st.entry_price = None
        st.qty = 0.0
        st.stop = None
        side_label = "LONG->SL" if st.side == "long" else "SHORT->SL"
        st.side = None
        st.trades_today += 1
        log_trade(side_label, symbol, entry, exitp, qty, fee, pnl, note)

def try_breakout_entry(symbol: str, candles: List[dict]) -> Optional[Tuple[str,float,float]]:
    """Returnerar (side, entry_price, initial_stop) om entry ska göras."""
    st = STATE[symbol]
    if not st.orb:
        return None
    if st.in_position:
        return None
    if ONLY_ONE_ENTRY_PER_ORB and st.has_entered_this_orb:
        return None

    c = candles[-1]
    m = 1.0 + CONFIG["min_orb_pct"]
    if st.orb.direction == "bullish":
        # Breakout upp: stängning över ORB-high * (1+min_orb_pct)
        if c["close"] > st.orb.high * m and ai_allows_entry(c, "bullish"):
            entry = c["close"]
            initial_stop = st.orb.low - STOP_OFFSET  # första SL = ORB low
            return ("long", entry, initial_stop)
    else:
        # Breakout ned: stängning under ORB-low * (1-min_orb_pct)
        if c["close"] < st.orb.low * (1.0 - CONFIG["min_orb_pct"]) and ai_allows_entry(c, "bearish"):
            entry = c["close"]
            initial_stop = st.orb.high + STOP_OFFSET  # första SL = ORB high
            return ("short", entry, initial_stop)
    return None

def place_order_mock(symbol: str, side: str, price: float, qty: float):
    # Mock-order: vi loggar vid exit för PnL; notiser skickas separat.
    pass

def place_order_live(symbol: str, side: str, price: float, qty: float):
    # Placeholder – lägg in KuCoin live-order om/när du vill.
    pass

def send_bot_message(context: ContextTypes.DEFAULT_TYPE, text: str):
    try:
        if CONFIG["chat_id"] is not None:
            context.bot.send_message(chat_id=CONFIG["chat_id"], text=text)
            CONFIG["last_heartbeat"] = now_ts()
    except Exception:
        if now_ts() - CONFIG["last_heartbeat"] > TELEGRAM_FAILSAFE_MIN * 60:
            CONFIG["is_trading"] = False

# ========== Körloop ==========
def trading_tick():
    """Hämtar data och uppdaterar samtliga symboler; returnerar lista med notiser per symbol."""
    notes = []
    for symbol in SYMBOLS:
        try:
            end_ts = now_ts()
            start_ts = end_ts - 3 * 3600  # 3 timmar historik
            candles = fetch_kucoin_candles(symbol, start_ts, end_ts, INTERVAL)
            if len(candles) < 6:
                continue
            st = STATE[symbol]

            # 1) Ny ORB om färgskifte på senaste candle
            prev = candles[-2]
            last = candles[-1]
            new_orb = orb_on_color_flip(prev, last)
            if new_orb:
                st.orb = new_orb
                st.has_entered_this_orb = False

            # 2) Trailing stop på varje NY candle (föregående candle)
            update_trailing_stop(symbol, prev)

            # 3) Stop-check
            maybe_exit_by_stop(symbol, last["close"], DEFAULT_FEE)

            # 4) Entry (bara en per ORB)
            if CONFIG["is_trading"]:
                br = try_breakout_entry(symbol, candles)
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

                    if CONFIG["mock_mode"]:
                        place_order_mock(symbol, side.upper(), price, qty)
                    else:
                        place_order_live(symbol, side.upper(), price, qty)

                    notes.append(f"✅ {'MOCK' if CONFIG['mock_mode'] else 'LIVE'}: {side.upper()} {symbol} @ {price:.6f} x {qty} | SL={stop_init:.6f}")

        except Exception as e:
            print("trading_tick error:", e)
            traceback.print_exc()
    return notes

def run_trading_job():
    async def job(context: ContextTypes.DEFAULT_TYPE):
        notes = trading_tick()
        for n in notes:
            send_bot_message(context, n)
    return job

# ========== Backtest ==========
def backtest_symbol(symbol: str, period: str, fee_rate: float, ai_mode: str) -> Dict:
    secs = parse_period(period)
    end_ts = now_ts()
    start_ts = end_ts - secs
    candles = fetch_kucoin_candles(symbol, start_ts, end_ts, INTERVAL)
    if len(candles) < 15:
        return {"symbol": symbol, "trades": 0, "pnl": 0.0, "details": []}

    min_orb = min_orb_pct_for_mode(ai_mode)
    pnl_total = 0.0
    details = []

    in_pos = False
    side = None
    entry = qty = stop = None
    has_entered_this_orb = False
    orb: Optional[ORBWindow] = None

    for i in range(1, len(candles)):
        cprev = candles[i-1]
        c = candles[i]

        # ORB på färgskifte
        new_orb = orb_on_color_flip(cprev, c)
        if new_orb:
            orb = new_orb
            has_entered_this_orb = False

        # Trailing på varje nytt candle
        if in_pos and TRAIL_ON_EACH_CANDLE:
            if side == "long":
                stop = max(stop, cprev["low"] - STOP_OFFSET)
            else:
                stop = min(stop, cprev["high"] + STOP_OFFSET)

        # Stop?
        if in_pos:
            hit = (side == "long" and c["close"] <= stop) or (side == "short" and c["close"] >= stop)
            if hit:
                fee = (entry * qty * fee_rate) + (abs(stop) * qty * fee_rate)
                pnl = (stop - entry) * qty - fee if side == "long" else (entry - stop) * qty - fee
                pnl_total += pnl
                details.append({
                    "ts": c["ts"], "event": f"{side.upper()}->SL", "side": side,
                    "entry": entry, "exit": stop, "qty": qty, "fee": fee, "pnl": pnl
                })
                in_pos = False
                side = None
                entry = qty = stop = None
                continue

        # Entry?
        if orb and (not in_pos) and (not (ONLY_ONE_ENTRY_PER_ORB and has_entered_this_orb)):
            allow = True
            if orb.direction == "bullish":
                trigger = c["close"] > orb.high * (1.0 + min_orb)
                if ai_mode == "försiktig":
                    allow = (not is_doji(c)) and bullish(c)
                if trigger and allow:
                    side = "long"
                    entry = c["close"]
                    qty = TRADE_SIZE_USDT / entry
                    stop = orb.low - STOP_OFFSET
                    in_pos = True
                    has_entered_this_orb = True
                    details.append({"ts": c["ts"], "event": "BUY", "side": side, "entry": entry, "qty": qty, "stop_init": stop})
            else:
                trigger = c["close"] < orb.low * (1.0 - min_orb)
                if ai_mode == "försiktig":
                    allow = (not is_doji(c)) and bearish(c)
                if trigger and allow:
                    side = "short"
                    entry = c["close"]
                    qty = TRADE_SIZE_USDT / entry
                    stop = orb.high + STOP_OFFSET
                    in_pos = True
                    has_entered_this_orb = True
                    details.append({"ts": c["ts"], "event": "SELL", "side": side, "entry": entry, "qty": qty, "stop_init": stop})

    # Exportera till CSV – EN fil med både long & short
    with open(BACKTEST_EXPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts","event","side","entry","exit","qty","fee","pnl"])
        for d in details:
            w.writerow([
                fmt_ts(d["ts"]),
                d.get("event",""),
                d.get("side",""),
                f"{d.get('entry','')}",
                f"{d.get('exit','')}",
                f"{d.get('qty','')}",
                f"{d.get('fee','')}",
                f"{d.get('pnl','')}",
            ])

    return {
        "symbol": symbol,
        "trades": sum(1 for d in details if "->SL" in d.get("event","")),
        "pnl": pnl_total,
        "details": details
    }

# ========== Telegram ==========
HELP_TXT = (
    "Kommandon:\n"
    "/start – starta trading (mock som standard, kräver bekräftelse)\n"
    "/stop – stoppa trading\n"
    "/status – visa status (AI-läge, mock/live, ORB, positioner)\n"
    "/set_ai <aggressiv|neutral|försiktig> – AI-läge och breakout-buffert\n"
    "/set_live – byt till LIVE (med bekräftelse)\n"
    "/mock_trade <SYMBOL> [long|short] – skapar testposition med initial SL på ORB-linje\n"
    "/backtest <SYMBOL> <PERIOD> [avgift] – ex: /backtest btcusdt 3d 0.001\n"
    "/export_csv – exporterar senaste backtest till CSV (long & short i samma fil)\n"
    "/help – visar denna hjälp\n"
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["chat_id"] = update.effective_chat.id
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("JA – Starta trading", callback_data="confirm_start")],
        [InlineKeyboardButton("AVBRYT", callback_data="cancel_start")]
    ])
    await update.message.reply_text(
        "Vill du starta trading?\nLäge: MOCK (default). Byt till live med /set_live senare.",
        reply_markup=kb
    )

async def start_confirm_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "confirm_start":
        CONFIG["is_trading"] = True
        await q.edit_message_text("Trading är igång ✅ (mock).")
    else:
        await q.edit_message_text("Start avbruten.")

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["is_trading"] = False
    await update.message.reply_text("Trading stoppad. Kommandon fungerar fortfarande.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["chat_id"] = update.effective_chat.id
    lines = [
        f"Bot: {BOT_NAME}",
        f"Mode: {'MOCK' if CONFIG['mock_mode'] else 'LIVE'}",
        f"AI-läge: {CONFIG['ai_mode']}",
        f"Trading: {'ON' if CONFIG['is_trading'] else 'OFF'}",
        f"Min ORB: {CONFIG['min_orb_pct']*100:.4f}%"
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
    if len(context.args) < 1:
        await update.message.reply_text("Använd: /set_ai aggressiv|neutral|försiktig")
        return
    mode = context.args[0].lower()
    if mode not in ["aggressiv","neutral","försiktig"]:
        await update.message.reply_text("Fel läge. Välj: aggressiv, neutral, försiktig.")
        return
    CONFIG["ai_mode"] = mode
    CONFIG["min_orb_pct"] = min_orb_pct_for_mode(mode)
    await update.message.reply_text(f"AI-läge satt till {mode}. Min ORB: {CONFIG['min_orb_pct']*100:.4f}%")

async def set_live_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("JA – Byt till LIVE", callback_data="confirm_live")],
        [InlineKeyboardButton("Avbryt", callback_data="cancel_live")],
    ])
    await update.message.reply_text("Är du säker på att du vill byta till LIVE-läge?", reply_markup=kb)

async def live_confirm_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "confirm_live":
        CONFIG["mock_mode"] = False
        await q.edit_message_text("LIVE-läge aktiverat ⚠️")
    else:
        await q.edit_message_text("Live-bytet avbröts.")

async def mock_trade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /mock_trade SYMBOL [long|short]
    if len(context.args) < 1:
        await update.message.reply_text("Använd: /mock_trade SYMBOL [long|short]")
        return
    symbol = context.args[0].upper()
    side = context.args[1].lower() if len(context.args) > 1 else "long"
    if symbol not in SYMBOLS:
        await update.message.reply_text(f"Symbolen {symbol} måste vara en av: {', '.join(SYMBOLS)}")
        return
    if side not in ["long","short"]:
        await update.message.reply_text("Sida måste vara long eller short.")
        return

    end_ts = now_ts()
    candles = fetch_kucoin_candles(symbol, end_ts-1800, end_ts, INTERVAL)
    if len(candles) < 3:
        await update.message.reply_text("För lite data för mock_trade.")
        return

    # Sätt en ORB från senaste färgskifte om finns, annars använd senaste candle som referens
    orb = None
    for i in range(len(candles)-2, 0, -1):
        orb = orb_on_color_flip(candles[i-1], candles[i])
        if orb:
            break
    st = STATE[symbol]
    st.orb = orb

    price = candles[-1]["close"]
    qty = compute_qty(price)
    st.in_position = True
    st.side = side
    st.entry_price = price
    st.qty = qty
    if side == "long":
        st.stop = (orb.low if orb else candles[-2]["low"]) - STOP_OFFSET
    else:
        st.stop = (orb.high if orb else candles[-2]["high"]) + STOP_OFFSET
    st.has_entered_this_orb = True
    await update.message.reply_text(f"✅ MOCK: {side.UPPER()} {symbol} @ {price:.6f} x {qty} | SL={st.stop:.6f}")

async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Använd: /backtest SYMBOL PERIOD [avgift]\nEx: /backtest btcusdt 3d 0.001")
        return
    symbol = context.args[0].upper()
    period = context.args[1]
    fee = DEFAULT_FEE if len(context.args) < 3 else safe_float(context.args[2], DEFAULT_FEE)
    await update.message.reply_text(f"Kör backtest {symbol} {period} (fee={fee}) ...")
    try:
        res = backtest_symbol(symbol, period, fee, CONFIG["ai_mode"])
        await update.message.reply_text(
            f"Backtest klart: {symbol}\nTrades: {res['trades']}\nPnL: {res['pnl']:.4f} USDT\n"
            f"CSV sparad till {BACKTEST_EXPORT} (hämta via /export_csv)."
        )
    except Exception as e:
        await update.message.reply_text(f"Backtest fel: {e}")

async def export_csv_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(BACKTEST_EXPORT):
        await update.message.reply_text("Ingen backtestexport hittades.")
        return
    await update.message.reply_document(document=open(BACKTEST_EXPORT, "rb"),
                                        filename=BACKTEST_EXPORT,
                                        caption="Backtest CSV (long & short)")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TXT)

# Heartbeat
async def heartbeat_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        if CONFIG["chat_id"] and CONFIG["is_trading"]:
            await context.bot.send_chat_action(chat_id=CONFIG["chat_id"], action="typing")
            CONFIG["last_heartbeat"] = now_ts()
    except Exception:
        if now_ts() - CONFIG["last_heartbeat"] > TELEGRAM_FAILSAFE_MIN * 60:
            CONFIG["is_trading"] = False

# ========== Render keep-alive HTTP server ==========
def start_keepalive_http():
    """Startar en minimal HTTP-server (PORT env) så Render-webprocesser inte somnar."""
    port = int(os.getenv("PORT", "8000"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")

        def log_message(self, format, *args):
            # tysta serverns egna logs
            return

    server = HTTPServer(("", port), Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    print(f"[keep-alive] HTTP server lyssnar på :{port}")

# ========== Main ==========
def main():
    # Starta Render keep-alive (gör inget om PORT inte används i en worker)
    start_keepalive_http()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(start_confirm_cb, pattern="^confirm_start|cancel_start$"))

    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("set_ai", set_ai_cmd))
    app.add_handler(CommandHandler("set_live", set_live_cmd))
    app.add_handler(CallbackQueryHandler(live_confirm_cb, pattern="^confirm_live|cancel_live$"))

    app.add_handler(CommandHandler("mock_trade", mock_trade_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("export_csv", export_csv_cmd))
    app.add_handler(CommandHandler("help", help_cmd))

    # Trading-jobb var 10:e sekund
    app.job_queue.run_repeating(run_trading_job(), interval=10, first=5)
    # Heartbeat
    app.job_queue.run_repeating(heartbeat_job, interval=HEARTBEAT_SEC, first=5)

    print(f"{BOT_NAME} startar...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
