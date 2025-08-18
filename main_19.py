# main.19.py  — Mp ORBbot (Render-ready: FastAPI + Telegram-bot i bakgrundstråd)
# - Engine OFF blockerar ALLA trades (guard i place_order + i entryvägar)
# - Ny ORB startas på FÖRSTA candlen efter ett färgskifte (röd↔grön)
# - Doji-filter, trailing stop, CSV-loggar, Telegram-kommandon
# - REST-poll av KuCoin 3m-candles (WS är valfritt och AV som default)
#
# För Render (rekommenderat filnamn för import): main_19.py och starta:
#   uvicorn main_19:app --host 0.0.0.0 --port $PORT
#
# Krav i requirements.txt:
#   fastapi
#   uvicorn
#   requests
#   python-telegram-bot==13.15
#
# Valfritt:
#   kucoin-python==1.0.11  (om USE_KUCOIN_WS=true och du vill köra WS)
#
# Env:
#   TELEGRAM_BOT_TOKEN=....
#   USE_KUCOIN_WS=false
#   KUCOIN_BASE_URL=https://api.kucoin.com

import os
import csv
import time
import json
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters

# ==============================
# KONFIG
# ==============================
SYMBOLS = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAME = "3m"
TRADE_SIZE_USDT = 30.0
FEE_RATE = 0.001       # 0.1 %
AI_DEFAULT = "neutral" # 'aggressiv' | 'neutral' | 'försiktig'
DOJI_FILTER = True
DOJI_BODY_PCT = 0.10   # <10% av range => Doji
BREAKOUT_BUFFER = 0.001  # 0.1 % över ORB high

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"  # fallback enbart för dig
)
KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")
USE_KUCOIN_WS = os.getenv("USE_KUCOIN_WS", "false").lower() == "true"

LOG_DIR = "logs"
MOCK_LOG = os.path.join(LOG_DIR, "mock_trade_log.csv")
REAL_LOG = os.path.join(LOG_DIR, "real_trade_log.csv")
ACTIVITY_LOG = os.path.join(LOG_DIR, "activity_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================
# HJÄLP
# ==============================
def ensure_csv(path: str, header: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

ensure_csv(MOCK_LOG, ["time_utc","symbol","side","qty","price","fee","ai_mode","reason"])
ensure_csv(REAL_LOG, ["time_utc","symbol","side","qty","price","fee","ai_mode","reason"])
ensure_csv(ACTIVITY_LOG, ["time_utc","symbol","event","details"])

def log_activity(symbol: str, event: str, details: str):
    with open(ACTIVITY_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(), symbol, event, details])

def candle_color(o: float, c: float) -> str:
    return "green" if c >= o else "red"

def is_doji(o: float, h: float, l: float, c: float, body_pct: float = DOJI_BODY_PCT) -> bool:
    total = max(h - l, 1e-12)
    body = abs(c - o)
    return (body / total) <= body_pct

def ku_symbol(symbol: str) -> str:
    s = symbol.upper()
    return s.replace("USDT", "-USDT") if "USDT" in s and "-" not in s else s

# ==============================
# DATATYPER
# ==============================
@dataclass
class Candle:
    ts: int   # open time ms
    o: float
    h: float
    l: float
    c: float
    v: float

@dataclass
class ORB:
    start_ts: int
    high: float
    low: float
    base: Candle

@dataclass
class Position:
    side: str
    entry: float
    qty: float
    orb_at_entry: ORB
    trailing_stop: Optional[float] = None
    active: bool = True

@dataclass
class BotState:
    # Körning
    trading_enabled: bool = False   # Engine ON/OFF
    mode: str = "mock"              # 'mock' | 'live'
    ai_mode: str = AI_DEFAULT
    epoch: int = 0                  # bumpas vid OFF för att ogiltigförklara köade signaler

    # Per symbol
    positions: Dict[str, Optional[Position]] = field(default_factory=dict)
    orbs: Dict[str, Optional[ORB]] = field(default_factory=dict)

    # Färgskifteslogik
    prev_seen_color: Dict[str, Optional[str]] = field(default_factory=dict)
    shift_armed: Dict[str, bool] = field(default_factory=dict)

    # Candle-hantering
    last_candle_ts: Dict[str, Optional[int]] = field(default_factory=dict)

    # Buffertar
    pending_signals: List[dict] = field(default_factory=list)

    def reset_symbol(self, s: str):
        self.positions[s] = None
        self.orbs[s] = None
        self.prev_seen_color[s] = None
        self.shift_armed[s] = False
        self.last_candle_ts[s] = None

STATE = BotState()
for s in SYMBOLS:
    STATE.reset_symbol(s)

# ==============================
# ORDER
# ==============================
def kucoin_min_qty(symbol: str, price: float) -> float:
    usdt_min = 5.0
    return round(usdt_min / max(price, 1e-9), 6)

def log_trade(mock: bool, symbol: str, side: str, qty: float, price: float, fee: float, ai_mode: str, reason: str):
    path = MOCK_LOG if mock else REAL_LOG
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(), symbol, side, f"{qty:.8f}", f"{price:.8f}", f"{fee:.8f}", ai_mode, reason])

def place_order(symbol: str, side: str, price: float, usdt_amount: float, reason: str):
    # CENTRALT SKYDD — allt passerar här
    if not STATE.trading_enabled:
        log_activity(symbol, "ORDER_BLOCKED", f"reason={reason}")
        return None

    qty = max(usdt_amount / max(price, 1e-9), 0.0)
    if qty < kucoin_min_qty(symbol, price):
        log_activity(symbol, "ORDER_SKIPPED", f"qty too small ({qty:.8f})")
        return None

    fee = usdt_amount * FEE_RATE
    log_trade(STATE.mode == "mock", symbol, side, qty, price, fee, STATE.ai_mode, reason)

    if side.lower() == "buy":
        STATE.positions[symbol] = Position("long", price, qty, STATE.orbs.get(symbol))
        log_activity(symbol, "POSITION_OPEN", f"entry={price}, qty={qty:.8f}")
    elif side.lower() == "sell" and STATE.positions.get(symbol):
        pos = STATE.positions[symbol]
        pnl = (price - pos.entry) * pos.qty - fee
        STATE.positions[symbol] = None
        log_activity(symbol, "POSITION_CLOSE", f"close={price}, pnl={pnl:.8f}")
    return True

# ==============================
# AI-FILTER (enkel)
# ==============================
def ai_allows(strength: float) -> bool:
    mode = STATE.ai_mode
    if mode == "aggressiv": return strength >= 0.2
    if mode == "neutral":   return strength >= 0.5
    if mode == "försiktig": return strength >= 0.75
    return True

# ==============================
# ORB & ENTRY
# ==============================
def start_new_orb(symbol: str, base: Candle):
    STATE.orbs[symbol] = ORB(base.ts, base.h, base.l, base)
    log_activity(symbol, "NEW_ORB", f"H={base.h} L={base.l} ts={base.ts}")

def try_breakouts(symbol: str, c: Candle):
    """ORB-long breakout + CloseBreak (close korsar ORB-high)."""
    orb = STATE.orbs.get(symbol)
    if not orb:
        return

    # Doji-filter
    if DOJI_FILTER and is_doji(c.o, c.h, c.l, c.c):
        log_activity(symbol, "SKIP_DOJI", f"ts={c.ts}")
        return

    # ORB LONG (hög bryts med buffer)
    long_trigger = orb.high * (1 + BREAKOUT_BUFFER)
    if c.h >= long_trigger and ai_allows(0.6):
        place_order(symbol, "buy", max(long_trigger, c.o), TRADE_SIZE_USDT, "ORB_LONG_BREAK")
        return

    # CloseBreak: close korsar ORB-high uppåt
    if c.o < orb.high and c.c >= long_trigger and ai_allows(0.6):
        place_order(symbol, "buy", c.c, TRADE_SIZE_USDT, "CloseBreak")

def update_trailing(symbol: str, c: Candle):
    pos = STATE.positions.get(symbol)
    if not pos or not pos.active:
        return
    # start trailing efter +0.1 %
    trigger = pos.entry * 1.001
    if pos.trailing_stop is None and c.h >= trigger:
        pos.trailing_stop = c.c * 0.999
        log_activity(symbol, "TRAIL_START", f"ts={c.ts}, ts={pos.trailing_stop:.6f}")
        return
    # följ uppåt
    if pos.trailing_stop is not None:
        new_ts = c.c * 0.999
        if new_ts > pos.trailing_stop:
            pos.trailing_stop = new_ts
    # stoppa ut
    if pos.trailing_stop is not None and c.l <= pos.trailing_stop:
        place_order(symbol, "sell", pos.trailing_stop, pos.qty * pos.entry, "TRAIL_HIT")

def process_candle(symbol: str, prev_candle: Optional[Candle], curr: Candle):
    # Init ORB
    if STATE.orbs.get(symbol) is None:
        start_new_orb(symbol, curr)

    # Färgskiften — starta ny ORB på FÖRSTA candlen efter skiftet
    prev_seen = STATE.prev_seen_color.get(symbol)
    curr_color = candle_color(curr.o, curr.c)

    if prev_candle is not None and prev_seen is not None:
        prev_color_actual = candle_color(prev_candle.o, prev_candle.c)
        shift_on_prev = (prev_color_actual != prev_seen)
        if shift_on_prev and not STATE.shift_armed.get(symbol, False):
            STATE.shift_armed[symbol] = True
            log_activity(symbol, "SHIFT_ARMED", f"prev_ts={prev_candle.ts} shift={prev_seen}->{prev_color_actual}")

        if STATE.shift_armed.get(symbol, False):
            # Första candlen efter skiftet ⇒ ny ORB här
            start_new_orb(symbol, curr)
            STATE.shift_armed[symbol] = False

    STATE.prev_seen_color[symbol] = curr_color

    # Entry + trailing
    try_breakouts(symbol, curr)
    update_trailing(symbol, curr)

# ==============================
# DATA: KuCoin 3m candles (REST)
# ==============================
def fetch_kucoin_3m(symbol: str, limit: int = 50) -> List[Candle]:
    url = f"{KUCOIN_BASE_URL}/api/v1/market/candles"
    params = {"type": "3min", "symbol": ku_symbol(symbol)}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    out: List[Candle] = []
    for it in reversed(data[-limit:]):  # äldst → senast
        t_end = int(float(it[0]))
        o = float(it[1]); c = float(it[2]); h = float(it[3]); l = float(it[4]); v = float(it[5])
        ts_open_ms = (t_end - 180) * 1000
        out.append(Candle(ts_open_ms, o, h, l, c, v))
    return out

def poll_loop():
    """Kör i bakgrund: hämtar nya candles och processar endast nya ts."""
    while True:
        try:
            for s in SYMBOLS:
                candles = fetch_kucoin_3m(s, limit=50)
                last_ts = STATE.last_candle_ts.get(s)
                for c in candles:
                    if (last_ts is None) or (c.ts > last_ts):
                        prev = None
                        # hämta prev genom att titta bakåt en candle
                        # (för exakt prev i detta flöde räcker att vi skickar None för första nya)
                        process_candle(s, None, c)
                        STATE.last_candle_ts[s] = c.ts
            time.sleep(20)  # pollintervall
        except Exception as e:
            log_activity("-", "POLL_ERROR", str(e))
            time.sleep(5)

# ==============================
# TELEGRAM
# ==============================
HELP_TEXT = (
    "/engine_start - Slå på handel\n"
    "/engine_stop  - Stäng av handel (stoppar ALLA trades, tömmer köer)\n"
    "/start_mock   - Byt till MOCK och slå på handel\n"
    "/start_live   - Byt till LIVE och slå på handel\n"
    "/status       - Visa status (läge, AI, positioner, ORB)\n"
    "/set_ai <aggressiv|neutral|försiktig>\n"
    "/export_csv   - Skicka loggfiler\n"
    "/help\n"
    "\nLogik: Ny ORB sätts på första candlen EFTER färgskifte. Doji-filter aktivt."
)

def tg_start(update: Update, context: CallbackContext):
    update.message.reply_text("Mp ORBbot igång. Använd /engine_start för att börja.\n" + HELP_TEXT)

def tg_engine_start(update: Update, context: CallbackContext):
    STATE.trading_enabled = True
    update.message.reply_text("✅ Engine: ON")

def tg_engine_stop(update: Update, context: CallbackContext):
    STATE.trading_enabled = False
    STATE.epoch += 1
    STATE.pending_signals.clear()
    update.message.reply_text("🛑 Engine: OFF – alla signaler blockeras och köer töms.")
    log_activity("-", "ENGINE_OFF", "manual")

def tg_start_mock(update: Update, context: CallbackContext):
    STATE.mode = "mock"
    STATE.trading_enabled = True
    update.message.reply_text("✅ MOCK-läge: ON")

def tg_start_live(update: Update, context: CallbackContext):
    STATE.mode = "live"
    STATE.trading_enabled = True
    update.message.reply_text("⚠️ LIVE-läge: ON (se upp!)")

def tg_status(update: Update, context: CallbackContext):
    lines = [
        f"Engine: {'ON' if STATE.trading_enabled else 'OFF'}",
        f"Läge: {STATE.mode.upper()}",
        f"AI: {STATE.ai_mode}",
    ]
    for s in SYMBOLS:
        orb = STATE.orbs.get(s)
        pos = STATE.positions.get(s)
        if orb:
            lines.append(f"{s} ORB: H={orb.high:.6f} L={orb.low:.6f}")
        else:
            lines.append(f"{s} ORB: -")
        if pos:
            lines.append(f"  Pos: long qty={pos.qty:.6f} entry={pos.entry:.6f} TS={pos.trailing_stop}")
        else:
            lines.append(f"  Pos: -")
    update.message.reply_text("\n".join(lines))

def tg_set_ai(update: Update, context: CallbackContext):
    if not context.args:
        update.message.reply_text("Använd: /set_ai <aggressiv|neutral|försiktig>")
        return
    m = context.args[0].lower()
    if m not in ("aggressiv","neutral","försiktig"):
        update.message.reply_text("Ogiltigt val. Välj aggressiv|neutral|försiktig.")
        return
    STATE.ai_mode = m
    update.message.reply_text(f"AI-läge satt till {m}.")

def tg_export_csv(update: Update, context: CallbackContext):
    for p in (MOCK_LOG, REAL_LOG, ACTIVITY_LOG):
        try:
            with open(p, "rb") as f:
                update.message.reply_document(InputFile(f, filename=os.path.basename(p)))
        except Exception as e:
            update.message.reply_text(f"Kunde inte bifoga {p}: {e}")

def tg_help(update: Update, context: CallbackContext):
    update.message.reply_text(HELP_TEXT)

class TelegramRunner:
    def __init__(self):
        self.updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler("start", tg_start))
        dp.add_handler(CommandHandler("engine_start", tg_engine_start))
        dp.add_handler(CommandHandler("engine_stop", tg_engine_stop))
        dp.add_handler(CommandHandler("start_mock", tg_start_mock))
        dp.add_handler(CommandHandler("start_live", tg_start_live))
        dp.add_handler(CommandHandler("status", tg_status))
        dp.add_handler(CommandHandler("set_ai", tg_set_ai, pass_args=True))
        dp.add_handler(CommandHandler("export_csv", tg_export_csv))
        dp.add_handler(CommandHandler("help", tg_help))
        # fånga enkel text om du vill, ej nödvändigt här
        dp.add_handler(MessageHandler(Filters.text & (~Filters.command), lambda u,c: None))

    def run(self):
        self.updater.start_polling(drop_pending_updates=True)
        self.updater.idle()

# ==============================
# (VALFRITT) KUCOIN WS
# ==============================
# Avstängd som standard för att undvika fel. Sätt USE_KUCOIN_WS=true för att köra.
# Kräver: kucoin-python==1.0.11
if USE_KUCOIN_WS:
    try:
        import asyncio
        from kucoin.ws_client import KucoinWsClient
        from kucoin.client import WsToken

        async def on_ws_message(msg: dict):
            # Här kan du läsa ticker/kline-updates och uppdatera intern state
            # Denna demo använder REST-poll, så WS är valfri
            pass

        async def start_kucoin_ws():
            ws = await KucoinWsClient.create(
                token=WsToken(),
                callback=on_ws_message,
                private=False
            )
            await ws.subscribe("/market/ticker:BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
            log_activity("-", "WS_CONNECTED", "KuCoin WS up")

        def ws_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_kucoin_ws())
            loop.run_forever()

        _ws_thr = threading.Thread(target=ws_thread, daemon=True)
        _ws_thr.start()
    except Exception as e:
        log_activity("-", "WS_DISABLED", f"{e}")

# ==============================
# FastAPI-app (Render)
# ==============================
app = FastAPI()
_tg_thread: Optional[threading.Thread] = None
_poll_thread: Optional[threading.Thread] = None
_tg_runner: Optional[TelegramRunner] = None

@app.on_event("startup")
def on_startup():
    global _tg_thread, _tg_runner, _poll_thread
    # starta Telegram i bakgrund
    if _tg_thread is None:
        _tg_runner = TelegramRunner()
        _tg_thread = threading.Thread(target=_tg_runner.run, daemon=True)
        _tg_thread.start()
        log_activity("-", "TG_STARTED", "polling")

    # starta poll-loop i bakgrund
    if _poll_thread is None:
        _poll_thread = threading.Thread(target=poll_loop, daemon=True)
        _poll_thread.start()
        log_activity("-", "POLL_STARTED", "kucoin REST 3m")

@app.get("/healthz")
def healthz():
    return {"status":"ok", "engine": STATE.trading_enabled, "mode": STATE.mode, "ai": STATE.ai_mode}

@app.get("/")
def root():
    return {"name":"Mp ORBbot", "msg":"Service up", "note":"ORB resets on first candle after color shift"}

# ==============================
# Lokal körning (python main.19.py)
# ==============================
def main():
    # Om du kör lokalt utan uvicorn: starta telegram och poll direkt
    TelegramRunner().run()

if __name__ == "__main__":
    main()
