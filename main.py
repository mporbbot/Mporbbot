# main.py — Mp ORBbot
# Ny ORB startas på FÖRSTA candlen som kommer DIREKT efter ett färgskifte.
# Python 3.10+
#
# Installera:
#   pip install python-telegram-bot==13.15 requests
#
# Miljövariabler (rekommenderat):
#   TELEGRAM_BOT_TOKEN=...
#   KUCOIN_BASE_URL=https://api.kucoin.com
#
# OBS: För säkerhet bör token sättas som env. Vi har fallback här enbart för din bekvämlighet.

import os
import csv
import time
import math
import json
import threading
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta

import requests
from telegram import Update, ParseMode, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters

# ==============================
# KONFIG
# ==============================
SYMBOLS = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAME = "3m"
TRADE_SIZE_USDT = 30.0
DEFAULT_FEE_RATE = 0.001  # 0.1%
DEFAULT_AI_MODE = "neutral"  # 'aggressiv' | 'neutral' | 'försiktig'
DEFAULT_USE_MOCK = True   # starta i mock-läge
DOJI_FILTER = True
DOJI_BODY_PCT = 0.1       # <10% av range => Doji
BREAKOUT_BUFFER = 0.001   # 0.1% över/under ORB för att undvika touch

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"  # fallback enligt din tidigare instruktion
)

# KuCoin (publik endpoint räcker för candles)
KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

# Loggar
LOG_DIR = "logs"
MOCK_LOG = os.path.join(LOG_DIR, "mock_trade_log.csv")
REAL_LOG = os.path.join(LOG_DIR, "real_trade_log.csv")
ACTIVITY_LOG = os.path.join(LOG_DIR, "activity_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# Failsafe: stoppa handel om Telegram-kontakten tappas > 5 min
TELEGRAM_HEARTBEAT_LIMIT_SEC = 300

# ==============================
# HJÄLP
# ==============================
def now_ts_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def candle_color(o: float, c: float) -> str:
    return "green" if c >= o else "red"

def is_doji(o: float, h: float, l: float, c: float, body_pct: float = DOJI_BODY_PCT) -> bool:
    total_range = max(h - l, 1e-12)
    body = abs(c - o)
    return (body / total_range) <= body_pct

def pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b - a) / a

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

def log_trade(mock: bool, symbol: str, side: str, qty: float, price: float, fee: float, ai_mode: str, reason: str):
    path = MOCK_LOG if mock else REAL_LOG
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.utcnow().isoformat(), symbol, side,
            f"{qty:.8f}", f"{price:.8f}", f"{fee:.8f}", ai_mode, reason
        ])

def ku_symbol(symbol: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    s = symbol.upper()
    if s.endswith("USDT"):
        return s[:-4] + "-" + "USDT"
    if "-" in s:
        return s
    return s

# ==============================
# DATATYPER
# ==============================
@dataclass
class Candle:
    ts: int   # open time (ms)
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
    base_candle: Candle

@dataclass
class Position:
    side: str  # "long"
    entry_price: float
    qty: float
    orb_at_entry: ORB
    trailing_stop: Optional[float] = None
    is_active: bool = True

@dataclass
class BotState:
    ai_mode: str = DEFAULT_AI_MODE
    use_mock: bool = DEFAULT_USE_MOCK
    fee_rate: float = DEFAULT_FEE_RATE

    # Run/failsafe
    trading_enabled: bool = False   # endast true efter bekräftelse
    last_telegram_heartbeat: float = field(default_factory=lambda: time.time())

    positions: Dict[str, Optional[Position]] = field(default_factory=dict)
    orbs: Dict[str, Optional[ORB]] = field(default_factory=dict)
    # För färgskifteslogik:
    # - prev_seen_color: sparad färg för föregående candle (per symbol)
    prev_seen_color: Dict[str, Optional[str]] = field(default_factory=dict)
    # - shift_armed: True om vi upptäckte skifte på förra candlen och ska starta ny ORB på denna candle
    shift_armed: Dict[str, bool] = field(default_factory=dict)
    # - pending_shift: textinfo om riktning (för logg)
    pending_shift: Dict[str, Optional[str]] = field(default_factory=dict)

    trades_today: Dict[str, int] = field(default_factory=dict)

    def start_new_orb(self, symbol: str, base_candle: Candle):
        self.orbs[symbol] = ORB(
            start_ts=base_candle.ts,
            high=base_candle.h,
            low=base_candle.l,
            base_candle=base_candle
        )
        log_activity(symbol, "NEW_ORB", f"ts={base_candle.ts}, H={base_candle.h}, L={base_candle.l}")

# ==============================
# NY ORB-LOGIK EFTER FÄRGSKIFTE
# ==============================
def should_start_new_orb(prev_candle: Candle, curr_candle: Candle) -> bool:
    """
    Vi startar ny ORB exakt på första candlen EFTER ett färgskifte.
    Denna funktion kallas endast när shift_armed=True, så vi returnerar True.
    """
    return True

# ==============================
# KUCOIN DATA (publik candles)
# ==============================
def kucoin_get_candles(symbol: str, minutes: int) -> List[Candle]:
    """
    Hämtar sista 'minutes' minuterna som 3-minuterscandles från KuCoin publik endpoint.
    """
    sym = ku_symbol(symbol)
    # KuCoin typ: "3min"
    # Return: list of [time, open, close, high, low, volume, turnover]
    # time är slutet av candle i sekunder.
    url = f"{KUCOIN_BASE_URL}/api/v1/market/candles"
    params = {"type": "3min", "symbol": sym}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    arr = data.get("data", [])
    candles: List[Candle] = []
    # KuCoin returnerar senaste först. Vi vänder ordningen.
    for it in reversed(arr):
        t_end_sec = int(float(it[0]))
        c = float(it[2]); h = float(it[3]); l = float(it[4]); o = float(it[1]); v = float(it[5])
        # approximera open time: 3 minuter före slutet
        ts_ms = (t_end_sec - 180) * 1000
        candles.append(Candle(ts=ts_ms, o=o, h=h, l=l, c=c, v=v))
    # Begränsa i tid (minutes)
    max_candles = max(1, minutes // 3)
    return candles[-max_candles:]

# ==============================
# ORDER & STORLEK
# ==============================
def kucoin_min_qty(symbol: str, price: float) -> float:
    # Enkel approx: 5 USDT min
    usdt_min = 5.0
    qty = usdt_min / max(price, 1e-9)
    return round(qty, 6)

def place_order(state: BotState, symbol: str, side: str, price: float, usdt_amount: float, reason: str):
    if not state.trading_enabled:
        log_activity(symbol, "ORDER_BLOCKED", "trading_enabled=False")
        return None

    qty = max(usdt_amount / max(price, 1e-9), 0.0)
    min_qty = kucoin_min_qty(symbol, price)
    if qty < min_qty:
        log_activity(symbol, "ORDER_SKIPPED", f"qty {qty:.8f} < min_qty {min_qty:.8f}")
        return None

    fee = usdt_amount * state.fee_rate
    log_trade(state.use_mock, symbol, side, qty, price, fee, state.ai_mode, reason)

    if side.lower() == "buy":
        state.positions[symbol] = Position(
            side="long",
            entry_price=price,
            qty=qty,
            orb_at_entry=state.orbs.get(symbol),
            trailing_stop=None,
            is_active=True
        )
        log_activity(symbol, "POSITION_OPEN", f"price={price}, qty={qty:.8f}")
    elif side.lower() == "sell" and state.positions.get(symbol):
        pos = state.positions[symbol]
        pnl = (price - pos.entry_price) * pos.qty - fee
        state.positions[symbol] = None
        state.trades_today[symbol] = state.trades_today.get(symbol, 0) + 1
        log_activity(symbol, "POSITION_CLOSE", f"price={price}, pnl={pnl:.8f}")
    return True

# ==============================
# AI-FILTER (enkel)
# ==============================
def ai_allows_trade(state: BotState, strength: float) -> bool:
    mode = state.ai_mode
    if mode == "aggressiv":
        return strength >= 0.2
    if mode == "neutral":
        return strength >= 0.5
    if mode == "försiktig":
        return strength >= 0.75
    return True

# ==============================
# BREAKOUTLOGIK
# ==============================
def try_breakout(state: BotState, symbol: str, c: Candle):
    orb = state.orbs.get(symbol)
    if orb is None:
        return

    # Doji-filter
    if DOJI_FILTER and is_doji(c.o, c.h, c.l, c.c):
        log_activity(symbol, "SKIP_DOJI", f"ts={c.ts}")
        return

    # Long-breakout
    long_trigger = orb.high * (1 + BREAKOUT_BUFFER)
    if c.h >= long_trigger:
        if ai_allows_trade(state, strength=0.6):
            place_order(state, symbol, "buy", price=max(long_trigger, c.o), usdt_amount=TRADE_SIZE_USDT, reason="ORB_LONG_BREAK")
        return

    # (Short kan läggas till här om du vill)

# ==============================
# TRAILING STOP
# ==============================
def update_trailing_and_exits(state: BotState, symbol: str, c: Candle):
    pos = state.positions.get(symbol)
    if not pos or not pos.is_active:
        return
    # Starta trailing när priset är +0.1% över entry
    trigger = pos.entry_price * 1.001
    if pos.trailing_stop is None and c.h >= trigger:
        pos.trailing_stop = c.c * 0.999  # 0.1% under nuvarande close
        log_activity(symbol, "TRAIL_START", f"ts={c.ts}, ts_value={pos.trailing_stop:.8f}")
        return
    # Följ med uppåt
    if pos.trailing_stop is not None:
        new_ts = c.c * 0.999
        if new_ts > pos.trailing_stop:
            pos.trailing_stop = new_ts
    # Stoppa ut
    if pos.trailing_stop is not None and c.l <= pos.trailing_stop:
        place_order(state, symbol, "sell", price=pos.trailing_stop, usdt_amount=pos.qty*pos.entry_price, reason="TRAIL_HIT")

# ==============================
# HUVUDLOOP PER CANDLE
# ==============================
def process_candle(state: BotState, symbol: str, prev_candle: Optional[Candle], curr_candle: Candle):
    # Init ORB om saknas
    if state.orbs.get(symbol) is None:
        state.start_new_orb(symbol, curr_candle)

    # Färgdetektion
    prev_seen = state.prev_seen_color.get(symbol)
    curr_color = candle_color(curr_candle.o, curr_candle.c)

    # Om vi har en tidigare candle: kontrollera om ett skifte inträffade på förra candlen
    if prev_candle is not None and prev_seen is not None:
        prev_candle_color = candle_color(prev_candle.o, prev_candle.c)
        # Skifte inträffade på förra candlen om dess färg != den föregående sparade färgen
        color_shift_on_prev = (prev_candle_color != prev_seen)

        if color_shift_on_prev and not state.shift_armed.get(symbol, False):
            # Armera skiftet: Ny ORB ska starta på NÄSTA candle (den vi processar nu)
            state.shift_armed[symbol] = True
            state.pending_shift[symbol] = f"{prev_seen}_to_{prev_candle_color}"
            log_activity(symbol, "SHIFT_ARMED", f"on_prev ts={prev_candle.ts}, shift={state.pending_shift[symbol]}")

        # Om armerad: starta ny ORB nu på curr_candle (första efter skiftet)
        if state.shift_armed.get(symbol, False):
            if should_start_new_orb(prev_candle, curr_candle):
                state.start_new_orb(symbol, curr_candle)
                state.shift_armed[symbol] = False
                state.pending_shift[symbol] = None

    # Uppdatera sparad färg till nuvarande candles färg
    state.prev_seen_color[symbol] = curr_color

    # Handelslogik
    try_breakout(state, symbol, curr_candle)
    update_trailing_and_exits(state, symbol, curr_candle)

# ==============================
# BACKTEST / CSV
# ==============================
def load_csv_klines(path: str) -> List[Candle]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(Candle(
                ts=int(r["ts"]),
                o=float(r["o"]), h=float(r["h"]), l=float(r["l"]), c=float(r["c"]),
                v=float(r.get("v", 0.0)),
            ))
    return out

def run_backtest_on_candles(state: BotState, symbol: str, candles: List[Candle]):
    prev = None
    # Enable trading for backtest without Telegram confirmation but still respect mock/live flag
    trading_enabled_backup = state.trading_enabled
    state.trading_enabled = True
    for c in candles:
        process_candle(state, symbol, prev, c)
        prev = c
    state.trading_enabled = trading_enabled_backup

def parse_period_to_minutes(period: str) -> int:
    p = period.strip().lower()
    if p.endswith("d"):
        return int(p[:-1]) * 24 * 60
    if p.endswith("h"):
        return int(p[:-1]) * 60
    if p.endswith("m"):
        return int(p[:-1])
    # default: minutes
    return int(p)

# ==============================
# STATUS / PnL (enkel)
# ==============================
def todays_pnl_from_log(path: str) -> float:
    today = datetime.utcnow().date()
    pnl = 0.0
    # Enkel approx: summera sälj - köp - avgift, per rad.
    # (Obs: inte positionsparning; räcker som snabbstatus.)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = datetime.fromisoformat(r["time_utc"])
            if t.date() != today:
                continue
            side = r["side"].lower()
            price = float(r["price"])
            qty = float(r["qty"])
            fee = float(r["fee"])
            amount = price * qty
            if side == "buy":
                pnl -= amount
                pnl -= fee
            elif side == "sell":
                pnl += amount
                pnl -= fee
    return pnl

def status_text(state: BotState) -> str:
    mock = state.use_mock
    pnl_today = todays_pnl_from_log(MOCK_LOG if mock else REAL_LOG)
    lines = [
        f"AI-läge: {state.ai_mode}",
        f"Läge: {'MOCK' if mock else 'LIVE'}",
        f"Handel påslagen: {state.trading_enabled}",
        f"Dagens PnL ({'mock' if mock else 'real'}): {pnl_today:.2f} USDT",
        "Aktiva positioner:"
    ]
    for s in SYMBOLS:
        pos = state.positions.get(s)
        orb = state.orbs.get(s)
        if pos:
            lines.append(f" - {s}: long qty={pos.qty:.6f} entry={pos.entry_price:.6f} TS={pos.trailing_stop}")
        else:
            lines.append(f" - {s}: (ingen)")
        if orb:
            lines.append(f"   ORB: H={orb.high:.6f} L={orb.low:.6f} start_ts={orb.start_ts}")
    return "\n".join(lines)

# ==============================
# TELEGRAM
# ==============================
HELP_TEXT = (
    "/start - begär bekräftelse och slår på handel (MOCK som standard)\n"
    "/stop - stänger av handel (kommandon fungerar ändå)\n"
    "/status - visa status (AI-läge, PnL idag, positioner, ORB)\n"
    "/set_ai <aggressiv|neutral|försiktig> - sätt AI-läge\n"
    "/backtest <SYMBOL> <PERIOD ex 3d|12h|90m> [FEE ex 0.001] - kör backtest med KuCoin-data\n"
    "/mock_trade <SYMBOL> [PRICE] - kör en omedelbar köp+stäng för test\n"
    "/export_csv - skickar loggfilerna (mock/real/activity)\n"
    "/help - den här hjälpen\n"
    "\nNy ORB startas på första candlen EFTER färgskifte."
)

class TelegramBot:
    def __init__(self):
        self.state = BotState()
        for s in SYMBOLS:
            self.state.positions[s] = None
            self.state.orbs[s] = None
            self.state.prev_seen_color[s] = None
            self.state.shift_armed[s] = False
            self.state.pending_shift[s] = None
            self.state.trades_today[s] = 0

        self.updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
        dp = self.updater.dispatcher

        dp.add_handler(CommandHandler("start", self.on_start))
        dp.add_handler(CommandHandler("stop", self.on_stop))
        dp.add_handler(CommandHandler("status", self.on_status))
        dp.add_handler(CommandHandler("set_ai", self.on_set_ai, pass_args=True))
        dp.add_handler(CommandHandler("backtest", self.on_backtest, pass_args=True))
        dp.add_handler(CommandHandler("mock_trade", self.on_mock_trade, pass_args=True))
        dp.add_handler(CommandHandler("export_csv", self.on_export_csv))
        dp.add_handler(CommandHandler("help", self.on_help))
        # Fånga "JA" bekräftelser
        dp.add_handler(MessageHandler(Filters.text & (~Filters.command), self.on_text))

        # Heartbeat/failsafe
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    # ===== Kommandon =====
    def on_start(self, update: Update, context: CallbackContext):
        self.state.last_telegram_heartbeat = time.time()
        # Bekräftelse innan handel aktiveras
        mode = "MOCK" if self.state.use_mock else "LIVE"
        update.message.reply_text(
            f"Bekräfta för att slå på handel ({mode}). Svara exakt med:\n"
            f"JA {mode}\n\nVill du byta läge först? Använd /toggle_mode eller /stop + /set_ai osv.",
        )

    def on_stop(self, update: Update, context: CallbackContext):
        self.state.trading_enabled = False
        update.message.reply_text("Handel AV. Kommandon fungerar fortfarande.")
        log_activity("-", "TRADING_DISABLED", "manual /stop")

    def on_status(self, update: Update, context: CallbackContext):
        self.state.last_telegram_heartbeat = time.time()
        update.message.reply_text(status_text(self.state))

    def on_set_ai(self, update: Update, context: CallbackContext):
        self.state.last_telegram_heartbeat = time.time()
        if not context.args:
            update.message.reply_text("Använd: /set_ai <aggressiv|neutral|försiktig>")
            return
        mode = context.args[0].lower()
        if mode not in ("aggressiv","neutral","försiktig"):
            update.message.reply_text("Ogiltigt läge. Välj aggressiv|neutral|försiktig.")
            return
        self.state.ai_mode = mode
        update.message.reply_text(f"AI-läge satt till {mode}.")

    def on_backtest(self, update: Update, context: CallbackContext):
        self.state.last_telegram_heartbeat = time.time()
        try:
            if len(context.args) < 2:
                update.message.reply_text("Använd: /backtest <SYMBOL> <PERIOD ex 3d|12h|90m> [FEE ex 0.001]")
                return
            symbol = context.args[0].upper()
            period = context.args[1]
            fee = float(context.args[2]) if len(context.args) >= 3 else self.state.fee_rate

            minutes = parse_period_to_minutes(period)
            self.state.fee_rate = fee

            update.message.reply_text(f"Hämtar {period} {TIMEFRAME}-data för {symbol} från KuCoin…")
            candles = kucoin_get_candles(symbol, minutes)
            if not candles:
                update.message.reply_text("Fick inga candles.")
                return

            # Temporärt slå på handel för backtest
            was_enabled = self.state.trading_enabled
            self.state.trading_enabled = True
            # Nollställ symbolens state för ren backtest
            self.state.positions[symbol] = None
            self.state.orbs[symbol] = None
            self.state.prev_seen_color[symbol] = None
            self.state.shift_armed[symbol] = False
            self.state.pending_shift[symbol] = None

            run_backtest_on_candles(self.state, symbol, candles)
            self.state.trading_enabled = was_enabled

            update.message.reply_text(f"Klar. Kolla loggar i {LOG_DIR}/. Avgift: {fee}")
        except Exception as e:
            update.message.reply_text(f"Fel i backtest: {e}")

    def on_mock_trade(self, update: Update, context: CallbackContext):
        self.state.last_telegram_heartbeat = time.time()
        if len(context.args) < 1:
            update.message.reply_text("Använd: /mock_trade <SYMBOL> [PRICE]")
            return
        symbol = context.args[0].upper()
        price: Optional[float] = None
        if len(context.args) >= 2:
            try:
                price = float(context.args[1])
            except:
                update.message.reply_text("Ogiltigt pris.")
                return
        # Hämta ungefärligt pris via candles om ej angivet
        if price is None:
            try:
                c = kucoin_get_candles(symbol, 6)[-1]
                price = c.c
            except:
                update.message.reply_text("Kunde inte hämta pris.")
                return

        if not self.state.trading_enabled:
            update.message.reply_text("Handel är AV. Skriv /start och bekräfta med 'JA MOCK' för mockläge.")
            return

        ok = place_order(self.state, symbol, "buy", price, TRADE_SIZE_USDT, "MANUAL_MOCK_BUY")
        if ok:
            place_order(self.state, symbol, "sell", price*1.001, TRADE_SIZE_USDT, "MANUAL_MOCK_SELL")
            update.message.reply_text("Mocktrade genomförd.")
        else:
            update.message.reply_text("Kunde inte genomföra mocktrade (min_qty?).")

    def on_export_csv(self, update: Update, context: CallbackContext):
        self.state.last_telegram_heartbeat = time.time()
        files = [MOCK_LOG, REAL_LOG, ACTIVITY_LOG]
        for p in files:
            try:
                with open(p, "rb") as f:
                    update.message.reply_document(InputFile(f, filename=os.path.basename(p)))
            except Exception as e:
                update.message.reply_text(f"Kunde inte bifoga {p}: {e}")

    def on_help(self, update: Update, context: CallbackContext):
        self.state.last_telegram_heartbeat = time.time()
        update.message.reply_text(HELP_TEXT)

    def on_text(self, update: Update, context: CallbackContext):
        self.state.last_telegram_heartbeat = time.time()
        txt = (update.message.text or "").strip().upper()
        # Bekräftelser
        if txt in ("JA MOCK", "JA LIVE"):
            mode = "MOCK" if "MOCK" in txt else "LIVE"
            self.state.use_mock = (mode == "MOCK")
            self.state.trading_enabled = True
            update.message.reply_text(f"Handel PÅ i {mode}-läge. Kör /status.")
            log_activity("-", "TRADING_ENABLED", f"mode={mode}")
            return

    # ===== Failsafe heartbeat (Telegram-kontakt) =====
    def _heartbeat_loop(self):
        while True:
            try:
                now = time.time()
                # Om handel på och vi inte hört något på > limit -> stäng av handel
                if self.state.trading_enabled and (now - self.state.last_telegram_heartbeat > TELEGRAM_HEARTBEAT_LIMIT_SEC):
                    self.state.trading_enabled = False
                    log_activity("-", "TRADING_DISABLED", "failsafe: telegram heartbeat timeout")
                time.sleep(5)
            except Exception:
                traceback.print_exc()
                time.sleep(5)

    def run(self):
        self.updater.start_polling(drop_pending_updates=True)
        self.updater.idle()

# ==============================
# CLI (frivilligt)
# ==============================
def main():
    bot = TelegramBot()
    print("Mp ORBbot igång. Använd Telegram-kommandon. /help för lista.")
    bot.run()

if __name__ == "__main__":
    main()
