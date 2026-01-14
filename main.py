# main.py
# Mp ORBbot – Momentum Strategy (stable loop, no JobQueue dependency)
# Works with: python-telegram-bot==20.3 and requests
# IMPORTANT: Set env var TELEGRAM_TOKEN (or TELEGRAM_BOT_TOKEN)

import os
import csv
import time
import json
import math
import threading
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# -----------------------------
# Config / Defaults
# -----------------------------
DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

ENGINE_TICK_SECONDS = 3
PRICE_POLL_COOLDOWN_PER_COIN = 5  # seconds between coin checks (soft throttle)
KUCOIN_BASE = "https://api.kucoin.com"

# Mock realism
DEFAULT_FEE_RATE = 0.001       # 0.10% per side
DEFAULT_SLIPPAGE = 0.0002      # 0.02% each side (approx)
DEFAULT_SPREAD_BPS = 2         # 2 bps base (0.02%) fallback if no bid/ask available

LOG_DIR = "."
MOCK_LOG_FILE = os.path.join(LOG_DIR, "mock_trade_log.csv")
REAL_LOG_FILE = os.path.join(LOG_DIR, "real_trade_log.csv")  # reserved for live later

# Healthcheck server (for platforms that expect PORT open)
DEFAULT_PORT = int(os.getenv("PORT", "8080"))

# -----------------------------
# Helpers
# -----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def ts_unix() -> int:
    return int(time.time())

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def safe_float(s, default=0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default

def ensure_csv_headers(path: str, headers: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)

def append_csv(path: str, row: List) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)

# -----------------------------
# Indicators (simple, fast)
# -----------------------------
def ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 1:
        return values[:]
    k = 2 / (period + 1)
    out = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out

def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return [50.0] * len(values)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period
    out = [50.0] * (period)

    def rs_to_rsi(rs):
        return 100.0 - (100.0 / (1.0 + rs))

    rs = (avg_gain / avg_loss) if avg_loss > 0 else 999.0
    out.append(rs_to_rsi(rs))

    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss > 0 else 999.0
        out.append(rs_to_rsi(rs))

    # pad to same length
    while len(out) < len(values):
        out.insert(0, 50.0)
    return out[-len(values):]

# -----------------------------
# KuCoin data (public)
# -----------------------------
def kucoin_get(path: str, params: Optional[dict] = None, timeout: int = 10) -> dict:
    url = KUCOIN_BASE + path
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_best_bid_ask(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    # /api/v1/market/orderbook/level1?symbol=BTC-USDT
    try:
        data = kucoin_get("/api/v1/market/orderbook/level1", {"symbol": symbol})
        if data.get("code") != "200000":
            return None, None
        d = data["data"]
        bid = safe_float(d.get("bestBid"))
        ask = safe_float(d.get("bestAsk"))
        if bid <= 0 or ask <= 0:
            return None, None
        return bid, ask
    except Exception:
        return None, None

def get_klines(symbol: str, tf: str = "1min", limit: int = 120) -> List[dict]:
    """
    KuCoin kline: /api/v1/market/candles?symbol=BTC-USDT&type=1min
    Returns newest->oldest; we convert to oldest->newest dicts.
    Each item: [time, open, close, high, low, volume, turnover]
    """
    type_map = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
    }
    ktype = type_map.get(tf, tf)
    data = kucoin_get("/api/v1/market/candles", {"symbol": symbol, "type": ktype})
    if data.get("code") != "200000":
        return []
    arr = data.get("data", [])[:limit]
    out = []
    for row in reversed(arr):
        out.append({
            "t": int(row[0]),
            "o": safe_float(row[1]),
            "c": safe_float(row[2]),
            "h": safe_float(row[3]),
            "l": safe_float(row[4]),
            "v": safe_float(row[5]),
        })
    return out

# -----------------------------
# State
# -----------------------------
@dataclass
class Settings:
    engine_on: bool = False
    notify: bool = True
    trade_mode: str = "mock"  # "mock" or "live" (live reserved)
    coins: List[str] = None

    timeframe: str = "1m"     # momentum works best on 1m-5m
    stake_usdt: float = 30.0

    # Momentum knobs
    threshold: float = 0.003  # 0.30% breakout strength (used as momentum trigger)
    tp: float = 0.0045        # 0.45%
    sl: float = 0.0028        # 0.28%
    trail_activate: float = 0.0035  # +0.35%
    trail_dist: float = 0.0020      # 0.20%

    max_positions: int = 5
    cooldown_seconds: int = 30

    # Costs
    fee_rate: float = DEFAULT_FEE_RATE
    slippage: float = DEFAULT_SLIPPAGE

    def __post_init__(self):
        if self.coins is None:
            self.coins = DEFAULT_COINS[:]

@dataclass
class Position:
    symbol: str
    side: str  # "LONG"
    qty: float
    entry_price: float
    entry_time: int
    tp_price: float
    sl_price: float
    trail_on: bool = False
    trail_stop: Optional[float] = None
    high_water: float = 0.0  # for trailing

@dataclass
class PnL:
    trades: int = 0
    net_usdt: float = 0.0

STATE = {
    "settings": Settings(),
    "positions": {},   # symbol -> Position
    "pnl": PnL(),
    "last_trade_ts": {},  # symbol -> unix
    "last_check_ts": {},  # symbol -> unix
}

# CSV headers (simple + Skatteverket-friendly)
CSV_HEADERS = [
    "timestamp_unix",
    "timestamp_utc",
    "exchange",
    "mode",
    "symbol",
    "side",
    "qty",
    "stake_usdt",
    "entry_price",
    "exit_price",
    "gross_pnl_usdt",
    "fees_usdt",
    "slippage_usdt",
    "net_pnl_usdt",
    "reason",
]

ensure_csv_headers(MOCK_LOG_FILE, CSV_HEADERS)
ensure_csv_headers(REAL_LOG_FILE, CSV_HEADERS)

# -----------------------------
# Mock execution (bid/ask + fees + slippage)
# -----------------------------
def compute_qty(stake_usdt: float, price: float) -> float:
    if price <= 0:
        return 0.0
    return stake_usdt / price

def fee(amount_usdt: float, fee_rate: float) -> float:
    return abs(amount_usdt) * fee_rate

def apply_slippage(price: float, side: str, slip: float) -> float:
    # For buy: price up; for sell: price down
    if side == "BUY":
        return price * (1.0 + slip)
    return price * (1.0 - slip)

def mock_fill_prices(symbol: str) -> Tuple[float, float]:
    """
    Returns (buy_price, sell_price) using best bid/ask if possible, otherwise approximate spread.
    """
    bid, ask = get_best_bid_ask(symbol)
    if bid and ask and ask > bid:
        return ask, bid
    # fallback: use last price from level1 and approximate spread
    try:
        data = kucoin_get("/api/v1/market/orderbook/level1", {"symbol": symbol})
        last = safe_float(data["data"].get("price"))
        if last <= 0:
            raise ValueError("no last")
        spread = last * (DEFAULT_SPREAD_BPS / 10000.0)
        buy = last + spread / 2
        sell = last - spread / 2
        return buy, sell
    except Exception:
        return 0.0, 0.0

def log_trade(mode: str, symbol: str, side: str, qty: float, stake: float,
              entry_price: float, exit_price: float,
              gross: float, fees: float, slip_cost: float, net: float, reason: str) -> None:
    row = [
        ts_unix(),
        utc_now().isoformat(),
        "KuCoin",
        mode,
        symbol,
        side,
        f"{qty:.12f}",
        f"{stake:.2f}",
        f"{entry_price:.12f}",
        f"{exit_price:.12f}",
        f"{gross:.6f}",
        f"{fees:.6f}",
        f"{slip_cost:.6f}",
        f"{net:.6f}",
        reason,
    ]
    append_csv(MOCK_LOG_FILE if mode == "mock" else REAL_LOG_FILE, row)

# -----------------------------
# Momentum Strategy (LONG only)
# -----------------------------
def momentum_signal(candles: List[dict], threshold: float) -> bool:
    """
    LONG signal:
    - Price above EMA20 & EMA50
    - RSI(14) > 55
    - Breaks above last N highs by threshold (momentum)
    """
    if len(candles) < 60:
        return False

    closes = [c["c"] for c in candles]
    highs = [c["h"] for c in candles]

    e20 = ema(closes, 20)[-1]
    e50 = ema(closes, 50)[-1]
    r = rsi(closes, 14)[-1]
    last = closes[-1]

    if not (last > e20 > 0 and last > e50 > 0):
        return False
    if r < 55:
        return False

    # Breakout above recent high
    lookback = 20
    prev_high = max(highs[-(lookback+1):-1])
    if prev_high <= 0:
        return False

    # require "meaningful" push above high
    return last >= prev_high * (1.0 + threshold)

def make_levels(entry: float, tp: float, sl: float) -> Tuple[float, float]:
    return entry * (1.0 + tp), entry * (1.0 - sl)

# -----------------------------
# Trading engine
# -----------------------------
async def engine_tick(app: Application) -> None:
    s: Settings = STATE["settings"]
    positions: Dict[str, Position] = STATE["positions"]

    # loop coins, but throttle per coin
    for symbol in list(s.coins):
        now = ts_unix()
        last_check = STATE["last_check_ts"].get(symbol, 0)
        if now - last_check < PRICE_POLL_COOLDOWN_PER_COIN:
            continue
        STATE["last_check_ts"][symbol] = now

        # manage existing position
        if symbol in positions:
            await manage_position(app, symbol)
            continue

        # open new positions only if under max
        if len(positions) >= s.max_positions:
            continue

        # cooldown after trade
        last_trade = STATE["last_trade_ts"].get(symbol, 0)
        if now - last_trade < s.cooldown_seconds:
            continue

        await maybe_open(app, symbol)

async def maybe_open(app: Application, symbol: str) -> None:
    s: Settings = STATE["settings"]
    positions: Dict[str, Position] = STATE["positions"]

    # fetch candles in a thread (requests is blocking)
    candles = await asyncio.to_thread(get_klines, symbol, s.timeframe, 120)
    if not candles:
        return

    if not momentum_signal(candles, s.threshold):
        return

    # Determine fill (mock: buy at ask + slippage)
    buy_px_raw, sell_px_raw = await asyncio.to_thread(mock_fill_prices, symbol)
    if buy_px_raw <= 0:
        return

    entry = apply_slippage(buy_px_raw, "BUY", s.slippage)
    qty = compute_qty(s.stake_usdt, entry)
    if qty <= 0:
        return

    tp_price, sl_price = make_levels(entry, s.tp, s.sl)

    pos = Position(
        symbol=symbol,
        side="LONG",
        qty=qty,
        entry_price=entry,
        entry_time=ts_unix(),
        tp_price=tp_price,
        sl_price=sl_price,
        trail_on=False,
        trail_stop=None,
        high_water=entry,
    )
    positions[symbol] = pos

    if s.notify:
        txt = (
            f"🟢 ENTRY {symbol} LONG @ {entry:.6f}\n"
            f"TP={tp_price:.6f} | SL={sl_price:.6f}\n"
            f"stake={s.stake_usdt:.2f} | TF={s.timeframe} | thr={pct(s.threshold)}"
        )
        try:
            await app.bot.send_message(chat_id=get_admin_chat_id(app), text=txt)
        except Exception:
            pass

async def manage_position(app: Application, symbol: str) -> None:
    s: Settings = STATE["settings"]
    positions: Dict[str, Position] = STATE["positions"]
    pnl: PnL = STATE["pnl"]

    pos = positions.get(symbol)
    if not pos:
        return

    buy_px_raw, sell_px_raw = await asyncio.to_thread(mock_fill_prices, symbol)
    if sell_px_raw <= 0:
        return

    # mark-to-market on sell side (exit would be at bid - slippage)
    px = apply_slippage(sell_px_raw, "SELL", s.slippage)

    # trailing activation
    if not pos.trail_on:
        if px >= pos.entry_price * (1.0 + s.trail_activate):
            pos.trail_on = True
            pos.high_water = px
            pos.trail_stop = px * (1.0 - s.trail_dist)
    else:
        if px > pos.high_water:
            pos.high_water = px
            pos.trail_stop = pos.high_water * (1.0 - s.trail_dist)

    # exit checks
    reason = None
    if px <= pos.sl_price:
        reason = "SL"
    elif px >= pos.tp_price:
        reason = "TP"
    elif pos.trail_on and pos.trail_stop is not None and px <= pos.trail_stop:
        reason = "TRAIL_STOP"

    if not reason:
        return

    # close position (mock)
    stake = s.stake_usdt
    gross = (px - pos.entry_price) * pos.qty

    # fees: on notional buy and sell
    notional_buy = pos.entry_price * pos.qty
    notional_sell = px * pos.qty
    fees = fee(notional_buy, s.fee_rate) + fee(notional_sell, s.fee_rate)

    # slippage cost approx (already baked into prices), track as 0 for simplicity
    slip_cost = 0.0
    net = gross - fees - slip_cost

    pnl.trades += 1
    pnl.net_usdt += net

    STATE["last_trade_ts"][symbol] = ts_unix()

    # log
    log_trade(
        mode=s.trade_mode,
        symbol=symbol,
        side=pos.side,
        qty=pos.qty,
        stake=stake,
        entry_price=pos.entry_price,
        exit_price=px,
        gross=gross,
        fees=fees,
        slip_cost=slip_cost,
        net=net,
        reason=reason,
    )

    # notify
    if s.notify:
        txt = (
            f"🔴 EXIT {symbol} @ {px:.6f}\n"
            f"Net {net:+.4f} USDT | {reason}\n"
            f"Trades={pnl.trades} | Total NET={pnl.net_usdt:+.4f} USDT"
        )
        try:
            await app.bot.send_message(chat_id=get_admin_chat_id(app), text=txt)
        except Exception:
            pass

    # remove position
    positions.pop(symbol, None)

# -----------------------------
# Telegram UI
# -----------------------------
def main_keyboard() -> ReplyKeyboardMarkup:
    kb = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/threshold"), KeyboardButton("/stake")],
        [KeyboardButton("/tp"), KeyboardButton("/sl")],
        [KeyboardButton("/coins"), KeyboardButton("/trade_mode")],
        [KeyboardButton("/notify"), KeyboardButton("/export_csv")],
        [KeyboardButton("/close_all"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True)

def parse_pct_arg(arg: str) -> Optional[float]:
    # accepts "0.3" meaning 0.3% OR "0.003" meaning 0.3%? We'll support both.
    v = safe_float(arg, None)
    if v is None:
        return None
    if v > 1.0:
        # treat as percent, e.g. 0.3 -> 0.3%? Actually 0.3 is not >1. Keep.
        pass
    # If user inputs 0.30 (likely percent), interpret as 0.30%
    if v >= 0.01:
        return v / 100.0
    return v

def fmt_settings() -> str:
    s: Settings = STATE["settings"]
    pnl: PnL = STATE["pnl"]
    positions: Dict[str, Position] = STATE["positions"]
    coins = ", ".join(s.coins)
    open_pos = ", ".join(list(positions.keys())) if positions else "none"
    return (
        f"ENGINE: {'ON' if s.engine_on else 'OFF'}\n"
        f"Strategy: Momentum Breakout (LONG only)\n"
        f"Trade mode: {s.trade_mode}\n"
        f"Timeframe: {s.timeframe}\n"
        f"Threshold (/threshold): {pct(s.threshold)}\n"
        f"Stake per trade (/stake): {s.stake_usdt:.2f} USDT\n"
        f"TP/SL: {pct(s.tp)} / {pct(s.sl)}\n"
        f"Trailing: activate {pct(s.trail_activate)} | dist {pct(s.trail_dist)}\n"
        f"Cooldown/coin: {s.cooldown_seconds}s | Max positions: {s.max_positions}\n"
        f"Fees: {pct(s.fee_rate)} per side | Slippage: {pct(s.slippage)} per side\n"
        f"Trades: {pnl.trades}\n"
        f"Total NET PnL: {pnl.net_usdt:+.4f} USDT\n"
        f"Coins ({len(s.coins)}): [{coins}]\n"
        f"Open positions: {open_pos}\n"
        f"Notify: {'ON' if s.notify else 'OFF'}"
    )

# Admin chat id handling (first user who uses /start becomes admin)
def get_admin_chat_id(app: Application) -> Optional[int]:
    return app.bot_data.get("admin_chat_id")

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if app.bot_data.get("admin_chat_id") is None:
        app.bot_data["admin_chat_id"] = update.effective_chat.id

    await update.message.reply_text(
        "✅ Mp ORBbot är igång.\nAnvänd knapparna nedan.",
        reply_markup=main_keyboard()
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(fmt_settings(), reply_markup=main_keyboard())

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pnl: PnL = STATE["pnl"]
    await update.message.reply_text(
        f"Total NET PnL: {pnl.net_usdt:+.4f} USDT\nTrades: {pnl.trades}",
        reply_markup=main_keyboard(),
    )

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE["settings"].engine_on = True
    await update.message.reply_text("✅ ENGINE ON", reply_markup=main_keyboard())

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE["settings"].engine_on = False
    await update.message.reply_text("🛑 ENGINE OFF", reply_markup=main_keyboard())

async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: Settings = STATE["settings"]
    s.notify = not s.notify
    await update.message.reply_text(f"Notify: {'ON' if s.notify else 'OFF'}", reply_markup=main_keyboard())

async def cmd_trade_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: Settings = STATE["settings"]
    # keep safe: only mock for now
    if context.args and context.args[0].lower() in ("mock", "live"):
        s.trade_mode = context.args[0].lower()
    await update.message.reply_text(
        f"Trade mode: {s.trade_mode}\n(OBS: live är reserverat för senare)",
        reply_markup=main_keyboard(),
    )

async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: Settings = STATE["settings"]
    if context.args:
        v = parse_pct_arg(context.args[0])
        if v is None:
            await update.message.reply_text("Ex: /threshold 0.30  (dvs 0.30%)", reply_markup=main_keyboard())
            return
        s.threshold = clamp(v, 0.0001, 0.02)
    await update.message.reply_text(f"Threshold nu: {pct(s.threshold)}\nEx: /threshold 0.30", reply_markup=main_keyboard())

async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: Settings = STATE["settings"]
    if context.args:
        v = safe_float(context.args[0], None)
        if v is None or v <= 0:
            await update.message.reply_text("Ex: /stake 30", reply_markup=main_keyboard())
            return
        s.stake_usdt = float(v)
    await update.message.reply_text(f"Stake per trade nu: {s.stake_usdt:.2f} USDT\nEx: /stake 30", reply_markup=main_keyboard())

async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: Settings = STATE["settings"]
    if context.args:
        v = parse_pct_arg(context.args[0])
        if v is None:
            await update.message.reply_text("Ex: /tp 0.45  (dvs 0.45%)", reply_markup=main_keyboard())
            return
        s.tp = clamp(v, 0.0001, 0.05)
    await update.message.reply_text(f"TP nu: {pct(s.tp)}\nEx: /tp 0.45", reply_markup=main_keyboard())

async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: Settings = STATE["settings"]
    if context.args:
        v = parse_pct_arg(context.args[0])
        if v is None:
            await update.message.reply_text("Ex: /sl 0.30  (dvs 0.30%)", reply_markup=main_keyboard())
            return
        s.sl = clamp(v, 0.0001, 0.05)
    await update.message.reply_text(f"SL nu: {pct(s.sl)}\nEx: /sl 0.30", reply_markup=main_keyboard())

async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s: Settings = STATE["settings"]
    if context.args:
        # Accept: /coins BTC-USDT ETH-USDT ...
        new = []
        for a in context.args:
            a = a.strip().upper()
            if "-" not in a:
                continue
            new.append(a)
        if new:
            s.coins = new
            await update.message.reply_text(f"✅ Coins uppdaterade ({len(new)}): {s.coins}", reply_markup=main_keyboard())
            return
    await update.message.reply_text(
        "Ange coins så här:\n/coins BTC-USDT ETH-USDT XRP-USDT\n\nNuvarande:\n" + ", ".join(s.coins),
        reply_markup=main_keyboard(),
    )

async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE["positions"].clear()
    await update.message.reply_text("✅ Stängde alla (mock) positioner (cleared).", reply_markup=main_keyboard())

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE["pnl"] = PnL()
    await update.message.reply_text("✅ PnL reset.", reply_markup=main_keyboard())

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Send mock_trade_log.csv as document
    path = MOCK_LOG_FILE
    if not os.path.exists(path):
        await update.message.reply_text("Ingen CSV hittades än.", reply_markup=main_keyboard())
        return
    try:
        await update.message.reply_document(document=open(path, "rb"), filename=os.path.basename(path))
    except Exception as e:
        await update.message.reply_text(f"Kunde inte skicka CSV: {e}", reply_markup=main_keyboard())

# -----------------------------
# Healthcheck tiny HTTP server
# -----------------------------
def start_health_server():
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/health", "/healthz"):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"OK\n")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            return  # quiet

    try:
        with socketserver.TCPServer(("", DEFAULT_PORT), Handler) as httpd:
            httpd.serve_forever()
    except Exception:
        # If port binding fails, ignore (worker type apps may not need it)
        pass

# -----------------------------
# App lifecycle
# -----------------------------
async def engine_loop(app: Application) -> None:
    # runs forever inside PTB event loop (no JobQueue needed)
    while True:
        try:
            if STATE["settings"].engine_on:
                await engine_tick(app)
        except Exception:
            # never crash the process
            pass
        await asyncio.sleep(ENGINE_TICK_SECONDS)

async def post_init(app: Application) -> None:
    # Start engine loop
    app.create_task(engine_loop(app))

def build_app(token: str) -> Application:
    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))

    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))

    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("sl", cmd_sl))

    app.add_handler(CommandHandler("coins", cmd_coins))
    app.add_handler(CommandHandler("trade_mode", cmd_trade_mode))
    app.add_handler(CommandHandler("notify", cmd_notify))

    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))

    return app

def get_token() -> str:
    token = (
        os.getenv("TELEGRAM_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM")
        or ""
    ).strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN env var")
    return token

def main() -> None:
    # Start health server thread (helps if platform expects PORT open)
    t = threading.Thread(target=start_health_server, daemon=True)
    t.start()

    token = get_token()
    app = build_app(token)

    # IMPORTANT: prevents conflict issues if there were queued updates
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
