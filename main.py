# main.py
# Mp ORBbot (Momentum edition) — single-file, no aiohttp
# Works with requirements.txt:
#   requests
#   python-telegram-bot==20.3
#
# ✅ Momentum strategy (LONG only) using KuCoin public data
# ✅ Mock + live (live needs KuCoin keys)
# ✅ Fees + bid/ask spread + optional mock slippage
# ✅ Telegram buttons for: coins, stake, tp, sl, threshold, timeframe, trade_mode, notify
# ✅ CSV export: mock_trade_log.csv + real_trade_log.csv
# ✅ DO App Platform friendly: starts tiny HTTP server on $PORT for health checks

import os
import csv
import json
import time
import hmac
import base64
import hashlib
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mporb_momentum")

# ----------------------------
# Files
# ----------------------------
STATE_FILE = "state.json"
MOCK_LOG_FILE = "mock_trade_log.csv"
REAL_LOG_FILE = "real_trade_log.csv"

# ----------------------------
# KuCoin endpoints (public)
# ----------------------------
KUCOIN_BASE = "https://api.kucoin.com"

# KuCoin candle types:
# 1min, 3min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 6hour, 8hour, 12hour, 1day, 1week
TF_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
}

# ----------------------------
# Default config
# ----------------------------
DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

DEFAULTS = {
    "engine_on": False,
    "trade_mode": "mock",          # "mock" or "live"
    "notify": True,
    "coins": DEFAULT_COINS[:5],
    "max_positions": 5,
    "stake_usdt": 30.0,            # per trade
    "timeframe": "1m",             # 1m / 3m / 5m / 15m
    "threshold_pct": 0.10,         # breakout strength in % (close must be above prevHigh by this %)
    "tp_pct": 0.40,                # take profit in %
    "sl_pct": 0.25,                # stop loss in %
    "trail_activate_pct": 0.30,    # activate trailing after this profit %
    "trail_dist_pct": 0.18,        # trailing distance %
    "spread_max_pct": 0.20,        # skip entries if spread > this %
    "cooldown_sec": 60,            # per-coin cooldown after exit/entry attempt
    "slippage_pct_mock": 0.02,     # mock slippage (percent) applied to fills (each side)
    "fee_rate": 0.001,             # fee per side (0.1%)
    "breakout_lookback": 20,       # N candles high breakout
    "vol_lookback": 20,            # volume avg length
    "vol_factor": 1.20,            # volume must be > avg * factor
    "ema_fast": 9,
    "ema_slow": 21,
}

# Telegram confirmation window for sensitive actions
CONFIRM_WINDOW_SEC = 30

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Position:
    symbol: str
    qty: float
    entry_px: float
    entry_time: float
    mode: str  # mock/live
    tp_px: float
    sl_px: float
    trail_active: bool = False
    trail_stop_px: Optional[float] = None
    highest_px: Optional[float] = None


# ----------------------------
# State
# ----------------------------
_state_lock = threading.Lock()
STATE: Dict = {}

_session = requests.Session()
_session.headers.update({"User-Agent": "MpORBot/1.0"})


def utc_now_ts() -> float:
    return time.time()


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            for k, v in DEFAULTS.items():
                s.setdefault(k, v)
            s.setdefault("positions", {})
            s.setdefault("pnl_net", 0.0)
            s.setdefault("trades", 0)
            s.setdefault("cooldowns", {})
            s.setdefault("pending_confirm", {})
            return s
        except Exception:
            log.exception("Failed to load state, using defaults.")
    s = dict(DEFAULTS)
    s["positions"] = {}
    s["pnl_net"] = 0.0
    s["trades"] = 0
    s["cooldowns"] = {}
    s["pending_confirm"] = {}
    return s


def save_state() -> None:
    with _state_lock:
        tmp = dict(STATE)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(tmp, f, indent=2)


def ensure_csv_headers(path: str) -> None:
    if os.path.exists(path):
        return
    headers = [
        "timestamp_unix",
        "timestamp_utc",
        "exchange",
        "mode",
        "symbol",
        "side",
        "qty",
        "stake_usdt",
        "entry_px",
        "exit_px",
        "gross_pnl_usdt",
        "fees_usdt",
        "slippage_usdt",
        "spread_cost_usdt",
        "net_pnl_usdt",
        "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(headers)


def append_trade_log(path: str, row: List) -> None:
    ensure_csv_headers(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ----------------------------
# Indicators
# ----------------------------
def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


# ----------------------------
# KuCoin public data
# ----------------------------
def kucoin_level1(symbol: str) -> Optional[Tuple[float, float, float]]:
    """Returns (bid, ask, last)."""
    try:
        r = _session.get(f"{KUCOIN_BASE}/api/v1/market/orderbook/level1", params={"symbol": symbol}, timeout=10)
        j = r.json()
        if j.get("code") != "200000":
            return None
        data = j["data"]
        bid = float(data["bestBid"])
        ask = float(data["bestAsk"])
        last = float(data["price"])
        return bid, ask, last
    except Exception:
        return None


def kucoin_candles(symbol: str, tf: str, limit: int) -> Optional[List[Dict]]:
    """
    Returns candles newest->oldest on KuCoin.
    Candle format from KuCoin: [time, open, close, high, low, volume, turnover]
    """
    try:
        ktype = TF_MAP.get(tf, "1min")
        r = _session.get(
            f"{KUCOIN_BASE}/api/v1/market/candles",
            params={"symbol": symbol, "type": ktype},
            timeout=10,
        )
        j = r.json()
        if j.get("code") != "200000":
            return None
        arr = j["data"]  # newest first
        arr = arr[: max(limit, 30)]
        out = []
        for c in arr:
            out.append({
                "ts": int(c[0]),
                "open": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "vol": float(c[5]),
            })
        return out
    except Exception:
        return None


# ----------------------------
# KuCoin private (LIVE) — optional
# ----------------------------
def _kucoin_sign(secret: str, str_to_sign: str) -> str:
    return base64.b64encode(hmac.new(secret.encode("utf-8"), str_to_sign.encode("utf-8"), hashlib.sha256).digest()).decode("utf-8")


def kucoin_private_request(method: str, path: str, body: str = "") -> Dict:
    """
    Minimal KuCoin REST signer.
    Needs env:
      KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE
    """
    key = os.getenv("KUCOIN_KEY", "").strip()
    secret = os.getenv("KUCOIN_SECRET", "").strip()
    passphrase = os.getenv("KUCOIN_PASSPHRASE", "").strip()

    if not (key and secret and passphrase):
        raise RuntimeError("Missing KuCoin live keys (KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE).")

    now_ms = str(int(time.time() * 1000))
    str_to_sign = now_ms + method.upper() + path + body
    signature = _kucoin_sign(secret, str_to_sign)
    passphrase_signed = _kucoin_sign(secret, passphrase)

    headers = {
        "KC-API-KEY": key,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": now_ms,
        "KC-API-PASSPHRASE": passphrase_signed,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }

    url = KUCOIN_BASE + path
    if method.upper() == "GET":
        r = _session.get(url, headers=headers, timeout=15)
    elif method.upper() == "POST":
        r = _session.post(url, headers=headers, data=body, timeout=15)
    else:
        raise RuntimeError("Unsupported method")
    return r.json()


def kucoin_place_market_buy(symbol: str, funds_usdt: float) -> Tuple[float, float]:
    """Market BUY using funds (USDT). Returns (filled_qty_base, avg_px)."""
    body = json.dumps({
        "clientOid": f"mporb-{int(time.time()*1000)}",
        "side": "buy",
        "symbol": symbol,
        "type": "market",
        "funds": str(funds_usdt),
    })
    j = kucoin_private_request("POST", "/api/v1/orders", body=body)
    if j.get("code") != "200000":
        raise RuntimeError(f"KuCoin order failed: {j}")

    order_id = j["data"]["orderId"]
    info = kucoin_private_request("GET", f"/api/v1/orders/{order_id}")
    if info.get("code") != "200000":
        raise RuntimeError(f"KuCoin order fetch failed: {info}")
    d = info["data"]
    deal_size = float(d.get("dealSize") or 0.0)
    deal_funds = float(d.get("dealFunds") or 0.0)
    avg_px = (deal_funds / deal_size) if deal_size > 0 else 0.0
    return deal_size, avg_px


def kucoin_place_market_sell(symbol: str, size_base: float) -> Tuple[float, float]:
    """Market SELL using size (base). Returns (filled_qty_base, avg_px)."""
    body = json.dumps({
        "clientOid": f"mporb-{int(time.time()*1000)}",
        "side": "sell",
        "symbol": symbol,
        "type": "market",
        "size": str(size_base),
    })
    j = kucoin_private_request("POST", "/api/v1/orders", body=body)
    if j.get("code") != "200000":
        raise RuntimeError(f"KuCoin order failed: {j}")

    order_id = j["data"]["orderId"]
    info = kucoin_private_request("GET", f"/api/v1/orders/{order_id}")
    if info.get("code") != "200000":
        raise RuntimeError(f"KuCoin order fetch failed: {info}")
    d = info["data"]
    deal_size = float(d.get("dealSize") or 0.0)
    deal_funds = float(d.get("dealFunds") or 0.0)
    avg_px = (deal_funds / deal_size) if deal_size > 0 else 0.0
    return deal_size, avg_px


# ----------------------------
# Momentum strategy (LONG)
# ----------------------------
def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0


def can_trade_symbol(symbol: str) -> bool:
    now = utc_now_ts()
    cd = STATE.get("cooldowns", {}).get(symbol, 0)
    return now >= cd


def set_cooldown(symbol: str, seconds: int) -> None:
    STATE.setdefault("cooldowns", {})
    STATE["cooldowns"][symbol] = utc_now_ts() + max(0, seconds)


def momentum_signal(symbol: str) -> Tuple[bool, str, Dict]:
    """Returns (should_buy, reason, debug)"""
    tf = STATE["timeframe"]
    look = max(int(STATE["breakout_lookback"]) + 5, 40)

    candles = kucoin_candles(symbol, tf, look)
    if not candles or len(candles) < (int(STATE["breakout_lookback"]) + 10):
        return False, "no_candles", {}

    ob = kucoin_level1(symbol)
    if not ob:
        return False, "no_orderbook", {}
    bid, ask, last = ob

    sp = spread_pct(bid, ask)
    if sp > float(STATE["spread_max_pct"]):
        return False, f"spread_too_high({sp:.3f}%)", {"spread_pct": sp}

    # reverse to old->new for indicators
    c = list(reversed(candles))
    closes = [x["close"] for x in c]
    highs = [x["high"] for x in c]
    vols = [x["vol"] for x in c]

    ema_fast_v = ema(closes[-(int(STATE["ema_fast"]) + 50):], int(STATE["ema_fast"]))
    ema_slow_v = ema(closes[-(int(STATE["ema_slow"]) + 50):], int(STATE["ema_slow"]))
    if ema_fast_v is None or ema_slow_v is None:
        return False, "no_ema", {}

    if not (ema_fast_v > ema_slow_v):
        return False, "ema_not_up", {"ema_fast": ema_fast_v, "ema_slow": ema_slow_v}

    N = int(STATE["breakout_lookback"])
    last_close = closes[-1]
    last_vol = vols[-1]
    prev_high = max(highs[-(N + 1):-1])  # prev N candles, exclude current

    strength_req = float(STATE["threshold_pct"]) / 100.0
    if last_close <= prev_high * (1.0 + strength_req):
        return False, "no_break_strength", {"close": last_close, "prev_high": prev_high}

    vN = int(STATE["vol_lookback"])
    avg_vol = sum(vols[-(vN + 1):-1]) / max(1, vN)
    if avg_vol > 0 and last_vol < avg_vol * float(STATE["vol_factor"]):
        return False, "vol_too_low", {"vol": last_vol, "avg_vol": avg_vol}

    dbg = {
        "close": last_close,
        "prev_high": prev_high,
        "ema_fast": ema_fast_v,
        "ema_slow": ema_slow_v,
        "spread_pct": sp,
        "vol": last_vol,
        "avg_vol": avg_vol,
    }
    return True, "momentum_break", dbg


# ----------------------------
# Execution (fees/slippage/spread)
# ----------------------------
def fee(amount_usdt: float) -> float:
    return float(amount_usdt) * float(STATE["fee_rate"])


def mock_fill_buy(symbol: str, stake_usdt: float) -> Tuple[float, float, float, float]:
    """Buy at ask + slippage. Returns (qty, fill_px, slippage_cost_usdt, spread_cost_usdt)."""
    ob = kucoin_level1(symbol)
    if not ob:
        return 0.0, 0.0, 0.0, 0.0
    bid, ask, _ = ob
    slip = float(STATE["slippage_pct_mock"]) / 100.0
    fill_px = ask * (1.0 + slip)
    qty = stake_usdt / fill_px if fill_px > 0 else 0.0
    mid = (bid + ask) / 2.0
    spread_cost = qty * max(0.0, (ask - mid))
    slip_cost = qty * max(0.0, (fill_px - ask))
    return qty, fill_px, slip_cost, spread_cost


def mock_fill_sell(symbol: str, qty_base: float) -> Tuple[float, float, float, float]:
    """Sell at bid - slippage. Returns (proceeds, fill_px, slippage_cost_usdt, spread_cost_usdt)."""
    ob = kucoin_level1(symbol)
    if not ob:
        return 0.0, 0.0, 0.0, 0.0
    bid, ask, _ = ob
    slip = float(STATE["slippage_pct_mock"]) / 100.0
    fill_px = bid * (1.0 - slip)
    proceeds = qty_base * fill_px
    mid = (bid + ask) / 2.0
    spread_cost = qty_base * max(0.0, (mid - bid))
    slip_cost = qty_base * max(0.0, (bid - fill_px))
    return proceeds, fill_px, slip_cost, spread_cost


def open_position(symbol: str, context: ContextTypes.DEFAULT_TYPE, reason: str, dbg: Dict) -> None:
    with _state_lock:
        if symbol in STATE["positions"]:
            return
        if len(STATE["positions"]) >= int(STATE["max_positions"]):
            return

    stake = float(STATE["stake_usdt"])
    mode = STATE["trade_mode"]

    if mode == "mock":
        qty, entry_px, slip_cost, spread_cost = mock_fill_buy(symbol, stake)
        if qty <= 0 or entry_px <= 0:
            return
        entry_fee = fee(stake)

        tp_px = entry_px * (1.0 + float(STATE["tp_pct"]) / 100.0)
        sl_px = entry_px * (1.0 - float(STATE["sl_pct"]) / 100.0)

        pos = Position(
            symbol=symbol,
            qty=qty,
            entry_px=entry_px,
            entry_time=utc_now_ts(),
            mode="mock",
            tp_px=tp_px,
            sl_px=sl_px,
            trail_active=False,
            trail_stop_px=None,
            highest_px=entry_px,
        )

        with _state_lock:
            STATE["positions"][symbol] = asdict(pos)
            STATE.setdefault("_entry_costs", {})
            STATE["_entry_costs"][symbol] = {
                "stake": stake,
                "entry_fee": entry_fee,
                "entry_slip": slip_cost,
                "entry_spread": spread_cost,
            }
            set_cooldown(symbol, int(STATE["cooldown_sec"]))
            save_state()

        if STATE["notify"] and STATE.get("_chat_id"):
            msg = (
                f"🟢 ENTRY {symbol} LONG @ {entry_px:.6f}\n"
                f"Reason: {reason}\n"
                f"TP={tp_px:.6f} ({STATE['tp_pct']:.2f}%) | SL={sl_px:.6f} ({STATE['sl_pct']:.2f}%)\n"
                f"TF={STATE['timeframe']} | threshold={STATE['threshold_pct']:.2f}% | spread={dbg.get('spread_pct',0):.3f}%"
            )
            context.application.create_task(
                context.bot.send_message(chat_id=STATE["_chat_id"], text=msg)
            )
        return

    # LIVE
    try:
        filled_qty, avg_px = kucoin_place_market_buy(symbol, stake)
        if filled_qty <= 0 or avg_px <= 0:
            return

        tp_px = avg_px * (1.0 + float(STATE["tp_pct"]) / 100.0)
        sl_px = avg_px * (1.0 - float(STATE["sl_pct"]) / 100.0)

        pos = Position(
            symbol=symbol,
            qty=filled_qty,
            entry_px=avg_px,
            entry_time=utc_now_ts(),
            mode="live",
            tp_px=tp_px,
            sl_px=sl_px,
            trail_active=False,
            trail_stop_px=None,
            highest_px=avg_px,
        )

        with _state_lock:
            STATE["positions"][symbol] = asdict(pos)
            set_cooldown(symbol, int(STATE["cooldown_sec"]))
            save_state()

        if STATE["notify"] and STATE.get("_chat_id"):
            msg = f"🟢 LIVE ENTRY {symbol} LONG @ {avg_px:.6f}\nTP={tp_px:.6f} | SL={sl_px:.6f} | TF={STATE['timeframe']}"
            context.application.create_task(context.bot.send_message(chat_id=STATE["_chat_id"], text=msg))

    except Exception as e:
        log.exception("Live entry failed")
        if STATE["notify"] and STATE.get("_chat_id"):
            context.application.create_task(
                context.bot.send_message(chat_id=STATE["_chat_id"], text=f"⚠️ LIVE entry failed: {e}")
            )


def update_trailing(pos: Position, last: float) -> Position:
    if pos.highest_px is None:
        pos.highest_px = pos.entry_px
    if last > pos.highest_px:
        pos.highest_px = last

    if not pos.trail_active:
        if last >= pos.entry_px * (1.0 + float(STATE["trail_activate_pct"]) / 100.0):
            pos.trail_active = True

    if pos.trail_active:
        dist = float(STATE["trail_dist_pct"]) / 100.0
        ts = pos.highest_px * (1.0 - dist)
        if pos.trail_stop_px is None or ts > pos.trail_stop_px:
            pos.trail_stop_px = ts
    return pos


def close_position(symbol: str, context: ContextTypes.DEFAULT_TYPE, reason: str) -> None:
    with _state_lock:
        pd = STATE["positions"].get(symbol)
        if not pd:
            return
        pos = Position(**pd)

    exit_ts = utc_now_ts()
    exchange = "KuCoin"

    if pos.mode == "mock":
        proceeds, exit_px, slip_exit, spread_exit = mock_fill_sell(symbol, pos.qty)
        if proceeds <= 0 or exit_px <= 0:
            return

        entry_costs = STATE.get("_entry_costs", {}).get(symbol, {})
        stake = float(entry_costs.get("stake", pos.qty * pos.entry_px))
        entry_fee = float(entry_costs.get("entry_fee", fee(stake)))
        entry_slip = float(entry_costs.get("entry_slip", 0.0))
        entry_spread = float(entry_costs.get("entry_spread", 0.0))

        exit_fee = fee(proceeds)

        gross = proceeds - stake
        fees_total = entry_fee + exit_fee
        slip_total = entry_slip + slip_exit
        spread_total = entry_spread + spread_exit
        net = gross - fees_total - slip_total - spread_total

        with _state_lock:
            STATE["pnl_net"] = float(STATE["pnl_net"]) + float(net)
            STATE["trades"] = int(STATE["trades"]) + 1
            STATE["positions"].pop(symbol, None)
            STATE.get("_entry_costs", {}).pop(symbol, None)
            set_cooldown(symbol, int(STATE["cooldown_sec"]))
            save_state()

        append_trade_log(
            MOCK_LOG_FILE,
            [
                int(exit_ts),
                iso_utc(exit_ts),
                exchange,
                "mock",
                symbol,
                "LONG",
                f"{pos.qty:.12f}",
                f"{stake:.2f}",
                f"{pos.entry_px:.12f}",
                f"{exit_px:.12f}",
                f"{gross:.6f}",
                f"{fees_total:.6f}",
                f"{slip_total:.6f}",
                f"{spread_total:.6f}",
                f"{net:.6f}",
                reason,
            ],
        )

        if STATE["notify"] and STATE.get("_chat_id"):
            emoji = "🎯" if net >= 0 else "🟥"
            msg = (
                f"{emoji} EXIT {symbol} @ {exit_px:.6f}\n"
                f"Net {net:+.4f} USDT | reason={reason}\n"
                f"Total NET PnL: {STATE['pnl_net']:+.4f} USDT | Trades: {STATE['trades']}"
            )
            context.application.create_task(context.bot.send_message(chat_id=STATE["_chat_id"], text=msg))
        return

    # LIVE exit
    try:
        filled_qty, avg_px = kucoin_place_market_sell(symbol, pos.qty)
        if filled_qty <= 0 or avg_px <= 0:
            return

        stake = pos.qty * pos.entry_px
        proceeds = filled_qty * avg_px
        gross = proceeds - stake
        fees_total = fee(stake) + fee(proceeds)
        net = gross - fees_total

        with _state_lock:
            STATE["pnl_net"] = float(STATE["pnl_net"]) + float(net)
            STATE["trades"] = int(STATE["trades"]) + 1
            STATE["positions"].pop(symbol, None)
            set_cooldown(symbol, int(STATE["cooldown_sec"]))
            save_state()

        append_trade_log(
            REAL_LOG_FILE,
            [
                int(exit_ts),
                iso_utc(exit_ts),
                exchange,
                "live",
                symbol,
                "LONG",
                f"{filled_qty:.12f}",
                f"{stake:.2f}",
                f"{pos.entry_px:.12f}",
                f"{avg_px:.12f}",
                f"{gross:.6f}",
                f"{fees_total:.6f}",
                "0.000000",
                "0.000000",
                f"{net:.6f}",
                reason,
            ],
        )

        if STATE["notify"] and STATE.get("_chat_id"):
            emoji = "🎯" if net >= 0 else "🟥"
            msg = f"{emoji} LIVE EXIT {symbol} @ {avg_px:.6f}\nNet {net:+.4f} USDT | reason={reason}"
            context.application.create_task(context.bot.send_message(chat_id=STATE["_chat_id"], text=msg))
    except Exception as e:
        log.exception("Live exit failed")
        if STATE["notify"] and STATE.get("_chat_id"):
            context.application.create_task(context.bot.send_message(chat_id=STATE["_chat_id"], text=f"⚠️ LIVE exit failed: {e}"))


# ----------------------------
# Engine tick
# ----------------------------
async def engine_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    with _state_lock:
        if not STATE.get("engine_on"):
            return
        coins = list(STATE.get("coins", []))
        positions = dict(STATE.get("positions", {}))
        max_pos = int(STATE.get("max_positions", 1))

    # Manage open positions
    for sym, pd in positions.items():
        pos = Position(**pd)
        ob = kucoin_level1(sym)
        if not ob:
            continue
        bid, ask, last = ob

        pos = update_trailing(pos, last)
        exit_px_trigger = bid  # conservative for LONG exits

        if exit_px_trigger <= pos.sl_px:
            with _state_lock:
                STATE["positions"][sym] = asdict(pos)
                save_state()
            close_position(sym, context, "SL")
            continue

        if exit_px_trigger >= pos.tp_px:
            with _state_lock:
                STATE["positions"][sym] = asdict(pos)
                save_state()
            close_position(sym, context, "TP")
            continue

        if pos.trail_active and pos.trail_stop_px is not None and exit_px_trigger <= pos.trail_stop_px:
            with _state_lock:
                STATE["positions"][sym] = asdict(pos)
                save_state()
            close_position(sym, context, "TRAIL_STOP")
            continue

        with _state_lock:
            STATE["positions"][sym] = asdict(pos)
            save_state()

    # Open new positions if capacity
    with _state_lock:
        if len(STATE.get("positions", {})) >= max_pos:
            return

    for sym in coins:
        with _state_lock:
            if sym in STATE["positions"]:
                continue
            if len(STATE["positions"]) >= max_pos:
                break

        if not can_trade_symbol(sym):
            continue

        ok, reason, dbg = momentum_signal(sym)
        if ok:
            open_position(sym, context, reason, dbg)


# ----------------------------
# Telegram UI (inline buttons)
# ----------------------------
def menu_keyboard() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("📊 status", callback_data="cmd:status"),
         InlineKeyboardButton("💰 pnl", callback_data="cmd:pnl")],
        [InlineKeyboardButton("✅ ENGINE ON", callback_data="cmd:engine_on"),
         InlineKeyboardButton("🛑 ENGINE OFF", callback_data="cmd:engine_off")],
        [InlineKeyboardButton("🧠 TF", callback_data="menu:tf"),
         InlineKeyboardButton("🎯 TP", callback_data="menu:tp")],
        [InlineKeyboardButton("🛡️ SL", callback_data="menu:sl"),
         InlineKeyboardButton("⚙️ THRESH", callback_data="menu:th")],
        [InlineKeyboardButton("💵 STAKE", callback_data="menu:stake"),
         InlineKeyboardButton("🪙 COINS", callback_data="menu:coins")],
        [InlineKeyboardButton("🔁 MODE", callback_data="menu:mode"),
         InlineKeyboardButton("🔔 NOTIFY", callback_data="cmd:notify")],
        [InlineKeyboardButton("📤 export_csv", callback_data="cmd:export_csv"),
         InlineKeyboardButton("🧹 reset_pnl", callback_data="cmd:reset_pnl")],
        [InlineKeyboardButton("❌ close_all", callback_data="cmd:close_all")],
    ]
    return InlineKeyboardMarkup(kb)


def options_keyboard(options: List[Tuple[str, str]]) -> InlineKeyboardMarkup:
    rows = []
    for i in range(0, len(options), 2):
        row = []
        for j in range(i, min(i + 2, len(options))):
            label, data = options[j]
            row.append(InlineKeyboardButton(label, callback_data=data))
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:main")])
    return InlineKeyboardMarkup(rows)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    with _state_lock:
        STATE["_chat_id"] = chat_id
        save_state()
    await update.message.reply_text("✅ Mp ORBbot (Momentum) online.\nUse menu:", reply_markup=menu_keyboard())


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with _state_lock:
        engine_on = STATE["engine_on"]
        mode = STATE["trade_mode"]
        tf = STATE["timeframe"]
        coins = STATE["coins"]
        pnl = float(STATE["pnl_net"])
        trades = int(STATE["trades"])
        th = float(STATE["threshold_pct"])
        tp = float(STATE["tp_pct"])
        sl = float(STATE["sl_pct"])
        stake = float(STATE["stake_usdt"])
        spread_max = float(STATE["spread_max_pct"])
        cooldown = int(STATE["cooldown_sec"])
        max_pos = int(STATE["max_positions"])
        notify = bool(STATE["notify"])
        pos_syms = list(STATE["positions"].keys())

    live_ok = bool(os.getenv("KUCOIN_KEY")) and bool(os.getenv("KUCOIN_SECRET")) and bool(os.getenv("KUCOIN_PASSPHRASE"))

    text = (
        f"ENGINE: {'ON' if engine_on else 'OFF'}\n"
        f"Strategy: Momentum Breakout (LONG only)\n"
        f"Trade mode: {mode} (live keys: {'OK' if live_ok else 'missing'})\n"
        f"TF: {tf}\n"
        f"Threshold: {th:.2f}% (break strength)\n"
        f"TP/SL: {tp:.2f}% / {sl:.2f}%\n"
        f"Trailing: activate +{STATE['trail_activate_pct']:.2f}% | dist {STATE['trail_dist_pct']:.2f}%\n"
        f"Spread max: {spread_max:.2f}%\n"
        f"Stake/trade: {stake:.2f} USDT\n"
        f"Cooldown: {cooldown}s | Max positions: {max_pos}\n"
        f"Trades: {trades}\n"
        f"Total NET PnL: {pnl:+.4f} USDT\n"
        f"Coins ({len(coins)}): {coins}\n"
        f"Open positions: {pos_syms if pos_syms else 'none'}\n"
        f"Notify: {'ON' if notify else 'OFF'}"
    )
    if update.message:
        await update.message.reply_text(text, reply_markup=menu_keyboard())
    else:
        await context.bot.send_message(chat_id=STATE["_chat_id"], text=text, reply_markup=menu_keyboard())


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with _state_lock:
        pnl = float(STATE["pnl_net"])
        trades = int(STATE["trades"])
    if update.message:
        await update.message.reply_text(f"Total NET PnL: {pnl:+.4f} USDT\nTrades: {trades}", reply_markup=menu_keyboard())


def set_pending(action: str, chat_id: int) -> None:
    STATE.setdefault("pending_confirm", {})
    STATE["pending_confirm"][str(chat_id)] = {"action": action, "ts": utc_now_ts()}
    save_state()


def check_pending(chat_id: int) -> Optional[str]:
    p = STATE.get("pending_confirm", {}).get(str(chat_id))
    if not p:
        return None
    if utc_now_ts() - float(p.get("ts", 0)) > CONFIRM_WINDOW_SEC:
        STATE["pending_confirm"].pop(str(chat_id), None)
        save_state()
        return None
    return p.get("action")


async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    set_pending("engine_on", chat_id)
    await update.message.reply_text("Skriv **JA** inom 30 sek för att starta motorn.", parse_mode="Markdown", reply_markup=menu_keyboard())


async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with _state_lock:
        STATE["engine_on"] = False
        save_state()
    await update.message.reply_text("🛑 ENGINE OFF", reply_markup=menu_keyboard())


async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with _state_lock:
        STATE["notify"] = not bool(STATE["notify"])
        v = STATE["notify"]
        save_state()
    await update.message.reply_text(f"Notify: {'ON' if v else 'OFF'}", reply_markup=menu_keyboard())


async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with _state_lock:
        STATE["pnl_net"] = 0.0
        STATE["trades"] = 0
        save_state()
    await update.message.reply_text("✅ PnL reset.", reply_markup=menu_keyboard())


async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with _state_lock:
        syms = list(STATE["positions"].keys())
    if not syms:
        await update.message.reply_text("Open positions: none", reply_markup=menu_keyboard())
        return
    for s in syms:
        close_position(s, context, "CLOSE_ALL")
    await update.message.reply_text("✅ Close all sent.", reply_markup=menu_keyboard())


async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    files = []
    if os.path.exists(MOCK_LOG_FILE):
        files.append(MOCK_LOG_FILE)
    if os.path.exists(REAL_LOG_FILE):
        files.append(REAL_LOG_FILE)

    if not files:
        await update.message.reply_text("Inga loggar ännu.", reply_markup=menu_keyboard())
        return

    for f in files:
        try:
            await update.message.reply_document(document=open(f, "rb"), filename=os.path.basename(f))
        except Exception as e:
            await update.message.reply_text(f"Kunde inte skicka {f}: {e}", reply_markup=menu_keyboard())


async def cmd_trade_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.strip().split()
    if len(parts) == 1:
        await update.message.reply_text(f"Trade mode nu: {STATE['trade_mode']}\nEx: /trade_mode mock  eller  /trade_mode live", reply_markup=menu_keyboard())
        return
    mode = parts[1].lower()
    if mode not in ("mock", "live"):
        await update.message.reply_text("Använd: /trade_mode mock  eller  /trade_mode live", reply_markup=menu_keyboard())
        return
    if mode == "live":
        set_pending("trade_mode_live", update.effective_chat.id)
        await update.message.reply_text("Skriv **JA** inom 30 sek för att slå på LIVE-läge.", parse_mode="Markdown", reply_markup=menu_keyboard())
        return
    with _state_lock:
        STATE["trade_mode"] = "mock"
        save_state()
    await update.message.reply_text("✅ Trade mode: mock", reply_markup=menu_keyboard())


async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.split()
    if len(parts) == 1:
        await update.message.reply_text(
            f"Threshold nu: {STATE['threshold_pct']:.2f}%\nEx: /threshold 0.10",
            reply_markup=options_keyboard([
                ("0.05%", "set:th:0.05"), ("0.10%", "set:th:0.10"),
                ("0.15%", "set:th:0.15"), ("0.20%", "set:th:0.20"),
                ("0.30%", "set:th:0.30"), ("0.50%", "set:th:0.50"),
            ]),
        )
        return
    try:
        v = float(parts[1])
        with _state_lock:
            STATE["threshold_pct"] = v
            save_state()
        await update.message.reply_text(f"✅ Threshold satt till {v:.2f}%", reply_markup=menu_keyboard())
    except Exception:
        await update.message.reply_text("Ogiltigt. Ex: /threshold 0.10", reply_markup=menu_keyboard())


async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.split()
    if len(parts) == 1:
        await update.message.reply_text(
            f"Stake per trade nu: {STATE['stake_usdt']:.2f} USDT\nEx: /stake 30",
            reply_markup=options_keyboard([
                ("10", "set:stake:10"), ("20", "set:stake:20"),
                ("30", "set:stake:30"), ("50", "set:stake:50"),
                ("100", "set:stake:100"), ("200", "set:stake:200"),
            ]),
        )
        return
    try:
        v = float(parts[1])
        with _state_lock:
            STATE["stake_usdt"] = v
            save_state()
        await update.message.reply_text(f"✅ Stake satt till {v:.2f} USDT", reply_markup=menu_keyboard())
    except Exception:
        await update.message.reply_text("Ogiltigt. Ex: /stake 30", reply_markup=menu_keyboard())


async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.split()
    if len(parts) == 1:
        await update.message.reply_text(
            f"TP nu: {STATE['tp_pct']:.2f}%\nEx: /tp 0.40",
            reply_markup=options_keyboard([
                ("0.25%", "set:tp:0.25"), ("0.35%", "set:tp:0.35"),
                ("0.40%", "set:tp:0.40"), ("0.50%", "set:tp:0.50"),
                ("0.70%", "set:tp:0.70"), ("1.00%", "set:tp:1.00"),
            ]),
        )
        return
    try:
        v = float(parts[1])
        with _state_lock:
            STATE["tp_pct"] = v
            save_state()
        await update.message.reply_text(f"✅ TP satt till {v:.2f}%", reply_markup=menu_keyboard())
    except Exception:
        await update.message.reply_text("Ogiltigt. Ex: /tp 0.40", reply_markup=menu_keyboard())


async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.split()
    if len(parts) == 1:
        await update.message.reply_text(
            f"SL nu: {STATE['sl_pct']:.2f}%\nEx: /sl 0.25",
            reply_markup=options_keyboard([
                ("0.15%", "set:sl:0.15"), ("0.20%", "set:sl:0.20"),
                ("0.25%", "set:sl:0.25"), ("0.30%", "set:sl:0.30"),
                ("0.40%", "set:sl:0.40"), ("0.60%", "set:sl:0.60"),
            ]),
        )
        return
    try:
        v = float(parts[1])
        with _state_lock:
            STATE["sl_pct"] = v
            save_state()
        await update.message.reply_text(f"✅ SL satt till {v:.2f}%", reply_markup=menu_keyboard())
    except Exception:
        await update.message.reply_text("Ogiltigt. Ex: /sl 0.25", reply_markup=menu_keyboard())


async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.split()
    if len(parts) == 1:
        await update.message.reply_text(
            f"Timeframe nu: {STATE['timeframe']}\nEx: /timeframe 1m",
            reply_markup=options_keyboard([
                ("1m", "set:tf:1m"), ("3m", "set:tf:3m"),
                ("5m", "set:tf:5m"), ("15m", "set:tf:15m"),
            ]),
        )
        return
    tf = parts[1].lower()
    if tf not in TF_MAP:
        await update.message.reply_text("Ogiltigt. Använd 1m/3m/5m/15m", reply_markup=menu_keyboard())
        return
    with _state_lock:
        STATE["timeframe"] = tf
        save_state()
    await update.message.reply_text(f"✅ Timeframe satt till {tf}", reply_markup=menu_keyboard())


async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.split(maxsplit=1)
    if len(parts) == 1:
        await update.message.reply_text(
            f"Coins nu ({len(STATE['coins'])}): {STATE['coins']}\nEx: /coins BTC-USDT,ETH-USDT,XRP-USDT",
            reply_markup=options_keyboard([
                ("Top5 default", "set:coins:BTC-USDT,ETH-USDT,XRP-USDT,ADA-USDT,LINK-USDT"),
                ("3 coins", "set:coins:BTC-USDT,ETH-USDT,XRP-USDT"),
                ("Alt mix", "set:coins:SOL-USDT,AVAX-USDT,ADA-USDT,LINK-USDT,XRP-USDT"),
                ("Meme-ish", "set:coins:DOGE-USDT,SHIB-USDT,PEPE-USDT"),
            ]),
        )
        return
    raw = parts[1]
    items = [x.strip().upper().replace("/", "-") for x in raw.split(",") if x.strip()]
    norm = []
    for it in items:
        if "-" not in it:
            it = it + "-USDT"
        norm.append(it)
    with _state_lock:
        STATE["coins"] = norm
        save_state()
    await update.message.reply_text(f"✅ Coins uppdaterade ({len(norm)}): {norm}", reply_markup=menu_keyboard())


async def text_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    txt = update.message.text.strip().upper()
    action = check_pending(chat_id)
    if not action:
        return
    # clear pending either way
    with _state_lock:
        STATE["pending_confirm"].pop(str(chat_id), None)
        save_state()

    if txt != "JA":
        await update.message.reply_text("Avbrutet.", reply_markup=menu_keyboard())
        return

    if action == "engine_on":
        with _state_lock:
            STATE["engine_on"] = True
            save_state()
        await update.message.reply_text("✅ ENGINE ON", reply_markup=menu_keyboard())
        return

    if action == "trade_mode_live":
        live_ok = bool(os.getenv("KUCOIN_KEY")) and bool(os.getenv("KUCOIN_SECRET")) and bool(os.getenv("KUCOIN_PASSPHRASE"))
        if not live_ok:
            await update.message.reply_text("❌ Live keys saknas. Sätt KUCOIN_KEY / KUCOIN_SECRET / KUCOIN_PASSPHRASE.", reply_markup=menu_keyboard())
            return
        with _state_lock:
            STATE["trade_mode"] = "live"
            save_state()
        await update.message.reply_text("⚠️ LIVE-läge aktivt.", reply_markup=menu_keyboard())
        return


# ----------------------------
# Callback handler (buttons)
# ----------------------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    await q.answer()
    data = q.data or ""

    if data == "menu:main":
        await q.edit_message_reply_markup(reply_markup=menu_keyboard())
        return

    if data.startswith("cmd:"):
        cmd = data.split(":", 1)[1]
        # trigger command effects via calling handlers
        if cmd == "status":
            await cmd_status(Update(update.update_id, message=q.message), context)
        elif cmd == "pnl":
            await cmd_pnl(Update(update.update_id, message=q.message), context)
        elif cmd == "engine_on":
            # need message-like update; easiest: send a prompt
            chat_id = q.message.chat_id
            set_pending("engine_on", chat_id)
            await q.message.reply_text("Skriv **JA** inom 30 sek för att starta motorn.", parse_mode="Markdown", reply_markup=menu_keyboard())
        elif cmd == "engine_off":
            with _state_lock:
                STATE["engine_on"] = False
                save_state()
            await q.message.reply_text("🛑 ENGINE OFF", reply_markup=menu_keyboard())
        elif cmd == "notify":
            await cmd_notify(Update(update.update_id, message=q.message), context)
        elif cmd == "export_csv":
            await q.message.reply_text("Använd /export_csv för att få filer.", reply_markup=menu_keyboard())
        elif cmd == "reset_pnl":
            await cmd_reset_pnl(Update(update.update_id, message=q.message), context)
        elif cmd == "close_all":
            await cmd_close_all(Update(update.update_id, message=q.message), context)
        return

    if data.startswith("menu:"):
        m = data.split(":", 1)[1]
        if m == "tf":
            kb = options_keyboard([("1m", "set:tf:1m"), ("3m", "set:tf:3m"), ("5m", "set:tf:5m"), ("15m", "set:tf:15m")])
            await q.edit_message_reply_markup(reply_markup=kb)
        elif m == "tp":
            kb = options_keyboard([("0.25%", "set:tp:0.25"), ("0.35%", "set:tp:0.35"), ("0.40%", "set:tp:0.40"),
                                  ("0.50%", "set:tp:0.50"), ("0.70%", "set:tp:0.70"), ("1.00%", "set:tp:1.00")])
            await q.edit_message_reply_markup(reply_markup=kb)
        elif m == "sl":
            kb = options_keyboard([("0.15%", "set:sl:0.15"), ("0.20%", "set:sl:0.20"), ("0.25%", "set:sl:0.25"),
                                  ("0.30%", "set:sl:0.30"), ("0.40%", "set:sl:0.40"), ("0.60%", "set:sl:0.60")])
            await q.edit_message_reply_markup(reply_markup=kb)
        elif m == "th":
            kb = options_keyboard([("0.05%", "set:th:0.05"), ("0.10%", "set:th:0.10"), ("0.15%", "set:th:0.15"),
                                  ("0.20%", "set:th:0.20"), ("0.30%", "set:th:0.30"), ("0.50%", "set:th:0.50")])
            await q.edit_message_reply_markup(reply_markup=kb)
        elif m == "stake":
            kb = options_keyboard([("10", "set:stake:10"), ("20", "set:stake:20"), ("30", "set:stake:30"),
                                  ("50", "set:stake:50"), ("100", "set:stake:100"), ("200", "set:stake:200")])
            await q.edit_message_reply_markup(reply_markup=kb)
        elif m == "coins":
            kb = options_keyboard([
                ("Top5", "set:coins:BTC-USDT,ETH-USDT,XRP-USDT,ADA-USDT,LINK-USDT"),
                ("3 coins", "set:coins:BTC-USDT,ETH-USDT,XRP-USDT"),
                ("Alt mix", "set:coins:SOL-USDT,AVAX-USDT,ADA-USDT,LINK-USDT,XRP-USDT"),
                ("Meme-ish", "set:coins:DOGE-USDT,SHIB-USDT,PEPE-USDT"),
            ])
            await q.edit_message_reply_markup(reply_markup=kb)
        elif m == "mode":
            kb = options_keyboard([("mock", "set:mode:mock"), ("live", "set:mode:live")])
            await q.edit_message_reply_markup(reply_markup=kb)
        return

    if data.startswith("set:"):
        _, field, val = data.split(":", 2)

        if field == "tf":
            if val in TF_MAP:
                with _state_lock:
                    STATE["timeframe"] = val
                    save_state()
                await q.message.reply_text(f"✅ TF satt till {val}", reply_markup=menu_keyboard())
            return

        if field in ("tp", "sl", "th"):
            v = float(val)
            with _state_lock:
                if field == "tp":
                    STATE["tp_pct"] = v
                elif field == "sl":
                    STATE["sl_pct"] = v
                else:
                    STATE["threshold_pct"] = v
                save_state()
            await q.message.reply_text(f"✅ {field.upper()} satt till {v:.2f}%", reply_markup=menu_keyboard())
            return

        if field == "stake":
            v = float(val)
            with _state_lock:
                STATE["stake_usdt"] = v
                save_state()
            await q.message.reply_text(f"✅ Stake satt till {v:.2f} USDT", reply_markup=menu_keyboard())
            return

        if field == "coins":
            items = [x.strip().upper() for x in val.split(",") if x.strip()]
            with _state_lock:
                STATE["coins"] = items
                save_state()
            await q.message.reply_text(f"✅ Coins uppdaterade ({len(items)}): {items}", reply_markup=menu_keyboard())
            return

        if field == "mode":
            if val == "mock":
                with _state_lock:
                    STATE["trade_mode"] = "mock"
                    save_state()
                await q.message.reply_text("✅ Trade mode: mock", reply_markup=menu_keyboard())
                return
            if val == "live":
                set_pending("trade_mode_live", q.message.chat_id)
                await q.message.reply_text("Skriv **JA** inom 30 sek för LIVE-läge.", parse_mode="Markdown", reply_markup=menu_keyboard())
                return


# ----------------------------
# Tiny HTTP server for DO healthcheck
# ----------------------------
def start_http_server() -> None:
    try:
        import http.server
        import socketserver

        port = int(os.getenv("PORT", "8080"))
        handler = http.server.SimpleHTTPRequestHandler

        class QuietHandler(handler):
            def log_message(self, format, *args):
                return

        def run():
            with socketserver.TCPServer(("", port), QuietHandler) as httpd:
                httpd.serve_forever()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        log.info("HTTP server started on port %s", port)
    except Exception:
        log.exception("Failed to start HTTP server (non-fatal)")


# ----------------------------
# Main
# ----------------------------
def build_app(token: str) -> Application:
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))

    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("trade_mode", cmd_trade_mode))
    app.add_handler(CommandHandler("notify", cmd_notify))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))

    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("sl", cmd_sl))
    app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    app.add_handler(CommandHandler("coins", cmd_coins))

    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_confirm))

    # Engine loop every 3 seconds
    app.job_queue.run_repeating(engine_tick, interval=3, first=3)

    return app


def main() -> None:
    global STATE
    STATE = load_state()

    token = (os.getenv("TELEGRAM_TOKEN", "").strip() or os.getenv("TELEGRAM_BOT_TOKEN", "").strip())
    if not token:
        raise RuntimeError("Sätt TELEGRAM_TOKEN i environment (eller TELEGRAM_BOT_TOKEN).")

    ensure_csv_headers(MOCK_LOG_FILE)
    ensure_csv_headers(REAL_LOG_FILE)

    start_http_server()

    app = build_app(token)

    log.info("Starting bot polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
