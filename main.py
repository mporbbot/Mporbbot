# =========================
# MpORBbot — Liquidity Sweep Reversal (LONG only) — KuCoin (public) + Telegram
# Single file main.py (PART 1/2)
# =========================

import os
import csv
import time
import json
import math
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mporbbot_sweep_long")

# =========================
# KuCoin
# =========================
KUCOIN_BASE = "https://api.kucoin.com"

# =========================
# Strategy meta
# =========================
STRATEGY_NAME = "Liquidity Sweep Reversal (LONG only)"
STRATEGY_CODE = "SWEEP_REV_LONG"
EXCHANGE_NAME = "KuCoin"

# =========================
# Defaults
# =========================
DEFAULT_COINS = [
    "BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT",
    "SOL-USDT", "BNB-USDT", "AVAX-USDT", "DOGE-USDT", "LTC-USDT",
]

TF_TREND = "15min"
TF_ENTRY = "5min"

TREND_CANDLES = 220
ENTRY_CANDLES = 220

ENGINE_LOOP_SEC = 4
COOLDOWN_PER_COIN_SEC = 45
MAX_POSITIONS = 4

DEFAULT_FEE_RATE_PER_SIDE = 0.0010      # 0.10% per side
DEFAULT_SLIPPAGE_RATE_PER_SIDE = 0.0002 # 0.02% per side

DEFAULT_STAKE_USDT = 30.0
DEFAULT_RISK_MULT = 1.0  # scales position size (mock only here, but kept)

# Sweep / structure settings
DEFAULT_PIVOT_LEFT = 3
DEFAULT_PIVOT_RIGHT = 3
DEFAULT_PIVOT_MAX_AGE_BARS = 60

DEFAULT_SWEEP_PCT = 0.06      # sweep below pivot low by >= 0.06%
DEFAULT_RECLAIM_PCT = 0.03    # close back above pivot low by >= 0.03%

# Confirmation
DEFAULT_TREND_FILTER = True
DEFAULT_ATR_CONFIRM = True
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_EXPAND_MULT = 1.10  # ATR_now >= ATR_sma * this
DEFAULT_MIN_BODY_PCT = 0.05     # candle body >= 0.05% of close

# Risk / exits
DEFAULT_SL_ATR_MULT = 2.2
DEFAULT_TP1_R_MULT = 1.2          # take partial at 1.2R
DEFAULT_TP1_CLOSE_PCT = 0.50      # close 50% at TP1
DEFAULT_TRAIL_ON_PCT = 1.20       # activate trailing after +1.20%
DEFAULT_TRAIL_DIST_PCT = 0.60     # trailing distance 0.60%

DEFAULT_MAX_HOLD_MIN = 24 * 60    # 24h
DEFAULT_STALL_MIN = 180           # if after 180 min not green, bail
DEFAULT_STALL_MIN_PROFIT_PCT = 0.05  # consider "not green" if < +0.05%

# Files
MOCK_LOG_PATH = "mock_trade_log.csv"
REAL_LOG_PATH = "real_trade_log.csv"
STATE_PATH = "bot_state.json"

# =========================
# Helpers
# =========================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def ts_unix() -> int:
    return int(time.time())

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def pct_change(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b - a) / a * 100.0

# =========================
# Indicators
# =========================
def ema(values: List[float], period: int) -> Optional[float]:
    if period <= 1 or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def true_range(h: float, l: float, prev_c: float) -> float:
    return max(h - l, abs(h - prev_c), abs(l - prev_c))

def atr(candles: List[List[str]], period: int = 14) -> Optional[float]:
    # candles: [time, open, close, high, low, volume, turnover] strings
    if len(candles) < period + 2:
        return None
    trs: List[float] = []
    for i in range(-period, 0):
        c = candles[i]
        prev = candles[i - 1]
        h = safe_float(c[3])
        l = safe_float(c[4])
        prev_close = safe_float(prev[2])
        trs.append(true_range(h, l, prev_close))
    return sum(trs) / len(trs) if trs else None

def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

# =========================
# Pivot detection
# =========================
def find_last_pivot_low(
    lows: List[float],
    left: int,
    right: int,
    max_age_bars: int
) -> Optional[Tuple[int, float]]:
    """
    Returns (index, pivot_low) where index refers to lows[] index.
    Finds the most recent pivot low in the last max_age_bars bars,
    excluding the last 'right' bars (needs future bars to confirm).
    """
    n = len(lows)
    if n < left + right + 5:
        return None
    end = n - right - 1
    start = max(left, n - max_age_bars)
    for i in range(end, start - 1, -1):
        pivot = lows[i]
        ok = True
        for j in range(i - left, i):
            if lows[j] <= pivot:
                ok = False
                break
        if not ok:
            continue
        for j in range(i + 1, i + right + 1):
            if lows[j] <= pivot:
                ok = False
                break
        if ok:
            return (i, pivot)
    return None

# =========================
# KuCoin public client
# =========================
class KuCoinPublic:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MpORBbot/1.0"})

    def get_level1(self, symbol: str) -> Dict:
        url = f"{KUCOIN_BASE}/api/v1/market/orderbook/level1"
        r = self.session.get(url, params={"symbol": symbol}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        return data["data"]

    def get_candles(self, symbol: str, ktype: str, limit: int) -> List[List[str]]:
        sec_map = {
            "1min": 60, "3min": 180, "5min": 300, "15min": 900,
            "30min": 1800, "1hour": 3600,
        }
        sec = sec_map.get(ktype, 60)
        end_at = ts_unix()
        start_at = end_at - sec * (limit + 10)

        url = f"{KUCOIN_BASE}/api/v1/market/candles"
        r = self.session.get(
            url,
            params={"symbol": symbol, "type": ktype, "startAt": start_at, "endAt": end_at},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        arr = data["data"] or []
        arr = list(reversed(arr))  # ASC
        if len(arr) > limit:
            arr = arr[-limit:]
        return arr

# =========================
# Models
# =========================
@dataclass
class BotSettings:
    coins: List[str]
    trade_mode: str          # mock | live (live placeholder)
    engine_on: bool
    stake_usdt: float
    risk_mult: float

    fee_rate_per_side: float
    slippage_rate_per_side: float

    max_positions: int
    cooldown_per_coin_sec: int
    notify: bool

    # Sweep params
    pivot_left: int
    pivot_right: int
    pivot_max_age_bars: int
    sweep_pct: float
    reclaim_pct: float

    # Confirm
    trend_filter: bool
    atr_confirm: bool
    atr_period: int
    atr_expand_mult: float
    min_body_pct: float

    # Exits
    sl_atr_mult: float
    tp1_r_mult: float
    tp1_close_pct: float
    trail_on_pct: float
    trail_dist_pct: float
    max_hold_min: int
    stall_min: int
    stall_min_profit_pct: float

@dataclass
class Position:
    symbol: str
    side: str  # LONG only
    qty: float
    stake_usdt: float
    entry_price: float
    entry_ts: int

    pivot_low: float
    atr_at_entry: float

    sl_price: float

    tp1_price: float
    tp1_done: bool

    trail_active: bool
    trail_highest: float
    trail_stop: float

@dataclass
class PnL:
    trades: int
    net_usdt: float

@dataclass
class CoinState:
    cooldown_until: int

# =========================
# CSV logging
# =========================
CSV_HEADERS = [
    "timestamp_unix",
    "timestamp_utc",
    "exchange",
    "mode",
    "strategy",
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

def ensure_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

def log_trade_csv(
    path: str,
    mode: str,
    symbol: str,
    side: str,
    qty: float,
    stake_usdt: float,
    entry_price: float,
    exit_price: float,
    gross_pnl: float,
    fees: float,
    slippage: float,
    net_pnl: float,
    reason: str,
):
    ensure_csv(path)
    row = [
        ts_unix(),
        utc_now().isoformat(),
        EXCHANGE_NAME,
        mode,
        STRATEGY_CODE,
        symbol,
        side,
        f"{qty:.12f}",
        f"{stake_usdt:.2f}",
        f"{entry_price:.12f}",
        f"{exit_price:.12f}",
        f"{gross_pnl:.6f}",
        f"{fees:.6f}",
        f"{slippage:.6f}",
        f"{net_pnl:.6f}",
        reason,
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# =========================
# State persistence
# =========================
def load_state() -> Dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: Dict):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Failed to save state: {e}")

# =========================
# Engine: Liquidity Sweep Reversal (LONG only)
# =========================
class SweepLongEngine:
    def __init__(self, kucoin: KuCoinPublic):
        self.kucoin = kucoin

        self.settings = BotSettings(
            coins=DEFAULT_COINS.copy(),
            trade_mode="mock",
            engine_on=False,
            stake_usdt=DEFAULT_STAKE_USDT,
            risk_mult=DEFAULT_RISK_MULT,

            fee_rate_per_side=DEFAULT_FEE_RATE_PER_SIDE,
            slippage_rate_per_side=DEFAULT_SLIPPAGE_RATE_PER_SIDE,

            max_positions=MAX_POSITIONS,
            cooldown_per_coin_sec=COOLDOWN_PER_COIN_SEC,
            notify=True,

            pivot_left=DEFAULT_PIVOT_LEFT,
            pivot_right=DEFAULT_PIVOT_RIGHT,
            pivot_max_age_bars=DEFAULT_PIVOT_MAX_AGE_BARS,
            sweep_pct=DEFAULT_SWEEP_PCT,
            reclaim_pct=DEFAULT_RECLAIM_PCT,

            trend_filter=DEFAULT_TREND_FILTER,
            atr_confirm=DEFAULT_ATR_CONFIRM,
            atr_period=DEFAULT_ATR_PERIOD,
            atr_expand_mult=DEFAULT_ATR_EXPAND_MULT,
            min_body_pct=DEFAULT_MIN_BODY_PCT,

            sl_atr_mult=DEFAULT_SL_ATR_MULT,
            tp1_r_mult=DEFAULT_TP1_R_MULT,
            tp1_close_pct=DEFAULT_TP1_CLOSE_PCT,
            trail_on_pct=DEFAULT_TRAIL_ON_PCT,
            trail_dist_pct=DEFAULT_TRAIL_DIST_PCT,
            max_hold_min=DEFAULT_MAX_HOLD_MIN,
            stall_min=DEFAULT_STALL_MIN,
            stall_min_profit_pct=DEFAULT_STALL_MIN_PROFIT_PCT,
        )

        self.positions: Dict[str, Position] = {}
        self.coin_state: Dict[str, CoinState] = {c: CoinState(cooldown_until=0) for c in self.settings.coins}
        self.pnl = PnL(trades=0, net_usdt=0.0)

        self._load_persisted()

    def _load_persisted(self):
        st = load_state()
        if not st:
            return
        try:
            s = st.get("settings", {})
            for key in asdict(self.settings).keys():
                if key in s:
                    setattr(self.settings, key, s[key])

            p = st.get("pnl", {})
            if "trades" in p and "net_usdt" in p:
                self.pnl = PnL(trades=int(p["trades"]), net_usdt=float(p["net_usdt"]))

            for c in self.settings.coins:
                if c not in self.coin_state:
                    self.coin_state[c] = CoinState(cooldown_until=0)

        except Exception as e:
            log.warning(f"Failed to load persisted state: {e}")

    def persist(self):
        save_state({"settings": asdict(self.settings), "pnl": asdict(self.pnl)})

    def set_coins(self, coins: List[str]):
        coins = [c.strip().upper() for c in coins if c.strip()]
        self.settings.coins = coins
        for c in coins:
            if c not in self.coin_state:
                self.coin_state[c] = CoinState(cooldown_until=0)
        for sym in list(self.positions.keys()):
            if sym not in coins:
                del self.positions[sym]
        self.persist()

    def _calc_qty(self, stake_usdt: float, price: float) -> float:
        return (stake_usdt / price) if price > 0 else 0.0

    def _calc_costs_mock(self, notional_usdt: float) -> Tuple[float, float]:
        fees = notional_usdt * self.settings.fee_rate_per_side
        slip = notional_usdt * self.settings.slippage_rate_per_side
        return fees, slip

    def _apply_slippage_price(self, price: float, side: str) -> float:
        slip = self.settings.slippage_rate_per_side
        if side == "BUY":
            return price * (1 + slip)
        return price * (1 - slip)

    def _level1_prices(self, symbol: str) -> Tuple[float, float, float]:
        d = self.kucoin.get_level1(symbol)
        last = safe_float(d.get("price"))
        bid = safe_float(d.get("bestBid"))
        ask = safe_float(d.get("bestAsk"))
        if bid <= 0:
            bid = last
        if ask <= 0:
            ask = last
        return last, bid, ask

    def open_positions_count(self) -> int:
        return len(self.positions)

    def can_trade_coin(self, symbol: str) -> bool:
        cs = self.coin_state.get(symbol)
        if not cs:
            return True
        return ts_unix() >= cs.cooldown_until

    def mark_cooldown(self, symbol: str):
        cs = self.coin_state.setdefault(symbol, CoinState(cooldown_until=0))
        cs.cooldown_until = ts_unix() + int(self.settings.cooldown_per_coin_sec)

    def _trend_ok(self, closes_15m: List[float]) -> bool:
        if not self.settings.trend_filter:
            return True
        e20 = ema(closes_15m, 20)
        e50 = ema(closes_15m, 50)
        if e20 is None or e50 is None:
            return False
        return e20 > e50

    def _atr_ok(self, candles_5m: List[List[str]]) -> bool:
        if not self.settings.atr_confirm:
            return True
        a = atr(candles_5m, self.settings.atr_period)
        if a is None:
            return False
        # compare ATR to its SMA over same period (simple proxy)
        atr_series: List[float] = []
        for i in range(self.settings.atr_period + 5, len(candles_5m)):
            sub = candles_5m[:i]
            v = atr(sub, self.settings.atr_period)
            if v is not None:
                atr_series.append(v)
        if len(atr_series) < self.settings.atr_period:
            return True
        a_sma = sma(atr_series, self.settings.atr_period)
        if a_sma is None or a_sma <= 0:
            return True
        return a >= a_sma * float(self.settings.atr_expand_mult)

    def _min_body_ok(self, o: float, c: float) -> bool:
        if c <= 0:
            return False
        body = abs(c - o)
        body_pct = (body / c) * 100.0
        return body_pct >= float(self.settings.min_body_pct)

    def _sweep_signal_long(
        self,
        candles_5m: List[List[str]],
        candles_15m: List[List[str]],
    ) -> Optional[Tuple[float, float]]:
        """
        Returns (pivot_low, atr_now) if LONG signal.
        Logic:
          - Find recent pivot low
          - Latest candle sweeps below pivot_low by sweep_pct
          - Latest candle closes back above pivot_low by reclaim_pct
          - Bullish candle with minimum body size
          - Optional: trend filter (15m EMA20 > EMA50)
          - Optional: ATR expansion confirm
        """
        if len(candles_5m) < 50 or len(candles_15m) < 80:
            return None

        closes_15m = [safe_float(x[2]) for x in candles_15m]
        if not self._trend_ok(closes_15m):
            return None

        if not self._atr_ok(candles_5m):
            return None

        opens = [safe_float(x[1]) for x in candles_5m]
        closes = [safe_float(x[2]) for x in candles_5m]
        highs = [safe_float(x[3]) for x in candles_5m]
        lows = [safe_float(x[4]) for x in candles_5m]

        pv = find_last_pivot_low(
            lows,
            int(self.settings.pivot_left),
            int(self.settings.pivot_right),
            int(self.settings.pivot_max_age_bars),
        )
        if not pv:
            return None
        _, pivot_low = pv

        o = opens[-1]
        c = closes[-1]
        l = lows[-1]

        if c <= 0 or pivot_low <= 0:
            return None

        if not self._min_body_ok(o, c):
            return None

        bullish = c > o and c > closes[-2]

        sweep_level = pivot_low * (1 - self.settings.sweep_pct / 100.0)
        reclaim_level = pivot_low * (1 + self.settings.reclaim_pct / 100.0)

        swept = l <= sweep_level
        reclaimed = c >= reclaim_level

        if not (swept and reclaimed and bullish):
            return None

        a = atr(candles_5m, self.settings.atr_period)
        if a is None or a <= 0:
            return None

        return (pivot_low, a)

    def try_open_long(self, symbol: str) -> Optional[str]:
        if symbol in self.positions:
            return None
        if self.open_positions_count() >= self.settings.max_positions:
            return None
        if not self.can_trade_coin(symbol):
            return None

        candles_15m = self.kucoin.get_candles(symbol, TF_TREND, TREND_CANDLES)
        candles_5m = self.kucoin.get_candles(symbol, TF_ENTRY, ENTRY_CANDLES)

        sig = self._sweep_signal_long(candles_5m, candles_15m)
        if not sig:
            return None
        pivot_low, a = sig

        last, bid, ask = self._level1_prices(symbol)
        entry_raw = ask if ask > 0 else last
        entry_price = entry_raw

        if self.settings.trade_mode == "mock":
            entry_price = self._apply_slippage_price(entry_price, "BUY")

        stake = float(self.settings.stake_usdt) * float(self.settings.risk_mult)
        qty = self._calc_qty(stake, entry_price)
        if qty <= 0:
            return None

        # SL: min(pivot_low - small buffer, entry - ATR*mult)
        pivot_sl = pivot_low * (1 - 0.0010)  # 0.10% under pivot low
        atr_sl = entry_price - a * float(self.settings.sl_atr_mult)
        sl_price = min(pivot_sl, atr_sl)

        risk_per_unit = max(0.0, entry_price - sl_price)
        if risk_per_unit <= 0:
            return None

        tp1_price = entry_price + risk_per_unit * float(self.settings.tp1_r_mult)

        pos = Position(
            symbol=symbol,
            side="LONG",
            qty=qty,
            stake_usdt=stake,
            entry_price=entry_price,
            entry_ts=ts_unix(),
            pivot_low=pivot_low,
            atr_at_entry=a,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp1_done=False,
            trail_active=False,
            trail_highest=entry_price,
            trail_stop=entry_price * (1 - self.settings.trail_dist_pct / 100.0),
        )
        self.positions[symbol] = pos
        self.mark_cooldown(symbol)
        self.persist()

        return (
            f"🟢 SWEEP LONG {symbol} @ {entry_price:.6f}\n"
            f"pivot_low={pivot_low:.6f} | ATR={a:.6f}\n"
            f"SL={sl_price:.6f} | TP1={tp1_price:.6f}\n"
            f"trail_on={fmt_pct(self.settings.trail_on_pct)} dist={fmt_pct(self.settings.trail_dist_pct)}"
        )

    def _close_position(self, symbol: str, reason: str, exit_qty: Optional[float] = None) -> Optional[Tuple[str, float]]:
        """
        Close full position if exit_qty is None.
        Return (message, net_pnl_delta).
        """
        pos = self.positions.get(symbol)
        if not pos:
            return None

        last, bid, ask = self._level1_prices(symbol)
        exit_raw = bid if bid > 0 else last
        exit_price = exit_raw

        if self.settings.trade_mode == "mock":
            exit_price = self._apply_slippage_price(exit_price, "SELL")

        qty_to_close = pos.qty if exit_qty is None else float(exit_qty)
        qty_to_close = clamp(qty_to_close, 0.0, pos.qty)
        if qty_to_close <= 0:
            return None

        entry_notional = qty_to_close * pos.entry_price
        exit_notional = qty_to_close * exit_price
        gross_pnl = exit_notional - entry_notional

        fees = 0.0
        slip_cost = 0.0
        if self.settings.trade_mode == "mock":
            fee_in, slip_in = self._calc_costs_mock(entry_notional)
            fee_out, slip_out = self._calc_costs_mock(exit_notional)
            fees = fee_in + fee_out
            slip_cost = slip_in + slip_out

        net_pnl = gross_pnl - fees - slip_cost

        # logging only when fully closed
        fully_closed = (abs(qty_to_close - pos.qty) < 1e-12)

        if fully_closed:
            self.pnl.trades += 1
            self.pnl.net_usdt += net_pnl

            path = MOCK_LOG_PATH if self.settings.trade_mode == "mock" else REAL_LOG_PATH
            log_trade_csv(
                path,
                self.settings.trade_mode,
                symbol,
                "LONG",
                qty_to_close,
                pos.stake_usdt,
                pos.entry_price,
                exit_price,
                gross_pnl,
                fees,
                slip_cost,
                net_pnl,
                reason,
            )
            del self.positions[symbol]
            self.mark_cooldown(symbol)
        else:
            # partial: reduce qty, keep position alive, still count pnl in total net
            pos.qty -= qty_to_close
            self.pnl.net_usdt += net_pnl
            self.mark_cooldown(symbol)

        self.persist()

        emoji = "🎯" if net_pnl >= 0 else "🟥"
        msg = (
            f"{emoji} EXIT {symbol} @ {exit_price:.6f}\n"
            f"net {net_pnl:+.4f} USDT | reason={reason}\n"
            f"(gross {gross_pnl:+.4f} | fees {fees:.4f} | slip {slip_cost:.4f})"
        )
        return (msg, net_pnl)

    def update_positions(self) -> List[str]:
        msgs: List[str] = []
        now = ts_unix()

        for sym, pos in list(self.positions.items()):
            last, bid, ask = self._level1_prices(sym)
            price = last

            # Hard SL
            if price <= pos.sl_price:
                r = self._close_position(sym, "SL")
                if r:
                    msgs.append(r[0])
                continue

            # Max hold
            hold_min = (now - pos.entry_ts) / 60.0
            if hold_min >= float(self.settings.max_hold_min):
                r = self._close_position(sym, "MAX_HOLD")
                if r:
                    msgs.append(r[0])
                continue

            # Stall bailout (if not green enough after stall_min)
            if hold_min >= float(self.settings.stall_min):
                profit_pct = pct_change(pos.entry_price, price)
                if profit_pct < float(self.settings.stall_min_profit_pct):
                    r = self._close_position(sym, "STALL_BAILOUT")
                    if r:
                        msgs.append(r[0])
                    continue

            # TP1 partial
            if (not pos.tp1_done) and price >= pos.tp1_price:
                close_qty = pos.qty * float(self.settings.tp1_close_pct)
                r = self._close_position(sym, "TP1_PARTIAL", exit_qty=close_qty)
                if r:
                    msgs.append(r[0])
                    pos.tp1_done = True
                continue

            # Arm trailing after +trail_on_pct
            if (not pos.trail_active) and pct_change(pos.entry_price, price) >= float(self.settings.trail_on_pct):
                pos.trail_active = True
                pos.trail_highest = price
                pos.trail_stop = price * (1 - self.settings.trail_dist_pct / 100.0)
                msgs.append(f"🟡 TRAIL ARMED {sym} | price={price:.6f} | trail_stop={pos.trail_stop:.6f}")
                self.persist()
                continue

            # Trailing management
            if pos.trail_active:
                if price > pos.trail_highest:
                    pos.trail_highest = price
                    pos.trail_stop = pos.trail_highest * (1 - self.settings.trail_dist_pct / 100.0)

                if price <= pos.trail_stop:
                    r = self._close_position(sym, "TRAIL_STOP")
                    if r:
                        msgs.append(r[0])
                    continue

        return msgs

    def step(self) -> List[str]:
        if not self.settings.engine_on:
            return []
        out: List[str] = []

        # exits first
        out.extend(self.update_positions())

        # entries
        for sym in self.settings.coins:
            if sym in self.positions:
                continue
            if self.open_positions_count() >= self.settings.max_positions:
                break
            try:
                m = self.try_open_long(sym)
                if m:
                    out.append(m)
            except Exception as e:
                log.warning(f"Entry check error {sym}: {e}")

        return out

    def status_text(self) -> str:
        s = self.settings
        open_pos = list(self.positions.keys())
        return (
            f"<b>ENGINE:</b> {'ON ✅' if s.engine_on else 'OFF ⛔️'}\n"
            f"<b>Strategy:</b> {STRATEGY_NAME} ({STRATEGY_CODE})\n"
            f"<b>Trade mode:</b> {s.trade_mode} (KuCoin keys: {'OK' if os.getenv('KUCOIN_API_KEY') else 'missing'})\n"
            f"<b>Trend TF:</b> {TF_TREND} | <b>Entry TF:</b> {TF_ENTRY}\n"
            f"<b>Trend filter:</b> {'ON' if s.trend_filter else 'OFF'} | <b>ATR confirm:</b> {'ON' if s.atr_confirm else 'OFF'}\n\n"
            f"<b>Stake:</b> {s.stake_usdt:.2f} USDT | <b>Risk mult:</b> {s.risk_mult:.2f}\n"
            f"<b>Max pos:</b> {s.max_positions} | <b>Cooldown:</b> {s.cooldown_per_coin_sec}s\n"
            f"<b>Fees:</b> {s.fee_rate_per_side*100:.2f}%/side | <b>Slip:</b> {s.slippage_rate_per_side*100:.2f}%/side\n\n"
            f"<b>Pivot:</b> L={s.pivot_left} R={s.pivot_right} age≤{s.pivot_max_age_bars} bars\n"
            f"<b>Sweep:</b> {fmt_pct(s.sweep_pct)} below | <b>Reclaim:</b> {fmt_pct(s.reclaim_pct)} above\n"
            f"<b>Min body:</b> {fmt_pct(s.min_body_pct)} | <b>ATR expand:</b> x{s.atr_expand_mult:.2f}\n\n"
            f"<b>SL:</b> ATR x{s.sl_atr_mult:.2f} | <b>TP1:</b> {s.tp1_r_mult:.2f}R (close {int(s.tp1_close_pct*100)}%)\n"
            f"<b>Trail:</b> on {fmt_pct(s.trail_on_pct)} | dist {fmt_pct(s.trail_dist_pct)}\n"
            f"<b>Hold:</b> max {s.max_hold_min}m | stall {s.stall_min}m (<{fmt_pct(s.stall_min_profit_pct)})\n\n"
            f"<b>Trades:</b> {self.pnl.trades}\n"
            f"<b>Total NET PnL:</b> {self.pnl.net_usdt:+.4f} USDT\n"
            f"<b>Coins ({len(s.coins)}):</b> {s.coins}\n"
            f"<b>Open positions:</b> {open_pos if open_pos else 'none'}\n"
            f"<b>Notify:</b> {'ON' if s.notify else 'OFF'}\n"
        )

# =========================
# Telegram UI
# =========================
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["/status", "/pnl"],
        ["/engine_on", "/engine_off"],
        ["/coins", "/mode"],
        ["/stake", "/risk"],
        ["/trend", "/atr_confirm"],
        ["/pivot", "/sweep"],
        ["/reclaim", "/trail"],
        ["/sl_atr", "/tp1"],
        ["/cooldown", "/maxpos"],
        ["/notify", "/export_csv"],
        ["/close_all", "/reset_pnl"],
    ],
    resize_keyboard=True,
)

ENGINE: Optional[SweepLongEngine] = None

def _parse_one_float(args: List[str]) -> Optional[float]:
    if not args:
        return None
    try:
        return float(str(args[0]).replace("%", "").strip())
    except Exception:
        return None

def _parse_two_ints(args: List[str]) -> Optional[Tuple[int, int]]:
    if len(args) < 2:
        return None
    try:
        return int(float(args[0])), int(float(args[1]))
    except Exception:
        return None

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        "MpORBbot\n\n"
        "Strategi: Liquidity Sweep Reversal (LONG only)\n"
        "Entry: sweep under pivot-low + reclaim above pivot-low (med trend/ATR-filter)\n"
        "Exit: SL (ATR/pivot), TP1 partial, trailing, stall-bailout, max-hold\n\n"
        "Använd knapparna nedan och skriv valfria värden, t.ex:\n"
        "/sweep 0.08\n"
        "/reclaim 0.04\n"
        "/trail 1.20 0.60\n"
        "/sl_atr 2.2\n"
        "/tp1 1.2 50\n",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id
    await update.message.reply_text(ENGINE.status_text(), parse_mode=ParseMode.HTML, reply_markup=MAIN_KEYBOARD)

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Total NET PnL: {ENGINE.pnl.net_usdt:+.4f} USDT\nTrades: {ENGINE.pnl.trades}",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.engine_on = True
    ENGINE.persist()
    await update.message.reply_text("ENGINE ON", reply_markup=MAIN_KEYBOARD)

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.engine_on = False
    ENGINE.persist()
    await update.message.reply_text("ENGINE OFF", reply_markup=MAIN_KEYBOARD)

async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.notify = not ENGINE.settings.notify
    ENGINE.persist()
    await update.message.reply_text(f"Notify: {'ON' if ENGINE.settings.notify else 'OFF'}", reply_markup=MAIN_KEYBOARD)

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.pnl = PnL(trades=0, net_usdt=0.0)
    ENGINE.persist()
    await update.message.reply_text("PnL reset", reply_markup=MAIN_KEYBOARD)

async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msgs = []
    for sym in list(ENGINE.positions.keys()):
        try:
            r = ENGINE._close_position(sym, "MANUAL_CLOSE_ALL")
            if r:
                msgs.append(r[0])
        except Exception as e:
            msgs.append(f"Close error {sym}: {e}")
    if not msgs:
        msgs = ["Inga öppna positioner."]
    for m in msgs:
        await update.message.reply_text(m, reply_markup=MAIN_KEYBOARD)

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    path = MOCK_LOG_PATH if ENGINE.settings.trade_mode == "mock" else REAL_LOG_PATH
    ensure_csv(path)
    try:
        await update.message.reply_document(
            document=open(path, "rb"),
            filename=os.path.basename(path),
            caption="Här är trade-loggen (CSV).",
        )
    except Exception as e:
        await update.message.reply_text(f"Kunde inte skicka CSV: {e}", reply_markup=MAIN_KEYBOARD)

async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(f"Mode nu: {ENGINE.settings.trade_mode}\nSkriv: /mode mock", reply_markup=MAIN_KEYBOARD)
        return
    m = str(args[0]).lower().strip()
    if m not in ("mock", "live"):
        await update.message.reply_text("Ogiltigt mode. Använd: mock eller live", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.trade_mode = m
    ENGINE.persist()
    await update.message.reply_text(f"Mode satt till: {ENGINE.settings.trade_mode}", reply_markup=MAIN_KEYBOARD)

async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "Skicka:\n/coins BTC-USDT ETH-USDT SOL-USDT\n\n"
            f"Nu: {ENGINE.settings.coins}",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.set_coins(args)
    await update.message.reply_text(f"Coins uppdaterade ({len(ENGINE.settings.coins)}): {ENGINE.settings.coins}", reply_markup=MAIN_KEYBOARD)

async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _parse_one_float(context.args)
    if v is None:
        await update.message.reply_text(f"Stake nu: {ENGINE.settings.stake_usdt:.2f}\nSkriv: /stake 30", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.stake_usdt = clamp(v, 1.0, 100000.0)
    ENGINE.persist()
    await update.message.reply_text(f"Stake satt till: {ENGINE.settings.stake_usdt:.2f} USDT", reply_markup=MAIN_KEYBOARD)

async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _parse_one_float(context.args)
    if v is None:
        await update.message.reply_text(f"Risk mult nu: {ENGINE.settings.risk_mult:.2f}\nSkriv: /risk 1.0", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.risk_mult = clamp(v, 0.10, 10.0)
    ENGINE.persist()
    await update.message.reply_text(f"Risk mult satt till: {ENGINE.settings.risk_mult:.2f}", reply_markup=MAIN_KEYBOARD)

async def cmd_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(f"Trendfilter: {'ON' if ENGINE.settings.trend_filter else 'OFF'}\nSkriv: /trend on eller /trend off", reply_markup=MAIN_KEYBOARD)
        return
    v = str(args[0]).lower().strip()
    ENGINE.settings.trend_filter = (v in ("on", "1", "true", "yes"))
    ENGINE.persist()
    await update.message.reply_text(f"Trendfilter: {'ON' if ENGINE.settings.trend_filter else 'OFF'}", reply_markup=MAIN_KEYBOARD)

async def cmd_atr_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(f"ATR confirm: {'ON' if ENGINE.settings.atr_confirm else 'OFF'}\nSkriv: /atr_confirm on/off", reply_markup=MAIN_KEYBOARD)
        return
    v = str(args[0]).lower().strip()
    ENGINE.settings.atr_confirm = (v in ("on", "1", "true", "yes"))
    ENGINE.persist()
    await update.message.reply_text(f"ATR confirm: {'ON' if ENGINE.settings.atr_confirm else 'OFF'}", reply_markup=MAIN_KEYBOARD)

async def cmd_pivot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            f"Pivot nu: L={ENGINE.settings.pivot_left} R={ENGINE.settings.pivot_right} age≤{ENGINE.settings.pivot_max_age_bars}\n"
            "Skriv:\n/pivot 3 3\n/pivot_age 60",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    pr = _parse_two_ints(context.args)
    if not pr:
        await update.message.reply_text("Skriv: /pivot 3 3", reply_markup=MAIN_KEYBOARD)
        return
    l, r = pr
    ENGINE.settings.pivot_left = int(clamp(l, 1, 20))
    ENGINE.settings.pivot_right = int(clamp(r, 1, 20))
    ENGINE.persist()
    await update.message.reply_text(f"Pivot satt: L={ENGINE.settings.pivot_left} R={ENGINE.settings.pivot_right}", reply_markup=MAIN_KEYBOARD)

async def cmd_pivot_age(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _parse_one_float(context.args)
    if v is None:
        await update.message.reply_text(f"Pivot max age nu: {ENGINE.settings.pivot_max_age_bars} bars\nSkriv: /pivot_age 60", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.pivot_max_age_bars = int(clamp(int(v), 10, 400))
    ENGINE.persist()
    await update.message.reply_text(f"Pivot max age satt: {ENGINE.settings.pivot_max_age_bars} bars", reply_markup=MAIN_KEYBOARD)

async def cmd_sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _parse_one_float(context.args)
    if v is None:
        await update.message.reply_text(f"Sweep nu: {fmt_pct(ENGINE.settings.sweep_pct)}\nSkriv: /sweep 0.08", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.sweep_pct = clamp(v, 0.01, 5.0)
    ENGINE.persist()
    await update.message.reply_text(f"Sweep satt: {fmt_pct(ENGINE.settings.sweep_pct)}", reply_markup=MAIN_KEYBOARD)

async def cmd_reclaim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _parse_one_float(context.args)
    if v is None:
        await update.message.reply_text(f"Reclaim nu: {fmt_pct(ENGINE.settings.reclaim_pct)}\nSkriv: /reclaim 0.04", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.reclaim_pct = clamp(v, 0.00, 5.0)
    ENGINE.persist()
    await update.message.reply_text(f"Reclaim satt: {fmt_pct(ENGINE.settings.reclaim_pct)}", reply_markup=MAIN_KEYBOARD)

async def cmd_trail(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text(
            f"Trail nu: on {fmt_pct(ENGINE.settings.trail_on_pct)} dist {fmt_pct(ENGINE.settings.trail_dist_pct)}\n"
            "Skriv: /trail 1.20 0.60",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    try:
        on = float(str(context.args[0]).replace("%", ""))
        dist = float(str(context.args[1]).replace("%", ""))
    except Exception:
        await update.message.reply_text("Skriv: /trail 1.20 0.60", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.trail_on_pct = clamp(on, 0.05, 25.0)
    ENGINE.settings.trail_dist_pct = clamp(dist, 0.05, 25.0)
    ENGINE.persist()
    await update.message.reply_text(
        f"Trail satt: on {fmt_pct(ENGINE.settings.trail_on_pct)} dist {fmt_pct(ENGINE.settings.trail_dist_pct)}",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_sl_atr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _parse_one_float(context.args)
    if v is None:
        await update.message.reply_text(f"SL ATR mult nu: {ENGINE.settings.sl_atr_mult:.2f}\nSkriv: /sl_atr 2.2", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.sl_atr_mult = clamp(v, 0.5, 10.0)
    ENGINE.persist()
    await update.message.reply_text(f"SL ATR mult satt: {ENGINE.settings.sl_atr_mult:.2f}", reply_markup=MAIN_KEYBOARD)

async def cmd_tp1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text(
            f"TP1 nu: {ENGINE.settings.tp1_r_mult:.2f}R | close {int(ENGINE.settings.tp1_close_pct*100)}%\n"
            "Skriv: /tp1 1.2 50",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    try:
        rmult = float(str(context.args[0]).replace("R", "").strip())
        pct = float(str(context.args[1]).replace("%", "").strip())
    except Exception:
        await update.message.reply_text("Skriv: /tp1 1.2 50", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.tp1_r_mult = clamp(rmult, 0.2, 10.0)
    ENGINE.settings.tp1_close_pct = clamp(pct / 100.0, 0.05, 0.95)
    ENGINE.persist()
    await update.message.reply_text(
        f"TP1 satt: {ENGINE.settings.tp1_r_mult:.2f}R | close {int(ENGINE.settings.tp1_close_pct*100)}%",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_cooldown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _parse_one_float(context.args)
    if v is None:
        await update.message.reply_text(f"Cooldown nu: {ENGINE.settings.cooldown_per_coin_sec}s\nSkriv: /cooldown 45", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.cooldown_per_coin_sec = int(clamp(int(v), 0, 3600))
    ENGINE.persist()
    await update.message.reply_text(f"Cooldown satt: {ENGINE.settings.cooldown_per_coin_sec}s", reply_markup=MAIN_KEYBOARD)

async def cmd_maxpos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _parse_one_float(context.args)
    if v is None:
        await update.message.reply_text(f"Max positions nu: {ENGINE.settings.max_positions}\nSkriv: /maxpos 4", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.max_positions = int(clamp(int(v), 1, 25))
    ENGINE.persist()
    await update.message.reply_text(f"Max positions satt: {ENGINE.settings.max_positions}", reply_markup=MAIN_KEYBOARD)

# =========================
# Engine Loop
# =========================
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if ENGINE is None:
                await asyncio.sleep(2)
                continue

            # run blocking step() in executor
            loop = asyncio.get_running_loop()
            msgs: List[str] = await loop.run_in_executor(None, ENGINE.step)

            if msgs and ENGINE.settings.notify:
                chat_id = app.bot_data.get("chat_id")
                if chat_id:
                    for m in msgs:
                        try:
                            await app.bot.send_message(chat_id=chat_id, text=m)
                        except Exception as e:
                            log.warning(f"Notify failed: {e}")

        except Exception as e:
            log.error(f"Engine loop error: {e}")

        await asyncio.sleep(ENGINE_LOOP_SEC)

# =========================
# Main
# =========================
async def post_init(app: Application):
    # Running loop exists here
    asyncio.create_task(engine_loop(app))

def main():
    global ENGINE

    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN (or BOT_TOKEN) in environment variables.")

    kucoin = KuCoinPublic(timeout=10)
    ENGINE = SweepLongEngine(kucoin)

    app = Application.builder().token(token).post_init(post_init).build()

    # Basic
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("notify", cmd_notify))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))

    # Config
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("coins", cmd_coins))
    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("trend", cmd_trend))
    app.add_handler(CommandHandler("atr_confirm", cmd_atr_confirm))
    app.add_handler(CommandHandler("pivot", cmd_pivot))
    app.add_handler(CommandHandler("pivot_age", cmd_pivot_age))
    app.add_handler(CommandHandler("sweep", cmd_sweep))
    app.add_handler(CommandHandler("reclaim", cmd_reclaim))
    app.add_handler(CommandHandler("trail", cmd_trail))
    app.add_handler(CommandHandler("sl_atr", cmd_sl_atr))
    app.add_handler(CommandHandler("tp1", cmd_tp1))
    app.add_handler(CommandHandler("cooldown", cmd_cooldown))
    app.add_handler(CommandHandler("maxpos", cmd_maxpos))

    log.info("Bot starting (SWEEP LONG only)...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
# =========================
# MpORBbot — Liquidity Sweep Reversal (LONG only)
# Single file main.py (PART 2/2)
# (This part is intentionally empty — all code is contained in PART 1/2)
# =========================
