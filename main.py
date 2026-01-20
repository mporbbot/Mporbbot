import os
import csv
import time
import json
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mp_orbbot")

# =========================
# Meta
# =========================
EXCHANGE_NAME = "KuCoin"
KUCOIN_BASE = "https://api.kucoin.com"

# =========================
# Defaults / Settings
# =========================
DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

TF_TREND = "5min"
TF_ENTRY = "1min"

TREND_CANDLES = 210
ENTRY_CANDLES = 240

ENGINE_LOOP_SEC = 3
COOLDOWN_PER_COIN_SEC = 30
MAX_POSITIONS = 5

DEFAULT_FEE_RATE_PER_SIDE = 0.0010       # 0.10% per side
DEFAULT_SLIPPAGE_RATE_PER_SIDE = 0.0002  # 0.02% per side

# Pullback Momentum defaults
DEFAULT_PULLBACK_THRESHOLD_PCT = 0.12
DEFAULT_TP_PCT = 0.60
DEFAULT_SL_PCT = 0.35
DEFAULT_TRAIL_ACTIVATE_PCT = 0.35
DEFAULT_TRAIL_DIST_PCT = 0.20

DEFAULT_STAKE_USDT = 30.0

# Mean Reversion defaults
DEFAULT_MR_BAND_PCT = 0.30      # entry when price <= VWAP*(1-band)
DEFAULT_MR_EXIT_BAND_PCT = 0.05 # exit when price >= VWAP*(1-exit_band)
DEFAULT_MR_RSI_MAX = 40         # RSI <= this for entry
DEFAULT_MR_VWAP_LOOKBACK_MIN = 60  # VWAP lookback (1m candles)
DEFAULT_MR_REQUIRE_RECLAIM_EMA = True

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

def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# =========================
# Indicators
# =========================
def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 1:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += abs(diff)
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def vwap_from_candles(highs: List[float], lows: List[float], closes: List[float], vols: List[float]) -> Optional[float]:
    n = min(len(highs), len(lows), len(closes), len(vols))
    if n < 2:
        return None
    pv_sum = 0.0
    v_sum = 0.0
    for i in range(-n, 0):
        v = vols[i]
        if v <= 0:
            continue
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        pv_sum += tp * v
        v_sum += v
    if v_sum <= 0:
        return None
    return pv_sum / v_sum

def max_n(values: List[float], n: int) -> Optional[float]:
    if not values:
        return None
    n = min(n, len(values))
    return max(values[-n:])

# =========================
# KuCoin client (public)
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
# Trade Models
# =========================
@dataclass
class BotSettings:
    coins: List[str]
    trade_mode: str  # "mock" or "live"
    engine_on: bool
    stake_usdt: float
    notify: bool

    # strategy selector
    strategy: str  # "PULLBACK" or "MEAN_REVERSION"

    # shared risk
    tp_pct: float
    sl_pct: float
    trail_activate_pct: float
    trail_dist_pct: float

    # costs
    fee_rate_per_side: float
    slippage_rate_per_side: float

    max_positions: int
    cooldown_per_coin_sec: int

    # pullback momentum params
    pullback_threshold_pct: float
    trend_ema_fast: int
    trend_ema_slow: int
    entry_ema: int
    pullback_lookback_min: int
    min_trend_slope_pct: float
    trend_slope_lookback: int

    # mean reversion params
    mr_band_pct: float
    mr_exit_band_pct: float
    mr_rsi_max: float
    mr_vwap_lookback_min: int
    mr_require_reclaim_ema: bool
    mr_trend_filter_on: bool  # keep from catching falling knives

@dataclass
class Position:
    symbol: str
    side: str  # "LONG"
    qty: float
    stake_usdt: float
    entry_price: float
    entry_ts: int

    sl_price: float
    tp_arm_price: float

    trail_active: bool
    trail_highest: float
    trail_stop: float

    last_update_ts: int

@dataclass
class PnL:
    trades: int
    net_usdt: float

@dataclass
class CoinState:
    last_signal_ts: int
    cooldown_until: int

# =========================
# CSV Logging
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

def log_trade(
    path: str,
    mode: str,
    strategy: str,
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
        strategy,
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
# Core Engine
# =========================
class TradingEngine:
    def __init__(self, kucoin: KuCoinPublic):
        self.kucoin = kucoin

        self.settings = BotSettings(
            coins=DEFAULT_COINS.copy(),
            trade_mode="mock",
            engine_on=False,
            stake_usdt=DEFAULT_STAKE_USDT,
            notify=True,

            strategy="PULLBACK",  # default

            tp_pct=DEFAULT_TP_PCT,
            sl_pct=DEFAULT_SL_PCT,
            trail_activate_pct=DEFAULT_TRAIL_ACTIVATE_PCT,
            trail_dist_pct=DEFAULT_TRAIL_DIST_PCT,

            fee_rate_per_side=DEFAULT_FEE_RATE_PER_SIDE,
            slippage_rate_per_side=DEFAULT_SLIPPAGE_RATE_PER_SIDE,

            max_positions=MAX_POSITIONS,
            cooldown_per_coin_sec=COOLDOWN_PER_COIN_SEC,

            pullback_threshold_pct=DEFAULT_PULLBACK_THRESHOLD_PCT,
            trend_ema_fast=20,
            trend_ema_slow=50,
            entry_ema=20,
            pullback_lookback_min=30,
            min_trend_slope_pct=0.02,
            trend_slope_lookback=6,

            mr_band_pct=DEFAULT_MR_BAND_PCT,
            mr_exit_band_pct=DEFAULT_MR_EXIT_BAND_PCT,
            mr_rsi_max=DEFAULT_MR_RSI_MAX,
            mr_vwap_lookback_min=DEFAULT_MR_VWAP_LOOKBACK_MIN,
            mr_require_reclaim_ema=DEFAULT_MR_REQUIRE_RECLAIM_EMA,
            mr_trend_filter_on=True,
        )

        self.positions: Dict[str, Position] = {}
        self.coin_state: Dict[str, CoinState] = {c: CoinState(0, 0) for c in self.settings.coins}
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
                    self.coin_state[c] = CoinState(0, 0)
        except Exception as e:
            log.warning(f"Failed to load persisted state: {e}")

    def persist(self):
        save_state({"settings": asdict(self.settings), "pnl": asdict(self.pnl)})

    def set_coins(self, coins: List[str]):
        coins = [c.strip().upper() for c in coins if c.strip()]
        self.settings.coins = coins
        for c in coins:
            if c not in self.coin_state:
                self.coin_state[c] = CoinState(0, 0)
        for sym in list(self.positions.keys()):
            if sym not in coins:
                del self.positions[sym]
        self.persist()

    def open_positions_count(self) -> int:
        return len(self.positions)

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

    def can_trade_coin(self, symbol: str) -> bool:
        cs = self.coin_state.get(symbol)
        if cs is None:
            return True
        return ts_unix() >= cs.cooldown_until

    def mark_cooldown(self, symbol: str):
        cs = self.coin_state.setdefault(symbol, CoinState(0, 0))
        cs.cooldown_until = ts_unix() + int(self.settings.cooldown_per_coin_sec)

    # ===== Trend filter (5m) shared =====
    def _trend_ok_5m(self, closes_5m: List[float]) -> bool:
        fast = ema(closes_5m, self.settings.trend_ema_fast)
        slow = ema(closes_5m, self.settings.trend_ema_slow)
        if fast is None or slow is None:
            return False
        if fast <= slow:
            return False
        last = closes_5m[-1]

        look = self.settings.trend_slope_lookback
        if len(closes_5m) < self.settings.trend_ema_fast + look:
            return False
        fast_prev = ema(closes_5m[:-look], self.settings.trend_ema_fast)
        if fast_prev is None:
            return False

        slope_pct = (fast - fast_prev) / fast_prev * 100.0
        if slope_pct < self.settings.min_trend_slope_pct:
            return False

        if last < fast:
            return False
        return True

    # ===== Strategy 1: Pullback Momentum =====
    def _pullback_signal_1m(self, closes_1m: List[float]) -> bool:
        if len(closes_1m) < 60:
            return False

        lb = max(10, int(self.settings.pullback_lookback_min))
        recent_high = max_n(closes_1m, lb)
        if recent_high is None:
            return False

        last = closes_1m[-1]
        prev = closes_1m[-2]

        depth_pct = (recent_high - last) / recent_high * 100.0
        if depth_pct < self.settings.pullback_threshold_pct:
            return False

        e = ema(closes_1m, self.settings.entry_ema)
        if e is None:
            return False

        # reclaim EMA
        if not (last > e and prev <= e):
            return False

        return True

    # ===== Strategy 2: Mean Reversion (VWAP + RSI) =====
    def _mean_reversion_signal_1m(
        self,
        highs_1m: List[float],
        lows_1m: List[float],
        closes_1m: List[float],
        vols_1m: List[float],
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
        """
        Entry when price is below VWAP by band and RSI is low.
        Optional: require reclaim of EMA to avoid catching falling knife.
        Returns: (ok, vwap, rsi, ema_entry)
        """
        if len(closes_1m) < 80:
            return (False, None, None, None)

        lb = max(20, int(self.settings.mr_vwap_lookback_min))
        lb = min(lb, len(closes_1m))

        v = vwap_from_candles(
            highs_1m[-lb:], lows_1m[-lb:], closes_1m[-lb:], vols_1m[-lb:]
        )
        if v is None or v <= 0:
            return (False, None, None, None)

        last = closes_1m[-1]
        prev = closes_1m[-2]

        band = self.settings.mr_band_pct / 100.0
        entry_level = v * (1 - band)
        if last > entry_level:
            return (False, v, None, None)

        r = rsi(closes_1m, 14)
        if r is None:
            return (False, v, None, None)
        if r > self.settings.mr_rsi_max:
            return (False, v, r, None)

        e = ema(closes_1m, self.settings.entry_ema)
        if e is None:
            return (False, v, r, None)

        if self.settings.mr_require_reclaim_ema:
            # require bounce: cross/reclaim above EMA or at least rising + above EMA
            if not (last > e and prev <= e):
                return (False, v, r, e)

        return (True, v, r, e)

    # ===== Position mgmt =====
    def _calc_qty(self, stake_usdt: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return stake_usdt / price

    def try_open_long(self, symbol: str) -> Optional[str]:
        if symbol in self.positions:
            return None
        if self.open_positions_count() >= self.settings.max_positions:
            return None
        if not self.can_trade_coin(symbol):
            return None

        # candles
        candles_5m = self.kucoin.get_candles(symbol, TF_TREND, TREND_CANDLES)
        candles_1m = self.kucoin.get_candles(symbol, TF_ENTRY, ENTRY_CANDLES)

        closes_5m = [safe_float(c[2]) for c in candles_5m]

        closes_1m = [safe_float(c[2]) for c in candles_1m]
        highs_1m = [safe_float(c[3]) for c in candles_1m]
        lows_1m  = [safe_float(c[4]) for c in candles_1m]
        vols_1m  = [safe_float(c[5]) for c in candles_1m]

        # trend filter
        trend_ok = self._trend_ok_5m(closes_5m)

        # strategy switch
        if self.settings.strategy == "PULLBACK":
            if not trend_ok:
                return None
            if not self._pullback_signal_1m(closes_1m):
                return None
            entry_reason = "PULLBACK_RECLAIM"
            extra_info = f"Entry: pullback ≥ {fmt_pct(self.settings.pullback_threshold_pct)} + reclaim EMA{self.settings.entry_ema}"

        else:  # MEAN_REVERSION
            if self.settings.mr_trend_filter_on and not trend_ok:
                return None
            ok, v, r, e = self._mean_reversion_signal_1m(highs_1m, lows_1m, closes_1m, vols_1m)
            if not ok:
                return None
            entry_reason = "MR_VWAP_RSI"
            extra_info = f"Entry: price under VWAP by {fmt_pct(self.settings.mr_band_pct)} + RSI≤{self.settings.mr_rsi_max:.0f}"

        # Entry price: use ask (marketable buy)
        last, bid, ask = self._level1_prices(symbol)
        entry_price = ask if ask > 0 else last
        if self.settings.trade_mode == "mock":
            entry_price = self._apply_slippage_price(entry_price, "BUY")

        qty = self._calc_qty(self.settings.stake_usdt, entry_price)
        if qty <= 0:
            return None

        sl_price = entry_price * (1 - self.settings.sl_pct / 100.0)
        tp_arm_price = entry_price * (1 + self.settings.tp_pct / 100.0)

        pos = Position(
            symbol=symbol,
            side="LONG",
            qty=qty,
            stake_usdt=self.settings.stake_usdt,
            entry_price=entry_price,
            entry_ts=ts_unix(),
            sl_price=sl_price,
            tp_arm_price=tp_arm_price,
            trail_active=False,
            trail_highest=entry_price,
            trail_stop=entry_price * (1 - self.settings.trail_dist_pct / 100.0),
            last_update_ts=ts_unix(),
        )
        self.positions[symbol] = pos
        self.mark_cooldown(symbol)
        self.persist()

        return (
            f"🟢 ENTRY {symbol} LONG @ {entry_price:.6f}\n"
            f"stake={pos.stake_usdt:.2f} | qty={qty:.8f}\n"
            f"SL={sl_price:.6f} | TP(arm trail)={tp_arm_price:.6f}\n"
            f"Strategy={self.settings.strategy} | {extra_info}\n"
            f"reason={entry_reason}"
        )

    def close_position(self, symbol: str, reason: str) -> Optional[str]:
        pos = self.positions.get(symbol)
        if not pos:
            return None

        last, bid, ask = self._level1_prices(symbol)
        exit_price = bid if bid > 0 else last  # realistic sell hits bid
        if self.settings.trade_mode == "mock":
            exit_price = self._apply_slippage_price(exit_price, "SELL")

        entry_notional = pos.qty * pos.entry_price
        exit_notional = pos.qty * exit_price
        gross_pnl = exit_notional - entry_notional

        fees = 0.0
        slip_cost = 0.0
        if self.settings.trade_mode == "mock":
            fee_entry, slip_entry = self._calc_costs_mock(entry_notional)
            fee_exit, slip_exit = self._calc_costs_mock(exit_notional)
            fees = fee_entry + fee_exit
            slip_cost = slip_entry + slip_exit

        net_pnl = gross_pnl - fees - slip_cost

        self.pnl.trades += 1
        self.pnl.net_usdt += net_pnl

        path = MOCK_LOG_PATH if self.settings.trade_mode == "mock" else REAL_LOG_PATH
        log_trade(
            path=path,
            mode=self.settings.trade_mode,
            strategy=self.settings.strategy,
            symbol=symbol,
            side="LONG",
            qty=pos.qty,
            stake_usdt=pos.stake_usdt,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            fees=fees,
            slippage=slip_cost,
            net_pnl=net_pnl,
            reason=reason,
        )

        del self.positions[symbol]
        self.mark_cooldown(symbol)
        self.persist()

        emoji = "🎯" if net_pnl >= 0 else "🟥"
        return (
            f"{emoji} EXIT {symbol} @ {exit_price:.6f}\n"
            f"Net {net_pnl:+.4f} USDT | reason={reason}\n"
            f"(gross {gross_pnl:+.4f} | fees {fees:.4f} | slip {slip_cost:.4f})"
        )

    def update_positions(self) -> List[str]:
        msgs = []
        for sym, pos in list(self.positions.items()):
            last, bid, ask = self._level1_prices(sym)
            price = last

            # Hard SL
            if price <= pos.sl_price:
                m = self.close_position(sym, "SL")
                if m:
                    msgs.append(m)
                continue

            # Mean Reversion exit: revert to VWAP band
            if self.settings.strategy == "MEAN_REVERSION":
                try:
                    candles_1m = self.kucoin.get_candles(sym, TF_ENTRY, ENTRY_CANDLES)
                    closes_1m = [safe_float(c[2]) for c in candles_1m]
                    highs_1m  = [safe_float(c[3]) for c in candles_1m]
                    lows_1m   = [safe_float(c[4]) for c in candles_1m]
                    vols_1m   = [safe_float(c[5]) for c in candles_1m]

                    lb = max(20, int(self.settings.mr_vwap_lookback_min))
                    lb = min(lb, len(closes_1m))
                    v = vwap_from_candles(highs_1m[-lb:], lows_1m[-lb:], closes_1m[-lb:], vols_1m[-lb:])
                    if v:
                        exit_band = self.settings.mr_exit_band_pct / 100.0
                        target = v * (1 - exit_band)
                        if price >= target:
                            m = self.close_position(sym, "VWAP_REVERT")
                            if m:
                                msgs.append(m)
                            continue
                except Exception as e:
                    log.warning(f"MR exit check error {sym}: {e}")

            # Arm trailing once TP reached
            if (not pos.trail_active) and price >= pos.tp_arm_price:
                pos.trail_active = True
                pos.trail_highest = price
                pos.trail_stop = price * (1 - self.settings.trail_dist_pct / 100.0)
                pos.last_update_ts = ts_unix()
                msgs.append(f"🟡 TRAIL ARMED {sym} | price={price:.6f} | trail_stop={pos.trail_stop:.6f}")
                continue

            if pos.trail_active:
                if price > pos.trail_highest:
                    pos.trail_highest = price
                    pos.trail_stop = pos.trail_highest * (1 - self.settings.trail_dist_pct / 100.0)
                if price <= pos.trail_stop:
                    m = self.close_position(sym, "TRAIL_STOP")
                    if m:
                        msgs.append(m)
                    continue

            pos.last_update_ts = ts_unix()

        if msgs:
            self.persist()
        return msgs

    def step(self) -> List[str]:
        if not self.settings.engine_on:
            return []
        out = []
        out.extend(self.update_positions())

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

        base = (
            f"<b>ENGINE:</b> {'ON ✅' if s.engine_on else 'OFF ⛔️'}\n"
            f"<b>Strategy:</b> {('Pullback Momentum' if s.strategy=='PULLBACK' else 'Mean Reversion (VWAP+RSI)')} ({s.strategy})\n"
            f"<b>Trade mode:</b> {s.trade_mode} (live keys: {'OK' if os.getenv('KUCOIN_KEY') else 'missing'})\n"
            f"<b>Timeframes:</b> trend={TF_TREND} | entry={TF_ENTRY}\n\n"
            f"<b>TP (arm trail)</b> (/tp): {fmt_pct(s.tp_pct)}\n"
            f"<b>SL</b> (/sl): {fmt_pct(s.sl_pct)}\n"
            f"<b>Trailing</b>: activate {fmt_pct(s.trail_activate_pct)} | dist {fmt_pct(s.trail_dist_pct)}\n\n"
            f"<b>Stake</b> (/stake): {s.stake_usdt:.2f} USDT\n"
            f"<b>Cooldown/coin:</b> {s.cooldown_per_coin_sec}s | <b>Max positions:</b> {s.max_positions}\n"
            f"<b>Fees:</b> {s.fee_rate_per_side*100:.2f}%/side | <b>Slippage:</b> {s.slippage_rate_per_side*100:.2f}%/side\n\n"
        )

        if s.strategy == "PULLBACK":
            base += (
                f"<b>Pullback threshold</b> (/threshold): {fmt_pct(s.pullback_threshold_pct)}\n"
                f"<b>Trend filter:</b> EMA{s.trend_ema_fast}>{s.trend_ema_slow} + slope\n"
                f"<b>Entry:</b> pullback + reclaim EMA{s.entry_ema}\n\n"
            )
        else:
            base += (
                f"<b>MR band</b> (/mr_band): {fmt_pct(s.mr_band_pct)} under VWAP\n"
                f"<b>MR exit band</b> (/mr_exit): {fmt_pct(s.mr_exit_band_pct)} under VWAP\n"
                f"<b>MR RSI max</b> (/mr_rsi): {s.mr_rsi_max:.0f}\n"
                f"<b>MR VWAP lookback:</b> {s.mr_vwap_lookback_min} min\n"
                f"<b>MR Trend filter:</b> {'ON' if s.mr_trend_filter_on else 'OFF'}\n\n"
            )

        base += (
            f"<b>Trades:</b> {self.pnl.trades}\n"
            f"<b>Total NET PnL:</b> {self.pnl.net_usdt:+.4f} USDT\n"
            f"<b>Coins ({len(s.coins)}):</b> {s.coins}\n"
            f"<b>Open positions:</b> {open_pos if open_pos else 'none'}\n"
            f"<b>Notify:</b> {'ON' if s.notify else 'OFF'}\n"
        )
        return base

# =========================
# Telegram UI
# =========================
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["/status", "/pnl"],
        ["/engine_on", "/engine_off"],
        ["/strategy", "/threshold"],
        ["/mr_band", "/mr_rsi"],
        ["/mr_exit", "/stake"],
        ["/tp", "/sl"],
        ["/coins", "/trade_mode"],
        ["/notify", "/export_csv"],
        ["/close_all", "/reset_pnl"],
    ],
    resize_keyboard=True,
)

def preset_keyboard(kind: str) -> InlineKeyboardMarkup:
    if kind == "threshold":
        options = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
        rows, row = [], []
        for v in options:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_threshold:{v}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "tp":
        options = [0.25, 0.30, 0.45, 0.60, 0.80, 1.00]
        rows, row = [], []
        for v in options:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_tp:{v}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "sl":
        options = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        rows, row = [], []
        for v in options:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_sl:{v}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "stake":
        options = [10, 20, 30, 50, 75, 100]
        rows, row = [], []
        for v in options:
            row.append(InlineKeyboardButton(f"{v} USDT", callback_data=f"set_stake:{v}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "trade_mode":
        rows = [[
            InlineKeyboardButton("mock", callback_data="set_mode:mock"),
            InlineKeyboardButton("live", callback_data="set_mode:live"),
        ]]
        return InlineKeyboardMarkup(rows)

    if kind == "strategy":
        rows = [[
            InlineKeyboardButton("PULLBACK", callback_data="set_strategy:PULLBACK"),
            InlineKeyboardButton("MEAN_REVERSION", callback_data="set_strategy:MEAN_REVERSION"),
        ]]
        return InlineKeyboardMarkup(rows)

    if kind == "mr_band":
        options = [0.15, 0.20, 0.30, 0.40, 0.60, 0.80]
        rows, row = [], []
        for v in options:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_mr_band:{v}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "mr_exit":
        options = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]
        rows, row = [], []
        for v in options:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_mr_exit:{v}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "mr_rsi":
        options = [30, 35, 40, 45, 50]
        rows = [[InlineKeyboardButton(f"{v}", callback_data=f"set_mr_rsi:{v}") for v in options]]
        return InlineKeyboardMarkup(rows)

    return InlineKeyboardMarkup([])

# =========================
# Telegram Handlers
# =========================
ENGINE: Optional[TradingEngine] = None

async def set_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        "Mp ORBbot är igång ✅\n\n"
        "Nu kan du köra:\n"
        "• Pullback Momentum (trend 5m + pullback 1m)\n"
        "• Mean Reversion (VWAP + RSI)\n\n"
        "Använd knapparna här nere.",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(ENGINE.status_text(), parse_mode=ParseMode.HTML, reply_markup=MAIN_KEYBOARD)

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"Total NET PnL: {ENGINE.pnl.net_usdt:+.4f} USDT\nTrades: {ENGINE.pnl.trades}",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    ENGINE.settings.engine_on = True
    ENGINE.persist()
    await update.message.reply_text("✅ ENGINE ON", reply_markup=MAIN_KEYBOARD)

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    ENGINE.settings.engine_on = False
    ENGINE.persist()
    await update.message.reply_text("⛔️ ENGINE OFF", reply_markup=MAIN_KEYBOARD)

async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    ENGINE.settings.notify = not ENGINE.settings.notify
    ENGINE.persist()
    await update.message.reply_text(f"Notify: {'ON' if ENGINE.settings.notify else 'OFF'}", reply_markup=MAIN_KEYBOARD)

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    ENGINE.pnl = PnL(trades=0, net_usdt=0.0)
    ENGINE.persist()
    await update.message.reply_text("PnL reset ✅", reply_markup=MAIN_KEYBOARD)

async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    msgs = []
    for sym in list(ENGINE.positions.keys()):
        try:
            m = ENGINE.close_position(sym, "MANUAL_CLOSE_ALL")
            if m:
                msgs.append(m)
        except Exception as e:
            msgs.append(f"Close error {sym}: {e}")
    if not msgs:
        msgs = ["Inga öppna positioner."]
    for m in msgs:
        await update.message.reply_text(m, reply_markup=MAIN_KEYBOARD)

async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"Pullback threshold nu: {fmt_pct(ENGINE.settings.pullback_threshold_pct)}\nVälj preset:",
        reply_markup=preset_keyboard("threshold"),
    )

async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"TP (armar trailing) nu: {fmt_pct(ENGINE.settings.tp_pct)}\nVälj preset:",
        reply_markup=preset_keyboard("tp"),
    )

async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"SL nu: {fmt_pct(ENGINE.settings.sl_pct)}\nVälj preset:",
        reply_markup=preset_keyboard("sl"),
    )

async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"Stake per trade nu: {ENGINE.settings.stake_usdt:.2f} USDT\nVälj preset:",
        reply_markup=preset_keyboard("stake"),
    )

async def cmd_trade_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"Trade mode nu: {ENGINE.settings.trade_mode}\nVälj:",
        reply_markup=preset_keyboard("trade_mode"),
    )

async def cmd_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"Strategi nu: {ENGINE.settings.strategy}\nVälj:",
        reply_markup=preset_keyboard("strategy"),
    )

async def cmd_mr_band(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"MR band (under VWAP) nu: {fmt_pct(ENGINE.settings.mr_band_pct)}\nVälj preset:",
        reply_markup=preset_keyboard("mr_band"),
    )

async def cmd_mr_rsi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"MR RSI max nu: {ENGINE.settings.mr_rsi_max:.0f}\nVälj preset:",
        reply_markup=preset_keyboard("mr_rsi"),
    )

async def cmd_mr_exit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    await update.message.reply_text(
        f"MR exit band (under VWAP) nu: {fmt_pct(ENGINE.settings.mr_exit_band_pct)}\nVälj preset:",
        reply_markup=preset_keyboard("mr_exit"),
    )

async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
    args = context.args
    if not args:
        await update.message.reply_text(
            "Skicka så här:\n"
            "/coins BTC-USDT ETH-USDT SOL-USDT\n\n"
            f"Nu: {ENGINE.settings.coins}",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.set_coins(args)
    await update.message.reply_text(
        f"✅ Coins uppdaterade ({len(ENGINE.settings.coins)}): {ENGINE.settings.coins}",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_chat_id(update, context)
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

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    try:
        if data.startswith("set_threshold:"):
            v = float(data.split(":")[1])
            ENGINE.settings.pullback_threshold_pct = clamp(v, 0.01, 2.0)
            ENGINE.persist()
            await q.edit_message_text(f"✅ Threshold satt till {v:.2f}%")

        elif data.startswith("set_tp:"):
            v = float(data.split(":")[1])
            ENGINE.settings.tp_pct = clamp(v, 0.05, 5.0)
            ENGINE.persist()
            await q.edit_message_text(f"✅ TP (arm trail) satt till {v:.2f}%")

        elif data.startswith("set_sl:"):
            v = float(data.split(":")[1])
            ENGINE.settings.sl_pct = clamp(v, 0.05, 10.0)
            ENGINE.persist()
            await q.edit_message_text(f"✅ SL satt till {v:.2f}%")

        elif data.startswith("set_stake:"):
            v = float(data.split(":")[1])
            ENGINE.settings.stake_usdt = clamp(v, 1.0, 100000.0)
            ENGINE.persist()
            await q.edit_message_text(f"✅ Stake satt till {v:.2f} USDT")

        elif data.startswith("set_mode:"):
            mode = data.split(":")[1].strip()
            if mode == "live":
                if not os.getenv("KUCOIN_KEY") or not os.getenv("KUCOIN_SECRET") or not os.getenv("KUCOIN_PASSPHRASE"):
                    await q.edit_message_text("❌ Live-nycklar saknas i env vars. Sätt KUCOIN_KEY/KUCOIN_SECRET/KUCOIN_PASSPHRASE.")
                    return
                ENGINE.settings.trade_mode = "live"
                ENGINE.persist()
                await q.edit_message_text("⚠️ Trade mode satt till LIVE. (Orderläggning kan byggas vidare.)")
            else:
                ENGINE.settings.trade_mode = "mock"
                ENGINE.persist()
                await q.edit_message_text("✅ Trade mode satt till MOCK")

        elif data.startswith("set_strategy:"):
            v = data.split(":")[1].strip().upper()
            if v not in ("PULLBACK", "MEAN_REVERSION"):
                await q.edit_message_text("❌ Okänd strategi.")
                return
            ENGINE.settings.strategy = v
            ENGINE.persist()
            await q.edit_message_text(f"✅ Strategy satt till {v}")

        elif data.startswith("set_mr_band:"):
            v = float(data.split(":")[1])
            ENGINE.settings.mr_band_pct = clamp(v, 0.01, 5.0)
            ENGINE.persist()
            await q.edit_message_text(f"✅ MR band satt till {v:.2f}%")

        elif data.startswith("set_mr_exit:"):
            v = float(data.split(":")[1])
            ENGINE.settings.mr_exit_band_pct = clamp(v, 0.0, 2.0)
            ENGINE.persist()
            await q.edit_message_text(f"✅ MR exit band satt till {v:.2f}%")

        elif data.startswith("set_mr_rsi:"):
            v = float(data.split(":")[1])
            ENGINE.settings.mr_rsi_max = clamp(v, 10.0, 90.0)
            ENGINE.persist()
            await q.edit_message_text(f"✅ MR RSI max satt till {v:.0f}")

    except Exception as e:
        await q.edit_message_text(f"Fel: {e}")

# =========================
# Engine Loop (no JobQueue)
# =========================
async def engine_loop(app: Application):
    while True:
        try:
            if ENGINE is not None:
                msgs = await asyncio.get_event_loop().run_in_executor(None, ENGINE.step)
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

async def post_init(app: Application):
    asyncio.create_task(engine_loop(app))

# =========================
# Main
# =========================
def build_app(token: str) -> Application:
    return Application.builder().token(token).post_init(post_init).build()

def main():
    global ENGINE
    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN (or BOT_TOKEN) in environment variables.")

    kucoin = KuCoinPublic(timeout=10)
    ENGINE = TradingEngine(kucoin)

    app = build_app(token)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("notify", cmd_notify))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))

    app.add_handler(CommandHandler("strategy", cmd_strategy))
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("mr_band", cmd_mr_band))
    app.add_handler(CommandHandler("mr_rsi", cmd_mr_rsi))
    app.add_handler(CommandHandler("mr_exit", cmd_mr_exit))

    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("sl", cmd_sl))
    app.add_handler(CommandHandler("stake", cmd_stake))

    app.add_handler(CommandHandler("trade_mode", cmd_trade_mode))
    app.add_handler(CommandHandler("coins", cmd_coins))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))

    app.add_handler(CallbackQueryHandler(on_callback))

    log.info("Bot starting...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
