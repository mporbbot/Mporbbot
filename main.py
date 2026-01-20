import os
import csv
import time
import math
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
log = logging.getLogger("mp_orbbot_pullback")

# =========================
# Strategy: Pullback Momentum (LONG only)
# =========================
STRATEGY_NAME = "Pullback Momentum (LONG only)"
EXCHANGE_NAME = "KuCoin"

# KuCoin endpoints (public)
KUCOIN_BASE = "https://api.kucoin.com"

# =========================
# Defaults / Settings
# =========================
DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

# Timeframes
TF_TREND = "5min"   # trend filter
TF_ENTRY = "1min"   # entry signals

# Candle lookbacks
TREND_CANDLES = 210  # for EMA50 on 5m etc
ENTRY_CANDLES = 210  # for 1m EMA + pullback checks

# Trade engine loop
ENGINE_LOOP_SEC = 3
COOLDOWN_PER_COIN_SEC = 30
MAX_POSITIONS = 5

# Realistic mock costs (defaults)
DEFAULT_FEE_RATE_PER_SIDE = 0.0010     # 0.10% per side
DEFAULT_SLIPPAGE_RATE_PER_SIDE = 0.0002  # 0.02% per side (approx)

# Pullback threshold (how deep pullback must be from recent high, in %)
DEFAULT_PULLBACK_THRESHOLD_PCT = 0.12  # 0.12% default

# TP/SL in % (TP arms trailing; SL is hard stop)
DEFAULT_TP_PCT = 0.60
DEFAULT_SL_PCT = 0.35

# Trailing
DEFAULT_TRAIL_ACTIVATE_PCT = 0.35
DEFAULT_TRAIL_DIST_PCT = 0.20

# Stake
DEFAULT_STAKE_USDT = 30.0

# Files
MOCK_LOG_PATH = "mock_trade_log.csv"
REAL_LOG_PATH = "real_trade_log.csv"  # (live placeholder)
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

def max_n(values: List[float], n: int) -> Optional[float]:
    if len(values) < 1:
        return None
    n = min(n, len(values))
    return max(values[-n:])

def min_n(values: List[float], n: int) -> Optional[float]:
    if len(values) < 1:
        return None
    n = min(n, len(values))
    return min(values[-n:])

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
        """
        KuCoin candles endpoint:
        /api/v1/market/candles?symbol=BTC-USDT&type=1min&startAt=..&endAt=..
        Returns array of [time, open, close, high, low, volume, turnover] (strings)
        Sorted DESC by default. We'll reverse to ASC.
        """
        # estimate seconds per candle
        sec_map = {
            "1min": 60, "3min": 180, "5min": 300, "15min": 900,
            "30min": 1800, "1hour": 3600,
        }
        sec = sec_map.get(ktype, 60)
        end_at = ts_unix()
        start_at = end_at - sec * (limit + 5)

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
        # KuCoin returns DESC time; reverse to ASC
        arr = list(reversed(arr))
        # keep last limit
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

    pullback_threshold_pct: float  # /threshold
    tp_pct: float  # /tp (arms trailing)
    sl_pct: float  # /sl

    trail_activate_pct: float
    trail_dist_pct: float

    fee_rate_per_side: float
    slippage_rate_per_side: float

    max_positions: int
    cooldown_per_coin_sec: int
    notify: bool

    # trend/entry parameters
    trend_ema_fast: int
    trend_ema_slow: int
    entry_ema: int
    pullback_lookback_min: int  # minutes (1m candles) for "recent high"
    min_trend_slope_pct: float  # minimum EMA20 slope over last X
    trend_slope_lookback: int   # candles for slope check

@dataclass
class Position:
    symbol: str
    side: str  # "LONG"
    qty: float
    stake_usdt: float
    entry_price: float
    entry_ts: int

    sl_price: float
    tp_arm_price: float  # reaching this arms trailing

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
            w = csv.writer(f)
            w.writerow(CSV_HEADERS)

def log_trade(
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
        STRATEGY_NAME,
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
        w = csv.writer(f)
        w.writerow(row)

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
class PullbackMomentumEngine:
    def __init__(self, kucoin: KuCoinPublic):
        self.kucoin = kucoin

        self.settings = BotSettings(
            coins=DEFAULT_COINS.copy(),
            trade_mode="mock",
            engine_on=False,
            stake_usdt=DEFAULT_STAKE_USDT,

            pullback_threshold_pct=DEFAULT_PULLBACK_THRESHOLD_PCT,
            tp_pct=DEFAULT_TP_PCT,
            sl_pct=DEFAULT_SL_PCT,

            trail_activate_pct=DEFAULT_TRAIL_ACTIVATE_PCT,
            trail_dist_pct=DEFAULT_TRAIL_DIST_PCT,

            fee_rate_per_side=DEFAULT_FEE_RATE_PER_SIDE,
            slippage_rate_per_side=DEFAULT_SLIPPAGE_RATE_PER_SIDE,

            max_positions=MAX_POSITIONS,
            cooldown_per_coin_sec=COOLDOWN_PER_COIN_SEC,
            notify=True,

            trend_ema_fast=20,
            trend_ema_slow=50,
            entry_ema=20,
            pullback_lookback_min=30,   # recent high over last 30 minutes
            min_trend_slope_pct=0.02,   # 0.02% slope requirement (small but filters chop)
            trend_slope_lookback=6,     # on 5m candles => last 30 min
        )

        self.positions: Dict[str, Position] = {}
        self.coin_state: Dict[str, CoinState] = {
            c: CoinState(last_signal_ts=0, cooldown_until=0) for c in self.settings.coins
        }

        self.pnl = PnL(trades=0, net_usdt=0.0)

        # load persisted settings
        self._load_persisted()

    def _load_persisted(self):
        st = load_state()
        if not st:
            return
        try:
            s = st.get("settings", {})
            # Only load safe fields
            for key in asdict(self.settings).keys():
                if key in s:
                    setattr(self.settings, key, s[key])
            # normalize coin_state for new coins
            for c in self.settings.coins:
                if c not in self.coin_state:
                    self.coin_state[c] = CoinState(last_signal_ts=0, cooldown_until=0)
        except Exception as e:
            log.warning(f"Failed to load persisted state: {e}")

    def persist(self):
        st = {
            "settings": asdict(self.settings),
            "pnl": asdict(self.pnl),
        }
        save_state(st)

    def set_coins(self, coins: List[str]):
        coins = [c.strip().upper() for c in coins if c.strip()]
        self.settings.coins = coins
        # keep state objects
        for c in coins:
            if c not in self.coin_state:
                self.coin_state[c] = CoinState(last_signal_ts=0, cooldown_until=0)
        # remove positions of coins not in list (safety)
        for sym in list(self.positions.keys()):
            if sym not in coins:
                del self.positions[sym]
        self.persist()

    def open_positions_count(self) -> int:
        return len(self.positions)

    def _calc_costs_mock(self, notional_usdt: float) -> Tuple[float, float]:
        # fees per side
        fees = notional_usdt * self.settings.fee_rate_per_side
        slip = notional_usdt * self.settings.slippage_rate_per_side
        return fees, slip

    def _apply_slippage_price(self, price: float, side: str, is_entry: bool) -> float:
        """
        Mock slippage: if buying -> price worse (higher), if selling -> price worse (lower)
        """
        slip = self.settings.slippage_rate_per_side
        if side == "BUY":
            return price * (1 + slip)
        else:
            return price * (1 - slip)

    # ---------- Signals ----------
    def _trend_ok_5m(self, closes_5m: List[float]) -> bool:
        fast = ema(closes_5m, self.settings.trend_ema_fast)
        slow = ema(closes_5m, self.settings.trend_ema_slow)
        if fast is None or slow is None:
            return False
        if fast <= slow:
            return False
        last = closes_5m[-1]

        # slope check: fast EMA rising
        look = self.settings.trend_slope_lookback
        if len(closes_5m) < self.settings.trend_ema_fast + look:
            return False
        # quick slope: compare fast EMA computed on earlier window vs now
        fast_now = fast
        fast_prev = ema(closes_5m[:-look], self.settings.trend_ema_fast)
        if fast_prev is None:
            return False
        slope_pct = (fast_now - fast_prev) / fast_prev * 100.0
        if slope_pct < self.settings.min_trend_slope_pct:
            return False

        # price should be above fast EMA (avoid buying under EMA)
        if last < fast_now:
            return False

        return True

    def _pullback_signal_1m(self, closes_1m: List[float]) -> bool:
        """
        Pullback momentum entry logic:
        1) Find recent high over last N minutes.
        2) Require current close is down from that high by >= pullback_threshold_pct.
        3) Then require a "reclaim" of 1m EMA (cross up) to confirm pullback ended.
        """
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

        # reclaim/cross above EMA: last > EMA and prev <= EMA
        if not (last > e and prev <= e):
            return False

        return True

    # ---------- Entry/Exit ----------
    def _calc_qty(self, stake_usdt: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return stake_usdt / price

    def _level1_prices(self, symbol: str) -> Tuple[float, float, float]:
        d = self.kucoin.get_level1(symbol)
        last = safe_float(d.get("price"))
        bid = safe_float(d.get("bestBid"))
        ask = safe_float(d.get("bestAsk"))
        # fallback
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
        cs = self.coin_state.setdefault(symbol, CoinState(last_signal_ts=0, cooldown_until=0))
        cs.cooldown_until = ts_unix() + int(self.settings.cooldown_per_coin_sec)

    def try_open_long(self, symbol: str) -> Optional[str]:
        if symbol in self.positions:
            return None
        if self.open_positions_count() >= self.settings.max_positions:
            return None
        if not self.can_trade_coin(symbol):
            return None

        # Fetch candles
        candles_5m = self.kucoin.get_candles(symbol, TF_TREND, TREND_CANDLES)
        candles_1m = self.kucoin.get_candles(symbol, TF_ENTRY, ENTRY_CANDLES)

        closes_5m = [safe_float(c[2]) for c in candles_5m]  # close
        closes_1m = [safe_float(c[2]) for c in candles_1m]

        if not self._trend_ok_5m(closes_5m):
            return None
        if not self._pullback_signal_1m(closes_1m):
            return None

        # Entry price: use ask for realistic (marketable buy in mock)
        last, bid, ask = self._level1_prices(symbol)
        entry_raw = ask if ask > 0 else last
        entry_price = entry_raw

        if self.settings.trade_mode == "mock":
            entry_price = self._apply_slippage_price(entry_price, side="BUY", is_entry=True)

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
            f"Trend: 5m EMA{self.settings.trend_ema_fast}>{self.settings.trend_ema_slow} + slope ok\n"
            f"Entry: 1m pullback ≥ {fmt_pct(self.settings.pullback_threshold_pct)} from recent high + reclaim EMA{self.settings.entry_ema}"
        )

    def close_position(self, symbol: str, reason: str) -> Optional[str]:
        pos = self.positions.get(symbol)
        if not pos:
            return None

        last, bid, ask = self._level1_prices(symbol)
        exit_raw = bid if bid > 0 else last  # realistic sell hits bid
        exit_price = exit_raw

        if self.settings.trade_mode == "mock":
            exit_price = self._apply_slippage_price(exit_price, side="SELL", is_entry=False)

        # PnL calc (spot)
        entry_notional = pos.qty * pos.entry_price
        exit_notional = pos.qty * exit_price
        gross_pnl = exit_notional - entry_notional

        fees = 0.0
        slip_cost = 0.0
        if self.settings.trade_mode == "mock":
            # fees + slippage cost (modeled as notional-based)
            fee_entry, slip_entry = self._calc_costs_mock(entry_notional)
            fee_exit, slip_exit = self._calc_costs_mock(exit_notional)
            fees = fee_entry + fee_exit
            slip_cost = slip_entry + slip_exit

        net_pnl = gross_pnl - fees - slip_cost

        # update totals
        self.pnl.trades += 1
        self.pnl.net_usdt += net_pnl

        # log
        if self.settings.trade_mode == "mock":
            log_trade(
                MOCK_LOG_PATH, "mock", symbol, "LONG",
                pos.qty, pos.stake_usdt,
                pos.entry_price, exit_price,
                gross_pnl, fees, slip_cost, net_pnl,
                reason,
            )
        else:
            log_trade(
                REAL_LOG_PATH, "live", symbol, "LONG",
                pos.qty, pos.stake_usdt,
                pos.entry_price, exit_price,
                gross_pnl, fees, slip_cost, net_pnl,
                reason,
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

            # Arm trailing once TP reached (this replaces hard TP with trailing-run)
            if (not pos.trail_active) and price >= pos.tp_arm_price:
                pos.trail_active = True
                pos.trail_highest = price
                pos.trail_stop = price * (1 - self.settings.trail_dist_pct / 100.0)
                pos.last_update_ts = ts_unix()
                msgs.append(
                    f"🟡 TRAIL ARMED {sym} | price={price:.6f} | trail_stop={pos.trail_stop:.6f}"
                )
                continue

            # Trailing management
            if pos.trail_active:
                if price > pos.trail_highest:
                    pos.trail_highest = price
                    pos.trail_stop = pos.trail_highest * (1 - self.settings.trail_dist_pct / 100.0)
                    pos.last_update_ts = ts_unix()

                if price <= pos.trail_stop:
                    m = self.close_position(sym, "TRAIL_STOP")
                    if m:
                        msgs.append(m)
                    continue

            # persist occasionally
            pos.last_update_ts = ts_unix()

        if msgs:
            self.persist()
        return msgs

    def step(self) -> List[str]:
        """
        One engine tick: manage exits then scan entries
        """
        if not self.settings.engine_on:
            return []

        out = []

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

    # ---------- Status ----------
    def status_text(self) -> str:
        s = self.settings
        open_pos = list(self.positions.keys())

        return (
            f"<b>ENGINE:</b> {'ON ✅' if s.engine_on else 'OFF ⛔️'}\n"
            f"<b>Strategy:</b> {STRATEGY_NAME}\n"
            f"<b>Trade mode:</b> {s.trade_mode} (live keys: {'OK' if os.getenv('KUCOIN_KEY') else 'missing'})\n"
            f"<b>Timeframes:</b> trend={TF_TREND} | entry={TF_ENTRY}\n\n"
            f"<b>Pullback threshold</b> (/threshold): {fmt_pct(s.pullback_threshold_pct)}\n"
            f"<b>TP (arms trailing)</b> (/tp): {fmt_pct(s.tp_pct)}\n"
            f"<b>SL</b> (/sl): {fmt_pct(s.sl_pct)}\n"
            f"<b>Trailing dist</b>: {fmt_pct(s.trail_dist_pct)}\n\n"
            f"<b>Stake</b> (/stake): {s.stake_usdt:.2f} USDT\n"
            f"<b>Cooldown/coin:</b> {s.cooldown_per_coin_sec}s | <b>Max positions:</b> {s.max_positions}\n"
            f"<b>Fees:</b> {s.fee_rate_per_side*100:.2f}%/side | <b>Slippage:</b> {s.slippage_rate_per_side*100:.2f}%/side\n\n"
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
        ["/threshold", "/stake"],
        ["/tp", "/sl"],
        ["/coins", "/trade_mode"],
        ["/notify", "/export_csv"],
        ["/close_all", "/reset_pnl"],
    ],
    resize_keyboard=True,
)

def preset_keyboard(kind: str) -> InlineKeyboardMarkup:
    """
    Inline presets for quick settings.
    """
    if kind == "threshold":
        options = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
        rows = []
        row = []
        for v in options:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_threshold:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "tp":
        options = [0.30, 0.45, 0.60, 0.80, 1.00]
        rows = [[InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_tp:{v}") for v in options]]
        return InlineKeyboardMarkup(rows)

    if kind == "sl":
        options = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        rows = []
        row = []
        for v in options:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_sl:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "stake":
        options = [10, 20, 30, 50, 75, 100]
        rows = []
        row = []
        for v in options:
            row.append(InlineKeyboardButton(f"{v} USDT", callback_data=f"set_stake:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "trade_mode":
        rows = [
            [
                InlineKeyboardButton("mock", callback_data="set_mode:mock"),
                InlineKeyboardButton("live", callback_data="set_mode:live"),
            ]
        ]
        return InlineKeyboardMarkup(rows)

    return InlineKeyboardMarkup([])

# =========================
# Telegram Handlers
# =========================
ENGINE: Optional[PullbackMomentumEngine] = None

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Mp ORBbot är igång ✅\n\n"
        "Nu kör vi: Pullback Momentum (LONG only)\n"
        "Trendfilter: 5m | Entry: 1m pullback + reclaim EMA\n\n"
        "Använd knapparna här nere.",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(ENGINE.status_text(), parse_mode=ParseMode.HTML, reply_markup=MAIN_KEYBOARD)

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Total NET PnL: {ENGINE.pnl.net_usdt:+.4f} USDT\nTrades: {ENGINE.pnl.trades}",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.engine_on = True
    ENGINE.persist()
    await update.message.reply_text("✅ ENGINE ON", reply_markup=MAIN_KEYBOARD)

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.engine_on = False
    ENGINE.persist()
    await update.message.reply_text("⛔️ ENGINE OFF", reply_markup=MAIN_KEYBOARD)

async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.notify = not ENGINE.settings.notify
    ENGINE.persist()
    await update.message.reply_text(f"Notify: {'ON' if ENGINE.settings.notify else 'OFF'}", reply_markup=MAIN_KEYBOARD)

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.pnl = PnL(trades=0, net_usdt=0.0)
    ENGINE.persist()
    await update.message.reply_text("PnL reset ✅", reply_markup=MAIN_KEYBOARD)

async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    await update.message.reply_text(
        f"Pullback threshold nu: {fmt_pct(ENGINE.settings.pullback_threshold_pct)}\n"
        f"Välj preset:",
        reply_markup=preset_keyboard("threshold"),
    )

async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"TP (armar trailing) nu: {fmt_pct(ENGINE.settings.tp_pct)}\nVälj preset:",
        reply_markup=preset_keyboard("tp"),
    )

async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"SL nu: {fmt_pct(ENGINE.settings.sl_pct)}\nVälj preset:",
        reply_markup=preset_keyboard("sl"),
    )

async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Stake per trade nu: {ENGINE.settings.stake_usdt:.2f} USDT\nVälj preset:",
        reply_markup=preset_keyboard("stake"),
    )

async def cmd_trade_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Trade mode nu: {ENGINE.settings.trade_mode}\nVälj:",
        reply_markup=preset_keyboard("trade_mode"),
    )

async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # user can pass symbols after /coins
    # Example: /coins BTC-USDT ETH-USDT SOL-USDT
    args = context.args
    if not args:
        await update.message.reply_text(
            "Skicka så här:\n"
            "/coins BTC-USDT ETH-USDT SOL-USDT\n\n"
            f"Nu: {ENGINE.settings.coins}",
            reply_markup=MAIN_KEYBOARD,
        )
        return

    coins = args
    ENGINE.set_coins(coins)
    await update.message.reply_text(f"✅ Coins uppdaterade ({len(ENGINE.settings.coins)}): {ENGINE.settings.coins}", reply_markup=MAIN_KEYBOARD)

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
            ENGINE.settings.sl_pct = clamp(v, 0.05, 5.0)
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
                # safety: require env keys
                if not os.getenv("KUCOIN_KEY") or not os.getenv("KUCOIN_SECRET") or not os.getenv("KUCOIN_PASSPHRASE"):
                    await q.edit_message_text("❌ Live-nycklar saknas i env vars. Sätt KUCOIN_KEY/KUCOIN_SECRET/KUCOIN_PASSPHRASE.")
                    return
                ENGINE.settings.trade_mode = "live"
                ENGINE.persist()
                await q.edit_message_text("⚠️ Trade mode satt till LIVE. (Säkerhetslogik för orderläggning kan byggas vidare.)")
            else:
                ENGINE.settings.trade_mode = "mock"
                ENGINE.persist()
                await q.edit_message_text("✅ Trade mode satt till MOCK")

    except Exception as e:
        await q.edit_message_text(f"Fel: {e}")

# =========================
# Engine Loop Job
# =========================
async def engine_job(context: ContextTypes.DEFAULT_TYPE):
    if ENGINE is None:
        return
    msgs = []
    try:
        # run in executor (requests is blocking)
        msgs = await asyncio.get_event_loop().run_in_executor(None, ENGINE.step)
    except Exception as e:
        log.error(f"Engine step error: {e}")
        return

    if not msgs:
        return

    if ENGINE.settings.notify:
        chat_id = context.application.bot_data.get("chat_id")
        if chat_id:
            for m in msgs:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=m)
                except Exception as e:
                    log.warning(f"Notify failed: {e}")

# =========================
# Main
# =========================
def build_app(token: str) -> Application:
    app = Application.builder().token(token).build()

    # handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("notify", cmd_notify))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))

    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("sl", cmd_sl))
    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("trade_mode", cmd_trade_mode))
    app.add_handler(CommandHandler("coins", cmd_coins))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))

    app.add_handler(CallbackQueryHandler(on_callback))

    return app

async def post_init(app: Application):
    # store a chat_id for push notifications
    # We'll set it on first /start or /status message.
    pass

async def set_chat_id_on_any_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id

def main():
    global ENGINE
    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN (or BOT_TOKEN) in environment variables.")

    kucoin = KuCoinPublic(timeout=10)
    ENGINE = PullbackMomentumEngine(kucoin)

    app = build_app(token)

    # always capture chat_id when user interacts
    app.add_handler(CommandHandler("start", set_chat_id_on_any_message), group=0)
    app.add_handler(CommandHandler("status", set_chat_id_on_any_message), group=0)
    app.add_handler(CommandHandler("pnl", set_chat_id_on_any_message), group=0)

    # schedule engine loop
    # NOTE: requires python-telegram-bot[job-queue]
    if app.job_queue is None:
        raise RuntimeError(
            "JobQueue is None. Install python-telegram-bot[job-queue]==20.3 in requirements.txt"
        )

    app.job_queue.run_repeating(engine_job, interval=ENGINE_LOOP_SEC, first=3)

    log.info("Bot starting...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
