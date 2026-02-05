# =========================
# Mp ORBbot — Mean Reversion (VWAP + RSI) + Momentum/Swing Exit — LONG only
# Single file main.py (DEL 1/2)
# python-telegram-bot==20.3 (NO JobQueue needed)
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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mp_orbbot_mr2")

# =========================
# Strategy identity
# =========================
STRATEGY_NAME = "Mean Reversion (VWAP+RSI) + Momentum/Swing Exit"
STRATEGY_CODE = "MR_MOMO_EXIT"
EXCHANGE_NAME = "KuCoin"
KUCOIN_BASE = "https://api.kucoin.com"

# =========================
# Defaults
# =========================
DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

TF_TREND = "5min"
TF_ENTRY = "1min"

TREND_CANDLES = 220
ENTRY_CANDLES = 220

ENGINE_LOOP_SEC = 3
COOLDOWN_PER_COIN_SEC = 30
MAX_POSITIONS = 5

DEFAULT_FEE_RATE_PER_SIDE = 0.0010      # 0.10% per side
DEFAULT_SLIPPAGE_RATE_PER_SIDE = 0.0002 # 0.02% per side
DEFAULT_STAKE_USDT = 30.0

# Mean reversion settings (percent)
DEFAULT_MR_BAND_PCT = 0.80       # entry if price is >= this % under VWAP
DEFAULT_MR_EXIT_PCT = 0.20       # "reclaim" exit when dev >= -exit_pct
DEFAULT_MR_RSI_MAX = 35          # entry requires RSI <= this
DEFAULT_MR_VWAP_LOOKBACK_MIN = 60
DEFAULT_MR_TREND_FILTER = True
DEFAULT_MR_TREND_STRICT = False  # if ON: don't take MR trades when trend filter fails

# Risk / exits
DEFAULT_TP_PCT = 1.00            # when move >= TP, arm trailing if not already
DEFAULT_SL_PCT = 0.70

DEFAULT_TRAIL_ACTIVATE_PCT = 1.50  # arm trailing when move >= this
DEFAULT_TRAIL_DIST_PCT = 0.60      # trailing stop distance

# Extra: momentum-based “bailout” when trade stalls
DEFAULT_STALL_MINUTES = 180        # after N minutes, if still under VWAP and momentum weak -> exit
DEFAULT_MIN_ROC_EXIT = -0.15       # 1m ROC% threshold used with stall logic

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

def yesno(s: str) -> Optional[bool]:
    t = (s or "").strip().lower()
    if t in ("on", "1", "true", "yes", "ja"):
        return True
    if t in ("off", "0", "false", "no", "nej"):
        return False
    return None

# =========================
# Indicators
# =========================
def ema_series(values: List[float], period: int) -> List[float]:
    if period <= 1 or len(values) == 0:
        return values[:]
    k = 2 / (period + 1)
    out = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out

def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return ema_series(values, period)[-1]

def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 2:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += -diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def roc_pct(closes: List[float], lookback: int = 3) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    a = closes[-1]
    b = closes[-1 - lookback]
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

def vwap(candles: List[List[str]]) -> Optional[float]:
    """
    candles: [time, open, close, high, low, volume, turnover]
    VWAP approx using typical price * volume (volume field)
    """
    if not candles:
        return None
    pv = 0.0
    vol = 0.0
    for c in candles:
        high = safe_float(c[3])
        low = safe_float(c[4])
        close = safe_float(c[2])
        volume = safe_float(c[5])
        tp = (high + low + close) / 3.0
        pv += tp * volume
        vol += volume
    if vol <= 0:
        return safe_float(candles[-1][2])
    return pv / vol

# =========================
# KuCoin public client (blocking, used via executor)
# =========================
class KuCoinPublic:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MpORBbot/2.0"})

    def get_level1(self, symbol: str) -> Dict:
        url = f"{KUCOIN_BASE}/api/v1/market/orderbook/level1"
        r = self.session.get(url, params={"symbol": symbol}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        return data["data"]

    def get_candles(self, symbol: str, ktype: str, limit: int) -> List[List[str]]:
        sec_map = {"1min": 60, "3min": 180, "5min": 300, "15min": 900, "30min": 1800, "1hour": 3600}
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
        arr = list(reversed(arr))  # oldest -> newest
        if len(arr) > limit:
            arr = arr[-limit:]
        return arr

# =========================
# Models
# =========================
@dataclass
class BotSettings:
    coins: List[str]
    trade_mode: str
    engine_on: bool
    stake_usdt: float

    tp_pct: float
    sl_pct: float
    trail_activate_pct: float
    trail_dist_pct: float

    fee_rate_per_side: float
    slippage_rate_per_side: float

    max_positions: int
    cooldown_per_coin_sec: int
    notify: bool

    mr_band_pct: float
    mr_exit_pct: float
    mr_rsi_max: int
    mr_vwap_lookback_min: int
    mr_trend_filter: bool
    mr_trend_strict: bool

    stall_minutes: int
    min_roc_exit: float

@dataclass
class Position:
    symbol: str
    side: str
    qty: float
    stake_usdt: float
    entry_price: float
    entry_ts: int

    sl_price: float
    tp_arm_price: float

    trail_active: bool
    trail_highest: float
    trail_stop: float

    entry_vwap: float
    entry_rsi: float
    entry_dev_pct: float

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
# Engine
# =========================
class Engine:
    def __init__(self, kucoin: KuCoinPublic):
        self.kucoin = kucoin

        self.settings = BotSettings(
            coins=DEFAULT_COINS.copy(),
            trade_mode="mock",
            engine_on=False,
            stake_usdt=DEFAULT_STAKE_USDT,

            tp_pct=DEFAULT_TP_PCT,
            sl_pct=DEFAULT_SL_PCT,
            trail_activate_pct=DEFAULT_TRAIL_ACTIVATE_PCT,
            trail_dist_pct=DEFAULT_TRAIL_DIST_PCT,

            fee_rate_per_side=DEFAULT_FEE_RATE_PER_SIDE,
            slippage_rate_per_side=DEFAULT_SLIPPAGE_RATE_PER_SIDE,

            max_positions=MAX_POSITIONS,
            cooldown_per_coin_sec=COOLDOWN_PER_COIN_SEC,
            notify=True,

            mr_band_pct=DEFAULT_MR_BAND_PCT,
            mr_exit_pct=DEFAULT_MR_EXIT_PCT,
            mr_rsi_max=DEFAULT_MR_RSI_MAX,
            mr_vwap_lookback_min=DEFAULT_MR_VWAP_LOOKBACK_MIN,
            mr_trend_filter=DEFAULT_MR_TREND_FILTER,
            mr_trend_strict=DEFAULT_MR_TREND_STRICT,

            stall_minutes=DEFAULT_STALL_MINUTES,
            min_roc_exit=DEFAULT_MIN_ROC_EXIT,
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

    # ----- Trend filter
    def _trend_ok_5m(self, closes_5m: List[float]) -> bool:
        if not self.settings.mr_trend_filter:
            return True
        e20 = ema(closes_5m, 20)
        e50 = ema(closes_5m, 50)
        if e20 is None or e50 is None:
            return False
        return e20 > e50

    # ----- MR signal
    def _mr_signal(
        self,
        candles_1m: List[List[str]],
        closes_1m: List[float],
        closes_5m: List[float],
    ) -> Tuple[bool, float, float, float, float]:
        """
        Returns: (signal, vwap_price, rsi_val, dev_pct, roc3)
        dev_pct is (price - vwap)/vwap*100 (negative means under)
        """
        if len(closes_1m) < 30:
            return (False, 0.0, 50.0, 0.0, 0.0)

        look = max(10, int(self.settings.mr_vwap_lookback_min))
        candles_slice = candles_1m[-look:] if len(candles_1m) >= look else candles_1m
        vw = vwap(candles_slice)
        if vw is None or vw <= 0:
            return (False, 0.0, 50.0, 0.0, 0.0)

        price = closes_1m[-1]
        r = rsi(closes_1m, 14)
        if r is None:
            return (False, vw, 50.0, 0.0, 0.0)

        dev = (price - vw) / vw * 100.0
        roc3 = roc_pct(closes_1m, 3)

        trend_ok = self._trend_ok_5m(closes_5m)

        if self.settings.mr_trend_strict and not trend_ok:
            return (False, vw, r, dev, roc3)

        if dev <= -self.settings.mr_band_pct and r <= float(self.settings.mr_rsi_max) and trend_ok:
            return (True, vw, r, dev, roc3)

        return (False, vw, r, dev, roc3)

    def try_open_long(self, symbol: str) -> Optional[str]:
        if symbol in self.positions:
            return None
        if self.open_positions_count() >= self.settings.max_positions:
            return None
        if not self.can_trade_coin(symbol):
            return None

        candles_5m = self.kucoin.get_candles(symbol, TF_TREND, TREND_CANDLES)
        candles_1m = self.kucoin.get_candles(symbol, TF_ENTRY, ENTRY_CANDLES)

        closes_5m = [safe_float(c[2]) for c in candles_5m]
        closes_1m = [safe_float(c[2]) for c in candles_1m]

        signal, vw, r, dev, roc3 = self._mr_signal(candles_1m, closes_1m, closes_5m)
        if not signal:
            return None

        last, bid, ask = self._level1_prices(symbol)
        entry_raw = ask if ask > 0 else last
        entry_price = entry_raw

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
            entry_vwap=vw,
            entry_rsi=r,
            entry_dev_pct=dev,
        )
        self.positions[symbol] = pos
        self.mark_cooldown(symbol)
        self.persist()

        return (
            f"ENTRY {symbol} LONG @ {entry_price:.6f}\n"
            f"VWAP={vw:.6f} | dev={dev:.2f}% | RSI={r:.1f} | roc3={roc3:.2f}%\n"
            f"stake={pos.stake_usdt:.2f} | qty={qty:.8f}\n"
            f"SL={sl_price:.6f} | TP(arm trail)={tp_arm_price:.6f}"
        )

    def close_position(self, symbol: str, reason: str, exit_hint_price: Optional[float] = None) -> Optional[str]:
        pos = self.positions.get(symbol)
        if not pos:
            return None

        if exit_hint_price is None:
            last, bid, ask = self._level1_prices(symbol)
            exit_raw = bid if bid > 0 else last
        else:
            exit_raw = exit_hint_price

        exit_price = exit_raw
        if self.settings.trade_mode == "mock":
            exit_price = self._apply_slippage_price(exit_price, "SELL")

        entry_notional = pos.qty * pos.entry_price
        exit_notional = pos.qty * exit_price
        gross_pnl = exit_notional - entry_notional

        fees = 0.0
        slip_cost = 0.0
        if self.settings.trade_mode == "mock":
            fee_in, slip_in = self._calc_costs_mock(entry_notional)
            fee_out, slip_out = self._calc_costs_mock(exit_notional)
            fees = fee_in + fee_out
            slip_cost = slip_in + slip_out

        net_pnl = gross_pnl - fees - slip_cost

        self.pnl.trades += 1
        self.pnl.net_usdt += net_pnl

        path = MOCK_LOG_PATH if self.settings.trade_mode == "mock" else REAL_LOG_PATH
        log_trade_csv(
            path,
            self.settings.trade_mode,
            symbol,
            "LONG",
            pos.qty,
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
        self.persist()

        return (
            f"EXIT {symbol} @ {exit_price:.6f}\n"
            f"Net {net_pnl:+.4f} USDT | reason={reason}\n"
            f"(gross {gross_pnl:+.4f} | fees {fees:.4f} | slip {slip_cost:.4f})"
        )

    def update_positions(self) -> List[str]:
        msgs: List[str] = []
        s = self.settings

        for sym, pos in list(self.positions.items()):
            # fetch candles to evaluate VWAP reclaim + momentum
            candles_1m = self.kucoin.get_candles(sym, TF_ENTRY, ENTRY_CANDLES)
            closes_1m = [safe_float(c[2]) for c in candles_1m]
            price = closes_1m[-1] if closes_1m else pos.entry_price

            look = max(10, int(s.mr_vwap_lookback_min))
            vw = vwap(candles_1m[-look:]) if len(candles_1m) >= 10 else pos.entry_vwap
            if vw is None or vw <= 0:
                vw = pos.entry_vwap if pos.entry_vwap > 0 else price

            dev = (price - vw) / vw * 100.0 if vw > 0 else 0.0
            r = rsi(closes_1m, 14) or 50.0
            roc3 = roc_pct(closes_1m, 3)

            # --- SL
            if price <= pos.sl_price:
                m = self.close_position(sym, "SL", exit_hint_price=price)
                if m:
                    msgs.append(m)
                continue

            # --- Fast MR reclaim exit (scalp)
            # exit when price has reclaimed VWAP area (dev >= -mr_exit_pct)
            if dev >= -s.mr_exit_pct:
                m = self.close_position(sym, "MR_RECLAIM_EXIT", exit_hint_price=price)
                if m:
                    msgs.append(m)
                continue

            # --- Trailing arm conditions:
            move_pct = (price - pos.entry_price) / pos.entry_price * 100.0 if pos.entry_price > 0 else 0.0

            # arm if move >= trail_activate_pct OR move >= tp_pct (backup)
            if (not pos.trail_active) and (move_pct >= s.trail_activate_pct or move_pct >= s.tp_pct):
                pos.trail_active = True
                pos.trail_highest = price
                pos.trail_stop = price * (1 - s.trail_dist_pct / 100.0)
                msgs.append(f"TRAIL ARMED {sym} | move={move_pct:.2f}% | stop={pos.trail_stop:.6f}")
                continue

            # --- Trailing management
            if pos.trail_active:
                if price > pos.trail_highest:
                    pos.trail_highest = price
                    pos.trail_stop = pos.trail_highest * (1 - s.trail_dist_pct / 100.0)
                if price <= pos.trail_stop:
                    m = self.close_position(sym, "TRAIL_STOP", exit_hint_price=price)
                    if m:
                        msgs.append(m)
                    continue

            # --- Stall / momentum bailout (longer hold logic)
            # after stall_minutes, if still under VWAP and momentum weak/negative -> exit to avoid bleed
            age_min = (ts_unix() - pos.entry_ts) / 60.0
            if age_min >= max(30, s.stall_minutes):
                if dev < -s.mr_exit_pct and roc3 <= s.min_roc_exit:
                    m = self.close_position(sym, "STALL_MOMO_BAILOUT", exit_hint_price=price)
                    if m:
                        msgs.append(m)
                    continue

        if msgs:
            self.persist()
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
            f"<b>Trade mode:</b> {s.trade_mode}\n"
            f"<b>Timeframes:</b> trend={TF_TREND} | entry={TF_ENTRY}\n\n"
            f"<b>TP</b> (/tp): {fmt_pct(s.tp_pct)}\n"
            f"<b>SL</b> (/sl): {fmt_pct(s.sl_pct)}\n"
            f"<b>Trailing:</b> activate {fmt_pct(s.trail_activate_pct)} | dist {fmt_pct(s.trail_dist_pct)}\n\n"
            f"<b>Stake</b> (/stake): {s.stake_usdt:.2f} USDT\n"
            f"<b>Cooldown/coin:</b> {s.cooldown_per_coin_sec}s | <b>Max positions:</b> {s.max_positions}\n"
            f"<b>Fees:</b> {s.fee_rate_per_side*100:.2f}%/side | <b>Slippage:</b> {s.slippage_rate_per_side*100:.2f}%/side\n\n"
            f"<b>MR band</b> (/mr_band): {fmt_pct(s.mr_band_pct)} under VWAP\n"
            f"<b>MR exit</b> (/mr_exit): {fmt_pct(s.mr_exit_pct)} under VWAP\n"
            f"<b>MR RSI max</b> (/mr_rsi): {s.mr_rsi_max}\n"
            f"<b>MR VWAP lookback:</b> {s.mr_vwap_lookback_min} min\n"
            f"<b>MR Trend filter:</b> {'ON' if s.mr_trend_filter else 'OFF'}\n"
            f"<b>MR Trend strict:</b> {'ON' if s.mr_trend_strict else 'OFF'}\n\n"
            f"<b>Stall exit:</b> {s.stall_minutes}m | roc3 <= {s.min_roc_exit:.2f}%\n\n"
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
        ["/mr_band", "/mr_exit"],
        ["/mr_rsi", "/mr_trend"],
        ["/mr_trend_strict", "/vwap_lookback"],
        ["/stake", "/tp"],
        ["/sl", "/trail_activate"],
        ["/trail_dist", "/stall"],
        ["/coins", "/notify"],
        ["/close_all", "/reset_pnl"],
        ["/export_csv", "/help"],
    ],
    resize_keyboard=True,
)

def preset_keyboard(kind: str) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []

    def pack(values: List[str], cb_prefix: str):
        r: List[InlineKeyboardButton] = []
        for i, v in enumerate(values, 1):
            r.append(InlineKeyboardButton(v, callback_data=f"{cb_prefix}:{v}"))
            if i % 3 == 0:
                rows.append(r)
                r = []
        if r:
            rows.append(r)

    if kind == "mr_band":
        pack([f"{x:.2f}" for x in [0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00]], "set_mr_band")
    elif kind == "mr_exit":
        pack([f"{x:.2f}" for x in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]], "set_mr_exit")
    elif kind == "mr_rsi":
        pack([str(x) for x in [25, 30, 35, 40, 45, 50]], "set_mr_rsi")
    elif kind == "tp":
        pack([f"{x:.2f}" for x in [0.60, 0.80, 1.00, 1.20, 1.50, 2.00]], "set_tp")
    elif kind == "sl":
        pack([f"{x:.2f}" for x in [0.35, 0.50, 0.70, 1.00, 1.50, 2.00]], "set_sl")
    elif kind == "stake":
        pack([str(x) for x in [10, 20, 30, 50, 75, 100]], "set_stake")
    elif kind == "trail_activate":
        pack([f"{x:.2f}" for x in [0.60, 1.00, 1.50, 2.00, 3.00, 5.00]], "set_trail_activate")
    elif kind == "trail_dist":
        pack([f"{x:.2f}" for x in [0.20, 0.30, 0.60, 1.00, 1.50, 2.00]], "set_trail_dist")

    return InlineKeyboardMarkup(rows)

# =========================
# Globals
# =========================
ENGINE: Optional[Engine] = None
ENGINE_TASK: Optional[asyncio.Task] = None

# =========================
# Engine loop (async wrapper)
# =========================
async def engine_loop(app: Application):
    """
    Runs forever in background while ENGINE.settings.engine_on is True.
    Uses run_in_executor because requests is blocking.
    """
    assert ENGINE is not None
    loop = asyncio.get_running_loop()

    while True:
        await asyncio.sleep(ENGINE_LOOP_SEC)

        # If engine switched off, just idle
        if not ENGINE.settings.engine_on:
            continue

        try:
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
# =========================
# Mp ORBbot — main.py (DEL 2/2)
# =========================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        "Mp ORBbot ✅\n\n"
        "Strategi: Mean Reversion (VWAP+RSI) + Momentum/Swing exit (LONG only)\n"
        "• Entry: pris under VWAP-band + RSI <= max (+ trendfilter)\n"
        "• Exit: MR reclaim (snabb) eller trailing (swing) eller stall-bailout\n\n"
        "Tips: skriv valfria värden!\n"
        "/mr_band 1.20\n"
        "/mr_exit 0.25\n"
        "/mr_rsi 35\n"
        "/trail_activate 1.50\n"
        "/trail_dist 0.60\n",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Kommandon:\n"
        "/status /pnl\n"
        "/engine_on /engine_off\n"
        "/mr_band [pct]\n"
        "/mr_exit [pct]\n"
        "/mr_rsi [int]\n"
        "/mr_trend on|off\n"
        "/mr_trend_strict on|off\n"
        "/vwap_lookback [min]\n"
        "/stake [usdt]\n"
        "/tp [pct]\n"
        "/sl [pct]\n"
        "/trail_activate [pct]\n"
        "/trail_dist [pct]\n"
        "/stall [minutes] [min_roc]\n"
        "/coins BTC-USDT ETH-USDT ...\n"
        "/notify (toggle)\n"
        "/close_all\n"
        "/reset_pnl\n"
        "/export_csv\n",
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

async def _ensure_engine_task(app: Application):
    global ENGINE_TASK
    if ENGINE_TASK is None or ENGINE_TASK.done():
        ENGINE_TASK = asyncio.create_task(engine_loop(app))

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.engine_on = True
    ENGINE.persist()
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id
    await _ensure_engine_task(context.application)
    await update.message.reply_text("ENGINE ON ✅", reply_markup=MAIN_KEYBOARD)

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.engine_on = False
    ENGINE.persist()
    await update.message.reply_text("ENGINE OFF ⛔️", reply_markup=MAIN_KEYBOARD)

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

async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "Skicka så här:\n/coins BTC-USDT ETH-USDT SOL-USDT\n\n"
            f"Nu: {ENGINE.settings.coins}",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.set_coins(args)
    await update.message.reply_text(f"Coins uppdaterade ({len(ENGINE.settings.coins)}): {ENGINE.settings.coins}", reply_markup=MAIN_KEYBOARD)

# ----- Numeric setter helper (accepts typing, not only presets)
async def _set_float(update: Update, name: str, lo: float, hi: float, unit: str = ""):
    parts = (update.message.text or "").split()
    if len(parts) == 1:
        await update.message.reply_text("Skriv värde också. Ex: /mr_band 1.20", reply_markup=MAIN_KEYBOARD)
        return
    try:
        v = float(parts[1])
        v = clamp(v, lo, hi)
        setattr(ENGINE.settings, name, v)
        ENGINE.persist()
        await update.message.reply_text(f"{name} satt till {v:.2f}{unit}", reply_markup=MAIN_KEYBOARD)
    except Exception:
        await update.message.reply_text("Fel format. Ex: /mr_band 1.20", reply_markup=MAIN_KEYBOARD)

async def _set_int(update: Update, name: str, lo: int, hi: int):
    parts = (update.message.text or "").split()
    if len(parts) == 1:
        await update.message.reply_text("Skriv värde också. Ex: /mr_rsi 35", reply_markup=MAIN_KEYBOARD)
        return
    try:
        v = int(float(parts[1]))
        v = int(clamp(v, lo, hi))
        setattr(ENGINE.settings, name, v)
        ENGINE.persist()
        await update.message.reply_text(f"{name} satt till {v}", reply_markup=MAIN_KEYBOARD)
    except Exception:
        await update.message.reply_text("Fel format. Ex: /mr_rsi 35", reply_markup=MAIN_KEYBOARD)

# ----- Commands (show presets + allow typing)
async def cmd_mr_band(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_float(update, "mr_band_pct", 0.05, 10.0, "%")
    await update.message.reply_text(
        f"MR band nu: {fmt_pct(ENGINE.settings.mr_band_pct)} under VWAP\n"
        "Välj preset eller skriv själv: /mr_band 1.20",
        reply_markup=preset_keyboard("mr_band"),
    )

async def cmd_mr_exit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_float(update, "mr_exit_pct", 0.00, 10.0, "%")
    await update.message.reply_text(
        f"MR exit nu: {fmt_pct(ENGINE.settings.mr_exit_pct)} under VWAP\n"
        "Välj preset eller skriv själv: /mr_exit 0.25",
        reply_markup=preset_keyboard("mr_exit"),
    )

async def cmd_mr_rsi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_int(update, "mr_rsi_max", 5, 95)
    await update.message.reply_text(
        f"MR RSI max nu: {ENGINE.settings.mr_rsi_max}\n"
        "Välj preset eller skriv själv: /mr_rsi 35",
        reply_markup=preset_keyboard("mr_rsi"),
    )

async def cmd_mr_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) == 1:
        await update.message.reply_text(
            f"MR Trend filter är nu: {'ON' if ENGINE.settings.mr_trend_filter else 'OFF'}\n"
            "Skriv: /mr_trend on eller /mr_trend off",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    yn = yesno(parts[1])
    if yn is None:
        await update.message.reply_text("Skriv: /mr_trend on/off", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.mr_trend_filter = yn
    ENGINE.persist()
    await update.message.reply_text(f"MR Trend filter satt till {'ON' if yn else 'OFF'}", reply_markup=MAIN_KEYBOARD)

async def cmd_mr_trend_strict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) == 1:
        await update.message.reply_text(
            f"MR Trend strict är nu: {'ON' if ENGINE.settings.mr_trend_strict else 'OFF'}\n"
            "Skriv: /mr_trend_strict on eller /mr_trend_strict off",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    yn = yesno(parts[1])
    if yn is None:
        await update.message.reply_text("Skriv: /mr_trend_strict on/off", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.mr_trend_strict = yn
    ENGINE.persist()
    await update.message.reply_text(f"MR Trend strict satt till {'ON' if yn else 'OFF'}", reply_markup=MAIN_KEYBOARD)

async def cmd_vwap_lookback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_int(update, "mr_vwap_lookback_min", 10, 600)
    await update.message.reply_text(
        f"VWAP lookback nu: {ENGINE.settings.mr_vwap_lookback_min} min\n"
        "Skriv: /vwap_lookback 90",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_float(update, "stake_usdt", 1.0, 100000.0, " USDT")
    await update.message.reply_text(
        f"Stake nu: {ENGINE.settings.stake_usdt:.2f} USDT\n"
        "Välj preset eller skriv själv: /stake 30",
        reply_markup=preset_keyboard("stake"),
    )

async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_float(update, "tp_pct", 0.05, 50.0, "%")
    await update.message.reply_text(
        f"TP nu: {fmt_pct(ENGINE.settings.tp_pct)}\n"
        "Välj preset eller skriv själv: /tp 1.20",
        reply_markup=preset_keyboard("tp"),
    )

async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_float(update, "sl_pct", 0.05, 50.0, "%")
    await update.message.reply_text(
        f"SL nu: {fmt_pct(ENGINE.settings.sl_pct)}\n"
        "Välj preset eller skriv själv: /sl 0.70",
        reply_markup=preset_keyboard("sl"),
    )

async def cmd_trail_activate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_float(update, "trail_activate_pct", 0.05, 50.0, "%")
    await update.message.reply_text(
        f"Trail activate nu: {fmt_pct(ENGINE.settings.trail_activate_pct)}\n"
        "Välj preset eller skriv själv: /trail_activate 1.50",
        reply_markup=preset_keyboard("trail_activate"),
    )

async def cmd_trail_dist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split()
    if len(parts) > 1:
        return await _set_float(update, "trail_dist_pct", 0.05, 50.0, "%")
    await update.message.reply_text(
        f"Trail dist nu: {fmt_pct(ENGINE.settings.trail_dist_pct)}\n"
        "Välj preset eller skriv själv: /trail_dist 0.60",
        reply_markup=preset_keyboard("trail_dist"),
    )

async def cmd_stall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /stall 180 -0.15
    parts = (update.message.text or "").split()
    if len(parts) == 1:
        await update.message.reply_text(
            f"Stall nu: {ENGINE.settings.stall_minutes} min | min_roc_exit={ENGINE.settings.min_roc_exit:.2f}%\n"
            "Skriv: /stall 180 -0.15",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    try:
        mins = int(float(parts[1]))
        mins = int(clamp(mins, 30, 20000))
        ENGINE.settings.stall_minutes = mins
        if len(parts) >= 3:
            roc = float(parts[2])
            ENGINE.settings.min_roc_exit = clamp(roc, -10.0, 10.0)
        ENGINE.persist()
        await update.message.reply_text(
            f"Stall satt: {ENGINE.settings.stall_minutes} min | min_roc_exit={ENGINE.settings.min_roc_exit:.2f}%",
            reply_markup=MAIN_KEYBOARD,
        )
    except Exception:
        await update.message.reply_text("Fel format. Ex: /stall 180 -0.15", reply_markup=MAIN_KEYBOARD)

# ----- Callback presets
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    try:
        key, val = data.split(":", 1)

        if key == "set_mr_band":
            v = float(val)
            ENGINE.settings.mr_band_pct = clamp(v, 0.05, 10.0)
            ENGINE.persist()
            await q.edit_message_text(f"MR band satt till {v:.2f}%")

        elif key == "set_mr_exit":
            v = float(val)
            ENGINE.settings.mr_exit_pct = clamp(v, 0.00, 10.0)
            ENGINE.persist()
            await q.edit_message_text(f"MR exit satt till {v:.2f}%")

        elif key == "set_mr_rsi":
            v = int(float(val))
            ENGINE.settings.mr_rsi_max = int(clamp(v, 5, 95))
            ENGINE.persist()
            await q.edit_message_text(f"MR RSI max satt till {ENGINE.settings.mr_rsi_max}")

        elif key == "set_stake":
            v = float(val)
            ENGINE.settings.stake_usdt = clamp(v, 1.0, 100000.0)
            ENGINE.persist()
            await q.edit_message_text(f"Stake satt till {v:.2f} USDT")

        elif key == "set_tp":
            v = float(val)
            ENGINE.settings.tp_pct = clamp(v, 0.05, 50.0)
            ENGINE.persist()
            await q.edit_message_text(f"TP satt till {v:.2f}%")

        elif key == "set_sl":
            v = float(val)
            ENGINE.settings.sl_pct = clamp(v, 0.05, 50.0)
            ENGINE.persist()
            await q.edit_message_text(f"SL satt till {v:.2f}%")

        elif key == "set_trail_activate":
            v = float(val)
            ENGINE.settings.trail_activate_pct = clamp(v, 0.05, 50.0)
            ENGINE.persist()
            await q.edit_message_text(f"Trail activate satt till {v:.2f}%")

        elif key == "set_trail_dist":
            v = float(val)
            ENGINE.settings.trail_dist_pct = clamp(v, 0.05, 50.0)
            ENGINE.persist()
            await q.edit_message_text(f"Trail dist satt till {v:.2f}%")

        else:
            await q.edit_message_text("Ok")

    except Exception as e:
        await q.edit_message_text(f"Fel: {e}")

# =========================
# Main
# =========================
def main():
    global ENGINE

    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN (or TELEGRAM_BOT_TOKEN / BOT_TOKEN) in env vars.")

    kucoin = KuCoinPublic(timeout=10)
    ENGINE = Engine(kucoin)

    app = Application.builder().token(token).build()

    # basic
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))

    # engine
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))

    # ops
    app.add_handler(CommandHandler("notify", cmd_notify))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("coins", cmd_coins))

    # settings
    app.add_handler(CommandHandler("mr_band", cmd_mr_band))
    app.add_handler(CommandHandler("mr_exit", cmd_mr_exit))
    app.add_handler(CommandHandler("mr_rsi", cmd_mr_rsi))
    app.add_handler(CommandHandler("mr_trend", cmd_mr_trend))
    app.add_handler(CommandHandler("mr_trend_strict", cmd_mr_trend_strict))
    app.add_handler(CommandHandler("vwap_lookback", cmd_vwap_lookback))

    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("sl", cmd_sl))
    app.add_handler(CommandHandler("trail_activate", cmd_trail_activate))
    app.add_handler(CommandHandler("trail_dist", cmd_trail_dist))
    app.add_handler(CommandHandler("stall", cmd_stall))

    app.add_handler(CallbackQueryHandler(on_callback))

    log.info("Bot starting (MR + Momo/Swing Exit)...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
