# MpORBbot - Compression Breakout + Reclaim (Wave-3 style) - LONG only
# Single-file main.py (PART 1/2)
# KuCoin public data for signals + Telegram control
#
# Notes:
# - No profit guarantee. This is a rules-based strategy with risk controls.
# - Designed to catch "impulse start" after compression + breakout/reclaim.
#
# Env vars:
#   TELEGRAM_TOKEN (required)
# Optional (for future live trading):
#   KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE

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

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mporbbot_breakout")

# -------------------------
# Constants
# -------------------------
KUCOIN_BASE = "https://api.kucoin.com"
EXCHANGE_NAME = "KuCoin"
STRATEGY_NAME = "Compression Breakout + Reclaim (Wave-3 style)"
STRATEGY_CODE = "WAVE3_BREAKOUT"

STATE_PATH = "bot_state.json"
MOCK_LOG_PATH = "mock_trade_log.csv"
REAL_LOG_PATH = "real_trade_log.csv"

# -------------------------
# Helpers
# -------------------------
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

def fmt(x: float, n: int = 6) -> str:
    return f"{x:.{n}f}"

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

# -------------------------
# CSV Logging
# -------------------------
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

# -------------------------
# Indicators
# -------------------------
def ema(values: List[float], period: int) -> Optional[float]:
    if period <= 1 or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

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

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 2:
        return None
    trs: List[float] = []
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        pc = closes[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period

def pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

def corr(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 10:
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(len(a)):
        xa = a[i] - ma
        xb = b[i] - mb
        num += xa * xb
        da += xa * xa
        db += xb * xb
    if da <= 0 or db <= 0:
        return None
    return num / math.sqrt(da * db)

# -------------------------
# KuCoin Public Client
# -------------------------
class KuCoinPublic:
    def __init__(self, timeout: int = 12):
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
        sec_map = {
            "1min": 60, "3min": 180, "5min": 300, "15min": 900,
            "30min": 1800, "1hour": 3600, "4hour": 14400,
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
        arr = data.get("data") or []
        arr = list(reversed(arr))  # oldest -> newest
        if len(arr) > limit:
            arr = arr[-limit:]
        return arr

    def get_all_tickers(self) -> List[Dict]:
        url = f"{KUCOIN_BASE}/api/v1/market/allTickers"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        return (data.get("data") or {}).get("ticker") or []

# -------------------------
# Models
# -------------------------
@dataclass
class BotSettings:
    trade_mode: str          # mock | live (live trading not implemented here)
    engine_on: bool
    notify: bool

    # stake/risk
    stake_usdt: float
    risk_mult: float  # scales stake

    # costs (mock realism)
    fee_rate_per_side: float
    slippage_rate_per_side: float

    # mode presets
    mode: str  # active | swing | hybrid

    # universe
    universe_size: int
    universe_refresh_min: int
    vol_filter_top_n: int
    corr_window_bars: int
    include_btc: bool
    want_negative_corr: bool

    # timeframes
    tf_trend: str
    tf_entry: str

    # filters
    trend_filter: bool

    # compression / breakout / reclaim
    comp_bars: int
    comp_max_pct: float
    break_buffer_pct: float
    retest_buffer_pct: float
    watch_expire_min: int

    # confirmation
    require_atr_expand: bool
    atr_period: int
    atr_expand_ratio: float
    require_volume_expand: bool
    volume_expand_ratio: float

    # exits
    sl_atr_mult: float
    trail_activate_pct: float
    trail_dist_pct: float
    stall_bail_min: int
    stall_bail_pct: float

    # concurrency
    max_positions: int
    cooldown_per_coin_sec: int
    loop_sec: int

@dataclass
class Position:
    symbol: str
    side: str
    qty: float
    stake_usdt: float
    entry_price: float
    entry_ts: int

    sl_price: float

    trail_active: bool
    trail_highest: float
    trail_stop: float

    # context
    entry_tf: str
    box_high: float
    box_low: float
    entry_reason: str

@dataclass
class WatchState:
    # after breakout, watch for reclaim / continuation entry
    active: bool
    created_ts: int
    expire_ts: int
    box_high: float
    box_low: float
    broke_out: bool

@dataclass
class PnL:
    trades: int
    net_usdt: float

@dataclass
class CoinState:
    cooldown_until: int

# -------------------------
# Telegram UI
# -------------------------
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["/status", "/pnl"],
        ["/engine_on", "/engine_off"],
        ["/mode", "/universe"],
        ["/trend", "/refresh_universe"],
        ["/stake", "/risk"],
        ["/comp_pct", "/comp_bars"],
        ["/break_buf", "/retest_buf"],
        ["/trail_activate", "/trail_dist"],
        ["/sl_atr", "/universe_size"],
    ],
    resize_keyboard=True,
)

# -------------------------
# Engine
# -------------------------
class Wave3BreakoutEngine:
    def __init__(self, kucoin: KuCoinPublic):
        self.kucoin = kucoin

        self.settings = BotSettings(
            trade_mode="mock",
            engine_on=False,
            notify=True,

            stake_usdt=30.0,
            risk_mult=1.0,

            fee_rate_per_side=0.0010,
            slippage_rate_per_side=0.0002,

            mode="hybrid",

            universe_size=25,
            universe_refresh_min=60,
            vol_filter_top_n=45,
            corr_window_bars=64,
            include_btc=True,
            want_negative_corr=True,

            tf_trend="15min",
            tf_entry="5min",

            trend_filter=True,

            comp_bars=48,
            comp_max_pct=1.20,
            break_buffer_pct=0.05,
            retest_buffer_pct=0.10,
            watch_expire_min=240,

            require_atr_expand=True,
            atr_period=14,
            atr_expand_ratio=1.20,
            require_volume_expand=False,
            volume_expand_ratio=1.30,

            sl_atr_mult=2.2,
            trail_activate_pct=1.20,
            trail_dist_pct=0.60,
            stall_bail_min=240,
            stall_bail_pct=-0.20,

            max_positions=4,
            cooldown_per_coin_sec=45,
            loop_sec=5,
        )

        self.positions: Dict[str, Position] = {}
        self.coin_state: Dict[str, CoinState] = {}
        self.watch: Dict[str, WatchState] = {}
        self.pnl = PnL(trades=0, net_usdt=0.0)

        self.universe: List[str] = []
        self._last_universe_refresh = 0

        self._load_persisted()
        self._apply_mode_presets(self.settings.mode, persist=False)
        self.refresh_universe(force=True)

    # ---------- persistence
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
            self.universe = st.get("universe", []) or []
        except Exception as e:
            log.warning(f"Failed to load persisted state: {e}")

    def persist(self):
        save_state({
            "settings": asdict(self.settings),
            "pnl": asdict(self.pnl),
            "universe": self.universe,
        })

    # ---------- costs / qty
    def _calc_qty(self, stake_usdt: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return stake_usdt / price

    def _calc_costs_mock(self, notional_usdt: float) -> Tuple[float, float]:
        fees = notional_usdt * self.settings.fee_rate_per_side
        slip = notional_usdt * self.settings.slippage_rate_per_side
        return fees, slip

    def _apply_slippage_price(self, price: float, side: str) -> float:
        slip = self.settings.slippage_rate_per_side
        if side == "BUY":
            return price * (1 + slip)
        return price * (1 - slip)

    # ---------- market
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

    # ---------- mode presets
    def _apply_mode_presets(self, mode: str, persist: bool = True):
        mode = (mode or "").lower().strip()
        if mode not in ("active", "swing", "hybrid"):
            mode = "hybrid"

        self.settings.mode = mode

        if mode == "active":
            self.settings.tf_trend = "15min"
            self.settings.tf_entry = "3min"
            self.settings.comp_bars = 60
            self.settings.comp_max_pct = 1.00
            self.settings.break_buffer_pct = 0.05
            self.settings.retest_buffer_pct = 0.10
            self.settings.max_positions = 5
            self.settings.cooldown_per_coin_sec = 30
            self.settings.trail_activate_pct = 0.90
            self.settings.trail_dist_pct = 0.45
            self.settings.sl_atr_mult = 2.0
            self.settings.loop_sec = 4

        elif mode == "swing":
            self.settings.tf_trend = "1hour"
            self.settings.tf_entry = "15min"
            self.settings.comp_bars = 48
            self.settings.comp_max_pct = 1.60
            self.settings.break_buffer_pct = 0.08
            self.settings.retest_buffer_pct = 0.15
            self.settings.max_positions = 2
            self.settings.cooldown_per_coin_sec = 120
            self.settings.trail_activate_pct = 1.80
            self.settings.trail_dist_pct = 0.90
            self.settings.sl_atr_mult = 2.6
            self.settings.loop_sec = 8

        else:  # hybrid
            self.settings.tf_trend = "15min"
            self.settings.tf_entry = "5min"
            self.settings.comp_bars = 48
            self.settings.comp_max_pct = 1.20
            self.settings.break_buffer_pct = 0.05
            self.settings.retest_buffer_pct = 0.10
            self.settings.max_positions = 4
            self.settings.cooldown_per_coin_sec = 45
            self.settings.trail_activate_pct = 1.20
            self.settings.trail_dist_pct = 0.60
            self.settings.sl_atr_mult = 2.2
            self.settings.loop_sec = 5

        if persist:
            self.persist()

    # ---------- universe selection
    def _is_good_symbol(self, sym: str) -> bool:
        # Keep simple filters: USDT pairs, exclude leveraged and weird tokens
        if not sym.endswith("-USDT"):
            return False
        bad = ("3L-", "3S-", "UP-", "DOWN-", "BULL-", "BEAR-", "HALF-", "HEDGE-")
        for b in bad:
            if b in sym:
                return False
        return True

    def refresh_universe(self, force: bool = False) -> List[str]:
        now = ts_unix()
        if (not force) and (now - self._last_universe_refresh) < self.settings.universe_refresh_min * 60:
            return self.universe

        tickers = self.kucoin.get_all_tickers()
        # sort by 24h quote volume if available (volValue) else by vol
        candidates: List[Tuple[str, float]] = []
        for t in tickers:
            sym = (t.get("symbol") or "").upper()
            if not self._is_good_symbol(sym):
                continue
            vol_value = safe_float(t.get("volValue"), 0.0)
            vol = safe_float(t.get("vol"), 0.0)
            score = vol_value if vol_value > 0 else vol
            if score <= 0:
                continue
            candidates.append((sym, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in candidates[: max(20, self.settings.vol_filter_top_n)]]

        # always include BTC-USDT if requested
        if self.settings.include_btc and "BTC-USDT" not in top:
            top = ["BTC-USDT"] + top

        # build correlation groups vs BTC
        chosen: List[str] = []
        btc = "BTC-USDT"
        btc_rets: Optional[List[float]] = None

        try:
            btc_c = self.kucoin.get_candles(btc, self.settings.tf_trend, self.settings.corr_window_bars + 5)
            btc_closes = [safe_float(c[2]) for c in btc_c]
            btc_rets = []
            for i in range(1, len(btc_closes)):
                btc_rets.append(math.log(btc_closes[i] / btc_closes[i - 1]) if btc_closes[i - 1] > 0 else 0.0)
        except Exception as e:
            log.warning(f"Universe: failed BTC correlation base: {e}")

        scored: List[Tuple[str, float, float]] = []  # (sym, corr, volrank proxy)
        for idx, sym in enumerate(top):
            if sym == btc:
                continue
            cval = 0.0
            if btc_rets is not None:
                try:
                    cs = self.kucoin.get_candles(sym, self.settings.tf_trend, self.settings.corr_window_bars + 5)
                    closes = [safe_float(c[2]) for c in cs]
                    rets: List[float] = []
                    for i in range(1, len(closes)):
                        rets.append(math.log(closes[i] / closes[i - 1]) if closes[i - 1] > 0 else 0.0)
                    L = min(len(rets), len(btc_rets))
                    c0 = corr(rets[-L:], btc_rets[-L:])
                    if c0 is not None:
                        cval = c0
                except Exception:
                    cval = 0.0
            scored.append((sym, cval, float(len(top) - idx)))

        # groups: negative corr, low corr, positive corr
        neg = sorted([x for x in scored if x[1] <= -0.15], key=lambda x: (x[1], -x[2]))
        low = sorted([x for x in scored if abs(x[1]) < 0.20], key=lambda x: (-x[2], abs(x[1])))
        pos = sorted([x for x in scored if x[1] >= 0.50], key=lambda x: (-x[1], -x[2]))

        # mix selection
        size = int(clamp(self.settings.universe_size, 5, 120))
        if self.settings.include_btc:
            chosen.append("BTC-USDT")

        # pick from each bucket
        def take(bucket: List[Tuple[str, float, float]], n: int):
            nonlocal chosen
            for sym, cval, _ in bucket:
                if len(chosen) >= size:
                    return
                if sym not in chosen:
                    chosen.append(sym)
                if len(chosen) >= size:
                    return

        # allocate roughly: 40% pos (trend followers), 40% neg/low (diversifiers), rest from volume top
        n_pos = int(size * 0.40)
        n_div = int(size * 0.40)

        take(pos, n_pos)
        if self.settings.want_negative_corr:
            take(neg, n_div // 2)
            take(low, n_div - (n_div // 2))
        else:
            take(low, n_div)

        # fill remaining from top-volume list
        for sym in top:
            if len(chosen) >= size:
                break
            if sym not in chosen:
                chosen.append(sym)

        self.universe = chosen
        self._last_universe_refresh = now

        # init per-coin state
        for s in self.universe:
            if s not in self.coin_state:
                self.coin_state[s] = CoinState(cooldown_until=0)
            if s not in self.watch:
                self.watch[s] = WatchState(active=False, created_ts=0, expire_ts=0, box_high=0.0, box_low=0.0, broke_out=False)

        self.persist()
        return self.universe

    # ---------- coin cooldown
    def can_trade_coin(self, symbol: str) -> bool:
        cs = self.coin_state.get(symbol)
        if not cs:
            return True
        return ts_unix() >= cs.cooldown_until

    def mark_cooldown(self, symbol: str):
        cs = self.coin_state.setdefault(symbol, CoinState(cooldown_until=0))
        cs.cooldown_until = ts_unix() + int(self.settings.cooldown_per_coin_sec)

    # ---------- trend filter
    def _trend_ok(self, closes_trend: List[float]) -> bool:
        if not self.settings.trend_filter:
            return True
        e20 = ema(closes_trend, 20)
        e50 = ema(closes_trend, 50)
        if e20 is None or e50 is None:
            return False
        return e20 > e50

    # ---------- compression detection (triangle-like box)
    def _box(self, highs: List[float], lows: List[float], bars: int) -> Tuple[float, float, float]:
        b = min(len(highs), len(lows), bars)
        h = max(highs[-b:])
        l = min(lows[-b:])
        mid = (h + l) / 2.0 if (h + l) > 0 else 1.0
        height_pct = (h - l) / mid * 100.0
        return h, l, height_pct

    def _atr_expand_ok(self, atr_now: float, atr_hist: List[float]) -> bool:
        if not self.settings.require_atr_expand:
            return True
        if len(atr_hist) < 10:
            return False
        base = sum(atr_hist[-10:]) / 10.0
        if base <= 0:
            return False
        return atr_now >= base * self.settings.atr_expand_ratio

    def _volume_expand_ok(self, vol_now: float, vol_hist: List[float]) -> bool:
        if not self.settings.require_volume_expand:
            return True
        if len(vol_hist) < 10:
            return False
        base = sum(vol_hist[-10:]) / 10.0
        if base <= 0:
            return False
        return vol_now >= base * self.settings.volume_expand_ratio

    # ---------- entry logic
    def _compute_series(self, candles: List[List[str]]) -> Tuple[List[float], List[float], List[float], List[float]]:
        # candle: [time, open, close, high, low, volume, turnover]
        opens = [safe_float(c[1]) for c in candles]
        closes = [safe_float(c[2]) for c in candles]
        highs = [safe_float(c[3]) for c in candles]
        lows = [safe_float(c[4]) for c in candles]
        vols = [safe_float(c[5]) for c in candles]
        return opens, closes, highs, lows, vols

    def open_positions_count(self) -> int:
        return len(self.positions)

    def _stake_effective(self) -> float:
        return float(self.settings.stake_usdt) * float(self.settings.risk_mult)

    def _try_entries_for_symbol(self, symbol: str) -> List[str]:
        out: List[str] = []
        if symbol in self.positions:
            return out
        if self.open_positions_count() >= int(self.settings.max_positions):
            return out
        if not self.can_trade_coin(symbol):
            return out

        # pull data
        c_trend = self.kucoin.get_candles(symbol, self.settings.tf_trend, max(210, self.settings.comp_bars + 60))
        c_entry = self.kucoin.get_candles(symbol, self.settings.tf_entry, max(210, self.settings.comp_bars + 60))

        _, closes_t, highs_t, lows_t, _ = self._compute_series(c_trend)
        _, closes_e, highs_e, lows_e, vols_e = self._compute_series(c_entry)

        if len(closes_t) < 60 or len(closes_e) < (self.settings.comp_bars + 20):
            return out

        # trend filter
        if not self._trend_ok(closes_t):
            return out

        # compression box on entry TF
        box_high, box_low, box_h_pct = self._box(highs_e, lows_e, int(self.settings.comp_bars))
        if box_h_pct > float(self.settings.comp_max_pct):
            # too wide, no compression
            # if there is active watch, let it expire naturally
            return out

        # ATR and volume
        atr_now = atr(highs_e, lows_e, closes_e, int(self.settings.atr_period)) or 0.0
        atr_hist: List[float] = []
        # quick ATR history approximation over last 30 bars
        for k in range(25, 5, -1):
            a = atr(highs_e[:-k], lows_e[:-k], closes_e[:-k], int(self.settings.atr_period))
            if a is not None:
                atr_hist.append(a)

        vol_now = vols_e[-1]
        vol_hist = vols_e[:-1]

        last_close = closes_e[-1]

        break_buf = float(self.settings.break_buffer_pct) / 100.0
        retest_buf = float(self.settings.retest_buffer_pct) / 100.0

        # A) Breakout Entry (aggressive)
        breakout_level = box_high * (1.0 + break_buf)
        breakout = last_close >= breakout_level

        # Update watch state when breakout happens
        w = self.watch.get(symbol)
        if w is None:
            w = WatchState(active=False, created_ts=0, expire_ts=0, box_high=0.0, box_low=0.0, broke_out=False)
            self.watch[symbol] = w

        now = ts_unix()
        if breakout:
            # Start/refresh watch
            w.active = True
            w.created_ts = now
            w.expire_ts = now + int(self.settings.watch_expire_min) * 60
            w.box_high = box_high
            w.box_low = box_low
            w.broke_out = True

        # Entry decision:
        # - If breakout AND confirmations ok -> enter
        # - Else if watch active and price reclaimed above box_high after retest -> enter
        entered = False
        entry_reason = ""

        if breakout and self._atr_expand_ok(atr_now, atr_hist) and self._volume_expand_ok(vol_now, vol_hist):
            entered = True
            entry_reason = "BREAKOUT"
        else:
            # B) Reclaim Entry (wave-3 style)
            if w.active and w.broke_out and now <= w.expire_ts:
                # retest condition: price went near/under box_high, then reclaimed
                # we approximate: last close above (box_high * (1 + small buffer)), and recent low dipped to <= box_high*(1+retest_buf)
                recent_lows = lows_e[-8:]
                dipped_near = min(recent_lows) <= w.box_high * (1.0 + retest_buf)
                reclaimed = last_close >= w.box_high * (1.0 + break_buf)
                if dipped_near and reclaimed:
                    entered = True
                    entry_reason = "RECLAIM"
            # expire watch
            if w.active and now > w.expire_ts:
                w.active = False
                w.broke_out = False

        if not entered:
            return out

        # execute entry
        last, bid, ask = self._level1_prices(symbol)
        entry_raw = ask if ask > 0 else last
        entry_price = entry_raw
        if self.settings.trade_mode == "mock":
            entry_price = self._apply_slippage_price(entry_price, "BUY")

        stake = self._stake_effective()
        qty = self._calc_qty(stake, entry_price)
        if qty <= 0:
            return out

        # SL using ATR below entry, but also not above box_low
        sl = entry_price - (atr_now * float(self.settings.sl_atr_mult))
        # protect: place SL below box_low if that is lower (wider stop) only if reasonable
        if box_low > 0:
            sl = min(sl, box_low * (1.0 - 0.001))  # slightly under box low
        sl = max(0.0, sl)

        pos = Position(
            symbol=symbol,
            side="LONG",
            qty=qty,
            stake_usdt=stake,
            entry_price=entry_price,
            entry_ts=now,
            sl_price=sl,
            trail_active=False,
            trail_highest=entry_price,
            trail_stop=entry_price * (1.0 - float(self.settings.trail_dist_pct) / 100.0),
            entry_tf=self.settings.tf_entry,
            box_high=box_high,
            box_low=box_low,
            entry_reason=entry_reason,
        )

        self.positions[symbol] = pos
        self.mark_cooldown(symbol)
        # once in position, disable watch
        w.active = False
        w.broke_out = False
        self.persist()

        out.append(
            f"ENTRY {symbol} LONG @ {fmt(entry_price, 6)} | {entry_reason}\n"
            f"TF={self.settings.tf_entry} box={fmt(box_low, 6)}-{fmt(box_high, 6)} ({box_h_pct:.2f}%)\n"
            f"ATR={fmt(atr_now, 6)} SL={fmt(sl, 6)} stake={stake:.2f} qty={qty:.8f}"
        )
        return out

    # ---------- exits & position management
    def close_position(self, symbol: str, reason: str) -> Optional[str]:
        pos = self.positions.get(symbol)
        if not pos:
            return None

        last, bid, ask = self._level1_prices(symbol)
        exit_raw = bid if bid > 0 else last
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

        tag = "WIN" if net_pnl >= 0 else "LOSS"
        return (
            f"EXIT {symbol} @ {fmt(exit_price, 6)} | {tag}\n"
            f"Net {net_pnl:+.4f} USDT | reason={reason}\n"
            f"(gross {gross_pnl:+.4f} | fees {fees:.4f} | slip {slip_cost:.4f})"
        )

    def update_positions(self) -> List[str]:
        msgs: List[str] = []
        now = ts_unix()

        for sym, pos in list(self.positions.items()):
            last, bid, ask = self._level1_prices(sym)
            price = last

            # Hard SL
            if pos.sl_price > 0 and price <= pos.sl_price:
                m = self.close_position(sym, "SL_ATR")
                if m:
                    msgs.append(m)
                continue

            # Arm trailing after activate threshold
            gain_pct = pct_change(price, pos.entry_price)
            if (not pos.trail_active) and gain_pct >= float(self.settings.trail_activate_pct):
                pos.trail_active = True
                pos.trail_highest = price
                pos.trail_stop = price * (1.0 - float(self.settings.trail_dist_pct) / 100.0)
                msgs.append(f"TRAIL_ARM {sym} price={fmt(price,6)} stop={fmt(pos.trail_stop,6)}")
                continue

            # Trailing management
            if pos.trail_active:
                if price > pos.trail_highest:
                    pos.trail_highest = price
                    pos.trail_stop = pos.trail_highest * (1.0 - float(self.settings.trail_dist_pct) / 100.0)

                if price <= pos.trail_stop:
                    m = self.close_position(sym, "TRAIL_STOP")
                    if m:
                        msgs.append(m)
                    continue

            # Stall bailout: if too long and not moving
            age_min = (now - pos.entry_ts) / 60.0
            if age_min >= float(self.settings.stall_bail_min):
                if gain_pct <= float(self.settings.stall_bail_pct):
                    m = self.close_position(sym, "STALL_BAIL")
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

        # refresh universe occasionally
        try:
            self.refresh_universe(force=False)
        except Exception as e:
            log.warning(f"Universe refresh error: {e}")

        # entries
        for sym in list(self.universe):
            if sym in self.positions:
                continue
            if self.open_positions_count() >= int(self.settings.max_positions):
                break
            try:
                out.extend(self._try_entries_for_symbol(sym))
            except Exception as e:
                log.warning(f"Entry error {sym}: {e}")

        return out

    # ---------- status
    def status_text(self) -> str:
        s = self.settings
        open_pos = list(self.positions.keys())
        u = self.universe[:]
        if len(u) > 30:
            u = u[:30] + ["..."]

        return (
            f"<b>ENGINE:</b> {'ON' if s.engine_on else 'OFF'}\n"
            f"<b>Strategy:</b> {STRATEGY_NAME} ({STRATEGY_CODE})\n"
            f"<b>Mode:</b> {s.mode}\n"
            f"<b>Trade mode:</b> {s.trade_mode}\n"
            f"<b>Trend TF:</b> {s.tf_trend} | <b>Entry TF:</b> {s.tf_entry}\n"
            f"<b>Trend filter:</b> {'ON' if s.trend_filter else 'OFF'}\n\n"
            f"<b>Compression:</b> bars={s.comp_bars} | max_box={fmt_pct(s.comp_max_pct)}\n"
            f"<b>Buffers:</b> break={fmt_pct(s.break_buffer_pct)} | retest={fmt_pct(s.retest_buffer_pct)}\n"
            f"<b>Confirm:</b> ATR_expand={'ON' if s.require_atr_expand else 'OFF'}\n"
            f"<b>Stops:</b> SL_ATR x{s.sl_atr_mult:.2f} | trail_on={fmt_pct(s.trail_activate_pct)} | dist={fmt_pct(s.trail_dist_pct)}\n\n"
            f"<b>Stake:</b> {s.stake_usdt:.2f} USDT | <b>Risk mult:</b> {s.risk_mult:.2f}\n"
            f"<b>Max pos:</b> {s.max_positions} | <b>Cooldown:</b> {s.cooldown_per_coin_sec}s\n"
            f"<b>Fees:</b> {s.fee_rate_per_side*100:.2f}%/side | <b>Slip:</b> {s.slippage_rate_per_side*100:.2f}%/side\n\n"
            f"<b>Trades:</b> {self.pnl.trades}\n"
            f"<b>Total NET PnL:</b> {self.pnl.net_usdt:+.4f} USDT\n"
            f"<b>Universe:</b> size={len(self.universe)} | refresh={s.universe_refresh_min}m\n"
            f"<b>Coins:</b> {u}\n"
            f"<b>Open positions:</b> {open_pos if open_pos else 'none'}\n"
            f"<b>Notify:</b> {'ON' if s.notify else 'OFF'}\n"
        )

# Global engine
ENGINE: Optional[Wave3BreakoutEngine] = None
# -------------------------
# Telegram Handlers
# -------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id

    await update.message.reply_text(
        "MpORBbot running.\n"
        "Strategy: Compression Breakout + Reclaim (wave-3 style), LONG only.\n\n"
        "Tips:\n"
        "/mode hybrid|active|swing\n"
        "/universe_size 30\n"
        "/trend on|off\n"
        "/comp_pct 1.20\n"
        "/comp_bars 48\n"
        "/break_buf 0.05\n"
        "/retest_buf 0.10\n"
        "/trail_activate 1.20\n"
        "/trail_dist 0.60\n",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id
    if ENGINE is None:
        await update.message.reply_text("Engine not ready.", reply_markup=MAIN_KEYBOARD)
        return
    await update.message.reply_text(ENGINE.status_text(), parse_mode=ParseMode.HTML, reply_markup=MAIN_KEYBOARD)

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ENGINE is None:
        await update.message.reply_text("Engine not ready.", reply_markup=MAIN_KEYBOARD)
        return
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

async def cmd_universe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ENGINE is None:
        await update.message.reply_text("Engine not ready.", reply_markup=MAIN_KEYBOARD)
        return
    u = ENGINE.universe
    if not u:
        await update.message.reply_text("Universe empty (try /refresh_universe).", reply_markup=MAIN_KEYBOARD)
        return
    # show up to 60
    show = u[:60]
    extra = "" if len(u) <= 60 else f"\n... (+{len(u)-60} till)"
    await update.message.reply_text("Coins igång:\n" + ", ".join(show) + extra, reply_markup=MAIN_KEYBOARD)

async def cmd_refresh_universe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ENGINE.refresh_universe(force=True)
        await update.message.reply_text(f"Universe refreshed. size={len(ENGINE.universe)}", reply_markup=MAIN_KEYBOARD)
    except Exception as e:
        await update.message.reply_text(f"Universe refresh failed: {e}", reply_markup=MAIN_KEYBOARD)

async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(f"Mode nu: {ENGINE.settings.mode}\nSkriv: /mode active|hybrid|swing", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE._apply_mode_presets(args[0], persist=True)
    await update.message.reply_text(f"Mode satt till: {ENGINE.settings.mode}", reply_markup=MAIN_KEYBOARD)

async def cmd_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        ENGINE.settings.trend_filter = not ENGINE.settings.trend_filter
        ENGINE.persist()
        await update.message.reply_text(f"Trend filter: {'ON' if ENGINE.settings.trend_filter else 'OFF'}", reply_markup=MAIN_KEYBOARD)
        return
    v = args[0].lower().strip()
    ENGINE.settings.trend_filter = (v in ("on", "1", "true", "yes"))
    ENGINE.persist()
    await update.message.reply_text(f"Trend filter: {'ON' if ENGINE.settings.trend_filter else 'OFF'}", reply_markup=MAIN_KEYBOARD)

# ---- numeric setters (allow typed values)
async def _set_float(update: Update, label: str, attr: str, lo: float, hi: float):
    args = update.message.text.split()
    if len(args) < 2:
        cur = getattr(ENGINE.settings, attr)
        await update.message.reply_text(f"{label} nu: {cur}", reply_markup=MAIN_KEYBOARD)
        return
    v = safe_float(args[1], None)
    if v is None:
        await update.message.reply_text("Ogiltigt nummer.", reply_markup=MAIN_KEYBOARD)
        return
    v = clamp(float(v), lo, hi)
    setattr(ENGINE.settings, attr, v)
    ENGINE.persist()
    await update.message.reply_text(f"{label} satt till {v}", reply_markup=MAIN_KEYBOARD)

async def _set_int(update: Update, label: str, attr: str, lo: int, hi: int):
    args = update.message.text.split()
    if len(args) < 2:
        cur = getattr(ENGINE.settings, attr)
        await update.message.reply_text(f"{label} nu: {cur}", reply_markup=MAIN_KEYBOARD)
        return
    try:
        v = int(float(args[1]))
    except Exception:
        await update.message.reply_text("Ogiltigt nummer.", reply_markup=MAIN_KEYBOARD)
        return
    v = int(clamp(float(v), lo, hi))
    setattr(ENGINE.settings, attr, v)
    ENGINE.persist()
    await update.message.reply_text(f"{label} satt till {v}", reply_markup=MAIN_KEYBOARD)

async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_float(update, "Stake (USDT)", "stake_usdt", 1.0, 100000.0)

async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_float(update, "Risk mult", "risk_mult", 0.10, 5.0)

async def cmd_universe_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_int(update, "Universe size", "universe_size", 5, 120)
    # refresh after size change
    try:
        ENGINE.refresh_universe(force=True)
    except Exception:
        pass

async def cmd_comp_bars(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_int(update, "Compression bars", "comp_bars", 20, 200)

async def cmd_comp_pct(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_float(update, "Compression max pct", "comp_max_pct", 0.30, 8.0)

async def cmd_break_buf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_float(update, "Break buffer pct", "break_buffer_pct", 0.00, 2.0)

async def cmd_retest_buf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_float(update, "Retest buffer pct", "retest_buffer_pct", 0.00, 3.0)

async def cmd_trail_activate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_float(update, "Trail activate pct", "trail_activate_pct", 0.05, 10.0)

async def cmd_trail_dist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_float(update, "Trail dist pct", "trail_dist_pct", 0.05, 15.0)

async def cmd_sl_atr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _set_float(update, "SL ATR mult", "sl_atr_mult", 0.8, 6.0)

# -------------------------
# Background engine loop
# -------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if ENGINE is None:
                await asyncio.sleep(2)
                continue

            # run blocking step in executor
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

        await asyncio.sleep(int(ENGINE.settings.loop_sec))

async def post_init(app: Application):
    # runs inside a running event loop in python-telegram-bot v20+
    app.create_task(engine_loop(app))

# -------------------------
# Main
# -------------------------
def main():
    global ENGINE
    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN (or BOT_TOKEN) in environment variables.")

    kucoin = KuCoinPublic(timeout=12)
    ENGINE = Wave3BreakoutEngine(kucoin)

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))

    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("universe", cmd_universe))
    app.add_handler(CommandHandler("refresh_universe", cmd_refresh_universe))
    app.add_handler(CommandHandler("universe_size", cmd_universe_size))

    app.add_handler(CommandHandler("trend", cmd_trend))
    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("risk", cmd_risk))

    app.add_handler(CommandHandler("comp_bars", cmd_comp_bars))
    app.add_handler(CommandHandler("comp_pct", cmd_comp_pct))
    app.add_handler(CommandHandler("break_buf", cmd_break_buf))
    app.add_handler(CommandHandler("retest_buf", cmd_retest_buf))

    app.add_handler(CommandHandler("trail_activate", cmd_trail_activate))
    app.add_handler(CommandHandler("trail_dist", cmd_trail_dist))
    app.add_handler(CommandHandler("sl_atr", cmd_sl_atr))

    log.info("Bot starting...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
