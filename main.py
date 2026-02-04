# Mp ORBbot — Mean Reversion (VWAP + RSI) — LONG only
# Single file main.py (paste PART 1/2 then PART 2/2 into the SAME file)

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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mp_orbbot_mr")

# -------------------------
# Strategy / Exchange
# -------------------------
STRATEGY_NAME = "Mean Reversion (VWAP+RSI)"
STRATEGY_CODE = "MEAN_REVERSION"
EXCHANGE_NAME = "KuCoin"
KUCOIN_BASE = "https://api.kucoin.com"

# -------------------------
# Defaults
# -------------------------
DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

TF_TREND = "5min"   # KuCoin API ktype
TF_ENTRY = "1min"

TREND_CANDLES = 210
ENTRY_CANDLES = 210

ENGINE_LOOP_SEC = 3
COOLDOWN_PER_COIN_SEC = 30
MAX_POSITIONS = 5

DEFAULT_FEE_RATE_PER_SIDE = 0.0010       # 0.10% per side
DEFAULT_SLIPPAGE_RATE_PER_SIDE = 0.0002  # 0.02% per side
DEFAULT_STAKE_USDT = 30.0

# MR settings (percent)
DEFAULT_MR_BAND_PCT = 0.60          # entry if price <= VWAP - band%
DEFAULT_MR_EXIT_PCT = 0.05          # exit when price >= VWAP - exit%
DEFAULT_MR_RSI_MAX = 40             # entry requires RSI <= this
DEFAULT_MR_VWAP_LOOKBACK_MIN = 60   # VWAP computed over last N minutes (1m candles)
DEFAULT_MR_TREND_FILTER = True      # optional trend filter
DEFAULT_MR_TREND_STRICT = False     # strict filter adds slope+close>EMA

# Risk / exits
DEFAULT_SL_PCT = 0.35

# "TP (arm trail)" means: when move reaches TP_PCT, trailing is armed.
DEFAULT_TP_PCT = 0.60
DEFAULT_TRAIL_DIST_PCT = 0.20

# Files
MOCK_LOG_PATH = "mock_trade_log.csv"
REAL_LOG_PATH = "real_trade_log.csv"
STATE_PATH = "bot_state.json"

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

def parse_pct_token(tok: str) -> float:
    """
    Accepts: "0.8", "0.8%", "1", "1.00%"
    Returns numeric percent as float.
    """
    t = tok.strip().replace("%", "")
    return float(t)

def parse_onoff(tok: str) -> bool:
    t = tok.strip().lower()
    return t in ("1", "true", "yes", "on", "ja")

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

def vwap(candles: List[List[str]]) -> Optional[float]:
    """
    candles (KuCoin): [time, open, close, high, low, volume, turnover] as strings
    VWAP approx using typical_price * volume.
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

def slope_pct(values: List[float], lookback: int = 6) -> float:
    if len(values) < lookback + 1:
        return 0.0
    a = values[-lookback - 1]
    b = values[-1]
    if a == 0:
        return 0.0
    return (b - a) / a * 100.0

# -------------------------
# KuCoin public client (blocking requests)
# -------------------------
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
        arr = list(reversed(arr))  # ASC oldest->newest
        if len(arr) > limit:
            arr = arr[-limit:]
        return arr

# -------------------------
# Models
# -------------------------
@dataclass
class BotSettings:
    coins: List[str]
    trade_mode: str          # mock | live (live not implemented here)
    engine_on: bool
    stake_usdt: float

    tp_pct: float                  # arms trailing when reached
    sl_pct: float                  # hard stop
    trail_dist_pct: float          # trailing distance

    fee_rate_per_side: float
    slippage_rate_per_side: float

    max_positions: int
    cooldown_per_coin_sec: int
    notify: bool

    # Mean Reversion params
    mr_band_pct: float
    mr_exit_pct: float
    mr_rsi_max: int
    mr_vwap_lookback_min: int
    mr_trend_filter: bool
    mr_trend_strict: bool

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

    # MR context (for debug)
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

# -------------------------
# CSV logging
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
# State persistence
# -------------------------
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
# Engine: Mean Reversion (VWAP + RSI)
# -------------------------
class MeanReversionEngine:
    def __init__(self, kucoin: KuCoinPublic):
        self.kucoin = kucoin

        self.settings = BotSettings(
            coins=DEFAULT_COINS.copy(),
            trade_mode="mock",
            engine_on=False,
            stake_usdt=DEFAULT_STAKE_USDT,

            tp_pct=DEFAULT_TP_PCT,
            sl_pct=DEFAULT_SL_PCT,
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

    # Trend filter (optional)
    def _trend_ok_5m(self, closes_5m: List[float]) -> bool:
        if not self.settings.mr_trend_filter:
            return True

        e20 = ema(closes_5m, 20)
        e50 = ema(closes_5m, 50)
        if e20 is None or e50 is None:
            return False

        ok = (e20 > e50)

        if not self.settings.mr_trend_strict:
            return ok

        # strict: also require positive EMA20 slope and last close above EMA20
        e20_series_approx = closes_5m[-60:] if len(closes_5m) >= 60 else closes_5m[:]
        e20_now = ema(e20_series_approx, 20)
        if e20_now is None:
            return False

        slope = slope_pct(e20_series_approx, lookback=6)
        last_close = closes_5m[-1]
        return ok and (slope > 0) and (last_close > e20_now)

    # MR Signal
    def _mr_signal(
        self,
        candles_1m: List[List[str]],
        closes_1m: List[float],
        closes_5m: List[float],
    ) -> Tuple[bool, float, float, float]:
        """
        Returns: (signal, vwap_price, rsi_val, dev_pct_under_vwap)
        dev_pct = (price - vwap)/vwap*100 (negative means under VWAP)
        """
        if len(closes_1m) < 30:
            return (False, 0.0, 50.0, 0.0)

        look = max(10, int(self.settings.mr_vwap_lookback_min))
        candles_slice = candles_1m[-look:] if len(candles_1m) >= look else candles_1m
        vw = vwap(candles_slice)
        if vw is None or vw <= 0:
            return (False, 0.0, 50.0, 0.0)

        price = closes_1m[-1]
        r = rsi(closes_1m, 14)
        if r is None:
            return (False, vw, 50.0, 0.0)

        dev = (price - vw) / vw * 100.0  # negative under VWAP

        # Trend filter
        if not self._trend_ok_5m(closes_5m):
            return (False, vw, r, dev)

        # Entry condition
        if dev <= -self.settings.mr_band_pct and r <= float(self.settings.mr_rsi_max):
            return (True, vw, r, dev)

        return (False, vw, r, dev)

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

        signal, vw, r, dev = self._mr_signal(candles_1m, closes_1m, closes_5m)
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

        pos = Position(
            symbol=symbol,
            side="LONG",
            qty=qty,
            stake_usdt=self.settings.stake_usdt,
            entry_price=entry_price,
            entry_ts=ts_unix(),
            sl_price=sl_price,
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
            f"VWAP={vw:.6f} dev={dev:.2f}% RSI={r:.1f}\n"
            f"stake={pos.stake_usdt:.2f} qty={qty:.8f} SL={sl_price:.6f}"
        )
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

        return (
            f"EXIT {symbol} @ {exit_price:.6f}\n"
            f"net {net_pnl:+.4f} USDT reason={reason}\n"
            f"(gross {gross_pnl:+.4f} fees {fees:.4f} slip {slip_cost:.4f})"
        )

    def update_positions(self) -> List[str]:
        msgs: List[str] = []
        s = self.settings

        for sym, pos in list(self.positions.items()):
            last, bid, ask = self._level1_prices(sym)
            price = last

            # Hard SL
            if price <= pos.sl_price:
                m = self.close_position(sym, "SL")
                if m:
                    msgs.append(m)
                continue

            # MR Exit: reclaim VWAP (within exit band or above)
            # exit when dev >= -mr_exit_pct
            dev = (price - pos.entry_vwap) / pos.entry_vwap * 100.0 if pos.entry_vwap > 0 else 0.0
            if dev >= -s.mr_exit_pct:
                m = self.close_position(sym, "MR_EXIT_VWAP_RECLAIM")
                if m:
                    msgs.append(m)
                continue

            # Arm trailing when move >= TP_PCT (TP is "arm trailing")
            move_pct = (price - pos.entry_price) / pos.entry_price * 100.0 if pos.entry_price > 0 else 0.0
            if (not pos.trail_active) and (move_pct >= s.tp_pct):
                pos.trail_active = True
                pos.trail_highest = price
                pos.trail_stop = price * (1 - s.trail_dist_pct / 100.0)
                msgs.append(f"TRAIL ARMED {sym} move={move_pct:.2f}% stop={pos.trail_stop:.6f}")
                continue

            # Trailing management
            if pos.trail_active:
                if price > pos.trail_highest:
                    pos.trail_highest = price
                    pos.trail_stop = pos.trail_highest * (1 - s.trail_dist_pct / 100.0)

                if price <= pos.trail_stop:
                    m = self.close_position(sym, "TRAIL_STOP")
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
        live_ok = "OK" if (os.getenv("KUCOIN_KEY") or os.getenv("KUCOIN_API_KEY")) else "missing"

        return (
            f"<b>ENGINE:</b> {'ON' if s.engine_on else 'OFF'}\n"
            f"<b>Strategy:</b> {STRATEGY_NAME} ({STRATEGY_CODE})\n"
            f"<b>Trade mode:</b> {s.trade_mode} (live keys: {live_ok})\n"
            f"<b>Timeframes:</b> trend={TF_TREND} | entry={TF_ENTRY}\n\n"
            f"<b>TP (arm trail)</b> (/tp): {fmt_pct(s.tp_pct)}\n"
            f"<b>SL</b> (/sl): {fmt_pct(s.sl_pct)}\n"
            f"<b>Trail</b> (/trail_dist): dist {fmt_pct(s.trail_dist_pct)}\n\n"
            f"<b>Stake</b> (/stake): {s.stake_usdt:.2f} USDT\n"
            f"<b>Cooldown/coin:</b> {s.cooldown_per_coin_sec}s | <b>Max positions:</b> {s.max_positions}\n"
            f"<b>Fees:</b> {s.fee_rate_per_side*100:.2f}%/side | <b>Slippage:</b> {s.slippage_rate_per_side*100:.2f}%/side\n\n"
            f"<b>MR band</b> (/mr_band): {fmt_pct(s.mr_band_pct)} under VWAP\n"
            f"<b>MR exit</b> (/mr_exit): {fmt_pct(s.mr_exit_pct)} under VWAP\n"
            f"<b>MR RSI max</b> (/mr_rsi): {s.mr_rsi_max}\n"
            f"<b>MR VWAP lookback:</b> {s.mr_vwap_lookback_min} min\n"
            f"<b>MR Trend filter</b> (/mr_trend): {'ON' if s.mr_trend_filter else 'OFF'}\n"
            f"<b>MR Trend strict</b> (/mr_trend_strict): {'ON' if s.mr_trend_strict else 'OFF'}\n\n"
            f"<b>Trades:</b> {self.pnl.trades}\n"
            f"<b>Total NET PnL:</b> {self.pnl.net_usdt:+.4f} USDT\n"
            f"<b>Coins ({len(s.coins)}):</b> {s.coins}\n"
            f"<b>Open positions:</b> {open_pos if open_pos else 'none'}\n"
            f"<b>Notify:</b> {'ON' if s.notify else 'OFF'}\n"
        )

# -------------------------
# Telegram UI
# -------------------------
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["/status", "/pnl"],
        ["/engine_on", "/engine_off"],
        ["/mr_band", "/mr_exit"],
        ["/mr_rsi", "/stake"],
        ["/tp", "/sl"],
        ["/trail_dist", "/trail_activate"],
        ["/mr_trend", "/mr_trend_strict"],
        ["/coins", "/notify"],
        ["/close_all", "/reset_pnl"],
        ["/export_csv"],
    ],
    resize_keyboard=True,
)

def preset_keyboard(kind: str) -> InlineKeyboardMarkup:
    if kind == "mr_band":
        opts = [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50, 2.00, 3.00]
        rows, row = [], []
        for v in opts:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_mr_band:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "mr_exit":
        opts = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
        rows, row = [], []
        for v in opts:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_mr_exit:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "mr_rsi":
        opts = [20, 25, 30, 35, 40, 45, 50]
        rows, row = [], []
        for v in opts:
            row.append(InlineKeyboardButton(f"{v}", callback_data=f"set_mr_rsi:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "stake":
        opts = [10, 20, 30, 50, 75, 100]
        rows, row = [], []
        for v in opts:
            row.append(InlineKeyboardButton(f"{v} USDT", callback_data=f"set_stake:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "tp":
        opts = [0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00]
        rows, row = [], []
        for v in opts:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_tp:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "sl":
        opts = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 1.00, 1.50]
        rows, row = [], []
        for v in opts:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_sl:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    if kind == "trail_dist":
        opts = [0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
        rows, row = [], []
        for v in opts:
            row.append(InlineKeyboardButton(f"{v:.2f}%", callback_data=f"set_trail_dist:{v}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(rows)

    return InlineKeyboardMarkup([])

# -------------------------
# Telegram Handlers
# -------------------------
ENGINE: Optional[MeanReversionEngine] = None

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id

    await update.message.reply_text(
        "Mp ORBbot (Mean Reversion VWAP+RSI) LONG only.\n\n"
        "You can set custom values:\n"
        "/mr_band 0.80\n"
        "/mr_exit 0.20\n"
        "/mr_rsi 30\n"
        "/tp 1.20\n"
        "/sl 0.70\n"
        "/trail_dist 0.60\n"
        "/mr_trend off\n"
        "/mr_trend_strict on\n\n"
        "Use the buttons below too.",
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
            m = ENGINE.close_position(sym, "MANUAL_CLOSE_ALL")
            if m:
                msgs.append(m)
        except Exception as e:
            msgs.append(f"Close error {sym}: {e}")
    if not msgs:
        msgs = ["No open positions."]
    for m in msgs:
        await update.message.reply_text(m, reply_markup=MAIN_KEYBOARD)

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    path = MOCK_LOG_PATH if ENGINE.settings.trade_mode == "mock" else REAL_LOG_PATH
    ensure_csv(path)
    try:
        await update.message.reply_document(
            document=open(path, "rb"),
            filename=os.path.basename(path),
            caption="Trade log CSV.",
        )
    except Exception as e:
        await update.message.reply_text(f"Could not send CSV: {e}", reply_markup=MAIN_KEYBOARD)

async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage:\n/coins BTC-USDT ETH-USDT SOL-USDT\n\n"
            f"Current: {ENGINE.settings.coins}",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.set_coins(args)
    await update.message.reply_text(f"Coins updated ({len(ENGINE.settings.coins)}): {ENGINE.settings.coins}", reply_markup=MAIN_KEYBOARD)

# ---- Settings commands: accept value OR show presets

async def cmd_mr_band(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            v = parse_pct_token(context.args[0])
            ENGINE.settings.mr_band_pct = clamp(v, 0.01, 10.0)
            ENGINE.persist()
            await update.message.reply_text(f"MR band set to {ENGINE.settings.mr_band_pct:.2f}%", reply_markup=MAIN_KEYBOARD)
            return
        except Exception:
            pass
    await update.message.reply_text(
        f"MR band current: {fmt_pct(ENGINE.settings.mr_band_pct)} under VWAP\nPick preset or type: /mr_band 1.50",
        reply_markup=preset_keyboard("mr_band"),
    )

async def cmd_mr_exit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            v = parse_pct_token(context.args[0])
            ENGINE.settings.mr_exit_pct = clamp(v, 0.00, 10.0)
            ENGINE.persist()
            await update.message.reply_text(f"MR exit set to {ENGINE.settings.mr_exit_pct:.2f}%", reply_markup=MAIN_KEYBOARD)
            return
        except Exception:
            pass
    await update.message.reply_text(
        f"MR exit current: {fmt_pct(ENGINE.settings.mr_exit_pct)} under VWAP\nPick preset or type: /mr_exit 0.20",
        reply_markup=preset_keyboard("mr_exit"),
    )

async def cmd_mr_rsi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            v = int(float(context.args[0]))
            ENGINE.settings.mr_rsi_max = int(clamp(v, 5, 95))
            ENGINE.persist()
            await update.message.reply_text(f"MR RSI max set to {ENGINE.settings.mr_rsi_max}", reply_markup=MAIN_KEYBOARD)
            return
        except Exception:
            pass
    await update.message.reply_text(
        f"MR RSI max current: {ENGINE.settings.mr_rsi_max}\nPick preset or type: /mr_rsi 30",
        reply_markup=preset_keyboard("mr_rsi"),
    )

async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            v = float(context.args[0])
            ENGINE.settings.stake_usdt = clamp(v, 1.0, 100000.0)
            ENGINE.persist()
            await update.message.reply_text(f"Stake set to {ENGINE.settings.stake_usdt:.2f} USDT", reply_markup=MAIN_KEYBOARD)
            return
        except Exception:
            pass
    await update.message.reply_text(
        f"Stake current: {ENGINE.settings.stake_usdt:.2f} USDT\nPick preset or type: /stake 30",
        reply_markup=preset_keyboard("stake"),
    )

async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            v = parse_pct_token(context.args[0])
            ENGINE.settings.tp_pct = clamp(v, 0.05, 25.0)
            ENGINE.persist()
            await update.message.reply_text(f"TP (arm trail) set to {ENGINE.settings.tp_pct:.2f}%", reply_markup=MAIN_KEYBOARD)
            return
        except Exception:
            pass
    await update.message.reply_text(
        f"TP (arm trail) current: {fmt_pct(ENGINE.settings.tp_pct)}\nPick preset or type: /tp 1.20",
        reply_markup=preset_keyboard("tp"),
    )

async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            v = parse_pct_token(context.args[0])
            ENGINE.settings.sl_pct = clamp(v, 0.05, 25.0)
            ENGINE.persist()
            await update.message.reply_text(f"SL set to {ENGINE.settings.sl_pct:.2f}%", reply_markup=MAIN_KEYBOARD)
            return
        except Exception:
            pass
    await update.message.reply_text(
        f"SL current: {fmt_pct(ENGINE.settings.sl_pct)}\nPick preset or type: /sl 0.70",
        reply_markup=preset_keyboard("sl"),
    )

async def cmd_trail_dist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            v = parse_pct_token(context.args[0])
            ENGINE.settings.trail_dist_pct = clamp(v, 0.05, 25.0)
            ENGINE.persist()
            await update.message.reply_text(f"Trail distance set to {ENGINE.settings.trail_dist_pct:.2f}%", reply_markup=MAIN_KEYBOARD)
            return
        except Exception:
            pass
    await update.message.reply_text(
        f"Trail distance current: {fmt_pct(ENGINE.settings.trail_dist_pct)}\nPick preset or type: /trail_dist 0.60",
        reply_markup=preset_keyboard("trail_dist"),
    )

# /trail_activate is an alias for /tp (arm-trail level), because your UI expects it.
async def cmd_trail_activate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            v = parse_pct_token(context.args[0])
            ENGINE.settings.tp_pct = clamp(v, 0.05, 25.0)
            ENGINE.persist()
            await update.message.reply_text(f"Trail activate (arm) set to {ENGINE.settings.tp_pct:.2f}%", reply_markup=MAIN_KEYBOARD)
            return
        except Exception:
            pass
    await update.message.reply_text(
        f"Trail activate (arm) current: {fmt_pct(ENGINE.settings.tp_pct)}\nType: /trail_activate 1.50",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_mr_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        ENGINE.settings.mr_trend_filter = parse_onoff(context.args[0])
        ENGINE.persist()
        await update.message.reply_text(f"MR trend filter: {'ON' if ENGINE.settings.mr_trend_filter else 'OFF'}", reply_markup=MAIN_KEYBOARD)
        return
    await update.message.reply_text(
        f"MR trend filter is {'ON' if ENGINE.settings.mr_trend_filter else 'OFF'}\nType: /mr_trend on  OR  /mr_trend off",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_mr_trend_strict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        ENGINE.settings.mr_trend_strict = parse_onoff(context.args[0])
        ENGINE.persist()
        await update.message.reply_text(f"MR trend strict: {'ON' if ENGINE.settings.mr_trend_strict else 'OFF'}", reply_markup=MAIN_KEYBOARD)
        return
    await update.message.reply_text(
        f"MR trend strict is {'ON' if ENGINE.settings.mr_trend_strict else 'OFF'}\nType: /mr_trend_strict on  OR  /mr_trend_strict off",
        reply_markup=MAIN_KEYBOARD,
    )

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = (q.data or "").strip()

    try:
        if data.startswith("set_mr_band:"):
            v = float(data.split(":")[1])
            ENGINE.settings.mr_band_pct = clamp(v, 0.01, 10.0)
            ENGINE.persist()
            await q.edit_message_text(f"MR band set to {ENGINE.settings.mr_band_pct:.2f}%")

        elif data.startswith("set_mr_exit:"):
            v = float(data.split(":")[1])
            ENGINE.settings.mr_exit_pct = clamp(v, 0.0, 10.0)
            ENGINE.persist()
            await q.edit_message_text(f"MR exit set to {ENGINE.settings.mr_exit_pct:.2f}%")

        elif data.startswith("set_mr_rsi:"):
            v = int(float(data.split(":")[1]))
            ENGINE.settings.mr_rsi_max = int(clamp(v, 5, 95))
            ENGINE.persist()
            await q.edit_message_text(f"MR RSI max set to {ENGINE.settings.mr_rsi_max}")

        elif data.startswith("set_stake:"):
            v = float(data.split(":")[1])
            ENGINE.settings.stake_usdt = clamp(v, 1.0, 100000.0)
            ENGINE.persist()
            await q.edit_message_text(f"Stake set to {ENGINE.settings.stake_usdt:.2f} USDT")

        elif data.startswith("set_tp:"):
            v = float(data.split(":")[1])
            ENGINE.settings.tp_pct = clamp(v, 0.05, 25.0)
            ENGINE.persist()
            await q.edit_message_text(f"TP (arm trail) set to {ENGINE.settings.tp_pct:.2f}%")

        elif data.startswith("set_sl:"):
            v = float(data.split(":")[1])
            ENGINE.settings.sl_pct = clamp(v, 0.05, 25.0)
            ENGINE.persist()
            await q.edit_message_text(f"SL set to {ENGINE.settings.sl_pct:.2f}%")

        elif data.startswith("set_trail_dist:"):
            v = float(data.split(":")[1])
            ENGINE.settings.trail_dist_pct = clamp(v, 0.05, 25.0)
            ENGINE.persist()
            await q.edit_message_text(f"Trail dist set to {ENGINE.settings.trail_dist_pct:.2f}%")

    except Exception as e:
        await q.edit_message_text(f"Error: {e}")

# -------------------------
# Engine Loop (JobQueue tick)
# -------------------------
async def engine_tick(context: ContextTypes.DEFAULT_TYPE):
    """
    Called by PTB JobQueue. Uses thread executor because requests is blocking.
    """
    if ENGINE is None:
        return

    loop = asyncio.get_running_loop()
    try:
        msgs: List[str] = await loop.run_in_executor(None, ENGINE.step)
    except Exception as e:
        log.error(f"ENGINE step error: {e}")
        return

    if msgs and ENGINE.settings.notify:
        chat_id = context.application.bot_data.get("chat_id")
        if chat_id:
            for m in msgs:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=m)
                except Exception as e:
                    log.warning(f"Notify failed: {e}")

async def post_init(app: Application):
    # Start repeating job safely (no "no running event loop")
    app.job_queue.run_repeating(engine_tick, interval=ENGINE_LOOP_SEC, first=2)

# -------------------------
# Main
# -------------------------
def build_app(token: str) -> Application:
    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))

    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))

    app.add_handler(CommandHandler("notify", cmd_notify))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("coins", cmd_coins))

    app.add_handler(CommandHandler("mr_band", cmd_mr_band))
    app.add_handler(CommandHandler("mr_exit", cmd_mr_exit))
    app.add_handler(CommandHandler("mr_rsi", cmd_mr_rsi))
    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("sl", cmd_sl))

    app.add_handler(CommandHandler("trail_dist", cmd_trail_dist))
    app.add_handler(CommandHandler("trail_activate", cmd_trail_activate))

    app.add_handler(CommandHandler("mr_trend", cmd_mr_trend))
    app.add_handler(CommandHandler("mr_trend_strict", cmd_mr_trend_strict))

    app.add_handler(CallbackQueryHandler(on_callback))
    return app

def main():
    global ENGINE

    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN (or BOT_TOKEN / TELEGRAM_BOT_TOKEN) in env vars.")

    kucoin = KuCoinPublic(timeout=10)
    ENGINE = MeanReversionEngine(kucoin)

    app = build_app(token)
    log.info("Bot starting (Mean Reversion only)...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
