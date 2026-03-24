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
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mporbbot_v2_pro")

KUCOIN_BASE = "https://api.kucoin.com"
EXCHANGE_NAME = "KuCoin"
STRATEGY_NAME = "Sweep Reclaim Breakout V2 PRO"
STRATEGY_CODE = "SWEEP_RECLAIM_BREAK_V2"

STATE_PATH = "bot_state.json"
MOCK_LOG_PATH = "mock_trade_log.csv"
REAL_LOG_PATH = "real_trade_log.csv"

DEFAULT_COINS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT",
    "XRP-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT",
    "DOGE-USDT", "LTC-USDT", "APT-USDT", "NEAR-USDT",
]

TF_TREND = "1hour"
TF_SETUP = "15min"
TF_TRIGGER = "5min"

TREND_CANDLES = 260
SETUP_CANDLES = 220
TRIGGER_CANDLES = 260

ENGINE_LOOP_SEC = 5

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
        log.warning(f"save_state failed: {e}")


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


def atr_from_ohlc(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 2:
        return None
    trs = []
    for i in range(1, len(closes)):
        trs.append(true_range(highs[i], lows[i], closes[i - 1]))
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def find_last_pivot_low(lows: List[float], left: int, right: int, max_age_bars: int) -> Optional[Tuple[int, float]]:
    n = len(lows)
    if n < left + right + 5:
        return None
    end = n - right - 1
    start = max(left, n - max_age_bars)
    for i in range(end, start - 1, -1):
        p = lows[i]
        ok = True
        for j in range(i - left, i):
            if lows[j] <= p:
                ok = False
                break
        if not ok:
            continue
        for j in range(i + 1, i + right + 1):
            if lows[j] <= p:
                ok = False
                break
        if ok:
            return i, p
    return None


class KuCoinPublic:
    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MpORBbotV2/1.0"})

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
            "1min": 60,
            "3min": 180,
            "5min": 300,
            "15min": 900,
            "30min": 1800,
            "1hour": 3600,
            "4hour": 14400,
        }
        sec = sec_map.get(ktype, 300)
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
        arr = data.get("data") or []
        arr = list(reversed(arr))
        if len(arr) > limit:
            arr = arr[-limit:]
        return arr


@dataclass
class BotSettings:
    coins: List[str]
    trade_mode: str
    engine_on: bool
    notify: bool

    stake_usdt: float
    risk_mult: float

    fee_rate_per_side: float
    slippage_rate_per_side: float

    max_positions: int
    cooldown_per_coin_sec: int

    trend_filter: bool
    trend_ema_fast: int
    trend_ema_slow: int

    pivot_left: int
    pivot_right: int
    pivot_max_age_bars: int

    min_sweep_pct: float
    min_reclaim_close_pct: float
    break_buffer_pct: float
    setup_expire_min: int

    atr_period: int
    atr_filter_on: bool
    sl_atr_mult: float

    break_even_r: float
    trail_start_r: float
    trail_atr_mult: float

    max_hold_min: int
    stall_min: int
    stall_min_r: float


@dataclass
class Position:
    symbol: str
    side: str
    qty: float
    stake_usdt: float
    entry_price: float
    entry_ts: int

    sweep_low: float
    pivot_low: float
    reclaim_high: float

    atr_at_entry: float
    risk_per_unit: float

    sl_price: float
    break_even_done: bool

    trail_active: bool
    trail_stop: float
    trail_highest: float


@dataclass
class PnL:
    trades: int
    net_usdt: float


@dataclass
class CoinState:
    cooldown_until: int


@dataclass
class PendingSetup:
    symbol: str
    pivot_low: float
    sweep_low: float
    reclaim_high: float
    setup_ts: int
    expires_ts: int
    atr_at_setup: float


class SweepBreakProEngine:
    def __init__(self, kucoin: KuCoinPublic):
        self.kucoin = kucoin

        self.settings = BotSettings(
            coins=DEFAULT_COINS.copy(),
            trade_mode="mock",
            engine_on=False,
            notify=True,

            stake_usdt=30.0,
            risk_mult=1.0,

            fee_rate_per_side=0.0010,
            slippage_rate_per_side=0.0002,

            max_positions=3,
            cooldown_per_coin_sec=120,

            trend_filter=True,
            trend_ema_fast=20,
            trend_ema_slow=50,

            pivot_left=3,
            pivot_right=3,
            pivot_max_age_bars=80,

            min_sweep_pct=0.35,
            min_reclaim_close_pct=0.20,
            break_buffer_pct=0.08,
            setup_expire_min=180,

            atr_period=14,
            atr_filter_on=True,
            sl_atr_mult=3.0,

            break_even_r=1.0,
            trail_start_r=2.2,
            trail_atr_mult=1.8,

            max_hold_min=72 * 60,
            stall_min=12 * 60,
            stall_min_r=0.30,
        )

        self.positions: Dict[str, Position] = {}
        self.coin_state: Dict[str, CoinState] = {c: CoinState(cooldown_until=0) for c in self.settings.coins}
        self.pending: Dict[str, PendingSetup] = {}
        self.pnl = PnL(trades=0, net_usdt=0.0)

        self._load_persisted()

    def _load_persisted(self):
        st = load_state()
        if not st:
            return
        try:
            s = st.get("settings", {})
            for k in asdict(self.settings).keys():
                if k in s:
                    setattr(self.settings, k, s[k])

            p = st.get("pnl", {})
            if "trades" in p and "net_usdt" in p:
                self.pnl = PnL(trades=int(p["trades"]), net_usdt=float(p["net_usdt"]))

            for c in self.settings.coins:
                if c not in self.coin_state:
                    self.coin_state[c] = CoinState(cooldown_until=0)
        except Exception as e:
            log.warning(f"_load_persisted failed: {e}")

    def persist(self):
        save_state({
            "settings": asdict(self.settings),
            "pnl": asdict(self.pnl),
        })

    def set_coins(self, coins: List[str]):
        coins = [c.strip().upper() for c in coins if c.strip()]
        self.settings.coins = coins
        for c in coins:
            if c not in self.coin_state:
                self.coin_state[c] = CoinState(cooldown_until=0)
        for sym in list(self.positions.keys()):
            if sym not in coins:
                del self.positions[sym]
        for sym in list(self.pending.keys()):
            if sym not in coins:
                del self.pending[sym]
        self.persist()

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

    def _trend_ok(self, symbol: str) -> bool:
        if not self.settings.trend_filter:
            return True
        candles = self.kucoin.get_candles(symbol, TF_TREND, TREND_CANDLES)
        closes = [safe_float(c[2]) for c in candles]
        e_fast = ema(closes, int(self.settings.trend_ema_fast))
        e_slow = ema(closes, int(self.settings.trend_ema_slow))
        if e_fast is None or e_slow is None:
            return False
        if closes[-1] <= 0:
            return False
        return e_fast > e_slow and closes[-1] > e_fast

    def _build_setup_from_15m(self, symbol: str) -> Optional[PendingSetup]:
        candles = self.kucoin.get_candles(symbol, TF_SETUP, SETUP_CANDLES)
        if len(candles) < 60:
            return None

        opens = [safe_float(c[1]) for c in candles]
        closes = [safe_float(c[2]) for c in candles]
        highs = [safe_float(c[3]) for c in candles]
        lows = [safe_float(c[4]) for c in candles]

        if not self._trend_ok(symbol):
            return None

        pv = find_last_pivot_low(
            lows,
            int(self.settings.pivot_left),
            int(self.settings.pivot_right),
            int(self.settings.pivot_max_age_bars),
        )
        if not pv:
            return None

        pivot_idx, pivot_low = pv

        i = len(candles) - 2
        if i <= pivot_idx:
            return None

        o = opens[i]
        c = closes[i]
        h = highs[i]
        l = lows[i]

        if c <= 0 or pivot_low <= 0:
            return None

        min_sweep_level = pivot_low * (1 - float(self.settings.min_sweep_pct) / 100.0)
        min_reclaim_close = pivot_low * (1 + float(self.settings.min_reclaim_close_pct) / 100.0)

        swept = l <= min_sweep_level
        reclaimed = c >= min_reclaim_close
        bullish_close = c > o and c > closes[i - 1]

        if not (swept and reclaimed and bullish_close):
            return None

        a = atr_from_ohlc(highs, lows, closes, int(self.settings.atr_period))
        if a is None or a <= 0:
            return None

        if self.settings.atr_filter_on:
            recent_atrs = []
            for k in range(40, 15, -1):
                sub_h = highs[:-k]
                sub_l = lows[:-k]
                sub_c = closes[:-k]
                v = atr_from_ohlc(sub_h, sub_l, sub_c, int(self.settings.atr_period))
                if v is not None:
                    recent_atrs.append(v)
            if recent_atrs:
                avg_atr = sum(recent_atrs[-10:]) / min(len(recent_atrs[-10:]), 10)
                if a < avg_atr * 0.9:
                    return None

        expires = ts_unix() + int(self.settings.setup_expire_min) * 60

        return PendingSetup(
            symbol=symbol,
            pivot_low=pivot_low,
            sweep_low=l,
            reclaim_high=h,
            setup_ts=ts_unix(),
            expires_ts=expires,
            atr_at_setup=a,
        )

    def _entry_trigger_hit(self, symbol: str, setup: PendingSetup) -> bool:
        candles = self.kucoin.get_candles(symbol, TF_TRIGGER, TRIGGER_CANDLES)
        if len(candles) < 20:
            return False

        closes = [safe_float(c[2]) for c in candles]
        highs = [safe_float(c[3]) for c in candles]
        lows = [safe_float(c[4]) for c in candles]

        atr_now = atr_from_ohlc(highs, lows, closes, int(self.settings.atr_period))
        if atr_now is None or atr_now <= 0:
            return False

        trigger_level = setup.reclaim_high * (1 + float(self.settings.break_buffer_pct) / 100.0)
        return closes[-2] >= trigger_level or closes[-1] >= trigger_level

    def _arm_pending_setup_if_any(self, symbol: str) -> Optional[str]:
        if symbol in self.pending:
            p = self.pending[symbol]
            if ts_unix() > p.expires_ts:
                del self.pending[symbol]
            return None

        setup = self._build_setup_from_15m(symbol)
        if not setup:
            return None

        self.pending[symbol] = setup
        return (
            f"SETUP {symbol}\n"
            f"pivot_low={setup.pivot_low:.6f} sweep_low={setup.sweep_low:.6f}\n"
            f"reclaim_high={setup.reclaim_high:.6f}\n"
            f"waiting break > {setup.reclaim_high * (1 + self.settings.break_buffer_pct/100):.6f}"
        )

    def try_open_long(self, symbol: str) -> Optional[str]:
        if symbol in self.positions:
            return None
        if self.open_positions_count() >= int(self.settings.max_positions):
            return None
        if not self.can_trade_coin(symbol):
            return None

        self._arm_pending_setup_if_any(symbol)

        setup = self.pending.get(symbol)
        if not setup:
            return None

        if ts_unix() > setup.expires_ts:
            del self.pending[symbol]
            return None

        if not self._entry_trigger_hit(symbol, setup):
            return None

        last, bid, ask = self._level1_prices(symbol)
        entry_raw = ask if ask > 0 else last
        entry_price = entry_raw

        if self.settings.trade_mode == "mock":
            entry_price = self._apply_slippage_price(entry_price, "BUY")

        stake = float(self.settings.stake_usdt) * float(self.settings.risk_mult)
        qty = self._calc_qty(stake, entry_price)
        if qty <= 0:
            return None

        atr_entry = max(setup.atr_at_setup, entry_price * 0.003)
        sl_candidate_1 = setup.sweep_low - atr_entry * float(self.settings.sl_atr_mult)
        sl_candidate_2 = setup.pivot_low - atr_entry * float(self.settings.sl_atr_mult)
        sl_price = min(sl_candidate_1, sl_candidate_2)

        risk_per_unit = entry_price - sl_price
        if risk_per_unit <= 0:
            return None

        pos = Position(
            symbol=symbol,
            side="LONG",
            qty=qty,
            stake_usdt=stake,
            entry_price=entry_price,
            entry_ts=ts_unix(),

            sweep_low=setup.sweep_low,
            pivot_low=setup.pivot_low,
            reclaim_high=setup.reclaim_high,

            atr_at_entry=atr_entry,
            risk_per_unit=risk_per_unit,

            sl_price=sl_price,
            break_even_done=False,

            trail_active=False,
            trail_stop=0.0,
            trail_highest=entry_price,
        )
        self.positions[symbol] = pos
        self.mark_cooldown(symbol)
        del self.pending[symbol]
        self.persist()

        return (
            f"ENTRY {symbol} LONG @ {entry_price:.6f}\n"
            f"pivot_low={pos.pivot_low:.6f} sweep_low={pos.sweep_low:.6f}\n"
            f"reclaim_high={pos.reclaim_high:.6f}\n"
            f"SL={pos.sl_price:.6f} | 1R={pos.risk_per_unit:.6f}"
        )

    def _close_position(self, symbol: str, reason: str) -> Optional[Tuple[str, float]]:
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

        msg = (
            f"EXIT {symbol} @ {exit_price:.6f}\n"
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

            if price <= pos.sl_price:
                r = self._close_position(sym, "SL")
                if r:
                    msgs.append(r[0])
                continue

            current_r = (price - pos.entry_price) / pos.risk_per_unit if pos.risk_per_unit > 0 else 0.0

            if (not pos.break_even_done) and current_r >= float(self.settings.break_even_r):
                pos.sl_price = max(pos.sl_price, pos.entry_price)
                pos.break_even_done = True
                msgs.append(f"BE {sym} stop -> {pos.sl_price:.6f}")
                self.persist()

            if price > pos.trail_highest:
                pos.trail_highest = price

            if (not pos.trail_active) and current_r >= float(self.settings.trail_start_r):
                pos.trail_active = True
                pos.trail_stop = pos.trail_highest - pos.atr_at_entry * float(self.settings.trail_atr_mult)
                msgs.append(f"TRAIL_ON {sym} stop={pos.trail_stop:.6f}")
                self.persist()

            if pos.trail_active:
                new_stop = pos.trail_highest - pos.atr_at_entry * float(self.settings.trail_atr_mult)
                if new_stop > pos.trail_stop:
                    pos.trail_stop = new_stop
                if price <= pos.trail_stop:
                    r = self._close_position(sym, "TRAIL_STOP")
                    if r:
                        msgs.append(r[0])
                    continue

            hold_min = (now - pos.entry_ts) / 60.0

            if hold_min >= float(self.settings.stall_min):
                if current_r < float(self.settings.stall_min_r):
                    r = self._close_position(sym, "STALL_EXIT")
                    if r:
                        msgs.append(r[0])
                    continue

            if hold_min >= float(self.settings.max_hold_min):
                r = self._close_position(sym, "MAX_HOLD")
                if r:
                    msgs.append(r[0])
                continue

        return msgs

    def step(self) -> List[str]:
        if not self.settings.engine_on:
            return []
        out: List[str] = []

        out.extend(self.update_positions())

        for sym in self.settings.coins:
            if sym not in self.positions:
                try:
                    setup_msg = self._arm_pending_setup_if_any(sym)
                    if setup_msg and self.settings.notify:
                        out.append(setup_msg)
                except Exception as e:
                    log.warning(f"setup error {sym}: {e}")

        for sym in self.settings.coins:
            if sym in self.positions:
                continue
            if self.open_positions_count() >= int(self.settings.max_positions):
                break
            try:
                m = self.try_open_long(sym)
                if m:
                    out.append(m)
            except Exception as e:
                log.warning(f"entry error {sym}: {e}")

        return out

    def status_text(self) -> str:
        s = self.settings
        open_pos = list(self.positions.keys())
        pending = list(self.pending.keys())
        return (
            f"<b>ENGINE:</b> {'ON' if s.engine_on else 'OFF'}\n"
            f"<b>Strategy:</b> {STRATEGY_NAME} ({STRATEGY_CODE})\n"
            f"<b>Trade mode:</b> {s.trade_mode}\n"
            f"<b>TF:</b> trend={TF_TREND} | setup={TF_SETUP} | trigger={TF_TRIGGER}\n\n"
            f"<b>Stake:</b> {s.stake_usdt:.2f} USDT | <b>Risk mult:</b> {s.risk_mult:.2f}\n"
            f"<b>Max pos:</b> {s.max_positions} | <b>Cooldown:</b> {s.cooldown_per_coin_sec}s\n"
            f"<b>Fees:</b> {s.fee_rate_per_side*100:.2f}%/side | <b>Slip:</b> {s.slippage_rate_per_side*100:.2f}%/side\n\n"
            f"<b>Trend filter:</b> {'ON' if s.trend_filter else 'OFF'} | EMA {s.trend_ema_fast}>{s.trend_ema_slow}\n"
            f"<b>Pivot:</b> L={s.pivot_left} R={s.pivot_right} age<={s.pivot_max_age_bars}\n"
            f"<b>Sweep:</b> {fmt_pct(s.min_sweep_pct)} | <b>Reclaim close:</b> {fmt_pct(s.min_reclaim_close_pct)}\n"
            f"<b>Break buffer:</b> {fmt_pct(s.break_buffer_pct)} | <b>Setup expire:</b> {s.setup_expire_min}m\n"
            f"<b>ATR filter:</b> {'ON' if s.atr_filter_on else 'OFF'} | period={s.atr_period}\n"
            f"<b>SL ATR:</b> x{s.sl_atr_mult:.2f}\n"
            f"<b>BE:</b> {s.break_even_r:.2f}R | <b>Trail start:</b> {s.trail_start_r:.2f}R | <b>Trail ATR:</b> x{s.trail_atr_mult:.2f}\n"
            f"<b>Stall:</b> {s.stall_min}m @ <{s.stall_min_r:.2f}R | <b>Max hold:</b> {s.max_hold_min}m\n\n"
            f"<b>Trades:</b> {self.pnl.trades}\n"
            f"<b>Total NET PnL:</b> {self.pnl.net_usdt:+.4f} USDT\n"
            f"<b>Coins ({len(s.coins)}):</b> {s.coins}\n"
            f"<b>Pending setups:</b> {pending if pending else 'none'}\n"
            f"<b>Open positions:</b> {open_pos if open_pos else 'none'}\n"
            f"<b>Notify:</b> {'ON' if s.notify else 'OFF'}\n"
        )


MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["/status", "/pnl"],
        ["/engine_on", "/engine_off"],
        ["/coins", "/stake"],
        ["/risk", "/trend"],
        ["/sweep", "/reclaim"],
        ["/breakbuf", "/sl_atr"],
        ["/trail", "/cooldown"],
        ["/maxpos", "/notify"],
        ["/export_csv", "/close_all"],
        ["/reset_pnl"],
    ],
    resize_keyboard=True,
)

ENGINE: Optional[SweepBreakProEngine] = None


def _one_float(args: List[str]) -> Optional[float]:
    if not args:
        return None
    try:
        return float(str(args[0]).replace("%", "").strip())
    except Exception:
        return None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        "MpORBbot V2 PRO\n\n"
        "Logic:\n"
        "1) 15m sweep under pivot low\n"
        "2) 15m reclaim closes back above pivot\n"
        "3) 5m breaks reclaim high\n"
        "4) entry long\n\n"
        "Defaults are already tuned for bigger moves.\n"
        "Turn on with /engine_on",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        ENGINE.status_text(),
        parse_mode=ParseMode.HTML,
        reply_markup=MAIN_KEYBOARD,
    )


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
    await update.message.reply_text(
        f"Notify: {'ON' if ENGINE.settings.notify else 'OFF'}",
        reply_markup=MAIN_KEYBOARD,
    )


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
            msgs.append(f"close error {sym}: {e}")
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
            caption="Trade log CSV",
        )
    except Exception as e:
        await update.message.reply_text(f"CSV send failed: {e}", reply_markup=MAIN_KEYBOARD)


async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Use:\n/coins BTC-USDT ETH-USDT SOL-USDT\n\n"
            f"Now: {ENGINE.settings.coins}",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.set_coins(context.args)
    await update.message.reply_text(
        f"Coins updated ({len(ENGINE.settings.coins)}): {ENGINE.settings.coins}",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _one_float(context.args)
    if v is None:
        await update.message.reply_text(
            f"Stake now: {ENGINE.settings.stake_usdt:.2f}\nUse: /stake 30",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.settings.stake_usdt = clamp(v, 1.0, 100000.0)
    ENGINE.persist()
    await update.message.reply_text(
        f"Stake set to {ENGINE.settings.stake_usdt:.2f} USDT",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _one_float(context.args)
    if v is None:
        await update.message.reply_text(
            f"Risk mult now: {ENGINE.settings.risk_mult:.2f}\nUse: /risk 1.0",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.settings.risk_mult = clamp(v, 0.10, 5.0)
    ENGINE.persist()
    await update.message.reply_text(
        f"Risk mult set to {ENGINE.settings.risk_mult:.2f}",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            f"Trend filter: {'ON' if ENGINE.settings.trend_filter else 'OFF'}\nUse: /trend on or /trend off",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    v = str(context.args[0]).lower().strip()
    ENGINE.settings.trend_filter = v in ("on", "1", "true", "yes")
    ENGINE.persist()
    await update.message.reply_text(
        f"Trend filter: {'ON' if ENGINE.settings.trend_filter else 'OFF'}",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _one_float(context.args)
    if v is None:
        await update.message.reply_text(
            f"Sweep now: {fmt_pct(ENGINE.settings.min_sweep_pct)}\nUse: /sweep 0.35",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.settings.min_sweep_pct = clamp(v, 0.05, 5.0)
    ENGINE.persist()
    await update.message.reply_text(
        f"Sweep set to {fmt_pct(ENGINE.settings.min_sweep_pct)}",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_reclaim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _one_float(context.args)
    if v is None:
        await update.message.reply_text(
            f"Reclaim now: {fmt_pct(ENGINE.settings.min_reclaim_close_pct)}\nUse: /reclaim 0.20",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.settings.min_reclaim_close_pct = clamp(v, 0.01, 5.0)
    ENGINE.persist()
    await update.message.reply_text(
        f"Reclaim set to {fmt_pct(ENGINE.settings.min_reclaim_close_pct)}",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_breakbuf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _one_float(context.args)
    if v is None:
        await update.message.reply_text(
            f"Break buffer now: {fmt_pct(ENGINE.settings.break_buffer_pct)}\nUse: /breakbuf 0.08",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.settings.break_buffer_pct = clamp(v, 0.01, 3.0)
    ENGINE.persist()
    await update.message.reply_text(
        f"Break buffer set to {fmt_pct(ENGINE.settings.break_buffer_pct)}",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_sl_atr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _one_float(context.args)
    if v is None:
        await update.message.reply_text(
            f"SL ATR mult now: {ENGINE.settings.sl_atr_mult:.2f}\nUse: /sl_atr 3.0",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.settings.sl_atr_mult = clamp(v, 0.5, 10.0)
    ENGINE.persist()
    await update.message.reply_text(
        f"SL ATR mult set to {ENGINE.settings.sl_atr_mult:.2f}",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_trail(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text(
            f"Trail now: start {ENGINE.settings.trail_start_r:.2f}R | ATR {ENGINE.settings.trail_atr_mult:.2f}\nUse: /trail 2.2 1.8",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    try:
        start_r = float(str(context.args[0]).strip())
        atr_mult = float(str(context.args[1]).strip())
    except Exception:
        await update.message.reply_text("Use: /trail 2.2 1.8", reply_markup=MAIN_KEYBOARD)
        return
    ENGINE.settings.trail_start_r = clamp(start_r, 0.5, 20.0)
    ENGINE.settings.trail_atr_mult = clamp(atr_mult, 0.5, 10.0)
    ENGINE.persist()
    await update.message.reply_text(
        f"Trail set: start {ENGINE.settings.trail_start_r:.2f}R | ATR {ENGINE.settings.trail_atr_mult:.2f}",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_cooldown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _one_float(context.args)
    if v is None:
        await update.message.reply_text(
            f"Cooldown now: {ENGINE.settings.cooldown_per_coin_sec}s\nUse: /cooldown 120",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.settings.cooldown_per_coin_sec = int(clamp(int(v), 0, 3600))
    ENGINE.persist()
    await update.message.reply_text(
        f"Cooldown set to {ENGINE.settings.cooldown_per_coin_sec}s",
        reply_markup=MAIN_KEYBOARD,
    )


async def cmd_maxpos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = _one_float(context.args)
    if v is None:
        await update.message.reply_text(
            f"Max positions now: {ENGINE.settings.max_positions}\nUse: /maxpos 3",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.settings.max_positions = int(clamp(int(v), 1, 20))
    ENGINE.persist()
    await update.message.reply_text(
        f"Max positions set to {ENGINE.settings.max_positions}",
        reply_markup=MAIN_KEYBOARD,
    )


async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if ENGINE is None:
                await asyncio.sleep(2)
                continue

            loop = asyncio.get_running_loop()
            msgs: List[str] = await loop.run_in_executor(None, ENGINE.step)

            if msgs and ENGINE.settings.notify:
                chat_id = app.bot_data.get("chat_id")
                if chat_id:
                    for m in msgs:
                        try:
                            await app.bot.send_message(chat_id=chat_id, text=m)
                        except Exception as e:
                            log.warning(f"notify failed: {e}")

        except Exception as e:
            log.error(f"engine loop error: {e}")

        await asyncio.sleep(ENGINE_LOOP_SEC)


async def post_init(app: Application):
    asyncio.create_task(engine_loop(app))


def main():
    global ENGINE

    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN or BOT_TOKEN")

    kucoin = KuCoinPublic(timeout=12)
    ENGINE = SweepBreakProEngine(kucoin)

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
    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("trend", cmd_trend))
    app.add_handler(CommandHandler("sweep", cmd_sweep))
    app.add_handler(CommandHandler("reclaim", cmd_reclaim))
    app.add_handler(CommandHandler("breakbuf", cmd_breakbuf))
    app.add_handler(CommandHandler("sl_atr", cmd_sl_atr))
    app.add_handler(CommandHandler("trail", cmd_trail))
    app.add_handler(CommandHandler("cooldown", cmd_cooldown))
    app.add_handler(CommandHandler("maxpos", cmd_maxpos))

    log.info("Starting V2 PRO...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
