# =========================
# Mp ORBbot — Trend Pullback Reclaim (EMA + RSI + ATR) — LONG only (MOCK)
# Single file main.py (PART 1/2)
# =========================

import os
import csv
import json
import time
import math
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mp_orbbot_tpr")

# -------------------------
# KuCoin public
# -------------------------
KUCOIN_BASE = "https://api.kucoin.com"

TF_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "4h": "4hour",
}

# -------------------------
# Files
# -------------------------
MOCK_LOG_PATH = "mock_trade_log.csv"
STATE_PATH = "bot_state.json"

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

# -------------------------
# Strategy constants
# -------------------------
STRATEGY_NAME = "Trend Pullback Reclaim (EMA+RSI+ATR)"
STRATEGY_CODE = "TPR_EMA_RSI_ATR"
EXCHANGE_NAME = "KuCoin"
TRADE_MODE = "mock"  # mock only in this version (safe)

DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

# Timeframes
DEFAULT_TF_TREND = "1h"
DEFAULT_TF_ENTRY = "5m"

# candles to fetch
DEFAULT_TREND_CANDLES = 260
DEFAULT_ENTRY_CANDLES = 260

ENGINE_LOOP_SEC = 4

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

def ensure_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

def log_trade_csv(
    path: str,
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
        TRADE_MODE,
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
        log.warning(f"Failed to save state: {e}")

# -------------------------
# Indicators
# -------------------------
def ema_series(values: List[float], period: int) -> List[float]:
    if period <= 1 or len(values) < 2:
        return values[:]
    k = 2.0 / (period + 1.0)
    out = []
    e = values[0]
    for v in values:
        e = v * k + e * (1 - k)
        out.append(e)
    return out

def ema_last(values: List[float], period: int) -> Optional[float]:
    if period <= 1 or len(values) < period:
        return None
    return ema_series(values, period)[-1]

def rsi_last(closes: List[float], period: int = 14) -> Optional[float]:
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

def atr_last(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 2 or len(highs) != len(lows) or len(lows) != len(closes):
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
    return sum(trs[-period:]) / float(period)

# -------------------------
# KuCoin client (public)
# -------------------------
class KuCoinPublicAsync:
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=KUCOIN_BASE,
            timeout=timeout,
            headers={"User-Agent": "MpORBbot/TPR"},
        )

    async def close(self):
        await self.client.aclose()

    async def get_level1(self, symbol: str) -> Tuple[float, float, float]:
        r = await self.client.get("/api/v1/market/orderbook/level1", params={"symbol": symbol})
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        d = data["data"] or {}
        last = safe_float(d.get("price"))
        bid = safe_float(d.get("bestBid"))
        ask = safe_float(d.get("bestAsk"))
        if bid <= 0:
            bid = last
        if ask <= 0:
            ask = last
        return last, bid, ask

    async def get_candles(self, symbol: str, tf: str, limit: int) -> List[List[str]]:
        ktype = TF_MAP.get(tf, tf)
        end_at = ts_unix()
        # rough seconds for timeframe
        sec_map = {
            "1min": 60, "3min": 180, "5min": 300, "15min": 900, "30min": 1800,
            "1hour": 3600, "4hour": 14400,
        }
        sec = sec_map.get(ktype, 60)
        start_at = end_at - sec * (limit + 10)

        r = await self.client.get(
            "/api/v1/market/candles",
            params={"symbol": symbol, "type": ktype, "startAt": start_at, "endAt": end_at},
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")

        arr = data.get("data") or []
        arr = list(reversed(arr))  # ASC
        if len(arr) > limit:
            arr = arr[-limit:]
        return arr

# -------------------------
# Models
# -------------------------
@dataclass
class BotSettings:
    coins: List[str]
    engine_on: bool
    notify: bool

    tf_trend: str
    tf_entry: str
    trend_candles: int
    entry_candles: int

    stake_usdt: float
    max_positions: int
    cooldown_sec: int

    # Costs (mock)
    fee_rate_per_side: float       # e.g. 0.001 = 0.10%
    slippage_rate_per_side: float  # e.g. 0.0002 = 0.02%

    # Trend filter (trend tf)
    ema_fast: int
    ema_slow: int
    ema_trend: int
    trend_slope_lookback: int      # candles for slope on EMA trend
    min_slope_pct: float           # e.g. 0.02% (per lookback window)

    # Entry (entry tf)
    entry_ema: int
    rsi_period: int
    rsi_pullback_max: float        # entry requires RSI <= this
    require_pullback_below_ema: bool

    # Exits
    atr_period: int
    sl_atr_mult: float
    tp1_atr_mult: float
    trail_atr_mult: float
    time_stop_hours: float

@dataclass
class Position:
    symbol: str
    qty: float
    stake_usdt: float
    entry_price: float
    entry_ts: int

    sl_price: float
    tp1_price: float

    trail_active: bool
    trail_stop: float
    highest: float

    atr_at_entry: float
    info: str

@dataclass
class PnL:
    trades: int
    net_usdt: float

@dataclass
class CoinState:
    cooldown_until: int

# -------------------------
# Engine
# -------------------------
class TrendPullbackEngine:
    def __init__(self, kc: KuCoinPublicAsync):
        self.kc = kc

        self.settings = BotSettings(
            coins=DEFAULT_COINS.copy(),
            engine_on=False,
            notify=True,

            tf_trend=DEFAULT_TF_TREND,
            tf_entry=DEFAULT_TF_ENTRY,
            trend_candles=DEFAULT_TREND_CANDLES,
            entry_candles=DEFAULT_ENTRY_CANDLES,

            stake_usdt=30.0,
            max_positions=3,
            cooldown_sec=30,

            fee_rate_per_side=0.0010,
            slippage_rate_per_side=0.0002,

            ema_fast=50,
            ema_slow=200,
            ema_trend=200,
            trend_slope_lookback=10,
            min_slope_pct=0.03,

            entry_ema=50,
            rsi_period=14,
            rsi_pullback_max=40.0,
            require_pullback_below_ema=True,

            atr_period=14,
            sl_atr_mult=1.2,
            tp1_atr_mult=0.9,
            trail_atr_mult=1.0,
            time_stop_hours=18.0,
        )

        self.positions: Dict[str, Position] = {}
        self.pnl = PnL(trades=0, net_usdt=0.0)
        self.coin_state: Dict[str, CoinState] = {c: CoinState(cooldown_until=0) for c in self.settings.coins}

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

            pos = st.get("positions", {})
            if isinstance(pos, dict):
                for sym, pd in pos.items():
                    if not isinstance(pd, dict):
                        continue
                    self.positions[sym] = Position(**pd)

            for c in self.settings.coins:
                if c not in self.coin_state:
                    self.coin_state[c] = CoinState(cooldown_until=0)

        except Exception as e:
            log.warning(f"Failed to load persisted state: {e}")

    def persist(self):
        save_state({
            "settings": asdict(self.settings),
            "pnl": asdict(self.pnl),
            "positions": {k: asdict(v) for k, v in self.positions.items()},
        })

    def open_positions_count(self) -> int:
        return len(self.positions)

    def can_trade_coin(self, symbol: str) -> bool:
        cs = self.coin_state.get(symbol)
        if not cs:
            return True
        return ts_unix() >= cs.cooldown_until

    def mark_cooldown(self, symbol: str):
        cs = self.coin_state.setdefault(symbol, CoinState(cooldown_until=0))
        cs.cooldown_until = ts_unix() + int(self.settings.cooldown_sec)

    def _calc_qty(self, stake_usdt: float, price: float) -> float:
        return (stake_usdt / price) if price > 0 else 0.0

    def _apply_slippage_price(self, price: float, side: str) -> float:
        slip = self.settings.slippage_rate_per_side
        if side.upper() == "BUY":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _costs_mock(self, notional_usdt: float) -> Tuple[float, float]:
        fees = notional_usdt * self.settings.fee_rate_per_side
        slip = notional_usdt * self.settings.slippage_rate_per_side
        return fees, slip

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

    async def _trend_ok(self, symbol: str) -> Tuple[bool, str]:
        s = self.settings
        candles = await self.kc.get_candles(symbol, s.tf_trend, s.trend_candles)
        closes = [safe_float(c[2]) for c in candles if safe_float(c[2]) > 0]
        if len(closes) < max(s.ema_slow, s.ema_trend) + 5:
            return False, "trend: not enough candles"

        e_fast = ema_last(closes, s.ema_fast)
        e_slow = ema_last(closes, s.ema_slow)
        e_trend_series = ema_series(closes, s.ema_trend)

        if e_fast is None or e_slow is None or len(e_trend_series) < s.trend_slope_lookback + 2:
            return False, "trend: ema calc failed"

        # price above slow EMA and fast above slow
        price = closes[-1]
        if not (price > e_slow and e_fast > e_slow):
            return False, "trend: not above ema"

        # slope on trend EMA
        a = e_trend_series[-1]
        b = e_trend_series[-1 - s.trend_slope_lookback]
        if b <= 0:
            return False, "trend: bad slope base"
        slope_pct = (a - b) / b * 100.0
        if slope_pct < s.min_slope_pct:
            return False, f"trend: slope {slope_pct:.3f}% < {s.min_slope_pct:.3f}%"

        return True, f"trend ok | slope={slope_pct:.3f}%"

    async def _entry_signal(self, symbol: str) -> Tuple[bool, str, float, float, float, float]:
        """
        Returns (signal, reason, last_price, entry_ema, rsi, atr)
        Uses entry timeframe candles.
        """
        s = self.settings
        candles = await self.kc.get_candles(symbol, s.tf_entry, s.entry_candles)
        closes = [safe_float(c[2]) for c in candles]
        highs = [safe_float(c[3]) for c in candles]
        lows = [safe_float(c[4]) for c in candles]
        if len(closes) < max(s.entry_ema, s.atr_period, s.rsi_period) + 5:
            return False, "entry: not enough candles", 0.0, 0.0, 50.0, 0.0

        e = ema_last(closes, s.entry_ema)
        r = rsi_last(closes, s.rsi_period)
        a = atr_last(highs, lows, closes, s.atr_period)

        if e is None or r is None or a is None or a <= 0:
            return False, "entry: indicator calc failed", closes[-1], e or 0.0, r or 50.0, a or 0.0

        last_price = closes[-1]
        prev_price = closes[-2]
        prev_e = ema_series(closes, s.entry_ema)[-2]

        # "Pullback then reclaim": previous candle below EMA, current closes above EMA
        reclaim = (prev_price < prev_e) and (last_price > e)

        # optionally require pullback dipped below EMA at some point recently
        if s.require_pullback_below_ema:
            recent_min = min(closes[-10:])
            if recent_min > e:
                return False, "entry: no pullback under EMA", last_price, e, r, a

        if not reclaim:
            return False, "entry: no reclaim", last_price, e, r, a

        if r > s.rsi_pullback_max:
            return False, f"entry: rsi {r:.1f} > {s.rsi_pullback_max:.1f}", last_price, e, r, a

        return True, "entry: reclaim+lowRSI", last_price, e, r, a

    async def try_open_long(self, symbol: str) -> Optional[str]:
        s = self.settings
        if symbol in self.positions:
            return None
        if self.open_positions_count() >= s.max_positions:
            return None
        if not self.can_trade_coin(symbol):
            return None

        ok_trend, trend_reason = await self._trend_ok(symbol)
        if not ok_trend:
            return None

        sig, reason, last_close, entry_ema, rsi_v, atr_v = await self._entry_signal(symbol)
        if not sig:
            return None

        last, bid, ask = await self.kc.get_level1(symbol)
        entry_raw = ask if ask > 0 else (last_close if last_close > 0 else last)
        entry_price = self._apply_slippage_price(entry_raw, "BUY")

        qty = self._calc_qty(s.stake_usdt, entry_price)
        if qty <= 0:
            return None

        sl_price = entry_price - (atr_v * s.sl_atr_mult)
        tp1_price = entry_price + (atr_v * s.tp1_atr_mult)

        pos = Position(
            symbol=symbol,
            qty=qty,
            stake_usdt=s.stake_usdt,
            entry_price=entry_price,
            entry_ts=ts_unix(),
            sl_price=sl_price,
            tp1_price=tp1_price,
            trail_active=False,
            trail_stop=entry_price - (atr_v * s.trail_atr_mult),
            highest=entry_price,
            atr_at_entry=atr_v,
            info=f"{reason} | {trend_reason} | ema={entry_ema:.6f} rsi={rsi_v:.1f} atr={atr_v:.6f}",
        )

        self.positions[symbol] = pos
        self.mark_cooldown(symbol)
        self.persist()

        return (
            f"ENTRY {symbol} LONG @ {entry_price:.6f}\n"
            f"{pos.info}\n"
            f"stake={pos.stake_usdt:.2f} qty={qty:.8f}\n"
            f"SL={sl_price:.6f} | TP1={tp1_price:.6f} | trail_atr={s.trail_atr_mult:.2f}"
        )

    async def close_position(self, symbol: str, reason: str) -> Optional[str]:
        pos = self.positions.get(symbol)
        if not pos:
            return None

        last, bid, _ask = await self.kc.get_level1(symbol)
        exit_raw = bid if bid > 0 else last
        exit_price = self._apply_slippage_price(exit_raw, "SELL")

        entry_notional = pos.qty * pos.entry_price
        exit_notional = pos.qty * exit_price
        gross = exit_notional - entry_notional

        fee_in, slip_in = self._costs_mock(entry_notional)
        fee_out, slip_out = self._costs_mock(exit_notional)
        fees = fee_in + fee_out
        slip = slip_in + slip_out

        net = gross - fees - slip

        self.pnl.trades += 1
        self.pnl.net_usdt += net

        log_trade_csv(
            MOCK_LOG_PATH,
            symbol,
            "LONG",
            pos.qty,
            pos.stake_usdt,
            pos.entry_price,
            exit_price,
            gross,
            fees,
            slip,
            net,
            reason,
        )

        del self.positions[symbol]
        self.mark_cooldown(symbol)
        self.persist()

        tag = "EXIT" if net >= 0 else "EXIT"
        return (
            f"{tag} {symbol} @ {exit_price:.6f}\n"
            f"Net {net:+.4f} USDT | reason={reason}\n"
            f"(gross {gross:+.4f} | fees {fees:.4f} | slip {slip:.4f})"
        )

    async def update_positions(self) -> List[str]:
        s = self.settings
        msgs: List[str] = []
        now = ts_unix()

        for sym, pos in list(self.positions.items()):
            last, _bid, _ask = await self.kc.get_level1(sym)
            price = last if last > 0 else pos.entry_price

            # highest
            if price > pos.highest:
                pos.highest = price

            # time stop
            age_sec = now - pos.entry_ts
            if age_sec >= int(s.time_stop_hours * 3600):
                m = await self.close_position(sym, "TIME_STOP")
                if m:
                    msgs.append(m)
                continue

            # hard stop
            if price <= pos.sl_price:
                m = await self.close_position(sym, "SL_ATR")
                if m:
                    msgs.append(m)
                continue

            # TP1 arms trailing + moves stop to BE-ish
            if (not pos.trail_active) and price >= pos.tp1_price:
                pos.trail_active = True
                # move stop up near entry (break-even minus tiny buffer)
                be = pos.entry_price * 0.999
                pos.sl_price = max(pos.sl_price, be)
                # set initial trail stop by ATR
                pos.trail_stop = pos.highest - (pos.atr_at_entry * s.trail_atr_mult)
                msgs.append(f"TRAIL_ARMED {sym} | price={price:.6f} | sl={pos.sl_price:.6f} | trail={pos.trail_stop:.6f}")
                self.persist()
                continue

            # trailing management
            if pos.trail_active:
                new_trail = pos.highest - (pos.atr_at_entry * s.trail_atr_mult)
                if new_trail > pos.trail_stop:
                    pos.trail_stop = new_trail

                if price <= pos.trail_stop:
                    m = await self.close_position(sym, "TRAIL_STOP_ATR")
                    if m:
                        msgs.append(m)
                    continue

        return msgs

    async def step(self) -> List[str]:
        if not self.settings.engine_on:
            return []

        out: List[str] = []
        # exits first
        out.extend(await self.update_positions())

        # entries
        for sym in self.settings.coins:
            if self.open_positions_count() >= self.settings.max_positions:
                break
            if sym in self.positions:
                continue
            try:
                m = await self.try_open_long(sym)
                if m:
                    out.append(m)
            except Exception as e:
                log.warning(f"Entry error {sym}: {e}")

        if out:
            self.persist()
        return out

    def status_text(self) -> str:
        s = self.settings
        open_pos = list(self.positions.keys())
        return (
            f"<b>ENGINE:</b> {'ON' if s.engine_on else 'OFF'}\n"
            f"<b>Strategy:</b> {STRATEGY_NAME} ({STRATEGY_CODE})\n"
            f"<b>Mode:</b> MOCK (safe)\n"
            f"<b>Timeframes:</b> trend={s.tf_trend} | entry={s.tf_entry}\n\n"
            f"<b>Stake</b> (/set stake): {s.stake_usdt:.2f} USDT\n"
            f"<b>Max positions</b> (/set max_pos): {s.max_positions}\n"
            f"<b>Cooldown</b> (/set cooldown): {s.cooldown_sec}s\n"
            f"<b>Fees:</b> {s.fee_rate_per_side*100:.2f}%/side | <b>Slippage:</b> {s.slippage_rate_per_side*100:.2f}%/side\n\n"
            f"<b>Trend filter:</b> EMA{s.ema_fast} > EMA{s.ema_slow} and price > EMA{s.ema_slow}\n"
            f"<b>Trend slope:</b> EMA{s.ema_trend} slope over {s.trend_slope_lookback} candles >= {s.min_slope_pct:.3f}%\n"
            f"<b>Entry:</b> reclaim above EMA{s.entry_ema} + RSI({s.rsi_period}) <= {s.rsi_pullback_max:.1f}\n\n"
            f"<b>Exits:</b> SL = {s.sl_atr_mult:.2f}*ATR | TP1 = {s.tp1_atr_mult:.2f}*ATR | Trail = {s.trail_atr_mult:.2f}*ATR\n"
            f"<b>Time stop:</b> {s.time_stop_hours:.1f}h\n\n"
            f"<b>Trades:</b> {self.pnl.trades}\n"
            f"<b>Total NET PnL:</b> {self.pnl.net_usdt:+.4f} USDT\n"
            f"<b>Coins ({len(s.coins)}):</b> {s.coins}\n"
            f"<b>Open positions:</b> {open_pos if open_pos else 'none'}\n"
            f"<b>Notify:</b> {'ON' if s.notify else 'OFF'}\n"
        )
# =========================
# main.py (PART 2/2) — Telegram + Commands + Engine loop
# =========================

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["/status", "/pnl"],
        ["/engine_on", "/engine_off"],
        ["/set", "/coins"],
        ["/export_csv", "/notify"],
        ["/help"],
    ],
    resize_keyboard=True,
)

ENGINE: Optional[TrendPullbackEngine] = None
KC: Optional[KuCoinPublicAsync] = None

def help_text() -> str:
    return (
        "Kommandon:\n"
        "/status — visar inställningar + PnL\n"
        "/engine_on, /engine_off — start/stop\n"
        "/pnl — total PnL\n"
        "/export_csv — skickar mock_trade_log.csv\n"
        "/notify — toggle notifications\n"
        "/coins BTC-USDT ETH-USDT ... — byt coins\n"
        "\n"
        "Ändra valfria parametrar med:\n"
        "/set key value\n"
        "\n"
        "Viktiga keys:\n"
        "stake (USDT)\n"
        "max_pos (int)\n"
        "cooldown (sek)\n"
        "fee (t.ex 0.001)\n"
        "slip (t.ex 0.0002)\n"
        "tf_trend (1h/4h)  tf_entry (5m/15m)\n"
        "min_slope_pct (t.ex 0.03)\n"
        "rsi_max (t.ex 40)\n"
        "sl_atr (t.ex 1.2)\n"
        "tp1_atr (t.ex 0.9)\n"
        "trail_atr (t.ex 1.0)\n"
        "time_stop_h (t.ex 18)\n"
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        "Mp ORBbot (TPR) igång.\n"
        "Strategi: Trend Pullback Reclaim (EMA+RSI+ATR), LONG only, MOCK.\n"
        "Kör /engine_on för att börja.\n\n"
        + help_text(),
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(help_text(), reply_markup=MAIN_KEYBOARD)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        context.application.bot_data["chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        ENGINE.status_text(),
        parse_mode=ParseMode.HTML,
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

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Total NET PnL: {ENGINE.pnl.net_usdt:+.4f} USDT\nTrades: {ENGINE.pnl.trades}",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.settings.notify = not ENGINE.settings.notify
    ENGINE.persist()
    await update.message.reply_text(f"Notify: {'ON' if ENGINE.settings.notify else 'OFF'}", reply_markup=MAIN_KEYBOARD)

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_csv(MOCK_LOG_PATH)
    try:
        await update.message.reply_document(
            document=open(MOCK_LOG_PATH, "rb"),
            filename=os.path.basename(MOCK_LOG_PATH),
            caption="Här är trade-loggen (CSV).",
        )
    except Exception as e:
        await update.message.reply_text(f"Kunde inte skicka CSV: {e}", reply_markup=MAIN_KEYBOARD)

async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "Använd:\n/coins BTC-USDT ETH-USDT SOL-USDT\n\n"
            f"Nu: {ENGINE.settings.coins}",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    ENGINE.set_coins(args)
    await update.message.reply_text(
        f"Coins uppdaterade ({len(ENGINE.settings.coins)}): {ENGINE.settings.coins}",
        reply_markup=MAIN_KEYBOARD,
    )

async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /set key value
    """
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Använd: /set key value\nEx: /set stake 30\nEx: /set rsi_max 38\n\n" + help_text(),
            reply_markup=MAIN_KEYBOARD,
        )
        return

    key = str(context.args[0]).strip().lower()
    val_s = str(context.args[1]).strip()

    s = ENGINE.settings
    try:
        if key == "stake":
            s.stake_usdt = float(val_s)
        elif key == "max_pos":
            s.max_positions = int(float(val_s))
        elif key == "cooldown":
            s.cooldown_sec = int(float(val_s))
        elif key == "fee":
            s.fee_rate_per_side = float(val_s)
        elif key == "slip":
            s.slippage_rate_per_side = float(val_s)

        elif key == "tf_trend":
            v = val_s.lower()
            if v not in ("1h", "4h"):
                raise ValueError("tf_trend must be 1h or 4h")
            s.tf_trend = v
        elif key == "tf_entry":
            v = val_s.lower()
            if v not in ("5m", "15m"):
                raise ValueError("tf_entry must be 5m or 15m")
            s.tf_entry = v

        elif key == "min_slope_pct":
            s.min_slope_pct = float(val_s)
        elif key == "rsi_max":
            s.rsi_pullback_max = float(val_s)

        elif key == "sl_atr":
            s.sl_atr_mult = float(val_s)
        elif key == "tp1_atr":
            s.tp1_atr_mult = float(val_s)
        elif key == "trail_atr":
            s.trail_atr_mult = float(val_s)

        elif key == "time_stop_h":
            s.time_stop_hours = float(val_s)

        else:
            raise ValueError("Unknown key. Use /help to see keys.")

        # clamp some values
        s.stake_usdt = clamp(s.stake_usdt, 1.0, 1000000.0)
        s.max_positions = int(clamp(float(s.max_positions), 1, 10))
        s.cooldown_sec = int(clamp(float(s.cooldown_sec), 0, 3600))
        s.fee_rate_per_side = clamp(s.fee_rate_per_side, 0.0, 0.02)
        s.slippage_rate_per_side = clamp(s.slippage_rate_per_side, 0.0, 0.01)

        s.rsi_pullback_max = clamp(s.rsi_pullback_max, 5.0, 60.0)
        s.min_slope_pct = clamp(s.min_slope_pct, 0.0, 5.0)

        s.sl_atr_mult = clamp(s.sl_atr_mult, 0.3, 10.0)
        s.tp1_atr_mult = clamp(s.tp1_atr_mult, 0.2, 10.0)
        s.trail_atr_mult = clamp(s.trail_atr_mult, 0.2, 10.0)
        s.time_stop_hours = clamp(s.time_stop_hours, 0.5, 240.0)

        ENGINE.persist()
        await update.message.reply_text(f"OK: {key} = {val_s}", reply_markup=MAIN_KEYBOARD)

    except Exception as e:
        await update.message.reply_text(f"Fel: {e}", reply_markup=MAIN_KEYBOARD)

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if ENGINE is None:
                await asyncio.sleep(2)
                continue

            msgs = await ENGINE.step()

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
    # running event loop exists here
    app.create_task(engine_loop(app))

def main():
    global ENGINE, KC

    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN (or BOT_TOKEN / TELEGRAM_BOT_TOKEN) env var.")

    async def _build_and_run():
        global ENGINE, KC
        KC = KuCoinPublicAsync(timeout=10.0)
        ENGINE = TrendPullbackEngine(KC)

        app = Application.builder().token(token).post_init(post_init).build()

        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("help", cmd_help))
        app.add_handler(CommandHandler("status", cmd_status))
        app.add_handler(CommandHandler("engine_on", cmd_engine_on))
        app.add_handler(CommandHandler("engine_off", cmd_engine_off))
        app.add_handler(CommandHandler("pnl", cmd_pnl))
        app.add_handler(CommandHandler("notify", cmd_notify))
        app.add_handler(CommandHandler("export_csv", cmd_export_csv))
        app.add_handler(CommandHandler("coins", cmd_coins))
        app.add_handler(CommandHandler("set", cmd_set))

        log.info("Bot starting (TPR, MOCK)...")
        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        # keep alive
        while True:
            await asyncio.sleep(3600)

    try:
        asyncio.run(_build_and_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
