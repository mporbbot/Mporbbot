import os
import csv
import time
import json
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("high_frequency_long_bot")


KUCOIN_BASE = "https://api.kucoin.com"

STRATEGY_NAME = "High Frequency Trend LONG"
STRATEGY_CODE = "HF_TREND_LONG_V1"

STATE_PATH = "bot_state.json"
MOCK_LOG_PATH = "mock_trade_log.csv"

TREND_TF = "5min"
ENTRY_TF = "1min"

TREND_CANDLES = 260
ENTRY_CANDLES = 180

ENGINE_LOOP_SEC = 5


DEFAULT_COINS = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "BNB-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "DOGE-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "LTC-USDT",
    "NEAR-USDT",
    "APT-USDT",
    "SUI-USDT",
    "DOT-USDT",
    "TRX-USDT",
    "BCH-USDT",
    "UNI-USDT",
    "FIL-USDT",
    "ARB-USDT",
    "OP-USDT",
    "ATOM-USDT",
    "INJ-USDT",
    "AAVE-USDT",
    "ETC-USDT",
    "ICP-USDT",
]


CSV_HEADERS = [
    "timestamp",
    "symbol",
    "setup",
    "entry",
    "exit",
    "gross_move_pct",
    "gross_pnl",
    "fees",
    "slippage",
    "net_pnl",
    "reason",
]


def now_ts():
    return int(time.time())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ema(values, period):
    if len(values) < period:
        return None

    k = 2.0 / (period + 1.0)
    value = values[0]

    for x in values[1:]:
        value = x * k + value * (1.0 - k)

    return value


def atr(highs, lows, closes, period=14):
    if len(closes) < period + 2:
        return None

    ranges = []

    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        ranges.append(tr)

    return sum(ranges[-period:]) / period


def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def highest(values, bars, exclude_last=True):
    if exclude_last:
        values = values[:-1]

    if not values:
        return None

    return max(values[-bars:])


def lowest(values, bars, exclude_last=True):
    if exclude_last:
        values = values[:-1]

    if not values:
        return None

    return min(values[-bars:])


def is_doji(o, h, l, c, max_body_ratio=0.18):
    total_range = h - l

    if total_range <= 0:
        return True

    body = abs(c - o)

    return body / total_range <= max_body_ratio


def pct_move(a, b):
    if a <= 0:
        return 0.0

    return (b - a) / a * 100.0


def ensure_csv():
    if not os.path.exists(MOCK_LOG_PATH):
        with open(
            MOCK_LOG_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            csv.writer(f).writerow(CSV_HEADERS)


def write_trade(
    symbol,
    setup,
    entry,
    exit_price,
    gross_move,
    gross_pnl,
    fees,
    slip,
    net,
    reason
):
    ensure_csv()

    with open(
        MOCK_LOG_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        csv.writer(f).writerow([
            utc_now(),
            symbol,
            setup,
            f"{entry:.12f}",
            f"{exit_price:.12f}",
            f"{gross_move:.4f}",
            f"{gross_pnl:.6f}",
            f"{fees:.6f}",
            f"{slip:.6f}",
            f"{net:.6f}",
            reason,
        ])


def save_json(data):
    try:
        with open(
            STATE_PATH,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )
    except Exception as e:
        log.error(f"State save error: {e}")


def load_json():
    try:
        if not os.path.exists(STATE_PATH):
            return {}

        with open(
            STATE_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except Exception:
        return {}


class KuCoinPublic:

    def __init__(self):
        self.session = requests.Session()

    def candles(
        self,
        symbol,
        timeframe,
        limit
    ):
        seconds = {
            "1min": 60,
            "3min": 180,
            "5min": 300,
            "15min": 900,
            "30min": 1800,
            "1hour": 3600,
        }[timeframe]

        end = now_ts()
        start = end - seconds * (limit + 5)

        r = self.session.get(
            f"{KUCOIN_BASE}/api/v1/market/candles",
            params={
                "symbol": symbol,
                "type": timeframe,
                "startAt": start,
                "endAt": end,
            },
            timeout=10,
        )

        r.raise_for_status()

        data = r.json()

        if data.get("code") != "200000":
            raise RuntimeError(str(data))

        candles = list(reversed(data["data"]))

        return candles[-limit:]

    def ticker(self, symbol):

        r = self.session.get(
            f"{KUCOIN_BASE}/api/v1/market/orderbook/level1",
            params={"symbol": symbol},
            timeout=10,
        )

        r.raise_for_status()

        d = r.json()["data"]

        last = safe_float(d.get("price"))
        bid = safe_float(d.get("bestBid")) or last
        ask = safe_float(d.get("bestAsk")) or last

        return last, bid, ask


@dataclass
class Settings:

    engine_on: bool
    coins: List[str]

    stake_usdt: float
    max_positions: int

    fee_side: float
    slippage_side: float

    cooldown_sec: int

    ema_fast: int
    ema_slow: int
    ema_long: int

    volume_mult: float
    volume_lookback: int

    breakout_bars: int
    breakout_buffer_pct: float

    micro_bars: int
    micro_max_range_pct: float

    pullback_distance_pct: float

    atr_period: int
    sl_atr_mult: float

    hard_sl_max_pct: float

    break_even_trigger_pct: float
    break_even_offset_pct: float

    trail_start_pct: float
    trail_distance_pct: float

    take_profit_pct: float

    max_hold_min: int

    doji_filter: bool
    extreme_candle_atr: float


@dataclass
class Position:

    symbol: str
    setup: str

    qty: float
    stake: float

    entry: float
    entry_ts: int

    atr_entry: float

    stop: float

    highest: float

    be_done: bool
    trail_active: bool
    trail_stop: float


@dataclass
class PnlStats:

    trades: int = 0
    wins: int = 0
    losses: int = 0
    net: float = 0.0


class TradingEngine:

    def __init__(self):

        self.api = KuCoinPublic()

        self.settings = Settings(
            engine_on=False,

            coins=DEFAULT_COINS.copy(),

            stake_usdt=30.0,

            max_positions=5,

            fee_side=0.0010,

            slippage_side=0.0002,

            cooldown_sec=30,

            ema_fast=20,
            ema_slow=50,
            ema_long=200,

            volume_mult=1.10,
            volume_lookback=20,

            breakout_bars=10,

            breakout_buffer_pct=0.025,

            micro_bars=6,

            micro_max_range_pct=0.45,

            pullback_distance_pct=0.18,

            atr_period=14,

            sl_atr_mult=1.0,

            hard_sl_max_pct=0.55,

            break_even_trigger_pct=0.38,

            break_even_offset_pct=0.06,

            trail_start_pct=0.52,

            trail_distance_pct=0.22,

            take_profit_pct=0.90,

            max_hold_min=120,

            doji_filter=True,

            extreme_candle_atr=2.8,
        )

        self.positions: Dict[str, Position] = {}

        self.cooldowns: Dict[str, int] = {}

        self.last_signal_candle: Dict[str, int] = {}

        self.stats = PnlStats()

        self.load_state()

    def load_state(self):

        data = load_json()

        if not data:
            return

        try:

            saved_settings = data.get(
                "settings",
                {}
            )

            for key in asdict(self.settings):

                if key in saved_settings:

                    setattr(
                        self.settings,
                        key,
                        saved_settings[key]
                    )

            s = data.get(
                "stats",
                {}
            )

            self.stats = PnlStats(
                trades=int(s.get("trades", 0)),
                wins=int(s.get("wins", 0)),
                losses=int(s.get("losses", 0)),
                net=float(s.get("net", 0)),
            )

        except Exception as e:
            log.error(f"State load error: {e}")

    def save_state(self):

        save_json({
            "settings": asdict(
                self.settings
            ),
            "stats": asdict(
                self.stats
            ),
        })

    def trend_data(self, symbol):

        candles = self.api.candles(
            symbol,
            TREND_TF,
            TREND_CANDLES
        )

        closes = [
            safe_float(x[2])
            for x in candles
        ]

        fast = ema(
            closes,
            self.settings.ema_fast
        )

        slow = ema(
            closes,
            self.settings.ema_slow
        )

        long_ema = ema(
            closes,
            self.settings.ema_long
        )

        if (
            fast is None
            or slow is None
            or long_ema is None
        ):
            return None

        price = closes[-2]

        strong_trend = (
            fast > slow
            and price > fast
        )

        ema200_positive = (
            price > long_ema
        )

        return {
            "price": price,
            "fast": fast,
            "slow": slow,
            "ema200": long_ema,
            "trend": strong_trend,
            "ema200_positive": ema200_positive,
        }

    def entry_data(self, symbol):

        candles = self.api.candles(
            symbol,
            ENTRY_TF,
            ENTRY_CANDLES
        )

        opens = [
            safe_float(x[1])
            for x in candles
        ]

        closes = [
            safe_float(x[2])
            for x in candles
        ]

        highs = [
            safe_float(x[3])
            for x in candles
        ]

        lows = [
            safe_float(x[4])
            for x in candles
        ]

        volumes = [
            safe_float(x[5])
            for x in candles
        ]

        candle_times = [
            int(x[0])
            for x in candles
        ]

        return (
            opens,
            highs,
            lows,
            closes,
            volumes,
            candle_times
        )

    def volume_ok(
        self,
        volumes,
        index
    ):

        if index < self.settings.volume_lookback:
            return False

        avg = sum(
            volumes[
                index - self.settings.volume_lookback:index
            ]
        ) / self.settings.volume_lookback

        if avg <= 0:
            return True

        return (
            volumes[index]
            >= avg * self.settings.volume_mult
        )

    def candle_filters_ok(
        self,
        opens,
        highs,
        lows,
        closes,
        index,
        atr_value
    ):

        if self.settings.doji_filter:

            if is_doji(
                opens[index],
                highs[index],
                lows[index],
                closes[index]
            ):
                return False

        tr = max(
            highs[index] - lows[index],
            abs(
                highs[index]
                - closes[index - 1]
            ),
            abs(
                lows[index]
                - closes[index - 1]
            ),
        )

        if (
            atr_value > 0
            and tr
            > atr_value
            * self.settings.extreme_candle_atr
        ):
            return False

        return True

    def detect_signal(
        self,
        symbol
    ):

        trend = self.trend_data(
            symbol
        )

        if not trend:
            return None

        if not trend["trend"]:
            return None

        (
            opens,
            highs,
            lows,
            closes,
            volumes,
            times,
        ) = self.entry_data(symbol)

        i = len(closes) - 2

        atr_value = atr(
            highs,
            lows,
            closes,
            self.settings.atr_period
        )

        if not atr_value:
            return None

        if not self.candle_filters_ok(
            opens,
            highs,
            lows,
            closes,
            i,
            atr_value
        ):
            return None

        candle_time = times[i]

        if (
            self.last_signal_candle.get(symbol)
            == candle_time
        ):
            return None

        volume_confirm = self.volume_ok(
            volumes,
            i
        )

        ema20_entry = ema(
            closes[:i + 1],
            20
        )

        if not ema20_entry:
            return None

        setup = None


        # --------------------------------
        # SETUP 1 - TREND BREAKOUT
        # --------------------------------

        previous_high = max(
            highs[
                max(
                    0,
                    i - self.settings.breakout_bars
                ):i
            ]
        )

        breakout_trigger = (
            previous_high
            * (
                1
                + self.settings.breakout_buffer_pct
                / 100
            )
        )

        breakout = (
            closes[i]
            >= breakout_trigger
            and closes[i]
            > opens[i]
            and volume_confirm
        )

        if breakout:
            setup = "TREND_BREAKOUT"


        # --------------------------------
        # SETUP 2 - PULLBACK RECLAIM
        # --------------------------------

        if setup is None:

            distance_pct = (
                abs(
                    lows[i] - ema20_entry
                )
                / ema20_entry
                * 100
            )

            touched_ema = (
                distance_pct
                <= self.settings.pullback_distance_pct
                or lows[i] <= ema20_entry
            )

            reclaimed = (
                closes[i] > ema20_entry
                and closes[i] > opens[i]
                and closes[i] > closes[i - 1]
            )

            pullback_volume_ok = True

            if i >= 20:

                avg_vol = (
                    sum(
                        volumes[i - 20:i]
                    )
                    / 20
                )

                pullback_volume_ok = (
                    volumes[i]
                    >= avg_vol * 0.85
                )

            if (
                touched_ema
                and reclaimed
                and pullback_volume_ok
            ):
                setup = "PULLBACK_RECLAIM"


        # --------------------------------
        # SETUP 3 - MICRO BREAKOUT
        # --------------------------------

        if setup is None:

            start = max(
                0,
                i - self.settings.micro_bars
            )

            micro_high = max(
                highs[start:i]
            )

            micro_low = min(
                lows[start:i]
            )

            micro_range_pct = (
                (
                    micro_high
                    - micro_low
                )
                / closes[i]
                * 100
            )

            micro_break = (
                micro_range_pct
                <= self.settings.micro_max_range_pct
                and closes[i]
                > micro_high
                and closes[i]
                > opens[i]
                and volume_confirm
            )

            if micro_break:
                setup = "MICRO_BREAKOUT"


        if setup is None:
            return None

        self.last_signal_candle[
            symbol
        ] = candle_time

        return {
            "setup": setup,
            "atr": atr_value,
            "ema200_positive": trend[
                "ema200_positive"
            ],
        }  
        def can_open(
        self,
        symbol
    ):

        if symbol in self.positions:
            return False

        if len(
            self.positions
        ) >= self.settings.max_positions:
            return False

        cooldown = self.cooldowns.get(
            symbol,
            0
        )

        if now_ts() < cooldown:
            return False

        return True

    def open_position(
        self,
        symbol,
        signal
    ):

        if not self.can_open(symbol):
            return None

        last, bid, ask = self.api.ticker(
            symbol
        )

        if ask <= 0:
            return None

        entry = (
            ask
            * (
                1
                + self.settings.slippage_side
            )
        )

        stake = self.settings.stake_usdt

        qty = stake / entry

        atr_stop = (
            entry
            - signal["atr"]
            * self.settings.sl_atr_mult
        )

        max_stop = (
            entry
            * (
                1
                - self.settings.hard_sl_max_pct
                / 100
            )
        )

        stop = max(
            atr_stop,
            max_stop
        )

        self.positions[
            symbol
        ] = Position(
            symbol=symbol,
            setup=signal["setup"],
            qty=qty,
            stake=stake,
            entry=entry,
            entry_ts=now_ts(),
            atr_entry=signal["atr"],
            stop=stop,
            highest=entry,
            be_done=False,
            trail_active=False,
            trail_stop=0.0,
        )

        self.cooldowns[
            symbol
        ] = (
            now_ts()
            + self.settings.cooldown_sec
        )

        return (
            f"ENTRY {symbol}\n"
            f"Setup: {signal['setup']}\n"
            f"Pris: {entry:.6f}\n"
            f"SL: {stop:.6f}"
        )

    def close_position(
        self,
        symbol,
        reason
    ):

        pos = self.positions.get(
            symbol
        )

        if not pos:
            return None

        last, bid, ask = self.api.ticker(
            symbol
        )

        exit_price = (
            bid
            * (
                1
                - self.settings.slippage_side
            )
        )

        entry_value = (
            pos.qty
            * pos.entry
        )

        exit_value = (
            pos.qty
            * exit_price
        )

        gross = (
            exit_value
            - entry_value
        )

        fees = (
            entry_value
            * self.settings.fee_side
            + exit_value
            * self.settings.fee_side
        )

        slip_cost = (
            entry_value
            * self.settings.slippage_side
            + exit_value
            * self.settings.slippage_side
        )

        net = (
            gross
            - fees
            - slip_cost
        )

        move = pct_move(
            pos.entry,
            exit_price
        )

        write_trade(
            symbol=symbol,
            setup=pos.setup,
            entry=pos.entry,
            exit_price=exit_price,
            gross_move=move,
            gross_pnl=gross,
            fees=fees,
            slip=slip_cost,
            net=net,
            reason=reason,
        )

        self.stats.trades += 1

        if net > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        self.stats.net += net

        del self.positions[
            symbol
        ]

        self.cooldowns[
            symbol
        ] = (
            now_ts()
            + self.settings.cooldown_sec
        )

        self.save_state()

        return (
            f"EXIT {symbol}\n"
            f"Pris: {exit_price:.6f}\n\n"
            f"Gross: {move:+.2f}%\n"
            f"Net PnL: {net:+.4f} USDT\n"
            f"Reason: {reason}"
        )

    def manage_positions(
        self
    ):

        messages = []

        for symbol in list(
            self.positions.keys()
        ):

            pos = self.positions[
                symbol
            ]

            try:

                last, bid, ask = self.api.ticker(
                    symbol
                )

                if last <= 0:
                    continue

                if last > pos.highest:
                    pos.highest = last

                profit_pct = pct_move(
                    pos.entry,
                    last
                )


                # -------------------------
                # STOP LOSS
                # -------------------------

                if last <= pos.stop:

                    msg = self.close_position(
                        symbol,
                        "STOP_LOSS"
                    )

                    if msg:
                        messages.append(
                            msg
                        )

                    continue


                # -------------------------
                # BREAK EVEN
                # -------------------------

                if (
                    not pos.be_done
                    and profit_pct
                    >= self.settings.break_even_trigger_pct
                ):

                    be_price = (
                        pos.entry
                        * (
                            1
                            + self.settings.break_even_offset_pct
                            / 100
                        )
                    )

                    pos.stop = max(
                        pos.stop,
                        be_price
                    )

                    pos.be_done = True


                # -------------------------
                # TRAILING
                # -------------------------

                if (
                    not pos.trail_active
                    and profit_pct
                    >= self.settings.trail_start_pct
                ):

                    pos.trail_active = True

                if pos.trail_active:

                    trail = (
                        pos.highest
                        * (
                            1
                            - self.settings.trail_distance_pct
                            / 100
                        )
                    )

                    if trail > pos.trail_stop:

                        pos.trail_stop = trail

                    if (
                        pos.trail_stop
                        > pos.stop
                    ):

                        pos.stop = (
                            pos.trail_stop
                        )


                # -------------------------
                # TAKE PROFIT
                # -------------------------

                if (
                    profit_pct
                    >= self.settings.take_profit_pct
                ):

                    msg = self.close_position(
                        symbol,
                        "TAKE_PROFIT"
                    )

                    if msg:
                        messages.append(
                            msg
                        )

                    continue


                # -------------------------
                # TIME EXIT
                # -------------------------

                minutes = (
                    now_ts()
                    - pos.entry_ts
                ) / 60

                if (
                    minutes
                    >= self.settings.max_hold_min
                ):

                    msg = self.close_position(
                        symbol,
                        "TIME_EXIT"
                    )

                    if msg:
                        messages.append(
                            msg
                        )

            except Exception as e:

                log.warning(
                    f"Position error {symbol}: {e}"
                )

        return messages

    def step(
        self
    ):

        if not self.settings.engine_on:
            return []

        messages = []

        messages.extend(
            self.manage_positions()
        )

        for symbol in self.settings.coins:

            if len(
                self.positions
            ) >= self.settings.max_positions:
                break

            if not self.can_open(
                symbol
            ):
                continue

            try:

                signal = self.detect_signal(
                    symbol
                )

                if signal:

                    msg = self.open_position(
                        symbol,
                        signal
                    )

                    if msg:
                        messages.append(
                            msg
                        )

            except Exception as e:

                log.warning(
                    f"Signal error {symbol}: {e}"
                )

        return messages

    def status(
        self
    ):

        s = self.settings

        winrate = 0.0

        if self.stats.trades:

            winrate = (
                self.stats.wins
                / self.stats.trades
                * 100
            )

        return (
            f"<b>{STRATEGY_NAME}</b>\n\n"
            f"Engine: {'ON' if s.engine_on else 'OFF'}\n"
            f"Mode: MOCK\n\n"

            f"Trend TF: {TREND_TF}\n"
            f"Entry TF: {ENTRY_TF}\n"
            f"Trend: EMA{s.ema_fast} > EMA{s.ema_slow}\n"
            f"EMA200: soft filter only\n\n"

            f"Setups:\n"
            f"1. Trend Breakout\n"
            f"2. Pullback Reclaim\n"
            f"3. Micro Breakout\n\n"

            f"Volume: x{s.volume_mult:.2f}\n"
            f"Breakout bars: {s.breakout_bars}\n"
            f"Break buffer: {s.breakout_buffer_pct:.3f}%\n\n"

            f"SL: {s.sl_atr_mult:.2f} ATR\n"
            f"Max SL: {s.hard_sl_max_pct:.2f}%\n"
            f"BE: +{s.break_even_trigger_pct:.2f}%\n"
            f"Trail: +{s.trail_start_pct:.2f}% / {s.trail_distance_pct:.2f}%\n"
            f"TP: +{s.take_profit_pct:.2f}%\n\n"

            f"Stake: {s.stake_usdt:.2f} USDT\n"
            f"Max positions: {s.max_positions}\n"
            f"Coins: {len(s.coins)}\n\n"

            f"Trades: {self.stats.trades}\n"
            f"Wins: {self.stats.wins}\n"
            f"Losses: {self.stats.losses}\n"
            f"Winrate: {winrate:.1f}%\n"
            f"Net PnL: {self.stats.net:+.4f} USDT\n\n"

            f"Open: {list(self.positions.keys()) if self.positions else 'none'}"
        )


ENGINE: Optional[
    TradingEngine
] = None


KEYBOARD = ReplyKeyboardMarkup(
    [
        [
            "/status",
            "/pnl",
            "/trades"
        ],
        [
            "/open",
            "/engine_on",
            "/engine_off"
        ],
        [
            "/stake",
            "/coins",
            "/maxpos"
        ],
        [
            "/volume",
            "/tp",
            "/sl"
        ],
        [
            "/trail",
            "/cooldown",
            "/reset"
        ],
    ],
    resize_keyboard=True,
)


def read_rows():

    ensure_csv()

    with open(
        MOCK_LOG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return list(
            csv.DictReader(f)
        )


async def cmd_start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    context.application.bot_data[
        "chat_id"
    ] = update.effective_chat.id

    await update.message.reply_text(
        "High Frequency Trend LONG Bot\n\n"
        "Tre entrymodeller:\n"
        "Trend Breakout\n"
        "Pullback Reclaim\n"
        "Micro Breakout\n\n"
        "Starta med /engine_on",
        reply_markup=KEYBOARD
    )


async def cmd_status(
    update,
    context
):

    await update.message.reply_text(
        ENGINE.status(),
        parse_mode=ParseMode.HTML,
        reply_markup=KEYBOARD
    )


async def cmd_engine_on(
    update,
    context
):

    ENGINE.settings.engine_on = True

    ENGINE.save_state()

    await update.message.reply_text(
        "ENGINE ON",
        reply_markup=KEYBOARD
    )


async def cmd_engine_off(
    update,
    context
):

    ENGINE.settings.engine_on = False

    ENGINE.save_state()

    await update.message.reply_text(
        "ENGINE OFF",
        reply_markup=KEYBOARD
    )


async def cmd_pnl(
    update,
    context
):

    await update.message.reply_text(
        f"Trades: {ENGINE.stats.trades}\n"
        f"Wins: {ENGINE.stats.wins}\n"
        f"Losses: {ENGINE.stats.losses}\n"
        f"Net: {ENGINE.stats.net:+.4f} USDT",
        reply_markup=KEYBOARD
    )


async def cmd_open(
    update,
    context
):

    if not ENGINE.positions:

        await update.message.reply_text(
            "Inga öppna trades.",
            reply_markup=KEYBOARD
        )

        return

    parts = []

    for symbol, pos in ENGINE.positions.items():

        try:

            last, _, _ = ENGINE.api.ticker(
                symbol
            )

            move = pct_move(
                pos.entry,
                last
            )

            parts.append(
                f"{symbol}\n"
                f"Setup: {pos.setup}\n"
                f"Entry: {pos.entry:.6f}\n"
                f"Nu: {last:.6f}\n"
                f"Move: {move:+.2f}%\n"
                f"Stop: {pos.stop:.6f}"
            )

        except Exception:

            pass

    await update.message.reply_text(
        "\n\n".join(parts),
        reply_markup=KEYBOARD
    )


async def cmd_trades(
    update,
    context
):

    data = read_rows()

    if not data:

        await update.message.reply_text(
            "Inga trades ännu.",
            reply_markup=KEYBOARD
        )

        return

    parts = []

    for r in reversed(
        data[-10:]
    ):

        parts.append(
            f"ENTRY {r['symbol']}\n"
            f"{safe_float(r['entry']):.6f}\n"
            f"EXIT {r['symbol']}\n"
            f"{safe_float(r['exit']):.6f}\n"
            f"{safe_float(r['gross_move_pct']):+.2f}%\n"
            f"{safe_float(r['net_pnl']):+.4f} USDT\n"
            f"{r['setup']}"
        )

    await update.message.reply_text(
        "\n\n".join(parts)[:3900],
        reply_markup=KEYBOARD
    )


async def cmd_stake(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            f"Stake: {ENGINE.settings.stake_usdt:.2f} USDT\n"
            f"Använd /stake 30",
            reply_markup=KEYBOARD
        )

        return

    ENGINE.settings.stake_usdt = clamp(
        float(context.args[0]),
        1,
        100000
    )

    ENGINE.save_state()

    await update.message.reply_text(
        f"Stake: {ENGINE.settings.stake_usdt:.2f} USDT",
        reply_markup=KEYBOARD
    )


async def cmd_maxpos(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            f"Max positions: {ENGINE.settings.max_positions}",
            reply_markup=KEYBOARD
        )

        return

    ENGINE.settings.max_positions = int(
        clamp(
            int(context.args[0]),
            1,
            20
        )
    )

    ENGINE.save_state()

    await update.message.reply_text(
        f"Max positions: {ENGINE.settings.max_positions}",
        reply_markup=KEYBOARD
    )


async def cmd_volume(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            f"Volume multiplier: {ENGINE.settings.volume_mult:.2f}",
            reply_markup=KEYBOARD
        )

        return

    ENGINE.settings.volume_mult = clamp(
        float(context.args[0]),
        0.5,
        3.0
    )

    ENGINE.save_state()

    await update.message.reply_text(
        f"Volume multiplier: {ENGINE.settings.volume_mult:.2f}",
        reply_markup=KEYBOARD
    )


async def cmd_tp(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            f"TP: {ENGINE.settings.take_profit_pct:.2f}%",
            reply_markup=KEYBOARD
        )

        return

    ENGINE.settings.take_profit_pct = clamp(
        float(context.args[0]),
        0.25,
        10
    )

    ENGINE.save_state()

    await update.message.reply_text(
        f"TP: {ENGINE.settings.take_profit_pct:.2f}%",
        reply_markup=KEYBOARD
    )


async def cmd_sl(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            f"SL ATR: {ENGINE.settings.sl_atr_mult:.2f}",
            reply_markup=KEYBOARD
        )

        return

    ENGINE.settings.sl_atr_mult = clamp(
        float(context.args[0]),
        0.3,
        5
    )

    ENGINE.save_state()

    await update.message.reply_text(
        f"SL ATR: {ENGINE.settings.sl_atr_mult:.2f}",
        reply_markup=KEYBOARD
    )


async def cmd_trail(
    update,
    context
):

    if len(context.args) < 2:

        await update.message.reply_text(
            f"Trail start: {ENGINE.settings.trail_start_pct:.2f}%\n"
            f"Distance: {ENGINE.settings.trail_distance_pct:.2f}%\n\n"
            f"Använd /trail 0.52 0.22",
            reply_markup=KEYBOARD
        )

        return

    ENGINE.settings.trail_start_pct = clamp(
        float(context.args[0]),
        0.2,
        10
    )

    ENGINE.settings.trail_distance_pct = clamp(
        float(context.args[1]),
        0.05,
        5
    )

    ENGINE.save_state()

    await update.message.reply_text(
        f"Trail start: {ENGINE.settings.trail_start_pct:.2f}%\n"
        f"Distance: {ENGINE.settings.trail_distance_pct:.2f}%",
        reply_markup=KEYBOARD
    )


async def cmd_cooldown(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            f"Cooldown: {ENGINE.settings.cooldown_sec}s",
            reply_markup=KEYBOARD
        )

        return

    ENGINE.settings.cooldown_sec = int(
        clamp(
            int(context.args[0]),
            0,
            3600
        )
    )

    ENGINE.save_state()

    await update.message.reply_text(
        f"Cooldown: {ENGINE.settings.cooldown_sec}s",
        reply_markup=KEYBOARD
    )


async def cmd_coins(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            "\n".join(
                ENGINE.settings.coins
            ),
            reply_markup=KEYBOARD
        )

        return

    ENGINE.settings.coins = [
        x.upper()
        for x in context.args
    ]

    ENGINE.save_state()

    await update.message.reply_text(
        f"Coins uppdaterade: {len(ENGINE.settings.coins)}",
        reply_markup=KEYBOARD
    )


async def cmd_reset(
    update,
    context
):

    ENGINE.stats = PnlStats()

    ENGINE.save_state()

    await update.message.reply_text(
        "Statistik nollställd.",
        reply_markup=KEYBOARD
    )


async def engine_loop(
    app
):

    await asyncio.sleep(3)

    while True:

        try:

            messages = await asyncio.get_running_loop().run_in_executor(
                None,
                ENGINE.step
            )

            chat_id = app.bot_data.get(
                "chat_id"
            )

            if chat_id:

                for msg in messages:

                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=msg
                    )

        except Exception as e:

            log.error(
                f"Engine loop error: {e}"
            )

        await asyncio.sleep(
            ENGINE_LOOP_SEC
        )


async def post_init(
    app
):

    asyncio.create_task(
        engine_loop(app)
    )


def main():

    global ENGINE

    token = os.getenv(
        "TELEGRAM_TOKEN"
    ) or os.getenv(
        "BOT_TOKEN"
    )

    if not token:

        raise RuntimeError(
            "Missing TELEGRAM_TOKEN"
        )

    ENGINE = TradingEngine()

    app = (
        Application
        .builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    handlers = [
        ("start", cmd_start),
        ("status", cmd_status),
        ("pnl", cmd_pnl),
        ("trades", cmd_trades),
        ("open", cmd_open),
        ("engine_on", cmd_engine_on),
        ("engine_off", cmd_engine_off),
        ("stake", cmd_stake),
        ("maxpos", cmd_maxpos),
        ("volume", cmd_volume),
        ("tp", cmd_tp),
        ("sl", cmd_sl),
        ("trail", cmd_trail),
        ("cooldown", cmd_cooldown),
        ("coins", cmd_coins),
        ("reset", cmd_reset),
    ]

    for name, func in handlers:

        app.add_handler(
            CommandHandler(
                name,
                func
            )
        )

    log.info(
        "Starting High Frequency Trend LONG Bot"
    )

    app.run_polling(
        close_loop=False
    )


if __name__ == "__main__":
    main()
