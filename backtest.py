import csv
import json
import os
import time
from bisect import bisect_left
from itertools import product

import requests


# ============================================================
# MPORBBOT BACKTEST V10
# MEAN REVERSION 2.0
#
# 15m = bullish / neutral regime
# 5m  = oversold pullback
# 1m  = early reversal confirmation
#
# Exit:
# target = return toward 5m EMA20
# target is clamped between MIN_TARGET and 1.20%
#
# NEW UNTOUCHED HISTORICAL PERIOD
# ~175 -> 140 days ago
# ============================================================

VERSION = "V10_MEAN_REVERSION_2"

BASE = "https://api.kucoin.com"

WARMUP_DAYS = 5
TRAIN_DAYS = 20
HOLDOUT_DAYS = 10

WINDOW_DAYS = (
    WARMUP_DAYS
    + TRAIN_DAYS
    + HOLDOUT_DAYS
)

EXCLUDE_RECENT_DAYS = 140

STAKE = 100.0

FEE_SIDE = 0.0010
SLIP_SIDE = 0.0002

MAX_HOLD_MINUTES = 120
COOLDOWN_MINUTES = 15

MAX_TARGET_PCT = 1.20

CHECKPOINT_FILE = "backtest_v10_checkpoint.json"
RESULT_FILE = "backtest_results.csv"
COIN_FILE = "backtest_v10_coins.csv"

COINS = [
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


# ============================================================
# V10 GRID
#
# RSI              3
# ATR deviation    2
# volume           2
# min target       3
# stop             2
#
# TOTAL = 72
# ============================================================

RSI_VALUES = [
    35.0,
    40.0,
    45.0,
]

DEVIATION_ATR_VALUES = [
    0.35,
    0.55,
]

CAP_VOLUME_VALUES = [
    0.80,
    1.00,
]

MIN_TARGET_VALUES = [
    0.45,
    0.60,
    0.75,
]

STOP_VALUES = [
    0.35,
    0.50,
]

CONFIGS = list(product(
    RSI_VALUES,
    DEVIATION_ATR_VALUES,
    CAP_VOLUME_VALUES,
    MIN_TARGET_VALUES,
    STOP_VALUES,
))

TOTAL_CONFIGS = len(CONFIGS)

session = requests.Session()


# ============================================================
# HELPERS
# ============================================================

def sf(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def ema_series(values, period):
    result = [None] * len(values)

    if not values:
        return result

    k = 2.0 / (period + 1.0)
    current = values[0]

    for i, value in enumerate(values):
        if i > 0:
            current = (
                value * k
                + current * (1.0 - k)
            )

        if i >= period - 1:
            result[i] = current

    return result


def atr_series(
    highs,
    lows,
    closes,
    period=14,
):
    n = len(closes)

    result = [None] * n
    tr = [0.0] * n

    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    running = 0.0

    for i in range(1, n):
        running += tr[i]

        if i > period:
            running -= tr[i - period]

        if i >= period:
            result[i] = running / period

    return result


def rsi_series(
    closes,
    period=14,
):
    result = [None] * len(closes)

    if len(closes) <= period:
        return result

    gains = [0.0] * len(closes)
    losses = [0.0] * len(closes)

    for i in range(1, len(closes)):
        change = (
            closes[i]
            - closes[i - 1]
        )

        if change > 0:
            gains[i] = change

        elif change < 0:
            losses[i] = -change

    avg_gain = (
        sum(gains[1:period + 1])
        / period
    )

    avg_loss = (
        sum(losses[1:period + 1])
        / period
    )

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss

        result[period] = (
            100.0
            - 100.0 / (1.0 + rs)
        )

    for i in range(
        period + 1,
        len(closes),
    ):
        avg_gain = (
            avg_gain * (period - 1)
            + gains[i]
        ) / period

        avg_loss = (
            avg_loss * (period - 1)
            + losses[i]
        ) / period

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss

            result[i] = (
                100.0
                - 100.0 / (1.0 + rs)
            )

    return result


def volume_average(
    volumes,
    period=20,
):
    result = [None] * len(volumes)

    prefix = [0.0]

    for value in volumes:
        prefix.append(
            prefix[-1] + value
        )

    for i in range(
        period,
        len(volumes),
    ):
        result[i] = (
            prefix[i]
            - prefix[i - period]
        ) / period

    return result


def bollinger(
    closes,
    period=20,
    std_mult=2.0,
):
    middle = [None] * len(closes)
    lower = [None] * len(closes)
    upper = [None] * len(closes)

    for i in range(
        period - 1,
        len(closes),
    ):
        window = closes[
            i - period + 1:
            i + 1
        ]

        mean = sum(window) / period

        variance = sum(
            (x - mean) ** 2
            for x in window
        ) / period

        std = variance ** 0.5

        middle[i] = mean
        lower[i] = (
            mean - std * std_mult
        )
        upper[i] = (
            mean + std * std_mult
        )

    return (
        middle,
        lower,
        upper,
    )


# ============================================================
# KUCOIN
# ============================================================

def request_candles(
    symbol,
    timeframe,
    start_ts,
    end_exclusive,
):
    seconds = {
        "1min": 60,
        "5min": 300,
        "15min": 900,
    }[timeframe]

    rows = {}

    cursor = (
        end_exclusive
        - seconds
    )

    while cursor >= start_ts:
        chunk_start = max(
            start_ts,
            cursor - seconds * 1490,
        )

        params = {
            "symbol": symbol,
            "type": timeframe,
            "startAt": chunk_start,
            "endAt": cursor,
        }

        success = False

        for attempt in range(8):
            try:
                response = session.get(
                    BASE + "/api/v1/market/candles",
                    params=params,
                    timeout=30,
                )

                if response.status_code == 429:
                    wait = (
                        3 + attempt * 2
                    )

                    print(
                        f"{symbol} {timeframe}: "
                        f"rate limit {wait}s",
                        flush=True,
                    )

                    time.sleep(wait)
                    continue

                response.raise_for_status()

                payload = response.json()

                if (
                    payload.get("code")
                    != "200000"
                ):
                    raise RuntimeError(
                        str(payload)
                    )

                for row in payload.get(
                    "data",
                    [],
                ):
                    ts = int(row[0])

                    if (
                        start_ts
                        <= ts
                        < end_exclusive
                    ):
                        rows[ts] = row

                success = True
                break

            except Exception as exc:
                wait = (
                    2 + attempt * 2
                )

                print(
                    f"{symbol} {timeframe} "
                    f"retry {attempt + 1}/8: "
                    f"{exc}",
                    flush=True,
                )

                time.sleep(wait)

        if not success:
            raise RuntimeError(
                f"Download failed "
                f"{symbol} {timeframe}"
            )

        cursor = (
            chunk_start
            - seconds
        )

        time.sleep(0.10)

    return [
        rows[key]
        for key in sorted(rows)
    ]


# ============================================================
# PREPARE
# ============================================================

def prepare(rows):
    times = []
    opens = []
    closes = []
    highs = []
    lows = []
    volumes = []

    for row in rows:
        times.append(
            int(row[0])
        )

        opens.append(
            sf(row[1])
        )

        closes.append(
            sf(row[2])
        )

        highs.append(
            sf(row[3])
        )

        lows.append(
            sf(row[4])
        )

        volumes.append(
            sf(row[5])
        )

    (
        bb_mid,
        bb_low,
        bb_high,
    ) = bollinger(
        closes,
        20,
        2.0,
    )

    return {
        "times": times,
        "opens": opens,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,

        "ema20":
            ema_series(
                closes,
                20,
            ),

        "ema50":
            ema_series(
                closes,
                50,
            ),

        "ema200":
            ema_series(
                closes,
                200,
            ),

        "atr":
            atr_series(
                highs,
                lows,
                closes,
                14,
            ),

        "rsi":
            rsi_series(
                closes,
                14,
            ),

        "volavg":
            volume_average(
                volumes,
                20,
            ),

        "bb_mid": bb_mid,
        "bb_low": bb_low,
        "bb_high": bb_high,
    }


def build_lookup(data):
    return {
        ts: i
        for i, ts
        in enumerate(data["times"])
    }


def get_closed_index(
    signal_open,
    timeframe_seconds,
    lookup,
):
    signal_close = (
        signal_open + 60
    )

    last_closed_open = (
        (
            signal_close
            - timeframe_seconds
        )
        // timeframe_seconds
    ) * timeframe_seconds

    return lookup.get(
        last_closed_open
    )


# ============================================================
# 15M REGIME
#
# V10 is less restrictive than V9.
#
# We allow:
# 1. EMA50 > EMA200
# OR
# 2. EMA20 > EMA50 while price is still near EMA200
#
# We still reject clear bearish structure.
# ============================================================

def regime_15m(
    data,
    index,
):
    if (
        index is None
        or index < 205
    ):
        return False

    close = data["closes"][index]

    e20 = data["ema20"][index]
    e50 = data["ema50"][index]
    e200 = data["ema200"][index]

    old50 = data["ema50"][
        index - 4
    ]

    old200 = data["ema200"][
        index - 4
    ]

    if (
        e20 is None
        or e50 is None
        or e200 is None
        or old50 is None
        or old200 is None
        or e50 <= 0
        or e200 <= 0
    ):
        return False

    slope50 = (
        (e50 - old50)
        / old50
        * 100.0
    )

    slope200 = (
        (e200 - old200)
        / old200
        * 100.0
    )

    bullish_structure = (
        e50 > e200
    )

    recovering_structure = (
        e20 > e50
        and close
        > e200 * 0.99
    )

    if not (
        bullish_structure
        or recovering_structure
    ):
        return False

    # Reject a strongly falling larger trend.
    if slope50 < -0.08:
        return False

    if slope200 < -0.04:
        return False

    # Don't buy a major structural breakdown.
    if close < e200 * 0.975:
        return False

    # Don't buy an already massively extended market.
    extension = (
        (close - e200)
        / e200
        * 100.0
    )

    if extension > 15.0:
        return False

    return True


# ============================================================
# 5M OVERSOLD
#
# V9 was too restrictive.
#
# V10:
# - searches last 6 closed candles
# - RSI 35 / 40 / 45
# - smaller ATR deviation allowed
# - volume 0.80 / 1.00
# - doesn't require huge capitulation
#
# Still requires recovery before entry.
# ============================================================

def oversold_5m(
    data,
    index,
    rsi_limit,
    deviation_atr,
    cap_volume,
):
    if (
        index is None
        or index < 205
    ):
        return None

    c = data["closes"]
    o = data["opens"]
    h = data["highs"]
    l = data["lows"]
    v = data["volumes"]

    ema20 = data["ema20"]
    ema50 = data["ema50"]

    atr = data["atr"]
    rsi = data["rsi"]
    volavg = data["volavg"]
    bb_low = data["bb_low"]

    start = max(
        205,
        index - 6,
    )

    oversold_index = None
    lowest_rsi = 999.0

    for j in range(
        start,
        index + 1,
    ):
        if (
            ema20[j] is None
            or ema50[j] is None
            or atr[j] is None
            or rsi[j] is None
            or volavg[j] is None
            or bb_low[j] is None
            or atr[j] <= 0
        ):
            continue

        deviation = (
            ema20[j] - l[j]
        ) / atr[j]

        bollinger_hit = (
            l[j]
            <= bb_low[j] * 1.001
        )

        atr_hit = (
            deviation
            >= deviation_atr
        )

        rsi_hit = (
            rsi[j]
            <= rsi_limit
        )

        volume_hit = (
            v[j]
            >= volavg[j]
            * cap_volume
        )

        if (
            rsi_hit
            and (
                bollinger_hit
                or atr_hit
            )
            and volume_hit
        ):
            if rsi[j] < lowest_rsi:
                lowest_rsi = rsi[j]
                oversold_index = j

    if oversold_index is None:
        return None

    if (
        ema20[index] is None
        or ema50[index] is None
        or rsi[index] is None
        or atr[index] is None
    ):
        return None

    # ========================================================
    # RECOVERY
    # ========================================================

    recovery_points = 0

    # Green 5m candle
    if c[index] > o[index]:
        recovery_points += 1

    # Higher close
    if c[index] > c[index - 1]:
        recovery_points += 1

    # RSI recovery
    previous_rsi = rsi[
        max(
            oversold_index,
            index - 1,
        )
    ]

    if (
        previous_rsi is not None
        and rsi[index]
        > previous_rsi
    ):
        recovery_points += 1

    # Reclaim previous midpoint
    previous_mid = (
        h[index - 1]
        + l[index - 1]
    ) / 2.0

    if c[index] > previous_mid:
        recovery_points += 1

    # Need 2 of 4.
    # Earlier than V9, but still requires reversal evidence.
    if recovery_points < 2:
        return None

    # Avoid full breakdown below EMA50.
    if c[index] < (
        ema50[index] * 0.975
    ):
        return None

    # Don't chase if mean reversion already completed.
    extension = (
        c[index]
        - ema20[index]
    ) / ema20[index] * 100.0

    if extension > 0.35:
        return None

    return {
        "mean": ema20[index],
        "oversold_index":
            oversold_index,
    }


# ============================================================
# 1M ENTRY
#
# Earlier than V9:
# - green candle
# - decent body
# - close high in candle
# - RSI rising
# - break previous 1m high
#
# No need to break TWO highs anymore.
# ============================================================

def entry_1m(
    data1,
    data5,
    data15,
    lookup5,
    lookup15,
    i,
    rsi_limit,
    deviation_atr,
    cap_volume,
):
    if i < 220:
        return None

    t = data1["times"]
    o = data1["opens"]
    c = data1["closes"]
    h = data1["highs"]
    l = data1["lows"]
    v = data1["volumes"]

    atr = data1["atr"]
    rsi = data1["rsi"]
    volavg = data1["volavg"]
    ema20 = data1["ema20"]

    if (
        atr[i] is None
        or rsi[i] is None
        or rsi[i - 1] is None
        or volavg[i] is None
        or ema20[i] is None
        or c[i] <= 0
    ):
        return None

    # ========================================================
    # 15M REGIME
    # ========================================================

    idx15 = get_closed_index(
        t[i],
        900,
        lookup15,
    )

    if not regime_15m(
        data15,
        idx15,
    ):
        return None

    # ========================================================
    # 5M OVERSOLD
    # ========================================================

    idx5 = get_closed_index(
        t[i],
        300,
        lookup5,
    )

    setup5 = oversold_5m(
        data5,
        idx5,
        rsi_limit,
        deviation_atr,
        cap_volume,
    )

    if setup5 is None:
        return None

    # ========================================================
    # 1M VOLATILITY
    # ========================================================

    atr_pct = (
        atr[i]
        / c[i]
        * 100.0
    )

    if (
        atr_pct < 0.025
        or atr_pct > 0.80
    ):
        return None

    candle_range = (
        h[i] - l[i]
    )

    if candle_range <= 0:
        return None

    body = (
        c[i] - o[i]
    )

    # Green candle
    if body <= 0:
        return None

    body_ratio = (
        body
        / candle_range
    )

    # V9 used 0.35.
    # V10 allows earlier reversal.
    if body_ratio < 0.25:
        return None

    close_location = (
        c[i] - l[i]
    ) / candle_range

    if close_location < 0.60:
        return None

    # Reject giant spike.
    if candle_range > (
        atr[i] * 2.5
    ):
        return None

    # RSI must turn upward.
    if rsi[i] <= rsi[i - 1]:
        return None

    # Don't enter while 1m is still extremely weak.
    if rsi[i] < 32.0:
        return None

    # Don't chase completed rebound.
    if rsi[i] > 70.0:
        return None

    # Break previous 1m high.
    if c[i] <= h[i - 1]:
        return None

    # Some participation required.
    if v[i] < (
        volavg[i] * 0.75
    ):
        return None

    # Avoid chasing too far from 1m EMA20.
    extension = (
        c[i] - ema20[i]
    ) / ema20[i] * 100.0

    if extension > 0.45:
        return None

    return {
        "atr": atr[i],
        "mean5":
            setup5["mean"],
    }


# ============================================================
# RESULT OBJECT
# ============================================================

def blank_result():
    return {
        "trades": 0,
        "wins": 0,

        "gross_market": 0.0,
        "fees": 0.0,
        "slippage": 0.0,
        "net": 0.0,

        "gross_wins": 0.0,
        "gross_losses": 0.0,

        "mfe_sum": 0.0,
        "mae_sum": 0.0,

        "target_sum": 0.0,

        "equity": 0.0,
        "peak": 0.0,
        "max_dd": 0.0,

        "tp_exits": 0,
        "stop_exits": 0,
        "time_exits": 0,
    }


def add_result(
    target,
    source,
):
    for key in (
        "trades",
        "wins",
        "gross_market",
        "fees",
        "slippage",
        "net",
        "gross_wins",
        "gross_losses",
        "mfe_sum",
        "mae_sum",
        "target_sum",
        "tp_exits",
        "stop_exits",
        "time_exits",
    ):
        target[key] += source[key]

    target["max_dd"] = max(
        target["max_dd"],
        source["max_dd"],
    )


def metrics(result):
    trades = result["trades"]

    if trades > 0:
        winrate = (
            result["wins"]
            / trades
            * 100.0
        )

        avg_net = (
            result["net"]
            / trades
        )

        avg_mfe = (
            result["mfe_sum"]
            / trades
        )

        avg_mae = (
            result["mae_sum"]
            / trades
        )

        avg_target = (
            result["target_sum"]
            / trades
        )

    else:
        winrate = 0.0
        avg_net = 0.0
        avg_mfe = 0.0
        avg_mae = 0.0
        avg_target = 0.0

    if result["gross_losses"] > 0:
        pf = (
            result["gross_wins"]
            / result["gross_losses"]
        )

    elif result["gross_wins"] > 0:
        pf = 999.0

    else:
        pf = 0.0

    return {
        "winrate": winrate,
        "avg_net": avg_net,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "avg_target": avg_target,
        "pf": pf,
    }


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    data1,
    data5,
    data15,
    lookup5,
    lookup15,
    config,
    start_ts,
    end_ts,
):
    (
        rsi_limit,
        deviation_atr,
        cap_volume,
        min_target_pct,
        stop_pct,
    ) = config

    t = data1["times"]
    o = data1["opens"]
    c = data1["closes"]
    h = data1["highs"]
    l = data1["lows"]

    result = blank_result()

    i = max(
        220,
        bisect_left(
            t,
            start_ts,
        ),
    )

    while i < len(c) - 2:
        if t[i] >= end_ts:
            break

        signal = entry_1m(
            data1,
            data5,
            data15,
            lookup5,
            lookup15,
            i,
            rsi_limit,
            deviation_atr,
            cap_volume,
        )

        if signal is None:
            i += 1
            continue

        # ====================================================
        # ENTRY NEXT 1M OPEN
        # ====================================================

        entry_i = i + 1

        if (
            entry_i >= len(c)
            or t[entry_i] >= end_ts
        ):
            break

        raw_entry = o[entry_i]

        if raw_entry <= 0:
            i += 1
            continue

        actual_entry = (
            raw_entry
            * (1.0 + SLIP_SIDE)
        )

        qty = (
            STAKE
            / actual_entry
        )

        # ====================================================
        # DYNAMIC MEAN TARGET
        #
        # First choice = 5m EMA20.
        #
        # But target must be large enough to pay costs.
        # Minimum target is config parameter.
        # Maximum target = 1.20%.
        # ====================================================

        mean5 = signal["mean5"]

        natural_target_pct = (
            (mean5 - actual_entry)
            / actual_entry
            * 100.0
        )

        target_pct = max(
            min_target_pct,
            natural_target_pct,
        )

        target_pct = min(
            target_pct,
            MAX_TARGET_PCT,
        )

        target = (
            actual_entry
            * (
                1.0
                + target_pct / 100.0
            )
        )

        stop = (
            actual_entry
            * (
                1.0
                - stop_pct / 100.0
            )
        )

        highest = actual_entry
        lowest = actual_entry

        exit_i = None
        raw_exit = None
        actual_exit = None
        exit_reason = None

        last_i = min(
            len(c) - 1,
            entry_i
            + MAX_HOLD_MINUTES,
        )

        j = entry_i

        while j <= last_i:
            if t[j] >= end_ts:
                exit_i = max(
                    entry_i,
                    j - 1,
                )

                raw_exit = c[exit_i]

                actual_exit = (
                    raw_exit
                    * (
                        1.0
                        - SLIP_SIDE
                    )
                )

                exit_reason = "TIME"
                break

            highest = max(
                highest,
                h[j],
            )

            lowest = min(
                lowest,
                l[j],
            )

            # Conservative:
            # if stop and target both hit same candle,
            # STOP wins.
            if l[j] <= stop:
                exit_i = j
                raw_exit = stop

                actual_exit = (
                    raw_exit
                    * (
                        1.0
                        - SLIP_SIDE
                    )
                )

                exit_reason = "STOP"
                break

            if h[j] >= target:
                exit_i = j
                raw_exit = target

                actual_exit = (
                    raw_exit
                    * (
                        1.0
                        - SLIP_SIDE
                    )
                )

                exit_reason = "TP"
                break

            j += 1

        if actual_exit is None:
            exit_i = min(
                last_i,
                len(c) - 1,
            )

            while (
                exit_i > entry_i
                and t[exit_i] >= end_ts
            ):
                exit_i -= 1

            raw_exit = c[exit_i]

            actual_exit = (
                raw_exit
                * (
                    1.0
                    - SLIP_SIDE
                )
            )

            exit_reason = "TIME"

        # ====================================================
        # MFE / MAE
        # ====================================================

        mfe = max(
            0.0,
            (
                highest
                - actual_entry
            )
            / actual_entry
            * 100.0,
        )

        mae = max(
            0.0,
            (
                actual_entry
                - lowest
            )
            / actual_entry
            * 100.0,
        )

        # ====================================================
        # COSTS
        # ====================================================

        raw_entry_value = (
            qty * raw_entry
        )

        raw_exit_value = (
            qty * raw_exit
        )

        actual_entry_value = (
            qty * actual_entry
        )

        actual_exit_value = (
            qty * actual_exit
        )

        gross_market = (
            raw_exit_value
            - raw_entry_value
        )

        slippage_cost = (
            actual_entry_value
            - raw_entry_value
        ) + (
            raw_exit_value
            - actual_exit_value
        )

        fees = (
            actual_entry_value
            * FEE_SIDE
            + actual_exit_value
            * FEE_SIDE
        )

        # Slippage already embedded in fills.
        # Do not subtract twice.
        net = (
            actual_exit_value
            - actual_entry_value
            - fees
        )

        result["trades"] += 1

        result["gross_market"] += (
            gross_market
        )

        result["fees"] += fees

        result["slippage"] += (
            slippage_cost
        )

        result["net"] += net

        result["mfe_sum"] += mfe
        result["mae_sum"] += mae

        result["target_sum"] += (
            target_pct
        )

        if exit_reason == "TP":
            result["tp_exits"] += 1

        elif exit_reason == "STOP":
            result["stop_exits"] += 1

        else:
            result["time_exits"] += 1

        if net > 0:
            result["wins"] += 1
            result["gross_wins"] += net

        else:
            result["gross_losses"] += (
                abs(net)
            )

        result["equity"] += net

        result["peak"] = max(
            result["peak"],
            result["equity"],
        )

        result["max_dd"] = max(
            result["max_dd"],
            result["peak"]
            - result["equity"],
        )

        next_time = (
            t[exit_i]
            + COOLDOWN_MINUTES * 60
        )

        i = bisect_left(
            t,
            next_time,
        )

    return result


# ============================================================
# CHECKPOINT
# ============================================================

def make_signature(
    download_end,
):
    return {
        "version": VERSION,
        "download_end": download_end,
        "stake": STAKE,
        "coins": COINS,
        "configs": [
            list(config)
            for config in CONFIGS
        ],
    }


def save_checkpoint(
    signature,
    completed,
    train_by_coin,
    holdout_by_coin,
):
    temp = (
        CHECKPOINT_FILE
        + ".tmp"
    )

    payload = {
        "signature": signature,
        "completed": completed,
        "train_by_coin":
            train_by_coin,
        "holdout_by_coin":
            holdout_by_coin,
    }

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
        )

    os.replace(
        temp,
        CHECKPOINT_FILE,
    )


def load_checkpoint(
    signature,
):
    if not os.path.exists(
        CHECKPOINT_FILE
    ):
        return None

    try:
        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )

        if (
            payload.get("signature")
            != signature
        ):
            return None

        return payload

    except Exception:
        return None


# ============================================================
# AGGREGATE
# ============================================================

def aggregate_config(
    by_coin,
    config_index,
):
    total = blank_result()

    positive = 0
    negative = 0
    active = 0

    for symbol in COINS:
        if symbol not in by_coin:
            continue

        result = (
            by_coin[symbol][
                config_index
            ]
        )

        add_result(
            total,
            result,
        )

        if result["trades"] > 0:
            active += 1

            if result["net"] > 0:
                positive += 1

            elif result["net"] < 0:
                negative += 1

    return (
        total,
        positive,
        negative,
        active,
    )


# ============================================================
# RESULTS
# ============================================================

def write_results(
    train_by_coin,
    holdout_by_coin,
):
    candidates = []

    for idx, config in enumerate(
        CONFIGS
    ):
        (
            train,
            train_positive,
            train_negative,
            train_active,
        ) = aggregate_config(
            train_by_coin,
            idx,
        )

        tm = metrics(train)

        # Enough data to rank.
        eligible = (
            train["trades"] >= 40
            and train_active >= 8
        )

        if not eligible:
            continue

        (
            holdout,
            hold_positive,
            hold_negative,
            hold_active,
        ) = aggregate_config(
            holdout_by_coin,
            idx,
        )

        candidates.append({
            "idx": idx,

            "train": train,
            "holdout": holdout,

            "train_positive":
                train_positive,

            "train_negative":
                train_negative,

            "train_active":
                train_active,

            "hold_positive":
                hold_positive,

            "hold_negative":
                hold_negative,

            "hold_active":
                hold_active,

            "train_pf":
                tm["pf"],
        })

    # IMPORTANT:
    # Ranking only uses TRAIN.
    # HOLDOUT does not influence selection.
    candidates.sort(
        key=lambda item: (
            item["train"]["net"],
            item["train_pf"],
            item["train_positive"],
        ),
        reverse=True,
    )

    headers = [
        # Main.py / Telegram compatibility
        "rank",
        "trades",
        "wins",
        "winrate",
        "net_usdt",
        "profit_factor",
        "max_drawdown",
        "tp_pct",
        "sl_atr",
        "volume_mult",
        "lookback",

        # V10
        "stop_pct",
        "rsi_limit",
        "deviation_atr",
        "cap_volume",
        "stake",
        "avg_target_pct",

        "gross_market",
        "fees",
        "slippage",
        "avg_net_trade",
        "trades_per_week",
        "avg_mfe_pct",
        "avg_mae_pct",

        "tp_exits",
        "stop_exits",
        "time_exits",

        "positive_coins",
        "negative_coins",
        "active_coins",

        "train_trades",
        "train_wins",
        "train_winrate",
        "train_net",
        "train_pf",
        "train_gross",
        "train_fees",
        "train_slippage",
        "train_avg_net",
        "train_avg_mfe",
        "train_avg_mae",
        "train_avg_target",

        "train_positive_coins",
        "train_negative_coins",
        "train_active_coins",
    ]

    rows = []

    for rank, item in enumerate(
        candidates[:10],
        start=1,
    ):
        idx = item["idx"]

        (
            rsi_limit,
            deviation_atr,
            cap_volume,
            min_target_pct,
            stop_pct,
        ) = CONFIGS[idx]

        train = item["train"]
        holdout = item["holdout"]

        tm = metrics(train)
        hm = metrics(holdout)

        row = {
            "rank": rank,

            "trades":
                holdout["trades"],

            "wins":
                holdout["wins"],

            "winrate":
                hm["winrate"],

            "net_usdt":
                holdout["net"],

            "profit_factor":
                hm["pf"],

            "max_drawdown":
                holdout["max_dd"],

            # Telegram displays this as TP.
            # In V10 this means MIN TARGET.
            "tp_pct":
                min_target_pct,

            "sl_atr": 0.0,

            "volume_mult":
                cap_volume,

            "lookback": 6,

            "stop_pct":
                stop_pct,

            "rsi_limit":
                rsi_limit,

            "deviation_atr":
                deviation_atr,

            "cap_volume":
                cap_volume,

            "stake":
                STAKE,

            "avg_target_pct":
                hm["avg_target"],

            "gross_market":
                holdout[
                    "gross_market"
                ],

            "fees":
                holdout["fees"],

            "slippage":
                holdout["slippage"],

            "avg_net_trade":
                hm["avg_net"],

            "trades_per_week":
                holdout["trades"]
                / (
                    HOLDOUT_DAYS
                    / 7.0
                ),

            "avg_mfe_pct":
                hm["avg_mfe"],

            "avg_mae_pct":
                hm["avg_mae"],

            "tp_exits":
                holdout[
                    "tp_exits"
                ],

            "stop_exits":
                holdout[
                    "stop_exits"
                ],

            "time_exits":
                holdout[
                    "time_exits"
                ],

            "positive_coins":
                item[
                    "hold_positive"
                ],

            "negative_coins":
                item[
                    "hold_negative"
                ],

            "active_coins":
                item[
                    "hold_active"
                ],

            "train_trades":
                train["trades"],

            "train_wins":
                train["wins"],

            "train_winrate":
                tm["winrate"],

            "train_net":
                train["net"],

            "train_pf":
                tm["pf"],

            "train_gross":
                train[
                    "gross_market"
                ],

            "train_fees":
                train["fees"],

            "train_slippage":
                train[
                    "slippage"
                ],

            "train_avg_net":
                tm["avg_net"],

            "train_avg_mfe":
                tm["avg_mfe"],

            "train_avg_mae":
                tm["avg_mae"],

            "train_avg_target":
                tm["avg_target"],

            "train_positive_coins":
                item[
                    "train_positive"
                ],

            "train_negative_coins":
                item[
                    "train_negative"
                ],

            "train_active_coins":
                item[
                    "train_active"
                ],
        }

        rows.append(row)

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: (
                    f"{value:.6f}"
                    if isinstance(
                        value,
                        float,
                    )
                    else value
                )
                for key, value
                in row.items()
            })

    # ========================================================
    # PER-COIN RESULT FOR TRAIN RANK #1
    # ========================================================

    if candidates:
        best_idx = (
            candidates[0]["idx"]
        )

        fields = [
            "symbol",

            "train_trades",
            "train_net",
            "train_pf",
            "train_wr",
            "train_gross",

            "holdout_trades",
            "holdout_net",
            "holdout_pf",
            "holdout_wr",
            "holdout_gross",
        ]

        with open(
            COIN_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()

            for symbol in COINS:
                train = (
                    train_by_coin[
                        symbol
                    ][best_idx]
                )

                hold = (
                    holdout_by_coin[
                        symbol
                    ][best_idx]
                )

                tm = metrics(train)
                hm = metrics(hold)

                writer.writerow({
                    "symbol":
                        symbol,

                    "train_trades":
                        train["trades"],

                    "train_net":
                        f"{train['net']:.6f}",

                    "train_pf":
                        f"{tm['pf']:.6f}",

                    "train_wr":
                        f"{tm['winrate']:.6f}",

                    "train_gross":
                        f"{train['gross_market']:.6f}",

                    "holdout_trades":
                        hold["trades"],

                    "holdout_net":
                        f"{hold['net']:.6f}",

                    "holdout_pf":
                        f"{hm['pf']:.6f}",

                    "holdout_wr":
                        f"{hm['winrate']:.6f}",

                    "holdout_gross":
                        f"{hold['gross_market']:.6f}",
                })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    existing = None

    if os.path.exists(
        CHECKPOINT_FILE
    ):
        try:
            with open(
                CHECKPOINT_FILE,
                "r",
                encoding="utf-8",
            ) as file:
                existing = json.load(
                    file
                )

        except Exception:
            existing = None

    if (
        existing
        and isinstance(
            existing.get(
                "signature"
            ),
            dict,
        )
    ):
        download_end = int(
            existing[
                "signature"
            ].get(
                "download_end",
                0,
            )
        )

    else:
        now = int(time.time())
        now -= now % 60

        download_end = (
            now
            - EXCLUDE_RECENT_DAYS
            * 86400
        )

    download_start = (
        download_end
        - WINDOW_DAYS * 86400
    )

    train_start = (
        download_start
        + WARMUP_DAYS * 86400
    )

    train_end = (
        train_start
        + TRAIN_DAYS * 86400
    )

    holdout_start = train_end
    holdout_end = download_end

    signature = make_signature(
        download_end
    )

    print(
        "======================================",
        flush=True,
    )

    print(
        "MPORBBOT BACKTEST V10",
        flush=True,
    )

    print(
        "MEAN REVERSION 2.0",
        flush=True,
    )

    print(
        "15m REGIME -> 5m OVERSOLD -> 1m REVERSAL",
        flush=True,
    )

    print(
        "EXIT TOWARD 5m EMA20",
        flush=True,
    )

    print(
        f"Stake: {STAKE:.0f} USDT",
        flush=True,
    )

    print(
        f"Configs: {TOTAL_CONFIGS}",
        flush=True,
    )

    print(
        "5d WARMUP + 20d TRAIN + 10d HOLDOUT",
        flush=True,
    )

    print(
        "NEW PERIOD: ~175 -> 140 DAYS AGO",
        flush=True,
    )

    print(
        "MIN TARGET: 0.45 / 0.60 / 0.75%",
        flush=True,
    )

    print(
        "MAX TARGET: 1.20%",
        flush=True,
    )

    print(
        "STOP: 0.35 / 0.50%",
        flush=True,
    )

    print(
        "RSI: 35 / 40 / 45",
        flush=True,
    )

    print(
        "VOLUME: 0.80 / 1.00",
        flush=True,
    )

    print(
        "FULL FEES + SLIPPAGE",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    checkpoint = load_checkpoint(
        signature
    )

    if checkpoint:
        completed = checkpoint[
            "completed"
        ]

        train_by_coin = checkpoint[
            "train_by_coin"
        ]

        holdout_by_coin = checkpoint[
            "holdout_by_coin"
        ]

        print(
            f"RESUME "
            f"{len(completed)}/25",
            flush=True,
        )

    else:
        completed = []
        train_by_coin = {}
        holdout_by_coin = {}

    # ========================================================
    # ONE COIN AT A TIME
    # ========================================================

    for coin_no, symbol in enumerate(
        COINS,
        start=1,
    ):
        if symbol in completed:
            print(
                f"{coin_no}/25 "
                f"{symbol}: checkpoint OK",
                flush=True,
            )

            continue

        print(
            "",
            flush=True,
        )

        print(
            f"{coin_no}/25 "
            f"{symbol}: downloading 1m",
            flush=True,
        )

        rows1 = request_candles(
            symbol,
            "1min",
            download_start,
            download_end,
        )

        print(
            f"{symbol}: "
            f"{len(rows1)} 1m candles",
            flush=True,
        )

        print(
            f"{coin_no}/25 "
            f"{symbol}: downloading 5m",
            flush=True,
        )

        rows5 = request_candles(
            symbol,
            "5min",
            download_start,
            download_end,
        )

        print(
            f"{symbol}: "
            f"{len(rows5)} 5m candles",
            flush=True,
        )

        print(
            f"{coin_no}/25 "
            f"{symbol}: downloading 15m",
            flush=True,
        )

        rows15 = request_candles(
            symbol,
            "15min",
            download_start,
            download_end,
        )

        print(
            f"{symbol}: "
            f"{len(rows15)} 15m candles",
            flush=True,
        )

        if (
            len(rows1) < 1000
            or len(rows5) < 500
            or len(rows15) < 250
        ):
            print(
                f"{symbol}: "
                f"insufficient data",
                flush=True,
            )

            train_by_coin[
                symbol
            ] = [
                blank_result()
                for _ in CONFIGS
            ]

            holdout_by_coin[
                symbol
            ] = [
                blank_result()
                for _ in CONFIGS
            ]

            completed.append(
                symbol
            )

            save_checkpoint(
                signature,
                completed,
                train_by_coin,
                holdout_by_coin,
            )

            continue

        data1 = prepare(rows1)
        data5 = prepare(rows5)
        data15 = prepare(rows15)

        lookup5 = build_lookup(
            data5
        )

        lookup15 = build_lookup(
            data15
        )

        del rows1
        del rows5
        del rows15

        train_results = []
        holdout_results = []

        for idx, config in enumerate(
            CONFIGS
        ):
            train_result = simulate(
                data1,
                data5,
                data15,
                lookup5,
                lookup15,
                config,
                train_start,
                train_end,
            )

            holdout_result = simulate(
                data1,
                data5,
                data15,
                lookup5,
                lookup15,
                config,
                holdout_start,
                holdout_end,
            )

            train_results.append(
                train_result
            )

            holdout_results.append(
                holdout_result
            )

            if (
                (idx + 1) % 12 == 0
                or (
                    idx + 1
                    == TOTAL_CONFIGS
                )
            ):
                tm = metrics(
                    train_result
                )

                hm = metrics(
                    holdout_result
                )

                print(
                    f"{symbol}: "
                    f"{idx + 1}/"
                    f"{TOTAL_CONFIGS} | "
                    f"TRAIN "
                    f"{train_result['net']:+.2f} "
                    f"PF {tm['pf']:.2f} "
                    f"{train_result['trades']}t | "
                    f"HOLD "
                    f"{holdout_result['net']:+.2f} "
                    f"PF {hm['pf']:.2f} "
                    f"{holdout_result['trades']}t",
                    flush=True,
                )

        train_by_coin[
            symbol
        ] = train_results

        holdout_by_coin[
            symbol
        ] = holdout_results

        completed.append(
            symbol
        )

        save_checkpoint(
            signature,
            completed,
            train_by_coin,
            holdout_by_coin,
        )

        print(
            f"CHECKPOINT "
            f"{len(completed)}/25",
            flush=True,
        )

        del data1
        del data5
        del data15
        del lookup5
        del lookup15

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    rows = write_results(
        train_by_coin,
        holdout_by_coin,
    )

    print(
        "",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    print(
        "V10 MEAN REVERSION 2.0 COMPLETE",
        flush=True,
    )

    print(
        "RANKED ON TRAIN ONLY",
        flush=True,
    )

    print(
        "HOLDOUT WAS UNTOUCHED",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    if not rows:
        print(
            "No configuration had enough trades.",
            flush=True,
        )

    for row in rows[:5]:
        print(
            f"TRAIN RANK #{row['rank']} | "
            f"TRAIN "
            f"{row['train_net']:+.2f} "
            f"PF {row['train_pf']:.2f} | "
            f"HOLDOUT "
            f"{row['net_usdt']:+.2f} "
            f"PF "
            f"{row['profit_factor']:.2f}",
            flush=True,
        )

        print(
            f"  HOLDOUT "
            f"{row['trades']} trades | "
            f"WR "
            f"{row['winrate']:.1f}% | "
            f"{row['trades_per_week']:.0f}/week",
            flush=True,
        )

        print(
            f"  Gross "
            f"{row['gross_market']:+.2f} | "
            f"fees -{row['fees']:.2f} | "
            f"slip -{row['slippage']:.2f}",
            flush=True,
        )

        print(
            f"  Avg net/trade "
            f"{row['avg_net_trade']:+.3f} USDT",
            flush=True,
        )

        print(
            f"  MFE "
            f"{row['avg_mfe_pct']:.2f}% | "
            f"MAE "
            f"{row['avg_mae_pct']:.2f}% | "
            f"AVG TARGET "
            f"{row['avg_target_pct']:.2f}%",
            flush=True,
        )

        print(
            f"  MIN TARGET "
            f"{row['tp_pct']:.2f}% | "
            f"STOP "
            f"{row['stop_pct']:.2f}% | "
            f"RSI "
            f"{row['rsi_limit']:.0f} | "
            f"DEV "
            f"{row['deviation_atr']:.2f} ATR | "
            f"VOL "
            f"{row['cap_volume']:.2f}",
            flush=True,
        )

        print(
            f"  EXITS: "
            f"TP {row['tp_exits']} | "
            f"STOP {row['stop_exits']} | "
            f"TIME {row['time_exits']}",
            flush=True,
        )

        print(
            f"  COINS: "
            f"{row['positive_coins']} positive / "
            f"{row['negative_coins']} negative / "
            f"{row['active_coins']} active",
            flush=True,
        )

    try:
        os.remove(
            CHECKPOINT_FILE
        )
    except OSError:
        pass


if __name__ == "__main__":
    main()
