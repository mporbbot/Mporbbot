import csv
import json
import os
import time
from bisect import bisect_left
from itertools import product

import requests


# ============================================================
# MPORBBOT BACKTEST V5
#
# STRATEGY:
# 5m strong trend
# -> 1m breakout
# -> real retest
# -> confirmation
# -> LONG
#
# EMA_RECLAIM removed.
#
# 20 days TRAIN
# 10 days untouched TEST
#
# Low-memory:
# one coin at a time
# ============================================================

VERSION = "V5"

BASE = "https://api.kucoin.com"

DAYS = 30
TRAIN_DAYS = 20

STAKE = 100.0

FEE_SIDE = 0.0010
SLIP_SIDE = 0.0002

CHECKPOINT_FILE = "backtest_v5_checkpoint.json"
RESULT_FILE = "backtest_results.csv"
TRADE_FILE = "backtest_v5_trade_stats.csv"

COINS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT",
    "XRP-USDT", "ADA-USDT", "DOGE-USDT", "LINK-USDT",
    "AVAX-USDT", "LTC-USDT", "NEAR-USDT", "APT-USDT",
    "SUI-USDT", "DOT-USDT", "TRX-USDT", "BCH-USDT",
    "UNI-USDT", "FIL-USDT", "ARB-USDT", "OP-USDT",
    "ATOM-USDT", "INJ-USDT", "AAVE-USDT", "ETC-USDT",
    "ICP-USDT",
]


# ============================================================
# V5 GRID
#
# We deliberately test fewer things.
#
# TP:
# 1.0 / 1.5 / 2.0 / 2.5%
#
# SL:
# 0.8 / 1.0 / 1.2 ATR
#
# Breakout volume:
# 1.20 / 1.40 x average
#
# Retest:
# within 4 or 6 candles
#
# 48 configurations
# ============================================================

TP_VALUES = [
    1.00,
    1.50,
    2.00,
    2.50,
]

SL_ATR_VALUES = [
    0.8,
    1.0,
    1.2,
]

VOLUME_VALUES = [
    1.20,
    1.40,
]

RETEST_WINDOWS = [
    4,
    6,
]

CONFIGS = list(product(
    TP_VALUES,
    SL_ATR_VALUES,
    VOLUME_VALUES,
    RETEST_WINDOWS,
))

TOTAL_CONFIGS = len(CONFIGS)

BREAKOUT_LOOKBACK = 12

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

        if i:
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
    period=14
):

    n = len(closes)

    result = [None] * n
    tr = [0.0] * n

    for i in range(1, n):

        tr[i] = max(
            highs[i] - lows[i],
            abs(
                highs[i] - closes[i - 1]
            ),
            abs(
                lows[i] - closes[i - 1]
            ),
        )

    running = 0.0

    for i in range(1, n):

        running += tr[i]

        if i > period:
            running -= tr[
                i - period
            ]

        if i >= period:
            result[i] = (
                running / period
            )

    return result


def previous_volume_average(
    volumes,
    period=20
):

    result = [None] * len(volumes)

    prefix = [0.0]

    for value in volumes:
        prefix.append(
            prefix[-1] + value
        )

    for i in range(
        period,
        len(volumes)
    ):

        result[i] = (
            prefix[i]
            - prefix[i - period]
        ) / period

    return result


# ============================================================
# DOWNLOAD
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
    }[timeframe]

    last_open = (
        end_exclusive
        - seconds
    )

    rows = {}

    cursor = last_open

    while cursor >= start_ts:

        chunk_start = max(
            start_ts,
            cursor
            - seconds * 1490
        )

        params = {
            "symbol": symbol,
            "type": timeframe,
            "startAt": chunk_start,
            "endAt": cursor,
        }

        success = False

        for attempt in range(7):

            try:

                response = session.get(
                    BASE
                    + "/api/v1/market/candles",
                    params=params,
                    timeout=25,
                )

                if (
                    response.status_code
                    == 429
                ):

                    time.sleep(
                        2 + attempt * 2
                    )

                    continue

                response.raise_for_status()

                payload = (
                    response.json()
                )

                if (
                    payload.get("code")
                    != "200000"
                ):

                    raise RuntimeError(
                        str(payload)
                    )

                for row in payload.get(
                    "data",
                    []
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

                print(
                    f"{symbol} "
                    f"{timeframe} "
                    f"retry "
                    f"{attempt + 1}/7: "
                    f"{exc}",
                    flush=True,
                )

                time.sleep(
                    2 + attempt * 2
                )

        if not success:

            raise RuntimeError(
                f"Download failed "
                f"{symbol} "
                f"{timeframe}"
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
# DATA
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

    return {
        "times": times,
        "opens": opens,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,

        "ema20": ema_series(
            closes,
            20
        ),

        "ema50": ema_series(
            closes,
            50
        ),

        "ema200": ema_series(
            closes,
            200
        ),

        "atr": atr_series(
            highs,
            lows,
            closes,
            14
        ),

        "volavg":
            previous_volume_average(
                volumes,
                20
            ),
    }


# ============================================================
# 5 MINUTE TREND
# ============================================================

def build_5m_lookup(data5):

    return {
        ts: i
        for i, ts
        in enumerate(
            data5["times"]
        )
    }


def get_closed_5m_index(
    signal_open,
    lookup
):

    signal_close = (
        signal_open + 60
    )

    bucket = (
        (
            signal_close - 300
        )
        // 300
    ) * 300

    return lookup.get(
        bucket
    )


def strong_trend(
    data5,
    index
):

    if (
        index is None
        or index < 205
    ):
        return False

    e20 = (
        data5["ema20"][index]
    )

    e50 = (
        data5["ema50"][index]
    )

    e200 = (
        data5["ema200"][index]
    )

    old20 = (
        data5["ema20"][
            index - 4
        ]
    )

    old50 = (
        data5["ema50"][
            index - 4
        ]
    )

    if (
        e20 is None
        or e50 is None
        or e200 is None
        or old20 is None
        or old50 is None
    ):
        return False

    close = (
        data5["closes"][index]
    )

    slope20 = (
        e20 - old20
    ) / old20 * 100.0

    slope50 = (
        e50 - old50
    ) / old50 * 100.0

    separation = (
        e20 - e50
    ) / e50 * 100.0

    # Stronger requirements than V4.
    return (
        close > e20
        and close > e50
        and close > e200

        and e20 > e50

        and slope20 >= 0.035
        and slope50 > 0.0

        and separation >= 0.025
    )


# ============================================================
# RESULT
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

        "max_dd": 0.0,

        "mfe_sum": 0.0,
        "mae_sum": 0.0,

        "mfe_max": 0.0,
        "mae_max": 0.0,
    }


def add_result(
    target,
    source
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
    ):

        target[key] += (
            source[key]
        )

    target["max_dd"] = max(
        target["max_dd"],
        source["max_dd"]
    )

    target["mfe_max"] = max(
        target["mfe_max"],
        source["mfe_max"]
    )

    target["mae_max"] = max(
        target["mae_max"],
        source["mae_max"]
    )


def metrics(result):

    trades = result["trades"]

    wr = (
        result["wins"]
        / trades
        * 100.0
        if trades
        else 0.0
    )

    if (
        result["gross_losses"]
        > 0
    ):

        pf = (
            result["gross_wins"]
            / result["gross_losses"]
        )

    elif (
        result["gross_wins"]
        > 0
    ):

        pf = 999.0

    else:

        pf = 0.0

    avg = (
        result["net"]
        / trades
        if trades
        else 0.0
    )

    avg_mfe = (
        result["mfe_sum"]
        / trades
        if trades
        else 0.0
    )

    avg_mae = (
        result["mae_sum"]
        / trades
        if trades
        else 0.0
    )

    return (
        wr,
        pf,
        avg,
        avg_mfe,
        avg_mae,
    )


# ============================================================
# SIGNAL
# ============================================================

def find_break_retest(
    data1,
    data5,
    lookup5,
    i,
    volume_mult,
    retest_window,
):

    o = data1["opens"]
    h = data1["highs"]
    l = data1["lows"]
    c = data1["closes"]
    v = data1["volumes"]

    times = data1["times"]

    ema20 = data1["ema20"]
    atr = data1["atr"]
    volavg = data1["volavg"]

    if i < 220:
        return None

    if (
        ema20[i] is None
        or atr[i] is None
        or volavg[i] is None
        or c[i] <= 0
    ):
        return None

    idx5 = get_closed_5m_index(
        times[i],
        lookup5
    )

    if not strong_trend(
        data5,
        idx5
    ):
        return None

    # ========================================================
    # CURRENT CONFIRMATION CANDLE QUALITY
    # ========================================================

    atr_pct = (
        atr[i]
        / c[i]
        * 100.0
    )

    if (
        atr_pct < 0.07
        or atr_pct > 0.65
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

    if body <= 0:
        return None

    body_ratio = (
        body
        / candle_range
    )

    # Strong green confirmation candle.
    if body_ratio < 0.50:
        return None

    # Close should be in upper part
    # of the confirmation candle.
    close_position = (
        c[i] - l[i]
    ) / candle_range

    if close_position < 0.72:
        return None

    if (
        candle_range
        > atr[i] * 1.8
    ):
        return None

    # Confirmation volume must also
    # be decent.
    if (
        v[i]
        < volavg[i] * 1.05
    ):
        return None

    # Don't chase far above 1m EMA20.
    ema_distance = (
        c[i] - ema20[i]
    ) / ema20[i] * 100.0

    if (
        ema_distance < 0
        or ema_distance > 0.45
    ):
        return None

    # ========================================================
    # SEARCH FOR RECENT BREAKOUT
    # ========================================================

    first_breakout = max(
        BREAKOUT_LOOKBACK + 2,
        i - retest_window - 1,
    )

    last_breakout = (
        i - 2
    )

    for b in range(
        first_breakout,
        last_breakout + 1
    ):

        if (
            volavg[b] is None
            or atr[b] is None
        ):
            continue

        previous_high = max(
            h[
                b
                - BREAKOUT_LOOKBACK:
                b
            ]
        )

        # Breakout must close clearly
        # above resistance.
        breakout_level = (
            previous_high
            * (
                1.0
                + 0.04 / 100.0
            )
        )

        b_range = (
            h[b] - l[b]
        )

        if b_range <= 0:
            continue

        b_body = (
            c[b] - o[b]
        )

        b_body_ratio = (
            b_body / b_range
        )

        breakout_ok = (
            b_body > 0
            and b_body_ratio >= 0.45

            and c[b]
            >= breakout_level

            and v[b]
            >= volavg[b]
            * volume_mult
        )

        if not breakout_ok:
            continue

        # Don't accept gigantic
        # breakout candles.
        if (
            b_range
            > atr[b] * 2.0
        ):
            continue

        # ====================================================
        # RETEST
        # ====================================================

        retest_index = None
        retest_low = None

        failed = False

        for r in range(
            b + 1,
            i
        ):

            # If price closes too far below
            # the old resistance, breakout
            # is considered failed.
            if (
                c[r]
                < previous_high
                * (
                    1.0
                    - 0.15 / 100.0
                )
            ):

                failed = True
                break

            # Real retest:
            # price comes back close
            # to breakout level.
            low_distance = (
                l[r]
                - previous_high
            ) / previous_high * 100.0

            if (
                -0.12
                <= low_distance
                <= 0.18
            ):

                retest_index = r
                retest_low = l[r]

        if failed:
            continue

        if retest_index is None:
            continue

        # Confirmation must come
        # AFTER retest.
        if retest_index >= i:
            continue

        # ====================================================
        # CONFIRMATION
        # ====================================================

        confirmation = (
            c[i] > h[i - 1]
            and c[i] > previous_high
            and c[i] > c[i - 1]
        )

        if not confirmation:
            continue

        # We want actual movement
        # from the retest low.
        bounce = (
            c[i] - retest_low
        ) / retest_low * 100.0

        if bounce < 0.10:
            continue

        return {
            "atr": atr[i],
            "retest_low": retest_low,
        }

    return None


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    data1,
    data5,
    lookup5,
    config,
    start_ts,
    end_ts,
):

    (
        tp_pct,
        sl_atr,
        volume_mult,
        retest_window,
    ) = config

    o = data1["opens"]
    h = data1["highs"]
    l = data1["lows"]
    c = data1["closes"]
    t = data1["times"]

    result = blank_result()

    equity = 0.0
    peak = 0.0

    i = max(
        220,
        bisect_left(
            t,
            start_ts
        )
    )

    while i < len(c) - 2:

        if t[i] >= end_ts:
            break

        signal = find_break_retest(
            data1,
            data5,
            lookup5,
            i,
            volume_mult,
            retest_window,
        )

        if signal is None:

            i += 1
            continue

        entry_i = i + 1

        if (
            entry_i >= len(c)
            or t[entry_i]
            >= end_ts
        ):
            break

        # ====================================================
        # ENTRY
        # ====================================================

        raw_entry = (
            o[entry_i]
        )

        if raw_entry <= 0:

            i += 1
            continue

        actual_entry = (
            raw_entry
            * (
                1.0
                + SLIP_SIDE
            )
        )

        qty = (
            STAKE
            / actual_entry
        )

        # ====================================================
        # STOP
        # ====================================================

        atr_stop = (
            actual_entry
            - signal["atr"]
            * sl_atr
        )

        # Structure stop below retest.
        structure_stop = (
            signal["retest_low"]
            * (
                1.0
                - 0.06 / 100.0
            )
        )

        # Never risk more than 0.80%.
        hard_stop = (
            actual_entry
            * (
                1.0
                - 0.80 / 100.0
            )
        )

        stop = max(
            atr_stop,
            structure_stop,
            hard_stop,
        )

        # Avoid stops tighter than 0.25%.
        tightest_allowed = (
            actual_entry
            * (
                1.0
                - 0.25 / 100.0
            )
        )

        stop = min(
            stop,
            tightest_allowed
        )

        target = (
            actual_entry
            * (
                1.0
                + tp_pct / 100.0
            )
        )

        # ====================================================
        # TRADE MANAGEMENT
        # ====================================================

        highest = (
            actual_entry
        )

        lowest = (
            actual_entry
        )

        be_active = False
        trail_active = False

        exit_i = None
        raw_exit = None
        actual_exit = None

        # Up to four hours.
        last_i = min(
            len(c) - 1,
            entry_i + 240
        )

        j = entry_i

        while j <= last_i:

            if t[j] >= end_ts:

                exit_i = max(
                    entry_i,
                    j - 1
                )

                raw_exit = (
                    c[exit_i]
                )

                actual_exit = (
                    raw_exit
                    * (
                        1.0
                        - SLIP_SIDE
                    )
                )

                break

            highest = max(
                highest,
                h[j]
            )

            lowest = min(
                lowest,
                l[j]
            )

            # Conservative:
            # stop checked before TP.
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

                break

            best_profit_pct = (
                highest
                - actual_entry
            ) / actual_entry * 100.0

            # Later break-even.
            if (
                not be_active
                and best_profit_pct
                >= 0.75
            ):

                be_active = True

                stop = max(
                    stop,
                    actual_entry
                    * (
                        1.0
                        + 0.10 / 100.0
                    )
                )

            # Trailing only after
            # a meaningful move.
            if (
                not trail_active
                and best_profit_pct
                >= 1.20
            ):

                trail_active = True

            if trail_active:

                trail_stop = (
                    highest
                    * (
                        1.0
                        - 0.45 / 100.0
                    )
                )

                stop = max(
                    stop,
                    trail_stop
                )

            j += 1

        # ====================================================
        # TIME EXIT
        # ====================================================

        if actual_exit is None:

            exit_i = min(
                last_i,
                len(c) - 1
            )

            while (
                exit_i > entry_i
                and t[exit_i]
                >= end_ts
            ):

                exit_i -= 1

            raw_exit = (
                c[exit_i]
            )

            actual_exit = (
                raw_exit
                * (
                    1.0
                    - SLIP_SIDE
                )
            )

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
            * 100.0
        )

        mae = max(
            0.0,
            (
                actual_entry
                - lowest
            )
            / actual_entry
            * 100.0
        )

        # ====================================================
        # COST BREAKDOWN
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

        net = (
            actual_exit_value
            - actual_entry_value
            - fees
        )

        result["trades"] += 1

        result[
            "gross_market"
        ] += gross_market

        result[
            "slippage"
        ] += slippage_cost

        result[
            "fees"
        ] += fees

        result["net"] += net

        result[
            "mfe_sum"
        ] += mfe

        result[
            "mae_sum"
        ] += mae

        result[
            "mfe_max"
        ] = max(
            result["mfe_max"],
            mfe
        )

        result[
            "mae_max"
        ] = max(
            result["mae_max"],
            mae
        )

        if net > 0:

            result["wins"] += 1

            result[
                "gross_wins"
            ] += net

        else:

            result[
                "gross_losses"
            ] += abs(net)

        equity += net

        peak = max(
            peak,
            equity
        )

        result[
            "max_dd"
        ] = max(
            result["max_dd"],
            peak - equity
        )

        # One position per coin.
        i = max(
            i + 1,
            exit_i + 1
        )

    return result


# ============================================================
# CHECKPOINT
# ============================================================

def signature(
    end_exclusive
):

    return {
        "version": VERSION,
        "days": DAYS,
        "train_days": TRAIN_DAYS,
        "stake": STAKE,
        "fee": FEE_SIDE,
        "slip": SLIP_SIDE,
        "end": end_exclusive,
        "coins": COINS,

        "configs": [
            list(x)
            for x in CONFIGS
        ],
    }


def save_checkpoint(
    sig,
    completed,
    train_totals,
    test_totals,
):

    temp = (
        CHECKPOINT_FILE
        + ".tmp"
    )

    payload = {
        "signature": sig,
        "completed": completed,
        "train_totals":
            train_totals,
        "test_totals":
            test_totals,
    }

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file
        )

    os.replace(
        temp,
        CHECKPOINT_FILE
    )


def load_checkpoint(sig):

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

            data = json.load(file)

        if (
            data.get("signature")
            != sig
        ):

            return None

        return data

    except Exception:
        return None


# ============================================================
# RESULTS
# ============================================================

def write_results(
    train_totals,
    test_totals,
):

    indexes = list(
        range(
            TOTAL_CONFIGS
        )
    )

    # Rank ONLY by TRAIN.
    #
    # First require at least
    # a reasonable number of trades.

    eligible = [
        idx
        for idx in indexes
        if train_totals[
            idx
        ]["trades"] >= 100
    ]

    if not eligible:
        eligible = indexes

    eligible.sort(
        key=lambda idx: (
            train_totals[
                idx
            ]["net"],

            metrics(
                train_totals[idx]
            )[1],
        ),
        reverse=True,
    )

    headers = [
        "rank",

        "tp_pct",
        "sl_atr",
        "volume_mult",
        "retest_window",

        # TEST
        "trades",
        "wins",
        "winrate",
        "net_usdt",
        "profit_factor",
        "max_drawdown",

        "gross_market",
        "fees",
        "slippage",

        "avg_net_trade",
        "trades_per_week",

        "avg_mfe_pct",
        "avg_mae_pct",
        "max_mfe_pct",
        "max_mae_pct",

        # TRAIN
        "train_trades",
        "train_winrate",
        "train_net",
        "train_pf",

        "train_gross",
        "train_fees",
        "train_slippage",

        "train_avg_mfe",
        "train_avg_mae",
    ]

    rows = []

    for rank, idx in enumerate(
        eligible[:10],
        start=1
    ):

        train = (
            train_totals[idx]
        )

        test = (
            test_totals[idx]
        )

        (
            train_wr,
            train_pf,
            train_avg,
            train_mfe,
            train_mae,
        ) = metrics(train)

        (
            test_wr,
            test_pf,
            test_avg,
            test_mfe,
            test_mae,
        ) = metrics(test)

        (
            tp,
            sl,
            vol,
            retest,
        ) = CONFIGS[idx]

        row = {
            "rank":
                rank,

            "tp_pct":
                tp,

            "sl_atr":
                sl,

            "volume_mult":
                vol,

            "retest_window":
                retest,

            "trades":
                test["trades"],

            "wins":
                test["wins"],

            "winrate":
                test_wr,

            "net_usdt":
                test["net"],

            "profit_factor":
                test_pf,

            "max_drawdown":
                test["max_dd"],

            "gross_market":
                test[
                    "gross_market"
                ],

            "fees":
                test["fees"],

            "slippage":
                test["slippage"],

            "avg_net_trade":
                test_avg,

            "trades_per_week":
                test["trades"]
                / (10.0 / 7.0),

            "avg_mfe_pct":
                test_mfe,

            "avg_mae_pct":
                test_mae,

            "max_mfe_pct":
                test["mfe_max"],

            "max_mae_pct":
                test["mae_max"],

            "train_trades":
                train["trades"],

            "train_winrate":
                train_wr,

            "train_net":
                train["net"],

            "train_pf":
                train_pf,

            "train_gross":
                train[
                    "gross_market"
                ],

            "train_fees":
                train["fees"],

            "train_slippage":
                train["slippage"],

            "train_avg_mfe":
                train_mfe,

            "train_avg_mae":
                train_mae,
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
                        float
                    )
                    else value
                )

                for key, value
                in row.items()
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
            dict
        )
    ):

        end_exclusive = int(
            existing[
                "signature"
            ].get(
                "end",
                0
            )
        )

    else:

        end_exclusive = int(
            time.time()
        )

        end_exclusive -= (
            end_exclusive % 60
        )

    sig = signature(
        end_exclusive
    )

    start_ts = (
        end_exclusive
        - DAYS * 86400
    )

    train_end = (
        start_ts
        + TRAIN_DAYS * 86400
    )

    print(
        "======================================",
        flush=True,
    )

    print(
        "MPORBBOT BACKTEST V5",
        flush=True,
    )

    print(
        "SELECTIVE BREAK-RETEST",
        flush=True,
    )

    print(
        "EMA_RECLAIM REMOVED",
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
        "20d TRAIN + 10d untouched TEST",
        flush=True,
    )

    print(
        "MFE / MAE enabled",
        flush=True,
    )

    print(
        "LOW-MEMORY / CHECKPOINT",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    checkpoint = (
        load_checkpoint(sig)
    )

    if checkpoint:

        completed = (
            checkpoint[
                "completed"
            ]
        )

        train_totals = (
            checkpoint[
                "train_totals"
            ]
        )

        test_totals = (
            checkpoint[
                "test_totals"
            ]
        )

        print(
            f"RESUME "
            f"{len(completed)}/25",
            flush=True,
        )

    else:

        completed = []

        train_totals = [
            blank_result()
            for _ in CONFIGS
        ]

        test_totals = [
            blank_result()
            for _ in CONFIGS
        ]

    # ========================================================
    # ONE COIN AT A TIME
    # ========================================================

    for coin_no, symbol in enumerate(
        COINS,
        start=1
    ):

        if symbol in completed:

            print(
                f"{coin_no}/25 "
                f"{symbol}: "
                f"checkpoint OK",
                flush=True,
            )

            continue

        print(
            "",
            flush=True,
        )

        print(
            f"{coin_no}/25 "
            f"{symbol}: "
            f"downloading 1m",
            flush=True,
        )

        rows1 = request_candles(
            symbol,
            "1min",
            start_ts,
            end_exclusive,
        )

        print(
            f"{symbol}: "
            f"{len(rows1)} "
            f"1m candles",
            flush=True,
        )

        print(
            f"{coin_no}/25 "
            f"{symbol}: "
            f"downloading 5m",
            flush=True,
        )

        rows5 = request_candles(
            symbol,
            "5min",
            start_ts,
            end_exclusive,
        )

        print(
            f"{symbol}: "
            f"{len(rows5)} "
            f"5m candles",
            flush=True,
        )

        if (
            len(rows1) < 1000
            or len(rows5) < 300
        ):

            print(
                f"{symbol}: "
                f"too little data",
                flush=True,
            )

            completed.append(
                symbol
            )

            save_checkpoint(
                sig,
                completed,
                train_totals,
                test_totals,
            )

            continue

        data1 = prepare(
            rows1
        )

        data5 = prepare(
            rows5
        )

        lookup5 = (
            build_5m_lookup(
                data5
            )
        )

        del rows1
        del rows5

        # ====================================================
        # CONFIGS
        # ====================================================

        for idx, config in enumerate(
            CONFIGS
        ):

            train_result = simulate(
                data1,
                data5,
                lookup5,
                config,
                start_ts,
                train_end,
            )

            test_result = simulate(
                data1,
                data5,
                lookup5,
                config,
                train_end,
                end_exclusive,
            )

            add_result(
                train_totals[idx],
                train_result,
            )

            add_result(
                test_totals[idx],
                test_result,
            )

            if (
                (idx + 1) % 8 == 0
                or idx + 1
                == TOTAL_CONFIGS
            ):

                (
                    wr,
                    pf,
                    avg,
                    mfe,
                    mae,
                ) = metrics(
                    train_totals[idx]
                )

                print(
                    f"{symbol}: "
                    f"{idx + 1}/"
                    f"{TOTAL_CONFIGS} | "
                    f"TRAIN "
                    f"{train_totals[idx]['net']:+.2f} | "
                    f"{train_totals[idx]['trades']} trades | "
                    f"PF {pf:.2f} | "
                    f"MFE {mfe:.2f}% | "
                    f"MAE {mae:.2f}%",
                    flush=True,
                )

        completed.append(
            symbol
        )

        save_checkpoint(
            sig,
            completed,
            train_totals,
            test_totals,
        )

        eligible_now = [
            idx
            for idx in range(
                TOTAL_CONFIGS
            )
            if train_totals[
                idx
            ]["trades"] >= 10
        ]

        if eligible_now:

            best = max(
                eligible_now,
                key=lambda idx: (
                    train_totals[
                        idx
                    ]["net"],
                    metrics(
                        train_totals[
                            idx
                        ]
                    )[1],
                ),
            )

            (
                wr,
                pf,
                avg,
                mfe,
                mae,
            ) = metrics(
                train_totals[best]
            )

            print(
                f"CHECKPOINT "
                f"{len(completed)}/25 | "
                f"BEST TRAIN "
                f"{train_totals[best]['net']:+.2f} | "
                f"{train_totals[best]['trades']} trades | "
                f"PF {pf:.2f} | "
                f"MFE {mfe:.2f}% | "
                f"MAE {mae:.2f}%",
                flush=True,
            )

        else:

            print(
                f"CHECKPOINT "
                f"{len(completed)}/25",
                flush=True,
            )

        del data1
        del data5
        del lookup5

    # ========================================================
    # FINAL
    # ========================================================

    rows = write_results(
        train_totals,
        test_totals,
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
        "V5 COMPLETE",
        flush=True,
    )

    print(
        "Ranking selected ONLY on TRAIN",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    for row in rows[:5]:

        print(
            f"TRAIN RANK "
            f"#{row['rank']} | "
            f"TRAIN "
            f"{row['train_net']:+.2f} "
            f"PF {row['train_pf']:.2f} | "
            f"TEST "
            f"{row['net_usdt']:+.2f} "
            f"PF "
            f"{row['profit_factor']:.2f}",
            flush=True,
        )

        print(
            f"  TEST "
            f"{row['trades']} trades | "
            f"WR "
            f"{row['winrate']:.1f}% | "
            f"{row['trades_per_week']:.0f}/week",
            flush=True,
        )

        print(
            f"  Gross "
            f"{row['gross_market']:+.2f} | "
            f"fees "
            f"-{row['fees']:.2f} | "
            f"slip "
            f"-{row['slippage']:.2f}",
            flush=True,
        )

        print(
            f"  Avg MFE "
            f"{row['avg_mfe_pct']:.2f}% | "
            f"Avg MAE "
            f"{row['avg_mae_pct']:.2f}%",
            flush=True,
        )

        print(
            f"  TP "
            f"{row['tp_pct']:.2f}% | "
            f"SL "
            f"{row['sl_atr']:.1f} ATR | "
            f"VOL "
            f"{row['volume_mult']:.2f} | "
            f"RETEST "
            f"{row['retest_window']}",
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
