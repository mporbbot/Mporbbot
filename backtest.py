import csv
import json
import os
import time
from bisect import bisect_left
from itertools import product

import requests


# ============================================================
# MPORBBOT BACKTEST V4
# Breakout -> Retest -> Confirmation
# EMA20 Pullback -> Reclaim
# ============================================================

VERSION = "V4"

BASE = "https://api.kucoin.com"

DAYS = 30
TRAIN_DAYS = 20

STAKE = 100.0

FEE_SIDE = 0.0010
SLIP_SIDE = 0.0002

CHECKPOINT_FILE = "backtest_v4_checkpoint.json"
RESULT_FILE = "backtest_results.csv"
SETUP_FILE = "backtest_setup_results.csv"


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
# V4 PARAMETER GRID
#
# 4 TP
# 3 SL
# 2 volume
# 2 breakout lookback
# 2 retest windows
#
# = 96 configs
# ============================================================

TP_VALUES = [
    0.70,
    1.00,
    1.30,
    1.60,
]

SL_ATR_VALUES = [
    0.8,
    1.0,
    1.2,
]

VOLUME_VALUES = [
    1.05,
    1.15,
]

LOOKBACK_VALUES = [
    8,
    12,
]

RETEST_WINDOW_VALUES = [
    4,
    7,
]


CONFIGS = list(product(
    TP_VALUES,
    SL_ATR_VALUES,
    VOLUME_VALUES,
    LOOKBACK_VALUES,
    RETEST_WINDOW_VALUES,
))

TOTAL_CONFIGS = len(CONFIGS)

session = requests.Session()


# ============================================================
# BASIC HELPERS
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

        if i == 0:
            current = value
        else:
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
                highs[i]
                - closes[i - 1]
            ),
            abs(
                lows[i]
                - closes[i - 1]
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
# DOWNLOAD KUCOIN
# ============================================================

def request_candles(
    symbol,
    timeframe,
    start_ts,
    end_exclusive
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
                        2
                        + attempt * 2
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
                    f"{symbol} {timeframe} "
                    f"retry "
                    f"{attempt + 1}/7: "
                    f"{exc}",
                    flush=True,
                )

                time.sleep(
                    2
                    + attempt * 2
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
# PREPARE DATA
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
# 5M TREND
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
    signal_1m_open,
    lookup
):

    signal_close = (
        signal_1m_open
        + 60
    )

    bucket = (
        (
            signal_close
            - 300
        )
        // 300
    ) * 300

    return lookup.get(
        bucket
    )


def trend_ok(
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

    old_e20 = (
        data5["ema20"][
            index - 3
        ]
    )

    if (
        e20 is None
        or e50 is None
        or e200 is None
        or old_e20 is None
        or old_e20 <= 0
    ):
        return False

    close = (
        data5["closes"][index]
    )

    slope_pct = (
        (
            e20 - old_e20
        )
        / old_e20
        * 100.0
    )

    # Stronger trend than V3.
    return (
        e20 > e50
        and close > e20
        and close > e200
        and slope_pct >= 0.020
    )


# ============================================================
# RESULT OBJECTS
# ============================================================

def blank_setup():

    return {
        "trades": 0,
        "wins": 0,

        "gross_market": 0.0,

        "fees": 0.0,
        "slippage": 0.0,

        "net": 0.0,

        "gross_wins": 0.0,
        "gross_losses": 0.0,
    }


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

        "setups": {
            "BREAK_RETEST":
                blank_setup(),

            "EMA_RECLAIM":
                blank_setup(),
        },
    }


def record_trade(
    result,
    setup,
    raw_entry,
    raw_exit,
    actual_entry,
    actual_exit,
    qty
):

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

    result["gross_market"] += (
        gross_market
    )

    result["slippage"] += (
        slippage_cost
    )

    result["fees"] += fees

    result["net"] += net

    setup_result = (
        result["setups"][setup]
    )

    setup_result[
        "trades"
    ] += 1

    setup_result[
        "gross_market"
    ] += gross_market

    setup_result[
        "slippage"
    ] += slippage_cost

    setup_result[
        "fees"
    ] += fees

    setup_result[
        "net"
    ] += net

    if net > 0:

        result["wins"] += 1

        result[
            "gross_wins"
        ] += net

        setup_result[
            "wins"
        ] += 1

        setup_result[
            "gross_wins"
        ] += net

    else:

        loss = abs(net)

        result[
            "gross_losses"
        ] += loss

        setup_result[
            "gross_losses"
        ] += loss

    return net


# ============================================================
# V4 SIGNAL SEARCH
# ============================================================

def find_signal(
    data1,
    data5,
    lookup5,
    i,
    volume_mult,
    lookback,
    retest_window
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

    if (
        i < max(
            220,
            lookback + 2
        )
    ):
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

    if not trend_ok(
        data5,
        idx5
    ):
        return None

    atr_pct = (
        atr[i]
        / c[i]
        * 100.0
    )

    # Don't trade dead markets,
    # but also avoid crazy volatility.
    if (
        atr_pct < 0.06
        or atr_pct > 0.70
    ):
        return None

    candle_range = (
        h[i] - l[i]
    )

    if candle_range <= 0:
        return None

    if (
        candle_range
        > atr[i] * 2.3
    ):
        return None

    body_ratio = (
        abs(
            c[i] - o[i]
        )
        / candle_range
    )

    if body_ratio < 0.20:
        return None

    # ========================================================
    # SETUP 1:
    # BREAKOUT -> RETEST -> CONFIRMATION
    # ========================================================
    #
    # We look backwards from current confirmation candle
    # to see whether a breakout happened recently.

    for breakout_i in range(
        max(
            lookback + 2,
            i - retest_window
        ),
        i
    ):

        if (
            breakout_i
            >= len(c)
        ):
            continue

        if (
            volavg[breakout_i]
            is None
        ):
            continue

        previous_high = max(
            h[
                breakout_i
                - lookback:
                breakout_i
            ]
        )

        breakout_level = (
            previous_high
            * (
                1.0
                + 0.02 / 100.0
            )
        )

        breakout_green = (
            c[breakout_i]
            > o[breakout_i]
        )

        breakout_volume = (
            v[breakout_i]
            >= volavg[breakout_i]
            * volume_mult
        )

        breakout = (
            breakout_green
            and breakout_volume
            and c[breakout_i]
            >= breakout_level
        )

        if not breakout:
            continue

        # Need at least one candle
        # between breakout and confirmation.

        if (
            i
            <= breakout_i + 1
        ):
            continue

        # Retest must happen AFTER breakout.
        retest_found = False
        retest_low = None

        for r in range(
            breakout_i + 1,
            i
        ):

            # Retest tolerance:
            # may dip 0.12% below breakout level.
            lower_limit = (
                previous_high
                * (
                    1.0
                    - 0.12 / 100.0
                )
            )

            upper_limit = (
                previous_high
                * (
                    1.0
                    + 0.18 / 100.0
                )
            )

            touched = (
                l[r] <= upper_limit
                and l[r] >= lower_limit
            )

            # Don't accept a failed breakout
            # with a meaningful close below level.
            held = (
                c[r]
                >= previous_high
                * (
                    1.0
                    - 0.08 / 100.0
                )
            )

            if (
                touched
                and held
            ):

                retest_found = True

                retest_low = l[r]

        if not retest_found:
            continue

        confirmation_green = (
            c[i] > o[i]
        )

        confirmation_strength = (
            c[i] > c[i - 1]
            and c[i] > previous_high
        )

        # Confirmation shouldn't be
        # too stretched above EMA20.
        distance_ema = (
            (
                c[i]
                - ema20[i]
            )
            / ema20[i]
            * 100.0
        )

        not_extended = (
            0 <= distance_ema <= 0.65
        )

        confirmation_volume = (
            v[i]
            >= volavg[i] * 0.90
        )

        if (
            confirmation_green
            and confirmation_strength
            and not_extended
            and confirmation_volume
        ):

            return {
                "setup":
                    "BREAK_RETEST",

                "atr":
                    atr[i],

                "signal_i":
                    i,

                "structure_stop":
                    retest_low,
            }

    # ========================================================
    # SETUP 2:
    # EMA20 PULLBACK -> RECLAIM
    # ========================================================

    if i >= 3:

        # Previous candle must pull
        # into/through EMA20.
        previous_touch = (
            ema20[i - 1]
            is not None
            and l[i - 1]
            <= ema20[i - 1]
            * (
                1.0
                + 0.08 / 100.0
            )
        )

        # But don't allow a deep breakdown.
        previous_held = (
            ema20[i - 1]
            is not None
            and c[i - 1]
            >= ema20[i - 1]
            * (
                1.0
                - 0.18 / 100.0
            )
        )

        confirmation = (
            c[i] > o[i]
            and c[i] > ema20[i]
            and c[i] > c[i - 1]
            and h[i] > h[i - 1]
        )

        confirmation_volume = (
            v[i]
            >= volavg[i]
            * 0.95
        )

        distance_ema = (
            (
                c[i]
                - ema20[i]
            )
            / ema20[i]
            * 100.0
        )

        not_extended = (
            0
            <= distance_ema
            <= 0.40
        )

        # Avoid repeated tiny sideways
        # EMA touches.
        prior_above = (
            ema20[i - 2]
            is not None
            and c[i - 2]
            > ema20[i - 2]
        )

        if (
            previous_touch
            and previous_held
            and confirmation
            and confirmation_volume
            and not_extended
            and prior_above
        ):

            return {
                "setup":
                    "EMA_RECLAIM",

                "atr":
                    atr[i],

                "signal_i":
                    i,

                "structure_stop":
                    l[i - 1],
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
    end_ts
):

    (
        tp_pct,
        sl_atr,
        volume_mult,
        lookback,
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

        signal = find_signal(
            data1,
            data5,
            lookup5,
            i,
            volume_mult,
            lookback,
            retest_window,
        )

        if signal is None:

            i += 1
            continue

        entry_i = i + 1

        if (
            entry_i
            >= len(c)
            or t[entry_i]
            >= end_ts
        ):
            break

        # ====================================================
        # RAW ENTRY
        # ====================================================
        #
        # Used for showing market PnL
        # BEFORE trading friction.

        raw_entry = (
            o[entry_i]
        )

        if raw_entry <= 0:

            i += 1
            continue

        # ====================================================
        # ACTUAL ENTRY
        # ====================================================

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

        atr_stop = (
            actual_entry
            - signal["atr"]
            * sl_atr
        )

        # Max initial risk = 0.65%
        hard_stop = (
            actual_entry
            * (
                1.0
                - 0.65 / 100.0
            )
        )

        # Structure stop is useful
        # for retests/reclaims.
        structure_stop = (
            signal[
                "structure_stop"
            ]
            * (
                1.0
                - 0.05 / 100.0
            )
        )

        stop = max(
            atr_stop,
            hard_stop,
            structure_stop,
        )

        # Don't accept an absurdly tight stop.
        minimum_stop_distance = (
            actual_entry
            * (
                1.0
                - 0.18 / 100.0
            )
        )

        stop = min(
            stop,
            minimum_stop_distance
        )

        target = (
            actual_entry
            * (
                1.0
                + tp_pct / 100.0
            )
        )

        highest = (
            actual_entry
        )

        be_active = False
        trail_active = False

        exit_i = None
        raw_exit = None
        actual_exit = None

        # Give these setups more time
        # than the old scalping model.
        last_i = min(
            len(c) - 1,
            entry_i + 180
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

            # ================================================
            # STOP FIRST
            # ================================================

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

            # ================================================
            # TP
            # ================================================

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

            highest = max(
                highest,
                h[j]
            )

            best_profit = (
                (
                    highest
                    - actual_entry
                )
                / actual_entry
                * 100.0
            )

            # ================================================
            # BREAK EVEN
            #
            # Later than old bot.
            # ================================================

            if (
                not be_active
                and best_profit
                >= 0.55
            ):

                be_active = True

                stop = max(
                    stop,
                    actual_entry
                    * (
                        1.0
                        + 0.08 / 100.0
                    )
                )

            # ================================================
            # TRAILING
            #
            # Don't trail tiny moves.
            # ================================================

            if (
                not trail_active
                and best_profit
                >= 0.85
            ):

                trail_active = True

            if trail_active:

                trail_stop = (
                    highest
                    * (
                        1.0
                        - 0.30 / 100.0
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

        net = record_trade(
            result,
            signal["setup"],
            raw_entry,
            raw_exit,
            actual_entry,
            actual_exit,
            qty,
        )

        equity += net

        peak = max(
            peak,
            equity
        )

        result["max_dd"] = max(
            result["max_dd"],
            peak - equity
        )

        # Only one open trade
        # per symbol at once.
        i = max(
            i + 1,
            exit_i + 1
        )

    return result


# ============================================================
# AGGREGATION
# ============================================================

def add_result(
    total,
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
    ):

        total[key] += (
            source[key]
        )

    total["max_dd"] = max(
        total["max_dd"],
        source["max_dd"]
    )

    for setup in (
        "BREAK_RETEST",
        "EMA_RECLAIM",
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
        ):

            total[
                "setups"
            ][setup][key] += (
                source[
                    "setups"
                ][setup][key]
            )


def metrics(result):

    trades = (
        result["trades"]
    )

    wins = (
        result["wins"]
    )

    winrate = (
        wins
        / trades
        * 100.0
        if trades
        else 0.0
    )

    if (
        result[
            "gross_losses"
        ] > 0
    ):

        pf = (
            result[
                "gross_wins"
            ]
            / result[
                "gross_losses"
            ]
        )

    elif (
        result[
            "gross_wins"
        ] > 0
    ):

        pf = 999.0

    else:

        pf = 0.0

    average = (
        result["net"]
        / trades
        if trades
        else 0.0
    )

    return (
        winrate,
        pf,
        average
    )


# ============================================================
# CHECKPOINT
# ============================================================

def make_signature(
    end_exclusive
):

    return {
        "version":
            VERSION,

        "days":
            DAYS,

        "train_days":
            TRAIN_DAYS,

        "stake":
            STAKE,

        "fee":
            FEE_SIDE,

        "slippage":
            SLIP_SIDE,

        "end_exclusive":
            end_exclusive,

        "coins":
            COINS,

        "configs": [
            list(config)
            for config in CONFIGS
        ],
    }


def save_checkpoint(
    signature,
    completed,
    train_totals,
    test_totals,
):

    temp = (
        CHECKPOINT_FILE
        + ".tmp"
    )

    payload = {
        "signature":
            signature,

        "completed":
            completed,

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
            file,
        )

    os.replace(
        temp,
        CHECKPOINT_FILE,
    )


def load_checkpoint(
    signature
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

            payload = (
                json.load(file)
            )

        if (
            payload.get(
                "signature"
            )
            != signature
        ):

            print(
                "Old V4 checkpoint "
                "does not match. "
                "Starting fresh.",
                flush=True,
            )

            return None

        return payload

    except Exception as exc:

        print(
            f"Checkpoint error: "
            f"{exc}",
            flush=True,
        )

        return None


# ============================================================
# WRITE FINAL RESULTS
# ============================================================

def write_results(
    train_totals,
    test_totals,
):

    ranked = list(
        range(
            TOTAL_CONFIGS
        )
    )

    # IMPORTANT:
    # Ranking is TRAIN only.

    ranked.sort(
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
        "lookback",
        "retest_window",

        # Current Telegram
        # /backtest_result compatibility
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

        "train_trades",
        "train_winrate",
        "train_net",
        "train_pf",
        "train_gross",
        "train_fees",
        "train_slippage",

        "break_retest_trades",
        "break_retest_winrate",
        "break_retest_net",
        "break_retest_pf",

        "ema_reclaim_trades",
        "ema_reclaim_winrate",
        "ema_reclaim_net",
        "ema_reclaim_pf",
    ]

    rows = []

    # Only display top 10
    # selected on TRAIN.

    for rank, idx in enumerate(
        ranked[:10],
        start=1,
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
        ) = metrics(train)

        (
            test_wr,
            test_pf,
            test_avg,
        ) = metrics(test)

        break_result = (
            test["setups"]
            ["BREAK_RETEST"]
        )

        ema_result = (
            test["setups"]
            ["EMA_RECLAIM"]
        )

        (
            break_wr,
            break_pf,
            _
        ) = metrics(
            break_result
        )

        (
            ema_wr,
            ema_pf,
            _
        ) = metrics(
            ema_result
        )

        (
            tp,
            sl,
            vol,
            lookback,
            retest_window,
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

            "lookback":
                lookback,

            "retest_window":
                retest_window,

            # TEST
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
                test[
                    "slippage"
                ],

            "avg_net_trade":
                test_avg,

            "trades_per_week":
                test["trades"]
                / (10.0 / 7.0),

            # TRAIN
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
                train[
                    "slippage"
                ],

            # BREAK RETEST
            "break_retest_trades":
                break_result[
                    "trades"
                ],

            "break_retest_winrate":
                break_wr,

            "break_retest_net":
                break_result[
                    "net"
                ],

            "break_retest_pf":
                break_pf,

            # EMA RECLAIM
            "ema_reclaim_trades":
                ema_result[
                    "trades"
                ],

            "ema_reclaim_winrate":
                ema_wr,

            "ema_reclaim_net":
                ema_result[
                    "net"
                ],

            "ema_reclaim_pf":
                ema_pf,
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
    # SEPARATE SETUP CSV
    # ========================================================

    with open(
        SETUP_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fields = [
            "rank",
            "period",
            "setup",
            "trades",
            "wins",
            "winrate",
            "gross_market",
            "fees",
            "slippage",
            "net_usdt",
            "profit_factor",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for rank, idx in enumerate(
            ranked[:10],
            start=1,
        ):

            for (
                period_name,
                result
            ) in (
                (
                    "TRAIN",
                    train_totals[idx],
                ),
                (
                    "TEST",
                    test_totals[idx],
                ),
            ):

                for setup in (
                    "BREAK_RETEST",
                    "EMA_RECLAIM",
                ):

                    s = (
                        result[
                            "setups"
                        ][setup]
                    )

                    wr, pf, _ = (
                        metrics(s)
                    )

                    writer.writerow({
                        "rank":
                            rank,

                        "period":
                            period_name,

                        "setup":
                            setup,

                        "trades":
                            s["trades"],

                        "wins":
                            s["wins"],

                        "winrate":
                            f"{wr:.6f}",

                        "gross_market":
                            f"{s['gross_market']:.6f}",

                        "fees":
                            f"{s['fees']:.6f}",

                        "slippage":
                            f"{s['slippage']:.6f}",

                        "net_usdt":
                            f"{s['net']:.6f}",

                        "profit_factor":
                            f"{pf:.6f}",
                    })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    # If V4 checkpoint exists,
    # preserve exactly the same
    # historical window.

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

                existing = (
                    json.load(file)
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

        end_exclusive = int(
            existing[
                "signature"
            ].get(
                "end_exclusive",
                0,
            )
        )

    else:

        end_exclusive = int(
            time.time()
        )

        end_exclusive -= (
            end_exclusive
            % 60
        )

    signature = (
        make_signature(
            end_exclusive
        )
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
        "MPORBBOT BACKTEST V4",
        flush=True,
    )

    print(
        "BREAK -> RETEST -> CONFIRM",
        flush=True,
    )

    print(
        "EMA20 PULLBACK -> RECLAIM",
        flush=True,
    )

    print(
        "LOW-MEMORY / CHECKPOINT",
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
        "Fees + slippage included",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    checkpoint = (
        load_checkpoint(
            signature
        )
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
        start=1,
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
                signature,
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
        # 96 CONFIGS
        # ====================================================

        for idx, config in enumerate(
            CONFIGS
        ):

            train_result = (
                simulate(
                    data1,
                    data5,
                    lookup5,
                    config,
                    start_ts,
                    train_end,
                )
            )

            test_result = (
                simulate(
                    data1,
                    data5,
                    lookup5,
                    config,
                    train_end,
                    end_exclusive,
                )
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
                (idx + 1) % 12 == 0
                or idx + 1
                == TOTAL_CONFIGS
            ):

                wr, pf, avg = (
                    metrics(
                        train_totals[
                            idx
                        ]
                    )
                )

                print(
                    f"{symbol}: "
                    f"{idx + 1}/"
                    f"{TOTAL_CONFIGS} "
                    f"configs | "
                    f"TRAIN "
                    f"{train_totals[idx]['net']:+.2f} | "
                    f"WR {wr:.1f}% | "
                    f"PF {pf:.2f}",
                    flush=True,
                )

        completed.append(
            symbol
        )

        save_checkpoint(
            signature,
            completed,
            train_totals,
            test_totals,
        )

        best_idx = max(
            range(
                TOTAL_CONFIGS
            ),
            key=lambda x: (
                train_totals[
                    x
                ]["net"],

                metrics(
                    train_totals[x]
                )[1],
            ),
        )

        best_wr, best_pf, _ = (
            metrics(
                train_totals[
                    best_idx
                ]
            )
        )

        print(
            f"CHECKPOINT "
            f"{len(completed)}/25 | "
            f"Best TRAIN "
            f"{train_totals[best_idx]['net']:+.2f} | "
            f"WR {best_wr:.1f}% | "
            f"PF {best_pf:.2f}",
            flush=True,
        )

        del data1
        del data5
        del lookup5

    # ========================================================
    # FINISHED
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
        "V4 COMPLETE",
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
            f"{row['profit_factor']:.2f} | "
            f"WR "
            f"{row['winrate']:.1f}% | "
            f"{row['trades']} trades",
            flush=True,
        )

        print(
            f"  TEST gross "
            f"{row['gross_market']:+.2f} | "
            f"fees "
            f"-{row['fees']:.2f} | "
            f"slip "
            f"-{row['slippage']:.2f}",
            flush=True,
        )

        print(
            f"  BREAK_RETEST "
            f"{row['break_retest_net']:+.2f} "
            f"({row['break_retest_trades']}) | "
            f"EMA_RECLAIM "
            f"{row['ema_reclaim_net']:+.2f} "
            f"({row['ema_reclaim_trades']})",
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
