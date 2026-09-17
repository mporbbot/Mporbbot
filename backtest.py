import csv
import json
import os
import time
from bisect import bisect_left
from itertools import product

import requests


# ============================================================
# MPORBBOT BACKTEST V8
#
# NEW STRATEGY:
#   15m = market regime
#   5m  = pullback / reclaim
#   1m  = precise breakout entry
#
# Historical period:
#   approximately 100 -> 65 days ago
#
# 5d warmup
# 20d TRAIN
# 10d HOLDOUT
#
# V7 used approximately 65 -> 30 days ago.
# Therefore V8 uses a DIFFERENT historical period.
# ============================================================

VERSION = "V8"

BASE = "https://api.kucoin.com"

WARMUP_DAYS = 5
TRAIN_DAYS = 20
HOLDOUT_DAYS = 10

WINDOW_DAYS = (
    WARMUP_DAYS
    + TRAIN_DAYS
    + HOLDOUT_DAYS
)

# V8 ends 65 days before today.
EXCLUDE_RECENT_DAYS = 65

STAKE = 100.0

FEE_SIDE = 0.0010
SLIP_SIDE = 0.0002

CHECKPOINT_FILE = "backtest_v8_checkpoint.json"
RESULT_FILE = "backtest_results.csv"
COIN_FILE = "backtest_v8_coins.csv"

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
# V8 PARAMETER GRID
#
# 4 TP
# 3 STOP
# 2 VOLUME
#
# = 24 configs
#
# No aggressive break-even.
# No trailing.
#
# First we want to know if the ENTRY itself has edge.
# ============================================================

TP_VALUES = [
    0.80,
    1.00,
    1.25,
    1.60,
]

STOP_VALUES = [
    0.30,
    0.40,
    0.50,
]

VOLUME_VALUES = [
    1.10,
    1.30,
]

CONFIGS = list(product(
    TP_VALUES,
    STOP_VALUES,
    VOLUME_VALUES,
))

TOTAL_CONFIGS = len(CONFIGS)

# 1m entry breakout
ENTRY_BREAKOUT_BARS = 5

# How far back we search for a 5m pullback
PULLBACK_5M_BARS = 6

# Maximum time in position
MAX_HOLD_MINUTES = 240

# After a trade closes, do not immediately re-enter
COOLDOWN_MINUTES = 20

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
            result[i] = (
                running / period
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

    last_open = (
        end_exclusive
        - seconds
    )

    cursor = last_open

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

                    wait = 3 + attempt * 2

                    print(
                        f"{symbol} {timeframe}: "
                        f"rate limit - wait {wait}s",
                        flush=True,
                    )

                    time.sleep(wait)
                    continue

                response.raise_for_status()

                payload = response.json()

                if payload.get("code") != "200000":
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

                wait = 2 + attempt * 2

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
            chunk_start - seconds
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
            20,
        ),

        "ema50": ema_series(
            closes,
            50,
        ),

        "ema200": ema_series(
            closes,
            200,
        ),

        "atr": atr_series(
            highs,
            lows,
            closes,
            14,
        ),

        "volavg": volume_average(
            volumes,
            20,
        ),
    }


# ============================================================
# CLOSED CANDLE LOOKUP
# ============================================================

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

    # 1m signal candle is closed at signal_open + 60.
    #
    # Only a higher timeframe candle fully closed before
    # that moment may be used.

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
# 15M MARKET REGIME
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

    e20 = data["ema20"][index]
    e50 = data["ema50"][index]
    e200 = data["ema200"][index]

    old20 = data["ema20"][
        index - 3
    ]

    old50 = data["ema50"][
        index - 3
    ]

    if (
        e20 is None
        or e50 is None
        or e200 is None
        or old20 is None
        or old50 is None
        or old20 <= 0
        or old50 <= 0
        or e50 <= 0
    ):
        return False

    close = data["closes"][index]

    slope20 = (
        (e20 - old20)
        / old20
        * 100.0
    )

    slope50 = (
        (e50 - old50)
        / old50
        * 100.0
    )

    separation = (
        (e20 - e50)
        / e50
        * 100.0
    )

    distance_200 = (
        (close - e200)
        / e200
        * 100.0
    )

    return (
        close > e20
        and close > e50
        and close > e200

        and e20 > e50
        and e50 > e200

        and slope20 >= 0.04
        and slope50 >= 0.01

        and separation >= 0.04

        and distance_200 >= 0.10
    )


# ============================================================
# 5M PULLBACK + RECLAIM
# ============================================================

def setup_5m(
    data,
    index,
):

    if (
        index is None
        or index < 205
    ):
        return False

    e20 = data["ema20"][index]
    e50 = data["ema50"][index]

    if (
        e20 is None
        or e50 is None
        or e20 <= 0
        or e50 <= 0
    ):
        return False

    close = data["closes"][index]
    open_ = data["opens"][index]

    # Current 5m candle should have reclaimed EMA20.
    if close <= e20:
        return False

    if close <= open_:
        return False

    # Do not enter when price has already exploded far above EMA20.
    extension = (
        (close - e20)
        / e20
        * 100.0
    )

    if extension > 0.55:
        return False

    # Search recent 5m candles for pullback.
    start = max(
        1,
        index - PULLBACK_5M_BARS,
    )

    touched = False
    deep_failure = False

    for j in range(
        start,
        index + 1,
    ):

        ej20 = data["ema20"][j]
        ej50 = data["ema50"][j]

        if (
            ej20 is None
            or ej50 is None
        ):
            continue

        low = data["lows"][j]
        close_j = data["closes"][j]

        # Pullback touches EMA20 area.
        distance20 = (
            abs(low - ej20)
            / ej20
            * 100.0
        )

        if (
            low <= ej20
            or distance20 <= 0.18
        ):
            touched = True

        # Reject if pullback closes clearly below EMA50.
        if (
            close_j
            < ej50 * (
                1.0 - 0.12 / 100.0
            )
        ):
            deep_failure = True

    if not touched:
        return False

    if deep_failure:
        return False

    # Momentum must be returning.
    if index >= 2:

        if (
            close
            <= data["closes"][index - 1]
        ):
            return False

    return True


# ============================================================
# 1M ENTRY
# ============================================================

def entry_1m(
    data1,
    data5,
    data15,
    lookup5,
    lookup15,
    i,
    volume_mult,
):

    if i < 220:
        return None

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
        ema20[i] is None
        or atr[i] is None
        or volavg[i] is None
        or c[i] <= 0
    ):
        return None

    # ========================================================
    # 15M REGIME
    # ========================================================

    idx15 = get_closed_index(
        times[i],
        900,
        lookup15,
    )

    if not regime_15m(
        data15,
        idx15,
    ):
        return None

    # ========================================================
    # 5M SETUP
    # ========================================================

    idx5 = get_closed_index(
        times[i],
        300,
        lookup5,
    )

    if not setup_5m(
        data5,
        idx5,
    ):
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
        atr_pct < 0.06
        or atr_pct > 0.60
    ):
        return None

    # ========================================================
    # 1M CANDLE QUALITY
    # ========================================================

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
        body / candle_range
    )

    if body_ratio < 0.45:
        return None

    close_location = (
        (c[i] - l[i])
        / candle_range
    )

    if close_location < 0.70:
        return None

    if (
        candle_range
        > atr[i] * 2.0
    ):
        return None

    # ========================================================
    # VOLUME
    # ========================================================

    if (
        v[i]
        < volavg[i] * volume_mult
    ):
        return None

    # ========================================================
    # BREAKOUT
    # ========================================================

    previous_high = max(
        h[
            i - ENTRY_BREAKOUT_BARS:
            i
        ]
    )

    breakout_level = (
        previous_high
        * (
            1.0
            + 0.02 / 100.0
        )
    )

    if c[i] < breakout_level:
        return None

    # Avoid entering if 1m is already extremely extended.
    distance_ema20 = (
        (c[i] - ema20[i])
        / ema20[i]
        * 100.0
    )

    if (
        distance_ema20 < 0
        or distance_ema20 > 0.45
    ):
        return None

    return {
        "atr": atr[i],
    }


# ============================================================
# RESULT HELPERS
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

    else:

        winrate = 0.0
        avg_net = 0.0
        avg_mfe = 0.0
        avg_mae = 0.0

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
        tp_pct,
        stop_pct,
        volume_mult,
    ) = config

    o = data1["opens"]
    h = data1["highs"]
    l = data1["lows"]
    c = data1["closes"]
    t = data1["times"]

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
            volume_mult,
        )

        if signal is None:

            i += 1
            continue

        # ====================================================
        # ENTRY
        #
        # Signal candle closes.
        # Entry at next 1m candle open.
        # Adverse slippage included.
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
            STAKE / actual_entry
        )

        stop = (
            actual_entry
            * (
                1.0
                - stop_pct / 100.0
            )
        )

        target = (
            actual_entry
            * (
                1.0
                + tp_pct / 100.0
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
            entry_i + MAX_HOLD_MINUTES,
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
                    * (1.0 - SLIP_SIDE)
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
            # if both stop and target are touched
            # in the same 1m candle, STOP wins.

            if l[j] <= stop:

                exit_i = j
                raw_exit = stop

                actual_exit = (
                    raw_exit
                    * (1.0 - SLIP_SIDE)
                )

                exit_reason = "STOP"
                break

            if h[j] >= target:

                exit_i = j
                raw_exit = target

                actual_exit = (
                    raw_exit
                    * (1.0 - SLIP_SIDE)
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
                * (1.0 - SLIP_SIDE)
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
            actual_entry_value * FEE_SIDE
            + actual_exit_value * FEE_SIDE
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

        result["fees"] += fees

        result["slippage"] += (
            slippage_cost
        )

        result["net"] += net

        result["mfe_sum"] += mfe
        result["mae_sum"] += mae

        if exit_reason == "TP":
            result["tp_exits"] += 1

        elif exit_reason == "STOP":
            result["stop_exits"] += 1

        else:
            result["time_exits"] += 1

        if net > 0:

            result["wins"] += 1

            result["gross_wins"] += (
                net
            )

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

        # ====================================================
        # COOLDOWN
        # ====================================================

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

def make_signature(download_end):

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
        CHECKPOINT_FILE + ".tmp"
    )

    payload = {
        "signature": signature,
        "completed": completed,
        "train_by_coin": train_by_coin,
        "holdout_by_coin": holdout_by_coin,
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


def load_checkpoint(signature):

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

            payload = json.load(file)

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
# WRITE FINAL RESULTS
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

        # Avoid ranking a config based on
        # just a handful of lucky trades.

        eligible = (
            train["trades"] >= 30
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

    # ========================================================
    # IMPORTANT:
    #
    # HOLDOUT is NOT used to rank configs.
    #
    # Ranking:
    # 1. TRAIN net
    # 2. TRAIN PF
    # 3. number of positive TRAIN coins
    # ========================================================

    candidates.sort(
        key=lambda item: (
            item["train"]["net"],
            item["train_pf"],
            item["train_positive"],
        ),
        reverse=True,
    )

    headers = [
        # Compatible with Telegram result reader
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

        # V8
        "stop_pct",
        "stake",
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
            tp,
            stop_pct,
            volume,
        ) = CONFIGS[idx]

        train = item["train"]
        holdout = item["holdout"]

        tm = metrics(train)
        hm = metrics(holdout)

        row = {
            "rank": rank,

            # HOLDOUT
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

            "tp_pct":
                tp,

            # compatibility only
            "sl_atr":
                0.0,

            "volume_mult":
                volume,

            "lookback":
                ENTRY_BREAKOUT_BARS,

            "stop_pct":
                stop_pct,

            "stake":
                STAKE,

            "gross_market":
                holdout["gross_market"],

            "fees":
                holdout["fees"],

            "slippage":
                holdout["slippage"],

            "avg_net_trade":
                hm["avg_net"],

            "trades_per_week":
                holdout["trades"]
                / (HOLDOUT_DAYS / 7.0),

            "avg_mfe_pct":
                hm["avg_mfe"],

            "avg_mae_pct":
                hm["avg_mae"],

            "tp_exits":
                holdout["tp_exits"],

            "stop_exits":
                holdout["stop_exits"],

            "time_exits":
                holdout["time_exits"],

            "positive_coins":
                item["hold_positive"],

            "negative_coins":
                item["hold_negative"],

            "active_coins":
                item["hold_active"],

            # TRAIN
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
                train["gross_market"],

            "train_fees":
                train["fees"],

            "train_slippage":
                train["slippage"],

            "train_avg_net":
                tm["avg_net"],

            "train_avg_mfe":
                tm["avg_mfe"],

            "train_avg_mae":
                tm["avg_mae"],

            "train_positive_coins":
                item["train_positive"],

            "train_negative_coins":
                item["train_negative"],

            "train_active_coins":
                item["train_active"],
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
                    if isinstance(value, float)
                    else value
                )
                for key, value
                in row.items()
            })

    # ========================================================
    # PER COIN RESULT FOR TRAIN-RANK #1
    # ========================================================

    if rows:

        best_idx = (
            candidates[0]["idx"]
        )

        fields = [
            "symbol",

            "train_trades",
            "train_net",
            "train_pf",
            "train_wr",

            "holdout_trades",
            "holdout_net",
            "holdout_pf",
            "holdout_wr",
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

                    "holdout_trades":
                        hold["trades"],

                    "holdout_net":
                        f"{hold['net']:.6f}",

                    "holdout_pf":
                        f"{hm['pf']:.6f}",

                    "holdout_wr":
                        f"{hm['winrate']:.6f}",
                })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # KEEP SAME HISTORICAL WINDOW ON RESUME
    # ========================================================

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

                existing = json.load(file)

        except Exception:
            existing = None

    if (
        existing
        and isinstance(
            existing.get("signature"),
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

        now -= (
            now % 60
        )

        download_end = (
            now
            - EXCLUDE_RECENT_DAYS * 86400
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
        "MPORBBOT BACKTEST V8",
        flush=True,
    )

    print(
        "15m REGIME -> 5m PULLBACK -> 1m ENTRY",
        flush=True,
    )

    print(
        "DIFFERENT HISTORICAL PERIOD FROM V7",
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
        "TP: 0.80 / 1.00 / 1.25 / 1.60%",
        flush=True,
    )

    print(
        "STOP: 0.30 / 0.40 / 0.50%",
        flush=True,
    )

    print(
        "NO TRAILING / NO EARLY BREAK-EVEN",
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

        completed = (
            checkpoint["completed"]
        )

        train_by_coin = (
            checkpoint[
                "train_by_coin"
            ]
        )

        holdout_by_coin = (
            checkpoint[
                "holdout_by_coin"
            ]
        )

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

        # ====================================================
        # 1M
        # ====================================================

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

        # ====================================================
        # 5M
        # ====================================================

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

        # ====================================================
        # 15M
        # ====================================================

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
                f"{symbol}: insufficient data",
                flush=True,
            )

            train_by_coin[symbol] = [
                blank_result()
                for _ in CONFIGS
            ]

            holdout_by_coin[symbol] = [
                blank_result()
                for _ in CONFIGS
            ]

            completed.append(symbol)

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

        # ====================================================
        # CONFIGS
        # ====================================================

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
                (idx + 1) % 6 == 0
                or idx + 1 == TOTAL_CONFIGS
            ):

                tm = metrics(
                    train_result
                )

                hm = metrics(
                    holdout_result
                )

                print(
                    f"{symbol}: "
                    f"{idx + 1}/{TOTAL_CONFIGS} | "
                    f"TRAIN {train_result['net']:+.2f} "
                    f"PF {tm['pf']:.2f} "
                    f"{train_result['trades']}t | "
                    f"HOLD {holdout_result['net']:+.2f} "
                    f"{holdout_result['trades']}t",
                    flush=True,
                )

        train_by_coin[
            symbol
        ] = train_results

        holdout_by_coin[
            symbol
        ] = holdout_results

        completed.append(symbol)

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
        "V8 COMPLETE",
        flush=True,
    )

    print(
        "TRAIN RANKING - HOLDOUT UNTOUCHED",
        flush=True,
    )

    print(
        "15m -> 5m -> 1m STRATEGY",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    if not rows:

        print(
            "No configuration had enough "
            "trades across enough coins.",
            flush=True,
        )

    for row in rows[:5]:

        print(
            f"TRAIN RANK #{row['rank']} | "
            f"TRAIN {row['train_net']:+.2f} "
            f"PF {row['train_pf']:.2f} | "
            f"HOLDOUT {row['net_usdt']:+.2f} "
            f"PF {row['profit_factor']:.2f}",
            flush=True,
        )

        print(
            f"  HOLDOUT "
            f"{row['trades']} trades | "
            f"WR {row['winrate']:.1f}% | "
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
            f"MAE {row['avg_mae_pct']:.2f}%",
            flush=True,
        )

        print(
            f"  TP {row['tp_pct']:.2f}% | "
            f"STOP {row['stop_pct']:.2f}% | "
            f"VOL {row['volume_mult']:.2f}",
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
