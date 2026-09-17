import csv
import json
import os
import time
from bisect import bisect_left
from itertools import product

import requests


# ============================================================
# MPORBBOT BACKTEST V7
#
# IMPORTANT:
# Previous V3-V6 tests used approximately the latest 30 days.
#
# V7 deliberately DOES NOT use those days.
#
# Download window:
#   ~65 days ago -> ~30 days ago
#
# Inside that historical window:
#   5 days WARMUP
#   20 days TRAIN
#   10 days HOLDOUT
#
# HOLDOUT is never used for ranking/config selection.
#
# Strategy:
# 5m strong trend
# -> 1m breakout
# -> retest
# -> confirmation
# -> LONG
#
# No per-coin cherry picking.
# All coins participate.
# ============================================================

VERSION = "V7"

BASE = "https://api.kucoin.com"

WARMUP_DAYS = 5
TRAIN_DAYS = 20
HOLDOUT_DAYS = 10

WINDOW_DAYS = (
    WARMUP_DAYS
    + TRAIN_DAYS
    + HOLDOUT_DAYS
)

# End historical test 30 days before now.
EXCLUDE_RECENT_DAYS = 30

STAKE = 100.0

FEE_SIDE = 0.0010
SLIP_SIDE = 0.0002

CHECKPOINT_FILE = "backtest_v7_checkpoint.json"
RESULT_FILE = "backtest_results.csv"
COIN_FILE = "backtest_v7_coins.csv"

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
# V7 GRID
#
# V5/V6 showed approximately:
# Avg MFE ~0.49-0.50%
# Avg MAE ~0.13-0.28%
#
# Therefore V7 tests exits around the actual observed move.
#
# 4 TP
# 3 STOP
# 2 VOLUME
# 2 BREAK-EVEN
#
# = 48 configs
# ============================================================

TP_VALUES = [
    0.35,
    0.45,
    0.55,
    0.65,
]

STOP_VALUES = [
    0.20,
    0.25,
    0.30,
]

VOLUME_VALUES = [
    1.20,
    1.40,
]

BE_TRIGGER_VALUES = [
    0.35,
    0.45,
]

CONFIGS = list(product(
    TP_VALUES,
    STOP_VALUES,
    VOLUME_VALUES,
    BE_TRIGGER_VALUES,
))

TOTAL_CONFIGS = len(CONFIGS)

BREAKOUT_LOOKBACK = 12
RETEST_WINDOW = 6

# Break-even stop is deliberately above entry.
# With 0.10% fee each side + slippage, ordinary "entry price"
# break-even is still a losing trade.
BE_LOCK_PCT = 0.22

MAX_HOLD_MINUTES = 90

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
            result[i] = (
                running / period
            )

    return result


def previous_volume_average(
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
# KUCOIN DOWNLOAD
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
                        f"rate limit, wait {wait}s",
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
                f"Download failed: "
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

        "volavg": previous_volume_average(
            volumes,
            20,
        ),
    }


# ============================================================
# 5M TREND
# ============================================================

def build_5m_lookup(data5):

    return {
        ts: i
        for i, ts
        in enumerate(data5["times"])
    }


def get_closed_5m_index(
    signal_open,
    lookup5,
):

    # A 1m candle opening at t is closed at t+60.
    # Only use a 5m candle that is fully closed by then.

    signal_close = (
        signal_open + 60
    )

    bucket = (
        (
            signal_close - 300
        )
        // 300
    ) * 300

    return lookup5.get(bucket)


def strong_trend(
    data5,
    index,
):

    if (
        index is None
        or index < 205
    ):
        return False

    e20 = data5["ema20"][index]
    e50 = data5["ema50"][index]
    e200 = data5["ema200"][index]

    old20 = data5["ema20"][
        index - 4
    ]

    old50 = data5["ema50"][
        index - 4
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

    close = data5["closes"][index]

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
# RESULTS
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
    ):

        target[key] += source[key]

    # This is worst single-coin DD,
    # not exact portfolio DD.
    target["max_dd"] = max(
        target["max_dd"],
        source["max_dd"],
    )


def metrics(result):

    trades = result["trades"]

    if trades:

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
        "pf": pf,
        "avg_net": avg_net,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
    }


# ============================================================
# ENTRY
# ============================================================

def find_signal(
    data1,
    data5,
    lookup5,
    i,
    volume_mult,
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
        or ema20[i] <= 0
    ):
        return None

    idx5 = get_closed_5m_index(
        times[i],
        lookup5,
    )

    if not strong_trend(
        data5,
        idx5,
    ):
        return None

    # ========================================================
    # CONFIRMATION CANDLE QUALITY
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
        body / candle_range
    )

    if body_ratio < 0.50:
        return None

    close_position = (
        (c[i] - l[i])
        / candle_range
    )

    if close_position < 0.72:
        return None

    if (
        candle_range
        > atr[i] * 1.8
    ):
        return None

    if (
        v[i]
        < volavg[i] * 1.05
    ):
        return None

    ema_distance = (
        (c[i] - ema20[i])
        / ema20[i]
        * 100.0
    )

    if (
        ema_distance < 0
        or ema_distance > 0.45
    ):
        return None

    # ========================================================
    # FIND BREAKOUT
    # ========================================================

    first_breakout = max(
        BREAKOUT_LOOKBACK + 2,
        i - RETEST_WINDOW - 1,
    )

    last_breakout = (
        i - 2
    )

    for b in range(
        first_breakout,
        last_breakout + 1,
    ):

        if (
            volavg[b] is None
            or atr[b] is None
        ):
            continue

        previous_high = max(
            h[
                b - BREAKOUT_LOOKBACK:
                b
            ]
        )

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

        if b_body <= 0:
            continue

        b_body_ratio = (
            b_body / b_range
        )

        if (
            b_body_ratio < 0.45
            or c[b] < breakout_level
            or v[b]
            < volavg[b] * volume_mult
        ):
            continue

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
            i,
        ):

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

            low_distance = (
                (l[r] - previous_high)
                / previous_high
                * 100.0
            )

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

        bounce = (
            (c[i] - retest_low)
            / retest_low
            * 100.0
        )

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
        stop_pct,
        volume_mult,
        be_trigger,
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

        signal = find_signal(
            data1,
            data5,
            lookup5,
            i,
            volume_mult,
        )

        if signal is None:

            i += 1
            continue

        # Entry at next candle open.
        # This approximates the live bot detecting the
        # closed signal candle and entering immediately after.

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

        # ====================================================
        # FIXED INITIAL STOP
        #
        # Directly test 0.20 / 0.25 / 0.30%.
        # ====================================================

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

        be_active = False

        exit_i = None
        raw_exit = None
        actual_exit = None

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

                break

            highest = max(
                highest,
                h[j],
            )

            lowest = min(
                lowest,
                l[j],
            )

            # Conservative same-candle ordering:
            # old stop is checked before TP.

            if l[j] <= stop:

                exit_i = j
                raw_exit = stop

                actual_exit = (
                    raw_exit
                    * (1.0 - SLIP_SIDE)
                )

                break

            if h[j] >= target:

                exit_i = j
                raw_exit = target

                actual_exit = (
                    raw_exit
                    * (1.0 - SLIP_SIDE)
                )

                break

            best_profit_pct = (
                (highest - actual_entry)
                / actual_entry
                * 100.0
            )

            # Break-even update becomes effective
            # after current candle's stop/TP check.
            # This avoids pretending we know intrabar order.

            if (
                not be_active
                and best_profit_pct
                >= be_trigger
            ):

                be_active = True

                stop = max(
                    stop,
                    actual_entry
                    * (
                        1.0
                        + BE_LOCK_PCT / 100.0
                    ),
                )

            j += 1

        # ====================================================
        # TIME EXIT
        # ====================================================

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

        # ====================================================
        # MFE / MAE
        # ====================================================

        mfe = max(
            0.0,
            (
                highest - actual_entry
            )
            / actual_entry
            * 100.0,
        )

        mae = max(
            0.0,
            (
                actual_entry - lowest
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

        # Only one open position per coin.
        i = max(
            i + 1,
            exit_i + 1,
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
        "fee": FEE_SIDE,
        "slip": SLIP_SIDE,
        "coins": COINS,
        "configs": [
            list(x)
            for x in CONFIGS
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
# AGGREGATE ALL COINS
# ============================================================

def aggregate_config(
    by_coin,
    config_index,
):

    total = blank_result()

    positive_coins = 0
    negative_coins = 0
    flat_coins = 0

    coins_with_trades = 0

    for symbol in COINS:

        if symbol not in by_coin:
            continue

        result = (
            by_coin[symbol][config_index]
        )

        add_result(
            total,
            result,
        )

        if result["trades"] > 0:

            coins_with_trades += 1

            if result["net"] > 0:
                positive_coins += 1

            elif result["net"] < 0:
                negative_coins += 1

            else:
                flat_coins += 1

    return (
        total,
        positive_coins,
        negative_coins,
        flat_coins,
        coins_with_trades,
    )


# ============================================================
# RESULTS
# ============================================================

def write_results(
    train_by_coin,
    holdout_by_coin,
):

    candidates = []

    for idx, config in enumerate(CONFIGS):

        (
            train,
            train_positive,
            train_negative,
            train_flat,
            train_active,
        ) = aggregate_config(
            train_by_coin,
            idx,
        )

        train_m = metrics(train)

        # Robustness requirements.
        #
        # We do NOT require TRAIN to be profitable here,
        # because that could leave us with tiny samples again.
        #
        # But there must be enough activity across several coins.

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
            hold_flat,
            hold_active,
        ) = aggregate_config(
            holdout_by_coin,
            idx,
        )

        candidates.append({
            "idx": idx,
            "train": train,
            "holdout": holdout,

            "train_positive": train_positive,
            "train_negative": train_negative,
            "train_active": train_active,

            "hold_positive": hold_positive,
            "hold_negative": hold_negative,
            "hold_active": hold_active,

            "train_pf": train_m["pf"],
        })

    # ========================================================
    # RANK ONLY USING TRAIN
    #
    # First net PnL,
    # then PF,
    # then number of profitable coins.
    #
    # HOLDOUT is NOT part of sorting.
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
        # Compatibility with current Telegram reader
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

        # V7
        "stop_pct",
        "be_trigger",
        "stake",

        "gross_market",
        "fees",
        "slippage",
        "avg_net_trade",
        "trades_per_week",
        "avg_mfe_pct",
        "avg_mae_pct",

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
            be_trigger,
        ) = CONFIGS[idx]

        train = item["train"]
        holdout = item["holdout"]

        tm = metrics(train)
        hm = metrics(holdout)

        row = {
            "rank": rank,

            # HOLDOUT result
            "trades": holdout["trades"],
            "wins": holdout["wins"],
            "winrate": hm["winrate"],
            "net_usdt": holdout["net"],
            "profit_factor": hm["pf"],
            "max_drawdown": holdout["max_dd"],

            "tp_pct": tp,

            # Compatibility field only.
            # V7 uses fixed % stop instead of ATR stop.
            "sl_atr": 0.0,

            "volume_mult": volume,
            "lookback": BREAKOUT_LOOKBACK,

            "stop_pct": stop_pct,
            "be_trigger": be_trigger,
            "stake": STAKE,

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
    # PER-COIN FILE FOR TOP TRAIN CONFIG
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
                    "symbol": symbol,

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
    # FIX HISTORICAL WINDOW
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
        "MPORBBOT BACKTEST V7",
        flush=True,
    )

    print(
        "HISTORICAL HOLDOUT TEST",
        flush=True,
    )

    print(
        "RECENT 30 DAYS EXCLUDED",
        flush=True,
    )

    print(
        "NO COIN CHERRY PICKING",
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
        "TP: 0.35 / 0.45 / 0.55 / 0.65%",
        flush=True,
    )

    print(
        "STOP: 0.20 / 0.25 / 0.30%",
        flush=True,
    )

    print(
        "HOLDOUT NOT USED FOR RANKING",
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
    # PROCESS ONE COIN AT A TIME
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
            f"{symbol}: downloading historical 1m",
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
            f"{symbol}: downloading historical 5m",
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

        if (
            len(rows1) < 1000
            or len(rows5) < 300
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

        lookup5 = build_5m_lookup(
            data5
        )

        del rows1
        del rows5

        train_results = []
        holdout_results = []

        for idx, config in enumerate(
            CONFIGS
        ):

            train_result = simulate(
                data1,
                data5,
                lookup5,
                config,
                train_start,
                train_end,
            )

            holdout_result = simulate(
                data1,
                data5,
                lookup5,
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
                (idx + 1) % 8 == 0
                or idx + 1 == TOTAL_CONFIGS
            ):

                tm = metrics(
                    train_result
                )

                print(
                    f"{symbol}: "
                    f"{idx + 1}/{TOTAL_CONFIGS} | "
                    f"TRAIN "
                    f"{train_result['net']:+.2f} | "
                    f"{train_result['trades']} trades | "
                    f"PF {tm['pf']:.2f} | "
                    f"MFE {tm['avg_mfe']:.2f}% | "
                    f"MAE {tm['avg_mae']:.2f}%",
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
        del lookup5

    # ========================================================
    # FINAL
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
        "V7 COMPLETE",
        flush=True,
    )

    print(
        "RANKING BASED ON TRAIN ONLY",
        flush=True,
    )

    print(
        "HOLDOUT = OLDER UNSEEN PERIOD",
        flush=True,
    )

    print(
        "ALL COINS INCLUDED",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    if not rows:

        print(
            "No config had enough TRAIN trades "
            "across enough coins.",
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
            f"VOL {row['volume_mult']:.2f} | "
            f"BE {row['be_trigger']:.2f}%",
            flush=True,
        )

        print(
            f"  HOLDOUT COINS: "
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
