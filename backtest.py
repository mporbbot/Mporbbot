import os
import csv
import zipfile
from bisect import bisect_left, bisect_right
from itertools import product
from pathlib import PurePosixPath

import requests


# ============================================================
# MPORBBOT V10.2 KRAKEN
#
# LONG ONLY
#
# 15m regime
# -> 5m oversold / mean-reversion setup
# -> 1m reversal
# -> post-only style maker entry
#
# IMPORTANT:
# 5m EMA20 is NO LONGER the profit target.
#
# We test fixed targets:
# 1.50 / 2.00 / 2.50 / 3.00 %
#
# Stops:
# 0.80 / 1.00 / 1.20 %
#
# 12 configurations.
#
# Ranking = TRAIN ONLY.
# HOLDOUT remains untouched.
# ============================================================

VERSION = "V10.2_KRAKEN_FIXED_TARGETS"

STAKE = 100.0

# Conservative Kraken Tier 1 spot crypto fees
MAKER_FEE = 0.0040       # 0.40%
TAKER_FEE = 0.0080       # 0.80%

# Only applied to taker exits
TAKER_SLIPPAGE = 0.0005  # 0.05%

# Maker entry
ENTRY_LIMIT_OFFSET = 0.0005  # 0.05% below signal close
ENTRY_WAIT_MINUTES = 3

# Give larger targets time to develop
MAX_HOLD_MINUTES = 360

# Prevent immediate repeated entries after a closed trade
COOLDOWN_MINUTES = 20


# ============================================================
# ENTRY PARAMETERS
#
# Fixed from V10.1 diagnostic.
# We are NOT optimising entry yet.
# ============================================================

RSI_LIMIT = 40.0
DEVIATION_ATR = 0.40
VOLUME_MULT = 0.80


# ============================================================
# EXIT GRID
# ============================================================

TP_VALUES = [
    1.50,
    2.00,
    2.50,
    3.00,
]

STOP_VALUES = [
    0.80,
    1.00,
    1.20,
]

CONFIGS = list(
    product(
        TP_VALUES,
        STOP_VALUES,
    )
)

TOTAL_CONFIGS = len(CONFIGS)


# ============================================================
# KRAKEN ARCHIVE
# ============================================================

ARCHIVE_FILE = "Kraken_OHLCVT_2026Q2.zip"

ARCHIVE_URL = (
    "https://assets.kraken.com/marketing/institutions/"
    "Kraken_OHLCVT_2026Q2.zip"
)

RESULT_FILE = "backtest_results.csv"
COIN_FILE = "backtest_v10_2_coins.csv"


# ============================================================
# SAME PERIOD AS V10 / V10.1
#
# Warmup: Apr 1 -> Apr 6
# TRAIN:  Apr 6 -> Apr 26
# HOLD:   Apr 26 -> May 6
# ============================================================

DATA_START = 1775001600

TRAIN_START = 1775433600
TRAIN_END = 1777161600

HOLDOUT_START = TRAIN_END
HOLDOUT_END = 1778025600


# ============================================================
# ASSETS
# ============================================================

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "LINK",
    "AVAX",
    "LTC",
    "NEAR",
    "APT",
    "SUI",
    "DOT",
    "TRX",
    "BCH",
    "UNI",
    "FIL",
    "ARB",
    "OP",
    "ATOM",
    "INJ",
    "AAVE",
    "ETC",
    "ICP",
]


ALIASES = {
    "BTC": ["XBT", "BTC"],
    "DOGE": ["XDG", "DOGE"],
    "ETH": ["ETH", "XETH"],
    "XRP": ["XRP", "XXRP"],
    "LTC": ["LTC", "XLTC"],
    "ETC": ["ETC", "XETC"],
}


# ============================================================
# DOWNLOAD ARCHIVE
# ============================================================

def download_archive():

    if os.path.exists(ARCHIVE_FILE):

        size_mb = (
            os.path.getsize(ARCHIVE_FILE)
            / 1024
            / 1024
        )

        print(
            f"Kraken archive exists: {size_mb:.1f} MB",
            flush=True,
        )

        return

    print(
        "Downloading Kraken Q2 2026 OHLCVT...",
        flush=True,
    )

    temp = ARCHIVE_FILE + ".part"

    with requests.get(
        ARCHIVE_URL,
        stream=True,
        timeout=120,
    ) as response:

        response.raise_for_status()

        downloaded = 0
        last_print = 0

        with open(temp, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                f.write(chunk)

                downloaded += len(chunk)

                if (
                    downloaded - last_print
                    >= 20 * 1024 * 1024
                ):

                    last_print = downloaded

                    print(
                        f"Downloaded "
                        f"{downloaded / 1024 / 1024:.0f} MB",
                        flush=True,
                    )

    os.replace(
        temp,
        ARCHIVE_FILE,
    )

    print(
        "Download complete.",
        flush=True,
    )


# ============================================================
# ZIP HELPERS
# ============================================================

def basename(name):

    return PurePosixPath(
        name
    ).name.upper()


def build_zip_index(zf):

    result = {}

    for name in zf.namelist():

        b = basename(name)

        if b.endswith(".CSV"):

            result[b] = name

    return result


def resolve_pair(
    asset,
    zip_index,
):

    aliases = ALIASES.get(
        asset,
        [asset],
    )

    # Prefer USD, USDT fallback
    for quote in [
        "USD",
        "USDT",
    ]:

        for alias in aliases:

            key = (
                f"{alias}{quote}_1.CSV"
            )

            if key in zip_index:

                return (
                    f"{alias}{quote}"
                )

    return None


def load_csv(
    zf,
    zip_index,
    pair,
    interval,
):

    key = (
        f"{pair}_{interval}.CSV"
    )

    filename = zip_index.get(
        key
    )

    if filename is None:

        return []

    rows = []

    with zf.open(filename) as f:

        for raw in f:

            try:

                parts = (
                    raw.decode("utf-8")
                    .strip()
                    .split(",")
                )

                if len(parts) < 7:

                    continue

                ts = int(
                    float(parts[0])
                )

                if not (
                    DATA_START
                    <= ts
                    < HOLDOUT_END
                ):

                    continue

                rows.append([
                    ts,
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    int(float(parts[6])),
                ])

            except Exception:

                continue

    rows.sort(
        key=lambda x: x[0]
    )

    return rows


# ============================================================
# EMA
# ============================================================

def ema_series(
    values,
    period,
):

    result = [
        None
    ] * len(values)

    if not values:

        return result

    alpha = (
        2.0
        / (period + 1.0)
    )

    current = values[0]

    for i, value in enumerate(
        values
    ):

        if i > 0:

            current = (
                value * alpha
                + current
                * (1.0 - alpha)
            )

        if i >= period - 1:

            result[i] = current

    return result


# ============================================================
# ATR
# ============================================================

def atr_series(
    highs,
    lows,
    closes,
    period=14,
):

    n = len(closes)

    result = [None] * n
    tr = [0.0] * n

    for i in range(
        1,
        n,
    ):

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

    for i in range(
        1,
        n,
    ):

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


# ============================================================
# RSI
# ============================================================

def rsi_series(
    closes,
    period=14,
):

    n = len(closes)

    result = [None] * n

    if n <= period:

        return result

    gains = [0.0] * n
    losses = [0.0] * n

    for i in range(
        1,
        n,
    ):

        change = (
            closes[i]
            - closes[i - 1]
        )

        if change > 0:

            gains[i] = change

        elif change < 0:

            losses[i] = -change

    avg_gain = (
        sum(
            gains[
                1:period + 1
            ]
        )
        / period
    )

    avg_loss = (
        sum(
            losses[
                1:period + 1
            ]
        )
        / period
    )

    if avg_loss == 0:

        result[period] = 100.0

    else:

        rs = (
            avg_gain
            / avg_loss
        )

        result[period] = (
            100.0
            - 100.0
            / (1.0 + rs)
        )

    for i in range(
        period + 1,
        n,
    ):

        avg_gain = (
            avg_gain
            * (period - 1)
            + gains[i]
        ) / period

        avg_loss = (
            avg_loss
            * (period - 1)
            + losses[i]
        ) / period

        if avg_loss == 0:

            result[i] = 100.0

        else:

            rs = (
                avg_gain
                / avg_loss
            )

            result[i] = (
                100.0
                - 100.0
                / (1.0 + rs)
            )

    return result


# ============================================================
# VOLUME AVERAGE
# ============================================================

def volume_average(
    volumes,
    period=20,
):

    result = [
        None
    ] * len(volumes)

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
            - prefix[
                i - period
            ]
        ) / period

    return result


# ============================================================
# BOLLINGER LOWER
# ============================================================

def bollinger_lower(
    closes,
    period=20,
):

    result = [
        None
    ] * len(closes)

    for i in range(
        period - 1,
        len(closes),
    ):

        window = closes[
            i - period + 1:
            i + 1
        ]

        mean = (
            sum(window)
            / period
        )

        variance = (
            sum(
                (x - mean) ** 2
                for x in window
            )
            / period
        )

        std = (
            variance ** 0.5
        )

        result[i] = (
            mean
            - 2.0 * std
        )

    return result


# ============================================================
# PREPARE DATA
# ============================================================

def prepare(rows):

    times = [
        x[0]
        for x in rows
    ]

    opens = [
        x[1]
        for x in rows
    ]

    highs = [
        x[2]
        for x in rows
    ]

    lows = [
        x[3]
        for x in rows
    ]

    closes = [
        x[4]
        for x in rows
    ]

    volumes = [
        x[5]
        for x in rows
    ]

    return {

        "times": times,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
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

        "bb_low":
            bollinger_lower(
                closes,
                20,
            ),
    }


# ============================================================
# LAST FULLY CLOSED HIGHER-TIMEFRAME CANDLE
# ============================================================

def closed_index(
    data,
    signal_open,
    timeframe_seconds,
):

    signal_close = (
        signal_open + 60
    )

    latest_open = (
        signal_close
        - timeframe_seconds
    )

    i = (
        bisect_right(
            data["times"],
            latest_open,
        )
        - 1
    )

    if i < 0:

        return None

    return i


# ============================================================
# 15M REGIME
# ============================================================

def regime_15m(
    data,
    i,
):

    if (
        i is None
        or i < 205
    ):

        return False

    close = data[
        "closes"
    ][i]

    e20 = data[
        "ema20"
    ][i]

    e50 = data[
        "ema50"
    ][i]

    e200 = data[
        "ema200"
    ][i]

    old50 = data[
        "ema50"
    ][i - 4]

    old200 = data[
        "ema200"
    ][i - 4]

    if any(
        x is None
        for x in (
            e20,
            e50,
            e200,
            old50,
            old200,
        )
    ):

        return False

    if (
        e50 <= 0
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

    bullish = (
        e50 > e200
    )

    recovering = (
        e20 > e50
        and close
        > e200 * 0.99
    )

    if not (
        bullish
        or recovering
    ):

        return False

    if slope50 < -0.08:

        return False

    if slope200 < -0.04:

        return False

    if close < (
        e200 * 0.975
    ):

        return False

    extension = (
        (close - e200)
        / e200
        * 100.0
    )

    if extension > 15.0:

        return False

    return True


# ============================================================
# 5M OVERSOLD / MEAN-REVERSION SETUP
# ============================================================

def setup_5m(
    data,
    i,
):

    if (
        i is None
        or i < 205
    ):

        return False

    t = data["times"]
    o = data["opens"]
    h = data["highs"]
    l = data["lows"]
    c = data["closes"]
    v = data["volumes"]

    e20 = data["ema20"]
    e50 = data["ema50"]

    atr = data["atr"]
    rsi = data["rsi"]

    volavg = data[
        "volavg"
    ]

    bb = data[
        "bb_low"
    ]

    oversold_i = None

    start = max(
        205,
        i - 6,
    )

    for j in range(
        start,
        i + 1,
    ):

        # Avoid treating a signal behind a large data gap
        # as recent.
        if (
            t[i] - t[j]
            > 35 * 60
        ):

            continue

        values = (
            e20[j],
            e50[j],
            atr[j],
            rsi[j],
            volavg[j],
            bb[j],
        )

        if any(
            x is None
            for x in values
        ):

            continue

        if atr[j] <= 0:

            continue

        deviation = (
            e20[j] - l[j]
        ) / atr[j]

        bb_hit = (
            l[j]
            <= bb[j] * 1.001
        )

        deviation_hit = (
            deviation
            >= DEVIATION_ATR
        )

        if (
            rsi[j] <= RSI_LIMIT
            and (
                bb_hit
                or deviation_hit
            )
            and v[j]
            >= volavg[j]
            * VOLUME_MULT
        ):

            oversold_i = j

    if oversold_i is None:

        return False

    if any(
        x is None
        for x in (
            e20[i],
            e50[i],
            rsi[i],
            rsi[i - 1],
        )
    ):

        return False

    recovery = 0

    if c[i] > o[i]:

        recovery += 1

    if c[i] > c[i - 1]:

        recovery += 1

    if rsi[i] > rsi[i - 1]:

        recovery += 1

    previous_mid = (
        h[i - 1]
        + l[i - 1]
    ) / 2.0

    if c[i] > previous_mid:

        recovery += 1

    if recovery < 2:

        return False

    # Reject structural collapse
    if c[i] < (
        e50[i] * 0.97
    ):

        return False

    # Don't chase a move already far above short mean
    if c[i] > (
        e20[i] * 1.003
    ):

        return False

    return True


# ============================================================
# 1M REVERSAL
# ============================================================

def reversal_1m(
    data,
    i,
):

    if i < 220:

        return False

    o = data["opens"]
    h = data["highs"]
    l = data["lows"]
    c = data["closes"]
    v = data["volumes"]

    atr = data["atr"]
    rsi = data["rsi"]

    va = data[
        "volavg"
    ]

    e20 = data[
        "ema20"
    ]

    required = (
        atr[i],
        rsi[i],
        rsi[i - 1],
        va[i],
        e20[i],
    )

    if any(
        x is None
        for x in required
    ):

        return False

    if (
        c[i] <= 0
        or atr[i] <= 0
    ):

        return False

    atr_pct = (
        atr[i]
        / c[i]
        * 100.0
    )

    if not (
        0.025
        <= atr_pct
        <= 0.90
    ):

        return False

    candle_range = (
        h[i] - l[i]
    )

    if candle_range <= 0:

        return False

    body = (
        c[i] - o[i]
    )

    if body <= 0:

        return False

    body_ratio = (
        body
        / candle_range
    )

    if body_ratio < 0.25:

        return False

    close_location = (
        c[i] - l[i]
    ) / candle_range

    if close_location < 0.60:

        return False

    if candle_range > (
        atr[i] * 2.5
    ):

        return False

    if rsi[i] <= rsi[i - 1]:

        return False

    if not (
        32.0
        <= rsi[i]
        <= 72.0
    ):

        return False

    # Break previous 1m high
    if c[i] <= h[i - 1]:

        return False

    if v[i] < (
        va[i] * 0.70
    ):

        return False

    extension = (
        (c[i] - e20[i])
        / e20[i]
        * 100.0
    )

    if extension > 0.50:

        return False

    return True


# ============================================================
# COMPLETE ENTRY SIGNAL
# ============================================================

def entry_signal(
    data1,
    data5,
    data15,
    i,
):

    t = data1["times"]

    idx15 = closed_index(
        data15,
        t[i],
        900,
    )

    if not regime_15m(
        data15,
        idx15,
    ):

        return False

    idx5 = closed_index(
        data5,
        t[i],
        300,
    )

    if not setup_5m(
        data5,
        idx5,
    ):

        return False

    if not reversal_1m(
        data1,
        i,
    ):

        return False

    return True


# ============================================================
# RESULT OBJECT
# ============================================================

def blank_result():

    return {
        "signals": 0,
        "unfilled": 0,

        "trades": 0,
        "wins": 0,

        "gross": 0.0,
        "fees": 0.0,
        "slippage": 0.0,
        "net": 0.0,

        "positive_net": 0.0,
        "negative_net": 0.0,

        "tp": 0,
        "stop": 0,
        "time": 0,

        "mfe_sum": 0.0,
        "mae_sum": 0.0,

        "equity": 0.0,
        "peak": 0.0,
        "max_dd": 0.0,
    }


# ============================================================
# METRICS
# ============================================================

def metrics(r):

    trades = r["trades"]

    if trades > 0:

        winrate = (
            r["wins"]
            / trades
            * 100.0
        )

        avg_net = (
            r["net"]
            / trades
        )

        avg_mfe = (
            r["mfe_sum"]
            / trades
        )

        avg_mae = (
            r["mae_sum"]
            / trades
        )

    else:

        winrate = 0.0
        avg_net = 0.0
        avg_mfe = 0.0
        avg_mae = 0.0

    if r[
        "negative_net"
    ] > 0:

        pf = (
            r["positive_net"]
            / r["negative_net"]
        )

    elif r[
        "positive_net"
    ] > 0:

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
# SIMULATION
# ============================================================

def simulate(
    data1,
    data5,
    data15,
    start_ts,
    end_ts,
    tp_pct,
    stop_pct,
):

    result = blank_result()

    t = data1["times"]
    h = data1["highs"]
    l = data1["lows"]
    c = data1["closes"]

    i = max(
        220,
        bisect_left(
            t,
            start_ts,
        ),
    )

    while (
        i < len(t) - 2
        and t[i] < end_ts
    ):

        if not entry_signal(
            data1,
            data5,
            data15,
            i,
        ):

            i += 1
            continue

        result[
            "signals"
        ] += 1

        # ====================================================
        # MAKER ENTRY
        # ====================================================

        entry_limit = (
            c[i]
            * (
                1.0
                - ENTRY_LIMIT_OFFSET
            )
        )

        signal_close_time = (
            t[i] + 60
        )

        deadline = (
            signal_close_time
            + ENTRY_WAIT_MINUTES
            * 60
        )

        entry_i = None

        j = i + 1

        while (
            j < len(t)
            and t[j] < deadline
            and t[j] < end_ts
        ):

            if l[j] <= entry_limit:

                entry_i = j
                break

            j += 1

        if entry_i is None:

            result[
                "unfilled"
            ] += 1

            i += 1
            continue

        entry = entry_limit

        target = (
            entry
            * (
                1.0
                + tp_pct / 100.0
            )
        )

        stop = (
            entry
            * (
                1.0
                - stop_pct / 100.0
            )
        )

        qty = (
            STAKE
            / entry
        )

        highest = entry
        lowest = entry

        raw_exit = None
        actual_exit = None
        exit_i = None
        exit_reason = None
        exit_fee_rate = None

        max_exit_time = (
            t[entry_i]
            + MAX_HOLD_MINUTES
            * 60
        )

        j = entry_i

        # ====================================================
        # POSITION LOOP
        # ====================================================

        while (
            j < len(t)
            and t[j] < end_ts
            and t[j] <= max_exit_time
        ):

            highest = max(
                highest,
                h[j],
            )

            lowest = min(
                lowest,
                l[j],
            )

            # Conservative same-candle assumption:
            # STOP happens before TP.
            if l[j] <= stop:

                raw_exit = stop

                actual_exit = (
                    stop
                    * (
                        1.0
                        - TAKER_SLIPPAGE
                    )
                )

                exit_i = j

                exit_reason = (
                    "STOP"
                )

                exit_fee_rate = (
                    TAKER_FEE
                )

                break

            if h[j] >= target:

                raw_exit = target

                actual_exit = target

                exit_i = j

                exit_reason = (
                    "TP"
                )

                # Resting limit
                exit_fee_rate = (
                    MAKER_FEE
                )

                break

            j += 1

        # ====================================================
        # TIME EXIT
        # ====================================================

        if actual_exit is None:

            exit_i = (
                bisect_right(
                    t,
                    min(
                        max_exit_time,
                        end_ts - 1,
                    ),
                )
                - 1
            )

            if exit_i < entry_i:

                exit_i = entry_i

            raw_exit = (
                c[exit_i]
            )

            actual_exit = (
                raw_exit
                * (
                    1.0
                    - TAKER_SLIPPAGE
                )
            )

            exit_reason = (
                "TIME"
            )

            exit_fee_rate = (
                TAKER_FEE
            )

        # ====================================================
        # COSTS
        # ====================================================

        entry_value = (
            qty * entry
        )

        raw_exit_value = (
            qty * raw_exit
        )

        actual_exit_value = (
            qty * actual_exit
        )

        gross = (
            raw_exit_value
            - entry_value
        )

        entry_fee = (
            entry_value
            * MAKER_FEE
        )

        exit_fee = (
            actual_exit_value
            * exit_fee_rate
        )

        fees = (
            entry_fee
            + exit_fee
        )

        slippage = max(
            0.0,
            raw_exit_value
            - actual_exit_value,
        )

        # Slippage already included in actual_exit.
        # Do NOT subtract it again.
        net = (
            actual_exit_value
            - entry_value
            - fees
        )

        # ====================================================
        # MFE / MAE
        # ====================================================

        mfe = max(
            0.0,
            (
                highest - entry
            )
            / entry
            * 100.0,
        )

        mae = max(
            0.0,
            (
                entry - lowest
            )
            / entry
            * 100.0,
        )

        # ====================================================
        # RECORD
        # ====================================================

        result[
            "trades"
        ] += 1

        result[
            "gross"
        ] += gross

        result[
            "fees"
        ] += fees

        result[
            "slippage"
        ] += slippage

        result[
            "net"
        ] += net

        result[
            "mfe_sum"
        ] += mfe

        result[
            "mae_sum"
        ] += mae

        if net > 0:

            result[
                "wins"
            ] += 1

            result[
                "positive_net"
            ] += net

        elif net < 0:

            result[
                "negative_net"
            ] += abs(net)

        if exit_reason == "TP":

            result[
                "tp"
            ] += 1

        elif exit_reason == "STOP":

            result[
                "stop"
            ] += 1

        else:

            result[
                "time"
            ] += 1

        result[
            "equity"
        ] += net

        result[
            "peak"
        ] = max(
            result["peak"],
            result["equity"],
        )

        drawdown = (
            result["peak"]
            - result["equity"]
        )

        result[
            "max_dd"
        ] = max(
            result["max_dd"],
            drawdown,
        )

        # ====================================================
        # COOLDOWN
        # ====================================================

        next_allowed = (
            t[exit_i]
            + COOLDOWN_MINUTES
            * 60
        )

        i = bisect_left(
            t,
            next_allowed,
        )

    return result


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def add_result(
    total,
    r,
):

    for key in [
        "signals",
        "unfilled",
        "trades",
        "wins",
        "gross",
        "fees",
        "slippage",
        "net",
        "positive_net",
        "negative_net",
        "tp",
        "stop",
        "time",
        "mfe_sum",
        "mae_sum",
    ]:

        total[key] += r[key]

    total[
        "max_dd"
    ] = max(
        total["max_dd"],
        r["max_dd"],
    )


def aggregate(
    results_by_coin,
    config_index,
):

    total = blank_result()

    positive_coins = 0
    negative_coins = 0
    active_coins = 0

    for asset, results in (
        results_by_coin.items()
    ):

        if config_index >= len(
            results
        ):

            continue

        r = results[
            config_index
        ]

        add_result(
            total,
            r,
        )

        if r[
            "trades"
        ] > 0:

            active_coins += 1

            if r["net"] > 0:

                positive_coins += 1

            elif r["net"] < 0:

                negative_coins += 1

    return (
        total,
        positive_coins,
        negative_coins,
        active_coins,
    )


# ============================================================
# WRITE RESULTS
# ============================================================

def write_results(
    train_by_coin,
    hold_by_coin,
):

    candidates = []

    for idx, config in enumerate(
        CONFIGS
    ):

        (
            tp_pct,
            stop_pct,
        ) = config

        (
            train,
            train_pos,
            train_neg,
            train_active,
        ) = aggregate(
            train_by_coin,
            idx,
        )

        (
            hold,
            hold_pos,
            hold_neg,
            hold_active,
        ) = aggregate(
            hold_by_coin,
            idx,
        )

        tm = metrics(
            train
        )

        hm = metrics(
            hold
        )

        # Don't exclude small samples completely.
        # Mark them instead.
        eligible = (
            train["trades"] >= 20
            and train_active >= 6
        )

        candidates.append({
            "idx": idx,

            "tp_pct":
                tp_pct,

            "stop_pct":
                stop_pct,

            "train":
                train,

            "hold":
                hold,

            "tm":
                tm,

            "hm":
                hm,

            "train_pos":
                train_pos,

            "train_neg":
                train_neg,

            "train_active":
                train_active,

            "hold_pos":
                hold_pos,

            "hold_neg":
                hold_neg,

            "hold_active":
                hold_active,

            "eligible":
                eligible,
        })

    # ========================================================
    # RANK ON TRAIN ONLY
    #
    # Eligible configurations first.
    # HOLDOUT is NEVER part of sorting.
    # ========================================================

    candidates.sort(
        key=lambda x: (
            1 if x["eligible"] else 0,
            x["train"]["net"],
            x["tm"]["pf"],
            x["train_pos"],
        ),
        reverse=True,
    )

    fields = [
        # Existing Telegram compatibility
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

        # V10.2
        "stop_pct",
        "eligible",

        "signals",
        "unfilled",

        "gross_market",
        "fees",
        "slippage",
        "avg_net_trade",

        "avg_mfe_pct",
        "avg_mae_pct",

        "trades_per_week",

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
        "train_max_dd",

        "train_gross",
        "train_fees",
        "train_slippage",
        "train_avg_net",
        "train_avg_mfe",
        "train_avg_mae",

        "train_signals",
        "train_unfilled",

        "train_tp",
        "train_stop",
        "train_time",

        "train_positive_coins",
        "train_negative_coins",
        "train_active_coins",

        "maker_fee_pct",
        "taker_fee_pct",
        "taker_slippage_pct",
    ]

    output = []

    for rank, item in enumerate(
        candidates,
        start=1,
    ):

        train = item[
            "train"
        ]

        hold = item[
            "hold"
        ]

        tm = item["tm"]
        hm = item["hm"]

        row = {
            "rank":
                rank,

            # HOLDOUT displayed here
            "trades":
                hold["trades"],

            "wins":
                hold["wins"],

            "winrate":
                hm["winrate"],

            "net_usdt":
                hold["net"],

            "profit_factor":
                hm["pf"],

            "max_drawdown":
                hold["max_dd"],

            "tp_pct":
                item["tp_pct"],

            "sl_atr":
                0.0,

            "volume_mult":
                VOLUME_MULT,

            "lookback":
                6,

            "stop_pct":
                item["stop_pct"],

            "eligible":
                1 if item[
                    "eligible"
                ] else 0,

            "signals":
                hold["signals"],

            "unfilled":
                hold["unfilled"],

            "gross_market":
                hold["gross"],

            "fees":
                hold["fees"],

            "slippage":
                hold["slippage"],

            "avg_net_trade":
                hm["avg_net"],

            "avg_mfe_pct":
                hm["avg_mfe"],

            "avg_mae_pct":
                hm["avg_mae"],

            "trades_per_week":
                hold["trades"]
                / (10.0 / 7.0),

            "tp_exits":
                hold["tp"],

            "stop_exits":
                hold["stop"],

            "time_exits":
                hold["time"],

            "positive_coins":
                item["hold_pos"],

            "negative_coins":
                item["hold_neg"],

            "active_coins":
                item["hold_active"],

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

            "train_max_dd":
                train["max_dd"],

            "train_gross":
                train["gross"],

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

            "train_signals":
                train["signals"],

            "train_unfilled":
                train["unfilled"],

            "train_tp":
                train["tp"],

            "train_stop":
                train["stop"],

            "train_time":
                train["time"],

            "train_positive_coins":
                item["train_pos"],

            "train_negative_coins":
                item["train_neg"],

            "train_active_coins":
                item["train_active"],

            "maker_fee_pct":
                MAKER_FEE * 100.0,

            "taker_fee_pct":
                TAKER_FEE * 100.0,

            "taker_slippage_pct":
                TAKER_SLIPPAGE
                * 100.0,
        }

        output.append(
            row
        )

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in output:

            clean = {}

            for key, value in (
                row.items()
            ):

                if isinstance(
                    value,
                    float,
                ):

                    clean[key] = (
                        f"{value:.6f}"
                    )

                else:

                    clean[key] = value

            writer.writerow(
                clean
            )

    # ========================================================
    # PER COIN RESULTS FOR TRAIN-RANK #1
    # ========================================================

    if output:

        best_idx = candidates[
            0
        ]["idx"]

        with open(
            COIN_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            fields_coin = [
                "asset",

                "train_trades",
                "train_net",
                "train_pf",
                "train_wr",

                "hold_trades",
                "hold_net",
                "hold_pf",
                "hold_wr",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fields_coin,
            )

            writer.writeheader()

            for asset in sorted(
                train_by_coin
            ):

                tr = train_by_coin[
                    asset
                ][best_idx]

                ho = hold_by_coin[
                    asset
                ][best_idx]

                tm = metrics(
                    tr
                )

                hm = metrics(
                    ho
                )

                writer.writerow({
                    "asset":
                        asset,

                    "train_trades":
                        tr["trades"],

                    "train_net":
                        f"{tr['net']:.6f}",

                    "train_pf":
                        f"{tm['pf']:.6f}",

                    "train_wr":
                        f"{tm['winrate']:.6f}",

                    "hold_trades":
                        ho["trades"],

                    "hold_net":
                        f"{ho['net']:.6f}",

                    "hold_pf":
                        f"{hm['pf']:.6f}",

                    "hold_wr":
                        f"{hm['winrate']:.6f}",
                })

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================",
        flush=True,
    )

    print(
        "MPORBBOT V10.2 KRAKEN",
        flush=True,
    )

    print(
        "FIXED TARGET BACKTEST",
        flush=True,
    )

    print(
        "LONG ONLY",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    print(
        f"Stake: {STAKE:.0f} USD",
        flush=True,
    )

    print(
        f"Configs: {TOTAL_CONFIGS}",
        flush=True,
    )

    print(
        "TP: 1.50 / 2.00 / 2.50 / 3.00%",
        flush=True,
    )

    print(
        "SL: 0.80 / 1.00 / 1.20%",
        flush=True,
    )

    print(
        "Entry: MAKER limit -0.05%",
        flush=True,
    )

    print(
        "Entry wait: 3 minutes",
        flush=True,
    )

    print(
        "TP exit: MAKER",
        flush=True,
    )

    print(
        "STOP/TIME exit: TAKER",
        flush=True,
    )

    print(
        "Maker fee: 0.40%",
        flush=True,
    )

    print(
        "Taker fee: 0.80%",
        flush=True,
    )

    print(
        "Taker slippage: 0.05%",
        flush=True,
    )

    print(
        f"Max hold: {MAX_HOLD_MINUTES} min",
        flush=True,
    )

    print(
        "20d TRAIN + 10d untouched HOLDOUT",
        flush=True,
    )

    print(
        "RANKING USES TRAIN ONLY",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    download_archive()

    train_by_coin = {}
    hold_by_coin = {}

    with zipfile.ZipFile(
        ARCHIVE_FILE,
        "r",
    ) as zf:

        zip_index = (
            build_zip_index(
                zf
            )
        )

        print(
            f"Archive CSV files: "
            f"{len(zip_index)}",
            flush=True,
        )

        for number, asset in enumerate(
            ASSETS,
            start=1,
        ):

            pair = resolve_pair(
                asset,
                zip_index,
            )

            if pair is None:

                print(
                    f"{number}/25 "
                    f"{asset}: NO DATA",
                    flush=True,
                )

                continue

            print(
                "",
                flush=True,
            )

            print(
                f"{number}/25 "
                f"{asset} -> {pair}",
                flush=True,
            )

            rows1 = load_csv(
                zf,
                zip_index,
                pair,
                1,
            )

            rows5 = load_csv(
                zf,
                zip_index,
                pair,
                5,
            )

            rows15 = load_csv(
                zf,
                zip_index,
                pair,
                15,
            )

            print(
                f"1m={len(rows1)} | "
                f"5m={len(rows5)} | "
                f"15m={len(rows15)}",
                flush=True,
            )

            if (
                len(rows1) < 5000
                or len(rows5) < 1000
                or len(rows15) < 350
            ):

                print(
                    "INSUFFICIENT DATA - SKIP",
                    flush=True,
                )

                continue

            data1 = prepare(
                rows1
            )

            data5 = prepare(
                rows5
            )

            data15 = prepare(
                rows15
            )

            train_results = []
            hold_results = []

            for idx, (
                tp_pct,
                stop_pct,
            ) in enumerate(
                CONFIGS,
                start=1,
            ):

                train = simulate(
                    data1,
                    data5,
                    data15,
                    TRAIN_START,
                    TRAIN_END,
                    tp_pct,
                    stop_pct,
                )

                hold = simulate(
                    data1,
                    data5,
                    data15,
                    HOLDOUT_START,
                    HOLDOUT_END,
                    tp_pct,
                    stop_pct,
                )

                train_results.append(
                    train
                )

                hold_results.append(
                    hold
                )

                tm = metrics(
                    train
                )

                hm = metrics(
                    hold
                )

                print(
                    f"{pair}: "
                    f"{idx}/{TOTAL_CONFIGS} | "
                    f"TP {tp_pct:.2f} "
                    f"SL {stop_pct:.2f} | "
                    f"TRAIN "
                    f"{train['net']:+.2f} "
                    f"PF {tm['pf']:.2f} "
                    f"{train['trades']}t | "
                    f"HOLD "
                    f"{hold['net']:+.2f} "
                    f"PF {hm['pf']:.2f} "
                    f"{hold['trades']}t",
                    flush=True,
                )

            train_by_coin[
                asset
            ] = train_results

            hold_by_coin[
                asset
            ] = hold_results

            print(
                f"COMPLETE {number}/25",
                flush=True,
            )

            del rows1
            del rows5
            del rows15

            del data1
            del data5
            del data15

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    rows = write_results(
        train_by_coin,
        hold_by_coin,
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
        "V10.2 KRAKEN COMPLETE",
        flush=True,
    )

    print(
        "RANKED ON TRAIN ONLY",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    for row in rows[:5]:

        print(
            "",
            flush=True,
        )

        print(
            f"RANK #{row['rank']} "
            f"| ELIGIBLE {row['eligible']}",
            flush=True,
        )

        print(
            f"TP {row['tp_pct']:.2f}% "
            f"| SL {row['stop_pct']:.2f}%",
            flush=True,
        )

        print(
            f"TRAIN: "
            f"{row['train_net']:+.2f} USD | "
            f"PF {row['train_pf']:.2f} | "
            f"WR {row['train_winrate']:.1f}% | "
            f"{row['train_trades']} trades",
            flush=True,
        )

        print(
            f"HOLD:  "
            f"{row['net_usdt']:+.2f} USD | "
            f"PF {row['profit_factor']:.2f} | "
            f"WR {row['winrate']:.1f}% | "
            f"{row['trades']} trades",
            flush=True,
        )

        print(
            f"HOLD gross "
            f"{row['gross_market']:+.2f} | "
            f"fees -{row['fees']:.2f} | "
            f"slip -{row['slippage']:.2f}",
            flush=True,
        )

        print(
            f"AVG NET/TRADE "
            f"{row['avg_net_trade']:+.3f} USD",
            flush=True,
        )

        print(
            f"MFE {row['avg_mfe_pct']:.2f}% | "
            f"MAE {row['avg_mae_pct']:.2f}%",
            flush=True,
        )

        print(
            f"EXITS: "
            f"TP {row['tp_exits']} | "
            f"STOP {row['stop_exits']} | "
            f"TIME {row['time_exits']}",
            flush=True,
        )

        print(
            f"SIGNALS {row['signals']} | "
            f"UNFILLED {row['unfilled']}",
            flush=True,
        )

        print(
            f"COINS: "
            f"{row['positive_coins']} positive / "
            f"{row['negative_coins']} negative / "
            f"{row['active_coins']} active",
            flush=True,
        )

        print(
            f"RATE: "
            f"{row['trades_per_week']:.1f} trades/week",
            flush=True,
        )


if __name__ == "__main__":

    main()
