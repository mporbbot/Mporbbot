import os
import csv
import json
import time
import zipfile
from bisect import bisect_left, bisect_right
from itertools import product
from pathlib import PurePosixPath

import requests


# ============================================================
# MPORBBOT BACKTEST V10 - KRAKEN
#
# LONG ONLY
# 15m regime -> 5m mean reversion -> 1m reversal
#
# ENTRY:
# Post-only style limit buy -> MAKER
#
# TP:
# Resting limit sell -> MAKER
#
# STOP / TIME:
# Market style exit -> TAKER + slippage
#
# DATA:
# Official Kraken Q2 2026 OHLCVT archive
# ============================================================

VERSION = "V10_KRAKEN_MEAN_REVERSION"

STAKE = 100.0

# Current Kraken base-tier assumptions
MAKER_FEE = 0.0040       # 0.40%
TAKER_FEE = 0.0080       # 0.80%

# Slippage only on taker exits
TAKER_SLIPPAGE = 0.0005  # 0.05%

# Maker entry is placed slightly below signal close
ENTRY_LIMIT_OFFSET = 0.0005  # 0.05%

# Wait max 3 minutes for maker entry
ENTRY_WAIT_MINUTES = 3

MAX_HOLD_MINUTES = 180
COOLDOWN_MINUTES = 15

# Dynamic target is frozen 5m EMA20,
# but never more than +3%
MAX_TARGET_PCT = 3.00

# Slightly below EMA20 to improve TP fill probability
MEAN_TARGET_BUFFER = 0.0005

ARCHIVE_FILE = "Kraken_OHLCVT_2026Q2.zip"

KRAKEN_HOST = "assets.kraken.com"

ARCHIVE_URL = (
    "https://"
    + KRAKEN_HOST
    + "/marketing/institutions/"
    + ARCHIVE_FILE
)

CHECKPOINT_FILE = "backtest_v10_kraken_checkpoint.json"
RESULT_FILE = "backtest_results.csv"
COIN_FILE = "backtest_v10_kraken_coins.csv"


# ============================================================
# FIXED TEST PERIOD
#
# Q2 2026
#
# Warmup:   Apr 1 -> Apr 6
# TRAIN:    Apr 6 -> Apr 26
# HOLDOUT:  Apr 26 -> May 6
#
# This finishes before V9's tested period.
# ============================================================

DATA_START = 1775001600      # 2026-04-01 00:00 UTC
TRAIN_START = 1775433600     # 2026-04-06 00:00 UTC
TRAIN_END = 1777161600       # 2026-04-26 00:00 UTC
HOLDOUT_START = TRAIN_END
HOLDOUT_END = 1778025600     # 2026-05-06 00:00 UTC


# ============================================================
# COINS
#
# Kraken history exports can use XBT for BTC
# and XDG for DOGE.
#
# Code automatically prefers USD pairs.
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
# PARAMETER GRID
#
# 2 RSI
# 2 deviation
# 2 volume
# 3 min target
# 3 stop
#
# = 72 configs
#
# Targets are larger than old KuCoin strategy because
# Kraken costs are higher.
# ============================================================

RSI_VALUES = [
    35.0,
    40.0,
]

DEVIATION_ATR_VALUES = [
    0.40,
    0.70,
]

VOLUME_VALUES = [
    0.80,
    1.00,
]

MIN_TARGET_VALUES = [
    1.20,
    1.50,
    1.80,
]

STOP_VALUES = [
    0.70,
    0.90,
    1.10,
]

CONFIGS = list(product(
    RSI_VALUES,
    DEVIATION_ATR_VALUES,
    VOLUME_VALUES,
    MIN_TARGET_VALUES,
    STOP_VALUES,
))

TOTAL_CONFIGS = len(CONFIGS)


# ============================================================
# DOWNLOAD
# ============================================================

def download_archive():
    if os.path.exists(ARCHIVE_FILE):
        size_mb = os.path.getsize(ARCHIVE_FILE) / 1024 / 1024

        print(
            f"Kraken archive already exists: "
            f"{size_mb:.1f} MB",
            flush=True,
        )
        return

    print(
        "Downloading official Kraken Q2 2026 OHLCVT archive...",
        flush=True,
    )

    temp_file = ARCHIVE_FILE + ".part"

    with requests.get(
        ARCHIVE_URL,
        stream=True,
        timeout=120,
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        downloaded = 0
        last_print = 0

        with open(temp_file, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if downloaded - last_print >= 20 * 1024 * 1024:
                    last_print = downloaded

                    if total > 0:
                        pct = downloaded / total * 100

                        print(
                            f"Download: "
                            f"{downloaded / 1024 / 1024:.0f} MB "
                            f"({pct:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(
                            f"Download: "
                            f"{downloaded / 1024 / 1024:.0f} MB",
                            flush=True,
                        )

    os.replace(
        temp_file,
        ARCHIVE_FILE,
    )

    print(
        "Kraken archive downloaded.",
        flush=True,
    )


# ============================================================
# ZIP HELPERS
# ============================================================

def basename(name):
    return PurePosixPath(name).name.upper()


def build_zip_index(zf):
    index = {}

    for name in zf.namelist():
        b = basename(name)

        if b.endswith(".CSV"):
            index[b] = name

    return index


def resolve_pair(asset, zip_index):
    aliases = ALIASES.get(
        asset,
        [asset],
    )

    # Prefer USD.
    # USDT is fallback.
    quotes = [
        "USD",
        "USDT",
    ]

    for quote in quotes:
        for alias in aliases:

            candidate = (
                f"{alias}{quote}_1.CSV"
            )

            if candidate in zip_index:
                return (
                    f"{alias}{quote}",
                    quote,
                )

    return None, None


def find_csv(
    pair,
    interval,
    zip_index,
):
    key = (
        f"{pair}_{interval}.CSV"
    )

    return zip_index.get(key)


# ============================================================
# LOAD KRAKEN CSV
#
# Format:
# timestamp, open, high, low, close, volume, trades
#
# No header.
# ============================================================

def load_kraken_csv(
    zf,
    zip_index,
    pair,
    interval,
):
    filename = find_csv(
        pair,
        interval,
        zip_index,
    )

    if filename is None:
        return []

    rows = []

    with zf.open(filename) as raw:

        for byte_line in raw:
            try:
                line = byte_line.decode(
                    "utf-8"
                ).strip()

                if not line:
                    continue

                parts = line.split(",")

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
                    float(parts[1]),  # open
                    float(parts[2]),  # high
                    float(parts[3]),  # low
                    float(parts[4]),  # close
                    float(parts[5]),  # volume
                    int(float(parts[6])),
                ])

            except Exception:
                continue

    rows.sort(
        key=lambda x: x[0]
    )

    return rows


# ============================================================
# INDICATORS
# ============================================================

def ema_series(values, period):
    result = [None] * len(values)

    if not values:
        return result

    alpha = 2.0 / (
        period + 1.0
    )

    current = values[0]

    for i, value in enumerate(values):

        if i > 0:
            current = (
                value * alpha
                + current * (
                    1.0 - alpha
                )
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

    for i in range(1, n):

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


def volume_average(
    volumes,
    period=20,
):
    result = [None] * len(
        volumes
    )

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


def bollinger_lower(
    closes,
    period=20,
):
    result = [None] * len(
        closes
    )

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

        variance = sum(
            (x - mean) ** 2
            for x in window
        ) / period

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
        row[0]
        for row in rows
    ]

    opens = [
        row[1]
        for row in rows
    ]

    highs = [
        row[2]
        for row in rows
    ]

    lows = [
        row[3]
        for row in rows
    ]

    closes = [
        row[4]
        for row in rows
    ]

    volumes = [
        row[5]
        for row in rows
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
#
# Works even if Kraken has missing empty candles.
# ============================================================

def closed_index(
    data,
    signal_open,
    timeframe_seconds,
):
    signal_close = (
        signal_open + 60
    )

    latest_allowed_open = (
        signal_close
        - timeframe_seconds
    )

    idx = bisect_right(
        data["times"],
        latest_allowed_open,
    ) - 1

    if idx < 0:
        return None

    return idx


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
# 5M MEAN REVERSION SETUP
# ============================================================

def setup_5m(
    data,
    i,
    rsi_limit,
    deviation_atr,
    volume_mult,
):
    if (
        i is None
        or i < 205
    ):
        return None

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
    volavg = data["volavg"]
    bb = data["bb_low"]

    oversold_i = None
    lowest_rsi = 999.0

    start = max(
        205,
        i - 6,
    )

    for j in range(
        start,
        i + 1,
    ):

        # Don't allow an old setup
        # hidden behind large data gaps.
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

        lower_band_hit = (
            l[j]
            <= bb[j] * 1.001
        )

        deviation_hit = (
            deviation
            >= deviation_atr
        )

        if (
            rsi[j] <= rsi_limit
            and (
                lower_band_hit
                or deviation_hit
            )
            and v[j]
            >= volavg[j]
            * volume_mult
        ):
            if (
                rsi[j]
                < lowest_rsi
            ):
                lowest_rsi = rsi[j]
                oversold_i = j

    if oversold_i is None:
        return None

    if any(
        x is None
        for x in (
            e20[i],
            e50[i],
            rsi[i],
        )
    ):
        return None

    recovery = 0

    if c[i] > o[i]:
        recovery += 1

    if c[i] > c[i - 1]:
        recovery += 1

    if (
        rsi[i - 1] is not None
        and rsi[i]
        > rsi[i - 1]
    ):
        recovery += 1

    previous_mid = (
        h[i - 1]
        + l[i - 1]
    ) / 2.0

    if c[i] > previous_mid:
        recovery += 1

    if recovery < 2:
        return None

    # Avoid structural collapse
    if c[i] < (
        e50[i] * 0.97
    ):
        return None

    # Reversion already completed?
    if c[i] > (
        e20[i] * 1.003
    ):
        return None

    return {
        "mean5": e20[i],
        "oversold_rsi":
            lowest_rsi,
    }


# ============================================================
# 1M REVERSAL
# ============================================================

def entry_signal(
    data1,
    data5,
    data15,
    i,
    config,
):
    (
        rsi_limit,
        deviation_atr,
        volume_mult,
        min_target,
        stop_pct,
    ) = config

    if i < 220:
        return None

    t = data1["times"]
    o = data1["opens"]
    h = data1["highs"]
    l = data1["lows"]
    c = data1["closes"]
    v = data1["volumes"]

    atr = data1["atr"]
    rsi = data1["rsi"]
    va = data1["volavg"]
    e20 = data1["ema20"]

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
        return None

    if c[i] <= 0:
        return None

    idx15 = closed_index(
        data15,
        t[i],
        900,
    )

    if not regime_15m(
        data15,
        idx15,
    ):
        return None

    idx5 = closed_index(
        data5,
        t[i],
        300,
    )

    setup = setup_5m(
        data5,
        idx5,
        rsi_limit,
        deviation_atr,
        volume_mult,
    )

    if setup is None:
        return None

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

    if body_ratio < 0.25:
        return None

    close_location = (
        c[i] - l[i]
    ) / candle_range

    if close_location < 0.60:
        return None

    if candle_range > (
        atr[i] * 2.5
    ):
        return None

    # RSI turning upward
    if rsi[i] <= rsi[i - 1]:
        return None

    if not (
        32.0
        <= rsi[i]
        <= 72.0
    ):
        return None

    # Early confirmation:
    # break previous 1m high
    if c[i] <= h[i - 1]:
        return None

    if v[i] < (
        va[i] * 0.70
    ):
        return None

    extension = (
        c[i] - e20[i]
    ) / e20[i] * 100.0

    if extension > 0.50:
        return None

    return {
        "signal_close": c[i],
        "mean5": setup[
            "mean5"
        ],
        "min_target":
            min_target,
        "stop_pct":
            stop_pct,
    }


# ============================================================
# RESULT
# ============================================================

def blank_result():
    return {
        "trades": 0,
        "wins": 0,

        "net": 0.0,
        "gross": 0.0,

        "fees": 0.0,
        "slippage": 0.0,

        "positive_net": 0.0,
        "negative_net": 0.0,

        "mfe_sum": 0.0,
        "mae_sum": 0.0,
        "target_sum": 0.0,

        "tp": 0,
        "stop": 0,
        "time": 0,

        "unfilled": 0,
        "no_room": 0,

        "equity": 0.0,
        "peak": 0.0,
        "max_dd": 0.0,
    }


def metrics(r):
    trades = r["trades"]

    if trades:
        wr = (
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

        avg_target = (
            r["target_sum"]
            / trades
        )

    else:
        wr = 0.0
        avg_net = 0.0
        avg_mfe = 0.0
        avg_mae = 0.0
        avg_target = 0.0

    if r["negative_net"] > 0:
        pf = (
            r["positive_net"]
            / r["negative_net"]
        )

    elif r["positive_net"] > 0:
        pf = 999.0

    else:
        pf = 0.0

    return {
        "wr": wr,
        "pf": pf,
        "avg_net": avg_net,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "avg_target":
            avg_target,
    }


# ============================================================
# SIMULATE
# ============================================================

def simulate(
    data1,
    data5,
    data15,
    config,
    start_ts,
    end_ts,
):
    result = blank_result()

    t = data1["times"]
    o = data1["opens"]
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

    while i < len(t) - 2:

        if t[i] >= end_ts:
            break

        signal = entry_signal(
            data1,
            data5,
            data15,
            i,
            config,
        )

        if signal is None:
            i += 1
            continue

        # ====================================================
        # POST-ONLY STYLE MAKER ENTRY
        #
        # Place buy limit 0.05% below signal close.
        # It must actually trade through our price within
        # next 3 minutes.
        # ====================================================

        entry_limit = (
            signal["signal_close"]
            * (
                1.0
                - ENTRY_LIMIT_OFFSET
            )
        )

        signal_close_time = (
            t[i] + 60
        )

        entry_deadline = (
            signal_close_time
            + ENTRY_WAIT_MINUTES
            * 60
        )

        entry_i = None

        j = i + 1

        while (
            j < len(t)
            and t[j]
            < entry_deadline
            and t[j]
            < end_ts
        ):

            # Require actual touch.
            if l[j] <= entry_limit:
                entry_i = j
                break

            j += 1

        if entry_i is None:
            result["unfilled"] += 1
            i += 1
            continue

        entry = entry_limit

        # ====================================================
        # TARGET = FROZEN 5M EMA20
        #
        # Important:
        # We do NOT invent a target if EMA20 is too close.
        # There must be enough real room to the mean.
        # ====================================================

        mean_target = (
            signal["mean5"]
            * (
                1.0
                - MEAN_TARGET_BUFFER
            )
        )

        natural_room_pct = (
            (
                mean_target
                - entry
            )
            / entry
            * 100.0
        )

        min_target = signal[
            "min_target"
        ]

        if (
            natural_room_pct
            < min_target
        ):
            result["no_room"] += 1

            i = entry_i + 1
            continue

        target_pct = min(
            natural_room_pct,
            MAX_TARGET_PCT,
        )

        target = (
            entry
            * (
                1.0
                + target_pct
                / 100.0
            )
        )

        stop = (
            entry
            * (
                1.0
                - signal[
                    "stop_pct"
                ]
                / 100.0
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

        while j < len(t):

            if t[j] >= end_ts:
                break

            if t[j] > max_exit_time:
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
            # if stop and TP hit in same candle,
            # stop is assumed first.
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
                exit_reason = "STOP"
                exit_fee_rate = (
                    TAKER_FEE
                )

                break

            if h[j] >= target:

                raw_exit = target
                actual_exit = target

                exit_i = j
                exit_reason = "TP"

                # Resting limit target
                exit_fee_rate = (
                    MAKER_FEE
                )

                break

            j += 1

        # ====================================================
        # TIME EXIT
        # ====================================================

        if actual_exit is None:

            exit_i = bisect_right(
                t,
                min(
                    max_exit_time,
                    end_ts - 1,
                ),
            ) - 1

            if exit_i < entry_i:
                exit_i = entry_i

            raw_exit = c[exit_i]

            actual_exit = (
                raw_exit
                * (
                    1.0
                    - TAKER_SLIPPAGE
                )
            )

            exit_reason = "TIME"
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

        result["trades"] += 1

        result["gross"] += gross
        result["fees"] += fees
        result["slippage"] += (
            slippage
        )

        result["net"] += net

        result["mfe_sum"] += mfe
        result["mae_sum"] += mae

        result["target_sum"] += (
            target_pct
        )

        if net > 0:
            result["wins"] += 1
            result[
                "positive_net"
            ] += net

        elif net < 0:
            result[
                "negative_net"
            ] += abs(net)

        if exit_reason == "TP":
            result["tp"] += 1

        elif exit_reason == "STOP":
            result["stop"] += 1

        else:
            result["time"] += 1

        result["equity"] += net

        result["peak"] = max(
            result["peak"],
            result["equity"],
        )

        dd = (
            result["peak"]
            - result["equity"]
        )

        result["max_dd"] = max(
            result["max_dd"],
            dd,
        )

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
# SERIALIZE / AGGREGATE
# ============================================================

def add_result(total, r):
    for key in (
        "trades",
        "wins",
        "net",
        "gross",
        "fees",
        "slippage",
        "positive_net",
        "negative_net",
        "mfe_sum",
        "mae_sum",
        "target_sum",
        "tp",
        "stop",
        "time",
        "unfilled",
        "no_room",
    ):
        total[key] += r[key]

    total["max_dd"] = max(
        total["max_dd"],
        r["max_dd"],
    )


def aggregate(
    by_coin,
    config_index,
):
    total = blank_result()

    positive = 0
    negative = 0
    active = 0

    for symbol, results in (
        by_coin.items()
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

        if r["trades"] > 0:

            active += 1

            if r["net"] > 0:
                positive += 1

            elif r["net"] < 0:
                negative += 1

    return (
        total,
        positive,
        negative,
        active,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    completed,
    resolved_pairs,
    train_by_coin,
    holdout_by_coin,
):
    payload = {
        "version": VERSION,
        "completed": completed,
        "resolved_pairs":
            resolved_pairs,
        "train_by_coin":
            train_by_coin,
        "holdout_by_coin":
            holdout_by_coin,
    }

    temp = (
        CHECKPOINT_FILE
        + ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
        )

    os.replace(
        temp,
        CHECKPOINT_FILE,
    )


def load_checkpoint():
    if not os.path.exists(
        CHECKPOINT_FILE
    ):
        return None

    try:
        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if (
            data.get("version")
            != VERSION
        ):
            return None

        return data

    except Exception:
        return None


# ============================================================
# WRITE RESULTS
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
            train_pos,
            train_neg,
            train_active,
        ) = aggregate(
            train_by_coin,
            idx,
        )

        tm = metrics(train)

        # Need reasonable sample
        if (
            train["trades"] < 30
            or train_active < 8
        ):
            continue

        (
            hold,
            hold_pos,
            hold_neg,
            hold_active,
        ) = aggregate(
            holdout_by_coin,
            idx,
        )

        candidates.append({
            "idx": idx,

            "train": train,
            "hold": hold,

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

            "train_pf":
                tm["pf"],
        })

    # HOLDOUT NEVER USED FOR RANKING
    candidates.sort(
        key=lambda x: (
            x["train"]["net"],
            x["train_pf"],
            x["train_pos"],
        ),
        reverse=True,
    )

    headers = [
        # Telegram compatibility
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

        # Kraken V10
        "stop_pct",
        "rsi_limit",
        "deviation_atr",
        "min_target_pct",
        "avg_target_pct",

        "maker_fee_pct",
        "taker_fee_pct",

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

        "unfilled_entries",
        "no_room_signals",

        "positive_coins",
        "negative_coins",
        "active_coins",

        "train_trades",
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

    output_rows = []

    for rank, item in enumerate(
        candidates[:10],
        start=1,
    ):

        (
            rsi_limit,
            deviation_atr,
            volume_mult,
            min_target,
            stop_pct,
        ) = CONFIGS[
            item["idx"]
        ]

        train = item["train"]
        hold = item["hold"]

        tm = metrics(train)
        hm = metrics(hold)

        row = {
            "rank": rank,

            "trades":
                hold["trades"],

            "wins":
                hold["wins"],

            "winrate":
                hm["wr"],

            "net_usdt":
                hold["net"],

            "profit_factor":
                hm["pf"],

            "max_drawdown":
                hold["max_dd"],

            "tp_pct":
                min_target,

            "sl_atr":
                0.0,

            "volume_mult":
                volume_mult,

            "lookback":
                6,

            "stop_pct":
                stop_pct,

            "rsi_limit":
                rsi_limit,

            "deviation_atr":
                deviation_atr,

            "min_target_pct":
                min_target,

            "avg_target_pct":
                hm["avg_target"],

            "maker_fee_pct":
                MAKER_FEE * 100,

            "taker_fee_pct":
                TAKER_FEE * 100,

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

            "unfilled_entries":
                hold["unfilled"],

            "no_room_signals":
                hold["no_room"],

            "positive_coins":
                item["hold_pos"],

            "negative_coins":
                item["hold_neg"],

            "active_coins":
                item[
                    "hold_active"
                ],

            "train_trades":
                train["trades"],

            "train_winrate":
                tm["wr"],

            "train_net":
                train["net"],

            "train_pf":
                tm["pf"],

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

            "train_avg_target":
                tm["avg_target"],

            "train_positive_coins":
                item["train_pos"],

            "train_negative_coins":
                item["train_neg"],

            "train_active_coins":
                item[
                    "train_active"
                ],
        }

        output_rows.append(
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
            fieldnames=headers,
        )

        writer.writeheader()

        for row in output_rows:

            writer.writerow({
                k: (
                    f"{v:.6f}"
                    if isinstance(
                        v,
                        float,
                    )
                    else v
                )
                for k, v
                in row.items()
            })

    # ========================================================
    # PER COIN - BEST TRAIN CONFIG
    # ========================================================

    if candidates:

        best_idx = candidates[
            0
        ]["idx"]

        with open(
            COIN_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

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

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            for symbol in sorted(
                train_by_coin
            ):

                tr = train_by_coin[
                    symbol
                ][best_idx]

                ho = holdout_by_coin[
                    symbol
                ][best_idx]

                tm = metrics(tr)
                hm = metrics(ho)

                writer.writerow({
                    "symbol":
                        symbol,

                    "train_trades":
                        tr["trades"],

                    "train_net":
                        f"{tr['net']:.6f}",

                    "train_pf":
                        f"{tm['pf']:.6f}",

                    "train_wr":
                        f"{tm['wr']:.6f}",

                    "holdout_trades":
                        ho["trades"],

                    "holdout_net":
                        f"{ho['net']:.6f}",

                    "holdout_pf":
                        f"{hm['pf']:.6f}",

                    "holdout_wr":
                        f"{hm['wr']:.6f}",
                })

    return output_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========================================",
        flush=True,
    )

    print(
        "MPORBBOT BACKTEST V10 - KRAKEN",
        flush=True,
    )

    print(
        "MEAN REVERSION / LONG ONLY",
        flush=True,
    )

    print(
        "Official Kraken Q2 2026 OHLCVT",
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
        "ENTRY: post-only style MAKER",
        flush=True,
    )

    print(
        "TP: MAKER limit",
        flush=True,
    )

    print(
        "STOP/TIME: TAKER",
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
        "Targets: minimum 1.20 / 1.50 / 1.80%",
        flush=True,
    )

    print(
        "Stops: 0.70 / 0.90 / 1.10%",
        flush=True,
    )

    print(
        "5d WARMUP + 20d TRAIN + 10d HOLDOUT",
        flush=True,
    )

    print(
        "HOLDOUT NEVER USED FOR RANKING",
        flush=True,
    )

    print(
        "========================================",
        flush=True,
    )

    download_archive()

    checkpoint = (
        load_checkpoint()
    )

    if checkpoint:

        completed = checkpoint[
            "completed"
        ]

        resolved_pairs = checkpoint[
            "resolved_pairs"
        ]

        train_by_coin = checkpoint[
            "train_by_coin"
        ]

        holdout_by_coin = checkpoint[
            "holdout_by_coin"
        ]

        print(
            f"Checkpoint: "
            f"{len(completed)} assets completed",
            flush=True,
        )

    else:

        completed = []
        resolved_pairs = {}
        train_by_coin = {}
        holdout_by_coin = {}

    with zipfile.ZipFile(
        ARCHIVE_FILE,
        "r",
    ) as zf:

        zip_index = build_zip_index(
            zf
        )

        print(
            f"Archive contains "
            f"{len(zip_index)} CSV files",
            flush=True,
        )

        for asset_no, asset in enumerate(
            ASSETS,
            start=1,
        ):

            if asset in completed:
                print(
                    f"{asset_no}/{len(ASSETS)} "
                    f"{asset}: checkpoint OK",
                    flush=True,
                )
                continue

            pair, quote = resolve_pair(
                asset,
                zip_index,
            )

            if pair is None:

                print(
                    f"{asset_no}/{len(ASSETS)} "
                    f"{asset}: no USD/USDT pair "
                    f"in Kraken Q2 archive - skipped",
                    flush=True,
                )

                completed.append(
                    asset
                )

                save_checkpoint(
                    completed,
                    resolved_pairs,
                    train_by_coin,
                    holdout_by_coin,
                )

                continue

            resolved_pairs[
                asset
            ] = pair

            print(
                "",
                flush=True,
            )

            print(
                f"{asset_no}/{len(ASSETS)} "
                f"{asset} -> {pair}",
                flush=True,
            )

            rows1 = load_kraken_csv(
                zf,
                zip_index,
                pair,
                1,
            )

            rows5 = load_kraken_csv(
                zf,
                zip_index,
                pair,
                5,
            )

            rows15 = load_kraken_csv(
                zf,
                zip_index,
                pair,
                15,
            )

            print(
                f"{pair}: "
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
                    f"{pair}: insufficient history - skipped",
                    flush=True,
                )

                completed.append(
                    asset
                )

                save_checkpoint(
                    completed,
                    resolved_pairs,
                    train_by_coin,
                    holdout_by_coin,
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

            del rows1
            del rows5
            del rows15

            train_results = []
            hold_results = []

            for idx, config in enumerate(
                CONFIGS
            ):

                tr = simulate(
                    data1,
                    data5,
                    data15,
                    config,
                    TRAIN_START,
                    TRAIN_END,
                )

                ho = simulate(
                    data1,
                    data5,
                    data15,
                    config,
                    HOLDOUT_START,
                    HOLDOUT_END,
                )

                train_results.append(
                    tr
                )

                hold_results.append(
                    ho
                )

                if (
                    (idx + 1) % 12 == 0
                    or idx + 1
                    == TOTAL_CONFIGS
                ):

                    tm = metrics(tr)
                    hm = metrics(ho)

                    print(
                        f"{pair}: "
                        f"{idx + 1}/{TOTAL_CONFIGS} | "
                        f"TRAIN "
                        f"{tr['net']:+.2f} "
                        f"PF {tm['pf']:.2f} "
                        f"{tr['trades']}t | "
                        f"HOLD "
                        f"{ho['net']:+.2f} "
                        f"PF {hm['pf']:.2f} "
                        f"{ho['trades']}t",
                        flush=True,
                    )

            train_by_coin[
                asset
            ] = train_results

            holdout_by_coin[
                asset
            ] = hold_results

            completed.append(
                asset
            )

            save_checkpoint(
                completed,
                resolved_pairs,
                train_by_coin,
                holdout_by_coin,
            )

            print(
                f"CHECKPOINT "
                f"{len(completed)}/{len(ASSETS)}",
                flush=True,
            )

            del data1
            del data5
            del data15

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
        "========================================",
        flush=True,
    )

    print(
        "V10 KRAKEN COMPLETE",
        flush=True,
    )

    print(
        "RANKED ON TRAIN ONLY",
        flush=True,
    )

    print(
        "========================================",
        flush=True,
    )

    if not rows:

        print(
            "No configuration reached "
            "30 TRAIN trades across 8 assets.",
            flush=True,
        )

    for row in rows[:5]:

        print(
            f"TRAIN RANK #{row['rank']}",
            flush=True,
        )

        print(
            f"TRAIN "
            f"{row['train_net']:+.2f} | "
            f"PF {row['train_pf']:.2f} | "
            f"{row['train_trades']} trades",
            flush=True,
        )

        print(
            f"HOLDOUT "
            f"{row['net_usdt']:+.2f} | "
            f"PF {row['profit_factor']:.2f} | "
            f"{row['trades']} trades | "
            f"WR {row['winrate']:.1f}%",
            flush=True,
        )

        print(
            f"Gross "
            f"{row['gross_market']:+.2f} | "
            f"Fees -{row['fees']:.2f} | "
            f"Slippage -{row['slippage']:.2f}",
            flush=True,
        )

        print(
            f"Avg net/trade "
            f"{row['avg_net_trade']:+.3f} USD | "
            f"{row['trades_per_week']:.0f}/week",
            flush=True,
        )

        print(
            f"MFE "
            f"{row['avg_mfe_pct']:.2f}% | "
            f"MAE "
            f"{row['avg_mae_pct']:.2f}% | "
            f"AVG TARGET "
            f"{row['avg_target_pct']:.2f}%",
            flush=True,
        )

        print(
            f"MIN TARGET "
            f"{row['min_target_pct']:.2f}% | "
            f"STOP "
            f"{row['stop_pct']:.2f}% | "
            f"RSI "
            f"{row['rsi_limit']:.0f} | "
            f"DEV "
            f"{row['deviation_atr']:.2f} ATR | "
            f"VOL "
            f"{row['volume_mult']:.2f}",
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
            f"COINS: "
            f"{row['positive_coins']} positive / "
            f"{row['negative_coins']} negative / "
            f"{row['active_coins']} active",
            flush=True,
        )

        print(
            "",
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
