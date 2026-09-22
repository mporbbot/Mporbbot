import os
import csv
import zipfile
from bisect import bisect_left, bisect_right
from pathlib import PurePosixPath

import requests


# ============================================================
# MPORBBOT V10.1 KRAKEN - SIGNAL DIAGNOSTIC
#
# PURPOSE:
# Find exactly why V10 produced ZERO trades.
#
# This is NOT an optimisation backtest.
# It counts every stage of the signal chain.
# ============================================================

VERSION = "V10.1_KRAKEN_DIAGNOSTIC"

ARCHIVE_FILE = "Kraken_OHLCVT_2026Q2.zip"

ARCHIVE_URL = (
    "https://assets.kraken.com/marketing/institutions/"
    "Kraken_OHLCVT_2026Q2.zip"
)

RESULT_FILE = "backtest_results.csv"
DIAG_FILE = "backtest_v10_1_diagnostic.csv"


# ============================================================
# SAME PERIOD AS V10
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
# DIAGNOSTIC BASELINE
#
# Deliberately use the LOOSEST V10 filters so we can see
# where signals disappear.
# ============================================================

RSI_LIMIT = 40.0
DEVIATION_ATR = 0.40
VOLUME_MULT = 0.80

ENTRY_LIMIT_OFFSET = 0.0005
ENTRY_WAIT_MINUTES = 3


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
# DOWNLOAD
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


# ============================================================
# ZIP
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
# INDICATORS
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
            prefix[-1]
            + value
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

        result[i] = (
            mean
            - 2.0
            * variance ** 0.5
        )

    return result


# ============================================================
# PREPARE
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
# CLOSED HIGHER TIMEFRAME INDEX
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

    if (
        (close - e200)
        / e200
        * 100.0
        > 15.0
    ):
        return False

    return True


# ============================================================
# 5M SETUP
# ============================================================

def setup_5m(
    data,
    i,
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

    volavg = data[
        "volavg"
    ]

    bb = data[
        "bb_low"
    ]

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

        dev_hit = (
            deviation
            >= DEVIATION_ATR
        )

        if (
            rsi[j]
            <= RSI_LIMIT
            and (
                bb_hit
                or dev_hit
            )
            and v[j]
            >= volavg[j]
            * VOLUME_MULT
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

    if c[i] < (
        e50[i] * 0.97
    ):
        return None

    if c[i] > (
        e20[i] * 1.003
    ):
        return None

    return {
        "mean5":
            e20[i],

        "lowest_rsi":
            lowest_rsi,
    }


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

    if (
        body
        / candle_range
        < 0.25
    ):
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

    if c[i] <= h[i - 1]:
        return False

    if v[i] < (
        va[i] * 0.70
    ):
        return False

    extension = (
        c[i] - e20[i]
    ) / e20[i] * 100.0

    if extension > 0.50:
        return False

    return True


# ============================================================
# COUNTERS
# ============================================================

def blank_diag():

    return {
        "candles": 0,

        "regime": 0,

        "setup5": 0,

        "reversal": 0,

        "maker_fill": 0,

        "room_040": 0,
        "room_060": 0,
        "room_080": 0,
        "room_100": 0,
        "room_120": 0,
        "room_150": 0,
        "room_180": 0,

        "room_sum": 0.0,
        "room_count": 0,
        "room_max": 0.0,

        "unfilled": 0,
    }


def add_diag(
    total,
    d,
):

    for key in (
        "candles",
        "regime",
        "setup5",
        "reversal",
        "maker_fill",

        "room_040",
        "room_060",
        "room_080",
        "room_100",
        "room_120",
        "room_150",
        "room_180",

        "room_count",
        "unfilled",
    ):

        total[key] += d[key]

    total[
        "room_sum"
    ] += d["room_sum"]

    total[
        "room_max"
    ] = max(
        total["room_max"],
        d["room_max"],
    )


# ============================================================
# DIAGNOSTIC SCAN
# ============================================================

def scan_period(
    data1,
    data5,
    data15,
    start_ts,
    end_ts,
):

    result = blank_diag()

    t = data1["times"]
    c = data1["closes"]
    l = data1["lows"]

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

        result[
            "candles"
        ] += 1

        idx15 = closed_index(
            data15,
            t[i],
            900,
        )

        if not regime_15m(
            data15,
            idx15,
        ):

            i += 1
            continue

        result[
            "regime"
        ] += 1

        idx5 = closed_index(
            data5,
            t[i],
            300,
        )

        setup = setup_5m(
            data5,
            idx5,
        )

        if setup is None:

            i += 1
            continue

        result[
            "setup5"
        ] += 1

        if not reversal_1m(
            data1,
            i,
        ):

            i += 1
            continue

        result[
            "reversal"
        ] += 1

        # ====================================================
        # SAME MAKER ENTRY AS V10
        # ====================================================

        entry_limit = (
            c[i]
            * (
                1.0
                - ENTRY_LIMIT_OFFSET
            )
        )

        deadline = (
            t[i]
            + 60
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

        result[
            "maker_fill"
        ] += 1

        # ====================================================
        # HOW MUCH ROOM REALLY EXISTS TO FROZEN 5M EMA20?
        # ====================================================

        mean_target = (
            setup["mean5"]
            * 0.9995
        )

        room = (
            (
                mean_target
                - entry_limit
            )
            / entry_limit
            * 100.0
        )

        result[
            "room_sum"
        ] += room

        result[
            "room_count"
        ] += 1

        result[
            "room_max"
        ] = max(
            result["room_max"],
            room,
        )

        if room >= 0.40:
            result["room_040"] += 1

        if room >= 0.60:
            result["room_060"] += 1

        if room >= 0.80:
            result["room_080"] += 1

        if room >= 1.00:
            result["room_100"] += 1

        if room >= 1.20:
            result["room_120"] += 1

        if room >= 1.50:
            result["room_150"] += 1

        if room >= 1.80:
            result["room_180"] += 1

        i += 1

    return result


# ============================================================
# PRINT
# ============================================================

def print_diag(
    name,
    d,
):

    if d[
        "room_count"
    ] > 0:

        avg_room = (
            d["room_sum"]
            / d["room_count"]
        )

    else:

        avg_room = 0.0

    print(
        "",
        flush=True,
    )

    print(
        f"--- {name} ---",
        flush=True,
    )

    print(
        f"1M CANDLES:       {d['candles']}",
        flush=True,
    )

    print(
        f"15M REGIME PASS:  {d['regime']}",
        flush=True,
    )

    print(
        f"5M SETUP PASS:    {d['setup5']}",
        flush=True,
    )

    print(
        f"1M REVERSAL PASS: {d['reversal']}",
        flush=True,
    )

    print(
        f"MAKER FILLED:     {d['maker_fill']}",
        flush=True,
    )

    print(
        f"UNFILLED:         {d['unfilled']}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        f"ROOM >= 0.40%:    {d['room_040']}",
        flush=True,
    )

    print(
        f"ROOM >= 0.60%:    {d['room_060']}",
        flush=True,
    )

    print(
        f"ROOM >= 0.80%:    {d['room_080']}",
        flush=True,
    )

    print(
        f"ROOM >= 1.00%:    {d['room_100']}",
        flush=True,
    )

    print(
        f"ROOM >= 1.20%:    {d['room_120']}",
        flush=True,
    )

    print(
        f"ROOM >= 1.50%:    {d['room_150']}",
        flush=True,
    )

    print(
        f"ROOM >= 1.80%:    {d['room_180']}",
        flush=True,
    )

    print(
        f"AVG ROOM:         {avg_room:.3f}%",
        flush=True,
    )

    print(
        f"MAX ROOM:         {d['room_max']:.3f}%",
        flush=True,
    )


# ============================================================
# WRITE DIAGNOSTIC CSV
# ============================================================

def write_diag_csv(rows):

    fields = [
        "asset",
        "pair",
        "period",

        "candles",
        "regime",
        "setup5",
        "reversal",
        "maker_fill",
        "unfilled",

        "room_040",
        "room_060",
        "room_080",
        "room_100",
        "room_120",
        "room_150",
        "room_180",

        "avg_room",
        "max_room",
    ]

    with open(
        DIAG_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# WRITE TELEGRAM-COMPATIBLE RESULT
#
# This is diagnostic, not a strategy result.
# It lets /backtest_result see that the run completed.
# ============================================================

def write_result_csv(
    train,
    hold,
):

    fields = [
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
    ]

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

        writer.writerow({
            "rank": 1,

            # In diagnostic mode this means
            # potential entries with >=1.20% room.
            "trades":
                hold["room_120"],

            "wins": 0,
            "winrate": 0.0,
            "net_usdt": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,

            "tp_pct": 1.20,
            "sl_atr": 0.0,

            "volume_mult":
                VOLUME_MULT,

            "lookback": 6,
        })


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================",
        flush=True,
    )

    print(
        "MPORBBOT V10.1 KRAKEN DIAGNOSTIC",
        flush=True,
    )

    print(
        "WHY DID V10 PRODUCE ZERO TRADES?",
        flush=True,
    )

    print(
        "NO PARAMETER OPTIMISATION",
        flush=True,
    )

    print(
        f"RSI <= {RSI_LIMIT:.0f}",
        flush=True,
    )

    print(
        f"DEV >= {DEVIATION_ATR:.2f} ATR",
        flush=True,
    )

    print(
        f"VOL >= {VOLUME_MULT:.2f}x",
        flush=True,
    )

    print(
        "Checking real room to 5m EMA20:",
        flush=True,
    )

    print(
        "0.40 / 0.60 / 0.80 / 1.00 / "
        "1.20 / 1.50 / 1.80%",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    download_archive()

    total_train = blank_diag()
    total_hold = blank_diag()

    csv_rows = []

    with zipfile.ZipFile(
        ARCHIVE_FILE,
        "r",
    ) as zf:

        zip_index = build_zip_index(
            zf
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
                    f"{number}/25 {asset}: "
                    f"NO USD/USDT DATA",
                    flush=True,
                )

                continue

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

            if (
                len(rows1) < 5000
                or len(rows5) < 1000
                or len(rows15) < 350
            ):

                print(
                    f"{number}/25 {pair}: "
                    f"INSUFFICIENT DATA",
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

            train = scan_period(
                data1,
                data5,
                data15,
                TRAIN_START,
                TRAIN_END,
            )

            hold = scan_period(
                data1,
                data5,
                data15,
                HOLDOUT_START,
                HOLDOUT_END,
            )

            add_diag(
                total_train,
                train,
            )

            add_diag(
                total_hold,
                hold,
            )

            for period_name, d in (
                ("TRAIN", train),
                ("HOLDOUT", hold),
            ):

                if d[
                    "room_count"
                ]:

                    avg_room = (
                        d["room_sum"]
                        / d["room_count"]
                    )

                else:

                    avg_room = 0.0

                csv_rows.append({
                    "asset":
                        asset,

                    "pair":
                        pair,

                    "period":
                        period_name,

                    "candles":
                        d["candles"],

                    "regime":
                        d["regime"],

                    "setup5":
                        d["setup5"],

                    "reversal":
                        d["reversal"],

                    "maker_fill":
                        d["maker_fill"],

                    "unfilled":
                        d["unfilled"],

                    "room_040":
                        d["room_040"],

                    "room_060":
                        d["room_060"],

                    "room_080":
                        d["room_080"],

                    "room_100":
                        d["room_100"],

                    "room_120":
                        d["room_120"],

                    "room_150":
                        d["room_150"],

                    "room_180":
                        d["room_180"],

                    "avg_room":
                        f"{avg_room:.6f}",

                    "max_room":
                        f"{d['room_max']:.6f}",
                })

            print(
                f"{number}/25 {pair} | "
                f"TRAIN "
                f"R:{train['regime']} "
                f"S:{train['setup5']} "
                f"REV:{train['reversal']} "
                f"FILL:{train['maker_fill']} "
                f"ROOM1.2:{train['room_120']} | "
                f"HOLD "
                f"R:{hold['regime']} "
                f"S:{hold['setup5']} "
                f"REV:{hold['reversal']} "
                f"FILL:{hold['maker_fill']} "
                f"ROOM1.2:{hold['room_120']}",
                flush=True,
            )

            del rows1
            del rows5
            del rows15

            del data1
            del data5
            del data15

    write_diag_csv(
        csv_rows
    )

    write_result_csv(
        total_train,
        total_hold,
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
        "V10.1 DIAGNOSTIC COMPLETE",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    print_diag(
        "TRAIN TOTAL",
        total_train,
    )

    print_diag(
        "HOLDOUT TOTAL",
        total_hold,
    )

    print(
        "",
        flush=True,
    )

    print(
        "IMPORTANT:",
        flush=True,
    )

    print(
        "ROOM numbers are potential signals, "
        "NOT completed trades.",
        flush=True,
    )

    print(
        "Next strategy change will be based "
        "on these diagnostics.",
        flush=True,
    )


if __name__ == "__main__":
    main()
