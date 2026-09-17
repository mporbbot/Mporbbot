import csv
import json
import os
import time
from bisect import bisect_left
from itertools import product

import requests


# ============================================================
# MPORBBOT BACKTEST V6
#
# V5 selective BREAK -> RETEST -> CONFIRM
# +
# exits matched to observed MFE/MAE
# +
# TRAIN-only coin selection
#
# TEST remains untouched for selection.
# ============================================================

VERSION = "V6"

BASE = "https://api.kucoin.com"

DAYS = 30
TRAIN_DAYS = 20

STAKE = 100.0

FEE_SIDE = 0.0010
SLIP_SIDE = 0.0002

CHECKPOINT_FILE = "backtest_v6_checkpoint.json"
RESULT_FILE = "backtest_results.csv"
COIN_RESULT_FILE = "backtest_v6_coins.csv"

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
# V6 GRID
#
# V5 showed average MFE ~0.50%, MAE ~0.28%.
#
# TP is therefore moved much closer:
# 0.45 / 0.60 / 0.75 / 0.90 %
#
# Initial stop is controlled by ATR, but also bounded.
#
# BE trigger:
# 0.35 / 0.45 %
#
# Breakout volume:
# 1.20 / 1.40
#
# = 4 * 3 * 2 * 2 = 48 configs
# ============================================================

TP_VALUES = [
    0.45,
    0.60,
    0.75,
    0.90,
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

BE_VALUES = [
    0.35,
    0.45,
]

CONFIGS = list(product(
    TP_VALUES,
    SL_ATR_VALUES,
    VOLUME_VALUES,
    BE_VALUES,
))

TOTAL_CONFIGS = len(CONFIGS)

BREAKOUT_LOOKBACK = 12
RETEST_WINDOW = 6

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


def atr_series(highs, lows, closes, period=14):

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


def previous_volume_average(volumes, period=20):

    result = [None] * len(volumes)

    prefix = [0.0]

    for value in volumes:
        prefix.append(prefix[-1] + value)

    for i in range(period, len(volumes)):

        result[i] = (
            prefix[i] - prefix[i - period]
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

    last_open = end_exclusive - seconds

    rows = {}
    cursor = last_open

    while cursor >= start_ts:

        chunk_start = max(
            start_ts,
            cursor - seconds * 1490
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
                    BASE + "/api/v1/market/candles",
                    params=params,
                    timeout=25,
                )

                if response.status_code == 429:

                    time.sleep(2 + attempt * 2)
                    continue

                response.raise_for_status()

                payload = response.json()

                if payload.get("code") != "200000":
                    raise RuntimeError(str(payload))

                for row in payload.get("data", []):

                    ts = int(row[0])

                    if start_ts <= ts < end_exclusive:
                        rows[ts] = row

                success = True
                break

            except Exception as exc:

                print(
                    f"{symbol} {timeframe} "
                    f"retry {attempt + 1}/7: {exc}",
                    flush=True,
                )

                time.sleep(2 + attempt * 2)

        if not success:

            raise RuntimeError(
                f"Download failed {symbol} {timeframe}"
            )

        cursor = chunk_start - seconds
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

        times.append(int(row[0]))
        opens.append(sf(row[1]))
        closes.append(sf(row[2]))
        highs.append(sf(row[3]))
        lows.append(sf(row[4]))
        volumes.append(sf(row[5]))

    return {
        "times": times,
        "opens": opens,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,

        "ema20": ema_series(closes, 20),
        "ema50": ema_series(closes, 50),
        "ema200": ema_series(closes, 200),

        "atr": atr_series(
            highs,
            lows,
            closes,
            14
        ),

        "volavg": previous_volume_average(
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
        for i, ts in enumerate(data5["times"])
    }


def get_closed_5m_index(signal_open, lookup):

    signal_close = signal_open + 60

    bucket = (
        ((signal_close - 300) // 300)
        * 300
    )

    return lookup.get(bucket)


def strong_trend(data5, index):

    if index is None or index < 205:
        return False

    e20 = data5["ema20"][index]
    e50 = data5["ema50"][index]
    e200 = data5["ema200"][index]

    old20 = data5["ema20"][index - 4]
    old50 = data5["ema50"][index - 4]

    if (
        e20 is None
        or e50 is None
        or e200 is None
        or old20 is None
        or old50 is None
        or old20 <= 0
        or old50 <= 0
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
    }


def add_result(target, source):

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

    target["max_dd"] = max(
        target["max_dd"],
        source["max_dd"]
    )


def metrics(result):

    trades = result["trades"]

    winrate = (
        result["wins"] / trades * 100.0
        if trades
        else 0.0
    )

    if result["gross_losses"] > 0:

        pf = (
            result["gross_wins"]
            / result["gross_losses"]
        )

    elif result["gross_wins"] > 0:
        pf = 999.0

    else:
        pf = 0.0

    avg_net = (
        result["net"] / trades
        if trades
        else 0.0
    )

    avg_mfe = (
        result["mfe_sum"] / trades
        if trades
        else 0.0
    )

    avg_mae = (
        result["mae_sum"] / trades
        if trades
        else 0.0
    )

    return (
        winrate,
        pf,
        avg_net,
        avg_mfe,
        avg_mae,
    )


# ============================================================
# V5 BREAK-RETEST ENTRY
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
    ):
        return None

    idx5 = get_closed_5m_index(
        times[i],
        lookup5
    )

    if not strong_trend(data5, idx5):
        return None

    atr_pct = (
        atr[i] / c[i] * 100.0
    )

    if (
        atr_pct < 0.07
        or atr_pct > 0.65
    ):
        return None

    candle_range = h[i] - l[i]

    if candle_range <= 0:
        return None

    body = c[i] - o[i]

    if body <= 0:
        return None

    body_ratio = body / candle_range

    if body_ratio < 0.50:
        return None

    close_position = (
        (c[i] - l[i])
        / candle_range
    )

    if close_position < 0.72:
        return None

    if candle_range > atr[i] * 1.8:
        return None

    if v[i] < volavg[i] * 1.05:
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

    first_breakout = max(
        BREAKOUT_LOOKBACK + 2,
        i - RETEST_WINDOW - 1,
    )

    last_breakout = i - 2

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
                b - BREAKOUT_LOOKBACK:
                b
            ]
        )

        breakout_level = (
            previous_high
            * (1.0 + 0.04 / 100.0)
        )

        b_range = h[b] - l[b]

        if b_range <= 0:
            continue

        b_body = c[b] - o[b]

        if b_body <= 0:
            continue

        b_body_ratio = (
            b_body / b_range
        )

        breakout_ok = (
            b_body_ratio >= 0.45
            and c[b] >= breakout_level
            and v[b] >= (
                volavg[b]
                * volume_mult
            )
        )

        if not breakout_ok:
            continue

        if b_range > atr[b] * 2.0:
            continue

        retest_index = None
        retest_low = None

        failed = False

        for r in range(
            b + 1,
            i
        ):

            if (
                c[r]
                < previous_high
                * (1.0 - 0.15 / 100.0)
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
# SIMULATE
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
        be_trigger,
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
        bisect_left(t, start_ts)
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
        # INITIAL STOP
        #
        # V5 average MAE was ~0.28%.
        #
        # Allow roughly 0.25-0.45%
        # depending on structure/ATR.
        # ====================================================

        atr_stop = (
            actual_entry
            - signal["atr"] * sl_atr
        )

        structure_stop = (
            signal["retest_low"]
            * (1.0 - 0.05 / 100.0)
        )

        hard_max_stop = (
            actual_entry
            * (1.0 - 0.45 / 100.0)
        )

        stop = max(
            atr_stop,
            structure_stop,
            hard_max_stop,
        )

        # Never start tighter than 0.25%.
        tightest_stop = (
            actual_entry
            * (1.0 - 0.25 / 100.0)
        )

        stop = min(
            stop,
            tightest_stop
        )

        target = (
            actual_entry
            * (1.0 + tp_pct / 100.0)
        )

        highest = actual_entry
        lowest = actual_entry

        be_active = False
        trail_active = False

        exit_i = None
        raw_exit = None
        actual_exit = None

        # 120 minutes.
        last_i = min(
            len(c) - 1,
            entry_i + 120
        )

        j = entry_i

        while j <= last_i:

            if t[j] >= end_ts:

                exit_i = max(
                    entry_i,
                    j - 1
                )

                raw_exit = c[exit_i]

                actual_exit = (
                    raw_exit
                    * (1.0 - SLIP_SIDE)
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

            # Conservative ordering.
            # Existing stop first.
            if l[j] <= stop:

                exit_i = j
                raw_exit = stop

                actual_exit = (
                    raw_exit
                    * (1.0 - SLIP_SIDE)
                )

                break

            # TP
            if h[j] >= target:

                exit_i = j
                raw_exit = target

                actual_exit = (
                    raw_exit
                    * (1.0 - SLIP_SIDE)
                )

                break

            best_profit = (
                (highest - actual_entry)
                / actual_entry
                * 100.0
            )

            # =================================================
            # BREAK EVEN
            #
            # Trigger at 0.35/0.45%.
            #
            # Lock enough to offset part
            # of trading friction.
            # =================================================

            if (
                not be_active
                and best_profit >= be_trigger
            ):

                be_active = True

                stop = max(
                    stop,
                    actual_entry
                    * (1.0 + 0.12 / 100.0)
                )

            # =================================================
            # TRAILING
            #
            # Only relevant for TP 0.75/0.90.
            # =================================================

            if (
                not trail_active
                and best_profit >= 0.60
            ):

                trail_active = True

            if trail_active:

                trail_stop = (
                    highest
                    * (1.0 - 0.25 / 100.0)
                )

                stop = max(
                    stop,
                    trail_stop
                )

            j += 1

        if actual_exit is None:

            exit_i = min(
                last_i,
                len(c) - 1
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

        mfe = max(
            0.0,
            (
                (highest - actual_entry)
                / actual_entry
                * 100.0
            )
        )

        mae = max(
            0.0,
            (
                (actual_entry - lowest)
                / actual_entry
                * 100.0
            )
        )

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

        result["gross_market"] += gross_market
        result["slippage"] += slippage_cost
        result["fees"] += fees
        result["net"] += net

        result["mfe_sum"] += mfe
        result["mae_sum"] += mae

        if net > 0:

            result["wins"] += 1
            result["gross_wins"] += net

        else:

            result["gross_losses"] += abs(net)

        equity += net

        peak = max(
            peak,
            equity
        )

        result["max_dd"] = max(
            result["max_dd"],
            peak - equity
        )

        i = max(
            i + 1,
            exit_i + 1
        )

    return result


# ============================================================
# CHECKPOINT
# ============================================================

def make_signature(end_exclusive):

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
    signature,
    completed,
    train_by_coin,
    test_by_coin,
):

    temp = (
        CHECKPOINT_FILE + ".tmp"
    )

    payload = {
        "signature": signature,
        "completed": completed,
        "train_by_coin": train_by_coin,
        "test_by_coin": test_by_coin,
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
# TRAIN-ONLY COIN SELECTION
# ============================================================

def select_coins(
    train_by_coin,
    config_index,
):

    candidates = []

    for symbol in COINS:

        if symbol not in train_by_coin:
            continue

        result = (
            train_by_coin[symbol][config_index]
        )

        wr, pf, avg, mfe, mae = (
            metrics(result)
        )

        # Require enough evidence.
        if result["trades"] < 5:
            continue

        # IMPORTANT:
        # Selection is TRAIN only.
        if (
            result["net"] > 0
            and pf > 1.05
            and avg > 0
        ):

            candidates.append(
                (
                    symbol,
                    result["net"],
                    pf,
                    result["trades"],
                )
            )

    candidates.sort(
        key=lambda x: (
            x[1],
            x[2],
        ),
        reverse=True,
    )

    # Limit concentration.
    return [
        item[0]
        for item in candidates[:12]
    ]


# ============================================================
# AGGREGATE SELECTED COINS
# ============================================================

def aggregate_selected(
    by_coin,
    selected,
    config_index,
):

    total = blank_result()

    for symbol in selected:

        add_result(
            total,
            by_coin[
                symbol
            ][config_index]
        )

    return total


# ============================================================
# WRITE RESULTS
# ============================================================

def write_results(
    train_by_coin,
    test_by_coin,
):

    candidates = []

    for idx, config in enumerate(CONFIGS):

        selected = select_coins(
            train_by_coin,
            idx
        )

        if not selected:
            continue

        train_total = (
            aggregate_selected(
                train_by_coin,
                selected,
                idx,
            )
        )

        test_total = (
            aggregate_selected(
                test_by_coin,
                selected,
                idx,
            )
        )

        train_metrics = metrics(
            train_total
        )

        candidates.append({
            "idx": idx,
            "selected": selected,
            "train": train_total,
            "test": test_total,
            "train_pf": train_metrics[1],
        })

    # Rank using TRAIN only.
    candidates.sort(
        key=lambda item: (
            item["train"]["net"],
            item["train_pf"],
        ),
        reverse=True,
    )

    headers = [
        "rank",
        "tp_pct",
        "sl_atr",
        "volume_mult",
        "be_trigger",

        "selected_coins",
        "coin_count",

        # TEST fields compatible
        # with Telegram result reader.
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

        # TRAIN
        "train_trades",
        "train_winrate",
        "train_net",
        "train_pf",
        "train_avg_net",
    ]

    rows = []

    for rank, item in enumerate(
        candidates[:10],
        start=1
    ):

        idx = item["idx"]

        (
            tp,
            sl,
            volume,
            be,
        ) = CONFIGS[idx]

        train = item["train"]
        test = item["test"]

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

        selected = item["selected"]

        row = {
            "rank": rank,

            "tp_pct": tp,
            "sl_atr": sl,
            "volume_mult": volume,
            "be_trigger": be,

            "selected_coins":
                ",".join(selected),

            "coin_count":
                len(selected),

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
                test["gross_market"],

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

            "train_trades":
                train["trades"],

            "train_winrate":
                train_wr,

            "train_net":
                train["net"],

            "train_pf":
                train_pf,

            "train_avg_net":
                train_avg,
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
    # PER-COIN DIAGNOSTICS
    # ========================================================

    with open(
        COIN_RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fields = [
            "rank",
            "symbol",
            "selected_by_train",

            "train_trades",
            "train_net",
            "train_pf",
            "train_wr",

            "test_trades",
            "test_net",
            "test_pf",
            "test_wr",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for rank, item in enumerate(
            candidates[:10],
            start=1
        ):

            idx = item["idx"]
            selected = item["selected"]

            for symbol in COINS:

                train = (
                    train_by_coin[
                        symbol
                    ][idx]
                )

                test = (
                    test_by_coin[
                        symbol
                    ][idx]
                )

                train_wr, train_pf, _, _, _ = (
                    metrics(train)
                )

                test_wr, test_pf, _, _, _ = (
                    metrics(test)
                )

                writer.writerow({
                    "rank": rank,
                    "symbol": symbol,

                    "selected_by_train":
                        symbol in selected,

                    "train_trades":
                        train["trades"],

                    "train_net":
                        f"{train['net']:.6f}",

                    "train_pf":
                        f"{train_pf:.6f}",

                    "train_wr":
                        f"{train_wr:.6f}",

                    "test_trades":
                        test["trades"],

                    "test_net":
                        f"{test['net']:.6f}",

                    "test_pf":
                        f"{test_pf:.6f}",

                    "test_wr":
                        f"{test_wr:.6f}",
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

                existing = json.load(file)

        except Exception:
            existing = None

    if (
        existing
        and isinstance(
            existing.get("signature"),
            dict
        )
    ):

        end_exclusive = int(
            existing[
                "signature"
            ].get("end", 0)
        )

    else:

        end_exclusive = int(
            time.time()
        )

        end_exclusive -= (
            end_exclusive % 60
        )

    signature = make_signature(
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
        "MPORBBOT BACKTEST V6",
        flush=True,
    )

    print(
        "SELECTIVE BREAK-RETEST",
        flush=True,
    )

    print(
        "TRAIN-ONLY COIN FILTER",
        flush=True,
    )

    print(
        "TEST UNTOUCHED",
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
        "TP 0.45-0.90%",
        flush=True,
    )

    print(
        "BE 0.35/0.45%",
        flush=True,
    )

    print(
        "20d TRAIN + 10d TEST",
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

        test_by_coin = (
            checkpoint[
                "test_by_coin"
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
        test_by_coin = {}

    # ========================================================
    # PROCESS EACH COIN
    # ========================================================

    for coin_no, symbol in enumerate(
        COINS,
        start=1
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
            start_ts,
            end_exclusive,
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
            start_ts,
            end_exclusive,
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

            train_by_coin[symbol] = [
                blank_result()
                for _ in CONFIGS
            ]

            test_by_coin[symbol] = [
                blank_result()
                for _ in CONFIGS
            ]

            completed.append(symbol)

            save_checkpoint(
                signature,
                completed,
                train_by_coin,
                test_by_coin,
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
        test_results = []

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

            train_results.append(
                train_result
            )

            test_results.append(
                test_result
            )

            if (
                (idx + 1) % 8 == 0
                or idx + 1 == TOTAL_CONFIGS
            ):

                wr, pf, avg, mfe, mae = (
                    metrics(train_result)
                )

                print(
                    f"{symbol}: "
                    f"{idx + 1}/{TOTAL_CONFIGS} | "
                    f"TRAIN {train_result['net']:+.2f} | "
                    f"{train_result['trades']} trades | "
                    f"PF {pf:.2f} | "
                    f"MFE {mfe:.2f}% | "
                    f"MAE {mae:.2f}%",
                    flush=True,
                )

        train_by_coin[
            symbol
        ] = train_results

        test_by_coin[
            symbol
        ] = test_results

        completed.append(symbol)

        save_checkpoint(
            signature,
            completed,
            train_by_coin,
            test_by_coin,
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
    # FINAL TRAIN SELECTION -> TEST
    # ========================================================

    rows = write_results(
        train_by_coin,
        test_by_coin,
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
        "V6 COMPLETE",
        flush=True,
    )

    print(
        "COINS + CONFIG SELECTED ON TRAIN ONLY",
        flush=True,
    )

    print(
        "TEST WAS NOT USED FOR SELECTION",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    if not rows:

        print(
            "No TRAIN configuration "
            "passed coin-selection rules.",
            flush=True,
        )

    for row in rows[:5]:

        print(
            f"TRAIN RANK #{row['rank']} | "
            f"TRAIN {row['train_net']:+.2f} "
            f"PF {row['train_pf']:.2f} | "
            f"TEST {row['net_usdt']:+.2f} "
            f"PF {row['profit_factor']:.2f}",
            flush=True,
        )

        print(
            f"  TEST {row['trades']} trades | "
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
            f"  Avg MFE "
            f"{row['avg_mfe_pct']:.2f}% | "
            f"MAE {row['avg_mae_pct']:.2f}%",
            flush=True,
        )

        print(
            f"  TP {row['tp_pct']:.2f}% | "
            f"SL {row['sl_atr']:.1f} ATR | "
            f"VOL {row['volume_mult']:.2f} | "
            f"BE {row['be_trigger']:.2f}%",
            flush=True,
        )

        print(
            f"  COINS ({row['coin_count']}): "
            f"{row['selected_coins']}",
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
