import csv
import time
import requests
from itertools import product
from datetime import datetime, timezone


# ============================================================
# V3 BACKTEST
# ============================================================

BASE = "https://api.kucoin.com"

DAYS = 30
TRAIN_DAYS = 20

STAKE = 100.0

FEE = 0.0010
SLIP = 0.0002

RESULT_FILE = "backtest_results.csv"

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
# PARAMETER GRID
# ============================================================

TP_VALUES = [
    0.55,
    0.75,
    1.00,
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

TRAIL_START_VALUES = [
    0.45,
    0.60,
]

CONFIGS = list(
    product(
        TP_VALUES,
        SL_ATR_VALUES,
        VOLUME_VALUES,
        LOOKBACK_VALUES,
        TRAIL_START_VALUES,
    )
)

TOTAL_CONFIGS = len(CONFIGS)


# ============================================================
# HELPERS
# ============================================================

session = requests.Session()


def sf(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def ema_series(values, period):

    result = [None] * len(values)

    if not values:
        return result

    k = 2.0 / (period + 1)

    current = values[0]

    for i, value in enumerate(values):

        if i == 0:
            current = value

        else:
            current = (
                value * k
                + current * (1 - k)
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
    trs = [0.0] * n

    for i in range(1, n):

        trs[i] = max(
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

        running += trs[i]

        if i > period:
            running -= trs[i - period]

        if i >= period:
            result[i] = (
                running / period
            )

    return result


def volume_average(
    volumes,
    period=20
):

    result = [None] * len(volumes)

    running = 0.0

    for i, value in enumerate(volumes):

        running += value

        if i >= period:
            running -= volumes[i - period]

        if i >= period - 1:
            result[i] = (
                running / period
            )

    return result


# ============================================================
# DOWNLOAD
# ============================================================

def request_candles(
    symbol,
    timeframe,
    days
):

    tf_seconds = {
        "1min": 60,
        "5min": 300,
    }[timeframe]

    now = int(time.time())

    # Don't use current forming candle
    end = (
        now
        - now % tf_seconds
        - tf_seconds
    )

    start = (
        end
        - days * 86400
    )

    all_rows = {}

    cursor = end

    while cursor > start:

        chunk_start = max(
            start,
            cursor
            - tf_seconds * 1490
        )

        params = {
            "symbol": symbol,
            "type": timeframe,
            "startAt": chunk_start,
            "endAt": cursor,
        }

        success = False

        for attempt in range(6):

            try:

                r = session.get(
                    BASE
                    + "/api/v1/market/candles",
                    params=params,
                    timeout=20,
                )

                if r.status_code == 429:

                    time.sleep(
                        2 + attempt * 2
                    )

                    continue

                r.raise_for_status()

                payload = r.json()

                if (
                    payload.get("code")
                    != "200000"
                ):
                    raise RuntimeError(
                        str(payload)
                    )

                for row in payload["data"]:

                    ts = int(row[0])

                    if start <= ts <= end:
                        all_rows[ts] = row

                success = True
                break

            except Exception as e:

                print(
                    f"{symbol} {timeframe} "
                    f"retry: {e}",
                    flush=True
                )

                time.sleep(
                    2 + attempt * 2
                )

        if not success:

            raise RuntimeError(
                f"Download failed "
                f"{symbol} {timeframe}"
            )

        cursor = (
            chunk_start
            - tf_seconds
        )

        time.sleep(0.08)

    return [
        all_rows[k]
        for k in sorted(all_rows)
    ]


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare(rows):

    times = []
    opens = []
    closes = []
    highs = []
    lows = []
    volumes = []

    for r in rows:

        times.append(int(r[0]))
        opens.append(sf(r[1]))
        closes.append(sf(r[2]))
        highs.append(sf(r[3]))
        lows.append(sf(r[4]))
        volumes.append(sf(r[5]))

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

        "volavg": volume_average(
            volumes,
            20
        ),
    }


# ============================================================
# MAP 1M -> CLOSED 5M CANDLE
# ============================================================

def build_5m_lookup(data5):

    lookup = {}

    times = data5["times"]

    for i, ts in enumerate(times):

        lookup[ts] = i

    return lookup


def get_5m_index(
    timestamp,
    lookup
):

    # We only use a fully closed 5m candle.
    bucket = (
        timestamp
        - timestamp % 300
        - 300
    )

    return lookup.get(bucket)


# ============================================================
# TREND FILTER
# ============================================================

def trend_ok(
    data5,
    index
):

    if index is None:
        return False

    if index < 205:
        return False

    e20 = data5["ema20"][index]
    e50 = data5["ema50"][index]
    e200 = data5["ema200"][index]

    if (
        e20 is None
        or e50 is None
        or e200 is None
    ):
        return False

    close = (
        data5["closes"][index]
    )

    # EMA20 slope over 3 completed 5m candles.
    old_e20 = (
        data5["ema20"][index - 3]
    )

    if old_e20 is None:
        return False

    slope_pct = (
        (
            e20 - old_e20
        )
        / old_e20
        * 100
    )

    return (
        e20 > e50
        and close > e20
        and close > e200
        and slope_pct > 0.015
    )


# ============================================================
# SINGLE CONFIG TEST
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
        sl_mult,
        volume_mult,
        lookback,
        trail_start,
    ) = config

    o = data1["opens"]
    h = data1["highs"]
    l = data1["lows"]
    c = data1["closes"]
    v = data1["volumes"]
    t = data1["times"]

    ema20 = data1["ema20"]
    atr = data1["atr"]
    volavg = data1["volavg"]

    trades = 0
    wins = 0

    net_total = 0.0

    gross_wins = 0.0
    gross_losses = 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    setups = {
        "BREAKOUT": {
            "trades": 0,
            "net": 0.0,
        },
        "PULLBACK": {
            "trades": 0,
            "net": 0.0,
        },
        "MICRO": {
            "trades": 0,
            "net": 0.0,
        },
    }

    i = 220

    while i < len(c) - 2:

        if t[i] < start_ts:

            i += 1
            continue

        if t[i] >= end_ts:
            break

        if (
            ema20[i] is None
            or atr[i] is None
            or volavg[i] is None
        ):

            i += 1
            continue

        idx5 = get_5m_index(
            t[i],
            lookup5
        )

        if not trend_ok(
            data5,
            idx5
        ):

            i += 1
            continue

        # --------------------------------
        # VOLATILITY FILTER
        # --------------------------------

        atr_pct = (
            atr[i]
            / c[i]
            * 100
        )

        if (
            atr_pct < 0.055
            or atr_pct > 0.80
        ):

            i += 1
            continue

        candle_range = (
            h[i] - l[i]
        )

        # Avoid huge impulse candles
        if (
            candle_range
            > atr[i] * 2.2
        ):

            i += 1
            continue

        body = abs(
            c[i] - o[i]
        )

        if candle_range <= 0:

            i += 1
            continue

        body_ratio = (
            body
            / candle_range
        )

        if body_ratio < 0.22:

            i += 1
            continue

        green = (
            c[i] > o[i]
        )

        volume_ok = (
            volavg[i] > 0
            and v[i]
            >= volavg[i]
            * volume_mult
        )

        setup = None

        # ====================================================
        # BREAKOUT
        # ====================================================

        previous_high = max(
            h[
                i - lookback:i
            ]
        )

        breakout_buffer = (
            previous_high
            * 0.00015
        )

        if (
            green
            and volume_ok
            and c[i]
            > previous_high
            + breakout_buffer
        ):

            setup = "BREAKOUT"

        # ====================================================
        # PULLBACK RECLAIM
        # ====================================================

        if setup is None:

            distance = (
                abs(
                    l[i]
                    - ema20[i]
                )
                / ema20[i]
                * 100
            )

            touched = (
                l[i] <= ema20[i]
                or distance <= 0.12
            )

            reclaim = (
                c[i] > ema20[i]
                and green
                and c[i] > c[i - 1]
            )

            # Require previous candle to have
            # actually pulled back.
            prior_pullback = (
                l[i - 1]
                < l[i - 2]
                or c[i - 1]
                < c[i - 2]
            )

            pull_volume = (
                v[i]
                >= volavg[i]
                * 0.95
            )

            if (
                touched
                and reclaim
                and prior_pullback
                and pull_volume
            ):

                setup = "PULLBACK"

        # ====================================================
        # MICRO BREAKOUT
        # ====================================================

        if setup is None:

            mh = max(
                h[i - 6:i]
            )

            ml = min(
                l[i - 6:i]
            )

            micro_range = (
                (
                    mh - ml
                )
                / c[i]
                * 100
            )

            if (
                micro_range <= 0.35
                and green
                and volume_ok
                and c[i] > mh
            ):

                setup = "MICRO"

        if setup is None:

            i += 1
            continue

        # ====================================================
        # ENTRY NEXT 1M OPEN
        # ====================================================

        entry_i = i + 1

        if t[entry_i] >= end_ts:
            break

        entry = (
            o[entry_i]
            * (
                1 + SLIP
            )
        )

        if entry <= 0:

            i += 1
            continue

        qty = (
            STAKE / entry
        )

        atr_stop = (
            entry
            - atr[i]
            * sl_mult
        )

        hard_stop = (
            entry
            * (
                1 - 0.0060
            )
        )

        stop = max(
            atr_stop,
            hard_stop
        )

        target = (
            entry
            * (
                1
                + tp_pct / 100
            )
        )

        highest = entry

        be_active = False
        trail_active = False

        exit_price = None
        exit_i = None

        # Max 120 minutes
        last_i = min(
            len(c) - 1,
            entry_i + 120
        )

        j = entry_i

        while j <= last_i:

            if t[j] >= end_ts:
                break

            # Conservative ordering.
            if l[j] <= stop:

                exit_price = (
                    stop
                    * (
                        1 - SLIP
                    )
                )

                exit_i = j
                break

            if h[j] >= target:

                exit_price = (
                    target
                    * (
                        1 - SLIP
                    )
                )

                exit_i = j
                break

            highest = max(
                highest,
                h[j]
            )

            profit_pct = (
                (
                    highest
                    - entry
                )
                / entry
                * 100
            )

            # Break-even after +0.38%
            if (
                not be_active
                and profit_pct >= 0.38
            ):

                be_active = True

                stop = max(
                    stop,
                    entry * 1.0006
                )

            # Trailing
            if (
                not trail_active
                and profit_pct
                >= trail_start
            ):

                trail_active = True

            if trail_active:

                trailing_stop = (
                    highest
                    * (
                        1 - 0.22 / 100
                    )
                )

                stop = max(
                    stop,
                    trailing_stop
                )

            j += 1

        if exit_price is None:

            if j >= len(c):
                exit_i = len(c) - 1
            else:
                exit_i = min(
                    j,
                    last_i
                )

            exit_price = (
                c[exit_i]
                * (
                    1 - SLIP
                )
            )

        entry_value = (
            qty * entry
        )

        exit_value = (
            qty * exit_price
        )

        gross = (
            exit_value
            - entry_value
        )

        fees = (
            entry_value * FEE
            + exit_value * FEE
        )

        # Slippage is already included in entry/exit.
        net = (
            gross - fees
        )

        trades += 1

        setups[setup][
            "trades"
        ] += 1

        setups[setup][
            "net"
        ] += net

        if net > 0:

            wins += 1
            gross_wins += net

        else:

            gross_losses += abs(net)

        net_total += net

        equity += net

        peak = max(
            peak,
            equity
        )

        max_dd = max(
            max_dd,
            peak - equity
        )

        # One open position per symbol.
        i = (
            exit_i + 1
        )

    winrate = (
        wins / trades * 100
        if trades
        else 0.0
    )

    if gross_losses > 0:

        pf = (
            gross_wins
            / gross_losses
        )

    elif gross_wins > 0:

        pf = 999.0

    else:

        pf = 0.0

    return {
        "trades": trades,
        "wins": wins,
        "winrate": winrate,
        "net": net_total,
        "pf": pf,
        "dd": max_dd,
        "setups": setups,
    }


# ============================================================
# AGGREGATION
# ============================================================

def blank_total():

    return {
        "trades": 0,
        "wins": 0,
        "net": 0.0,
        "gross_wins": 0.0,
        "gross_losses": 0.0,
        "dd": 0.0,

        "breakout_trades": 0,
        "breakout_net": 0.0,

        "pullback_trades": 0,
        "pullback_net": 0.0,

        "micro_trades": 0,
        "micro_net": 0.0,
    }


def add_result(
    total,
    result
):

    total["trades"] += (
        result["trades"]
    )

    total["wins"] += (
        result["wins"]
    )

    total["net"] += (
        result["net"]
    )

    # Reconstruct approximate gross
    # components from PF + net.
    pf = result["pf"]
    net = result["net"]

    if (
        pf > 0
        and pf < 999
        and abs(pf - 1.0) > 1e-9
    ):

        loss = (
            net
            / (
                pf - 1
            )
        )

        if loss > 0:

            win = (
                pf * loss
            )

            total[
                "gross_wins"
            ] += win

            total[
                "gross_losses"
            ] += loss

    elif (
        pf >= 999
        and net > 0
    ):

        total[
            "gross_wins"
        ] += net

    total["dd"] += (
        result["dd"]
    )

    s = result["setups"]

    total[
        "breakout_trades"
    ] += s["BREAKOUT"]["trades"]

    total[
        "breakout_net"
    ] += s["BREAKOUT"]["net"]

    total[
        "pullback_trades"
    ] += s["PULLBACK"]["trades"]

    total[
        "pullback_net"
    ] += s["PULLBACK"]["net"]

    total[
        "micro_trades"
    ] += s["MICRO"]["trades"]

    total[
        "micro_net"
    ] += s["MICRO"]["net"]


def finish(total):

    trades = total["trades"]
    wins = total["wins"]

    wr = (
        wins / trades * 100
        if trades
        else 0.0
    )

    if total["gross_losses"] > 0:

        pf = (
            total["gross_wins"]
            / total["gross_losses"]
        )

    elif total["gross_wins"] > 0:

        pf = 999.0

    else:

        pf = 0.0

    total["winrate"] = wr
    total["pf"] = pf

    return total


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================",
        flush=True
    )

    print(
        "MPORBBOT BACKTEST V3",
        flush=True
    )

    print(
        "Stake: 100 USDT",
        flush=True
    )

    print(
        "5m trend + 1m entries",
        flush=True
    )

    print(
        "20d TRAIN + 10d TEST",
        flush=True
    )

    print(
        f"Configs: {TOTAL_CONFIGS}",
        flush=True
    )

    print(
        "======================================",
        flush=True
    )

    # --------------------------------------------------------
    # Download once
    # --------------------------------------------------------

    coin_data = {}

    for number, symbol in enumerate(
        COINS,
        start=1
    ):

        print(
            f"{number}/25 Downloading {symbol} 1m",
            flush=True
        )

        try:

            rows1 = request_candles(
                symbol,
                "1min",
                DAYS
            )

            print(
                f"{symbol}: {len(rows1)} 1m candles",
                flush=True
            )

            print(
                f"{number}/25 Downloading {symbol} 5m",
                flush=True
            )

            rows5 = request_candles(
                symbol,
                "5min",
                DAYS
            )

            print(
                f"{symbol}: {len(rows5)} 5m candles",
                flush=True
            )

            coin_data[symbol] = {
                "1m": prepare(rows1),
                "5m": prepare(rows5),
            }

        except Exception as e:

            print(
                f"SKIP {symbol}: {e}",
                flush=True
            )

    if not coin_data:

        raise RuntimeError(
            "No coin data downloaded."
        )

    # Use the common data boundary.
    latest_end = min(
        data["1m"]["times"][-1]
        for data in coin_data.values()
    )

    full_start = (
        latest_end
        - DAYS * 86400
    )

    train_end = (
        full_start
        + TRAIN_DAYS * 86400
    )

    test_end = latest_end + 60

    print(
        "",
        flush=True
    )

    print(
        "DOWNLOAD COMPLETE",
        flush=True
    )

    print(
        "Starting TRAIN...",
        flush=True
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_results = []

    for config_no, config in enumerate(
        CONFIGS,
        start=1
    ):

        total = blank_total()

        for symbol, data in coin_data.items():

            lookup5 = build_5m_lookup(
                data["5m"]
            )

            r = simulate(
                data["1m"],
                data["5m"],
                lookup5,
                config,
                full_start,
                train_end,
            )

            add_result(
                total,
                r
            )

        total = finish(
            total
        )

        train_results.append(
            {
                "config": config,
                "result": total,
            }
        )

        print(
            f"TRAIN {config_no}/{TOTAL_CONFIGS} | "
            f"Trades {total['trades']} | "
            f"WR {total['winrate']:.1f}% | "
            f"Net {total['net']:+.2f} | "
            f"PF {total['pf']:.2f}",
            flush=True
        )

    # Require meaningful sample.
    eligible = [
        x
        for x in train_results
        if x["result"]["trades"] >= 300
    ]

    if not eligible:

        eligible = train_results

    # Selection is based ONLY on TRAIN.
    eligible.sort(
        key=lambda x: (
            x["result"]["net"],
            x["result"]["pf"],
        ),
        reverse=True
    )

    # Test top 10 train configs on untouched data.
    finalists = eligible[:10]

    print(
        "",
        flush=True
    )

    print(
        "TRAIN COMPLETE",
        flush=True
    )

    print(
        "Testing top 10 on untouched TEST period...",
        flush=True
    )

    final_rows = []

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    for rank, candidate in enumerate(
        finalists,
        start=1
    ):

        config = candidate["config"]

        test_total = blank_total()

        for symbol, data in coin_data.items():

            lookup5 = build_5m_lookup(
                data["5m"]
            )

            r = simulate(
                data["1m"],
                data["5m"],
                lookup5,
                config,
                train_end,
                test_end,
            )

            add_result(
                test_total,
                r
            )

        test_total = finish(
            test_total
        )

        train = candidate[
            "result"
        ]

        (
            tp,
            sl,
            vol,
            lb,
            trail,
        ) = config

        final_rows.append({
            "rank": rank,

            "tp_pct": tp,
            "sl_atr": sl,
            "volume_mult": vol,
            "lookback": lb,
            "trail_start": trail,

            "trades": test_total["trades"],
            "wins": test_total["wins"],
            "winrate": test_total["winrate"],
            "net_usdt": test_total["net"],
            "profit_factor": test_total["pf"],
            "max_drawdown": test_total["dd"],

            "train_trades": train["trades"],
            "train_winrate": train["winrate"],
            "train_net": train["net"],
            "train_pf": train["pf"],

            "breakout_trades":
                test_total["breakout_trades"],

            "breakout_net":
                test_total["breakout_net"],

            "pullback_trades":
                test_total["pullback_trades"],

            "pullback_net":
                test_total["pullback_net"],

            "micro_trades":
                test_total["micro_trades"],

            "micro_net":
                test_total["micro_net"],
        })

        print(
            f"TEST #{rank} | "
            f"Trades {test_total['trades']} | "
            f"WR {test_total['winrate']:.1f}% | "
            f"Net {test_total['net']:+.2f} | "
            f"PF {test_total['pf']:.2f}",
            flush=True
        )

    # Important:
    # rank remains TRAIN ranking.
    # We do NOT choose a winner by looking at TEST.
    headers = [
        "rank",
        "tp_pct",
        "sl_atr",
        "volume_mult",
        "lookback",
        "trail_start",

        "trades",
        "wins",
        "winrate",
        "net_usdt",
        "profit_factor",
        "max_drawdown",

        "train_trades",
        "train_winrate",
        "train_net",
        "train_pf",

        "breakout_trades",
        "breakout_net",

        "pullback_trades",
        "pullback_net",

        "micro_trades",
        "micro_net",
    ]

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=headers
        )

        writer.writeheader()

        for row in final_rows:

            writer.writerow({
                k: (
                    f"{v:.4f}"
                    if isinstance(v, float)
                    else v
                )
                for k, v in row.items()
            })

    print(
        "",
        flush=True
    )

    print(
        "======================================",
        flush=True
    )

    print(
        "V3 COMPLETE",
        flush=True
    )

    print(
        "TEST data was NOT used for selection.",
        flush=True
    )

    print(
        "======================================",
        flush=True
    )

    for row in final_rows[:5]:

        print(
            f"TRAIN RANK #{row['rank']} | "
            f"TRAIN {row['train_net']:+.2f} | "
            f"TEST {row['net_usdt']:+.2f} | "
            f"TEST WR {row['winrate']:.1f}% | "
            f"TEST PF {row['profit_factor']:.2f}",
            flush=True
        )

        print(
            f"  Breakout: "
            f"{row['breakout_net']:+.2f} "
            f"({row['breakout_trades']} trades)",
            flush=True
        )

        print(
            f"  Pullback: "
            f"{row['pullback_net']:+.2f} "
            f"({row['pullback_trades']} trades)",
            flush=True
        )

        print(
            f"  Micro: "
            f"{row['micro_net']:+.2f} "
            f"({row['micro_trades']} trades)",
            flush=True
        )


if __name__ == "__main__":
    main()
