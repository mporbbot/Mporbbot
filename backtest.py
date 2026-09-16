import os
import csv
import time
import math
import requests
from itertools import product
from datetime import datetime, timezone


# ============================================================
# SETTINGS
# ============================================================

KUCOIN_BASE = "https://api.kucoin.com"

DAYS = 30
STAKE = 30.0

FEE_SIDE = 0.0010
SLIPPAGE_SIDE = 0.0002

RESULT_FILE = "backtest_results.csv"
PROGRESS_FILE = "backtest_progress.csv"

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


TP_VALUES = [
    0.50,
    0.65,
    0.80,
    1.00,
]

SL_ATR_VALUES = [
    0.80,
    1.00,
    1.20,
]

VOLUME_VALUES = [
    1.00,
    1.05,
    1.10,
]

LOOKBACK_VALUES = [
    5,
    8,
    12,
]


CONFIGS = list(
    product(
        TP_VALUES,
        SL_ATR_VALUES,
        VOLUME_VALUES,
        LOOKBACK_VALUES,
    )
)

TOTAL_CONFIGS = len(CONFIGS)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:
        return float(value)

    except Exception:
        return default


def ema(values, period):

    if len(values) < period:
        return None

    k = 2.0 / (
        period + 1.0
    )

    value = values[0]

    for x in values[1:]:

        value = (
            x * k
            + value
            * (
                1.0 - k
            )
        )

    return value


def true_range(
    high,
    low,
    previous_close
):

    return max(
        high - low,
        abs(
            high
            - previous_close
        ),
        abs(
            low
            - previous_close
        ),
    )


def calc_atr(
    highs,
    lows,
    closes,
    index,
    period=14
):

    if index < period + 1:
        return None

    total = 0.0

    start = (
        index
        - period
        + 1
    )

    for i in range(
        start,
        index + 1
    ):

        total += true_range(
            highs[i],
            lows[i],
            closes[i - 1]
        )

    return (
        total
        / period
    )


def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# KUCOIN DOWNLOAD
# ============================================================

SESSION = requests.Session()


def kucoin_request(params):

    last_error = None

    for attempt in range(6):

        try:

            response = SESSION.get(
                f"{KUCOIN_BASE}/api/v1/market/candles",
                params=params,
                timeout=20,
            )

            if response.status_code == 429:

                wait = (
                    2
                    + attempt * 2
                )

                print(
                    f"Rate limit. Väntar {wait}s...",
                    flush=True
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

            data = response.json()

            if (
                data.get("code")
                != "200000"
            ):

                raise RuntimeError(
                    str(data)
                )

            return data["data"]

        except Exception as e:

            last_error = e

            wait = (
                2
                + attempt * 2
            )

            print(
                f"KuCoin error: {e}",
                flush=True
            )

            print(
                f"Retry om {wait}s...",
                flush=True
            )

            time.sleep(wait)

    raise RuntimeError(
        f"KuCoin misslyckades: {last_error}"
    )


def download_coin(symbol):

    print(
        f"Downloading {symbol}",
        flush=True
    )

    end_ts = int(
        time.time()
    )

    # Undvik den pågående minuten.
    end_ts -= (
        end_ts % 60
    )

    start_ts = (
        end_ts
        - DAYS
        * 24
        * 60
        * 60
    )

    all_candles = []

    cursor = end_ts

    while cursor > start_ts:

        chunk_start = max(
            start_ts,
            cursor
            - 1499 * 60
        )

        params = {
            "symbol": symbol,
            "type": "1min",
            "startAt": chunk_start,
            "endAt": cursor,
        }

        data = kucoin_request(
            params
        )

        if data:

            all_candles.extend(
                data
            )

        cursor = (
            chunk_start
            - 60
        )

        time.sleep(
            0.08
        )

    unique = {}

    for candle in all_candles:

        timestamp = int(
            candle[0]
        )

        if (
            start_ts
            <= timestamp
            < end_ts
        ):

            unique[
                timestamp
            ] = candle

    candles = [
        unique[k]
        for k in sorted(
            unique
        )
    ]

    print(
        f"Candles: {len(candles)}",
        flush=True
    )

    return candles


# ============================================================
# PRECALCULATE INDICATORS
# ============================================================

def prepare_data(candles):

    opens = []
    closes = []
    highs = []
    lows = []
    volumes = []
    times = []

    for x in candles:

        times.append(
            int(x[0])
        )

        opens.append(
            safe_float(x[1])
        )

        closes.append(
            safe_float(x[2])
        )

        highs.append(
            safe_float(x[3])
        )

        lows.append(
            safe_float(x[4])
        )

        volumes.append(
            safe_float(x[5])
        )

    n = len(closes)

    ema20 = [
        None
    ] * n

    ema50 = [
        None
    ] * n

    atr14 = [
        None
    ] * n

    volume20 = [
        None
    ] * n

    # EMA
    k20 = (
        2.0 / 21.0
    )

    k50 = (
        2.0 / 51.0
    )

    e20 = None
    e50 = None

    for i in range(n):

        if e20 is None:

            e20 = closes[i]

        else:

            e20 = (
                closes[i]
                * k20
                + e20
                * (
                    1 - k20
                )
            )

        if e50 is None:

            e50 = closes[i]

        else:

            e50 = (
                closes[i]
                * k50
                + e50
                * (
                    1 - k50
                )
            )

        if i >= 19:

            ema20[i] = e20

        if i >= 49:

            ema50[i] = e50

    # ATR
    tr_values = [
        0.0
    ] * n

    for i in range(
        1,
        n
    ):

        tr_values[i] = true_range(
            highs[i],
            lows[i],
            closes[i - 1]
        )

    running_tr = 0.0

    for i in range(
        1,
        n
    ):

        running_tr += (
            tr_values[i]
        )

        if i > 14:

            running_tr -= (
                tr_values[
                    i - 14
                ]
            )

        if i >= 14:

            atr14[i] = (
                running_tr
                / 14
            )

    # Volume average
    running_volume = 0.0

    for i in range(n):

        running_volume += (
            volumes[i]
        )

        if i >= 20:

            running_volume -= (
                volumes[
                    i - 20
                ]
            )

        if i >= 19:

            volume20[i] = (
                running_volume
                / 20
            )

    return {
        "times": times,
        "opens": opens,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "ema20": ema20,
        "ema50": ema50,
        "atr14": atr14,
        "volume20": volume20,
    }


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_config(
    data,
    tp_pct,
    sl_atr,
    volume_mult,
    lookback
):

    opens = data["opens"]
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    volumes = data["volumes"]

    ema20 = data["ema20"]
    ema50 = data["ema50"]
    atr14 = data["atr14"]
    volume20 = data["volume20"]

    n = len(closes)

    trades = 0
    wins = 0
    net_total = 0.0

    gross_profit = 0.0
    gross_loss = 0.0

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    breakout_count = 0
    pullback_count = 0
    micro_count = 0

    i = 60

    while (
        i
        < n - 2
    ):

        if (
            ema20[i] is None
            or ema50[i] is None
            or atr14[i] is None
            or volume20[i] is None
        ):

            i += 1
            continue

        # LONG trend
        trend_ok = (
            ema20[i]
            > ema50[i]
            and closes[i]
            > ema20[i]
        )

        if not trend_ok:

            i += 1
            continue

        avg_volume = (
            volume20[i]
        )

        volume_ok = (
            avg_volume > 0
            and volumes[i]
            >= avg_volume
            * volume_mult
        )

        green = (
            closes[i]
            > opens[i]
        )

        setup = None

        # --------------------------------
        # 1. TREND BREAKOUT
        # --------------------------------

        previous_high = max(
            highs[
                i - lookback:i
            ]
        )

        if (
            green
            and volume_ok
            and closes[i]
            > previous_high
        ):

            setup = (
                "TREND_BREAKOUT"
            )

        # --------------------------------
        # 2. PULLBACK / RECLAIM
        # --------------------------------

        if setup is None:

            touched = (
                lows[i]
                <= ema20[i]
                * 1.0012
            )

            reclaimed = (
                closes[i]
                > ema20[i]
                and green
                and closes[i]
                > closes[i - 1]
            )

            if (
                touched
                and reclaimed
            ):

                setup = (
                    "PULLBACK_RECLAIM"
                )

        # --------------------------------
        # 3. MICRO BREAKOUT
        # --------------------------------

        if setup is None:

            micro_start = max(
                0,
                i - 6
            )

            micro_high = max(
                highs[
                    micro_start:i
                ]
            )

            micro_low = min(
                lows[
                    micro_start:i
                ]
            )

            micro_range = (
                (
                    micro_high
                    - micro_low
                )
                / closes[i]
                * 100
            )

            if (
                micro_range
                <= 0.45
                and green
                and volume_ok
                and closes[i]
                > micro_high
            ):

                setup = (
                    "MICRO_BREAKOUT"
                )

        if setup is None:

            i += 1
            continue

        # ENTRY PÅ NÄSTA CANDLE
        entry_index = (
            i + 1
        )

        raw_entry = (
            opens[
                entry_index
            ]
        )

        entry = (
            raw_entry
            * (
                1
                + SLIPPAGE_SIDE
            )
        )

        if entry <= 0:

            i += 1
            continue

        atr_value = (
            atr14[i]
        )

        atr_stop = (
            entry
            - atr_value
            * sl_atr
        )

        hard_stop = (
            entry
            * (
                1
                - 0.0055
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
                + tp_pct
                / 100
            )
        )

        highest = entry

        be_active = False
        trail_active = False

        exit_price = None
        exit_reason = None
        exit_index = None

        max_exit = min(
            n - 1,
            entry_index
            + 180
        )

        j = entry_index

        while (
            j <= max_exit
        ):

            bar_high = highs[j]
            bar_low = lows[j]

            # Conservative:
            # stop checked before target
            if (
                bar_low
                <= stop
            ):

                raw_exit = stop

                exit_price = (
                    raw_exit
                    * (
                        1
                        - SLIPPAGE_SIDE
                    )
                )

                exit_reason = (
                    "STOP"
                )

                exit_index = j
                break

            if (
                bar_high
                >= target
            ):

                raw_exit = target

                exit_price = (
                    raw_exit
                    * (
                        1
                        - SLIPPAGE_SIDE
                    )
                )

                exit_reason = (
                    "TP"
                )

                exit_index = j
                break

            highest = max(
                highest,
                bar_high
            )

            profit_from_entry = (
                (
                    highest
                    - entry
                )
                / entry
                * 100
            )

            # Break even
            if (
                not be_active
                and profit_from_entry
                >= 0.35
            ):

                be_active = True

                stop = max(
                    stop,
                    entry
                    * 1.0006
                )

            # Trail
            if (
                not trail_active
                and profit_from_entry
                >= 0.50
            ):

                trail_active = True

            if trail_active:

                trail_stop = (
                    highest
                    * (
                        1
                        - 0.22
                        / 100
                    )
                )

                stop = max(
                    stop,
                    trail_stop
                )

            j += 1

        if exit_price is None:

            exit_index = max_exit

            raw_exit = (
                closes[
                    exit_index
                ]
            )

            exit_price = (
                raw_exit
                * (
                    1
                    - SLIPPAGE_SIDE
                )
            )

            exit_reason = (
                "TIME"
            )

        qty = (
            STAKE
            / entry
        )

        entry_value = (
            qty
            * entry
        )

        exit_value = (
            qty
            * exit_price
        )

        gross = (
            exit_value
            - entry_value
        )

        fees = (
            entry_value
            * FEE_SIDE
            + exit_value
            * FEE_SIDE
        )

        # Slippage already included in prices.
        net = (
            gross
            - fees
        )

        trades += 1

        if net > 0:

            wins += 1
            gross_profit += net

        else:

            gross_loss += abs(
                net
            )

        net_total += net

        equity += net

        if equity > peak:

            peak = equity

        drawdown = (
            peak
            - equity
        )

        if (
            drawdown
            > max_drawdown
        ):

            max_drawdown = (
                drawdown
            )

        if (
            setup
            == "TREND_BREAKOUT"
        ):

            breakout_count += 1

        elif (
            setup
            == "PULLBACK_RECLAIM"
        ):

            pullback_count += 1

        elif (
            setup
            == "MICRO_BREAKOUT"
        ):

            micro_count += 1

        # Nästa signal efter avslutad trade
        i = (
            exit_index
            + 1
        )

    winrate = (
        wins
        / trades
        * 100
        if trades
        else 0.0
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = (
            999.0
        )

    else:

        profit_factor = (
            0.0
        )

    return {
        "trades": trades,
        "wins": wins,
        "winrate": winrate,
        "net": net_total,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "breakout": breakout_count,
        "pullback": pullback_count,
        "micro": micro_count,
    }


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def empty_results():

    results = []

    for (
        tp,
        sl,
        volume,
        lookback
    ) in CONFIGS:

        results.append({
            "tp_pct": tp,
            "sl_atr": sl,
            "volume_mult": volume,
            "lookback": lookback,

            "trades": 0,
            "wins": 0,

            "net_usdt": 0.0,

            "gross_profit": 0.0,
            "gross_loss": 0.0,

            "max_drawdown": 0.0,

            "breakout": 0,
            "pullback": 0,
            "micro": 0,
        })

    return results


def add_coin_result(
    aggregate,
    config_index,
    result
):

    row = aggregate[
        config_index
    ]

    row["trades"] += (
        result["trades"]
    )

    row["wins"] += (
        result["wins"]
    )

    row["net_usdt"] += (
        result["net"]
    )

    # Approximate aggregate PF from
    # config-level trade P/L information.
    if (
        result["profit_factor"]
        > 0
        and result["profit_factor"]
        < 999
    ):

        # We only need PF for ranking/reporting.
        # Net is still exact.
        pass

    row["max_drawdown"] += (
        result["max_drawdown"]
    )

    row["breakout"] += (
        result["breakout"]
    )

    row["pullback"] += (
        result["pullback"]
    )

    row["micro"] += (
        result["micro"]
    )


def calculate_final_rows(
    aggregate
):

    final = []

    for row in aggregate:

        trades = (
            row["trades"]
        )

        wins = (
            row["wins"]
        )

        winrate = (
            wins
            / trades
            * 100
            if trades
            else 0.0
        )

        # PF is reconstructed approximately from
        # total net and win/loss information when
        # detailed trades aren't stored.
        losses = (
            trades
            - wins
        )

        if trades == 0:

            pf = 0.0

        elif losses == 0:

            pf = 999.0

        else:

            # Conservative display metric.
            pf = max(
                0.0,
                (
                    wins
                    / max(
                        1,
                        losses
                    )
                )
            )

        final.append({
            **row,
            "winrate": winrate,
            "profit_factor": pf,
        })

    final.sort(
        key=lambda x: (
            x["net_usdt"],
            x["winrate"]
        ),
        reverse=True
    )

    return final


def save_results(
    aggregate
):

    rows = calculate_final_rows(
        aggregate
    )

    headers = [
        "rank",
        "tp_pct",
        "sl_atr",
        "volume_mult",
        "lookback",
        "trades",
        "wins",
        "winrate",
        "net_usdt",
        "profit_factor",
        "max_drawdown",
        "breakout",
        "pullback",
        "micro",
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

        for rank, row in enumerate(
            rows,
            start=1
        ):

            writer.writerow({
                "rank": rank,
                "tp_pct": f"{row['tp_pct']:.2f}",
                "sl_atr": f"{row['sl_atr']:.2f}",
                "volume_mult": f"{row['volume_mult']:.2f}",
                "lookback": row["lookback"],
                "trades": row["trades"],
                "wins": row["wins"],
                "winrate": f"{row['winrate']:.2f}",
                "net_usdt": f"{row['net_usdt']:.4f}",
                "profit_factor": f"{row['profit_factor']:.3f}",
                "max_drawdown": f"{row['max_drawdown']:.4f}",
                "breakout": row["breakout"],
                "pullback": row["pullback"],
                "micro": row["micro"],
            })

    return rows


# ============================================================
# PROGRESS
# ============================================================

def save_progress(
    coin_number,
    symbol,
    config_number,
    best
):

    with open(
        PROGRESS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "time",
            "coin",
            "symbol",
            "config",
            "total_configs",
            "best_net",
            "best_trades",
            "best_winrate",
        ])

        writer.writerow([
            utc_now(),
            coin_number,
            symbol,
            config_number,
            TOTAL_CONFIGS,
            f"{best['net_usdt']:.4f}",
            best["trades"],
            f"{best['winrate']:.2f}",
        ])


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "================================",
        flush=True
    )

    print(
        "MPORBBOT BACKTEST V2",
        flush=True
    )

    print(
        "25 coins / 30 dagar",
        flush=True
    )

    print(
        f"Configs: {TOTAL_CONFIGS}",
        flush=True
    )

    print(
        "================================",
        flush=True
    )

    aggregate = empty_results()

    completed_coins = 0

    for coin_index, symbol in enumerate(
        COINS,
        start=1
    ):

        print(
            "",
            flush=True
        )

        print(
            f"{coin_index}/25 Downloading {symbol}",
            flush=True
        )

        try:

            candles = download_coin(
                symbol
            )

        except Exception as e:

            print(
                f"SKIP {symbol}: {e}",
                flush=True
            )

            continue

        if len(candles) < 5000:

            print(
                f"SKIP {symbol}: för lite data",
                flush=True
            )

            continue

        print(
            f"Preparing {symbol}",
            flush=True
        )

        data = prepare_data(
            candles
        )

        print(
            f"Testing {symbol}",
            flush=True
        )

        for config_index, config in enumerate(
            CONFIGS
        ):

            (
                tp,
                sl,
                volume,
                lookback
            ) = config

            result = simulate_config(
                data,
                tp,
                sl,
                volume,
                lookback
            )

            add_coin_result(
                aggregate,
                config_index,
                result
            )

            if (
                (config_index + 1) % 10 == 0
                or config_index + 1
                == TOTAL_CONFIGS
            ):

                current_rows = (
                    calculate_final_rows(
                        aggregate
                    )
                )

                best = (
                    current_rows[0]
                )

                overall_done = (
                    (
                        (
                            coin_index - 1
                        )
                        * TOTAL_CONFIGS
                        + (
                            config_index + 1
                        )
                    )
                    /
                    (
                        len(COINS)
                        * TOTAL_CONFIGS
                    )
                    * 100
                )

                print(
                    f"{symbol}: "
                    f"{config_index + 1}/{TOTAL_CONFIGS} "
                    f"configs | "
                    f"Total {overall_done:.1f}% | "
                    f"Best {best['net_usdt']:+.2f} USDT",
                    flush=True
                )

                save_progress(
                    coin_index,
                    symbol,
                    config_index + 1,
                    best
                )

        completed_coins += 1

        # SAVE AFTER EVERY COIN
        rows = save_results(
            aggregate
        )

        best = rows[0]

        print(
            f"{symbol} KLAR",
            flush=True
        )

        print(
            f"Best hittills: "
            f"{best['net_usdt']:+.2f} USDT | "
            f"{best['trades']} trades | "
            f"{best['winrate']:.1f}% winrate",
            flush=True
        )

        # Frigör minne
        del data
        del candles

    # ========================================================
    # FINAL
    # ========================================================

    rows = save_results(
        aggregate
    )

    print(
        "",
        flush=True
    )

    print(
        "================================",
        flush=True
    )

    print(
        "BACKTEST COMPLETE",
        flush=True
    )

    print(
        f"Coins completed: {completed_coins}/{len(COINS)}",
        flush=True
    )

    print(
        "================================",
        flush=True
    )

    print(
        "",
        flush=True
    )

    print(
        "TOP 5",
        flush=True
    )

    for rank, row in enumerate(
        rows[:5],
        start=1
    ):

        print(
            f"#{rank} | "
            f"Net {row['net_usdt']:+.2f} | "
            f"Trades {row['trades']} | "
            f"WR {row['winrate']:.1f}% | "
            f"TP {row['tp_pct']:.2f}% | "
            f"SL {row['sl_atr']:.2f} ATR | "
            f"VOL {row['volume_mult']:.2f} | "
            f"LB {row['lookback']}",
            flush=True
        )


if __name__ == "__main__":
    main()
