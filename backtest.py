import csv
import json
import os
import time
from bisect import bisect_left
from itertools import product

import requests

VERSION = "V3.1"
BASE = "https://api.kucoin.com"
DAYS = 30
TRAIN_DAYS = 20
STAKE = 100.0
FEE = 0.0010
SLIP = 0.0002

CHECKPOINT_FILE = "backtest_v31_checkpoint.json"
RESULT_FILE = "backtest_results.csv"
SETUP_FILE = "backtest_setup_results.csv"

COINS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "ADA-USDT", "DOGE-USDT", "LINK-USDT", "AVAX-USDT", "LTC-USDT",
    "NEAR-USDT", "APT-USDT", "SUI-USDT", "DOT-USDT", "TRX-USDT",
    "BCH-USDT", "UNI-USDT", "FIL-USDT", "ARB-USDT", "OP-USDT",
    "ATOM-USDT", "INJ-USDT", "AAVE-USDT", "ETC-USDT", "ICP-USDT",
]

TP_VALUES = [0.55, 0.75, 1.00]
SL_ATR_VALUES = [0.8, 1.0, 1.2]
VOLUME_VALUES = [1.05, 1.15]
LOOKBACK_VALUES = [8, 12]
TRAIL_START_VALUES = [0.45, 0.60]

CONFIGS = list(product(
    TP_VALUES,
    SL_ATR_VALUES,
    VOLUME_VALUES,
    LOOKBACK_VALUES,
    TRAIL_START_VALUES,
))

TOTAL_CONFIGS = len(CONFIGS)

session = requests.Session()


def sf(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def ema_series(values, period):
    out = [None] * len(values)

    if not values:
        return out

    k = 2.0 / (period + 1.0)
    current = values[0]

    for i, value in enumerate(values):
        if i:
            current = value * k + current * (1.0 - k)

        if i >= period - 1:
            out[i] = current

    return out


def atr_series(highs, lows, closes, period=14):
    n = len(closes)

    out = [None] * n
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
            out[i] = running / period

    return out


def previous_volume_average(volumes, period=20):
    # Genomsnitt av FÖREGÅENDE candles.
    # Aktuell candle räknas alltså inte med.

    out = [None] * len(volumes)

    prefix = [0.0]

    for value in volumes:
        prefix.append(prefix[-1] + value)

    for i in range(period, len(volumes)):
        out[i] = (
            prefix[i] - prefix[i - period]
        ) / period

    return out


def request_candles(
    symbol,
    timeframe,
    start_ts,
    end_exclusive
):
    tf_seconds = {
        "1min": 60,
        "5min": 300
    }[timeframe]

    last_open = end_exclusive - tf_seconds

    all_rows = {}

    cursor = last_open

    while cursor >= start_ts:

        chunk_start = max(
            start_ts,
            cursor - tf_seconds * 1490
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
                r = session.get(
                    BASE + "/api/v1/market/candles",
                    params=params,
                    timeout=25,
                )

                if r.status_code == 429:
                    time.sleep(2 + attempt * 2)
                    continue

                r.raise_for_status()

                payload = r.json()

                if payload.get("code") != "200000":
                    raise RuntimeError(
                        str(payload)
                    )

                for row in payload.get("data", []):
                    ts = int(row[0])

                    if (
                        start_ts
                        <= ts
                        < end_exclusive
                    ):
                        all_rows[ts] = row

                success = True
                break

            except Exception as exc:
                print(
                    f"{symbol} {timeframe} "
                    f"retry {attempt + 1}/7: {exc}",
                    flush=True
                )

                time.sleep(
                    2 + attempt * 2
                )

        if not success:
            raise RuntimeError(
                f"Download failed: "
                f"{symbol} {timeframe}"
            )

        cursor = (
            chunk_start
            - tf_seconds
        )

        time.sleep(0.10)

    return [
        all_rows[k]
        for k in sorted(all_rows)
    ]


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

        "volavg": previous_volume_average(
            volumes,
            20
        ),
    }


def build_5m_lookup(data5):

    return {
        ts: i
        for i, ts
        in enumerate(data5["times"])
    }


def get_closed_5m_index(
    signal_1m_open,
    lookup5
):
    # En 1m-signal vid timestamp t
    # är känd när candle stängt t+60.
    #
    # Vi använder senaste HELT stängda
    # 5m-candle och aldrig en candle
    # som fortfarande formas.

    signal_close = (
        signal_1m_open + 60
    )

    bucket_open = (
        (
            signal_close - 300
        )
        // 300
    ) * 300

    return lookup5.get(
        bucket_open
    )


def trend_ok(data5, index):

    if (
        index is None
        or index < 205
    ):
        return False

    e20 = data5["ema20"][index]
    e50 = data5["ema50"][index]
    e200 = data5["ema200"][index]

    old_e20 = (
        data5["ema20"][index - 3]
    )

    if (
        None in (
            e20,
            e50,
            e200,
            old_e20
        )
        or old_e20 <= 0
    ):
        return False

    close = (
        data5["closes"][index]
    )

    # EMA20 ska luta uppåt
    # över tre stängda 5m candles.

    slope_pct = (
        e20 - old_e20
    ) / old_e20 * 100.0

    return (
        e20 > e50
        and close > e20
        and close > e200
        and slope_pct > 0.015
    )


def blank_setup():

    return {
        "trades": 0,
        "wins": 0,
        "net": 0.0,
        "gross_wins": 0.0,
        "gross_losses": 0.0,
    }


def blank_result():

    return {
        "trades": 0,
        "wins": 0,
        "net": 0.0,
        "gross_wins": 0.0,
        "gross_losses": 0.0,
        "max_dd": 0.0,

        "setups": {
            "BREAKOUT": blank_setup(),
            "PULLBACK": blank_setup(),
            "MICRO": blank_setup(),
        },
    }


def record_trade(
    result,
    setup,
    net
):

    result["trades"] += 1
    result["net"] += net

    s = result["setups"][setup]

    s["trades"] += 1
    s["net"] += net

    if net > 0:

        result["wins"] += 1
        result["gross_wins"] += net

        s["wins"] += 1
        s["gross_wins"] += net

    else:

        loss = abs(net)

        result["gross_losses"] += loss
        s["gross_losses"] += loss


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

        if t[i] < start_ts:
            i += 1
            continue

        if t[i] >= end_ts:
            break

        if (
            ema20[i] is None
            or atr[i] is None
            or volavg[i] is None
            or c[i] <= 0
        ):
            i += 1
            continue

        idx5 = get_closed_5m_index(
            t[i],
            lookup5
        )

        if not trend_ok(
            data5,
            idx5
        ):
            i += 1
            continue

        # ==========================================
        # VOLATILITY FILTER
        # ==========================================

        atr_pct = (
            atr[i]
            / c[i]
            * 100.0
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

        if (
            candle_range <= 0
            or candle_range
            > atr[i] * 2.2
        ):
            i += 1
            continue

        body_ratio = (
            abs(c[i] - o[i])
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

        # ==========================================
        # BREAKOUT
        # ==========================================

        previous_high = max(
            h[i - lookback:i]
        )

        breakout_trigger = (
            previous_high
            * (
                1.0
                + 0.015 / 100.0
            )
        )

        if (
            green
            and volume_ok
            and c[i]
            >= breakout_trigger
        ):
            setup = "BREAKOUT"

        # ==========================================
        # PULLBACK RECLAIM
        # ==========================================

        if setup is None:

            distance_pct = (
                abs(
                    l[i]
                    - ema20[i]
                )
                / ema20[i]
                * 100.0
            )

            touched = (
                l[i] <= ema20[i]
                or distance_pct <= 0.12
            )

            reclaimed = (
                c[i] > ema20[i]
                and green
                and c[i] > c[i - 1]
            )

            prior_pullback = (
                l[i - 1] < l[i - 2]
                or c[i - 1] < c[i - 2]
            )

            pull_volume = (
                volavg[i] > 0
                and v[i]
                >= volavg[i] * 0.95
            )

            if (
                touched
                and reclaimed
                and prior_pullback
                and pull_volume
            ):
                setup = "PULLBACK"

        # ==========================================
        # MICRO BREAKOUT
        # ==========================================

        if setup is None:

            mh = max(
                h[i - 6:i]
            )

            ml = min(
                l[i - 6:i]
            )

            micro_range_pct = (
                mh - ml
            ) / c[i] * 100.0

            if (
                micro_range_pct <= 0.35
                and green
                and volume_ok
                and c[i] > mh
            ):
                setup = "MICRO"

        if setup is None:
            i += 1
            continue

        # ==========================================
        # ENTRY
        # ==========================================
        #
        # Signalen kommer från en stängd 1m candle.
        # Entry sker på nästa 1m open.
        # Slippage läggs på entryn.

        entry_i = i + 1

        if (
            entry_i >= len(c)
            or t[entry_i] >= end_ts
        ):
            break

        entry = (
            o[entry_i]
            * (1.0 + SLIP)
        )

        if entry <= 0:
            i += 1
            continue

        qty = (
            STAKE / entry
        )

        # ==========================================
        # INITIAL STOP
        # ==========================================

        atr_stop = (
            entry
            - atr[i] * sl_mult
        )

        hard_stop = (
            entry
            * (
                1.0
                - 0.60 / 100.0
            )
        )

        stop = max(
            atr_stop,
            hard_stop
        )

        target = (
            entry
            * (
                1.0
                + tp_pct / 100.0
            )
        )

        highest = entry

        be_active = False
        trail_active = False

        exit_price = None
        exit_i = None

        # Max 120 minuter
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

                exit_price = (
                    c[exit_i]
                    * (1.0 - SLIP)
                )

                break

            # ======================================
            # STOP
            # ======================================
            #
            # Konservativ intrabar-modell:
            # gammal stop kontrolleras innan
            # ny trailing/BE beräknas.

            if l[j] <= stop:

                exit_i = j

                exit_price = (
                    stop
                    * (1.0 - SLIP)
                )

                break

            # ======================================
            # TAKE PROFIT
            # ======================================

            if h[j] >= target:

                exit_i = j

                exit_price = (
                    target
                    * (1.0 - SLIP)
                )

                break

            highest = max(
                highest,
                h[j]
            )

            best_profit_pct = (
                highest - entry
            ) / entry * 100.0

            # ======================================
            # BREAK EVEN
            # ======================================

            if (
                not be_active
                and best_profit_pct
                >= 0.38
            ):

                be_active = True

                stop = max(
                    stop,
                    entry
                    * (
                        1.0
                        + 0.06 / 100.0
                    )
                )

            # ======================================
            # TRAILING
            # ======================================

            if (
                not trail_active
                and best_profit_pct
                >= trail_start
            ):
                trail_active = True

            if trail_active:

                trail_stop = (
                    highest
                    * (
                        1.0
                        - 0.22 / 100.0
                    )
                )

                stop = max(
                    stop,
                    trail_stop
                )

            j += 1

        # ==========================================
        # TIME EXIT
        # ==========================================

        if exit_price is None:

            exit_i = min(
                last_i,
                len(c) - 1
            )

            if t[exit_i] >= end_ts:

                while (
                    exit_i > entry_i
                    and t[exit_i]
                    >= end_ts
                ):
                    exit_i -= 1

            exit_price = (
                c[exit_i]
                * (1.0 - SLIP)
            )

        # ==========================================
        # PNL
        # ==========================================

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

        # Slippage är redan inbyggd i
        # entry och exit och dras därför
        # INTE en gång till.

        net = (
            gross - fees
        )

        record_trade(
            result,
            setup,
            net
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

        # Samma coin kan inte öppna
        # ytterligare position förrän
        # den gamla är stängd.

        i = max(
            i + 1,
            exit_i + 1
        )

    return result


def empty_total():
    return blank_result()


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

    total["gross_wins"] += (
        result["gross_wins"]
    )

    total["gross_losses"] += (
        result["gross_losses"]
    )

    # Detta är största drawdown
    # från en enskild coin.
    # Inte portfolio-DD.

    total["max_dd"] = max(
        total["max_dd"],
        result["max_dd"]
    )

    for name in (
        "BREAKOUT",
        "PULLBACK",
        "MICRO"
    ):

        dst = total["setups"][name]
        src = result["setups"][name]

        for key in (
            "trades",
            "wins",
            "net",
            "gross_wins",
            "gross_losses"
        ):
            dst[key] += src[key]


def metrics(total):

    trades = total["trades"]
    wins = total["wins"]

    wr = (
        wins / trades * 100.0
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

    avg = (
        total["net"] / trades
        if trades
        else 0.0
    )

    return (
        wr,
        pf,
        avg
    )


def setup_metrics(s):

    trades = s["trades"]

    wr = (
        s["wins"]
        / trades
        * 100.0
        if trades
        else 0.0
    )

    if s["gross_losses"] > 0:

        pf = (
            s["gross_wins"]
            / s["gross_losses"]
        )

    elif s["gross_wins"] > 0:

        pf = 999.0

    else:

        pf = 0.0

    return wr, pf


def signature(end_exclusive):

    return {
        "version": VERSION,
        "days": DAYS,
        "train_days": TRAIN_DAYS,
        "stake": STAKE,
        "fee": FEE,
        "slip": SLIP,
        "end_exclusive": end_exclusive,
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
    test_totals
):

    tmp = (
        CHECKPOINT_FILE
        + ".tmp"
    )

    payload = {
        "signature": sig,
        "completed": completed,
        "train_totals": train_totals,
        "test_totals": test_totals,
    }

    with open(
        tmp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f
        )

    os.replace(
        tmp,
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
            encoding="utf-8"
        ) as f:

            payload = json.load(f)

        if (
            payload.get("signature")
            != sig
        ):

            print(
                "Old checkpoint does not "
                "match this run; starting fresh.",
                flush=True
            )

            return None

        if (
            len(
                payload.get(
                    "train_totals",
                    []
                )
            )
            != TOTAL_CONFIGS
        ):
            return None

        if (
            len(
                payload.get(
                    "test_totals",
                    []
                )
            )
            != TOTAL_CONFIGS
        ):
            return None

        return payload

    except Exception as exc:

        print(
            f"Checkpoint could not "
            f"be loaded: {exc}",
            flush=True
        )

        return None


def write_results(
    train_totals,
    test_totals
):

    ranked = list(
        range(TOTAL_CONFIGS)
    )

    # ==========================================
    # RANKING ENDAST PÅ TRAIN
    # ==========================================

    ranked.sort(
        key=lambda idx: (
            train_totals[idx]["net"],
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
        "trail_start",

        "trades",
        "wins",
        "winrate",
        "net_usdt",
        "profit_factor",
        "max_drawdown",
        "avg_net_trade",
        "trades_per_week",

        "train_trades",
        "train_winrate",
        "train_net",
        "train_pf",
        "train_avg_net_trade",

        "breakout_trades",
        "breakout_net",
        "breakout_winrate",
        "breakout_pf",

        "pullback_trades",
        "pullback_net",
        "pullback_winrate",
        "pullback_pf",

        "micro_trades",
        "micro_net",
        "micro_winrate",
        "micro_pf",
    ]

    rows = []

    for rank, idx in enumerate(
        ranked[:10],
        start=1
    ):

        (
            tp,
            sl,
            vol,
            lb,
            trail
        ) = CONFIGS[idx]

        train = (
            train_totals[idx]
        )

        test = (
            test_totals[idx]
        )

        (
            train_wr,
            train_pf,
            train_avg
        ) = metrics(train)

        (
            test_wr,
            test_pf,
            test_avg
        ) = metrics(test)

        bwr, bpf = setup_metrics(
            test["setups"]["BREAKOUT"]
        )

        pwr, ppf = setup_metrics(
            test["setups"]["PULLBACK"]
        )

        mwr, mpf = setup_metrics(
            test["setups"]["MICRO"]
        )

        rows.append({

            "rank": rank,

            "tp_pct": tp,
            "sl_atr": sl,
            "volume_mult": vol,
            "lookback": lb,
            "trail_start": trail,

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

            "train_avg_net_trade":
                train_avg,

            # BREAKOUT TEST
            "breakout_trades":
                test["setups"]
                ["BREAKOUT"]
                ["trades"],

            "breakout_net":
                test["setups"]
                ["BREAKOUT"]
                ["net"],

            "breakout_winrate":
                bwr,

            "breakout_pf":
                bpf,

            # PULLBACK TEST
            "pullback_trades":
                test["setups"]
                ["PULLBACK"]
                ["trades"],

            "pullback_net":
                test["setups"]
                ["PULLBACK"]
                ["net"],

            "pullback_winrate":
                pwr,

            "pullback_pf":
                ppf,

            # MICRO TEST
            "micro_trades":
                test["setups"]
                ["MICRO"]
                ["trades"],

            "micro_net":
                test["setups"]
                ["MICRO"]
                ["net"],

            "micro_winrate":
                mwr,

            "micro_pf":
                mpf,
        })

    # ==========================================
    # MAIN RESULT FILE
    # ==========================================

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

        for row in rows:

            writer.writerow({
                k: (
                    f"{v:.6f}"
                    if isinstance(v, float)
                    else v
                )
                for k, v
                in row.items()
            })

    # ==========================================
    # SETUP RESULT FILE
    # ==========================================

    with open(
        SETUP_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        setup_headers = [
            "rank",
            "period",
            "setup",
            "trades",
            "wins",
            "winrate",
            "net_usdt",
            "profit_factor"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=setup_headers
        )

        writer.writeheader()

        for rank, idx in enumerate(
            ranked[:10],
            start=1
        ):

            for (
                period_name,
                total
            ) in (
                (
                    "TRAIN",
                    train_totals[idx]
                ),
                (
                    "TEST",
                    test_totals[idx]
                )
            ):

                for setup_name in (
                    "BREAKOUT",
                    "PULLBACK",
                    "MICRO"
                ):

                    s = (
                        total["setups"]
                        [setup_name]
                    )

                    wr, pf = (
                        setup_metrics(s)
                    )

                    writer.writerow({

                        "rank": rank,
                        "period":
                            period_name,

                        "setup":
                            setup_name,

                        "trades":
                            s["trades"],

                        "wins":
                            s["wins"],

                        "winrate":
                            f"{wr:.6f}",

                        "net_usdt":
                            f"{s['net']:.6f}",

                        "profit_factor":
                            f"{pf:.6f}",
                    })

    return ranked, rows


def main():

    # ==========================================
    # FAST BOUNDARY
    # ==========================================
    #
    # Om checkpoint finns använder vi
    # exakt samma tidsperiod när vi fortsätter.

    existing = None

    if os.path.exists(
        CHECKPOINT_FILE
    ):

        try:

            with open(
                CHECKPOINT_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                existing = json.load(f)

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
            existing["signature"].get(
                "end_exclusive",
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
        flush=True
    )

    print(
        "MPORBBOT BACKTEST V3.1",
        flush=True
    )

    print(
        "LOW-MEMORY / CHECKPOINT",
        flush=True
    )

    print(
        f"Stake: {STAKE:.0f} USDT",
        flush=True
    )

    print(
        f"Configs: {TOTAL_CONFIGS}",
        flush=True
    )

    print(
        "20d TRAIN + 10d untouched TEST",
        flush=True
    )

    print(
        "======================================",
        flush=True
    )

    checkpoint = (
        load_checkpoint(sig)
    )

    if checkpoint:

        completed = (
            checkpoint["completed"]
        )

        train_totals = (
            checkpoint["train_totals"]
        )

        test_totals = (
            checkpoint["test_totals"]
        )

        print(
            f"RESUME: "
            f"{len(completed)}/25 "
            f"coins already complete",
            flush=True
        )

    else:

        completed = []

        train_totals = [
            empty_total()
            for _ in CONFIGS
        ]

        test_totals = [
            empty_total()
            for _ in CONFIGS
        ]

    # ==========================================
    # EN COIN I TAGET
    # ==========================================

    for coin_no, symbol in enumerate(
        COINS,
        start=1
    ):

        if symbol in completed:

            print(
                f"{coin_no}/25 {symbol}: "
                f"checkpoint OK, skipping",
                flush=True
            )

            continue

        print(
            "",
            flush=True
        )

        print(
            f"{coin_no}/25 {symbol}: "
            f"downloading 1m",
            flush=True
        )

        rows1 = request_candles(
            symbol,
            "1min",
            start_ts,
            end_exclusive
        )

        print(
            f"{symbol}: "
            f"{len(rows1)} "
            f"1m candles",
            flush=True
        )

        print(
            f"{coin_no}/25 {symbol}: "
            f"downloading 5m",
            flush=True
        )

        rows5 = request_candles(
            symbol,
            "5min",
            start_ts,
            end_exclusive
        )

        print(
            f"{symbol}: "
            f"{len(rows5)} "
            f"5m candles",
            flush=True
        )

        if (
            len(rows1) < 1000
            or len(rows5) < 300
        ):

            print(
                f"{symbol}: "
                f"too little data, skipping",
                flush=True
            )

            completed.append(
                symbol
            )

            save_checkpoint(
                sig,
                completed,
                train_totals,
                test_totals
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

        # Rensa rådata direkt
        del rows1
        del rows5

        # ======================================
        # ALLA 72 CONFIGS PÅ DENNA COIN
        # ======================================

        for idx, config in enumerate(
            CONFIGS
        ):

            train_r = simulate(
                data1,
                data5,
                lookup5,
                config,
                start_ts,
                train_end
            )

            test_r = simulate(
                data1,
                data5,
                lookup5,
                config,
                train_end,
                end_exclusive
            )

            add_result(
                train_totals[idx],
                train_r
            )

            add_result(
                test_totals[idx],
                test_r
            )

            if (
                (idx + 1) % 12 == 0
                or idx + 1
                == TOTAL_CONFIGS
            ):

                twr, tpf, _ = metrics(
                    train_totals[idx]
                )

                print(
                    f"{symbol}: "
                    f"{idx + 1}/"
                    f"{TOTAL_CONFIGS} configs | "
                    f"TRAIN total "
                    f"{train_totals[idx]['net']:+.2f} | "
                    f"WR {twr:.1f}% | "
                    f"PF {tpf:.2f}",
                    flush=True
                )

        # ======================================
        # CHECKPOINT EFTER VARJE COIN
        # ======================================

        completed.append(
            symbol
        )

        save_checkpoint(
            sig,
            completed,
            train_totals,
            test_totals
        )

        best_so_far = max(
            range(TOTAL_CONFIGS),
            key=lambda x: (
                train_totals[x]["net"],
                metrics(
                    train_totals[x]
                )[1]
            )
        )

        bwr, bpf, _ = metrics(
            train_totals[
                best_so_far
            ]
        )

        print(
            f"CHECKPOINT "
            f"{len(completed)}/25 | "
            f"Best TRAIN so far "
            f"{train_totals[best_so_far]['net']:+.2f} | "
            f"WR {bwr:.1f}% | "
            f"PF {bpf:.2f}",
            flush=True
        )

        # ======================================
        # VIKTIGT FÖR LITEN DIGITALOCEAN WORKER
        # ======================================

        del data1
        del data5
        del lookup5

    # ==========================================
    # ALLA COINS KLARA
    # ==========================================

    ranked, rows = write_results(
        train_totals,
        test_totals
    )

    print(
        "",
        flush=True
    )

    print(
        "======================================",
        flush=True
    )

    print(
        "V3.1 COMPLETE",
        flush=True
    )

    print(
        "Ranking is based ONLY on TRAIN.",
        flush=True
    )

    print(
        "TEST was not used to select "
        "the ranking.",
        flush=True
    )

    print(
        "max_drawdown = worst "
        "single-coin DD, not portfolio DD.",
        flush=True
    )

    print(
        "======================================",
        flush=True
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
            f"TEST WR "
            f"{row['winrate']:.1f}% | "
            f"{row['trades']} trades",
            flush=True
        )

        print(
            f"  BREAKOUT "
            f"{row['breakout_net']:+.2f} | "
            f"PULLBACK "
            f"{row['pullback_net']:+.2f} | "
            f"MICRO "
            f"{row['micro_net']:+.2f}",
            flush=True
        )

    # Lyckad körning.
    # Ta bort checkpoint så nästa /backtest
    # börjar med en ny 30-dagarsperiod.

    try:
        os.remove(
            CHECKPOINT_FILE
        )

    except OSError:
        pass


if __name__ == "__main__":
    main()
