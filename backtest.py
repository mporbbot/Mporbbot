import requests
import time
import csv
import itertools

BASE = "https://api.kucoin.com"

DAYS = 30
STAKE = 30.0

FEE = 0.0010
SLIP = 0.0002

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


def ema(x, n):
    if len(x) < n:
        return None

    k = 2 / (n + 1)
    e = x[0]

    for z in x[1:]:
        e = z * k + e * (1 - k)

    return e


def atr(h, l, c, n=14):
    if len(c) < n + 2:
        return None

    tr = []

    for i in range(1, len(c)):
        tr.append(
            max(
                h[i] - l[i],
                abs(h[i] - c[i - 1]),
                abs(l[i] - c[i - 1]),
            )
        )

    return sum(tr[-n:]) / n


def fetch(sym):
    sec = 60

    end = int(time.time())
    start = end - DAYS * 86400

    out = []

    session = requests.Session()

    cur = start

    while cur < end:

        chunk_end = min(
            end,
            cur + sec * 1499
        )

        r = session.get(
            BASE + "/api/v1/market/candles",
            params={
                "symbol": sym,
                "type": "1min",
                "startAt": cur,
                "endAt": chunk_end,
            },
            timeout=20,
        )

        r.raise_for_status()

        j = r.json()

        if j.get("code") != "200000":
            raise RuntimeError(j)

        out += j.get("data", [])

        cur = chunk_end + sec

        time.sleep(0.06)

    unique = {
        int(x[0]): x
        for x in out
    }

    return [
        unique[k]
        for k in sorted(unique)
    ]


def simulate(sym, rows, cfg):

    o = [float(x[1]) for x in rows]
    c = [float(x[2]) for x in rows]
    h = [float(x[3]) for x in rows]
    l = [float(x[4]) for x in rows]
    v = [float(x[5]) for x in rows]

    values = []

    setups = {
        "BREAKOUT": [],
        "PULLBACK": [],
        "MICRO": [],
    }

    i = 60

    while i < len(c) - 2:

        fast = ema(
            c[max(0, i - 150):i],
            20
        )

        slow = ema(
            c[max(0, i - 180):i],
            50
        )

        if (
            fast is None
            or slow is None
            or fast <= slow
        ):
            i += 1
            continue

        avg_volume = (
            sum(v[i - 20:i]) / 20
        )

        previous_high = max(
            h[i - cfg["lb"]:i]
        )

        breakout = (
            c[i] > previous_high * 1.0002
            and c[i] > o[i]
            and v[i] >= avg_volume * cfg["vol"]
        )

        e20 = ema(
            c[max(0, i - 60):i + 1],
            20
        )

        pullback = False

        if e20:

            pullback = (
                l[i] <= e20 * 1.0018
                and c[i] > e20
                and c[i] > o[i]
                and c[i] > c[i - 1]
            )

        micro_high = max(
            h[i - 6:i]
        )

        micro_low = min(
            l[i - 6:i]
        )

        micro_range = (
            (micro_high - micro_low)
            / c[i]
            * 100
        )

        micro = (
            micro_range <= 0.60
            and c[i] > micro_high
            and c[i] > o[i]
            and v[i]
            >= avg_volume
            * max(
                1.0,
                cfg["vol"] - 0.05
            )
        )

        if breakout:
            setup = "BREAKOUT"

        elif pullback:
            setup = "PULLBACK"

        elif micro:
            setup = "MICRO"

        else:
            i += 1
            continue

        a = atr(
            h[:i + 1],
            l[:i + 1],
            c[:i + 1]
        )

        if not a:
            i += 1
            continue

        # Entry sker på nästa candle.
        # Slippage läggs in i själva execution-priset.

        entry = (
            o[i + 1]
            * (1 + SLIP)
        )

        qty = STAKE / entry

        stop = max(
            entry - a * cfg["sl"],
            entry * 0.9945
        )

        tp = (
            entry
            * (
                1
                + cfg["tp"] / 100
            )
        )

        highest = entry

        j = i + 1

        exit_price = None

        while j < min(
            len(c) - 1,
            i + 181
        ):

            # Konservativ ordning:
            # stop kontrolleras före TP
            # om båda träffas i samma candle.

            if l[j] <= stop:

                exit_price = (
                    stop
                    * (1 - SLIP)
                )

                break

            highest = max(
                highest,
                h[j]
            )

            move = (
                (highest - entry)
                / entry
                * 100
            )

            # Break-even

            if move >= 0.35:

                stop = max(
                    stop,
                    entry * 1.0005
                )

            # Trailing

            if move >= 0.50:

                trail = (
                    highest
                    * (
                        1
                        - 0.22 / 100
                    )
                )

                stop = max(
                    stop,
                    trail
                )

            # Take profit

            if h[j] >= tp:

                exit_price = (
                    tp
                    * (1 - SLIP)
                )

                break

            j += 1

        # Time exit

        if exit_price is None:

            exit_price = (
                c[min(j, len(c) - 1)]
                * (1 - SLIP)
            )

        gross = (
            exit_price - entry
        ) * qty

        fees = (
            entry * qty
            + exit_price * qty
        ) * FEE

        # Viktigt:
        # slippage är redan inbyggd
        # i entry och exit.
        # Den dras därför INTE igen.

        net = gross - fees

        values.append(net)

        setups[setup].append(net)

        # Ingen ny position förrän
        # den gamla är avslutad.

        i = j + 1

    return values, setups


def metrics(values):

    if not values:
        return (
            0,
            0,
            0,
            0,
            0
        )

    wins = [
        x for x in values
        if x > 0
    ]

    losses = [
        x for x in values
        if x < 0
    ]

    gross_profit = sum(wins)

    gross_loss = abs(
        sum(losses)
    )

    equity = 0
    peak = 0
    max_dd = 0

    for x in values:

        equity += x

        peak = max(
            peak,
            equity
        )

        max_dd = max(
            max_dd,
            peak - equity
        )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = 999

    return (
        len(values),
        100
        * len(wins)
        / len(values),
        sum(values),
        profit_factor,
        max_dd,
    )


def main():

    data = {}

    print()
    print("DOWNLOADING HISTORICAL DATA")
    print("===========================")
    print()

    for n, symbol in enumerate(
        COINS,
        1
    ):

        try:

            print(
                f"{n}/{len(COINS)} "
                f"Downloading {symbol}"
            )

            data[symbol] = fetch(
                symbol
            )

            print(
                "Candles:",
                len(data[symbol])
            )

        except Exception as e:

            print(
                "ERROR",
                symbol,
                e
            )

    print()
    print("DOWNLOAD COMPLETE")
    print()

    grid = []

    for (
        tp,
        sl,
        volume,
        lookback
    ) in itertools.product(

        [
            0.50,
            0.65,
            0.80,
            1.00
        ],

        [
            0.8,
            1.0,
            1.2
        ],

        [
            1.00,
            1.05,
            1.10
        ],

        [
            5,
            8,
            12
        ],
    ):

        grid.append({
            "tp": tp,
            "sl": sl,
            "vol": volume,
            "lb": lookback,
        })

    results = []

    print(
        "TESTING",
        len(grid),
        "CONFIGURATIONS"
    )

    print()

    for number, cfg in enumerate(
        grid,
        1
    ):

        all_trades = []

        setup_results = {
            "BREAKOUT": [],
            "PULLBACK": [],
            "MICRO": [],
        }

        for symbol, rows in data.items():

            trades, setups = simulate(
                symbol,
                rows,
                cfg
            )

            all_trades += trades

            for setup_name in setup_results:

                setup_results[
                    setup_name
                ] += setups[
                    setup_name
                ]

        result = metrics(
            all_trades
        )

        results.append(
            (
                result,
                cfg,
                setup_results
            )
        )

        print(
            f"{number}/{len(grid)} "
            f"Trades={result[0]} "
            f"WR={result[1]:.1f}% "
            f"Net={result[2]:+.2f} "
            f"PF={result[3]:.2f} "
            f"DD={result[4]:.2f}"
        )

    results.sort(
        key=lambda x: (
            x[0][2],
            x[0][3]
        ),
        reverse=True
    )

    with open(
        "backtest_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "trades",
            "winrate",
            "net_usdt",
            "profit_factor",
            "max_drawdown",
            "tp_pct",
            "sl_atr",
            "volume_mult",
            "lookback",
        ])

        for rank, (
            result,
            cfg,
            setup_results
        ) in enumerate(
            results,
            1
        ):

            writer.writerow([
                rank,
                result[0],
                f"{result[1]:.2f}",
                f"{result[2]:.4f}",
                f"{result[3]:.4f}",
                f"{result[4]:.4f}",
                cfg["tp"],
                cfg["sl"],
                cfg["vol"],
                cfg["lb"],
            ])

    print()
    print("============================")
    print("TOP 5 CONFIGURATIONS")
    print("============================")

    for rank, (
        result,
        cfg,
        setup_results
    ) in enumerate(
        results[:5],
        1
    ):

        print()
        print(
            "RANK",
            rank
        )

        print(
            "TP:",
            cfg["tp"]
        )

        print(
            "SL ATR:",
            cfg["sl"]
        )

        print(
            "Volume:",
            cfg["vol"]
        )

        print(
            "Lookback:",
            cfg["lb"]
        )

        print()

        print(
            "ALL TRADES"
        )

        print(
            "Trades:",
            result[0]
        )

        print(
            "Winrate:",
            f"{result[1]:.2f}%"
        )

        print(
            "Net:",
            f"{result[2]:+.4f} USDT"
        )

        print(
            "Profit Factor:",
            f"{result[3]:.3f}"
        )

        print(
            "Max Drawdown:",
            f"{result[4]:.4f} USDT"
        )

        print()

        for setup_name in [
            "BREAKOUT",
            "PULLBACK",
            "MICRO"
        ]:

            setup_metric = metrics(
                setup_results[
                    setup_name
                ]
            )

            print(
                setup_name
            )

            print(
                "  Trades:",
                setup_metric[0]
            )

            print(
                "  Winrate:",
                f"{setup_metric[1]:.2f}%"
            )

            print(
                "  Net:",
                f"{setup_metric[2]:+.4f}"
            )

            print(
                "  PF:",
                f"{setup_metric[3]:.3f}"
            )

    print()
    print("============================")
    print("FINISHED")
    print("============================")
    print()

    print(
        "Results saved to:"
    )

    print(
        "backtest_results.csv"
    )

    print()

    print(
        "Do not change the trading bot "
        "until these results have been "
        "checked on a separate unseen period."
    )


if __name__ == "__main__":
    main()
