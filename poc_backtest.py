
"""Historical candle-based POC backtest.
Run: python poc_backtest.py 14 100 ETH-USDT
"""
import sys
import time
import json
import csv
from pathlib import Path
from main import Bot

OUT = Path('poc_backtest_results.json')
TRADES = Path('poc_backtest_trades.csv')


def history(bot, symbol, tf, days):
    seconds = {'5min': 300, '1hour': 3600}[tf]
    end = int(time.time()) // seconds * seconds

    warmup = 230 * 3600 if tf == '1hour' else 320 * 300
    start = end - (days * 86400 + warmup)

    all_rows = {}
    cursor = start

    while cursor < end:
        to = min(end, cursor + seconds * 1450)

        raw = bot.get(
            '/api/v1/market/candles',
            {
                'symbol': symbol,
                'type': tf,
                'startAt': cursor,
                'endAt': to
            }
        )

        for c in raw:
            t = int(c[0])

            if t + seconds <= end:
                all_rows[t] = {
                    'time': t,
                    'open': float(c[1]),
                    'close': float(c[2]),
                    'high': float(c[3]),
                    'low': float(c[4]),
                    'volume': float(c[5])
                }

        cursor = to + 1
        time.sleep(0.22)

    return sorted(
        all_rows.values(),
        key=lambda c: c['time']
    )


def test_symbol(bot, symbol, days, stake):
    five = history(bot, symbol, '5min', days)
    hour = history(bot, symbol, '1hour', days)

    print(
        symbol,
        'candles',
        len(five),
        '5m',
        len(hour),
        '1h',
        flush=True
    )

    if len(five) < 320 or len(hour) < 200:
        return []

    hidx = 0
    position = None
    trades = []
    fee = 0.001
    slip = 0.0002
    cutoff = int(time.time()) - days * 86400

    for i in range(301, len(five) - 1):
        bar = five[i]
        nextbar = five[i + 1]

        if bar['time'] < cutoff:
            continue

        if position:
            p = position

            # Kontrollera befintlig stop före
            # uppdatering av trailing.
            if bar['low'] <= p['stop']:
                px = min(
                    bar['open'],
                    p['stop']
                ) * (1 - slip)

                gross = p['qty'] * (px - p['entry'])
                fees = fee * (
                    stake + p['qty'] * px
                )
                net = gross - fees

                trades.append({
                    'symbol': symbol,
                    'entry_time': p['time'],
                    'exit_time': bar['time'],
                    'entry': p['entry'],
                    'exit': px,
                    'net': net,
                    'reason': (
                        'TRAIL'
                        if p['active']
                        else 'STOP'
                    )
                })

                position = None
                continue

            risk = p['entry'] - p['initial_stop']

            if (
                not p['active']
                and bar['high'] >= p['entry'] + 2 * risk
            ):
                p['active'] = True
                p['stop'] = max(
                    p['stop'],
                    p['entry'] + risk
                )

            if p['active']:
                p['highest'] = max(
                    p['highest'],
                    bar['high']
                )

                p['stop'] = max(
                    p['stop'],
                    p['entry'] + risk,
                    p['highest'] * 0.995
                )

            continue

        # Använd endast färdigstängda 1h-candles.
        while (
            hidx < len(hour)
            and hour[hidx]['time'] + 3600
            <= bar['time'] + 300
        ):
            hidx += 1

        hs = hour[max(0, hidx - 220):hidx]

        sig = Bot.signal_from_history(
            five[max(0, i - 319):i + 1],
            hs
        )

        if not sig:
            continue

        # Entry vid nästa candles öppning.
        entry = nextbar['open'] * (1 + slip)
        stop = sig['stop']

        if entry <= stop:
            continue

        risk_pct = (entry - stop) / entry

        if not 0.0015 <= risk_pct <= 0.025:
            continue

        if entry > sig['close'] * 1.004:
            continue

        position = {
            'entry': entry,
            'qty': stake / entry,
            'initial_stop': stop,
            'stop': stop,
            'highest': entry,
            'active': False,
            'time': nextbar['time']
        }

    # Öppna positioner räknas inte som avslutad PnL.
    return trades


def main():
    days = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else 14
    )

    stake = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else 100
    )

    symbols = (
        sys.argv[3].split(',')
        if len(sys.argv) > 3
        else ['ETH-USDT']
    )

    bot = Bot()
    all_trades = []

    for symbol in symbols:
        try:
            all_trades.extend(
                test_symbol(
                    bot,
                    symbol,
                    days,
                    stake
                )
            )
        except Exception as exc:
            print(
                f'ERROR {symbol}: {exc}',
                flush=True
            )
            raise

    all_trades.sort(
        key=lambda t: t['exit_time']
    )

    net = sum(
        t['net']
        for t in all_trades
    )

    wins = sum(
        t['net'] > 0
        for t in all_trades
    )

    equity = 0.0
    peak = 0.0
    drawdown = 0.0

    for t in all_trades:
        equity += t['net']
        peak = max(peak, equity)
        drawdown = max(
            drawdown,
            peak - equity
        )

    summary = {
        'days': days,
        'symbols': symbols,
        'trades': len(all_trades),
        'wins': wins,
        'winrate': (
            100 * wins / len(all_trades)
            if all_trades
            else 0
        ),
        'net_usdt': net,
        'max_drawdown_usdt': drawdown,
        'stake_usdt': stake
    }

    with TRADES.open(
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'symbol',
                'entry_time',
                'exit_time',
                'entry',
                'exit',
                'net',
                'reason'
            ]
        )

        writer.writeheader()
        writer.writerows(all_trades)

    OUT.write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8'
    )

    print(
        'RESULT',
        json.dumps(summary),
        flush=True
    )


if __name__ == '__main__':
    main()
