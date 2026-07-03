import csv
import os
from datetime import datetime, timezone

from config import MOCK_LOG, SIGNAL_LOG

j
def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def create_logs():

    if not os.path.exists(MOCK_LOG):

        with open(MOCK_LOG, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "time",
                "symbol",
                "side",
                "entry",
                "exit",
                "size_usdt",
                "fee_entry",
                "fee_exit",
                "pnl",
                "reason",
                "entry_type"
            ])

    if not os.path.exists(SIGNAL_LOG):

        with open(SIGNAL_LOG, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "time",
                "symbol",
                "signal",
                "reason"
            ])


def log_signal(symbol, signal, reason):

    with open(SIGNAL_LOG, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            now(),
            symbol,
            signal,
            reason
        ])


def log_trade(
    symbol,
    side,
    entry,
    exit_price,
    size_usdt,
    fee_rate,
    reason,
    entry_type,
):

    fee_entry = size_usdt * fee_rate

    qty = (size_usdt - fee_entry) / entry

    exit_value = qty * exit_price

    fee_exit = exit_value * fee_rate

    pnl = exit_value - fee_exit - size_usdt

    with open(MOCK_LOG, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            now(),
            symbol,
            side,
            round(entry, 6),
            round(exit_price, 6),
            size_usdt,
            round(fee_entry, 4),
            round(fee_exit, 4),
            round(pnl, 4),
            reason,
            entry_type
        ])

    return pnl


def get_statistics():

    create_logs()

    trades = 0
    wins = 0
    pnl = 0

    with open(MOCK_LOG, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:

            trades += 1

            trade_pnl = float(row["pnl"])

            pnl += trade_pnl

            if trade_pnl > 0:
                wins += 1

    winrate = 0

    if trades > 0:
        winrate = wins / trades * 100

    return {
        "trades": trades,
        "wins": wins,
        "winrate": round(winrate, 2),
        "pnl": round(pnl, 4),
    }
