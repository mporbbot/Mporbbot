import csv
import os
from datetime import datetime, timezone

from config import MOCK_LOG, SIGNAL_LOG


def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def ensure_logs():
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
                "pnl_usdt",
                "exit_reason",
                "entry_type",
                "orb_high",
                "orb_low"
            ])

    if not os.path.exists(SIGNAL_LOG):
        with open(SIGNAL_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "symbol",
                "signal",
                "reason",
                "entry_type",
                "price",
                "orb_high",
                "orb_low"
            ])


def log_signal(symbol, signal, reason, entry_type="", price="", orb_high="", orb_low=""):
    ensure_logs()

    with open(SIGNAL_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now_str(),
            symbol,
            signal,
            reason,
            entry_type,
            price,
            orb_high,
            orb_low
        ])


def log_trade(
    symbol,
    side,
    entry,
    exit_price,
    size_usdt,
    fee_rate,
    exit_reason,
    entry_type,
    orb_high,
    orb_low
):
    ensure_logs()

    fee_entry = size_usdt * fee_rate
    qty = (size_usdt - fee_entry) / entry
    exit_value = qty * exit_price
    fee_exit = exit_value * fee_rate
    pnl = exit_value - fee_exit - size_usdt

    with open(MOCK_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now_str(),
            symbol,
            side,
            round(entry, 8),
            round(exit_price, 8),
            size_usdt,
            round(fee_entry, 6),
            round(fee_exit, 6),
            round(pnl, 6),
            exit_reason,
            entry_type,
            round(orb_high, 8),
            round(orb_low, 8)
        ])

    return pnl


def get_stats():
    ensure_logs()

    trades = 0
    wins = 0
    total_pnl = 0.0

    with open(MOCK_LOG, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pnl = float(row["pnl_usdt"])
            total_pnl += pnl
            trades += 1

            if pnl > 0:
                wins += 1

    winrate = 0

    if trades > 0:
        winrate = wins / trades * 100

    return {
        "trades": trades,
        "wins": wins,
        "winrate": round(winrate, 2),
        "pnl": round(total_pnl, 4)
    }


def get_history(limit=20):
    ensure_logs()

    with open(MOCK_LOG, "r") as f:
        rows = list(csv.DictReader(f))

    return rows[-limit:]
