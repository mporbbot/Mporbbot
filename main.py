
"""POC LONG mock bot – Telegram och V1/V2-backtest."""

import os
import csv
import json
import time
import asyncio
import logging
import math
import subprocess
import sys

from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import requests
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("poc_mock")

STATE = Path("poc_mock_state.json")
LOG = Path("poc_mock_trades.csv")
BASE = "https://api.kucoin.com"

PERIOD = {
    "5min": 300,
    "1hour": 3600,
}

HEADERS = [
    "time", "symbol", "entry", "exit", "stake",
    "fees", "net_pnl", "reason", "initial_stop"
]

BT_PROCESS = None
BT_LOG = Path("poc_backtest.log")
BT_RESULT = Path("poc_backtest_results.json")

KEYBOARD = ReplyKeyboardMarkup(
    [
        ["/status", "/pnl", "/open"],
        ["/engine_on", "/engine_off"],
        ["/stake", "/coins", "/maxpos"],
        ["/backtest", "/backtest_status", "/backtest_result"],
        ["/trades", "/reset"],
    ],
    resize_keyboard=True,
)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def ema(vals, period):
    if len(vals) < period:
        return None

    v = sum(vals[:period]) / period

    for x in vals[period:]:
        v += (x - v) * 2 / (period + 1)

    return v


@dataclass
class Position:
    symbol: str
    entry: float
    qty: float
    stake: float
    initial_stop: float
    stop: float
    highest: float
    active: bool
    opened: int
    poc: float


class Bot:

    def __init__(self):
        self.session = requests.Session()
        self.engine_on = False
        self.coins = ["ETH-USDT"]
        self.stake = 100.0
        self.maxpos = 1

        self.fee = 0.001
        self.slippage = 0.0002

        self.positions = {}
        self.last_signal = {}

        self.stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net": 0.0,
        }

        self.chat_id = None
        self.load()

    def load(self):
        if not STATE.exists():
            return

        try:
            d = json.loads(STATE.read_text())

            self.engine_on = bool(
                d.get("engine_on", False)
            )
            self.coins = d.get("coins", self.coins)
            self.stake = float(d.get("stake", self.stake))
            self.maxpos = int(d.get("maxpos", self.maxpos))

            self.positions = {
                s: Position(**p)
                for s, p in d.get("positions", {}).items()
            }

            self.last_signal = d.get("last_signal", {})
            self.stats = d.get("stats", self.stats)
            self.chat_id = d.get("chat_id")

        except Exception:
            log.exception("Could not load state")
            self.engine_on = False

    def save(self):
        data = {
            "engine_on": self.engine_on,
            "coins": self.coins,
            "stake": self.stake,
            "maxpos": self.maxpos,
            "positions": {
                s: asdict(p)
                for s, p in self.positions.items()
            },
            "last_signal": self.last_signal,
            "stats": self.stats,
            "chat_id": self.chat_id,
        }

        temp = STATE.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(STATE)

    def get(self, endpoint, params):
        r = self.session.get(
            BASE + endpoint,
            params=params,
            timeout=12,
        )

        r.raise_for_status()
        d = r.json()

        if (
            d.get("code") != "200000"
            or d.get("data") is None
        ):
            raise RuntimeError(str(d)[:200])

        return d["data"]

    def candles(self, symbol, tf, count):
        end = int(time.time())

        raw = self.get(
            "/api/v1/market/candles",
            {
                "symbol": symbol,
                "type": tf,
                "startAt": end - PERIOD[tf] * (count + 12),
                "endAt": end,
            },
        )

        rows = []

        for c in reversed(raw):
            t = int(c[0])

            if t + PERIOD[tf] > end:
                continue

            rows.append({
                "time": t,
                "open": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "volume": float(c[5]),
            })

        return rows[-count:]

    def ticker(self, symbol):
        d = self.get(
            "/api/v1/market/orderbook/level1",
            {"symbol": symbol},
        )

        last = float(d.get("price") or 0)
        bid = float(d.get("bestBid") or last)
        ask = float(d.get("bestAsk") or last)

        if min(last, bid, ask) <= 0:
            raise RuntimeError("Invalid ticker")

        return last, bid, ask

    @staticmethod
    def poc(candles, bins=80):
        low = min(c["low"] for c in candles)
        high = max(c["high"] for c in candles)

        if high <= low:
            return None

        step = (high - low) / bins
        volumes = [0.0] * bins

        for c in candles:
            typical = (
                c["high"] + c["low"] + c["close"]
            ) / 3

            idx = max(
                0,
                min(
                    bins - 1,
                    int((typical - low) / step),
                ),
            )

            volumes[idx] += c["volume"]

        if max(volumes) <= 0:
            return None

        return low + (
            volumes.index(max(volumes)) + 0.5
        ) * step

    def signal(self, symbol):
        five = self.candles(symbol, "5min", 320)
        hour = self.candles(symbol, "1hour", 220)

        if not five:
            return None

        if str(five[-1]["time"]) == str(
            self.last_signal.get(symbol)
        ):
            return None

        return self.signal_from_history(five, hour)

    @staticmethod
    def signal_from_history(five, hour):
        if len(five) < 302 or len(hour) < 200:
            return None

        last = five[-1]

        trend = ema(
            [x["close"] for x in hour],
            200,
        )

        if trend is None or last["close"] <= trend:
            return None

        for i in range(
            len(five) - 2,
            max(288, len(five) - 14) - 1,
            -1,
        ):
            level = Bot.poc(five[i - 288:i])

            if level is None:
                continue

            br = five[i]

            if not (
                five[i - 1]["close"] <= level
                and br["close"] > level * 1.0005
            ):
                continue

            subsequent = five[i + 1:]

            if (
                len(subsequent) < 2
                or len(subsequent) > 12
            ):
                continue

            retest = subsequent[:-1]

            if not any(
                c["low"] <= level * 1.002
                and c["high"] >= level * 0.998
                for c in retest
            ):
                continue

            if any(
                c["close"] < level * 0.998
                for c in subsequent
            ):
                continue

            if not (
                last["close"] > last["open"]
                and last["close"] > subsequent[-2]["high"]
                and level < last["close"] <= level * 1.004
            ):
                continue

            stop = min(
                c["low"] for c in retest
            ) * 0.9995

            risk = (
                last["close"] - stop
            ) / last["close"]

            if not 0.0015 <= risk <= 0.025:
                continue

            return {
                "close": last["close"],
                "stop": stop,
                "poc": level,
                "time": last["time"],
            }

        return None

    def close(self, symbol, reason, bid):
        p = self.positions[symbol]

        exit_price = bid * (1 - self.slippage)

        gross = p.qty * (
            exit_price - p.entry
        )

        fees = self.fee * (
            p.stake + p.qty * exit_price
        )

        net = gross - fees

        LOG.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fresh = (
            not LOG.exists()
            or LOG.stat().st_size == 0
        )

        with LOG.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            w = csv.writer(f)

            if fresh:
                w.writerow(HEADERS)

            w.writerow([
                stamp(),
                symbol,
                p.entry,
                exit_price,
                p.stake,
                fees,
                net,
                reason,
                p.initial_stop,
            ])

        self.stats["trades"] += 1

        self.stats[
            "wins" if net > 0 else "losses"
        ] += 1

        self.stats["net"] += net

        del self.positions[symbol]
        self.save()

        return (
            f"EXIT {symbol} | MOCK\n"
            f"Pris: {exit_price:.6f}\n"
            f"Netto: {net:+.4f} USDT\n"
            f"Orsak: {reason}"
        )

    def step(self):
        if not self.engine_on:
            return []

        messages = []

        for symbol in list(self.positions):
            try:
                p = self.positions[symbol]

                last, bid, _ = self.ticker(symbol)

                if last > p.highest:
                    p.highest = last

                risk = p.entry - p.initial_stop

                if risk <= 0:
                    log.error("Invalid risk: %s", symbol)
                    continue

                if (
                    not p.active
                    and last >= p.entry + 2 * risk
                ):
                    p.active = True

                    p.stop = max(
                        p.stop,
                        p.entry + risk,
                    )

                    messages.append(
                        f"2R NÅTT {symbol} | MOCK\n"
                        f"Trailing aktiv\n"
                        f"SL minst +1R: {p.stop:.6f}"
                    )

                if p.active:
                    p.stop = max(
                        p.stop,
                        p.entry + risk,
                        p.highest * 0.995,
                    )

                self.save()

                if bid <= p.stop:
                    messages.append(
                        self.close(
                            symbol,
                            "TRAIL" if p.active else "STOP_LOSS",
                            bid,
                        )
                    )

            except Exception:
                log.exception("Position error: %s", symbol)

        for symbol in self.coins:
            if len(self.positions) >= self.maxpos:
                break

            if symbol in self.positions:
                continue

            try:
                s = self.signal(symbol)

                if not s:
                    continue

                _, _, ask = self.ticker(symbol)

                entry = ask * (1 + self.slippage)

                if entry > s["close"] * 1.004:
                    continue

                stop = s["stop"]

                if (
                    stop <= 0
                    or stop >= entry
                    or not 0.0015 <=
                    (entry - stop) / entry <= 0.025
                ):
                    continue

                p = Position(
                    symbol,
                    entry,
                    self.stake / entry,
                    self.stake,
                    stop,
                    stop,
                    entry,
                    False,
                    int(time.time()),
                    s["poc"],
                )

                self.positions[symbol] = p
                self.last_signal[symbol] = s["time"]

                self.save()

                messages.append(
                    f"ENTRY {symbol} | MOCK\n"
                    f"Insats: {self.stake:.2f} USDT\n"
                    f"Entry: {entry:.6f}\n"
                    f"POC: {s['poc']:.6f}\n"
                    f"SL: {stop:.6f}\n"
                    f"2R: {entry + 2 * (entry - stop):.6f}\n"
                    f"Vid 2R: trailing 0,5%"
                )

            except Exception:
                log.exception("Signal error: %s", symbol)

        return messages

    def status(self):
        n = self.stats["trades"]
        wins = self.stats["wins"]

        winrate = (
            wins / n * 100
            if n else 0
        )

        return (
            "POC LONG | ENDAST MOCK\n"
            f"Engine: {'ON' if self.engine_on else 'OFF'}\n"
            f"Coins: {', '.join(self.coins)}\n"
            f"Stake: {self.stake:.2f} USDT\n"
            f"Max positioner: {self.maxpos}\n"
            "POC: 24h / 5min\n"
            "Trend: 1h EMA200\n"
            "Trailing: +2R\n"
            "Skyddad nivå: +1R\n"
            "Trail: 0,50%\n"
            f"Trades: {n}\n"
            f"Vinster: {wins}\n"
            f"Winrate: {winrate:.1f}%\n"
            f"Netto: {self.stats['net']:+.4f} USDT\n"
            f"Öppna: {len(self.positions)}"
        )


ENGINE = Bot()


async def start(update, context):
    ENGINE.chat_id = update.effective_chat.id
    ENGINE.save()

    await update.message.reply_text(
        "POC LONG – endast mocktrading.\n"
        + ENGINE.status(),
        reply_markup=KEYBOARD,
    )


async def status(update, context):
    await update.message.reply_text(
        ENGINE.status(),
        reply_markup=KEYBOARD,
    )


async def on(update, context):
    ENGINE.engine_on = True
    ENGINE.save()

    await update.message.reply_text(
        "MOCK ENGINE ON",
        reply_markup=KEYBOARD,
    )


async def off(update, context):
    ENGINE.engine_on = False
    ENGINE.save()

    await update.message.reply_text(
        "ENGINE OFF",
        reply_markup=KEYBOARD,
    )


async def stake(update, context):
    if context.args:
        try:
            x = float(context.args[0])

            if not math.isfinite(x) or not 1 <= x <= 100000:
                raise ValueError()

            ENGINE.stake = x
            ENGINE.save()

        except ValueError:
            return await update.message.reply_text(
                "Använd /stake 100"
            )

    await update.message.reply_text(
        f"Stake: {ENGINE.stake:.2f} USDT",
        reply_markup=KEYBOARD,
    )


async def coins(update, context):
    if context.args:
        if ENGINE.positions:
            return await update.message.reply_text(
                "Vänta tills positionerna är stängda."
            )

        symbols = [
            s.upper()
            for s in context.args
        ]

        if (
            len(symbols) > 25
            or any(
                not s.endswith("-USDT")
                or len(s) > 25
                for s in symbols
            )
        ):
            return await update.message.reply_text(
                "Använd /coins ETH-USDT BTC-USDT"
            )

        ENGINE.coins = list(dict.fromkeys(symbols))
        ENGINE.save()

    await update.message.reply_text(
        "Coins: " + ", ".join(ENGINE.coins),
        reply_markup=KEYBOARD,
    )


async def maxpos(update, context):
    if context.args:
        try:
            x = int(context.args[0])

            if not 1 <= x <= 20:
                raise ValueError()

            ENGINE.maxpos = x
            ENGINE.save()

        except ValueError:
            return await update.message.reply_text(
                "Använd /maxpos 1"
            )

    await update.message.reply_text(
        f"Max positioner: {ENGINE.maxpos}",
        reply_markup=KEYBOARD,
    )


async def opened(update, context):
    if not ENGINE.positions:
        return await update.message.reply_text(
            "Inga öppna mocktrades.",
            reply_markup=KEYBOARD,
        )

    lines = []

    for symbol, p in ENGINE.positions.items():
        target = p.entry + 2 * (
            p.entry - p.initial_stop
        )

        lines.append(
            f"{symbol}\n"
            f"Entry: {p.entry:.6f}\n"
            f"SL: {p.stop:.6f}\n"
            f"2R: {target:.6f}\n"
            f"Trailing: {'ON' if p.active else 'OFF'}"
        )

    await update.message.reply_text(
        "\n\n".join(lines),
        reply_markup=KEYBOARD,
    )


async def pnl(update, context):
    await update.message.reply_text(
        ENGINE.status(),
        reply_markup=KEYBOARD,
    )


async def trades(update, context):
    if not LOG.exists():
        return await update.message.reply_text(
            "Inga avslutade trades.",
            reply_markup=KEYBOARD,
        )

    with LOG.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-10:]

    text = "\n".join(
        f"{r['symbol']} "
        f"{float(r['net_pnl']):+.4f} USDT "
        f"({r['reason']})"
        for r in rows
    )

    await update.message.reply_text(
        text or "Inga trades",
        reply_markup=KEYBOARD,
    )


async def reset(update, context):
    if ENGINE.positions:
        return await update.message.reply_text(
            "Kan inte nollställa med öppna positioner."
        )

    ENGINE.stats = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "net": 0.0,
    }

    ENGINE.save()

    await update.message.reply_text(
        "Statistik nollställd. CSV finns kvar.",
        reply_markup=KEYBOARD,
    )


async def backtest(update, context):
    global BT_PROCESS

    if (
        BT_PROCESS is not None
        and BT_PROCESS.poll() is None
    ):
        return await update.message.reply_text(
            "Backtest kör redan. /backtest_status"
        )

    try:
        days = (
            int(context.args[0])
            if context.args else 14
        )

        if not 2 <= days <= 60:
            raise ValueError()

    except ValueError:
        return await update.message.reply_text(
            "Använd /backtest 14 (2–60 dagar)"
        )

    BT_RESULT.unlink(missing_ok=True)

    with BT_LOG.open(
        "w",
        encoding="utf-8",
    ) as out:

        BT_PROCESS = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "poc_backtest.py",
                str(days),
                str(ENGINE.stake),
                ",".join(ENGINE.coins),
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
        )

    await update.message.reply_text(
        f"POC-backtest startat\n"
        f"Dagar: {days}\n"
        f"Coins: {len(ENGINE.coins)}\n\n"
        "Kolla /backtest_status",
        reply_markup=KEYBOARD,
    )


async def backtest_status(update, context):
    running = (
        BT_PROCESS is not None
        and BT_PROCESS.poll() is None
    )

    tail = (
        BT_LOG.read_text(encoding="utf-8")[-3200:]
        if BT_LOG.exists()
        else "Ingen logg ännu."
    )

    await update.message.reply_text(
        ("KÖR\n" if running else "STOPPAD/KLAR\n")
        + tail,
        reply_markup=KEYBOARD,
    )


async def backtest_result(update, context):
    if (
        BT_PROCESS is not None
        and BT_PROCESS.poll() is None
    ):
        return await update.message.reply_text(
            "Backtest kör fortfarande. Se /backtest_status"
        )

    if not BT_RESULT.exists():
        return await update.message.reply_text(
            "Inget färdigt resultat. Se /backtest_status"
        )

    try:
        d = json.loads(
            BT_RESULT.read_text(encoding="utf-8")
        )

        parts = [
            f"POC BACKTEST {d['days']} DAGAR",
            f"Coins: {', '.join(d['symbols'])}",
            f"Insats: {d['stake_usdt']:.2f} USDT",
        ]

        for version in ("V1", "V2"):
            r = d[version]

            parts.append(
                f"\n{version}\n"
                f"Trades: {r['trades']}\n"
                f"Vinster: {r['wins']}\n"
                f"Winrate: {r['winrate']:.1f}%\n"
                f"Netto: {r['net_usdt']:+.4f} USDT\n"
                f"Max DD: {r['max_drawdown_usdt']:.4f} USDT"
            )

        diff = (
            d["V2"]["net_usdt"]
            - d["V1"]["net_usdt"]
        )

        parts.append(
            f"\nSkillnad V2 - V1: {diff:+.4f} USDT"
        )

        parts.append(
            "Candlebaserad simulering, inte verkliga avslut."
        )

        await update.message.reply_text(
            "\n".join(parts),
            reply_markup=KEYBOARD,
        )

    except (
        KeyError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as e:

        await update.message.reply_text(
            f"Felaktigt backtestresultat: {e}"
        )


async def loop(app):
    await asyncio.sleep(3)

    while True:
        try:
            messages = await asyncio.to_thread(
                ENGINE.step
            )

            if ENGINE.chat_id:
                for message in messages:
                    await app.bot.send_message(
                        chat_id=ENGINE.chat_id,
                        text=message,
                    )

        except Exception:
            log.exception("Engine loop error")

        await asyncio.sleep(10)


async def post_init(app):
    app.bot_data["engine_task"] = (
        asyncio.create_task(loop(app))
    )


async def post_shutdown(app):
    task = app.bot_data.get("engine_task")

    if task:
        task.cancel()


def main():
    token = (
        os.getenv("TELEGRAM_TOKEN")
        or os.getenv("BOT_TOKEN")
    )

    if not token:
        raise RuntimeError(
            "Missing TELEGRAM_TOKEN or BOT_TOKEN"
        )

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    handlers = [
        ("start", start),
        ("status", status),
        ("pnl", pnl),
        ("open", opened),
        ("trades", trades),
        ("engine_on", on),
        ("engine_off", off),
        ("stake", stake),
        ("coins", coins),
        ("maxpos", maxpos),
        ("reset", reset),
        ("backtest", backtest),
        ("backtest_status", backtest_status),
        ("backtest_result", backtest_result),
    ]

    for name, function in handlers:
        app.add_handler(
            CommandHandler(name, function)
        )

    app.run_polling()


if __name__ == "__main__":
    main()
