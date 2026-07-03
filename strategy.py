from datetime import datetime, timezone

from config import (
    ORB_HOUR_UTC,
    ORB_MINUTE_UTC,
    BREAKOUT_BUFFER,
    USE_TREND_FILTER,
    USE_VOLUME_FILTER,
    RISK_REWARD,
)


class ORBStrategy:

    def __init__(self, exchange):

        self.exchange = exchange

        self.state = {}

    def get_state(self, symbol):

        if symbol not in self.state:

            self.state[symbol] = {
                "breakout_seen": False,
                "retest_seen": False,
                "orb": None,
                "reason": ""
            }

        return self.state[symbol]

    def get_orb(self, symbol):

        candles = self.exchange.fetch_candles(symbol, "15m", 100)

        today = datetime.now(timezone.utc).date()

        for c in candles:

            t = datetime.fromtimestamp(
                c["time"] / 1000,
                tz=timezone.utc
            )

            if (
                t.date() == today
                and t.hour == ORB_HOUR_UTC
                and t.minute == ORB_MINUTE_UTC
            ):

                return {
                    "high": c["high"],
                    "low": c["low"]
                }

        return None

    def trend_ok(self, symbol):

        if not USE_TREND_FILTER:
            return True

        candles = self.exchange.fetch_candles(
            symbol,
            "1h",
            250
        )

        closes = [c["close"] for c in candles]

        ema = sum(closes[-200:]) / 200

        return closes[-1] > ema

    def volume_ok(self, candles):

        if not USE_VOLUME_FILTER:
            return True

        latest = candles[-1]["volume"]

        avg = sum(
            c["volume"]
            for c in candles[-21:-1]
        ) / 20

        return latest > avg

    def check(self, symbol):

        state = self.get_state(symbol)

        orb = self.get_orb(symbol)

        if orb is None:

            state["reason"] = "Ingen ORB ännu"

            return None

        candles15 = self.exchange.fetch_candles(
            symbol,
            "15m",
            10
        )

        candles5 = self.exchange.fetch_candles(
            symbol,
            "5m",
            20
        )

        last15 = candles15[-1]

        last5 = candles5[-1]

        breakout_level = orb["high"] * (
            1 + BREAKOUT_BUFFER
        )

        # ---------------------------------
        # 1. 15m breakout
        # ---------------------------------

        if last15["close"] > breakout_level:

            state["breakout_seen"] = True

        if not state["breakout_seen"]:

            state["reason"] = "Väntar på 15m breakout"

            return None

        # ---------------------------------
        # 2. 5m retest
        # ---------------------------------

        if last5["low"] <= orb["high"]:

            state["retest_seen"] = True

        if not state["retest_seen"]:

            state["reason"] = "Väntar på 5m retest"

            return None

        # ---------------------------------
        # 3. reclaim
        # ---------------------------------

        if last5["close"] <= orb["high"]:

            state["reason"] = "Ingen reclaim"

            return None

        if last5["close"] <= last5["open"]:

            state["reason"] = "Inte bullish"

            return None

        if not self.trend_ok(symbol):

            state["reason"] = "Trendfilter"

            return None

        if not self.volume_ok(candles5):

            state["reason"] = "Volymfilter"

            return None

        entry = last5["close"]

        stop = orb["low"]

        risk = entry - stop

        if risk <= 0:

            state["reason"] = "Ogiltig risk"

            return None

        tp = entry + risk * RISK_REWARD

        state["reason"] = "ENTRY"

        return {

            "entry": entry,

            "stop": stop,

            "tp": tp,

            "orb_high": orb["high"],

            "orb_low": orb["low"],

            "entry_type": "ORB_RETEST"

        }

    def debug(self, symbol):

        state = self.get_state(symbol)

        return {

            "reason": state["reason"],

            "breakout": state["breakout_seen"],

            "retest": state["retest_seen"],

            "orb": state["orb"]

        }
