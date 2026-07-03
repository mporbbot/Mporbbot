from datetime import datetime, timezone

from config import (
    ORB_HOUR_UTC,
    ORB_MINUTE_UTC,
    BREAKOUT_BUFFER,
    MIN_ORB_PERCENT,
    MAX_ORB_PERCENT,
    RISK_REWARD,
    USE_TREND_FILTER,
    USE_VOLUME_FILTER,
    VOLUME_MULTIPLIER,
)

from logger import log_signal


class ORBStrategy:
    def __init__(self, exchange):
        self.exchange = exchange
        self.state = {}

    def get_state(self, symbol):
        if symbol not in self.state:
            self.state[symbol] = {
                "breakout_seen": False,
                "pullback_seen": False,
                "pullback_high": None,
                "reason": "Ingen analys ännu",
                "orb_high": None,
                "orb_low": None,
                "last_price": None,
            }
        return self.state[symbol]

    def get_orb(self, symbol):
        candles = self.exchange.fetch_candles(symbol, "15m", 120)
        today = datetime.now(timezone.utc).date()

        for c in candles:
            t = datetime.fromtimestamp(c["time"] / 1000, tz=timezone.utc)

            if (
                t.date() == today
                and t.hour == ORB_HOUR_UTC
                and t.minute == ORB_MINUTE_UTC
            ):
                orb_size = (c["high"] - c["low"]) / c["low"]

                if orb_size < MIN_ORB_PERCENT:
                    return None, "ORB för liten"

                if orb_size > MAX_ORB_PERCENT:
                    return None, "ORB för stor"

                return {
                    "high": c["high"],
                    "low": c["low"],
                    "time": t,
                    "size": orb_size,
                }, "OK"

        return None, "Ingen ORB hittad"

    def trend_ok(self, symbol):
        if not USE_TREND_FILTER:
            return True

        candles = self.exchange.fetch_candles(symbol, "1h", 250)

        if len(candles) < 200:
            return False

        closes = [c["close"] for c in candles]
        ema200 = sum(closes[-200:]) / 200

        return closes[-1] > ema200

    def volume_ok(self, candles):
        if not USE_VOLUME_FILTER:
            return True

        if len(candles) < 21:
            return False

        latest = candles[-1]["volume"]
        avg = sum(c["volume"] for c in candles[-21:-1]) / 20

        return latest > avg * VOLUME_MULTIPLIER

    def check(self, symbol):
        state = self.get_state(symbol)

        orb, orb_reason = self.get_orb(symbol)

        if not orb:
            state["reason"] = orb_reason
            log_signal(symbol, "NO_TRADE", orb_reason)
            return None

        state["orb_high"] = orb["high"]
        state["orb_low"] = orb["low"]

        candles15 = self.exchange.fetch_candles(symbol, "15m", 50)
        candles5 = self.exchange.fetch_candles(symbol, "5m", 80)

        last15 = candles15[-1]
        last5 = candles5[-1]

        state["last_price"] = last5["close"]

        if not self.trend_ok(symbol):
            state["reason"] = "Trendfilter nekar"
            log_signal(
                symbol,
                "NO_TRADE",
                state["reason"],
                price=last5["close"],
                orb_high=orb["high"],
                orb_low=orb["low"],
            )
            return None

        if not self.volume_ok(candles5):
            state["reason"] = "Volymfilter nekar"
            log_signal(
                symbol,
                "NO_TRADE",
                state["reason"],
                price=last5["close"],
                orb_high=orb["high"],
                orb_low=orb["low"],
            )
            return None

        breakout_level = orb["high"] * (1 + BREAKOUT_BUFFER)

        # 1. 15m breakout
        if last15["close"] > breakout_level:
            state["breakout_seen"] = True

        if not state["breakout_seen"]:
            state["reason"] = "Väntar på 15m breakout"
            log_signal(
                symbol,
                "NO_TRADE",
                state["reason"],
                price=last5["close"],
                orb_high=orb["high"],
                orb_low=orb["low"],
            )
            return None

        # 2. Första 5m pullback efter breakout
        # Pullback = röd eller svag 5m-candle som fortfarande håller sig över ORB high
        red_candle = last5["close"] < last5["open"]
        weak_candle = abs(last5["close"] - last5["open"]) / last5["open"] < 0.0005
        holds_above_orb = last5["close"] > orb["high"]

        if not state["pullback_seen"]:
            if (red_candle or weak_candle) and holds_above_orb:
                state["pullback_seen"] = True
                state["pullback_high"] = last5["high"]
                state["reason"] = "Pullback hittad, väntar på continuation"
            else:
                state["reason"] = "15m breakout klar, väntar på första 5m pullback"

            log_signal(
                symbol,
                "NO_TRADE",
                state["reason"],
                price=last5["close"],
                orb_high=orb["high"],
                orb_low=orb["low"],
            )
            return None

        # 3. Entry när pris stänger över pullback-candlens high
        if last5["close"] <= state["pullback_high"]:
            state["reason"] = "Pullback klar, väntar på break över pullback-high"
            log_signal(
                symbol,
                "NO_TRADE",
                state["reason"],
                price=last5["close"],
                orb_high=orb["high"],
                orb_low=orb["low"],
            )
            return None

        entry = last5["close"]
        stop = orb["low"]
        risk = entry - stop

        if risk <= 0:
            state["reason"] = "Ogiltig risk"
            log_signal(
                symbol,
                "NO_TRADE",
                state["reason"],
                price=entry,
                orb_high=orb["high"],
                orb_low=orb["low"],
            )
            return None

        tp = entry + risk * RISK_REWARD

        state["reason"] = "TRADE GODKÄND"

        log_signal(
            symbol,
            "TRADE",
            "15m breakout + 5m pullback + continuation",
            "ORB_PULLBACK_CONTINUATION",
            entry,
            orb["high"],
            orb["low"],
        )

        return {
            "entry": entry,
            "stop": stop,
            "tp": tp,
            "orb_high": orb["high"],
            "orb_low": orb["low"],
            "entry_type": "ORB_PULLBACK_CONTINUATION",
        }

    def debug(self, symbol):
        state = self.get_state(symbol)

        return {
            "reason": state["reason"],
            "breakout_seen": state["breakout_seen"],
            "retest_seen": state["pullback_seen"],
            "orb_high": state["orb_high"],
            "orb_low": state["orb_low"],
            "last_price": state["last_price"],
            "pullback_high": state["pullback_high"],
        }
