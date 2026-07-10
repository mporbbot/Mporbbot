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


# Retest får ligga högst 0,20 % från ORB High
RETEST_TOLERANCE = 0.002

# Entry får ligga högst 0,50 % över ORB High
MAX_ENTRY_DISTANCE = 0.005

# Retest måste ske inom 12 stängda 5m-candles efter breakout
MAX_RETEST_CANDLES = 12

# Stop strax under retestens lägsta punkt
STOP_BUFFER = 0.0005


class ORBStrategy:
    def __init__(self, exchange):
        self.exchange = exchange
        self.state = {}

    def get_state(self, symbol):
        if symbol not in self.state:
            self.state[symbol] = {
                "orb_key": None,
                "cycle_id": 0,
                "breakout_seen": False,
                "breakout_time": None,
                "retest_seen": False,
                "retest_time": None,
                "retest_high": None,
                "retest_low": None,
                "last_entry_time": None,
                "waiting_for_new_cycle": False,
                "reason": "Ingen analys ännu",
                "orb_high": None,
                "orb_low": None,
                "entry_distance": None,
            }

        return self.state[symbol]

    @staticmethod
    def candle_time(candle):
        return datetime.fromtimestamp(
            candle["time"] / 1000,
            tz=timezone.utc,
        )

    def get_orb(self, symbol):
        candles = self.exchange.fetch_candles(
            symbol,
            "15m",
            150,
        )

        # Sista candle kan fortfarande vara öppen
        closed_candles = candles[:-1]
        today = datetime.now(timezone.utc).date()

        for candle in closed_candles:
            candle_time = self.candle_time(candle)

            if (
                candle_time.date() == today
                and candle_time.hour == ORB_HOUR_UTC
                and candle_time.minute == ORB_MINUTE_UTC
            ):
                orb_size = (
                    candle["high"] - candle["low"]
                ) / candle["low"]

                if orb_size < MIN_ORB_PERCENT:
                    return None, "ORB för liten"

                if orb_size > MAX_ORB_PERCENT:
                    return None, "ORB för stor"

                return {
                    "high": candle["high"],
                    "low": candle["low"],
                    "time": candle_time,
                    "timestamp": candle["time"],
                    "end_timestamp": (
                        candle["time"] + 15 * 60 * 1000
                    ),
                    "key": candle_time.strftime("%Y-%m-%d %H:%M"),
                }, "OK"

        return None, "Ingen färdig ORB hittad"

    def reset_for_new_orb(self, state, orb):
        if state["orb_key"] == orb["key"]:
            return

        state.update({
            "orb_key": orb["key"],
            "cycle_id": 0,
            "breakout_seen": False,
            "breakout_time": None,
            "retest_seen": False,
            "retest_time": None,
            "retest_high": None,
            "retest_low": None,
            "last_entry_time": None,
            "waiting_for_new_cycle": False,
            "reason": "Ny ORB skapad",
            "orb_high": orb["high"],
            "orb_low": orb["low"],
            "entry_distance": None,
        })

    def trend_ok(self, symbol):
        if not USE_TREND_FILTER:
            return True

        candles = self.exchange.fetch_candles(
            symbol,
            "1h",
            220,
        )

        closed_candles = candles[:-1]

        if len(closed_candles) < 200:
            return False

        closes = [
            candle["close"]
            for candle in closed_candles[-200:]
        ]

        average_200 = sum(closes) / len(closes)

        return closes[-1] > average_200

    def volume_ok(self, candles5, signal_index):
        if not USE_VOLUME_FILTER:
            return True

        if signal_index < 20:
            return False

        signal_volume = candles5[signal_index]["volume"]

        previous_volumes = [
            candle["volume"]
            for candle in candles5[
                signal_index - 20:signal_index
            ]
        ]

        average_volume = (
            sum(previous_volumes) / len(previous_volumes)
        )

        return (
            signal_volume
            >= average_volume * VOLUME_MULTIPLIER
        )

    def find_first_breakout(
        self,
        candles15,
        orb,
        earliest_timestamp,
    ):
        breakout_level = (
            orb["high"] * (1 + BREAKOUT_BUFFER)
        )

        for candle in candles15:
            if candle["time"] < orb["end_timestamp"]:
                continue

            if (
                earliest_timestamp is not None
                and candle["time"] <= earliest_timestamp
            ):
                continue

            if candle["close"] > breakout_level:
                return candle

        return None

    def find_new_retest_cycle(
        self,
        candles5,
        orb,
        breakout,
        last_entry_time,
    ):
        breakout_end = (
            breakout["time"] + 15 * 60 * 1000
        )

        earliest_time = breakout_end

        if last_entry_time is not None:
            earliest_time = max(
                earliest_time,
                last_entry_time + 5 * 60 * 1000,
            )

        candles_after_breakout = [
            candle
            for candle in candles5
            if candle["time"] >= earliest_time
        ]

        search_candles = candles_after_breakout[
            :MAX_RETEST_CANDLES
        ]

        retest = None
        retest_index = None

        for index, candle in enumerate(search_candles):
            distance = (
                abs(candle["low"] - orb["high"])
                / orb["high"]
            )

            inside_zone = (
                distance <= RETEST_TOLERANCE
            )

            closes_above_orb = (
                candle["close"] > orb["high"]
            )

            # Retest får sticka ner något under nivån,
            # men måste återta och stänga ovanför.
            if inside_zone and closes_above_orb:
                retest = candle
                retest_index = index
                break

        if retest is None:
            return (
                None,
                None,
                "Väntar på nytt 5m-återtest av ORB High",
            )

        continuation_candles = search_candles[
            retest_index + 1:
        ]

        for candle in continuation_candles:
            bullish = candle["close"] > candle["open"]

            breaks_retest_high = (
                candle["close"] > retest["high"]
            )

            entry_distance = (
                candle["close"] - orb["high"]
            ) / orb["high"]

            if entry_distance > MAX_ENTRY_DISTANCE:
                return (
                    retest,
                    None,
                    "Entry nekad: priset för långt över ORB",
                )

            if bullish and breaks_retest_high:
                return retest, candle, "ENTRY"

        return (
            retest,
            None,
            "Återtest klart, väntar på grön continuation",
        )

    def check(self, symbol):
        state = self.get_state(symbol)

        orb, orb_reason = self.get_orb(symbol)

        if orb is None:
            state["reason"] = orb_reason
            log_signal(
                symbol,
                "NO_TRADE",
                orb_reason,
            )
            return None

        self.reset_for_new_orb(state, orb)

        state["orb_high"] = orb["high"]
        state["orb_low"] = orb["low"]

        candles15 = self.exchange.fetch_candles(
            symbol,
            "15m",
            100,
        )[:-1]

        candles5 = self.exchange.fetch_candles(
            symbol,
            "5m",
            200,
        )[:-1]

        earliest_breakout = None

        if state["last_entry_time"] is not None:
            earliest_breakout = state["last_entry_time"]

        breakout = self.find_first_breakout(
            candles15,
            orb,
            earliest_breakout,
        )

        # Efter en genomförd trade kan samma breakout fortfarande
        # användas om priset gör ett helt nytt återtest av ORB.
        if breakout is None and state["breakout_time"] is not None:
            for candle in candles15:
                if candle["time"] == state["breakout_time"]:
                    breakout = candle
                    break

        if breakout is None:
            state["breakout_seen"] = False
            state["reason"] = (
                "Väntar på stängd 15m-breakout"
            )

            log_signal(
                symbol,
                "NO_TRADE",
                state["reason"],
                price=candles5[-1]["close"],
                orb_high=orb["high"],
                orb_low=orb["low"],
            )
            return None

        state["breakout_seen"] = True
        state["breakout_time"] = breakout["time"]

        retest, entry_candle, entry_reason = (
            self.find_new_retest_cycle(
                candles5,
                orb,
                breakout,
                state["last_entry_time"],
            )
        )

        if retest is not None:
            state["retest_seen"] = True
            state["retest_time"] = retest["time"]
            state["retest_high"] = retest["high"]
            state["retest_low"] = retest["low"]
        else:
            state["retest_seen"] = False
            state["retest_time"] = None
            state["retest_high"] = None
            state["retest_low"] = None

        if entry_candle is None:
            state["reason"] = entry_reason

            log_signal(
                symbol,
                "NO_TRADE",
                entry_reason,
                price=candles5[-1]["close"],
                orb_high=orb["high"],
                orb_low=orb["low"],
            )
            return None

        if not self.trend_ok(symbol):
            state["reason"] = "Trendfilter nekar entry"
            return None

        entry_index = next(
            (
                index
                for index, candle in enumerate(candles5)
                if candle["time"] == entry_candle["time"]
            ),
            -1,
        )

        if entry_index < 0:
            state["reason"] = (
                "Kunde inte hitta entry-candlen"
            )
            return None

        if not self.volume_ok(
            candles5,
            entry_index,
        ):
            state["reason"] = "Volymfilter nekar entry"
            return None

        entry = entry_candle["close"]

        entry_distance = (
            entry - orb["high"]
        ) / orb["high"]

        state["entry_distance"] = entry_distance

        stop = (
            retest["low"] * (1 - STOP_BUFFER)
        )

        risk = entry - stop

        if risk <= 0:
            state["reason"] = "Ogiltigt stop-avstånd"
            return None

        take_profit = (
            entry + risk * RISK_REWARD
        )

        state["cycle_id"] += 1
        state["last_entry_time"] = entry_candle["time"]
        state["waiting_for_new_cycle"] = True
        state["reason"] = "TRADE GODKÄND"

        log_signal(
            symbol,
            "TRADE",
            (
                "15m-breakout + nytt ORB-återtest "
                "+ grön continuation"
            ),
            "ORB_LEVEL_RETEST",
            entry,
            orb["high"],
            orb["low"],
        )

        return {
            "entry": entry,
            "stop": stop,
            "tp": take_profit,
            "orb_high": orb["high"],
            "orb_low": orb["low"],
            "entry_type": "ORB_LEVEL_RETEST",
        }

    def debug(self, symbol):
        state = self.get_state(symbol)

        entry_distance = state.get(
            "entry_distance"
        )

        if entry_distance is not None:
            entry_distance = (
                f"{entry_distance * 100:.3f}%"
            )

        breakout_time = state.get(
            "breakout_time"
        )

        if isinstance(breakout_time, int):
            breakout_time = datetime.fromtimestamp(
                breakout_time / 1000,
                tz=timezone.utc,
            ).strftime("%Y-%m-%d %H:%M")

        retest_time = state.get("retest_time")

        if isinstance(retest_time, int):
            retest_time = datetime.fromtimestamp(
                retest_time / 1000,
                tz=timezone.utc,
            ).strftime("%Y-%m-%d %H:%M")

        return {
            "reason": state["reason"],
            "breakout_seen": state["breakout_seen"],
            "retest_seen": state["retest_seen"],
            "orb_high": state["orb_high"],
            "orb_low": state["orb_low"],
            "breakout_time": breakout_time,
            "retest_time": retest_time,
            "retest_high": state["retest_high"],
            "retest_low": state["retest_low"],
            "entry_distance": entry_distance,
            "cycle_id": state["cycle_id"],
        }
