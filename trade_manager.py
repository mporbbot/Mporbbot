from datetime import datetime, timezone, timedelta

from config import (
    TRADE_SIZE_USDT,
    FEE_RATE,
    TRAIL_PERCENT,
    COOLDOWN_MINUTES,
    USE_TRAILING_STOP,
)

from logger import log_trade


class TradeManager:
    def __init__(self, exchange):
        self.exchange = exchange
        self.open_positions = {}
        self.cooldowns = {}
        self.last_trade_time = None

    def now(self):
        return datetime.now(timezone.utc)

    def now_str(self):
        return self.now().strftime("%Y-%m-%d %H:%M:%S")

    def can_open(self, symbol, max_open_trades):
        if symbol in self.open_positions:
            return False

        if len(self.open_positions) >= max_open_trades:
            return False

        if symbol in self.cooldowns and self.now() < self.cooldowns[symbol]:
            return False

        return True

    def open_trade(self, symbol, setup):
        position = {
            "symbol": symbol,
            "entry": setup["entry"],
            "stop": setup["stop"],
            "tp": setup["tp"],
            "trail": setup["entry"] * (1 - TRAIL_PERCENT),
            "orb_high": setup["orb_high"],
            "orb_low": setup["orb_low"],
            "entry_type": setup["entry_type"],
            "opened": self.now_str(),
        }

        self.open_positions[symbol] = position
        self.last_trade_time = self.now_str()

        return position

    def manage_trades(self):
        closed = []

        for symbol, pos in list(self.open_positions.items()):
            price = self.exchange.fetch_price(symbol)

            if USE_TRAILING_STOP:
                new_trail = price * (1 - TRAIL_PERCENT)

                if new_trail > pos["trail"]:
                    pos["trail"] = new_trail

            reason = None

            if price <= pos["stop"]:
                reason = "STOP_LOSS"
                self.cooldowns[symbol] = (
                    self.now() + timedelta(minutes=COOLDOWN_MINUTES)
                )

            elif USE_TRAILING_STOP and price <= pos["trail"]:
                reason = "TRAIL_STOP"

            elif price >= pos["tp"]:
                reason = "TAKE_PROFIT"

            if reason:
                pnl = log_trade(
                    symbol=symbol,
                    side="LONG",
                    entry=pos["entry"],
                    exit_price=price,
                    size_usdt=TRADE_SIZE_USDT,
                    fee_rate=FEE_RATE,
                    exit_reason=reason,
                    entry_type=pos["entry_type"],
                    orb_high=pos["orb_high"],
                    orb_low=pos["orb_low"],
                )

                closed.append({
                    "symbol": symbol,
                    "entry": pos["entry"],
                    "exit": price,
                    "pnl": pnl,
                    "reason": reason,
                    "entry_type": pos["entry_type"],
                })

                del self.open_positions[symbol]

        return closed

    def open_summary(self):
        if not self.open_positions:
            return "Inga öppna trades."

        msg = "📌 Öppna trades\n\n"

        for symbol, p in self.open_positions.items():
            msg += (
                f"{symbol}\n"
                f"Entry: {round(p['entry'], 4)}\n"
                f"SL: {round(p['stop'], 4)}\n"
                f"Trail: {round(p['trail'], 4)}\n"
                f"TP: {round(p['tp'], 4)}\n"
                f"Typ: {p['entry_type']}\n"
                f"Öppnad: {p['opened']}\n\n"
            )

        return msg
