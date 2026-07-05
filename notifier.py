from storage import load_chat_id


def fmt(value):
    if value is None:
        return "Ej aktiv"
    return round(value, 4)


class Notifier:
    def __init__(self, app):
        self.app = app

    async def send(self, text):
        chat_id = load_chat_id()

        if not chat_id:
            print("Ingen chat sparad ännu.")
            return

        try:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=text
            )
        except Exception as e:
            print(f"Telegram error: {e}")

    async def startup(self):
        await self.send(
            "🤖 MP ORB Bot startad\n\n"
            "Status: ONLINE\n"
            "Mode: MOCK"
        )

    async def entry(self, symbol, position):
        await self.send(
            f"🟢 ENTRY\n\n"
            f"Coin: {symbol}\n"
            f"Typ: {position['entry_type']}\n\n"
            f"Entry: {fmt(position['entry'])}\n"
            f"SL: {fmt(position['stop'])}\n"
            f"TP: {fmt(position['tp'])}\n"
            f"Trail: {fmt(position.get('trail'))}"
        )

    async def exit(self, trade):
        emoji = "✅" if trade["pnl"] > 0 else "❌"

        await self.send(
            f"{emoji} EXIT\n\n"
            f"{trade['symbol']}\n\n"
            f"Reason: {trade['reason']}\n"
            f"Entry: {fmt(trade['entry'])}\n"
            f"Exit: {fmt(trade['exit'])}\n"
            f"PnL: {round(trade['pnl'], 4)} USDT"
        )

    async def signal(self, symbol, reason):
        await self.send(
            f"📡 Signal\n\n"
            f"{symbol}\n\n"
            f"{reason}"
        )

    async def error(self, message):
        await self.send(
            f"⚠️ ERROR\n\n"
            f"{message}"
        )
