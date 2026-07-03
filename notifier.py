from config import CHAT_ID_FILE


def save_chat_id(chat_id):
    with open(CHAT_ID_FILE, "w") as f:
        f.write(str(chat_id))


def load_chat_id():
    try:
        with open(CHAT_ID_FILE, "r") as f:
            return f.read().strip()
    except:
        return None


class Notifier:

    def __init__(self, app):
        self.app = app

    async def send(self, text):
        chat_id = load_chat_id()

        if not chat_id:
            print("Ingen chat_id sparad ännu.")
            return

        try:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=text
            )
        except Exception as e:
            print("Telegram notify error:", e)

    async def entry(self, symbol, pos):
        await self.send(
            f"🟢 NEW MOCK LONG\n\n"
            f"{symbol}\n"
            f"Entry: {pos['entry']}\n"
            f"SL: {pos['stop']}\n"
            f"TP: {pos['tp']}\n"
            f"Trail: {pos['trail']}\n\n"
            f"Typ: {pos['entry_type']}"
        )

    async def exit(self, trade):
        await self.send(
            f"🔴 EXIT {trade['reason']}\n\n"
            f"{trade['symbol']}\n"
            f"Entry: {trade['entry']}\n"
            f"Exit: {trade['exit']}\n"
            f"PnL: {round(trade['pnl'], 4)} USDT\n"
            f"Typ: {trade['entry_type']}"
        )
