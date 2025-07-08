
from bot.ai import get_ai_mode
from bot.trade import mock_trade
from bot.utils import send_message

async def handle_message(data):
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if text.startswith("/status"):
        await send_message(chat_id, "🤖 Status: AI-läge: neutral. Inga aktiva trades.")
    elif text.startswith("/start"):
        await send_message(chat_id, "🟢 Boten är igång med mocktrades.")
    elif text.startswith("/stop"):
        await send_message(chat_id, "🔴 Boten stoppad.")
    elif text.startswith("/set_ai"):
        await send_message(chat_id, "⚙️ AI-läge uppdaterat.")
    elif text.startswith("/backtest"):
        await send_message(chat_id, "🔄 Kör backtest på 5 coins...")
    elif text.startswith("/export_csv"):
        await send_message(chat_id, "📤 CSV-export skickas...")
    elif text.startswith("/mock_trade"):
        await send_message(chat_id, "📊 Mocktrade genomförd.")
    else:
        await send_message(chat_id, "❓ Okänt kommando. Skriv /status.")

    return {"ok": True}
