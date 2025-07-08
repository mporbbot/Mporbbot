
from fastapi import FastAPI, Request
import os
import httpx

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

@app.get("/")
async def root():
    return {"message": "Mp ORBbot is live!"}

@app.post(WEBHOOK_PATH)
async def telegram_webhook(req: Request):
    data = await req.json()

    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "").lower()

        if text == "/status":
            reply = "✅ Mp ORBbot är igång!\nAI-läge: neutral\nInga aktiva trades ännu."
        elif text.startswith("/set_ai"):
            reply = "🤖 AI-läge uppdaterat!"
        elif text == "/backtest":
            reply = "🔄 Kör backtest på 5 coins...\nDetta kan ta en stund."
        elif text == "/export_csv":
            reply = "📁 Senaste backtest-resultat exporterad som CSV."
        elif text == "/help":
            reply = "📘 Kommandon:\n/status – visa status\n/set_ai [läge] – ändra AI-läge\n/backtest – kör backtest på alla 5 coins\n/export_csv – skicka senaste CSV\n/help – visa denna hjälptext"
        else:
            reply = "🤖 Jag förstår inte kommandot. Skriv /help för hjälp."

        await send_telegram_message(chat_id, reply)

    return {"ok": True}

async def send_telegram_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
