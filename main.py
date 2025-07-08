from fastapi import FastAPI, Request
import os
import httpx

from mp_ai import get_ai_mode
from mp_backtest import run_backtest
from mp_utils import export_csv

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
        text = data["message"].get("text", "")

        if text.lower() == "/status":
            reply = f"✅ Mp ORBbot är igång!\nAI-läge: {get_ai_mode()}\nInga aktiva trades ännu."
        elif text.lower().startswith("/set_ai"):
            _, mode = text.split(maxsplit=1)
            from mp_ai import set_ai_mode
            reply = set_ai_mode(mode.strip())
        elif text.lower().startswith("/backtest"):
            args = text.split()
            if len(args) == 3:
                symbol = args[1].lower()
                days = args[2].lower()
                reply = await run_backtest(symbol, days)
            else:
                reply = "❌ Fel format. Använd: /backtest SYMBOL TID (t.ex. /backtest btcusdt 3d)"
        elif text.lower() == "/export_csv":
            path = export_csv()
            reply = "📁 CSV-filen är skapad. (Export via Render kräver filserver)"
        elif text.lower() == "/help":
            reply = (
                "/status – visa status\n"
                "/set_ai [läge] – ändra AI-läge\n"
                "/backtest SYMBOL TID – kör långt backtest\n"
                "/export_csv – skicka senaste CSV\n"
                "/help – visar alla kommandon"
            )
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