from fastapi import FastAPI, Request
import httpx

app = FastAPI()

# Telegram-token hårdkodad
TELEGRAM_BOT_TOKEN = "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"
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
            reply = "✅ Mp ORBbot är igång!\nAI-läge: neutral\nInga aktiva trades ännu."
        else:
            reply = "🤖 Jag förstår inte kommandot. Skriv /status för att se botstatus."

        await send_telegram_message(chat_id, reply)

    return {"ok": True}

async def send_telegram_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
