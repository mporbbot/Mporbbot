from fastapi import FastAPI, Request
import os
import httpx

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

AI_MODE = "neutral"
LAST_CSV = "backtest_results.csv"

@app.get("/")
async def root():
    return {"message": "Mp ORBbot is live!"}

@app.post(WEBHOOK_PATH)
async def telegram_webhook(req: Request):
    global AI_MODE
    data = await req.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "").lower().strip()

        if text == "/status":
            reply = f"✅ Mp ORBbot är igång!\nAI-läge: {AI_MODE}\nInga aktiva trades ännu."
        elif text.startswith("/set_ai"):
            mode = text.replace("/set_ai", "").strip()
            if mode in ["neutral", "aggressiv", "försiktig"]:
                AI_MODE = mode
                reply = f"✅ AI-läge uppdaterat till: {mode}"
            else:
                reply = "⚠️ Ogiltigt läge. Använd: neutral, aggressiv eller försiktig."
        elif text == "/backtest":
            reply = "🔄 Kör backtest på 5 coins...
✅ Klart! Exporterar CSV..."
        elif text == "/export_csv":
            reply = "📎 Här är senaste CSV-filen."
        elif text == "/help":
            reply = (
                "📘 Kommandon:
"
                "/status – visa status
"
                "/set_ai [läge] – ändra AI-läge
"
                "/backtest – kör dummy backtest
"
                "/export_csv – skicka senaste CSV
"
                "/help – visa denna hjälp"
            )
        else:
            reply = "🤖 Okänt kommando. Skriv /help för tillgängliga kommandon."

        await send_telegram_message(chat_id, reply)
    return {"ok": True}

async def send_telegram_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
