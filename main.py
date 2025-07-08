
from fastapi import FastAPI, Request
from bot.core import handle_message
import os

app = FastAPI()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_PATH = f"/{TOKEN}"

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_message(data)

@app.get("/")
async def root():
    return {"message": "Mp ORBbot is live!"}
