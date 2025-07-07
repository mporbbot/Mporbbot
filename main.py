
from fastapi import FastAPI, Request
import httpx, csv, io, datetime

app = FastAPI()

TOKEN = "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"  # hårdkodad token
WEBHOOK_PATH = f"/{TOKEN}"
API = f"https://api.telegram.org/bot{TOKEN}"

AI_MODE = "neutral"
last_backtest = {}

@app.get("/")
async def root():
    return {"message": "Mp ORBbot is live!"}

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()

    if "message" not in data:
        return {"ok": True}

    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    if text.lower() == "/help":
        msg = "/status – visa status\n/set_ai [läge] – ändra AI-läge\n/backtest – kör dummy backtest\n/export_csv – skicka senaste CSV"
        await send(chat_id, msg)
    elif text.lower().startswith("/set_ai"):
        global AI_MODE
        mode = text.split(maxsplit=1)[1] if len(text.split()) > 1 else ""
        if mode in {"neutral", "försiktig", "aggressiv"}:
            AI_MODE = mode
            await send(chat_id, f"AI-läge ändrat till {AI_MODE}")
        else:
            await send(chat_id, "Använd neutral, försiktig eller aggressiv.")
    elif text.lower() == "/status":
        await send(chat_id, f"✅ Mp ORBbot är igång!\nAI-läge: {AI_MODE}\nInga aktiva trades ännu.")
    elif text.lower() == "/backtest":
        results, csv_bytes = dummy_backtest()
        global last_backtest
        last_backtest = {"csv": csv_bytes}
        out = "\n".join([f"{k}: {v}" for k, v in results.items()])
        await send(chat_id, f"📊 Backtest klart:\n{out}")
    elif text.lower() == "/export_csv":
        if last_backtest.get("csv"):
            await send(chat_id, "Här är CSV-filen:", doc=last_backtest["csv"], filename="backtest.csv")
        else:
            await send(chat_id, "Ingen backtest gjord ännu. Kör /backtest först.")
    else:
        await send(chat_id, "Okänt kommando. Skriv /help")

    return {"ok": True}

def dummy_backtest():
    now = datetime.datetime.utcnow()
    results = {"from": (now - datetime.timedelta(days=7)).date(),
               "to": now.date(),
               "trades": 10, "win_rate": "60%", "pnl": "+3.1%"}
    buf = io.StringIO()
    w = csv.writer(buf); w.writerow(["date", "pair", "side", "pnl_pct"])
    for i in range(10):
        w.writerow([(now - datetime.timedelta(days=i)).date(), "BTCUSDT",
                    "LONG" if i % 2 == 0 else "SHORT", (0.5 if i % 2 == 0 else -0.3)])
    return results, buf.getvalue().encode()

async def send(chat_id, text, doc=None, filename=None):
    async with httpx.AsyncClient() as c:
        if doc:
            await c.post(f"{API}/sendDocument",
                         data={"chat_id": str(chat_id)},
                         files={"document": (filename, doc)})
        else:
            await c.post(f"{API}/sendMessage",
                         json={"chat_id": chat_id, "text": text})
