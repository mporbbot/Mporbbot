# main_orb_v34.py
import os, asyncio, logging
from typing import Optional
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, ContextTypes, CommandHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("orb_v34")

# ==== ENV ====
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID", "").strip()
ENTRY_MODE = (os.getenv("ENTRY_MODE", "close") or "close").lower()  # default close
TIMEFRAME = os.getenv("TIMEFRAME", "5m")

if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

# ==== STATE ====
ENGINE_RUNNING = False
SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT").split(",")
ALLOWED_TFS = ["1m", "3m", "5m", "15m", "30m", "1h"]

# ==== TELEGRAM APP ====
tg_app: Application = Application.builder().token(BOT_TOKEN).build()

def main_keyboard() -> ReplyKeyboardMarkup:
    # Viktigt: knapplabels = EXAKT samma som kommandona
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/mock"), KeyboardButton("/live")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ==== COMMANDS ====
async def send_menu(update: Update, text: str):
    if update.message:
        await update.message.reply_text(text, reply_markup=main_keyboard())

@tg_app.command("start")
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "🔧 ORB v34 startad. Använd /engine_on för att börja.")

@tg_app.command("status")
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "📊 Status\n"
        f"• Engine: {'AKTIV' if ENGINE_RUNNING else 'STOPPAD'}\n"
        f"• Entry mode: {ENTRY_MODE}\n"
        f"• Timeframe: {TIMEFRAME}\n"
        f"• Symbols: {', '.join(SYMBOLS)}"
    )
    await send_menu(update, txt)

@tg_app.command("engine_on")
async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_RUNNING
    ENGINE_RUNNING = True
    await send_menu(update, "✅ Engine är nu AKTIV.")

@tg_app.command("engine_off")
async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_RUNNING
    ENGINE_RUNNING = False
    await send_menu(update, "🛑 Engine stoppad.")

@tg_app.command("mock")
async def cmd_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "🧪 Mock-läge valt (ingen riktig handel).")

@tg_app.command("live")
async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "💼 Live-läge markerat (KuCoin).")

@tg_app.command("entrymode")
async def cmd_entrymode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENTRY_MODE
    # Växla mellan close/tick
    ENTRY_MODE = "tick" if ENTRY_MODE == "close" else "close"
    await send_menu(update, f"🔁 Entry mode satt till: {ENTRY_MODE}")

@tg_app.command("timeframe")
async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Inline-knappar för val
    kb = InlineKeyboardMarkup.from_row(
        [InlineKeyboardButton(tf, callback_data=f"tf:{tf}") for tf in ALLOWED_TFS]
    )
    await update.message.reply_text(f"⏱ Välj timeframe (nu: {TIMEFRAME})", reply_markup=kb)

@tg_app.callback_query_handler()
async def on_cbq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TIMEFRAME
    data = update.callback_query.data if update.callback_query else ""
    if data.startswith("tf:"):
        tf = data.split(":", 1)[1]
        if tf in ALLOWED_TFS:
            TIMEFRAME = tf
            await update.callback_query.answer(f"Timeframe: {tf}", show_alert=False)
            await update.callback_query.edit_message_text(f"⏱ Timeframe satt till {tf}")
        else:
            await update.callback_query.answer("Ogiltigt timeframe", show_alert=True)

@tg_app.command("panic")
async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_RUNNING
    ENGINE_RUNNING = False
    await send_menu(update, "🛑 PANIC: Engine stoppad och alla positioner (mock) anses stängda.")

# Dummy-kommandon för knapparna som finns i menyn (så de “svarar”)
@tg_app.command("pnl")
async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "📈 PnL: 0.00 USDT (mock).")

@tg_app.command("reset_pnl")
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "♻️ PnL återställt (mock).")

@tg_app.command("orb_on")
async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "🟩 ORB: PÅ (mock).")

@tg_app.command("orb_off")
async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_menu(update, "🟥 ORB: AV (mock).")

# ==== FASTAPI + WEBHOOK ====
app = FastAPI()

class _TGUpdate(BaseModel):
    update_id: Optional[int] = None  # pydantic placeholder; vi skickar rå JSON vidare

@app.on_event("startup")
async def _startup():
    log.info("Initierar Telegram Application…")
    await tg_app.initialize()
    await tg_app.start()
    # Sätt webhook
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    await tg_app.bot.set_webhook(url)
    log.info(f"Webhook satt: {url}")

@app.on_event("shutdown")
async def _shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    return {"ok": True, "engine": ENGINE_RUNNING, "entry": ENTRY_MODE, "tf": TIMEFRAME}

@app.post(f"/webhook/{{token}}")
async def webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return Response(status_code=403)
    body = await request.json()
    update = Update.de_json(body, tg_app.bot)
    await tg_app.process_update(update)
    return Response(status_code=200)
