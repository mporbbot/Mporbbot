# main_orb_v34.py
import os
import json
import logging
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ----------------------------
# Miljö / inställningar
# ----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("orb_v34")

def _getenv_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()

def _getenv_int(name: str, default: int = 0) -> int:
    try:
        raw = os.getenv(name, str(default))
        # tillåt “0   # kommentar”
        return int(str(raw).split()[0])
    except Exception:
        return default

# Token (stöd för båda namnen)
BOT_TOKEN = _getenv_str("TELEGRAM_BOT_TOKEN") or _getenv_str("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN")

WEBHOOK_BASE = _getenv_str("WEBHOOK_BASE") or _getenv_str("RENDER_EXTERNAL_URL")
if not WEBHOOK_BASE:
    # Render sätter oftast inte RENDER_EXTERNAL_URL för python-webbservice,
    # men vi försöker gissa från service-namnet om någon glömt sätta WEBHOOK_BASE.
    # Hellre faila tydligt:
    raise RuntimeError("Saknar WEBHOOK_BASE (t.ex. https://mporbbot.onrender.com)")

OWNER_CHAT_ID = _getenv_int("OWNER_CHAT_ID", 0)

# Trading-relaterade (stubbar här – din motor kan läsa samma envs)
SYMBOLS = _getenv_str("SYMBOLS", "BTC-USDT,ETH-USDT").replace(" ", "")
TIMEFRAME = _getenv_str("TIMEFRAME", "1m")
LIVE = _getenv_int("LIVE", 0)  # 0=mock, 1=live

# ----------------------------
# Enkel state för demon
# ----------------------------
class State(BaseModel):
    engine_on: bool = False
    live_mode: bool = LIVE == 1
    entry_mode: str = "close"   # starta som 'close' (inte tick)
    timeframe: str = TIMEFRAME
    symbols: List[str] = []
    pnl_usdt: float = 0.0
    orb_enabled: bool = True

state = State(symbols=[s for s in SYMBOLS.split(",") if s])

# ----------------------------
# Telegram Application
# ----------------------------
tg_app: Optional[Application] = None

# Hjälptexter
HELP_TEXT = (
    "📋 **Kommandon**\n"
    "/status – visa status (sparar chat_id)\n"
    "/engine_on – aktivera motorn\n"
    "/engine_off – stäng av motorn\n"
    "/start_mock – mock-läge\n"
    "/start_live – live-läge (KuCoin)\n"
    "/entry_mode – välj entry: tick/close\n"
    "/timeframe – välj timeframe (1m/3m/5m/15m/1h/1d)\n"
    "/pnl – visa PnL\n"
    "/reset_pnl – nollställ PnL\n"
    "/trailing – visa trailing-inställningar (stub)\n"
    "/orb_on – slå på ORB-filter\n"
    "/orb_off – slå av ORB-filter\n"
    "/panic – stäng/stoppa allt (stub)\n"
)

# ------------- Handlers -------------
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(HELP_TEXT, parse_mode="Markdown")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update.effective_chat else None
    head = "✅ Status"
    rows = [
        f"engine_on: {state.engine_on}",
        f"live_mode: {state.live_mode}",
        f"entry_mode: {state.entry_mode}",
        f"timeframe: {state.timeframe}",
        f"symbols: {', '.join(state.symbols) if state.symbols else '-'}",
        f"pnl: {state.pnl_usdt:.2f} USDT",
        f"orb_enabled: {state.orb_enabled}",
        f"owner_chat_id: {OWNER_CHAT_ID or '-'}",
        f"your_chat_id: {chat_id or '-'}",
    ]
    await update.effective_chat.send_message(f"{head}\n" + "\n".join(rows))

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.engine_on = True
    await update.effective_chat.send_message("✅ Engine är nu **AKTIV**.")
async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.engine_on = False
    await update.effective_chat.send_message("🛑 Engine **stoppad**.")

async def cmd_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.live_mode = False
    await update.effective_chat.send_message("🧪 Mock-läge **AKTIVT**.")
async def cmd_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.live_mode = True
    await update.effective_chat.send_message("💹 Live-läge **AKTIVT** (KuCoin).")

def _entry_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Entry: close", callback_data="entry:close"),
            InlineKeyboardButton("Entry: tick",  callback_data="entry:tick"),
        ]
    ])

async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(
        f"Aktuell entry: **{state.entry_mode}**\nVälj nytt läge:",
        reply_markup=_entry_keyboard(),
        parse_mode="Markdown",
    )

def _tf_keyboard() -> InlineKeyboardMarkup:
    row1 = [
        InlineKeyboardButton("1m", callback_data="tf:1m"),
        InlineKeyboardButton("3m", callback_data="tf:3m"),
        InlineKeyboardButton("5m", callback_data="tf:5m"),
    ]
    row2 = [
        InlineKeyboardButton("15m", callback_data="tf:15m"),
        InlineKeyboardButton("1h", callback_data="tf:1h"),
        InlineKeyboardButton("1d", callback_data="tf:1d"),
    ]
    return InlineKeyboardMarkup([row1, row2])

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(
        f"Aktuellt timeframe: **{state.timeframe}**\nVälj nytt timeframe:",
        reply_markup=_tf_keyboard(),
        parse_mode="Markdown",
    )

async def cb_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.callback_query:
        return
    q = update.callback_query
    data = q.data or ""
    try:
        if data.startswith("entry:"):
            _, mode = data.split(":", 1)
            if mode in ("tick", "close"):
                state.entry_mode = mode
                await q.edit_message_text(f"✅ Entry-läge satt till **{mode}**.", parse_mode="Markdown")
            else:
                await q.answer("Ogiltigt entry-läge")
        elif data.startswith("tf:"):
            _, tf = data.split(":", 1)
            if tf in ("1m","3m","5m","15m","1h","1d"):
                state.timeframe = tf
                await q.edit_message_text(f"⏱️ Timeframe satt till **{tf}**.", parse_mode="Markdown")
            else:
                await q.answer("Ogiltigt timeframe")
        else:
            await q.answer("Okänt val")
    except Exception as e:
        log.exception("Callback error")
        await q.answer("Fel")

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.engine_on = False
    # Här skulle du stänga alla öppna positioner i din motor.
    await update.effective_chat.send_message("🛑 PANIC: motor stoppad och (stub) stäng alla positioner.")

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(f"📈 PnL: {state.pnl_usdt:.2f} USDT")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.pnl_usdt = 0.0
    await update.effective_chat.send_message("✅ PnL återställd till 0.00 USDT.")

async def cmd_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Stub – visa bara placeholder. Koppla senare till riktiga envs/logic.
    await update.effective_chat.send_message("🔧 Trailing: (stub) inga parametrar exponerade ännu.")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.orb_enabled = True
    await update.effective_chat.send_message("🟢 ORB-filter: PÅ")
async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.orb_enabled = False
    await update.effective_chat.send_message("⚪️ ORB-filter: AV")

# ----------------------------
# FastAPI + webhook
# ----------------------------
app = FastAPI(title="Mporbbot/ORB v34")

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.on_event("startup")
async def on_startup():
    global tg_app
    log.info("Initierar Telegram Application…")
    tg_app = Application.builder().token(BOT_TOKEN).build()

    # Handlers
    tg_app.add_handler(CommandHandler(["help", "start"], cmd_help))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    tg_app.add_handler(CommandHandler("start_mock", cmd_start_mock))
    tg_app.add_handler(CommandHandler("start_live", cmd_start_live))
    tg_app.add_handler(CommandHandler(["entry_mode", "entrymode"], cmd_entry_mode))
    tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    tg_app.add_handler(CommandHandler("panic", cmd_panic))
    tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
    tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    tg_app.add_handler(CommandHandler("trailing", cmd_trailing))
    tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
    tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
    tg_app.add_handler(CallbackQueryHandler(cb_buttons))

    # Starta appen (krävs för process_update)
    await tg_app.initialize()
    await tg_app.start()

    # Sätt webhook
    wh_url = f"{WEBHOOK_BASE.rstrip('/')}/webhook/{BOT_TOKEN}"
    await tg_app.bot.set_webhook(wh_url)
    log.info(f"Webhook satt: {wh_url}")

@app.on_event("shutdown")
async def on_shutdown():
    if tg_app:
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=False)
        except Exception:
            pass
        await tg_app.stop()
        await tg_app.shutdown()

@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return PlainTextResponse("bad token", status_code=403)
    data = await request.json()
    try:
        update = Update.de_json(data, bot=tg_app.bot)  # type: ignore
        await tg_app.process_update(update)            # kräver initialize/start (fixat i startup)
    except Exception:
        log.exception("Kunde inte processa update")
    return "OK"
