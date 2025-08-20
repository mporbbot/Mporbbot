# filename: main_1.py
import os
import asyncio
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ========= Config / ENV =========
BOT_TOKEN: Final[str] = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_BASE: Final[str] = os.getenv("WEBHOOK_BASE", "")
OWNER_CHAT_ID: Final[int] = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

LIVE: int = int(str(os.getenv("LIVE", "0")).strip()[:1] or "0")  # 0 mock, 1 live
SYMBOLS: str = os.getenv(
    "SYMBOLS",
    "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT",
)

if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")
if not OWNER_CHAT_ID:
    raise RuntimeError("Saknar OWNER_CHAT_ID")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_BASE.rstrip('/')}{WEBHOOK_PATH}"

# ========= Bot state (minne) =========
state = {
    "engine_active": False,
    "entry_mode": "close",  # default = close
    "orb_on": True,
    "live": bool(LIVE),
    "symbols": [s.strip() for s in SYMBOLS.split(",") if s.strip()],
    "pnl": 0.0,
}

# ========= Telegram Application =========
app = FastAPI()
tg_app: Application | None = None

# --------- Reply Keyboard (huvudmenyn) ----------
MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[
        ["/status"],
        ["/engine_start", "/engine_stop"],
        ["/start_mock", "/start_live"],
        ["/entry_mode", "/trailing"],
        ["/pnl", "/reset_pnl"],
        ["/orb_on", "/orb_off"],
        ["/panic"],
    ],
    resize_keyboard=True,
)

# ========= Hjälp-funktioner =========
def only_owner(update: Update) -> bool:
    chat_id = update.effective_chat.id if update.effective_chat else 0
    return chat_id == OWNER_CHAT_ID

async def reject_if_not_owner(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    if not only_owner(update):
        await ctx.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ Endast ägaren får använda kommandon.",
        )
        return True
    return False

# ========= Handlers =========
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    text = (
        "🛠️ ORB-bot laddad.\n"
        "Använd knapparna nedan. Standard-entry är **close**.\n"
        "Tips: /set_entry för att byta snabbt via knapp."
    )
    await update.message.reply_text(text, reply_markup=MAIN_KB, parse_mode="Markdown")

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    msg = (
        "🧭 Kommandon:\n"
        "/start – visa meny\n"
        "/status – visa status\n"
        "/engine_start – starta engine (ingen live-handel ännu)\n"
        "/engine_stop – stoppa engine\n"
        "/start_mock – mockläge (inte implementerat, svarar bara)\n"
        "/start_live – liveläge (inte implementerat, svarar bara)\n"
        "/entry_mode – visa nuvarande entry-läge\n"
        "/set_entry – byt entry (knapp: close/tick)\n"
        "/trailing – (inte implementerat, svarar bara)\n"
        "/pnl – visa dagens PnL (mock)\n"
        "/reset_pnl – nollställer PnL (mock)\n"
        "/orb_on /orb_off – togglar ORB-logik (mock)\n"
        "/panic – snabb stop av engine\n"
    )
    await update.message.reply_text(msg, reply_markup=MAIN_KB)

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    s = state
    msg = (
        "📊 Status:\n"
        f"• Engine: {'AKTIV' if s['engine_active'] else 'AV'}\n"
        f"• Entry-mode: {s['entry_mode']}\n"
        f"• ORB: {'ON' if s['orb_on'] else 'OFF'}\n"
        f"• Läge: {'LIVE' if s['live'] else 'MOCK'}\n"
        f"• Symbols: {', '.join(s['symbols'])}\n"
        f"• PnL: {s['pnl']:.2f} USDT"
    )
    await update.message.reply_text(msg, reply_markup=MAIN_KB)

async def cmd_engine_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    state["engine_active"] = True
    await update.message.reply_text("✅ Engine startad.", reply_markup=MAIN_KB)

async def cmd_engine_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    state["engine_active"] = False
    await update.message.reply_text("🛑 Engine stoppad.", reply_markup=MAIN_KB)

async def cmd_panic(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    state["engine_active"] = False
    await update.message.reply_text("🧯 PANIC: Engine OFF. Alla aktiviteter stoppade.", reply_markup=MAIN_KB)

async def cmd_entry_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    await update.message.reply_text(f"Entry-mode: **{state['entry_mode']}**", parse_mode="Markdown", reply_markup=MAIN_KB)

async def cmd_set_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("close", callback_data="entry:close"),
                InlineKeyboardButton("tick", callback_data="entry:tick"),
            ]
        ]
    )
    await update.message.reply_text("Välj entry-läge:", reply_markup=kb)

async def cb_entry_choice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.callback_query is None:
        return
    cq = update.callback_query
    if not only_owner(update):
        await cq.answer("Endast ägaren får ändra.")
        return
    if cq.data and cq.data.startswith("entry:"):
        choice = cq.data.split(":", 1)[1]
        if choice in ("close", "tick"):
            state["entry_mode"] = choice
            await cq.answer(f"Entry-läge satt: {choice}")
            await cq.edit_message_text(f"Entry-läge satt: **{choice}**", parse_mode="Markdown")
        else:
            await cq.answer("Ogiltigt val")

# ----- stubs (svarar bara) -----
async def cmd_start_mock(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    state["live"] = False
    await update.message.reply_text("ℹ️ MOCK-läge markerat (plats-hållare).", reply_markup=MAIN_KB)

async def cmd_start_live(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    state["live"] = True
    await update.message.reply_text("ℹ️ LIVE-läge markerat (plats-hållare).", reply_markup=MAIN_KB)

async def cmd_trailing(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    await update.message.reply_text("Trailing: ej implementerat ännu.", reply_markup=MAIN_KB)

async def cmd_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    await update.message.reply_text(f"Aktuellt PnL (mock): {state['pnl']:.2f} USDT", reply_markup=MAIN_KB)

async def cmd_reset_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    state["pnl"] = 0.0
    await update.message.reply_text("PnL återställt till 0.00 USDT.", reply_markup=MAIN_KB)

async def cmd_orb_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    state["orb_on"] = True
    await update.message.reply_text("🟢 ORB: ON", reply_markup=MAIN_KB)

async def cmd_orb_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await reject_if_not_owner(update, ctx):
        return
    state["orb_on"] = False
    await update.message.reply_text("🔴 ORB: OFF", reply_markup=MAIN_KB)

# ========= FastAPI x PTB wiring =========
@app.on_event("startup")
async def _on_startup():
    global tg_app
    tg_app = Application.builder().token(BOT_TOKEN).build()

    # Handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("help", cmd_help))
    tg_app.add_handler(CommandHandler("status", cmd_status))

    tg_app.add_handler(CommandHandler("engine_start", cmd_engine_start))
    tg_app.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    tg_app.add_handler(CommandHandler("panic", cmd_panic))

    tg_app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    tg_app.add_handler(CommandHandler("set_entry", cmd_set_entry))
    tg_app.add_handler(CallbackQueryHandler(cb_entry_choice, pattern=r"^entry:(close|tick)$"))

    tg_app.add_handler(CommandHandler("start_mock", cmd_start_mock))
    tg_app.add_handler(CommandHandler("start_live", cmd_start_live))
    tg_app.add_handler(CommandHandler("trailing", cmd_trailing))
    tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
    tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
    tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))

    # Initiera & starta PTB i webhook-läge
    await tg_app.initialize()
    await tg_app.start()
    await tg_app.bot.set_webhook(WEBHOOK_URL)

@app.on_event("shutdown")
async def _on_shutdown():
    if tg_app:
        await tg_app.stop()
        await tg_app.shutdown()

@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if tg_app is None:
        return JSONResponse({"ok": False, "error": "tg_app not ready"}, status_code=503)
    data = await request.json()
    update = Update.de_json(data=data, bot=tg_app.bot)
    await tg_app.process_update(update)
    return JSONResponse({"ok": True})
