# main_v23.py
import os
import json
import math
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("orb_v23")

# =============================================================================
# Config / ENV
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()

# Trading / strategy defaults
DEFAULT_TIMEFRAME_MIN = 1            # 1 | 3 | 5 | 15
DOJI_BODY_PCT = 0.10                 # body <= 10% av range => doji
LONGS_ENABLED = True
SHORTS_ENABLED = False
TRAILING_ENABLED = True
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]

# Mock trading
TRADE_SIZE_USDT = 30.0

# Price polling
POLL_INTERVAL_SEC = 8

# =============================================================================
# Helpers
# =============================================================================
def kucoin_symbol(sym: str) -> str:
    # KuCoin REST använder t.ex. BTC-USDT
    if sym.endswith("USDT"):
        return sym[:-4] + "-USDT"
    return sym

def kline_type(minutes: int) -> str:
    return {1: "1min", 3: "3min", 5: "5min", 15: "15min"}.get(minutes, "1min")

def candle_color(o: float, c: float) -> Optional[str]:
    if c > o:
        return "green"
    elif c < o:
        return "red"
    else:
        return None  # doji eller flat

def is_doji(o: float, h: float, l: float, c: float) -> bool:
    rng = max(h - l, 0.0)
    body = abs(c - o)
    if rng == 0:
        return True
    return (body / rng) <= DOJI_BODY_PCT

def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.1f}"
    if p >= 10:
        return f"{p:,.2f}"
    return f"{p:.6f}"

def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"

def make_panel_keyboard(curr_tf: int) -> InlineKeyboardMarkup:
    tf_row = [
        InlineKeyboardButton("1m" + (" ✅" if curr_tf == 1 else ""), callback_data="tf:1"),
        InlineKeyboardButton("3m" + (" ✅" if curr_tf == 3 else ""), callback_data="tf:3"),
        InlineKeyboardButton("5m" + (" ✅" if curr_tf == 5 else ""), callback_data="tf:5"),
        InlineKeyboardButton("15m" + (" ✅" if curr_tf == 15 else ""), callback_data="tf:15"),
    ]
    row2 = [
        InlineKeyboardButton("ORB ON", callback_data="orb:on"),
        InlineKeyboardButton("ORB OFF", callback_data="orb:off"),
    ]
    row3 = [
        InlineKeyboardButton("TRAIL ON", callback_data="trail:on"),
        InlineKeyboardButton("TRAIL OFF", callback_data="trail:off"),
    ]
    return InlineKeyboardMarkup([tf_row, row2, row3])

def buy_card(symbol: str, price: float, orb_h: float, orb_l: float, tf: int) -> str:
    return (
        f"🟩 **BUY** {symbol}\n"
        f"TF: {tf}m | Entry: `{fmt_price(price)}`\n"
        f"ORB: H `{fmt_price(orb_h)}`  L `{fmt_price(orb_l)}`\n"
        f"SL (init): `{fmt_price(orb_l)}`  Trailing: {'ON' if TRAILING_ENABLED else 'OFF'}"
    )

def sell_card(symbol: str, price: float, reason: str, pnl_usd: float, tf: int) -> str:
    emoji = "🟥" if reason.lower() == "stop" else "🟧"
    return (
        f"{emoji} **SELL** {symbol}\n"
        f"TF: {tf}m | Exit: `{fmt_price(price)}` | Orsak: {reason}\n"
        f"PnL: {fmt_usd(pnl_usd)}"
    )

# =============================================================================
# Strategy state
# =============================================================================
@dataclass
class Position:
    side: str = ""                # "long" | "short" | ""
    qty: float = 0.0
    entry: float = 0.0
    stop: float = 0.0

@dataclass
class SymbolState:
    last_color: Optional[str] = None           # senaste icke-doji färg
    shift_armed: bool = False                  # skifte upptäckt, väntar på första candle efter skifte
    orb_high: float = 0.0
    orb_low: float = 0.0
    orb_ts: Optional[int] = None               # ms
    orb_dir: Optional[str] = None              # "bull" | "bear"
    pos: Position = field(default_factory=Position)
    realized_pnl: float = 0.0

@dataclass
class GlobalState:
    timeframe_min: int = DEFAULT_TIMEFRAME_MIN
    orb_enabled: bool = True
    trailing_enabled: bool = TRAILING_ENABLED
    longs_enabled: bool = LONGS_ENABLED
    shorts_enabled: bool = SHORTS_ENABLED
    symbols: List[str] = field(default_factory=lambda: SYMBOLS.copy())
    # Telegram
    admin_chat_id: Optional[int] = None

GS = GlobalState()
SYMS: Dict[str, SymbolState] = {s: SymbolState() for s in GS.symbols}

# =============================================================================
# Telegram app (webhook-driven)
# =============================================================================
app = FastAPI()
_application: Optional[Application] = None

@app.get("/")
async def root():
    return {"ok": True, "version": "v23", "tf": GS.timeframe_min, "orb": GS.orb_enabled}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/tg/webhook")
async def tg_webhook(request: Request):
    global _application
    if TELEGRAM_BOT_TOKEN == "" or _application is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    # valfri säkerhet via hemlig header
    if WEBHOOK_SECRET:
        header = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if header != WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret")

    data = await request.json()
    update = Update.de_json(data, _application.bot)
    await _application.process_update(update)
    return JSONResponse({"ok": True})

# =============================================================================
# Price / Klines
# =============================================================================
async def fetch_last_closed_candle(sym: str, tf_min: int) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Returnera (ts_ms, open, high, low, close) för SENAST STÄNGDA candle.
    """
    ktype = kline_type(tf_min)
    pair = kucoin_symbol(sym)
    url = f"https://api.kucoin.com/api/v1/market/candles?type={ktype}&symbol={pair}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            arr = r.json().get("data", [])
            if not arr:
                return None
            # KuCoin returnerar reverse chronological (nyaste först)
            it = arr[1] if len(arr) > 1 else arr[0]  # index 0 kan vara "forming"; ta [1] om finns
            # format: [time, open, close, high, low, volume, turnover]
            ts_s = int(it[0])
            o = float(it[1]); c = float(it[2]); h = float(it[3]); l = float(it[4])
            return (ts_s * 1000, o, h, l, c)
    except Exception as e:
        log.warning(f"[{sym}] fetch candle failed: {e}")
        return None

async def fetch_last_price(sym: str) -> Optional[float]:
    pair = kucoin_symbol(sym)
    url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={pair}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json().get("data", {})
            return float(data.get("price"))
    except Exception as e:
        log.warning(f"[{sym}] fetch price failed: {e}")
        return None

# =============================================================================
# Core ORB logic
# =============================================================================
async def process_symbol(sym: str, tf_min: int, bot_notify):
    st = SYMS[sym]
    ck = await fetch_last_closed_candle(sym, tf_min)
    if not ck:
        return
    ts, o, h, l, c = ck

    # Doji-filter
    if is_doji(o, h, l, c):
        log.info(f"{sym} SKIP_DOJI ts={ts}")
        return

    col = candle_color(o, c)  # red/green
    prev = st.last_color

    # 1) Kolla om färgskifte sker nu
    if col and prev and col != prev:
        st.shift_armed = True
        st.orb_dir = "bull" if col == "green" else "bear"
        log.info(f"{sym} SHIFT detected -> armed {st.orb_dir}")

    # Uppdatera last_color (för icke-doji)
    if col:
        st.last_color = col

    # 2) Sätt ny ORB på FÖRSTA icke-doji-candle efter skifte
    if st.shift_armed and col:
        st.orb_high = h
        st.orb_low = l
        st.orb_ts = ts
        st.shift_armed = False
        log.info(f"{sym} NEW_ORB: H={st.orb_high} L={st.orb_low} ts={st.orb_ts}")

    # 3) Entry / Exit
    if not GS.orb_enabled or st.orb_ts is None:
        return

    # Om position finns, traila stop och ev. exit
    if st.pos.side:
        if GS.trailing_enabled and st.pos.side == "long":
            # Flytta upp stop till candle.low om högre (aldrig sänka)
            if l > st.pos.stop:
                st.pos.stop = l
        if st.pos.side == "short" and GS.trailing_enabled:
            # För short: traila ned stop till candle.high om lägre
            if h < st.pos.stop:
                st.pos.stop = h

        # Exit på stop
        if st.pos.side == "long" and l <= st.pos.stop:
            exit_price = st.pos.stop
            pnl = (exit_price - st.pos.entry) * st.pos.qty
            st.realized_pnl += pnl
            await bot_notify(sell_card(sym, exit_price, "Stop", pnl, GS.timeframe_min))
            st.pos = Position()
            return

        if st.pos.side == "short" and h >= st.pos.stop:
            exit_price = st.pos.stop
            pnl = (st.pos.entry - exit_price) * st.pos.qty
            st.realized_pnl += pnl
            await bot_notify(sell_card(sym, exit_price, "Stop", pnl, GS.timeframe_min))
            st.pos = Position()
            return

    # Om ingen position, leta entry på ORB-break
    if not st.pos.side:
        # Long-entry
        if GS.longs_enabled and st.orb_dir == "bull" and h >= st.orb_high:
            price = max(st.orb_high, c)
            qty = TRADE_SIZE_USDT / max(price, 1e-9)
            st.pos = Position(side="long", qty=qty, entry=price, stop=st.orb_low)
            await bot_notify(buy_card(sym, price, st.orb_high, st.orb_low, GS.timeframe_min))
            return

        # Short-entry
        if GS.shorts_enabled and st.orb_dir == "bear" and l <= st.orb_low:
            price = min(st.orb_low, c)
            qty = TRADE_SIZE_USDT / max(price, 1e-9)
            st.pos = Position(side="short", qty=qty, entry=price, stop=st.orb_high)
            await bot_notify(buy_card(sym, price, st.orb_high, st.orb_low, GS.timeframe_min).replace("BUY", "SELL (SHORT)"))
            return

# =============================================================================
# Background worker
# =============================================================================
async def trading_loop(send_fn):
    while True:
        try:
            tasks = [process_symbol(sym, GS.timeframe_min, send_fn) for sym in GS.symbols]
            await asyncio.gather(*tasks)
        except Exception as e:
            log.exception(f"trading_loop error: {e}")
        await asyncio.sleep(POLL_INTERVAL_SEC)

# =============================================================================
# Telegram handlers
# =============================================================================
async def tg_send_text(chat_id: Optional[int], text: str):
    if _application and chat_id:
        try:
            await _application.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except Exception as e:
            log.warning(f"send_message failed: {e}")

async def tg_send(text: str):
    await tg_send_text(GS.admin_chat_id, text)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    GS.admin_chat_id = update.effective_chat.id
    kb = make_panel_keyboard(GS.timeframe_min)
    await update.message.reply_text(
        "✅ Mp ORBbot v23 kör (webhook). Välj timeframe och toggles:",
        reply_markup=kb
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [
        f"TF: {GS.timeframe_min}m | ORB: {'ON' if GS.orb_enabled else 'OFF'} | TRAIL: {'ON' if GS.trailing_enabled else 'OFF'}",
        f"Longs: {'ON' if GS.longs_enabled else 'OFF'} | Shorts: {'ON' if GS.shorts_enabled else 'OFF'}",
    ]
    # PnL summering
    total = 0.0
    for s in GS.symbols:
        total += SYMS[s].realized_pnl
    lines.append(f"Realized PnL: {fmt_usd(total)} (mock)")
    await update.message.reply_text("\n".join(lines))

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "📖 Kommandon:\n"
        "/start – öppna panel & sätt chat\n"
        "/status – status + PnL\n"
        "/tf <1|3|5|15> – byt timeframe\n"
        "/orb_on, /orb_off – toggla ORB\n"
        "/trailing <on|off> – toggla trailing SL\n"
        "/longs <on|off>, /shorts <on|off>\n"
        "/panic – stäng ev. positioner direkt\n"
        "/reset_pnl – nollställ PnL\n"
        "/help – denna hjälp\n\n"
        "Entry: ORB-break efter skifte (ny ORB på första icke-doji efter färgskifte).\n"
        "Stop: Sätts vid ORB-botten vid köp, flyttas upp varje candle (aldrig ned)."
    )
    await update.message.reply_text(txt)

async def cmd_tf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /tf 1|3|5|15")
        return
    try:
        v = int(context.args[0])
        if v in (1, 3, 5, 15):
            GS.timeframe_min = v
            await update.message.reply_text(f"⏱ Timeframe satt till {v}m", reply_markup=make_panel_keyboard(GS.timeframe_min))
        else:
            await update.message.reply_text("Endast 1,3,5,15 tillåtna.")
    except Exception:
        await update.message.reply_text("Använd: /tf 1|3|5|15")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    GS.orb_enabled = True
    await update.message.reply_text("✅ ORB ON")

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    GS.orb_enabled = False
    await update.message.reply_text("🛑 ORB OFF")

async def cmd_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /trailing on|off")
        return
    v = context.args[0].lower()
    if v in ("on", "off"):
        GS.trailing_enabled = (v == "on")
        await update.message.reply_text(f"Trailing: {'ON' if GS.trailing_enabled else 'OFF'}")
    else:
        await update.message.reply_text("Använd: /trailing on|off")

async def cmd_longs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /longs on|off")
        return
    v = context.args[0].lower()
    if v in ("on", "off"):
        GS.longs_enabled = (v == "on")
        await update.message.reply_text(f"Longs: {'ON' if GS.longs_enabled else 'OFF'}")
    else:
        await update.message.reply_text("Använd: /longs on|off")

async def cmd_shorts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /shorts on|off")
        return
    v = context.args[0].lower()
    if v in ("on", "off"):
        GS.shorts_enabled = (v == "on")
        await update.message.reply_text(f"Shorts: {'ON' if GS.shorts_enabled else 'OFF'}")
    else:
        await update.message.reply_text("Använd: /shorts on|off")

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    closed = 0
    for s in GS.symbols:
        st = SYMS[s]
        if st.pos.side:
            # stäng på market (senaste pris)
            px = await fetch_last_price(s) or st.pos.entry
            pnl = (px - st.pos.entry) * st.pos.qty if st.pos.side == "long" else (st.pos.entry - px) * st.pos.qty
            st.realized_pnl += pnl
            await tg_send(sell_card(s, px, "Panic", pnl, GS.timeframe_min))
            st.pos = Position()
            closed += 1
    await update.message.reply_text(f"⚠️ Panic close klar. Stängda positioner: {closed}")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for s in GS.symbols:
        SYMS[s].realized_pnl = 0.0
    await update.message.reply_text("🔄 PnL nollställd.")

# Inline-knappar
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    await q.answer()
    data = q.data or ""
    if data.startswith("tf:"):
        v = int(data.split(":")[1])
        if v in (1, 3, 5, 15):
            GS.timeframe_min = v
        await q.edit_message_reply_markup(reply_markup=make_panel_keyboard(GS.timeframe_min))
    elif data == "orb:on":
        GS.orb_enabled = True
        await q.edit_message_reply_markup(reply_markup=make_panel_keyboard(GS.timeframe_min))
    elif data == "orb:off":
        GS.orb_enabled = False
        await q.edit_message_reply_markup(reply_markup=make_panel_keyboard(GS.timeframe_min))
    elif data == "trail:on":
        GS.trailing_enabled = True
        await q.edit_message_reply_markup(reply_markup=make_panel_keyboard(GS.timeframe_min))
    elif data == "trail:off":
        GS.trailing_enabled = False
        await q.edit_message_reply_markup(reply_markup=make_panel_keyboard(GS.timeframe_min))

# =============================================================================
# Telegram bootstrap (webhook)
# =============================================================================
async def init_telegram_and_webhook():
    global _application
    if not TELEGRAM_BOT_TOKEN:
        log.error("[TG] No TELEGRAM_BOT_TOKEN set. Skipping Telegram.")
        return

    _application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers
    _application.add_handler(CommandHandler("start", cmd_start))
    _application.add_handler(CommandHandler("status", cmd_status))
    _application.add_handler(CommandHandler("help", cmd_help))
    _application.add_handler(CommandHandler("tf", cmd_tf))
    _application.add_handler(CommandHandler("orb_on", cmd_orb_on))
    _application.add_handler(CommandHandler("orb_off", cmd_orb_off))
    _application.add_handler(CommandHandler("trailing", cmd_trailing))
    _application.add_handler(CommandHandler("longs", cmd_longs))
    _application.add_handler(CommandHandler("shorts", cmd_shorts))
    _application.add_handler(CommandHandler("panic", cmd_panic))
    _application.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    _application.add_handler(CallbackQueryHandler(on_callback))

    await _application.initialize()
    await _application.start()

    # Knappar i Telegram-menu
    try:
        await _application.bot.set_my_commands([
            BotCommand("start", "Starta & panel"),
            BotCommand("status", "Visa status & PnL"),
            BotCommand("tf", "Byt timeframe: /tf 1|3|5|15"),
            BotCommand("orb_on", "Aktivera ORB"),
            BotCommand("orb_off", "Avaktivera ORB"),
            BotCommand("trailing", "Trailing SL on/off"),
            BotCommand("longs", "Longs on/off"),
            BotCommand("shorts", "Shorts on/off"),
            BotCommand("panic", "Stäng positioner"),
            BotCommand("reset_pnl", "Nollställ PnL"),
            BotCommand("help", "Hjälp"),
        ])
    except Exception as e:
        log.warning(f"set_my_commands failed: {e}")

    # Webhook URL
    if not PUBLIC_URL:
        log.error("[TG] RENDER_EXTERNAL_URL saknas. Sätt PUBLIC URL i env för webhook.")
        return

    wh_url = f"{PUBLIC_URL}/tg/webhook"
    try:
        # städa ev. gammal webhook
        await _application.bot.delete_webhook(drop_pending_updates=True)
        await _application.bot.set_webhook(url=wh_url, secret_token=(WEBHOOK_SECRET or None))
        log.info(f"[TG] Webhook set to {wh_url} (secret set: {bool(WEBHOOK_SECRET)})")
    except Exception as e:
        log.error(f"[TG] set_webhook failed: {e}")

# =============================================================================
# FastAPI lifecycle
# =============================================================================
@app.on_event("startup")
async def on_startup():
    await init_telegram_and_webhook()
    # starta trading loop
    asyncio.create_task(trading_loop(tg_send))
    log.info("Startup complete (v23).")

@app.on_event("shutdown")
async def on_shutdown():
    global _application
    if _application:
        try:
            await _application.bot.delete_webhook()
        except Exception:
            pass
        await _application.stop()
        _application = None
    log.info("Shutdown complete.")
