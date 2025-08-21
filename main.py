# main.py
# En-filslösning: FastAPI + Telegram webhook + ORB mock spot-bot (KuCoin prisfeed)
# - ORB: breakout efter slutförd ORB-fönster (ex: första 20min varje timme)
# - Entry-läge: "close" (vid candle close) eller "tick" (på varje prisuppdatering)
# - Trailing stop som följer pris uppåt och låser vinst
# - Endast long & spot i mock-läge (LIVE=0). Live-order är markerade "TODO".

import os
import asyncio
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler
)

# ======== Miljövariabler / Standarder ========
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")
if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN i Environment")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

WEBHOOK_URL = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"

LIVE = int((os.getenv("LIVE") or "0").strip() or "0")  # 0 = mock, 1 = live (TODO orderläggning)
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
SYMBOLS_RAW = os.getenv("SYMBOLS", DEFAULT_SYMBOLS)
STAKE_USDT = float(os.getenv("STAKE_USDT", "50"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.5"))       # trailing i %
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.7"))  # hård SL i %
ORB_WINDOW_MIN = int(os.getenv("ORB_WINDOW_MIN", "20"))
ORB_RESET = (os.getenv("ORB_RESET", "hour")).lower()   # 'hour' eller 'day'
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "20m")  # används för candle-aggregatorn

# ======== Hjälp: symbolformat (KuCoin använder bindestreck) ========
def to_kucoin_symbol(sym: str) -> Optional[str]:
    s = sym.strip().upper().replace("-", "")
    # Hantera ev. missar som "XRPUS" -> "XRPUSDT"
    if s.endswith("US"):
        s += "DT"
    if not s.endswith("USDT"):
        return None
    base = s[:-4]
    return f"{base}-USDT"

def parse_symbols(raw: str):
    out = []
    for t in raw.split(","):
        k = to_kucoin_symbol(t)
        if k:
            out.append(k)
    return out

SYMBOLS = parse_symbols(SYMBOLS_RAW)
if not SYMBOLS:
    SYMBOLS = parse_symbols(DEFAULT_SYMBOLS)

# ======== Global state ========
state: Dict[str, Any] = {
    "engine_on": False,                 # start i "off" (du ville default stängd)
    "entry_mode": "close",              # "close" eller "tick"
    "timeframe": DEFAULT_TIMEFRAME,     # t.ex. "20m"
    "symbols": SYMBOLS,                 # KuCoin-format: "BTC-USDT"
    "positions": {},                    # per symbol: dict med mock-position
    "orb": {},                          # per symbol: { window_start, window_end, high, low, completed }
    "candles": {},                      # per symbol: aktuell aggregator för timeframe
    "last_msg_chat": None,              # ev. senaste chat-id vi kan pinga
}

# ======== Telegram setup ========
tg_app: Application = (
    Application.builder()
    .token(BOT_TOKEN)
    .concurrent_updates(True)
    .build()
)

# ======== FastAPI ========
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.update_queue.put(update)
    return JSONResponse({"ok": True})

# ======== Prisfeed (KuCoin) ========
def kucoin_get_price(symbol: str) -> Optional[float]:
    # symbol: "BTC-USDT"
    try:
        r = requests.get(
            "https://api.kucoin.com/api/v1/market/orderbook/level1",
            params={"symbol": symbol},
            timeout=6,
        )
        r.raise_for_status()
        data = r.json()
        return float(data["data"]["price"])
    except Exception:
        return None

# ======== Timeframe & candle-aggregator ========
def parse_timeframe_to_seconds(tf: str) -> int:
    t = tf.strip().lower()
    if t.endswith("m"):
        return int(t[:-1]) * 60
    if t.endswith("h"):
        return int(t[:-1]) * 3600
    if t.endswith("s"):
        return int(t[:-1])
    # fallback minuter
    return int(t) * 60

def candle_bucket_start(ts: datetime, tf_sec: int) -> datetime:
    # align to tf grid
    epoch = int(ts.timestamp())
    start = epoch - (epoch % tf_sec)
    return datetime.fromtimestamp(start, tz=timezone.utc)

def update_candle(sym: str, price: float, now: datetime):
    """Bygger en enkel OHLC för nuvarande timeframe."""
    tf_sec = parse_timeframe_to_seconds(state["timeframe"])
    bstart = candle_bucket_start(now, tf_sec)
    c = state["candles"].get(sym)
    if not c or c["start"] != bstart:
        # ny candle -> flytta ev. prev_close
        state["candles"][sym] = {
            "start": bstart,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "tf_sec": tf_sec,
        }
        # när ny candle startar är förra stängd => "close" utvärdering görs vid nästa varv
        return "new"
    else:
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price
        return "update"

def candle_is_closed(sym: str, now: datetime) -> bool:
    c = state["candles"].get(sym)
    if not c:
        return False
    return now >= (c["start"] + timedelta(seconds=c["tf_sec"]))

# ======== ORB-fönster ========
def orb_window_for_now(now: datetime) -> (datetime, datetime):
    if ORB_RESET == "day":
        day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        return day_start, day_start + timedelta(minutes=ORB_WINDOW_MIN)
    # default 'hour'
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    return hour_start, hour_start + timedelta(minutes=ORB_WINDOW_MIN)

def ensure_orb(sym: str, now: datetime):
    ow = state["orb"].get(sym)
    ws, we = orb_window_for_now(now)
    if not ow or ow["window_start"] != ws:
        state["orb"][sym] = {
            "window_start": ws,
            "window_end": we,
            "high": -math.inf,
            "low": math.inf,
            "completed": False,
        }

def update_orb(sym: str, price: float, now: datetime):
    ensure_orb(sym, now)
    ow = state["orb"][sym]
    if now < ow["window_end"]:
        ow["high"] = max(ow["high"], price)
        ow["low"] = min(ow["low"], price)
    elif not ow["completed"]:
        # stäng ORB när fönstret löpt ut
        ow["completed"] = True

# ======== Entry/Exit-regler ========
def should_enter_long(sym: str, price: float, now: datetime, closed_candle: bool) -> bool:
    ow = state["orb"].get(sym)
    if not ow or not ow["completed"]:
        return False
    if state["entry_mode"] == "close" and not closed_candle:
        return False
    # Breakout över ORB-high
    return price > ow["high"] and ow["high"] != -math.inf

def compute_trailing_stop(entry: float, max_price: float) -> float:
    # trail i procent av max pris över entry (låser vinst)
    trail = max_price * (1.0 - TRAIL_PCT / 100.0)
    # hård SL från entry (om pris aldrig gick upp)
    hard = entry * (1.0 - STOP_LOSS_PCT / 100.0)
    return max(trail, hard)

def should_exit(sym: str, price: float) -> bool:
    pos = state["positions"].get(sym)
    if not pos:
        return False
    return price <= pos["stop_price"]

# ======== Mock orderhantering ========
async def mock_buy(sym: str, price: float, chat_id: Optional[int]):
    qty = round(STAKE_USDT / price, 6)
    pos = {
        "symbol": sym,
        "entry_price": price,
        "qty": qty,
        "max_price": price,
        "stop_price": compute_trailing_stop(price, price),
        "opened_at": datetime.now(timezone.utc),
    }
    state["positions"][sym] = pos
    if chat_id:
        await tg_app.bot.send_message(
            chat_id,
            f"🟢 ENTRY LONG {sym}\nPris: {price:.4f}\nQty: {qty}\nORB breakout över high.",
        )

async def mock_sell(sym: str, price: float, chat_id: Optional[int], reason: str):
    pos = state["positions"].pop(sym, None)
    if not pos:
        return
    pnl = (price - pos["entry_price"]) * pos["qty"]
    if chat_id:
        await tg_app.bot.send_message(
            chat_id,
            f"🔴 EXIT {sym} ({reason})\nPris: {price:.4f}\nPnL: {pnl:.2f} USDT",
        )

# ======== Handelsloop ========
async def engine_loop():
    await tg_app.initialize()  # säkerställ att bot finns (ifall vi pingar)
    chat_id = state["last_msg_chat"]  # kan bli None tills någon kör /start
    while True:
        try:
            if not state["engine_on"]:
                await asyncio.sleep(1.0)
                continue

            now = datetime.now(timezone.utc)
            for sym in list(state["symbols"]):
                px = kucoin_get_price(sym)
                if px is None:
                    continue

                # bygg candle
                stamp = update_candle(sym, px, now)
                closed_candle_now = candle_is_closed(sym, now)
                # uppdatera ORB high/low och ev. stäng fönster
                update_orb(sym, px, now)

                # uppdatera trailing
                pos = state["positions"].get(sym)
                if pos:
                    if px > pos["max_price"]:
                        pos["max_price"] = px
                        pos["stop_price"] = compute_trailing_stop(pos["entry_price"], pos["max_price"])

                    # exit?
                    if should_exit(sym, px):
                        await mock_sell(sym, px, chat_id, reason="Trailing stop")

                else:
                    # entry?
                    if should_enter_long(sym, px, now, closed_candle_now):
                        await mock_buy(sym, px, chat_id)

            await asyncio.sleep(2.0)  # polling
        except Exception as e:
            # tyst återhämtning
            try:
                if state["last_msg_chat"]:
                    await tg_app.bot.send_message(state["last_msg_chat"], f"⚠️ Engine fel: {e}")
            except Exception:
                pass
            await asyncio.sleep(2.0)

# ======== Telegram: UI/kommandon ========
def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶️ Engine ON", callback_data="engine_on"),
            InlineKeyboardButton("⏹ Engine OFF", callback_data="engine_off"),
        ],
        [
            InlineKeyboardButton("Entry: close", callback_data="entry_close"),
            InlineKeyboardButton("Entry: tick", callback_data="entry_tick"),
        ],
        [
            InlineKeyboardButton("TF: 1m", callback_data="tf_1m"),
            InlineKeyboardButton("TF: 5m", callback_data="tf_5m"),
            InlineKeyboardButton("TF: 15m", callback_data="tf_15m"),
        ],
        [
            InlineKeyboardButton("TF: 20m", callback_data="tf_20m"),
            InlineKeyboardButton("TF: 30m", callback_data="tf_30m"),
            InlineKeyboardButton("TF: 1h", callback_data="tf_1h"),
        ],
        [
            InlineKeyboardButton("Panic close", callback_data="panic"),
            InlineKeyboardButton("Status", callback_data="status"),
        ],
    ])

async def set_bot_commands():
    cmds = [
        BotCommand("start", "Visa meny och spara chat"),
        BotCommand("status", "Visa status"),
        BotCommand("engine_on", "Starta motor"),
        BotCommand("engine_off", "Stoppa motor"),
        BotCommand("ping", "Svarstest"),
    ]
    await tg_app.bot.set_my_commands(cmds)

async def cmd_start(update: Update, _):
    chat_id = update.effective_chat.id
    state["last_msg_chat"] = chat_id
    await update.message.reply_text(
        "Hej! ORB-boten är online.\nVälj nedan:",
        reply_markup=main_menu_kb()
    )

async def cmd_status(update: Update, _):
    await send_status(update.effective_chat.id)

async def cmd_engine_on(update: Update, _):
    state["engine_on"] = True
    await update.message.reply_text("Engine: ON ✅")

async def cmd_engine_off(update: Update, _):
    state["engine_on"] = False
    await update.message.reply_text("Engine: OFF ⏹")

async def cmd_ping(update: Update, _):
    await update.message.reply_text("pong 🏓")

async def send_status(chat_id: int):
    lines = []
    lines.append(f"Engine: {'ON' if state['engine_on'] else 'OFF'}")
    lines.append(f"Entry mode: {state['entry_mode']}")
    lines.append(f"Timeframe: {state['timeframe']}")
    lines.append(f"Symbols: {', '.join(state['symbols'])}")
    lines.append(f"ORB: window={ORB_WINDOW_MIN}min reset={ORB_RESET}")
    if state["positions"]:
        lines.append("Positioner:")
        for sym, p in state["positions"].items():
            lines.append(
                f" - {sym} qty={p['qty']} entry={p['entry_price']:.4f} "
                f"max={p['max_price']:.4f} stop={p['stop_price']:.4f}"
            )
    else:
        lines.append("Positioner: inga")
    await tg_app.bot.send_message(chat_id, "\n".join(lines))

async def on_button(update: Update, _):
    q = update.callback_query
    data = q.data
    await q.answer()

    if data == "engine_on":
        state["engine_on"] = True
        await q.edit_message_text("Engine: ON ✅", reply_markup=main_menu_kb())
    elif data == "engine_off":
        state["engine_on"] = False
        await q.edit_message_text("Engine: OFF ⏹", reply_markup=main_menu_kb())

    elif data == "entry_close":
        state["entry_mode"] = "close"
        await q.edit_message_text("Entry mode: close", reply_markup=main_menu_kb())
    elif data == "entry_tick":
        state["entry_mode"] = "tick"
        await q.edit_message_text("Entry mode: tick", reply_markup=main_menu_kb())

    elif data.startswith("tf_"):
        tf = data.split("_", 1)[1]
        # validera
        ok = tf in ["1m", "5m", "15m", "20m", "30m", "1h"]
        if ok:
            state["timeframe"] = tf
            # nollställ candleaggregatorer så vi inte blandar olika TF
            state["candles"].clear()
            await q.edit_message_text(f"Timeframe satt till {tf}", reply_markup=main_menu_kb())
        else:
            await q.edit_message_text(f"Ogiltig timeframe: {tf}", reply_markup=main_menu_kb())

    elif data == "panic":
        # Stäng alla mock-positioner på marknadspris
        now = datetime.now(timezone.utc)
        closed = []
        for sym in list(state["positions"].keys()):
            px = kucoin_get_price(sym) or state["positions"][sym]["entry_price"]
            await mock_sell(sym, px, update.effective_chat.id, reason="Panic close")
            closed.append(sym)
        msg = "Panic close utförd." if closed else "Inga positioner att stänga."
        await q.edit_message_text(msg, reply_markup=main_menu_kb())

    elif data == "status":
        await send_status(update.effective_chat.id)

# ======== FastAPI livscykel ========
@app.on_event("startup")
async def on_startup():
    # Telegram init + webhook
    await tg_app.initialize()
    await tg_app.bot.set_webhook(WEBHOOK_URL, allowed_updates=["message", "callback_query"])
    await tg_app.start()
    await set_bot_commands()

    # Telegram handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    tg_app.add_handler(CommandHandler("ping", cmd_ping))
    tg_app.add_handler(CallbackQueryHandler(on_button))

    # starta handelsloop som bakgrunds-task
    asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

# ======== (valfri) HTTP-endpoint för debugging ========
@app.get("/status", response_class=PlainTextResponse)
async def http_status():
    lines = []
    lines.append(f"Engine: {'ON' if state['engine_on'] else 'OFF'}")
    lines.append(f"Entry: {state['entry_mode']}")
    lines.append(f"TF: {state['timeframe']}")
    lines.append(f"Symbols: {', '.join(state['symbols'])}")
    return "\n".join(lines)
