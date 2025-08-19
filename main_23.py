import os
import time
import math
import queue
import json
import threading
import logging
import asyncio
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# Telegram (python-telegram-bot v20.x)
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes
)

# --------------------------- Konfiguration ---------------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]
DEFAULT_TF_MIN = 1                      # 1|3|5|15
ENTRY_MODE = "TICK"                     # "TICK" eller "CLOSE"
TRAIL_ENABLED = True                    # trailing via candle-lows
ORB_MASTER = True                       # ORB på/av
MOCK_OR_LIVE = "mock"                   # "mock" | "live" (ingen riktig order här)

KUCOIN_ALL_TICKERS = "https://api.kucoin.com/api/v1/market/allTickers"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# --------------------------- Logger ---------------------------

LOG = logging.getLogger("orb_v23")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# --------------------------- Delad state ---------------------------

state_lock = threading.RLock()

class Position(BaseModel):
    symbol: str
    size: float
    entry: float
    stop: float
    reason: str
    opened_ts: float

class ORB(BaseModel):
    high: float
    low: float
    start_ts: int   # öppningstid för ORB-candlen (epoch ms)

class EngineState(BaseModel):
    engine_on: bool = False
    mode: str = MOCK_OR_LIVE
    tf_min: int = DEFAULT_TF_MIN
    entry_mode: str = ENTRY_MODE
    trail_on: bool = TRAIL_ENABLED
    orb_master: bool = ORB_MASTER
    pnl_day: float = 0.0

ENGINE = EngineState()
LAST_PRICE = {s: None for s in SYMBOLS}
POSITIONS: dict[str, Position | None] = {s: None for s in SYMBOLS}
ORBS: dict[str, ORB | None] = {s: None for s in SYMBOLS}

# Candle-aggregator för trailing (lows)
# Håller pågående candle och senast stängda low för respektive symbol
CANDLE = {
    s: {
        "open_ts": None,
        "high": None,
        "low": None,
        "close": None,
        "last_closed_low": None,
    } for s in SYMBOLS
}

# Telegram outbox (för säkra skick från andra trådar)
outbox = queue.Queue()

# Telegram application + loop (sätts i TG-tråden)
_tg_app = None
_tg_loop: asyncio.AbstractEventLoop | None = None

# --------------------------- Hjälp ---------------------------

def utc_now():
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def pct(a, b):
    try:
        return 100.0 * (a - b) / b
    except Exception:
        return 0.0

def fmt_num(x, sym):
    if sym.endswith("USDT"):
        # Grov avrundning per symbol
        if sym.startswith("BTC"): return f"{x:.1f}"
        if sym.startswith("ETH"): return f"{x:.2f}"
        if sym.startswith("XRP"): return f"{x:.4f}"
        if sym.startswith("ADA"): return f"{x:.4f}"
        if sym.startswith("LINK"): return f"{x:.4f}"
    return f"{x:.6f}"

def tf_bucket(epoch_ms: int, tf_min: int) -> int:
    """Returnera starttid (ms) för timeframe-bucket."""
    tf_ms = tf_min * 60_000
    return (epoch_ms // tf_ms) * tf_ms

def symbols_to_kucoin(s: str) -> str:
    # BTCUSDT -> BTC-USDT
    base = s[:-4]
    quote = s[-4:]
    return f"{base}-{quote}"

# --------------------------- Telegram UI ---------------------------

def base_keyboard():
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_start"), KeyboardButton("/engine_stop")],
        [KeyboardButton("/start_mock"), KeyboardButton("/start_live")],
        [KeyboardButton("/entry_mode"), KeyboardButton("/trailing")],
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/orb_on"), KeyboardButton("/orb_off")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def tf_keyboard():
    rows = [
        [KeyboardButton("/tf_1"), KeyboardButton("/tf_3"),
         KeyboardButton("/tf_5"), KeyboardButton("/tf_15")],
        [KeyboardButton("/status")]
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def send_outbox(chat_id: int | None, text: str):
    outbox.put({"chat_id": chat_id, "text": text})

async def _tg_outbox_worker():
    global _tg_app
    while True:
        item = await asyncio.get_event_loop().run_in_executor(None, outbox.get)
        try:
            if _tg_app is not None and item.get("chat_id"):
                await _tg_app.bot.send_message(
                    chat_id=item["chat_id"],
                    text=item["text"],
                    parse_mode=None,
                    reply_markup=base_keyboard()
                )
        except Exception as e:
            LOG.exception(f"[TG] send failed: {e}")

def buy_card(sym: str, entry: float, stop: float, size: float, reason: str):
    now = utc_now().strftime("%H:%M:%S UTC")
    pct_sl = pct(stop, entry)
    return (
        f"🟢 BUY {sym}\n"
        f"time={now}\n"
        f"entry={fmt_num(entry, sym)}  stop={fmt_num(stop, sym)} ({pct_sl:+.2f}%)\n"
        f"size={size:.6f}\n"
        f"reason={reason}"
    )

def sell_card(sym: str, price: float, entry: float, size: float, why: str):
    now = utc_now().strftime("%H:%M:%S UTC")
    pl = (price - entry) * size
    plp = pct(price, entry)
    return (
        f"🔴 SELL {sym}\n"
        f"time={now}\n"
        f"exit={fmt_num(price, sym)}  PnL={pl:+.4f} USDT ({plp:+.2f}%)\n"
        f"size={size:.6f}\n"
        f"reason={why}"
    )

def status_card():
    with state_lock:
        mode = ENGINE.mode
        tf = ENGINE.tf_min
        entry_mode = ENGINE.entry_mode
        trail = "ON" if ENGINE.trail_on else "OFF"
        orb = "ON" if ENGINE.orb_master else "OFF"
        kp = "ON"  # heartbeat on
        pnl = ENGINE.pnl_day
        eng = "ON" if ENGINE.engine_on else "OFF"

        lines = []
        lines.append(f"Mode: {mode}   Engine: {eng}")
        lines.append(f"TF: {tf}m   Symbols:")
        lines.append(",".join(SYMBOLS))
        lines.append(f"Entry: {entry_mode}   Trail: {trail}")
        lines.append(f"Keepalive: {kp}   DayPnL: {pnl:+.4f} USDT")
        lines.append(f"ORB master: {orb}")
        for s in SYMBOLS:
            pos = POSITIONS[s]
            mark = "✅" if pos else "❌"
            stop_txt = "-"
            if pos:
                stop_txt = fmt_num(pos.stop, s)
            lines.append(f"{s}: pos={mark}  stop={stop_txt} | ORB: { 'ON' if ENGINE.orb_master else 'OFF'}")
        return "\n".join(lines)

# --------------------------- Telegram handlers ---------------------------

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Mporbbot v23 online ✅\nAnvänd knapparna eller skriv kommandon.",
        reply_markup=base_keyboard(),
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(status_card(), reply_markup=base_keyboard())

async def cmd_engine_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.engine_on = True
    await cmd_status(update, ctx)

async def cmd_engine_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.engine_on = False
    await cmd_status(update, ctx)

async def cmd_start_mock(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.mode = "mock"
        ENGINE.engine_on = True
    await cmd_status(update, ctx)

async def cmd_start_live(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.mode = "live"  # (ingen riktig order här)
        ENGINE.engine_on = True
    await cmd_status(update, ctx)

async def cmd_entry_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.entry_mode = "CLOSE" if ENGINE.entry_mode == "TICK" else "TICK"
    await cmd_status(update, ctx)

async def cmd_trailing(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.trail_on = not ENGINE.trail_on
    await cmd_status(update, ctx)

async def cmd_orb_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.orb_master = True
    await cmd_status(update, ctx)

async def cmd_orb_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.orb_master = False
    await cmd_status(update, ctx)

async def cmd_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"DayPnL: {ENGINE.pnl_day:+.4f} USDT", reply_markup=base_keyboard())

async def cmd_reset_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with state_lock:
        ENGINE.pnl_day = 0.0
    await cmd_pnl(update, ctx)

async def cmd_panic(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # stäng allt på mark
    chat_id = update.effective_chat.id if update.effective_chat else None
    with state_lock:
        for s in SYMBOLS:
            pos = POSITIONS[s]
            if pos and LAST_PRICE[s]:
                price = float(LAST_PRICE[s])
                ENGINE.pnl_day += (price - pos.entry) * pos.size
                txt = sell_card(s, price, pos.entry, pos.size, "panic")
                send_outbox(chat_id, txt)
                POSITIONS[s] = None
    await cmd_status(update, ctx)

# timeframe snabbkommandon
async def cmd_tf_generic(update: Update, ctx: ContextTypes.DEFAULT_TYPE, tf: int):
    with state_lock:
        ENGINE.tf_min = tf
        # Rensa pågående candle-range för ren start
        for s in SYMBOLS:
            CANDLE[s] = {"open_ts": None, "high": None, "low": None, "close": None, "last_closed_low": None}
            ORBS[s] = None
    await update.message.reply_text(f"Timeframe satt till {tf}m.", reply_markup=tf_keyboard())
    await cmd_status(update, ctx)

async def cmd_tf_1(update: Update, ctx: ContextTypes.DEFAULT_TYPE):  return await cmd_tf_generic(update, ctx, 1)
async def cmd_tf_3(update: Update, ctx: ContextTypes.DEFAULT_TYPE):  return await cmd_tf_generic(update, ctx, 3)
async def cmd_tf_5(update: Update, ctx: ContextTypes.DEFAULT_TYPE):  return await cmd_tf_generic(update, ctx, 5)
async def cmd_tf_15(update: Update, ctx: ContextTypes.DEFAULT_TYPE): return await cmd_tf_generic(update, ctx, 15)

# --------------------------- Pris & ORB-motor ---------------------------

def update_candle(sym: str, price: float, now_ms: int, tf_min: int):
    bucket = tf_bucket(now_ms, tf_min)
    c = CANDLE[sym]
    if c["open_ts"] is None:
        c["open_ts"] = bucket
        c["high"] = price
        c["low"] = price
        c["close"] = price
        return False  # ingen stängning än

    if bucket != c["open_ts"]:
        # candle stänger -> uppdatera trailing source
        c["last_closed_low"] = c["low"]
        # starta ny candle
        c["open_ts"] = bucket
        c["high"] = price
        c["low"] = price
        c["close"] = price
        return True

    # uppdatera pågående
    c["high"] = max(c["high"], price)
    c["low"] = min(c["low"], price)
    c["close"] = price
    return False

def maybe_new_orb(sym: str, tf_min: int) -> None:
    # Sätt ORB = första candle efter bytet – här använder vi första candle i varje bucket som "ORB"
    c = CANDLE[sym]
    if ORBS[sym] is None and c["open_ts"] is not None and c["high"] is not None and c["low"] is not None:
        ORBS[sym] = ORB(high=c["high"], low=c["low"], start_ts=c["open_ts"])
        LOG.info(f"{sym} NEW_ORB: H={c['high']} L={c['low']} ts={c['open_ts']}")

def check_entry(sym: str, price: float, closed_just_now: bool, chat_id: int | None):
    if not ENGINE.orb_master:
        return
    orb = ORBS[sym]
    if orb is None:
        return

    pos = POSITIONS[sym]
    if pos:
        return  # redan inne

    # Entryvillkor
    if ENGINE.entry_mode == "TICK":
        cond = price > orb.high
        reason = "TickBreak"
    else:
        # CLOSE: kräv att senaste candle close > ORB high, så bara vid candle-stängning
        cond = closed_just_now and CANDLE[sym]["close"] is not None and CANDLE[sym]["close"] > orb.high
        reason = "CloseBreak"

    if cond:
        # storlek (mock): använd 100 USDT riskbudget
        qty = max(1e-8, 100.0 / price)
        stop = float(orb.low)  # din regel: initialt ORB-botten
        POSITIONS[sym] = Position(symbol=sym, size=qty, entry=price, stop=stop, reason=reason, opened_ts=time.time())
        LOG.info(f"{sym} OPEN {qty} @ {price} stop={stop}")
        send_outbox(chat_id, buy_card(sym, price, stop, qty, reason))

def check_trailing_and_stop(sym: str, price: float, chat_id: int | None):
    pos = POSITIONS[sym]
    if not pos:
        return

    # STOP hit?
    if price <= pos.stop:
        # stäng position
        pl = (pos.size * (pos.stop - pos.entry))
        ENGINE.pnl_day += pl
        send_outbox(chat_id, sell_card(sym, pos.stop, pos.entry, pos.size, "stop"))
        POSITIONS[sym] = None
        return

    # Trailing via candle-lows
    if ENGINE.trail_on:
        low = CANDLE[sym]["last_closed_low"]
        if low is not None and low > pos.stop:
            pos.stop = low  # flytta upp
            LOG.info(f"{sym} trail stop -> {pos.stop}")

def engine_loop():
    chat_id_for_cards = None  # sätts från /start första gången någon skriver
    last_sender = [None]

    async def remember_chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        # kopplas till /start för att ha en default chat_id för korten
        if update.effective_chat:
            last_sender[0] = update.effective_chat.id

    # Bind “remember_chat” i Telegram-appen när den är redo
    # (görs senare i tg_worker)

    with httpx.Client(timeout=5) as client:
        while True:
            try:
                with state_lock:
                    engine_on = ENGINE.engine_on
                    tf = ENGINE.tf_min

                if engine_on:
                    r = client.get(KUCOIN_ALL_TICKERS)
                    data = r.json()

                    now_ms = int(time.time() * 1000)
                    tick_by_sym = {}
                    for item in data.get("data", {}).get("ticker", []):
                        # item["symbol"] är "BTC-USDT"
                        ksym = item.get("symbol", "")
                        price = float(item.get("last", "0") or 0)
                        sym = ksym.replace("-", "")
                        if sym in SYMBOLS:
                            tick_by_sym[sym] = price

                    for sym in SYMBOLS:
                        if sym not in tick_by_sym:
                            continue
                        price = tick_by_sym[sym]
                        LAST_PRICE[sym] = price

                        closed = update_candle(sym, price, now_ms, tf)

                        # sätt ORB om saknas (första candle)
                        if ORBS[sym] is None:
                            maybe_new_orb(sym, tf)

                        # efter varje candle-stängning, uppdatera trailing och testa CLOSE-entry
                        check_trailing_and_stop(sym, price, last_sender[0])
                        check_entry(sym, price, closed, last_sender[0])

                time.sleep(1.0)
            except Exception as e:
                LOG.exception(f"engine error: {e}")
                time.sleep(2.0)

# --------------------------- Telegram worker ---------------------------

def telegram_worker():
    global _tg_app, _tg_loop

    if not TELEGRAM_BOT_TOKEN:
        LOG.error("[TG] No token – hoppar över Telegram. Sätt TELEGRAM_BOT_TOKEN.")
        return

    # Egen event-loop i denna tråd (fixar dina tidigare loop/signal-fel)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _tg_loop = loop

    async def main():
        global _tg_app
        _tg_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        # Kommandon
        _tg_app.add_handler(CommandHandler("start", cmd_start))
        _tg_app.add_handler(CommandHandler("status", cmd_status))
        _tg_app.add_handler(CommandHandler("engine_start", cmd_engine_start))
        _tg_app.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
        _tg_app.add_handler(CommandHandler("start_mock", cmd_start_mock))
        _tg_app.add_handler(CommandHandler("start_live", cmd_start_live))
        _tg_app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
        _tg_app.add_handler(CommandHandler("trailing", cmd_trailing))
        _tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
        _tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
        _tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
        _tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
        _tg_app.add_handler(CommandHandler("panic", cmd_panic))

        _tg_app.add_handler(CommandHandler("tf_1", cmd_tf_1))
        _tg_app.add_handler(CommandHandler("tf_3", cmd_tf_3))
        _tg_app.add_handler(CommandHandler("tf_5", cmd_tf_5))
        _tg_app.add_handler(CommandHandler("tf_15", cmd_tf_15))

        # Outbox worker i TG-loopen
        loop.create_task(_tg_outbox_worker())

        # Kör polling utan signals (vi är i bakgrundstråd)
        await _tg_app.run_polling(drop_pending_updates=True, stop_signals=None)

    try:
        # Sätt /setMyCommands för menyn i Telegram
        # (PTB gör det inte automatiskt här, men menyn byggs av knapparna också).
        loop.run_until_complete(main())
    finally:
        try:
            loop.stop()
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass

# --------------------------- FastAPI ---------------------------

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "bot": "mporbbot v23", "engine": ENGINE.engine_on}

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": utc_now().isoformat()}

# --------------------------- start ---------------------------

def start_all():
    # Starta TG-tråd
    t1 = threading.Thread(target=telegram_worker, name="telegram", daemon=True)
    t1.start()

    # Starta engine-tråd
    t2 = threading.Thread(target=engine_loop, name="engine", daemon=True)
    t2.start()

# Uvicorn startup hook
@app.on_event("startup")
def on_startup():
    LOG.info("Startup: kicking workers …")
    start_all()
