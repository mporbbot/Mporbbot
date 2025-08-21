import os
import json
import math
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
)
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    ContextTypes
)

# =======================
# Konfiguration (ENV)
# =======================
BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
USE_WEBHOOK = bool(WEBHOOK_BASE)

DEFAULT_SYMBOLS = "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT"
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "1m")  # 1m,3m,5m,15m,30m,1h
ENTRY_MODE = os.getenv("ENTRY_MODE", "close").lower()  # "close" eller "tick"

TRADE_USDT = float(os.getenv("TRADE_USDT", "50"))
MIN_GREEN_BODY_PCT = float(os.getenv("MIN_GREEN_BODY_PCT", "0.05"))  # % minsta "grön kropp"

POLL_SEC = 5  # engine-tick


# =======================
# States
# =======================
engine_on: bool = False
orb_enabled: bool = True
entry_mode: str = ENTRY_MODE
timeframe: str = TIMEFRAME

# PnL
pnl_by_symbol: Dict[str, float] = {s: 0.0 for s in SYMBOLS}

# telegram chat id för svar (sätts av /start eller /status första gången)
main_chat_id: Optional[int] = None

# KuCoin tf-map
KUCOIN_TF = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
}

# Position & ORB
@dataclass
class Position:
    symbol: str
    entry: float
    qty: float
    stop: float
    meta: Dict

@dataclass
class OrbState:
    high: float
    low: float
    index: int  # candle-index när ORB skapades

open_pos: Dict[str, Position] = {}
orb_state: Dict[str, OrbState] = {}

# Telegram app
tg_app: Application = ApplicationBuilder().token(BOT_TOKEN).build()

# FastAPI app
app = FastAPI()


# =======================
# Hjälpfunktioner
# =======================
def fmt_usd(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1:
        return f"{sign}{x:.2f} USDT"
    return f"{sign}{x:.4f} USDT"

def candle_body_pct(c) -> float:
    if c["open"] == 0:
        return 0.0
    return abs(c["close"] - c["open"]) / c["open"] * 100.0

def is_green_strict(c) -> bool:
    return c["close"] > c["open"] and candle_body_pct(c) >= MIN_GREEN_BODY_PCT

def is_red(c) -> bool:
    return c["close"] < c["open"]

def build_kbd() -> InlineKeyboardMarkup:
    # Bara neder-raden knapparna (som du vill)
    rows = [
        [InlineKeyboardButton("/status", callback_data="noop")],
        [InlineKeyboardButton("/engine_on", callback_data="noop"),
         InlineKeyboardButton("/engine_off", callback_data="noop")],
        [InlineKeyboardButton("/entrymode", callback_data="noop"),
         InlineKeyboardButton("/timeframe", callback_data="noop")],
        [InlineKeyboardButton("/orb_on", callback_data="noop"),
         InlineKeyboardButton("/orb_off", callback_data="noop")],
        [InlineKeyboardButton("/panic", callback_data="noop")],
    ]
    return InlineKeyboardMarkup(rows)


async def send(chat_id: int, text: str):
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text, reply_markup=build_kbd())
    except Exception:
        pass


# =======================
# KuCoin data
# =======================
async def kucoin_klines(symbol: str, tf: str, limit: int = 60) -> List[dict]:
    """
    Returnerar lista av stängda candles i stigande tidordning.
    KuCoin returnerar i fallande ordning, vi vänder.
    """
    k_tf = KUCOIN_TF.get(tf, "1min")
    url = f"https://api.kucoin.com/api/v1/market/candles?type={k_tf}&symbol={symbol}&count={limit}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json().get("data", [])
    # data: [ [time, open, close, high, low, volume, turnover], ... ] nyast först
    candles = []
    for row in reversed(data):
        t, o, c, h, l = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        candles.append({"ts": t, "open": o, "high": h, "low": l, "close": c})
    return candles

async def kucoin_last_price(symbol: str) -> float:
    url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json().get("data", {})
    return float(data.get("price"))


# =======================
# ORB-logik (NY!)
# =======================
def maybe_start_or_replace_orb(symbol: str, prev_c: dict, cur_c: dict, cur_idx: int):
    """Starta/ersätt ORB när vi ser röd -> grön. (Första gröna = ORB-ljus)"""
    if is_red(prev_c) and is_green_strict(cur_c):
        orb_state[symbol] = OrbState(high=cur_c["high"], low=cur_c["low"], index=cur_idx)

def has_orb(symbol: str) -> bool:
    return symbol in orb_state

def confirms_break_over_orb(symbol: str, cur_c: dict) -> bool:
    """Kräv GRÖN close > ORB-high (kan vara valfri senare grön candle)."""
    s = orb_state.get(symbol)
    if not s:
        return False
    return is_green_strict(cur_c) and cur_c["close"] > s.high

async def try_orb_entry(symbol: str, candles: List[dict], now_idx: int, chat_id: int):
    """Kalla vid stängd candle när vi INTE har position och ORB är på."""
    if len(candles) < 2:
        return
    prev_c, cur_c = candles[-2], candles[-1]

    # 1) Upptäck ny ORB om röd -> grön. Ersätt alltid med senaste.
    maybe_start_or_replace_orb(symbol, prev_c, cur_c, now_idx)

    # 2) Har vi en aktiv ORB? Vänta på valfri senare GRÖN close > ORB-high => entry
    if has_orb(symbol) and confirms_break_over_orb(symbol, cur_c):
        s = orb_state[symbol]
        entry_price = cur_c["close"]
        stop_price = s.low
        qty = TRADE_USDT / entry_price
        open_pos[symbol] = Position(
            symbol=symbol,
            entry=entry_price,
            qty=qty,
            stop=stop_price,
            meta={"orb_high": s.high, "orb_low": s.low, "orb_index": s.index, "trigger_index": now_idx}
        )
        orb_state.pop(symbol, None)
        msg = (
            f"🟢 ENTRY LONG (close) {symbol} @ {entry_price:.6f}\n"
            f"ORB(H:{s.high:.6f} L:{s.low:.6f}) | SL={stop_price:.6f} | QTY={qty:.6f}"
        )
        await send(chat_id, msg)


# =======================
# Trailing & exit
# =======================
async def handle_trailing_and_exit(symbol: str, candles: List[dict], chat_id: int):
    """Flytta SL till föregående stängda candles low om det höjer SL. Exit om pris < stop."""
    pos = open_pos.get(symbol)
    if not pos:
        return

    if len(candles) >= 2:
        last_closed = candles[-1]
        prev_closed = candles[-2]
        new_stop = max(pos.stop, prev_closed["low"])
        if new_stop > pos.stop:
            pos.stop = new_stop
            await send(chat_id, f"🛠️ SL flyttad {symbol} -> {pos.stop:.6f}")

    # Tick/exit-koll
    price = candles[-1]["close"]
    if entry_mode == "tick":
        try:
            price = await kucoin_last_price(symbol)
        except Exception:
            pass

    if price <= pos.stop:
        # Exit
        pnl = (price - pos.entry) * pos.qty
        pnl_by_symbol[symbol] += pnl
        await send(chat_id, f"🔴 EXIT {symbol} @ {price:.6f} | PnL: {fmt_usd(pnl)} ❌")
        open_pos.pop(symbol, None)


# =======================
# Telegram handlers
# =======================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global main_chat_id
    main_chat_id = update.effective_chat.id
    await tg_app.bot.set_my_commands([
        BotCommand("status", "Visa status"),
        BotCommand("engine_on", "Starta motorn"),
        BotCommand("engine_off", "Stoppa motorn"),
        BotCommand("entrymode", "Växla close/tick"),
        BotCommand("timeframe", "Välj timeframe"),
        BotCommand("orb_on", "Sätt ORB på"),
        BotCommand("orb_off", "Sätt ORB av"),
        BotCommand("panic", "Stäng alla positioner"),
    ])
    await send(main_chat_id, "✅ Bot igång. Använd knapparna.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global main_chat_id
    main_chat_id = update.effective_chat.id

    orb_txt = "PÅ (första gröna efter röd → köp när senare grön stänger > ORB-high)" if orb_enabled else "AV"
    pos_lines = []
    for s in SYMBOLS:
        if s in open_pos:
            p = open_pos[s]
            pos_lines.append(f"• {s}: LONG @ {p.entry:.6f} | SL={p.stop:.6f}")
    pos_block = "\n".join(pos_lines) if pos_lines else "inga"

    pnl_total = sum(pnl_by_symbol.values())
    msg = (
        f"Engine: {'ON' if engine_on else 'OFF'}\n"
        f"Entry mode: {entry_mode}\n"
        f"Timeframe: {timeframe}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"ORB: {orb_txt}\n"
        f"Positioner: {pos_block}\n"
        f"PnL total: {fmt_usd(pnl_total)}"
    )
    await send(main_chat_id, msg)

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_on, main_chat_id
    main_chat_id = update.effective_chat.id
    engine_on = True
    await send(main_chat_id, "Engine: ON ✅")

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_on, main_chat_id
    main_chat_id = update.effective_chat.id
    engine_on = False
    await send(main_chat_id, "Engine: OFF ⛔")

async def cmd_entrymode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global entry_mode, main_chat_id
    main_chat_id = update.effective_chat.id
    entry_mode = "tick" if entry_mode == "close" else "close"
    await send(main_chat_id, f"Entry mode satt till: {entry_mode}")

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global timeframe, main_chat_id
    main_chat_id = update.effective_chat.id
    # Tolka argument: /timeframe 3m osv
    args = context.args
    if args and args[0] in KUCOIN_TF:
        timeframe = args[0]
        await send(main_chat_id, f"Timeframe satt till {timeframe}")
    else:
        await send(main_chat_id, "Skriv t.ex. `/timeframe 1m` eller `3m` `5m` `15m` `30m` `1h`")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global orb_enabled, main_chat_id
    main_chat_id = update.effective_chat.id
    orb_enabled = True
    await send(main_chat_id, "🟢 ORB: PÅ.")

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global orb_enabled, main_chat_id
    main_chat_id = update.effective_chat.id
    orb_enabled = False
    orb_state.clear()
    await send(main_chat_id, "⚫ ORB: AV.")

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global main_chat_id
    main_chat_id = update.effective_chat.id
    # stäng mock-positioner på senaste close
    for s in list(open_pos.keys()):
        pos = open_pos.pop(s)
        # vi försöker hämta last price – fall back close via klines
        price = pos.entry
        try:
            price = await kucoin_last_price(s)
        except Exception:
            try:
                ks = await kucoin_klines(s, timeframe, limit=2)
                price = ks[-1]["close"]
            except Exception:
                pass
        pnl = (price - pos.entry) * pos.qty
        pnl_by_symbol[s] += pnl
        await send(main_chat_id, f"🔴 EXIT (panic) {s} @ {price:.6f} | PnL: {fmt_usd(pnl)} ❌")
    await send(main_chat_id, "Panic close klart.")

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("orb_on", cmd_orb_on))
tg_app.add_handler(CommandHandler("orb_off", cmd_orb_off))
tg_app.add_handler(CommandHandler("panic", cmd_panic))


# =======================
# Engine loop
# =======================
async def engine_loop():
    await asyncio.sleep(2.0)  # liten delay efter start
    while True:
        try:
            if engine_on and main_chat_id is not None:
                for symbol in SYMBOLS:
                    # Hämta candles
                    try:
                        kl = await kucoin_klines(symbol, timeframe, limit=60)
                    except Exception as e:
                        await send(main_chat_id, f"[{symbol}] Fel vid klines: {e}")
                        continue

                    # Vi jobbar med STÄNGDA candles → använd hela listan (sista är stängd)
                    if symbol not in open_pos:
                        if orb_enabled and entry_mode == "close":
                            await try_orb_entry(symbol, kl, len(kl)-1, main_chat_id)
                        elif orb_enabled and entry_mode == "tick":
                            # Vi använder fortfarande stängd candle för logik,
                            # men entrypriset bekräftas av tick (lastprice) i trailing/exit.
                            await try_orb_entry(symbol, kl, len(kl)-1, main_chat_id)
                    else:
                        await handle_trailing_and_exit(symbol, kl, main_chat_id)
        except Exception:
            # svälj oväntade fel och fortsätt loopen
            pass

        await asyncio.sleep(POLL_SEC)


# =======================
# FastAPI + webhook
# =======================
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return PlainTextResponse("Bad token", status_code=403)
    body = await request.json()
    update = Update.de_json(body, tg_app.bot)
    await tg_app.process_update(update)
    return "OK"

@app.on_event("startup")
async def on_startup():
    # starta engine
    asyncio.create_task(engine_loop())
    # webhook / polling
    if USE_WEBHOOK:
        url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
        await tg_app.bot.set_webhook(url)
    else:
        # starta polling i bakgrunden
        asyncio.create_task(tg_app.initialize())
        asyncio.create_task(tg_app.start())
        # ingen stop här; FastAPI lever ändå

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await tg_app.stop()
        await tg_app.shutdown()
    except Exception:
        pass
