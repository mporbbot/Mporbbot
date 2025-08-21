# main_v37.py
# ORB v37 : bygger på v36 men med:
# - ORB = första GRÖNA candle efter en RÖD
# - ENTRY (close) = första efterföljande GRÖNA candle som STÄNGER ÖVER ORB-high
# - (valfritt) ENTRY (tick) = intrabar om priset bryter ORB-high
# - SL = ORB-low vid entry och flyttas upp till "nästa candle low" (trail candle-for-candle)
# - Mock spot "fills" mot KuCoin livepriser. Ingen Binance används.
# - PnL logg + summering i /status
# - Enbart botten-knappar (inga topp-rutor)

import os
import asyncio
import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, Response
from telegram import Update, ReplyKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes

# ------------------ ENV / DEFAULTS ------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").strip()  # t.ex. https://mporbbot.onrender.com
SYMBOLS_ENV = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
DEFAULT_TF = os.getenv("TIMEFRAME", "1m").lower()     # '1m' eller '3m'
ENTRY_MODE = os.getenv("ENTRY_MODE", "close").lower() # 'close' eller 'tick'
USDT_PER_TRADE = float(os.getenv("USDT_PER_TRADE", "100"))

# För "grön kropp" filter om du vill (0.0 = av)
MIN_GREEN_BODY_PCT = float(os.getenv("MIN_GREEN_BODY_PCT", "0.0"))  # ex 0.05 = 5%

ENGINE_POLL_SEC = int(os.getenv("ENGINE_POLL_SEC", "4"))

SYMBOLS = [s.strip().upper() for s in SYMBOLS_ENV.split(",") if s.strip()]
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# ------------------ STATE ------------------

@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    sl: float
    entry_ts: int
    last_trail_index: Optional[int] = None  # candle-index vi senast trailade på

@dataclass
class SymState:
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_index: Optional[int] = None  # index där ORB definierades
    pos: Optional[Position] = None

@dataclass
class GlobalState:
    engine_on: bool = False
    entry_mode: str = ENTRY_MODE  # 'close' / 'tick'
    timeframe: str = DEFAULT_TF
    orb_on: bool = True
    pnl_total: float = 0.0
    by_symbol: Dict[str, SymState] = field(default_factory=lambda: {s: SymState() for s in SYMBOLS})

STATE = GlobalState()

# ------------------ TELEGRAM ------------------

KB = ReplyKeyboardMarkup(
    [
        ["/status"],
        ["/engine_on", "/engine_off"],
        ["/entrymode", "/timeframe"],
        ["/orb_on", "/orb_off"],
        ["/panic"],
    ],
    resize_keyboard=True
)

tg_app: Application = Application.builder().token(BOT_TOKEN).build()

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(
        text=(
            "✅ ORB v37 igång.\n"
            "Styr med knapparna nedan.\n\n"
            f"TF: {STATE.timeframe}\n"
            f"Entry: {STATE.entry_mode}\n"
            f"Symbols: {', '.join(SYMBOLS)}\n"
            "ORB: PÅ (första gröna efter röd)"
        ),
        reply_markup=KB
    )

def fmt_usd(x: float) -> str:
    s = f"{x:.4f}"
    if abs(x) >= 1000:
        s = f"{x:,.2f}"
    return s

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Entry mode: {STATE.entry_mode}",
        f"Timeframe: {STATE.timeframe}",
        f"Symbols: {', '.join(SYMBOLS)}",
        "ORB: PÅ (första gröna efter röd)",
        f"PnL total: {fmt_usd(STATE.pnl_total)} USDT"
    ]
    await update.effective_chat.send_message("\n".join(lines), reply_markup=KB)

async def engine_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = True
    await update.effective_chat.send_message("Engine: ON ✅", reply_markup=KB)

async def engine_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = False
    await update.effective_chat.send_message("Engine: OFF ⬜️", reply_markup=KB)

async def entrymode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.entry_mode = "tick" if STATE.entry_mode == "close" else "close"
    await update.effective_chat.send_message(f"Entry mode satt till: {STATE.entry_mode}", reply_markup=KB)

async def timeframe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Enkel toggle mellan 1m <-> 3m (din favorit)
    STATE.timeframe = "3m" if STATE.timeframe == "1m" else "1m"
    await update.effective_chat.send_message(f"Timeframe satt till {STATE.timeframe}", reply_markup=KB)

async def orb_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.orb_on = True
    await update.effective_chat.send_message("ORB: PÅ (första gröna efter röd)", reply_markup=KB)

async def orb_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.orb_on = False
    await update.effective_chat.send_message("ORB: AV", reply_markup=KB)

async def panic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Stäng alla mock-positioner till nuvarande pris
    for sym in SYMBOLS:
        s = STATE.by_symbol[sym]
        if s.pos:
            price = await get_last_price(sym)
            pnl = (price - s.pos.entry_price) * s.pos.qty
            STATE.pnl_total += pnl
            await send_trade_exit(sym, price, pnl)
            s.pos = None
    await update.effective_chat.send_message("Panic close skickat (mock).", reply_markup=KB)

tg_app.add_handler(CommandHandler("start", start_cmd))
tg_app.add_handler(CommandHandler("status", status_cmd))
tg_app.add_handler(CommandHandler("engine_on", engine_on_cmd))
tg_app.add_handler(CommandHandler("engine_off", engine_off_cmd))
tg_app.add_handler(CommandHandler("entrymode", entrymode_cmd))
tg_app.add_handler(CommandHandler("timeframe", timeframe_cmd))
tg_app.add_handler(CommandHandler("orb_on", orb_on_cmd))
tg_app.add_handler(CommandHandler("orb_off", orb_off_cmd))
tg_app.add_handler(CommandHandler("panic", panic_cmd))

# Vi sätter bot-kommandon (visas i Telegrams meny)
BOT_COMMANDS = [
    BotCommand("status", "Visa status"),
    BotCommand("engine_on", "Starta motorn"),
    BotCommand("engine_off", "Stoppa motorn"),
    BotCommand("entrymode", "Växla entry: tick/close"),
    BotCommand("timeframe", "Växla TF: 1m/3m"),
    BotCommand("orb_on", "ORB på"),
    BotCommand("orb_off", "ORB av"),
    BotCommand("panic", "Stäng alla positioner"),
]

# ------------------ KUCOIN (HTTPX) ------------------

_client: Optional[httpx.AsyncClient] = None

def ku_tf(tf: str) -> str:
    return TF_MAP.get(tf, "1min")

async def ku_get_candles(symbol: str, tf: str, limit: int = 120) -> List[Dict]:
    """Returnerar candles som lista äldst->nyast. Fält: open, close, high, low (float)."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=10)

    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"type": ku_tf(tf), "symbol": symbol}
    r = await _client.get(url, params=params)
    r.raise_for_status()
    data = r.json()["data"]  # nyast först
    out = []
    for item in reversed(data):
        # KuCoin: [time, open, close, high, low, volume, turnover]
        ts, o, c, h, l = int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4])
        out.append({"ts": ts, "open": o, "close": c, "high": h, "low": l})
        if len(out) >= limit:
            break
    return out

async def get_last_price(symbol: str) -> float:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=10)
    url = "https://api.kucoin.com/api/v1/market/orderbook/level1"
    r = await _client.get(url, params={"symbol": symbol})
    r.raise_for_status()
    return float(r.json()["data"]["price"])

# ------------------ ORB / ENTRY LOGIK ------------------

def is_green(c): return c["close"] > c["open"]
def is_red(c): return c["close"] < c["open"]

def has_min_green_body(c) -> bool:
    if MIN_GREEN_BODY_PCT <= 0:
        return True
    body = c["close"] - c["open"]
    rng = c["high"] - c["low"]
    if rng <= 0:
        return False
    return (body / rng) >= MIN_GREEN_BODY_PCT

def find_orb(candles: List[Dict]) -> Optional[Tuple[int, float, float]]:
    """
    Hitta senaste ORB = första GRÖNA candle efter en RÖD.
    Returnerar (index, orb_high, orb_low). candles äldst->nyast (alla stängda).
    """
    # Gå bakifrån: hitta ett par (red -> green)
    for i in range(len(candles) - 1, 0, -1):
        prev = candles[i - 1]
        cur = candles[i]
        if is_red(prev) and is_green(cur) and has_min_green_body(cur):
            return i, cur["high"], cur["low"]
    return None

def first_break_after_orb(candles: List[Dict], orb_index: int, orb_high: float) -> Optional[int]:
    """
    Hitta första GRÖNA candle efter orb_index som STÄNGER ÖVER orb_high.
    Returnerar index, annars None.
    """
    for j in range(orb_index + 1, len(candles)):
        cj = candles[j]
        if is_green(cj) and cj["close"] > orb_high:
            return j
    return None

# ------------------ TRADE UTILS ------------------

async def send_trade_entry(symbol: str, price: float, orb_h: float, orb_l: float, sl: float, qty: float):
    txt = (
        f"🟢 ENTRY LONG (close) {symbol} @ {price:.6f}\n"
        f"ORB(H:{orb_h:.6f} L:{orb_l:.6f}) | SL={sl:.6f} | QTY={qty:.6f}"
    )
    await tg_app.bot.send_message(chat_id=context_chat_id(), text=txt)

async def send_trade_exit(symbol: str, price: float, pnl: float):
    mark = "✅" if pnl >= 0 else "❌"
    txt = f"🔴 EXIT {symbol} @ {price:.6f} | PnL: {pnl:.4f} USDT {mark}"
    await tg_app.bot.send_message(chat_id=context_chat_id(), text=txt)

async def send_trail(symbol: str, new_sl: float):
    txt = f"🔧 SL flyttad {symbol} -> {new_sl:.6f}"
    await tg_app.bot.send_message(chat_id=context_chat_id(), text=txt)

# Vi sparar senaste chat_id (från /start eller annat kommando) i minnet
_LAST_CHAT_ID: Optional[int] = None
def context_chat_id() -> int:
    # fallback om vi inte har chat ännu (kan sättas manuellt här om du vill)
    return _LAST_CHAT_ID or 0

async def remember_chat(update: Update):
    global _LAST_CHAT_ID
    if update and update.effective_chat:
        _LAST_CHAT_ID = update.effective_chat.id

# Hooka in “remember”
async def _remember_wrapper(func):
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await remember_chat(update)
        return await func(update, context)
    return wrapped

# Wrappar alla kommandon så vi sparar chat_id
tg_app.add_handler(CommandHandler("start", await _remember_wrapper(start_cmd)))
tg_app.add_handler(CommandHandler("status", await _remember_wrapper(status_cmd)))
tg_app.add_handler(CommandHandler("engine_on", await _remember_wrapper(engine_on_cmd)))
tg_app.add_handler(CommandHandler("engine_off", await _remember_wrapper(engine_off_cmd)))
tg_app.add_handler(CommandHandler("entrymode", await _remember_wrapper(entrymode_cmd)))
tg_app.add_handler(CommandHandler("timeframe", await _remember_wrapper(timeframe_cmd)))
tg_app.add_handler(CommandHandler("orb_on", await _remember_wrapper(orb_on_cmd)))
tg_app.add_handler(CommandHandler("orb_off", await _remember_wrapper(orb_off_cmd)))
tg_app.add_handler(CommandHandler("panic", await _remember_wrapper(panic_cmd)))

# ------------------ ENGINE ------------------

async def handle_symbol(symbol: str):
    s = STATE.by_symbol[symbol]
    candles = await ku_get_candles(symbol, STATE.timeframe, limit=200)
    if len(candles) < 5:
        return

    last_closed = candles[-1]  # KuCoin klines är stängda
    # 1) Om position finns -> trail & ev. exit
    if s.pos:
        # Trail SL till "nästa candle low": använd senaste stängda candlen
        if (s.pos.last_trail_index is None) or (s.pos.last_trail_index < len(candles) - 1):
            candidate = last_closed["low"]
            if candidate > s.pos.sl:
                s.pos.sl = candidate
                s.pos.last_trail_index = len(candles) - 1
                await send_trail(symbol, s.pos.sl)

        # Exit om pris <= SL
        price = await get_last_price(symbol)
        if price <= s.pos.sl:
            pnl = (price - s.pos.entry_price) * s.pos.qty
            STATE.pnl_total += pnl
            await send_trade_exit(symbol, price, pnl)
            s.pos = None

        return  # klart för öppna positioner

    # 2) Ingen position: definiera ORB (första gröna efter röd)
    if not STATE.orb_on:
        s.orb_high = s.orb_low = s.orb_index = None
        return

    orb = find_orb(candles)
    if not orb:
        s.orb_high = s.orb_low = s.orb_index = None
        return

    orb_idx, orb_h, orb_l = orb
    s.orb_high, s.orb_low, s.orb_index = orb_h, orb_l, orb_idx

    # 3) ENTRY-LOGIK
    if STATE.entry_mode == "close":
        first_break = first_break_after_orb(candles, orb_idx, orb_h)
        if first_break is not None and first_break == len(candles) - 1:
            # Enter nu på stängningspriset av senaste candle
            price = last_closed["close"]
            qty = max(USDT_PER_TRADE / price, 0.000001)
            s.pos = Position(symbol=symbol, entry_price=price, qty=qty, sl=orb_l, entry_ts=last_closed["ts"])
            await send_trade_entry(symbol, price, orb_h, orb_l, s.pos.sl, qty)

    else:  # tick
        # Intrabar om livepris bryter ORB-high och senaste candle är grön
        price = await get_last_price(symbol)
        if price >= orb_h and is_green(last_closed):
            price = float(price)
            qty = max(USDT_PER_TRADE / price, 0.000001)
            s.pos = Position(symbol=symbol, entry_price=price, qty=qty, sl=orb_l, entry_ts=last_closed["ts"])
            await send_trade_entry(symbol, price, orb_h, orb_l, s.pos.sl, qty)

async def engine_loop():
    # Sätt bot-kommandon i Telegram
    try:
        await tg_app.bot.set_my_commands(BOT_COMMANDS)
    except Exception:
        pass

    while True:
        try:
            if STATE.engine_on:
                for sym in SYMBOLS:
                    try:
                        await handle_symbol(sym)
                    except Exception as e:
                        try:
                            await tg_app.bot.send_message(
                                chat_id=context_chat_id(),
                                text=f"[{sym}] Fel i engine: {e}"
                            )
                        except Exception:
                            pass
            await asyncio.sleep(ENGINE_POLL_SEC)
        except asyncio.CancelledError:
            break
        except Exception as e:
            try:
                await tg_app.bot.send_message(chat_id=context_chat_id(), text=f"Engine-loop fel: {e}")
            except Exception:
                pass
            await asyncio.sleep(5)

# ------------------ FASTAPI + WEBHOOK ------------------

app = FastAPI()

@app.get("/")
async def root():
    return {"ok": True, "service": "mporbbot v37", "time": datetime.now(timezone.utc).isoformat()}

@app.post("/webhook/{token}")
async def telegram_webhook(request: Request, token: str):
    if token != BOT_TOKEN:
        return Response(status_code=403)
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return Response(status_code=200)

@app.on_event("startup")
async def on_startup():
    # sätt webhook
    if WEBHOOK_BASE:
        url = WEBHOOK_BASE.rstrip("/") + f"/webhook/{BOT_TOKEN}"
        try:
            await tg_app.bot.set_webhook(url)
        except Exception:
            pass
    # starta engine
    app.state.engine_task = asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def on_shutdown():
    try:
        app.state.engine_task.cancel()
    except Exception:
        pass
    if _client:
        await _client.aclose()
