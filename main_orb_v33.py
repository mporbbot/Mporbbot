# main_orb_v33.py
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)


# ========= Miljö =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))  # sätt i Render
PING_URL = os.getenv("PING_URL", "")

SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT").split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")  # 1m eller 3m
ENTRY_MODE = os.getenv("ENTRY_MODE", "tick").lower()  # tick | close
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "50"))
LIVE = os.getenv("LIVE", "0") == "1"

KU_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KU_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KU_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

SERVICE_URL = os.getenv("RENDER_EXTERNAL_URL", os.getenv("SERVICE_URL", "")) or "https://mporbbot.onrender.com"
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{SERVICE_URL.rstrip('/')}{WEBHOOK_PATH}"

KU_BASE = "https://api.kucoin.com"  # public endpoints only for this bot


# ========= Hjälpare =========
def to_ku_symbol(s: str) -> str:
    s = s.replace("-", "").upper()
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s

def tf_to_ku(tf: str) -> str:
    return {"1m": "1min", "3m": "3min"}.get(tf, "1min")

def fmt_usd(x: float) -> str:
    return f"{x:.4f} USDT"

def green(candle: dict) -> bool:
    return float(candle["close"]) > float(candle["open"])

def red(candle: dict) -> bool:
    return float(candle["close"]) < float(candle["open"])

async def http_get_json(client: httpx.AsyncClient, url: str, params: Dict = None) -> dict:
    r = await client.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# ========= Datamodeller =========
@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0
    stop: float = 0.0
    open_time: float = 0.0
    pnl: float = 0.0  # realiserad för symbolen

@dataclass
class SymbolState:
    symbol: str = ""
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_time: Optional[int] = None
    in_pos: bool = False
    pos: Position = field(default_factory=Position)
    last_candle_ts: Optional[int] = None  # millisekunder
    entry_mode: str = ENTRY_MODE

# ========= Global bot-state =========
engine_on: bool = False
states: Dict[str, SymbolState] = {s: SymbolState(symbol=s) for s in SYMBOLS}
day_pnl: float = 0.0
app = FastAPI()
tg_app: Optional[Application] = None
engine_task: Optional[asyncio.Task] = None


# ========= KuCoin marknadsdata =========
async def fetch_candles(client: httpx.AsyncClient, symbol: str, limit: int = 50) -> List[dict]:
    ku_sym = to_ku_symbol(symbol)
    params = {"type": tf_to_ku(TIMEFRAME), "symbol": ku_sym}
    data = await http_get_json(client, f"{KU_BASE}/api/v1/market/candles", params=params)
    # KuCoin returnerar nyaste först: [time,open,close,high,low,vol,turnover]
    rows = data.get("data", [])
    candles = []
    for r in rows[:limit]:
        candles.append({
            "ts": int(float(r[0]) * 1000),  # sek -> ms
            "open": float(r[1]),
            "close": float(r[2]),
            "high": float(r[3]),
            "low": float(r[4]),
            "vol": float(r[5]),
        })
    candles.reverse()  # äldst -> nyast
    return candles

async def fetch_ticker_price(client: httpx.AsyncClient, symbol: str) -> float:
    ku_sym = to_ku_symbol(symbol)
    data = await http_get_json(client, f"{KU_BASE}/api/v1/market/orderbook/level1", params={"symbol": ku_sym})
    return float(data["data"]["price"])


# ========= ORB/Entry/Stop-logik =========
def find_orb_from_series(state: SymbolState, candles: List[dict]) -> None:
    """
    Sätt ORB till första gröna efter en röd i den givna serien (om inte redan satt).
    """
    if state.orb_high is not None:
        return
    for i in range(1, len(candles)):
        prev, cur = candles[i - 1], candles[i]
        if red(prev) and green(cur):
            state.orb_high = cur["high"]
            state.orb_low = cur["low"]
            state.orb_time = cur["ts"]
            break

def new_candle_closed(state: SymbolState, candles: List[dict]) -> Tuple[bool, Optional[dict], Optional[dict]]:
    """
    Returnerar (is_new, prev_candle, last_closed)
    """
    if len(candles) < 2:
        return False, None, None
    last_closed = candles[-2]  # sista stängda
    prev = candles[-3] if len(candles) >= 3 else None
    ts = last_closed["ts"]
    is_new = (state.last_candle_ts is None) or (ts > state.last_candle_ts)
    if is_new:
        state.last_candle_ts = ts
    return is_new, prev, last_closed

async def try_entry(state: SymbolState, candles: List[dict], live_price: float) -> Optional[str]:
    if state.in_pos or state.orb_high is None:
        return None

    # Entry-regler
    if state.entry_mode == "tick":
        cond = live_price > state.orb_high
    else:  # close
        if len(candles) < 2:
            return None
        last_closed = candles[-2]
        cond = last_closed["close"] > state.orb_high

    if cond:
        # beräkna storlek
        price = live_price
        qty = round(MOCK_TRADE_USDT / price, 6)
        state.in_pos = True
        state.pos = Position(qty=qty, entry_price=price, stop=state.orb_low, open_time=time.time())
        msg = (
            f"🟢 <b>BUY {state.symbol}</b>\n"
            f"entry={price:.6f}  stop={state.orb_low:.6f}\n"
            f"size={qty:.6f} ({MOCK_TRADE_USDT} USDT)\n"
            f"reason={'TickBreak' if state.entry_mode=='tick' else 'CloseBreak'}"
        )
        return msg
    return None

def trail_stop_each_candle(state: SymbolState, last_closed: dict) -> Optional[str]:
    """
    Flytta stop till föregående candles low om det är högre än nuvarande stop.
    """
    if not state.in_pos:
        return None
    new_stop = float(last_closed["low"])
    if new_stop > state.pos.stop:
        old = state.pos.stop
        state.pos.stop = new_stop
        return f"🔧 Trail {state.symbol}: stop {old:.6f} ➜ {new_stop:.6f}"
    return None

def check_exit_by_stop(state: SymbolState, live_price: float) -> Optional[str]:
    global day_pnl
    if not state.in_pos:
        return None
    if live_price <= state.pos.stop:
        # Exit (market)
        pnl = (live_price - state.pos.entry_price) * state.pos.qty
        day_pnl += pnl
        state.pos.pnl += pnl
        msg = (
            f"🔴 <b>EXIT {state.symbol}</b>\n"
            f"price={live_price:.6f}  pnl={fmt_usd(pnl)}\n"
            f"dayPnL={fmt_usd(day_pnl)}"
        )
        # reset
        state.in_pos = False
        state.pos = Position()
        # Nollställ ORB så vi väntar på nästa setup
        state.orb_high = None
        state.orb_low = None
        state.orb_time = None
        return msg
    return None


# ========= Telegram-kommandon =========
def owner_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user and update.effective_user.id != OWNER_CHAT_ID:
            return
        return await func(update, context)
    return wrapper

@owner_only
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(
        "🔧 ORB v33 startad.\nAnvänd /engine_on för att börja.\n"
        "Prova även /help, /status, /ping.",
        parse_mode=ParseMode.HTML,
    )

@owner_only
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(
        "<b>Kommandon</b>\n"
        "/start – starta boten\n"
        "/help – denna hjälp\n"
        "/engine_on – aktivera engine\n"
        "/engine_off – stäng av engine\n"
        "/status – visa status\n"
        "/entry_mode – visa/ändra entry (tick/close)\n"
        "/pnl – visa dagens PnL\n"
        "/reset_pnl – nollställ PnL\n"
        "/panic – stänger eventuella mock-positioner",
        parse_mode=ParseMode.HTML,
    )

@owner_only
async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_on, engine_task
    if engine_on:
        await update.effective_chat.send_message("✅ Engine är redan AKTIV.")
        return
    engine_on = True
    engine_task = asyncio.create_task(run_engine_loop())
    await update.effective_chat.send_message("✅ Engine är nu AKTIV (mock-läge)." if not LIVE else "✅ Engine (LIVE) är AKTIV.")

@owner_only
async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_on
    engine_on = False
    await update.effective_chat.send_message("🛑 Engine stoppad.")

@owner_only
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [
        f"Mode: {'live' if LIVE else 'mock'}   Engine: {'ON' if engine_on else 'OFF'}",
        f"TF: {TIMEFRAME}   Symbols: {','.join(SYMBOLS)}",
        f"Entry: {ENTRY_MODE.upper()}   DayPnL: {fmt_usd(day_pnl)}",
    ]
    for s in SYMBOLS:
        st = states[s]
        pos = "✅" if st.in_pos else "—"
        stop = f"{st.pos.stop:.6f}" if st.in_pos else "-"
        orb = "ON" if st.orb_high is not None else "OFF"
        lines.append(f"{s}: pos={pos}  stop={stop} | ORB: {orb}")
    await update.effective_chat.send_message("\n".join(lines))

@owner_only
async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENTRY_MODE
    if context.args:
        mode = context.args[0].lower()
        if mode in ("tick", "close"):
            ENTRY_MODE = mode
            for st in states.values():
                st.entry_mode = mode
            await update.effective_chat.send_message(f"Entry-läge satt till <b>{mode}</b>.", parse_mode=ParseMode.HTML)
            return
    await update.effective_chat.send_message(f"Aktuellt entry-läge: <b>{ENTRY_MODE}</b>\nAnvänd: /entry_mode tick | close",
                                            parse_mode=ParseMode.HTML)

@owner_only
async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(f"PnL\nDayPnL: {fmt_usd(day_pnl)}")

@owner_only
async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global day_pnl
    day_pnl = 0.0
    for st in states.values():
        st.pos.pnl = 0.0
    await update.effective_chat.send_message("✅ DayPnL nollställd.")

@owner_only
async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # stäng mock-positioner och nollställ ORB
    closed = []
    for st in states.values():
        if st.in_pos:
            st.in_pos = False
            st.pos = Position()
            closed.append(st.symbol)
        st.orb_high = st.orb_low = st.orb_time = None
    if closed:
        await update.effective_chat.send_message("🛑 PANIC – stängde: " + ", ".join(closed))
    else:
        await update.effective_chat.send_message("🛑 PANIC – inga öppna positioner.")

@owner_only
async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message("pong")


# ========= Engine loop =========
async def run_engine_loop():
    async with httpx.AsyncClient() as client:
        while engine_on:
            try:
                for sym, st in states.items():
                    candles = await fetch_candles(client, sym, limit=60)
                    if not candles:
                        continue

                    # Ny stängd candle?
                    is_new, prev, last_closed = new_candle_closed(st, candles)

                    # ORB (första gröna efter röd)
                    if st.orb_high is None:
                        find_orb_from_series(st, candles)

                    # Entry?
                    live_price = await fetch_ticker_price(client, sym)
                    if not st.in_pos and st.orb_high is not None:
                        msg = await try_entry(st, candles, live_price)
                        if msg and tg_app:
                            await tg_app.bot.send_message(chat_id=OWNER_CHAT_ID, text=msg, parse_mode=ParseMode.HTML)

                    # Trail stop per candle
                    if st.in_pos and is_new and last_closed:
                        tmsg = trail_stop_each_candle(st, last_closed)
                        if tmsg and tg_app:
                            await tg_app.bot.send_message(chat_id=OWNER_CHAT_ID, text=tmsg)

                    # Exit på stop
                    if st.in_pos:
                        emsg = check_exit_by_stop(st, live_price)
                        if emsg and tg_app:
                            await tg_app.bot.send_message(chat_id=OWNER_CHAT_ID, text=emsg, parse_mode=ParseMode.HTML)

                # enkel keep-alive ping om satt
                if PING_URL:
                    try:
                        await client.get(PING_URL, timeout=5)
                    except Exception:
                        pass

                await asyncio.sleep(1.2)
            except Exception as e:
                if tg_app:
                    await tg_app.bot.send_message(chat_id=OWNER_CHAT_ID, text=f"⚠️ Engine fel: {e}")
                await asyncio.sleep(2.0)


# ========= FastAPI + Telegram webhook =========
class _TGUpdate(BaseModel):
    __root__: dict

@app.on_event("startup")
async def on_startup():
    global tg_app
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN saknas")
    tg = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()

    tg.add_handler(CommandHandler("start", cmd_start))
    tg.add_handler(CommandHandler("help", cmd_help))
    tg.add_handler(CommandHandler("engine_on", cmd_engine_on))
    tg.add_handler(CommandHandler("engineoff", cmd_engine_off))
    tg.add_handler(CommandHandler("engine_off", cmd_engine_off))
    tg.add_handler(CommandHandler("status", cmd_status))
    tg.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    tg.add_handler(CommandHandler("pnl", cmd_pnl))
    tg.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    tg.add_handler(CommandHandler("panic", cmd_panic))
    tg.add_handler(CommandHandler("ping", cmd_ping))

    await tg.initialize()
    await tg.start()  # krävs för process_update i webhook-läge
    # sätt webhook
    await tg.bot.set_webhook(WEBHOOK_URL, allowed_updates=Update.ALL_TYPES)
    tg_app = tg

@app.on_event("shutdown")
async def on_shutdown():
    global tg_app, engine_on, engine_task
    engine_on = False
    if engine_task:
        try:
            engine_task.cancel()
        except Exception:
            pass
        engine_task = None
    if tg_app:
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=False)
        except Exception:
            pass
        await tg_app.stop()
        await tg_app.shutdown()
        tg_app = None

@app.get("/")
async def root():
    return {"ok": True, "version": "v33", "engine": engine_on, "tf": TIMEFRAME, "entry": ENTRY_MODE}

@app.post(WEBHOOK_PATH)
async def telegram_webhook(req: Request):
    body = await req.json()
    update = Update.de_json(body, tg_app.bot)
    # filtrera bara ägaren
    if update.effective_user and OWNER_CHAT_ID and update.effective_user.id != OWNER_CHAT_ID:
        return {"ok": True}
    await tg_app.process_update(update)
    return {"ok": True}
