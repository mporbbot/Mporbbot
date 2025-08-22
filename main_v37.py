# main_v37.py
# FastAPI + Telegram webhook-bot (python-telegram-bot v20.x)
# Bygger på v36-layouten men med ny ORB & ENTRY-logik:
#   - ORB = första gröna candlen för dagen (close > open)
#   - Entry = första candle (#2 eller senare) som STÄNGER över ORB-high
#   - Stop = föregående candles low (trail), skyddar break-even efter vinst
#   - Bara "nedre" reply-knapparna (samma som v36)

import os
import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import httpx

# Telegram v20
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

# === ENV ===
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "").strip()  # valfri

if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN (lägg i Render Environment)")

if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_BASE}{WEBHOOK_PATH}"

# === KONFIG ===
DEFAULT_SYMBOLS = ["BTCUSDT"]            # spot/USDT-par
DEFAULT_TIMEFRAME = "1m"                 # vi kör 1m som i v36 om inget annat sattes
POLL_SEC = 5                             # mock-pris polling
MIN_GREEN_BODY_PCT = 0.05                # 5% kropp för att räknas som "grön"? (close>open räcker; pct används bara informativt)

# ====== MOCK/PRIS ======
async def fetch_candle(symbol: str, tf: str) -> Dict:
    """
    MOCK: hämtar "senaste candle" (OHLC) med ett enkelt prisflöde från Binance klona (public agg).
    Vill du byta till riktig exchange – byt här.
    Retur: dict: {open, high, low, close, ts}
    """
    # Minimal mock via Binance public kluster (utan API-nyckel). Faller tillbaka till "stegande" pris om fel.
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit=2"
    o,h,l,c,ts = None, None, None, None, None
    try:
        async with httpx.AsyncClient(timeout=5) as cli:
            r = await cli.get(url)
            r.raise_for_status()
            data = r.json()
            last = data[-1]
            o = float(last[1]); h = float(last[2]); l = float(last[3]); c = float(last[4])
            ts = int(last[6])
    except Exception:
        # fallback: syntetisk candle
        now = dt.datetime.utcnow().timestamp()
        base = 50000.0
        c = base + (now % 1200)  # svajar
        o = c * 0.999
        h = max(o, c) * 1.0005
        l = min(o, c) * 0.9995
        ts = int(now*1000)
    return {"open": o, "high": h, "low": l, "close": c, "ts": ts}

# ====== STATE ======
@dataclass
class Position:
    side: str = "long"
    entry: float = 0.0
    size: float = 0.0
    active: bool = False
    max_price: float = 0.0       # för break-even låsning

@dataclass
class SymbolState:
    timeframe: str = DEFAULT_TIMEFRAME
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_day: Optional[str] = None           # "YYYY-MM-DD" UTC
    orb_candle_ts: Optional[int] = None     # ts för första gröna candle
    entry_armed: bool = False               # armerad att köpa när candle stänger > orb_high
    position: Position = field(default_factory=Position)
    realized_pnl: float = 0.0
    last_candle: Optional[Dict] = None

class EngineState(BaseModel):
    running: bool = False
    entry_mode: str = "close"     # kvar från v36 (vi kör alltid close-krav ändå)
    symbols: List[str] = DEFAULT_SYMBOLS

ENGINE = EngineState()
SYMBOLS: Dict[str, SymbolState] = {s: SymbolState() for s in ENGINE.symbols}

# ====== ORB LOGIK ======
def _utc_day(ts_ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ts_ms/1000).strftime("%Y-%m-%d")

def is_green(c: Dict) -> bool:
    return c["close"] > c["open"]

def compute_orb_if_needed(sym: str, cndl: Dict) -> None:
    """
    Sätter ORB till första GRÖNA candlen för DAGEN (UTC).
    ORB-high/-low = high/low från den första gröna candlen.
    Entry blir armerad först efter att ORB hittats (dvs från nästa candle och framåt).
    """
    st = SYMBOLS[sym]
    day = _utc_day(cndl["ts"])

    # Reset vid ny dag
    if st.orb_day and day != st.orb_day:
        st.orb_high = st.orb_low = None
        st.orb_candle_ts = None
        st.entry_armed = False

    # Om ORB saknas – leta första gröna candlen
    if st.orb_high is None:
        if is_green(cndl):
            st.orb_high = cndl["high"]
            st.orb_low = cndl["low"]
            st.orb_day = day
            st.orb_candle_ts = cndl["ts"]
            st.entry_armed = False  # ARMERA på NÄSTA candle i loopen
    else:
        # ORB finns redan dagens datum – inget att göra
        st.orb_day = day

def ready_to_long(sym: str, prev_close: float, curr_closed: Dict) -> bool:
    """
    Entry-rule: efter att ORB (första gröna) är satt, får vi köpa
    på första CANDLE **EFTER** ORB-candlen som stänger över ORB-high.
    Vi säkerställer att curr_closed.ts != orb_candle_ts.
    """
    st = SYMBOLS[sym]
    if st.orb_high is None or st.orb_candle_ts is None:
        return False
    if curr_closed["ts"] == st.orb_candle_ts:
        return False  # får inte köpa på samma candle som definierade ORB
    # “Close”-entry: kräver close > ORB-high
    return curr_closed["close"] > st.orb_high

def trail_stop(sym: str, prev_closed: Dict) -> float:
    """
    Stop följer föregående candles low (klassisk candle-trail).
    """
    return prev_closed["low"]

def apply_be_lock(pos: Position, stop: float) -> float:
    """
    Break-even låsning: om vi är i vinst, höj stop minst till entry.
    """
    if pos.active and pos.max_price > pos.entry and stop < pos.entry:
        return pos.entry
    return stop

# ====== TELEGRAM (samma knapplayout som v36) ======
def keyboard_v36() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("Engine ON"), KeyboardButton("Engine OFF")],
        [KeyboardButton("Status"), KeyboardButton("PnL")],
        [KeyboardButton("Set Entry: close"), KeyboardButton("Set Entry: tick")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

async def send(chat_id: int, text: str, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await ctx.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard_v36())
    except Exception:
        pass

# ====== ENGINE LOOP ======
async def engine_loop(app: Application):
    await asyncio.sleep(1)
    while True:
        try:
            if not ENGINE.running:
                await asyncio.sleep(POLL_SEC)
                continue

            for sym in list(SYMBOLS.keys()):
                st = SYMBOLS[sym]
                tf = st.timeframe
                # Hämta två candles så vi kan använda föregående closed för stop-logik
                c2 = await fetch_candle(sym, tf)
                await asyncio.sleep(0.2)
                c1 = st.last_candle or c2  # föregående (fallback)
                st.last_candle = c2

                # Sätt ORB om saknas (första gröna)
                compute_orb_if_needed(sym, c2)

                # Entry (long) – endast om vi inte redan är inne
                if not st.position.active and st.orb_high is not None:
                    if ready_to_long(sym, c1["close"], c2):
                        # "köp"
                        st.position = Position(
                            side="long",
                            entry=c2["close"],
                            size=1.0,     # 1x mock
                            active=True,
                            max_price=c2["close"]
                        )
                        # skicka Entry-notis
                        if ADMIN_CHAT_ID:
                            try:
                                await app.bot.send_message(
                                    chat_id=int(ADMIN_CHAT_ID),
                                    text=f"ENTRY {sym} @ {st.position.entry:.2f} (ORB-high {st.orb_high:.2f})"
                                )
                            except Exception:
                                pass

                # Om aktiv position – uppdatera trail & ev exit
                if st.position.active:
                    pos = st.position
                    pos.max_price = max(pos.max_price, c2["close"])
                    stop = trail_stop(sym, c1)
                    stop = apply_be_lock(pos, stop)

                    # exit om priset bryter under stop
                    if c2["low"] <= stop:
                        exit_px = stop
                        # realiserad PnL
                        pnl = (exit_px - pos.entry) * pos.size
                        st.realized_pnl += pnl
                        # stäng
                        st.position = Position()  # reset
                        if ADMIN_CHAT_ID:
                            try:
                                await app.bot.send_message(
                                    chat_id=int(ADMIN_CHAT_ID),
                                    text=f"EXIT {sym} @ {exit_px:.2f} | PnL {pnl:.2f} | Tot: {st.realized_pnl:.2f}"
                                )
                            except Exception:
                                pass

            await asyncio.sleep(POLL_SEC)
        except Exception as e:
            # swallow & fortsätt
            await asyncio.sleep(POLL_SEC)

# ====== TG HANDLERS (v36-stil) ======
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "MporbBot v37 – ORB = första GRÖNA candle.\nEntry = candle #2+ som STÄNGER över ORB-high.\nStop = förra candlens low (BE-lås i vinst).",
        reply_markup=keyboard_v36()
    )

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    text = update.message.text.strip().lower()

    if text == "engine on":
        ENGINE.running = True
        await update.message.reply_text("Engine: ON", reply_markup=keyboard_v36())
        return

    if text == "engine off":
        ENGINE.running = False
        await update.message.reply_text("Engine: OFF", reply_markup=keyboard_v36())
        return

    if text == "status":
        lines = [f"Engine: {'ON' if ENGINE.running else 'OFF'}",
                 f"Symbols: {', '.join(ENGINE.symbols)}"]
        for s in ENGINE.symbols:
            st = SYMBOLS[s]
            lines.append(
                f"{s} | ORB: {st.orb_high:.2f if st.orb_high else '–'} / {st.orb_low:.2f if st.orb_low else '–'}"
            )
            if st.position.active:
                lines.append(f"  Pos: long @{st.position.entry:.2f}, max {st.position.max_price:.2f}")
            else:
                lines.append("  Pos: none")
        await update.message.reply_text("\n".join(lines), reply_markup=keyboard_v36())
        return

    if text == "pnl":
        lines = []
        tot = 0.0
        for s in ENGINE.symbols:
            r = SYMBOLS[s].realized_pnl
            tot += r
            lines.append(f"{s}: {r:.2f}")
        lines.append(f"TOTAL: {tot:.2f}")
        await update.message.reply_text("\n".join(lines), reply_markup=keyboard_v36())
        return

    if text == "set entry: close":
        ENGINE.entry_mode = "close"   # kvar för kompatibilitet
        await update.message.reply_text("Entry-läge: close", reply_markup=keyboard_v36())
        return

    if text == "set entry: tick":
        ENGINE.entry_mode = "tick"    # påverkar inte ORB-regeln; kvar för v36-kompat
        await update.message.reply_text("Entry-läge: tick (OBS: ORB kräver ändå close över high)", reply_markup=keyboard_v36())
        return

    # fallback
    await update.message.reply_text("Ok 👌", reply_markup=keyboard_v36())

# ====== FASTAPI + WEBHOOK ======
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    global tg_app
    tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Handlers (v36-stil)
    tg_app.add_handler(CommandHandler("start", start_cmd))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # sätt webhook
    await tg_app.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=["message"])
    # starta tg-app (processar inkommande i bakgrund)
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())

    # starta engine
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

class TgUpdate(BaseModel):
    update_id: int | None = None
    # vi skickar igenom rå payload till PTB
    # (PTB parsar själv)

@app.post(WEBHOOK_PATH)
async def tg_webhook(req: Request):
    raw = await req.json()
    update = Update.de_json(raw, tg_app.bot)
    await tg_app.process_update(update)
    return PlainTextResponse("OK")

@app.get("/")
async def health():
    return PlainTextResponse("OK")
