# main_orb_v34.py
# ORB-bot (KuCoin, mock orders, live-priser)
# - ORB = första gröna candle efter röd (per symbol/timeframe)
# - Entry: close (candle close > ORB-high) eller tick (livepris bryter ORB-high)
# - Stop: initial = ORB-low (long). Trailing per candle: efter låströskel flyttas stop till föregående candle low (aldrig ned).
# - Telegram: meny med knappar (Engine ON/OFF, Entry Close/Tick, Timeframe, Status)

import os
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
)

# ====== ENV ======
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
OWNER_CHAT_ID = int((os.getenv("OWNER_CHAT_ID", "0") or "0").strip() or 0)

LIVE = int((os.getenv("LIVE", "0") or "0").strip() or 0)              # 0=mock, 1=live (ej aktiverat här)
ALLOW_SHORTS = int((os.getenv("ALLOW_SHORTS", "0") or "0").strip() or 0)

SYMBOLS_ENV = os.getenv("SYMBOLS", "BTCUSDT")
DEFAULT_SYMBOLS = [s.strip().upper() for s in SYMBOLS_ENV.split(",") if s.strip()]
DEFAULT_ENTRY = "close"    # "close" eller "tick" (startläge = close)
DEFAULT_TF = "3m"          # 1m/3m/5m/15m/30m

if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN (TELEGRAM_BOT_TOKEN).")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://din-app.onrender.com)")

# ====== FastAPI & Telegram Application ======
app = FastAPI(title="mporbbot ORB v34")
tg_app = Application.builder().token(BOT_TOKEN).build()

# ====== Pydantic modell för Telegram webhook ======
class _TGUpdate(BaseModel):
    update_id: Optional[int] = None
    message: Optional[dict] = None
    edited_message: Optional[dict] = None
    channel_post: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    inline_query: Optional[dict] = None
    chosen_inline_result: Optional[dict] = None
    callback_query: Optional[dict] = None
    shipping_query: Optional[dict] = None
    pre_checkout_query: Optional[dict] = None
    poll: Optional[dict] = None
    poll_answer: Optional[dict] = None
    my_chat_member: Optional[dict] = None
    chat_member: Optional[dict] = None
    chat_join_request: Optional[dict] = None

# ====== Hjälp ======

KU_TF_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
}

def to_kucoin_symbol(sym: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    sym = sym.upper()
    if "-" in sym:
        return sym
    if sym.endswith("USDT"):
        return sym[:-4] + "-USDT"
    if sym.endswith("USD"):
        return sym[:-3] + "-USD"
    if len(sym) > 3:
        return sym[:-3] + "-" + sym[-3:]
    return sym

async def ku_level1_price(session: httpx.AsyncClient, symbol: str) -> Optional[float]:
    """Hämta best bid/ask (vi använder 'price') från KuCoin public API."""
    try:
        r = await session.get("https://api.kucoin.com/api/v1/market/orderbook/level1",
                              params={"symbol": symbol}, timeout=10)
        d = r.json()
        if d.get("code") == "200000":
            price = float(d["data"]["price"])
            return price
    except Exception:
        return None
    return None

async def ku_recent_candles(session: httpx.AsyncClient, symbol: str, tf: str, limit: int = 5) -> List[Tuple[int,float,float,float,float]]:
    """
    Returnerar lista med candles: [(ts, open, high, low, close)], äldst->nyast.
    KuCoin svarar som listor: [time, open, close, high, low, volume, turnover]
    Vi konverterar och sorterar stigande på tid.
    """
    ku_tf = KU_TF_MAP.get(tf, "3min")
    try:
        r = await session.get("https://api.kucoin.com/api/v1/market/candles",
                              params={"symbol": symbol, "type": ku_tf}, timeout=10)
        d = r.json()
        if d.get("code") == "200000":
            raw = d["data"][:limit]  # KuCoin returnerar nyast först
            candles = []
            for row in raw:
                ts = int(row[0])
                o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
                candles.append((ts, o, h, l, c))
            candles.sort(key=lambda x: x[0])  # äldst->nyast
            return candles
    except Exception:
        pass
    return []

def is_green(candle) -> bool:
    _, o, h, l, c = candle
    return c > o

def is_red(candle) -> bool:
    _, o, h, l, c = candle
    return c < o

# ====== Trading-state ======
@dataclass
class Position:
    side: str                  # "long" / "short"
    entry: float
    size: float                # kvantitet (mockad)
    stop: float                # aktuell stop
    max_fav_price: float       # högsta pris (long) / lägsta pris (short) efter entry
    locked: bool = False       # blev "låst" (i vinst)

@dataclass
class SymbolState:
    symbol: str
    timeframe: str = DEFAULT_TF
    entry_mode: str = DEFAULT_ENTRY  # "close" / "tick"
    engine_on: bool = False
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_ts: Optional[int] = None     # tidsstämpel för ORB-candlen
    pos: Optional[Position] = None
    last_candle_ts: Optional[int] = None  # senaste stängda candle ts vi processade

@dataclass
class GlobalState:
    symbols: Dict[str, SymbolState] = field(default_factory=dict)
    allow_shorts: bool = bool(ALLOW_SHORTS)
    live: bool = bool(LIVE)
    lock_trigger_bp: float = 0.001   # 0.1% vinst innan trailing börjar låsa
    qty_usd: float = 50.0            # mockad storlek per trade

STATE = GlobalState()

for s in DEFAULT_SYMBOLS:
    STATE.symbols[s] = SymbolState(symbol=s, timeframe=DEFAULT_TF, entry_mode=DEFAULT_ENTRY, engine_on=False)

# ====== Telegram UI ======
def main_menu_markup(st: SymbolState) -> InlineKeyboardMarkup:
    # Engine
    eng_btn = InlineKeyboardButton(("🟢 Engine ON" if st.engine_on else "🔴 Engine OFF"),
                                   callback_data=f"ENGINE:{st.symbol}:{'off' if st.engine_on else 'on'}")
    # Entry
    e_close = InlineKeyboardButton(("✅ Entry: Close" if st.entry_mode=='close' else "Entry: Close"),
                                   callback_data=f"ENTRY:{st.symbol}:close")
    e_tick  = InlineKeyboardButton(("✅ Entry: Tick" if st.entry_mode=='tick' else "Entry: Tick"),
                                   callback_data=f"ENTRY:{st.symbol}:tick")
    # Timeframe
    tf_buttons = []
    for tf in ["1m","3m","5m","15m","30m"]:
        label = f"✅ TF: {tf}" if st.timeframe == tf else tf
        tf_buttons.append(InlineKeyboardButton(label, callback_data=f"TF:{st.symbol}:{tf}"))

    rows = [
        [eng_btn],
        [e_close, e_tick],
        tf_buttons[:3],
        tf_buttons[3:],
        [InlineKeyboardButton("🔎 Status", callback_data=f"STATUS:{st.symbol}")]
    ]
    return InlineKeyboardMarkup(rows)

async def safe_send(text: str):
    if OWNER_CHAT_ID:
        try:
            await tg_app.bot.send_message(chat_id=OWNER_CHAT_ID, text=text)
        except Exception:
            pass

def fmt_state(st: SymbolState) -> str:
    pos = st.pos
    ptxt = "Ingen" if not pos else f"{pos.side.upper()} @ {pos.entry:.4f} | stop {pos.stop:.4f} | locked={pos.locked}"
    orb = "saknas" if st.orb_high is None else f"H:{st.orb_high:.4f} / L:{st.orb_low:.4f}"
    return (f"📊 {st.symbol} | TF {st.timeframe} | Entry {st.entry_mode}\n"
            f"Engine: {'ON 🟢' if st.engine_on else 'OFF 🔴'}\n"
            f"ORB: {orb}\n"
            f"Pos: {ptxt}")

# ====== Telegram Handlers ======
async def cmd_start(update: Update, context):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID:
        return
    lines = ["Hej! ORB v34 redo. Välj symbol för inställningar:"]
    buttons = []
    row = []
    for s in STATE.symbols.values():
        row.append(InlineKeyboardButton(s.symbol, callback_data=f"OPENMENU:{s.symbol}"))
        if len(row) == 3:
            buttons.append(row); row=[]
    if row: buttons.append(row)
    await update.message.reply_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(buttons))

async def cbk(update: Update, context):
    if not update.effective_chat or update.effective_chat.id != OWNER_CHAT_ID:
        return
    q = update.callback_query
    if not q or not q.data: return
    parts = q.data.split(":")
    try:
        action = parts[0]
        if action == "OPENMENU":
            sym = parts[1]
            st = STATE.symbols.get(sym)
            if not st: return
            await q.answer("Meny")
            await q.message.edit_text(fmt_state(st), reply_markup=main_menu_markup(st))
            return

        if action == "ENGINE":
            sym, mode = parts[1], parts[2]
            st = STATE.symbols.get(sym)
            if not st: return
            st.engine_on = (mode == "on")
            if not st.engine_on:
                # Stäng ev mock-position
                st.pos = None
            await q.answer(f"Engine {mode.upper()}")
            await q.message.edit_text(fmt_state(st), reply_markup=main_menu_markup(st))
            return

        if action == "ENTRY":
            sym, mode = parts[1], parts[2]
            st = STATE.symbols.get(sym)
            if not st: return
            st.entry_mode = mode
            await q.answer(f"Entry: {mode}")
            await q.message.edit_text(fmt_state(st), reply_markup=main_menu_markup(st))
            return

        if action == "TF":
            sym, tf = parts[1], parts[2]
            st = STATE.symbols.get(sym)
            if not st: return
            st.timeframe = tf
            # Reset ORB vid byte TF
            st.orb_high = st.orb_low = st.orb_ts = None
            st.last_candle_ts = None
            await q.answer(f"TF = {tf}")
            await q.message.edit_text(fmt_state(st), reply_markup=main_menu_markup(st))
            return

        if action == "STATUS":
            sym = parts[1]
            st = STATE.symbols.get(sym)
            if not st: return
            await q.answer("Status")
            await q.message.edit_text(fmt_state(st), reply_markup=main_menu_markup(st))
            return

    except Exception as e:
        try:
            await q.answer("Fel")
        except Exception:
            pass

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CallbackQueryHandler(cbk))

# ====== ORB/Trading-logik ======
async def process_candles_for_symbol(session: httpx.AsyncClient, st: SymbolState):
    """
    - Hämtar senaste candles
    - Om ORB saknas: leta första 'grön efter röd' i de 5 senaste (äldst->nyast)
    - Hanterar entry på close (på senaste stängda candle) eller tick (livepris-korsning)
    - Uppdaterar trailing stop per ny candle efter låströskel
    - Exit på tick när pris <= stop (long)
    """
    sym_ku = to_kucoin_symbol(st.symbol)
    candles = await ku_recent_candles(session, sym_ku, st.timeframe, limit=6)
    if len(candles) < 3:
        return

    # identifiera senaste stängda candle och föregående
    closed_prev = candles[-2]
    closed_last = candles[-1]

    # Sätt ORB om saknas: första 'grön efter röd' i fönstret
    if st.orb_high is None:
        for i in range(1, len(candles)):
            c_prev = candles[i-1]
            c_cur  = candles[i]
            if is_red(c_prev) and is_green(c_cur):
                _, o,h,l,c = c_cur
                st.orb_high = h
                st.orb_low  = l
                st.orb_ts   = c_cur[0]
                await safe_send(f"🧭 ORB satt {st.symbol} [{st.timeframe}] H:{h:.4f} L:{l:.4f}")
                break

    # Trailing stop per candle (endast long i denna version)
    if st.pos and st.pos.side == "long":
        # Lås efter vinst ≥ lock_trigger_bp, annars vänta
        current_price = await ku_level1_price(session, sym_ku)
        if current_price:
            fav = max(st.pos.max_fav_price, current_price)
            st.pos.max_fav_price = fav
            up = (fav / st.pos.entry) - 1.0
            if (not st.pos.locked) and (up >= STATE.lock_trigger_bp):
                st.pos.locked = True
        # När ny candle har stängt, och locked=True → flytta stop till föregående candle low (aldrig ner)
        if st.pos.locked:
            if st.last_candle_ts is None or closed_last[0] != st.last_candle_ts:
                prev_low = closed_prev[3]  # [ts, o, h, l, c] -> l = index 3
                if prev_low > st.pos.stop:
                    st.pos.stop = prev_low
                    await safe_send(f"🔒 Trailing stop -> {st.symbol} {st.pos.stop:.4f}")
                st.last_candle_ts = closed_last[0]

    # Entry & Exit bara om engine_on
    if not st.engine_on or st.orb_high is None:
        return

    # Exit (tick mot stop)
    if st.pos and st.pos.side == "long":
        price = await ku_level1_price(session, sym_ku)
        if price and price <= st.pos.stop:
            # stäng mock-position
            pnl = (price - st.pos.entry) / st.pos.entry
            await safe_send(f"⬇️ EXIT LONG {st.symbol} @ {price:.4f} | PnL {pnl*100:.2f}%")
            st.pos = None
            return

    # Entry (bara long i denna version)
    if st.pos is None:
        # close-läge: använd senaste stängda candle (closed_last)
        if st.entry_mode == "close":
            _, o,h,l,c = closed_last
            if c > st.orb_high:
                entry_price = c
                size = max(STATE.qty_usd / entry_price, 0.0001)
                st.pos = Position(side="long", entry=entry_price, size=size,
                                  stop=st.orb_low, max_fav_price=entry_price)
                st.last_candle_ts = closed_last[0]
                await safe_send(f"🟢 ENTRY LONG (close) {st.symbol} @ {entry_price:.4f}\n"
                                f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | Stop {st.pos.stop:.4f}")
        else:
            # tick-läge: om livepris bryter ORB-high
            price = await ku_level1_price(session, sym_ku)
            if price and price > st.orb_high:
                entry_price = price
                size = max(STATE.qty_usd / entry_price, 0.0001)
                st.pos = Position(side="long", entry=entry_price, size=size,
                                  stop=st.orb_low, max_fav_price=entry_price)
                st.last_candle_ts = closed_last[0]  # lagra senast kända closed ts
                await safe_send(f"🟢 ENTRY LONG (tick) {st.symbol} @ {entry_price:.4f}\n"
                                f"ORB(H:{st.orb_high:.4f} L:{st.orb_low:.4f}) | Stop {st.pos.stop:.4f}")

# ====== Bakgrundsloop ======
async def trading_loop():
    await asyncio.sleep(1.0)
    async with httpx.AsyncClient(timeout=10) as session:
        while True:
            tasks = [process_candles_for_symbol(session, st) for st in STATE.symbols.values()]
            try:
                await asyncio.gather(*tasks)
            except Exception:
                pass
            await asyncio.sleep(2.5)  # frekvens

# ====== FastAPI routes ======
@app.get("/")
async def root():
    return {"ok": True, "service": "ORB v34"}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return Response(status_code=403)
    data = await request.json()
    upd = _TGUpdate(**data)
    await tg_app.process_update(Update.de_json(data, tg_app.bot))
    return Response(status_code=200)

# ====== Startup / Shutdown ======
async def set_bot_commands():
    try:
        await tg_app.bot.set_my_commands([
            ("start", "Visa meny & välj symbol"),
        ])
    except Exception:
        pass

@app.on_event("startup")
async def _startup():
    # Initiera Telegram Application
    await tg_app.initialize()
    await tg_app.start()

    await set_bot_commands()

    # Sätt webhook
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    try:
        await tg_app.bot.set_webhook(url)
        await safe_send("🔧 ORB v34 startad (webhook). Engine är OFF. Välj /start → symbol → slå på Engine.")
    except Exception:
        pass

    # Starta trading-loop
    asyncio.create_task(trading_loop())

@app.on_event("shutdown")
async def _shutdown():
    try:
        await tg_app.bot.delete_webhook()
    except Exception:
        pass
    await tg_app.stop()
    await tg_app.shutdown()
