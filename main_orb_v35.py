# main_orb_v35.py
# Komplett FastAPI + Telegram webhook + ORB-mocktrading med livepriser
# python-telegram-bot v20, FastAPI, httpx
import os
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

# ------------------------ ENV & KONFIG ------------------------

def env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")

def env_list(name: str, default: List[str]) -> List[str]:
    v = os.getenv(name)
    if not v:
        return default
    items = [s.strip().upper() for s in v.split(",") if s.strip()]
    # Filtrera bara symboler som slutar på USDT (enkel sanity)
    items = [s for s in items if s.endswith("USDT")]
    return items or default

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN")
if not WEBHOOK_BASE or not WEBHOOK_BASE.startswith("http"):
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "5397586616") or "5397586616")
LIVE = env_bool("LIVE", False)  # Live-knapp visas men riktiga ordrar är INTE implementerade här

SYMBOLS = env_list("SYMBOLS",
    ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]
)

# Timeframes vi stödjer (Binance publika API)
VALID_TF = ["1m", "3m", "5m", "15m", "30m"]
DEFAULT_TF = "1m"

# Mock-trade parametrar
LOCK_PROFIT_PCT = 0.1 / 100.0   # lås till break-even efter +0.1%
DEFAULT_QTY_USD = 50.0          # "storlek" per symbol i USD för mock
POLL_SEC = 3.0                  # hur ofta vi pollar pris/klines

# ------------------------ STATE ------------------------

class Position:
    def __init__(self, entry: float, qty: float, stop: float):
        self.entry = entry
        self.qty = qty
        self.stop = stop
        self.locked_to_be = False  # låst till break-even?

class ORB:
    def __init__(self, high: float, low: float, ts_close: int):
        self.high = high
        self.low = low
        self.ts_close = ts_close  # stängningstid (ms) för ORB-candlen

class SymbolState:
    def __init__(self):
        self.orb: Optional[ORB] = None
        self.position: Optional[Position] = None
        self.last_closed_open: Optional[float] = None  # för att hitta röd->grön
        self.last_closed_close: Optional[float] = None
        self.last_closed_ts: Optional[int] = None

class GlobalState:
    def __init__(self):
        self.engine_on: bool = False
        self.entry_mode: str = "close"  # "close" eller "tick"
        self.timeframe: str = DEFAULT_TF
        self.mock: bool = True          # mock-flag
        self.symbols: List[str] = SYMBOLS.copy()
        self.task: Optional[asyncio.Task] = None
        self.by_symbol: Dict[str, SymbolState] = {s: SymbolState() for s in self.symbols}

GS = GlobalState()

# ------------------------ HJÄLPFUNKTIONER ------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def menu_keyboard() -> InlineKeyboardMarkup:
    # Rad1: Engine ON/OFF
    # Rad2: Entry Close / Tick
    # Rad3: Timeframe (öppnar val)
    # Rad4: Mock/Live (visa status), Status
    # Rad5: Panic
    rows = [
        [
            InlineKeyboardButton("🚀 Engine ON", callback_data="engine:on"),
            InlineKeyboardButton("🛑 Engine OFF", callback_data="engine:off"),
        ],
        [
            InlineKeyboardButton("Entry: Close", callback_data="entry:close"),
            InlineKeyboardButton("Entry: Tick", callback_data="entry:tick"),
        ],
        [
            InlineKeyboardButton(f"Timeframe: {GS.timeframe}", callback_data="tf:choose"),
        ],
        [
            InlineKeyboardButton(f"Mode: {'MOCK' if GS.mock else 'LIVE'}", callback_data="mode:noop"),
            InlineKeyboardButton("📊 Status", callback_data="status:show"),
        ],
        [
            InlineKeyboardButton("❗ Panic (stop + flat)", callback_data="panic:now"),
        ],
    ]
    return InlineKeyboardMarkup(rows)

def timeframe_keyboard() -> InlineKeyboardMarkup:
    rows = []
    row = []
    for tf in VALID_TF:
        row.append(InlineKeyboardButton(tf, callback_data=f"tf:set:{tf}"))
        if len(row) == 5:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Tillbaka", callback_data="menu")])
    return InlineKeyboardMarkup(rows)

def only_owner(update: Update) -> bool:
    chat_id = None
    if update.effective_chat:
        chat_id = update.effective_chat.id
    elif update.callback_query and update.callback_query.message:
        chat_id = update.callback_query.message.chat_id
    return chat_id == OWNER_CHAT_ID

def fmt_state() -> str:
    lines = [
        f"⏱ {now_iso()}",
        f"Engine: {'ON' if GS.engine_on else 'OFF'}",
        f"Entry: {GS.entry_mode.upper()}",
        f"TF: {GS.timeframe}",
        f"Mode: {'MOCK' if GS.mock else 'LIVE'}",
        f"Symbols: {', '.join(GS.symbols)}",
    ]
    # Ev positioner
    openpos = []
    for sym, st in GS.by_symbol.items():
        if st.position:
            p = st.position
            openpos.append(f"{sym}: entry={p.entry:.4f} stop={p.stop:.4f} BE_locked={p.locked_to_be}")
    if openpos:
        lines.append("Öppna mock-positioner:")
        lines += [" • " + s for s in openpos]
    return "\n".join(lines)

async def send_owner(app: Application, text: str):
    try:
        await app.bot.send_message(chat_id=OWNER_CHAT_ID, text=text)
    except Exception:
        pass

# ------------------------ DATA (Binance publikt API) ------------------------

BINANCE = "https://api.binance.com"

async def binance_klines(client: httpx.AsyncClient, symbol: str, interval: str, limit: int=3) -> List[List]:
    # Returnerar lista med klines (öppettid, open, high, low, close, …, sluttid)
    url = f"{BINANCE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = await client.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

async def binance_price(client: httpx.AsyncClient, symbol: str) -> float:
    url = f"{BINANCE}/api/v3/ticker/price"
    params = {"symbol": symbol}
    r = await client.get(url, params=params, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])

# ------------------------ ORB & TRADINGLOGIK ------------------------

def is_green(open_: float, close_: float) -> bool:
    return close_ > open_

def is_red(open_: float, close_: float) -> bool:
    return close_ < open_

def usd_qty_to_units(usd: float, price: float) -> float:
    if price <= 0:
        return 0.0
    return round(usd / price, 6)

async def check_symbol(symbol: str, client: httpx.AsyncClient, app: Application):
    st = GS.by_symbol.setdefault(symbol, SymbolState())

    # Hämta de 3 senaste (så vi säkert har 2 stängda + 1 pågående)
    try:
        kl = await binance_klines(client, symbol, GS.timeframe, limit=3)
    except Exception as e:
        await send_owner(app, f"[{symbol}] Fel vid klines: {e}")
        return

    if len(kl) < 2:
        return

    # Senaste stängda är näst sist
    last_closed = kl[-2]
    (open_time_ms,
     o, h, l, c,
     _vol,
     close_time_ms,
     *_rest) = (
        last_closed[0],
        float(last_closed[1]),
        float(last_closed[2]),
        float(last_closed[3]),
        float(last_closed[4]),
        last_closed[5],
        last_closed[6],
        *last_closed[7:]
    )

    # 1) Bygg ORB: "första gröna efter röd"
    if st.last_closed_ts != close_time_ms:
        # ny stängd candle; uppdatera minnet
        prev_open = st.last_closed_open
        prev_close = st.last_closed_close

        if prev_open is not None and prev_close is not None:
            if is_red(prev_open, prev_close) and is_green(o, c) and st.orb is None and st.position is None:
                st.orb = ORB(high=h, low=l, ts_close=close_time_ms)
                await send_owner(app, f"🔷 [{symbol}] Ny ORB hittad (röd→grön). High={h:.4f}, Low={l:.4f}")

        st.last_closed_open = o
        st.last_closed_close = c
        st.last_closed_ts = close_time_ms

    # 2) Entry
    if st.position is None and st.orb:
        if GS.entry_mode == "close":
            # entry när en STÄNGD candle bryter över ORB.high
            if c > st.orb.high:
                entry = c
                qty = usd_qty_to_units(DEFAULT_QTY_USD, entry)
                stop = st.orb.low
                st.position = Position(entry=entry, qty=qty, stop=stop)
                await send_owner(app, f"✅ [{symbol}] LONG (close-break) entry={entry:.4f}, stop={stop:.4f} qty≈{qty}")
        else:
            # entry på tick-break (nuvarande pris)
            try:
                px = await binance_price(client, symbol)
            except Exception as e:
                await send_owner(app, f"[{symbol}] Fel vid tick: {e}")
                px = None
            if px and px > st.orb.high:
                entry = px
                qty = usd_qty_to_units(DEFAULT_QTY_USD, entry)
                stop = st.orb.low
                st.position = Position(entry=entry, qty=qty, stop=stop)
                await send_owner(app, f"✅ [{symbol}] LONG (tick-break) entry={entry:.4f}, stop={stop:.4f} qty≈{qty}")

    # 3) Trailing stop (på stängd candle) & BE-låsning
    if st.position:
        pos = st.position

        # Lås till BE efter liten vinst (på högsta priset i stängd candle)
        if not pos.locked_to_be and (float(h) >= pos.entry * (1.0 + LOCK_PROFIT_PCT)):
            pos.stop = max(pos.stop, pos.entry)  # lås minst till BE
            pos.locked_to_be = True
            await send_owner(app, f"🔒 [{symbol}] Stop låst till BE eller högre (nu {pos.stop:.4f})")

        # Traila till candle-low (endast uppåt, aldrig nedåt)
        if float(l) > pos.stop:
            pos.stop = float(l)
            await send_owner(app, f"📈 [{symbol}] Trailing stop uppdaterad: {pos.stop:.4f}")

        # 4) Stop-ut om tick <= stop
        try:
            px_now = await binance_price(client, symbol)
        except Exception:
            px_now = None

        if px_now and px_now <= pos.stop:
            exit_price = pos.stop
            pnl = (exit_price - pos.entry) * pos.qty
            await send_owner(app, f"❌ [{symbol}] Stop-out @ {exit_price:.4f} (PnL ≈ {pnl:.2f} USD)")
            st.position = None
            # börja om från nästa ORB (nollställ ORB så vi letar ny röd→grön)
            st.orb = None

# ------------------------ ENGINE LOOP ------------------------

async def engine_loop(app: Application):
    await send_owner(app, "🟢 Engine startad")
    async with httpx.AsyncClient() as client:
        while GS.engine_on:
            # Se till att symbol-listan i state har varsitt SymbolState
            for s in GS.symbols:
                GS.by_symbol.setdefault(s, SymbolState())

            tasks = [check_symbol(s, client, app) for s in GS.symbols]
            # Kör allt parallellt men svälj fel per symbol
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # (valfritt logga exceptions)
            await asyncio.sleep(POLL_SEC)
    await send_owner(app, "🔴 Engine stoppad")

def start_engine(app: Application):
    if GS.task and not GS.task.done():
        return
    GS.engine_on = True
    GS.task = asyncio.create_task(engine_loop(app))

def stop_engine():
    GS.engine_on = False
    if GS.task and not GS.task.done():
        GS.task.cancel()
    GS.task = None
    # Låt mock-positioner ligga kvar om man vill – Panic stänger dem.

def panic_flat():
    # stäng alla mock-positioner och nollställ ORB
    for s, st in GS.by_symbol.items():
        st.position = None
        st.orb = None

# ------------------------ TELEGRAM HANDLERS ------------------------

async def cmd_start(update: Update, context):
    if not only_owner(update):
        return
    await update.effective_message.reply_text(
        "Hej! ORB-bot online.\n\nAnvänd knapparna nedan eller /menu.",
        reply_markup=menu_keyboard()
    )

async def cmd_menu(update: Update, context):
    if not only_owner(update):
        return
    await update.effective_message.reply_text("Meny:", reply_markup=menu_keyboard())

async def cmd_status(update: Update, context):
    if not only_owner(update):
        return
    await update.effective_message.reply_text(fmt_state())

async def cmd_engine_on(update: Update, context):
    if not only_owner(update):
        return
    start_engine(context.application)
    await update.effective_message.reply_text("Engine: ON", reply_markup=menu_keyboard())

async def cmd_engine_off(update: Update, context):
    if not only_owner(update):
        return
    stop_engine()
    await update.effective_message.reply_text("Engine: OFF", reply_markup=menu_keyboard())

async def cmd_entrymode(update: Update, context):
    if not only_owner(update):
        return
    await update.effective_message.reply_text(
        f"Aktuellt: {GS.entry_mode.upper()}\nVälj nedan:",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("Close", callback_data="entry:close"),
            InlineKeyboardButton("Tick",  callback_data="entry:tick"),
        ]])
    )

async def cmd_timeframe(update: Update, context):
    if not only_owner(update):
        return
    await update.effective_message.reply_text(
        f"Aktuellt TF: {GS.timeframe}. Välj nytt:",
        reply_markup=timeframe_keyboard()
    )

async def cmd_panic(update: Update, context):
    if not only_owner(update):
        return
    stop_engine()
    panic_flat()
    await update.effective_message.reply_text("PANIC utförd: engine OFF + alla mock-positioner stängda.")

async def on_callback(update: Update, context):
    if not only_owner(update):
        return
    q = update.callback_query
    data = q.data or ""
    await q.answer()

    if data == "menu":
        await q.edit_message_text("Meny:", reply_markup=menu_keyboard())
        return

    if data == "engine:on":
        start_engine(context.application)
        await q.edit_message_text("Engine: ON", reply_markup=menu_keyboard())
        return

    if data == "engine:off":
        stop_engine()
        await q.edit_message_text("Engine: OFF", reply_markup=menu_keyboard())
        return

    if data.startswith("entry:"):
        mode = data.split(":")[1]
        if mode in ("close", "tick"):
            GS.entry_mode = mode
            await q.edit_message_text(f"Entry-mode satt till: {mode.upper()}", reply_markup=menu_keyboard())
        return

    if data == "tf:choose":
        await q.edit_message_text(f"Välj timeframe (nu {GS.timeframe}):", reply_markup=timeframe_keyboard())
        return

    if data.startswith("tf:set:"):
        tf = data.split(":")[2]
        if tf in VALID_TF:
            GS.timeframe = tf
            # Nollställ ORB per symbol så vi börjar rent med ny TF
            for st in GS.by_symbol.values():
                st.orb = None
                st.last_closed_ts = None
            await q.edit_message_text(f"Timeframe satt till {tf}.", reply_markup=menu_keyboard())
        else:
            await q.edit_message_text(f"Ogiltigt TF.", reply_markup=menu_keyboard())
        return

    if data == "status:show":
        await q.edit_message_text(fmt_state(), reply_markup=menu_keyboard())
        return

    if data == "panic:now":
        stop_engine()
        panic_flat()
        await q.edit_message_text("PANIC utförd: engine OFF + alla mock-positioner stängda.", reply_markup=menu_keyboard())
        return

    if data == "mode:noop":
        # Bara visning i denna version
        await q.answer(f"Läge: {'MOCK' if GS.mock else 'LIVE'}", show_alert=False)
        return

async def unknown(update: Update, context):
    if not only_owner(update):
        return
    await update.effective_message.reply_text("Okänt kommando. Testa /menu.")

# ------------------------ FASTAPI + WEBHOOK ------------------------

app = FastAPI()
tg_app: Optional[Application] = None

@app.on_event("startup")
async def on_startup():
    global tg_app
    tg_app = Application.builder().token(BOT_TOKEN).build()
    # Handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("menu", cmd_menu))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
    tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    tg_app.add_handler(CommandHandler("panic", cmd_panic))
    tg_app.add_handler(CallbackQueryHandler(on_callback))
    tg_app.add_handler(MessageHandler(filters.COMMAND, unknown))

    # Initiera & starta PTB
    await tg_app.initialize()
    await tg_app.start()

    # Sätt webhook
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    await tg_app.bot.set_webhook(url)
    # Bekräftelse till ägaren
    try:
        await tg_app.bot.send_message(
            OWNER_CHAT_ID,
            f"✅ Startad på {WEBHOOK_BASE}\nMode: {'MOCK' if GS.mock else 'LIVE'}\nTF: {GS.timeframe}\nEntry: {GS.entry_mode}\nSymbols: {', '.join(GS.symbols)}"
        )
    except Exception:
        pass

@app.on_event("shutdown")
async def on_shutdown():
    global tg_app
    stop_engine()
    if tg_app:
        try:
            await tg_app.bot.delete_webhook()
        except Exception:
            pass
        await tg_app.stop()
        await tg_app.shutdown()

@app.get("/")
async def root():
    return {"ok": True, "ts": now_iso(), "engine": GS.engine_on, "tf": GS.timeframe, "entry": GS.entry_mode}

@app.post(f"/webhook/{{token}}")
async def webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return PlainTextResponse("forbidden", status_code=403)
    data = await request.json()
    # PTB v20: de_json med bot
    update = Update.de_json(data, tg_app.bot)  # type: ignore
    await tg_app.process_update(update)        # type: ignore
    return JSONResponse({"ok": True})
