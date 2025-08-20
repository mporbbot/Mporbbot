import os
import time
import asyncio
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Literal, List

import csv
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI

# =========================
# KONFIG
# =========================
TF = os.getenv("TF", "1min")  # 1min, 3min, 5min ...
SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT").split(",")
KUCOIN_API = "https://api.kucoin.com"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

KUCOIN_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

# Position size per symbol (spot)
DEFAULT_QTY = {
    "BTC-USDT": 0.00025,
    "ETH-USDT": 0.007,
    "ADA-USDT": 100.0,
    "LINK-USDT": 1.2,
    "XRP-USDT": 30.0,
}

Side = Literal["buy", "sell"]
EntryMode = Literal["tick", "close"]

# =========================
# BROKER-IF
# =========================
class IBroker:
    async def market(self, symbol: str, side: Side, qty: float) -> str:
        raise NotImplementedError

class MockBroker(IBroker):
    async def market(self, symbol: str, side: Side, qty: float) -> str:
        return f"mock-{symbol}-{side}-{time.time_ns()}"

class KucoinBroker(IBroker):
    def __init__(self, key: str, secret: str, passphrase: str):
        from kucoin.client import Client
        self.c = Client(key, secret, passphrase)

    async def market(self, symbol: str, side: Side, qty: float) -> str:
        sym = symbol.replace("-", "")
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: self.c.create_market_order(sym, side, size=str(qty))
        )
        return str(resp.get("orderId", ""))

# =========================
# DATASTRUKTURER
# =========================
@dataclass
class Candle:
    t: int
    o: float
    h: float
    l: float
    c: float

@dataclass
class SymState:
    # ORB
    orb_set: bool = False
    orb_high: float = 0.0
    orb_low: float = 0.0
    # Position
    in_pos: bool = False
    qty: float = 0.0
    entry_px: float = 0.0
    stop: float = 0.0
    # Entry-logik
    entry_mode: EntryMode = "tick"

class EngineState:
    def __init__(self):
        self.engine_on: bool = False
        self.mode: Literal["mock", "live"] = "mock"
        self.symbols: Dict[str, SymState] = {s: SymState() for s in SYMBOLS}
        self.broker: IBroker = MockBroker()
        self.chat_id: Optional[int] = None
        self.qty: Dict[str, float] = DEFAULT_QTY.copy()

STATE = EngineState()

# Trade log (för K4-export)
TRADE_LOG: List[Dict] = []  # varje rad: dict

# =========================
# HJÄLPARE
# =========================
def fmt(v: float) -> str:
    try:
        return f"{v:,.4f}"
    except Exception:
        return str(v)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

async def fetch_klines(symbol: str, count: int = 3) -> List[Candle]:
    url = f"{KUCOIN_API}/api/v1/market/candles"
    params = {"type": TF, "symbol": symbol}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])

    cs: List[Candle] = []
    for row in data[: max(50, count + 2)]:
        t, o, c, h, l = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        cs.append(Candle(t=t, o=o, h=h, l=l, c=c))
    cs = list(reversed(cs))  # stigande tid

    # exkludera pågående sista candle
    if len(cs) >= 2:
        closed = cs[:-1]
    else:
        closed = cs
    return closed[-count:] if len(closed) >= count else closed

def is_green(c: Candle) -> bool:
    return c.c > c.o

def is_red(c: Candle) -> bool:
    return c.c < c.o

# =========================
# TELEGRAM
# =========================
TG_APP = None
TG_STARTED = False

from telegram import Update, BotCommand, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

def ok(b: bool) -> str:
    return "✅" if b else "❌"

async def tg_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    any_sym = next(iter(STATE.symbols.values()))
    lines = [
        f"Mode: {STATE.mode}   Engine: {ok(STATE.engine_on)}   TF: {TF}   Entry={any_sym.entry_mode.upper()}",
        "ORB: Första GRÖNA candle efter en RÖD",
        "Stop: Trail till SENASTE candle-LOW (long), flyttas aldrig ner."
    ]
    for s, st in STATE.symbols.items():
        orb = f"[{fmt(st.orb_low)} - {fmt(st.orb_high)}]" if st.orb_set else "-"
        stop = fmt(st.stop) if st.in_pos else "-"
        lines.append(f"{s}: pos={ok(st.in_pos)} stop={stop} ORB={orb}")
    await update.message.reply_text("\n".join(lines))

async def tg_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = True
    await update.message.reply_text("Engine: ON")

async def tg_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = False
    await update.message.reply_text("Engine: OFF")

async def tg_mode_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.mode = "mock"
    STATE.broker = MockBroker()
    await update.message.reply_text("Mode: MOCK (orders simuleras)")

async def tg_mode_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
        await update.message.reply_text("LIVE misslyckades: API-nycklar saknas i env.")
        return
    STATE.mode = "live"
    STATE.broker = KucoinBroker(KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE)
    await update.message.reply_text("Mode: LIVE (KuCoin spot)")

async def tg_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for st in STATE.symbols.values():
        st.entry_mode = "close" if st.entry_mode == "tick" else "tick"
    await update.message.reply_text(f"Entry-mode: {next(iter(STATE.symbols.values())).entry_mode}")

async def tg_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # stäng alla positioner + logga
    for sym, st in STATE.symbols.items():
        if st.in_pos and st.qty > 0:
            live = await get_live_price(sym)
            try:
                oid = await STATE.broker.market(sym, "sell", st.qty)
            except Exception as e:
                await update.message.reply_text(f"{sym} PANIC misslyckades: {e}")
                continue
            pnl = (live - st.entry_px) * st.qty
            log_trade(sym, st.qty, sell_px=live, buy_px=st.entry_px, reason="PANIC", order_id=oid)
            await tg_notify(f"🛑 {sym} PANIC SELL @{fmt(live)} pnl≈{fmt(pnl)} oid={oid}")
            st.in_pos = False
            st.qty = 0.0
            st.stop = 0.0
            st.entry_px = 0.0
    await update.message.reply_text("PANIC: Alla positioner stängda/loggade.")

async def tg_k4export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Skriv CSV och skicka
    if not TRADE_LOG:
        await update.message.reply_text("Inga trades att exportera ännu.")
        return
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"/tmp/k4_{ts}.csv"
    write_k4_csv(path, TRADE_LOG)
    try:
        await TG_APP.bot.send_document(
            chat_id=update.effective_chat.id,
            document=InputFile(path),
            caption=f"K4-export {ts} (Övriga tillgångar)"
        )
    except Exception as e:
        await update.message.reply_text(f"Kunde inte skicka filen: {e}")

def start_telegram_once():
    global TG_APP, TG_STARTED
    if TG_STARTED or not TELEGRAM_BOT_TOKEN:
        return
    TG_STARTED = True

    async def _run():
        global TG_APP
        TG_APP = (
            ApplicationBuilder()
            .token(TELEGRAM_BOT_TOKEN)
            .concurrent_updates(True)
            .build()
        )
        TG_APP.add_handler(CommandHandler("status", tg_status))
        TG_APP.add_handler(CommandHandler("engine_on", tg_engine_on))
        TG_APP.add_handler(CommandHandler("engine_off", tg_engine_off))
        TG_APP.add_handler(CommandHandler("mock", tg_mode_mock))
        TG_APP.add_handler(CommandHandler("live", tg_mode_live))
        TG_APP.add_handler(CommandHandler("entrymode", tg_entry_mode))
        TG_APP.add_handler(CommandHandler("panic", tg_panic))
        TG_APP.add_handler(CommandHandler("k4export", tg_k4export))

        await TG_APP.bot.set_my_commands(
            [
                BotCommand("status", "Visa status (sparar chat_id)"),
                BotCommand("engine_on", "Starta motorn"),
                BotCommand("engine_off", "Stoppa motorn"),
                BotCommand("mock", "Mock-läge"),
                BotCommand("live", "Live-läge (KuCoin)"),
                BotCommand("entrymode", "Växla entry: tick/close"),
                BotCommand("panic", "Stäng alla positioner"),
                BotCommand("k4export", "Skicka K4-CSV för loggade trades"),
            ]
        )

        await TG_APP.initialize()
        await TG_APP.start()
        await TG_APP.updater.start_polling(drop_pending_updates=True)
        await TG_APP.updater.wait()

    def _bg():
        asyncio.run(_run())

    t = threading.Thread(target=_bg, daemon=True)
    t.start()

async def tg_notify(text: str):
    if TELEGRAM_BOT_TOKEN and STATE.chat_id:
        try:
            await TG_APP.bot.send_message(chat_id=STATE.chat_id, text=text)
        except Exception:
            pass

# =========================
# ORB/ENTRY/STOP/LOGG
# =========================
def log_trade(symbol: str, qty: float, sell_px: float, buy_px: float, reason: str, order_id: str):
    proceeds = sell_px * qty
    cost = buy_px * qty
    pnl = proceeds - cost
    TRADE_LOG.append(
        {
            "timestamp_utc": utc_now_iso(),
            "symbol": symbol,
            "qty": qty,
            "buy_px": round(buy_px, 8),
            "sell_px": round(sell_px, 8),
            "proceeds": round(proceeds, 8),
            "cost": round(cost, 8),
            "pnl": round(pnl, 8),
            "reason": reason,
            "order_id": order_id,
            "mode": STATE.mode,
        }
    )

def write_k4_csv(path: str, rows: List[Dict]):
    # Enkel K4-CSV för Avsnitt D (Övriga tillgångar)
    # Kolumner: Datum, Tillgång, Antal, Försäljningspris, Omkostnadsbelopp, Vinst/Förlust, Orsak, OrderId, Mode
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Datum (UTC)", "Tillgång", "Antal", "Försäljningspris (SEK/USD?)",
                    "Omkostnadsbelopp", "Vinst/Förlust", "Orsak", "OrderId", "Mode"])
        for r in rows:
            w.writerow([
                r["timestamp_utc"],
                r["symbol"],
                r["qty"],
                r["proceeds"],
                r["cost"],
                r["pnl"],
                r["reason"],
                r["order_id"],
                r["mode"],
            ])

async def compute_orb(st: SymState, closed: List[Candle]) -> None:
    if len(closed) < 2:
        return
    for i in range(len(closed) - 1):
        c_prev = closed[i]
        c_cur = closed[i + 1]
        if is_red(c_prev) and is_green(c_cur):
            st.orb_high = c_cur.h
            st.orb_low = c_cur.l
            st.orb_set = True
    # om inget nytt par hittas, behåll tidigare ORB

async def maybe_enter(symbol: str, st: SymState, closed: List[Candle], live_price: float):
    if st.in_pos or not st.orb_set:
        return

    should_enter = False
    reason = ""
    if st.entry_mode == "tick":
        if live_price > st.orb_high:
            should_enter = True
            reason = "TickBreak"
    else:
        if len(closed) >= 1 and closed[-1].c > st.orb_high:
            should_enter = True
            reason = "CloseBreak"

    if not should_enter:
        return

    qty = STATE.qty.get(symbol, 0.0)
    try:
        oid = await STATE.broker.market(symbol, "buy", qty)
    except Exception as e:
        await tg_notify(f"❌ {symbol} BUY misslyckades: {e}")
        return

    st.in_pos = True
    st.qty = qty
    st.entry_px = live_price
    st.stop = max(st.stop, closed[-1].l) if st.stop else closed[-1].l
    await tg_notify(f"✅ {symbol} ENTRY LONG @{fmt(live_price)} (qty={qty}) reason={reason} ORB=[{fmt(st.orb_low)}-{fmt(st.orb_high)}] oid={oid}")

async def maybe_trail_stop(symbol: str, st: SymState, closed: List[Candle], live_price: float):
    if not st.in_pos:
        return

    # trail uppåt till senaste candle-low
    if len(closed) >= 1:
        new_low = closed[-1].l
        if new_low > st.stop:
            st.stop = new_low
            await tg_notify(f"↗️ {symbol} STOP trailad till {fmt(st.stop)}")

    # stop-trigger
    if live_price <= st.stop:
        try:
            oid = await STATE.broker.market(symbol, "sell", st.qty)
        except Exception as e:
            await tg_notify(f"❌ {symbol} STOP SELL misslyckades: {e}")
            return
        pnl = (live_price - st.entry_px) * st.qty
        log_trade(symbol, st.qty, sell_px=live_price, buy_px=st.entry_px, reason="STOP", order_id=oid)
        await tg_notify(f"🛑 {symbol} STOP HIT @{fmt(live_price)} pnl≈{fmt(pnl)} oid={oid}")
        st.in_pos = False
        st.qty = 0.0
        st.stop = 0.0
        st.entry_px = 0.0

# =========================
# DATASLINGA
# =========================
async def get_live_price(symbol: str) -> float:
    c = await fetch_klines(symbol, count=1)
    return c[-1].c if c else 0.0

async def symbol_loop(symbol: str):
    st = STATE.symbols[symbol]
    while True:
        try:
            closed = await fetch_klines(symbol, count=3)
            if not closed:
                await asyncio.sleep(1.0)
                continue

            await compute_orb(st, closed)
            live = closed[-1].c

            if STATE.engine_on:
                await maybe_enter(symbol, st, closed, live)
                await maybe_trail_stop(symbol, st, closed, live)

        except Exception:
            pass

        await asyncio.sleep(3.0)

async def engine_loop():
    tasks = [asyncio.create_task(symbol_loop(s)) for s in SYMBOLS]
    await asyncio.gather(*tasks)

# =========================
# FASTAPI
# =========================
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    start_telegram_once()
    asyncio.create_task(engine_loop())

@app.get("/")
async def root():
    return {
        "ok": True,
        "mode": STATE.mode,
        "engine_on": STATE.engine_on,
        "tf": TF,
        "symbols": SYMBOLS,
        "entry_mode": next(iter(STATE.symbols.values())).entry_mode if STATE.symbols else "tick",
        "trades": len(TRADE_LOG),
    }
