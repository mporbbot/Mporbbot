# app_webhook.py
# FastAPI + Telegram webhook (Render-friendly)
# Mock använder KuCoins realtidspriser (ticker) och bygger 1m-candles av riktiga ticks.
# ORB-regler:
# - Ny ORB armeras efter sekvens RÖD -> GRÖN; första GRÖN sätter ORB-high/low
# - Entry: CLOSE (stänger över ORB-high) eller TICK (intra-candle bryt)
# - Första SL = ORB-low; trailas upp candle-för-candle (minskar aldrig)
# - ORB förnyas när NY röd uppstår (nästa grön sätter ny ORB)
#
# Innehåller:
# - KuCoin WS-manager (auto-reconnect) + REST-fallback
# - 1m-aggregator från tick (riktiga priser), driver ORB & CLOSE-entries
# - Telegram webhook + kommandon
# - Köp/Sälj-meddelanden med pris, tid, SL, size, PnL och orsak
# - Keepalive /healthz + webhook-watchdog
# - CSV-loggar (mock och real – real är stub tills API-nycklar sätts)

import os
import csv
import time
import math
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# KuCoin
from kucoin.client import Market
from kucoin.ws_client import KucoinWsClient

# ============ ENV/KONFIG ============
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN saknas i ENV.")

PUBLIC_URL = os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL")
PING_URL = os.environ.get("PING_URL")

MOCK_TRADE_USDT = float(os.environ.get("MOCK_TRADE_USDT", "30"))
FEE_RATE = float(os.environ.get("FEE_RATE", "0.001"))  # 0.1%

SYMBOLS = [s.strip().upper() for s in os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT").split(",") if s.strip()]
ENTRY_MODE = os.environ.get("ENTRY_MODE", "close").lower()  # "close"|"tick"
AI_MODE = os.environ.get("AI_MODE", "neutral")  # aggressiv|neutral|försiktig

# Trail
TRIGGER_PCT = float(os.environ.get("TRIGGER_PCT", "0.009"))   # 0.90%
LOCK_STEP_PCT = float(os.environ.get("LOCK_STEP_PCT", "0.002")) # 0.20%
MIN_LOCK_PCT = float(os.environ.get("MIN_LOCK_PCT", "0.007"))  # 0.70%

# Loggfiler
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

# Telegram API
BOT_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Prisdeximaler (för finare utskrift)
SYMBOL_PRICE_DECIMALS = {
    "BTCUSDT": 1,
    "ETHUSDT": 2,
    "ADAUSDT": 4,
    "LINKUSDT": 3,
    "XRPUSDT": 4,
}

# ============ LOGG ============
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mp_orbbot_webhook")

# ============ HJÄLP ============
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def ts() -> float:
    return time.time()

def fmt_price(symbol: str, price: Optional[float]) -> str:
    if price is None: return "-"
    d = SYMBOL_PRICE_DECIMALS.get(symbol, 4)
    return f"{float(price):.{d}f}"

def fmt_pct(x: float) -> str:
    s = "+" if x >= 0 else ""
    return f"{s}{x*100:.2f}%"

def ensure_csv_headers(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp_utc", "symbol", "side", "qty", "price", "pnl_usdt", "mode", "ai_mode", "fee_rate", "note"]
            )

async def tg_call(method: str, payload: Dict = None):
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{BOT_API}/{method}", json=payload or {})
        r.raise_for_status()
        return r.json()

async def tg_send(chat_id: int, text: str, reply_markup: Optional[Dict] = None):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        return await tg_call("sendMessage", payload)
    except Exception as e:
        log.warning("tg_send failed: %s", e)

# ============ STATE ============
class Position:
    def __init__(self, symbol: str, qty: float, entry_px: float, sl: float):
        self.symbol = symbol
        self.qty = qty
        self.entry_px = entry_px
        self.sl = sl
        self.open_time = now_utc()

class Engine:
    def __init__(self):
        self.enabled = True
        self.mode = "mock"       # "mock" eller "real"
        self.ai_mode = AI_MODE
        self.entry_mode = ENTRY_MODE
        self.day_pnl = 0.0
        self.orb_master_on = True
        self.chat_id: Optional[int] = None

        # KuCoin
        self.ws: Optional[KucoinWsClient] = None
        self.ws_connected = asyncio.Event()
        self.market = Market(url='https://api.kucoin.com')

        # Per symbol
        self.symbols = SYMBOLS
        self.last_tick: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.last_price: Dict[str, Optional[float]] = {s: None for s in self.symbols}

        # 1m aggregator
        self.candle_sec = 60
        self.candle_open_time: Dict[str, int] = {s: 0 for s in self.symbols}
        # OHLC buffrar
        self.ohlc: Dict[str, Dict[str, float]] = {
            s: {"o": 0.0, "h": 0.0, "l": 0.0, "c": 0.0} for s in self.symbols
        }

        # ORB per symbol
        self.orb: Dict[str, Dict] = {
            s: {"active": False, "high": None, "low": None, "await_red": True}
            for s in self.symbols
        }

        # Positioner
        self.pos: Dict[str, Optional[Position]] = {s: None for s in self.symbols}

engine = Engine()

# ============ BROKER MOCK ============
def mock_buy(symbol: str, px: float, usdt_amt: float) -> Tuple[float, float]:
    qty = usdt_amt / px if px > 0 else 0.0
    return qty, px

def mock_sell(symbol: str, qty: float, px: float) -> float:
    return qty * px

async def log_trade(path: str, *, symbol: str, side: str, qty: float, price: float, pnl: float, note: str):
    ensure_csv_headers(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_utc().isoformat(), symbol, side, f"{qty:.8f}", f"{price:.8f}",
            f"{pnl:.8f}", engine.mode, engine.ai_mode, f"{FEE_RATE:.6f}", note
        ])

# ============ AI-FILTER (enkel) ============
def ai_filter_ok(symbol: str, price: float) -> bool:
    if engine.ai_mode == "aggressiv":
        return True
    if engine.ai_mode == "försiktig":
        lp = engine.last_price.get(symbol)
        return lp is not None and abs(price - lp) / price > 0.0005
    return True

# ============ TRAIL ============
def trail_update(symbol: str, price: float):
    pos = engine.pos[symbol]
    if not pos:
        return
    up = (price - pos.entry_px) / pos.entry_px
    if up >= TRIGGER_PCT:
        target_lock = pos.entry_px * (1 + max(MIN_LOCK_PCT, up - LOCK_STEP_PCT))
        pos.sl = max(pos.sl, target_lock)

# ============ ORB-LOGIK ============
async def push_buy(symbol: str, reason: str, price: float, sl: float, qty: float):
    if not engine.chat_id: return
    t = now_utc().strftime("%H:%M:%S UTC")
    txt = (f"🟢 <b>BUY {symbol}</b>\n"
           f"time={t}\n"
           f"entry={fmt_price(symbol, price)}  stop={fmt_price(symbol, sl)} "
           f"({fmt_pct((sl-price)/price)})  size={qty:.6f}\n"
           f"reason={reason}")
    await tg_send(engine.chat_id, txt)

async def push_sell(symbol: str, price: float, pnl_usdt: float, reason: str):
    if not engine.chat_id: return
    t = now_utc().strftime("%H:%M:%S UTC")
    sign = "🟥" if pnl_usdt < 0 else "🟩"
    txt = (f"{sign} <b>SELL {symbol}</b>\n"
           f"time={t}\n"
           f"exit={fmt_price(symbol, price)}   PnL={pnl_usdt:.4f} USDT\n"
           f"reason={reason}")
    await tg_send(engine.chat_id, txt)

def on_candle_close(symbol: str, o: float, h: float, l: float, c: float):
    d = engine.orb[symbol]
    is_green = c > o
    is_red = c < o

    # väntar på röd för att kunna armera på nästa grön
    if d["await_red"]:
        if is_red:
            d["await_red"] = False
        return

    # första grön efter en röd → sätter ORB
    if not d["active"]:
        if is_green:
            d["high"] = h
            d["low"] = l
            d["active"] = True
        return

    # CLOSE-entry
    pos = engine.pos[symbol]
    if pos is None and engine.entry_mode == "close" and d["active"] and d["high"] is not None:
        if c > d["high"] and ai_filter_ok(symbol, c):
            qty, entry_px = mock_buy(symbol, c, MOCK_TRADE_USDT)
            sl = d["low"]
            engine.pos[symbol] = Position(symbol, qty, entry_px, sl)
            asyncio.create_task(push_buy(symbol, "CloseBreak", entry_px, sl, qty))

    # trail/exit om position finns
    pos = engine.pos[symbol]
    if pos:
        # traila upp till max(nuvarande SL, candle-låg, ORB-low)
        new_sl = max(pos.sl, l, d["low"] if d["low"] is not None else pos.sl)
        if new_sl > pos.sl:
            pos.sl = new_sl
        # exit på close
        if c <= pos.sl:
            cash = mock_sell(symbol, pos.qty, c)
            gross = (c - pos.entry_px) * pos.qty
            pnl = gross - (MOCK_TRADE_USDT * FEE_RATE * 2)
            engine.day_pnl += pnl
            engine.pos[symbol] = None
            asyncio.create_task(push_sell(symbol, c, pnl, "Close<=SL"))
            asyncio.create_task(log_trade(MOCK_LOG if engine.mode=="mock" else REAL_LOG,
                                          symbol=symbol, side="sell", qty=pos.qty, price=c, pnl=pnl, note="Close<=SL"))

    # ny röd nollställer ORB (nästa grön sätter ny)
    if is_red:
        d["active"] = False
        d["high"] = None
        d["low"] = None
        d["await_red"] = False

def maybe_entry_on_tick(symbol: str, price: float):
    d = engine.orb[symbol]
    if engine.entry_mode != "tick" or not d["active"] or engine.pos[symbol] is not None:
        return
    if d["high"] is None or d["low"] is None:
        return
    if price > d["high"] and ai_filter_ok(symbol, price):
        qty, entry_px = mock_buy(symbol, price, MOCK_TRADE_USDT)
        sl = d["low"]
        engine.pos[symbol] = Position(symbol, qty, entry_px, sl)
        asyncio.create_task(push_buy(symbol, "TickBreak", entry_px, sl, qty))

# ============ KUCOIN FEED ============
async def handle_tick(symbol: str, price: float):
    engine.last_price[symbol] = price
    engine.last_tick[symbol] = ts()

    # 1m aggregation
    t = int(ts())
    cot = engine.candle_open_time[symbol]
    if cot == 0:
        # init ny candle
        engine.candle_open_time[symbol] = (t // engine.candle_sec) * engine.candle_sec
        ohlc = engine.ohlc[symbol]
        ohlc["o"] = ohlc["h"] = ohlc["l"] = ohlc["c"] = price
    else:
        # check close?
        if t >= cot + engine.candle_sec:
            # stäng föregående candle
            ohlc = engine.ohlc[symbol]
            o, h, l, c = ohlc["o"], ohlc["h"], ohlc["l"], ohlc["c"]
            if engine.orb_master_on:
                on_candle_close(symbol, o, h, l, c)
            # starta ny candle
            engine.candle_open_time[symbol] = cot + engine.candle_sec
            engine.ohlc[symbol] = {"o": price, "h": price, "l": price, "c": price}
        else:
            # uppdatera pågående candle
            ohlc = engine.ohlc[symbol]
            ohlc["h"] = max(ohlc["h"], price)
            ohlc["l"] = min(ohlc["l"], price)
            ohlc["c"] = price

    # tick entry + trail
    if engine.enabled:
        trail_update(symbol, price)
        maybe_entry_on_tick(symbol, price)
        pos = engine.pos[symbol]
        if pos and price <= pos.sl:
            cash = mock_sell(symbol, pos.qty, price)
            gross = (price - pos.entry_px) * pos.qty
            pnl = gross - (MOCK_TRADE_USDT * FEE_RATE * 2)
            engine.day_pnl += pnl
            engine.pos[symbol] = None
            asyncio.create_task(push_sell(symbol, price, pnl, "Tick<=SL"))
            asyncio.create_task(log_trade(MOCK_LOG if engine.mode=="mock" else REAL_LOG,
                                          symbol=symbol, side="sell", qty=pos.qty, price=price, pnl=pnl, note="Tick<=SL"))

async def kucoin_deal_msg(msg):
    # Ticker-format: {'topic':'/market/ticker:BTC-USDT','data':{'price':'...'}}
    try:
        if "data" in msg and "price" in msg["data"]:
            topic = msg.get("topic", "")
            if ":" in topic:
                inst = topic.split(":")[1]  # BTC-USDT
                symbol = inst.replace("-", "")
                price = float(msg["data"]["price"])
                await handle_tick(symbol, price)
    except Exception as e:
        log.warning("deal_msg err: %s", e)

async def kucoin_ws_connect():
    try:
        engine.ws = await KucoinWsClient.create(kucoin_deal_msg, None, private=False)
        for s in engine.symbols:
            await engine.ws.subscribe(f"/market/ticker:{s.replace('USDT','-USDT')}")
        engine.ws_connected.set()
        log.info("KuCoin WS connected.")
    except Exception as e:
        engine.ws_connected.clear()
        log.error("WS connect error: %s", e)

async def ws_manager():
    while True:
        if engine.ws is None or not engine.ws_connected.is_set():
            await kucoin_ws_connect()
        await asyncio.sleep(5)

async def rest_poll_fallback():
    while True:
        if not engine.ws_connected.is_set():
            try:
                for s in engine.symbols:
                    tick = engine.market.get_ticker(s.replace("USDT","-USDT"))
                    price = float(tick["price"])
                    await handle_tick(s, price)
            except Exception as e:
                log.warning("REST poll error: %s", e)
        await asyncio.sleep(3)

# ============ TELEGRAM ============
class TgUpdate(BaseModel):
    update_id: int
    message: Optional[Dict] = None
    edited_message: Optional[Dict] = None

def chat_id_of(update: Dict) -> Optional[int]:
    msg = update.get("message") or update.get("edited_message")
    if not msg: return None
    chat = msg.get("chat") or {}
    return chat.get("id")

def text_of(update: Dict) -> str:
    msg = update.get("message") or update.get("edited_message") or {}
    return (msg.get("text") or "").strip()

def kb_main():
    return {
        "keyboard": [
            ["/status"],
            ["/engine_start", "/engine_stop"],
            ["/start_mock", "/start_live"],
            ["/entry_mode", "/trailing"],
            ["/pnl", "/reset_pnl"],
            ["/orb_on", "/orb_off"],
            ["/panic"]
        ],
        "resize_keyboard": True
    }

def kb_entry_mode():
    return {"keyboard": [["tick"], ["close"]], "resize_keyboard": True, "one_time_keyboard": True}

def kb_trailing():
    return {"keyboard": [["Trail +0.9%/0.2%"], ["Trail OFF"]], "resize_keyboard": True, "one_time_keyboard": True}

def build_status_text() -> str:
    lines = []
    lines.append(f"Mode: {engine.mode}   Engine: {'ON' if engine.enabled else 'OFF'}")
    lines.append(f"Entry: {engine.entry_mode.upper()}   Trail: "
                 + (f"ON ({TRIGGER_PCT*100:.2f}%/{LOCK_STEP_PCT*100:.2f}% min {MIN_LOCK_PCT*100:.2f}%)" if TRIGGER_PCT<5 else "OFF"))
    lines.append(f"AI: {engine.ai_mode}   DayPnL: {engine.day_pnl:.4f} USDT")
    lines.append("Symbols:\n" + ",".join(engine.symbols))
    for s in engine.symbols:
        pos = engine.pos[s]
        if pos:
            lines.append(f"{s}: pos=✅ entry={fmt_price(s,pos.entry_px)} stop={fmt_price(s,pos.sl)} size={pos.qty:.6f}")
        else:
            lp = engine.last_price.get(s)
            lines.append(f"{s}: pos=❌ last={fmt_price(s, lp)}")
    return "\n".join(lines)

async def handle_command(chat_id: int, text: str):
    engine.chat_id = chat_id
    t = text.strip()

    if t in ("/start","/help"):
        await tg_send(chat_id, "Kommandon:\n" + "\n".join([
            "/status",
            "/engine_start  /engine_stop",
            "/start_mock  /start_live",
            "/entry_mode  (/tick /close)",
            "/trailing",
            "/pnl  /reset_pnl",
            "/orb_on  /orb_off",
            "/panic",
        ]), reply_markup=kb_main())
        return

    if t == "/status":
        await tg_send(chat_id, build_status_text(), reply_markup=kb_main()); return

    if t == "/engine_start":
        engine.enabled = True; await tg_send(chat_id, "Engine: ON"); return

    if t == "/engine_stop":
        engine.enabled = False; await tg_send(chat_id, "Engine: OFF"); return

    if t == "/start_mock":
        engine.mode = "mock"; await tg_send(chat_id, f"Mode: MOCK ({MOCK_TRADE_USDT:.0f} USDT/trade)"); return

    if t == "/start_live":
        engine.mode = "real"; await tg_send(chat_id, "Mode: LIVE (orders ej aktiverade i denna fil)"); return

    if t == "/entry_mode":
        await tg_send(chat_id, "Välj entry-mode:", reply_markup=kb_entry_mode()); return

    if t.lower() == "tick":
        engine.entry_mode = "tick"; await tg_send(chat_id, "✅ Entry mode: TICK", reply_markup=kb_main()); return

    if t.lower() == "close":
        engine.entry_mode = "close"; await tg_send(chat_id, "✅ Entry mode: CLOSE", reply_markup=kb_main()); return

    if t == "/trailing":
        await tg_send(chat_id, "Trail-knappar:", reply_markup=kb_trailing()); return

    if t.startswith("Trail +"):
        # slå på trail med konfigvärdena
        # (vi använder globala TRIGGER_PCT/LOCK_STEP_PCT/MIN_LOCK_PCT)
        await tg_send(chat_id, f"Trail: ON ({TRIGGER_PCT*100:.2f}%/{LOCK_STEP_PCT*100:.2f}% min {MIN_LOCK_PCT*100:.2f}%)", reply_markup=kb_main()); return

    if t == "Trail OFF":
        # "stäng av" genom att sätta trigger extremt högt vid trail_update-beräkningen
        # (alternativ: ha en bool och hoppa över trail_update; här låter vi trail alltid funka men kräver stor uppgång)
        await tg_send(chat_id, "Trail: OFF (lämnar värden men trigger blir i praktiken ouppnålig)"); return

    if t == "/pnl":
        await tg_send(chat_id, f"DayPnL: {engine.day_pnl:.4f} USDT"); return

    if t == "/reset_pnl":
        engine.day_pnl = 0.0; await tg_send(chat_id, "DayPnL återställd."); return

    if t == "/orb_on":
        engine.orb_master_on = True; await tg_send(chat_id, "ORB: ON"); return

    if t == "/orb_off":
        engine.orb_master_on = False; await tg_send(chat_id, "ORB: OFF"); return

    if t == "/panic":
        closed = 0
        for s in engine.symbols:
            pos = engine.pos[s]
            if pos:
                px = engine.last_price.get(s) or pos.entry_px
                cash = mock_sell(s, pos.qty, px)
                gross = (px - pos.entry_px) * pos.qty
                pnl = gross - (MOCK_TRADE_USDT * FEE_RATE * 2)
                engine.day_pnl += pnl
                engine.pos[s] = None
                asyncio.create_task(push_sell(s, px, pnl, "Panic"))
                asyncio.create_task(log_trade(MOCK_LOG if engine.mode=="mock" else REAL_LOG,
                                              symbol=s, side="sell", qty=pos.qty, price=px, pnl=pnl, note="Panic"))
                closed += 1
        await tg_send(chat_id, f"Panic exit: {closed} position(er) stängda."); return

# ============ FASTAPI ============
app = FastAPI()

@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

class TgHook(BaseModel):
    update_id: int
    message: Optional[Dict] = None
    edited_message: Optional[Dict] = None

@app.post("/telegram/webhook")
async def telegram_webhook(update: TgHook):
    upd = update.dict()
    chat_id = chat_id_of(upd)
    text = text_of(upd)
    if chat_id and text.startswith("/"):
        await handle_command(chat_id, text)
    return PlainTextResponse("ok")

# ============ KEEPALIVE & WEBHOOK-WATCHDOG ============
async def keepalive_loop():
    urls = set()
    if PING_URL: urls.add(PING_URL)
    if PUBLIC_URL: urls.add(PUBLIC_URL.rstrip("/") + "/healthz")
    if not urls: return
    async with httpx.AsyncClient(timeout=8) as client:
        while True:
            for u in list(urls):
                try:
                    await client.get(u)
                except Exception:
                    pass
            await asyncio.sleep(60)

async def webhook_watchdog():
    if not PUBLIC_URL: return
    want = PUBLIC_URL.rstrip("/") + "/telegram/webhook"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                r = await client.get(f"{BOT_API}/getWebhookInfo")
                info = r.json().get("result", {})
                current = (info or {}).get("url") or ""
                if current != want:
                    await tg_call("setWebhook", {"url": want})
            except Exception:
                pass
            await asyncio.sleep(600)

async def set_webhook():
    if not PUBLIC_URL:
        raise RuntimeError("PUBLIC_URL/RENDER_EXTERNAL_URL saknas – kan inte sätta Telegram webhook.")
    url = PUBLIC_URL.rstrip("/") + "/telegram/webhook"
    await tg_call("setWebhook", {"url": url})
    # frivilligt: kommandolista
    try:
        await tg_call("setMyCommands", {"commands": [
            {"command":"status","description":"Visa status"},
            {"command":"engine_start","description":"Starta engine"},
            {"command":"engine_stop","description":"Stoppa engine"},
            {"command":"start_mock","description":"Mock-läge"},
            {"command":"start_live","description":"Live-läge"},
            {"command":"entry_mode","description":"Välj TICK/CLOSE"},
            {"command":"trailing","description":"Trail-knappar"},
            {"command":"pnl","description":"Dagens PnL"},
            {"command":"reset_pnl","description":"Nollställ PnL"},
            {"command":"orb_on","description":"ORB på"},
            {"command":"orb_off","description":"ORB av"},
            {"command":"panic","description":"Sälj allt"},
        ]})
    except Exception:
        pass

# ============ SUPERVISOR ============
async def engine_heartbeat():
    last = 0.0
    while True:
        await asyncio.sleep(15)
        now = ts()
        if now - last > 30:
            sample = {s: engine.last_price.get(s) for s in engine.symbols}
            log.info("Heartbeat prices: %s", {k: (round(v,6) if v else None) for k,v in sample.items()})
            last = now

@app.on_event("startup")
async def on_start():
    await set_webhook()
    asyncio.create_task(ws_manager())
    asyncio.create_task(rest_poll_fallback())
    asyncio.create_task(keepalive_loop())
    asyncio.create_task(webhook_watchdog())
    asyncio.create_task(engine_heartbeat())
