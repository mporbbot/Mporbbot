# main.py
# FastAPI + Telegram webhook (Render-friendly)
# ORB-strategi enligt specifikation:
# - Ny ORB definieras av första GRÖN candle efter en röd
# - ORB-high/low = den gröna candlens high/low
# - Entry sker när en candle STÄNGER över ORB-high (entry_mode=CLOSE) eller bryter ORB-high intra-candle (entry_mode=TICK)
# - Första SL sätts till ORB-low; SL trailas upp candle-för-candle (lägsta nivå = senaste candle-låga eller bättre)
# - ORB ligger kvar tills en NY röd candle kommer (då väntar vi på nästa grön för ny ORB)
#
# Obs: Detta är en kompakt, Render-anpassad version med mock-orderbok.
# Du kan ansluta riktig exchange i blocket "LiveBroker" (KuCoin) om/ när du vill.

import os
import json
import time
import hmac
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

from pydantic import BaseModel

# Telegram (webhook via httpx)
import httpx

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
BOT_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
PUBLIC_URL = os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL")
PING_URL = os.environ.get("PING_URL")
MOCK_TRADE_USDT = float(os.environ.get("MOCK_TRADE_USDT", "100"))

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN saknas i ENV.")

# ---------- Enkel marknadsdatasimulator ----------
# I mock-läge streamar vi 'ticks' med en simpel prisgenerator per symbol.
# (För livekoppling, plugga in realdata här.)

SYMBOLS_DEFAULT = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]

class TickerFeed:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        # startpriser (ungefärliga)
        self.price = {
            "BTCUSDT": 117000.0,
            "ETHUSDT": 4400.0,
            "ADAUSDT": 0.95,
            "LINKUSDT": 15.0,
            "XRPUSDT": 0.55,
        }
        self.candle_tf_sec = 60
        self.last_candle_open_t = int(time.time() // self.candle_tf_sec) * self.candle_tf_sec
        # OHLC för aktiv candle per symbol
        self.ohlc = {s: {"o": self.price[s], "h": self.price[s], "l": self.price[s], "c": self.price[s]} for s in symbols}

    def step(self):
        # liten randomwalk
        for s in self.symbols:
            import random
            delta = (random.random() - 0.5) * (0.001 * self.price[s])  # 0.1% sväng
            self.price[s] = max(0.0001, self.price[s] + delta)
            # uppdatera aktiv candle
            ohlc = self.ohlc[s]
            ohlc["h"] = max(ohlc["h"], self.price[s])
            ohlc["l"] = min(ohlc["l"], self.price[s])
            ohlc["c"] = self.price[s]

        # kolla candle close
        now_t = int(time.time())
        if now_t >= self.last_candle_open_t + self.candle_tf_sec:
            # rulla candles
            closed = {}
            for s in self.symbols:
                closed[s] = self.ohlc[s]
                # starta ny
                p = self.price[s]
                self.ohlc[s] = {"o": p, "h": p, "l": p, "c": p}
            self.last_candle_open_t += self.candle_tf_sec
            return closed  # closed candles dict
        return None

# ---------- ORB & Trading Engine ----------
class Position:
    def __init__(self, symbol: str, qty: float, entry_px: float, sl: float):
        self.symbol = symbol
        self.qty = qty
        self.entry_px = entry_px
        self.sl = sl
        self.open_time = datetime.now(timezone.utc)

class EngineState:
    def __init__(self):
        self.enabled = True
        self.mode = "mock"  # "mock"|"live"
        self.symbols = SYMBOLS_DEFAULT.copy()
        self.entry_mode = "close"  # "close"|"tick"
        # trailing parametrar
        self.trig = 0.009  # 0.9%
        self.avst = 0.002  # 0.2%
        self.min_trail = 0.007  # 0.7%
        self.day_pnl = 0.0

        # ORB master
        self.orb_master_on = True
        # per symbol ORB status
        # {symbol: {"active": True/False, "high": float, "low": float, "await_red": bool, "last_red_close_t": ts}}
        self.orb: Dict[str, Dict] = {s: {"active": False, "high": None, "low": None, "await_red": True} for s in self.symbols}

        # pos & stops
        self.pos: Dict[str, Optional[Position]] = {s: None for s in self.symbols}

state = EngineState()

# ---------- Broker (mock & live stub) ----------
class MockBroker:
    def __init__(self):
        self.cash = 100000.0

    def buy(self, symbol: str, px: float, usdt_amt: float) -> Tuple[float, float]:
        qty = usdt_amt / px if px > 0 else 0.0
        return qty, px

    def sell(self, symbol: str, qty: float, px: float) -> float:
        return qty * px

mock_broker = MockBroker()

# ---- Telegram utils ----
async def tg_call(method: str, payload: Dict = None):
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{BOT_API}/{method}", json=payload or {})
        r.raise_for_status()
        return r.json()

async def tg_send(chat_id: int, text: str, reply_markup: Optional[Dict] = None):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    return await tg_call("sendMessage", payload)

def kb_main():
    # Visa bara baslista; när man trycker visar vi sekundära knappar
    return {
        "keyboard": [
            ["/status"],
            ["/engine_start", "/engine_stop"],
            ["/start_mock", "/start_live"],
            ["/symbols", "/timeframe"],
            ["/entry_mode", "/trailing"],
            ["/pnl", "/reset_pnl"],
            ["/export_csv", "/export_k4"],
            ["/keepalive_on", "/keepalive_off"],
            ["/orb_on", "/orb_off"],
            ["/panic"]
        ],
        "resize_keyboard": True
    }

def kb_entry_mode():
    return {"keyboard": [["tick"], ["close"]], "resize_keyboard": True, "one_time_keyboard": True}

def kb_timeframe():
    return {"keyboard": [["TF 1m"], ["TF 3m"], ["TF 5m"]], "resize_keyboard": True, "one_time_keyboard": True}

def kb_trailing():
    return {"keyboard": [["Trail +0.9%/0.2%"], ["Trail OFF"]], "resize_keyboard": True, "one_time_keyboard": True}

# ---- ORB logik ----
def update_orb_on_closed(symbol: str, closed_candle: Dict):
    """Hantera ny stängd candle ⇒ uppdatera ORB & ev. entry/exit."""
    o, h, l, c = closed_candle["o"], closed_candle["h"], closed_candle["l"], closed_candle["c"]

    d = state.orb[symbol]
    # identifiera röd/grön
    is_green = c > o
    is_red = c < o

    # Om vi väntar på röd: så fort vi ser röd markerar vi att nästa GRÖN definierar ny ORB.
    if d["await_red"]:
        if is_red:
            d["await_red"] = False  # nästa grön armerar ORB
        return

    # Letar första grön efter en röd → sätt ORB på den gröna candeln
    if not d["active"]:
        if is_green:
            d["high"] = h
            d["low"] = l
            d["active"] = True
        return

    # ORB är aktiv. Entry när candle stänger över ORB-high (om entry_mode=CLOSE)
    pos = state.pos[symbol]
    if pos is None and state.entry_mode == "close" and d["active"]:
        if c > (d["high"] or 0):
            # BUY
            qty, entry_px = mock_broker.buy(symbol, c, MOCK_TRADE_USDT)
            sl = d["low"]
            state.pos[symbol] = Position(symbol, qty, entry_px, sl)
    # Trail SL uppåt efter stängning om vi har position
    pos = state.pos[symbol]
    if pos:
        # För enkel trailing: sätt SL till max(nuvarande SL, candle-låga, ORB-low) men inte nedåt
        new_sl = max(pos.sl, l, d["low"])
        pos.sl = new_sl
        # Exit om stängning under SL
        if c <= pos.sl:
            cash = mock_broker.sell(symbol, pos.qty, c)
            pnl = (c - pos.entry_px) * pos.qty
            state.day_pnl += pnl
            state.pos[symbol] = None
            # Efter exit ligger ORB kvar. Om en röd candle kommer, återställ ORB-armen.
            # (Din regel: ORB förnyas först när ny röd + efterföljande grön uppstår.)
    # Om en RÖD candle dyker upp (vilken som helst), armera för ny ORB på nästa grön
    if is_red:
        d["active"] = False
        d["high"] = None
        d["low"] = None
        d["await_red"] = False  # vi har redan röd; nästa grön sätter ny ORB

def maybe_entry_on_tick(symbol: str, price: float):
    # Entry intra-candle om TICK-läge och brytning över ORB-high
    if state.entry_mode != "tick":
        return
    d = state.orb[symbol]
    if not d["active"] or state.pos[symbol] is not None:
        return
    if d["high"] is None or d["low"] is None:
        return
    if price > d["high"]:
        qty, entry_px = mock_broker.buy(symbol, price, MOCK_TRADE_USDT)
        sl = d["low"]
        state.pos[symbol] = Position(symbol, qty, entry_px, sl)

# ---------- FastAPI ----------
app = FastAPI()

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
    return msg.get("text") or ""

def html_status() -> str:
    lines = []
    lines.append(f"Mode: {state.mode}   Engine: {'ON' if state.enabled else 'OFF'}")
    lines.append(f"TF: 1m   Symbols: {','.join(state.symbols)}")
    lines.append(f"Entry: {state.entry_mode.upper()}   Trail: ON ({state.trig*100:.2f}%/{state.avst*100:.2f}% min {state.min_trail*100:.2f}%)")
    lines.append(f"Keepalive: {'ON' if PING_URL else 'OFF'}   DayPnL: {state.day_pnl:.4f} USDT")
    lines.append(f"ORB master: {'ON' if state.orb_master_on else 'OFF'}")
    for s in state.symbols:
        pos = state.pos[s]
        orb_flag = "ON" if state.orb_master_on else "OFF"
        if pos:
            lines.append(f"{s}: pos=✅ stop={pos.sl:.4f} | ORB: {orb_flag}")
        else:
            lines.append(f"{s}: pos=❌ stop=- | ORB: {orb_flag}")
    return "\n".join(lines)

async def handle_command(chat_id: int, text: str):
    t = text.strip()
    if t == "/start" or t == "/help":
        await tg_send(chat_id, "Kommandon:\n" + "\n".join([
            "/status",
            "/engine_start  /engine_stop",
            "/start_mock  /start_live",
            "/symbols (visa)  — nu: " + ",".join(state.symbols),
            "/timeframe (välj)",
            "/entry_mode (tick|close)",
            "/trailing",
            "/pnl  /reset_pnl",
            "/export_csv  /export_k4",
            "/keepalive_on  /keepalive_off",
            "/orb_on  /orb_off",
            "/panic",
        ]), reply_markup=kb_main())
        return

    if t == "/status":
        await tg_send(chat_id, html_status(), reply_markup=kb_main())
        return

    if t == "/engine_start":
        state.enabled = True
        await tg_send(chat_id, "Engine: ON")
        return

    if t == "/engine_stop":
        state.enabled = False
        await tg_send(chat_id, "Engine: OFF")
        return

    if t == "/start_mock":
        state.mode = "mock"
        await tg_send(chat_id, "Mode: MOCK")
        return

    if t == "/start_live":
        state.mode = "live"
        await tg_send(chat_id, "Mode: LIVE (stub; koppla exchange innan skarpt)")
        return

    if t == "/symbols":
        await tg_send(chat_id, f"Symbols: {','.join(state.symbols)}")
        return

    if t == "/timeframe":
        await tg_send(chat_id, "Timeframe satt till 1m", reply_markup=kb_timeframe())
        return

    if t == "/entry_mode":
        await tg_send(chat_id, "Välj entry_mode:", reply_markup=kb_entry_mode())
        return

    if t.lower() == "tick":
        state.entry_mode = "tick"
        await tg_send(chat_id, "✅ Entry mode set to: TICK", reply_markup=kb_main())
        return

    if t.lower() == "close":
        state.entry_mode = "close"
        await tg_send(chat_id, "✅ Entry mode set to: CLOSE", reply_markup=kb_main())
        return

    if t == "/trailing":
        await tg_send(chat_id, "Trail-knappar:", reply_markup=kb_trailing())
        return

    if t.startswith("Trail +"):
        state.trig = 0.009
        state.avst = 0.002
        state.min_trail = 0.007
        await tg_send(chat_id, "Trail: ON (0.90%/0.20% min 0.70%)", reply_markup=kb_main())
        return

    if t == "Trail OFF":
        # (Behåller parametrar men tolka som av)
        state.trig = 9.99
        await tg_send(chat_id, "Trail: OFF", reply_markup=kb_main())
        return

    if t == "/pnl":
        await tg_send(chat_id, f"DayPnL: {state.day_pnl:.4f} USDT")
        return

    if t == "/reset_pnl":
        state.day_pnl = 0.0
        await tg_send(chat_id, "DayPnL återställd.")
        return

    if t == "/export_csv":
        await tg_send(chat_id, "Export CSV (mock): Inga avslut loggade i denna demo.")
        return

    if t == "/export_k4":
        await tg_send(chat_id, "Export K4 (mock): Inga affärer loggade i denna demo.")
        return

    if t == "/keepalive_on":
        await tg_send(chat_id, "Keepalive ping används om PING_URL finns i ENV.")
        return

    if t == "/keepalive_off":
        await tg_send(chat_id, "Keepalive kan stängas av genom att ta bort PING_URL i ENV.")
        return

    if t == "/orb_on":
        state.orb_master_on = True
        await tg_send(chat_id, "ORB: ON (för alla valda symboler)")
        return

    if t == "/orb_off":
        state.orb_master_on = False
        await tg_send(chat_id, "ORB: OFF")
        return

    if t == "/panic":
        # sälj allt (mock)
        closed = 0
        for s in state.symbols:
            pos = state.pos[s]
            if pos:
                px = feed.price[s]
                cash = mock_broker.sell(s, pos.qty, px)
                pnl = (px - pos.entry_px) * pos.qty
                state.day_pnl += pnl
                state.pos[s] = None
                closed += 1
        await tg_send(chat_id, f"Panic exit: {closed} position(er) stängda.")
        return

    # okända texter ignoreras tyst

@app.post("/telegram/webhook")
async def telegram_webhook(update: TgUpdate):
    upd = update.dict()
    chat_id = chat_id_of(upd)
    text = text_of(upd)

    if text.startswith("/"):
        await handle_command(chat_id, text)
        return PlainTextResponse("ok")

    # annars ignorerar vi vanlig chatt
    return PlainTextResponse("ok")

@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.get("/healthz")
async def health():
    return PlainTextResponse("ok")

# ---------- Startup: set webhook + start bakgrundsloop ----------
async def set_webhook():
    if not PUBLIC_URL:
        raise RuntimeError("PUBLIC_URL/RENDER_EXTERNAL_URL saknas – kan inte sätta Telegram webhook.")
    url = PUBLIC_URL.rstrip("/") + "/telegram/webhook"
    await tg_call("setWebhook", {"url": url})
    # Valfritt: minska till 1 uppdatering åt gången
    await tg_call("setMyCommands", {"commands": [
        {"command": "status", "description": "Visa status"},
        {"command": "engine_start", "description": "Starta engine"},
        {"command": "engine_stop", "description": "Stoppa engine"},
        {"command": "start_mock", "description": "Mock-läge"},
        {"command": "start_live", "description": "Live-läge"},
        {"command": "symbols", "description": "Visa symboler"},
        {"command": "timeframe", "description": "Välj timeframe"},
        {"command": "entry_mode", "description": "Välj tick/close"},
        {"command": "trailing", "description": "Trail-knappar"},
        {"command": "pnl", "description": "Dagens PnL"},
        {"command": "reset_pnl", "description": "Nollställ PnL"},
        {"command": "export_csv", "description": "Export CSV"},
        {"command": "export_k4", "description": "Export K4"},
        {"command": "keepalive_on", "description": "Keepalive ON"},
        {"command": "keepalive_off", "description": "Keepalive OFF"},
        {"command": "orb_on", "description": "ORB på"},
        {"command": "orb_off", "description": "ORB av"},
        {"command": "panic", "description": "Sälj allt"},
    ]})

async def keepalive_loop():
    # ping Render/annan url så appen hålls vaken ibland
    if not PING_URL:
        return
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await client.get(PING_URL, timeout=10)
            except Exception:
                pass
            await asyncio.sleep(300)  # var 5:e minut

async def engine_loop():
    # enkel loop: ticka priser och generera 1m-candles
    while True:
        await asyncio.sleep(1.0)
        if not state.enabled:
            continue
        # uppdatera mock-priser
        closed = feed.step()
        # tick entry
        if state.orb_master_on:
            for s in state.symbols:
                maybe_entry_on_tick(s, feed.price[s])
        # candle close → ORB-logik + trail/exit
        if closed:
            for s, cndl in closed.items():
                if state.orb_master_on:
                    update_orb_on_closed(s, cndl)

@app.on_event("startup")
async def on_start():
    await set_webhook()
    # starta loopar
    asyncio.create_task(engine_loop())
    if PING_URL:
        asyncio.create_task(keepalive_loop())

# initiera mock-feed
feed = TickerFeed(SYMBOLS_DEFAULT)
