# main_v11.py
# Mock-trading på KuCoins LIVE-priser (1m candles + tick)
# - Ny ORB = första GRÖN efter RÖD; ORB-high/low = den gröna candlens high/low
# - Entry: CLOSE (stänger över ORB-high) eller TICK (intra-candle bryter)
# - Första SL = ORB-low; sedan följer SL upp med varje candles låg (minskar aldrig)
# - ORB nollas efter ny röd; nästa grön sätter ny
# Innehåller:
# - Telegram webhook (Render-vänlig)
# - Self-healing engine, keepalive, webhook-watchdog
# - Tydliga BUY/SELL-pushar
# - /status, /engine_start/stop, /start_mock, /entry_mode, /trailing, /pnl, /reset_pnl
#   /export_csv, /export_k4, /orb_on/off, /panic
# - Symboler: BTCUSDT, ETHUSDT, ADAUSDT, LINKUSDT, XRPUSDT

import os, time, asyncio, logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# ---------- ENV & Konfig ----------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN saknas i ENV.")
BOT_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

PUBLIC_URL = os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL")
PING_URL = os.environ.get("PING_URL")
MOCK_TRADE_USDT = float(os.environ.get("MOCK_TRADE_USDT", "100"))

# trailing standard
TRIGGER_PCT = 0.009   # 0.9%
LOCK_STEP_PCT = 0.002 # 0.2%
MIN_LOCK_PCT = 0.007  # 0.7%

SYMBOLS_DEFAULT = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]

# ---------- Logg ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# ---------- Formatteringar ----------
SYMBOL_PRICE_DECIMALS = {
    "BTCUSDT": 1,
    "ETHUSDT": 2,
    "ADAUSDT": 4,
    "LINKUSDT": 3,
    "XRPUSDT": 4,
}
def fmt_price(symbol: str, price: Optional[float]) -> str:
    if price is None:
        return "-"
    d = SYMBOL_PRICE_DECIMALS.get(symbol.upper(), 4)
    return f"{float(price):.{d}f}"

def fmt_pct(x: float) -> str:
    s = "+" if x >= 0 else ""
    return f"{s}{x*100:.2f}%"

# ---------- Telegram utils ----------
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

# ---------- KuCoin publikt prisflöde (REST polling) ----------
# Candles (1min): GET https://api.kucoin.com/api/v1/market/candles?type=1min&symbol=BTC-USDT
# Ticker:         GET https://api.kucoin.com/api/v1/market/orderbook/level1?symbol=BTC-USDT

KUCOIN_BASE = "https://api.kucoin.com"

def to_kucoin_symbol(sym: str) -> str:
    # BTCUSDT -> BTC-USDT
    sym = sym.upper()
    if sym.endswith("USDT"):
        return sym[:-4] + "-USDT"
    return sym.replace("USDT", "-USDT")

async def kc_get_candle_1m(symbol: str) -> Optional[Dict]:
    # returnerar senaste stängda 1m-candle för symbol
    ks = to_kucoin_symbol(symbol)
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    params = {"type": "1min", "symbol": ks}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        lst = data.get("data") or []
        if not lst:
            return None
        # KuCoin returns list of [time, open, close, high, low, volume, turnover] strings, time in ms/1000?
        # Official doc: time is unix timestamp in seconds (STR). We'll parse floats.
        # Take the FIRST item (most recent candle)
        t, o, c, h, l, *_ = lst[0]
        o, h, l, c = float(o), float(h), float(l), float(c)
        ts = int(t)
        return {"t": ts, "o": o, "h": h, "l": l, "c": c}

async def kc_get_tick_price(symbol: str) -> Optional[float]:
    ks = to_kucoin_symbol(symbol)
    url = f"{KUCOIN_BASE}/api/v1/market/orderbook/level1"
    params = {"symbol": ks}
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data") or {}
        px = data.get("price")
        if px is None:
            return None
        return float(px)

# ---------- Broker (mock) ----------
class Position:
    def __init__(self, symbol: str, qty: float, entry_px: float, sl: float):
        self.symbol = symbol
        self.qty = qty
        self.entry_px = entry_px
        self.sl = sl
        self.open_time = datetime.now(timezone.utc)

class MockBroker:
    def buy(self, symbol: str, px: float, usdt_amt: float) -> Tuple[float, float]:
        qty = usdt_amt / px if px > 0 else 0.0
        return qty, px
    def sell(self, symbol: str, qty: float, px: float) -> float:
        return qty * px

mock_broker = MockBroker()

# ---------- EngineState ----------
class EngineState:
    def __init__(self):
        self.enabled = True
        self.mode = "mock"  # mock/live (orders). Vi kör mock på livepris.
        self.symbols = SYMBOLS_DEFAULT.copy()
        self.entry_mode = "close"  # "close"|"tick"
        self.trig = TRIGGER_PCT
        self.avst = LOCK_STEP_PCT
        self.min_trail = MIN_LOCK_PCT
        self.day_pnl = 0.0
        self.keepalive = True if PING_URL else False

        self.orb_master_on = True
        # per symbol ORB state
        self.orb: Dict[str, Dict] = {
            s: {"active": False, "high": None, "low": None, "await_red": True, "last_close_t": 0}
            for s in self.symbols
        }
        self.pos: Dict[str, Optional[Position]] = {s: None for s in self.symbols}
        self.chat_id: Optional[int] = None

        # live feed cache
        self.tick: Dict[str, float] = {s: None for s in self.symbols}

state = EngineState()

# ---------- Keyboards ----------
def kb_main():
    return {
        "keyboard": [
            ["/status"],
            ["/engine_start", "/engine_stop"],
            ["/start_mock"],
            ["/entry_mode", "/trailing"],
            ["/pnl", "/reset_pnl"],
            ["/export_csv", "/export_k4"],
            ["/orb_on", "/orb_off"],
            ["/panic"]
        ],
        "resize_keyboard": True
    }
def kb_entry_mode():
    return {"keyboard": [["tick"], ["close"]], "resize_keyboard": True, "one_time_keyboard": True}
def kb_trailing():
    return {"keyboard": [["Trail +0.9%/0.2%"], ["Trail OFF"]], "resize_keyboard": True, "one_time_keyboard": True}

# ---------- Status ----------
def build_status_text() -> str:
    mode = "mock (live price)"
    engine_on = "ON" if state.enabled else "OFF"
    entry_mode = state.entry_mode.upper()
    trail_txt = "OFF"
    if state.trig < 5:
        trail_txt = f"ON ({state.trig*100:.2f}%/{state.avst*100:.2f}% min {state.min_trail*100:.2f}%)"
    hdr = [
        f"Mode: {mode}   Engine: {engine_on}",
        f"TF: 1m   Symbols:",
        ",".join(state.symbols),
        f"Entry: {entry_mode}   Trail: {trail_txt}",
        f"Keepalive: {'ON' if state.keepalive else 'OFF'}   DayPnL: {state.day_pnl:.4f} USDT",
        f"ORB master: {'ON' if state.orb_master_on else 'OFF'}",
    ]
    lines = []
    for s in state.symbols:
        pos = state.pos[s]
        orb_flag = "ON" if state.orb_master_on else "OFF"
        if pos:
            entry = fmt_price(s, pos.entry_px)
            stp = fmt_price(s, pos.sl)
            dlt = ""
            if pos.entry_px:
                dlt = f" ({fmt_pct((pos.sl - pos.entry_px)/pos.entry_px)})"
            size = f"{pos.qty:.6f}"
            lines.append(f"{s}: pos=✅ entry={entry} stop={stp}{dlt} size={size} | ORB: {orb_flag}")
        else:
            lines.append(f"{s}: pos=❌ entry=- stop=- size=- | ORB: {orb_flag}")
    return "\n".join(hdr + lines)

# ---------- Pushar ----------
async def push_buy(symbol: str, reason: str, price: float, sl: float, qty: float):
    if not state.chat_id: return
    txt = (f"🟢 <b>BUY {symbol} [MOCK @ live]</b>\n"
           f"entry={fmt_price(symbol, price)}  stop={fmt_price(symbol, sl)} "
           f"({fmt_pct((sl-price)/price)})  size={qty:.6f}\n"
           f"reason={reason}")
    await tg_send(state.chat_id, txt)

async def push_sell(symbol: str, price: float, pnl_usdt: float, reason: str):
    if not state.chat_id: return
    sign = "🟥" if pnl_usdt < 0 else "🟩"
    txt = (f"{sign} <b>SELL {symbol} [MOCK @ live]</b>\n"
           f"exit={fmt_price(symbol, price)}   PnL={pnl_usdt:.4f} USDT\n"
           f"reason={reason}")
    await tg_send(state.chat_id, txt)

# ---------- ORB på candle close ----------
def update_orb_on_closed(symbol: str, closed_candle: Dict):
    o, h, l, c, t = closed_candle["o"], closed_candle["h"], closed_candle["l"], closed_candle["c"], closed_candle["t"]
    d = state.orb[symbol]
    d["last_close_t"] = t
    is_green = c > o
    is_red = c < o

    # väntar på röd för att kunna armera på nästa grön
    if d["await_red"]:
        if is_red:
            d["await_red"] = False
        return

    # leta första grön efter röd → sätt ORB på den gröna
    if not d["active"]:
        if is_green:
            d["high"] = h
            d["low"] = l
            d["active"] = True
        return

    # entry på CLOSE om c > ORB-high och ingen position
    pos = state.pos[symbol]
    if pos is None and state.entry_mode == "close" and d["active"] and d["high"] is not None:
        if c > d["high"]:
            qty, entry_px = mock_broker.buy(symbol, c, MOCK_TRADE_USDT)
            sl = d["low"]
            state.pos[symbol] = Position(symbol, qty, entry_px, sl)
            asyncio.create_task(push_buy(symbol, "CloseBreak", entry_px, sl, qty))

    # trail & exit om position finns
    pos = state.pos[symbol]
    if pos:
        new_sl = max(pos.sl, l, d["low"] if d["low"] is not None else pos.sl)
        if new_sl > pos.sl:
            pos.sl = new_sl
        # exit på close under/likamed SL
        if c <= pos.sl:
            cash = mock_broker.sell(symbol, pos.qty, c)
            pnl = (c - pos.entry_px) * pos.qty
            state.day_pnl += pnl
            state.pos[symbol] = None
            asyncio.create_task(push_sell(symbol, c, pnl, "Close<=SL"))

    # om röd candle → reset ORB (nästa grön sätter ny)
    if is_red:
        d["active"] = False
        d["high"] = None
        d["low"] = None
        d["await_red"] = False

# ---------- Tick-entry (intra candle) ----------
def maybe_entry_on_tick(symbol: str, price: Optional[float]):
    if state.entry_mode != "tick" or price is None:
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
        asyncio.create_task(push_buy(symbol, "TickBreak", entry_px, sl, qty))

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

def kb_main_menu():
    return kb_main()

async def handle_command(chat_id: int, text: str):
    state.chat_id = chat_id
    t = text.strip()

    if t in ("/start", "/help"):
        await tg_send(chat_id, "Kommandon:\n" + "\n".join([
            "/status",
            "/engine_start  /engine_stop",
            "/start_mock",
            "/entry_mode  (/tick /close)",
            "/trailing",
            "/pnl  /reset_pnl",
            "/export_csv  /export_k4",
            "/orb_on  /orb_off",
            "/panic",
        ]), reply_markup=kb_main_menu())
        return

    if t == "/status":
        await tg_send(chat_id, build_status_text(), reply_markup=kb_main_menu())
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
        await tg_send(chat_id, f"Mode: MOCK @ KuCoin live price ({MOCK_TRADE_USDT:.0f} USDT/trade)")
        return

    if t == "/entry_mode":
        await tg_send(chat_id, "Välj entry_mode:", reply_markup=kb_entry_mode())
        return

    if t.lower() == "tick":
        state.entry_mode = "tick"
        await tg_send(chat_id, "✅ Entry mode set to: TICK", reply_markup=kb_main_menu())
        return

    if t.lower() == "close":
        state.entry_mode = "close"
        await tg_send(chat_id, "✅ Entry mode set to: CLOSE", reply_markup=kb_main_menu())
        return

    if t == "/trailing":
        await tg_send(chat_id, "Trail-knappar:", reply_markup=kb_trailing())
        return

    if t.startswith("Trail +"):
        state.trig = TRIGGER_PCT
        state.avst = LOCK_STEP_PCT
        state.min_trail = MIN_LOCK_PCT
        await tg_send(chat_id, "Trail: ON (0.90%/0.20% min 0.70%)", reply_markup=kb_main_menu())
        return

    if t == "Trail OFF":
        state.trig = 99.0
        await tg_send(chat_id, "Trail: OFF", reply_markup=kb_main_menu())
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

    if t == "/orb_on":
        state.orb_master_on = True
        await tg_send(chat_id, "ORB: ON")
        return

    if t == "/orb_off":
        state.orb_master_on = False
        await tg_send(chat_id, "ORB: OFF")
        return

    if t == "/panic":
        closed = 0
        for s in state.symbols:
            pos = state.pos[s]
            if pos:
                px = state.tick.get(s)
                if px is None:
                    px = await kc_get_tick_price(s)
                cash = mock_broker.sell(s, pos.qty, px)
                pnl = (px - pos.entry_px) * pos.qty
                state.day_pnl += pnl
                state.pos[s] = None
                asyncio.create_task(push_sell(s, px, pnl, "Panic"))
                closed += 1
        await tg_send(chat_id, f"Panic exit: {closed} position(er) stängda.")
        return

@app.post("/telegram/webhook")
async def telegram_webhook(update: TgUpdate):
    upd = update.dict()
    chat_id = chat_id_of(upd)
    text = text_of(upd)
    if chat_id and text.startswith("/"):
        await handle_command(chat_id, text)
    return PlainTextResponse("ok")

@app.get("/")
async def root():
    return PlainTextResponse("OK")

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

# ---------- Keepalive + webhook-watchdog ----------
async def keepalive_loop():
    urls = set()
    if PING_URL:
        urls.add(PING_URL)
    if PUBLIC_URL:
        urls.add(PUBLIC_URL.rstrip("/") + "/healthz")
    if not urls:
        return
    async with httpx.AsyncClient(timeout=8) as client:
        while True:
            for u in list(urls):
                try:
                    await client.get(u)
                except Exception:
                    pass
            await asyncio.sleep(60)

async def webhook_watchdog():
    if not PUBLIC_URL:
        return
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
    await tg_call("setMyCommands", {"commands": [
        {"command": "status", "description": "Visa status"},
        {"command": "engine_start", "description": "Starta engine"},
        {"command": "engine_stop", "description": "Stoppa engine"},
        {"command": "start_mock", "description": "Mock-läge @ livepris"},
        {"command": "entry_mode", "description": "Välj tick/close"},
        {"command": "trailing", "description": "Trail-knappar"},
        {"command": "pnl", "description": "Dagens PnL"},
        {"command": "reset_pnl", "description": "Nollställ PnL"},
        {"command": "export_csv", "description": "Export CSV"},
        {"command": "export_k4", "description": "Export K4"},
        {"command": "orb_on", "description": "ORB på"},
        {"command": "orb_off", "description": "ORB av"},
        {"command": "panic", "description": "Sälj allt"},
    ]})

# ---------- Engine + Supervisor ----------
ENGINE_TASK: Optional[asyncio.Task] = None

async def engine_loop():
    last_candle_ts: Dict[str, int] = {s: 0 for s in state.symbols}
    last_heartbeat = 0.0
    while True:
        try:
            now = time.time()
            if now - last_heartbeat > 30:
                hb = {s: (round(state.tick[s], 6) if state.tick[s] else None) for s in state.symbols}
                log.info("Engine heartbeat: %s", hb)
                last_heartbeat = now

            await asyncio.sleep(1.0)
            if not state.enabled:
                continue

            # 1) hämta tickpriser (för tick-entry och status)
            for s in state.symbols:
                try:
                    px = await kc_get_tick_price(s)
                    state.tick[s] = px
                    if state.orb_master_on:
                        maybe_entry_on_tick(s, px)
                except Exception:
                    pass

            # 2) hämta senaste stängda 1m-candle per symbol och processa om ny
            for s in state.symbols:
                try:
                    cd = await kc_get_candle_1m(s)
                    if not cd:
                        continue
                    t = cd["t"]
                    if t != last_candle_ts[s]:
                        # ny stängd candle
                        last_candle_ts[s] = t
                        if state.orb_master_on:
                            update_orb_on_closed(s, cd)
                except Exception as e:
                    log.warning("candle fetch fail %s: %s", s, e)

        except Exception as e:
            log.exception("Engine fel: %s", e)
            await asyncio.sleep(2)

async def supervisor():
    global ENGINE_TASK
    backoff = 2
    while True:
        if ENGINE_TASK is None or ENGINE_TASK.done():
            try:
                if ENGINE_TASK and ENGINE_TASK.exception():
                    log.error("Engine dog: %s", ENGINE_TASK.exception())
                log.info("Startar engine-loop…")
                ENGINE_TASK = asyncio.create_task(engine_loop())
                backoff = 2
            except Exception as e:
                log.exception("Kunde inte starta engine: %s", e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
        await asyncio.sleep(3)

# ---------- Startup ----------
app = FastAPI()

@app.on_event("startup")
async def on_start():
    await set_webhook()
    asyncio.create_task(supervisor())
    asyncio.create_task(webhook_watchdog())
    if state.keepalive:
        asyncio.create_task(keepalive_loop())
