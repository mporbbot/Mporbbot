# main_v24.py
# Komplett bot: FastAPI + Telegram + ORB-motor + knappar/paneler + timeframe + trailing stop + SELL-kort.
# Körbar på Render: uvicorn main_v24:app --host 0.0.0.0 --port $PORT

from __future__ import annotations

import os
import time
import json
import math
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# ---- TELEGRAM (python-telegram-bot v20.x, asynk) ----
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ===================== LOGGNING =====================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("orb_v24")

# ===================== ENV/DEFAULTS =====================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT").replace(" ", "")
SYMBOLS = [s for s in DEFAULT_SYMBOLS.split(",") if s]

# Timeframes (Binance publika endpoints används för stabila candles/ticker)
TF_MAP = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
}

DOJI_BODY_RATIO = float(os.getenv("DOJI_BODY_RATIO", "0.2"))  # body <= 20% av range => doji
POLL_SEC = float(os.getenv("POLL_SEC", "1.0"))                 # motor loop-sleep
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10.0"))

# ===================== DATAMODELL =====================

@dataclass
class Candle:
    ts: int        # ms
    o: float
    h: float
    l: float
    c: float

@dataclass
class Position:
    side: str             # 'long' eller 'short'
    entry: float
    stop: float
    qty: float = 1.0
    open_ts: int = 0

@dataclass
class Orb:
    have_orb: bool = False
    high: float = 0.0
    low: float = 0.0
    ts: int = 0

@dataclass
class SymState:
    last_candle_ts: int = 0
    last_color: Optional[str] = None  # 'green' eller 'red'
    arm_new_orb: bool = False         # "första efter skifte" beväpning
    orb: Orb = field(default_factory=Orb)
    pos: Optional[Position] = None
    realized_pnl: float = 0.0

@dataclass
class EngineState:
    engine_on: bool = False
    orb_on: bool = True
    trailing_on: bool = True
    entry_mode: str = "both"  # 'long'|'short'|'both'
    timeframe_key: str = "1"  # "1","3","5","15"
    chat_id: Optional[int] = None
    panel_mode: int = 1       # 1|2 (två layouter)
    symbols: List[str] = field(default_factory=lambda: SYMBOLS.copy())

# ===================== GLOBAL STATE =====================

STATE = EngineState()
SYMS: Dict[str, SymState] = {s: SymState() for s in STATE.symbols}
STATE_LOCK = threading.Lock()

# ===================== HJÄLPFUNKTIONER =====================

def is_doji(c: Candle) -> bool:
    rng = max(0.0, c.h - c.l)
    body = abs(c.c - c.o)
    return rng > 0 and body <= DOJI_BODY_RATIO * rng

def fmt_price(x: float) -> str:
    if x >= 1000:  # tusental med 1-2 decimals
        return f"{x:,.1f}".replace(",", " ").replace(".", ",")
    if x >= 1:
        return f"{x:.2f}".replace(".", ",")
    return f"{x:.6f}".rstrip("0").rstrip(",").replace(".", ",")

def now_ms() -> int:
    return int(time.time() * 1000)

def binance_symbol(s: str) -> str:
    # vi använder redan "BTCUSDT" etc
    return s.upper().replace("-", "")

def tf_str() -> str:
    return TF_MAP.get(STATE.timeframe_key, "1m")

# ===================== HTTP KLIENT =====================

HTTP = httpx.Client(timeout=HTTP_TIMEOUT)

def get_ticker_price(sym: str) -> Optional[float]:
    # Binance publikt endpoint
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol(sym)}"
    try:
        r = HTTP.get(url)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        log.warning("Ticker misslyckades %s: %s", sym, e)
        return None

def get_latest_candle(sym: str, interval: str) -> Optional[Candle]:
    # Binance klines: [ openTime, open, high, low, close, volume, closeTime, ... ]
    url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol(sym)}&interval={interval}&limit=2"
    try:
        r = HTTP.get(url)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        k = data[-1]
        return Candle(
            ts=int(k[6]),  # closeTime ms
            o=float(k[1]),
            h=float(k[2]),
            l=float(k[3]),
            c=float(k[4]),
        )
    except Exception as e:
        log.warning("Klines misslyckades %s: %s", sym, e)
        return None

# ===================== TELEGRAM OUTBOX (HTTP) =====================

def tg_api(method: str, payload: dict) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    try:
        HTTP.post(url, json=payload)
    except Exception as e:
        log.warning("TG API fel %s: %s", method, e)

def send_msg(text: str, chat_id: Optional[int] = None, parse_mode: str = "HTML") -> None:
    cid = chat_id or STATE.chat_id
    if not cid or not TELEGRAM_BOT_TOKEN:
        return
    tg_api("sendMessage", {"chat_id": cid, "text": text, "parse_mode": parse_mode})

def set_tg_commands() -> None:
    if not TELEGRAM_BOT_TOKEN:
        return
    commands = [
        {"command": "status", "description": "Visa status"},
        {"command": "engine_start", "description": "Starta motor"},
        {"command": "engine_stop", "description": "Stoppa motor"},
        {"command": "entry_mode", "description": "Byt Entry-mode (long/short/båda)"},
        {"command": "trailing", "description": "Toggle trailing stop"},
        {"command": "orb_on", "description": "Aktivera ORB"},
        {"command": "orb_off", "description": "Avaktivera ORB"},
        {"command": "pnl", "description": "Visa PnL"},
        {"command": "reset_pnl", "description": "Nollställ PnL"},
        {"command": "timeframe", "description": "Välj timeframe (1/3/5/15)"},
        {"command": "panic", "description": "Stäng allt NU"},
    ]
    tg_api("setMyCommands", {"commands": commands})

def ensure_polling_mode() -> None:
    # Se till att webhook inte är satt (vi kör polling)
    if not TELEGRAM_BOT_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getWebhookInfo"
        resp = HTTP.post(url).json()
        if resp.get("ok") and resp.get("result", {}).get("url"):
            HTTP.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook")
            log.info("[TG] webhook deleted")
    except Exception as e:
        log.warning("[TG] webhook check fail: %s", e)

# ===================== UI: PANELER & KNAPPAR =====================

def build_keyboard(panel: int) -> InlineKeyboardMarkup:
    # Panel 1: kommandon grupperade
    if panel == 1:
        rows = [
            [
                InlineKeyboardButton("▶️ Engine", callback_data="ENGINE:START"),
                InlineKeyboardButton("⏹ Stop", callback_data="ENGINE:STOP"),
                InlineKeyboardButton("🚨 Panic", callback_data="PANIC"),
            ],
            [
                InlineKeyboardButton("ORB ON", callback_data="ORB:ON"),
                InlineKeyboardButton("ORB OFF", callback_data="ORB:OFF"),
                InlineKeyboardButton("⛓ Trailing", callback_data="TRAIL:TOGGLE"),
            ],
            [
                InlineKeyboardButton("Entry: Long", callback_data="MODE:LONG"),
                InlineKeyboardButton("Short", callback_data="MODE:SHORT"),
                InlineKeyboardButton("Båda", callback_data="MODE:BOTH"),
            ],
            [
                InlineKeyboardButton("TF 1m", callback_data="TF:1"),
                InlineKeyboardButton("3m", callback_data="TF:3"),
                InlineKeyboardButton("5m", callback_data="TF:5"),
                InlineKeyboardButton("15m", callback_data="TF:15"),
            ],
            [
                InlineKeyboardButton("💳 PnL", callback_data="PNL:SHOW"),
                InlineKeyboardButton("♻️ Reset", callback_data="PNL:RESET"),
            ],
            [
                InlineKeyboardButton("Byt till Panel 2", callback_data="PANEL:2"),
            ],
        ]
    else:
        # Panel 2: alternativ layout
        rows = [
            [
                InlineKeyboardButton("TF 1", callback_data="TF:1"),
                InlineKeyboardButton("TF 3", callback_data="TF:3"),
                InlineKeyboardButton("TF 5", callback_data="TF:5"),
                InlineKeyboardButton("TF 15", callback_data="TF:15"),
            ],
            [
                InlineKeyboardButton("Entry Long", callback_data="MODE:LONG"),
                InlineKeyboardButton("Short", callback_data="MODE:SHORT"),
                InlineKeyboardButton("Båda", callback_data="MODE:BOTH"),
            ],
            [
                InlineKeyboardButton("ORB ON", callback_data="ORB:ON"),
                InlineKeyboardButton("OFF", callback_data="ORB:OFF"),
                InlineKeyboardButton("Trailing", callback_data="TRAIL:TOGGLE"),
            ],
            [
                InlineKeyboardButton("▶️ Start", callback_data="ENGINE:START"),
                InlineKeyboardButton("⏹ Stop", callback_data="ENGINE:STOP"),
                InlineKeyboardButton("🚨 Panic", callback_data="PANIC"),
            ],
            [
                InlineKeyboardButton("PnL", callback_data="PNL:SHOW"),
                InlineKeyboardButton("Reset PnL", callback_data="PNL:RESET"),
            ],
            [
                InlineKeyboardButton("Byt till Panel 1", callback_data="PANEL:1"),
            ],
        ]
    return InlineKeyboardMarkup(rows)

def panel_text() -> str:
    with STATE_LOCK:
        t = (
            f"<b>MP ORB v24</b>\n"
            f"Engine: {'ON' if STATE.engine_on else 'OFF'}\n"
            f"Entry mode: {STATE.entry_mode.upper()}\n"
            f"Trailing: {'ON' if STATE.trailing_on else 'OFF'}\n"
            f"ORB: {'ON' if STATE.orb_on else 'OFF'}\n"
            f"TF: {tf_str()}\n"
            f"Symbols: {', '.join(STATE.symbols)}"
        )
    return t

# ===================== SIGNALKORT =====================

def send_buy_card(sym: str, pos: Position, orb: Orb) -> None:
    text = (
        f"🟢 <b>BUY</b>\n"
        f"Symbol: <b>{sym}</b>\n"
        f"Side: <b>{pos.side.upper()}</b>\n"
        f"Entry: <code>{fmt_price(pos.entry)}</code>\n"
        f"Stop: <code>{fmt_price(pos.stop)}</code> (ORB low/high)\n"
        f"TF: <code>{tf_str()}</code>\n"
        f"ORB: H=<code>{fmt_price(orb.high)}</code> L=<code>{fmt_price(orb.low)}</code>"
    )
    send_msg(text)

def send_sell_card(sym: str, exit_px: float, reason: str, pnl_abs: float, pos: Position) -> None:
    pnl_pct = (exit_px - pos.entry) / pos.entry * 100.0 if pos.entry else 0.0
    if pos.side == "short":
        pnl_pct = -pnl_pct
    text = (
        f"🔴 <b>SELL</b>\n"
        f"Symbol: <b>{sym}</b>\n"
        f"Reason: <b>{reason}</b>\n"
        f"Exit: <code>{fmt_price(exit_px)}</code>\n"
        f"PnL: <b>{fmt_price(pnl_abs)}</b>  ({pnl_pct:+.2f}%)"
    )
    send_msg(text)

# ===================== ORB-LOGIK =====================

def update_orb_on_shift(sym: str, c: Candle, st: SymState) -> None:
    # Candlefärg
    color = "green" if c.c >= c.o else "red"
    if st.last_color is None:
        st.last_color = color
        return

    # Om färg skiftade: beväpna nästa candle för ORB
    if color != st.last_color and not st.arm_new_orb:
        st.arm_new_orb = True
        st.last_color = color
        return

    # Om beväpnad: sätt ORB på "första röda eller gröna candle som görs efter skifte"
    if st.arm_new_orb:
        if is_doji(c):
            # hoppa över doji
            log.info("%s SKIP_DOJI: ts=%s", sym, c.ts)
            return
        st.orb.have_orb = True
        st.orb.high = c.h
        st.orb.low = c.l
        st.orb.ts = c.ts
        st.arm_new_orb = False
        log.info("%s NEW_ORB: H=%s L=%s ts=%s", sym, fmt_price(c.h), fmt_price(c.l), c.ts)

def maybe_open_position(sym: str, st: SymState, price: float) -> None:
    if st.pos or not st.orb.have_orb:
        return
    with STATE_LOCK:
        if not (STATE.engine_on and STATE.orb_on):
            return
        mode = STATE.entry_mode

    # Breakoutkontroll mot livepriset
    if mode in ("long", "both") and price > st.orb.high:
        stop = st.orb.low
        st.pos = Position(side="long", entry=price, stop=stop, open_ts=now_ms())
        send_buy_card(sym, st.pos, st.orb)
        log.info("%s ENTER LONG @ %s stop=%s", sym, price, stop)
        return

    if mode in ("short", "both") and price < st.orb.low:
        stop = st.orb.high
        st.pos = Position(side="short", entry=price, stop=stop, open_ts=now_ms())
        send_buy_card(sym, st.pos, st.orb)
        log.info("%s ENTER SHORT @ %s stop=%s", sym, price, stop)
        return

def maybe_trail_stop(sym: str, st: SymState, c: Candle) -> None:
    with STATE_LOCK:
        trailing = STATE.trailing_on
    if not trailing or not st.pos:
        return
    if st.pos.side == "long":
        new_stop = max(st.pos.stop, c.l)
        if new_stop > st.pos.stop:
            st.pos.stop = new_stop
            log.info("%s TRAIL LONG stop->%s", sym, new_stop)
    else:
        new_stop = min(st.pos.stop, c.h)
        if new_stop < st.pos.stop:
            st.pos.stop = new_stop
            log.info("%s TRAIL SHORT stop->%s", sym, new_stop)

def maybe_stop_out(sym: str, st: SymState, price: float) -> None:
    if not st.pos:
        return
    if st.pos.side == "long" and price <= st.pos.stop:
        pnl = (price - st.pos.entry) * st.pos.qty
        st.realized_pnl += pnl
        send_sell_card(sym, price, "STOP", pnl, st.pos)
        log.info("%s STOP LONG @ %s pnl=%s", sym, price, pnl)
        st.pos = None
        return
    if st.pos.side == "short" and price >= st.pos.stop:
        pnl = (st.pos.entry - price) * st.pos.qty
        st.realized_pnl += pnl
        send_sell_card(sym, price, "STOP", pnl, st.pos)
        log.info("%s STOP SHORT @ %s pnl=%s", sym, price, pnl)
        st.pos = None
        return

def panic_close_all(reason: str = "PANIC") -> None:
    for sym, st in SYMS.items():
        if not st.pos:
            continue
        px = get_ticker_price(sym)
        if px is None:
            continue
        if st.pos.side == "long":
            pnl = (px - st.pos.entry) * st.pos.qty
        else:
            pnl = (st.pos.entry - px) * st.pos.qty
        st.realized_pnl += pnl
        send_sell_card(sym, px, reason, pnl, st.pos)
        log.info("%s %s close @ %s pnl=%s", sym, reason, px, pnl)
        st.pos = None

# ===================== MOTOR =====================

def engine_worker():
    interval = tf_str()
    log.info("Engine startar. TF=%s, symbols=%s", interval, ",".join(STATE.symbols))
    while True:
        try:
            with STATE_LOCK:
                running = STATE.engine_on
                interval = tf_str()
            for sym in STATE.symbols:
                c = get_latest_candle(sym, interval)
                if not c:
                    continue

                s = SYMS[sym]
                # Ny candle?
                if c.ts != s.last_candle_ts:
                    s.last_candle_ts = c.ts
                    # Uppdatera ORB vid skifte
                    update_orb_on_shift(sym, c, s)
                    # Trailing på candle close
                    maybe_trail_stop(sym, s, c)

                # Livepris för break/stop
                px = get_ticker_price(sym)
                if px is None:
                    continue

                # Stop-check alltid, även om engine av
                maybe_stop_out(sym, s, px)

                # Entry endast om engine_on
                if running:
                    maybe_open_position(sym, s, px)

            time.sleep(POLL_SEC)
        except Exception as e:
            log.exception("Engine fel: %s", e)
            time.sleep(1.0)

# ===================== TELEGRAM HANDLERS =====================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with STATE_LOCK:
        STATE.chat_id = update.effective_chat.id
        txt = panel_text()
        kb = build_keyboard(STATE.panel_mode)
    await update.message.reply_text(txt, reply_markup=kb, parse_mode="HTML")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with STATE_LOCK:
        txt = panel_text()
    await update.message.reply_text(txt, parse_mode="HTML")

async def cmd_engine_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with STATE_LOCK:
        STATE.engine_on = True
    await update.message.reply_text("▶️ Engine: <b>ON</b>", parse_mode="HTML")

async def cmd_engine_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with STATE_LOCK:
        STATE.engine_on = False
    await update.message.reply_text("⏹ Engine: <b>OFF</b>", parse_mode="HTML")

async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Cykla LONG -> SHORT -> BOTH
    with STATE_LOCK:
        if STATE.entry_mode == "long":
            STATE.entry_mode = "short"
        elif STATE.entry_mode == "short":
            STATE.entry_mode = "both"
        else:
            STATE.entry_mode = "long"
        mode = STATE.entry_mode
    await update.message.reply_text(f"Entry mode: <b>{mode.upper()}</b>", parse_mode="HTML")

async def cmd_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with STATE_LOCK:
        STATE.trailing_on = not STATE.trailing_on
        t = "ON" if STATE.trailing_on else "OFF"
    await update.message.reply_text(f"Trailing: <b>{t}</b>", parse_mode="HTML")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with STATE_LOCK:
        STATE.orb_on = True
    await update.message.reply_text("ORB: <b>ON</b>", parse_mode="HTML")

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with STATE_LOCK:
        STATE.orb_on = False
    await update.message.reply_text("ORB: <b>OFF</b>", parse_mode="HTML")

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = []
    total = 0.0
    for sym, st in SYMS.items():
        lines.append(f"{sym}: {fmt_price(st.realized_pnl)}")
        total += st.realized_pnl
    lines.append(f"—\nTOTAL: <b>{fmt_price(total)}</b>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for st in SYMS.values():
        st.realized_pnl = 0.0
    await update.message.reply_text("PnL nollställd.")

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    panic_close_all("PANIC")
    await update.message.reply_text("🚨 Alla positioner stängda (PANIC).")

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("1m", callback_data="TF:1"),
        InlineKeyboardButton("3m", callback_data="TF:3"),
        InlineKeyboardButton("5m", callback_data="TF:5"),
        InlineKeyboardButton("15m", callback_data="TF:15"),
    ]])
    await update.message.reply_text("Välj timeframe:", reply_markup=kb)

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = (q.data or "").upper()
    await q.answer()
    if data.startswith("ENGINE:"):
        on = data.endswith("START")
        with STATE_LOCK:
            STATE.engine_on = on
        await q.edit_message_text(panel_text(), reply_markup=build_keyboard(STATE.panel_mode), parse_mode="HTML")
        return
    if data.startswith("PANEL:"):
        mode = 1 if data.endswith("1") else 2
        with STATE_LOCK:
            STATE.panel_mode = mode
        await q.edit_message_text(panel_text(), reply_markup=build_keyboard(STATE.panel_mode), parse_mode="HTML")
        return
    if data.startswith("TF:"):
        key = data.split(":")[1]
        if key in TF_MAP:
            with STATE_LOCK:
                STATE.timeframe_key = key
            await q.edit_message_text(panel_text(), reply_markup=build_keyboard(STATE.panel_mode), parse_mode="HTML")
        return
    if data.startswith("MODE:"):
        val = data.split(":")[1]
        with STATE_LOCK:
            if val == "LONG":
                STATE.entry_mode = "long"
            elif val == "SHORT":
                STATE.entry_mode = "short"
            else:
                STATE.entry_mode = "both"
        await q.edit_message_text(panel_text(), reply_markup=build_keyboard(STATE.panel_mode), parse_mode="HTML")
        return
    if data.startswith("TRAIL:"):
        with STATE_LOCK:
            STATE.trailing_on = not STATE.trailing_on
        await q.edit_message_text(panel_text(), reply_markup=build_keyboard(STATE.panel_mode), parse_mode="HTML")
        return
    if data.startswith("ORB:"):
        with STATE_LOCK:
            STATE.orb_on = data.endswith("ON")
        await q.edit_message_text(panel_text(), reply_markup=build_keyboard(STATE.panel_mode), parse_mode="HTML")
        return
    if data.startswith("PNL:SHOW"):
        lines = []
        total = 0.0
        for sym, st in SYMS.items():
            lines.append(f"{sym}: {fmt_price(st.realized_pnl)}")
            total += st.realized_pnl
        lines.append(f"—\nTOTAL: <b>{fmt_price(total)}</b>")
        await q.edit_message_text("\n".join(lines), parse_mode="HTML")
        return
    if data.startswith("PNL:RESET"):
        for st in SYMS.values():
            st.realized_pnl = 0.0
        await q.edit_message_text("PnL nollställd.")
        return
    if data.startswith("PANIC"):
        panic_close_all("PANIC")
        await q.edit_message_text("🚨 Alla positioner stängda (PANIC).")
        return

# ===================== FASTAPI =====================

app = FastAPI(title="Mporbbot v24")

@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return "Mporbbot v24 – OK"

@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"

@app.get("/status", response_class=PlainTextResponse)
def status() -> str:
    with STATE_LOCK:
        return json.dumps({
            "engine_on": STATE.engine_on,
            "entry_mode": STATE.entry_mode,
            "trailing_on": STATE.trailing_on,
            "orb_on": STATE.orb_on,
            "tf": tf_str(),
            "symbols": STATE.symbols,
        })

# ===================== STARTUP: ENGINE + TELEGRAM =====================

def telegram_worker():
    if not TELEGRAM_BOT_TOKEN:
        log.error("[TG] Ingen TELEGRAM_BOT_TOKEN – startas ej.")
        return
    try:
        ensure_polling_mode()
        set_tg_commands()

        # PTB i egen tråd: egen loop och stop_signals=None för att undvika signal-hook i tråd
        asyncio.set_event_loop(asyncio.new_event_loop())
        app_tg = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        app_tg.add_handler(CommandHandler("start", cmd_start))
        app_tg.add_handler(CommandHandler("status", cmd_status))
        app_tg.add_handler(CommandHandler("engine_start", cmd_engine_start))
        app_tg.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
        app_tg.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
        app_tg.add_handler(CommandHandler("trailing", cmd_trailing))
        app_tg.add_handler(CommandHandler("orb_on", cmd_orb_on))
        app_tg.add_handler(CommandHandler("orb_off", cmd_orb_off))
        app_tg.add_handler(CommandHandler("pnl", cmd_pnl))
        app_tg.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
        app_tg.add_handler(CommandHandler("panic", cmd_panic))
        app_tg.add_handler(CommandHandler("timeframe", cmd_timeframe))
        app_tg.add_handler(CallbackQueryHandler(on_button))

        log.info("[TG] run_polling startar …")
        app_tg.run_polling(drop_pending_updates=True, stop_signals=None)
    except Exception as e:
        log.exception("[TG] Worker kraschade: %s", e)

_started = False
_start_lock = threading.Lock()

@app.on_event("startup")
def on_startup():
    global _started
    with _start_lock:
        if _started:
            return
        # Kicka motor
        threading.Thread(target=engine_worker, name="engine", daemon=True).start()
        # Kicka Telegram
        threading.Thread(target=telegram_worker, name="telegram", daemon=True).start()
        _started = True
        log.info("Startup klart: engine + telegram trådar startade.")
