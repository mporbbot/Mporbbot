# main_v37_fix.py
import os
import json
import time
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler

# ========= Konfig =========
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")
if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

SYMBOL = os.getenv("SYMBOL", "BTC-USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "20min")
ENTRY_MODE = os.getenv("ENTRY_MODE", "close").lower()  # "close" eller "tick"
ENGINE_ON = os.getenv("ENGINE_ON", "false").lower() == "true"

# ORB-regel: röd candle följt av grön candle med minsta kropp
MIN_GREEN_BODY_PCT = float(os.getenv("MIN_GREEN_BODY_PCT", "0.02"))  # 2%

FETCH_INTERVAL_SEC = int(os.getenv("FETCH_INTERVAL_SEC", "8"))  # pollingintervall
CHAT_ID: Optional[int] = None  # fylls när /start körs (för meddelanden)

# ========= Globalt state =========
state: Dict[str, Any] = {
    "engine_on": ENGINE_ON,
    "entry_mode": ENTRY_MODE,        # "close" eller "tick" (vi använder close)
    "timeframe": TIMEFRAME,
    "symbol": SYMBOL,

    # ORB
    "orb_high": None,
    "orb_low": None,
    "orb_index": None,               # index för den gröna candle som slutför ORB

    # Position
    "position": None,                # dict: {"side":"long","entry":float,"stop":float,"qty":float,"time":ts}
    "realized_pnl": 0.0,

    # intern cache
    "last_candle_ts": None,
    "debug": False,
}

# ========= Telegram / FastAPI =========
app = FastAPI()
tg_app: Application = Application.builder().token(BOT_TOKEN).build()

def kb_main() -> InlineKeyboardMarkup:
    # Endast knapparna där nere (inga “topp-knappar”)
    rows = [
        [
            InlineKeyboardButton("Engine ON", callback_data="engine_on"),
            InlineKeyboardButton("Engine OFF", callback_data="engine_off"),
        ],
        [
            InlineKeyboardButton("Entry: close", callback_data="entry_close"),
            InlineKeyboardButton("Entry: tick", callback_data="entry_tick"),
        ],
        [
            InlineKeyboardButton("TF: 1m", callback_data="tf_1min"),
            InlineKeyboardButton("TF: 5m", callback_data="tf_5min"),
            InlineKeyboardButton("TF: 15m", callback_data="tf_15min"),
            InlineKeyboardButton("TF: 20m", callback_data="tf_20min"),
        ],
        [
            InlineKeyboardButton("PNL", callback_data="pnl"),
            InlineKeyboardButton("Status", callback_data="status"),
            InlineKeyboardButton("Debug ON", callback_data="debug_on"),
            InlineKeyboardButton("Debug OFF", callback_data="debug_off"),
        ],
    ]
    return InlineKeyboardMarkup(rows)

async def safe_send(text: str):
    if CHAT_ID is None:
        return
    try:
        await tg_app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception:
        pass

def dbg(msg: str):
    if state["debug"]:
        asyncio.create_task(safe_send(f"[DEBUG] {msg}"))

# ======== KuCoin candles ========
KU_PUBLIC = "https://api.kucoin.com"

async def fetch_klines(symbol: str, tf: str, limit: int = 200):
    # KuCoin API: GET /api/v1/market/candles?symbol=BTC-USDT&type=1min
    # Retur: [[time, open, close, high, low, volume, turnover], ...] i reverse chrono
    type_map = {
        "1m": "1min", "1min": "1min",
        "5m": "5min", "5min": "5min",
        "15m": "15min", "15min": "15min",
        "20m": "20min", "20min": "20min",
        "1h": "1hour", "1hour": "1hour",
    }
    tf_api = type_map.get(tf.lower(), tf)
    url = f"{KU_PUBLIC}/api/v1/market/candles"
    params = {"symbol": symbol, "type": tf_api}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # reverse chrono
        # Gör om till stigande tid
        data = list(reversed(data))[-limit:]
        candles = []
        for row in data:
            ts = int(row[0])  # sekund
            o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
            candles.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
        return candles

# ======== ORB/Entry/Trail/Exit ========
def try_set_orb(prev, cur, idx: int):
    """ Sätter ORB när vi har röd -> grön och den gröna har min kropp. """
    if state["orb_index"] is not None:
        return
    if prev["close"] < prev["open"] and cur["close"] > cur["open"]:
        body = (cur["close"] - cur["open"]) / cur["open"]
        if body >= MIN_GREEN_BODY_PCT:
            state["orb_high"] = max(prev["high"], cur["high"])
            state["orb_low"] = min(prev["low"], cur["low"])
            state["orb_index"] = idx
            dbg(f"Set ORB @i={idx} high={state['orb_high']:.8f} low={state['orb_low']:.8f} body={body*100:.2f}%")

def eligible_entry(i: int, cur) -> bool:
    """ Entry tidigast candle #2 efter ORB, candle grön, close > orb_high. """
    if state["orb_index"] is None:
        return False
    if i - state["orb_index"] < 2:
        return False
    if not (cur["close"] > cur["open"]):
        return False
    if not (cur["close"] > state["orb_high"]):
        return False
    return True

async def open_long(price: float):
    qty = 1.0  # mock kvantitet (du kan byta mot riktig sizing)
    state["position"] = {
        "side": "long",
        "entry": price,
        "stop": None,
        "qty": qty,
        "time": int(time.time()),
    }
    await safe_send(f"ENTRY LONG {state['symbol']} @ {price:.6f}")

def trail_stop(prev_candle):
    """ Flytta SL upp till föregående candles low om det höjer stop. Aldrig sänka. """
    pos = state["position"]
    if not pos or pos["side"] != "long":
        return
    new_sl = prev_candle["low"]
    if pos["stop"] is None or new_sl > pos["stop"]:
        pos["stop"] = new_sl
        dbg(f"Trail SL -> {new_sl:.8f} (prev.low)")

async def check_exit(cur):
    """ Stäng position om low <= stop. """
    pos = state["position"]
    if not pos:
        return
    if pos["side"] == "long" and pos["stop"] is not None:
        if cur["low"] <= pos["stop"]:
            exit_price = pos["stop"]
            pnl = (exit_price - pos["entry"]) * pos["qty"]
            state["realized_pnl"] += pnl
            await safe_send(f"EXIT LONG @ {exit_price:.6f} | PnL: {pnl:.6f} | Cum PnL: {state['realized_pnl']:.6f}")
            state["position"] = None
            # ORB kan antingen få ligga kvar eller nollas – vi låter den ligga kvar (din tidigare setup)

# ======== Engine loop ========
async def engine_loop():
    await asyncio.sleep(1)  # kort delay för att appen ska hinna starta
    dbg("Engine loop started")
    while True:
        try:
            if not state["engine_on"]:
                await asyncio.sleep(FETCH_INTERVAL_SEC)
                continue

            candles = await fetch_klines(state["symbol"], state["timeframe"], limit=300)
            if not candles or len(candles) < 3:
                await asyncio.sleep(FETCH_INTERVAL_SEC)
                continue

            # Processa nya candles (baserat på ts)
            last_ts = state["last_candle_ts"]
            # Vi kör logiken på sista fullt stängda candle (dvs näst sista i listan)
            # och använder prev för trail osv.
            for i in range(len(candles) - 1):  # lämna sista som "pågående"
                c = candles[i]
                if last_ts is not None and c["ts"] <= last_ts:
                    continue

                if i >= 1:
                    prev = candles[i - 1]
                    try_set_orb(prev, c, i)

                    # Trail upp SL (på varje ny stängd candle)
                    if state["position"]:
                        trail_stop(prev)

                    # Exit-check (ifall SL träffas på denna candle)
                    # OBS: vi tittar på cur.low mot stop – i realtid hade detta skett intrabar.
                    await check_exit(c)

                    # Entry enligt regel (endast close-läge stöds här – tick lämnas orörd)
                    if state["position"] is None and state["entry_mode"] == "close":
                        if eligible_entry(i, c):
                            await open_long(c["close"])

                state["last_candle_ts"] = c["ts"]

        except Exception as e:
            await safe_send(f"[Engine error] {e!r}")
        await asyncio.sleep(FETCH_INTERVAL_SEC)

# ======== Telegram handlers ========
async def start_cmd(update, context):
    global CHAT_ID
    CHAT_ID = update.effective_chat.id
    txt = (
        "MPorb v37-fix\n"
        f"Symbol: {state['symbol']} | TF: {state['timeframe']}\n"
        f"Engine: {'ON' if state['engine_on'] else 'OFF'} | Entry: {state['entry_mode']}\n"
        f"ORB: high={state['orb_high']} low={state['orb_low']} idx={state['orb_index']}\n"
        f"Realized PnL: {state['realized_pnl']:.6f}\n"
        f"Min green body: {MIN_GREEN_BODY_PCT*100:.1f}%\n"
        "—\n"
        "Styr allt via knapparna nedan."
    )
    await update.message.reply_text(txt, reply_markup=kb_main())

async def status_cmd(update, context):
    txt = (
        f"Engine: {'ON' if state['engine_on'] else 'OFF'}\n"
        f"Entry mode: {state['entry_mode']}\n"
        f"Symbol: {state['symbol']} | TF: {state['timeframe']}\n"
        f"ORB: high={state['orb_high']} low={state['orb_low']} idx={state['orb_index']}\n"
        f"Position: {json.dumps(state['position'])}\n"
        f"Realized PnL: {state['realized_pnl']:.6f}"
    )
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.edit_text(txt, reply_markup=kb_main())
    else:
        await update.message.reply_text(txt, reply_markup=kb_main())

async def callback_handler(update, context):
    q = update.callback_query
    data = q.data
    await q.answer()
    if data == "engine_on":
        state["engine_on"] = True
        await q.message.edit_text("Engine: ON", reply_markup=kb_main())
    elif data == "engine_off":
        state["engine_on"] = False
        await q.message.edit_text("Engine: OFF", reply_markup=kb_main())
    elif data == "entry_close":
        state["entry_mode"] = "close"
        await q.message.edit_text("Entry mode: close", reply_markup=kb_main())
    elif data == "entry_tick":
        state["entry_mode"] = "tick"
        await q.message.edit_text("Entry mode: tick (ej använd i denna fil)", reply_markup=kb_main())
    elif data == "tf_1min":
        state["timeframe"] = "1min"
        await q.message.edit_text("Timeframe: 1min", reply_markup=kb_main())
    elif data == "tf_5min":
        state["timeframe"] = "5min"
        await q.message.edit_text("Timeframe: 5min", reply_markup=kb_main())
    elif data == "tf_15min":
        state["timeframe"] = "15min"
        await q.message.edit_text("Timeframe: 15min", reply_markup=kb_main())
    elif data == "tf_20min":
        state["timeframe"] = "20min"
        await q.message.edit_text("Timeframe: 20min", reply_markup=kb_main())
    elif data == "pnl":
        await q.message.edit_text(f"Cumulative PnL: {state['realized_pnl']:.6f}", reply_markup=kb_main())
    elif data == "status":
        await status_cmd(update, context)
    elif data == "debug_on":
        state["debug"] = True
        await q.message.edit_text("DEBUG: ON", reply_markup=kb_main())
    elif data == "debug_off":
        state["debug"] = False
        await q.message.edit_text("DEBUG: OFF", reply_markup=kb_main())

# ======== FastAPI endpoints & startup ========
class TelegramUpdate(BaseModel):
    update_id: Optional[int] = None

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return Response(status_code=403)
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.update_queue.put(update)
    return Response(status_code=200)

@app.get("/")
async def root():
    return {"ok": True, "bot": "mporb v37-fix", "engine_on": state["engine_on"]}

@app.on_event("startup")
async def on_startup():
    # Telegram handlers
    tg_app.add_handler(CommandHandler("start", start_cmd))
    tg_app.add_handler(CommandHandler("status", status_cmd))
    tg_app.add_handler(CallbackQueryHandler(callback_handler))

    # Sätt webhook
    await tg_app.bot.set_webhook(url=f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")

    # Starta telegram-loop + engine-loop
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())
    asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await tg_app.stop()
        await tg_app.shutdown()
    except Exception:
        pass
