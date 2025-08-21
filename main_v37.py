# main_v37.py
import asyncio
import os
from typing import Dict, Any, Optional, List, Tuple

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    CallbackQueryHandler, ContextTypes
)

from kucoin.client import Market

# -----------------------------
# Miljö & grundkonfig
# -----------------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOL = os.getenv("SYMBOL", "BTC-USDT").upper()
BASE_USDT = float(os.getenv("BASE_USDT", "50"))
MOCK = os.getenv("MOCK", "1") != "0"

if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

# -----------------------------
# Globalt tillstånd
# -----------------------------
state: Dict[str, Any] = {
    "engine_on": False,
    "entry_mode": "close",         # "close" eller "tick"
    "stop_mode": "candle",         # "candle" eller "none"
    "timeframe": "20m",
    "orb": None,                   # {"time","index","high","low"}
    "position": None,              # {"entry","qty","stop","entry_time"}
    "realized_pnl": 0.0,
    "last_chat_id": None,
    "symbol": SYMBOL,
}

market = Market(url='https://api.kucoin.com')

# -----------------------------
# Hjälp: klines & resampling
# -----------------------------
KC_NATIVE = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1hour", "2h": "2hour", "4h": "4hour", "6h": "6hour", "8h": "8hour",
    "12h": "12hour", "1d": "1day", "1w": "1week"
}

def _kucoin_interval(tf: str) -> Tuple[str, int]:
    tf = tf.lower().replace("min", "m").replace("hour", "h")
    if tf in KC_NATIVE:
        return KC_NATIVE[tf], 1
    if tf in ("20m", "20min"):
        return KC_NATIVE["5m"], 4
    return KC_NATIVE["5m"], 1

def _candle_from_rows(rows: List[List[float]]) -> Dict[str, Any]:
    t = int(rows[0][0])
    o = float(rows[0][1])
    c = float(rows[-1][2])
    h = max(float(r[3]) for r in rows)
    l = min(float(r[4]) for r in rows)
    return {"t": t, "o": o, "h": h, "l": l, "c": c}

def _resample_klines(raw: List[List[str]], factor: int) -> List[Dict[str, Any]]:
    rows = [[int(r[0])*1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in raw]
    rows.sort(key=lambda x: x[0])
    out = []
    if factor <= 1:
        for r in rows:
            out.append({"t": r[0], "o": r[1], "h": r[3], "l": r[4], "c": r[2]})
        return out
    for i in range(0, len(rows), factor):
        chunk = rows[i:i+factor]
        if len(chunk) < factor:
            continue
        out.append(_candle_from_rows(chunk))
    return out

def get_klines(symbol: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    native, factor = _kucoin_interval(tf)
    fetch_limit = limit * factor if factor > 1 else limit
    data = market.get_kline(symbol, native, None, None, fetch_limit)
    candles = _resample_klines(data, factor)
    return candles[-limit:]

async def get_last_price(symbol: str) -> float:
    try:
        t = market.get_ticker(symbol)
        return float(t["price"])
    except Exception:
        kl = get_klines(symbol, "1m", 1)
        return kl[-1]["c"] if kl else 0.0

# -----------------------------
# ORB-regler (första gröna candle)
# -----------------------------
def find_orb_first_green(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for idx, c in enumerate(candles):
        if c["c"] > c["o"]:
            return {"time": c["t"], "index": idx, "high": c["h"], "low": c["l"]}
    return None

def is_green(c: Dict[str, Any], min_body_pct: float = 0.0) -> bool:
    body = c["c"] - c["o"]
    if body <= 0:
        return False
    rng = max(c["h"] - c["l"], 1e-12)
    return (body / rng) >= min_body_pct

# -----------------------------
# Tradinglogik (mock)
# -----------------------------
async def maybe_enter_long(candles: List[Dict[str, Any]], min_body_pct: float = 0.0):
    orb = state["orb"]
    if orb is None or state["position"] is not None:
        return

    latest = candles[-1]
    # måste vara efter orb-index
    if candles.index(latest) <= orb["index"]:
        return

    # close > ORB-high + grön kropp
    if latest["c"] > orb["high"] and is_green(latest, min_body_pct=min_body_pct):
        price = latest["c"]
        qty = BASE_USDT / price
        stop = latest["l"]  # initial stop = entrycandlens low
        state["position"] = {"entry": price, "qty": qty, "stop": stop, "entry_time": latest["t"]}
        await notify(f"🚀 ENTRY LONG {state['symbol']} @ {price:.4f}\nORB H:{orb['high']:.4f}  L:{orb['low']:.4f}\nStop: {stop:.4f}")

async def update_trailing_and_exit(candles: List[Dict[str, Any]]):
    pos = state["position"]
    if pos is None:
        return

    latest = candles[-1]
    prev = candles[-2] if len(candles) >= 2 else None

    if state["stop_mode"] == "candle" and prev:
        cand_stop = prev["l"]
        if cand_stop > pos["stop"]:
            pos["stop"] = cand_stop
            await notify(f"🔒 Trailing stop uppdaterad: {pos['stop']:.4f}")

    if latest["c"] <= pos["stop"]:
        exit_price = latest["c"]
        pnl = (exit_price - pos["entry"]) * pos["qty"]
        state["realized_pnl"] += pnl
        await notify(f"🏁 EXIT {state['symbol']} @ {exit_price:.4f}\nPnL: {pnl:.4f} USDT  (Tot: {state['realized_pnl']:.4f})")
        state["position"] = None

# -----------------------------
# Telegram UI (inline-knappar)
# -----------------------------
def build_keyboard() -> InlineKeyboardMarkup:
    eng = "ON" if state["engine_on"] else "OFF"
    entry = "Close" if state["entry_mode"] == "close" else "Tick"
    stop = "Candle" if state["stop_mode"] == "candle" else "None"

    row1 = [
        InlineKeyboardButton(f"Engine {eng}", callback_data="toggle_engine"),
        InlineKeyboardButton(f"Entry: {entry}", callback_data="toggle_entry"),
        InlineKeyboardButton(f"Stop: {stop}", callback_data="toggle_stop"),
    ]
    row2 = [
        InlineKeyboardButton("TF 5m", callback_data="tf:5m"),
        InlineKeyboardButton("TF 15m", callback_data="tf:15m"),
        InlineKeyboardButton("TF 20m", callback_data="tf:20m"),
        InlineKeyboardButton("TF 30m", callback_data="tf:30m"),
        InlineKeyboardButton("TF 1h", callback_data="tf:1h"),
    ]
    row3 = [
        InlineKeyboardButton("Status", callback_data="status"),
        InlineKeyboardButton("PnL", callback_data="pnl"),
        InlineKeyboardButton("Reset ORB", callback_data="reset_orb"),
    ]
    return InlineKeyboardMarkup([row1, row2, row3])

async def notify(text: str):
    chat_id = state.get("last_chat_id")
    if not chat_id:
        return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text, reply_markup=build_keyboard())
    except Exception:
        pass

def remember(fn):
    async def inner(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat:
            state["last_chat_id"] = update.effective_chat.id
        return await fn(update, context)
    return inner

@remember
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state["last_chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        f"MPORB Bot – {state['symbol']}\n"
        "• ORB: första gröna candle\n"
        "• Entry: candle CLOSE över ORB-high\n"
        "• Trailing stop: föregående candles low\n"
        "• Spot long (mock)\n",
        reply_markup=build_keyboard()
    )

@remember
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pos = state["position"]; orb = state["orb"]
    txt = [
        f"Engine: {'ON' if state['engine_on'] else 'OFF'}",
        f"Symbol: {state['symbol']}",
        f"TF: {state['timeframe']}",
        f"Entry mode: {state['entry_mode']}",
        f"Stop mode: {state['stop_mode']}",
        f"PnL Tot: {state['realized_pnl']:.4f} USDT",
    ]
    if orb:
        txt.append(f"ORB H:{orb['high']:.4f} L:{orb['low']:.4f} (idx {orb['index']})")
    else:
        txt.append("ORB: –")
    if pos:
        txt.append(f"Pos: entry {pos['entry']:.4f}, stop {pos['stop']:.4f}, qty {pos['qty']:.6f}")
    else:
        txt.append("Pos: –")
    await update.message.reply_text("\n".join(txt), reply_markup=build_keyboard())

@remember
async def engine_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state["engine_on"] = True
    await update.message.reply_text("Engine ON", reply_markup=build_keyboard())

@remember
async def engine_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state["engine_on"] = False
    await update.message.reply_text("Engine OFF", reply_markup=build_keyboard())

@remember
async def pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Realiserad PnL: {state['realized_pnl']:.4f} USDT", reply_markup=build_keyboard())

@remember
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    await query.answer()
    if data == "toggle_engine":
        state["engine_on"] = not state["engine_on"]
        await query.edit_message_text(f"Engine {'ON' if state['engine_on'] else 'OFF'}", reply_markup=build_keyboard())
    elif data == "toggle_entry":
        state["entry_mode"] = "tick" if state["entry_mode"] == "close" else "close"
        await query.edit_message_text(f"Entry-läge: {state['entry_mode']}", reply_markup=build_keyboard())
    elif data == "toggle_stop":
        state["stop_mode"] = "none" if state["stop_mode"] == "candle" else "candle"
        await query.edit_message_text(f"Stop-läge: {state['stop_mode']}", reply_markup=build_keyboard())
    elif data.startswith("tf:"):
        tf = data.split(":", 1)[1]
        state["timeframe"] = tf
        state["orb"] = None
        await query.edit_message_text(f"Timeframe satt: {tf}", reply_markup=build_keyboard())
    elif data == "status":
        await query.message.reply_text("Status begärd.", reply_markup=build_keyboard())
    elif data == "pnl":
        await query.message.reply_text(f"Realiserad PnL: {state['realized_pnl']:.4f} USDT", reply_markup=build_keyboard())
    elif data == "reset_orb":
        state["orb"] = None
        await query.edit_message_text("ORB nollställd – ny första grön candle krävs.", reply_markup=build_keyboard())

# -----------------------------
# Motor-loop
# -----------------------------
async def engine_loop():
    MIN_GREEN_BODY_PCT = 0.0
    while True:
        try:
            if state["engine_on"]:
                candles = get_klines(state["symbol"], state["timeframe"], limit=200)
                if len(candles) >= 3:
                    if state["orb"] is None:
                        orb = find_orb_first_green(candles)
                        if orb:
                            state["orb"] = orb
                            await notify(f"🟢 ORB funnen: H {orb['high']:.4f} / L {orb['low']:.4f}")

                    if state["entry_mode"] == "close":
                        await maybe_enter_long(candles, min_body_pct=MIN_GREEN_BODY_PCT)
                    else:
                        if state["orb"] and state["position"] is None:
                            px = await get_last_price(state["symbol"])
                            if px > state["orb"]["high"] and is_green(candles[-1], min_body_pct=MIN_GREEN_BODY_PCT):
                                qty = BASE_USDT / px
                                stop = candles[-1]["l"]
                                state["position"] = {"entry": px, "qty": qty, "stop": stop, "entry_time": candles[-1]["t"]}
                                await notify(f"⚡️ ENTRY (tick) LONG {state['symbol']} @ {px:.4f}\nStop: {stop:.4f}")

                    await update_trailing_and_exit(candles)
            await asyncio.sleep(5)
        except Exception as e:
            print("engine_loop error:", e)
            await asyncio.sleep(5)

# -----------------------------
# FastAPI + webhook
# -----------------------------
app = FastAPI()
tg_app: Application = ApplicationBuilder().token(BOT_TOKEN).build()

tg_app.add_handler(CommandHandler("start", remember(start_cmd)))
tg_app.add_handler(CommandHandler("status", remember(status_cmd)))
tg_app.add_handler(CommandHandler("engine_on", remember(engine_on_cmd)))
tg_app.add_handler(CommandHandler("engine_off", remember(engine_off_cmd)))
tg_app.add_handler(CommandHandler("pnl", remember(pnl_cmd)))
tg_app.add_handler(CallbackQueryHandler(remember(on_button)))

class TelegramUpdate(BaseModel):
    update_id: Optional[int] = None

@app.get("/")
async def root():
    return {"ok": True, "service": "mporbbot", "symbol": state["symbol"]}

@app.post(f"/webhook/{{token}}")
async def webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        raise HTTPException(status_code=404, detail="Not Found")
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.update_queue.put(update)
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    await tg_app.initialize()
    # VIKTIGT: starta worker som läser update_queue
    await tg_app.start()
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    try:
        await tg_app.bot.set_webhook(url=url, drop_pending_updates=True)
        print("Webhook set to:", url)
    except Exception as e:
        print("Failed to set webhook:", e)
    asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await tg_app.bot.delete_webhook()
    except Exception:
        pass
    await tg_app.stop()
    await tg_app.shutdown()
