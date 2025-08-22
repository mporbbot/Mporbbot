# main_v40.py
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from kucoin.client import Market
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# --------------------------
# LOGGING
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("mp_orbbot_v40")

# --------------------------
# ENV & SETTINGS
# --------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
CHAT_ID = os.getenv("CHAT_ID", "").strip()  # valfri pushkanal

if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN i miljövariablerna.")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://din-app.onrender.com).")

# Handel
SYMBOLS: List[str] = ["BTC-USDT", "ETH-USDT", "ADA-USDT", "XRP-USDT", "LINK-USDT"]
TIMEFRAME = "3min"
TRADE_SIZE_USDT = 30
MIN_GREEN_BODY_PCT = 0.05   # minsta kropp (%) för att gröna ORB-candlen ska gälla
ENGINE_TICK_SEC = 10

# --------------------------
# APPS
# --------------------------
app = FastAPI(title="MP ORB BOT v40")
ku = Market(url="https://api.kucoin.com")
tg_app = Application.builder().token(BOT_TOKEN).build()

# --------------------------
# STATE
# --------------------------
engine_running: bool = False
open_trades: Dict[str, Dict] = {}   # symbol -> {"entry","stop","entry_time","last_i"}
pnl_history: List[Dict] = []        # lista av avslut
last_index_cache: Dict[str, int] = {s: -1 for s in SYMBOLS}

# --------------------------
# DATA-HJÄLP
# --------------------------
def _df_from_klines(klines: List[List]) -> pd.DataFrame:
    cols = ["time", "open", "close", "high", "low", "volume", "turnover"]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open", "close", "high", "low", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.dropna().reset_index(drop=True)

def is_green(c): return c["close"] > c["open"]
def is_red(c):   return c["close"] < c["open"]
def body_pct(c): return abs((c["close"] - c["open"]) / c["open"]) * 100.0

def detect_orb(df: pd.DataFrame) -> Optional[Dict]:
    """Första GRÖNA direkt efter en RÖD, med minst kropp MIN_GREEN_BODY_PCT.
       ORB-high/low tas från den gröna candlen."""
    if len(df) < 2:
        return None
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        if is_red(prev) and is_green(curr) and body_pct(curr) >= MIN_GREEN_BODY_PCT:
            return {"high": float(curr["high"]), "low": float(curr["low"]), "index": i}
    return None

def find_entry_after_orb(df: pd.DataFrame, orb: Dict, after_index: int) -> Optional[Dict]:
    """Köp först när en SENARE candle (2,3,4,...) STÄNGER ÖVER ORB-high."""
    start = max(orb["index"] + 1, after_index + 1)
    for i in range(start, len(df)):
        c = df.iloc[i]
        if c["close"] > orb["high"]:
            return {"i": i, "price": float(c["close"])}
    return None

def trail_stop_to_last_candle_low(trade: Dict, last_candle: pd.Series) -> None:
    trade["stop"] = max(float(trade["stop"]), float(last_candle["low"]))

# --------------------------
# TRADES
# --------------------------
def _size_from_entry(entry_price: float) -> float:
    return TRADE_SIZE_USDT / entry_price if entry_price > 0 else 0.0

def close_trade(symbol: str, exit_price: float, reason: str) -> Optional[Dict]:
    trade = open_trades.pop(symbol, None)
    if not trade:
        return None
    size = _size_from_entry(trade["entry"])
    pnl = (exit_price - trade["entry"]) * size
    rec = {
        "symbol": symbol,
        "pnl": float(pnl),
        "exit_price": float(exit_price),
        "entry": float(trade["entry"]),
        "time": datetime.now(timezone.utc),
        "reason": reason,
    }
    pnl_history.append(rec)
    return rec

# --------------------------
# TELEGRAM (endast INLINE-knappar längst ner)
# --------------------------
def main_keyboard() -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton("Start Engine", callback_data="engine_on"),
        InlineKeyboardButton("Stop Engine", callback_data="engine_off"),
        InlineKeyboardButton("PnL", callback_data="pnl"),
    ]
    return InlineKeyboardMarkup([row])

async def tg_push(text: str) -> None:
    if not CHAT_ID:
        return
    try:
        # Skicka UTAN några extra knappar – bara ren text
        await tg_app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        log.warning(f"TG push fail: {e}")

# Huvudmeddelandet vi uppdaterar för att hålla knapparna samlade längst ner
main_message_id: Optional[int] = None
main_chat_id: Optional[int] = None

async def ensure_main_message(update: Update) -> None:
    """Säkerställ att vi har ett huvudmeddelande med knapparna längst ner."""
    global main_message_id, main_chat_id
    chat_id = update.effective_chat.id
    if main_message_id is None or main_chat_id != chat_id:
        msg = await update.effective_chat.send_message(
            "MP ORBbot – kontrollpanel",
            reply_markup=main_keyboard()
        )
        main_message_id, main_chat_id = msg.message_id, chat_id

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_main_message(update)

async def btn_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_running
    q = update.callback_query
    await q.answer()

    # säkerställ att vi alltid editerar "huvudmeddelandet" (med knapparna)
    global main_message_id, main_chat_id
    if main_message_id is None or main_chat_id != q.message.chat_id:
        msg = await q.message.chat.send_message("MP ORBbot – kontrollpanel", reply_markup=main_keyboard())
        main_message_id, main_chat_id = msg.message_id, q.message.chat_id

    if q.data == "engine_on":
        engine_running = True
        await q.message.chat.edit_message_text(
            "Engine STARTED ✅", message_id=main_message_id, reply_markup=main_keyboard()
        )
    elif q.data == "engine_off":
        engine_running = False
        await q.message.chat.edit_message_text(
            "Engine STOPPED ⛔", message_id=main_message_id, reply_markup=main_keyboard()
        )
    elif q.data == "pnl":
        if not pnl_history:
            msg = "Inga avslutade trades ännu."
        else:
            total = sum(x["pnl"] for x in pnl_history)
            lines = [f"Totalt PnL: {total:.2f} USDT"]
            for x in pnl_history[-10:]:
                lines.append(
                    f"{x['symbol']}  PnL {x['pnl']:.2f}  "
                    f"({x['reason']})  entry {x['entry']:.4f} → exit {x['exit_price']:.4f}"
                )
            msg = "\n".join(lines)
        await q.message.chat.edit_message_text(
            msg, message_id=main_message_id, reply_markup=main_keyboard()
        )

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CallbackQueryHandler(btn_handler))

# --------------------------
# ENGINE
# --------------------------
async def engine_loop():
    await asyncio.sleep(1)
    log.info("Engine loop började.")
    while True:
        try:
            if engine_running:
                for symbol in SYMBOLS:
                    try:
                        kl = ku.get_kline(symbol, TIMEFRAME)
                        if not kl or len(kl) < 3:
                            continue
                        df = _df_from_klines(kl)

                        # 1) ORB
                        orb = detect_orb(df)

                        # 2) ENTRY
                        if orb:
                            ent = find_entry_after_orb(df, orb, last_index_cache.get(symbol, -1))
                            if ent and symbol not in open_trades:
                                i = ent["i"]
                                entry_px = ent["price"]
                                last_index_cache[symbol] = i
                                open_trades[symbol] = {
                                    "entry": entry_px,
                                    "stop": float(orb["low"]),
                                    "entry_time": datetime.now(timezone.utc),
                                    "last_i": i,
                                }
                                msg = f"ENTRY {symbol} @ {entry_px:.6f} | stop {orb['low']:.6f}"
                                log.info(msg)
                                await tg_push(msg)

                        # 3) TRAIL + STOP
                        if symbol in open_trades:
                            last_c = df.iloc[-1]
                            trade = open_trades[symbol]
                            prev_stop = trade["stop"]
                            trail_stop_to_last_candle_low(trade, last_c)
                            new_stop = trade["stop"]
                            last_price = float(last_c["close"])

                            if last_price < new_stop:
                                res = close_trade(symbol, last_price, reason="STOP")
                                if res:
                                    msg = (f"EXIT {symbol} @ {res['exit_price']:.6f} | "
                                           f"PnL {res['pnl']:.2f} (STOP)")
                                    log.info(msg)
                                    await tg_push(msg)
                            elif new_stop > prev_stop:
                                log.info(f"TRAIL {symbol}: stop {prev_stop:.6f} → {new_stop:.6f}")

                    except Exception as syme:
                        log.error(f"{symbol} fel: {syme}")
            await asyncio.sleep(ENGINE_TICK_SEC)
        except Exception as e:
            log.error(f"Engine-loop fel: {e}")
            await asyncio.sleep(ENGINE_TICK_SEC)

# --------------------------
# FASTAPI
# --------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "MP ORBbot v40 OK"

@app.get("/health", response_class=JSONResponse)
async def health():
    return {"ok": True, "engine_running": engine_running, "symbols": SYMBOLS, "timeframe": TIMEFRAME}

@app.post("/webhook/{token}")
async def tg_webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")
    data = await request.json()
    try:
        update = Update.de_json(data, tg_app.bot)
        await tg_app.process_update(update)
    except Exception as e:
        log.error(f"Webhook update error: {e}")
    return {"ok": True}

# --------------------------
# LIFECYCLE
# --------------------------
@app.on_event("startup")
async def on_startup():
    await tg_app.initialize()
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    try:
        await tg_app.bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass
    await tg_app.bot.set_webhook(url=url, allowed_updates=["message", "callback_query"])
    log.info(f"Telegram webhook satt till: {url}")
    asyncio.create_task(engine_loop())
    log.info("Startup klar.")

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await tg_app.bot.delete_webhook()
    except Exception:
        pass
    await tg_app.shutdown()
    log.info("Shutdown klar.")
