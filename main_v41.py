# main_v41.py
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
from telegram.ext import (
    Application, CallbackQueryHandler, MessageHandler, CommandHandler, ContextTypes, filters
)

# --------------------------
# LOGGING
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("mp_orbbot_v41")

# --------------------------
# ENV
# --------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
CHAT_ID = os.getenv("CHAT_ID", "").strip()  # valfri push

if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://din-app.onrender.com)")

# --------------------------
# KONFIG
# --------------------------
SYMBOLS: List[str] = ["BTC-USDT", "ETH-USDT", "ADA-USDT", "XRP-USDT", "LINK-USDT"]
TIMEFRAME = "3min"
TRADE_SIZE_USDT = 30
MIN_GREEN_BODY_PCT = 0.05
ENGINE_TICK_SEC = 10

# --------------------------
# APPs
# --------------------------
app = FastAPI(title="MP ORB BOT v41")
ku = Market(url="https://api.kucoin.com")
tg_app = Application.builder().token(BOT_TOKEN).build()

# --------------------------
# STATE
# --------------------------
engine_running: bool = False
open_trades: Dict[str, Dict] = {}
pnl_history: List[Dict] = []
last_index_cache: Dict[str, int] = {s: -1 for s in SYMBOLS}

# Hålla EN panel i chatten så knappar alltid är nederst
main_message_id: Optional[int] = None
main_chat_id: Optional[int] = None

# --------------------------
# UI – BARA NEDER-KNAPPAR
# --------------------------
def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Start Engine", callback_data="engine_on"),
        InlineKeyboardButton("Stop Engine",  callback_data="engine_off"),
        InlineKeyboardButton("PnL",          callback_data="pnl"),
    ]])

async def ensure_panel(update: Update) -> None:
    """Skapa/återskapa panelen i den chat du skriver från."""
    global main_message_id, main_chat_id
    chat = update.effective_chat
    if not chat:
        return
    if main_message_id is None or main_chat_id != chat.id:
        msg = await chat.send_message("MP ORBbot – kontrollpanel", reply_markup=main_keyboard())
        main_message_id, main_chat_id = msg.message_id, chat.id

# Alla inkommande meddelanden: ignorera texten, visa/uppdatera panelen
async def ignore_and_show_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_panel(update)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Gör samma sak som v36: svara i chatten + visa panelen
    await update.message.reply_text("MP ORBbot – redo ✅", reply_markup=main_keyboard())
    await ensure_panel(update)

async def btn_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_running
    q = update.callback_query
    await q.answer()
    await ensure_panel(update)  # säkerställ att panelen finns

    if q.data == "engine_on":
        engine_running = True
        # uppdatera kontrollpanelen (enbart nederknappar)
        await tg_app.bot.edit_message_text(
            chat_id=main_chat_id, message_id=main_message_id,
            text="Engine STARTED ✅", reply_markup=main_keyboard()
        )
    elif q.data == "engine_off":
        engine_running = False
        await tg_app.bot.edit_message_text(
            chat_id=main_chat_id, message_id=main_message_id,
            text="Engine STOPPED ⛔", reply_markup=main_keyboard()
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
        await tg_app.bot.edit_message_text(
            chat_id=main_chat_id, message_id=main_message_id,
            text=msg, reply_markup=main_keyboard()
        )

# Endast nederknappar + /start + generisk message handler
tg_app.add_handler(CallbackQueryHandler(btn_handler))
tg_app.add_handler(CommandHandler("start", start_cmd))
tg_app.add_handler(MessageHandler(filters.ALL, ignore_and_show_panel))

async def tg_push(text: str) -> None:
    if not CHAT_ID:
        return
    try:
        await tg_app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        log.warning(f"TG push fail: {e}")

# --------------------------
# STRATEGY (ORB/entry enligt dina regler)
# --------------------------
def _df_from_klines(klines: List[List]) -> pd.DataFrame:
    # KuCoin: [time, open, close, high, low, volume, turnover]
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
    """
    Din ORB:
      - hitta den FÖRSTA gröna candle som kommer direkt efter en röd
      - den gröna måste ha minsta kropp MIN_GREEN_BODY_PCT
      - ORB-high/low sätts av DEN gröna candlen
    """
    if len(df) < 2:
        return None
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        if is_red(prev) and is_green(curr) and body_pct(curr) >= MIN_GREEN_BODY_PCT:
            return {"high": float(curr["high"]), "low": float(curr["low"]), "index": i}
    return None

def find_entry_after_orb(df: pd.DataFrame, orb: Dict, after_index: int) -> Optional[Dict]:
    """
    Entry-regel:
      - köp ENDAST när EN SENARE candle (2,3,4,...) STÄNGER ÖVER ORB-high
      - dvs i > orb['index'] och df.close[i] > ORB-high
    """
    start = max(orb["index"] + 1, after_index + 1)
    for i in range(start, len(df)):
        c = df.iloc[i]
        if c["close"] > orb["high"]:
            return {"i": i, "price": float(c["close"])}
    return None

def trail_stop_to_last_candle_low(trade: Dict, last_candle: pd.Series) -> None:
    """
    Stoppen följer med candle för candle (uppåt):
      - ny stop = max(gammal stop, senaste candle.low)
    """
    trade["stop"] = max(float(trade["stop"]), float(last_candle["low"]))

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
# ENGINE
# --------------------------
async def engine_loop():
    await asyncio.sleep(1)
    log.info("Engine loop startad.")
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

                        # 3) TRAIL/STOP
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
# FASTAPI ROUTER
# --------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "MP ORBbot v41 OK"

@app.get("/health", response_class=JSONResponse)
async def health():
    return {
        "ok": True,
        "engine_running": engine_running,
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME
    }

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

# Hjälprutt för att test-sända en notis till CHAT_ID
@app.get("/poke")
async def poke():
    if not CHAT_ID:
        return {"ok": False, "msg": "Ingen CHAT_ID satt."}
    try:
        await tg_push("Poke från servern 👋")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "err": str(e)}

# --------------------------
# STARTUP/SHUTDOWN
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
    await tg_app.start()
    log.info(f"Telegram webhook satt till: {url}")

    asyncio.create_task(engine_loop())
    log.info("Startup klar.")

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await tg_app.bot.delete_webhook()
    except Exception:
        pass
    await tg_app.stop()
    await tg_app.shutdown()
    log.info("Shutdown klar.")
