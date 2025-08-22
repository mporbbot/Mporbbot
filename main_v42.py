# main_v42.py
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from kucoin.client import Market
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# -------------------------- LOGGING --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("mp_orbbot_v42")

# -------------------------- ENV --------------------------
BOT_TOKEN   = os.getenv("BOT_TOKEN", "").strip()
WEBHOOK_BASE= os.getenv("WEBHOOK_BASE", "").rstrip("/")
CHAT_ID     = os.getenv("CHAT_ID", "").strip()   # valfri push

if not BOT_TOKEN:
    raise RuntimeError("Saknar BOT_TOKEN")
if not WEBHOOK_BASE:
    raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://din-app.onrender.com)")

# -------------------------- KONFIG --------------------------
SYMBOLS: List[str] = ["BTC-USDT", "ETH-USDT", "ADA-USDT", "XRP-USDT", "LINK-USDT"]
TIMEFRAME = "3min"            # KuCoin kline
TRADE_SIZE_USDT = 30
MIN_GREEN_BODY_PCT = 0.05     # minsta gröna kropp (%) för ORB-candle
ENGINE_TICK_SEC = 10

# -------------------------- APPs --------------------------
app   = FastAPI(title="MP ORB BOT v42")
ku    = Market(url="https://api.kucoin.com")
tg_app= Application.builder().token(BOT_TOKEN).build()

# -------------------------- STATE --------------------------
engine_running: bool = False
open_trades: Dict[str, Dict] = {}
pnl_history: List[Dict] = []
last_index_cache: Dict[str, int] = {s: -1 for s in SYMBOLS}

# -------------------------- TG helpers --------------------------
async def tg_push(text: str) -> None:
    if not CHAT_ID: return
    try:
        await tg_app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        log.warning(f"TG push fail: {e}")

# -------------------------- Strategi --------------------------
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
    # Första gröna efter röd, med minsta kropp
    if len(df) < 2:
        return None
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        if is_red(prev) and is_green(curr) and body_pct(curr) >= MIN_GREEN_BODY_PCT:
            return {"high": float(curr["high"]), "low": float(curr["low"]), "index": i}
    return None

def find_entry_after_orb(df: pd.DataFrame, orb: Dict, after_index: int) -> Optional[Dict]:
    # Köp när en senare candle stänger över ORB-high
    start = max(orb["index"] + 1, after_index + 1)
    for i in range(start, len(df)):
        c = df.iloc[i]
        if c["close"] > orb["high"]:
            return {"i": i, "price": float(c["close"])}
    return None

def trail_stop_to_last_candle_low(trade: Dict, last_candle: pd.Series) -> None:
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

# -------------------------- ENGINE --------------------------
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

                        # 2) ENTRY – vänta tills en senare candle stänger över ORB-high
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

                        # 3) TRAIL/STOP – stop följer senaste candle low uppåt
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

# -------------------------- Telegram KOMMANDON (endast “nere”-knappar) --------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("MP ORBbot – redo. Använd /status, /engine_on, /engine_off, /pnl, /panic.")
    log.info("/start från %s", update.effective_user.id)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        f"Engine: {'ON' if engine_running else 'OFF'}\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"Öppna positioner: {', '.join(open_trades.keys()) if open_trades else 'inga'}\n"
        f"Tot PnL: {sum(x['pnl'] for x in pnl_history):.2f} USDT"
    )
    await update.message.reply_text(txt)

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_running
    engine_running = True
    await update.message.reply_text("Engine STARTED ✅")

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_running
    engine_running = False
    await update.message.reply_text("Engine STOPPED ⛔")

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    await update.message.reply_text(msg)

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    closed = []
    for sym in list(open_trades.keys()):
        try:
            kl = ku.get_kline(sym, TIMEFRAME)
            df = _df_from_klines(kl)
            px = float(df.iloc[-1]["close"])
            res = close_trade(sym, px, reason="PANIC")
            if res: closed.append(f"{sym} PnL {res['pnl']:.2f}")
        except Exception as e:
            log.error(f"Panic close {sym} fel: {e}")
    if closed:
        await update.message.reply_text("Panic close:\n" + "\n".join(closed))
    else:
        await update.message.reply_text("Inga öppna positioner.")

# “infosvar” så de nedre knapparna känns levande
async def cmd_entrymode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Entry: köp när en senare candle stänger över ORB-high (din regel).")

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Timeframe: {TIMEFRAME}")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ORB är aktiv: första gröna efter röd med minsta kropp.")

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ORB kan inte stängas av i denna version.")

# registrera endast kommandon
tg_app.add_handler(CommandHandler("start",      cmd_start))
tg_app.add_handler(CommandHandler("status",     cmd_status))
tg_app.add_handler(CommandHandler("engine_on",  cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("pnl",        cmd_pnl))
tg_app.add_handler(CommandHandler("panic",      cmd_panic))
tg_app.add_handler(CommandHandler("entrymode",  cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe",  cmd_timeframe))
tg_app.add_handler(CommandHandler("orb_on",     cmd_orb_on))
tg_app.add_handler(CommandHandler("orb_off",    cmd_orb_off))

# -------------------------- FastAPI --------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "MP ORBbot v42 OK"

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
    log.info("Webhook hit ✅")
    try:
        update = Update.de_json(data, tg_app.bot)
        await tg_app.process_update(update)
    except Exception as e:
        log.error(f"Webhook update error: {e}")
    return {"ok": True}

@app.get("/reset_webhook")
async def reset_webhook():
    url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
    await tg_app.bot.delete_webhook(drop_pending_updates=True)
    await tg_app.bot.set_webhook(url=url, allowed_updates=["message", "callback_query"])
    return {"ok": True, "url": url}

# -------------------------- Startup/Shutdown --------------------------
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
