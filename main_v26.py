# main_v26.py
import os
import time
import json
import math
import queue
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

APP_NAME = "orb_v26"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s " + APP_NAME + ": %(message)s",
)
log = logging.getLogger(APP_NAME)

# =========================
# Konfig
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL = os.getenv("BASE_URL", "https://mporbbot.onrender.com").rstrip("/")

# Handla dessa spot-par (utan bindestreck i vår interna nyckel)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]

# State
STATE: Dict = {
    "orb_on": True,
    "entry_mode": "both",       # "long" | "short" | "both"
    "trailing": True,           # trailing stop aktiv
    "timeframe": "1m",          # "1m"|"3m"|"5m"|"15m"
    "risk_usdt": 30.0,          # per mocktrade
    "open_positions": {},       # symbol-> {side, entry, qty, stop, t_open}
    "realized_pnl_usdt": 0.0,
    "outbox": queue.Queue(),    # Telegram-outbox; telegram_bot.py läser denna
}

# KuCoin REST
KUCOIN_API = "https://api.kucoin.com/api/v1"

# ============
# FastAPI
# ============
app = FastAPI(title="ORB Bot v26")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK - " + APP_NAME

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

@app.get("/state", response_class=JSONResponse)
def get_state():
    s = dict(STATE)
    s["open_positions"] = {k: {kk: vv for kk, vv in v.items() if kk != "t_open"} for k, v in STATE["open_positions"].items()}
    return s

# =========================
# Hjälp
# =========================
def now_ts() -> int:
    return int(time.time())

def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.1f}"
    if p >= 1:
        return f"{p:.2f}"
    return f"{p:.4f}"

def fmt_usdt(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f} USDT"

def minute_tf_label(tf: str) -> str:
    return {"1m": "1", "3m": "3", "5m": "5", "15m": "15"}.get(tf, "1")

def to_kucoin_symbol(sym: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    if "-" in sym:
        return sym
    if sym.endswith("USDT"):
        return sym[:-4] + "-" + "USDT"
    # fallback
    return sym

def qty_for_trade(symbol: str, price: float) -> float:
    usdt = STATE["risk_usdt"]
    if price <= 0:
        return 0.0
    return max(0.0, usdt / price)

def candle_trail_long(current_stop: float, last_low: float) -> float:
    return max(current_stop, last_low)

def candle_trail_short(current_stop: float, last_high: float) -> float:
    return min(current_stop, last_high)

def make_buy_card(symbol: str, side: str, entry_price: float, qty: float) -> str:
    tf = STATE["timeframe"]
    return (
        f"🟢 <b>ENTRY {symbol}</b>\n"
        f"<b>Side:</b> {side.upper()} | <b>TF:</b> {minute_tf_label(tf)}m\n"
        f"<b>Entry:</b> {fmt_price(entry_price)} | <b>Qty:</b> {qty:.6f}\n"
        f"<b>Strategy:</b> ORB + TrailingStop"
    )

def make_sell_card(symbol: str, exit_price: float, qty: float, trade_pnl: float, reason: str) -> str:
    return (
        f"🔴 <b>EXIT {symbol}</b>\n"
        f"<b>Exit:</b> {fmt_price(exit_price)} | <b>Qty:</b> {qty:.6f}\n"
        f"<b>Reason:</b> {reason}\n"
        f"<b>Trade PnL:</b> {fmt_usdt(trade_pnl)}"
    )

def tg_send_text(text: str, chat_id: Optional[int] = None):
    try:
        STATE["outbox"].put_nowait({
            "type": "text",
            "text": text,
            "parse_mode": "HTML",
            "chat_id": chat_id,
        })
    except Exception as e:
        log.warning(f"[TG] Outbox queue issue: {e}")

# =========================
# KuCoin helpers
# =========================
async def kucoin_klines(symbol: str, tf: str, limit: int = 60) -> List[List]:
    """
    Returnerar listan (äldst->nyast) candle:
    KuCoin format per item: [time, open, close, high, low, volume, turnover] (alla str)
    """
    tf_map = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}
    k_tf = tf_map.get(tf, "1min")
    k_symbol = to_kucoin_symbol(symbol)
    url = f"{KUCOIN_API}/market/candles?type={k_tf}&symbol={k_symbol}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        log.info(f"HTTP Request: GET {url} \"{r.http_version} {r.status_code} {r.reason_phrase}\"")
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        kl = data["data"][:limit]
    # kucoin skickar nyast->äldst, vi vänder till äldst->nyast
    return sorted(kl, key=lambda x: int(x[0]))

async def kucoin_ticker(symbol: str) -> Optional[float]:
    k_symbol = to_kucoin_symbol(symbol)
    url = f"{KUCOIN_API}/market/orderbook/level1?symbol={k_symbol}"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url)
            log.info(f"HTTP Request: GET {url} \"{r.http_version} {r.status_code} {r.reason_phrase}\"")
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "200000":
                return None
            return float(data["data"]["price"])
    except Exception as e:
        log.warning(f"KuCoin ticker fail {symbol}: {e}")
        return None

# =========================
# ORB/Trading
# =========================
def compute_orb(candles: List[List]) -> Optional[Tuple[float, float]]:
    if not candles:
        return None
    first = candles[0]
    high = float(first[3])
    low = float(first[4])
    return (high, low)

async def try_entries_and_manage(symbol: str):
    tf = STATE["timeframe"]
    try:
        kl = await kucoin_klines(symbol, tf, limit=60)
    except Exception as e:
        log.warning(f"{symbol} klines error: {e}")
        return

    if not kl or len(kl) < 3:
        return

    orb = compute_orb(kl)
    if not orb:
        return

    orb_h, orb_l = orb
    last = kl[-1]
    prev = kl[-2]

    last_high = float(last[3])
    last_low = float(last[4])
    last_close = float(last[2])
    prev_high = float(prev[3])
    prev_low = float(prev[4])

    price = last_close
    pos = STATE["open_positions"].get(symbol)

    # Exit
    if pos:
        side = pos["side"]
        qty = pos["qty"]
        entry = pos["entry"]
        stop = pos["stop"]
        reason_exit = None

        if STATE["trailing"]:
            if side == "long":
                stop = candle_trail_long(stop, prev_low)
            else:
                stop = candle_trail_short(stop, prev_high)
            pos["stop"] = stop

        if side == "long":
            if price <= stop:
                reason_exit = "STOP HIT"
        else:
            if price >= stop:
                reason_exit = "STOP HIT"

        if reason_exit:
            exit_price = price
            trade_pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
            STATE["realized_pnl_usdt"] += trade_pnl
            del STATE["open_positions"][symbol]

            tg_send_text(make_sell_card(symbol, exit_price, qty, trade_pnl, reason_exit))
            log.info(f"{symbol} EXIT {side} @ {fmt_price(exit_price)} pnl={trade_pnl:.2f} ({reason_exit})")
            return

    # Entries
    if STATE["orb_on"] and pos is None:
        want_long = STATE["entry_mode"] in ("both", "long")
        want_short = STATE["entry_mode"] in ("both", "short")

        # Long break
        if want_long and last_high > orb_h and last_close > orb_h:
            qty = qty_for_trade(symbol, price)
            if qty > 0:
                stop = orb_l
                STATE["open_positions"][symbol] = {
                    "side": "long",
                    "entry": price,
                    "qty": qty,
                    "stop": stop,
                    "t_open": now_ts(),
                }
                tg_send_text(make_buy_card(symbol, "long", price, qty))
                log.info(f"{symbol} ENTRY LONG @ {fmt_price(price)} stop={fmt_price(stop)} qty={qty:.6f}")

        # Short break
        elif want_short and last_low < orb_l and last_close < orb_l:
            qty = qty_for_trade(symbol, price)
            if qty > 0:
                stop = orb_h
                STATE["open_positions"][symbol] = {
                    "side": "short",
                    "entry": price,
                    "qty": qty,
                    "stop": stop,
                    "t_open": now_ts(),
                }
                tg_send_text(make_buy_card(symbol, "short", price, qty))
                log.info(f"{symbol} ENTRY SHORT @ {fmt_price(price)} stop={fmt_price(stop)} qty={qty:.6f}")

# =========================
# Trading loop
# =========================
async def trade_loop():
    while True:
        try:
            for symbol in SYMBOLS:
                await try_entries_and_manage(symbol)
            await asyncio.sleep(4.0)
        except Exception as e:
            log.warning(f"trade_loop fel: {e}")
            await asyncio.sleep(2.0)

# =========================
# Startup
# =========================
@app.on_event("startup")
async def on_startup():
    log.info("Startup: init …")

    # Telegram-modulen (separerad i telegram_bot.py)
    if TELEGRAM_BOT_TOKEN:
        try:
            import telegram_bot  # din modul
            # ge den outboxen så den kan skicka vidare till Telegram
            telegram_bot.set_outbox(STATE["outbox"])
            # starta i egen tråd (INGA argument – den läser token själv från env)
            telegram_bot.start_in_thread()
            log.info("[TG] started.")
        except Exception as e:
            log.error(f"[TG] startup error: {e}")
    else:
        log.error("[TG] TELEGRAM_BOT_TOKEN saknas – Telegram startas inte.")

    # starta tradingloop
    loop = asyncio.get_event_loop()
    loop.create_task(trade_loop())

    log.info("Startup done.")

# =========================
# (valfritt) En enkel endpoint för att toggla saker snabbt
# =========================
@app.post("/toggle")
async def toggle(body: Dict):
    k = body.get("key")
    v = body.get("value")
    if k not in STATE:
        return JSONResponse({"ok": False, "msg": "unknown key"}, status_code=400)
    STATE[k] = v
    return {"ok": True, "state": {k: STATE[k]}}
