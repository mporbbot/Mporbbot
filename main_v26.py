import os
import time
import math
import logging
import asyncio
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

# our Telegram module
from telegram_bot import start_in_thread as tg_start_in_thread, send_text as tg_send_text, STATE as TG_STATE

APP_NAME = "orb_v26"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s " + APP_NAME + ": %(message)s",
)
log = logging.getLogger(APP_NAME)

# =========================
# Config
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# KuCoin public REST
KUCOIN_API = "https://api.kucoin.com/api/v1"
# Map Binance-style to KuCoin-style pairs
PAIR_MAP = {
    "BTCUSDT": "BTC-USDT",
    "ETHUSDT": "ETH-USDT",
    "ADAUSDT": "ADA-USDT",
    "LINKUSDT": "LINK-USDT",
    "XRPUSDT": "XRP-USDT",
}
SYMBOLS = list(PAIR_MAP.keys())

STATE: Dict = {
    "risk_usdt": float(os.getenv("RISK_USDT", "30")),
    "timeframe": "1m",          # keep in sync with tg STATE
    "trailing": True,
    "entry_mode": "both",
    "orb_on": True,
    "open_positions": {},       # symbol -> {side, entry, qty, stop, t_open}
    "realized_pnl_usdt": 0.0,
}

# =========================
# FastAPI
# =========================
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
    s["open_positions"] = list(STATE["open_positions"].keys())
    return s

# =========================
# Utils
# =========================
def now_ts() -> int:
    return int(time.time())

def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.1f}"
    if p >= 1:
        return f"{p:.2f}"
    return f"{p:.5f}"

def fmt_usdt(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f} USDT"

def minute_tf_label(tf: str) -> str:
    return {"1m": "1", "3m": "3", "5m": "5", "15m": "15"}.get(tf, "1")

def make_buy_card(symbol: str, side: str, entry_price: float, qty: float) -> str:
    tf = STATE["timeframe"]
    return (
        f"🟢 <b>BUY {symbol}</b>\n"
        f"<b>Side:</b> {side.upper()} | <b>TF:</b> {minute_tf_label(tf)}m\n"
        f"<b>Entry:</b> {fmt_price(entry_price)} | <b>Qty:</b> {qty:.6f}\n"
        f"<b>Strategy:</b> ORB + TrailingStop\n"
    )

def make_sell_card(symbol: str, exit_price: float, qty: float, trade_pnl: float, reason: str) -> str:
    return (
        f"🔴 <b>SELL {symbol}</b>\n"
        f"<b>Exit:</b> {fmt_price(exit_price)} | <b>Qty:</b> {qty:.6f}\n"
        f"<b>Reason:</b> {reason}\n"
        f"<b>Trade PnL:</b> {fmt_usdt(trade_pnl)}\n"
    )

# sync our config with Telegram toggles (simple mirror)
def pull_tele_state():
    STATE["timeframe"] = TG_STATE["timeframe"]
    STATE["entry_mode"] = TG_STATE["entry_mode"]
    STATE["trailing"]   = TG_STATE["trailing"]
    STATE["orb_on"]     = TG_STATE["orb_on"]

# =========================
# KuCoin helpers
# =========================
async def kucoin_klines(symbol: str, tf: str, limit: int = 60) -> List[List]:
    tf_map = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}
    k_tf = tf_map.get(tf, "1min")
    k_symbol = PAIR_MAP[symbol]
    url = f"{KUCOIN_API}/market/candles?type={k_tf}&symbol={k_symbol}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        kl = data["data"][:limit]
    # KuCoin format: [time, open, close, high, low, volume, turnover]
    kl_sorted = sorted(kl, key=lambda x: int(x[0]))
    return kl_sorted

async def kucoin_ticker(symbol: str) -> Optional[float]:
    k_symbol = PAIR_MAP[symbol]
    url = f"{KUCOIN_API}/market/orderbook/level1?symbol={k_symbol}"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "200000":
                return None
            return float(data["data"]["price"])
    except Exception as e:
        log.warning(f"KuCoin ticker fail {symbol}: {e}")
        return None

# =========================
# ORB logic
# =========================
def compute_orb(candles: List[List]) -> Optional[Tuple[float, float]]:
    if not candles:
        return None
    first = candles[0]
    high = float(first[3])
    low = float(first[4])
    return (high, low)

def qty_for_trade(price: float) -> float:
    if price <= 0:
        return 0.0
    return max(0.0, STATE["risk_usdt"] / price)

def trail_long(current_stop: float, last_low: float) -> float:
    return max(current_stop, last_low)

def trail_short(current_stop: float, last_high: float) -> float:
    return min(current_stop, last_high)

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
    last_low  = float(last[4])
    last_close = float(last[2])
    prev_high = float(prev[3])
    prev_low  = float(prev[4])
    price = last_close

    pos = STATE["open_positions"].get(symbol)

    # Exits
    if pos:
        side = pos["side"]
        qty = pos["qty"]
        entry = pos["entry"]
        stop = pos["stop"]
        reason_exit = None

        if STATE["trailing"]:
            if side == "long":
                stop = trail_long(stop, prev_low)
            else:
                stop = trail_short(stop, prev_high)
            pos["stop"] = stop

        if side == "long" and price <= stop:
            reason_exit = "STOP HIT"
        elif side == "short" and price >= stop:
            reason_exit = "STOP HIT"

        if reason_exit:
            exit_price = price
            pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
            STATE["realized_pnl_usdt"] += pnl
            del STATE["open_positions"][symbol]
            tg_send_text(make_sell_card(symbol, exit_price, qty, pnl, reason_exit))
            log.info(f"{symbol} EXIT {side} @ {fmt_price(exit_price)} pnl={pnl:.2f} ({reason_exit})")
            return

    # Entries
    if STATE["orb_on"] and pos is None:
        want_long  = STATE["entry_mode"] in ("both", "long")
        want_short = STATE["entry_mode"] in ("both", "short")

        # long break
        if want_long and last_high > orb_h and last_close > orb_h:
            q = qty_for_trade(price)
            if q > 0:
                stop = orb_l
                STATE["open_positions"][symbol] = {
                    "side": "long", "entry": price, "qty": q, "stop": stop, "t_open": now_ts()
                }
                tg_send_text(make_buy_card(symbol, "long", price, q))
                log.info(f"{symbol} ENTRY LONG @ {fmt_price(price)} stop={fmt_price(stop)} qty={q:.6f}")
        # short break
        elif want_short and last_low < orb_l and last_close < orb_l:
            q = qty_for_trade(price)
            if q > 0:
                stop = orb_h
                STATE["open_positions"][symbol] = {
                    "side": "short", "entry": price, "qty": q, "stop": stop, "t_open": now_ts()
                }
                tg_send_text(make_buy_card(symbol, "short", price, q))
                log.info(f"{symbol} ENTRY SHORT @ {fmt_price(price)} stop={fmt_price(stop)} qty={q:.6f}")

# =========================
# Background loops
# =========================
async def trade_loop():
    while True:
        try:
            # mirror Telegram toggles into our local STATE
            pull_tele_state()
            for symbol in SYMBOLS:
                await try_entries_and_manage(symbol)
            await asyncio.sleep(3.0)
        except Exception as e:
            log.warning(f"trade_loop error: {e}")
            await asyncio.sleep(2.0)

# =========================
# Startup
# =========================
@app.on_event("startup")
async def on_startup():
    # Start Telegram in its own thread (no event-loop fights with Uvicorn/uvloop)
    if TELEGRAM_BOT_TOKEN:
        tg_start_in_thread(TELEGRAM_BOT_TOKEN)
    else:
        log.error("TELEGRAM_BOT_TOKEN saknas – startar utan Telegram.")

    # kick trading loop
    loop = asyncio.get_event_loop()
    loop.create_task(trade_loop())
    log.info("Startup done.")
