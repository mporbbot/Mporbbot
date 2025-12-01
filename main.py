# -*- coding: utf-8 -*-
# MP ORB/MOMO Hybrid Bot – Full Version
# Features:
# - Momentum + Mean Reversion engine (multi-symbol)
# - MOCK + LIVE trading (KuCoin spot, LONG only in LIVE)
# - TestBuy (LONG only)
# - TP/SL/Trail handling
# - Close All
# - PnL tracking
# - CSV-export
# - Risk buttons (size, TP, SL, max_pos)
# - FastAPI + Telegram webhook
# - Clean, stable, fully deployable on DigitalOcean

import os
import io
import csv
import json
import time
import hmac
import uuid
import base64
import hashlib
import asyncio
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from telegram import (
    Update,
    KeyboardButton,
    ReplyKeyboardMarkup,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
)

import logging

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mp_orb_hybrid")

# --------------------------------------------------
# ENVIRONMENT VARIABLES
# --------------------------------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / BOT_TOKEN environment variable.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (
    os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
    .replace(" ", "")
).split(",")

DEFAULT_TFS = os.getenv("TIMEFRAMES", "3m").replace(" ", "").split(",")

MOCK_SIZE_USDT = float(os.getenv("MOCK_SIZE_USDT", "10"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))   # 0.1% per side
MAX_OPEN_POS = int(os.getenv("MAX_POS", "4"))

# Risk settings
TP_PCT = float(os.getenv("TP_PCT", "0.30"))
SL_PCT = float(os.getenv("SL_PCT", "0.50"))
TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "2.0"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "2.5"))

# Momentum sensitivity
ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.55"))

# Mean reversion parameters
MR_DEV_PCT = float(os.getenv("MR_DEV_PCT", "1.20"))
TREND_SLOPE_MIN = float(os.getenv("TREND_SLOPE_MIN", "0.20"))
RANGE_ATR_MAX = float(os.getenv("RANGE_ATR_MAX", "0.80"))

# Loss guard
LOSS_GUARD_ON_DEFAULT = True
LOSS_GUARD_N_DEFAULT = int(os.getenv("LOSS_GUARD_N", "2"))
LOSS_GUARD_PAUSE_MIN_DEFAULT = int(os.getenv("LOSS_GUARD_PAUSE_MIN", "15"))

# KuCoin Credentials (V3 API)
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KUCOIN_API_KEY_VERSION = os.getenv("KUCOIN_API_KEY_VERSION", "3")
KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

logger.info(
    "[KUCOIN CREDS] key_len=%d secret_len=%d pass_len=%d version=%s",
    len(KUCOIN_API_KEY or ""),
    len(KUCOIN_API_SECRET or ""),
    len(KUCOIN_API_PASSPHRASE or ""),
    KUCOIN_API_KEY_VERSION,
)

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min"
}

# --------------------------------------------------
# STATE CLASSES
# --------------------------------------------------
@dataclass
class Position:
    side: str
    entry_price: float
    qty: float
    opened_at: datetime
    high_water: float
    low_water: float
    trailing: bool
    regime: str
    reason: str

    usd_in: float = 0.0
    fee_in: float = 0.0
    entry_order_id: Optional[str] = None


@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)


@dataclass
class EngineState:
    engine_on: bool = False
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
    per_sym: Dict[str, SymState] = field(default_factory=dict)

    mock_size: float = MOCK_SIZE_USDT
    fee_side: float = FEE_PER_SIDE
    max_pos: int = MAX_OPEN_POS

    tp_pct: float = TP_PCT
    sl_pct: float = SL_PCT
    trail_start_pct: float = TRAIL_START_PCT
    trail_pct: float = TRAIL_PCT

    threshold: float = ENTRY_THRESHOLD
    mr_on: bool = True
    regime_auto: bool = True
    mr_dev_pct: float = MR_DEV_PCT
    trend_slope_min: float = TREND_SLOPE_MIN
    range_atr_max: float = RANGE_ATR_MAX

    trade_mode: str = "mock"   # mock | live

    # Loss guard
    loss_guard_on: bool = LOSS_GUARD_ON_DEFAULT
    loss_guard_n: int = LOSS_GUARD_N_DEFAULT
    loss_guard_pause_min: int = LOSS_GUARD_PAUSE_MIN_DEFAULT
    loss_streak: int = 0
    paused_until: Optional[datetime] = None

    chat_id: Optional[int] = None


STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

LAST_LIVE_ENTRY_INFO: Dict[str, dict] = {}
LAST_LIVE_EXIT_INFO: Dict[str, dict] = {}
# --------------------------------------------------
# FILE LOGGING HELPERS
# --------------------------------------------------

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def log_mock(row: Dict) -> None:
    fname = "mock_trade_log.csv"
    new = not os.path.exists(fname)
    ensure_dir(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "time", "symbol", "action", "side", "price", "qty",
                "gross", "fee_in", "fee_out", "net", "info"
            ]
        )
        if new:
            w.writeheader()
        w.writerow(row)


def log_real(row: Dict) -> None:
    fname = "real_trade_log.csv"
    new = not os.path.exists(fname)
    ensure_dir(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "time", "symbol", "action", "side", "price", "qty",
                "gross", "fee_in", "fee_out", "net", "info"
            ]
        )
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------
# KUCOIN PRIVATE REQUEST
# --------------------------------------------------

def kucoin_creds_ok() -> bool:
    return bool(KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE)


async def kucoin_private_request(
    method: str,
    path: str,
    body: Optional[dict] = None,
) -> Optional[dict]:

    if not kucoin_creds_ok():
        return None

    body = body or {}
    body_str = json.dumps(body, separators=(",", ":")) if method.upper() != "GET" else ""

    ts = str(int(time.time() * 1000))
    prehash = ts + method.upper() + path + body_str

    sign = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode(),
            prehash.encode(),
            hashlib.sha256
        ).digest()
    ).decode()

    passphrase_hashed = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode(),
            KUCOIN_API_PASSPHRASE.encode(),
            hashlib.sha256
        ).digest()
    ).decode()

    headers = {
        "KC-API-KEY": KUCOIN_API_KEY,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": passphrase_hashed,
        "KC-API-KEY-VERSION": KUCOIN_API_KEY_VERSION,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(base_url=KUCOIN_BASE_URL, timeout=10) as client:
            resp = await client.request(
                method.upper(),
                path,
                headers=headers,
                content=body_str or None
            )
    except Exception as e:
        return {"_http_status": None, "code": None, "msg": f"Request error: {e}"}

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text or ""}

    data["_http_status"] = resp.status_code
    return data


# --------------------------------------------------
# GET KUCOIN FILLS
# --------------------------------------------------

async def kucoin_get_fills_for_order(order_id: str) -> Optional[dict]:

    if not kucoin_creds_ok():
        return None

    path = f"/api/v1/fills?orderId={order_id}&tradeType=TRADE"

    fills: List[dict] = []

    for _ in range(6):
        data = await kucoin_private_request("GET", path)
        if not data:
            await asyncio.sleep(0.2)
            continue

        if data.get("_http_status") != 200 or data.get("code") != "200000":
            await asyncio.sleep(0.2)
            continue

        raw = data.get("data")
        if isinstance(raw, list):
            fills = raw
        elif isinstance(raw, dict) and "items" in raw:
            fills = raw["items"]
        else:
            fills = []

        if fills:
            break
        await asyncio.sleep(0.2)

    if not fills:
        return None

    total_size = 0.0
    total_funds = 0.0
    total_fee = 0.0

    for f in fills:
        try:
            total_size += float(f.get("size", 0))
            total_funds += float(f.get("funds", 0))
            total_fee += float(f.get("fee", 0))
        except:
            pass

    if total_size <= 0 or total_funds <= 0:
        return None

    avg_price = total_funds / total_size

    return {
        "orderId": order_id,
        "size": total_size,
        "funds": total_funds,
        "fee": total_fee,
        "avg_price": avg_price,
    }


# --------------------------------------------------
# PLACE MARKET ORDER (SPOT)
# --------------------------------------------------

async def kucoin_place_market_order(symbol: str, side: str, amount: float):
    """
    SIDE:
      buy  -> amount = USDT funds
      sell -> amount = asset amount
    """
    if not kucoin_creds_ok():
        return False, "Missing KuCoin credentials."

    side_l = side.lower()

    body = {
        "clientOid": str(uuid.uuid4()),
        "side": side_l,
        "symbol": symbol,
        "type": "market",
    }

    if side_l == "buy":
        body["funds"] = f"{amount:.2f}"
    else:
        # SELL needs clean qty
        body["size"] = f"{amount:.8f}"

    data = await kucoin_private_request("POST", "/api/v1/orders", body)

    if not data:
        return False, "No response from KuCoin."

    if data.get("_http_status") != 200 or data.get("code") != "200000":
        msg = data.get("msg") or data.get("raw") or "KuCoin error"
        return False, f"HTTP={data.get('_http_status')} msg={msg}"

    order_id = (data.get("data") or {}).get("orderId")
    if not order_id:
        return False, "orderId missing in response"

    fills_info = await kucoin_get_fills_for_order(order_id)

    if fills_info:
        if side_l == "buy":
            LAST_LIVE_ENTRY_INFO[symbol] = fills_info
        else:
            LAST_LIVE_EXIT_INFO[symbol] = fills_info

    return True, ""
# --------------------------------------------------
# MARKET DATA (KLINES + INDICATORS)
# --------------------------------------------------

async def get_klines(symbol: str, tf: str, limit: int = 100):
    tf_api = TF_MAP.get(tf, tf)
    params = {"symbol": symbol, "type": tf_api}

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]

    return data[::-1][:limit]  # reverse → oldest first


def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]

    k = 2.0 / (period + 1)
    out = []
    val = series[0]

    for x in series:
        val = val + k * (x - val)
        out.append(val)

    return out


def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return [50.0] * len(closes)

    gains, losses = [], []

    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0))
        losses.append(-min(ch, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    out = [50.0] * period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period

        rs = 999 if avg_loss == 0 else avg_gain / avg_loss
        val = 100 - 100/(1 + rs)

        out.append(val)

    return [50.0] + out


def compute_features(candles):
    closes = [float(c[2]) for c in candles]
    highs  = [float(c[3]) for c in candles]
    lows   = [float(c[4]) for c in candles]

    if len(closes) < 40:
        last = closes[-1]
        return {
            "close": last,
            "ema20": last,
            "ema50": last,
            "rsi": 50.0,
            "mom": 0.0,
            "atrp": 0.2,
            "trend_slope": 0.0
        }

    ema20_series = ema(closes, 20)
    ema50_series = ema(closes, 50)

    ema20 = ema20_series[-1]
    ema50 = ema50_series[-1]

    mom = (closes[-1] - closes[-6]) / (closes[-6] or 1) * 100

    rsi_last = rsi(closes, 14)[-1]

    # ATR%
    trs = []
    for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:]):
        tr = (h - l) / (c or 1) * 100
        trs.append(tr)

    atrp = sum(trs)/len(trs) if trs else 0.2

    # Trend-slope
    if len(ema20_series) > 6 and ema20_series[-6] != 0:
        slope = (ema20_series[-1] - ema20_series[-6]) / ema20_series[-6] * 100
    else:
        slope = 0.0

    return {
        "close": closes[-1],
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi_last,
        "mom": mom,
        "atrp": atrp,
        "trend_slope": slope
    }


# --------------------------------------------------
# ENTRIES – MOMENTUM SCORE
# --------------------------------------------------

def momentum_score(feats) -> float:
    ema20 = feats["ema20"]
    ema50 = feats["ema50"]
    mom = feats["mom"]
    rsi_val = feats["rsi"]

    # Trend direction
    if ema20 > ema50:
        trend = 1
    elif ema20 < ema50:
        trend = -1
    else:
        trend = 0

    rsi_dev = (rsi_val - 50) / 10

    score = trend * (abs(mom) / 0.1) + rsi_dev

    if mom < 0:
        score = -abs(score)
    else:
        score = abs(score)

    if trend < 0:
        score = -score

    return score


# --------------------------------------------------
# REGIME DECISION
# --------------------------------------------------

def decide_regime(feats) -> str:
    if not STATE.regime_auto:
        return "trend"

    atrp = feats.get("atrp", 0)
    slope = abs(feats.get("trend_slope", 0))

    if atrp >= STATE.range_atr_max and slope >= STATE.trend_slope_min:
        return "trend"

    if atrp <= STATE.range_atr_max and slope < STATE.trend_slope_min:
        return "range"

    return "trend"
# --------------------------------------------------
# POSITION MANAGEMENT (TP / SL / TRAILING)
# --------------------------------------------------

def _fee(amount_usdt: float) -> float:
    """Model-fee för mock, 0.1% per sida."""
    return amount_usdt * STATE.fee_side


def _log_trade(row: Dict) -> None:
    """Välj rätt logg beroende på mode."""
    if STATE.trade_mode == "live":
        log_real(row)
    else:
        log_mock(row)


def open_position(symbol: str, side: str, price: float,
                  st: SymState, regime: str, reason: str):
    """
    Öppnar en position.
    - MOCK: qty = mock_size / price
    - LIVE: använder verkliga fills från LAST_LIVE_ENTRY_INFO
    """

    if STATE.trade_mode == "live":
        info = LAST_LIVE_ENTRY_INFO.pop(symbol, None)
        if info:
            entry_price = float(info.get("avg_price") or price)
            qty = float(info.get("size") or 0)
            usd_in = float(info.get("funds") or 0)
            fee_in = float(info.get("fee") or 0)
            entry_order_id = info.get("orderId")
        else:
            entry_price = price
            qty = STATE.mock_size / price
            usd_in = qty * price
            fee_in = _fee(usd_in)
            entry_order_id = None
    else:
        entry_price = price
        qty = STATE.mock_size / price
        usd_in = qty * price
        fee_in = _fee(usd_in)
        entry_order_id = None

    st.pos = Position(
        side=side,
        entry_price=entry_price,
        qty=qty,
        opened_at=datetime.now(timezone.utc),
        high_water=entry_price,
        low_water=entry_price,
        trailing=False,
        regime=regime,
        reason=reason,
        usd_in=usd_in,
        fee_in=fee_in,
        entry_order_id=entry_order_id
    )

    _log_trade({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "action": "ENTRY",
        "side": side,
        "price": round(entry_price, 6),
        "qty": round(qty, 8),
        "gross": "",
        "fee_in": round(fee_in, 6),
        "fee_out": "",
        "net": "",
        "info": f"{reason};regime={regime};mode={STATE.trade_mode}"
    })


def close_position(symbol: str, st: SymState,
                   reason: str, approx_price: float = None) -> float:
    """
    Stänger en position och räknar korrekt PnL.
    - LIVE → använder KuCoin fills exakt
    - MOCK → model-fee + model-qty
    """

    if not st.pos:
        return 0.0

    pos = st.pos

    # --- LIVE MODE ---
    if STATE.trade_mode == "live":
        info = LAST_LIVE_EXIT_INFO.pop(symbol, None)

        if info:
            usd_in = pos.usd_in
            fee_in = pos.fee_in

            usd_out = float(info.get("funds") or 0)
            fee_out = float(info.get("fee") or 0)
            exit_price = float(info.get("avg_price") or approx_price or pos.entry_price)

            gross = usd_out - usd_in
            net = gross - fee_in - fee_out

        else:
            # fallback
            price = approx_price or pos.entry_price
            usd_in = pos.qty * pos.entry_price
            usd_out = pos.qty * price
            fee_in = _fee(usd_in)
            fee_out = _fee(usd_out)

            gross = pos.qty * (price - pos.entry_price)
            net = gross - fee_in - fee_out
            exit_price = price

    # --- MOCK MODE ---
    else:
        price = approx_price or pos.entry_price
        usd_in = pos.qty * pos.entry_price
        usd_out = pos.qty * price

        fee_in = _fee(usd_in)
        fee_out = _fee(usd_out)

        gross = pos.qty * (price - pos.entry_price)
        net = gross - fee_in - fee_out
        exit_price = price

    st.realized_pnl += net

    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": pos.side,
        "entry": pos.entry_price,
        "exit": exit_price,
        "gross": round(gross, 6),
        "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6),
        "net": round(net, 6),
        "reason": reason
    })

    _log_trade({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "action": "EXIT",
        "side": pos.side,
        "price": round(exit_price, 6),
        "qty": round(pos.qty, 8),
        "gross": round(gross, 6),
        "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6),
        "net": round(net, 6),
        "info": reason
    })

    st.pos = None
    return net
# --------------------------------------------------
# ENGINE LOOP (MAIN TRADING ENGINE)
# --------------------------------------------------

async def engine_loop(app: Application):
    """
    Huvudloopen – kör var 3:e sekund.
    Hämtar klines → räknar features → gör:
    ✔ Momentum entries
    ✔ Mean reversion entries
    ✔ TP
    ✔ SL
    ✔ Trailing stop
    ✔ Live-orderhantering
    """

    await asyncio.sleep(2)

    while True:
        try:
            if not STATE.engine_on:
                await asyncio.sleep(3)
                continue

            now = datetime.now(timezone.utc)

            # Loss guard paus
            if STATE.loss_guard_on and STATE.paused_until:
                if now < STATE.paused_until:
                    await asyncio.sleep(3)
                    continue
                else:
                    STATE.paused_until = None

            # -----------------------
            #     ENTRY LOGIC
            # -----------------------
            open_positions = [s for s, st in STATE.per_sym.items() if st.pos]
            can_open_more = len(open_positions) < STATE.max_pos

            if can_open_more:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if st.pos:
                        continue

                    tf = STATE.tfs[0]
                    try:
                        kl = await get_klines(sym, tf)
                    except:
                        continue

                    feats = compute_features(kl)
                    price = feats["close"]
                    regime = decide_regime(feats)

                    allow_shorts = STATE.allow_shorts if STATE.trade_mode == "mock" else False

                    # --- TREND / MOMENTUM ---
                    if regime == "trend":
                        score = momentum_score(feats)

                        # Long entry
                        if score > STATE.threshold:
                            if STATE.trade_mode == "live":
                                ok, err = await kucoin_place_market_order(sym, "buy", STATE.mock_size)
                                if not ok:
                                    if STATE.chat_id:
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"⚠️ LIVE BUY {sym} (MOMO) misslyckades.\n{err}"
                                        )
                                    continue

                            open_position(sym, "LONG", price, st, "trend", "MOMO")

                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 MOMO ENTRY {sym} LONG @ {price:.4f} | score={score:.2f}"
                                )
                            open_positions.append(sym)
                            if len(open_positions) >= STATE.max_pos:
                                break

                        # Short entry (Mock-only)
                        if allow_shorts and score < -STATE.threshold:
                            open_position(sym, "SHORT", price, st, "trend", "MOMO")
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔻 MOMO SHORT {sym} @ {price:.4f} | score={score:.2f}"
                                )
                            open_positions.append(sym)
                            if len(open_positions) >= STATE.max_pos:
                                break

                    # --- MEAN REVERSION ---
                    elif regime == "range" and STATE.mr_on:
                        ema20 = feats["ema20"]
                        if ema20 == 0:
                            continue

                        dev = (price - ema20) / ema20 * 100

                        # Long MR
                        if dev <= -STATE.mr_dev_pct:
                            if STATE.trade_mode == "live":
                                ok, err = await kucoin_place_market_order(sym, "buy", STATE.mock_size)
                                if not ok:
                                    if STATE.chat_id:
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"⚠️ LIVE BUY {sym} (MR) misslyckades.\n{err}"
                                        )
                                    continue

                            open_position(sym, "LONG", price, st, "range", "MR")

                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 MR ENTRY {sym} LONG @ {price:.4f} | dev={dev:.2f}%"
                                )
                            open_positions.append(sym)
                            if len(open_positions) >= STATE.max_pos:
                                break

                        # Short MR (Mock only)
                        if allow_shorts and dev >= STATE.mr_dev_pct:
                            open_position(sym, "SHORT", price, st, "range", "MR")
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔻 MR SHORT {sym} @ {price:.4f} | dev={dev:.2f}%"
                                )
                            open_positions.append(sym)
                            if len(open_positions) >= STATE.max_pos:
                                break

            # -----------------------
            #     POSITION MGMT
            # -----------------------
            for sym in STATE.symbols:
                st = STATE.per_sym[sym]
                pos = st.pos
                if not pos:
                    continue

                tf = STATE.tfs[0]
                try:
                    kl = await get_klines(sym, tf, limit=10)
                except:
                    continue

                feats = compute_features(kl)
                price = feats["close"]

                pos.high_water = max(pos.high_water, price)
                pos.low_water = min(pos.low_water, price)

                move_pct = (price - pos.entry_price) / pos.entry_price * 100
                if pos.side == "SHORT":
                    move_pct = -move_pct

                # --- TRAIL START ---
                if not pos.trailing and move_pct >= STATE.trail_start_pct:
                    pos.trailing = True
                    if STATE.chat_id:
                        await app.bot.send_message(
                            STATE.chat_id,
                            f"🔒 TRAIL ON {sym} | move {move_pct:.2f}%"
                        )

                # --- TAKE PROFIT ---
                if move_pct >= STATE.tp_pct and not pos.trailing:
                    if pos.side == "LONG" and STATE.trade_mode == "live":
                        ok, err = await kucoin_place_market_order(sym, "sell", pos.qty)
                        if not ok:
                            continue

                    net = close_position(sym, st, "TP", approx_price=price)

                    if STATE.chat_id:
                        await app.bot.send_message(
                            STATE.chat_id,
                            f"🎯 TP EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT"
                        )
                    continue

                # --- TRAILING STOP ---
                if pos.trailing:
                    if pos.side == "LONG":
                        stop_lvl = pos.high_water * (1 - STATE.trail_pct / 100)
                        if price <= stop_lvl:
                            if STATE.trade_mode == "live":
                                await kucoin_place_market_order(sym, "sell", pos.qty)
                            net = close_position(sym, st, "TRAIL", approx_price=price)

                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f}"
                                )
                            continue

                    else:
                        # Short trailing
                        stop_lvl = pos.low_water * (1 + STATE.trail_pct / 100)
                        if price >= stop_lvl:
                            net = close_position(sym, st, "TRAIL", approx_price=price)

                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🏁 TRAIL EXIT SHORT {sym} | Net {net:+.4f}"
                                )
                            continue

                # --- STOP LOSS ---
                if move_pct <= -STATE.sl_pct:
                    if pos.side == "LONG" and STATE.trade_mode == "live":
                        await kucoin_place_market_order(sym, "sell", pos.qty)

                    net = close_position(sym, st, "SL", approx_price=price)

                    if STATE.chat_id:
                        await app.bot.send_message(
                            STATE.chat_id,
                            f"⛔ SL EXIT {sym} @ {price:.4f} | Net {net:+.4f}"
                        )

            await asyncio.sleep(3)

        except Exception as e:
            logger.exception("Engine error: %s", e)
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)
# --------------------------------------------------
# TELEGRAM UI – KNAPPAR, STATUS, MENY
# --------------------------------------------------

def reply_kb() -> ReplyKeyboardMarkup:
    """
    Huvudmenyn som visas i Telegram.
    """
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/threshold")],
        [KeyboardButton("/risk"), KeyboardButton("/export_csv")],
        [KeyboardButton("/close_all"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/mode"), KeyboardButton("/test_buy")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def status_text() -> str:
    """
    Status-rapport med:
    ✔ Aktiva positioner
    ✔ Riskinställningar
    ✔ Engine status
    ✔ Regime-inställningar
    ✔ PnL-total
    """
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    pos_lines = []

    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            p = st.pos
            move_pct = (p.high_water - p.entry_price) / p.entry_price * 100
            pos_lines.append(
                f"{s}: {p.side} @ {p.entry_price:.4f} | qty={p.qty:.6f} | "
                f"hi={p.high_water:.4f} | lo={p.low_water:.4f} | "
                f"max_move≈{move_pct:.2f}% | regime={p.regime}, src={p.reason}"
            )

    regime_line = (
        "Regime: AUTO (trend + mean reversion)"
        if STATE.regime_auto else
        "Regime: TREND only"
    )

    mr_line = (
        f"MR: {'ON' if STATE.mr_on else 'OFF'} | dev={STATE.mr_dev_pct:.2f}% "
        f"| range_max_ATR={STATE.range_atr_max:.2f}%"
    )

    mode_line = (
        "Mode: LIVE SPOT (endast LONG)"
        if STATE.trade_mode == "live"
        else "Mode: MOCK (simulerat)"
    )

    loss_guard_line = (
        f"Loss-guard: ON | N={STATE.loss_guard_n} | pause={STATE.loss_guard_pause_min}m"
    )
    if STATE.loss_guard_on and STATE.paused_until:
        rest = STATE.paused_until.astimezone().strftime("%H:%M")
        loss_guard_line += f" | aktiv paus → {rest}"

    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        "Strategi: Hybrid Momentum + Mean Reversion",
        regime_line,
        mr_line,
        f"Trend-slope min: {STATE.trend_slope_min:.2f}%",
        loss_guard_line,
        mode_line,
        f"Momentum-threshold: {STATE.threshold:.2f}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Mock-size: {STATE.mock_size:.2f} USDT",
        f"Fee per sida (modell): {STATE.fee_side:.4%}",
        f"Risk → TP: {STATE.tp_pct:.2f}% | SL: {STATE.sl_pct:.2f}% | "
        f"Trail start: {STATE.trail_start_pct:.2f}% | Trail: {STATE.trail_pct:.2f}%",
        f"Max samtidiga positioner: {STATE.max_pos}",
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]

    return "\n".join(lines)


# --------------------------------------------------
# TELEGRAM COMMANDS
# --------------------------------------------------

tg_app = Application.builder().token(BOT_TOKEN).build()


async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(
        STATE.chat_id,
        "🤖 MP Bot v55 – Hybrid Momentum + Mean Reversion\n"
        "Standardläge: MOCK (simulerad handel)\n"
        "Starta engine med /engine_on\n"
        "Byt mellan MOCK/LIVE med /mode.\n"
        "Justera momentum-känslighet med /threshold 0.55",
        reply_markup=reply_kb(),
    )
    await tg_app.bot.send_message(STATE.chat_id, status_text())


async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(
        STATE.chat_id,
        status_text(),
        reply_markup=reply_kb()
    )


async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Engine ON ✅ ({STATE.trade_mode.upper()})",
        reply_markup=reply_kb(),
    )


async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(
        STATE.chat_id,
        "Engine OFF ⛔️",
        reply_markup=reply_kb(),
    )


async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip()
    parts = msg.split(" ", 1)

    if len(parts) == 2:
        tfs = [t.strip() for t in parts[1].split(",") if t.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(
                STATE.chat_id,
                f"Timeframes → {', '.join(STATE.tfs)}",
                reply_markup=reply_kb(),
            )
            return

    await tg_app.bot.send_message(
        STATE.chat_id,
        "Använd: /timeframe 1m,3m,5m",
        reply_markup=reply_kb(),
    )
async def cmd_threshold(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip().split()

    if len(msg) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Aktuellt momentum-threshold: {STATE.threshold:.2f}",
            reply_markup=reply_kb(),
        )
        return

    try:
        val = float(msg[1])
        if val <= 0:
            raise ValueError()
        STATE.threshold = val
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Threshold uppdaterad → {val:.2f}",
            reply_markup=reply_kb(),
        )
    except:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Fel värde. Exempel: /threshold 0.55",
            reply_markup=reply_kb(),
        )


# --------------------------------------------------
#  RISK / TP / SL / SIZE / MAX_POS
# --------------------------------------------------

def _yesno(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "on", "ja")


async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()

    # Avancerat kommando:
    # /risk set key value
    if len(toks) == 4 and toks[1] == "set":
        key = toks[2]
        val = toks[3]

        try:
            if key == "size":
                STATE.mock_size = float(val)
            elif key == "tp":
                STATE.tp_pct = float(val)
            elif key == "sl":
                STATE.sl_pct = float(val)
            elif key == "trail_start":
                STATE.trail_start_pct = float(val)
            elif key == "trail":
                STATE.trail_pct = float(val)
            elif key == "max_pos":
                STATE.max_pos = int(val)
            elif key == "allow_shorts":
                if STATE.trade_mode == "live":
                    await tg_app.bot.send_message(
                        STATE.chat_id,
                        "Shorts är inte tillåtna i LIVE-spot.",
                        reply_markup=reply_kb(),
                    )
                    return
                STATE.allow_shorts = _yesno(val)
            elif key == "mr_on":
                STATE.mr_on = _yesno(val)
            elif key == "mr_dev":
                STATE.mr_dev_pct = float(val)
            elif key == "regime_auto":
                STATE.regime_auto = _yesno(val)
            elif key == "trend_slope_min":
                STATE.trend_slope_min = float(val)
            elif key == "range_atr_max":
                STATE.range_atr_max = float(val)
            elif key == "loss_guard_on":
                STATE.loss_guard_on = _yesno(val)
            elif key == "loss_guard_n":
                STATE.loss_guard_n = int(val)
            elif key == "loss_guard_pause":
                STATE.loss_guard_pause_min = int(val)
            else:
                await tg_app.bot.send_message(
                    STATE.chat_id,
                    "Stödjer: size, tp, sl, trail_start, trail, max_pos, "
                    "allow_shorts, mr_on, mr_dev, regime_auto, trend_slope_min, "
                    "range_atr_max, loss_guard_on, loss_guard_n, loss_guard_pause",
                    reply_markup=reply_kb(),
                )
                return

            await tg_app.bot.send_message(
                STATE.chat_id, "Risk inställningar uppdaterade.", reply_markup=reply_kb()
            )
        except:
            await tg_app.bot.send_message(
                STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb()
            )

        return

    # Snabbknappar
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Size 10", callback_data="risk_size_10"),
                InlineKeyboardButton("Size 30", callback_data="risk_size_30"),
                InlineKeyboardButton("Size 50", callback_data="risk_size_50"),
            ],
            [
                InlineKeyboardButton("TP 0.3%", callback_data="risk_tp_0.3"),
                InlineKeyboardButton("TP 0.5%", callback_data="risk_tp_0.5"),
                InlineKeyboardButton("TP 1.0%", callback_data="risk_tp_1.0"),
            ],
            [
                InlineKeyboardButton("SL 0.5%", callback_data="risk_sl_0.5"),
                InlineKeyboardButton("SL 1.0%", callback_data="risk_sl_1.0"),
                InlineKeyboardButton("SL 2.0%", callback_data="risk_sl_2.0"),
            ],
            [
                InlineKeyboardButton("max_pos 1", callback_data="risk_maxpos_1"),
                InlineKeyboardButton("max_pos 2", callback_data="risk_maxpos_2"),
                InlineKeyboardButton("max_pos 4", callback_data="risk_maxpos_4"),
            ],
        ]
    )

    text = (
        f"Aktuellt Size: {STATE.mock_size:.2f} USDT\n"
        f"TP: {STATE.tp_pct:.2f}% | SL: {STATE.sl_pct:.2f}%\n"
        f"Max Pos: {STATE.max_pos}\n\n"
        "Välj knapp nedan eller använd:\n"
        "`/risk set key value`"
    )

    await tg_app.bot.send_message(
        STATE.chat_id, text, reply_markup=kb
    )


# --------------------------------------------------
# PNL / EXPORT
# --------------------------------------------------

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)

    lines = [f"📈 Total PnL (NET): {total:+.4f} USDT"]
    for sym in STATE.symbols:
        pnl = STATE.per_sym[sym].realized_pnl
        lines.append(f"• {sym}: {pnl:+.4f} USDT")

    await tg_app.bot.send_message(
        STATE.chat_id, "\n".join(lines), reply_markup=reply_kb()
    )


async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id

    rows = [
        ["time", "symbol", "side", "entry", "exit", "gross", "fee_in", "fee_out", "net", "reason"]
    ]

    for sym in STATE.symbols:
        for r in STATE.per_sym[sym].trades_log:
            rows.append([
                r["time"], r["symbol"], r["side"], r["entry"], r["exit"],
                r["gross"], r["fee_in"], r["fee_out"], r["net"], r["reason"]
            ])

    if len(rows) <= 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.")
        return

    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    buf.seek(0)

    await tg_app.bot.send_document(
        STATE.chat_id,
        io.BytesIO(buf.getvalue().encode("utf-8")),
        filename="trades_export.csv",
        caption="Export CSV",
    )
# --------------------------------------------------
# MODE (MOCK / LIVE)
# --------------------------------------------------

async def cmd_mode(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    parts = update.message.text.strip().split()

    # Enbart /mode → visa valknappar
    if len(parts) == 1:
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("MOCK", callback_data="mode_mock"),
                    InlineKeyboardButton("LIVE", callback_data="mode_live_confirm"),
                ]
            ]
        )

        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Aktuellt läge: {STATE.trade_mode.upper()}\nVälj nytt läge:",
            reply_markup=kb,
        )
        return

    # /mode mock
    if parts[1].lower() == "mock":
        STATE.trade_mode = "mock"
        STATE.allow_shorts = True  # Shorts endast i mock
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Läge satt till MOCK (simulerad handel).",
            reply_markup=reply_kb(),
        )
        return

    # /mode live eller /mode live JA
    if parts[1].lower() == "live":

        # Kräver bekräftelse
        if len(parts) < 3 or parts[2].upper() != "JA":
            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("✅ Ja, aktivera LIVE", callback_data="mode_live_yes"),
                        InlineKeyboardButton("❌ Nej", callback_data="mode_live_no"),
                    ]
                ]
            )

            await tg_app.bot.send_message(
                STATE.chat_id,
                "⚠️ Är du säker på att du vill aktivera LIVE?\n"
                "Detta skickar riktiga market-ordrar på KuCoin.",
                reply_markup=kb,
            )
            return

        # Bekräftelsen ”JA”
        if not kucoin_creds_ok():
            await tg_app.bot.send_message(
                STATE.chat_id,
                "❌ LIVE kunde inte aktiveras: KuCoin API-uppgifter saknas.",
                reply_markup=reply_kb(),
            )
            return

        STATE.trade_mode = "live"
        STATE.allow_shorts = False  # spot only long

        await tg_app.bot.send_message(
            STATE.chat_id,
            "✅ LIVE-läge AKTIVERAT (spot, endast LONG).\n"
            "Engine kommer nu att skicka riktiga market-ordrar.\n"
            "Testa gärna /test_buy BTC-USDT 5 först.",
            reply_markup=reply_kb(),
        )
        return

    await tg_app.bot.send_message(
        STATE.chat_id,
        "Använd: /mode, /mode mock, /mode live, /mode live JA",
        reply_markup=reply_kb(),
    )


# --------------------------------------------------
# CLOSE ALL POSITIONS
# --------------------------------------------------

async def cmd_close_all(update: Update, _):
    STATE.chat_id = update.effective_chat.id

    closed = 0
    total_net = 0.0

    for sym in STATE.symbols:
        st = STATE.per_sym[sym]
        if not st.pos:
            continue

        pos = st.pos

        # Hämta senaste pris
        tf = STATE.tfs[0]
        try:
            kl = await get_klines(sym, tf, limit=2)
            price = compute_features(kl)["close"]
        except:
            price = pos.entry_price

        # För live: sälj riktiga positionen
        if STATE.trade_mode == "live" and pos.side == "LONG":
            ok, err = await kucoin_place_market_order(sym, "sell", pos.qty)
            if not ok:
                await tg_app.bot.send_message(
                    STATE.chat_id,
                    f"⚠️ CLOSE_ALL misslyckades i LIVE för {sym}.\n{err}",
                )
                continue

        net = close_position(sym, st, reason="MANUAL_CLOSE", approx_price=price)
        closed += 1
        total_net += net

    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Close-all klart.\nStängda positioner: {closed}\nNetto: {total_net:+.4f} USDT",
        reply_markup=reply_kb(),
    )


# --------------------------------------------------
# RESET PNL
# --------------------------------------------------

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id

    for s in STATE.symbols:
        st = STATE.per_sym[s]
        st.realized_pnl = 0.0
        st.trades_log.clear()

    await tg_app.bot.send_message(
        STATE.chat_id,
        "PnL återställd (i RAM – loggfilerna påverkas ej).",
        reply_markup=reply_kb(),
    )


# --------------------------------------------------
# TEST BUY (Long only)
# --------------------------------------------------

async def cmd_test_buy(update: Update, _):
    """
    /test_buy SYMBOL [USDT]
    Exempel:
      /test_buy BTC-USDT
      /test_buy ETH-USDT 15

    - Öppnar LONG position i MOCK eller LIVE
    - Ingår i PnL
    - Engine tar över (TP/SL/trail)
    """
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip().split()

    if len(msg) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Använd: /test_buy SYMBOL [USDT]\nEx: /test_buy BTC-USDT 5",
            reply_markup=reply_kb(),
        )
        return

    symbol = msg[1].upper()
    if symbol not in STATE.symbols:
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Symbol {symbol} finns inte.\n"
            f"Aktuella symboler: {', '.join(STATE.symbols)}",
            reply_markup=reply_kb(),
        )
        return

    size = STATE.mock_size
    if len(msg) >= 3:
        try:
            size = float(msg[2])
        except:
            pass

    mode = "LIVE" if STATE.trade_mode == "live" else "MOCK"

    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"✅ Ja, köp {symbol} för {size:.2f} USDT ({mode})",
                    callback_data=f"testbuy_confirm|{symbol}|{size:.4f}",
                ),
                InlineKeyboardButton("❌ Nej", callback_data="testbuy_cancel"),
            ]
        ]
    )

    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Bekräfta test-köp:\n{symbol}\n{size:.2f} USDT ({mode})",
        reply_markup=kb,
    )
# --------------------------------------------------
# CALLBACK-KNAPPAR
# --------------------------------------------------

async def on_button(update: Update, _):
    query = update.callback_query
    data = (query.data or "").strip()
    STATE.chat_id = query.message.chat_id

    # -----------------------------
    # RISK-knappar
    # -----------------------------

    # Size
    if data.startswith("risk_size_"):
        await query.answer()
        try:
            size = float(data.split("_")[-1])
            STATE.mock_size = size
            await query.edit_message_text(f"Size uppdaterad till {size:.1f} USDT.")
        except:
            await query.edit_message_text("Felaktigt värde för size.")
        return

    # TP
    if data.startswith("risk_tp_"):
        await query.answer()
        try:
            tp = float(data.split("_")[-1])
            STATE.tp_pct = tp
            await query.edit_message_text(f"TP uppdaterad till {tp:.2f}%.")
        except:
            await query.edit_message_text("Fel vid uppdatering av TP.")
        return

    # SL
    if data.startswith("risk_sl_"):
        await query.answer()
        try:
            sl = float(data.split("_")[-1])
            STATE.sl_pct = sl
            await query.edit_message_text(f"SL uppdaterad till {sl:.2f}%.")
        except:
            await query.edit_message_text("Fel vid uppdatering av SL.")
        return

    # max_pos
    if data.startswith("risk_maxpos_"):
        await query.answer()
        try:
            val = int(data.split("_")[-1])
            STATE.max_pos = val
            await query.edit_message_text(f"max_pos uppdaterad till {val}.")
        except:
            await query.edit_message_text("Felaktigt max_pos värde.")
        return


    # -----------------------------
    # MODE (MOCK / LIVE)
    # -----------------------------

    if data == "mode_mock":
        await query.answer()
        STATE.trade_mode = "mock"
        STATE.allow_shorts = True
        await query.edit_message_text("Läge satt till MOCK.")
        return

    if data == "mode_live_confirm":
        await query.answer()
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Ja, aktivera LIVE", callback_data="mode_live_yes"),
                    InlineKeyboardButton("❌ Nej", callback_data="mode_live_no"),
                ]
            ]
        )
        await query.edit_message_text(
            "⚠️ Vill du slå på LIVE?\nDetta skickar riktiga ordrar på KuCoin.",
            reply_markup=kb,
        )
        return

    if data == "mode_live_no":
        await query.answer("Avbrutet.")
        await query.edit_message_text("LIVE-aktivering avbröts.")
        return

    if data == "mode_live_yes":
        await query.answer()
        if not kucoin_creds_ok():
            await query.edit_message_text(
                "❌ LIVE kan inte aktiveras: saknar KuCoin API-nycklar."
            )
            return

        STATE.trade_mode = "live"
        STATE.allow_shorts = False  # Spot only long

        await query.edit_message_text(
            "✅ LIVE-läge aktiverat.\nEndast LONG i LIVE.\nEngine använder riktiga market-ordrar."
        )
        return


    # -----------------------------
    # TESTBUY
    # -----------------------------

    if data == "testbuy_cancel":
        await query.answer("Avbrutet.")
        await query.edit_message_text("Test-köp avbrutet.")
        return

    if data.startswith("testbuy_confirm|"):
        await query.answer()

        parts = data.split("|")
        if len(parts) != 3:
            await query.edit_message_text("Felaktig testbuy-data.")
            return

        symbol = parts[1].upper()
        try:
            size = float(parts[2])
        except:
            size = STATE.mock_size

        # Hämta pris
        tf = STATE.tfs[0]
        try:
            kl = await get_klines(symbol, tf, limit=5)
            price = compute_features(kl)["close"]
        except:
            price = 0.0

        # LIVE buy
        if STATE.trade_mode == "live":
            ok, err = await kucoin_place_market_order(symbol, "buy", size)
            if not ok:
                await query.edit_message_text(
                    f"❌ LIVE test-köp misslyckades för {symbol}\n{err}"
                )
                return

        # MOCK eller LIVE → skapa position
        st = STATE.per_sym[symbol]
        open_position(
            symbol,
            "LONG",
            price,
            st,
            regime="trend",
            reason="TESTBUY",
        )

        mode = "LIVE" if STATE.trade_mode == "live" else "MOCK"

        await query.edit_message_text(
            f"✅ Test-köp öppnat:\n{symbol} @ {price:.4f} ({mode}).\n"
            "Engine hanterar nu positionen (TP/SL/Trail)."
        )
        return


    # -----------------------------
    # Fallback
    # -----------------------------
    await query.answer()
    await query.edit_message_text("Ok.")
# --------------------------------------------------
# REGISTRERA TELEGRAM-HANDLERS
# --------------------------------------------------

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("threshold", cmd_threshold))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("mode", cmd_mode))
tg_app.add_handler(CommandHandler("close_all", cmd_close_all))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("test_buy", cmd_test_buy))
tg_app.add_handler(CallbackQueryHandler(on_button))


# --------------------------------------------------
# FASTAPI / DIGITALOCEAN
# --------------------------------------------------

app = FastAPI()


class TgUpdate(BaseModel):
    update_id: Optional[int] = None


@app.on_event("startup")
async def on_startup():
    logger.info("Startup: init Telegram app, webhook + engine")

    # Initiera telegram-appen
    await tg_app.initialize()

    # Sätt webhook om WEBHOOK_BASE finns
    if WEBHOOK_BASE:
        url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
        await tg_app.bot.set_webhook(url)
        logger.info("Webhook satt till %s", url)
    else:
        logger.warning("WEBHOOK_BASE saknas – glöm inte sätta den i environment!")

    # Starta telegram-app
    await tg_app.start()

    # Starta trading-engine
    asyncio.create_task(engine_loop(tg_app))


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutdown: stoppar Telegram app")
    await tg_app.stop()
    await tg_app.shutdown()


@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    return (
        f"MP Bot v55 Hybrid Momo OK | "
        f"engine_on={STATE.engine_on} | mode={STATE.trade_mode} | "
        f"thr={STATE.threshold:.2f} | pnl_total={total:+.4f}"
    )


@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "threshold": STATE.threshold,
        "tfs": STATE.tfs,
        "symbols": STATE.symbols,
        "pnl_total": round(total, 6),
        "mr_on": STATE.mr_on,
        "regime_auto": STATE.regime_auto,
        "mode": STATE.trade_mode,
    }


@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    """
    Telegram skickar uppdateringar hit (webhook).
    DigitalOcean/Render pekar hit via WEBHOOK_BASE.
    """
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}


# --------------------------------------------------
# LOKAL KÖRNING (om du kör python main.py)
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
