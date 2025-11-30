# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MP Bot – Full version (MOCK + LIVE)
# TestBuy: 5 USDT per coin (manual coin selection)
# Fixed LIVE SELL (KuCoin precision, minSize, increment)
# Includes all functions, all commands, all menus exactly as before.
# ------------------------------------------------------------

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
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal

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

# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN/BOT_TOKEN")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (
    os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
    .replace(" ", "")
).split(",")

DEFAULT_TFS = (
    os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")
).split(",")

MOCK_SIZE_USDT = float(os.getenv("MOCK_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "4"))

TP_PCT = float(os.getenv("TP_PCT", "0.30"))
SL_PCT = float(os.getenv("SL_PCT", "0.50"))
TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "2.0"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "2.5"))

ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.55"))
ALLOW_SHORTS_DEFAULT = (
    os.getenv("ALLOW_SHORTS", "false").lower() in ("1", "true", "on", "yes")
)

MR_DEV_PCT = float(os.getenv("MR_DEV_PCT", "1.20"))
TREND_SLOPE_MIN = float(os.getenv("TREND_SLOPE_MIN", "0.20"))
RANGE_ATR_MAX = float(os.getenv("RANGE_ATR_MAX", "0.80"))

LOSS_GUARD_ON_DEFAULT = True
LOSS_GUARD_N_DEFAULT = int(os.getenv("LOSS_GUARD_N", "2"))
LOSS_GUARD_PAUSE_MIN_DEFAULT = int(os.getenv("LOSS_GUARD_PAUSE_MIN", "15"))

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KUCOIN_API_KEY_VERSION = os.getenv("KUCOIN_API_KEY_VERSION", "2")
KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}

# Cache för symbol-metadata (minSize, increment)
SYMBOL_META_CACHE: Dict[str, Dict[str, str]] = {}

# ------------------------------------------------------------
# STATE
# ------------------------------------------------------------

@dataclass
class Position:
    side: str
    entry_price: float
    qty: float
    opened_at: datetime
    high_water: float
    low_water: float
    trailing: bool = False
    regime: str = "trend"
    reason: str = "MOMO"


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
    threshold: float = ENTRY_THRESHOLD
    allow_shorts: bool = ALLOW_SHORTS_DEFAULT

    tp_pct: float = TP_PCT
    sl_pct: float = SL_PCT
    trail_start_pct: float = TRAIL_START_PCT
    trail_pct: float = TRAIL_PCT
    max_pos: int = MAX_OPEN_POS

    mr_on: bool = True
    regime_auto: bool = True
    mr_dev_pct: float = MR_DEV_PCT
    trend_slope_min: float = TREND_SLOPE_MIN
    range_atr_max: float = RANGE_ATR_MAX

    trade_mode: str = "mock"   # "mock" / "live"

    loss_guard_on: bool = LOSS_GUARD_ON_DEFAULT
    loss_guard_n: int = LOSS_GUARD_N_DEFAULT
    loss_guard_pause_min: int = LOSS_GUARD_PAUSE_MIN_DEFAULT
    loss_streak: int = 0
    paused_until: Optional[datetime] = None

    chat_id: Optional[int] = None


STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def log_mock(row: Dict):
    fname = "mock_trade_log.csv"
    new = not os.path.exists(fname)
    ensure_dir(fname)

    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "time","symbol","action","side","price","qty",
                "gross","fee_in","fee_out","net","info",
            ],
        )
        if new:
            w.writeheader()
        w.writerow(row)


def log_real(row: Dict):
    fname = "real_trade_log.csv"
    new = not os.path.exists(fname)
    ensure_dir(fname)

    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "time","symbol","action","side","price","qty",
                "gross","fee_in","fee_out","net","info",
            ],
        )
        if new:
            w.writeheader()
        w.writerow(row)


def _log_trade(row: Dict):
    if STATE.trade_mode == "live":
        log_real(row)
    else:
        log_mock(row)


def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

# ------------------------------------------------------------
# KUCOIN HELPERS
# ------------------------------------------------------------

def kucoin_creds_ok() -> bool:
    return bool(KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE)


async def kucoin_private_request(method: str, path: str, body: Optional[dict] = None):
    """
    KuCoin private API request with proper signing.
    """
    if not kucoin_creds_ok():
        return None

    body = body or {}
    body_str = json.dumps(body, separators=(",", ":"))
    ts = str(int(time.time() * 1000))

    prehash = ts + method.upper() + path + body_str
    sign = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")

    passphrase_hashed = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode("utf-8"),
            KUCOIN_API_PASSPHRASE.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")

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
            r = await client.request(
                method.upper(), path, headers=headers, content=body_str
            )
    except Exception as e:
        return {"_http_status": None, "msg": f"Request error: {e}"}

    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text or ""}

    data["_http_status"] = r.status_code
    return data# ------------------------------------------------------------
# MARKET METADATA (minSize, quantityIncrement)
# ------------------------------------------------------------

async def kucoin_load_symbol_meta(symbol: str):
    """
    Hämtar minSize, maxSize, quantityIncrement och priceIncrement för ett coin.
    Sparas i cache för snabbare anrop.
    """
    if symbol in SYMBOL_META_CACHE:
        return SYMBOL_META_CACHE[symbol]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.kucoin.com/api/v1/symbols")
            r.raise_for_status()
            syms = r.json().get("data", [])
    except Exception:
        return None

    for s in syms:
        if s.get("symbol") == symbol:
            SYMBOL_META_CACHE[symbol] = {
                "baseMinSize": s.get("baseMinSize", "0.000001"),
                "baseIncrement": s.get("baseIncrement", "0.000001"),
                "priceIncrement": s.get("priceIncrement", "0.000001"),
            }
            return SYMBOL_META_CACHE[symbol]

    return None


def round_to_increment(value: float, increment: str) -> float:
    """
    Rundar ett värde till rätt precision enligt KuCoin increment (str som '0.00001').
    """
    try:
        d_val = Decimal(str(value))
        d_inc = Decimal(increment)
        return float((d_val // d_inc) * d_inc)
    except Exception:
        return value


# ------------------------------------------------------------
# KUCOIN MARKET ORDER
# ------------------------------------------------------------

async def kucoin_place_market_buy(symbol: str, usdt_amount: float):
    """
    Market BUY (funds=USDT) – safe precision.
    """
    meta = await kucoin_load_symbol_meta(symbol)
    if not meta:
        return False, "No symbol meta from KuCoin"

    funds = float(usdt_amount)
    body = {
        "clientOid": str(uuid.uuid4()),
        "side": "buy",
        "symbol": symbol,
        "type": "market",
        "funds": f"{funds:.2f}",
    }

    data = await kucoin_private_request("POST", "/api/v1/orders", body)
    if not data:
        return False, "No response"

    if data.get("_http_status") != 200 or data.get("code") != "200000":
        return False, f"{data}"

    return True, ""


async def kucoin_place_market_sell(symbol: str, qty: float):
    """
    Market SELL – must follow baseIncrement + baseMinSize
    """

    meta = await kucoin_load_symbol_meta(symbol)
    if not meta:
        return False, "No symbol meta from KuCoin"

    inc = meta["baseIncrement"]
    min_size = float(meta["baseMinSize"])

    qty_fixed = round_to_increment(qty, inc)
    if qty_fixed < min_size:
        return False, f"Qty too small ({qty_fixed}), min={min_size}"

    body = {
        "clientOid": str(uuid.uuid4()),
        "side": "sell",
        "symbol": symbol,
        "type": "market",
        "size": f"{qty_fixed:.8f}",
    }

    data = await kucoin_private_request("POST", "/api/v1/orders", body)
    if not data:
        return False, "No response"

    if data.get("_http_status") != 200 or data.get("code") != "200000":
        return False, f"{data}"

    return True, ""


# ------------------------------------------------------------
# PUBLIC MARKET DATA
# ------------------------------------------------------------

async def get_klines(symbol: str, tf: str, limit: int = 100):
    tf_api = TF_MAP.get(tf, tf)
    params = {"symbol": symbol, "type": tf_api}

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])

    return data[::-1][:limit]


# ------------------------------------------------------------
# INDICATORS
# ------------------------------------------------------------

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series

    k = 2.0 / (period + 1.0)
    out = []
    val = series[0]

    for x in series:
        val = val + (x - val) * k
        out.append(val)

    return out


def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return [50] * len(closes)

    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(-min(diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    result = [50] * period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rs = 999
        else:
            rs = avg_gain / avg_loss

        rsi_val = 100 - (100 / (1 + rs))
        result.append(rsi_val)

    return [50] + result


def compute_features(candles):
    closes = [float(c[2]) for c in candles]
    highs = [float(c[3]) for c in candles]
    lows = [float(c[4]) for c in candles]

    if len(closes) < 40:
        last = closes[-1]
        return {
            "close": last,
            "ema20": last,
            "ema50": last,
            "rsi": 50,
            "mom": 0,
            "atrp": 0.2,
            "trend_slope": 0,
        }

    ema20_series = ema(closes, 20)
    ema50_series = ema(closes, 50)

    ema20 = ema20_series[-1]
    ema50 = ema50_series[-1]

    mom = (closes[-1] - closes[-6]) / (closes[-6] or 1) * 100
    rsi_val = rsi(closes, 14)[-1]

    trs = []
    for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:]):
        tr = (h - l) / (c or 1) * 100
        trs.append(tr)

    atrp = sum(trs) / len(trs)
    trend_slope = (
        (ema20_series[-1] - ema20_series[-6]) / (ema20_series[-6] or 1) * 100
        if len(ema20_series) > 6
        else 0
    )

    return {
        "close": closes[-1],
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi_val,
        "mom": mom,
        "atrp": atrp,
        "trend_slope": trend_slope,
    }


def momentum_score(f):
    ema20, ema50 = f["ema20"], f["ema50"]
    mom = f["mom"]
    rsi_val = f["rsi"]

    trend = 1 if ema20 > ema50 else -1 if ema20 < ema50 else 0
    rsi_dev = (rsi_val - 50) / 10

    score = trend * (abs(mom) / 0.1) + rsi_dev
    score = abs(score) if mom >= 0 else -abs(score)
    if trend < 0:
        score = -score

    return score# ------------------------------------------------------------
# REGIME DECISION (trend / range)
# ------------------------------------------------------------

def decide_regime(f):
    if not STATE.regime_auto:
        return "trend"

    atrp = f["atrp"]
    slope = abs(f["trend_slope"])

    if atrp >= STATE.range_atr_max and slope >= STATE.trend_slope_min:
        return "trend"

    if atrp <= STATE.range_atr_max and slope < STATE.trend_slope_min:
        return "range"

    return "trend"


# ------------------------------------------------------------
# POSITION CLASS
# ------------------------------------------------------------

@dataclass
class Position:
    side: str
    entry_price: float
    qty: float
    opened_at: datetime
    high_water: float
    low_water: float
    trailing: bool = False
    regime: str = "trend"
    reason: str = "MOMO"


@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)


@dataclass
class EngineState:
    engine_on: bool = False
    symbols: List[str] = field(
        default_factory=lambda: ["BTC-USDT", "ETH-USDT", "ADA-USDT", "XRP-USDT", "LINK-USDT"]
    )
    tfs: List[str] = field(default_factory=lambda: ["3m"])
    per_sym: Dict[str, SymState] = field(default_factory=dict)

    mock_size: float = 30.0
    fee_side: float = 0.001
    threshold: float = 0.55
    allow_shorts: bool = False

    tp_pct: float = 0.30
    sl_pct: float = 0.50
    trail_start_pct: float = 2.0
    trail_pct: float = 2.5
    max_pos: int = 4

    mr_on: bool = True
    regime_auto: bool = True
    mr_dev_pct: float = 1.2
    trend_slope_min: float = 0.20
    range_atr_max: float = 0.80

    trade_mode: str = "mock"

    loss_guard_on: bool = True
    loss_guard_n: int = 2
    loss_guard_pause_min: int = 15
    loss_streak: int = 0
    paused_until: Optional[datetime] = None

    chat_id: Optional[int] = None


STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()


# ------------------------------------------------------------
# LOG HELPERS
# ------------------------------------------------------------

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def log_mock(row: Dict):
    fname = "mock_trade_log.csv"
    new = not os.path.exists(fname)
    ensure_dir(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "time","symbol","action","side","price","qty",
                "gross","fee_in","fee_out","net","info"
            ],
        )
        if new:
            w.writeheader()
        w.writerow(row)


def log_real(row: Dict):
    fname = "real_trade_log.csv"
    new = not os.path.exists(fname)
    ensure_dir(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "time","symbol","action","side","price","qty",
                "gross","fee_in","fee_out","net","info"
            ],
        )
        if new:
            w.writeheader()
        w.writerow(row)


def _log_trade(row: Dict):
    if STATE.trade_mode == "live":
        log_real(row)
    else:
        log_mock(row)


# ------------------------------------------------------------
# POSITION MANAGEMENT (OPEN / CLOSE)
# ------------------------------------------------------------

def open_position(sym: str, side: str, price: float, st: SymState,
                  regime: str, reason: str):
    qty = STATE.mock_size / price if price > 0 else 0.0

    st.pos = Position(
        side=side,
        entry_price=price,
        qty=qty,
        opened_at=datetime.now(timezone.utc),
        high_water=price,
        low_water=price,
        trailing=False,
        regime=regime,
        reason=reason,
    )

    _log_trade({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "action": "ENTRY",
        "side": side,
        "price": round(price, 6),
        "qty": round(qty, 8),
        "gross": "",
        "fee_in": "",
        "fee_out": "",
        "net": "",
        "info": f"size_usdt={STATE.mock_size};regime={regime};reason={reason};mode={STATE.trade_mode}",
    })


def close_position(sym: str, price: float, st: SymState, reason: str) -> float:
    if not st.pos:
        return 0.0

    pos = st.pos
    usd_in = pos.qty * pos.entry_price
    usd_out = pos.qty * price

    fee_in = usd_in * STATE.fee_side
    fee_out = usd_out * STATE.fee_side

    if pos.side == "LONG":
        gross = pos.qty * (price - pos.entry_price)
    else:
        gross = pos.qty * (pos.entry_price - price)

    net = gross - fee_in - fee_out

    st.realized_pnl += net
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "side": pos.side,
        "entry": pos.entry_price,
        "exit": price,
        "gross": round(gross, 6),
        "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6),
        "net": round(net, 6),
        "reason": f"{reason};regime={pos.regime};src={pos.reason};mode={STATE.trade_mode}",
    })

    _log_trade({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "action": "EXIT",
        "side": pos.side,
        "price": round(price, 6),
        "qty": round(pos.qty, 8),
        "gross": round(gross, 6),
        "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6),
        "net": round(net, 6),
        "info": f"{reason};regime={pos.regime};src={pos.reason};mode={STATE.trade_mode}",
    })

    st.pos = None
    return net# ------------------------------------------------------------
# ENGINE LOOP
# ------------------------------------------------------------

async def engine_loop(app: Application):
    await asyncio.sleep(2)

    while True:
        try:
            now = datetime.now(timezone.utc)

            # ------------------------------------------------
            # LOSS-GUARD PAUSE
            # ------------------------------------------------
            if (
                STATE.engine_on
                and STATE.loss_guard_on
                and STATE.paused_until is not None
                and now < STATE.paused_until
            ):
                await asyncio.sleep(3)
                continue

            # ------------------------------------------------
            # -------------------   ENTRIES   -----------------
            # ------------------------------------------------
            if STATE.engine_on:
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                can_open_more = len(open_syms) < STATE.max_pos

                if can_open_more:
                    for sym in STATE.symbols:
                        st = STATE.per_sym[sym]
                        if st.pos:
                            continue

                        tf = STATE.tfs[0]
                        try:
                            kl = await get_klines(sym, tf, limit=80)
                        except Exception:
                            continue

                        feats = compute_features(kl)
                        price = feats["close"]
                        regime = decide_regime(feats)

                        allow_short = STATE.allow_shorts and STATE.trade_mode == "mock"

                        # -------- TREND / MOMENTUM --------
                        if regime == "trend":
                            score = momentum_score(feats)

                            # ---- LONG ----
                            if score > STATE.threshold:
                                if STATE.trade_mode == "live":
                                    ok, err = await kucoin_place_market_order(
                                        sym, "buy", STATE.mock_size
                                    )
                                    if not ok:
                                        if STATE.chat_id:
                                            await app.bot.send_message(
                                                STATE.chat_id,
                                                f"⚠️ LIVE BUY misslyckades {sym}\n{err}"
                                            )
                                        continue

                                open_position(sym, "LONG", price, st,
                                              regime="trend", reason="MOMO")

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 MOMO ENTRY {sym} LONG @ {price:.4f} | score={score:.2f}"
                                    )

                                open_syms.append(sym)
                                if len(open_syms) >= STATE.max_pos:
                                    break

                            # ---- SHORT (mock only) ----
                            elif allow_short and score < -STATE.threshold:
                                open_position(sym, "SHORT", price, st,
                                              regime="trend", reason="MOMO")

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 MOMO ENTRY {sym} SHORT @ {price:.4f} | score={score:.2f}"
                                    )

                                open_syms.append(sym)
                                if len(open_syms) >= STATE.max_pos:
                                    break

                        # -------- RANGE / MEAN REVERSION --------
                        else:
                            ema20 = feats["ema20"]
                            if ema20 == 0:
                                continue

                            dev = (price - ema20) / ema20 * 100.0

                            # ---- MR LONG ----
                            if dev <= -STATE.mr_dev_pct:
                                if STATE.trade_mode == "live":
                                    ok, err = await kucoin_place_market_order(
                                        sym, "buy", STATE.mock_size
                                    )
                                    if not ok:
                                        if STATE.chat_id:
                                            await app.bot.send_message(
                                                STATE.chat_id,
                                                f"⚠️ LIVE BUY misslyckades {sym}\n{err}"
                                            )
                                        continue

                                open_position(sym, "LONG", price, st,
                                              regime="range", reason="MR")

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 MR ENTRY {sym} LONG @ {price:.4f}"
                                    )

                                open_syms.append(sym)
                                if len(open_syms) >= STATE.max_pos:
                                    break

                            # ---- MR SHORT (mock only) ----
                            elif dev >= STATE.mr_dev_pct and allow_short:
                                open_position(sym, "SHORT", price, st,
                                              regime="range", reason="MR")

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 MR ENTRY {sym} SHORT @ {price:.4f}"
                                    )

                                open_syms.append(sym)
                                if len(open_syms) >= STATE.max_pos:
                                    break

                # ------------------------------------------------
                # -------------------   EXITS   ------------------
                # ------------------------------------------------
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos:
                        continue

                    tf = STATE.tfs[0]

                    try:
                        kl = await get_klines(sym, tf, limit=5)
                    except Exception:
                        continue

                    feats = compute_features(kl)
                    price = feats["close"]

                    pos = st.pos
                    pos.high_water = max(pos.high_water, price)
                    pos.low_water = min(pos.low_water, price)

                    # move_pct = profit in % (positive for LONG)
                    move_pct = (price - pos.entry_price) / pos.entry_price * 100.0
                    if pos.side == "SHORT":
                        move_pct = -move_pct

                    # -------- START TRAILING --------
                    if not pos.trailing and move_pct >= STATE.trail_start_pct:
                        pos.trailing = True
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🔒 TRAIL ON {sym} | move≈{move_pct:.2f}%"
                            )

                    # -------- TAKE PROFIT --------
                    if move_pct >= STATE.tp_pct and not pos.trailing:

                        if STATE.trade_mode == "live" and pos.side == "LONG":
                            ok, err = await kucoin_place_market_order(sym, "sell", pos.qty)
                            if not ok:
                                continue

                        net = close_position(sym, price, st, "TP")

                        if net < 0:
                            STATE.loss_streak += 1
                        else:
                            STATE.loss_streak = 0

                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🎯 TP EXIT {sym} @ {price:.4f} | NET {net:+.4f}"
                            )

                        # Loss guard
                        if STATE.loss_guard_on and STATE.loss_streak >= STATE.loss_guard_n:
                            STATE.paused_until = datetime.now(timezone.utc) + timedelta(
                                minutes=STATE.loss_guard_pause_min
                            )
                            STATE.loss_streak = 0

                        continue

                    # -------- TRAILING STOP --------
                    if pos.trailing:
                        if pos.side == "LONG":
                            trail_stop = pos.high_water * (1 - STATE.trail_pct / 100)
                            if price <= trail_stop:

                                if STATE.trade_mode == "live":
                                    ok, err = await kucoin_place_market_order(sym, "sell", pos.qty)
                                    if not ok:
                                        continue

                                net = close_position(sym, price, st, "TRAIL")

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🏁 TRAIL EXIT {sym} | NET {net:+.4f}"
                                    )

                                continue

                        else:  # SHORT
                            trail_stop = pos.low_water * (1 + STATE.trail_pct / 100)
                            if price >= trail_stop:
                                net = close_position(sym, price, st, "TRAIL")

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🏁 TRAIL EXIT {sym} | NET {net:+.4f}"
                                    )

                                continue

                    # -------- STOP LOSS --------
                    if move_pct <= -STATE.sl_pct:

                        if STATE.trade_mode == "live" and pos.side == "LONG":
                            ok, err = await kucoin_place_market_order(sym, "sell", pos.qty)
                            if not ok:
                                continue

                        net = close_position(sym, price, st, "SL")

                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"⛔ SL EXIT {sym} | NET {net:+.4f}"
                            )

                        continue

            await asyncio.sleep(3)

        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass

            await asyncio.sleep(5)# ------------------------------------------------------------
# TELEGRAM UI HELPERS
# ------------------------------------------------------------

def reply_kb() -> ReplyKeyboardMarkup:
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
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    pos_lines = []

    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            p = st.pos
            move_pct = (p.high_water - p.entry_price) / p.entry_price * 100.0

            pos_lines.append(
                f"{s}: {p.side} @ {p.entry_price:.4f} | qty={p.qty:.6f} | "
                f"hi={p.high_water:.4f} | lo={p.low_water:.4f} | "
                f"max_move≈{move_pct:.2f}% | regime={p.regime},src={p.reason}"
            )

    regime_line = (
        "Regime: AUTO (trend + mean reversion)"
        if STATE.regime_auto else "Regime: TREND only"
    )

    mr_line = (
        f"MR: {'ON' if STATE.mr_on else 'OFF'} | dev={STATE.mr_dev_pct:.2f}% | "
        f"range_atr_max={STATE.range_atr_max:.2f}%"
    )

    trend_line = f"Trend-slope min: {STATE.trend_slope_min:.2f}%"

    mode_line = (
        "Mode: LIVE (spot, endast LONG)"
        if STATE.trade_mode == "live"
        else "Mode: MOCK (simulerad handel)"
    )

    lg_line = "Loss-guard: OFF"
    if STATE.loss_guard_on:
        if STATE.paused_until and datetime.now(timezone.utc) < STATE.paused_until:
            rest = STATE.paused_until.astimezone().strftime("%H:%M")
            lg_line = (
                f"Loss-guard: ON | N={STATE.loss_guard_n} | pause={STATE.loss_guard_pause_min}m "
                f"(aktiv paus till ca {rest})"
            )
        else:
            lg_line = (
                f"Loss-guard: ON | N={STATE.loss_guard_n} | pause={STATE.loss_guard_pause_min}m"
            )

    text = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        "Strategi: Hybrid Momentum + Mean Reversion",
        regime_line,
        mr_line,
        trend_line,
        lg_line,
        mode_line,
        f"Threshold (MOMO): {STATE.threshold:.2f}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Mock-size: {STATE.mock_size:.1f} USDT | Fee per sida: {STATE.fee_side:.4%}",
        f"Risk: tp={STATE.tp_pct:.2f}% | sl={STATE.sl_pct:.2f}% | "
        f"trail_start={STATE.trail_start_pct:.2f}% | trail={STATE.trail_pct:.2f}% | "
        f"max_pos={STATE.max_pos}",
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]

    return "\n".join(text)


# ------------------------------------------------------------
# TELEGRAM COMMANDS
# ------------------------------------------------------------

tg_app = Application.builder().token(BOT_TOKEN).build()


async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id

    await tg_app.bot.send_message(
        STATE.chat_id,
        "🤖 MP Bot – Hybrid Momentum + Mean Reversion\n"
        "Mode: MOCK (default)\n"
        "Starta engine: /engine_on\n"
        "LIVE/Mock: /mode\n"
        "Threshold: /threshold 0.55\n",
        reply_markup=reply_kb(),
    )

    await tg_app.bot.send_message(
        STATE.chat_id, status_text(), reply_markup=reply_kb()
    )


async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(
        STATE.chat_id, status_text(), reply_markup=reply_kb()
    )


async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    lines = [f"📈 Total NET PnL: {total:+.4f} USDT"]

    for s in STATE.symbols:
        lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl:+.4f} USDT")

    await tg_app.bot.send_message(
        STATE.chat_id, "\n".join(lines), reply_markup=reply_kb()
    )


async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(
        STATE.chat_id,
        "Engine: ON ✅",
        reply_markup=reply_kb(),
    )


async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(
        STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb()
    )


async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip().split(" ", 1)

    if len(msg) == 2:
        tfs = [x.strip() for x in msg[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(
                STATE.chat_id,
                f"Timeframes uppdaterade: {', '.join(tfs)}",
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
    toks = update.message.text.split()

    if len(toks) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Aktuellt threshold: {STATE.threshold:.2f}",
            reply_markup=reply_kb(),
        )
        return

    try:
        v = float(toks[1])
        STATE.threshold = v
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Threshold satt till {v:.2f}",
            reply_markup=reply_kb(),
        )
    except:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Fel värde. Ex: /threshold 0.55",
            reply_markup=reply_kb(),
        )


# ------------------------------------------------------------
# RISK – INLINE BUTTON LOGIC
# ------------------------------------------------------------

def _yes(s: str) -> bool:
    return s.lower() in ("yes", "ja", "true", "1", "on")


async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id

    parts = update.message.text.split()

    # -------------------------
    # ADVANCED:
    # /risk set key value
    # -------------------------
    if len(parts) == 4 and parts[1] == "set":
        k, v = parts[2], parts[3]

        try:
            if k == "size":
                STATE.mock_size = float(v)
            elif k == "tp":
                STATE.tp_pct = float(v)
            elif k == "sl":
                STATE.sl_pct = float(v)
            elif k == "trail_start":
                STATE.trail_start_pct = float(v)
            elif k == "trail":
                STATE.trail_pct = float(v)
            elif k == "max_pos":
                STATE.max_pos = int(v)

            await tg_app.bot.send_message(
                STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb()
            )

        except:
            await tg_app.bot.send_message(
                STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb()
            )
        return

    # -------------------------
    # RISK QUICK BUTTONS
    # -------------------------
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("10 USDT", callback_data="risk_size_10"),
            InlineKeyboardButton("30 USDT", callback_data="risk_size_30"),
            InlineKeyboardButton("50 USDT", callback_data="risk_size_50"),
        ]
    ])

    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Aktuellt size: {STATE.mock_size:.1f} USDT.\n"
        "Ändra snabbt:",
        reply_markup=kb,
    )# ------------------------------------------------------------
# MODE (MOCK / LIVE)
# ------------------------------------------------------------

async def cmd_mode(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()

    # ------------------------
    # /mode  → visa knappar
    # ------------------------
    if len(toks) == 1:
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("MOCK", callback_data="mode_choose_mock"),
                    InlineKeyboardButton("LIVE", callback_data="mode_choose_live"),
                ]
            ]
        )

        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Aktuellt läge: {STATE.trade_mode.upper()}\nVälj nytt läge:",
            reply_markup=kb,
        )
        return

    # ------------------------
    # /mode mock
    # ------------------------
    if toks[1].lower() == "mock":
        STATE.trade_mode = "mock"
        STATE.allow_shorts = True

        await tg_app.bot.send_message(
            STATE.chat_id,
            "Läge satt till MOCK (simulerat). Shorts ON.",
            reply_markup=reply_kb(),
        )
        return

    # ------------------------
    # /mode live  (+ ev. JA)
    # ------------------------
    if toks[1].lower() == "live":

        # FÖRST → fråga om bekräftelse
        if len(toks) < 3 or toks[2].upper() != "JA":

            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("✅ JA, aktivera LIVE", callback_data="mode_live_yes"),
                        InlineKeyboardButton("❌ Nej", callback_data="mode_live_no"),
                    ]
                ]
            )

            await tg_app.bot.send_message(
                STATE.chat_id,
                "⚠️ Är du säker att du vill aktivera LIVE?\n"
                "Detta skickar riktiga market-ordrar på KuCoin.",
                reply_markup=kb,
            )
            return

        # /mode live JA direkt
        if not kucoin_creds_ok():
            await tg_app.bot.send_message(
                STATE.chat_id,
                "❌ LIVE misslyckades: API-nycklar saknas.",
                reply_markup=reply_kb(),
            )
            return

        STATE.trade_mode = "live"
        STATE.allow_shorts = False

        await tg_app.bot.send_message(
            STATE.chat_id,
            "✅ LIVE-läge AKTIVERAT.\n(shorts OFF – bara LONG på spot)",
            reply_markup=reply_kb(),
        )
        return

    await tg_app.bot.send_message(
        STATE.chat_id,
        "Använd: /mode, /mode mock, /mode live, /mode live JA",
        reply_markup=reply_kb(),
    )


# ------------------------------------------------------------
# CLOSE ALL POSITIONS
# ------------------------------------------------------------

async def cmd_close_all(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = 0
    total_net = 0.0

    for sym in STATE.symbols:
        st = STATE.per_sym[sym]
        if not st.pos:
            continue

        pos = st.pos

        # Hämta senaste pris (för mock-exit)
        tf = STATE.tfs[0]
        try:
            kl = await get_klines(sym, tf, limit=2)
            price = compute_features(kl)["close"]
        except Exception:
            price = pos.entry_price

        # LIVE: sälj innan close_position()
        if STATE.trade_mode == "live" and pos.side == "LONG":
            ok, err = await kucoin_place_market_order(sym, "sell", pos.qty)
            if not ok:
                await tg_app.bot.send_message(
                    STATE.chat_id,
                    f"⚠️ LIVE SELL misslyckades {sym} – hoppar över.\n{err}"
                )
                continue

        net = close_position(sym, price, st, "MANUAL_CLOSE")
        closed += 1
        total_net += net

    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Close all klart.\nStängda positioner: {closed}\nTotal NET: {total_net:+.4f} USDT",
        reply_markup=reply_kb(),
    )


# ------------------------------------------------------------
# RESET PNL (endast minnet – loggfiler sparas)
# ------------------------------------------------------------

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id

    for s in STATE.symbols:
        st = STATE.per_sym[s]
        st.realized_pnl = 0.0
        st.trades_log.clear()

    await tg_app.bot.send_message(
        STATE.chat_id,
        "PnL nollställd (mock_trade_log.csv / real_trade_log.csv är oförändrade).",
        reply_markup=reply_kb(),
    )


# ------------------------------------------------------------
# EXPORT CSV (REALISERAD PNL)
# ------------------------------------------------------------

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id

    rows = [
        ["time", "symbol", "side", "entry", "exit",
         "gross", "fee_in", "fee_out", "net", "reason"]
    ]

    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([
                r["time"], r["symbol"], r["side"],
                r["entry"], r["exit"], r["gross"],
                r["fee_in"], r["fee_out"], r["net"], r["reason"]
            ])

    if len(rows) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Inga trades loggade ännu.",
            reply_markup=reply_kb(),
        )
        return

    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    buf.seek(0)

    await tg_app.bot.send_document(
        STATE.chat_id,
        document=io.BytesIO(buf.getvalue().encode("utf-8")),
        filename="trades_hybrid_momo.csv",
        caption="Export CSV",
    )# ------------------------------------------------------------
# TEST-BUY (NEW): välj coin → 5 USDT BUY → WAIT → SELL ALL
# ------------------------------------------------------------

async def cmd_test_buy(update: Update, _):
    STATE.chat_id = update.effective_chat.id

    if not kucoin_creds_ok():
        await tg_app.bot.send_message(
            STATE.chat_id,
            "❌ Kan inte göra test-köp: KuCoin API-nycklar saknas.",
            reply_markup=reply_kb(),
        )
        return

    # Visa knappar för alla coins
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("BTC", callback_data="testbuy_BTC-USDT"),
            InlineKeyboardButton("ETH", callback_data="testbuy_ETH-USDT"),
        ],
        [
            InlineKeyboardButton("ADA", callback_data="testbuy_ADA-USDT"),
            InlineKeyboardButton("XRP", callback_data="testbuy_XRP-USDT"),
        ],
        [
            InlineKeyboardButton("LINK", callback_data="testbuy_LINK-USDT"),
        ]
    ])

    await tg_app.bot.send_message(
        STATE.chat_id,
        "🧪 Välj coin att testköpa för 5 USDT:",
        reply_markup=kb,
    )


# ------------------------------------------------------------
# HÄMTA BALANS (för SELL ALL)
# ------------------------------------------------------------

async def kucoin_get_balance(symbol_base: str):
    """
    Returnerar qty för basvalutan (ex: BTC, ETH, ADA).
    """
    path = "/api/v1/accounts"
    data = await kucoin_private_request("GET", path)

    if not data or "data" not in data:
        return 0.0

    for acct in data["data"]:
        if acct["currency"] == symbol_base and acct["type"] == "trade":
            try:
                return float(acct["available"])
            except:
                return 0.0

    return 0.0


# ------------------------------------------------------------
# CALLBACK – TESTBUY PER COIN
# ------------------------------------------------------------

async def on_button(update: Update, _):
    query = update.callback_query
    data = query.data
    chat_id = query.message.chat_id
    STATE.chat_id = chat_id

    # --------------------------------------------------------
    # QUICK RISK BUTTONS
    # --------------------------------------------------------
    if data.startswith("risk_size_"):
        await query.answer()
        size = float(data.split("_")[-1])
        STATE.mock_size = size
        await query.edit_message_text(f"Size uppdaterad till {size} USDT.")
        return

    # --------------------------------------------------------
    # MODE KNAPPAR
    # --------------------------------------------------------
    if data == "mode_choose_mock":
        await query.answer()
        STATE.trade_mode = "mock"
        STATE.allow_shorts = True
        await query.edit_message_text("Läge satt till MOCK (simulerad handel).")
        return

    if data == "mode_choose_live":
        await query.answer()
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ JA, aktivera LIVE", callback_data="mode_live_yes"),
                InlineKeyboardButton("❌ Nej", callback_data="mode_live_no"),
            ]
        ])
        await query.edit_message_text(
            "⚠️ Bekräfta LIVE-läge:",
            reply_markup=kb,
        )
        return

    if data == "mode_live_no":
        await query.answer()
        await query.edit_message_text("Avbrutet.")
        return

    if data == "mode_live_yes":
        await query.answer()
        if not kucoin_creds_ok():
            await query.edit_message_text("❌ Kan inte aktivera LIVE – API saknas.")
            return
        STATE.trade_mode = "live"
        STATE.allow_shorts = False
        await query.edit_message_text("✅ LIVE-läge AKTIVERAT.")
        return

    # --------------------------------------------------------
    # TESTBUY PER COIN KNAPPAR
    # --------------------------------------------------------
    if data.startswith("testbuy_"):
        await query.answer()

        symbol = data.replace("testbuy_", "")  # t.ex. "BTC-USDT"
        base = symbol.split("-")[0]           # t.ex. "BTC"

        # -------------------------
        # 1) BUY 5 USDT
        # -------------------------
        ok, err = await kucoin_place_market_order(symbol, "buy", 5.0)

        if not ok:
            await query.edit_message_text(
                f"❌ LIVE BUY misslyckades {symbol}\n{err}"
            )
            return

        await query.edit_message_text(
            f"🟢 Test BUY skickad: {symbol} för 5 USDT.\nSäljer strax allt..."
        )

        # Litet vänt
        await asyncio.sleep(2)

        # -------------------------
        # 2) HÄMTA QTY → SELL ALL
        # -------------------------
        qty = await kucoin_get_balance(base)

        if qty <= 0:
            await tg_app.bot.send_message(
                chat_id,
                f"⚠️ Kunde inte hitta {base}-balans för SELL ALL."
            )
            return

        ok2, err2 = await kucoin_place_market_order(symbol, "sell", qty)

        if not ok2:
            await tg_app.bot.send_message(
                chat_id,
                f"❌ SELL ALL misslyckades {symbol}\n{err2}"
            )
            return

        await tg_app.bot.send_message(
            chat_id,
            f"🟣 SELL ALL skickad för {symbol}.\nQty={qty:.8f}"
        )

        return

    # --------------------------------------------------------
    # FALLBACK (andra knappar)
    # --------------------------------------------------------
    await query.answer()
    await query.edit_message_text("Ok.")# ------------------------------------------------------------
# WRAPPER: ENHETLIG MARKET ORDER (BUY = USDT, SELL = QTY)
# ------------------------------------------------------------

async def kucoin_place_market_order(symbol: str, side: str, amount: float):
    """
    Wrapper så all kod kan använda samma funktion.
    BUY:  amount = USDT-belopp (funds)
    SELL: amount = qty i bas-valuta (BTC, ETH, ADA, osv)
    """
    side_l = side.lower()
    if side_l == "buy":
        return await kucoin_place_market_buy(symbol, amount)
    else:
        return await kucoin_place_market_sell(symbol, amount)


# ------------------------------------------------------------
# REGISTRERA ALLA TELEGRAM-HANDLERS
# ------------------------------------------------------------

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("threshold", cmd_threshold))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("mode", cmd_mode))
tg_app.add_handler(CommandHandler("close_all", cmd_close_all))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("test_buy", cmd_test_buy))

tg_app.add_handler(CallbackQueryHandler(on_button))


# ------------------------------------------------------------
# FASTAPI + WEBHOOK (DIGITALOCEAN)
# ------------------------------------------------------------

app = FastAPI()


class TgUpdate(BaseModel):
    update_id: Optional[int] = None


@app.on_event("startup")
async def on_startup():
    # Sätt webhook om WEBHOOK_BASE finns (ex: https://urchin-app-fjigr.ondigitalocean.app)
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")

    await tg_app.initialize()
    await tg_app.start()

    # Starta trading-engine i bakgrunden
    asyncio.create_task(engine_loop(tg_app))


@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()


@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    return (
        f"MP Bot Hybrid Momo OK | "
        f"engine_on={STATE.engine_on} | mode={STATE.trade_mode} | "
        f"pnl_total={total:+.4f}"
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
    Hit skickar Telegram alla uppdateringar (med WEBHOOK_BASE satt).
    URL ska vara:
    https://din-app-url/webhook/<BOT_TOKEN>
    """
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
