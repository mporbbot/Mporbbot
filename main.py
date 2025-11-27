# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MP Bot – v54 (Hybrid Momentum + Mean Reversion, MOCK + LIVE)
# ------------------------------------------------------------
# - Mock-trades (simulerade) OCH riktig spot-handel på KuCoin
# - Livepriser från KuCoin
# - Hybrid-strategi:
#   * Trend-läge: Simple Momentum
#   * Range-läge: Mean Reversion runt EMA20
#   * Automatisk växling per symbol (trend vs sidled)
# - TP/SL i % + enkel trailing stop
# - Loss-guard: pausa efter X förlorande trades i rad
# - Telegram + FastAPI (DigitalOcean-kompatibel, via polling)
#
# LÄGEN:
#   trade_mode = "mock"  -> bara simulering (default)
#   trade_mode = "live"  -> riktiga spot-ordrar på KuCoin (endast LONG)
#
# Byt läge i Telegram:
#   /mode              -> visa läge + knappar MOCK/LIVE
#   /mode mock         -> tvinga MOCK
#   /mode live         -> instruktion / knappar
#   /mode live JA      -> AKTIVERA LIVE (om API-nycklar finns)
#
# Snabbkommandon:
#   /test_buy          -> fråga Ja/Nej och gör LIVE test-köp BTC-USDT 5 USDT
#   /risk              -> visa status + knappar size 10 / 30 / 50 USDT
#
# Miljövariabler (valfria):
#   TELEGRAM_BOT_TOKEN  eller BOT_TOKEN  (om du inte vill hårdkoda)
#
#   KUCOIN_API_KEY
#   KUCOIN_API_SECRET
#   KUCOIN_API_PASSPHRASE
#   KUCOIN_API_KEY_VERSION  (2 eller 3 – matcha det som står i KuCoin, t.ex. "3")
#   KUCOIN_BASE_URL         (standard "https://api.kucoin.com")
#
#   SYMBOLS            (t.ex. "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
#   TIMEFRAMES         (t.ex. "1m,3m,5m")
#   MOCK_SIZE_USDT     (default 50.0)
#   FEE_PER_SIDE       (default 0.001 = 0.1 % per sida)
#   MAX_POS            (default 4)
#
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
from telegram.ext import Application, CommandHandler, CallbackQueryHandler

# -----------------------------
# ENV & BOT TOKEN
# -----------------------------

# Hårdkodad token (från BotFather) + env-fallback
HARDCODED_TOKEN = "8265069090:AAF8W7l3-MNwyV8_DBRiLxUeEJHtGevURUg"

BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("BOT_TOKEN")
    or HARDCODED_TOKEN
)

if not BOT_TOKEN:
    raise RuntimeError("Ingen Telegram-token satt (varken env eller hårdkodad).")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")  # används inte längre, men kvar om du vill byta till webhook

DEFAULT_SYMBOLS = (
    os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
    .replace(" ", "")
).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")

# Standardinställningar
MOCK_SIZE_USDT = float(os.getenv("MOCK_SIZE_USDT", "50"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "4"))

TP_PCT = float(os.getenv("TP_PCT", "0.30"))       # take-profit i %
SL_PCT = float(os.getenv("SL_PCT", "0.50"))       # stop-loss i %
TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "2.0"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "2.5"))

ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.55"))  # momentum-score
ALLOW_SHORTS_DEFAULT = (
    os.getenv("ALLOW_SHORTS", "false").lower() in ("1", "true", "on", "yes")
)

# Mean Reversion / regime defaults
MR_DEV_PCT = float(os.getenv("MR_DEV_PCT", "1.20"))              # avvikelse från EMA20 i %
TREND_SLOPE_MIN = float(os.getenv("TREND_SLOPE_MIN", "0.20"))    # min lutning EMA20 (i %)
RANGE_ATR_MAX = float(os.getenv("RANGE_ATR_MAX", "0.80"))        # max ATR% för range

# Loss-guard defaults (pausa efter X förluster i rad)
LOSS_GUARD_ON_DEFAULT = True
LOSS_GUARD_N_DEFAULT = int(os.getenv("LOSS_GUARD_N", "2"))
LOSS_GUARD_PAUSE_MIN_DEFAULT = int(os.getenv("LOSS_GUARD_PAUSE_MIN", "15"))

# KuCoin private API (för LIVE)
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KUCOIN_API_KEY_VERSION = os.getenv("KUCOIN_API_KEY_VERSION", "2")
KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

# Publika klines
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}

# -----------------------------
# STATE
# -----------------------------

@dataclass
class Position:
    side: str
    entry_price: float
    qty: float
    opened_at: datetime
    high_water: float
    low_water: float
    trailing: bool = False
    regime: str = "trend"   # "trend" eller "range"
    reason: str = "MOMO"    # "MOMO" eller "MR"

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
    threshold: float = ENTRY_THRESHOLD      # momentum-threshold
    allow_shorts: bool = ALLOW_SHORTS_DEFAULT

    tp_pct: float = TP_PCT
    sl_pct: float = SL_PCT
    trail_start_pct: float = TRAIL_START_PCT
    trail_pct: float = TRAIL_PCT
    max_pos: int = MAX_OPEN_POS

    # Hybrid / regime
    mr_on: bool = True
    regime_auto: bool = True
    mr_dev_pct: float = MR_DEV_PCT
    trend_slope_min: float = TREND_SLOPE_MIN
    range_atr_max: float = RANGE_ATR_MAX

    # Trade-mode: "mock" eller "live"
    trade_mode: str = "mock"  # "mock" (simulerad) eller "live" (spot)

    # Loss-guard
    loss_guard_on: bool = LOSS_GUARD_ON_DEFAULT
    loss_guard_n: int = LOSS_GUARD_N_DEFAULT
    loss_guard_pause_min: int = LOSS_GUARD_PAUSE_MIN_DEFAULT
    loss_streak: int = 0
    paused_until: Optional[datetime] = None

    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# -----------------------------
# Helpers (loggar etc)
# -----------------------------

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

def kucoin_creds_ok() -> bool:
    return bool(KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE)

async def kucoin_private_request(method: str, path: str, body: Optional[dict] = None):
    """
    Enkel KuCoin private-request (för LIVE-ordrar).
    Returnerar alltid ett dict med JSON + _http_status,
    även vid fel (så vi kan visa exakt fel från KuCoin).
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

    # passphrase måste också hashas
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
            resp = await client.request(
                method.upper(), path, headers=headers, content=body_str
            )
    except Exception as e:
        return {
            "_http_status": None,
            "code": None,
            "msg": f"Request error: {e}",
        }

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text or ""}

    data["_http_status"] = resp.status_code
    return data

async def kucoin_place_market_order(symbol: str, side: str, amount: float):
    """
    Skicka en enkel market-order på spot.

    side: "buy" eller "sell"
    amount:
        - För BUY: USDT-belopp (funds)
        - För SELL: mängd bas-valuta (size), t.ex. BTC

    Returnerar (ok: bool, err: str)
    """
    side_l = side.lower()
    body = {
        "clientOid": str(uuid.uuid4()),
        "side": side_l,
        "symbol": symbol,
        "type": "market",
    }

    if side_l == "buy":
        # Viktigt: KuCoin kräver 'funds' för market BUY (quote-valuta, t.ex. USDT)
        body["funds"] = f"{amount:.2f}"
    else:
        # Market SELL använder 'size' (bas-valuta, t.ex. BTC)
        body["size"] = f"{amount:.8f}"

    data = await kucoin_private_request("POST", "/api/v1/orders", body)
    if not data:
        return False, "Inget svar från KuCoin (data=None)."

    status = data.get("_http_status")
    code = data.get("code")
    msg = data.get("msg") or data.get("message") or data.get("raw") or ""

    if status != 200 or code != "200000":
        err_text = f"HTTP={status}, code={code}, msg={msg}"
        return False, err_text

    return True, ""

# -----------------------------
# Data & Indicators
# -----------------------------

async def get_klines(symbol: str, tf: str, limit: int = 100):
    tf_api = TF_MAP.get(tf, tf)
    params = {"symbol": symbol, "type": tf_api}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]
    return data[::-1][:limit]

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    k = 2.0 / (period + 1.0)
    out = []
    val = series[0]
    for x in series:
        val = (x - val) * k + val
        out.append(val)
    return out

def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    gains = []
    losses = []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(-min(ch, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out = [50.0] * period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rs = 999.0
        else:
            rs = avg_gain / avg_loss
        val = 100.0 - (100.0 / (1.0 + rs))
        out.append(val)
    return [50.0] + out

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
            "rsi": 50.0,
            "mom": 0.0,
            "atrp": 0.2,
            "trend_slope": 0.0,
        }

    ema20_series = ema(closes, 20)
    ema50_series = ema(closes, 50)
    ema20 = ema20_series[-1]
    ema50 = ema50_series[-1]

    mom = (closes[-1] - closes[-6]) / (closes[-6] or 1.0) * 100.0
    rsi_last = rsi(closes, 14)[-1]

    # ATR% (snitt av true-range/close * 100)
    trs = []
    for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:]):
        tr = (h - l) / (c or 1.0) * 100.0
        trs.append(tr)
    atrp = sum(trs) / len(trs) if trs else 0.2

    # trend-slope på EMA20, ca 5 candles tillbaka
    if len(ema20_series) > 6 and ema20_series[-6] != 0:
        trend_slope = (ema20_series[-1] - ema20_series[-6]) / ema20_series[-6] * 100.0
    else:
        trend_slope = 0.0

    return {
        "close": closes[-1],
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi_last,
        "mom": mom,
        "atrp": atrp,
        "trend_slope": trend_slope,
    }

def momentum_score(feats) -> float:
    ema20 = feats["ema20"]
    ema50 = feats["ema50"]
    mom = feats["mom"]
    rsi_val = feats["rsi"]
    # trend + momentum + rsi
    if ema20 > ema50:
        trend = 1.0
    elif ema20 < ema50:
        trend = -1.0
    else:
        trend = 0.0
    rsi_dev = (rsi_val - 50.0) / 10.0  # +/- 5 när RSI 0/100
    score = trend * (abs(mom) / 0.1) + rsi_dev
    # signen på mom avgör riktning
    if mom < 0:
        score = -abs(score)
    else:
        score = abs(score)
    if trend < 0:
        score = -score
    return score

def decide_regime(feats) -> str:
    """
    Returnerar 'trend' eller 'range' baserat på ATR och lutning.
    """
    if not STATE.regime_auto:
        return "trend"  # fallback = bara momentum

    atrp = feats.get("atrp", 0.0)
    slope = abs(feats.get("trend_slope", 0.0))

    # Hög ATR + tydlig lutning => trend
    if atrp >= STATE.range_atr_max and slope >= STATE.trend_slope_min:
        return "trend"
    # Låg ATR + platt lutning => range
    if atrp <= STATE.range_atr_max and slope < STATE.trend_slope_min:
        return "range"
    # Default: trend
    return "trend"

# -----------------------------
# Trading helpers (mock + live logging)
# -----------------------------

def _fee(amount_usdt: float) -> float
