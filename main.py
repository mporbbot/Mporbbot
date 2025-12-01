# -*- coding: utf-8 -*-
# MP Bot v55 – Hybrid Momentum + Mean Reversion
# MOCK + LIVE (KuCoin spot, endast LONG i LIVE)
# Live-PnL baserad på riktiga fills från KuCoin (funds + fee)
# /test_buy kan köpa valfritt coin och positionen ingår i PnL/engine

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
import logging

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mp_bot_v55")

# --------------------------------------------------
# ENV
# --------------------------------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (
    os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
    .replace(" ", "")
).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "3m").replace(" ", "")).split(",")

MOCK_SIZE_USDT = float(os.getenv("MOCK_SIZE_USDT", "10"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))  # 0.1% per sida
MAX_OPEN_POS = int(os.getenv("MAX_POS", "4"))

TP_PCT = float(os.getenv("TP_PCT", "0.30"))       # take-profit i %
SL_PCT = float(os.getenv("SL_PCT", "0.50"))       # stop-loss i %
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

# KuCoin private API
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KUCOIN_API_KEY_VERSION = os.getenv("KUCOIN_API_KEY_VERSION", "3")
KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}

logger.info(
    "[KUCOIN CREDS] key_len=%d, secret_len=%d, passphrase_len=%d, version='%s'",
    len(KUCOIN_API_KEY or ""),
    len(KUCOIN_API_SECRET or ""),
    len(KUCOIN_API_PASSPHRASE or ""),
    KUCOIN_API_KEY_VERSION,
)

# --------------------------------------------------
# STATE
# --------------------------------------------------
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
    # Live-exakt PnL-data:
    usd_in: float = 0.0        # verkligt spenderat USDT (entry)
    fee_in: float = 0.0        # verklig fee vid entry
    entry_order_id: Optional[str] = None  # KuCoin orderId för entry


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

    trade_mode: str = "mock"  # "mock" eller "live"

    loss_guard_on: bool = LOSS_GUARD_ON_DEFAULT
    loss_guard_n: int = LOSS_GUARD_N_DEFAULT
    loss_guard_pause_min: int = LOSS_GUARD_PAUSE_MIN_DEFAULT
    loss_streak: int = 0
    paused_until: Optional[datetime] = None

    chat_id: Optional[int] = None


STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# --------------------------------------------------
# Helpers & logging
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
                "gross", "fee_in", "fee_out", "net", "info",
            ],
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
                "gross", "fee_in", "fee_out", "net", "info",
            ],
        )
        if new:
            w.writeheader()
        w.writerow(row)


def kucoin_creds_ok() -> bool:
    return bool(KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE)


async def kucoin_private_request(
    method: str,
    path: str,
    body: Optional[dict] = None,
) -> Optional[dict]:
    """
    Enkel KuCoin private-request (V3) för Spot.
    path = t.ex. "/api/v1/orders" eller "/api/v1/fills?orderId=xxx&tradeType=TRADE"
    Returnerar dict med JSON + _http_status.
    """
    if not kucoin_creds_ok():
        return None

    body = body or {}
    body_str = json.dumps(body, separators=(",", ":")) if method.upper() != "GET" else ""
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
            resp = await client.request(
                method.upper(), path, headers=headers, content=body_str or None
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


async def kucoin_get_fills_for_order(order_id: str) -> Optional[dict]:
    """
    Hämtar fills för en spot-order (TRADES).
    Returnerar dict med:
      {
        "orderId": order_id,
        "size": total_size,
        "funds": total_funds,
        "fee": total_fee,
        "avg_price": avg_price
      }
    eller None vid fel.
    """
    if not kucoin_creds_ok():
        return None

    path = f"/api/v1/fills?orderId={order_id}&tradeType=TRADE"

    fills: List[dict] = []
    for _ in range(6):
        data = await kucoin_private_request("GET", path)
        if not data:
            await asyncio.sleep(0.2)
            continue

        status = data.get("_http_status")
        code = data.get("code")
        if status != 200 or code != "200000":
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
            total_size += float(f.get("size", 0) or 0)
            total_funds += float(f.get("funds", 0) or 0)
            total_fee += float(f.get("fee", 0) or 0)
        except Exception:
            continue

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


async def kucoin_place_market_order(symbol: str, side: str, amount: float):
    """
    Skicka en enkel market-order på spot.
    side: "buy" eller "sell".
    amount:
      - BUY: USDT-belopp (funds)
      - SELL: bas-kvantitet (size)

    Returnerar (ok: bool, err: str, info: Optional[dict]) där info vid ok är:
      {
        "orderId": ...,
        "size": ...,
        "funds": ...,
        "fee": ...,
        "avg_price": ...
      }
    """
    if not kucoin_creds_ok():
        return False, "Inga KuCoin-creds satta.", None

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
        body["size"] = f"{amount:.8f}"

    data = await kucoin_private_request("POST", "/api/v1/orders", body)
    if not data:
        return False, "Inget svar från KuCoin (data=None).", None

    status = data.get("_http_status")
    code = data.get("code")
    msg = data.get("msg") or data.get("message") or data.get("raw") or ""

    if status != 200 or code != "200000":
        err_text = f"HTTP={status}, code={code}, msg={msg}"
        return False, err_text, None

    od = data.get("data") or {}
    order_id = od.get("orderId") or od.get("id")
    if not order_id:
        return False, "OrderId saknas i KuCoin-svar.", None

    fills_info = await kucoin_get_fills_for_order(order_id)
    if not fills_info:
        return True, "", {
            "orderId": order_id,
            "size": float(body.get("size", 0) or 0),
            "funds": float(body.get("funds", amount) or amount),
            "fee": 0.0,
            "avg_price": None,
        }

    return True, "", fills_info


# --------------------------------------------------
# Data & Indicators
# --------------------------------------------------
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

    trs = []
    for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:]):
        tr = (h - l) / (c or 1.0) * 100.0
        trs.append(tr)
    atrp = sum(trs) / len(trs) if trs else 0.2

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

    if ema20 > ema50:
        trend = 1.0
    elif ema20 < ema50:
        trend = -1.0
    else:
        trend = 0.0

    rsi_dev = (rsi_val - 50.0) / 10.0
    score = trend * (abs(mom) / 0.1) + rsi_dev
    if mom < 0:
        score = -abs(score)
    else:
        score = abs(score)
    if trend < 0:
        score = -score
    return score


def decide_regime(feats) -> str:
    if not STATE.regime_auto:
        return "trend"

    atrp = feats.get("atrp", 0.0)
    slope = abs(feats.get("trend_slope", 0.0))

    if atrp >= STATE.range_atr_max and slope >= STATE.trend_slope_min:
        return "trend"
    if atrp <= STATE.range_atr_max and slope < STATE.trend_slope_min:
        return "range"
    return "trend"


# --------------------------------------------------
# Trading helpers
# --------------------------------------------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side


def _log_trade(row: Dict) -> None:
    if STATE.trade_mode == "live":
        log_real(row)
    else:
        log_mock(row)


def open_position(
    sym: str,
    side: str,
    price: float,
    st: SymState,
    regime: str,
    reason: str,
    live_info: Optional[dict] = None,
):
    """
    Öppna position.
    - MOCK: size = STATE.mock_size / price
    - LIVE: använder verkliga size/funds/fee från live_info
    """
    if STATE.trade_mode == "live" and live_info:
        qty = float(live_info.get("size", 0) or 0.0)
        entry_price = float(live_info.get("avg_price") or price)
        usd_in = float(live_info.get("funds", 0) or 0.0)
        fee_in = float(live_info.get("fee", 0) or 0.0)
        entry_order_id = live_info.get("orderId")
    else:
        entry_price = price
        qty = STATE.mock_size / price if price > 0 else 0.0
        usd_in = qty * entry_price
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
        entry_order_id=entry_order_id,
    )

    _log_trade(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "action": "ENTRY",
            "side": side,
            "price": round(entry_price, 6),
            "qty": round(qty, 8),
            "gross": "",
            "fee_in": round(fee_in, 6) if fee_in else "",
            "fee_out": "",
            "net": "",
            "info": f"size_usdt={STATE.mock_size};regime={regime};reason={reason};mode={STATE.trade_mode}",
        }
    )


def close_position(
    sym: str,
    st: SymState,
    reason: str,
    live_sell_info: Optional[dict] = None,
    approx_price: Optional[float] = None,
) -> float:
    """
    Stänger positionen och räknar PnL.
    - LIVE: använder verkliga funds/fee från buy + sell (live_sell_info)
    - MOCK: använder approx-pris + modell-fee.
    """
    if not st.pos:
        return 0.0

    pos = st.pos

    if STATE.trade_mode == "live" and live_sell_info:
        usd_in = pos.usd_in
        fee_in = pos.fee_in
        usd_out = float(live_sell_info.get("funds", 0) or 0.0)
        fee_out = float(live_sell_info.get("fee", 0) or 0.0)
        exit_price = float(live_sell_info.get("avg_price") or approx_price or pos.entry_price)
        gross = usd_out - usd_in
        net = gross - fee_in - fee_out
    else:
        price = approx_price or pos.entry_price
        usd_in = pos.qty * pos.entry_price
        usd_out = pos.qty * price
        fee_in = _fee(usd_in)
        fee_out = _fee(usd_out)
        if pos.side == "LONG":
            gross = pos.qty * (price - pos.entry_price)
        else:
            gross = pos.qty * (pos.entry_price - price)
        net = gross - fee_in - fee_out
        exit_price = price

    st.realized_pnl += net
    st.trades_log.append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "side": pos.side,
            "entry": pos.entry_price,
            "exit": exit_price,
            "gross": round(gross, 6),
            "fee_in": round(fee_in, 6),
            "fee_out": round(fee_out, 6),
            "net": round(net, 6),
            "reason": f"{reason};regime={pos.regime};src={pos.reason};mode={STATE.trade_mode}",
        }
    )
    _log_trade(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "action": "EXIT",
            "side": pos.side,
            "price": round(exit_price, 6),
            "qty": round(pos.qty, 8),
            "gross": round(gross, 6),
            "fee_in": round(fee_in, 6),
            "fee_out": round(fee_out, 6),
            "net": round(net, 6),
            "info": f"{reason};regime={pos.regime};src={pos.reason};mode={STATE.trade_mode}",
        }
    )
    st.pos = None
    return net


# --------------------------------------------------
# ENGINE LOOP
# --------------------------------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                now = datetime.now(timezone.utc)

                if (
                    STATE.loss_guard_on
                    and STATE.paused_until is not None
                    and now < STATE.paused_until
                ):
                    await asyncio.sleep(3)
                    continue

                # ------ Entries ------
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.max_pos:
                    for sym in STATE.symbols:
                        st = STATE.per_sym[sym]
                        if st.pos:
                            continue
                        tf = STATE.tfs[0] if STATE.tfs else "3m"
                        try:
                            kl = await get_klines(sym, tf, limit=80)
                        except Exception:
                            continue
                        feats = compute_features(kl)
                        price = feats["close"]
                        regime = decide_regime(feats)

                        allow_shorts_effective = (
                            STATE.allow_shorts if STATE.trade_mode == "mock" else False
                        )

                        if regime == "trend":
                            score = momentum_score(feats)
                            if score > STATE.threshold:
                                live_info = None
                                if STATE.trade_mode == "live":
                                    ok, err, info = await kucoin_place_market_order(
                                        sym, "buy", STATE.mock_size
                                    )
                                    if not ok:
                                        if STATE.chat_id:
                                            await app.bot.send_message(
                                                STATE.chat_id,
                                                f"⚠️ LIVE BUY {sym} misslyckades (MOMO).\n{err}",
                                            )
                                        continue
                                    live_info = info
                                    if live_info and live_info.get("avg_price"):
                                        price = live_info["avg_price"]

                                open_position(
                                    sym,
                                    "LONG",
                                    price,
                                    st,
                                    regime="trend",
                                    reason="MOMO",
                                    live_info=live_info,
                                )
                                open_syms.append(sym)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 MOMO ENTRY {sym} LONG @ {price:.4f} | "
                                        f"score={score:.2f} | thr={STATE.threshold:.2f} | "
                                        f"regime=trend,mode={STATE.trade_mode}",
                                    )
                                if len(open_syms) >= STATE.max_pos:
                                    break

                            elif allow_shorts_effective and score < -STATE.threshold:
                                open_position(
                                    sym,
                                    "SHORT",
                                    price,
                                    st,
                                    regime="trend",
                                    reason="MOMO",
                                    live_info=None,
                                )
                                open_syms.append(sym)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 MOMO ENTRY {sym} SHORT @ {price:.4f} | "
                                        f"score={score:.2f} | thr={STATE.threshold:.2f} | "
                                        f"regime=trend,mode={STATE.trade_mode}",
                                    )
                                if len(open_syms) >= STATE.max_pos:
                                    break

                        elif regime == "range" and STATE.mr_on:
                            ema20 = feats["ema20"]
                            if ema20 == 0:
                                continue
                            dev_pct = (price - ema20) / ema20 * 100.0

                            if dev_pct <= -STATE.mr_dev_pct:
                                live_info = None
                                if STATE.trade_mode == "live":
                                    ok, err, info = await kucoin_place_market_order(
                                        sym, "buy", STATE.mock_size
                                    )
                                    if not ok:
                                        if STATE.chat_id:
                                            await app.bot.send_message(
                                                STATE.chat_id,
                                                f"⚠️ LIVE BUY {sym} misslyckades (MR).\n{err}",
                                            )
                                        continue
                                    live_info = info
                                    if live_info and live_info.get("avg_price"):
                                        price = live_info["avg_price"]

                                open_position(
                                    sym,
                                    "LONG",
                                    price,
                                    st,
                                    regime="range",
                                    reason="MR",
                                    live_info=live_info,
                                )
                                open_syms.append(sym)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 MR ENTRY {sym} LONG @ {price:.4f} | "
                                        f"dev={dev_pct:.2f}% | regime=range,mode={STATE.trade_mode}",
                                    )
                                if len(open_syms) >= STATE.max_pos:
                                    break

                            elif dev_pct >= STATE.mr_dev_pct and allow_shorts_effective:
                                open_position(
                                    sym,
                                    "SHORT",
                                    price,
                                    st,
                                    regime="range",
                                    reason="MR",
                                    live_info=None,
                                )
                                open_syms.append(sym)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 MR ENTRY {sym} SHORT @ {price:.4f} | "
                                        f"dev={dev_pct:.2f}% | regime=range,mode={STATE.trade_mode}",
                                    )
                                if len(open_syms) >= STATE.max_pos:
                                    break

                # ------ Manage positions ------
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos:
                        continue
                    tf = STATE.tfs[0] if STATE.tfs else "3m"
                    try:
                        kl = await get_klines(sym, tf, limit=5)
                    except Exception:
                        continue
                    feats = compute_features(kl)
                    price = feats["close"]
                    pos = st.pos

                    pos.high_water = max(pos.high_water, price)
                    pos.low_water = min(pos.low_water, price)

                    move_pct = (price - pos.entry_price) / (pos.entry_price or 1.0) * 100.0
                    if pos.side == "SHORT":
                        move_pct = -move_pct

                    # Starta trailing
                    if (not pos.trailing) and move_pct >= STATE.trail_start_pct:
                        pos.trailing = True
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🔒 TRAIL ON {sym} | move≈{move_pct:.2f}% "
                                f"(regime={pos.regime},src={pos.reason},mode={STATE.trade_mode})",
                            )

                    # TP (ej trailing)
                    if move_pct >= STATE.tp_pct and not pos.trailing:
                        live_info = None
                        if STATE.trade_mode == "live" and pos.side == "LONG":
                            ok, err, info = await kucoin_place_market_order(
                                sym, "sell", pos.qty
                            )
                            if not ok:
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"⚠️ LIVE TP-sälj {sym} misslyckades.\n{err}",
                                    )
                                continue
                            live_info = info
                            if live_info and live_info.get("avg_price"):
                                price = live_info["avg_price"]

                        net = close_position(
                            sym,
                            st,
                            reason="TP",
                            live_sell_info=live_info,
                            approx_price=price,
                        )
                        if net < 0:
                            STATE.loss_streak += 1
                        else:
                            STATE.loss_streak = 0
                        if STATE.chat_id:
                            mark = "✅" if net >= 0 else "❌"
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🎯 TP EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}",
                            )
                        if (
                            STATE.loss_guard_on
                            and STATE.loss_streak >= STATE.loss_guard_n
                        ):
                            STATE.paused_until = datetime.now(timezone.utc) + timedelta(
                                minutes=STATE.loss_guard_pause_min
                            )
                            STATE.loss_streak = 0
                            if STATE.chat_id:
                                pause_t = STATE.paused_until.astimezone().strftime(
                                    "%H:%M"
                                )
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🛑 Loss-guard: pausar nya entries till ca {pause_t}.",
                                )
                        continue

                    # Trailing stop
                    if pos.trailing:
                        if pos.side == "LONG":
                            trail_stop = pos.high_water * (
                                1.0 - STATE.trail_pct / 100.0
                            )
                            if price <= trail_stop:
                                live_info = None
                                if STATE.trade_mode == "live":
                                    ok, err, info = await kucoin_place_market_order(
                                        sym, "sell", pos.qty
                                    )
                                    if not ok:
                                        if STATE.chat_id:
                                            await app.bot.send_message(
                                                STATE.chat_id,
                                                f"⚠️ LIVE TRAIL-sälj {sym} misslyckades.\n{err}",
                                            )
                                        continue
                                    live_info = info
                                    if live_info and live_info.get("avg_price"):
                                        price = live_info["avg_price"]

                                net = close_position(
                                    sym,
                                    st,
                                    reason="TRAIL",
                                    live_sell_info=live_info,
                                    approx_price=price,
                                )
                                if net < 0:
                                    STATE.loss_streak += 1
                                else:
                                    STATE.loss_streak = 0
                                if STATE.chat_id:
                                    mark = "✅" if net >= 0 else "❌"
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}",
                                    )
                                if (
                                    STATE.loss_guard_on
                                    and STATE.loss_streak >= STATE.loss_guard_n
                                ):
                                    STATE.paused_until = datetime.now(
                                        timezone.utc
                                    ) + timedelta(
                                        minutes=STATE.loss_guard_pause_min
                                    )
                                    STATE.loss_streak = 0
                                    if STATE.chat_id:
                                        pause_t = (
                                            STATE.paused_until.astimezone().strftime(
                                                "%H:%M"
                                            )
                                        )
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"🛑 Loss-guard: pausar nya entries till ca {pause_t}.",
                                        )
                                continue
                        else:
                            # SHORT trail (mock only)
                            trail_stop = pos.low_water * (
                                1.0 + STATE.trail_pct / 100.0
                            )
                            if price >= trail_stop:
                                net = close_position(
                                    sym,
                                    st,
                                    reason="TRAIL",
                                    live_sell_info=None,
                                    approx_price=price,
                                )
                                if net < 0:
                                    STATE.loss_streak += 1
                                else:
                                    STATE.loss_streak = 0
                                if STATE.chat_id:
                                    mark = "✅" if net >= 0 else "❌"
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}",
                                    )
                                if (
                                    STATE.loss_guard_on
                                    and STATE.loss_streak >= STATE.loss_guard_n
                                ):
                                    STATE.paused_until = datetime.now(
                                        timezone.utc
                                    ) + timedelta(
                                        minutes=STATE.loss_guard_pause_min
                                    )
                                    STATE.loss_streak = 0
                                    if STATE.chat_id:
                                        pause_t = (
                                            STATE.paused_until.astimezone().strftime(
                                                "%H:%M"
                                            )
                                        )
                                        await app.bot.send_message(
                                            STATE.chat_id,
                                            f"🛑 Loss-guard: pausar nya entries till ca {pause_t}.",
                                        )
                                continue

                    # SL
                    if move_pct <= -STATE.sl_pct:
                        live_info = None
                        if STATE.trade_mode == "live" and pos.side == "LONG":
                            ok, err, info = await kucoin_place_market_order(
                                sym, "sell", pos.qty
                            )
                            if not ok:
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"⚠️ LIVE SL-sälj {sym} misslyckades.\n{err}",
                                    )
                                continue
                            live_info = info
                            if live_info and live_info.get("avg_price"):
                                price = live_info["avg_price"]

                        net = close_position(
                            sym,
                            st,
                            reason="SL",
                            live_sell_info=live_info,
                            approx_price=price,
                        )
                        if net < 0:
                            STATE.loss_streak += 1
                        else:
                            STATE.loss_streak = 0
                        if STATE.chat_id:
                            mark = "✅" if net >= 0 else "❌"
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"⛔ SL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}",
                            )
                        if (
                            STATE.loss_guard_on
                            and STATE.loss_streak >= STATE.loss_guard_n
                        ):
                            STATE.paused_until = datetime.now(
                                timezone.utc
                            ) + timedelta(minutes=STATE.loss_guard_pause_min)
                            STATE.loss_streak = 0
                            if STATE.chat_id:
                                pause_t = STATE.paused_until.astimezone().strftime(
                                    "%H:%M"
                                )
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🛑 Loss-guard: pausar nya entries till ca {pause_t}.",
                                )
                        continue

            await asyncio.sleep(3)
        except Exception as e:
            logger.exception("Engine-fel: %s", e)
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except Exception:
                    pass
            await asyncio.sleep(5)


# --------------------------------------------------
# Telegram UI
# --------------------------------------------------
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
            move_pct = (p.high_water - p.entry_price) / (
                p.entry_price or 1.0
            ) * 100.0
            pos_lines.append(
                f"{s}: {p.side} @ {p.entry_price:.4f} | qty {p.qty:.6f} | "
                f"hi {p.high_water:.4f} | lo {p.low_water:.4f} | "
                f"max_move≈{move_pct:.2f}% | regime={p.regime},src={p.reason}",
            )

    regime_line = (
        "Regime: AUTO (trend + mean reversion)"
        if STATE.regime_auto
        else "Regime: TREND only"
    )
    mr_line = (
        f"MR: {'ON' if STATE.mr_on else 'OFF'} | dev={STATE.mr_dev_pct:.2f}% | "
        f"range_atr_max={STATE.range_atr_max:.2f}%"
    )
    trend_line = f"Trend-slope min: {STATE.trend_slope_min:.2f}%"

    mode_line = (
        "Läge: LIVE SPOT (endast LONG)"
        if STATE.trade_mode == "live"
        else "Läge: MOCK (simulerad handel)"
    )

    if STATE.loss_guard_on:
        if STATE.paused_until and datetime.now(timezone.utc) < STATE.paused_until:
            rest = STATE.paused_until.astimezone().strftime("%H:%M")
            lg_line = (
                f"Loss-guard: ON | N={STATE.loss_guard_n} | pause={STATE.loss_guard_pause_min}m "
                f"(aktiv paus till ca {rest})"
            )
        else:
            lg_line = (
                f"Loss-guard: ON | N={STATE.loss_guard_n} | "
                f"pause={STATE.loss_guard_pause_min}m"
            )
    else:
        lg_line = "Loss-guard: OFF"

    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        "Strategi: Hybrid Momentum + Mean Reversion (MOCK/LIVE)",
        regime_line,
        mr_line,
        trend_line,
        lg_line,
        mode_line,
        f"Threshold (MOMO): {STATE.threshold:.2f}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Mock-size: {STATE.mock_size:.1f} USDT | Fee per sida (modell): {STATE.fee_side:.4%}",
        f"Risk: tp={STATE.tp_pct:.2f}% | sl={STATE.sl_pct:.2f}% | "
        f"trail_start={STATE.trail_start_pct:.2f}% | trail={STATE.trail_pct:.2f}% | "
        f"shorts={'ON' if STATE.allow_shorts and STATE.trade_mode=='mock' else 'OFF'} | "
        f"max_pos={STATE.max_pos}",
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    return "\n".join(lines)


# --------------------------------------------------
# Telegram Application + Commands
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
        "Justera momentum-känslighet med /threshold 0.55 (lägre = fler trades)",
        reply_markup=reply_kb(),
    )
    await tg_app.bot.send_message(STATE.chat_id, status_text())


async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(
        STATE.chat_id, status_text(), reply_markup=reply_kb()
    )


async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Engine: ON ✅ ({STATE.trade_mode.upper()})",
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
    msg = update.message.text.strip()
    parts = msg.split(" ", 1)
    if len(parts) == 2:
        tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(
                STATE.chat_id,
                f"Timeframes satta: {', '.join(STATE.tfs)}",
                reply_markup=reply_kb(),
            )
            return
    await tg_app.bot.send_message(
        STATE.chat_id, "Använd: /timeframe 1m,3m,5m", reply_markup=reply_kb()
    )


async def cmd_threshold(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Aktuellt momentum-threshold: {STATE.threshold:.2f}",
            reply_markup=reply_kb(),
        )
        return
    try:
        val = float(toks[1])
        if val <= 0:
            raise ValueError()
        STATE.threshold = val
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Momentum-threshold uppdaterad: {val:.2f}",
            reply_markup=reply_kb(),
        )
    except Exception:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Fel värde. Ex: /threshold 0.55",
            reply_markup=reply_kb(),
        )


def _yesno(s: str) -> bool:
    return s.lower() in ("1", "true", "on", "yes", "ja")


async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()

    # Avancerat: /risk set key value
    if len(toks) == 4 and toks[0] == "/risk" and toks[1] == "set":
        key, val = toks[2], toks[3]
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
                        "Shorts är inte tillåtna i LIVE-spotläge. "
                        "Byt till /mode mock om du vill testa shorts i simulering.",
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
                    (
                        "Stödjer: size, tp, sl, trail_start, trail, max_pos, "
                        "allow_shorts, mr_on, mr_dev, regime_auto, "
                        "trend_slope_min, range_atr_max, "
                        "loss_guard_on, loss_guard_n, loss_guard_pause"
                    ),
                    reply_markup=reply_kb(),
                )
                return

            await tg_app.bot.send_message(
                STATE.chat_id, "Risk/Regime uppdaterad.", reply_markup=reply_kb()
            )
        except Exception:
            await tg_app.bot.send_message(
                STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb()
            )
        return

    # Normal /risk: snabbknappar
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
        f"Aktuellt size per trade: {STATE.mock_size:.1f} USDT\n"
        f"Aktuellt TP: {STATE.tp_pct:.2f}% | SL: {STATE.sl_pct:.2f}%\n"
        f"Aktuellt max_pos: {STATE.max_pos}\n\n"
        "Välj snabb-knapp här under, eller använd:\n"
        "`/risk set <nyckel> <värde>`\n"
        "Ex: `/risk set size 20` eller `/risk set max_pos 6`"
    )

    await tg_app.bot.send_message(
        STATE.chat_id, text, reply_markup=kb
    )


async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    lines = [f"📈 PnL total (NET): {total:+.4f} USDT"]
    for s in STATE.symbols:
        lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl:+.4f} USDT")
    await tg_app.bot.send_message(
        STATE.chat_id, "\n".join(lines), reply_markup=reply_kb()
    )


async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    rows = [
        [
            "time",
            "symbol",
            "side",
            "entry",
            "exit",
            "gross",
            "fee_in",
            "fee_out",
            "net",
            "reason",
        ]
    ]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append(
                [
                    r["time"],
                    r["symbol"],
                    r["side"],
                    r["entry"],
                    r["exit"],
                    r["gross"],
                    r["fee_in"],
                    r["fee_out"],
                    r["net"],
                    r["reason"],
                ]
            )
    if len(rows) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb()
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
    )


async def cmd_mode(update: Update, _):
    """
    /mode        -> visa läge + knappar MOCK/LIVE
    /mode mock   -> tvinga MOCK
    /mode live   -> fråga med inline-knappar
    /mode live JA -> fallback aktivera LIVE
    """
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
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

    target = toks[1].lower()
    if target == "mock":
        STATE.trade_mode = "mock"
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Läge satt till MOCK (endast simulering). "
            "Shorts kan aktiveras igen via /risk set allow_shorts on.",
            reply_markup=reply_kb(),
        )
        return

    if target == "live":
        if len(toks) < 3 or toks[2].upper() != "JA":
            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ JA, slå på LIVE", callback_data="mode_live_yes"
                        ),
                        InlineKeyboardButton("❌ NEJ", callback_data="mode_live_no"),
                    ]
                ]
            )
            await tg_app.bot.send_message(
                STATE.chat_id,
                "⚠️ Är du säker på att du vill slå på LIVE-spot?\n"
                "Detta skickar riktiga market-ordrar på KuCoin.",
                reply_markup=kb,
            )
            return

        if not kucoin_creds_ok():
            await tg_app.bot.send_message(
                STATE.chat_id,
                "❌ Kan inte aktivera LIVE: saknar KUCOIN_API_KEY/SECRET/PASSPHRASE i env.",
                reply_markup=reply_kb(),
            )
            return

        STATE.trade_mode = "live"
        STATE.allow_shorts = False
        await tg_app.bot.send_message(
            STATE.chat_id,
            "✅ LIVE-läge AKTIVERAT (spot, endast LONG).\n"
            "Engine kör samma logik som i mock, men skickar riktiga ordrar.\n"
            "Testa gärna med liten size först, t.ex. /risk set size 10",
            reply_markup=reply_kb(),
        )
        return

    await tg_app.bot.send_message(
        STATE.chat_id,
        "Använd: /mode, /mode mock, /mode live, /mode live JA",
        reply_markup=reply_kb(),
    )


async def cmd_close_all(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = 0
    total_net = 0.0
    for sym in STATE.symbols:
        st = STATE.per_sym[sym]
        if not st.pos:
            continue
        pos = st.pos
        tf = STATE.tfs[0] if STATE.tfs else "3m"
        try:
            kl = await get_klines(sym, tf, limit=2)
            price = compute_features(kl)["close"]
        except Exception:
            price = pos.entry_price

        live_info = None
        if STATE.trade_mode == "live" and pos.side == "LONG":
            ok, err, info = await kucoin_place_market_order(sym, "sell", pos.qty)
            if not ok:
                await tg_app.bot.send_message(
                    STATE.chat_id,
                    f"⚠️ LIVE close-all SELL {sym} misslyckades, hoppar över.\n{err}",
                )
                continue
            live_info = info
            if live_info and live_info.get("avg_price"):
                price = live_info["avg_price"]

        net = close_position(
            sym,
            st,
            reason="MANUAL_CLOSE",
            live_sell_info=live_info,
            approx_price=price,
        )
        closed += 1
        total_net += net

    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Close all klart. Antal positioner stängda: {closed}, "
        f"Netto PnL≈{total_net:+.4f} USDT",
        reply_markup=reply_kb(),
    )


async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        st.realized_pnl = 0.0
        st.trades_log.clear()
    await tg_app.bot.send_message(
        STATE.chat_id,
        "PnL återställd i minnet (loggfilerna mock_trade_log.csv/real_trade_log.csv påverkas inte).",
        reply_markup=reply_kb(),
    )


async def cmd_test_buy(update: Update, _):
    """
    /test_buy SYMBOL [USDT]

    Exempel:
      /test_buy BTC-USDT
      /test_buy ETH-USDT 15

    - Öppnar en position i valt coin (MOCK eller LIVE).
    - I LIVE används riktiga KuCoin-ordrar.
    - Positionen hanteras av engine (TP/SL/trail eller /close_all) och ingår i PnL.
    """
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()

    if len(toks) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Använd: /test_buy SYMBOL [USDT]\n"
            "Ex: /test_buy BTC-USDT 5\n"
            "Symbol måste finnas i botens symbol-lista.",
            reply_markup=reply_kb(),
        )
        return

    symbol = toks[1].upper()
    if symbol not in STATE.symbols:
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Symbol {symbol} finns inte i botens lista.\n"
            f"Aktuella symboler: {', '.join(STATE.symbols)}",
            reply_markup=reply_kb(),
        )
        return

    size = STATE.mock_size
    if len(toks) >= 3:
        try:
            size = float(toks[2])
        except Exception:
            pass

    mode_txt = "LIVE" if STATE.trade_mode == "live" else "MOCK"

    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"✅ JA, köp {symbol} {size:.2f} USDT ({mode_txt})",
                    callback_data=f"testbuy_yes|{symbol}|{size:.4f}",
                ),
                InlineKeyboardButton("❌ NEJ", callback_data="testbuy_no"),
            ]
        ]
    )

    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Vill du göra ett {mode_txt} test-köp i {symbol} för {size:.2f} USDT?",
        reply_markup=kb,
    )


# --------------------------------------------------
# Callback-knappar
# --------------------------------------------------
async def on_button(update: Update, _):
    query = update.callback_query
    data = (query.data or "").strip()
    chat_id = query.message.chat_id
    STATE.chat_id = chat_id

    # Risk: size
    if data.startswith("risk_size_"):
        await query.answer()
        try:
            size = float(data.split("_")[-1])
        except Exception:
            size = STATE.mock_size
        STATE.mock_size = size
        await query.edit_message_text(
            f"Size per trade uppdaterad till {size:.1f} USDT."
        )
        return

    # Risk: TP
    if data.startswith("risk_tp_"):
        await query.answer()
        try:
            tp_val = float(data.split("_")[-1])
            STATE.tp_pct = tp_val
            await query.edit_message_text(f"TP uppdaterad till {tp_val:.2f}%.")
        except Exception:
            await query.edit_message_text("Kunde inte tolka TP-värde.")
        return

    # Risk: SL
    if data.startswith("risk_sl_"):
        await query.answer()
        try:
            sl_val = float(data.split("_")[-1])
            STATE.sl_pct = sl_val
            await query.edit_message_text(f"SL uppdaterad till {sl_val:.2f}%.")
        except Exception:
            await query.edit_message_text("Kunde inte tolka SL-värde.")
        return

    # Risk: max_pos
    if data.startswith("risk_maxpos_"):
        await query.answer()
        try:
            val = int(data.split("_")[-1])
            if val <= 0:
                raise ValueError()
            STATE.max_pos = val
            await query.edit_message_text(f"max_pos uppdaterad till {val}.")
        except Exception:
            await query.edit_message_text("Kunde inte tolka max_pos-värde.")
        return

    # Mode-val
    if data == "mode_choose_mock":
        await query.answer()
        STATE.trade_mode = "mock"
        await query.edit_message_text(
            "Läge satt till MOCK (endast simulering). "
            "Shorts kan aktiveras igen via /risk set allow_shorts on."
        )
        return

    if data == "mode_choose_live":
        await query.answer()
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "✅ JA, slå på LIVE", callback_data="mode_live_yes"
                    ),
                    InlineKeyboardButton("❌ NEJ", callback_data="mode_live_no"),
                ]
            ]
        )
        await query.edit_message_text(
            "⚠️ Är du säker på att du vill slå på LIVE-spot?\n"
            "Detta skickar riktiga market-ordrar på KuCoin.",
            reply_markup=kb,
        )
        return

    if data == "mode_live_no":
        await query.answer("Avbrutet.")
        await query.edit_message_text("LIVE-aktivering avbruten.")
        return

    if data == "mode_live_yes":
        await query.answer()
        if not kucoin_creds_ok():
            await query.edit_message_text(
                "❌ Kan inte aktivera LIVE: KUCOIN_API_KEY/SECRET/PASSPHRASE saknas."
            )
            return
        STATE.trade_mode = "live"
        STATE.allow_shorts = False
        await query.edit_message_text(
            "✅ LIVE-läge AKTIVERAT (spot, endast LONG).\n"
            "Engine kör samma logik som i mock, men skickar riktiga ordrar.\n"
            "Testa gärna /test_buy för att se att allt fungerar."
        )
        return

    # Test-buy
    if data == "testbuy_no":
        await query.answer("Avbrutet.")
        await query.edit_message_text("Test-köp avbrutet.")
        return

    if data.startswith("testbuy_yes|"):
        await query.answer()
        parts = data.split("|")
        if len(parts) != 3:
            await query.edit_message_text("Felaktig test_buy-data.")
            return

        symbol = parts[1].upper()
        try:
            size = float(parts[2])
        except Exception:
            size = STATE.mock_size

        if symbol not in STATE.symbols:
            await query.edit_message_text(
                f"Symbol {symbol} finns inte i botens lista.\n"
                f"Aktuella symboler: {', '.join(STATE.symbols)}"
            )
            return

        # Hämta senaste pris (för MOCK eller fallback)
        tf = STATE.tfs[0] if STATE.tfs else "3m"
        try:
            kl = await get_klines(symbol, tf, limit=5)
            feats = compute_features(kl)
            price = feats["close"]
        except Exception:
            price = 0.0

        live_info = None
        if STATE.trade_mode == "live":
            if not kucoin_creds_ok():
                await query.edit_message_text(
                    "❌ KuCoin API-uppgifter saknas, kan inte göra LIVE test-köp."
                )
                return

            ok, err, info = await kucoin_place_market_order(symbol, "buy", size)
            if not ok:
                await query.edit_message_text(
                    f"❌ Testköp misslyckades för {symbol}.\nKuCoin fel: {err}"
                )
                return

            live_info = info
            if live_info and live_info.get("avg_price"):
                price = live_info["avg_price"]

        # Öppna position i boten (MOCK eller LIVE)
        st = STATE.per_sym[symbol]
        open_position(
            symbol,
            "LONG",
            price,
            st,
            regime="trend",
            reason="TESTBUY",
            live_info=live_info,
        )

        mode_txt = "LIVE" if STATE.trade_mode == "live" else "MOCK"
        await query.edit_message_text(
            f"✅ Test-köp öppnat i {symbol} @ {price:.4f} ({mode_txt}).\n"
            "Positionen hanteras nu av engine (TP/SL/trail eller /close_all) "
            "och ingår i PnL.",
        )
        return

    # Fallback
    await query.answer()
    await query.edit_message_text("Ok.")


# Registrera handlers
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
# FastAPI / DigitalOcean
# --------------------------------------------------
app = FastAPI()


class TgUpdate(BaseModel):
    update_id: Optional[int] = None


@app.on_event("startup")
async def on_startup():
    logger.info("Startup: init Telegram app, webhook + engine")
    await tg_app.initialize()
    if WEBHOOK_BASE:
        url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
        await tg_app.bot.set_webhook(url)
        logger.info("Webhook satt till %s", url)
    await tg_app.start()
    asyncio.create_task(engine_loop(tg_app))


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutdown: stop Telegram app")
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
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
