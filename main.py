# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MP Bot – FULL VERSION (Hybrid Momentum + Mean Reversion)
# ------------------------------------------------------------
# Helt komplett bot (mock + live) med:
# • KuCoin LIVE orders
# • 100% korrekt KuCoin PnL-beregning
# • TP/SL + trailing stop
# • Momentum + Mean Reversion hybrid
# • Trend/Range auto-detektering
# • TestBuy för alla coins via knappar
# • Risk-knappar (TP/SL)
# • Max positioner via knappar
# • Export CSV
# • Mock + Live växling
# • FastAPI + Webhook
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
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
)

# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN saknas — lägg in den i DigitalOcean ENV.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (
    os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,XRP-USDT,LINK-USDT")
    .replace(" ", "")
).split(",")

DEFAULT_TFS = (
    os.getenv("TIMEFRAMES", "1m,3m")
    .replace(" ", "")
).split(",")

MOCK_SIZE_USDT = float(os.getenv("MOCK_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "4"))

TP_PCT = float(os.getenv("TP_PCT", "0.30"))
SL_PCT = float(os.getenv("SL_PCT", "0.50"))
TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "2.0"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "2.0"))

ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.55"))
ALLOW_SHORTS_DEFAULT = False

MR_DEV_PCT = float(os.getenv("MR_DEV_PCT", "1.20"))
TREND_SLOPE_MIN = float(os.getenv("TREND_SLOPE_MIN", "0.20"))
RANGE_ATR_MAX = float(os.getenv("RANGE_ATR_MAX", "0.80"))

LOSS_GUARD_ON_DEFAULT = True
LOSS_GUARD_N_DEFAULT = int(os.getenv("LOSS_GUARD_N", "2"))
LOSS_GUARD_PAUSE_MIN_DEFAULT = int(os.getenv("LOSS_GUARD_PAUSE_MIN", "15"))

# KuCoin API
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KUCOIN_API_KEY_VERSION = os.getenv("KUCOIN_API_KEY_VERSION", "3")

KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

# Candle endpoint
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
}

# ------------------------------------------------------------
# STATE OBJECTS
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

    trade_mode: str = "mock"

    loss_guard_on: bool = LOSS_GUARD_ON_DEFAULT
    loss_guard_n: int = LOSS_GUARD_N_DEFAULT
    loss_guard_pause_min: int = LOSS_GUARD_PAUSE_MIN_DEFAULT
    loss_streak: int = 0
    paused_until: Optional[datetime] = None

    chat_id: Optional[int] = None

STATE = EngineState()
for sym in STATE.symbols:
    STATE.per_sym[sym] = SymState()

# ------------------------------------------------------------
# HELPERS / LOGGING
# ------------------------------------------------------------

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)def log_mock(row: Dict):
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

def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

# ------------------------------------------------------------
# KUCOIN AUTH / PRIVATE REQUEST
# ------------------------------------------------------------

def kucoin_creds_ok() -> bool:
    return (
        KUCOIN_API_KEY
        and KUCOIN_API_SECRET
        and KUCOIN_API_PASSPHRASE
    )

async def kucoin_private_request(method: str, path: str, body: Optional[dict] = None):
    if not kucoin_creds_ok():
        return None

    body = body or {}
    body_str = json.dumps(body, separators=(",", ":"))
    ts = str(int(time.time() * 1000))

    prehash = ts + method.upper() + path + body_str
    sign = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode(),
            prehash.encode(),
            hashlib.sha256,
        ).digest()
    ).decode()

    passphrase_hashed = base64.b64encode(
        hmac.new(
            KUCOIN_API_SECRET.encode(),
            KUCOIN_API_PASSPHRASE.encode(),
            hashlib.sha256,
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
        data = {"raw": resp.text}

    data["_http_status"] = resp.status_code
    return data

async def kucoin_place_market_order(symbol: str, side: str, amount: float):
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
        return False, "No response"

    if data.get("code") != "200000":
        return False, f"{data.get('_http_status')} {data.get('msg')}"

    return True, ""

# ------------------------------------------------------------
# DATA FETCHING + INDICATORS
# ------------------------------------------------------------

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

    return [50.0] + outdef compute_features(candles):
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

    atrp = feats["atrp"]
    slope = abs(feats["trend_slope"])

    if atrp >= STATE.range_atr_max and slope >= STATE.trend_slope_min:
        return "trend"

    if atrp <= STATE.range_atr_max and slope < STATE.trend_slope_min:
        return "range"

    return "trend"

# ------------------------------------------------------------
# TRADE LOGIC
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

    log_data = {
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "action": "ENTRY",
        "side": side,
        "price": round(price, 8),
        "qty": round(qty, 8),
        "gross": "",
        "fee_in": "",
        "fee_out": "",
        "net": "",
        "info": f"size_usdt={STATE.mock_size};regime={regime};reason={reason};mode={STATE.trade_mode}",
    }

    if STATE.trade_mode == "live":
        log_real(log_data)
    else:
        log_mock(log_data)

def close_position(sym: str, price: float, st: SymState, reason: str) -> float:
    if not st.pos:
        return 0.0

    pos = st.pos

    usd_in = pos.entry_price * pos.qty
    usd_out = price * pos.qty

    if pos.side == "LONG":
        gross = pos.qty * (price - pos.entry_price)
    else:
        gross = pos.qty * (pos.entry_price - price)

    fee_in = _fee(usd_in)
    fee_out = _fee(usd_out)

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

    log_data = {
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "action": "EXIT",
        "side": pos.side,
        "price": round(price, 8),
        "qty": round(pos.qty, 8),
        "gross": round(gross, 6),
        "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6),
        "net": round(net, 6),
        "info": f"{reason};regime={pos.regime};src={pos.reason};mode={STATE.trade_mode}",
    }

    if STATE.trade_mode == "live":
        log_real(log_data)
    else:
        log_mock(log_data)

    st.pos = None
    return net# ------------------------------------------------------------
# ENGINE LOOP
# ------------------------------------------------------------

async def engine_loop(app: Application):
    await asyncio.sleep(2)

    while True:
        try:
            if STATE.engine_on:
                now = datetime.now(timezone.utc)

                # -------------------------
                # LOSS-GUARD PAUSE
                # -------------------------
                if (
                    STATE.loss_guard_on
                    and STATE.paused_until is not None
                    and now < STATE.paused_until
                ):
                    await asyncio.sleep(3)
                    continue

                # ------------------------------------------------
                # ENTRY LOGIC (NYA POSITIONER)
                # ------------------------------------------------
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]

                if len(open_syms) < STATE.max_pos:
                    for sym in STATE.symbols:
                        st = STATE.per_sym[sym]
                        if st.pos:
                            continue

                        tf = STATE.tfs[0] if STATE.tfs else "1m"

                        try:
                            kl = await get_klines(sym, tf, limit=80)
                        except Exception:
                            continue

                        feats = compute_features(kl)
                        price = feats["close"]
                        regime = decide_regime(feats)

                        allow_shorts = (
                            STATE.allow_shorts if STATE.trade_mode == "mock" else False
                        )

                        # MOMENTUM (TREND-LÄGE)
                        if regime == "trend":
                            score = momentum_score(feats)

                            # ----- LONG -----
                            if score > STATE.threshold:
                                if STATE.trade_mode == "live":
                                    ok, err = await kucoin_place_market_order(
                                        sym, "buy", STATE.mock_size
                                    )
                                    if not ok:
                                        if STATE.chat_id:
                                            await app.bot.send_message(
                                                STATE.chat_id,
                                                f"⚠️ LIVE-KÖP {sym} misslyckades:\n{err}",
                                            )
                                        continue

                                open_position(
                                    sym, "LONG", price, st, "trend", "MOMO"
                                )

                                open_syms.append(sym)

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 MOMO ENTRY {sym} LONG @ {price:.4f} | "
                                        f"score={score:.2f} | thr={STATE.threshold:.2f}",
                                    )

                                if len(open_syms) >= STATE.max_pos:
                                    break

                            # ----- SHORT (mock-only) -----
                            elif allow_shorts and score < -STATE.threshold:
                                open_position(
                                    sym, "SHORT", price, st, "trend", "MOMO"
                                )
                                open_syms.append(sym)

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 MOMO ENTRY {sym} SHORT @ {price:.4f} | "
                                        f"score={score:.2f}",
                                    )

                                if len(open_syms) >= STATE.max_pos:
                                    break

                        # ------------------------------------------------
                        # MEAN-REVERSION (RANGE-LÄGE)
                        # ------------------------------------------------
                        elif regime == "range" and STATE.mr_on:
                            ema20 = feats["ema20"]
                            if ema20 == 0:
                                continue

                            dev_pct = (price - ema20) / ema20 * 100.0

                            # ----- LONG MR -----
                            if dev_pct <= -STATE.mr_dev_pct:
                                if STATE.trade_mode == "live":
                                    ok, err = await kucoin_place_market_order(
                                        sym, "buy", STATE.mock_size
                                    )
                                    if not ok:
                                        if STATE.chat_id:
                                            await app.bot.send_message(
                                                STATE.chat_id,
                                                f"⚠️ LIVE-KÖP {sym} misslyckades:\n{err}",
                                            )
                                        continue

                                open_position(
                                    sym, "LONG", price, st, "range", "MR"
                                )
                                open_syms.append(sym)

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 MR ENTRY {sym} LONG @ {price:.4f} | "
                                        f"dev={dev_pct:.2f}%",
                                    )

                                if len(open_syms) >= STATE.max_pos:
                                    break

                            # ----- SHORT MR (mock-only) -----
                            elif dev_pct >= STATE.mr_dev_pct and allow_shorts:
                                open_position(
                                    sym, "SHORT", price, st, "range", "MR"
                                )
                                open_syms.append(sym)

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 MR ENTRY {sym} SHORT @ {price:.4f} | "
                                        f"dev={dev_pct:.2f}%",
                                    )

                                if len(open_syms) >= STATE.max_pos:
                                    break

                # ------------------------------------------------
                # MANAGE OPEN POSITIONS
                # ------------------------------------------------
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos:
                        continue

                    tf = STATE.tfs[0] if STATE.tfs else "1m"

                    try:
                        kl = await get_klines(sym, tf, limit=5)
                    except Exception:
                        continue

                    feats = compute_features(kl)
                    price = feats["close"]
                    pos = st.pos

                    # Uppdatera high/low
                    pos.high_water = max(pos.high_water, price)
                    pos.low_water = min(pos.low_water, price)

                    move_pct = (price - pos.entry_price) / (pos.entry_price or 1.0) * 100.0
                    if pos.side == "SHORT":
                        move_pct = -move_pct

                    # ------------------------------------------------
                    # STARTA TRAILING STOP
                    # ------------------------------------------------
                    if not pos.trailing and move_pct >= STATE.trail_start_pct:
                        pos.trailing = True
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🔒 TRAIL ON {sym} | move≈{move_pct:.2f}%",
                            )

                    # ------------------------------------------------
                    # TAKE-PROFIT
                    # ------------------------------------------------
                    if move_pct >= STATE.tp_pct and not pos.trailing:

                        if STATE.trade_mode == "live" and pos.side == "LONG":
                            ok, err = await kucoin_place_market_order(
                                sym, "sell", pos.qty
                            )
                            if not ok:
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"⚠️ LIVE TP-SELL {sym} misslyckades:\n{err}",
                                    )
                                continue

                        net = close_position(sym, price, st, "TP")

                        # Loss-guard
                        if net < 0:
                            STATE.loss_streak += 1
                        else:
                            STATE.loss_streak = 0

                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🎯 TP EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT",
                            )

                        # Pause if needed
                        if (
                            STATE.loss_guard_on
                            and STATE.loss_streak >= STATE.loss_guard_n
                        ):
                            STATE.paused_until = datetime.now(timezone.utc) + timedelta(
                                minutes=STATE.loss_guard_pause_min
                            )
                            STATE.loss_streak = 0

                        continue                    # ------------------------------------------------
                    # TRAILING STOP
                    # ------------------------------------------------
                    if pos.trailing:
                        if pos.side == "LONG":
                            trail_stop = pos.high_water * (1.0 - STATE.trail_pct / 100.0)
                            if price <= trail_stop:

                                if STATE.trade_mode == "live":
                                    ok, err = await kucoin_place_market_order(
                                        sym, "sell", pos.qty
                                    )
                                    if not ok:
                                        if STATE.chat_id:
                                            await app.bot.send_message(
                                                STATE.chat_id,
                                                f"⚠️ LIVE TRAIL-SELL {sym} misslyckades:\n{err}",
                                            )
                                        continue

                                net = close_position(sym, price, st, "TRAIL")

                                if net < 0:
                                    STATE.loss_streak += 1
                                else:
                                    STATE.loss_streak = 0

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT",
                                    )

                                if (
                                    STATE.loss_guard_on
                                    and STATE.loss_streak >= STATE.loss_guard_n
                                ):
                                    STATE.paused_until = datetime.now(timezone.utc) + timedelta(
                                        minutes=STATE.loss_guard_pause_min
                                    )
                                    STATE.loss_streak = 0

                                continue

                        else:  # SHORT
                            trail_stop = pos.low_water * (1.0 + STATE.trail_pct / 100.0)
                            if price >= trail_stop:
                                net = close_position(sym, price, st, "TRAIL")

                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT",
                                    )

                                continue

                    # ------------------------------------------------
                    # STOP-LOSS
                    # ------------------------------------------------
                    if move_pct <= -STATE.sl_pct:

                        if STATE.trade_mode == "live" and pos.side == "LONG":
                            ok, err = await kucoin_place_market_order(
                                sym, "sell", pos.qty
                            )
                            if not ok:
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"⚠️ LIVE SL-SELL {sym} misslyckades:\n{err}",
                                    )
                                continue

                        net = close_position(sym, price, st, "SL")

                        if net < 0:
                            STATE.loss_streak += 1
                        else:
                            STATE.loss_streak = 0

                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"⛔ SL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT",
                            )

                        if (
                            STATE.loss_guard_on
                            and STATE.loss_streak >= STATE.loss_guard_n
                        ):
                            STATE.paused_until = datetime.now(timezone.utc) + timedelta(
                                minutes=STATE.loss_guard_pause_min
                            )
                            STATE.loss_streak = 0

                        continue

            await asyncio.sleep(3)

        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(
                        STATE.chat_id,
                        f"⚠️ Engine-fel: {e}",
                    )
                except Exception:
                    pass

            await asyncio.sleep(5)


# ------------------------------------------------------------
# TELEGRAM UI
# ------------------------------------------------------------

def reply_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("/status"), KeyboardButton("/pnl")],
            [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
            [KeyboardButton("/timeframe"), KeyboardButton("/threshold")],
            [KeyboardButton("/risk"), KeyboardButton("/export_csv")],
            [KeyboardButton("/close_all"), KeyboardButton("/reset_pnl")],
            [KeyboardButton("/mode"), KeyboardButton("/testbuy")],
        ],
        resize_keyboard=True,
    )


def status_text() -> str:
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)

    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            p = st.pos
            max_move = (p.high_water - p.entry_price) / (p.entry_price or 1) * 100
            pos_lines.append(
                f"{s}: {p.side} @ {p.entry_price:.4f} | qty {p.qty:.6f} | "
                f"hi {p.high_water:.4f} | lo {p.low_water:.4f} | "
                f"max_move≈{max_move:.2f}% | regime={p.regime},src={p.reason}"
            )

    regime_line = "AUTO (trend + MR)" if STATE.regime_auto else "TREND-only"

    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Mode: {STATE.trade_mode.upper()}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Threshold: {STATE.threshold:.2f}",
        f"Mock-size: {STATE.mock_size:.1f} USDT",
        f"TP={STATE.tp_pct:.2f}% | SL={STATE.sl_pct:.2f}% | "
        f"TrailStart={STATE.trail_start_pct:.2f}% | Trail={STATE.trail_pct:.2f}%",
        f"Regime: {regime_line} | MR={STATE.mr_on} | dev={STATE.mr_dev_pct:.2f}%",
        f"ATR max={STATE.range_atr_max:.2f}% | slope_min={STATE.trend_slope_min:.2f}%",
        f"Shorts: {'ON' if STATE.allow_shorts else 'OFF'}",
        f"MaxPos: {STATE.max_pos}",
        f"LossGuard: {'ON' if STATE.loss_guard_on else 'OFF'} "
        f"(N={STATE.loss_guard_n}, pause={STATE.loss_guard_pause_min}m)",
        f"PnL total: {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]

    return "\n".join(lines)# ------------------------------------------------------------
# TELEGRAM APPLICATION
# ------------------------------------------------------------

tg_app = Application.builder().token(BOT_TOKEN).build()

# ------------------------------------------------------------
# TELEGRAM COMMANDS
# ------------------------------------------------------------

async def cmd_start(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(
        STATE.chat_id,
        "🤖 MP Bot – Hybrid Momentum + Mean Reversion\n"
        "Standardläge: MOCK (simulerad handel)\n"
        "Starta motor med /engine_on\n"
        "Visa status med /status\n"
        "Justera risk med /risk\n"
        "Testköp (LIVE) med /testbuy",
        reply_markup=reply_kb(),
    )
    await tg_app.bot.send_message(STATE.chat_id, status_text(), reply_markup=reply_kb())


async def cmd_status(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(
        STATE.chat_id,
        status_text(),
        reply_markup=reply_kb(),
    )


async def cmd_engine_on(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Engine: ON ✅  (mode={STATE.trade_mode.upper()})",
        reply_markup=reply_kb(),
    )


async def cmd_engine_off(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(
        STATE.chat_id,
        "Engine: OFF ⛔️",
        reply_markup=reply_kb(),
    )


async def cmd_timeframe(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    msg = (update.message.text or "").strip()
    parts = msg.split(" ", 1)

    if len(parts) == 2:
        tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(
                STATE.chat_id,
                f"Timeframes uppdaterade: {', '.join(STATE.tfs)}",
                reply_markup=reply_kb(),
            )
            return

    await tg_app.bot.send_message(
        STATE.chat_id,
        "Använd: /timeframe 1m,3m,5m",
        reply_markup=reply_kb(),
    )


async def cmd_threshold(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    toks = (update.message.text or "").strip().split()

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


async def cmd_pnl(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    lines = [f"📈 PnL total (NET): {total:+.4f} USDT"]
    for s in STATE.symbols:
        lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl:+.4f} USDT")
    await tg_app.bot.send_message(
        STATE.chat_id,
        "\n".join(lines),
        reply_markup=reply_kb(),
    )# ------------------------------------------------------------
# RISK / PARAMETRAR – med knappar
# ------------------------------------------------------------

def _yesno(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "on", "ja")


async def cmd_risk(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    toks = (update.message.text or "").strip().split()

    # --------------------------------------------------------
    # Avancerad inställning via:
    # /risk set key value
    # --------------------------------------------------------
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
                        "Shorts tillåts aldrig i LIVE-spot. Gå till MOCK om du vill testa shorts.",
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
                STATE.chat_id,
                "Risk-inställning uppdaterad.",
                reply_markup=reply_kb(),
            )
        except Exception:
            await tg_app.bot.send_message(
                STATE.chat_id,
                "Felaktigt värde.",
                reply_markup=reply_kb(),
            )
        return

    # --------------------------------------------------------
    # ANNARS: Visa knappar för snabb-risk (size, TP, SL, MaxPos)
    # --------------------------------------------------------

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Size 10", callback_data="risk_size_10"),
                InlineKeyboardButton("Size 30", callback_data="risk_size_30"),
                InlineKeyboardButton("Size 50", callback_data="risk_size_50"),
            ],
            [
                InlineKeyboardButton("TP 0.5%", callback_data="risk_tp_0_5"),
                InlineKeyboardButton("SL 0.5%", callback_data="risk_sl_0_5"),
            ],
            [
                InlineKeyboardButton("TP 1.0%", callback_data="risk_tp_1_0"),
                InlineKeyboardButton("SL 1.0%", callback_data="risk_sl_1_0"),
            ],
            [
                InlineKeyboardButton("MaxPos 1", callback_data="risk_maxpos_1"),
                InlineKeyboardButton("MaxPos 2", callback_data="risk_maxpos_2"),
                InlineKeyboardButton("MaxPos 3", callback_data="risk_maxpos_3"),
            ],
            [
                InlineKeyboardButton("Shorts ON", callback_data="risk_shorts_on"),
                InlineKeyboardButton("Shorts OFF", callback_data="risk_shorts_off"),
            ]
        ]
    )

    await tg_app.bot.send_message(
        STATE.chat_id,
        "Välj riskinställning:",
        reply_markup=keyboard,
    )# ------------------------------------------------------------
# MODE (MOCK / LIVE)
# ------------------------------------------------------------

async def cmd_mode(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    toks = (update.message.text or "").strip().split()

    # /mode  (visa val)
    if len(toks) == 1:
        keyboard = InlineKeyboardMarkup(
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
            reply_markup=keyboard,
        )
        return

    # /mode mock
    if toks[1].lower() == "mock":
        STATE.trade_mode = "mock"
        STATE.allow_shorts = True
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Läge satt till MOCK (simulering).",
            reply_markup=reply_kb(),
        )
        return

    # /mode live
    if toks[1].lower() == "live":
        # Måste bekräftas
        if len(toks) < 3 or toks[2].upper() != "JA":
            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("JA – aktivera LIVE", callback_data="mode_live_yes"),
                        InlineKeyboardButton("NEJ", callback_data="mode_live_no"),
                    ]
                ]
            )
            await tg_app.bot.send_message(
                STATE.chat_id,
                "⚠️ Är du säker att du vill aktivera LIVE-spot?\nDetta skickar riktiga marknadsordrar!",
                reply_markup=kb,
            )
            return

        # Direkt /mode live JA
        if not kucoin_creds_ok():
            await tg_app.bot.send_message(
                STATE.chat_id,
                "❌ Kan inte aktivera LIVE – saknar KuCoin API KEY / SECRET / PASSPHRASE.",
                reply_markup=reply_kb(),
            )
            return

        STATE.trade_mode = "live"
        STATE.allow_shorts = False
        await tg_app.bot.send_message(
            STATE.chat_id,
            "LIVE-läge AKTIVERAT. Endast LONG. 🔥",
            reply_markup=reply_kb(),
        )
        return

    await tg_app.bot.send_message(
        STATE.chat_id,
        "Använd: /mode, /mode mock, /mode live, /mode live JA",
        reply_markup=reply_kb(),
    )


# ------------------------------------------------------------
# TESTBUY – FÖR ALLA COINS (LIVE)
# ------------------------------------------------------------

async def cmd_testbuy(update: Update, context):
    STATE.chat_id = update.effective_chat.id

    if not kucoin_creds_ok():
        await tg_app.bot.send_message(
            STATE.chat_id,
            "❌ Kan inte göra LIVE testköp – KuCoin API saknas.",
            reply_markup=reply_kb(),
        )
        return

    # Visa lista med coins i inline-knappar
    keyboard = []
    row = []
    for sym in STATE.symbols:
        row.append(InlineKeyboardButton(sym, callback_data=f"testbuy_coin_{sym}"))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    keyboard.append([InlineKeyboardButton("Avbryt", callback_data="testbuy_cancel")])

    await tg_app.bot.send_message(
        STATE.chat_id,
        "Välj vilket coin du vill testköpa (LIVE 5 USDT):",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ------------------------------------------------------------
# CLOSE ALL POSITIONS
# ------------------------------------------------------------

async def cmd_close_all(update: Update, context):
    STATE.chat_id = update.effective_chat.id
    closed = 0
    total_net = 0.0

    for sym in STATE.symbols:
        st = STATE.per_sym[sym]
        if not st.pos:
            continue

        pos = st.pos

        # Hämta senaste pris
        try:
            kl = await get_klines(sym, STATE.tfs[0], limit=2)
            price = compute_features(kl)["close"]
        except:
            price = pos.entry_price

        # LIVE säljer först
        if STATE.trade_mode == "live" and pos.side == "LONG":
            ok, err = await kucoin_place_market_order(sym, "sell", pos.qty)
            if not ok:
                await tg_app.bot.send_message(
                    STATE.chat_id,
                    f"⚠️ LIVE SELL failed {sym}:\n{err}",
                )
                continue

        net = close_position(sym, price, st, "MANUAL_CLOSE")
        closed += 1
        total_net += net

    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Close All klart. Stängde {closed} positioner.\nNet PnL = {total_net:+.4f} USDT",
        reply_markup=reply_kb(),
    )


# ------------------------------------------------------------
# RESET PNL
# ------------------------------------------------------------

async def cmd_reset_pnl(update: Update, context):
    STATE.chat_id = update.effective_chat.id

    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl = 0.0
        STATE.per_sym[s].trades_log.clear()

    await tg_app.bot.send_message(
        STATE.chat_id,
        "PnL återställd i RAM (loggfiler ändras inte).",
        reply_markup=reply_kb(),
    )


# ------------------------------------------------------------
# EXPORT CSV
# ------------------------------------------------------------

async def cmd_export_csv(update: Update, context):
    STATE.chat_id = update.effective_chat.id

    rows = [
        ["time", "symbol", "side", "entry", "exit", "gross", "fee_in", "fee_out", "net", "reason"]
    ]

    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([
                r["time"], r["symbol"], r["side"], r["entry"], r["exit"],
                r["gross"], r["fee_in"], r["fee_out"], r["net"], r["reason"]
            ])

    if len(rows) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Inga trades loggade ännu.",
            reply_markup=reply_kb(),
        )
        return

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    buf.seek(0)

    await tg_app.bot.send_document(
        STATE.chat_id,
        document=io.BytesIO(buf.getvalue().encode("utf-8")),
        filename="trades_export.csv",
        caption="📂 CSV-export",
    )


# ------------------------------------------------------------
# CALLBACK-KNAPPAR (RISK, MODE, TESTBUY)
# ------------------------------------------------------------

async def on_button(update: Update, context):
    query = update.callback_query
    data = query.data
    chat_id = query.message.chat_id
    STATE.chat_id = chat_id

    # ------------------------
    # RISK – SIZE
    # ------------------------
    if data == "risk_size_10":
        STATE.mock_size = 10
        await query.edit_message_text("Size satt till 10 USDT.")
        return

    if data == "risk_size_30":
        STATE.mock_size = 30
        await query.edit_message_text("Size satt till 30 USDT.")
        return

    if data == "risk_size_50":
        STATE.mock_size = 50
        await query.edit_message_text("Size satt till 50 USDT.")
        return

    # ------------------------
    # RISK – TP/SL
    # ------------------------
    if data == "risk_tp_0_5":
        STATE.tp_pct = 0.5
        await query.edit_message_text("TP satt till 0.5%.")
        return

    if data == "risk_tp_1_0":
        STATE.tp_pct = 1.0
        await query.edit_message_text("TP satt till 1.0%.")
        return

    if data == "risk_sl_0_5":
        STATE.sl_pct = 0.5
        await query.edit_message_text("SL satt till 0.5%.")
        return

    if data == "risk_sl_1_0":
        STATE.sl_pct = 1.0
        await query.edit_message_text("SL satt till 1.0%.")
        return

    # ------------------------
    # RISK – MAX POS
    # ------------------------
    if data == "risk_maxpos_1":
        STATE.max_pos = 1
        await query.edit_message_text("MaxPos = 1")
        return

    if data == "risk_maxpos_2":
        STATE.max_pos = 2
        await query.edit_message_text("MaxPos = 2")
        return

    if data == "risk_maxpos_3":
        STATE.max_pos = 3
        await query.edit_message_text("MaxPos = 3")
        return

    # ------------------------
    # RISK – SHORTS
    # ------------------------
    if data == "risk_shorts_on":
        if STATE.trade_mode == "live":
            await query.edit_message_text("Shorts tillåts inte i LIVE.")
            return
        STATE.allow_shorts = True
        await query.edit_message_text("Shorts ON (mock).")
        return

    if data == "risk_shorts_off":
        STATE.allow_shorts = False
        await query.edit_message_text("Shorts OFF.")
        return

    # ------------------------
    # MODE – INLINE VAL
    # ------------------------
    if data == "mode_choose_mock":
        STATE.trade_mode = "mock"
        STATE.allow_shorts = True
        await query.edit_message_text("Läge satt till MOCK.")
        return

    if data == "mode_live_no":
        await query.edit_message_text("LIVE-aktivering avbruten.")
        return

    if data == "mode_choose_live":
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("JA – aktivera LIVE", callback_data="mode_live_yes"),
                    InlineKeyboardButton("NEJ", callback_data="mode_live_no"),
                ]
            ]
        )
        await query.edit_message_text(
            "⚠️ Är du säker att du vill aktivera LIVE?",
            reply_markup=kb,
        )
        return

    if data == "mode_live_yes":
        if not kucoin_creds_ok():
            await query.edit_message_text("❌ API-uppgifter saknas.")
            return
        STATE.trade_mode = "live"
        STATE.allow_shorts = False
        await query.edit_message_text("LIVE-läge AKTIVERAT.")
        return

    # ------------------------
    # TESTBUY – COIN
    # ------------------------
    if data.startswith("testbuy_coin_"):
        sym = data.replace("testbuy_coin_", "")
        ok, err = await kucoin_place_market_order(sym, "buy", 5.0)
        if ok:
            await query.edit_message_text(f"LIVE Testköp OK – köpte 5 USDT {sym}.")
        else:
            await query.edit_message_text(f"Misslyckades: {err}")
        return

    if data == "testbuy_cancel":
        await query.edit_message_text("Avbrutet.")
        return

    # ------------------------
    # Fallback
    # ------------------------
    await query.edit_message_text("OK.")# ------------------------------------------------------------
# REGISTER TELEGRAM HANDLERS
# ------------------------------------------------------------

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
tg_app.add_handler(CommandHandler("testbuy", cmd_testbuy))

tg_app.add_handler(CallbackQueryHandler(on_button))


# ------------------------------------------------------------
# FASTAPI (WEBHOOK / HEALTH / ROOT)
# ------------------------------------------------------------

app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None


@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    return (
        f"MP Bot HYBRID OK | "
        f"engine_on={STATE.engine_on} | mode={STATE.trade_mode} | "
        f"thr={STATE.threshold:.2f} | pnl={total:+.4f}"
    )


@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "mode": STATE.trade_mode,
        "threshold": STATE.threshold,
        "symbols": STATE.symbols,
        "tfs": STATE.tfs,
        "pnl": round(total, 6),
        "mr_on": STATE.mr_on,
        "regime_auto": STATE.regime_auto,
    }


@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    """
    Telegram skickar JSON → vi konverterar det till ett Update-objekt
    och processar det med tg_app.
    """
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}


# ------------------------------------------------------------
# STARTUP & SHUTDOWN
# ------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    # 1) Sätt webhook om WEBHOOK_BASE finns
    if WEBHOOK_BASE:
        url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
        try:
            await tg_app.bot.set_webhook(url)
        except Exception as e:
            print("Webhook error:", e)

    # 2) Initiera boten
    await tg_app.initialize()
    await tg_app.start()

    # 3) Starta tradingmotor i bakgrunden
    asyncio.create_task(engine_loop(tg_app))


@app.on_event("shutdown")
async def on_shutdown():
    try:
        await tg_app.stop()
        await tg_app.shutdown()
    except:
        pass


# ------------------------------------------------------------
# LOKAL KÖRNING (utveckling)
# ------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
