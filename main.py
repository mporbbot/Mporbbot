# -*- coding: utf-8 -*-
# MP Hybrid ORB/MOMO Bot – Optimized v2 (NO SHORT, buttons restored)

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
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from telegram import (
    Update, KeyboardButton, ReplyKeyboardMarkup,
    InlineKeyboardButton, InlineKeyboardMarkup
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler
)

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mp_optimized")

# =====================================================
# ENVIRONMENT
# =====================================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("ERROR: TELEGRAM_BOT_TOKEN not provided")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

SYMBOLS = os.getenv(
    "SYMBOLS",
    "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT"
).replace(" ", "").split(",")

TFS = os.getenv("TIMEFRAMES", "3m").replace(" ", "").split(",")

MOCK_SIZE_USDT = float(os.getenv("MOCK_SIZE_USDT", "10"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))  # 0.1%
MAX_POS = int(os.getenv("MAX_POS", "4"))

TP_PCT = float(os.getenv("TP_PCT", "1"))
SL_PCT = float(os.getenv("SL_PCT", "0.5"))
TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "2"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "1.2"))

ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.55"))

MR_DEV_PCT = float(os.getenv("MR_DEV_PCT", "1.20"))
TREND_SLOPE_MIN = float(os.getenv("TREND_SLOPE_MIN", "0.25"))
RANGE_ATR_MAX = float(os.getenv("RANGE_ATR_MAX", "0.75"))

LOSS_GUARD_N = int(os.getenv("LOSS_GUARD_N", "2"))
LOSS_GUARD_PAUSE = int(os.getenv("LOSS_GUARD_PAUSE_MIN", "15"))

# KuCoin
KC_KEY = os.getenv("KUCOIN_API_KEY", "")
KC_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KC_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KC_VER = os.getenv("KUCOIN_API_KEY_VERSION", "3")

KC_BASE = "https://api.kucoin.com"
KC_KLINES = KC_BASE + "/api/v1/market/candles"
KC_LEVEL1 = KC_BASE + "/api/v1/market/orderbook/level1"

TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}

# =====================================================
# STATE
# =====================================================

@dataclass
class Position:
    side: str  # always LONG
    entry: float
    qty: float
    opened: datetime
    high: float
    low: float
    trailing: bool
    regime: str
    reason: str
    usd_in: float
    fee_in: float

@dataclass
class SymState:
    pos: Optional[Position] = None
    pnl: float = 0.0
    log: List[dict] = field(default_factory=list)

@dataclass
class BotState:
    engine: bool = False
    mode: str = "mock"  # mock | live
    symbols: List[str] = field(default_factory=lambda: SYMBOLS.copy())
    tfs: List[str] = field(default_factory=lambda: TFS.copy())
    sym: Dict[str, SymState] = field(default_factory=dict)

    mock_size: float = MOCK_SIZE_USDT
    fee_side: float = FEE_PER_SIDE
    max_pos: int = MAX_POS

    tp: float = TP_PCT
    sl: float = SL_PCT
    trail_on: float = TRAIL_START_PCT
    trail_pct: float = TRAIL_PCT

    threshold: float = ENTRY_THRESHOLD
    mr_dev: float = MR_DEV_PCT
    regime_auto: bool = True
    slope_min: float = TREND_SLOPE_MIN
    atr_max: float = RANGE_ATR_MAX

    # loss guard
    loss_n: int = LOSS_GUARD_N
    loss_pause: int = LOSS_GUARD_PAUSE
    loss_streak: int = 0
    paused_until: Optional[datetime] = None

    chat_id: Optional[int] = None

STATE = BotState()
for s in STATE.symbols:
    STATE.sym[s] = SymState()

LAST_ENTRY: Dict[str, dict] = {}
LAST_EXIT: Dict[str, dict] = {}

# =====================================================
# HELPERS
# =====================================================

def now() -> datetime:
    return datetime.now(timezone.utc)

def _fee(amount: float) -> float:
    return amount * STATE.fee_side

def log_trade(row: dict):
    fn = "real_trade_log.csv" if STATE.mode == "live" else "mock_trade_log.csv"
    new = not os.path.exists(fn)
    with open(fn, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)

# =====================================================
# KUCOIN PUBLIC DATA: LEVEL-1 FOR MOCK
# =====================================================

async def get_best_prices(symbol: str) -> Optional[Dict[str, float]]:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(KC_LEVEL1, params={"symbol": symbol})
            r.raise_for_status()
            d = r.json().get("data", {})
    except Exception:
        return None
    try:
        return {
            "price": float(d.get("price") or 0),
            "bestBid": float(d.get("bestBid") or 0),
            "bestAsk": float(d.get("bestAsk") or 0),
        }
    except Exception:
        return None

# =====================================================
# KUCOIN PRIVATE (LIVE)
# =====================================================

def kc_ok() -> bool:
    return bool(KC_KEY and KC_SECRET and KC_PASSPHRASE)

async def kc_priv(method: str, path: str, body: Optional[dict] = None):
    if not kc_ok():
        return None
    body = body or {}
    body_str = "" if method == "GET" else json.dumps(body, separators=(",", ":"))

    ts = str(int(time.time() * 1000))
    sign_payload = ts + method + path + body_str
    sign = base64.b64encode(
        hmac.new(KC_SECRET.encode(), sign_payload.encode(), hashlib.sha256).digest()
    ).decode()
    pass_h = base64.b64encode(
        hmac.new(KC_SECRET.encode(), KC_PASSPHRASE.encode(), hashlib.sha256).digest()
    ).decode()

    headers = {
        "KC-API-KEY": KC_KEY,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": pass_h,
        "KC-API-KEY-VERSION": KC_VER,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(base_url=KC_BASE, timeout=10) as c:
            r = await c.request(method, path, content=body_str or None, headers=headers)
    except Exception:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    data["_status"] = r.status_code
    return data

async def kc_order(symbol: str, side: str, amount: float):
    if not kc_ok():
        return False, "Missing KuCoin API keys."

    body = {
        "clientOid": str(uuid.uuid4()),
        "side": side,
        "symbol": symbol,
        "type": "market",
    }
    if side == "buy":
        body["funds"] = f"{amount:.2f}"
    else:
        body["size"] = f"{amount:.8f}"

    d = await kc_priv("POST", "/api/v1/orders", body)
    if not d or d.get("_status") != 200 or d.get("code") != "200000":
        return False, "Order failed"

    oid = (d.get("data") or {}).get("orderId")
    if not oid:
        return False, "Missing orderId"

    fills = await kc_priv("GET", f"/api/v1/fills?orderId={oid}&tradeType=TRADE")
    try:
        raw = fills.get("data")
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and "items" in raw:
            items = raw["items"]
        else:
            items = []
    except Exception:
        items = []

    size = sum(float(i.get("size") or 0) for i in items)
    funds = sum(float(i.get("funds") or 0) for i in items)
    fee = sum(float(i.get("fee") or 0) for i in items)
    avg = funds / size if size > 0 else 0

    info = {"size": size, "funds": funds, "fee": fee, "avg": avg, "oid": oid}

    if side == "buy":
        LAST_ENTRY[symbol] = info
    else:
        LAST_EXIT[symbol] = info

    return True, None

# =====================================================
# MARKET DATA + INDICATORS
# =====================================================

async def get_klines(symbol: str, tf: str, limit: int = 100):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(KC_KLINES, params={
                "symbol": symbol,
                "type": TF_MAP.get(tf, tf)
            })
            r.raise_for_status()
            data = r.json().get("data", [])
    except Exception:
        return []
    return data[::-1][:limit]

def ema(series: List[float], p: int) -> List[float]:
    if len(series) < 2:
        return series[:]
    k = 2 / (p + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 2:
        return [50.0] * len(closes)
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0))
        losses.append(-min(ch, 0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    out = [50.0] * period
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l != 0 else 999.0
        out.append(100 - 100 / (1 + rs))
    return [50.0] + out

def features(kl) -> dict:
    closes = [float(c[2]) for c in kl]
    highs = [float(c[3]) for c in kl]
    lows = [float(c[4]) for c in kl]

    if len(closes) < 40:
        x = closes[-1]
        return dict(close=x, ema20=x, ema50=x, rsi=50.0, mom=0.0, atrp=0.2, slope=0.0)

    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    mom = (closes[-1] - closes[-6]) / closes[-6] * 100
    r = rsi(closes, 14)[-1]

    trs = []
    for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:]):
        trs.append((h - l) / c * 100)
    atrp = sum(trs) / len(trs) if trs else 0.2

    slope = (e20[-1] - e20[-6]) / e20[-6] * 100 if e20[-6] else 0.0

    return dict(
        close=closes[-1],
        ema20=e20[-1],
        ema50=e50[-1],
        rsi=r,
        mom=mom,
        atrp=atrp,
        slope=slope,
    )

def momentum_score(f: dict) -> float:
    ema20 = f["ema20"]
    ema50 = f["ema50"]
    mom = f["mom"]
    rsi_val = f["rsi"]

    if ema20 > ema50:
        trend = 1
    elif ema20 < ema50:
        trend = -1
    else:
        trend = 0

    rsi_dev = (rsi_val - 50.0) / 10.0
    score = trend * abs(mom / 0.1) + rsi_dev
    return score if mom >= 0 else -abs(score)

def decide_regime(f: dict) -> str:
    if not STATE.regime_auto:
        return "trend"
    atrp = f["atrp"]
    slope = abs(f["slope"])
    if atrp >= STATE.atr_max and slope >= STATE.slope_min:
        return "trend"
    if atrp <= STATE.atr_max and slope < STATE.slope_min:
        return "range"
    return "trend"

# =====================================================
# POSITION MGMT
# =====================================================

def open_pos(sym: str, price: float, st: SymState,
             regime: str, reason: str,
             qty: Optional[float] = None,
             usd_in: Optional[float] = None,
             fee_in: Optional[float] = None):
    if qty is None:
        qty = STATE.mock_size / price
    if usd_in is None:
        usd_in = qty * price
    if fee_in is None:
        fee_in = _fee(usd_in)

    st.pos = Position(
        side="LONG",
        entry=price,
        qty=qty,
        opened=now(),
        high=price,
        low=price,
        trailing=False,
        regime=regime,
        reason=reason,
        usd_in=usd_in,
        fee_in=fee_in,
    )

    log_trade({
        "time": now().isoformat(),
        "symbol": sym,
        "action": "ENTRY",
        "side": "LONG",
        "price": price,
        "qty": qty,
        "fee_in": fee_in,
        "fee_out": "",
        "gross": "",
        "net": "",
        "info": reason,
    })

def close_pos(sym: str, st: SymState, reason: str, exit_price: float) -> float:
    if not st.pos:
        return 0.0
    p = st.pos

    if STATE.mode == "live":
        info = LAST_EXIT.get(sym)
        if info:
            usd_in = p.usd_in
            fee_in = p.fee_in
            usd_out = info["funds"]
            fee_out = info["fee"]
            gross = usd_out - usd_in
            net = gross - fee_in - fee_out
            exit_price = info["avg"]
        else:
            usd_in = p.entry * p.qty
            usd_out = p.qty * exit_price
            fee_in = _fee(usd_in)
            fee_out = _fee(usd_out)
            gross = usd_out - usd_in
            net = gross - fee_in - fee_out
    else:
        usd_in = p.entry * p.qty
        usd_out = p.qty * exit_price
        fee_in = _fee(usd_in)
        fee_out = _fee(usd_out)
        gross = usd_out - usd_in
        net = gross - fee_in - fee_out

    st.pnl += net
    st.log.append({
        "time": now().isoformat(),
        "symbol": sym,
        "side": "LONG",
        "entry": p.entry,
        "exit": exit_price,
        "gross": gross,
        "fee_in": fee_in,
        "fee_out": fee_out,
        "net": net,
        "reason": reason,
    })

    log_trade({
        "time": now().isoformat(),
        "symbol": sym,
        "action": "EXIT",
        "side": "LONG",
        "price": exit_price,
        "qty": p.qty,
        "gross": gross,
        "fee_in": fee_in,
        "fee_out": fee_out,
        "net": net,
        "info": reason,
    })

    st.pos = None
    return net

# =====================================================
# ENGINE LOOP
# =====================================================

async def engine_loop(app: Application):
    await asyncio.sleep(2)

    while True:
        if not STATE.engine:
            await asyncio.sleep(2)
            continue

        now_t = now()

        # Loss-guard pause
        if STATE.paused_until:
            if now_t < STATE.paused_until:
                await asyncio.sleep(2)
                continue
            else:
                STATE.paused_until = None
                STATE.loss_streak = 0

        # ENTRY
        used = sum(1 for s in STATE.sym if STATE.sym[s].pos)
        can_open = used < STATE.max_pos

        if can_open:
            for sym in STATE.symbols:
                st = STATE.sym[sym]
                if st.pos:
                    continue

                tf = STATE.tfs[0]
                try:
                    kl = await get_klines(sym, tf)
                    if not kl:
                        continue
                    f = features(kl)
                except Exception:
                    continue

                price = f["close"]
                regime = decide_regime(f)

                if regime == "trend":
                    score = momentum_score(f)
                    if score > STATE.threshold:
                        if STATE.mode == "live":
                            ok, err = await kc_order(sym, "buy", STATE.mock_size)
                            if not ok:
                                continue
                            info = LAST_ENTRY.get(sym, {})
                            entry_price = info.get("avg", price)
                            qty = info.get("size", STATE.mock_size / entry_price)
                            usd_in = info.get("funds", STATE.mock_size)
                            fee_in = info.get("fee", _fee(usd_in))
                        else:
                            best = await get_best_prices(sym)
                            entry_price = best["bestAsk"] if best and best.get("bestAsk") else price
                            qty = STATE.mock_size / entry_price
                            usd_in = STATE.mock_size
                            fee_in = _fee(usd_in)

                        open_pos(sym, entry_price, st, "trend", "MOMO", qty, usd_in, fee_in)
                        used += 1
                        if used >= STATE.max_pos:
                            break

                elif regime == "range":
                    ema20 = f["ema20"]
                    if ema20 == 0:
                        continue
                    dev = (price - ema20) / ema20 * 100
                    if dev <= -STATE.mr_dev:
                        if STATE.mode == "live":
                            ok, err = await kc_order(sym, "buy", STATE.mock_size)
                            if not ok:
                                continue
                            info = LAST_ENTRY.get(sym, {})
                            entry_price = info.get("avg", price)
                            qty = info.get("size", STATE.mock_size / entry_price)
                            usd_in = info.get("funds", STATE.mock_size)
                            fee_in = info.get("fee", _fee(usd_in))
                        else:
                            best = await get_best_prices(sym)
                            entry_price = best["bestAsk"] if best and best.get("bestAsk") else price
                            qty = STATE.mock_size / entry_price
                            usd_in = STATE.mock_size
                            fee_in = _fee(usd_in)

                        open_pos(sym, entry_price, st, "range", "MR", qty, usd_in, fee_in)
                        used += 1
                        if used >= STATE.max_pos:
                            break

        # POSITION MGMT
        for sym in STATE.symbols:
            st = STATE.sym[sym]
            p = st.pos
            if not p:
                continue

            tf = STATE.tfs[0]
            try:
                kl = await get_klines(sym, tf, limit=10)
                if not kl:
                    continue
                f = features(kl)
            except Exception:
                continue

            price = f["close"]
            p.high = max(p.high, price)
            p.low = min(p.low, price)

            move = (price - p.entry) / p.entry * 100

            # Trail start
            if not p.trailing and move >= STATE.trail_on:
                p.trailing = True
                if STATE.chat_id:
                    try:
                        await app.bot.send_message(
                            STATE.chat_id,
                            f"🔒 TRAIL ON {sym} | move {move:.2f}%"
                        )
                    except Exception:
                        pass

            # Take profit
            if move >= STATE.tp and not p.trailing:
                if STATE.mode == "live":
                    await kc_order(sym, "sell", p.qty)
                    info = LAST_EXIT.get(sym, {})
                    exit_price = info.get("avg", price)
                else:
                    best = await get_best_prices(sym)
                    exit_price = best["bestBid"] if best and best.get("bestBid") else price

                net = close_pos(sym, st, "TP", exit_price)
                if net < 0:
                    STATE.loss_streak += 1
                elif net > 0 and STATE.loss_streak > 0:
                    STATE.loss_streak = 0
                continue

            # Trailing stop
            if p.trailing:
                stop_lvl = p.high * (1 - STATE.trail_pct / 100)
                if price <= stop_lvl:
                    if STATE.mode == "live":
                        await kc_order(sym, "sell", p.qty)
                        info = LAST_EXIT.get(sym, {})
                        exit_price = info.get("avg", price)
                    else:
                        best = await get_best_prices(sym)
                        exit_price = best["bestBid"] if best and best.get("bestBid") else price

                    net = close_pos(sym, st, "TRAIL", exit_price)
                    if net < 0:
                        STATE.loss_streak += 1
                    elif net > 0 and STATE.loss_streak > 0:
                        STATE.loss_streak = 0
                    continue

            # Stop loss
            if move <= -STATE.sl:
                if STATE.mode == "live":
                    await kc_order(sym, "sell", p.qty)
                    info = LAST_EXIT.get(sym, {})
                    exit_price = info.get("avg", price)
                else:
                    best = await get_best_prices(sym)
                    exit_price = best["bestBid"] if best and best.get("bestBid") else price

                net = close_pos(sym, st, "SL", exit_price)
                if net < 0:
                    STATE.loss_streak += 1
                elif net > 0 and STATE.loss_streak > 0:
                    STATE.loss_streak = 0

                if STATE.loss_streak >= STATE.loss_n:
                    STATE.paused_until = now() + timedelta(minutes=STATE.loss_pause)
                    if STATE.chat_id:
                        try:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"⏸ Loss-guard aktiv – paus i {STATE.loss_pause} minuter."
                            )
                        except Exception:
                            pass
                continue

        await asyncio.sleep(2)

# =====================================================
# TELEGRAM UI
# =====================================================

def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/mode"), KeyboardButton("/risk")],
        [KeyboardButton("/test_buy"), KeyboardButton("/close_all")],
        [KeyboardButton("/export_csv")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def status_text() -> str:
    total = sum(STATE.sym[s].pnl for s in STATE.symbols)
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.sym[s]
        if st.pos:
            p = st.pos
            pos_lines.append(
                f"{s}: LONG @ {p.entry:.4f} | qty={p.qty:.6f} | "
                f"hi={p.high:.4f} lo={p.low:.4f} | {p.regime}/{p.reason}"
            )

    loss_line = (
        f"Loss-guard: N={STATE.loss_n}, pause={STATE.loss_pause}m, "
        f"streak={STATE.loss_streak}"
    )
    if STATE.paused_until:
        loss_line += f" | paus till {STATE.paused_until.astimezone().strftime('%H:%M')}"

    lines = [
        f"Engine: {'ON ✅' if STATE.engine else 'OFF ⛔️'}",
        f"Mode: {STATE.mode.upper()}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"TFs: {', '.join(STATE.tfs)}",
        f"Threshold: {STATE.threshold:.2f}",
        f"TP: {STATE.tp:.2f}% | SL: {STATE.sl:.2f}% | "
        f"Trail start: {STATE.trail_on:.2f}% | Trail: {STATE.trail_pct:.2f}%",
        f"Max pos: {STATE.max_pos}",
        loss_line,
        f"PnL total: {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    return "\n".join(lines)

# =====================================================
# TELEGRAM COMMANDS
# =====================================================

tg_app = Application.builder().token(BOT_TOKEN).build()

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(
        STATE.chat_id,
        "🤖 MP ORB/MOMO Bot – Optimized v2\nStandardläge: MOCK (simulerad).",
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
    STATE.engine = True
    await tg_app.bot.send_message(
        STATE.chat_id, "Engine ON ✅", reply_markup=reply_kb()
    )

async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine = False
    await tg_app.bot.send_message(
        STATE.chat_id, "Engine OFF ⛔️", reply_markup=reply_kb()
    )

async def cmd_mode(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    parts = update.message.text.strip().split()

    if len(parts) == 1:
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("MOCK", callback_data="mode_mock"),
                    InlineKeyboardButton("LIVE", callback_data="mode_live"),
                ]
            ]
        )
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Aktuellt läge: {STATE.mode.upper()}\nVälj nytt läge:",
            reply_markup=kb,
        )
        return

    if parts[1].lower() == "mock":
        STATE.mode = "mock"
        await tg_app.bot.send_message(
            STATE.chat_id, "Läge satt till MOCK.", reply_markup=reply_kb()
        )
        return

    if parts[1].lower() == "live":
        if not kc_ok():
            await tg_app.bot.send_message(
                STATE.chat_id,
                "❌ Kan inte aktivera LIVE: KuCoin-API uppgifter saknas.",
                reply_markup=reply_kb(),
            )
            return
        STATE.mode = "live"
        await tg_app.bot.send_message(
            STATE.chat_id,
            "✅ LIVE-läge aktiverat (spot, endast LONG).",
            reply_markup=reply_kb(),
        )
        return

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.sym[s].pnl for s in STATE.symbols)
    lines = [f"📈 Total PnL: {total:+.4f} USDT"]
    for s in STATE.symbols:
        lines.append(f"{s}: {STATE.sym[s].pnl:+.4f} USDT")
    await tg_app.bot.send_message(
        STATE.chat_id, "\n".join(lines), reply_markup=reply_kb()
    )

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    rows = [["time","symbol","side","entry","exit","gross","fee_in","fee_out","net","reason"]]
    for s in STATE.symbols:
        for r in STATE.sym[s].log:
            rows.append([
                r["time"], r["symbol"], r["side"], r["entry"], r["exit"],
                r["gross"], r["fee_in"], r["fee_out"], r["net"], r["reason"],
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
        reply_markup=reply_kb(),
    )

# ---------------- RISK COMMAND ----------------

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()

    # /risk set key value
    if len(toks) == 4 and toks[1] == "set":
        key = toks[2]
        val = toks[3]
        try:
            if key == "size":
                STATE.mock_size = float(val)
            elif key == "tp":
                STATE.tp = float(val)
            elif key == "sl":
                STATE.sl = float(val)
            elif key == "trail_on":
                STATE.trail_on = float(val)
            elif key == "trail":
                STATE.trail_pct = float(val)
            elif key == "max_pos":
                STATE.max_pos = int(val)
            elif key == "threshold":
                STATE.threshold = float(val)
            elif key == "mr_dev":
                STATE.mr_dev = float(val)
            elif key == "loss_n":
                STATE.loss_n = int(val)
            elif key == "loss_pause":
                STATE.loss_pause = int(val)
            else:
                raise ValueError()
            await tg_app.bot.send_message(
                STATE.chat_id, "Riskinställningar uppdaterade.", reply_markup=reply_kb()
            )
        except Exception:
            await tg_app.bot.send_message(
                STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb()
            )
        return

    # Visa knappar
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
        f"TP: {STATE.tp:.2f}% | SL: {STATE.sl:.2f}%\n"
        f"Trail start: {STATE.trail_on:.2f}% | Trail: {STATE.trail_pct:.2f}%\n"
        f"Max Pos: {STATE.max_pos}\n\n"
        "Välj knapp nedan eller använd:\n"
        "`/risk set key value`"
    )

    await tg_app.bot.send_message(
        STATE.chat_id, text, reply_markup=kb
    )

# ---------------- TIMEFRAME / THRESHOLD ----------------

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
    except Exception:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Fel värde. Exempel: /threshold 0.55",
            reply_markup=reply_kb(),
        )

# ---------------- TEST BUY (WITH CONFIRM BUTTONS) ----------------

async def cmd_test_buy(update: Update, _):
    """
    /test_buy SYMBOL [USDT]
    Visar bekräftelseknappar innan köp.
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
        except Exception:
            pass

    mode = "LIVE" if STATE.mode == "live" else "MOCK"

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

# ---------------- CLOSE ALL ----------------

async def cmd_close_all(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = 0
    total_net = 0.0

    for sym in STATE.symbols:
        st = STATE.sym[sym]
        p = st.pos
        if not p:
            continue

        tf = STATE.tfs[0]
        try:
            kl = await get_klines(sym, tf, limit=5)
            price = features(kl)["close"]
        except Exception:
            price = p.entry

        if STATE.mode == "live":
            await kc_order(sym, "sell", p.qty)
            info = LAST_EXIT.get(sym, {})
            exit_price = info.get("avg", price)
        else:
            best = await get_best_prices(sym)
            exit_price = best["bestBid"] if best and best.get("bestBid") else price

        net = close_pos(sym, st, "MANUAL_CLOSE", exit_price)
        closed += 1
        total_net += net

    await tg_app.bot.send_message(
        STATE.chat_id,
        f"Close-all klart.\nStängda positioner: {closed}\nNetto: {total_net:+.4f} USDT",
        reply_markup=reply_kb(),
    )

# =====================================================
# CALLBACK HANDLER (buttons)
# =====================================================

async def on_button(update: Update, _):
    query = update.callback_query
    data = (query.data or "").strip()
    STATE.chat_id = query.message.chat_id

    # --- MODE ---
    if data == "mode_mock":
        await query.answer()
        STATE.mode = "mock"
        await query.edit_message_text("Läge satt till MOCK.")
        return

    if data == "mode_live":
        await query.answer()
        if not kc_ok():
            await query.edit_message_text("❌ LIVE kan inte aktiveras: saknar KuCoin API-nycklar.")
            return
        STATE.mode = "live"
        await query.edit_message_text("✅ Läge satt till LIVE (spot, endast LONG).")
        return

    # --- RISK BUTTONS ---
    if data.startswith("risk_size_"):
        await query.answer()
        try:
            size = float(data.split("_")[-1])
            STATE.mock_size = size
            await query.edit_message_text(f"Size uppdaterad till {size:.1f} USDT.")
        except Exception:
            await query.edit_message_text("Felaktigt värde för size.")
        return

    if data.startswith("risk_tp_"):
        await query.answer()
        try:
            tp = float(data.split("_")[-1])
            STATE.tp = tp
            await query.edit_message_text(f"TP uppdaterad till {tp:.2f}%.")
        except Exception:
            await query.edit_message_text("Fel vid uppdatering av TP.")
        return

    if data.startswith("risk_sl_"):
        await query.answer()
        try:
            sl = float(data.split("_")[-1])
            STATE.sl = sl
            await query.edit_message_text(f"SL uppdaterad till {sl:.2f}%.")
        except Exception:
            await query.edit_message_text("Fel vid uppdatering av SL.")
        return

    if data.startswith("risk_maxpos_"):
        await query.answer()
        try:
            val = int(data.split("_")[-1])
            STATE.max_pos = val
            await query.edit_message_text(f"max_pos uppdaterad till {val}.")
        except Exception:
            await query.edit_message_text("Felaktigt max_pos värde.")
        return

    # --- TESTBUY BUTTONS ---
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
        except Exception:
            size = STATE.mock_size

        # Hämta baspris
        tf = STATE.tfs[0]
        try:
            kl = await get_klines(symbol, tf, limit=10)
            price = features(kl)["close"]
        except Exception:
            price = 0.0

        if STATE.mode == "live":
            ok, err = await kc_order(symbol, "buy", size)
            if not ok:
                await query.edit_message_text(
                    f"❌ LIVE test-köp misslyckades för {symbol}\n{err}"
                )
                return
            info = LAST_ENTRY.get(symbol, {})
            entry_price = info.get("avg", price)
            qty = info.get("size", size / entry_price if entry_price else 0.0)
            usd_in = info.get("funds", size)
            fee_in = info.get("fee", _fee(usd_in))
        else:
            best = await get_best_prices(symbol)
            entry_price = best["bestAsk"] if best and best.get("bestAsk") else price
            qty = size / entry_price if entry_price else 0.0
            usd_in = size
            fee_in = _fee(usd_in)

        st = STATE.sym[symbol]
        open_pos(symbol, entry_price, st, "trend", "TESTBUY", qty, usd_in, fee_in)

        mode = "LIVE" if STATE.mode == "live" else "MOCK"
        await query.edit_message_text(
            f"✅ Test-köp öppnat:\n{symbol} @ {entry_price:.4f} ({mode}).\n"
            "Engine hanterar nu positionen (TP/SL/Trail)."
        )
        return

    # Fallback
    await query.answer()
    await query.edit_message_text("Ok.")

# =====================================================
# REGISTER HANDLERS
# =====================================================

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("mode", cmd_mode))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("threshold", cmd_threshold))
tg_app.add_handler(CommandHandler("test_buy", cmd_test_buy))
tg_app.add_handler(CommandHandler("close_all", cmd_close_all))
tg_app.add_handler(CallbackQueryHandler(on_button))

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    log.info("Startup: init Telegram app, webhook + engine")
    await tg_app.initialize()
    if WEBHOOK_BASE:
        url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
        await tg_app.bot.set_webhook(url)
        log.info("Webhook satt till %s", url)
    else:
        log.warning("WEBHOOK_BASE saknas – glöm inte sätta den i environment!")
    await tg_app.start()
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    log.info("Shutdown: stoppar Telegram app")
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/", response_class=PlainTextResponse)
async def root():
    pnl = sum(STATE.sym[s].pnl for s in STATE.symbols)
    return (
        f"MP Bot Optimized v2 OK | engine={STATE.engine} | "
        f"mode={STATE.mode} | pnl={pnl:+.4f}"
    )

@app.get("/health", response_class=JSONResponse)
async def health():
    return {
        "ok": True,
        "engine": STATE.engine,
        "mode": STATE.mode,
        "symbols": STATE.symbols,
        "tfs": STATE.tfs,
        "pnl_total": round(sum(STATE.sym[s].pnl for s in STATE.symbols), 6),
        "loss_guard": {
            "n": STATE.loss_n,
            "pause_min": STATE.loss_pause,
            "streak": STATE.loss_streak,
            "paused_until": STATE.paused_until.isoformat() if STATE.paused_until else None,
        },
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

# =====================================================
# LOCAL RUN
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
