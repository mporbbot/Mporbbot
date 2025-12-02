# -*- coding: utf-8 -*-
# MP Hybrid ORB/MOMO Bot – Optimized v1 (NO SHORT)
# By ChatGPT – custom-built for Mp ORBbot

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

TFS = os.getenv("TIMEFRAMES", "3m").split(",")

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

LAST_ENTRY = {}
LAST_EXIT = {}

# =====================================================
# HELPERS
# =====================================================

def now():
    return datetime.now(timezone.utc)

def _fee(amount: float):
    return amount * STATE.fee_side

def log_trade(row: dict):
    if STATE.mode == "live":
        fn = "real_trade_log.csv"
    else:
        fn = "mock_trade_log.csv"
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
    except:
        return None
    try:
        return {
            "price": float(d.get("price") or 0),
            "bestBid": float(d.get("bestBid") or 0),
            "bestAsk": float(d.get("bestAsk") or 0),
        }
    except:
        return None

# =====================================================
# KUCOIN PRIVATE (LIVE)
# =====================================================

def kc_ok():
    return bool(KC_KEY and KC_SECRET and KC_PASSPHRASE)

async def kc_priv(method, path, body=None):
    if not kc_ok():
        return None
    body = body or {}
    body_str = "" if method == "GET" else json.dumps(body, separators=(",", ":"))

    ts = str(int(time.time() * 1000))
    sign = base64.b64encode(
        hmac.new(KC_SECRET.encode(), (ts + method + path + body_str).encode(), hashlib.sha256).digest()
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
    except Exception as e:
        return None

    try:
        data = r.json()
    except:
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
        return False, "Order fail"

    oid = (d.get("data") or {}).get("orderId")
    if not oid:
        return False, "Missing orderId"

    # fetch fills
    fills = await kc_priv("GET", f"/api/v1/fills?orderId={oid}&tradeType=TRADE")
    try:
        items = fills.get("data") or fills.get("data", {}).get("items") or []
        if not isinstance(items, list):
            items = []
    except:
        items = []

    size = sum(float(i.get("size") or 0) for i in items)
    funds = sum(float(i.get("funds") or 0) for i in items)
    fee = sum(float(i.get("fee") or 0) for i in items)
    avg = funds / size if size > 0 else 0

    info = {
        "size": size,
        "funds": funds,
        "fee": fee,
        "avg": avg,
        "oid": oid,
    }

    if side == "buy":
        LAST_ENTRY[symbol] = info
    else:
        LAST_EXIT[symbol] = info

    return True, None

# =====================================================
# MARKET DATA + INDICATORS
# =====================================================

async def get_klines(symbol: str, tf: str, limit=100):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(KC_KLINES, params={
                "symbol": symbol,
                "type": TF_MAP.get(tf, tf)
            })
            r.raise_for_status()
            data = r.json().get("data", [])
    except:
        return []
    return data[::-1][:limit]

def ema(series, p):
    if len(series) < 2: return series[:]
    k = 2 / (p + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def rsi(closes, period=14):
    if len(closes) < period + 2:
        return [50] * len(closes)
    gains = []
    losses = []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0))
        losses.append(-min(ch, 0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    out = [50] * period
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l != 0 else 999
        out.append(100 - 100 / (1 + rs))
    return [50] + out

def features(kl):
    closes = [float(c[2]) for c in kl]
    highs = [float(c[3]) for c in kl]
    lows = [float(c[4]) for c in kl]

    if len(closes) < 40:
        x = closes[-1]
        return dict(close=x, ema20=x, ema50=x, rsi=50, mom=0, atrp=0.2, slope=0)

    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    mom = (closes[-1] - closes[-6]) / closes[-6] * 100
    r = rsi(closes, 14)[-1]

    trs = []
    for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:]):
        trs.append((h - l) / c * 100)
    atrp = sum(trs) / len(trs)

    slope = (e20[-1] - e20[-6]) / e20[-6] * 100 if e20[-6] else 0

    return dict(
        close=closes[-1],
        ema20=e20[-1],
        ema50=e50[-1],
        rsi=r,
        mom=mom,
        atrp=atrp,
        slope=slope
    )

def momentum_score(f):
    t = 1 if f["ema20"] > f["ema50"] else -1 if f["ema20"] < f["ema50"] else 0
    rdev = (f["rsi"] - 50) / 10
    score = t * abs(f["mom"] / 0.1) + rdev
    return score if f["mom"] > 0 else -abs(score)

def decide_regime(f):
    if not STATE.regime_auto:
        return "trend"
    if f["atrp"] >= STATE.atr_max and abs(f["slope"]) >= STATE.slope_min:
        return "trend"
    if f["atrp"] <= STATE.atr_max and abs(f["slope"]) < STATE.slope_min:
        return "range"
    return "trend"

# =====================================================
# POSITION MGMT
# =====================================================

def open_pos(sym, price, st, regime, reason, qty=None, usd_in=None, fee_in=None):
    if qty is None:
        qty = STATE.mock_size / price
        usd_in = STATE.mock_size
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

def close_pos(sym, st, reason, exit_price):
    if not st.pos:
        return 0.0
    p = st.pos

    if STATE.mode == "live":
        info = LAST_EXIT.get(sym)
        if info:
            usd_out = info["funds"]
            fee_out = info["fee"]
            usd_in = p.usd_in
            fee_in = p.fee_in
            net = usd_out - usd_in - fee_in - fee_out
            exit_price = info["avg"]
        else:
            usd_in = p.entry * p.qty
            usd_out = p.qty * exit_price
            fee_in = _fee(usd_in)
            fee_out = _fee(usd_out)
            net = usd_out - usd_in - fee_in - fee_out
    else:
        usd_in = p.entry * p.qty
        usd_out = p.qty * exit_price
        fee_in = _fee(usd_in)
        fee_out = _fee(usd_out)
        net = usd_out - usd_in - fee_in - fee_out

    st.pnl += net
    st.log.append({
        "time": now().isoformat(),
        "symbol": sym,
        "side": "LONG",
        "entry": p.entry,
        "exit": exit_price,
        "gross": usd_out - usd_in,
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
        "gross": usd_out - usd_in,
        "fee_in": fee_in,
        "fee_out": fee_out,
        "net": net,
        "info": reason,
    })

    st.pos = None
    return net

# =====================================================
# ENGINE LOOP (OPTIMIZED)
# =====================================================

async def engine_loop(app: Application):
    await asyncio.sleep(1)

    while True:
        if not STATE.engine:
            await asyncio.sleep(2)
            continue

        now_t = now()

        # loss guard pause
        if STATE.paused_until:
            if now_t < STATE.paused_until:
                await asyncio.sleep(2)
                continue
            else:
                STATE.paused_until = None
                STATE.loss_streak = 0

        # -------------------------------
        # ENTRY LOGIC
        # -------------------------------
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
                except:
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
                            entry_price = best["bestAsk"] if best else price
                            qty = STATE.mock_size / entry_price
                            usd_in = STATE.mock_size
                            fee_in = _fee(usd_in)

                        open_pos(sym, entry_price, st, "trend", "MOMO", qty, usd_in, fee_in)
                        used += 1
                        if used >= STATE.max_pos:
                            break

                elif regime == "range":
                    dev = (price - f["ema20"]) / f["ema20"] * 100
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
                            entry_price = best["bestAsk"] if best else price
                            qty = STATE.mock_size / entry_price
                            usd_in = STATE.mock_size
                            fee_in = _fee(usd_in)

                        open_pos(sym, entry_price, st, "range", "MR", qty, usd_in, fee_in)
                        used += 1
                        if used >= STATE.max_pos:
                            break

        # -------------------------------
        # POSITION MANAGEMENT
        # -------------------------------
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
            except:
                continue

            price = f["close"]
            p.high = max(p.high, price)
            p.low = min(p.low, price)

            move = (price - p.entry) / p.entry * 100

            # TRAIL start
            if not p.trailing and move >= STATE.trail_on:
                p.trailing = True

            # TAKE PROFIT
            if move >= STATE.tp and not p.trailing:
                if STATE.mode == "live":
                    await kc_order(sym, "sell", p.qty)
                    info = LAST_EXIT.get(sym, {})
                    exit_price = info.get("avg", price)
                else:
                    best = await get_best_prices(sym)
                    exit_price = best["bestBid"] if best else price

                net = close_pos(sym, st, "TP", exit_price)
                STATE.loss_streak = 0 if net > 0 else STATE.loss_streak + 1
                continue

            # TRAILING STOP
            if p.trailing:
                stop_lvl = p.high * (1 - STATE.trail_pct / 100)
                if price <= stop_lvl:
                    if STATE.mode == "live":
                        await kc_order(sym, "sell", p.qty)
                        info = LAST_EXIT.get(sym, {})
                        exit_price = info.get("avg", price)
                    else:
                        best = await get_best_prices(sym)
                        exit_price = best["bestBid"] if best else price

                    net = close_pos(sym, st, "TRAIL", exit_price)
                    STATE.loss_streak = 0 if net > 0 else STATE.loss_streak + 1
                    continue

            # STOP LOSS
            if move <= -STATE.sl:
                if STATE.mode == "live":
                    await kc_order(sym, "sell", p.qty)
                    info = LAST_EXIT.get(sym, {})
                    exit_price = info.get("avg", price)
                else:
                    best = await get_best_prices(sym)
                    exit_price = best["bestBid"] if best else price

                net = close_pos(sym, st, "SL", exit_price)
                STATE.loss_streak = 0 if net > 0 else STATE.loss_streak + 1

                if STATE.loss_streak >= STATE.loss_n:
                    STATE.paused_until = now() + timedelta(minutes=STATE.loss_pause)
                continue

        await asyncio.sleep(2)

# =====================================================
# TELEGRAM UI
# =====================================================

def kb():
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/mode"), KeyboardButton("/risk")],
        [KeyboardButton("/test_buy"), KeyboardButton("/close_all")],
        [KeyboardButton("/export_csv")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def status_text():
    t = sum(STATE.sym[s].pnl for s in STATE.symbols)
    pos = []
    for s in STATE.symbols:
        st = STATE.sym[s]
        if st.pos:
            p = st.pos
            pos.append(
                f"{s}: {p.side} @ {p.entry:.4f} | qty={p.qty:.6f} | "
                f"high={p.high:.4f} low={p.low:.4f} | reg={p.regime}"
            )

    return (
        f"Engine: {'ON' if STATE.engine else 'OFF'}\n"
        f"Mode: {STATE.mode.upper()}\n"
        f"Threshold: {STATE.threshold}\n"
        f"TP:{STATE.tp}% SL:{STATE.sl}% Trail:{STATE.trail_on}%/{STATE.trail_pct}%\n"
        f"Symbols: {', '.join(STATE.symbols)}\n"
        f"Positions: {len(pos)}\n"
        f"PnL total: {t:+.4f} USDT\n"
        + "\n".join(pos)
    )

# =====================================================
# TELEGRAM COMMANDS
# =====================================================

tg = Application.builder().token(BOT_TOKEN).build()

async def cmd_start(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    await tg.bot.send_message(STATE.chat_id,
        "🤖 MP ORB/MOMO Bot – Optimized\nMOCK-mode som standard.",
        reply_markup=kb()
    )
    await tg.bot.send_message(STATE.chat_id, status_text())

async def cmd_status(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    await tg.bot.send_message(STATE.chat_id, status_text(), reply_markup=kb())

async def cmd_engine_on(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    STATE.engine = True
    await tg.bot.send_message(STATE.chat_id, "Engine ON", reply_markup=kb())

async def cmd_engine_off(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    STATE.engine = False
    await tg.bot.send_message(STATE.chat_id, "Engine OFF", reply_markup=kb())

async def cmd_mode(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    parts = u.message.text.split()

    if len(parts) == 1:
        kb2 = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("MOCK", callback_data="mode_mock"),
                InlineKeyboardButton("LIVE", callback_data="mode_live")
            ]
        ])
        await tg.bot.send_message(STATE.chat_id,
            f"Mode: {STATE.mode}\nVälj nytt läge:",
            reply_markup=kb2
        )
        return

    mode = parts[1].lower()
    if mode == "mock":
        STATE.mode = "mock"
        await tg.bot.send_message(STATE.chat_id, "Mode → MOCK", reply_markup=kb())
        return
    if mode == "live":
        if not kc_ok():
            await tg.bot.send_message(
                STATE.chat_id, "KuCoin API saknas, kan inte slå på LIVE.",
                reply_markup=kb()
            )
            return
        STATE.mode = "live"
        await tg.bot.send_message(
            STATE.chat_id, "LIVE-läge AKTIVERAT.",
            reply_markup=kb()
        )
        return

async def on_button(update: Update, _):
    q = update.callback_query
    data = q.data.strip()
    STATE.chat_id = q.message.chat_id

    if data == "mode_mock":
        STATE.mode = "mock"
        await q.edit_message_text("Mode satt → MOCK")
        return

    if data == "mode_live":
        if not kc_ok():
            await q.edit_message_text("Saknar KuCoin API-nycklar.")
            return
        STATE.mode = "live"
        await q.edit_message_text("Mode satt → LIVE")
        return

async def cmd_risk(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    msg = u.message.text.split()

    if len(msg) == 4 and msg[1] == "set":
        key = msg[2]
        val = msg[3]
        try:
            if key == "tp": STATE.tp = float(val)
            elif key == "sl": STATE.sl = float(val)
            elif key == "trail": STATE.trail_pct = float(val)
            elif key == "trail_on": STATE.trail_on = float(val)
            elif key == "threshold": STATE.threshold = float(val)
            elif key == "max_pos": STATE.max_pos = int(val)
            elif key == "mr_dev": STATE.mr_dev = float(val)
            await tg.bot.send_message(STATE.chat_id, "Risk-parameter uppdaterad.", reply_markup=kb())
        except:
            await tg.bot.send_message(STATE.chat_id, "Fel värde.", reply_markup=kb())
        return

    await tg.bot.send_message(
        STATE.chat_id,
        "Använd: /risk set tp|sl|trail|trail_on|threshold|max_pos|mr_dev VALUE",
        reply_markup=kb()
    )

async def cmd_pnl(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    total = sum(STATE.sym[s].pnl for s in STATE.symbols)
    lines = [f"Total PnL: {total:+.3f}"]
    for s in STATE.symbols:
        lines.append(f"{s}: {STATE.sym[s].pnl:+.3f}")
    await tg.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=kb())

async def cmd_export(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    rows = [["time","symbol","side","entry","exit","gross","fee_in","fee_out","net","reason"]]

    for s in STATE.symbols:
        for r in STATE.sym[s].log:
            rows.append([
                r["time"], r["symbol"], r["side"], r["entry"], r["exit"],
                r["gross"], r["fee_in"], r["fee_out"], r["net"], r["reason"]
            ])

    if len(rows) <= 1:
        await tg.bot.send_message(STATE.chat_id, "Inga trades loggade.")
        return

    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    await tg.bot.send_document(
        STATE.chat_id,
        io.BytesIO(buf.getvalue().encode()),
        filename="export.csv",
        caption="Trades CSV",
        reply_markup=kb()
    )

async def cmd_testbuy(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    msg = u.message.text.split()

    if len(msg) == 1:
        await tg.bot.send_message(
            STATE.chat_id,
            "Använd: /test_buy SYMBOL [USDT]",
            reply_markup=kb()
        )
        return

    sym = msg[1].upper()
    if sym not in STATE.symbols:
        await tg.bot.send_message(
            STATE.chat_id, "Symbol ej i listan.", reply_markup=kb()
        )
        return

    size = STATE.mock_size
    if len(msg) >= 3:
        try: size = float(msg[2])
        except: pass

    tf = STATE.tfs[0]
    try:
        kl = await get_klines(sym, tf, limit=10)
        pr = features(kl)["close"]
    except:
        pr = 0

    if STATE.mode == "live":
        ok, err = await kc_order(sym, "buy", size)
        info = LAST_ENTRY.get(sym, {})
        entry = info.get("avg", pr)
        qty = info.get("size", size / entry)
        usd_in = info.get("funds", size)
        fee_in = info.get("fee", _fee(usd_in))
    else:
        best = await get_best_prices(sym)
        entry = best["bestAsk"] if best else pr
        qty = size / entry
        usd_in = size
        fee_in = _fee(size)

    st = STATE.sym[sym]
    open_pos(sym, entry, st, "trend", "TESTBUY", qty, usd_in, fee_in)

    await tg.bot.send_message(
        STATE.chat_id,
        f"Test-buy {sym} @ {entry}",
        reply_markup=kb()
    )

async def cmd_close_all(u: Update, _):
    STATE.chat_id = u.effective_chat.id
    count = 0
    total = 0

    for sym in STATE.symbols:
        st = STATE.sym[sym]
        p = st.pos
        if not p:
            continue

        tf = STATE.tfs[0]
        try:
            kl = await get_klines(sym, tf, limit=5)
            pr = features(kl)["close"]
        except:
            pr = p.entry

        if STATE.mode == "live":
            await kc_order(sym, "sell", p.qty)
            info = LAST_EXIT.get(sym, {})
            exit_price = info.get("avg", pr)
        else:
            best = await get_best_prices(sym)
            exit_price = best["bestBid"] if best else pr

        n = close_pos(sym, st, "MANUAL_CLOSE", exit_price)
        count += 1
        total += n

    await tg.bot.send_message(
        STATE.chat_id,
        f"Stängde {count} positioner.\nNetto: {total:+.3f}",
        reply_markup=kb()
    )

# =====================================================
# REGISTER HANDLERS
# =====================================================

tg.add_handler(CommandHandler("start", cmd_start))
tg.add_handler(CommandHandler("status", cmd_status))
tg.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg.add_handler(CommandHandler("mode", cmd_mode))
tg.add_handler(CommandHandler("risk", cmd_risk))
tg.add_handler(CommandHandler("pnl", cmd_pnl))
tg.add_handler(CommandHandler("export_csv", cmd_export))
tg.add_handler(CommandHandler("test_buy", cmd_testbuy))
tg.add_handler(CommandHandler("close_all", cmd_close_all))
tg.add_handler(CallbackQueryHandler(on_button))

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()

@app.on_event("startup")
async def startup():
    await tg.initialize()
    if WEBHOOK_BASE:
        url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
        await tg.bot.set_webhook(url)
    await tg.start()
    asyncio.create_task(engine_loop(tg))

@app.on_event("shutdown")
async def shutdown():
    await tg.stop()
    await tg.shutdown()

@app.get("/", response_class=PlainTextResponse)
async def root():
    pnl = sum(STATE.sym[s].pnl for s in STATE.symbols)
    return f"MP Bot OK | engine={STATE.engine} | pnl={pnl:+.3f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    return {
        "ok": True,
        "engine": STATE.engine,
        "mode": STATE.mode,
        "symbols": STATE.symbols,
        "tfs": STATE.tfs,
        "paused_until": STATE.paused_until.isoformat() if STATE.paused_until else None,
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def webhook(request: Request):
    data = await request.json()
    upd = Update.de_json(data, tg.bot)
    await tg.process_update(upd)
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
        reload=False
    )
