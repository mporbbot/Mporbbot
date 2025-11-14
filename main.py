# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MP Bot – v52 (Hybrid Momentum + Mean Reversion, MOCK only)
# + Loss-guard (pause efter X förlusttrades i rad)
# ------------------------------------------------------------
# - Endast mock-trades (ingen riktig handel)
# - Livepriser från KuCoin
# - Hybrid-strategi:
#   * Trend-läge: Simple Momentum (som v50)
#   * Range-läge: Mean Reversion runt EMA20
#   * Automatisk växling per symbol (trend vs sidled)
# - TP/SL i % + enkel trailing stop
# - Loss-guard: pausar nya trades i symbol efter N förluster i rad
# - Telegram + FastAPI (Render-kompatibel)
# ------------------------------------------------------------

import os, io, csv, asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# -----------------------------
# ENV
# -----------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
                   .replace(" ", "")).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")

MOCK_SIZE_USDT = float(os.getenv("MOCK_SIZE_USDT", "50"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "4"))

TP_PCT = float(os.getenv("TP_PCT", "0.6"))       # take-profit
SL_PCT = float(os.getenv("SL_PCT", "0.6"))       # stop-loss
TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "2.0"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "2.5"))

ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.30"))  # momentum-score
ALLOW_SHORTS_DEFAULT = (os.getenv("ALLOW_SHORTS", "false").lower() in ("1", "true", "on", "yes"))

# Mean Reversion / regime defaults
MR_DEV_PCT = float(os.getenv("MR_DEV_PCT", "1.2"))        # avvikelse från EMA20 i %
TREND_SLOPE_MIN = float(os.getenv("TREND_SLOPE_MIN", "0.12"))  # min lutning EMA20 (i %)
RANGE_ATR_MAX = float(os.getenv("RANGE_ATR_MAX", "0.6"))  # max ATR% för range

# Loss-guard defaults
LOSS_GUARD_ON = os.getenv("LOSS_GUARD_ON", "true").lower() in ("1", "true", "on", "yes")
LOSS_GUARD_N = int(os.getenv("LOSS_GUARD_N", "3"))               # förlusttrades i rad
LOSS_GUARD_PAUSE_M = float(os.getenv("LOSS_GUARD_PAUSE_M", "10"))  # paus i minuter

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
    # Loss-guard
    loss_guard_on: bool = LOSS_GUARD_ON
    loss_guard_n: int = LOSS_GUARD_N
    loss_guard_pause_m: float = LOSS_GUARD_PAUSE_M
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# Loss-guard: antal förluster i rad / paus-tid per symbol
loss_streak: Dict[str, int] = defaultdict(int)
loss_pause_until: Dict[str, datetime] = defaultdict(
    lambda: datetime.fromtimestamp(0, tz=timezone.utc)
)

# -----------------------------
# Helpers
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
            fieldnames=["time","symbol","action","side","price","qty",
                        "gross","fee_in","fee_out","net","info"]
        )
        if new:
            w.writeheader()
        w.writerow(row)

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
    atrp = sum(trs)/len(trs) if trs else 0.2

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
    ema20 = feats["ema20"]; ema50 = feats["ema50"]
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
# Trading helpers (mock)
# -----------------------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

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
        reason=reason
    )
    log_mock({
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
        "info": f"size_usdt={STATE.mock_size};regime={regime};reason={reason}"
    })

def close_position(sym: str, price: float, st: SymState, reason: str) -> float:
    if not st.pos:
        return 0.0
    pos = st.pos
    usd_in = pos.qty * pos.entry_price
    usd_out = pos.qty * price
    fee_in = _fee(usd_in)
    fee_out = _fee(usd_out)
    if pos.side == "LONG":
        gross = pos.qty * (price - pos.entry_price)
    else:
        gross = pos.qty * (pos.entry_price - price)
    net = gross - fee_in - fee_out
    st.realized_pnl += net

    now = datetime.now(timezone.utc)

    # --- LOSS-GUARD uppdatering ---
    if STATE.loss_guard_on:
        if net < 0:
            loss_streak[sym] += 1
        else:
            loss_streak[sym] = 0

        if loss_streak[sym] >= STATE.loss_guard_n:
            pause_minutes = STATE.loss_guard_pause_m
            loss_pause_until[sym] = now + timedelta(minutes=pause_minutes)
            loss_streak[sym] = 0  # börja om efter pausen
    # --- /LOSS-GUARD ---

    st.trades_log.append({
        "time": now.isoformat(),
        "symbol": sym,
        "side": pos.side,
        "entry": pos.entry_price,
        "exit": price,
        "gross": round(gross, 6),
        "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6),
        "net": round(net, 6),
        "reason": f"{reason};regime={pos.regime};src={pos.reason}"
    })
    log_mock({
        "time": now.isoformat(),
        "symbol": sym,
        "action": "EXIT",
        "side": pos.side,
        "price": round(price, 6),
        "qty": round(pos.qty, 8),
        "gross": round(gross, 6),
        "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6),
        "net": round(net, 6),
        "info": f"{reason};regime={pos.regime};src={pos.reason}"
    })
    st.pos = None
    return net

# -----------------------------
# ENGINE
# -----------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                now = datetime.now(timezone.utc)

                # -------- Entries --------
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.max_pos:
                    for sym in STATE.symbols:
                        st = STATE.per_sym[sym]
                        if st.pos:
                            continue

                        # LOSS-GUARD: pausa nya trades i symbol under paus-fönstret
                        if STATE.loss_guard_on and now < loss_pause_until[sym]:
                            continue

                        tf = STATE.tfs[0] if STATE.tfs else "1m"
                        try:
                            kl = await get_klines(sym, tf, limit=80)
                        except Exception:
                            continue
                        feats = compute_features(kl)
                        price = feats["close"]
                        regime = decide_regime(feats)

                        if regime == "trend":
                            # Momentumläge (som v50)
                            score = momentum_score(feats)
                            if score > STATE.threshold:
                                # LONG
                                open_position(sym, "LONG", price, st, regime="trend", reason="MOMO")
                                open_syms.append(sym)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 MOMO ENTRY {sym} LONG @ {price:.4f} | "
                                        f"score={score:.2f} | thr={STATE.threshold:.2f} | "
                                        f"regime=trend"
                                    )
                                if len(open_syms) >= STATE.max_pos:
                                    break
                            elif STATE.allow_shorts and score < -STATE.threshold:
                                # SHORT
                                open_position(sym, "SHORT", price, st, regime="trend", reason="MOMO")
                                open_syms.append(sym)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 MOMO ENTRY {sym} SHORT @ {price:.4f} | "
                                        f"score={score:.2f} | thr={STATE.threshold:.2f} | "
                                        f"regime=trend"
                                    )
                                if len(open_syms) >= STATE.max_pos:
                                    break

                        elif regime == "range" and STATE.mr_on:
                            # Mean Reversion-läge
                            ema20 = feats["ema20"]
                            if ema20 == 0:
                                continue
                            dev_pct = (price - ema20) / ema20 * 100.0
                            # För långt under EMA20 => LONG
                            if dev_pct <= -STATE.mr_dev_pct:
                                open_position(sym, "LONG", price, st, regime="range", reason="MR")
                                open_syms.append(sym)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 MR ENTRY {sym} LONG @ {price:.4f} | "
                                        f"dev={dev_pct:.2f}% | regime=range"
                                    )
                                if len(open_syms) >= STATE.max_pos:
                                    break
                            # För långt över EMA20 => SHORT (om tillåtet)
                            elif dev_pct >= STATE.mr_dev_pct and STATE.allow_shorts:
                                open_position(sym, "SHORT", price, st, regime="range", reason="MR")
                                open_syms.append(sym)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔻 MR ENTRY {sym} SHORT @ {price:.4f} | "
                                        f"dev={dev_pct:.2f}% | regime=range"
                                    )
                                if len(open_syms) >= STATE.max_pos:
                                    break

                # -------- Manage positions --------
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

                    # uppdatera high/low
                    pos.high_water = max(pos.high_water, price)
                    pos.low_water = min(pos.low_water, price)

                    move_pct = (price - pos.entry_price) / (pos.entry_price or 1.0) * 100.0
                    if pos.side == "SHORT":
                        move_pct = -move_pct  # positivt = vinst även för short

                    # starta trailing om vi når trail_start_pct
                    if (not pos.trailing) and move_pct >= STATE.trail_start_pct:
                        pos.trailing = True
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🔒 TRAIL ON {sym} | move≈{move_pct:.2f}% "
                                f"(regime={pos.regime},src={pos.reason})"
                            )

                    # TP (om vi inte trailar)
                    if move_pct >= STATE.tp_pct and not pos.trailing:
                        net = close_position(sym, price, st, "TP")
                        if STATE.chat_id:
                            mark = "✅" if net >= 0 else "❌"
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🎯 TP EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}"
                            )
                        continue

                    # Trailing stop
                    if pos.trailing:
                        if pos.side == "LONG":
                            trail_stop = pos.high_water * (1.0 - STATE.trail_pct / 100.0)
                            if price <= trail_stop:
                                net = close_position(sym, price, st, "TRAIL")
                                if STATE.chat_id:
                                    mark = "✅" if net >= 0 else "❌"
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}"
                                    )
                                continue
                        else:  # SHORT
                            trail_stop = pos.low_water * (1.0 + STATE.trail_pct / 100.0)
                            if price >= trail_stop:
                                net = close_position(sym, price, st, "TRAIL")
                                if STATE.chat_id:
                                    mark = "✅" if net >= 0 else "❌"
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}"
                                    )
                                continue

                    # Fast SL
                    if move_pct <= -STATE.sl_pct:
                        net = close_position(sym, price, st, "SL")
                        if STATE.chat_id:
                            mark = "✅" if net >= 0 else "❌"
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"⛔ SL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}"
                            )
                        continue

            await asyncio.sleep(3)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# -----------------------------
# UI
# -----------------------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/threshold")],
        [KeyboardButton("/risk"), KeyboardButton("/export_csv")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def status_text() -> str:
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            p = st.pos
            move_pct = (p.high_water - p.entry_price) / (p.entry_price or 1.0) * 100.0
            pos_lines.append(
                f"{s}: {p.side} @ {p.entry_price:.4f} | qty {p.qty:.6f} | "
                f"hi {p.high_water:.4f} | lo {p.low_water:.4f} | "
                f"max_move≈{move_pct:.2f}% | regime={p.regime},src={p.reason}"
            )

    regime_line = f"Regime: AUTO (trend + mean reversion)" if STATE.regime_auto else "Regime: TREND only"
    mr_line = f"MR: {'ON' if STATE.mr_on else 'OFF'} | dev={STATE.mr_dev_pct:.2f}% | range_atr_max={STATE.range_atr_max:.2f}%"
    trend_line = f"Trend-slope min: {STATE.trend_slope_min:.2f}%"

    # Loss-guard status
    lg_line = (
        f"Loss-guard: {'ON' if STATE.loss_guard_on else 'OFF'} | "
        f"N={STATE.loss_guard_n} | pause={STATE.loss_guard_pause_m:.0f}m"
    )

    # Pausade symbols
    now = datetime.now(timezone.utc)
    paused = [
        f"{s} ({max(0, int((loss_pause_until[s] - now).total_seconds() // 60))}m)"
        for s in STATE.symbols
        if now < loss_pause_until[s]
    ]
    paused_line = ""
    if paused:
        paused_line = "Pausade (loss-guard): " + ", ".join(paused)

    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        "Strategi: Hybrid Momentum + Mean Reversion (MOCK)",
        regime_line,
        mr_line,
        trend_line,
        lg_line,
        f"Threshold (MOMO): {STATE.threshold:.2f}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Mock-size: {STATE.mock_size:.1f} USDT | Fee per sida: {STATE.fee_side:.4%}",
        f"Risk: tp={STATE.tp_pct:.2f}% | sl={STATE.sl_pct:.2f}% | "
        f"trail_start={STATE.trail_start_pct:.2f}% | trail={STATE.trail_pct:.2f}% | "
        f"shorts={'ON' if STATE.allow_shorts else 'OFF'} | max_pos={STATE.max_pos}",
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    if paused_line:
        lines.append(paused_line)
    return "\n".join(lines)

# -----------------------------
# Telegram commands
# -----------------------------
tg_app = Application.builder().token(BOT_TOKEN).build()

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(
        STATE.chat_id,
        "🤖 MP Bot v52 – Hybrid Momentum + Mean Reversion (MOCK only)\n"
        "Starta engine med /engine_on\n"
        "Justera momentum-känslighet med /threshold 0.30 (lägre = fler trades)",
        reply_markup=reply_kb()
    )
    await tg_app.bot.send_message(STATE.chat_id, status_text())

async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, status_text(), reply_markup=reply_kb())

async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(STATE.chat_id, "Engine: ON ✅ (mock)", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

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
                reply_markup=reply_kb()
            )
            return
    await tg_app.bot.send_message(
        STATE.chat_id,
        "Använd: /timeframe 1m,3m,5m",
        reply_markup=reply_kb()
    )

async def cmd_threshold(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks) == 1:
        await tg_app.bot.send_message(
            STATE.chat_id,
            f"Aktuellt momentum-threshold: {STATE.threshold:.2f}",
            reply_markup=reply_kb()
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
            reply_markup=reply_kb()
        )
    except:
        await tg_app.bot.send_message(
            STATE.chat_id,
            "Fel värde. Ex: /threshold 0.30",
            reply_markup=reply_kb()
        )

def _yesno(s: str) -> bool:
    return s.lower() in ("1", "true", "on", "yes")

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    # /risk set key val
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
            elif key == "loss_guard_pause_m":
                STATE.loss_guard_pause_m = float(val)
            else:
                await tg_app.bot.send_message(
                    STATE.chat_id,
                    ("Stödjer: size, tp, sl, trail_start, trail, max_pos, "
                     "allow_shorts, mr_on, mr_dev, regime_auto, trend_slope_min, "
                     "range_atr_max, loss_guard_on, loss_guard_n, loss_guard_pause_m"),
                    reply_markup=reply_kb()
                )
                return
            await tg_app.bot.send_message(STATE.chat_id, "Risk/Regime uppdaterad.", reply_markup=reply_kb())
        except:
            await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
        return

    await tg_app.bot.send_message(STATE.chat_id, status_text(), reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    lines = [f"📈 PnL total (NET): {total:+.4f} USDT"]
    for s in STATE.symbols:
        lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl:+.4f} USDT")
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    rows = [["time","symbol","side","entry","exit","gross","fee_in","fee_out","net","reason"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([
                r["time"], r["symbol"], r["side"], r["entry"], r["exit"],
                r["gross"], r["fee_in"], r["fee_out"], r["net"], r["reason"]
            ])
    if len(rows) == 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb())
        return
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    buf.seek(0)
    await tg_app.bot.send_document(
        STATE.chat_id,
        document=io.BytesIO(buf.getvalue().encode("utf-8")),
        filename="trades_hybrid_momo_v52.csv",
        caption="Export CSV"
    )

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

# -----------------------------
# FastAPI / Render
# -----------------------------
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    await tg_app.initialize()
    await tg_app.start()
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
    return (
        f"MP Bot v52 Hybrid Momo OK | "
        f"engine_on={STATE.engine_on} | thr={STATE.threshold:.2f} | pnl_total={total:+.4f}"
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
        "loss_guard_on": STATE.loss_guard_on,
        "loss_guard_n": STATE.loss_guard_n,
        "loss_guard_pause_m": STATE.loss_guard_pause_m,
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
