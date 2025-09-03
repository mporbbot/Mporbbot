# main_v44.py
# ------------------------------------------------------------
# MP Bot – v44 (MOCK + LIVE + K4 + AI-sparning + Watchdog)
# - Reply-keyboard (inga inline-knappar)
# - Mock & KuCoin Spot LIVE (bara LONG i live)
# - Grid/DCA + TP + Trailing + DD-stop
# - Enkel AI-score som kan sparas/laddas
# - /symbols add/remove/list
# - /export_k4 (CSV för Skatteverket K4, USDT -> SEK via env om satt)
# - Stabil loop med retries/timeout + watchdog och /kick
# - FastAPI webhook (Render)
# ------------------------------------------------------------
import os, io, csv, math, pickle, asyncio, contextlib, traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ---- KuCoin client (LIVE) ----
from kucoin.client import Client as KuClient

# -----------------------------
# ENV / SETTINGS
# -----------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
                   .replace(" ", "")).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))

GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.6"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.25"))
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.05"))

DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "2.0"))

# Trailing (låter vinsterna rinna)
TREND_ON = os.getenv("TREND_ON", "true").lower() in ("1", "true", "yes")
TREND_TRIGGER = float(os.getenv("TREND_TRIGGER_PCT", "0.9"))  # aktivera trail efter +0.9% mot snitt
TREND_TRAIL = float(os.getenv("TREND_TRAIL_PCT", "0.25"))     # trail-avstånd i %
TREND_BE = float(os.getenv("TREND_BE_PCT", "0.2"))            # flytta SL till BE efter +0.2% vinst

# AI-modell
AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "./data/ai_model.pkl")
os.makedirs(os.path.dirname(AI_MODEL_PATH) or ".", exist_ok=True)

# K4 valutakurs (frivillig). Om satt -> extra kolumn i SEK
K4_USDTSEK = os.getenv("K4_USDTSEK")
K4_USDTSEK = float(K4_USDTSEK) if K4_USDTSEK not in (None, "") else None

# KuCoin
KUCOIN_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KUCOIN_SANDBOX = os.getenv("KUCOIN_SANDBOX", "false").lower() in ("1", "true", "yes")

# Data
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# Global httpx-klient
HTTP: Optional[httpx.AsyncClient] = None
def _mk_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=3, read=6, write=6, pool=6),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
        headers={"User-Agent": "mp-bot/44"}
    )

# -----------------------------
# STATE
# -----------------------------
@dataclass
class TradeLeg:
    side: str
    price: float
    qty: float
    time: datetime

@dataclass
class Position:
    side: str                       # LONG (live: endast LONG)
    legs: List[TradeLeg] = field(default_factory=list)
    avg_price: float = 0.0
    target_price: float = 0.0
    safety_count: int = 0
    # trailing
    trail_active: bool = False
    trail_anchor: Optional[float] = None

    def qty_total(self) -> float:
        return sum(l.qty for l in self.legs)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)   # för CSV/K4
    next_step_pct: float = GRID_STEP_MIN_PCT

@dataclass
class EngineState:
    engine_on: bool = False
    mode_live: bool = False           # False=mock, True=live
    ai_on: bool = True
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
    per_sym: Dict[str, SymState] = field(default_factory=dict)
    position_size: float = POSITION_SIZE_USDT
    fee_side: float = FEE_PER_SIDE
    grid_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        max_safety=GRID_MAX_SAFETY,
        step_mult=GRID_STEP_MULT,
        step_min=GRID_STEP_MIN_PCT,
        size_mult=GRID_SIZE_MULT,
        tp=GRID_TP_PCT,
        tp_bonus=TP_SAFETY_BONUS,
    ))
    risk_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        dd=DD_STOP_PCT,
        max_pos=MAX_OPEN_POS,
        allow_shorts=False   # live spot: nej
    ))
    trend_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        on=TREND_ON, trigger=TREND_TRIGGER, trail=TREND_TRAIL, be=TREND_BE
    ))
    chat_id: Optional[int] = None
    # loop/watchdog
    last_loop_at: datetime = datetime.now(timezone.utc)
    engine_task: Optional[asyncio.Task] = None
    watchdog_task: Optional[asyncio.Task] = None
    heartbeat_task: Optional[asyncio.Task] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# AI
AI_WEIGHTS: Dict[Tuple[str, str], List[float]] = {}
AI_LEARN_RATE = 0.05

# KuCoin client (sync, körs via to_thread)
KU: Optional[KuClient] = None

# -----------------------------
# UI
# -----------------------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/start_mock"), KeyboardButton("/start_live")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/symbols")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/ai_on"), KeyboardButton("/ai_off")],
        [KeyboardButton("/save_ai"), KeyboardButton("/load_ai")],
        [KeyboardButton("/pnl"), KeyboardButton("/export_csv")],
        [KeyboardButton("/export_k4"), KeyboardButton("/panic")],
        [KeyboardButton("/reset_pnl"), KeyboardButton("/kick")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -----------------------------
# HELPERS
# -----------------------------
def _ensure_model_dir():
    d = os.path.dirname(AI_MODEL_PATH) or "."
    os.makedirs(d, exist_ok=True)

def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# -----------------------------
# DATA (KuCoin REST) + retries
# -----------------------------
async def _fetch_with_retries(url: str, params: dict, retries: int = 4, timeout_s: float = 7.0):
    assert HTTP is not None, "HTTP client not initialized"
    delay = 0.6
    last_exc = None
    for _ in range(retries):
        try:
            r = await HTTP.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            await asyncio.sleep(delay)
            delay *= 1.7
    raise last_exc or RuntimeError("network error")

async def get_klines(symbol: str, tf: str, limit: int = 60):
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    data = await _fetch_with_retries(KUCOIN_KLINES_URL, params, retries=4, timeout_s=7.0)
    rows = list(reversed(data["data"]))[-limit:]
    out = []
    for row in rows:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

# -----------------------------
# INDICATORS
# -----------------------------
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    k = 2 / (period + 1)
    out = []
    ema_val = series[0]
    for x in series:
        ema_val = (x - ema_val) * k + ema_val
        out.append(ema_val)
    return out

def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0.0))
        losses.append(-min(ch, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [50.0]*(period)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss else 999
        rsis.append(100 - (100/(1+rs)))
    return [50.0] + rsis

def features_from_candles(candles):
    closes = [c[-1] for c in candles]
    if len(closes) < 30:
        return {"mom": 0.0, "ema_fast": 0.0, "ema_slow": 0.0, "rsi": 50.0, "atrp": 0.2}
    ema_fast = ema(closes, 9)[-1]
    ema_slow = ema(closes, 21)[-1]
    mom = (closes[-1] - closes[-6]) / closes[-6] * 100.0 if closes[-6] else 0.0
    rsi_last = rsi(closes, 14)[-1]
    rng = [(c[2]-c[3]) / c[-1] * 100.0 if c[-1] else 0.0 for c in candles[-20:]]
    atrp = sum(rng)/len(rng) if rng else 0.2
    return {"mom": mom, "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": rsi_last, "atrp": max(0.02, min(3.0, atrp))}

# -----------------------------
# AI scorer
# -----------------------------
def _wkey(symbol: str, tf: str) -> Tuple[str, str]:
    return (symbol, tf)

def ai_score(symbol: str, tf: str, feats: Dict[str, float]) -> float:
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        AI_WEIGHTS[key] = [0.0, 0.7, 0.6, 0.1]
    w0, w_mom, w_ema, w_rsi = AI_WEIGHTS[key]
    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
    rsi_dev = (feats["rsi"] - 50.0) / 50.0
    raw = w0 + w_mom*feats["mom"] + w_ema*ema_diff + w_rsi*rsi_dev*100.0
    return max(-10, min(10, raw))

def ai_learn(symbol: str, tf: str, feats: Dict[str, float], pnl_net: float):
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        AI_WEIGHTS[key] = [0.0, 0.7, 0.6, 0.1]
    w = AI_WEIGHTS[key]
    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
    rsi_dev = (feats["rsi"] - 50.0) / 50.0
    grad = [1.0, feats["mom"], ema_diff, rsi_dev*100.0]
    lr = AI_LEARN_RATE * (1 if pnl_net >= 0 else -1)
    AI_WEIGHTS[key] = [w[i] + lr*grad[i] for i in range(4)]

def save_ai() -> str:
    try:
        _ensure_model_dir()
        with open(AI_MODEL_PATH, "wb") as f:
            pickle.dump(AI_WEIGHTS, f)
        return f"AI sparad: {AI_MODEL_PATH}"
    except Exception as e:
        return f"Fel vid sparning: {e}"

def load_ai() -> str:
    global AI_WEIGHTS
    try:
        _ensure_model_dir()
        with open(AI_MODEL_PATH, "rb") as f:
            AI_WEIGHTS = pickle.load(f)
        return f"AI laddad från: {AI_MODEL_PATH}"
    except Exception as e:
        return f"Kunde inte ladda AI: {e}"

# -----------------------------
# TRADING (mock + kucoin live)
# -----------------------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

def _enter_leg_mock(sym: str, side: str, price: float, usd_size: float, st: SymState) -> TradeLeg:
    qty = usd_size / price if price > 0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=now_utc())
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0)
        st.next_step_pct = STATE.grid_cfg["step_min"]
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        total_qty = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total_qty or 1.0)
    # TP
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
    if side == "LONG":
        st.pos.target_price = st.pos.avg_price * (1.0 + tp/100.0)
    else:
        st.pos.target_price = st.pos.avg_price * (1.0 - tp/100.0)
    return leg

async def _enter_leg_live(sym: str, usd_size: float, price_hint: float) -> TradeLeg:
    """Market BUY på KuCoin Spot med 'funds' i USDT. Returnerar approx leg."""
    global KU
    assert KU is not None, "KuCoin client not initialized"
    order = await asyncio.to_thread(KU.create_market_order, sym, 'buy', funds=f"{usd_size:.6f}")
    qty = usd_size / price_hint if price_hint > 0 else 0.0
    return TradeLeg(side="LONG", price=price_hint, qty=qty, time=now_utc())

def _activate_trailing_if_needed(st: SymState, price: float):
    if not STATE.trend_cfg["on"] or not st.pos:
        return
    avg = st.pos.avg_price
    up = (price - avg) / avg * 100.0
    if not st.pos.trail_active and up >= STATE.trend_cfg["trigger"]:
        st.pos.trail_active = True
        st.pos.trail_anchor = price
    if st.pos.trail_active:
        st.pos.trail_anchor = max(st.pos.trail_anchor or price, price)

def _trail_stop_hit(st: SymState, price: float) -> bool:
    if not STATE.trend_cfg["on"] or not st.pos or not st.pos.trail_active:
        return False
    anchor = st.pos.trail_anchor or price
    trail_pct = STATE.trend_cfg["trail"]
    be_pct = STATE.trend_cfg["be"]
    avg = st.pos.avg_price
    be_price = avg * (1 + be_pct/100.0)
    trail_price = anchor * (1 - trail_pct/100.0)
    stop_price = max(be_price, trail_price)
    return price <= stop_price

def _exit_all_mock(sym: str, price: float, st: SymState) -> float:
    if not st.pos: return 0.0
    gross = 0.0; fee_in = 0.0
    for leg in st.pos.legs:
        usd_in = leg.qty * leg.price
        fee_in += _fee(usd_in)
        gross += leg.qty * (price - leg.price)  # LONG
    usd_out = st.pos.qty_total() * price
    fee_out = _fee(usd_out)
    net = gross - fee_in - fee_out
    st.realized_pnl_net += net
    st.trades_log.append({
        "time": now_utc().isoformat(),
        "symbol": sym, "side": st.pos.side, "avg_price": st.pos.avg_price,
        "exit_price": price, "gross": round(gross,6),
        "fee_in": round(fee_in,6), "fee_out": round(fee_out,6),
        "net": round(net,6), "qty": round(st.pos.qty_total(), 10), "mode": "MOCK",
        "legs": len(st.pos.legs)
    })
    st.pos = None
    st.next_step_pct = STATE.grid_cfg["step_min"]
    return net

async def _exit_all_live(sym: str, price: float, st: SymState) -> float:
    global KU
    assert KU is not None and st.pos, "KuCoin client/pos saknas"
    qty = st.pos.qty_total()
    order = await asyncio.to_thread(KU.create_market_order, sym, 'sell', size=f"{qty:.9f}")
    return _exit_all_mock(sym, price, st)

# -----------------------------
# BESLUT & ENGINE
# -----------------------------
def score_vote(symbol: str, feats_per_tf: Dict[str, Dict[str, float]]) -> float:
    votes = 0.0
    for tf, feats in feats_per_tf.items():
        sc = ai_score(symbol, tf, feats) if STATE.ai_on else 0.0
        ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
        bias = (1.0 if ema_diff > 0 else -1.0) * min(1.0, abs(ema_diff)/1.5)
        votes += sc/10.0 + bias
    return votes

async def _engine_once(app: Application):
    open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
    if len(open_syms) < STATE.risk_cfg["max_pos"]:
        for sym in STATE.symbols:
            if STATE.per_sym[sym].pos:
                continue
            feats_per_tf = {}
            skip = False
            for tf in STATE.tfs:
                try:
                    kl = await get_klines(sym, tf, limit=60)
                except Exception:
                    skip = True; break
                feats_per_tf[tf] = features_from_candles(kl)
            if skip or not feats_per_tf:
                continue
            score = score_vote(sym, feats_per_tf)
            if score > 0.35:  # LONG
                st = STATE.per_sym[sym]
                last_px = (await get_klines(sym, STATE.tfs[0], limit=1))[-1][-1]
                if STATE.mode_live:
                    leg = await _enter_leg_live(sym, STATE.position_size, last_px)
                else:
                    leg = _enter_leg_mock(sym, "LONG", last_px, STATE.position_size, st)
                if STATE.mode_live:
                    if st.pos is None:
                        st.pos = Position(side="LONG", legs=[leg], avg_price=leg.price)
                        st.next_step_pct = STATE.grid_cfg["step_min"]
                    else:
                        st.pos.legs.append(leg)
                        st.pos.safety_count += 1
                        total_qty = st.pos.qty_total()
                        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total_qty or 1.0)
                    tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
                    st.pos.target_price = st.pos.avg_price * (1.0 + tp/100.0)

                if STATE.chat_id:
                    await app.bot.send_message(
                        STATE.chat_id,
                        f"🟢 ENTRY {sym} LONG @ {st.pos.legs[-1].price:.4f} | TP {st.pos.target_price:.4f} | "
                        f"{'LIVE' if STATE.mode_live else 'MOCK'}"
                    )
                open_syms.append(sym)
                if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                    break

    for sym in STATE.symbols:
        st = STATE.per_sym[sym]
        if not st.pos:
            continue
        tf0 = STATE.tfs[0] if STATE.tfs else "1m"
        try:
            kl = await get_klines(sym, tf0, limit=3)
        except Exception:
            continue
        price = kl[-1][-1]
        avg = st.pos.avg_price
        move_pct = (price-avg)/avg*100.0

        _activate_trailing_if_needed(st, price)

        if price >= st.pos.target_price:
            net = await (_exit_all_live(sym, price, st) if STATE.mode_live else asyncio.to_thread(_exit_all_mock, sym, price, st))
            if STATE.chat_id:
                mark = "✅" if net >= 0 else "❌"
                await app.bot.send_message(STATE.chat_id, f"🟤 EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}")
            feats = features_from_candles(kl)
            if STATE.ai_on:
                ai_learn(sym, tf0, feats, net)
            continue

        if _trail_stop_hit(st, price):
            net = await (_exit_all_live(sym, price, st) if STATE.mode_live else asyncio.to_thread(_exit_all_mock, sym, price, st))
            if STATE.chat_id:
                await app.bot.send_message(STATE.chat_id, f"🔒 TRAIL STOP {sym} @ {price:.4f} | Net {net:+.4f} USDT")
            feats = features_from_candles(kl)
            if STATE.ai_on:
                ai_learn(sym, tf0, feats, net)
            continue

        step = st.next_step_pct
        if move_pct <= -step and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
            usd = STATE.position_size * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
            if STATE.mode_live:
                leg = await _enter_leg_live(sym, usd, price)
                if st.pos is None:
                    st.pos = Position(side="LONG", legs=[leg], avg_price=leg.price)
                    st.next_step_pct = STATE.grid_cfg["step_min"]
                else:
                    st.pos.legs.append(leg)
                    st.pos.safety_count += 1
                    total_qty = st.pos.qty_total()
                    st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total_qty or 1.0)
                tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
                st.pos.target_price = st.pos.avg_price * (1.0 + tp/100.0)
            else:
                _enter_leg_mock(sym, "LONG", price, usd, st)
            if STATE.chat_id:
                await app.bot.send_message(
                    STATE.chat_id, f"🧩 DCA {sym} LONG @ {price:.4f} (leg {st.pos.safety_count}) | "
                                   f"{'LIVE' if STATE.mode_live else 'MOCK'}"
                )
            st.next_step_pct = max(STATE.grid_cfg["step_min"], st.next_step_pct * STATE.grid_cfg["step_mult"])

        if abs(move_pct) >= STATE.risk_cfg["dd"]:
            net = await (_exit_all_live(sym, price, st) if STATE.mode_live else asyncio.to_thread(_exit_all_mock, sym, price, st))
            if STATE.chat_id:
                mark = "✅" if net >= 0 else "❌"
                await app.bot.send_message(STATE.chat_id, f"⛔ STOP {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark} (DD)")

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                await asyncio.wait_for(_engine_once(app), timeout=25.0)
                STATE.last_loop_at = datetime.now(timezone.utc)
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            STATE.last_loop_at = datetime.now(timezone.utc)
        except Exception as e:
            if STATE.chat_id:
                with contextlib.suppress(Exception):
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
            await asyncio.sleep(5)

async def watchdog_loop(app: Application):
    while True:
        try:
            await asyncio.sleep(30)
            if not STATE.engine_on:
                continue
            stale = datetime.now(timezone.utc) - STATE.last_loop_at
            if stale > timedelta(minutes=2):
                if STATE.engine_task and not STATE.engine_task.done():
                    STATE.engine_task.cancel()
                    with contextlib.suppress(Exception):
                        await STATE.engine_task
                STATE.engine_task = asyncio.create_task(engine_loop(app))
                if STATE.chat_id:
                    with contextlib.suppress(Exception):
                        await app.bot.send_message(STATE.chat_id, "🛠️ Watchdog: engine startad om.")
        except Exception:
            await asyncio.sleep(10)

async def heartbeat_loop(app: Application):
    while True:
        try:
            await asyncio.sleep(600)
            if STATE.engine_on and STATE.chat_id:
                total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
                await app.bot.send_message(STATE.chat_id, f"💓 Heartbeat | PnL(NET) {total:+.4f} USDT")
        except Exception:
            await asyncio.sleep(30)

# -----------------------------
# TELEGRAM
# -----------------------------
tg_app = Application.builder().token(BOT_TOKEN).build()

def status_text() -> str:
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            pos_lines.append(
                f"{s}: LONG avg {st.pos.avg_price:.4f} → TP {st.pos.target_price:.4f} | legs {st.pos.safety_count} "
                f"{'TRAIL' if st.pos.trail_active else ''}"
            )
    g = STATE.grid_cfg; r = STATE.risk_cfg; t = STATE.trend_cfg
    return "\n".join([
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'} | Mode: {'LIVE' if STATE.mode_live else 'MOCK'}",
        f"AI: {'ON 🧠' if STATE.ai_on else 'OFF'}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} USDT | Fee per sida: {STATE.fee_side:.4%}",
        (f"Grid: max_safety={g['max_safety']} step_mult={g['step_mult']} step_min%={g['step_min']} "
         f"size_mult={g['size_mult']} tp%={g['tp']} (+{g['tp_bonus']}%/safety)"),
        (f"Risk: dd={r['dd']}% | max_pos={r['max_pos']} | shorts={'ON' if r['allow_shorts'] else 'OFF'}"),
        (f"Trailing: {'ON' if t['on'] else 'OFF'} | trigger={t['trigger']}% | trail={t['trail']}% | BE={t['be']}%"),
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
        f"Heartbeat: {STATE.last_loop_at.isoformat(timespec='seconds')}"
    ])

async def send_status(chat_id: int):
    await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

# --- commands ---
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v44 – redo ✅", reply_markup=reply_kb())
    await send_status(STATE.chat_id)

async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_status(STATE.chat_id)

async def cmd_start_mock(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.mode_live = False
    await tg_app.bot.send_message(STATE.chat_id, "Mock-läge: AKTIV ✅", reply_markup=reply_kb())

async def cmd_start_live(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
        await tg_app.bot.send_message(STATE.chat_id, "❌ KuCoin API-nycklar saknas (KUCOIN_API_KEY/SECRET/PASSPHRASE).")
        return
    global KU
    KU = KuClient(KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE, sandbox=KUCOIN_SANDBOX)
    STATE.mode_live = True
    await tg_app.bot.send_message(STATE.chat_id, f"LIVE-läge: AKTIV ✅ (sandbox={KUCOIN_SANDBOX})", reply_markup=reply_kb())

async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    if STATE.engine_task is None or STATE.engine_task.done():
        STATE.engine_task = asyncio.create_task(engine_loop(tg_app))
    await tg_app.bot.send_message(STATE.chat_id, "Engine: ON ✅", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

async def cmd_kick(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    if STATE.engine_task and not STATE.engine_task.done():
        STATE.engine_task.cancel()
        with contextlib.suppress(Exception):
            await STATE.engine_task
    STATE.engine_task = asyncio.create_task(engine_loop(tg_app))
    await tg_app.bot.send_message(STATE.chat_id, "♻️ Engine omstartad.", reply_markup=reply_kb())

async def cmd_ai_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.ai_on = True
    await tg_app.bot.send_message(STATE.chat_id, "AI: ON 🧠", reply_markup=reply_kb())

async def cmd_ai_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.ai_on = False
    await tg_app.bot.send_message(STATE.chat_id, "AI: OFF", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip()
    parts = msg.split(" ", 1)
    if len(parts) == 2:
        tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.tfs)}",
                                          reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, "Använd: /timeframe 1m,3m,5m,15m", reply_markup=reply_kb())

async def cmd_grid(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
    toks = text.split()
    if len(toks) == 4 and toks[0] == "/grid" and toks[1] == "set":
        key, val = toks[2], toks[3]
        if key in STATE.grid_cfg:
            try:
                STATE.grid_cfg[key] = float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Grid uppdaterad.", reply_markup=reply_kb())
            except:
                await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
        else:
            await tg_app.bot.send_message(STATE.chat_id, f"Okänd grid-nyckel: {key}", reply_markup=reply_kb())
            return
    g = STATE.grid_cfg
    await tg_app.bot.send_message(STATE.chat_id,
        ("Grid:\n"
         f"  max_safety={g['max_safety']}\n"
         f"  step_mult={g['step_mult']}\n"
         f"  step_min%={g['step_min']}\n"
         f"  size_mult={g['size_mult']}\n"
         f"  tp%={g['tp']} (+{g['tp_bonus']}%/safety)\n\n"
         "Ex: /grid set step_mult 0.7"),
        reply_markup=reply_kb())

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
    toks = text.split()
    if len(toks) == 4 and toks[0] == "/risk" and toks[1] == "set":
        key, val = toks[2], toks[3]
        if key in STATE.risk_cfg:
            try:
                if key == "max_pos":
                    STATE.risk_cfg[key] = int(val)
                elif key == "allow_shorts":
                    STATE.risk_cfg[key] = (val.lower() in ("1", "true", "on", "yes"))
                else:
                    STATE.risk_cfg[key] = float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb())
            except:
                await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
        else:
            await tg_app.bot.send_message(STATE.chat_id, f"Okänd risk-nyckel: {key}", reply_markup=reply_kb())
            return
    r = STATE.risk_cfg; t = STATE.trend_cfg
    await tg_app.bot.send_message(STATE.chat_id,
        (f"Risk:\n  dd={r['dd']}%\n  max_pos={r['max_pos']}\n  shorts={'ON' if r['allow_shorts'] else 'OFF'}\n\n"
         f"Trailing: on={t['on']} trigger={t['trigger']}% trail={t['trail']}% be={t['be']}%\n"
         "Ex: /risk set dd 2.0"),
        reply_markup=reply_kb())

async def cmd_symbols(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = update.message.text.strip()
    toks = txt.split()
    if len(toks) >= 2 and toks[0] == "/symbols":
        if len(toks) == 2 and toks[1].lower() in ("list",):
            await tg_app.bot.send_message(STATE.chat_id, "Symbols: " + ", ".join(STATE.symbols), reply_markup=reply_kb())
            return
        if len(toks) >= 3:
            action = toks[1].lower()
            sym = toks[2].upper()
            if action == "add":
                if sym not in STATE.symbols:
                    STATE.symbols.append(sym)
                    STATE.per_sym[sym] = SymState()
                    await tg_app.bot.send_message(STATE.chat_id, f"La till: {sym}", reply_markup=reply_kb())
                else:
                    await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns redan.", reply_markup=reply_kb())
                return
            if action == "remove":
                if sym in STATE.symbols:
                    STATE.symbols = [s for s in STATE.symbols if s != sym]
                    STATE.per_sym.pop(sym, None)
                    await tg_app.bot.send_message(STATE.chat_id, f"Tog bort: {sym}", reply_markup=reply_kb())
                else:
                    await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns ej.", reply_markup=reply_kb())
                return
    await tg_app.bot.send_message(STATE.chat_id,
        ("Symbols: " + ", ".join(STATE.symbols) +
         "\n/lista: /symbols list\n"
         "Lägg till/ta bort:\n  /symbols add LINK-USDT\n  /symbols remove LINK-USDT"),
        reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    lines = [f"📈 PnL total (NET): {total:+.4f} USDT"]
    for s in STATE.symbols:
        lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl_net:+.4f} USDT")
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    rows = [["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","qty","mode","legs"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r["time"], r["symbol"], r["side"], r["avg_price"], r["exit_price"],
                         r["gross"], r["fee_in"], r["fee_out"], r["net"], r["qty"], r["mode"], r["legs"]])
    if len(rows) == 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb())
        return
    buf = io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="trades.csv", caption="Export CSV")

async def cmd_export_k4(update: Update, _):
    """Skapar enkel K4-CSV. Antaganden:
       - Försäljningspris = qty * exit_price - fee_out
       - Omkostnadsbelopp = qty * avg_price + fee_in
       - Resultat = Försäljningspris - Omkostnadsbelopp
       - Valuta: USDT (extra SEK-kolumn om K4_USDTSEK är satt)
    """
    STATE.chat_id = update.effective_chat.id
    rows = [["datum","värdepapper","antal","försäljningspris_usdt","omkostnadsbelopp_usdt","avgifter_usdt","resultat_usdt"] +
            (["resultat_sek"] if K4_USDTSEK else [])]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            if r.get("side") != "LONG":  # vi loggar bara LONG i v44 (live)
                continue
            qty = float(r["qty"])
            exit_price = float(r["exit_price"])
            avg_price = float(r["avg_price"])
            fee_in = float(r["fee_in"]); fee_out = float(r["fee_out"])
            sales = qty * exit_price - fee_out
            cost = qty * avg_price + fee_in
            result = sales - cost
            cols = [r["time"][:10], r["symbol"], f"{qty:.8f}", f"{sales:.6f}", f"{cost:.6f}", f"{(fee_in+fee_out):.6f}", f"{result:.6f}"]
            if K4_USDTSEK:
                cols.append(f"{result * K4_USDTSEK:.2f}")
            rows.append(cols)
    if len(rows) == 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga stängda affärer ännu.", reply_markup=reply_kb()); return
    buf = io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="k4_export.csv", caption="K4-export (USDT)")

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            tf0 = STATE.tfs[0] if STATE.tfs else "1m"
            try:
                px = (await get_klines(s, tf0, limit=1))[-1][-1]
            except:
                continue
            if STATE.mode_live:
                net = await _exit_all_live(s, px, st)
            else:
                net = _exit_all_mock(s, px, st)
            closed.append(f"{s}:{net:+.4f}")
    msg = " | ".join(closed) if closed else "Inga positioner."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl_net = 0.0
        STATE.per_sym[s].trades_log.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL + loggar återställda.", reply_markup=reply_kb())

async def cmd_save_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, save_ai(), reply_markup=reply_kb())

async def cmd_load_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, load_ai(), reply_markup=reply_kb())

# registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("start_mock", cmd_start_mock))
tg_app.add_handler(CommandHandler("start_live", cmd_start_live))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("kick", cmd_kick))
tg_app.add_handler(CommandHandler("ai_on", cmd_ai_on))
tg_app.add_handler(CommandHandler("ai_off", cmd_ai_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("export_k4", cmd_export_k4))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
tg_app.add_handler(CommandHandler("load_ai", cmd_load_ai))

# -----------------------------
# FASTAPI WEBHOOK
# -----------------------------
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    global HTTP
    HTTP = _mk_client()
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    await tg_app.initialize()
    await tg_app.start()
    # starta loopar
    STATE.engine_task = asyncio.create_task(engine_loop(tg_app))
    STATE.watchdog_task = asyncio.create_task(watchdog_loop(tg_app))
    STATE.heartbeat_task = asyncio.create_task(heartbeat_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    for t in (STATE.engine_task, STATE.watchdog_task, STATE.heartbeat_task):
        if t and not t.done():
            t.cancel()
            with contextlib.suppress(Exception):
                await t
    await tg_app.stop()
    await tg_app.shutdown()
    global HTTP
    if HTTP is not None:
        with contextlib.suppress(Exception):
            await HTTP.aclose()
        HTTP = None

@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return f"MP Bot v44 OK | engine_on={STATE.engine_on} | mode={'LIVE' if STATE.mode_live else 'MOCK'} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "mode_live": STATE.mode_live,
        "tfs": STATE.tfs,
        "symbols": STATE.symbols,
        "last_loop_at": STATE.last_loop_at.isoformat(),
        "pnl_total": round(total, 6)
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
