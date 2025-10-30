# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MP Bot – v45 (Confirmed Momentum Hybrid - "premium")
# ------------------------------------------------------------
# - Bekräftad momentum-entry (confirm_mode) + trendfilter
# - Hybrid TP→Trail (sen aktivering) + bred trailing
# - Failsafes: early_cut / sl_max / max_dd_per_trade
# - Mock-handel, CSV-export (/export_csv), Backtest (/backtest)
# - Persistenta mock-loggar till CSV (mock_trade_log.csv)
# - Render-vänlig Telegram-start (await initialize/start)
# ------------------------------------------------------------

import os, io, csv, pickle, asyncio, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes

# -----------------------------
# ENV / SETTINGS
# -----------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
                   .replace(" ", "")).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "5m,15m,30m").replace(" ", "")).split(",")

# Storlek & avgifter
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "80"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))

# Risk / slots
MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "2.0"))

# Grid/DCA
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "2"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.50"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "1.2"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.2"))
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.55"))
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.03"))

# Entry-tröskel (AI/score)
ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "1.85"))

# Trailing / Hybrid
TRAIL_ON = (os.getenv("TRAIL_ON", "true").lower() in ("1","true","on","yes"))
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "7.0"))
TRAIL_MIN_PCT = float(os.getenv("TRAIL_MIN_PCT", "3.0"))
HYB_ON = (os.getenv("HYB_ON", "true").lower() in ("1","true","on","yes"))
HYB_START_PCT = float(os.getenv("HYB_START_PCT", "2.5"))
HYB_LOCK_PCT = float(os.getenv("HYB_LOCK_PCT", "0.25"))

# Failsafes
SL_MAX_PCT = float(os.getenv("SL_MAX_PCT", "1.10"))
EARLY_CUT_ON = (os.getenv("EARLY_CUT_ON", "false").lower() in ("1","true","on","yes"))
EARLY_CUT_THR = float(os.getenv("EARLY_CUT_THR", "0.55"))
EARLY_CUT_WIN_M = int(os.getenv("EARLY_CUT_WINDOW_M", "3"))
MAX_DD_TRADE_USDT = float(os.getenv("MAX_DD_TRADE_USDT", "3.0"))

# Confirmed Momentum + Trendfilter
CONFIRM_MODE = (os.getenv("CONFIRM_MODE", "true").lower() in ("1","true","on","yes"))
CONFIRM_CANDLES = int(os.getenv("CONFIRM_CANDLES", "2"))
CONFIRM_STRENGTH = float(os.getenv("CONFIRM_STRENGTH", "0.4"))
TREND_FILTER = (os.getenv("TREND_FILTER", "true").lower() in ("1","true","on","yes"))
TREND_LEN = int(os.getenv("TREND_LEN", "30"))
TREND_BIAS = float(os.getenv("TREND_BIAS", "0.6"))

AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "ai_model.pkl")

# KuCoin candles
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# -----------------------------
# STATE STRUCTURES
# -----------------------------
@dataclass
class TradeLeg:
    side: str
    price: float
    qty: float
    time: datetime

@dataclass
class Position:
    side: str  # "LONG" eller "SHORT"
    legs: List[TradeLeg] = field(default_factory=list)
    avg_price: float = 0.0
    target_price: float = 0.0
    safety_count: int = 0
    high_water: float = 0.0
    low_water: float = 0.0
    trailing_active: bool = False
    lock_price: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    def qty_total(self) -> float:
        return sum(l.qty for l in self.legs)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)
    last_signal_ts: Optional[int] = None
    next_step_pct: float = GRID_STEP_MIN_PCT

@dataclass
class EngineState:
    engine_on: bool = False
    ai_on: bool = True
    entry_threshold: float = ENTRY_THRESHOLD
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
    per_sym: Dict[str, SymState] = field(default_factory=dict)
    position_size: float = POSITION_SIZE_USDT
    fee_side: float = FEE_PER_SIDE
    grid_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        max_safety=GRID_MAX_SAFETY, step_mult=GRID_STEP_MULT, step_min=GRID_STEP_MIN_PCT,
        size_mult=GRID_SIZE_MULT, tp=GRID_TP_PCT, tp_bonus=TP_SAFETY_BONUS))
    risk_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        dd=DD_STOP_PCT, max_pos=MAX_OPEN_POS, allow_shorts=False,
        size=POSITION_SIZE_USDT,
        trail_on=TRAIL_ON, trail_mult=TRAIL_ATR_MULT, trail_min=TRAIL_MIN_PCT,
        hyb_on=HYB_ON, hyb_start=HYB_START_PCT, hyb_lock=HYB_LOCK_PCT,
        sl_max=SL_MAX_PCT, early_cut=EARLY_CUT_ON,
        early_cut_thr=EARLY_CUT_THR, early_cut_window_m=EARLY_CUT_WIN_M,
        max_dd_per_trade=MAX_DD_TRADE_USDT,
        confirm_mode=CONFIRM_MODE, confirm_candle=CONFIRM_CANDLES, confirm_strength=CONFIRM_STRENGTH,
        trend_filter=TREND_FILTER, trend_len=TREND_LEN, trend_bias=TREND_BIAS
    ))
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# AI-vikter
AI_WEIGHTS: Dict[Tuple[str, str], List[float]] = {}  # bias, mom, ema_diff, rsi_dev
AI_LEARN_RATE = 0.05

# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def log_mock_trade(row: Dict):
    """Append till mock_trade_log.csv (för Skatteverket)."""
    fname = "mock_trade_log.csv"
    newfile = not os.path.exists(fname)
    ensure_dir(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time","symbol","action","side","price","qty","avg_price","gross","fee_in","fee_out","net","info"])
        if newfile:
            w.writeheader()
        w.writerow(row)

# -----------------------------
# INDICATORS & FEATURES
# -----------------------------
async def fetch_klines(symbol: str, tf: str, limit: int = 80) -> List[List[str]]:
    tf_api = TF_MAP.get(tf, tf)
    params = {"symbol": symbol, "type": tf_api}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
    return data[::-1]  # senaste sist

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1: return series[:]
    k = 2.0 / (period + 1.0)
    out = []
    val = series[0]
    for x in series:
        val = (x - val) * k + val
        out.append(val)
    return out

def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1: return [50.0] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0.0)); losses.append(-min(ch, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [50.0] * period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else 999.0
        val = 100.0 - (100.0/(1.0+rs))
        rsis.append(val)
    return [50.0] + rsis

def features_from_candles(candles: List[List[str]]) -> Dict[str, float]:
    closes = [float(c[2]) for c in candles]
    if len(closes) < 30:
        return {"mom":0.0,"ema_fast":0.0,"ema_slow":0.0,"rsi":50.0,"atrp":0.25}
    ema9 = ema(closes, 9); ema21 = ema(closes, 21)
    mom = (closes[-1]-closes[-6])/(closes[-6] or 1)*100.0
    rsi_last = rsi(closes, 14)[-1]
    rng = []
    for c in candles[-20:]:
        high = float(c[3]); low = float(c[4]); close = float(c[2])
        rng.append((high-low)/(close or 1)*100.0)
    atrp = sum(rng)/len(rng) if rng else 0.25
    return {"mom":mom,"ema_fast":ema9[-1],"ema_slow":ema21[-1],"rsi":rsi_last,"atrp":max(0.05,min(4.0,atrp))}

# -----------------------------
# AI SCORE (enkel)
# -----------------------------
def _wkey(symbol: str, tf: str) -> Tuple[str, str]:
    return (symbol, tf)

def ai_score(symbol: str, tf: str, feats: Dict[str,float]) -> float:
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        AI_WEIGHTS[key] = [0.0, 0.7, 0.6, 0.1]
    w0, w_mom, w_ema, w_rsi = AI_WEIGHTS[key]
    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
    rsi_dev = (feats["rsi"] - 50.0) / 50.0
    raw = w0 + w_mom*feats["mom"] + w_ema*ema_diff + w_rsi*rsi_dev*100.0
    return max(-10.0, min(10.0, raw))

def ai_learn(symbol: str, tf: str, feats: Dict[str,float], pnl_net: float):
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        AI_WEIGHTS[key] = [0.0, 0.7, 0.6, 0.1]
    w = AI_WEIGHTS[key]
    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
    grad = [1.0, feats["mom"], ema_diff, (feats["rsi"]-50.0)]
    lr = AI_LEARN_RATE * (1 if pnl_net >= 0 else -1)
    AI_WEIGHTS[key] = [w[i] + lr*grad[i] for i in range(4)]

# -----------------------------
# ENTRY/EXIT HELPERS
# -----------------------------
def _trend_bias_from_tf(feats_hi: Dict[str,float], risk: Dict[str,float]) -> float:
    ema_diff_hi = (feats_hi["ema_fast"] - feats_hi["ema_slow"]) / (feats_hi["ema_slow"] or 1.0) * 100.0
    bias = float(risk.get("trend_bias", TREND_BIAS))
    return bias * (1.0 if ema_diff_hi > 0 else -1.0)

def _confirmed_move(closes: List[float], strength_pct: float, direction_long: bool, candles: int) -> bool:
    if len(closes) < candles+1: return False
    ref = closes[-(candles+1)]
    if direction_long:
        return (max(closes[-candles:]) - ref) / (ref or 1) * 100.0 >= strength_pct
    else:
        return (ref - min(closes[-candles:])) / (ref or 1) * 100.0 >= strength_pct

# -----------------------------
# TRADING HELPERS (mock)
# -----------------------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

def _enter_leg(sym: str, side: str, price: float, usd_size: float, st: SymState) -> TradeLeg:
    qty = usd_size / price if price > 0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0)
        st.next_step_pct = STATE.grid_cfg["step_min"]
        st.pos.high_water = price; st.pos.low_water = price
        st.pos.trailing_active = False; st.pos.lock_price = 0.0
        st.pos.opened_at = datetime.now(timezone.utc)
    else:
        st.pos.legs.append(leg); st.pos.safety_count += 1
        total_qty = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total_qty or 1.0)
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
    st.pos.target_price = (st.pos.avg_price * (1.0 + tp/100.0) if side=="LONG"
                           else st.pos.avg_price * (1.0 - tp/100.0))
    # log
    log_mock_trade({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym, "action": "ENTRY", "side": side, "price": round(price,6),
        "qty": round(qty,8), "avg_price": round(st.pos.avg_price,6),
        "gross": "", "fee_in": "", "fee_out": "", "net": "", "info": f"size_usdt={usd_size}"
    })
    return leg

def _exit_all(sym: str, price: float, st: SymState) -> float:
    if not st.pos: return 0.0
    gross = 0.0; fee_in = 0.0
    for leg in st.pos.legs:
        usd_in = leg.qty * leg.price
        fee_in += _fee(usd_in)
        if st.pos.side == "LONG":
            gross += leg.qty * (price - leg.price)
        else:
            gross += leg.qty * (leg.price - price)
    usd_out = st.pos.qty_total() * price
    fee_out = _fee(usd_out)
    net = gross - fee_in - fee_out
    st.realized_pnl_net += net
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym, "side": st.pos.side, "avg_price": st.pos.avg_price,
        "exit_price": price, "gross": round(gross,6), "fee_in": round(fee_in,6),
        "fee_out": round(fee_out,6), "net": round(net,6), "safety_legs": st.pos.safety_count
    })
    # log
    log_mock_trade({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym, "action": "EXIT", "side": st.pos.side, "price": round(price,6),
        "qty": round(st.pos.qty_total(),8), "avg_price": round(st.pos.avg_price,6),
        "gross": round(gross,6), "fee_in": round(fee_in,6), "fee_out": round(fee_out,6),
        "net": round(net,6), "info": f"safety_legs={st.pos.safety_count}"
    })
    st.pos = None
    st.next_step_pct = STATE.grid_cfg["step_min"]
    return net

# -----------------------------
# ENGINE LOOP (entries/exits)
# -----------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                # ===== ENTRIES =====
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < int(STATE.risk_cfg["max_pos"]):
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos: continue
                        feats_per_tf = {}; klines_per_tf = {}; last_price=None; skip=False
                        for tf in STATE.tfs:
                            try:
                                kl = await fetch_klines(sym, tf, 80)
                                last_price = float(kl[-1][2])
                            except Exception:
                                skip=True; break
                            klines_per_tf[tf] = kl
                            feats_per_tf[tf] = features_from_candles(kl)
                        if skip or not feats_per_tf or last_price is None: continue

                        # AI score + EMA-bias
                        score = 0.0
                        for tf, f in feats_per_tf.items():
                            sc = ai_score(sym, tf, f) if STATE.ai_on else 0.0
                            ema_diff = (f["ema_fast"] - f["ema_slow"]) / (f["ema_slow"] or 1.0) * 100.0
                            bias = (1.0 if ema_diff > 0 else -1.0) * min(1.0, abs(ema_diff)/1.5)
                            score += sc/10.0 + bias

                        risk = STATE.risk_cfg
                        th = float(STATE.entry_threshold)
                        allow_shorts = bool(risk.get("allow_shorts", False))

                        # Trendfilter (högsta TF)
                        if risk.get("trend_filter", True):
                            tf_hi = STATE.tfs[-1]
                            hi = feats_per_tf[tf_hi]
                            trend_bias = _trend_bias_from_tf(hi, risk)
                            score += trend_bias

                        do_long = score > th
                        do_short = (score < -th) and allow_shorts

                        # Confirmed momentum på minsta TF
                        if risk.get("confirm_mode", True):
                            confirm_n = int(risk.get("confirm_candle", 2))
                            confirm_str = float(risk.get("confirm_strength", 0.4))
                            tf0 = STATE.tfs[0] if STATE.tfs else "5m"
                            closes = [float(c[-1]) for c in klines_per_tf[tf0]]
                            if do_long:
                                do_long = _confirmed_move(closes, confirm_str, True, confirm_n)
                            if do_short:
                                do_short = _confirmed_move(closes, confirm_str, False, confirm_n)

                        if do_long:
                            st = STATE.per_sym[sym]
                            usd = float(risk.get("size", STATE.position_size))
                            _enter_leg(sym, "LONG", last_price, usd, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {last_price:.4f} | score={score:.2f} | thr={th:.2f} | TP {st.pos.target_price:.4f}")
                            open_syms.append(sym)
                            if len(open_syms) >= int(risk["max_pos"]): break

                        elif do_short:
                            st = STATE.per_sym[sym]
                            usd = float(risk.get("size", STATE.position_size))
                            _enter_leg(sym, "SHORT", last_price, usd, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔻 ENTRY {sym} SHORT @ {last_price:.4f} | score={score:.2f} | thr={-th:.2f} | TP {st.pos.target_price:.4f}")
                            open_syms.append(sym)
                            if len(open_syms) >= int(risk["max_pos"]): break

                # ===== EXITS / DCA / RISK =====
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos: continue
                    tf0 = STATE.tfs[0] if STATE.tfs else "5m"
                    try:
                        kl = await fetch_klines(sym, tf0, 3)
                    except Exception:
                        continue
                    price = float(kl[-1][2])
                    avg = st.pos.avg_price
                    move_pct = (price-avg)/(avg or 1)*100.0
                    qty = st.pos.qty_total()
                    feats_now = features_from_candles(kl)
                    atrp = feats_now["atrp"]

                    # trail width
                    trail_w = max(float(STATE.risk_cfg.get("trail_min",3.0)),
                                  atrp * float(STATE.risk_cfg.get("trail_mult",7.0)))
                    st.pos.high_water = max(st.pos.high_water, price)
                    st.pos.low_water  = min(st.pos.low_water, price)
                    if st.pos.side=="LONG":
                        trail_level = st.pos.high_water*(1.0 - trail_w/100.0)
                    else:
                        trail_level = st.pos.low_water*(1.0 + trail_w/100.0)

                    # HYBRID auto-aktiv
                    if bool(STATE.risk_cfg.get("hyb_on", True)) and not st.pos.trailing_active:
                        start = float(STATE.risk_cfg.get("hyb_start",2.5))
                        lock  = float(STATE.risk_cfg.get("hyb_lock",0.25))
                        if (st.pos.side=="LONG" and move_pct>=start) or (st.pos.side=="SHORT" and -move_pct>=start):
                            st.pos.trailing_active=True
                            st.pos.lock_price = (st.pos.avg_price*(1.0+lock/100.0) if st.pos.side=="LONG"
                                                 else st.pos.avg_price*(1.0-lock/100.0))
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔒 TRAIL ON {sym} | start={start:.2f}% lock={lock:.2f}% trail≈{trail_w:.2f}%")

                    # EARLY CUT
                    if bool(STATE.risk_cfg.get("early_cut", False)):
                        win_m = int(STATE.risk_cfg.get("early_cut_window_m",3))
                        thr   = float(STATE.risk_cfg.get("early_cut_thr",0.55))
                        if datetime.now(timezone.utc) - st.pos.opened_at <= timedelta(minutes=win_m):
                            hit = (st.pos.side=="LONG" and move_pct<=-thr) or (st.pos.side=="SHORT" and move_pct>=thr)
                            if hit:
                                net = _exit_all(sym, price, st)
                                if STATE.chat_id:
                                    mark = "✅" if net>=0 else "❌"
                                    await app.bot.send_message(STATE.chat_id,
                                        f"⚡ EARLY-CUT {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark}")
                                if STATE.ai_on: ai_learn(sym, tf0, feats_now, net)
                                continue

                    # EXIT via Trail/TP
                    did_exit=False
                    if bool(STATE.risk_cfg.get("trail_on", True)) and (st.pos.trailing_active or bool(STATE.risk_cfg.get("hyb_on", True))):
                        stop_lvl = trail_level
                        if st.pos.trailing_active and st.pos.lock_price>0:
                            stop_lvl = (max(stop_lvl, st.pos.lock_price) if st.pos.side=="LONG"
                                        else min(stop_lvl, st.pos.lock_price))
                        if (st.pos.side=="LONG" and price<=stop_lvl) or (st.pos.side=="SHORT" and price>=stop_lvl):
                            net = _exit_all(sym, price, st); did_exit=True
                            if STATE.chat_id:
                                mark = "✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id,
                                    f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark} | stop={stop_lvl:.4f} | trail={trail_w:.2f}%")
                            if STATE.ai_on: ai_learn(sym, tf0, feats_now, net)
                    else:
                        if (st.pos.side=="LONG" and price>=st.pos.target_price) or (st.pos.side=="SHORT" and price<=st.pos.target_price):
                            net = _exit_all(sym, price, st); did_exit=True
                            if STATE.chat_id:
                                mark = "✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id, f"🟤 EXIT {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark}")
                            if STATE.ai_on: ai_learn(sym, tf0, feats_now, net)
                    if did_exit: continue

                    # DCA
                    step = st.next_step_pct
                    need_dca = (st.pos.side=="LONG" and move_pct<=-step) or (st.pos.side=="SHORT" and move_pct>=step)
                    if need_dca and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        base = float(STATE.risk_cfg.get("size", STATE.position_size))
                        usd = base * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
                        _enter_leg(sym, st.pos.side, price, usd, st)
                        st.next_step_pct = max(STATE.grid_cfg["step_min"], st.next_step_pct * STATE.grid_cfg["step_mult"])
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id,
                                f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} | leg {st.pos.safety_count} | leg_size={usd:.2f} USDT | avg {st.pos.avg_price:.4f} | TP {st.pos.target_price:.4f}")

                    # MAX-SL i %
                    sl = float(STATE.risk_cfg.get("sl_max",0.0) or 0.0)
                    if sl>0:
                        hit = (st.pos.side=="LONG" and move_pct<=-sl) or (st.pos.side=="SHORT" and move_pct>=sl)
                        if hit:
                            net = _exit_all(sym, price, st)
                            if STATE.chat_id:
                                mark = "✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id,
                                    f"⛔ MAX-SL {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark}")
                            continue

                    # MAX DD i USDT
                    max_dd = float(STATE.risk_cfg.get("max_dd_per_trade",0.0) or 0.0)
                    if max_dd>0 and qty>0:
                        unreal = (qty*max(0.0, st.pos.avg_price-price) if st.pos.side=="LONG"
                                  else qty*max(0.0, price-st.pos.avg_price))
                        if unreal >= max_dd:
                            net = _exit_all(sym, price, st)
                            if STATE.chat_id:
                                mark = "✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id,
                                    f"🧯 MAX-DD EXIT {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark}")
                            continue

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)# -----------------------------
# AI SAVE/LOAD
# -----------------------------
def save_ai() -> str:
    try:
        with open(AI_MODEL_PATH, "wb") as f:
            pickle.dump(AI_WEIGHTS, f)
        return f"{AI_MODEL_PATH}"
    except Exception as e:
        return f"Fel vid sparning: {e}"

def load_ai() -> str:
    global AI_WEIGHTS
    try:
        with open(AI_MODEL_PATH, "rb") as f:
            AI_WEIGHTS = pickle.load(f)
        return f"{AI_MODEL_PATH}"
    except Exception as e:
        return f"Kunde inte ladda AI: {e}"

# -----------------------------
# TELEGRAM UI HELPERS
# -----------------------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/mode premium")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/threshold")],
        [KeyboardButton("/risk"), KeyboardButton("/export_csv")],
        [KeyboardButton("/backtest BTC-USDT 3d 0.001")]
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def status_text() -> str:
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    pos_lines=[]
    for s in STATE.symbols:
        st=STATE.per_sym[s]
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} → TP {st.pos.target_price:.4f} | legs {st.pos.safety_count} | trail={'ON' if st.pos.trailing_active else 'OFF'}")
    g=STATE.grid_cfg; r=STATE.risk_cfg
    lines=[
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"AI: {'ON 🧠' if STATE.ai_on else 'OFF'}",
        f"Entry-threshold: {STATE.entry_threshold:.2f}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size (default): {STATE.position_size:.1f} USDT | Fee per sida: {STATE.fee_side:.4%}",
        (f"Grid: max_safety={g['max_safety']} step_mult={g['step_mult']} step_min%={g['step_min']} "
         f"size_mult={g['size_mult']} tp%={g['tp']} (+{g['tp_bonus']}%/safety)"),
        (f"Risk: dd={r['dd']}% | max_pos={r['max_pos']} | size={r.get('size',STATE.position_size):.2f} USDT | "
         f"shorts={'ON' if r['allow_shorts'] else 'OFF'} | trail={'ON' if r.get('trail_on') else 'OFF'} "
         f"(min={r.get('trail_min',0):.2f}% x{r.get('trail_mult',1):.2f}) | "
         f"hybrid={'ON' if r.get('hyb_on') else 'OFF'} (start={r.get('hyb_start',0):.2f}% lock={r.get('hyb_lock',0):.2f}%)\n"
         f"  sl_max={r.get('sl_max',0):.2f}% | early_cut={'ON' if r.get('early_cut') else 'OFF'} "
         f"(thr={r.get('early_cut_thr',0):.2f}%/{r.get('early_cut_window_m',0)}m) | "
         f"max_dd_per_trade={r.get('max_dd_per_trade',0):.2f} USDT\n"
         f"  confirm_mode={'ON' if r.get('confirm_mode') else 'OFF'} | confirm_candle={int(r.get('confirm_candle',0))} | confirm_strength={r.get('confirm_strength',0):.2f}%\n"
         f"  trend_filter={'ON' if r.get('trend_filter') else 'OFF'} | trend_len={int(r.get('trend_len',0))} | trend_bias={r.get('trend_bias',0):.2f}"),
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

# -----------------------------
# TELEGRAM COMMANDS
# -----------------------------
tg_app = Application.builder().token(BOT_TOKEN).build()

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "🤖 Mp ORBbot v45 – Premium Mode\nAnvänd /mode premium och /engine_on för att starta.", reply_markup=reply_kb())
    await tg_app.bot.send_message(STATE.chat_id, status_text())

async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, status_text(), reply_markup=reply_kb())

async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(STATE.chat_id, "Engine: ON ✅", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

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
    parts = msg.split(" ",1)
    if len(parts)==2:
        tfs=[x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs=tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.tfs)}", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, "Använd: /timeframe 5m,15m,30m", reply_markup=reply_kb())

async def cmd_threshold(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks)==1:
        await tg_app.bot.send_message(STATE.chat_id, f"Aktuellt threshold: {STATE.entry_threshold:.2f}", reply_markup=reply_kb())
        return
    try:
        val=float(toks[1]); 
        if val<=0: raise ValueError()
        STATE.entry_threshold=val
        await tg_app.bot.send_message(STATE.chat_id, f"Threshold uppdaterad: {val:.2f}", reply_markup=reply_kb())
    except:
        await tg_app.bot.send_message(STATE.chat_id, "Fel värde. Ex: /threshold 1.85", reply_markup=reply_kb())

def _yesno(s:str)->bool: return s.lower() in ("1","true","on","yes","y")

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks)==4 and toks[0]=="/risk" and toks[1]=="set":
        key,val = toks[2], toks[3]
        if key in STATE.risk_cfg:
            try:
                if key=="max_pos": STATE.risk_cfg[key]=int(val)
                elif key in ("allow_shorts","trail_on","hyb_on","early_cut","confirm_mode","trend_filter"):
                    STATE.risk_cfg[key]=_yesno(val)
                elif key in ("confirm_candle","early_cut_window_m","trend_len"):
                    STATE.risk_cfg[key]=int(float(val))
                elif key=="size": STATE.risk_cfg[key]=float(val)
                else:
                    STATE.risk_cfg[key]=float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb())
            except:
                await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
        else:
            await tg_app.bot.send_message(STATE.chat_id, f"Okänd risk-nyckel: {key}", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, status_text(), reply_markup=reply_kb())

async def cmd_symbols(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = update.message.text.strip()
    toks = txt.split()
    if len(toks) >= 3 and toks[0] == "/symbols":
        if toks[1] == "add":
            sym = toks[2].upper()
            if sym not in STATE.symbols:
                STATE.symbols.append(sym)
                STATE.per_sym[sym] = SymState()
                await tg_app.bot.send_message(STATE.chat_id, f"La till: {sym}", reply_markup=reply_kb())
            else:
                await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns redan.", reply_markup=reply_kb())
            return
        if toks[1] == "remove":
            sym = toks[2].upper()
            if sym in STATE.symbols:
                STATE.symbols = [s for s in STATE.symbols if s != sym]
                STATE.per_sym.pop(sym, None)
                await tg_app.bot.send_message(STATE.chat_id, f"Tog bort: {sym}", reply_markup=reply_kb())
            else:
                await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns ej.", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id,
        ("Symbols: " + ", ".join(STATE.symbols) +
         "\nLägg till/ta bort:\n  /symbols add LINK-USDT\n  /symbols remove LINK-USDT"),
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
    rows = [["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r["time"], r["symbol"], r["side"], r["avg_price"], r["exit_price"],
                         r["gross"], r["fee_in"], r["fee_out"], r["net"], r["safety_legs"]])
    if len(rows) == 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb())
        return
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="trades_v45.csv", caption="Export CSV")

# -----------------------------
# BACKTEST (snabb, på klines)
# -----------------------------
def _parse_period(s: str) -> int:
    s=s.strip().lower()
    if s.endswith("d"): return int(float(s[:-1]) * 1440)
    if s.endswith("h"): return int(float(s[:-1]) * 60)
    if s.endswith("m"): return int(float(s[:-1]))
    return int(float(s))  # minutes

async def run_backtest(symbol: str, period: str, fee_per_side: float = None) -> Dict:
    """En enkel backtest som går igenom klines på minsta TF och simulerar premium-exits."""
    tf0 = STATE.tfs[0] if STATE.tfs else "5m"
    minutes = _parse_period(period)
    # hämta tillräckligt många candles
    need = max(200, int(minutes / (1 if tf0=="1m" else (3 if tf0=="3m" else (5 if tf0=="5m" else 15)))) + 50)
    kl = await fetch_klines(symbol, tf0, need)
    closes = [float(c[2]) for c in kl]
    if len(closes) < 50:
        return {"ok": False, "msg": "För lite data."}
    fee = fee_per_side if fee_per_side is not None else STATE.fee_side
    # policy
    th = float(STATE.entry_threshold)
    hyb_start = float(STATE.risk_cfg.get("hyb_start",2.5))
    hyb_lock  = float(STATE.risk_cfg.get("hyb_lock",0.25))
    trail_min = float(STATE.risk_cfg.get("trail_min",3.0))
    trail_mult= float(STATE.risk_cfg.get("trail_mult",7.0))
    sl_max    = float(STATE.risk_cfg.get("sl_max",1.1))

    pnl = 0.0; trades=0; wins=0; losses=0
    pos_avg=None; pos_high=None
    for i in range(30, len(closes)):
        # dummy score: momentum + ema-diff
        window = closes[i-6:i+1]
        mom = (window[-1]-window[0])/(window[0] or 1)*100.0
        ema9=ema(closes[:i+1],9)[-1]; ema21=ema(closes[:i+1],21)[-1]
        ema_diff = (ema9-ema21)/(ema21 or 1)*100.0
        score = max(-10,min(10, 0.7*mom + 0.6*ema_diff))
        price = window[-1]

        if pos_avg is None:
            if score>th:
                pos_avg=price; pos_high=price; trades+=1
                pnl -= fee*100*2  # approx som % (båda sidor)
        else:
            pos_high=max(pos_high, price)
            move_pct = (price-pos_avg)/(pos_avg or 1)*100.0
            # trail width approx
            trail_w = max(trail_min, 1.2*trail_mult)  # enkel approx utan ATR i backtest
            # hybrid start
            if move_pct>=hyb_start:
                lock_price = pos_avg*(1.0+hyb_lock/100.0)
                stop = max(pos_high*(1.0-trail_w/100.0), lock_price)
                if price<=stop:
                    # exit
                    p = (price-pos_avg)/(pos_avg or 1)*100.0
                    pnl += p
                    if p>=0: wins+=1
                    else: losses+=1
                    pos_avg=None; pos_high=None
            # sl_max
            if move_pct<=-sl_max and pos_avg is not None:
                p = (price-pos_avg)/(pos_avg or 1)*100.0
                pnl += p
                if p>=0: wins+=1
                else: losses+=1
                pos_avg=None; pos_high=None

    return {"ok": True, "symbol": symbol, "tf": tf0, "trades": trades, "wins": wins, "losses": losses, "pnl_pct": round(pnl,3)}

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    args = context.args
    if not args or len(args)<2:
        await tg_app.bot.send_message(STATE.chat_id, "Använd: /backtest SYMBOL PERIOD [fee]\nEx: /backtest BTC-USDT 3d 0.001", reply_markup=reply_kb())
        return
    sym = args[0].upper()
    period = args[1]
    fee = float(args[2]) if len(args)>=3 else None
    try:
        res = await run_backtest(sym, period, fee)
        if not res.get("ok"):
            await tg_app.bot.send_message(STATE.chat_id, f"Backtest misslyckades: {res.get('msg','')}")
            return
        await tg_app.bot.send_message(STATE.chat_id,
            f"🔬 Backtest {res['symbol']} @ {res['tf']}\n"
            f"Trades: {res['trades']} | Wins: {res['wins']} | Losses: {res['losses']}\n"
            f"Totalt PnL: {res['pnl_pct']} %", reply_markup=reply_kb())
    except Exception as e:
        await tg_app.bot.send_message(STATE.chat_id, f"Backtest-fel: {e}")

# -----------------------------
# MODE (preset)
# -----------------------------
async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    args = context.args
    sub = args[0].lower() if args else ""
    if sub == "premium":
        STATE.entry_threshold = 1.85
        STATE.risk_cfg.update(dict(
            trail_on=True, hyb_on=True,
            trail_min=3.0, trail_mult=7.0,
            hyb_start=2.50, hyb_lock=0.25,
            max_pos=5, size=80.0,
            sl_max=1.10, early_cut=False,
            allow_shorts=False,
            confirm_mode=True, confirm_candle=2, confirm_strength=0.4,
            trend_filter=True, trend_len=30, trend_bias=0.6
        ))
        STATE.tfs = ["5m","15m","30m"]
        await tg_app.bot.send_message(STATE.chat_id, "Mode satt: premium ✅", reply_markup=reply_kb())
        return
    await tg_app.bot.send_message(STATE.chat_id, "Använd: /mode premium", reply_markup=reply_kb())

# -----------------------------
# HANDLERS
# -----------------------------
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("ai_on", cmd_ai_on))
tg_app.add_handler(CommandHandler("ai_off", cmd_ai_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("threshold", cmd_threshold))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("backtest", cmd_backtest))
tg_app.add_handler(CommandHandler("mode", cmd_mode))

# -----------------------------
# FASTAPI WEBHOOK/APP
# -----------------------------
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    # Sätt webhook (om satt)
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    # Starta Telegram korrekt (await = Render-vänligt)
    await tg_app.initialize()
    await tg_app.start()
    # Starta engine efter att botten är igång
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return f"MP Bot v45 OK | engine_on={STATE.engine_on} | thr={STATE.entry_threshold:.2f} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {"ok": True, "engine_on": STATE.engine_on, "threshold": round(STATE.entry_threshold,4),
            "tfs": STATE.tfs, "symbols": STATE.symbols, "pnl_total": round(total,6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
# ----------------------------- END -----------------------------
