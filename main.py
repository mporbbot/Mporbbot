# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MP Bot – v60 (Reversal Sniper)
# - Mean-reversion/“sniper” entries (ersätter ORB-breakout)
# - Multitimeframe-bias + enkel AI-vikter (lär av PnL)
# - Grid/DCA, TP eller Hybrid→Trail, SL/Max-SL/Early-Cut/Max-DD
# - Samma Telegram-kommandon som tidigare (v47), inkl. /mode
# ------------------------------------------------------------

import os, io, csv, pickle, asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
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

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "2"))

# Grid/DCA
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "2"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.40"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "1.2"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.2"))
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.45"))
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.03"))

# Risk
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "2.0"))

# Max-SL / Early cut / Max DD
SL_MAX_PCT = float(os.getenv("SL_MAX_PCT", "0.5"))
EARLY_CUT_ON = (os.getenv("EARLY_CUT_ON", "true").lower() in ("1","true","on","yes"))
EARLY_CUT_THR = float(os.getenv("EARLY_CUT_THR", "0.55"))
EARLY_CUT_WIN_M = int(os.getenv("EARLY_CUT_WINDOW_M", "3"))
MAX_DD_TRADE_USDT = float(os.getenv("MAX_DD_TRADE_USDT", "2.5"))

# Entry-tröskel (används som “poäng” för sniper-score)
ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "1.25"))

# Trailing / Hybrid
TRAIL_ON = (os.getenv("TRAIL_ON", "true").lower() in ("1","true","on","yes"))
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "4.5"))
TRAIL_MIN_PCT = float(os.getenv("TRAIL_MIN_PCT", "1.20"))
HYB_ON = (os.getenv("HYB_ON", "true").lower() in ("1","true","on","yes"))
HYB_START_PCT = float(os.getenv("HYB_START_PCT", "1.25"))
HYB_LOCK_PCT = float(os.getenv("HYB_LOCK_PCT", "0.50"))

# Sniper-parametrar (styr entry)
SNIPER_DEV_MIN = float(os.getenv("SNIPER_DEV_MIN", "0.40"))  # min avvikelse från EMA21 i %
SNIPER_ATR_MIN = float(os.getenv("SNIPER_ATR_MIN", "0.20"))  # min ATR% för att undvika död marknad
SNIPER_RSI_TURN = float(os.getenv("SNIPER_RSI_TURN", "0.30"))# min RSI-vändning (delta) för att bekräfta reversal
SNIPER_MTF_BIAS = float(os.getenv("SNIPER_MTF_BIAS", "0.40"))# hur mycket högre TF:s trend påverkar score

AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "ai_model.pkl")

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1hour"}

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
    side: str
    legs: List[TradeLeg] = field(default_factory=list)
    avg_price: float = 0.0
    target_price: float = 0.0
    safety_count: int = 0
    high_water: float = 0.0
    low_water: float = 0.0
    trailing_active: bool = False
    lock_price: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    def qty_total(self) -> float: return sum(l.qty for l in self.legs)

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
        trail_on=TRAIL_ON, trail_mult=TRAIL_ATR_MULT, trail_min=TRAIL_MIN_PCT,
        hyb_on=HYB_ON, hyb_start=HYB_START_PCT, hyb_lock=HYB_LOCK_PCT,
        size=POSITION_SIZE_USDT,
        sl_max=SL_MAX_PCT, early_cut=EARLY_CUT_ON,
        early_cut_thr=EARLY_CUT_THR, early_cut_window_m=EARLY_CUT_WIN_M,
        max_dd_per_trade=MAX_DD_TRADE_USDT,
        # Sniper-parametrar justerbara via /risk set ...
        sniper_dev_min=SNIPER_DEV_MIN, sniper_atr_min=SNIPER_ATR_MIN,
        sniper_rsi_turn=SNIPER_RSI_TURN, sniper_mtf_bias=SNIPER_MTF_BIAS
    ))
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols: STATE.per_sym[s] = SymState()

# AI-vikter (för sniper-score)
AI_WEIGHTS: Dict[Tuple[str,str], List[float]] = {}  # bias, dev, rsi_turn, mom, trend
AI_LEARN_RATE = 0.05

# -----------------------------
# UI
# -----------------------------
def reply_kb():
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/threshold"), KeyboardButton("/symbols")],
        [KeyboardButton("/ai_on"), KeyboardButton("/ai_off")],
        [KeyboardButton("/save_ai"), KeyboardButton("/load_ai")],
        [KeyboardButton("/export_csv"), KeyboardButton("/panic")],
        [KeyboardButton("/reset_pnl"), KeyboardButton("/mode")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -----------------------------
# DATA & INDICATORS
# -----------------------------
async def get_klines(symbol: str, tf: str, limit: int = 60):
    k_tf = TF_MAP.get(tf, "5min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = list(reversed(r.json()["data"]))
    out = []
    for row in data[-limit:]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1: return series[:]
    k = 2 / (period + 1); out = []; val = series[0]
    for x in series:
        val = (x - val) * k + val; out.append(val)
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
        val = 100.0 - (100.0 / (1.0 + rs))
        rsis.append(val)
    return [50.0] + rsis

def features_from_candles(candles):
    closes = [c[-1] for c in candles]
    if len(closes) < 30:
        return {"mom":0.0,"ema_fast":0.0,"ema_slow":0.0,"rsi":50.0,"atrp":0.2,"dev":0.0,"rsi_slope":0.0}
    ema9 = ema(closes, 9); ema21 = ema(closes, 21)
    mom = (closes[-1]-closes[-6])/(closes[-6] or 1)*100.0
    rsi_series = rsi(closes, 14)
    rsi_last = rsi_series[-1]; rsi_prev = rsi_series[-2] if len(rsi_series)>1 else rsi_last
    rng = [(c[2]-c[3])/(c[-1] or 1)*100.0 for c in candles[-20:]]
    atrp = sum(rng)/len(rng) if rng else 0.2
    ema_fast = ema9[-1]; ema_slow = ema21[-1]
    dev = (closes[-1]-ema_slow)/(ema_slow or 1)*100.0  # pos=över EMA21, neg=under
    return {
        "mom": mom, "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": rsi_last, "atrp": max(0.02,min(3.0,atrp)),
        "dev": dev, "rsi_slope": (rsi_last - rsi_prev)
    }

# -----------------------------
# SNIPER SCORE (mean-reversion)
# -----------------------------
def _wkey(symbol: str, tf: str) -> Tuple[str, str]: return (symbol, tf)

def sniper_score(symbol: str, tf: str, feats: Dict[str,float], dir_long: bool) -> float:
    """
    dir_long=True  -> koop dip (dev<0), rsi vänder upp (> SNIPER_RSI_TURN), mom gärna negativ som fyller poäng
    dir_long=False -> sälj topp (dev>0), rsi vänder ner (< -SNIPER_RSI_TURN)
    """
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        # w0, w_dev, w_rsi_turn, w_mom, w_trend(ema_diff)
        AI_WEIGHTS[key] = [0.0, 0.9, 0.6, 0.3, 0.4]
    w0, w_dev, w_rsi, w_mom, w_trend = AI_WEIGHTS[key]

    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
    dev = feats["dev"]                     # +/- från EMA21
    rsi_turn = feats["rsi_slope"]          # RSI-ändring senaste candle
    mom = feats["mom"]
    atrp = feats["atrp"]

    # Filter: tillräcklig volatilitet + dev från EMA
    dev_ok = (abs(dev) >= STATE.risk_cfg.get("sniper_dev_min", SNIPER_DEV_MIN))
    atr_ok = (atrp >= STATE.risk_cfg.get("sniper_atr_min", SNIPER_ATR_MIN))
    if not (dev_ok and atr_ok):
        return -999.0  # blockera entry på denna TF

    if dir_long:
        # Ju mer under EMA21 (negativ dev), desto bättre. Positiv RSI-slope, negativ/neutral mom ökar score.
        raw = (w0
               + w_dev * max(0.0, -dev)
               + w_rsi * max(0.0, rsi_turn)
               + w_mom * max(0.0, -mom)      # nyligen nedtryckt gynnar reversion
               + w_trend * max(0.0, ema_diff))  # upptrend på TF ger bonus
    else:
        raw = (w0
               + w_dev * max(0.0, dev)
               + w_rsi * max(0.0, -rsi_turn)
               + w_mom * max(0.0, mom)
               + w_trend * max(0.0, -ema_diff))
    # Skala och kapa
    return max(-10.0, min(10.0, raw / 2.0))

def sniper_learn(symbol: str, tf: str, feats: Dict[str,float], pnl_net: float):
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        AI_WEIGHTS[key] = [0.0, 0.9, 0.6, 0.3, 0.4]
    w = AI_WEIGHTS[key]
    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
    grad = [1.0, abs(feats["dev"]), abs(feats["rsi_slope"]), abs(feats["mom"]), abs(ema_diff)]
    lr = AI_LEARN_RATE * (1 if pnl_net >= 0 else -1)
    AI_WEIGHTS[key] = [w[i] + lr*grad[i] for i in range(5)]

def save_ai() -> str:
    try:
        with open(AI_MODEL_PATH, "wb") as f: pickle.dump(AI_WEIGHTS, f)
        return f"AI sparad: {AI_MODEL_PATH}"
    except Exception as e:
        return f"Fel vid sparning: {e}"

def load_ai() -> str:
    global AI_WEIGHTS
    try:
        with open(AI_MODEL_PATH, "rb") as f: AI_WEIGHTS = pickle.load(f)
        return f"AI laddad från: {AI_MODEL_PATH}"
    except Exception as e:
        return f"Kunde inte ladda AI: {e}"

# -----------------------------
# TRADING HELPERS (mock)
# -----------------------------
def _fee(amount_usdt: float) -> float: return amount_usdt * FEE_PER_SIDE

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
    return leg

def _exit_all(sym: str, price: float, st: SymState) -> float:
    if not st.pos: return 0.0
    gross = 0.0; fee_in = 0.0
    for leg in st.pos.legs:
        usd_in = leg.qty * leg.price; fee_in += _fee(usd_in)
        gross += leg.qty * ((price - leg.price) if st.pos.side=="LONG" else (leg.price - price))
    usd_out = st.pos.qty_total()*price; fee_out = _fee(usd_out)
    net = gross - fee_in - fee_out
    st.realized_pnl_net += net
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym, "side": st.pos.side, "avg_price": st.pos.avg_price,
        "exit_price": price, "gross": round(gross,6), "fee_in": round(fee_in,6),
        "fee_out": round(fee_out,6), "net": round(net,6), "safety_legs": st.pos.safety_count
    })
    st.pos = None; st.next_step_pct = STATE.grid_cfg["step_min"]; return net

# -----------------------------
# ENGINE (med Reversal Sniper)
# -----------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                # Entries
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.risk_cfg["max_pos"]:
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos: continue
                        feats_per_tf = {}; last_price = None; skip=False
                        for tf in STATE.tfs:
                            try:
                                kl = await get_klines(sym, tf, limit=60); last_price = kl[-1][-1]
                            except Exception: skip=True; break
                            feats_per_tf[tf] = features_from_candles(kl)
                        if skip or not feats_per_tf or last_price is None: continue

                        # MTF-bias (högre TF sist i listan)
                        tf_hi = STATE.tfs[-1]
                        hi = feats_per_tf[tf_hi]
                        mtf_bias = STATE.risk_cfg.get("sniper_mtf_bias", SNIPER_MTF_BIAS)
                        hi_trend = (hi["ema_fast"] - hi["ema_slow"]) / (hi["ema_slow"] or 1) * 100.0

                        # Räkna score för LONG/SHORT (mean reversion)
                        score_long = 0.0; score_short = 0.0
                        for tf, f in feats_per_tf.items():
                            score_long += (sniper_score(sym, tf, f, True))
                            score_short += (sniper_score(sym, tf, f, False))
                        # MTF-bias: bonus om man köper dip i övergripande upptrend, eller säljer spike i nedtrend
                        score_long += mtf_bias * max(0.0, hi_trend)
                        score_short += mtf_bias * max(0.0, -hi_trend)

                        th = STATE.entry_threshold
                        allow_shorts = STATE.risk_cfg["allow_shorts"]

                        if score_long > th:
                            st = STATE.per_sym[sym]; usd = float(STATE.risk_cfg.get("size", STATE.position_size))
                            _enter_leg(sym, "LONG", last_price, usd, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {st.pos.legs[-1].price:.4f} | thr={th:.2f} | scoreL={score_long:.2f} | TP {st.pos.target_price:.4f}")
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]: break

                        elif allow_shorts and score_short > th:
                            st = STATE.per_sym[sym]; usd = float(STATE.risk_cfg.get("size", STATE.position_size))
                            _enter_leg(sym, "SHORT", last_price, usd, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔻 ENTRY {sym} SHORT @ {st.pos.legs[-1].price:.4f} | thr={th:.2f} | scoreS={score_short:.2f} | TP {st.pos.target_price:.4f}")
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]: break

                # Exits / DCA / Risk
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos: continue
                    tf0 = STATE.tfs[0] if STATE.tfs else "5m"
                    try: kl = await get_klines(sym, tf0, limit=3)
                    except Exception: continue
                    price = kl[-1][-1]; avg = st.pos.avg_price
                    move_pct = (price-avg)/(avg or 1)*100.0; qty = st.pos.qty_total()
                    feats_now = features_from_candles(kl); atrp = feats_now["atrp"]

                    # trail width
                    trail_w = max(STATE.risk_cfg.get("trail_min",1.2),
                                  atrp * STATE.risk_cfg.get("trail_mult",4.5))
                    st.pos.high_water = max(st.pos.high_water, price)
                    st.pos.low_water  = min(st.pos.low_water, price)
                    if st.pos.side=="LONG":
                        trail_level = st.pos.high_water*(1.0 - trail_w/100.0)
                    else:
                        trail_level = st.pos.low_water*(1.0 + trail_w/100.0)

                    # HYBRID auto-aktiv
                    if STATE.risk_cfg.get("hyb_on", True) and not st.pos.trailing_active:
                        start = STATE.risk_cfg.get("hyb_start",1.25)
                        lock  = STATE.risk_cfg.get("hyb_lock",0.50)
                        if (st.pos.side=="LONG" and move_pct>=start) or (st.pos.side=="SHORT" and -move_pct>=start):
                            st.pos.trailing_active=True
                            st.pos.lock_price = (st.pos.avg_price*(1.0+lock/100.0) if st.pos.side=="LONG"
                                                 else st.pos.avg_price*(1.0-lock/100.0))
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔒 TRAIL ON {sym} | start={start:.2f}% lock={lock:.2f}% trail≈{trail_w:.2f}%")

                    # EARLY CUT (inom x minuter)
                    if STATE.risk_cfg.get("early_cut", True):
                        win_m = int(STATE.risk_cfg.get("early_cut_window_m",3))
                        thr   = float(STATE.risk_cfg.get("early_cut_thr",0.55))
                        if datetime.now(timezone.utc) - st.pos.opened_at <= timedelta(minutes=win_m):
                            hit = (st.pos.side=="LONG" and move_pct<=-thr) or (st.pos.side=="SHORT" and move_pct>=thr)
                            if hit:
                                net = _exit_all(sym, price, st)
                                if STATE.chat_id:
                                    mark = "✅" if net>=0 else "❌"
                                    await app.bot.send_message(STATE.chat_id,
                                        f"⚡ EARLY-CUT {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark} (thr={thr:.2f}%/{win_m}m)")
                                if STATE.ai_on: sniper_learn(sym, tf0, feats_now, net)
                                continue

                    # EXIT via Trail/TP
                    did_exit=False
                    if STATE.risk_cfg.get("trail_on", True) and (st.pos.trailing_active or STATE.risk_cfg.get("hyb_on", True)):
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
                            if STATE.ai_on: sniper_learn(sym, tf0, feats_now, net)
                    else:
                        if (st.pos.side=="LONG" and price>=st.pos.target_price) or (st.pos.side=="SHORT" and price<=st.pos.target_price):
                            net = _exit_all(sym, price, st); did_exit=True
                            if STATE.chat_id:
                                mark = "✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id, f"🟤 EXIT {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark}")
                            if STATE.ai_on: sniper_learn(sym, tf0, feats_now, net)
                    if did_exit: continue

                    # DCA vid motrörelse
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

                    # HÅRD MAX-SL i %
                    sl = float(STATE.risk_cfg.get("sl_max",0.0) or 0.0)
                    if sl>0:
                        hit = (st.pos.side=="LONG" and move_pct<=-sl) or (st.pos.side=="SHORT" and move_pct>=sl)
                        if hit:
                            net = _exit_all(sym, price, st)
                            if STATE.chat_id:
                                mark = "✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id,
                                    f"⛔ MAX-SL {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark} (sl_max={sl:.2f}%)")
                            continue

                    # MAX DD i USDT
                    max_dd = float(STATE.risk_cfg.get("max_dd_per_trade",0.0) or 0.0)
                    if max_dd>0:
                        unreal = (qty*max(0.0, st.pos.avg_price-price) if st.pos.side=="LONG"
                                  else qty*max(0.0, price-st.pos.avg_price))
                        if unreal >= max_dd:
                            net = _exit_all(sym, price, st)
                            if STATE.chat_id:
                                mark = "✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id,
                                    f"🧯 MAX-DD EXIT {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark} (limit {max_dd:.2f})")
                            continue

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try: await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except: pass
            await asyncio.sleep(5)

# -----------------------------
# TELEGRAM (samma kommandon)
# -----------------------------
tg_app = Application.builder().token(BOT_TOKEN).build()

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
         f"  sniper_dev_min={r.get('sniper_dev_min',0):.2f}% | sniper_atr_min={r.get('sniper_atr_min',0):.2f}% | "
         f"sniper_rsi_turn={r.get('sniper_rsi_turn',0):.2f}% | sniper_mtf_bias={r.get('sniper_mtf_bias',0):.2f}"),
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

async def send_status(chat_id:int): await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

async def cmd_start(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v60 – Reversal Sniper ✅", reply_markup=reply_kb())
    await send_status(STATE.chat_id)

async def cmd_status(update:Update,_): STATE.chat_id = update.effective_chat.id; await send_status(STATE.chat_id)
async def cmd_engine_on(update:Update,_): STATE.chat_id=update.effective_chat.id; STATE.engine_on=True; await tg_app.bot.send_message(STATE.chat_id,"Engine: ON ✅",reply_markup=reply_kb())
async def cmd_engine_off(update:Update,_): STATE.chat_id=update.effective_chat.id; STATE.engine_on=False; await tg_app.bot.send_message(STATE.chat_id,"Engine: OFF ⛔️",reply_markup=reply_kb())
async def cmd_ai_on(update:Update,_): STATE.chat_id=update.effective_chat.id; STATE.ai_on=True; await tg_app.bot.send_message(STATE.chat_id,"AI: ON 🧠",reply_markup=reply_kb())
async def cmd_ai_off(update:Update,_): STATE.chat_id=update.effective_chat.id; STATE.ai_on=False; await tg_app.bot.send_message(STATE.chat_id,"AI: OFF",reply_markup=reply_kb())

async def cmd_timeframe(update:Update, context:ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip(); parts = msg.split(" ",1)
    if len(parts)==2:
        tfs=[x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs: STATE.tfs=tfs; await tg_app.bot.send_message(STATE.chat_id,f"Timeframes satta: {', '.join(STATE.tfs)}",reply_markup=reply_kb()); return
    await tg_app.bot.send_message(STATE.chat_id,"Använd: /timeframe 5m,15m,30m",reply_markup=reply_kb())

async def cmd_threshold(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks)==1:
        await tg_app.bot.send_message(STATE.chat_id,f"Aktuellt threshold: {STATE.entry_threshold:.2f}",reply_markup=reply_kb()); return
    try:
        val=float(toks[1]); 
        if val<=0: raise ValueError()
        STATE.entry_threshold=val
        await tg_app.bot.send_message(STATE.chat_id,f"Threshold uppdaterad: {val:.2f}",reply_markup=reply_kb())
    except:
        await tg_app.bot.send_message(STATE.chat_id,"Fel värde. Ex: /threshold 1.25",reply_markup=reply_kb())

async def cmd_grid(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks)==4 and toks[0]=="/grid" and toks[1]=="set":
        key,val=toks[2],toks[3]
        if key in STATE.grid_cfg:
            try: STATE.grid_cfg[key]=float(val); await tg_app.bot.send_message(STATE.chat_id,"Grid uppdaterad.",reply_markup=reply_kb())
            except: await tg_app.bot.send_message(STATE.chat_id,"Felaktigt värde.",reply_markup=reply_kb())
            return
        else:
            await tg_app.bot.send_message(STATE.chat_id,f"Okänd grid-nyckel: {key}",reply_markup=reply_kb()); return
    g=STATE.grid_cfg
    msg=("Grid:\n"
         f"  max_safety={g['max_safety']}\n"
         f"  step_mult={g['step_mult']}\n"
         f"  step_min%={g['step_min']}\n"
         f"  size_mult={g['size_mult']}\n"
         f"  tp%={g['tp']} (+{g['tp_bonus']}%/safety)\n\n"
         "Ex: /grid set tp 0.60")
    await tg_app.bot.send_message(STATE.chat_id,msg,reply_markup=reply_kb())

async def cmd_risk(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks)==4 and toks[0]=="/risk" and toks[1]=="set":
        key,val=toks[2],toks[3]
        if key in STATE.risk_cfg:
            try:
                if key=="max_pos": STATE.risk_cfg[key]=int(val)
                elif key in ("allow_shorts","trail_on","hyb_on","early_cut"):
                    STATE.risk_cfg[key]=(val.lower() in ("1","true","on","yes"))
                elif key=="size": STATE.risk_cfg[key]=float(val)
                else: STATE.risk_cfg[key]=float(val)
                await tg_app.bot.send_message(STATE.chat_id,"Risk uppdaterad.",reply_markup=reply_kb())
            except:
                await tg_app.bot.send_message(STATE.chat_id,"Felaktigt värde.",reply_markup=reply_kb())
            return
        else:
            await tg_app.bot.send_message(STATE.chat_id,f"Okänd risk-nyckel: {key}",reply_markup=reply_kb()); return
    r=STATE.risk_cfg
    msg=(f"Risk:\n"
         f"  dd={r['dd']}% | max_pos={r['max_pos']} | size={r.get('size',STATE.position_size):.2f} USDT | "
         f"shorts={'ON' if r['allow_shorts'] else 'OFF'}\n"
         f"  trail={'ON' if r.get('trail_on') else 'OFF'} (min={r.get('trail_min',0):.2f}% x{r.get('trail_mult',1):.2f})\n"
         f"  hybrid={'ON' if r.get('hyb_on') else 'OFF'} (start={r.get('hyb_start',0):.2f}% lock={r.get('hyb_lock',0):.2f}%)\n"
         f"  sl_max={r.get('sl_max',0):.2f}% | early_cut={'ON' if r.get('early_cut') else 'OFF'} (thr={r.get('early_cut_thr',0):.2f}%/{r.get('early_cut_window_m',0)}m)\n"
         f"  max_dd_per_trade={r.get('max_dd_per_trade',0):.2f} USDT\n"
         f"  sniper_dev_min={r.get('sniper_dev_min',0):.2f}% | sniper_atr_min={r.get('sniper_atr_min',0):.2f}% | "
         f"sniper_rsi_turn={r.get('sniper_rsi_turn',0):.2f}% | sniper_mtf_bias={r.get('sniper_mtf_bias',0):.2f}\n"
         "Ex: /risk set sl_max 0.6 | /risk set sniper_dev_min 0.5 | /risk set sniper_mtf_bias 0.5")
    await tg_app.bot.send_message(STATE.chat_id,msg,reply_markup=reply_kb())

async def cmd_symbols(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    txt = update.message.text.strip(); toks = txt.split()
    if len(toks)>=3 and toks[0]=="/symbols":
        if toks[1]=="add":
            sym=toks[2].upper()
            if sym not in STATE.symbols:
                STATE.symbols.append(sym); STATE.per_sym[sym]=SymState()
                await tg_app.bot.send_message(STATE.chat_id,f"La till: {sym}",reply_markup=reply_kb())
            else: await tg_app.bot.send_message(STATE.chat_id,f"{sym} finns redan.",reply_markup=reply_kb())
            return
        if toks[1]=="remove":
            sym=toks[2].upper()
            if sym in STATE.symbols:
                STATE.symbols=[s for s in STATE.symbols if s!=sym]; STATE.per_sym.pop(sym,None)
                await tg_app.bot.send_message(STATE.chat_id,f"Tog bort: {sym}",reply_markup=reply_kb())
            else: await tg_app.bot.send_message(STATE.chat_id,f"{sym} finns ej.",reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id,("Symbols: "+", ".join(STATE.symbols)+
        "\nLägg till/ta bort:\n  /symbols add LINK-USDT\n  /symbols remove LINK-USDT"),reply_markup=reply_kb())

async def cmd_pnl(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    lines=[f"📈 PnL total (NET): {total:+.4f} USDT"]
    for s in STATE.symbols: lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl_net:+.4f} USDT")
    await tg_app.bot.send_message(STATE.chat_id,"\n".join(lines),reply_markup=reply_kb())

async def cmd_export_csv(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    rows=[["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r["time"],r["symbol"],r["side"],r["avg_price"],r["exit_price"],
                         r["gross"],r["fee_in"],r["fee_out"],r["net"],r["safety_legs"]])
    if len(rows)==1:
        await tg_app.bot.send_message(STATE.chat_id,"Inga trades loggade ännu.",reply_markup=reply_kb()); return
    buf=io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id,document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="trades.csv",caption="Export CSV")

async def cmd_panic(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    closed=[]
    for s in STATE.symbols:
        st=STATE.per_sym[s]
        if st.pos:
            tf0=STATE.tfs[0] if STATE.tfs else "5m"
            try: kl=await get_klines(s, tf0, limit=1); px=kl[-1][-1]
            except: continue
            net=_exit_all(s, px, st); closed.append(f"{s}:{net:+.4f}")
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {' | '.join(closed) if closed else 'Inga positioner.'}", reply_markup=reply_kb())

async def cmd_reset_pnl(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols: STATE.per_sym[s].realized_pnl_net=0.0; STATE.per_sym[s].trades_log.clear()
    await tg_app.bot.send_message(STATE.chat_id,"PnL återställd.",reply_markup=reply_kb())

async def cmd_save_ai(update:Update,_): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id, save_ai(), reply_markup=reply_kb())
async def cmd_load_ai(update:Update,_): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id, load_ai(), reply_markup=reply_kb())

def _yesno(s:str)->bool: return s.lower() in ("1","true","on","yes","y")

async def cmd_mode(update:Update,_):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip(); toks = msg.split()
    if len(toks)==1:
        await tg_app.bot.send_message(STATE.chat_id,
            "Använd: /mode <precision|profitguard|safe-trail|show|set>\n"
            "Ex: /mode precision  |  /mode profitguard  |  /mode show\n"
            "    /mode set threshold 1.25   | /mode set tfs 5m,15m,30m", reply_markup=reply_kb()); return
    sub=toks[1].lower()
    if sub=="show":
        await tg_app.bot.send_message(STATE.chat_id,
            f"threshold={STATE.entry_threshold:.2f} | tfs={', '.join(STATE.tfs)} | "
            f"size={STATE.risk_cfg.get('size',STATE.position_size):.2f} | max_pos={STATE.risk_cfg['max_pos']}\n"
            f"sniper_dev_min={STATE.risk_cfg['sniper_dev_min']:.2f}% | sniper_atr_min={STATE.risk_cfg['sniper_atr_min']:.2f}% | "
            f"sniper_rsi_turn={STATE.risk_cfg['sniper_rsi_turn']:.2f}% | sniper_mtf_bias={STATE.risk_cfg['sniper_mtf_bias']:.2f}",
            reply_markup=reply_kb()); return
    if sub=="set":
        if len(toks)<4:
            await tg_app.bot.send_message(STATE.chat_id,
                "Använd: /mode set <nyckel> <värde>\nNycklar: threshold,size,max_pos,shorts,trail_on,trail_min,trail_mult,hyb_on,hyb_start,hyb_lock,sl_max,early_cut,early_cut_thr,early_cut_window_m,max_dd_per_trade,tfs,sniper_dev_min,sniper_atr_min,sniper_rsi_turn,sniper_mtf_bias",
                reply_markup=reply_kb()); return
        key=toks[2].lower(); val=" ".join(toks[3:]).strip()
        try:
            if key=="threshold": STATE.entry_threshold=float(val)
            elif key=="size": STATE.risk_cfg["size"]=float(val)
            elif key=="max_pos": STATE.risk_cfg["max_pos"]=int(float(val))
            elif key=="shorts": STATE.risk_cfg["allow_shorts"]=_yesno(val)
            elif key=="trail_on": STATE.risk_cfg["trail_on"]=_yesno(val)
            elif key=="trail_min": STATE.risk_cfg["trail_min"]=float(val)
            elif key=="trail_mult": STATE.risk_cfg["trail_mult"]=float(val)
            elif key=="hyb_on": STATE.risk_cfg["hyb_on"]=_yesno(val)
            elif key=="hyb_start": STATE.risk_cfg["hyb_start"]=float(val)
            elif key=="hyb_lock": STATE.risk_cfg["hyb_lock"]=float(val)
            elif key=="sl_max": STATE.risk_cfg["sl_max"]=float(val)
            elif key=="early_cut": STATE.risk_cfg["early_cut"]=_yesno(val)
            elif key=="early_cut_thr": STATE.risk_cfg["early_cut_thr"]=float(val)
            elif key=="early_cut_window_m": STATE.risk_cfg["early_cut_window_m"]=int(float(val))
            elif key=="max_dd_per_trade": STATE.risk_cfg["max_dd_per_trade"]=float(val)
            elif key=="tfs": STATE.tfs=[x.strip() for x in val.split(",") if x.strip()]
            elif key in ("sniper_dev_min","sniper_atr_min","sniper_rsi_turn","sniper_mtf_bias"):
                STATE.risk_cfg[key]=float(val)
            else:
                await tg_app.bot.send_message(STATE.chat_id,f"Okänd nyckel: {key}",reply_markup=reply_kb()); return
            await tg_app.bot.send_message(STATE.chat_id,"Mode/param uppdaterad ✅",reply_markup=reply_kb())
        except Exception:
            await tg_app.bot.send_message(STATE.chat_id,f"Fel värde ({key}): {val}",reply_markup=reply_kb())
        return
    if sub in ("precision","profitguard","safe-trail"):
        if sub=="precision":
            STATE.entry_threshold=1.25
            STATE.risk_cfg.update(dict(trail_on=True, hyb_on=True, trail_min=1.2, trail_mult=4.5,
                                       hyb_start=1.25, hyb_lock=0.5, max_pos=2, size=80.0,
                                       sniper_dev_min=0.45, sniper_atr_min=0.20, sniper_rsi_turn=0.30, sniper_mtf_bias=0.40))
            STATE.tfs=["5m","15m","30m"]
        elif sub=="profitguard":
            STATE.entry_threshold=1.35
            STATE.risk_cfg.update(dict(trail_on=True, hyb_on=True, trail_min=1.6, trail_mult=5.0,
                                       hyb_start=1.30, hyb_lock=0.55, max_pos=2, size=90.0,
                                       sl_max=0.6, early_cut=False,
                                       sniper_dev_min=0.50, sniper_atr_min=0.22, sniper_rsi_turn=0.30, sniper_mtf_bias=0.50))
            STATE.tfs=["5m","15m","30m"]
        elif sub=="safe-trail":
            STATE.entry_threshold=1.40
            STATE.risk_cfg.update(dict(trail_on=True, hyb_on=True, trail_min=1.8, trail_mult=5.0,
                                       hyb_start=1.40, hyb_lock=0.55, max_pos=2, size=80.0,
                                       sl_max=0.5, early_cut=True,
                                       sniper_dev_min=0.55, sniper_atr_min=0.25, sniper_rsi_turn=0.35, sniper_mtf_bias=0.40))
            STATE.tfs=["5m","15m","30m"]
        await tg_app.bot.send_message(STATE.chat_id,f"Mode satt: {sub} ✅",reply_markup=reply_kb())
        return
    await tg_app.bot.send_message(STATE.chat_id,f"Okänt mode: {sub}",reply_markup=reply_kb())

# register handlers
for name, func in [
    ('start', cmd_start), ('status', cmd_status),
    ('engine_on', cmd_engine_on), ('engine_off', cmd_engine_off),
    ('ai_on', cmd_ai_on), ('ai_off', cmd_ai_off),
    ('timeframe', cmd_timeframe), ('threshold', cmd_threshold),
    ('grid', cmd_grid), ('risk', cmd_risk), ('symbols', cmd_symbols),
    ('pnl', cmd_pnl), ('export_csv', cmd_export_csv), ('panic', cmd_panic),
    ('reset_pnl', cmd_reset_pnl), ('save_ai', cmd_save_ai), ('load_ai', cmd_load_ai),
    ('mode', cmd_mode),
]: tg_app.add_handler(CommandHandler(name, func))

# -----------------------------
# FASTAPI WEBHOOK
# -----------------------------
app = FastAPI()
class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    asyncio.create_task(tg_app.initialize()); asyncio.create_task(tg_app.start()); asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop(); await tg_app.shutdown()

@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return f"MP Bot v60 (Sniper) OK | engine_on={STATE.engine_on} | thr={STATE.entry_threshold:.2f} | pnl_total={total:+.4f}"

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
