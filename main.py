# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MP Bot – v46 (Break & Retest Strategy)
# ------------------------------------------------------------
# - Breakout över EMA-band (EMA20>EMA50) + lokalt hög/låg
# - Vänta RETEST inom fönster, entry på rejection mot EMA20/nivån
# - ATR-filter, Trend-filter, Early-cut, SL-max, Max-DD
# - Hybrid TP→Trail (sen aktivering) för att låta vinster gå
# - Mock-handel + CSV-export + Backtest (light) + Telegram UI
# - Render-vänlig Telegram-start (await initialize/start)
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

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "80"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "4"))
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "2.0"))

# Grid/TP
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "1"))  # BR = färre DCA som standard
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.60"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "1.3"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.2"))
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.50"))
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.03"))

# BR thresholds
ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "1.60"))  # BR-score-tröskel (högre=färre trades)

# Trailing / Hybrid
TRAIL_ON = (os.getenv("TRAIL_ON", "true").lower() in ("1","true","on","yes"))
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "7.0"))
TRAIL_MIN_PCT = float(os.getenv("TRAIL_MIN_PCT", "2.8"))
HYB_ON = (os.getenv("HYB_ON", "true").lower() in ("1","true","on","yes"))
HYB_START_PCT = float(os.getenv("HYB_START_PCT", "2.2"))
HYB_LOCK_PCT = float(os.getenv("HYB_LOCK_PCT", "0.25"))

# Failsafes
SL_MAX_PCT = float(os.getenv("SL_MAX_PCT", "0.60"))
EARLY_CUT_ON = (os.getenv("EARLY_CUT_ON", "true").lower() in ("1","true","on","yes"))
EARLY_CUT_THR = float(os.getenv("EARLY_CUT_THR", "0.50"))
EARLY_CUT_WIN_M = int(os.getenv("EARLY_CUT_WINDOW_M", "3"))
MAX_DD_TRADE_USDT = float(os.getenv("MAX_DD_TRADE_USDT", "2.50"))

# BR-parametrar
BR_BREAK_LOOKBACK = int(os.getenv("BR_BREAK_LOOKBACK", "12"))      # n senaste ljus för "lokalt hög/låg"
BR_CONFIRM_CANDLES = int(os.getenv("BR_CONFIRM_CANDLES", "2"))     # antal candles på break-stängning över/under band
BR_RETEST_WINDOW_M = int(os.getenv("BR_RETEST_WINDOW_M", "15"))    # tid fönster för retest
BR_RETEST_PULL_PCT = float(os.getenv("BR_RETEST_PULL_PCT", "0.25"))# hur långt priset ska “dra tillbaka” från break (%)
BR_REJECT_WICK_PCT = float(os.getenv("BR_REJECT_WICK_PCT", "0.10"))# wick-krav (rejection)

# Filter
TREND_FILTER = (os.getenv("TREND_FILTER", "true").lower() in ("1","true","on","yes"))
TREND_LEN = int(os.getenv("TREND_LEN", "45"))
TREND_BIAS = float(os.getenv("TREND_BIAS", "0.80"))
ATR_MIN = float(os.getenv("ATR_MIN", "0.25")) # min ATR% (för att undvika låg vol)

# Data
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
    side: str  # LONG/SHORT
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
class BRPlan:
    direction: str        # "LONG"/"SHORT"
    level: float          # breakoutnivå (lokal high/low)
    created_at: datetime
    expire_at: datetime

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)
    next_step_pct: float = GRID_STEP_MIN_PCT
    plan: Optional[BRPlan] = None        # aktivt retest-plan

@dataclass
class EngineState:
    engine_on: bool = False
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
    per_sym: Dict[str, SymState] = field(default_factory=dict)
    position_size: float = POSITION_SIZE_USDT
    fee_side: float = FEE_PER_SIDE
    entry_threshold: float = ENTRY_THRESHOLD
    grid_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        max_safety=GRID_MAX_SAFETY, step_mult=GRID_STEP_MULT, step_min=GRID_STEP_MIN_PCT,
        size_mult=GRID_SIZE_MULT, tp=GRID_TP_PCT, tp_bonus=TP_SAFETY_BONUS
    ))
    risk_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        dd=DD_STOP_PCT, max_pos=MAX_OPEN_POS, allow_shorts=False, size=POSITION_SIZE_USDT,
        # trail/hybrid
        trail_on=TRAIL_ON, trail_mult=TRAIL_ATR_MULT, trail_min=TRAIL_MIN_PCT,
        hyb_on=HYB_ON, hyb_start=HYB_START_PCT, hyb_lock=HYB_LOCK_PCT,
        # failsafes
        sl_max=SL_MAX_PCT, early_cut=EARLY_CUT_ON,
        early_cut_thr=EARLY_CUT_THR, early_cut_window_m=EARLY_CUT_WIN_M,
        max_dd_per_trade=MAX_DD_TRADE_USDT,
        # BR-parametrar
        br_break_lookback=BR_BREAK_LOOKBACK, br_confirm_candles=BR_CONFIRM_CANDLES,
        br_retest_window_m=BR_RETEST_WINDOW_M, br_retest_pull_pct=BR_RETEST_PULL_PCT,
        br_reject_wick_pct=BR_REJECT_WICK_PCT,
        # filter
        trend_filter=TREND_FILTER, trend_len=TREND_LEN, trend_bias=TREND_BIAS, atr_min=ATR_MIN
    ))
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# -----------------------------
# IO HELPERS
# -----------------------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def log_mock_trade(row: Dict):
    fname = "mock_trade_log.csv"
    newfile = not os.path.exists(fname)
    ensure_dir(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time","symbol","action","side","price","qty","avg_price","gross","fee_in","fee_out","net","info"])
        if newfile:
            w.writeheader()
        w.writerow(row)

# -----------------------------
# DATA & INDICATORS
# -----------------------------
async def fetch_klines(symbol: str, tf: str, limit: int = 120) -> List[List[str]]:
    tf_api = TF_MAP.get(tf, tf)
    params = {"symbol": symbol, "type": tf_api}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
    return data[::-1]  # senaste sist

def ema(series: List[float], period: int) -> List[float]:
    if not series or period<=1: return series[:]
    k = 2.0/(period+1.0); out=[]; val=series[0]
    for x in series:
        val = (x-val)*k + val
        out.append(val)
    return out

def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes)<period+1: return [50.0]*len(closes)
    gains=[]; losses=[]
    for i in range(1,len(closes)):
        ch = closes[i]-closes[i-1]
        gains.append(max(ch,0.0)); losses.append(-min(ch,0.0))
    ag = sum(gains[:period])/period; al = sum(losses[:period])/period
    rsis=[50.0]*period
    for i in range(period,len(gains)):
        ag=(ag*(period-1)+gains[i])/period; al=(al*(period-1)+losses[i])/period
        rs = (ag/al) if al!=0 else 999.0
        rsis.append(100.0 - 100.0/(1.0+rs))
    return [50.0]+rsis

def features_from_candles(candles: List[List[str]]) -> Dict[str, float]:
    closes=[float(c[2]) for c in candles]
    highs =[float(c[3]) for c in candles]
    lows  =[float(c[4]) for c in candles]
    if len(closes)<60:
        return {"ema20":0,"ema50":0,"atrp":0.3,"rsi":50,"hi":max(closes),"lo":min(closes)}
    ema20=ema(closes,20)[-1]; ema50=ema(closes,50)[-1]
    # ATR% approx
    rng=[(highs[i]-lows[i])/(closes[i] or 1)*100.0 for i in range(max(0,len(closes)-20), len(closes))]
    atrp = sum(rng)/len(rng) if rng else 0.3
    return {"ema20":ema20,"ema50":ema50,"atrp":max(0.05,min(5.0,atrp)),
            "rsi":rsi(closes,14)[-1], "hi":max(highs[-STATE.risk_cfg['br_break_lookback']:]),
            "lo":min(lows[-STATE.risk_cfg['br_break_lookback']:]), "close":closes[-1],
            "prev_close":closes[-2] if len(closes)>=2 else closes[-1],
            "high":highs[-1],"low":lows[-1]}

# -----------------------------
# TRADING HELPERS
# -----------------------------
def _fee(usdt: float) -> float:
    return usdt * STATE.fee_side

def _enter_leg(sym: str, side: str, price: float, usd_size: float, st: SymState):
    qty = usd_size/price if price>0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    if not st.pos:
        st.pos = Position(side=side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0)
        st.next_step_pct = STATE.grid_cfg["step_min"]
        st.pos.high_water=price; st.pos.low_water=price
        st.pos.trailing_active=False; st.pos.lock_price=0.0
        st.pos.opened_at = datetime.now(timezone.utc)
    else:
        st.pos.legs.append(leg); st.pos.safety_count+=1
        total_qty = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs)/(total_qty or 1.0)
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count*STATE.grid_cfg["tp_bonus"]
    st.pos.target_price = (st.pos.avg_price*(1+tp/100) if side=="LONG" else st.pos.avg_price*(1-tp/100))
    log_mock_trade({"time":datetime.now(timezone.utc).isoformat(),"symbol":sym,"action":"ENTRY",
                    "side":side,"price":round(price,6),"qty":round(qty,8),
                    "avg_price":round(st.pos.avg_price,6),"gross":"","fee_in":"","fee_out":"","net":"",
                    "info":f"size_usdt={usd_size}"})

def _exit_all(sym: str, price: float, st: SymState) -> float:
    if not st.pos: return 0.0
    gross=0.0; fee_in=0.0
    for leg in st.pos.legs:
        usd_in = leg.qty*leg.price
        fee_in += _fee(usd_in)
        if st.pos.side=="LONG": gross += leg.qty*(price-leg.price)
        else: gross += leg.qty*(leg.price-price)
    fee_out = _fee(st.pos.qty_total()*price)
    net = gross - fee_in - fee_out
    st.realized_pnl_net += net
    st.trades_log.append({"time":datetime.now(timezone.utc).isoformat(),"symbol":sym,"side":st.pos.side,
                          "avg_price":st.pos.avg_price,"exit_price":price,"gross":round(gross,6),
                          "fee_in":round(fee_in,6),"fee_out":round(fee_out,6),"net":round(net,6),
                          "safety_legs":st.pos.safety_count})
    log_mock_trade({"time":datetime.now(timezone.utc).isoformat(),"symbol":sym,"action":"EXIT",
                    "side":st.pos.side,"price":round(price,6),"qty":round(st.pos.qty_total(),8),
                    "avg_price":round(st.pos.avg_price,6),"gross":round(gross,6),
                    "fee_in":round(fee_in,6),"fee_out":round(fee_out,6),"net":round(net,6),
                    "info":f"safety_legs={st.pos.safety_count}"})
    st.pos=None; st.next_step_pct=STATE.grid_cfg["step_min"]
    return net

# -----------------------------
# BREAK & RETEST LOGIK
# -----------------------------
def _trend_bias(feats: Dict[str,float]) -> float:
    ema20, ema50 = feats["ema20"], feats["ema50"]
    if ema20==0 or ema50==0: return 0.0
    return 1.0 if ema20>ema50 else -1.0

def _br_score(feats: Dict[str,float], lookback_hi: float, lookback_lo: float) -> Tuple[float,str,float]:
    """Return (score, dir, break_level)."""
    ema20, ema50 = feats["ema20"], feats["ema50"]
    c  = feats["close"]; pc = feats["prev_close"]
    hi = lookback_hi; lo = lookback_lo
    dir_sign = 0
    # Break villkor: stäng över lokalt högsta + över EMA20/50 i bullish trend
    long_break  = (ema20>ema50) and (c>hi) and (pc>ema20)
    short_break = (ema20<ema50) and (c<lo) and (pc<ema20)
    if long_break: dir_sign = 1
    elif short_break: dir_sign = -1
    # score = hur långt över hi/under lo i % + ema-skillnad
    if dir_sign==1:
        overshoot = (c-hi)/(hi or 1)*100.0
        ema_gap = (ema20-ema50)/(ema50 or 1)*100.0
        score = overshoot + 0.5*ema_gap
        return (score,"LONG",hi)
    elif dir_sign==-1:
        undershoot = (lo-c)/(lo or 1)*100.0
        ema_gap = (ema50-ema20)/(ema20 or 1)*100.0
        score = undershoot + 0.5*ema_gap
        return (score,"SHORT",lo)
    return (0.0,"",0.0)

def _retest_ok(feats: Dict[str,float], plan: BRPlan, pull_pct: float, reject_wick_pct: float) -> bool:
    c=feats["close"]; low=feats["low"]; high=feats["high"]; ema20=feats["ema20"]
    lvl=plan.level
    if plan.direction=="LONG":
        pulled = (lvl - low)/(lvl or 1)*100.0 >= pull_pct
        # rejection: stänger över ema20 och wicken visar avvisning
        wick = ( (c - low) / (c or 1) * 100.0 ) >= reject_wick_pct
        return pulled and (c>ema20) and wick
    else:
        pulled = (high - lvl)/(lvl or 1)*100.0 >= pull_pct
        wick = ( (high - c) / (c or 1) * 100.0 ) >= reject_wick_pct
        return pulled and (c<ema20) and wick

# -----------------------------
# ENGINE
# -----------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                # ===== handle entries (create plan -> retest -> entry) =====
                open_syms = [s for s,st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < int(STATE.risk_cfg["max_pos"]):
                    for sym in STATE.symbols:
                        st = STATE.per_sym[sym]
                        if st.pos: continue
                        tf0 = STATE.tfs[0] if STATE.tfs else "5m"
                        try:
                            kl = await fetch_klines(sym, tf0, 120)
                        except Exception:
                            continue
                        feats = features_from_candles(kl)
                        # filter: ATR & trend
                        if STATE.risk_cfg.get("trend_filter", True):
                            if _trend_bias(feats) < 0 and not STATE.risk_cfg.get("allow_shorts", False):
                                # om bara LONG tillåts och trend är ned → hoppa
                                pass
                        if feats["atrp"] < float(STATE.risk_cfg.get("atr_min", 0.25)):
                            # för låg vol → hoppa
                            pass
                        # plan expired?
                        if st.plan and datetime.now(timezone.utc) > st.plan.expire_at:
                            st.plan=None
                        # finns en plan? testa retest
                        if st.plan:
                            if _retest_ok(feats, st.plan,
                                          float(STATE.risk_cfg["br_retest_pull_pct"]),
                                          float(STATE.risk_cfg["br_reject_wick_pct"])):
                                price = feats["close"]
                                usd = float(STATE.risk_cfg.get("size", STATE.position_size))
                                _enter_leg(sym, st.plan.direction, price, usd, st)
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id,
                                        f"🟢 ENTRY {sym} {st.plan.direction} @ {price:.4f} | retest lvl {st.plan.level:.4f}")
                                st.plan=None
                                open_syms.append(sym)
                                if len(open_syms) >= int(STATE.risk_cfg["max_pos"]):
                                    break
                        else:
                            # inga plan → se om vi får ett nytt break
                            score, direction, level = _br_score(feats, feats["hi"], feats["lo"])
                            if direction and score > STATE.entry_threshold:
                                expire = datetime.now(timezone.utc) + timedelta(minutes=int(STATE.risk_cfg["br_retest_window_m"]))
                                st.plan = BRPlan(direction=direction, level=level, created_at=datetime.now(timezone.utc), expire_at=expire)
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id,
                                        f"🧭 PLAN {sym} {direction} break @ {level:.4f} | score={score:.2f} | retest {STATE.risk_cfg['br_retest_pull_pct']}% inom {STATE.risk_cfg['br_retest_window_m']}m")

                # ===== manage exits / dca / risk =====
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos: continue
                    tf0 = STATE.tfs[0] if STATE.tfs else "5m"
                    try:
                        kl = await fetch_klines(sym, tf0, 3)
                    except Exception:
                        continue
                    feats = features_from_candles(kl)
                    price = feats["close"]; avg = st.pos.avg_price
                    move_pct = (price-avg)/(avg or 1)*100.0
                    atrp = feats["atrp"]

                    # trail width
                    trail_w = max(float(STATE.risk_cfg.get("trail_min",2.8)), atrp*float(STATE.risk_cfg.get("trail_mult",7.0)))
                    st.pos.high_water = max(st.pos.high_water, price)
                    st.pos.low_water  = min(st.pos.low_water,  price)
                    if st.pos.side=="LONG":
                        trail_level = st.pos.high_water*(1.0 - trail_w/100.0)
                    else:
                        trail_level = st.pos.low_water*(1.0 + trail_w/100.0)

                    # HYBRID start
                    if bool(STATE.risk_cfg.get("hyb_on", True)) and not st.pos.trailing_active:
                        start = float(STATE.risk_cfg.get("hyb_start",2.2))
                        lock  = float(STATE.risk_cfg.get("hyb_lock",0.25))
                        if (st.pos.side=="LONG" and move_pct>=start) or (st.pos.side=="SHORT" and -move_pct>=start):
                            st.pos.trailing_active=True
                            st.pos.lock_price = (st.pos.avg_price*(1+lock/100) if st.pos.side=="LONG"
                                                 else st.pos.avg_price*(1-lock/100))
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔒 TRAIL ON {sym} | start={start:.2f}% lock={lock:.2f}% trail≈{trail_w:.2f}%")

                    # Early cut
                    if bool(STATE.risk_cfg.get("early_cut", True)):
                        if datetime.now(timezone.utc) - st.pos.opened_at <= timedelta(minutes=int(STATE.risk_cfg.get("early_cut_window_m",3))):
                            thr = float(STATE.risk_cfg.get("early_cut_thr",0.5))
                            hit = (st.pos.side=="LONG" and move_pct<=-thr) or (st.pos.side=="SHORT" and move_pct>=thr)
                            if hit:
                                net=_exit_all(sym, price, st)
                                if STATE.chat_id:
                                    mark="✅" if net>=0 else "❌"
                                    await app.bot.send_message(STATE.chat_id, f"⚡ EARLY-CUT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}")
                                continue

                    # Trail/TP exit
                    did_exit=False
                    if bool(STATE.risk_cfg.get("trail_on", True)) and st.pos.trailing_active:
                        stop = trail_level
                        if st.pos.lock_price>0:
                            stop = max(stop, st.pos.lock_price) if st.pos.side=="LONG" else min(stop, st.pos.lock_price)
                        if (st.pos.side=="LONG" and price<=stop) or (st.pos.side=="SHORT" and price>=stop):
                            net=_exit_all(sym, price, st); did_exit=True
                            if STATE.chat_id:
                                mark="✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id,
                                    f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark} | stop={stop:.4f} trail={trail_w:.2f}%")
                    if did_exit: continue
                    if not st.pos.trailing_active:
                        if (st.pos.side=="LONG" and price>=st.pos.target_price) or (st.pos.side=="SHORT" and price<=st.pos.target_price):
                            net=_exit_all(sym, price, st)
                            if STATE.chat_id:
                                mark="✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id, f"🟤 EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}")
                            continue

                    # DCA (få – BR)
                    if st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        step = st.next_step_pct
                        need_dca = (st.pos.side=="LONG" and move_pct<=-step) or (st.pos.side=="SHORT" and move_pct>=step)
                        if need_dca:
                            base = float(STATE.risk_cfg.get("size", STATE.position_size))
                            usd = base*(STATE.grid_cfg["size_mult"]**st.pos.safety_count)
                            _enter_leg(sym, st.pos.side, price, usd, st)
                            st.next_step_pct = max(STATE.grid_cfg["step_min"], st.next_step_pct*STATE.grid_cfg["step_mult"])
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} | leg {st.pos.safety_count} | avg {st.pos.avg_price:.4f} | TP {st.pos.target_price:.4f}")

                    # Max SL
                    sl = float(STATE.risk_cfg.get("sl_max",0.6))
                    if sl>0:
                        hit = (st.pos.side=="LONG" and move_pct<=-sl) or (st.pos.side=="SHORT" and move_pct>=sl)
                        if hit:
                            net=_exit_all(sym, price, st)
                            if STATE.chat_id:
                                mark="✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id, f"⛔ MAX-SL {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}")
                            continue

                    # Max DD per trade (USDT)
                    max_dd=float(STATE.risk_cfg.get("max_dd_per_trade",0))
                    if max_dd>0 and st.pos.qty_total()>0:
                        unreal = (st.pos.qty_total()*max(0.0, st.pos.avg_price-price) if st.pos.side=="LONG"
                                  else st.pos.qty_total()*max(0.0, price-st.pos.avg_price))
                        if unreal >= max_dd:
                            net=_exit_all(sym, price, st)
                            if STATE.chat_id:
                                mark="✅" if net>=0 else "❌"
                                await app.bot.send_message(STATE.chat_id, f"🧯 MAX-DD EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}")

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try: await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except: pass
            await asyncio.sleep(5)

# -----------------------------
# UI HELPERS
# -----------------------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/mode br")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/threshold")],
        [KeyboardButton("/risk"), KeyboardButton("/export_csv")],
        [KeyboardButton("/pnl")]
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def status_text() -> str:
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    pos_lines=[]
    for s in STATE.symbols:
        st=STATE.per_sym[s]
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} → TP {st.pos.target_price:.4f} | legs {st.pos.safety_count} | trail={'ON' if st.pos.trailing_active else 'OFF'}")
        if st.plan:
            mins=int((st.plan.expire_at-datetime.now(timezone.utc)).total_seconds()/60)
            pos_lines.append(f"{s}: PLAN {st.plan.direction} lvl {st.plan.level:.4f} ({mins}m kvar)")
    g=STATE.grid_cfg; r=STATE.risk_cfg
    lines=[
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Strategi: Break & Retest",
        f"BR-threshold: {STATE.entry_threshold:.2f}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {r.get('size',STATE.position_size):.1f} USDT | Fee per sida: {STATE.fee_side:.4%}",
        (f"Grid: max_safety={g['max_safety']} step_mult={g['step_mult']} step_min%={g['step_min']} "
         f"size_mult={g['size_mult']} tp%={g['tp']} (+{g['tp_bonus']}%/safety)"),
        (f"Risk: dd={r['dd']}% | max_pos={r['max_pos']} | shorts={'ON' if r['allow_shorts'] else 'OFF'}\n"
         f"  trail={'ON' if r['trail_on'] else 'OFF'} (min={r['trail_min']}% x{r['trail_mult']}) | "
         f"hybrid={'ON' if r['hyb_on'] else 'OFF'} (start={r['hyb_start']}% lock={r['hyb_lock']}%)\n"
         f"  sl_max={r['sl_max']}% | early_cut={'ON' if r['early_cut'] else 'OFF'} "
         f"(thr={r['early_cut_thr']}%/{r['early_cut_window_m']}m) | max_dd={r['max_dd_per_trade']} USDT\n"
         f"  BR: lookback={r['br_break_lookback']} confirm={r['br_confirm_candles']} | "
         f"retest={r['br_retest_pull_pct']}%/{r['br_retest_window_m']}m | wick>={r['br_reject_wick_pct']}% | "
         f"trend_len={r['trend_len']} | atr_min={r['atr_min']}%"),
        f"PnL total (NET): {total:+.4f} USDT",
        "Queues/Pos: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

# -----------------------------
# TELEGRAM COMMANDS
# -----------------------------
tg_app = Application.builder().token(BOT_TOKEN).build()

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "🤖 Mp Bot v46 – Break & Retest\n/startat i mock-läge.\nTips: /mode br, /engine_on", reply_markup=reply_kb())
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

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip()
    parts = msg.split(" ",1)
    if len(parts)==2:
        tfs=[x.strip() for x in parts[1].split(",") if x.strip()]
        STATE.tfs=tfs
        await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.tfs)}", reply_markup=reply_kb())
    else:
        await tg_app.bot.send_message(STATE.chat_id, "Använd: /timeframe 5m,15m,30m", reply_markup=reply_kb())

async def cmd_threshold(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks)==1:
        await tg_app.bot.send_message(STATE.chat_id, f"Aktuellt BR-threshold: {STATE.entry_threshold:.2f}", reply_markup=reply_kb())
        return
    try:
        val=float(toks[1]); 
        if val<=0: raise ValueError()
        STATE.entry_threshold=val
        await tg_app.bot.send_message(STATE.chat_id, f"BR-threshold uppdaterad: {val:.2f}", reply_markup=reply_kb())
    except:
        await tg_app.bot.send_message(STATE.chat_id, "Fel värde. Ex: /threshold 1.60", reply_markup=reply_kb())

def _yesno(s:str)->bool: return s.lower() in ("1","true","on","yes","y")

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks)==4 and toks[0]=="/risk" and toks[1]=="set":
        key,val=toks[2],toks[3]
        if key in STATE.risk_cfg:
            try:
                if key=="max_pos": STATE.risk_cfg[key]=int(val)
                elif key in ("allow_shorts","trail_on","hyb_on","early_cut","trend_filter"):
                    STATE.risk_cfg[key]=_yesno(val)
                elif key in ("br_break_lookback","br_confirm_candles","br_retest_window_m","early_cut_window_m","trend_len"):
                    STATE.risk_cfg[key]=int(float(val))
                elif key in ("size","sl_max","br_retest_pull_pct","br_reject_wick_pct","atr_min",
                             "hyb_start","hyb_lock","trail_min","trail_mult","early_cut_thr"):
                    STATE.risk_cfg[key]=float(val)
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
    txt=update.message.text.strip(); toks=txt.split()
    if len(toks)>=3 and toks[0]=="/symbols":
        if toks[1]=="add":
            sym=toks[2].upper()
            if sym not in STATE.symbols:
                STATE.symbols.append(sym); STATE.per_sym[sym]=SymState()
                await tg_app.bot.send_message(STATE.chat_id, f"La till: {sym}", reply_markup=reply_kb())
            else:
                await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns redan.", reply_markup=reply_kb())
            return
        if toks[1]=="remove":
            sym=toks[2].upper()
            if sym in STATE.symbols:
                STATE.symbols=[s for s in STATE.symbols if s!=sym]; STATE.per_sym.pop(sym,None)
                await tg_app.bot.send_message(STATE.chat_id, f"Tog bort: {sym}", reply_markup=reply_kb())
            else:
                await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns ej.", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, "Använd:\n/symbols add LINK-USDT\n/symbols remove LINK-USDT", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    total=sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    lines=[f"📈 PnL total (NET): {total:+.4f} USDT"]
    for s in STATE.symbols:
        lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl_net:+.4f} USDT")
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    rows=[["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r["time"], r["symbol"], r["side"], r["avg_price"], r["exit_price"],
                         r["gross"], r["fee_in"], r["fee_out"], r["net"], r["safety_legs"]])
    if len(rows)==1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb()); return
    buf=io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="trades_v46_br.csv", caption="Export CSV")

async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id=update.effective_chat.id
    sub=(context.args[0].lower() if context.args else "")
    if sub=="br":
        # “säkrare” preset
        STATE.entry_threshold=1.60
        STATE.risk_cfg.update(dict(
            max_pos=3, size=70.0, allow_shorts=False,
            trail_on=True, trail_mult=7.0, trail_min=2.8,
            hyb_on=True, hyb_start=2.2, hyb_lock=0.25,
            sl_max=0.60, early_cut=True, early_cut_thr=0.50, early_cut_window_m=3,
            br_break_lookback=12, br_confirm_candles=2, br_retest_window_m=15,
            br_retest_pull_pct=0.25, br_reject_wick_pct=0.10,
            trend_filter=True, trend_len=45, trend_bias=0.80, atr_min=0.25
        ))
        await tg_app.bot.send_message(STATE.chat_id, "✅ Mode satt: br (Break & Retest)", reply_markup=reply_kb()); return
    await tg_app.bot.send_message(STATE.chat_id, "Använd: /mode br", reply_markup=reply_kb())

# Handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("threshold", cmd_threshold))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("mode", cmd_mode))

# -----------------------------
# FASTAPI APP
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
    total=sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return f"MP Bot v46 BR OK | engine_on={STATE.engine_on} | thr={STATE.entry_threshold:.2f} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total=sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {"ok":True,"engine_on":STATE.engine_on,"threshold":STATE.entry_threshold,
            "tfs":STATE.tfs,"symbols":STATE.symbols,"pnl_total":round(total,6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
# ----------------------------- END -----------------------------
