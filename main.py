# main_v42.py
# ------------------------------------------------------------
# MP Bot – v42
# - Byggd på v41 (reply keyboard, FastAPI webhook)
# - Trendfilter (EMA200), Volymfilter
# - Steg-TP: 50% ut på TP1, resten trailas
# - AI används endast som filter (kan av/på)
# - Tighter DD default, presets & kommandon
# ------------------------------------------------------------
import os
import io
import csv
import math
import pickle
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

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
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))

# Grid/DCA (vi håller det enkelt men tajtare)
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "1"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.25"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.6"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.20"))

# Exit & risk
TP1_PCT = float(os.getenv("TP1_PCT", "0.40"))          # 50% ut här
TRAIL_TRIGGER = float(os.getenv("TRAIL_TRIGGER", "0.40"))
TRAIL_STEP = float(os.getenv("TRAIL_STEP", "0.18"))
TRAIL_BE = float(os.getenv("TRAIL_BE", "0.05"))

DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "1.0"))   # tajtare default

# Filters
USE_TREND = True
USE_VOLUME = True
AI_FILTER_ON = True  # AI bara som filter
VOL_WIN = int(os.getenv("VOL_WIN", "20"))              # volymmedel fönster
TREND_EMA = 200

# AI model (filter)
DATA_DIR = os.getenv("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)
AI_MODEL_PATH = os.path.join(DATA_DIR, "ai_filter.pkl")

# KuCoin candles
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

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
    safety_count: int = 0
    # v42 exits
    half_taken: bool = False
    be_price: Optional[float] = None
    trail_anchor: Optional[float] = None
    trail_last_stop: Optional[float] = None

    def qty_total(self) -> float:
        return sum(l.qty for l in self.legs)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)
    next_step_pct: float = GRID_STEP_MIN_PCT
    cooldown_until_ms: int = 0

@dataclass
class EngineState:
    engine_on: bool = True
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
    ))
    risk_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        dd=DD_STOP_PCT,
        max_pos=MAX_OPEN_POS,
        allow_shorts=True
    ))
    filters: Dict[str, float] = field(default_factory=lambda: dict(
        min_atr=0.10,
        cool=5.0  # minutes
    ))
    strategy: str = "both"  # trend / fib / both
    ai_on: bool = True
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# AI filter weights (enkel linjär scorer), bara filter (ja/nej)
AI_W: Dict[str, List[float]] = {}  # per-symbol/tf: [w0, w_mom, w_ema, w_rsi]
AI_LR = 0.02

# -----------------------------
# UI
# -----------------------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/strategy"), KeyboardButton("/timeframe")],
        [KeyboardButton("/risk"), KeyboardButton("/trail")],
        [KeyboardButton("/filter"), KeyboardButton("/pnl")],
        [KeyboardButton("/symbols"), KeyboardButton("/export_csv")],
        [KeyboardButton("/ai_on"), KeyboardButton("/ai_off")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/preset active"), KeyboardButton("/preset stable")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -----------------------------
# DATA & INDICATORS
# -----------------------------
async def get_klines(symbol: str, tf: str, limit: int = 200):
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = list(reversed(r.json()["data"]))
    out = []
    for row in data[-limit:]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        vol = float(row[5]) if len(row) > 5 else 0.0
        out.append((t_ms, o, h, l, c, vol))
    return out

def ema(series: List[float], period: int) -> List[float]:
    if not series:
        return []
    k = 2/(period+1)
    out = []
    val = series[0]
    for x in series:
        val = (x - val)*k + val
        out.append(val)
    return out

def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period+1:
        return [50.0]*len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i]-closes[i-1]
        gains.append(max(ch,0.0)); losses.append(-min(ch,0.0))
    avg_g = sum(gains[:period])/period
    avg_l = sum(losses[:period])/period
    rsis = [50.0]*(period)
    for i in range(period, len(gains)):
        avg_g = (avg_g*(period-1)+gains[i])/period
        avg_l = (avg_l*(period-1)+losses[i])/period
        rs = 999 if avg_l==0 else avg_g/avg_l
        rsis.append(100 - 100/(1+rs))
    return [50.0]+rsis

def features(candles):
    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows  = [c[3] for c in candles]
    vols  = [c[5] for c in candles]
    if len(closes) < 210:
        return None
    ema200 = ema(closes, TREND_EMA)[-1]
    ema21  = ema(closes, 21)[-1]
    mom = (closes[-1]-closes[-6])/(closes[-6] or 1)*100
    rsi14 = rsi(closes, 14)[-1]
    atrp = sum((h-l)/(c or 1)*100 for (_,_,h,l,c,_) in candles[-20:])/20
    vol_mean = sum(vols[-VOL_WIN:])/VOL_WIN if len(vols)>=VOL_WIN else 0
    vol_last = vols[-1]
    # swings för fib (senaste 20)
    sw_high = max(highs[-20:]); sw_low = min(lows[-20:])
    return dict(ema200=ema200, ema21=ema21, mom=mom, rsi=rsi14,
                atrp=max(0.02, min(5.0, atrp)),
                vol_last=vol_last, vol_mean=vol_mean,
                sw_high=sw_high, sw_low=sw_low, close=closes[-1])

def ai_ok(sym: str, f: dict) -> bool:
    if not STATE.ai_on:
        return True
    key = sym
    if key not in AI_W:
        AI_W[key] = [0.0, 0.4, 0.3, 0.1]  # [bias, mom, ema_diff, rsi_dev]
    w0, wm, we, wr = AI_W[key]
    ema_diff = (f["ema21"]-f["ema200"])/(f["ema200"] or 1)*100
    rdev = (f["rsi"]-50)/50*100
    raw = w0 + wm*f["mom"] + we*ema_diff + wr*rdev
    return raw > -2.0  # bara filter: helt kass signal blockas

def ai_learn(sym: str, f: dict, pnl: float):
    # Enkel förstärkningsuppdatering
    if not STATE.ai_on:
        return
    key = sym
    if key not in AI_W:
        AI_W[key] = [0.0, 0.4, 0.3, 0.1]
    w0, wm, we, wr = AI_W[key]
    ema_diff = (f["ema21"]-f["ema200"])/(f["ema200"] or 1)*100
    rdev = (f["rsi"]-50)/50*100
    grad = [1.0, f["mom"], ema_diff, rdev]
    lr = AI_LR*(1 if pnl>=0 else -1)
    AI_W[key] = [w0+lr*grad[0], wm+lr*grad[1], we+lr*grad[2], wr+lr*grad[3]]

def save_ai() -> str:
    try:
        with open(AI_MODEL_PATH, "wb") as f:
            pickle.dump(AI_W, f)
        return f"AI (filter) sparad."
    except Exception as e:
        return f"Kunde inte spara AI: {e}"

def load_ai() -> str:
    global AI_W
    try:
        with open(AI_MODEL_PATH, "rb") as f:
            AI_W = pickle.load(f)
        return "AI (filter) laddad."
    except Exception as e:
        return f"Kunde inte ladda AI: {e}"

def _fee(usd: float) -> float:
    return usd*FEE_PER_SIDE

def _enter(sym: str, side: str, price: float, usd: float, st: SymState):
    qty = usd/price if price>0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, safety_count=0)
        st.next_step_pct = STATE.grid_cfg["step_min"]
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        tot = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs)/(tot or 1.0)
    return leg

def _exit_all(sym: str, price: float, st: SymState, reason="EXIT") -> float:
    if not st.pos: return 0.0
    gross = 0.0; fee_in=0.0
    for l in st.pos.legs:
        fee_in += _fee(l.qty*l.price)
        if st.pos.side=="LONG":
            gross += l.qty*(price-l.price)
        else:
            gross += l.qty*(l.price-price)
    fee_out = _fee(st.pos.qty_total()*price)
    net = gross - fee_in - fee_out
    st.realized_pnl_net += net
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym, "side": st.pos.side,
        "avg_price": st.pos.avg_price, "exit_price": price,
        "gross": round(gross,6), "fee_in": round(fee_in,6), "fee_out": round(fee_out,6),
        "net": round(net,6), "safety_legs": st.pos.safety_count, "reason": reason
    })
    st.pos = None
    st.next_step_pct = STATE.grid_cfg["step_min"]
    return net

def _scale_out_half(sym: str, price: float, st: SymState):
    """Sälj 50% och flytta SL till BE, sedan traila resten."""
    if not st.pos: return
    pos = st.pos
    qty = pos.qty_total()
    half = qty*0.5
    # räkna PnL för 50% exit
    # vi förenklar: anta halvan säljs från snittpriset
    gross_half = (price-pos.avg_price)*(half if pos.side=="LONG" else -half)
    gross_half = -gross_half  # korr: enklare att använda samma formel
    if pos.side=="LONG":
        gross_half = half*(price-pos.avg_price)
    else:
        gross_half = half*(pos.avg_price-price)
    fee_in = _fee(sum(l.qty*l.price for l in pos.legs))*0.5  # halva in-fee approx
    fee_out = _fee(half*price)
    net_half = gross_half - fee_in - fee_out
    st.realized_pnl_net += net_half
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym, "side": pos.side,
        "avg_price": pos.avg_price, "exit_price": price,
        "gross": round(gross_half,6), "fee_in": round(fee_in,6),
        "fee_out": round(fee_out,6), "net": round(net_half,6),
        "safety_legs": pos.safety_count, "reason": "TP1-50%"
    })
    # reducera legs proportionellt
    remain = qty - half
    factor = (remain/(qty or 1))
    new_legs = []
    for l in pos.legs:
        new_legs.append(TradeLeg(side=l.side, price=l.price, qty=l.qty*factor, time=l.time))
    pos.legs = new_legs
    pos.half_taken = True
    pos.be_price = pos.avg_price  # flytta SL till BE
    pos.trail_anchor = price
    pos.trail_last_stop = pos.avg_price

def _should_long(f: dict) -> bool:
    if USE_TREND and not (f["close"] > f["ema200"]):
        return False
    if USE_VOLUME and not (f["vol_last"] > f["vol_mean"]):
        return False
    return True

def _should_short(f: dict) -> bool:
    if USE_TREND and not (f["close"] < f["ema200"]):
        return False
    if USE_VOLUME and not (f["vol_last"] > f["vol_mean"]):
        return False
    return True

def _fib_entry_ok(side: str, f: dict) -> bool:
    # enkel fib pullback: 0.382-0.618 zon
    high, low, close = f["sw_high"], f["sw_low"], f["close"]
    if high <= low: return False
    diff = high - low
    f382 = low + 0.382*diff
    f618 = low + 0.618*diff
    if side=="LONG":
        return f382 <= close <= f618 and close > f["ema21"]
    else:
        # för short spegelvänt
        f382s = high - 0.382*diff
        f618s = high - 0.618*diff
        return f618s <= close <= f382s and close < f["ema21"]

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                open_syms = [s for s,st in STATE.per_sym.items() if st.pos]
                can_open = len(open_syms) < STATE.risk_cfg["max_pos"]
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    # Hantera öppna först (exit-logik)
                    tf0 = STATE.tfs[0] if STATE.tfs else "1m"
                    try:
                        kl = await get_klines(sym, tf0, limit=220)
                    except:
                        continue
                    f = features(kl)
                    if not f: 
                        continue
                    price = f["close"]

                    # exits
                    if st.pos:
                        avg = st.pos.avg_price
                        move_pct = (price-avg)/avg*100 if st.pos.side=="LONG" else (avg-price)/avg*100

                        # DD-stop
                        if move_pct <= -STATE.risk_cfg["dd"]:
                            net = _exit_all(sym, price, st, reason="DD")
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"⛔ STOP {sym} @ {price:.4f} | Net {net:+.4f}")
                            # AI feedback
                            ai_learn(sym, f, net)
                            continue

                        # TP1: 50% ut + BE
                        if (not st.pos.half_taken) and move_pct >= TP1_PCT:
                            _scale_out_half(sym, price, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"✅ TP1 {sym} 50% @ {price:.4f} (BE aktiv)")

                        # Trailing för resterande
                        if st.pos and st.pos.half_taken:
                            # uppdatera anchor
                            if st.pos.side=="LONG":
                                if price > (st.pos.trail_anchor or price):
                                    st.pos.trail_anchor = price
                                # räkna nytt stopp
                                if st.pos.trail_anchor:
                                    trail_stop = st.pos.trail_anchor*(1 - TRAIL_STEP/100)
                                    trail_stop = max(trail_stop, st.pos.be_price or st.pos.avg_price)
                                    st.pos.trail_last_stop = trail_stop
                                    if price <= trail_stop:
                                        net = _exit_all(sym, price, st, reason="TRAIL")
                                        if STATE.chat_id:
                                            await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f}")
                                        ai_learn(sym, f, net)
                                        continue
                            else:
                                if price < (st.pos.trail_anchor or price):
                                    st.pos.trail_anchor = price
                                if st.pos.trail_anchor:
                                    trail_stop = st.pos.trail_anchor*(1 + TRAIL_STEP/100)
                                    trail_stop = min(trail_stop, st.pos.be_price or st.pos.avg_price)
                                    st.pos.trail_last_stop = trail_stop
                                    if price >= trail_stop:
                                        net = _exit_all(sym, price, st, reason="TRAIL")
                                        if STATE.chat_id:
                                            await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f}")
                                        ai_learn(sym, f, net)
                                        continue

                        # DCA (tajt)
                        step = st.next_step_pct
                        need_dca = (move_pct <= -step) if st.pos.side=="LONG" else (move_pct <= -step)
                        if need_dca and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                            usd = STATE.position_size * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
                            _enter(sym, st.pos.side, price, usd, st)
                            st.next_step_pct = max(STATE.grid_cfg["step_min"], st.next_step_pct*STATE.grid_cfg["step_mult"])
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} (leg {st.pos.safety_count})")
                        continue  # gå till nästa symbol

                    # Inga öppna → kan vi öppna?
                    if not can_open:
                        continue

                    # filter: ATR & cooldown
                    if f["atrp"] < STATE.filters["min_atr"]:
                        continue
                    now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
                    if now_ms < st.cooldown_until_ms:
                        continue

                    # STRATEGIER
                    want_long = want_short = False
                    if STATE.strategy in ("trend", "both"):
                        if _should_long(f):  want_long = True
                        if STATE.risk_cfg["allow_shorts"] and _should_short(f): want_short = True
                    if STATE.strategy in ("fib", "both"):
                        want_long = want_long or _fib_entry_ok("LONG", f)
                        if STATE.risk_cfg["allow_shorts"]:
                            want_short = want_short or _fib_entry_ok("SHORT", f)

                    # AI filter
                    if AI_FILTER_ON and not ai_ok(sym, f):
                        want_long = want_short = False

                    # välj sida
                    side = None
                    if want_long and not want_short:
                        side = "LONG"
                    elif want_short and not want_long:
                        side = "SHORT"
                    elif want_long and want_short:
                        # välj enligt trend
                        side = "LONG" if f["close"] > f["ema200"] else "SHORT"

                    if side:
                        _enter(sym, side, f["close"], STATE.position_size, st)
                        st.cooldown_until_ms = now_ms + int(STATE.filters["cool"]*60*1000)
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🟢 ENTRY {sym} {side} @ {f['close']:.4f} | TP1 {TP1_PCT:.2f}% | DD {STATE.risk_cfg['dd']:.2f}%")
                        open_syms.append(sym)
                        can_open = len(open_syms) < STATE.risk_cfg["max_pos"]

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

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
            extra = []
            if st.pos.half_taken:
                extra.append("TP1✅")
                if st.pos.trail_last_stop:
                    extra.append(f"TS:{st.pos.trail_last_stop:.4f}")
            pos_lines.append(f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} legs {st.pos.safety_count} {' '.join(extra)}")
    return "\n".join([
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Strategy: {STATE.strategy} | AI-filter: {'ON' if STATE.ai_on else 'OFF'}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} | Fee: {STATE.fee_side:.4%}",
        f"Risk: dd={STATE.risk_cfg['dd']}% | max_pos={STATE.risk_cfg['max_pos']} | shorts={'ON' if STATE.risk_cfg['allow_shorts'] else 'OFF'}",
        f"Filters: atr>={STATE.filters['min_atr']} | cool={STATE.filters['cool']}m | trend(EMA{TREND_EMA})={'ON' if USE_TREND else 'OFF'} vol={'ON' if USE_VOLUME else 'OFF'}",
        f"TP1={TP1_PCT}% trail trig={TRAIL_TRIGGER}% step={TRAIL_STEP}% BE={TRAIL_BE}%",
        f"PnL total (NET): {total:+.4f}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ])

async def send_status(chat_id: int):
    await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v42 – redo ✅", reply_markup=reply_kb())
    await send_status(STATE.chat_id)

async def cmd_status(update: Update, _): STATE.chat_id=update.effective_chat.id; await send_status(STATE.chat_id)
async def cmd_engine_on(update: Update, _): STATE.chat_id=update.effective_chat.id; STATE.engine_on=True; await send_status(STATE.chat_id)
async def cmd_engine_off(update: Update, _): STATE.chat_id=update.effective_chat.id; STATE.engine_on=False; await send_status(STATE.chat_id)

async def cmd_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    mode = " ".join(update.message.text.split()[1:]).strip().lower()
    if mode in ("trend", "fib", "both"):
        STATE.strategy = mode
        await tg_app.bot.send_message(STATE.chat_id, f"Strategy: {mode}", reply_markup=reply_kb())
    else:
        await tg_app.bot.send_message(STATE.chat_id, "Använd: /strategy trend|fib|both", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    parts = update.message.text.split(" ", 1)
    if len(parts)==2:
        tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes: {', '.join(STATE.tfs)}", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, "Använd: /timeframe 1m,3m,5m", reply_markup=reply_kb())

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.split()
    if len(toks)==4 and toks[1]=="set":
        key, val = toks[2], toks[3]
        if key in STATE.risk_cfg:
            try:
                if key=="max_pos": STATE.risk_cfg[key]=int(val)
                elif key=="allow_shorts": STATE.risk_cfg[key]=val.lower() in ("1","on","true","yes")
                else: STATE.risk_cfg[key]=float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb())
                return
            except: pass
    r=STATE.risk_cfg
    await tg_app.bot.send_message(STATE.chat_id, f"Risk:\n dd={r['dd']}% | max_pos={r['max_pos']} | shorts={'ON' if r['allow_shorts'] else 'OFF'}\nEx: /risk set dd 1.0",
                                  reply_markup=reply_kb())

async def cmd_trail(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    toks=update.message.text.split()
    global TRAIL_TRIGGER, TRAIL_STEP, TRAIL_BE
    if len(toks)==4 and toks[1]=="set":
        try:
            if toks[2]=="trigger": TRAIL_TRIGGER=float(toks[3])
            elif toks[2]=="step": TRAIL_STEP=float(toks[3])
            elif toks[2]=="be": TRAIL_BE=float(toks[3])
            await tg_app.bot.send_message(STATE.chat_id, "Trail uppdaterad.", reply_markup=reply_kb()); return
        except: ...
    await tg_app.bot.send_message(STATE.chat_id, f"Trail:\n trigger={TRAIL_TRIGGER}% step={TRAIL_STEP}% be={TRAIL_BE}%\nEx: /trail set trigger 0.4",
                                  reply_markup=reply_kb())

async def cmd_filter(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    toks=update.message.text.split()
    if len(toks)==4 and toks[1]=="set":
        key,val=toks[2],toks[3]
        if key in STATE.filters:
            try:
                STATE.filters[key]=float(val)
                await tg_app.bot.send_message(STATE.chat_id,"Filter uppdaterat.", reply_markup=reply_kb()); return
            except: ...
    await tg_app.bot.send_message(STATE.chat_id,f"Filter:\n min_atr={STATE.filters['min_atr']} | cool={STATE.filters['cool']}m\nEx: /filter set cool 3",
                                  reply_markup=reply_kb())

async def cmd_symbols(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    toks=update.message.text.split()
    if len(toks)>=3 and toks[0]=="/symbols":
        act=toks[1].lower(); sym=toks[2].upper()
        if act=="add":
            if sym not in STATE.symbols:
                STATE.symbols.append(sym); STATE.per_sym[sym]=SymState()
                await tg_app.bot.send_message(STATE.chat_id, f"La till: {sym}", reply_markup=reply_kb()); return
            else:
                await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns redan.", reply_markup=reply_kb()); return
        if act=="remove":
            if sym in STATE.symbols:
                STATE.symbols=[s for s in STATE.symbols if s!=sym]; STATE.per_sym.pop(sym,None)
                await tg_app.bot.send_message(STATE.chat_id, f"Tog bort: {sym}", reply_markup=reply_kb()); return
            else:
                await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns ej.", reply_markup=reply_kb()); return
    await tg_app.bot.send_message(STATE.chat_id,"Använd: /symbols add BTC-USDT | /symbols remove BTC-USDT", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    lines=[f"📈 PnL total (NET): {total:+.4f}"]
    for s in STATE.symbols: lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl_net:+.4f}")
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    rows=[["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs","reason"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r["time"],r["symbol"],r["side"],r["avg_price"],r["exit_price"],r["gross"],r["fee_in"],r["fee_out"],r["net"],r["safety_legs"],r.get("reason","")])
    if len(rows)==1:
        await tg_app.bot.send_message(STATE.chat_id,"Inga trades loggade.", reply_markup=reply_kb()); return
    buf=io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, io.BytesIO(buf.getvalue().encode("utf-8")), filename="trades_v42.csv", caption="Export CSV")

async def cmd_panic(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    closed=[]
    for s in STATE.symbols:
        st=STATE.per_sym[s]
        if st.pos:
            tf0=STATE.tfs[0] if STATE.tfs else "1m"
            try: kl=await get_klines(s, tf0, limit=1); px=kl[-1][4]
            except: continue
            net=_exit_all(s, px, st, reason="PANIC"); closed.append(f"{s}:{net:+.4f}")
    msg=" | ".join(closed) if closed else "Inga positioner."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic: {msg}", reply_markup=reply_kb())

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    for s in STATE.symbols: STATE.per_sym[s].realized_pnl_net=0.0; STATE.per_sym[s].trades_log.clear()
    await tg_app.bot.send_message(STATE.chat_id,"PnL återställd.", reply_markup=reply_kb())

async def cmd_ai_on(update: Update, _): STATE.chat_id=update.effective_chat.id; STATE.ai_on=True; await tg_app.bot.send_message(STATE.chat_id,"AI-filter: ON", reply_markup=reply_kb())
async def cmd_ai_off(update: Update, _): STATE.chat_id=update.effective_chat.id; STATE.ai_on=False; await tg_app.bot.send_message(STATE.chat_id,"AI-filter: OFF", reply_markup=reply_kb())
async def cmd_save_ai(update: Update, _): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id, save_ai(), reply_markup=reply_kb())
async def cmd_load_ai(update: Update, _): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id, load_ai(), reply_markup=reply_kb())

async def cmd_trivia_preset(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    txt=update.message.text.strip().lower()
    if "active" in txt:
        # mycket aktiv men skyddad
        STATE.filters["min_atr"]=0.10; STATE.filters["cool"]=5
        STATE.risk_cfg["dd"]=1.0
        global TRAIL_TRIGGER, TRAIL_STEP, TRAIL_BE
        TRAIL_TRIGGER=0.40; TRAIL_STEP=0.18; TRAIL_BE=0.05
        await tg_app.bot.send_message(STATE.chat_id,"Preset ACTIVE laddad.", reply_markup=reply_kb())
    elif "stable" in txt:
        STATE.filters["min_atr"]=0.15; STATE.filters["cool"]=8
        STATE.risk_cfg["dd"]=0.9
        TRAIL_TRIGGER=0.55; TRAIL_STEP=0.22; TRAIL_BE=0.05
        await tg_app.bot.send_message(STATE.chat_id,"Preset STABLE laddad.", reply_markup=reply_kb())
    else:
        await tg_app.bot.send_message(STATE.chat_id,"/preset active | /preset stable", reply_markup=reply_kb())

# Handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("strategy", cmd_strategy))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("trail", cmd_trail))
tg_app.add_handler(CommandHandler("filter", cmd_filter))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("ai_on", cmd_ai_on))
tg_app.add_handler(CommandHandler("ai_off", cmd_ai_off))
tg_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
tg_app.add_handler(CommandHandler("load_ai", cmd_load_ai))
tg_app.add_handler(CommandHandler("preset", cmd_trivia_preset))

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
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return f"MP Bot v42 OK | engine_on={STATE.engine_on} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {"ok": True, "engine_on": STATE.engine_on, "tfs": STATE.tfs, "symbols": STATE.symbols, "pnl_total": round(total,6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
