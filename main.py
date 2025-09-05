# main_v42.py  — MP Bot v42
# - Based on v41 (reply keyboard, mock trading, grid/DCA, TP)
# - Adds: AI toggle, Trend+Fib strategier, trailing+BE, hård DD=1.5%
# - Fler entries (lägre trösklar), stabil engine-loop, AI sparning i ./data

import os, io, csv, math, pickle, asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------- ENV ----------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
                   .replace(" ","")).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ","")).split(",")

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))  # 0.1%
MAX_OPEN_POS = int(os.getenv("MAX_POS", "8"))

# Grid/DCA defaults (mer aktiv)
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "4"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.10"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.5"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.6"))
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.25"))
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.05"))

# Risk
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "1.5"))  # hård stop
ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "on").lower() in ("on","true","1")

# Trailing
TRAIL_TRIGGER = float(os.getenv("TRAIL_TRIGGER", "0.6"))
TRAIL_STEP = float(os.getenv("TRAIL_STEP", "0.20"))
TRAIL_BE = float(os.getenv("TRAIL_BE", "0.10"))

# AI
AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "./data/ai_model.pkl")
os.makedirs(os.path.dirname(AI_MODEL_PATH) or ".", exist_ok=True)

# KuCoin klines
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1hour"}

# ---------- STATE ----------
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
    high_water: float = 0.0  # för trailing LONG (högsta pris)
    low_water: float = 0.0   # för trailing SHORT (lägsta pris)
    be_set: bool = False

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
    strategy: str = "trend"  # "trend" eller "fib"
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
    per_sym: Dict[str, SymState] = field(default_factory=dict)
    position_size: float = POSITION_SIZE_USDT
    fee_side: float = FEE_PER_SIDE
    grid_cfg: Dict[str,float] = field(default_factory=lambda: dict(
        max_safety=GRID_MAX_SAFETY, step_mult=GRID_STEP_MULT, step_min=GRID_STEP_MIN_PCT,
        size_mult=GRID_SIZE_MULT, tp=GRID_TP_PCT, tp_bonus=TP_SAFETY_BONUS
    ))
    risk_cfg: Dict[str,float|int|bool] = field(default_factory=lambda: dict(
        dd=DD_STOP_PCT, max_pos=MAX_OPEN_POS, allow_shorts=ALLOW_SHORTS
    ))
    trail_cfg: Dict[str,float] = field(default_factory=lambda: dict(
        trigger=TRAIL_TRIGGER, step=TRAIL_STEP, be=TRAIL_BE
    ))
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ---- AI (enkel linjär scorer, valbar) ----
AI_WEIGHTS: Dict[Tuple[str,str], List[float]] = {}
AI_LEARN_RATE = 0.05

def save_ai() -> str:
    try:
        with open(AI_MODEL_PATH, "wb") as f:
            pickle.dump(AI_WEIGHTS, f)
        return f"AI sparad: {AI_MODEL_PATH}"
    except Exception as e:
        return f"Fel vid sparning: {e}"

def load_ai() -> str:
    global AI_WEIGHTS
    try:
        with open(AI_MODEL_PATH, "rb") as f:
            AI_WEIGHTS = pickle.load(f)
        return f"AI laddad från: {AI_MODEL_PATH}"
    except Exception as e:
        return f"Kunde inte ladda AI: {e}"

# ---------- INDICATORS ----------
async def get_klines(symbol: str, tf: str, limit: int = 120):
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = list(reversed(r.json()["data"]))
    out=[]
    for row in data[-limit:]:
        t_ms=int(row[0])*1000
        o=float(row[1]); c=float(row[2]); h=float(row[3]); l=float(row[4])
        out.append((t_ms,o,h,l,c))
    return out

def ema(series: List[float], period: int)->List[float]:
    if not series or period<=1: return series[:]
    k=2/(period+1); out=[]; v=series[0]
    for x in series: v = (x-v)*k+v; out.append(v)
    return out

def rsi(closes: List[float], period:int=14)->List[float]:
    if len(closes)<period+1: return [50.0]*len(closes)
    gains=[]; losses=[]
    for i in range(1,len(closes)):
        ch=closes[i]-closes[i-1]
        gains.append(max(ch,0.0)); losses.append(-min(ch,0.0))
    ag=sum(gains[:period])/period; al=sum(losses[:period])/period
    rsis=[50.0]*period
    for i in range(period,len(gains)):
        ag=(ag*(period-1)+gains[i])/period
        al=(al*(period-1)+losses[i])/period
        rs=999 if al==0 else ag/al
        rsis.append(100-(100/(1+rs)))
    return [50.0]+rsis

def features(candles)->Dict[str,float]:
    closes=[c[-1] for c in candles]
    if len(closes)<30: return {"mom":0.0,"ema_f":0.0,"ema_s":0.0,"rsi":50.0,"atrp":0.2}
    ef=ema(closes,9)[-1]; es=ema(closes,21)[-1]
    mom=(closes[-1]-closes[-6])/closes[-6]*100 if closes[-6]!=0 else 0.0
    rs=rsi(closes,14)[-1]
    rng=[(c[2]-c[3])/(c[-1] or 1)*100 for c in candles[-20:]]
    atrp=sum(rng)/len(rng) if rng else 0.2
    return {"mom":mom,"ema_f":ef,"ema_s":es,"rsi":rs,"atrp":max(0.02,min(3.0,atrp))}

def ai_score(symbol:str, tf:str, f:Dict[str,float])->float:
    key=(symbol,tf)
    if key not in AI_WEIGHTS: AI_WEIGHTS[key]=[0.0,0.7,0.6,0.1]
    w0,w_m,w_e,w_r=AI_WEIGHTS[key]
    ema_diff=(f["ema_f"]-f["ema_s"])/(f["ema_s"] or 1)*100
    rsi_dev=(f["rsi"]-50)/50
    raw=w0 + w_m*f["mom"] + w_e*ema_diff + w_r*rsi_dev*100
    return max(-10,min(10,raw))

def ai_learn(symbol:str, tf:str, f:Dict[str,float], pnl:float):
    key=(symbol,tf)
    if key not in AI_WEIGHTS: AI_WEIGHTS[key]=[0.0,0.7,0.6,0.1]
    w=AI_WEIGHTS[key]
    ema_diff=(f["ema_f"]-f["ema_s"])/(f["ema_s"] or 1)*100
    rsi_dev=(f["rsi"]-50)/50
    grad=[1.0, f["mom"], ema_diff, rsi_dev*100]
    lr=AI_LEARN_RATE*(1 if pnl>=0 else -1)
    AI_WEIGHTS[key]=[w[i]+lr*grad[i] for i in range(4)]

# ---------- TRADING HELPERS ----------
def _fee(usd:float)->float: return usd * FEE_PER_SIDE

def _set_target(st:SymState):
    tp = STATE.grid_cfg["tp"] + (st.pos.safety_count * STATE.grid_cfg["tp_bonus"])
    if st.pos.side=="LONG":
        st.pos.target_price = st.pos.avg_price * (1 + tp/100)
    else:
        st.pos.target_price = st.pos.avg_price * (1 - tp/100)

def _enter_leg(sym:str, side:str, price:float, usd:float, st:SymState)->TradeLeg:
    qty = usd/price if price>0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, safety_count=0)
        st.pos.high_water = price; st.pos.low_water = price
        st.pos.be_set=False
        st.next_step_pct = STATE.grid_cfg["step_min"]
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        tot=st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs)/(tot or 1.0)
    _set_target(st)
    return leg

def _exit_all(sym:str, price:float, st:SymState)->float:
    if not st.pos: return 0.0
    gross=0.0; fee_in=0.0
    for l in st.pos.legs:
        usd_in=l.qty*l.price; fee_in+=_fee(usd_in)
        if st.pos.side=="LONG":
            gross += l.qty*(price-l.price)
        else:
            gross += l.qty*(l.price-price)
    usd_out=st.pos.qty_total()*price
    fee_out=_fee(usd_out)
    net=gross-fee_in-fee_out
    st.realized_pnl_net += net
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym, "side": st.pos.side,
        "avg_price": st.pos.avg_price, "exit_price": price,
        "gross": round(gross,6), "fee_in": round(fee_in,6),
        "fee_out": round(fee_out,6), "net": round(net,6),
        "safety_legs": st.pos.safety_count
    })
    st.pos=None; st.next_step_pct=STATE.grid_cfg["step_min"]
    return net

# ---------- STRATEGIER ----------
def trend_vote(symbol:str, per_tf_feats:Dict[str,Dict[str,float]])->float:
    votes=0.0
    for tf,f in per_tf_feats.items():
        ema_diff=(f["ema_f"]-f["ema_s"])/(f["ema_s"] or 1)*100
        bias=(1.0 if ema_diff>0 else -1.0)*min(1.0, abs(ema_diff)/1.2)  # lite aggressiv
        votes += bias
        if STATE.ai_on: votes += ai_score(symbol, tf, f)/10.0
    return votes

def fib_vote(candles)->float:
    # enkel: mät senaste swing (20 bars). Om pullback 38.2–61.8 mot trenden → röst i trendens riktning
    closes=[c[-1] for c in candles[-20:]]
    if len(closes)<5: return 0.0
    hi=max(closes); lo=min(closes)
    uptrend = hi == closes[-1] or closes[-1] > closes[0]
    # fib nivåer
    fib382 = lo + 0.382*(hi-lo)
    fib500 = lo + 0.500*(hi-lo)
    fib618 = lo + 0.618*(hi-lo)
    px=closes[-1]
    vote=0.0
    if uptrend and fib382 <= px <= fib618: vote+=0.6
    if (not uptrend) and fib382 <= (hi - (px-lo)) <= fib618: vote-=0.6
    return vote

# ---------- ENGINE ----------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                open_syms=[s for s,st in STATE.per_sym.items() if st.pos]
                # ÖPPNA NYTT
                if len(open_syms) < int(STATE.risk_cfg["max_pos"]):
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos: continue
                        per_tf={}; candles_any=None; skip=False
                        for tf in STATE.tfs:
                            try:
                                kl=await asyncio.wait_for(get_klines(sym, tf, 80), timeout=8)
                                per_tf[tf]=features(kl)
                                if tf==STATE.tfs[0]: candles_any=kl
                            except Exception:
                                skip=True; break
                        if skip or not per_tf: continue
                        # rösta
                        vote = trend_vote(sym, per_tf) if STATE.strategy=="trend" else 0.0
                        if STATE.strategy=="fib" and candles_any:
                            vote += fib_vote(candles_any)
                        # lägre trösklar => fler trades
                        if vote > 0.15:
                            st=STATE.per_sym[sym]
                            price=candles_any[-1][-1]
                            _enter_leg(sym,"LONG",price,STATE.position_size,st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {price:.4f} | TP {st.pos.target_price:.4f} | ddSL~{STATE.risk_cfg['dd']:.2f}%"
                                )
                            open_syms.append(sym)
                            if len(open_syms)>=int(STATE.risk_cfg["max_pos"]): break
                        elif vote < -0.15 and STATE.risk_cfg["allow_shorts"]:
                            st=STATE.per_sym[sym]
                            price=candles_any[-1][-1]
                            _enter_leg(sym,"SHORT",price,STATE.position_size,st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔻 ENTRY {sym} SHORT @ {price:.4f} | TP {st.pos.target_price:.4f} | ddSL~{STATE.risk_cfg['dd']:.2f}%"
                                )
                            open_syms.append(sym)
                            if len(open_syms)>=int(STATE.risk_cfg["max_pos"]): break

                # HANTERA ÖPPNA
                for sym in STATE.symbols:
                    st=STATE.per_sym[sym]
                    if not st.pos: continue
                    tf0=STATE.tfs[0] if STATE.tfs else "1m"
                    try:
                        kl=await asyncio.wait_for(get_klines(sym, tf0, 3), timeout=6)
                    except Exception:
                        continue
                    price=kl[-1][-1]; avg=st.pos.avg_price
                    move_pct=(price-avg)/avg*100.0 if avg else 0.0

                    # High/Low-water för trailing
                    if st.pos.side=="LONG":
                        st.pos.high_water = max(st.pos.high_water, price)
                    else:
                        st.pos.low_water = min(st.pos.low_water, price) if st.pos.low_water else price

                    # Break-even aktivera när +be
                    if not st.pos.be_set:
                        if st.pos.side=="LONG" and move_pct>=STATE.trail_cfg["be"]:
                            st.pos.be_set=True
                        if st.pos.side=="SHORT" and -move_pct>=STATE.trail_cfg["be"]:
                            st.pos.be_set=True

                    # TP
                    if (st.pos.side=="LONG" and price>=st.pos.target_price) or \
                       (st.pos.side=="SHORT" and price<=st.pos.target_price):
                        net=_exit_all(sym, price, st)
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🟤 EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT")

                        # AI feedback (litet)
                        if STATE.ai_on:
                            f=features(kl)
                            ai_learn(sym, tf0, f, net)
                        continue

                    # Trailing-Stop
                    if st.pos.side=="LONG":
                        gain_from_entry = (st.pos.high_water-avg)/avg*100.0
                        if gain_from_entry>=STATE.trail_cfg["trigger"]:
                            trail_price = st.pos.high_water*(1-STATE.trail_cfg["step"]/100.0)
                            if st.pos.be_set:
                                trail_price = max(trail_price, avg*1.0002)  # BE+tick
                            if price<=trail_price:
                                net=_exit_all(sym, price, st)
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL {sym} LONG @ {price:.4f} | Net {net:+.4f} USDT")
                                continue
                    else: # SHORT
                        gain_from_entry = (avg-st.pos.low_water)/(avg or 1)*100.0
                        if gain_from_entry>=STATE.trail_cfg["trigger"]:
                            trail_price = st.pos.low_water*(1+STATE.trail_cfg["step"]/100.0)
                            if st.pos.be_set:
                                trail_price = min(trail_price, avg*0.9998)
                            if price>=trail_price:
                                net=_exit_all(sym, price, st)
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL {sym} SHORT @ {price:.4f} | Net {net:+.4f} USDT")
                                continue

                    # DCA
                    step=st.next_step_pct; need=False
                    if st.pos.side=="LONG" and move_pct<=-step: need=True
                    if st.pos.side=="SHORT" and move_pct>= step: need=True
                    if need and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        usd=STATE.position_size*(STATE.grid_cfg["size_mult"]**st.pos.safety_count)
                        _enter_leg(sym, st.pos.side, price, usd, st)
                        st.next_step_pct=max(STATE.grid_cfg["step_min"], st.next_step_pct*STATE.grid_cfg["step_mult"])
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} | leg {st.pos.safety_count}")

                    # Hård DD stop
                    if abs(move_pct)>=STATE.risk_cfg["dd"]:
                        net=_exit_all(sym, price, st)
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"⛔ STOP {sym} @ {price:.4f} | Net {net:+.4f} USDT (DD)")
            await asyncio.sleep(1.5)
        except Exception as e:
            if STATE.chat_id:
                try: await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except: pass
            await asyncio.sleep(5)

# ---------- UI ----------
def reply_kb()->ReplyKeyboardMarkup:
    rows=[
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/trail"), KeyboardButton("/strategy")],
        [KeyboardButton("/ai_on"), KeyboardButton("/ai_off")],
        [KeyboardButton("/save_ai"), KeyboardButton("/load_ai")],
        [KeyboardButton("/symbols"), KeyboardButton("/export_csv")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

tg_app = Application.builder().token(BOT_TOKEN).build()

def status_text()->str:
    total=sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    pos_lines=[]
    for s in STATE.symbols:
        st=STATE.per_sym[s]
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} → TP {st.pos.target_price:.4f} | legs {st.pos.safety_count}")
    g=STATE.grid_cfg; r=STATE.risk_cfg; t=STATE.trail_cfg
    lines=[
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'} | Mode: MOCK",
        f"AI: {'ON 🧠' if STATE.ai_on else 'OFF'} | Strategy: {STATE.strategy.upper()}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} USDT | Fee/mock: {STATE.fee_side:.4%}",
        (f"Grid: max_safety={g['max_safety']} step_mult={g['step_mult']} "
         f"step_min%={g['step_min']} size_mult={g['size_mult']} tp%={g['tp']} (+{g['tp_bonus']}%/safety)"),
        f"Trail: trigger={t['trigger']}% | trail={t['step']}% | be={t['be']}%",
        f"Risk: dd={r['dd']}% | max_pos={r['max_pos']} | shorts={'ON' if r['allow_shorts'] else 'OFF'}",
        f"PnL total (NET): {total:+.4f} USDT",
        "Open: "+(", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

async def send_status(chat_id:int):
    await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

# --- Commands ---
async def cmd_start(update:Update,_): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id,"MP Bot v42 – redo ✅",reply_markup=reply_kb()); await send_status(STATE.chat_id)
async def cmd_status(update:Update,_): STATE.chat_id=update.effective_chat.id; await send_status(STATE.chat_id)
async def cmd_engine_on(update:Update,_): STATE.chat_id=update.effective_chat.id; STATE.engine_on=True; await tg_app.bot.send_message(STATE.chat_id,"Engine: ON ✅",reply_markup=reply_kb())
async def cmd_engine_off(update:Update,_): STATE.chat_id=update.effective_chat.id; STATE.engine_on=False; await tg_app.bot.send_message(STATE.chat_id,"Engine: OFF ⛔️",reply_markup=reply_kb())
async def cmd_ai_on(update:Update,_): STATE.chat_id=update.effective_chat.id; STATE.ai_on=True; await tg_app.bot.send_message(STATE.chat_id,"AI: ON 🧠",reply_markup=reply_kb())
async def cmd_ai_off(update:Update,_): STATE.chat_id=update.effective_chat.id; STATE.ai_on=False; await tg_app.bot.send_message(STATE.chat_id,"AI: OFF",reply_markup=reply_kb())

async def cmd_timeframe(update:Update, context:ContextTypes.DEFAULT_TYPE):
    STATE.chat_id=update.effective_chat.id
    parts=update.message.text.strip().split(" ",1)
    if len(parts)==2:
        tfs=[x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs: STATE.tfs=tfs; await tg_app.bot.send_message(STATE.chat_id,f"Timeframes: {', '.join(STATE.tfs)}",reply_markup=reply_kb()); return
    await tg_app.bot.send_message(STATE.chat_id,"Använd: /timeframe 1m,3m,5m",reply_markup=reply_kb())

async def cmd_grid(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    toks=update.message.text.strip().split()
    if len(toks)==4 and toks[1]=="set":
        k=toks[2]; v=toks[3]
        if k in STATE.grid_cfg:
            try: STATE.grid_cfg[k]=float(v); await tg_app.bot.send_message(STATE.chat_id,"Grid uppdaterad.",reply_markup=reply_kb())
            except: await tg_app.bot.send_message(STATE.chat_id,"Felaktigt värde.",reply_markup=reply_kb())
            return
    g=STATE.grid_cfg
    await tg_app.bot.send_message(STATE.chat_id,(f"Grid:\n  max_safety={g['max_safety']}\n  step_mult={g['step_mult']}\n  step_min%={g['step_min']}\n  size_mult={g['size_mult']}\n  tp%={g['tp']} (+{g['tp_bonus']}%/safety)\nEx: /grid set step_min 0.12"),reply_markup=reply_kb())

async def cmd_risk(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    toks=update.message.text.strip().split()
    if len(toks)==4 and toks[1]=="set":
        k=toks[2]; v=toks[3]
        if k in STATE.risk_cfg:
            try:
                if k=="max_pos": STATE.risk_cfg[k]=int(v)
                elif k=="allow_shorts": STATE.risk_cfg[k]=(v.lower() in ("1","on","true","yes"))
                else: STATE.risk_cfg[k]=float(v)
                await tg_app.bot.send_message(STATE.chat_id,"Risk uppdaterad.",reply_markup=reply_kb())
            except: await tg_app.bot.send_message(STATE.chat_id,"Felaktigt värde.",reply_markup=reply_kb())
            return
    r=STATE.risk_cfg
    await tg_app.bot.send_message(STATE.chat_id,(f"Risk:\n  dd={r['dd']}%\n  max_pos={r['max_pos']}\n  shorts={'ON' if r['allow_shorts'] else 'OFF'}\nEx: /risk set dd 1.5"),reply_markup=reply_kb())

async def cmd_trail(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    toks=update.message.text.strip().split()
    if len(toks)==4 and toks[1]=="set":
        k=toks[2]; v=toks[3]
        if k in STATE.trail_cfg:
            try: STATE.trail_cfg[k]=float(v); await tg_app.bot.send_message(STATE.chat_id,"Trail uppdaterad.",reply_markup=reply_kb())
            except: await tg_app.bot.send_message(STATE.chat_id,"Felaktigt värde.",reply_markup=reply_kb())
            return
    t=STATE.trail_cfg
    await tg_app.bot.send_message(STATE.chat_id,(f"Trail:\n  trigger={t['trigger']}%\n  trail={t['step']}%\n  be={t['be']}%\nEx: /trail set trigger 0.8"),reply_markup=reply_kb())

async def cmd_strategy(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    toks=update.message.text.strip().split()
    if len(toks)==3 and toks[1]=="set" and toks[2] in ("trend","fib"):
        STATE.strategy=toks[2]; await tg_app.bot.send_message(STATE.chat_id,f"Strategy: {STATE.strategy}",reply_markup=reply_kb()); return
    await tg_app.bot.send_message(STATE.chat_id,f"Strategy: {STATE.strategy}\nVälj: /strategy set trend | /strategy set fib",reply_markup=reply_kb())

async def cmd_symbols(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    toks=update.message.text.strip().split()
    if len(toks)>=3 and toks[1] in ("add","remove"):
        sym=toks[2].upper()
        if toks[1]=="add":
            if sym not in STATE.symbols:
                STATE.symbols.append(sym); STATE.per_sym[sym]=SymState()
                await tg_app.bot.send_message(STATE.chat_id,f"La till: {sym}",reply_markup=reply_kb())
            else: await tg_app.bot.send_message(STATE.chat_id,f"{sym} finns redan.",reply_markup=reply_kb())
        else:
            if sym in STATE.symbols:
                STATE.symbols=[s for s in STATE.symbols if s!=sym]; STATE.per_sym.pop(sym,None)
                await tg_app.bot.send_message(STATE.chat_id,f"Tog bort: {sym}",reply_markup=reply_kb())
            else: await tg_app.bot.send_message(STATE.chat_id,f"{sym} finns ej.",reply_markup=reply_kb())
        return
    await tg_app.bot.send_message(STATE.chat_id,"Använd: /symbols add LINK-USDT | /symbols remove LINK-USDT",reply_markup=reply_kb())

async def cmd_pnl(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    total=sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    lines=[f"📈 PnL total (NET): {total:+.4f} USDT"]
    for s in STATE.symbols: lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl_net:+.4f} USDT")
    await tg_app.bot.send_message(STATE.chat_id,"\n".join(lines),reply_markup=reply_kb())

async def cmd_export_csv(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    rows=[["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r["time"],r["symbol"],r["side"],r["avg_price"],r["exit_price"],r["gross"],r["fee_in"],r["fee_out"],r["net"],r["safety_legs"]])
    if len(rows)==1: await tg_app.bot.send_message(STATE.chat_id,"Inga trades loggade ännu.",reply_markup=reply_kb()); return
    buf=io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")), filename="trades.csv", caption="Export CSV")

async def cmd_panic(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    closed=[]
    for s in STATE.symbols:
        st=STATE.per_sym[s]
        if not st.pos: continue
        tf0=STATE.tfs[0] if STATE.tfs else "1m"
        try: kl=await get_klines(s, tf0, 1); px=kl[-1][-1]
        except: continue
        net=_exit_all(s, px, st); closed.append(f"{s}:{net:+.4f}")
    await tg_app.bot.send_message(STATE.chat_id, "Panic close: "+(" | ".join(closed) if closed else "Inga positioner."), reply_markup=reply_kb())

async def cmd_reset_pnl(update:Update,_):
    STATE.chat_id=update.effective_chat.id
    for s in STATE.symbols: STATE.per_sym[s].realized_pnl_net=0.0; STATE.per_sym[s].trades_log.clear()
    await tg_app.bot.send_message(STATE.chat_id,"PnL återställd.",reply_markup=reply_kb())

async def cmd_save_ai(update:Update,_): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id, save_ai(), reply_markup=reply_kb())
async def cmd_load_ai(update:Update,_): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id, load_ai(), reply_markup=reply_kb())

# register
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("ai_on", cmd_ai_on))
tg_app.add_handler(CommandHandler("ai_off", cmd_ai_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("trail", cmd_trail))
tg_app.add_handler(CommandHandler("strategy", cmd_strategy))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
tg_app.add_handler(CommandHandler("load_ai", cmd_load_ai))

# ---------- FASTAPI ----------
app=FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int]=None

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
    total=sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return f"MP Bot v42 OK | engine_on={STATE.engine_on} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total=sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {"ok":True,"engine_on":STATE.engine_on,"tfs":STATE.tfs,"strategy":STATE.strategy,"symbols":STATE.symbols,"pnl_total":round(total,6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request:Request):
    data=await request.json()
    update=Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok":True}
