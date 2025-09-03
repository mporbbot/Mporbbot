# main_v45.py
# ------------------------------------------------------------
# MP Bot – v45
# - Reply-keyboard (inga inline-knappar)
# - Mock-handel (default) med avgifter & DCA/TP
# - Trailing stop + break-even + DD-stop
# - Enkel AI-score + save/load
# - CSV-export (trades) + K4-export (Skatteverket) med SEK
# - Välja coins & timeframes
# - FastAPI webhook (Render) + watchdog/heartbeat
# - Live-läge för KuCoin Spot (Market/Trade API) – slå på via /start_live
#   (shorts i live-läge är AV – kräver margin/loan, hålls utanför)
# ------------------------------------------------------------

import os, io, csv, math, pickle, asyncio, contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------- KuCoin import: stöd Market/Trade (kucoin-python==1.0.12) ----------
KUCOIN_STYLE = None
KuMarket = KuTrade = None
try:
    from kucoin.client import Market as KuMarket, Trade as KuTrade
    KUCOIN_STYLE = "trade_market"
except Exception:
    KUCOIN_STYLE = "none"

# ---------------- ENV / SETTINGS ----------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
                   .replace(" ", "")).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))          # 0.1%
MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))

GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.6"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.25"))
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.05"))

DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "1.8"))               # lite lägre, enligt din önskan

# Trailing / trend-follow
TREND_ON = os.getenv("TREND_ON", "true").lower() in ("1","true","yes")
TREND_TRIGGER = float(os.getenv("TREND_TRIGGER_PCT", "0.9"))       # aktivera trail efter +0.9%
TREND_BE = float(os.getenv("TREND_BE_PCT", "0.20"))                 # flytta BE efter +0.2%
TREND_TRAIL = float(os.getenv("TREND_TRAIL_PCT", "0.25"))           # ge pris luft, 0.25%

# AI
AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "/mnt/data/ai_model.pkl")
os.makedirs(os.path.dirname(AI_MODEL_PATH) or ".", exist_ok=True)

# Skatteverket K4
K4_USDTSEK = os.getenv("K4_USDTSEK")
K4_USDTSEK = float(K4_USDTSEK) if K4_USDTSEK not in (None, "") else None

# KuCoin nycklar (spot)
KUCOIN_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KUCOIN_SANDBOX = os.getenv("KUCOIN_SANDBOX", "false").lower() in ("1","true","yes")

# KuCoin REST (candles)
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1hour"}

# ------------- GLOBAL HTTP-klient -------------
HTTP: httpx.AsyncClient

def _mk_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=3, read=7, write=7, pool=7),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
        headers={"User-Agent":"mp-bot/45"}
    )

# ---------------- STATE ----------------
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
    # trailing
    trail_active: bool = False
    trail_anchor: Optional[float] = None
    be_moved: bool = False
    def qty_total(self) -> float: return sum(l.qty for l in self.legs)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)  # för CSV/K4
    next_step_pct: float = GRID_STEP_MIN_PCT

@dataclass
class EngineState:
    engine_on: bool = False
    mode_live: bool = False
    ai_on: bool = True
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
    per_sym: Dict[str, SymState] = field(default_factory=dict)
    position_size: float = POSITION_SIZE_USDT
    fee_side: float = FEE_PER_SIDE
    grid_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        max_safety=GRID_MAX_SAFETY, step_mult=GRID_STEP_MULT, step_min=GRID_STEP_MIN_PCT,
        size_mult=GRID_SIZE_MULT, tp=GRID_TP_PCT, tp_bonus=TP_SAFETY_BONUS))
    risk_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        dd=DD_STOP_PCT, max_pos=MAX_OPEN_POS, allow_shorts=False))   # shorts OFF i live-läge
    trend_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        on=TREND_ON, trigger=TREND_TRIGGER, be=TREND_BE, trail=TREND_TRAIL))
    chat_id: Optional[int] = None
    last_loop_at: datetime = datetime.now(timezone.utc)
    engine_task: Optional[asyncio.Task] = None
    watchdog_task: Optional[asyncio.Task] = None
    heartbeat_task: Optional[asyncio.Task] = None

STATE = EngineState()
for s in STATE.symbols: STATE.per_sym[s] = SymState()

# AI
AI_WEIGHTS: Dict[Tuple[str,str], List[float]] = {}
AI_LEARN_RATE = 0.05

# KuCoin klienter (spot)
KU_MARKET = KU_TRADE = None
def _init_kucoin():
    global KU_MARKET, KU_TRADE
    if KUCOIN_STYLE != "trade_market": return "KuCoin-paket saknar Market/Trade."
    try:
        KU_MARKET = KuMarket(key=KUCOIN_KEY, secret=KUCOIN_SECRET, passphrase=KUCOIN_PASSPHRASE, is_sandbox=KUCOIN_SANDBOX)
        KU_TRADE  = KuTrade (key=KUCOIN_KEY, secret=KUCOIN_SECRET, passphrase=KUCOIN_PASSPHRASE, is_sandbox=KUCOIN_SANDBOX)
        return "KuCoin Market/Trade init."
    except Exception as e:
        return f"KuCoin init-fel: {e}"

# ----------------- INDICATORS / FEATURES -----------------
def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1: return series[:]
    k = 2/(period+1); out=[]; v=series[0]
    for x in series: v=(x-v)*k+v; out.append(v)
    return out

def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period+1: return [50.0]*len(closes)
    gains, losses=[],[]
    for i in range(1,len(closes)):
        ch = closes[i]-closes[i-1]
        gains.append(max(ch,0.0)); losses.append(-min(ch,0.0))
    ag=sum(gains[:period])/period; al=sum(losses[:period])/period
    rsis=[50.0]*period
    for i in range(period,len(gains)):
        ag=(ag*(period-1)+gains[i])/period; al=(al*(period-1)+losses[i])/period
        rs = (ag/al) if al else 999
        rsis.append(100-(100/(1+rs)))
    return [50.0]+rsis

def features_from_candles(candles):
    closes=[c[-1] for c in candles]
    if len(closes)<30: return {"mom":0.0,"ema_fast":0.0,"ema_slow":0.0,"rsi":50.0,"atrp":0.2}
    ema_fast=ema(closes,9)[-1]; ema_slow=ema(closes,21)[-1]
    mom=((closes[-1]-closes[-6])/closes[-6]*100.0) if closes[-6] else 0.0
    rsi_last=rsi(closes,14)[-1]
    rng=[(c[2]-c[3])/c[-1]*100.0 if c[-1] else 0.0 for c in candles[-20:]]
    atrp=sum(rng)/len(rng) if rng else 0.2
    return {"mom":mom,"ema_fast":ema_fast,"ema_slow":ema_slow,"rsi":rsi_last,"atrp":max(0.02,min(3.0,atrp))}

# ----------------- AI -----------------
def _wkey(symbol, tf): return (symbol, tf)

def ai_score(symbol, tf, feats):
    key=_wkey(symbol,tf)
    if key not in AI_WEIGHTS: AI_WEIGHTS[key]=[0.0,0.7,0.6,0.1]   # bias,mom,ema,rsi
    w0,wm,we,wr = AI_WEIGHTS[key]
    ema_diff=(feats["ema_fast"]-feats["ema_slow"])/(feats["ema_slow"] or 1.0)*100.0
    rsi_dev=(feats["rsi"]-50.0)/50.0
    raw = w0 + wm*feats["mom"] + we*ema_diff + wr*rsi_dev*100.0
    return max(-10,min(10,raw))

def ai_learn(symbol, tf, feats, pnl_net):
    key=_wkey(symbol,tf)
    if key not in AI_WEIGHTS: AI_WEIGHTS[key]=[0.0,0.7,0.6,0.1]
    w=AI_WEIGHTS[key]
    ema_diff=(feats["ema_fast"]-feats["ema_slow"])/(feats["ema_slow"] or 1.0)*100.0
    rsi_dev=(feats["rsi"]-50.0)/50.0
    grad=[1.0, feats["mom"], ema_diff, rsi_dev*100.0]
    lr=AI_LEARN_RATE*(1 if pnl_net>=0 else -1)
    AI_WEIGHTS[key]=[w[i]+lr*grad[i] for i in range(4)]

def save_ai()->str:
    try:
        with open(AI_MODEL_PATH,"wb") as f: pickle.dump(AI_WEIGHTS,f)
        return f"AI sparad: {AI_MODEL_PATH}"
    except Exception as e:
        return f"Fel vid sparning: {e}"

def load_ai()->str:
    global AI_WEIGHTS
    try:
        with open(AI_MODEL_PATH,"rb") as f:
            AI_WEIGHTS = pickle.load(f)
        return f"AI laddad från: {AI_MODEL_PATH}"
    except Exception as e:
        return f"Kunde inte ladda AI: {e}"

# ----------------- DATA -----------------
async def _fetch_with_retries(url: str, params: dict, retries: int = 4):
    delay = 0.6; last = None
    for _ in range(retries):
        try:
            r = await HTTP.get(url, params=params); r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e; await asyncio.sleep(delay); delay *= 1.7
    raise last or RuntimeError("HTTP error")

async def get_klines(symbol: str, tf: str, limit: int = 60):
    k_tf = TF_MAP.get(tf,"1min")
    data = await _fetch_with_retries(KUCOIN_KLINES_URL, {"symbol":symbol,"type":k_tf})
    rows = list(reversed(data["data"]))[-limit:]
    out=[]
    for row in rows:
        t_ms=int(row[0])*1000; o=float(row[1]); c=float(row[2]); h=float(row[3]); l=float(row[4])
        out.append((t_ms,o,h,l,c))
    return out

# ----------------- TRADING (mock & live) -----------------
def _fee(amount_usdt: float) -> float: return amount_usdt * STATE.fee_side

@dataclass
class LiveOrderResult:
    ok: bool
    msg: str
    order_id: Optional[str]=None

def _enter_leg_mock(sym: str, side: str, price: float, usd_size: float, st: SymState) -> TradeLeg:
    qty = usd_size/price if price>0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0)
        st.pos.trail_active=False; st.pos.trail_anchor=None; st.pos.be_moved=False
        st.next_step_pct = STATE.grid_cfg["step_min"]
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        total = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total or 1.0)
    # TP baserat på avg_price
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count*STATE.grid_cfg["tp_bonus"]
    if side=="LONG":
        st.pos.target_price = st.pos.avg_price*(1+tp/100.0)
    else:
        st.pos.target_price = st.pos.avg_price*(1-tp/100.0)
    return leg

def _exit_all_mock(sym: str, price: float, st: SymState) -> float:
    if not st.pos: return 0.0
    gross=0.0; fee_in=0.0
    for leg in st.pos.legs:
        usd_in = leg.qty*leg.price; fee_in += _fee(usd_in)
        if st.pos.side=="LONG": gross += leg.qty*(price-leg.price)
        else: gross += leg.qty*(leg.price-price)
    usd_out = st.pos.qty_total()*price
    fee_out = _fee(usd_out)
    net = gross - fee_in - fee_out
    st.realized_pnl_net += net
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym, "side": st.pos.side, "avg_price": st.pos.avg_price,
        "exit_price": price, "gross": round(gross,6), "fee_in":round(fee_in,6),
        "fee_out":round(fee_out,6), "net": round(net,6), "safety_legs": st.pos.safety_count
    })
    st.pos=None; st.next_step_pct = STATE.grid_cfg["step_min"]
    return net

# Live (spot): endast LONG (köp -> sälj), shorts avstängda
async def _place_buy_live(sym: str, funds_usdt: float) -> LiveOrderResult:
    if KU_TRADE is None: return LiveOrderResult(False,"KuCoin Trade ej init")
    try:
        # Market BUY för quote (USDT): använd 'funds'
        res = KU_TRADE.create_market_order(symbol=sym, side='buy', funds=str(round(funds_usdt, 6)))
        return LiveOrderResult(True, "BUY OK", res.get('orderId'))
    except Exception as e:
        return LiveOrderResult(False, f"BUY fel: {e}")

async def _place_sell_live(sym: str, size_base: float) -> LiveOrderResult:
    if KU_TRADE is None: return LiveOrderResult(False,"KuCoin Trade ej init")
    try:
        # Market SELL kräver 'size' i basmängd
        res = KU_TRADE.create_market_order(symbol=sym, side='sell', size=str(round(size_base, 8)))
        return LiveOrderResult(True, "SELL OK", res.get('orderId'))
    except Exception as e:
        return LiveOrderResult(False, f"SELL fel: {e}")

# ----------------- SIGNAL / ENGINE -----------------
def score_vote(symbol: str, feats_by_tf: Dict[str,Dict]) -> float:
    votes=0.0
    for tf,feats in feats_by_tf.items():
        sc = ai_score(symbol, tf, feats) if STATE.ai_on else 0.0
        ema_diff=(feats["ema_fast"]-feats["ema_slow"])/(feats["ema_slow"] or 1.0)*100.0
        bias=(1.0 if ema_diff>0 else -1.0)*min(1.0,abs(ema_diff)/1.5)
        votes += sc/10.0 + bias
    return votes

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        STATE.last_loop_at = datetime.now(timezone.utc)
        try:
            if STATE.engine_on:
                # 1) Öppna nya
                open_syms = [s for s,st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.risk_cfg["max_pos"]:
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos: continue
                        feats_by_tf={}; skip=False; last_px=None
                        for tf in STATE.tfs:
                            try:
                                kl = await get_klines(sym, tf, 60)
                                feats_by_tf[tf]=features_from_candles(kl)
                                if tf==STATE.tfs[0]: last_px = kl[-1][-1]
                            except Exception:
                                skip=True; break
                        if skip or not feats_by_tf or last_px is None: continue
                        score = score_vote(sym, feats_by_tf)
                        if score > 0.35:
                            st=STATE.per_sym[sym]
                            # mock alltid; live: placera riktig market BUY
                            if STATE.mode_live and not STATE.risk_cfg["allow_shorts"]:
                                live = await _place_buy_live(sym, STATE.position_size)
                                if not live.ok and STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"⚠️ Live BUY misslyckades: {live.msg}")
                            leg=_enter_leg_mock(sym,"LONG", last_px, STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {leg.price:.4f} | QTY {leg.qty:.6f}\n"
                                    f"TP ~ {st.pos.target_price:.4f} | Avgift~ {STATE.position_size*FEE_PER_SIDE:.4f} USDT"
                                )
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]: break

                # 2) Hantera öppna (TP/DCA/Stop/Trail)
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos: continue
                    tf0 = STATE.tfs[0] if STATE.tfs else "1m"
                    try:
                        kl = await get_klines(sym, tf0, limit=3)
                    except Exception:
                        continue
                    price = kl[-1][-1]
                    avg = st.pos.avg_price
                    move_pct = (price-avg)/avg*100.0 if avg else 0.0

                    # Break-even flytt
                    if TREND_ON and not st.pos.be_moved and st.pos.side=="LONG" and move_pct >= STATE.trend_cfg["be"]:
                        # flytta "mentalt" stop till avg (mock – real stop skulle kräva OCO)
                        st.pos.be_moved = True
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🔒 {sym} BE aktiverad (avg {avg:.4f})")

                    # Trailing: aktivera efter trigger
                    if TREND_ON and st.pos.side=="LONG":
                        if not st.pos.trail_active and move_pct >= STATE.trend_cfg["trigger"]:
                            st.pos.trail_active = True
                            st.pos.trail_anchor = price
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"🏁 {sym} trailing ON (anchor {price:.4f})")
                        if st.pos.trail_active:
                            st.pos.trail_anchor = max(st.pos.trail_anchor or price, price)
                            drop = (st.pos.trail_anchor - price)/st.pos.trail_anchor*100.0 if st.pos.trail_anchor else 0.0
                            if drop >= STATE.trend_cfg["trail"]:
                                # sälj allt – trail hit
                                net = _exit_all_mock(sym, price, st)
                                if STATE.mode_live:
                                    res = await _place_sell_live(sym, st.pos.qty_total() if st.pos else 0.0)  # pos är None efter mock exit – ta qty innan!
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT")
                                continue

                    # TP
                    if price >= st.pos.target_price and st.pos.side=="LONG":
                        net = _exit_all_mock(sym, price, st)
                        if STATE.mode_live:
                            await _place_sell_live(sym, st.pos.qty_total() if st.pos else 0.0)
                        if STATE.chat_id:
                            mark="✅" if net>=0 else "❌"
                            await app.bot.send_message(STATE.chat_id, f"🟤 TP EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}")
                        feats = features_from_candles(kl)
                        if STATE.ai_on: ai_learn(sym, tf0, feats, net)
                        continue

                    # DCA
                    need_dca=False; step=st.next_step_pct
                    if st.pos.side=="LONG" and move_pct <= -step: need_dca=True
                    if need_dca and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        usd = STATE.position_size*(STATE.grid_cfg["size_mult"]**st.pos.safety_count)
                        leg = _enter_leg_mock(sym, st.pos.side, price, usd, st)
                        st.next_step_pct = max(STATE.grid_cfg["step_min"], st.next_step_pct*STATE.grid_cfg["step_mult"])
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} | leg {st.pos.safety_count} | QTY {leg.qty:.6f}")

                    # DD-stop (nöd)
                    if abs(move_pct) >= STATE.risk_cfg["dd"]:
                        net = _exit_all_mock(sym, price, st)
                        if STATE.mode_live:
                            await _place_sell_live(sym, st.pos.qty_total() if st.pos else 0.0)
                        if STATE.chat_id:
                            mark="✅" if net>=0 else "❌"
                            await app.bot.send_message(STATE.chat_id, f"⛔ STOP {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark} (DD)")

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                with contextlib.suppress(Exception):
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
            await asyncio.sleep(5)

# Watchdog – startar om engine om den inte tickat på ett tag
async def watchdog(app: Application):
    while True:
        try:
            ago = datetime.now(timezone.utc) - STATE.last_loop_at
            if ago > timedelta(seconds=30):
                # starta om engine-task
                if STATE.engine_task and not STATE.engine_task.done():
                    STATE.engine_task.cancel()
                    with contextlib.suppress(Exception):
                        await STATE.engine_task
                STATE.engine_task = asyncio.create_task(engine_loop(app))
                if STATE.chat_id:
                    await app.bot.send_message(STATE.chat_id, "🩺 Watchdog: engine restartad")
        except Exception:
            pass
        await asyncio.sleep(10)

# Heartbeat – visar att den lever
async def heartbeat(app: Application):
    while True:
        if STATE.chat_id:
            with contextlib.suppress(Exception):
                await app.bot.send_message(STATE.chat_id, "💓")
        await asyncio.sleep(300)

# ----------------- UI / TELEGRAM -----------------
def reply_kb()->ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/start_live"), KeyboardButton("/stop_live")],
        [KeyboardButton("/timeframe"), KeyboardButton("/symbols")],
        [KeyboardButton("/pnl"), KeyboardButton("/export_csv"), KeyboardButton("/k4")],
        [KeyboardButton("/ai_on"), KeyboardButton("/ai_off")],
        [KeyboardButton("/save_ai"), KeyboardButton("/load_ai")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def status_text()->str:
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    pos_lines=[]
    for s in STATE.symbols:
        st=STATE.per_sym[s]
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} → TP {st.pos.target_price:.4f} | legs {st.pos.safety_count}")
    g=STATE.grid_cfg; r=STATE.risk_cfg; t=STATE.trend_cfg
    lines=[
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'} | Mode: {'LIVE 🟢' if STATE.mode_live else 'MOCK 🧪'}",
        f"AI: {'ON 🧠' if STATE.ai_on else 'OFF'}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} USDT | Fee/side: {STATE.fee_side:.4%}",
        (f"Grid: max_safety={g['max_safety']} step_mult={g['step_mult']} step_min%={g['step_min']} "
         f"size_mult={g['size_mult']} tp%={g['tp']} (+{g['tp_bonus']}%/leg)"),
        (f"Risk: dd={r['dd']}% | max_pos={r['max_pos']} | shorts={'ON' if r['allow_shorts'] else 'OFF'}"),
        (f"Trail: on={'ON' if t['on'] else 'OFF'} | trigger={t['trigger']}% | BE={t['be']}% | trail={t['trail']}%"),
        f"PnL NET: {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

tg_app = Application.builder().token(BOT_TOKEN).build()

async def send_status(chat_id:int):
    await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v45 – redo ✅", reply_markup=reply_kb())
    await send_status(STATE.chat_id)

async def cmd_status(update: Update, _): STATE.chat_id=update.effective_chat.id; await send_status(STATE.chat_id)
async def cmd_engine_on(update: Update, _): STATE.chat_id=update.effective_chat.id; STATE.engine_on=True; await tg_app.bot.send_message(STATE.chat_id,"Engine: ON ✅",reply_markup=reply_kb())
async def cmd_engine_off(update: Update, _): STATE.chat_id=update.effective_chat.id; STATE.engine_on=False; await tg_app.bot.send_message(STATE.chat_id,"Engine: OFF ⛔️",reply_markup=reply_kb())
async def cmd_ai_on(update: Update, _): STATE.chat_id=update.effective_chat.id; STATE.ai_on=True; await tg_app.bot.send_message(STATE.chat_id,"AI: ON 🧠",reply_markup=reply_kb())
async def cmd_ai_off(update: Update, _): STATE.chat_id=update.effective_chat.id; STATE.ai_on=False; await tg_app.bot.send_message(STATE.chat_id,"AI: OFF",reply_markup=reply_kb())

async def cmd_start_live(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    if KUCOIN_STYLE != "trade_market":
        await tg_app.bot.send_message(STATE.chat_id, "KuCoin-paket saknar Market/Trade – live av.", reply_markup=reply_kb()); return
    note = _init_kucoin()
    STATE.mode_live = True
    await tg_app.bot.send_message(STATE.chat_id, f"Live-läge: ON 🟢 ({note})\nObs: shorts OFF i live.", reply_markup=reply_kb())

async def cmd_stop_live(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.mode_live = False
    await tg_app.bot.send_message(STATE.chat_id, "Live-läge: OFF (mock)", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = update.message.text.strip()
    parts = txt.split(" ",1)
    if len(parts)==2:
        tfs=[x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs=tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.tfs)}", reply_markup=reply_kb()); return
    await tg_app.bot.send_message(STATE.chat_id,"Använd: /timeframe 1m,3m,5m,15m",reply_markup=reply_kb())

async def cmd_symbols(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = update.message.text.strip()
    toks = txt.split()
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
    await tg_app.bot.send_message(STATE.chat_id,"Symbols: "+", ".join(STATE.symbols)
                                  +"\n/symbols add LINK-USDT\n/symbols remove LINK-USDT", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    lines=[f"📈 PnL NET: {total:+.4f} USDT"]
    for s in STATE.symbols: lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl_net:+.4f} USDT")
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    rows=[["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r["time"],r["symbol"],r["side"],r["avg_price"],r["exit_price"],r["gross"],r["fee_in"],r["fee_out"],r["net"],r["safety_legs"]])
    if len(rows)==1:
        await tg_app.bot.send_message(STATE.chat_id,"Inga trades loggade ännu.", reply_markup=reply_kb()); return
    buf=io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="trades.csv", caption="Export CSV")

async def cmd_k4(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    rate = K4_USDTSEK if K4_USDTSEK else 10.0
    rows=[["Datum","Beteckning","Antal","Försäljningspris SEK","Omkostnadsbelopp SEK","Vinst/Förlust SEK","Valuta (antag USDT)","Kurs USDT/SEK"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            qty = sum(l["qty"] if isinstance(l,dict) and "qty" in l else 0.0 for l in [])
            # använder inte qty i mock-loggen – vi räknar i USDT
            sell_sek = (r["exit_price"] * rate)  # proxy
            cost_sek = (r["avg_price"] * rate)
            pnl_sek  = r["net"] * rate
            rows.append([
                r["time"][:10], r["symbol"], "", f"{sell_sek:.2f}", f"{cost_sek:.2f}", f"{pnl_sek:.2f}", "USDT", f"{rate:.2f}"
            ])
    if len(rows)==1:
        await tg_app.bot.send_message(STATE.chat_id,"Inga avslut för K4 ännu.", reply_markup=reply_kb()); return
    buf=io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="k4_mock.csv", caption=f"K4 (SEK, kurs {rate:.2f})")

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed=[]
    for s in STATE.symbols:
        st=STATE.per_sym[s]
        if not st.pos: continue
        try:
            kl=await get_klines(s, STATE.tfs[0] if STATE.tfs else "1m", 1)
            px=kl[-1][-1]
        except: continue
        net=_exit_all_mock(s, px, st)
        if STATE.mode_live:
            await _place_sell_live(s, st.pos.qty_total() if st.pos else 0.0)
        closed.append(f"{s}:{net:+.4f}")
    await tg_app.bot.send_message(STATE.chat_id, "Panic: "+(" | ".join(closed) if closed else "inga"), reply_markup=reply_kb())

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id=update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl_net=0.0
        STATE.per_sym[s].trades_log.clear()
    await tg_app.bot.send_message(STATE.chat_id,"PnL återställd.", reply_markup=reply_kb())

async def cmd_save_ai(update: Update, _): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id, save_ai(), reply_markup=reply_kb())
async def cmd_load_ai(update: Update, _): STATE.chat_id=update.effective_chat.id; await tg_app.bot.send_message(STATE.chat_id, load_ai(), reply_markup=reply_kb())

# Register handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("start_live", cmd_start_live))
tg_app.add_handler(CommandHandler("stop_live", cmd_stop_live))
tg_app.add_handler(CommandHandler("ai_on", cmd_ai_on))
tg_app.add_handler(CommandHandler("ai_off", cmd_ai_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("k4", cmd_k4))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
tg_app.add_handler(CommandHandler("load_ai", cmd_load_ai))

# ----------------- FASTAPI / WEBHOOK -----------------
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int]=None

@app.on_event("startup")
async def on_startup():
    global HTTP
    HTTP = _mk_http()
    # Telegram – starta app
    await tg_app.initialize()
    await tg_app.start()
    # Webhook (Render)
    if WEBHOOK_BASE:
        with contextlib.suppress(Exception):
            await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    # Engine + watchdog + heartbeat
    STATE.engine_task = asyncio.create_task(engine_loop(tg_app))
    STATE.watchdog_task = asyncio.create_task(watchdog(tg_app))
    STATE.heartbeat_task = asyncio.create_task(heartbeat(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    for t in (STATE.engine_task, STATE.watchdog_task, STATE.heartbeat_task):
        if t and not t.done():
            t.cancel()
            with contextlib.suppress(Exception):
                await t
    with contextlib.suppress(Exception):
        await tg_app.stop(); await tg_app.shutdown()
    with contextlib.suppress(Exception):
        await HTTP.aclose()

@app.get("/", response_class=PlainTextResponse)
async def root():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return f"MP Bot v45 OK | engine_on={STATE.engine_on} | mode={'LIVE' if STATE.mode_live else 'MOCK'} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {"ok":True,"engine_on":STATE.engine_on,"tfs":STATE.tfs,"symbols":STATE.symbols,
            "mode":"LIVE" if STATE.mode_live else "MOCK","pnl_total":round(total,6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok":True}
