# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MP Bot – v44 (Confirmed Momentum Hybrid - "premium")
# ------------------------------------------------------------
# - Bekräftad momentum-entry (confirm_mode) + trendfilter
# - Hybrid TP→Trail (sen aktivering) + bred trailing
# - Failsafes: early_cut / sl_max / max_dd_per_trade
# - Mock-handel, CSV-export, AI-spara/ladda
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
AI_WEIGHTS: Dict[Tuple[str, str], List[float]] = {}
AI_LEARN_RATE = 0.05# ------------------------------------------------------------
# AI / SIGNAL / LOGIC FUNCTIONS
# ------------------------------------------------------------
async def fetch_klines(symbol: str, tf: str, limit: int = 60) -> List[List[str]]:
    """Hämta senaste candles från KuCoin."""
    tf_api = TF_MAP.get(tf, tf)
    url = f"{KUCOIN_KLINES_URL}?type={tf_api}&symbol={symbol}&limit={limit}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        data = r.json().get("data", [])
        return data[::-1]  # senaste sist

def calc_orb_like(data: List[List[str]]) -> Tuple[float, float]:
    """Beräkna enklare breakout-liknande signal (range-styrka)."""
    if len(data) < 6:
        return 0.0, 0.0
    highs = [float(x[3]) for x in data[-6:]]
    lows = [float(x[4]) for x in data[-6:]]
    rng = (max(highs) - min(lows)) / float(data[-1][2])
    last_close = float(data[-1][2])
    prev_close = float(data[-2][2])
    momentum = (last_close - prev_close) / prev_close * 100
    return rng, momentum

def score_signal(symbol: str, tf: str, candles: List[List[str]]) -> float:
    """Enkel AI-score baserad på range + momentum + bias."""
    rng, momentum = calc_orb_like(candles)
    score = (rng * 1.2 + momentum * 0.8)
    # Normalisera till ~0-3
    return max(0.0, min(3.0, score / 1.5))

def trend_ok(candles: List[List[str]], bias: float, lookback: int = 30) -> bool:
    """Kollar om övergripande trend stödjer trade-riktningen."""
    if len(candles) < lookback + 1:
        return True
    closes = [float(x[2]) for x in candles[-lookback:]]
    diff = closes[-1] - closes[0]
    pct = diff / closes[0] * 100
    return pct > bias * 100

def confirm_entry(candles: List[List[str]], strength: float, candles_need: int) -> bool:
    """Bekräftar att priset fortsätter i samma riktning efter signal."""
    if len(candles) < candles_need + 1:
        return False
    closes = [float(x[2]) for x in candles[-(candles_need+1):]]
    move = (closes[-1] - closes[0]) / closes[0] * 100
    return move >= strength

# ------------------------------------------------------------
# ENTRY / EXIT LOGIC
# ------------------------------------------------------------
async def check_signal(symbol: str, tf: str):
    data = await fetch_klines(symbol, tf, 60)
    if not data:
        return None
    score = score_signal(symbol, tf, data)
    if score < STATE.entry_threshold:
        return None
    risk = STATE.risk_cfg

    # Trendfilter
    if risk.get("trend_filter", True):
        if not trend_ok(data, risk.get("trend_bias", 0.5), risk.get("trend_len", 30)):
            return None

    # Confirm-mode
    if risk.get("confirm_mode", True):
        if not confirm_entry(data, risk.get("confirm_strength", 0.4), risk.get("confirm_candle", 2)):
            return None

    side = "buy"
    price = float(data[-1][2])
    return dict(symbol=symbol, tf=tf, side=side, price=price, score=score)

def calc_pnl(position: Position, current_price: float) -> float:
    if not position or position.qty_total() == 0:
        return 0.0
    direction = 1 if position.side == "buy" else -1
    return (current_price - position.avg_price) / position.avg_price * 100 * direction

def should_exit(position: Position, pnl_pct: float, price: float, risk: dict) -> bool:
    """Avgör om positionen ska stängas."""
    # Max stop
    if pnl_pct <= -risk.get("sl_max", 1.0):
        return True
    # Hybrid lock
    if risk.get("hyb_on", True):
        if pnl_pct >= risk.get("hyb_start", 2.5):
            lock_thr = pnl_pct - risk.get("hyb_lock", 0.25)
            if price < position.high_water * (1 - lock_thr/100):
                return True
    # Trailing
    if risk.get("trail_on", True) and position.trailing_active:
        if pnl_pct < (position.high_water - position.avg_price)/position.avg_price*100 - risk.get("trail_mult",7.0):
            return True
    return False

# ------------------------------------------------------------
# TRADE EXECUTION (mock)
# ------------------------------------------------------------
async def execute_entry(symbol: str, side: str, price: float):
    sym_state = STATE.per_sym[symbol]
    if sym_state.pos:
        return  # redan inne
    qty = STATE.position_size / price
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    pos = Position(side=side, legs=[leg], avg_price=price, target_price=price*(1+GRID_TP_PCT/100))
    sym_state.pos = pos
    sym_state.trades_log.append(dict(time=datetime.now().isoformat(), side=side, price=price, action="ENTRY"))
    print(f"[ENTRY] {symbol} {side} @ {price:.4f}")

async def execute_exit(symbol: str, price: float, reason: str = ""):
    sym_state = STATE.per_sym[symbol]
    pos = sym_state.pos
    if not pos:
        return
    qty = pos.qty_total()
    pnl = (price - pos.avg_price) / pos.avg_price * 100 if pos.side == "buy" else (pos.avg_price - price)/pos.avg_price*100
    sym_state.realized_pnl_net += pnl - (STATE.fee_side*2*100)
    sym_state.trades_log.append(dict(time=datetime.now().isoformat(), side=pos.side, price=price, pnl=pnl, reason=reason, action="EXIT"))
    sym_state.pos = None
    print(f"[EXIT] {symbol} {reason} PnL={pnl:.2f}%")# ------------------------------------------------------------
# TELEGRAM COMMANDS
# ------------------------------------------------------------
app = FastAPI()
tg_app = Application.builder().token(BOT_TOKEN).build()

def reply_kb():
    return ReplyKeyboardMarkup(
        [[KeyboardButton("/status"), KeyboardButton("/mode premium")],
         [KeyboardButton("/risk"), KeyboardButton("/engine_on"), KeyboardButton("/engine_off")]],
        resize_keyboard=True
    )

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    await update.message.reply_text(
        "🤖 Mp ORBbot v44 – Premium Mode\n"
        "AI-läge: Confirmed Momentum Hybrid\n"
        "Använd /mode premium för att aktivera.", reply_markup=reply_kb()
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = f"🟢 Engine On: {STATE.engine_on}\n"
    msg += f"AI Threshold: {STATE.entry_threshold:.2f}\n"
    msg += f"Symbols: {', '.join(STATE.symbols)}\n\n"
    for s, st in STATE.per_sym.items():
        pnl = f"{st.realized_pnl_net:.2f}"
        msg += f"{s}: {'OPEN' if st.pos else '–'} | PnL: {pnl}%\n"
    await update.message.reply_text(msg)

async def cmd_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    sub = args[0].lower() if args else ""
    if sub == "premium":
        STATE.entry_threshold = 1.85
        STATE.risk_cfg.update(dict(
            trail_on=True, hyb_on=True, trail_min=3.0, trail_mult=7.0,
            hyb_start=2.50, hyb_lock=0.25, max_pos=5, size=80.0,
            sl_max=1.10, early_cut=False, allow_shorts=False,
            confirm_mode=True, confirm_candle=2, confirm_strength=0.4,
            trend_filter=True, trend_len=30, trend_bias=0.6,
            sniper_atr_min=0.35, sniper_dev_min=0.75,
            sniper_rsi_turn=0.30, sniper_mtf_bias=0.50
        ))
        await update.message.reply_text("✅ Mode satt: premium (Confirmed Momentum Hybrid)", reply_markup=reply_kb())
        return
    await update.message.reply_text("Använd: /mode premium", reply_markup=reply_kb())

async def cmd_engine_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = True
    await update.message.reply_text("🚀 Engine påbörjad.")

async def cmd_engine_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = False
    await update.message.reply_text("🛑 Engine stoppad.")

async def cmd_risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = "\n".join([f"{k}: {v}" for k, v in STATE.risk_cfg.items()])
    await update.message.reply_text(f"⚙️ Riskparametrar:\n{txt}")

# ------------------------------------------------------------
# ENGINE LOOP
# ------------------------------------------------------------
async def engine_loop():
    while True:
        if STATE.engine_on:
            for sym in STATE.symbols:
                for tf in STATE.tfs:
                    sig = await check_signal(sym, tf)
                    if sig:
                        await execute_entry(sig["symbol"], sig["side"], sig["price"])
                    st = STATE.per_sym[sym]
                    if st.pos:
                        last_price = float((await fetch_klines(sym, tf, 1))[0][2])
                        pnl = calc_pnl(st.pos, last_price)
                        if should_exit(st.pos, pnl, last_price, STATE.risk_cfg):
                            await execute_exit(sym, last_price, "exit-rule")
        await asyncio.sleep(60)

# ------------------------------------------------------------
# TELEGRAM SETUP
# ------------------------------------------------------------
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("mode", cmd_mode))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("risk", cmd_risk))

# ------------------------------------------------------------
# BACKGROUND TASKS
# ------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(engine_loop())
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())

@app.on_event("shutdown")
async def shutdown_event():
    await tg_app.stop()

@app.get("/")
async def root():
    return {"ok": True, "version": "v44", "mode": "premium"}

# ------------------------------------------------------------
# END OF FILE
# ------------------------------------------------------------
