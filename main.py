# main_v41_full.py
# ------------------------------------------------------------
# MP Bot – v41 (FULL)
# - Reply-keyboard only (inga inline-knappar)
# - Mock-handel med avgifter per sida
# - Grid/DCA + TP
# - Long & short
# - Enkelt AI-score som kan läras, sparas & laddas
# - CSV-export, symbol-urval, timeframe-lista
# - FastAPI webhook (Render)
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
from fastapi import FastAPI, Request, HTTPException
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

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))       # per entry
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))                 # 0.1% default mock
MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))                            # maximalt antal symboler öppna samtidigt

# Grid/DCA standard
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))        # % mellan DCA-ben
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.5"))               # varje nästa steg * mult
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))               # varje nästa kvant. * mult
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.25"))                    # target vinst [%] från genomsnittligt pris
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.05"))            # +% per DCA-ben för att snabbare ta hem

# Risk
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "3.0"))                     # nöd-stopp om pris går x% emot genomsnittspris

# AI Model fil
AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "/mnt/data/ai_model.pkl")

# KuCoin REST (candles)
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# -----------------------------
# STATE
# -----------------------------
@dataclass
class TradeLeg:
    side: str        # "LONG" eller "SHORT"
    price: float
    qty: float
    time: datetime

@dataclass
class Position:
    side: str                     # "LONG"/"SHORT"
    legs: List[TradeLeg] = field(default_factory=list)
    avg_price: float = 0.0
    target_price: float = 0.0
    safety_count: int = 0

    def qty_total(self) -> float:
        return sum(l.qty for l in self.legs)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)   # för CSV
    last_signal_ts: Optional[int] = None                   # ms för att undvika spam
    # grid runtime
    next_step_pct: float = GRID_STEP_MIN_PCT

@dataclass
class EngineState:
    engine_on: bool = False
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
        allow_shorts=True
    ))
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# AI – väldigt enkel linjär modell per symbol som väger samman features
# weights[(symbol, tf)] = [w0_bias, w1, w2, ...]
AI_WEIGHTS: Dict[Tuple[str, str], List[float]] = {}
AI_LEARN_RATE = 0.05

# -----------------------------
# UI (reply keyboard)
# -----------------------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/ai_on"), KeyboardButton("/ai_off")],
        [KeyboardButton("/save_ai"), KeyboardButton("/load_ai")],
        [KeyboardButton("/symbols"), KeyboardButton("/export_csv")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -----------------------------
# DATA & INDICATORS
# -----------------------------
async def get_klines(symbol: str, tf: str, limit: int = 60):
    """Returnerar lista av tuples [(ts_ms, open, high, low, close)] äldst->nyast."""
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        # KuCoin returnerar nyast först; vi vänder till äldst->nyast
        data = list(reversed(r.json()["data"]))
    out = []
    for row in data[-limit:]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

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
    # glidande medel
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [50.0]*(period)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        if avg_loss == 0:
            rs = 999
        else:
            rs = avg_gain / avg_loss
        val = 100 - (100/(1+rs))
        rsis.append(val)
    return [50.0] + rsis  # längd = len(closes)

def features_from_candles(candles: List[Tuple[int,float,float,float,float]]) -> Dict[str, float]:
    closes = [c[-1] for c in candles]
    if len(closes) < 30:
        return {"mom": 0.0, "ema_fast": 0.0, "ema_slow": 0.0, "rsi": 50.0, "atrp": 0.2}

    ema_fast = ema(closes, 9)[-1]
    ema_slow = ema(closes, 21)[-1]
    mom = (closes[-1] - closes[-6]) / closes[-6] * 100.0 if closes[-6] != 0 else 0.0
    rsi_last = rsi(closes, 14)[-1]

    # ATR% approx (enkel): genomsnittlig (H-L)/Close i %
    rng = [(c[2]-c[3]) / c[-1] * 100.0 if c[-1] else 0.0 for c in candles[-20:]]
    atrp = sum(rng)/len(rng) if rng else 0.2

    return {
        "mom": mom,            # momentum %
        "ema_fast": ema_fast,  # value
        "ema_slow": ema_slow,  # value
        "rsi": rsi_last,       # 0..100
        "atrp": max(0.02, min(3.0, atrp))  # clamp 0.02..3.0
    }

# -----------------------------
# AI – enkel linjär scorer
# -----------------------------
def _wkey(symbol: str, tf: str) -> Tuple[str, str]:
    return (symbol, tf)

def ai_score(symbol: str, tf: str, feats: Dict[str, float]) -> float:
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        # init: bias, mom, ema_diff, rsi_dev
        AI_WEIGHTS[key] = [0.0, 0.7, 0.6, 0.1]
    w0, w_mom, w_ema, w_rsi = AI_WEIGHTS[key]
    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
    rsi_dev = (feats["rsi"] - 50.0) / 50.0  # -1..+1 ungefär
    raw = w0 + w_mom*feats["mom"] + w_ema*ema_diff + w_rsi*rsi_dev*100.0
    # normalisera lite
    return max(-10, min(10, raw))

def ai_learn(symbol: str, tf: str, feats: Dict[str, float], pnl_net: float):
    """Enkel förstärknings-update: +pnl => förstärka tecken på weights."""
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

# -----------------------------
# TRADING HELPERS (mock)
# -----------------------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * FEE_PER_SIDE

def _enter_leg(sym: str, side: str, price: float, usd_size: float, st: SymState) -> TradeLeg:
    qty = usd_size / price if price > 0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    # uppdatera pos
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0)
        st.next_step_pct = STATE.grid_cfg["step_min"]
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        total_qty = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total_qty or 1.0)
    # target
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
    if side == "LONG":
        st.pos.target_price = st.pos.avg_price * (1.0 + tp/100.0)
    else:
        st.pos.target_price = st.pos.avg_price * (1.0 - tp/100.0)
    return leg

def _exit_all(sym: str, price: float, st: SymState) -> float:
    """Stänger hela positionen och returnerar NET PnL (avgifter dragna)."""
    if not st.pos: return 0.0
    gross = 0.0
    fee_in = 0.0
    fee_out = 0.0
    for leg in st.pos.legs:
        usd_in = leg.qty * leg.price
        fee_in += _fee(usd_in)
        # PnL per leg (gross)
        if st.pos.side == "LONG":
            gross += leg.qty * (price - leg.price)
        else:
            gross += leg.qty * (leg.price - price)
    usd_out = st.pos.qty_total() * price
    fee_out = _fee(usd_out)
    net = gross - fee_in - fee_out

    # logg
    st.realized_pnl_net += net
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "side": st.pos.side,
        "avg_price": st.pos.avg_price,
        "exit_price": price,
        "gross": round(gross, 6),
        "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6),
        "net": round(net, 6),
        "safety_legs": st.pos.safety_count
    })
    st.pos = None
    st.next_step_pct = STATE.grid_cfg["step_min"]
    return net

# -----------------------------
# DECISION & ENGINE
# -----------------------------
def score_vote(symbol: str, feats_per_tf: Dict[str, Dict[str, float]]) -> float:
    """Returnera samlad score (+ = köp long, - = sälj/short)."""
    votes = 0.0
    for tf, feats in feats_per_tf.items():
        sc = ai_score(symbol, tf, feats) if STATE.ai_on else 0.0
        # lite heuristik även om AI av
        ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
        bias = (1.0 if ema_diff > 0 else -1.0) * min(1.0, abs(ema_diff)/1.5)
        votes += sc/10.0 + bias
    return votes

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                # hur många positioner öppna?
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.risk_cfg["max_pos"]:
                    # utvärdera signaler och öppna nya
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos:  # redan öppen
                            continue
                        feats_per_tf = {}
                        skip_sym = False
                        for tf in STATE.tfs:
                            try:
                                kl = await get_klines(sym, tf, limit=60)
                            except Exception:
                                skip_sym = True
                                break
                            feats_per_tf[tf] = features_from_candles(kl)
                        if skip_sym or not feats_per_tf:
                            continue
                        score = score_vote(sym, feats_per_tf)
                        # tröskel: > +0.35 long, < -0.35 short (om shorts tillåtna)
                        if score > 0.35:
                            # LONG
                            st = STATE.per_sym[sym]
                            _enter_leg(sym, "LONG", kl[-1][-1], STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {st.pos.legs[-1].price:.4f} | "
                                    f"SL ~ dd {STATE.risk_cfg['dd']:.2f}% | QTY {st.pos.legs[-1].qty:.6f}\n"
                                    f"TP ~ {st.pos.target_price:.4f} | Avgift~ {STATE.position_size*FEE_PER_SIDE:.4f} USDT"
                                )
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                                break
                        elif score < -0.35 and STATE.risk_cfg["allow_shorts"]:
                            st = STATE.per_sym[sym]
                            _enter_leg(sym, "SHORT", kl[-1][-1], STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔻 ENTRY {sym} SHORT @ {st.pos.legs[-1].price:.4f} | "
                                    f"SL ~ dd {STATE.risk_cfg['dd']:.2f}% | QTY {st.pos.legs[-1].qty:.6f}\n"
                                    f"TP ~ {st.pos.target_price:.4f} | Avgift~ {STATE.position_size*FEE_PER_SIDE:.4f} USDT"
                                )
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                                break

                # hantera öppna positioner (TP/DCA/Stop)
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos:
                        continue
                    # hämta senaste candle på snabbaste TF (för snabb reaktion)
                    tf0 = STATE.tfs[0] if STATE.tfs else "1m"
                    try:
                        kl = await get_klines(sym, tf0, limit=3)
                    except Exception:
                        continue
                    last = kl[-1]
                    price = last[-1]
                    avg = st.pos.avg_price
                    move_pct = (price-avg)/avg*100.0

                    # TP
                    if (st.pos.side == "LONG" and price >= st.pos.target_price) or \
                       (st.pos.side == "SHORT" and price <= st.pos.target_price):
                        net = _exit_all(sym, price, st)
                        if STATE.chat_id:
                            mark = "✅" if net >= 0 else "❌"
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🟤 EXIT {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark}"
                            )
                        # AI-update (använder features från alla TF vid entry-tid k.a.)
                        # här använder enkel feature på tf0 för feedback
                        feats = features_from_candles(kl)
                        if STATE.ai_on:
                            ai_learn(sym, tf0, feats, net)

                    else:
                        # DCA – om pris går emot med step% och säkerhetsben kvar
                        step = st.next_step_pct
                        need_dca = False
                        if st.pos.side == "LONG" and move_pct <= -step:
                            need_dca = True
                        if st.pos.side == "SHORT" and move_pct >= step:
                            need_dca = True
                        if need_dca and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                            usd = STATE.position_size * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
                            _enter_leg(sym, st.pos.side, price, usd, st)
                            st.next_step_pct = max(STATE.grid_cfg["step_min"],
                                                   st.next_step_pct * STATE.grid_cfg["step_mult"])
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} | "
                                    f"leg {st.pos.safety_count} | QTY {st.pos.legs[-1].qty:.6f}"
                                )

                        # DD-stop
                        if abs(move_pct) >= STATE.risk_cfg["dd"]:
                            net = _exit_all(sym, price, st)
                            if STATE.chat_id:
                                mark = "✅" if net >= 0 else "❌"
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"⛔ STOP {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark} (DD)"
                                )
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
            pos_lines.append(
                f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} "
                f"→ TP {st.pos.target_price:.4f} | legs {st.pos.safety_count}"
            )
    grid = STATE.grid_cfg
    risk = STATE.risk_cfg
    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"AI: {'ON 🧠' if STATE.ai_on else 'OFF'}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} USDT | Fee per sida: {STATE.fee_side:.4%}",
        (f"Grid: max_safety={grid['max_safety']} step_mult={grid['step_mult']} "
         f"step_min%={grid['step_min']} size_mult={grid['size_mult']} "
         f"tp%={grid['tp']} (+{grid['tp_bonus']}%/safety)"),
        f"Risk: dd={risk['dd']}% | max_pos={risk['max_pos']} | shorts={'ON' if risk['allow_shorts'] else 'OFF'}",
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

async def send_status(chat_id: int):
    await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v41 – redo ✅", reply_markup=reply_kb())
    await send_status(STATE.chat_id)

async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_status(STATE.chat_id)

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
    # format: /timeframe 1m,3m,5m
    msg = update.message.text.strip()
    parts = msg.split(" ", 1)
    if len(parts) == 2:
        tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.tfs)}",
                                          reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id,
        "Använd: /timeframe 1m,3m,5m", reply_markup=reply_kb())

async def cmd_grid(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
    # ex: /grid set step_mult 0.6
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
    r = STATE.risk_cfg
    await tg_app.bot.send_message(STATE.chat_id,
        (f"Risk:\n  dd={r['dd']}%\n  max_pos={r['max_pos']}\n  shorts={'ON' if r['allow_shorts'] else 'OFF'}\n"
         "Ex: /risk set dd 4.0   | /risk set max_pos 8   | /risk set allow_shorts on"),
        reply_markup=reply_kb())

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
    # bygga CSV i minnet
    rows = [["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r["time"], r["symbol"], r["side"], r["avg_price"], r["exit_price"],
                         r["gross"], r["fee_in"], r["fee_out"], r["net"], r["safety_legs"]])
    if len(rows) == 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb())
        return
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="trades.csv", caption="Export CSV")

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            tf0 = STATE.tfs[0] if STATE.tfs else "1m"
            try:
                kl = await get_klines(s, tf0, limit=1)
                px = kl[-1][-1]
            except:
                continue
            net = _exit_all(s, px, st)
            closed.append(f"{s}:{net:+.4f}")
    msg = " | ".join(closed) if closed else "Inga positioner."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl_net = 0.0
        STATE.per_sym[s].trades_log.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

async def cmd_save_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, save_ai(), reply_markup=reply_kb())

async def cmd_load_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, load_ai(), reply_markup=reply_kb())

# registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("ai_on", cmd_ai_on))
tg_app.add_handler(CommandHandler("ai_off", cmd_ai_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
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
    # webhook
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
    return f"MP Bot v41 OK | engine_on={STATE.engine_on} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tfs": STATE.tfs,
        "symbols": STATE.symbols,
        "pnl_total": round(total, 6)
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
