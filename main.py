# main_v41_hybrid.py
# Grid+DCA mock-trader med AI-filter, Telegram reply-knappar, FastAPI-webhook
# Krav: python-telegram-bot==20.*, fastapi, httpx, pandas, numpy, pydantic

import os
import json
import math
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler

# ───────────────────────────────────────────────────────────────────
# ENV
# ───────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")

TIMEFRAMES = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))  # 0.1% per sida

MODEL_PATH = os.getenv("AI_MODEL_PATH", "ai_model.json")

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1hour"}

# ───────────────────────────────────────────────────────────────────
# UI (bara reply-knappar)
# ───────────────────────────────────────────────────────────────────
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/ai_on"), KeyboardButton("/ai_off")],
        [KeyboardButton("/save_ai"), KeyboardButton("/load_ai")],
        [KeyboardButton("/export_csv")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ───────────────────────────────────────────────────────────────────
# Datatyper
# ───────────────────────────────────────────────────────────────────
@dataclass
class Leg:
    price: float
    qty: float

@dataclass
class Position:
    side: str                # "LONG" eller "SHORT"
    legs: List[Leg] = field(default_factory=list)
    tp_pct: float = 0.25     # take profit från snittpris (procent)
    sl: Optional[float] = None

    @property
    def qty(self) -> float:
        return sum(l.qty for l in self.legs)

    @property
    def avg(self) -> float:
        if not self.legs: return 0.0
        v = sum(l.price * l.qty for l in self.legs)
        return v / max(self.qty, 1e-12)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_gross: float = 0.0
    realized_fees: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    # Grid
    max_safety: int = 3
    step_mult: float = 0.5       # eskalering av grid-avstånd
    step_min_pct: float = 0.15   # min % avvikelse för DCA
    size_mult: float = 1.5       # ökning av storlek per DCA-steg
    tp_pct: float = 0.25         # TP från snitt (bas)
    allow_shorts: bool = True

@dataclass
class EngineState:
    engine_on: bool = True
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    tfs: List[str] = field(default_factory=lambda: TIMEFRAMES)
    pos_size: float = POSITION_SIZE_USDT
    fee_rate: float = FEE_RATE
    # risk
    dd_limit_pct: float = 6.0
    max_open_positions: int = 8
    # AI
    ai_on: bool = True
    ai_lr: float = 0.10
    ai_thresh: float = 0.0
    # State
    per_sym: Dict[str, SymState] = field(default_factory=dict)
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# AI-linjär modell per symbol
class AiModel:
    def __init__(self):
        self.weights: Dict[str, List[float]] = {}
        self.bias: Dict[str, float] = {}

AI = AiModel()

# ───────────────────────────────────────────────────────────────────
# Hjälp – data
# ───────────────────────────────────────────────────────────────────
async def get_klines(symbol: str, tf: str, limit: int = 150) -> pd.DataFrame:
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=12) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        rows = r.json()["data"]  # newest first
    cols = ["ts","open","close","high","low","vol","turn"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","close","high","low","vol","turn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="s", utc=True)
    df = df.dropna().iloc[::-1].reset_index(drop=True)  # old→new
    return df.tail(limit)

def ema(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) == 0: return a
    alpha = 2 / (n + 1)
    s = a[0]
    out = np.empty_like(a, dtype=float)
    for i, v in enumerate(a):
        s = v if i == 0 else v*alpha + s*(1-alpha)
        out[i] = s
    return out

def rsi(series: np.ndarray, n=14) -> np.ndarray:
    if len(series) < n+1: return np.full_like(series, np.nan)
    d = np.diff(series, prepend=series[0])
    up = np.where(d>0, d, 0.0)
    dn = np.where(d<0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rs = np.divide(ru, np.maximum(rd,1e-12))
    return 100 - (100/(1+rs))

def features_from_df(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low  = df["low"].to_numpy(dtype=float)
    vol  = df["vol"].to_numpy(dtype=float)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    slope = (ema50 - ema200)/np.maximum(ema200,1e-12)*100.0
    atr = pd.Series(high-low).rolling(14).mean().to_numpy()
    atrp = np.divide(atr, np.maximum(close,1e-12))*100.0
    r = rsi(close,14)
    dv = pd.Series(vol).pct_change().to_numpy()*100.0
    f = np.array([
        np.nan_to_num(slope[-1], 0.0),
        np.nan_to_num(atrp[-1], 0.0),
        np.nan_to_num((r[-1]-50.0)/20.0, 0.0),
        np.nan_to_num(dv[-1]/50.0, 0.0)
    ], dtype=float)
    # klippningar
    f[0] = np.clip(f[0]/2.0, -5, 5)
    f[1] = np.clip(f[1]/2.0, 0, 5)
    f[3] = np.clip(f[3], -5, 5)
    return f

def ai_score(symbol: str, f: np.ndarray) -> float:
    w = np.array(AI.weights.get(symbol, [0.0]*len(f)), dtype=float)
    b = float(AI.bias.get(symbol, 0.0))
    if len(w) != len(f):
        w = np.zeros_like(f)
        AI.weights[symbol] = w.tolist()
    return float(np.dot(w, f) + b)

def ai_update(symbol: str, f: np.ndarray, reward: float, lr: float):
    sign = 1.0 if reward > 0 else -1.0
    w = np.array(AI.weights.get(symbol, [0.0]*len(f)), dtype=float)
    b = float(AI.bias.get(symbol, 0.0))
    w = w + lr * sign * f
    b = b + lr * sign * 0.1
    AI.weights[symbol] = w.tolist()
    AI.bias[symbol] = b

def fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_rate

# ───────────────────────────────────────────────────────────────────
# Grid-regler
# ───────────────────────────────────────────────────────────────────
def need_new_dca(sym: str, price: float) -> bool:
    st = STATE.per_sym[sym]
    if not st.pos: return True
    p = st.pos
    steps_done = len(p.legs)-1
    if steps_done >= st.max_safety: return False
    step_pct = st.step_min_pct * (st.step_mult ** steps_done)
    if p.side == "LONG":
        return price <= p.avg * (1 - step_pct/100)
    else:
        return price >= p.avg * (1 + step_pct/100)

def tp_hit(sym: str, price: float) -> bool:
    st = STATE.per_sym[sym]
    p = st.pos
    if not p: return False
    safety_bonus = 0.05*(len(p.legs)-1)  # 0.05% extra per säkerhetsben
    tp_pct = st.tp_pct + safety_bonus
    if p.side == "LONG":
        return price >= p.avg*(1 + tp_pct/100)
    else:
        return price <= p.avg*(1 - tp_pct/100)

# ───────────────────────────────────────────────────────────────────
# Mock-orderhjälp (avgifter in/ut)
# ───────────────────────────────────────────────────────────────────
async def notify(text: str):
    if STATE.chat_id:
        try:
            await tg_app.bot.send_message(STATE.chat_id, text)
        except Exception:
            pass

def add_trade(sym: str, rec: Dict):
    STATE.per_sym[sym].trades.append(rec)
    if len(STATE.per_sym[sym].trades) > 500:
        STATE.per_sym[sym].trades = STATE.per_sym[sym].trades[-500:]

def pnl_total() -> float:
    gross = sum(STATE.per_sym[s].realized_gross for s in STATE.symbols)
    fees  = sum(STATE.per_sym[s].realized_fees for s in STATE.symbols)
    return gross - fees

def qty_from_usdt(price: float, size_usdt: float) -> float:
    return size_usdt/max(price,1e-12)

# ───────────────────────────────────────────────────────────────────
# Beslutslogik (trend + AI-filter)
# ───────────────────────────────────────────────────────────────────
def decide_side(df: pd.DataFrame) -> Optional[str]:
    close = df["close"].to_numpy(float)
    ema50 = ema(close,50)
    ema200 = ema(close,200)
    slope = (ema50[-1]-ema200[-1])/max(ema200[-1],1e-12)*100.0
    # enkel trendregel
    if slope > 0.1 and close[-1] > ema50[-1]:
        return "LONG"
    if slope < -0.1 and close[-1] < ema50[-1]:
        return "SHORT"
    return None

# ───────────────────────────────────────────────────────────────────
# ENGINE
# ───────────────────────────────────────────────────────────────────
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if not STATE.engine_on:
                await asyncio.sleep(2)
                continue

            # enkel "risk": stoppa om för många öppna
            open_cnt = sum(1 for s in STATE.symbols if STATE.per_sym[s].pos)
            if open_cnt >= STATE.max_open_positions:
                await asyncio.sleep(2)
                continue

            for sym in STATE.symbols:
                try:
                    # kombinera flera TF via medel av features
                    f_all = []
                    last_price = None
                    for tf in STATE.tfs:
                        df = await get_klines(sym, tf, limit=220)
                        if df.empty: continue
                        f_all.append(features_from_df(df))
                        last_price = float(df["close"].iloc[-1])
                    if not f_all or last_price is None:
                        continue
                    f_stack = np.vstack(f_all)
                    f = f_stack.mean(axis=0)

                    # AI-filter
                    allow = True
                    if STATE.ai_on:
                        score = ai_score(sym, f)
                        allow = (score > STATE.ai_thresh)

                    st = STATE.per_sym[sym]

                    # TP?
                    if st.pos and tp_hit(sym, last_price):
                        p = st.pos
                        entry_usdt = sum(l.price*l.qty for l in p.legs)
                        exit_usdt  = last_price * p.qty
                        gross = (exit_usdt - entry_usdt) if p.side=="LONG" else (entry_usdt - exit_usdt)
                        fees = fee(entry_usdt) + fee(exit_usdt)
                        st.realized_gross += gross
                        st.realized_fees  += fees
                        add_trade(sym, {"time":datetime.now(timezone.utc).isoformat(),
                                        "sym":sym,"side":p.side,"avg":p.avg,"exit":last_price,
                                        "gross":gross,"fees":fees,"net":gross-fees})
                        await notify(f"🧱 EXIT {sym} @ {last_price:.4f} | Net: {gross-fees:+.4f} USDT\n"
                                     f"(avgifter in:{fee(entry_usdt):.4f} ut:{fee(exit_usdt):.4f})")
                        # AI-belöning
                        if STATE.ai_on:
                            ai_update(sym, f, reward=(gross-fees), lr=STATE.ai_lr)
                        st.pos = None
                        continue

                    # DCA?
                    if st.pos and need_new_dca(sym, last_price):
                        steps_done = len(st.pos.legs)-1
                        size = STATE.pos_size * (st.size_mult ** steps_done)
                        q = qty_from_usdt(last_price, size)
                        st.pos.legs.append(Leg(last_price, q))
                        await notify(f"🧩 DCA {sym} {st.pos.side} @ {last_price:.4f} | "
                                     f"leg {len(st.pos.legs)} | QTY {q:.6f}")
                        continue

                    # Ny position?
                    if (not st.pos) and allow:
                        # trend-beslut
                        df_main = await get_klines(sym, STATE.tfs[0], limit=220)
                        side = decide_side(df_main)
                        if side is None:
                            continue
                        if side == "SHORT" and not st.allow_shorts:
                            continue
                        q = qty_from_usdt(last_price, STATE.pos_size)
                        st.pos = Position(side=side, tp_pct=st.tp_pct, sl=None, legs=[Leg(last_price, q)])
                        await notify(("🟢 ENTRY " if side=="LONG" else "🔻 ENTRY ")
                                     + f"{sym} {side} @ {last_price:.4f} | QTY {q:.6f} | "
                                     f"Avgift~{fee(STATE.pos_size):.4f} USDT")
                except Exception as e:
                    await notify(f"⚠️ Engine-fel: {e}")
            await asyncio.sleep(2)
        except Exception as e:
            await notify(f"⚠️ Engine-loop fel: {e}")
            await asyncio.sleep(3)

# ───────────────────────────────────────────────────────────────────
# Telegram
# ───────────────────────────────────────────────────────────────────
tg_app = Application.builder().token(BOT_TOKEN).build()

def pnl_lines() -> List[str]:
    lines = [f"PnL total (NET): {pnl_total():+.4f} USDT"]
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        net = st.realized_gross - st.realized_fees
        lines.append(f"• {s}: {net:+.4f} USDT")
    return lines

async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            pos_lines.append(f"{s} {st.pos.side} avg {st.pos.avg:.4f} qty {st.pos.qty:.6f} legs {len(st.pos.legs)}")
    txt = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Timeframes: {', '.join(STATE.tfs)}  # ändra med /timeframe",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.pos_size:.1f} USDT | Fee/side: {STATE.fee_rate*100:.4f}%",
        "Grid:",
        f"  max_safety={STATE.per_sym[STATE.symbols[0]].max_safety}",
        f"  step_mult={STATE.per_sym[STATE.symbols[0]].step_mult}",
        f"  step_min%={STATE.per_sym[STATE.symbols[0]].step_min_pct}",
        f"  size_mult={STATE.per_sym[STATE.symbols[0]].size_mult}",
        f"  tp%={STATE.per_sym[STATE.symbols[0]].tp_pct} (+0.05%/safety)",
        "AI:",
        f"  ai_on={STATE.ai_on} | lr={STATE.ai_lr:.2f} | thresh={STATE.ai_thresh:.2f}",
    ] + pnl_lines() + [f"Positioner: {', '.join(pos_lines) if pos_lines else 'inga'}"]
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(txt), reply_markup=reply_kb())

async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(STATE.chat_id, "Engine: ON ✅", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    # rotera mellan presets
    presets = [["1m","3m","5m"], ["3m","5m","15m"], ["1m","5m","15m"]]
    try:
        i = presets.index(STATE.tfs)
    except ValueError:
        i = 0
    STATE.tfs = presets[(i+1) % len(presets)]
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta till: {', '.join(STATE.tfs)}",
                                  reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(pnl_lines()), reply_markup=reply_kb())

async def cmd_grid(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    s0 = STATE.symbols[0]
    st = STATE.per_sym[s0]
    msg = (f"Grid:\n"
           f"max_safety={st.max_safety}\n"
           f"step_mult={st.step_mult}\n"
           f"step_min%={st.step_min_pct}\n"
           f"size_mult={st.size_mult}\n"
           f"tp%={st.tp_pct} (+0.05%/safety)\n"
           f"Ex:\n"
           f"/grid set step_min 0.10\n"
           f"/grid set step_mult 0.40\n"
           f"/grid set size_mult 1.2\n"
           f"/grid set tp 0.10\n"
           f"/grid set max_safety 5\n")
    await tg_app.bot.send_message(STATE.chat_id, msg, reply_markup=reply_kb())

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    msg = (f"Risk:\n"
           f"dd={STATE.dd_limit_pct:.1f}% | max_pos={STATE.max_open_positions}\n"
           f"Ex:\n/risk set dd 6\n/risk set max_pos 8")
    await tg_app.bot.send_message(STATE.chat_id, msg, reply_markup=reply_kb())

async def cmd_ai_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.ai_on = True
    await tg_app.bot.send_message(STATE.chat_id, "AI-filter: PÅ ✅", reply_markup=reply_kb())

async def cmd_ai_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.ai_on = False
    await tg_app.bot.send_message(STATE.chat_id, "AI-filter: AV ⛔️", reply_markup=reply_kb())

async def cmd_save_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    data = {"weights": AI.weights, "bias": AI.bias}
    with open(MODEL_PATH, "w") as f:
        json.dump(data, f)
    await tg_app.bot.send_message(STATE.chat_id, f"AI-modell sparad → {MODEL_PATH}", reply_markup=reply_kb())

async def cmd_load_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    try:
        with open(MODEL_PATH, "r") as f:
            data = json.load(f)
        AI.weights = data.get("weights", {})
        AI.bias = data.get("bias", {})
        await tg_app.bot.send_message(STATE.chat_id, "AI-modell laddad ✅", reply_markup=reply_kb())
    except Exception as e:
        await tg_app.bot.send_message(STATE.chat_id, f"Kunde inte ladda AI: {e}", reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    rows = []
    for s in STATE.symbols:
        rows += STATE.per_sym[s].trades
    if not rows:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades ännu.", reply_markup=reply_kb())
        return
    df = pd.DataFrame(rows)
    path = "trades_export.csv"
    df.to_csv(path, index=False)
    try:
        await tg_app.bot.send_document(STATE.chat_id, document=open(path, "rb"))
    except Exception as e:
        await tg_app.bot.send_message(STATE.chat_id, f"Export misslyckades: {e}", reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            last_price = None
            try:
                df = await get_klines(s, STATE.tfs[0], limit=2)
                last_price = float(df["close"].iloc[-1])
            except:
                pass
            if last_price is None:
                continue
            p = st.pos
            entry_usdt = sum(l.price*l.qty for l in p.legs)
            exit_usdt  = last_price * p.qty
            gross = (exit_usdt - entry_usdt) if p.side=="LONG" else (entry_usdt - exit_usdt)
            fees = fee(entry_usdt) + fee(exit_usdt)
            st.realized_gross += gross
            st.realized_fees  += fees
            add_trade(s, {"time":datetime.now(timezone.utc).isoformat(),
                          "sym":s,"side":p.side,"avg":p.avg,"exit":last_price,
                          "gross":gross,"fees":fees,"net":gross-fees,"panic":True})
            st.pos = None
            closed.append(f"{s}:{gross-fees:+.4f}")
    msg = "Panic close: " + (", ".join(closed) if closed else "inga positioner.")
    await tg_app.bot.send_message(STATE.chat_id, msg, reply_markup=reply_kb())

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        ss.realized_gross = 0.0
        ss.realized_fees = 0.0
        ss.trades.clear()
        ss.pos = None
    await tg_app.bot.send_message(STATE.chat_id, "PnL nollställt och positioner stängda.", reply_markup=reply_kb())

# grid/risk setters via enkla textkommandon
async def cmd_grid_set(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip().split()
    # format: /grid set <nyckel> <värde>
    if len(text) != 4 or text[1] != "set":
        await cmd_grid(update,_)
        return
    key, val = text[2], text[3]
    s0 = STATE.symbols[0]
    st = STATE.per_sym[s0]
    try:
        if key == "step_min":   v = float(val);  [setattr(STATE.per_sym[s], "step_min_pct", v) for s in STATE.symbols]
        elif key == "step_mult":v = float(val);  [setattr(STATE.per_sym[s], "step_mult", v) for s in STATE.symbols]
        elif key == "size_mult":v = float(val);  [setattr(STATE.per_sym[s], "size_mult", v) for s in STATE.symbols]
        elif key == "tp":       v = float(val);  [setattr(STATE.per_sym[s], "tp_pct", v) for s in STATE.symbols]
        elif key == "max_safety": v=int(val);    [setattr(STATE.per_sym[s], "max_safety", v) for s in STATE.symbols]
        else:
            await tg_app.bot.send_message(STATE.chat_id, f"Okänd nyckel: {key}", reply_markup=reply_kb()); return
        await tg_app.bot.send_message(STATE.chat_id, "Grid uppdaterad.", reply_markup=reply_kb())
        await cmd_grid(update,_)
    except Exception as e:
        await tg_app.bot.send_message(STATE.chat_id, f"Fel: {e}", reply_markup=reply_kb())

async def cmd_risk_set(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip().split()
    # /risk set dd 6  eller  /risk set max_pos 8
    if len(text) != 4 or text[1] != "set":
        await cmd_risk(update,_)
        return
    key, val = text[2], text[3]
    try:
        if key == "dd":
            STATE.dd_limit_pct = float(val)
        elif key == "max_pos":
            STATE.max_open_positions = int(val)
        else:
            await tg_app.bot.send_message(STATE.chat_id, f"Okänd nyckel: {key}", reply_markup=reply_kb()); return
        await tg_app.bot.send_message(STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb())
        await cmd_risk(update,_)
    except Exception as e:
        await tg_app.bot.send_message(STATE.chat_id, f"Fel: {e}", reply_markup=reply_kb())

# register
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("grid_set", cmd_grid_set))  # backup alias om du vill
tg_app.add_handler(CommandHandler("risk_set", cmd_risk_set))
tg_app.add_handler(CommandHandler("ai_on", cmd_ai_on))
tg_app.add_handler(CommandHandler("ai_off", cmd_ai_off))
tg_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
tg_app.add_handler(CommandHandler("load_ai", cmd_load_ai))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))

# vi fångar "/grid set ..." och "/risk set ..." med generiska handlers
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))

# ───────────────────────────────────────────────────────────────────
# FastAPI webhook
# ───────────────────────────────────────────────────────────────────
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

@app.get("/")
async def root():
    return {"ok": True, "engine_on": STATE.engine_on, "tfs": STATE.tfs,
            "net": round(pnl_total(), 6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
