# main_v36_ai.py
import os
import json
import math
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler

# ========= ENV / KONFIG =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")

TIMEFRAME = os.getenv("TIMEFRAME", "1m")  # "1m","3m","5m","15m","30m","1h"
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))

# Mock avgift i bps (0.1% = 10 bps). Avgift tas vid både entry och exit.
FEE_BPS = float(os.getenv("FEE_BPS", "10"))

# AI-parametrar
EPSILON = float(os.getenv("AI_EPSILON", "0.1"))      # utforskning
LR = float(os.getenv("AI_LR", "0.02"))               # inlärningstakt
WEIGHT_DECAY = float(os.getenv("AI_DECAY", "0.999"))  # motverka drift (per avslutad trade)
TP_PCT = float(os.getenv("TP_PCT", "0.35"))          # take profit i %
SL_PCT = float(os.getenv("SL_PCT", "0.35"))          # stop loss i %
MAX_HOLD = int(os.getenv("MAX_HOLD", "10"))          # max candles i position

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

MODEL_PATH = "/mnt/data/ai_model.json"

# ========= STATE =========
@dataclass
class Position:
    side: str           # "LONG" eller "SHORT"
    entry: float
    qty: float
    tp: float
    sl: float
    opened_at_ms: int
    bars_held: int = 0

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    realized_fees: float = 0.0
    last_decision_time_ms: int = 0

@dataclass
class EngineState:
    engine_on: bool = False
    timeframe: str = TIMEFRAME
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# AI-modell: vikter per symbol, action=0:LONG, 1:SHORT, 2:FLAT
# x = [1(bias), rsi/100, emaFast>emaSlow, dist_to_ema, vol_norm]
N_FEATS = 5
ModelType = Dict[str, List[List[float]]]  # symbol -> 3xN_WEIGHTS
MODEL: ModelType = {}

# HTTP-klient (återanvänds)
HTTP = httpx.AsyncClient(timeout=10)

# ========= UI (bara nederknappar) =========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/save_ai"), KeyboardButton("/load_ai")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ========= DATA =========
async def get_klines(symbol: str, tf: str, limit: int = 200) -> List[Tuple[int,float,float,float,float]]:
    """Returnerar [(startMs, open, high, low, close)] nyast först."""
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    r = await HTTP.get(KUCOIN_KLINES_URL, params=params)
    r.raise_for_status()
    data = r.json()["data"]  # newest first
    out = []
    for row in data[:limit]:
        t_ms = int(row[0]) * 1000
        # KuCoin: [time, open, close, high, low, volume, turnover]
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

def ema(values: List[float], length: int) -> List[float]:
    if not values: return []
    k = 2 / (length + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out

def rsi(values: List[float], length: int = 14) -> List[float]:
    if len(values) < length + 1:
        return [50.0] * len(values)
    gains = [0.0]*len(values)
    losses = [0.0]*len(values)
    for i in range(1, len(values)):
        ch = values[i] - values[i-1]
        gains[i] = max(ch, 0.0)
        losses[i] = max(-ch, 0.0)
    avg_gain = sum(gains[1:length+1])/length
    avg_loss = sum(losses[1:length+1])/length
    rs = (avg_gain / avg_loss) if avg_loss > 0 else float('inf')
    out = [50.0]*(length) + [100 - 100/(1+rs)]
    for i in range(length+1, len(values)):
        avg_gain = (avg_gain*(length-1) + gains[i]) / length
        avg_loss = (avg_loss*(length-1) + losses[i]) / length
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float('inf')
        out.append(100 - 100/(1+rs))
    return out

def features_from_candles(candles: List[Tuple[int,float,float,float,float]]):
    # candles: nyast först. Vi jobbar på SISTA STÄNGDA -> index 1 (0 = current forming)
    if len(candles) < 60:
        return None  # för lite data
    seq = list(reversed(candles))   # äldst -> nyast
    closes = [c[4] for c in seq]
    highs  = [c[2] for c in seq]
    lows   = [c[3] for c in seq]

    ema_fast = ema(closes, 9)
    ema_slow = ema(closes, 21)
    r = rsi(closes, 14)

    i = len(closes) - 2  # senaste STÄNGDA
    c = closes[i]
    ef = ema_fast[i]; es = ema_slow[i]; rr = r[i]

    dist_to_ema = (c - es) / es if es > 0 else 0.0
    atr_like = (sum([highs[j]-lows[j] for j in range(i-14, i)]) / 14.0) if i >= 14 else (highs[i]-lows[i])
    vol_norm = atr_like / c if c > 0 else 0.0
    trend_up = 1.0 if ef > es else 0.0

    x = [1.0, rr/100.0, trend_up, dist_to_ema, vol_norm]
    stamp_ms = seq[i][0]
    price = c
    return x, stamp_ms, price

# ========= AI-MODELL =========
def init_model_for_symbol(sym: str):
    if sym in MODEL: return
    # små slumpmässiga vikter (men deterministiskt)
    MODEL[sym] = [[0.0]*N_FEATS for _ in range(3)]

def save_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w") as f:
        json.dump({"model": MODEL}, f)

def load_model():
    global MODEL
    try:
        with open(MODEL_PATH, "r") as f:
            data = json.load(f)
            MODEL = data.get("model", {})
    except FileNotFoundError:
        pass

def choose_action(sym: str, x: List[float]) -> int:
    """Returnerar 0=LONG, 1=SHORT, 2=FLAT."""
    import random
    init_model_for_symbol(sym)
    if random.random() < EPSILON:
        return random.choice([0,1,2])
    scores = []
    for a in range(3):
        w = MODEL[sym][a]
        s = sum(w[i]*x[i] for i in range(N_FEATS))
        scores.append(s)
    # argmax
    best = max(range(3), key=lambda a: scores[a])
    return best

def update_model(sym: str, action: int, x: List[float], reward: float):
    init_model_for_symbol(sym)
    # vikt-decay
    for a in range(3):
        for i in range(N_FEATS):
            MODEL[sym][a][i] *= WEIGHT_DECAY
    # uppdatera valda actionens vikter
    for i in range(N_FEATS):
        MODEL[sym][action][i] += LR * reward * x[i]

# ========= FEES / PnL =========
def fee_from_notional(notional: float) -> float:
    return notional * (FEE_BPS / 10000.0)

def side_mult(side: str) -> float:
    return 1.0 if side == "LONG" else -1.0

# ========= ENGINE =========
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    save_tick = 0
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    try:
                        kl = await get_klines(sym, STATE.timeframe, limit=200)
                    except Exception as e:
                        # mild backoff vid 429 etc.
                        if STATE.chat_id:
                            try:
                                await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                            except: pass
                        await asyncio.sleep(3)
                        continue

                    feats = features_from_candles(kl)
                    if not feats:
                        continue
                    x, stamp_ms, price = feats

                    # Hantera öppen position
                    if st.pos:
                        st.pos.bars_held += 1
                        # TP/SL
                        if st.pos.side == "LONG":
                            hit_tp = price >= st.pos.tp
                            hit_sl = price <= st.pos.sl
                        else:
                            hit_tp = price <= st.pos.tp
                            hit_sl = price >= st.pos.sl
                        timeout = st.pos.bars_held >= MAX_HOLD
                        if hit_tp or hit_sl or timeout:
                            exit_px = price
                            notional = st.pos.qty * abs(exit_px)
                            fee_exit = fee_from_notional(notional)
                            # PnL (brutto)
                            gross = side_mult(st.pos.side) * (exit_px - st.pos.entry) * st.pos.qty
                            # entry fee beräknades på entry-notional
                            entry_notional = st.pos.qty * abs(st.pos.entry)
                            fee_entry = fee_from_notional(entry_notional)
                            net = gross - fee_entry - fee_exit

                            st.realized_pnl_net += net
                            st.realized_fees += (fee_entry + fee_exit)

                            # reward = net i USDT (kan normaliseras)
                            reward = net
                            action_idx = 0 if st.pos.side == "LONG" else 1
                            update_model(sym, action_idx, x, reward)

                            if STATE.chat_id:
                                sign = "✅" if net >= 0 else "❌"
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔴 EXIT {sym} @ {exit_px:.4f} | "
                                    f"Gross: {gross:+.4f}  Fees: {(fee_entry+fee_exit):.4f}  "
                                    f"Net: {net:+.4f} USDT {sign}"
                                )
                            st.pos = None

                    # Öppna ny position om ingen är öppen
                    if st.pos is None and stamp_ms > st.last_decision_time_ms:
                        act = choose_action(sym, x)  # 0/1/2
                        st.last_decision_time_ms = stamp_ms
                        if act in (0,1):
                            side = "LONG" if act == 0 else "SHORT"
                            qty = POSITION_SIZE_USDT / price if price > 0 else 0.0
                            # fees på entry (mock)
                            fee_entry = fee_from_notional(qty * price)

                            if side == "LONG":
                                tp = price * (1 + TP_PCT/100.0)
                                sl = price * (1 - SL_PCT/100.0)
                            else:
                                tp = price * (1 - TP_PCT/100.0)
                                sl = price * (1 + SL_PCT/100.0)

                            st.pos = Position(side=side, entry=price, qty=qty, tp=tp, sl=sl,
                                              opened_at_ms=stamp_ms)

                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {side} {sym} @ {price:.4f}\n"
                                    f"TP {tp:.4f} | SL {sl:.4f} | QTY {qty:.6f}\n"
                                    f"(Mock fee entry: {fee_entry:.4f} USDT)"
                                )

                # Spara modellen då och då
                save_tick = (save_tick + 1) % 20
                if save_tick == 0:
                    save_model()

            await asyncio.sleep(3)  # skona API:et
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ========= TELEGRAM =========
tg_app = Application.builder().token(BOT_TOKEN).build()

def fmt_status() -> str:
    total_net = sum(s.realized_pnl_net for s in STATE.per_sym.values())
    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Timeframe: {STATE.timeframe}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"PnL total (NET): {total_net:+.4f} USDT",
        "Positioner:",
    ]
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(f"• {s}: {st.pos.side} @ {st.pos.entry:.4f} "
                             f"| TP {st.pos.tp:.4f} SL {st.pos.sl:.4f} "
                             f"| held {st.pos.bars_held}")
    lines.append("\n".join(pos_lines) if pos_lines else "inga")
    return "\n".join(lines)

async def send_status(chat_id: int):
    await tg_app.bot.send_message(chat_id, fmt_status(), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    lines = ["📈 PnL / avgifter:"]
    total_net = 0.0; total_fees = 0.0
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        total_net += st.realized_pnl_net
        total_fees += st.realized_fees
        lines.append(f"• {s}: Net {st.realized_pnl_net:+.4f} | Fees {st.realized_fees:.4f}")
    lines.append(f"\nTOTAL Net: {total_net:+.4f} | TOTAL Fees: {total_fees:.4f}")
    lines.append(f"Avgiftssats (mock): {(FEE_BPS/100.0):.4f}% per sida")
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! AI-bot v36-AI redo ✅", reply_markup=reply_kb())
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
    save_model()

async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    order = ["1m","3m","5m","15m","30m","1h"]
    i = order.index(STATE.timeframe) if STATE.timeframe in order else 0
    STATE.timeframe = order[(i+1) % len(order)]
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframe satt till: {STATE.timeframe}", reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            # stäng på senaste pris-kännedom (entry närmast)
            exit_px = st.pos.entry  # neutralt
            fee_exit = fee_from_notional(abs(exit_px)*st.pos.qty)
            fee_entry = fee_from_notional(abs(st.pos.entry)*st.pos.qty)
            gross = side_mult(st.pos.side)*(exit_px - st.pos.entry)*st.pos.qty
            net = gross - fee_entry - fee_exit
            st.realized_pnl_net += net
            st.realized_fees += (fee_entry + fee_exit)
            closed.append(f"{s} net {net:+.4f}")
            st.pos = None
    await tg_app.bot.send_message(STATE.chat_id, "Panic close: " + (" | ".join(closed) if closed else "inga"),
                                  reply_markup=reply_kb())
    save_model()

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        st.realized_pnl_net = 0.0
        st.realized_fees = 0.0
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

async def cmd_save_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    save_model()
    await tg_app.bot.send_message(STATE.chat_id, "AI-modell sparad.", reply_markup=reply_kb())

async def cmd_load_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    load_model()
    await tg_app.bot.send_message(STATE.chat_id, "AI-modell laddad.", reply_markup=reply_kb())

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
tg_app.add_handler(CommandHandler("load_ai", cmd_load_ai))

# ========= FASTAPI WEBHOOK =========
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    load_model()
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    save_model()
    await tg_app.stop()
    await tg_app.shutdown()
    try:
        await HTTP.aclose()
    except:
        pass

@app.get("/")
async def root():
    total_net = sum(s.realized_pnl_net for s in STATE.per_sym.values())
    return {"ok": True, "engine_on": STATE.engine_on, "tf": STATE.timeframe,
            "pnl_total_net": round(total_net, 6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
