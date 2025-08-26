# main.py  — v36-bas + AI-policy med persistens + mock-trades m/ avgifter
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

# ========= ENV =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")

TIMEFRAME = os.getenv("TIMEFRAME", "1m")  # 1m,3m,5m,15m,30m,1h
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))  # mock-positionstorlek
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))  # 0.1% per sida (köp/sälj)

AI_STATE_PATH = os.getenv("AI_STATE_PATH", "ai_state.json")  # spar fil

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ========= STATE =========
@dataclass
class Position:
    side: str           # "LONG"
    entry: float
    stop: float
    qty: float
    entry_fee: float    # avgift USDT som betalats vid köp

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl: float = 0.0
    trades: List[Tuple[str, float]] = field(default_factory=list)  # (label, pnl)

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

# ========= AI-POLICY (enkel linjär policy + RL-liknande uppdatering) =========
# features: [1, mom1, mom3, mom5, rsi, vol_k, ema_ratio]
# action score = dot(w, features). action_long = score > 0
DEFAULT_W = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ALPHA = float(os.getenv("AI_LR", "0.01"))  # inlärningshastighet

# per symbol weights
AI_WEIGHTS: Dict[str, List[float]] = {s: DEFAULT_W[:] for s in SYMBOLS}

def ai_load():
    global AI_WEIGHTS
    try:
        with open(AI_STATE_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for s in SYMBOLS:
                if s in data and isinstance(data[s], list) and len(data[s]) == len(DEFAULT_W):
                    AI_WEIGHTS[s] = data[s]
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"AI load misslyckades: {e}")

def ai_save():
    try:
        os.makedirs(os.path.dirname(AI_STATE_PATH), exist_ok=True) if "/" in AI_STATE_PATH else None
        with open(AI_STATE_PATH, "w") as f:
            json.dump(AI_WEIGHTS, f)
    except Exception as e:
        print(f"AI save misslyckades: {e}")

def ema(series: List[float], period: int) -> float:
    if not series:
        return 0.0
    k = 2 / (period + 1)
    e = series[0]
    for x in series[1:]:
        e = x * k + e * (1 - k)
    return e

def rsi14(closes: List[float]) -> float:
    if len(closes) < 15:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-14, 0):
        ch = closes[i] - closes[i-1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def build_features(ohlc: List[Tuple[int,float,float,float,float]]) -> List[float]:
    """
    ohlc: [(ms, o, h, l, c)] nyast först
    """
    if len(ohlc) < 10:
        # nöd-features
        return [1.0, 0,0,0, 50.0, 0, 1.0]
    closes = [x[4] for x in reversed(ohlc)]  # äldst -> nyast
    highs  = [x[2] for x in reversed(ohlc)]
    lows   = [x[3] for x in reversed(ohlc)]
    vols   = [abs(highs[i]-lows[i]) for i in range(len(highs))]

    c = closes[-1]
    mom1 = c - closes[-2]
    mom3 = c - closes[-4]
    mom5 = c - closes[-6]
    r = rsi14(closes)
    vol_k = vols[-1] / (sum(vols[-6:])/6.0 + 1e-9)
    ema_fast = ema(closes[-20:], 8)
    ema_slow = ema(closes[-20:], 20)
    ema_ratio = ema_fast / (ema_slow + 1e-9)

    return [1.0, mom1, mom3, mom5, r, vol_k, ema_ratio]

def dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))

def ai_action(symbol: str, feats: List[float]) -> float:
    w = AI_WEIGHTS.get(symbol, DEFAULT_W)
    return dot(w, feats)

def ai_learn(symbol: str, feats: List[float], reward: float):
    # enkel policy gradient-liknande uppdatering (utan probabilistisk policy)
    w = AI_WEIGHTS.get(symbol, DEFAULT_W)
    AI_WEIGHTS[symbol] = [w_i + ALPHA * reward * f_i for w_i, f_i in zip(w, feats)]
    ai_save()

# ========= UTIL =========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe")],
        [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

async def get_klines(symbol: str, tf: str, limit: int = 30) -> List[Tuple[int,float,float,float,float]]:
    """Returnerar (startMs, open, high, low, close) — nyast först."""
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # newest first
    out = []
    for row in data[:limit]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

# ========= ENGINE =========
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    kl = await get_klines(sym, STATE.timeframe, limit=30)
                    if len(kl) < 6:
                        continue

                    last_closed = kl[1]  # näst nyaste är stängd
                    current     = kl[0]  # pågående (tick-simulering)
                    feats = build_features(kl[:20])
                    score = ai_action(sym, feats)
                    want_long = score > 0.0

                    # EXIT villkor: om vi är långa och "policy vänder" eller SL hit
                    if st.pos:
                        # trailing SL till föregående stängda candlens low
                        _, o,h,l,c = last_closed
                        new_sl = max(st.pos.stop, l)
                        sl_changed = new_sl > st.pos.stop
                        st.pos.stop = new_sl

                        # stop hit (tick-logik: om current low <= stop, annars om close < stop)
                        _, co, ch, cl, cc = current
                        stop_hit = (cl <= st.pos.stop) or (cc < st.pos.stop)

                        policy_flip = not want_long  # om modellen inte vill vara lång längre

                        if stop_hit or policy_flip:
                            # exit pris
                            exit_px = min(cc, st.pos.stop) if stop_hit else cc

                            # säljavgift
                            sell_fee = exit_px * st.pos.qty * FEE_RATE
                            gross = (exit_px - st.pos.entry) * st.pos.qty
                            pnl = gross - st.pos.entry_fee - sell_fee

                            st.realized_pnl += pnl
                            st.trades.append((f"{sym} LONG EXIT", pnl))
                            if len(st.trades) > 50:
                                st.trades = st.trades[-50:]
                            if STATE.chat_id:
                                sign = "✅" if pnl >= 0 else "❌"
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔴 EXIT {sym} @ {exit_px:.6f} | PnL: {pnl:+.4f} USDT {sign}"
                                )

                            # RL-belöning = pnl i USDT (eller normalized)
                            ai_learn(sym, feats, reward=pnl)

                            st.pos = None
                        else:
                            if sl_changed and STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔧 SL {sym} → {st.pos.stop:.6f}"
                                )

                    # ENTRY villkor: om vi inte har position och policy vill vara lång
                    if (st.pos is None) and want_long:
                        entry_px = last_closed[4]  # gå in på close av föregående stängda candle
                        qty = POSITION_SIZE_USDT / entry_px if entry_px > 0 else 0.0
                        # köpavgift
                        buy_fee = entry_px * qty * FEE_RATE
                        st.pos = Position("LONG", entry=entry_px, stop=last_closed[3], qty=qty, entry_fee=buy_fee)

                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🟢 ENTRY LONG {sym} @ {entry_px:.6f}\n"
                                f"SL={st.pos.stop:.6f} | QTY={qty:.6f} | Fee(köp)={buy_fee:.4f}"
                            )

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ========= TELEGRAM =========
tg_app = Application.builder().token(BOT_TOKEN).build()

def fmt_pnl_per_symbol() -> str:
    return ", ".join([f"{s}:{STATE.per_sym[s].realized_pnl:+.2f}" for s in STATE.symbols])

async def send_status(chat_id: int):
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(
                f"{s}: LONG @ {st.pos.entry:.6f} | SL {st.pos.stop:.6f} | "
                f"QTY {st.pos.qty:.6f}"
            )

    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Timeframe: {STATE.timeframe}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Position size: {POSITION_SIZE_USDT} USDT | Fee per sida: {FEE_RATE*100:.3f}%",
        f"PnL total: {total_pnl:+.4f} USDT",
        f"PnL per symbol: {fmt_pnl_per_symbol() or '-'}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    lines = [f"📈 PnL total: {total_pnl:+.4f} USDT"]
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        lines.append(f"• {s}: {ss.realized_pnl:+.4f} USDT")
    # senaste trades
    last = []
    for s in STATE.symbols:
        for lbl, p in STATE.per_sym[s].trades[-5:]:
            last.append((lbl, p))
    if last:
        lines.append("\nSenaste affärer:")
        for lbl, p in last[-10:]:
            lines.append(f"  - {lbl}: {p:+.4f} USDT")
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon (v36-stil, inga inline-knappar)
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! AI-bot redo ✅ (v36-bas, ingen ORB)", reply_markup=reply_kb())
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
            exit_px = st.pos.stop
            sell_fee = exit_px * st.pos.qty * FEE_RATE
            gross = (exit_px - st.pos.entry) * st.pos.qty
            pnl = gross - st.pos.entry_fee - sell_fee
            st.realized_pnl += pnl
            st.trades.append((f"{s} LONG (panic)", pnl))
            closed.append(f"{s} pnl {pnl:+.4f}")
            # lär policy av utfallet
            feats = [1,0,0,0,50,1,1]  # dummy (ingen ny data kopplad här)
            ai_learn(s, feats, reward=pnl)
            st.pos = None
    msg = " | ".join(closed) if closed else "Inga positioner öppna."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl = 0.0
        STATE.per_sym[s].trades.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))

# ========= FASTAPI / WEBHOOK =========
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    ai_load()
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()
    ai_save()

@app.get("/")
async def root():
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tf": STATE.timeframe,
        "pnl_total": round(total_pnl, 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
