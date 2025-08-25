# main_ai_fees.py
import os
import asyncio
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler

# ===================== ENV =====================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")

TIMEFRAME = os.getenv("TIMEFRAME", "1m")       # 1m,3m,5m,15m,30m,1h
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "100"))

# Handelsavgift per sida (entry & exit). 0.001 = 0.1% per fill
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))

# AI-inställningar
AI_LR = float(os.getenv("AI_LR", "0.05"))            # inlärningshastighet
AI_MIN_BARS = int(os.getenv("AI_MIN_BARS", "40"))    # min stängda candles för features
AI_THRESH_LONG = float(os.getenv("AI_THRESH_LONG", "0.60"))  # tröskel för long
AI_THRESH_SHORT = float(os.getenv("AI_THRESH_SHORT", "0.60")) # tröskel för short
ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "1") == "1"          # mock-short tillåten

# Stop/exit
ENGINE_TICK_SEC = int(os.getenv("ENGINE_TICK_SEC", "3"))
STOP_MODE = os.getenv("STOP_MODE", "close")          # close|tick
MAX_BARS_IN_TRADE = int(os.getenv("MAX_BARS_IN_TRADE", "80"))  # fail-safe exit

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ===================== STATE =====================
@dataclass
class Position:
    side: str              # "LONG" / "SHORT"
    entry: float           # inkl entry-avgift (justerat pris)
    stop: float
    qty: float
    bars_held: int = 0

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl: float = 0.0
    trades: List[Tuple[str, float]] = field(default_factory=list)
    last_train_ts: Optional[int] = None           # ms på senast tränade stängda candle
    prev_feat: Optional[np.ndarray] = None        # features(t-1)
    prev_close: Optional[float] = None            # close(t-1)
    w: Optional[np.ndarray] = None                # modellvikter (D+1 med bias)

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

# ===================== UI (endast neder-knappar) =====================
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/th_up"), KeyboardButton("/th_down")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ===================== DATA =====================
async def get_klines(symbol: str, tf: str, limit: int = 150) -> List[Tuple[int,float,float,float,float]]:
    """Returnerar lista av (startMs, open, high, low, close) – nyast först."""
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

# ===================== AI – features & modell =====================
def ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1.0)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    rs = np.zeros_like(closes)
    rsi_vals = np.zeros_like(closes)
    if len(closes) < period + 1:
        return rsi_vals
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs[period] = (avg_gain / (avg_loss + 1e-12))
    rsi_vals[period] = 100 - (100 / (1 + rs[period]))
    for i in range(period+1, len(closes)):
        avg_gain = (avg_gain*(period-1) + gains[i-1]) / period
        avg_loss = (avg_loss*(period-1) + losses[i-1]) / period
        rs[i] = (avg_gain / (avg_loss + 1e-12))
        rsi_vals[i] = 100 - (100 / (1 + rs[i]))
    return rsi_vals

def make_features(rows_closed: List[Tuple[int,float,float,float,float]]) -> np.ndarray:
    """Bygger features från stängda candles (kronologisk ordning)."""
    # rows_closed ska vara äldst -> nyast
    closes = np.array([r[4] for r in rows_closed], dtype=float)
    highs  = np.array([r[2] for r in rows_closed], dtype=float)
    lows   = np.array([r[3] for r in rows_closed], dtype=float)

    # avkastningar
    ret1 = np.zeros_like(closes)
    ret1[1:] = closes[1:] / closes[:-1] - 1.0
    ret3 = np.zeros_like(closes)
    ret3[3:] = closes[3:] / closes[:-3] - 1.0

    # trend/EMA
    ema_fast = ema(closes, 8)
    ema_slow = ema(closes, 21)
    ema_ratio = (ema_fast / (ema_slow + 1e-12)) - 1.0

    # volatilitet
    vol10 = np.zeros_like(closes)
    for i in range(10, len(closes)):
        vol10[i] = np.std(ret1[i-10:i])

    # RSI
    rsi14 = rsi(closes, 14) / 100.0  # 0..1

    # Normalisera några inputs
    feats = np.stack([
        ret1, ret3, ema_ratio, vol10,
        (highs - closes) / (closes + 1e-12),
        (closes - lows) / (closes + 1e-12),
        rsi14
    ], axis=1)

    # senaste rad (nyaste stängda candle) som features
    x = feats[-1]
    # klipp extrema
    x = np.clip(x, -0.2, 0.2)
    return x.astype(float)

def sigmoid(z: float) -> float:
    # numeriskt stabil
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def predict_prob(w: np.ndarray, x: np.ndarray) -> float:
    xb = np.append(x, 1.0)  # bias
    return sigmoid(float(np.dot(w, xb)))

def sgd_update(w: np.ndarray, x: np.ndarray, y: int, lr: float) -> np.ndarray:
    # logistisk regression – gradient = (y - p)*x
    xb = np.append(x, 1.0)
    p = sigmoid(float(np.dot(w, xb)))
    w = w + lr * (y - p) * xb
    return w

# ===================== Avgifts-hjälp =====================
def round_turn_fees(entry_px: float, exit_px: float, qty: float) -> float:
    return FEE_RATE * entry_px * qty + FEE_RATE * exit_px * qty

def qty_from_usdt(price: float) -> float:
    return (POSITION_SIZE_USDT / price) if price > 0 else 0.0

# ===================== Stop/exit =====================
def trail_stop_long(cur_stop: float, prev_closed: Tuple[int,float,float,float,float]) -> float:
    # flytta upp mot föregående stängda candlens low
    _, o,h,l,c = prev_closed
    return max(cur_stop, l)

def trail_stop_short(cur_stop: float, prev_closed: Tuple[int,float,float,float,float]) -> float:
    # flytta ned mot föregående stängda candlens high
    _, o,h,l,c = prev_closed
    return min(cur_stop, h)

def stop_hit_close_long(stop: float, closed: Tuple[int,float,float,float,float]) -> bool:
    return closed[4] < stop

def stop_hit_close_short(stop: float, closed: Tuple[int,float,float,float,float]) -> bool:
    return closed[4] > stop

def stop_hit_tick_long(stop: float, candle: Tuple[int,float,float,float,float]) -> bool:
    return candle[3] <= stop  # low <= stop

def stop_hit_tick_short(stop: float, candle: Tuple[int,float,float,float,float]) -> bool:
    return candle[2] >= stop  # high >= stop

# ===================== ENGINE =====================
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]

                    # Hämta minst AI_MIN_BARS + lite buffert (inkl "current")
                    kl = await get_klines(sym, STATE.timeframe, limit=max(150, AI_MIN_BARS+5))
                    if len(kl) < 3:
                        continue

                    # Nyast först enligt API:
                    current = kl[0]         # antagen pågående/nyaste
                    last_closed = kl[1]     # senaste stängda
                    closed_hist = kl[1:AI_MIN_BARS+2]  # en bunt stängda
                    if len(closed_hist) < AI_MIN_BARS:
                        continue
                    closed_hist = list(reversed(closed_hist))  # äldst -> nyast

                    # Init vikter
                    if st.w is None:
                        st.w = np.zeros(7 + 1, dtype=float)  # 7 features + bias

                    # 1) Träna online på tidigare bar när ny stängning kommit
                    last_ts = last_closed[0]
                    if st.last_train_ts is None or last_ts > st.last_train_ts:
                        x_now = make_features(closed_hist)           # x_t (på senaste stängda)
                        # Label y_{t-1}: om vi har prev_feat & prev_close, jämför nuvarande close mot prev_close
                        if st.prev_feat is not None and st.prev_close is not None:
                            y = 1 if (last_closed[4] > st.prev_close) else 0
                            st.w = sgd_update(st.w, st.prev_feat, y, AI_LR)
                        st.prev_feat = x_now
                        st.prev_close = last_closed[4]
                        st.last_train_ts = last_ts

                    # 2) Entrésignal (ingen position)
                    if st.pos is None and st.prev_feat is not None:
                        p_up = predict_prob(st.w, st.prev_feat)

                        # LONG-entry
                        if p_up >= AI_THRESH_LONG:
                            price = last_closed[4]
                            qty = qty_from_usdt(price)
                            # justera pris för entryavgift
                            entry_px = price * (1 + FEE_RATE)
                            # start-stopp: föreg. stängda low
                            sl = last_closed[3]
                            st.pos = Position("LONG", entry=entry_px, stop=sl, qty=qty)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY LONG {sym} @ {entry_px:.6f} (inkl avgift) | SL {sl:.6f} | p_up={p_up:.2f}"
                                )

                        # SHORT-entry (mock) om tillåtet
                        elif ALLOW_SHORTS and (1 - p_up) >= AI_THRESH_SHORT:
                            price = last_closed[4]
                            qty = qty_from_usdt(price)
                            # short: vi "säljer" först → räkna entryavgift
                            entry_px = price * (1 - FEE_RATE)  # effekt av avgift på intäkt
                            sl = last_closed[2]  # high
                            st.pos = Position("SHORT", entry=entry_px, stop=sl, qty=qty)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔴 ENTRY SHORT {sym} @ {entry_px:.6f} (inkl avgift) | SL {sl:.6f} | p_dn={(1-p_up):.2f}"
                                )

                    # 3) Hantera öppen position: trail + exit
                    if st.pos:
                        pos = st.pos
                        # Trailing
                        if pos.side == "LONG":
                            new_sl = trail_stop_long(pos.stop, last_closed)
                            if new_sl > pos.stop:
                                pos.stop = new_sl
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id,
                                        f"🔧 TRAIL {sym} LONG SL → {pos.stop:.6f}")
                        else:
                            new_sl = trail_stop_short(pos.stop, last_closed)
                            if new_sl < pos.stop:
                                pos.stop = new_sl
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id,
                                        f"🔧 TRAIL {sym} SHORT SL → {pos.stop:.6f}")

                        # Exit-regler
                        do_exit = False
                        exit_px_raw = last_closed[4]
                        if STOP_MODE == "close":
                            if pos.side == "LONG" and stop_hit_close_long(pos.stop, last_closed):
                                do_exit = True
                            if pos.side == "SHORT" and stop_hit_close_short(pos.stop, last_closed):
                                do_exit = True
                        else:  # tick
                            if pos.side == "LONG" and stop_hit_tick_long(pos.stop, current):
                                do_exit = True
                                exit_px_raw = pos.stop  # antag träff
                            if pos.side == "SHORT" and stop_hit_tick_short(pos.stop, current):
                                do_exit = True
                                exit_px_raw = pos.stop

                        # Fail-safe: max antal barer i trade
                        pos.bars_held += 1
                        if pos.bars_held >= MAX_BARS_IN_TRADE:
                            do_exit = True

                        if do_exit:
                            if pos.side == "LONG":
                                # sälj med avgift
                                exit_px = exit_px_raw * (1 - FEE_RATE)
                                pnl = (exit_px - pos.entry) * pos.qty
                            else:
                                # köp tillbaka med avgift
                                exit_px = exit_px_raw * (1 + FEE_RATE)
                                pnl = (pos.entry - exit_px) * pos.qty

                            st.realized_pnl += pnl
                            st.trades.append((f"{sym} {pos.side}", pnl))
                            if len(st.trades) > 80:
                                st.trades = st.trades[-80:]
                            if STATE.chat_id:
                                sign = "✅" if pnl >= 0 else "❌"
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"⏹ EXIT {sym} {pos.side} @ {exit_px:.6f} | PnL: {pnl:+.4f} USDT {sign}"
                                )
                            st.pos = None

            await asyncio.sleep(ENGINE_TICK_SEC)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ===================== TELEGRAM =====================
tg_app = Application.builder().token(BOT_TOKEN).build()

async def send_status(chat_id: int):
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            p = st.pos
            pos_lines.append(f"{s}: {p.side} @ {p.entry:.6f} | SL {p.stop:.6f} | QTY {p.qty:.6f} | bars {p.bars_held}")
    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"TF: {STATE.timeframe} | Fee/side: {FEE_RATE:.4%}",
        f"AI: lr={AI_LR} | th_long={AI_THRESH_LONG:.2f} | th_short={AI_THRESH_SHORT:.2f} | shorts={'ON' if ALLOW_SHORTS else 'OFF'}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"PnL total: {total_pnl:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    lines = [f"📈 PnL total: {total_pnl:+.4f} USDT"]
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        lines.append(f"• {s}: {ss.realized_pnl:+.4f} USDT")
    # senaste affärer
    last = []
    for s in STATE.symbols:
        for lbl, p in STATE.per_sym[s].trades[-5:]:
            last.append((lbl, p))
    if last:
        lines.append("\nSenaste affärer:")
        for lbl, p in last[-10:]:
            lines.append(f"  - {lbl}: {p:+.4f} USDT")
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon (endast neder-knappar)
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "🤖 AI-bot med avgifter redo ✅", reply_markup=reply_kb())
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
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframe → {STATE.timeframe}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        ss.realized_pnl = 0.0
        ss.trades.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            p = st.pos
            mkt = st.prev_close or p.entry
            if p.side == "LONG":
                exit_px = mkt * (1 - FEE_RATE)
                pnl = (exit_px - p.entry) * p.qty
            else:
                exit_px = mkt * (1 + FEE_RATE)
                pnl = (p.entry - exit_px) * p.qty
            st.realized_pnl += pnl
            st.trades.append((f"{s} {p.side} (panic)", pnl))
            closed.append(f"{s}:{p.side} {pnl:+.4f}")
            st.pos = None
    msg = " | ".join(closed) if closed else "Inga positioner öppna."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_th_up(update: Update, _):
    global AI_THRESH_LONG, AI_THRESH_SHORT
    STATE.chat_id = update.effective_chat.id
    AI_THRESH_LONG = min(0.95, AI_THRESH_LONG + 0.02)
    AI_THRESH_SHORT = min(0.95, AI_THRESH_SHORT + 0.02)
    await send_status(STATE.chat_id)

async def cmd_th_down(update: Update, _):
    global AI_THRESH_LONG, AI_THRESH_SHORT
    STATE.chat_id = update.effective_chat.id
    AI_THRESH_LONG = max(0.50, AI_THRESH_LONG - 0.02)
    AI_THRESH_SHORT = max(0.50, AI_THRESH_SHORT - 0.02)
    await send_status(STATE.chat_id)

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("th_up", cmd_th_up))
tg_app.add_handler(CommandHandler("th_down", cmd_th_down))

# ===================== FASTAPI WEBHOOK =====================
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    # Telegram webhook
    if WEBHOOK_BASE:
        await tg_app.initialize()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}",
                                     allowed_updates=["message", "callback_query"])
        await tg_app.start()
    # Engine
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await tg_app.bot.delete_webhook()
    except Exception:
        pass
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tf": STATE.timeframe,
        "fee_rate": FEE_RATE,
        "ai": {"lr": AI_LR, "th_long": AI_THRESH_LONG, "th_short": AI_THRESH_SHORT, "shorts": ALLOW_SHORTS},
        "pnl_total": round(total_pnl, 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
