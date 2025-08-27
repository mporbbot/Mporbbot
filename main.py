# main_v36_ai_learn.py
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

SYMBOLS = (
    os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
    .replace(" ", "")
    .split(",")
)

# Du kan ange flera TF separerat med komma, t.ex. "1m,3m,5m"
TIMEFRAMES = (
    os.getenv("TIMEFRAMES", "1m")
    .replace(" ", "")
    .split(",")
)

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
# Mock-avgift per sida (t.ex. 0.001 = 0.10% per köp OCH per sälj)
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))

MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/data/ai_model.json")

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ========= UI =========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/th_up"), KeyboardButton("/th_down")],
        [KeyboardButton("/save_model"), KeyboardButton("/load_model")],
        [KeyboardButton("/export_csv")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ========= DATA =========
async def get_klines(symbol: str, tf: str, limit: int = 30) -> List[Tuple[int,float,float,float,float,float]]:
    """Returnera lista av (startMs, open, high, low, close, volume) – nyast först."""
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # newest first
    out = []
    for row in data[:limit]:
        # KuCoin schema: [time, open, close, high, low, volume, turnover]
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4]); v = float(row[5])
        out.append((t_ms, o, h, l, c, v))
    return out

def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for x in values[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

# ========= AI-MODELL =========
@dataclass
class Weights:
    w: List[float] = field(default_factory=list)
    b: float = 0.0

    def score(self, x: List[float]) -> float:
        if len(self.w) != len(x):
            # Om feature-dim ändrats, nolla/resize
            self.w = [0.0] * len(x)
        return sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

    def update(self, x: List[float], desired_sign: int, lr: float = 0.05, l2: float = 1e-4):
        # Enkel perceptron-uppdatering med lite L2-dämpning
        if len(self.w) != len(x):
            self.w = [0.0] * len(x)
        for i in range(len(self.w)):
            self.w[i] = (1 - l2) * self.w[i] + lr * desired_sign * x[i]
        self.b = (1 - l2) * self.b + lr * desired_sign

@dataclass
class ModelStore:
    # Per symbol och timeframe egna vikter
    w: Dict[str, Dict[str, Weights]] = field(default_factory=dict)
    # Aggregat-vikter per TF (vikta röst/medel). Start 1.0
    tf_weight: Dict[str, float] = field(default_factory=dict)

    def ensure(self, symbol: str, tf: str, feat_dim: int):
        self.w.setdefault(symbol, {})
        if tf not in self.w[symbol]:
            self.w[symbol][tf] = Weights([0.0]*feat_dim, 0.0)
        # tf-vikt default 1.0
        self.tf_weight.setdefault(tf, 1.0)

    def save(self, path: str):
        js = {
            "w": {s: {tf: {"w": wt.w, "b": wt.b} for tf, wt in tfmap.items()} for s, tfmap in self.w.items()},
            "tf_weight": self.tf_weight,
        }
        with open(path, "w") as f:
            json.dump(js, f)

    def load(self, path: str):
        with open(path, "r") as f:
            js = json.load(f)
        self.tf_weight = js.get("tf_weight", {})
        self.w = {}
        for s, tfmap in js.get("w", {}).items():
            self.w[s] = {}
            for tf, wb in tfmap.items():
                self.w[s][tf] = Weights(wb.get("w", []), wb.get("b", 0.0))

MODEL = ModelStore()

# ========= STATE =========
@dataclass
class Position:
    side: str   # "LONG" eller "SHORT"
    entry: float
    stop: float
    qty: float
    fee_in: float
    entry_features: Dict[str, List[float]]   # per TF features för uppdatering
    entry_scores: Dict[str, float]
    entry_time: int

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_gross: float = 0.0
    realized_fees: float = 0.0
    last_decision_ts: int = 0  # senaste besluts-candle (gating)
    trades: List[Tuple[str, float]] = field(default_factory=list)  # ("LONG"/"SHORT", netPnL)

@dataclass
class EngineState:
    engine_on: bool = False
    timeframes: List[str] = field(default_factory=lambda: TIMEFRAMES.copy())
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    th_up: float = 0.20   # score >= th_up -> LONG
    th_down: float = -0.20  # score <= th_down -> SHORT
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

TRADES_CSV = "/mnt/data/trades_ai.csv"

# ========= FEATURES =========
def features_from_klines(kl: List[Tuple[int,float,float,float,float,float]]) -> Tuple[List[float], int]:
    """
    Beräkna features från SENASTE STÄNGDA candle (kl[1]) och historik.
    Returnerar (features_list, closed_ts)
    """
    if len(kl) < 6:
        return [], 0
    # Nyast först
    # kl[0] = aktuell forming, kl[1] = senast STÄNGDA
    last_closed = kl[1]
    prev = kl[2]
    prev2 = kl[3]

    t, o,h,l,c,v = last_closed
    _, po,ph,pl,pc,pv = prev
    _, p2o,p2h,p2l,p2c,p2v = prev2

    rng = max(1e-9, h - l)
    body = (c - o) / max(1e-9, o)
    ret1 = (c - pc) / max(1e-9, pc)
    ret2 = (pc - p2c) / max(1e-9, p2c)
    pos_in_range = (c - l) / rng
    upper_wick = (h - max(c, o)) / rng
    lower_wick = (min(c, o) - l) / rng
    vol_change = (v - pv) / max(1e-9, pv)

    closes = [row[4] for row in kl[::-1]]  # äldst->nyast
    ema9 = ema(closes, 9)[-2]  # motsvarar stängd candle (näst sista i ema-serien)
    above_ema = 1.0 if c > ema9 else 0.0
    cross_up = 1.0 if (pc <= ema(closes, 9)[-3] and c > ema9) else 0.0
    cross_dn = 1.0 if (pc >= ema(closes, 9)[-3] and c < ema9) else 0.0

    feats = [
        body, ret1, ret2, pos_in_range,
        upper_wick, lower_wick, vol_change,
        above_ema, cross_up, cross_dn
    ]
    return feats, int(t)

def aggregate_score(symbol: str, feats_per_tf: Dict[str, List[float]]) -> Tuple[float, Dict[str, float]]:
    scores = {}
    total = 0.0
    wsum = 0.0
    for tf, x in feats_per_tf.items():
        MODEL.ensure(symbol, tf, len(x))
        s = MODEL.w[symbol][tf].score(x)
        scores[tf] = s
        w = MODEL.tf_weight.get(tf, 1.0)
        total += w * s
        wsum += w
    agg = total / max(1e-9, wsum)
    return agg, scores

def qty_from_entry(px: float) -> float:
    return POSITION_SIZE_USDT / px if px > 0 else 0.0

# ========= ENGINE =========
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    try:
                        feats_per_tf: Dict[str, List[float]] = {}
                        closed_ts = 0
                        # hämta klines för alla TF
                        for tf in STATE.timeframes:
                            kl = await get_klines(sym, tf, limit=40)
                            f, ts = features_from_klines(kl)
                            if not f:
                                feats_per_tf = {}
                                break
                            feats_per_tf[tf] = f
                            # besluts-timestamp = snabbaste (min) candlets tid
                            if closed_ts == 0 or ts < closed_ts:
                                closed_ts = ts
                        if not feats_per_tf:
                            continue

                        st = STATE.per_sym[sym]

                        # Gating: bara 1 beslut per stängd candle
                        if closed_ts <= st.last_decision_ts:
                            continue
                        st.last_decision_ts = closed_ts

                        # 1) Trailing/Exit om position finns (per stängd candle)
                        if st.pos:
                            # trailing stop baserat på stängd candle:
                            # long: SL följer min(low, tidigare); short: max(high, tidigare)
                            kl_fast = await get_klines(sym, STATE.timeframes[0], limit=3)
                            last_closed_fast = kl_fast[1]
                            _, o,h,l,c,v = last_closed_fast

                            if st.pos.side == "LONG":
                                new_sl = max(st.pos.stop, l)
                                st.pos.stop = new_sl
                                # stop hit?
                                if c < st.pos.stop:
                                    exit_px = c
                                    fee_out = exit_px * st.pos.qty * FEE_PER_SIDE
                                    gross = (exit_px - st.pos.entry) * st.pos.qty
                                    net = gross - st.pos.fee_in - fee_out
                                    st.realized_gross += gross
                                    st.realized_fees += (st.pos.fee_in + fee_out)
                                    st.trades.append(("LONG", net))
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔴 EXIT {sym} @ {exit_px:.4f} | PnL Net: {net:+.4f} USDT "
                                        f"(avgifter in:{st.pos.fee_in:.4f} ut:{fee_out:.4f})"
                                    )
                                    # AI-uppdatering (reward)
                                    a = +1  # long
                                    r = 1 if net >= 0 else -1
                                    desired = a if r > 0 else -a
                                    for tf, x in st.pos.entry_features.items():
                                        MODEL.w[sym][tf].update(x, desired_sign=desired)
                                    st.pos = None

                            else:  # SHORT
                                new_sl = min(st.pos.stop, h)
                                st.pos.stop = new_sl
                                if c > st.pos.stop:
                                    exit_px = c
                                    fee_out = exit_px * st.pos.qty * FEE_PER_SIDE
                                    gross = (st.pos.entry - exit_px) * st.pos.qty
                                    net = gross - st.pos.fee_in - fee_out
                                    st.realized_gross += gross
                                    st.realized_fees += (st.pos.fee_in + fee_out)
                                    st.trades.append(("SHORT", net))
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔴 EXIT {sym} @ {exit_px:.4f} | PnL Net: {net:+.4f} USDT "
                                        f"(avgifter in:{st.pos.fee_in:.4f} ut:{fee_out:.4f})"
                                    )
                                    a = -1  # short
                                    r = 1 if net >= 0 else -1
                                    desired = a if r > 0 else -a
                                    for tf, x in st.pos.entry_features.items():
                                        MODEL.w[sym][tf].update(x, desired_sign=desired)
                                    st.pos = None

                        # 2) Entry (om ingen position)
                        if st.pos is None and feats_per_tf:
                            agg_score, s_per = aggregate_score(sym, feats_per_tf)
                            # Hämta entry px från snabbaste TFs senaste stängda
                            kl_fast = await get_klines(sym, STATE.timeframes[0], limit=3)
                            entry_px = kl_fast[1][4]

                            if agg_score >= STATE.th_up:
                                qty = qty_from_entry(entry_px)
                                fee_in = entry_px * qty * FEE_PER_SIDE
                                # sätt initial SL (försiktigt): senaste stängda low
                                sl = kl_fast[1][3]
                                st.pos = Position(
                                    side="LONG",
                                    entry=entry_px, stop=sl, qty=qty, fee_in=fee_in,
                                    entry_features=feats_per_tf.copy(),
                                    entry_scores={tf: s for tf, s in s_per.items()},
                                    entry_time=closed_ts
                                )
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {entry_px:.4f} | SL {sl:.4f} | "
                                    f"QTY={qty:.6f} | Avgift~{fee_in:.4f} | score {agg_score:+.3f}"
                                )

                            elif agg_score <= STATE.th_down:
                                qty = qty_from_entry(entry_px)
                                fee_in = entry_px * qty * FEE_PER_SIDE
                                sl = kl_fast[1][2]  # high
                                st.pos = Position(
                                    side="SHORT",
                                    entry=entry_px, stop=sl, qty=qty, fee_in=fee_in,
                                    entry_features=feats_per_tf.copy(),
                                    entry_scores=s_per,
                                    entry_time=closed_ts
                                )
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} SHORT @ {entry_px:.4f} | SL {sl:.4f} | "
                                    f"QTY={qty:.6f} | Avgift~{fee_in:.4f} | score {agg_score:+.3f}"
                                )

                        # Autosave modellen ibland (var ~30 beslut)
                        if (st.last_decision_ts // 1000) % (30*60) == 0:
                            try:
                                MODEL.save(MODEL_PATH)
                            except:
                                pass

                        await asyncio.sleep(0.05)  # mild rate-limit mellan symboler

                    except httpx.HTTPStatusError as he:
                        if STATE.chat_id:
                            try:
                                # Skicka kort fel (t.ex. 429/502)
                                await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {he}")
                            except:
                                pass
                        await asyncio.sleep(3)
                    except Exception as e:
                        if STATE.chat_id:
                            try:
                                await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                            except:
                                pass
                        await asyncio.sleep(3)

            await asyncio.sleep(8)  # 1m-5m TF funkar fint med 8s loop

        except Exception:
            await asyncio.sleep(5)

# ========= TELEGRAM =========
tg_app = Application.builder().token(BOT_TOKEN).build()

async def send_status(chat_id: int):
    total_net = sum((s.realized_gross - s.realized_fees) for s in STATE.per_sym.values())
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(
                f"{s}: {st.pos.side} @ {st.pos.entry:.4f} | SL {st.pos.stop:.4f} | QTY {st.pos.qty:.6f}"
            )
    tf_str = ", ".join(STATE.timeframes)
    sym_pnls = ", ".join(
        [f"{s}:{(st.realized_gross - st.realized_fees):+.4f}" for s, st in STATE.per_sym.items()]
    )
    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Timeframes: {tf_str}    # ändra med /timeframe",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Position size: {POSITION_SIZE_USDT:.1f} USDT | Fee per sida: {FEE_PER_SIDE*100:.4f}%",
        f"Trösklar: th_up={STATE.th_up:+.2f}  th_down={STATE.th_down:+.2f}",
        f"PnL total: {total_net:+.4f} USDT",
        f"PnL per symbol: {sym_pnls if sym_pnls else '-'}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    total_net = sum((s.realized_gross - s.realized_fees) for s in STATE.per_sym.values())
    lines = [f"📈 PnL total (net): {total_net:+.4f} USDT"]
    for s, st in STATE.per_sym.items():
        lines.append(f"• {s}: {(st.realized_gross - st.realized_fees):+.4f} USDT")
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

def append_trade_csv(row: Dict):
    hdr = "time,symbol,side,entry,exit,qty,fee_in,fee_out,gross,net\n"
    need_hdr = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a") as f:
        if need_hdr:
            f.write(hdr)
        f.write(",".join(str(row[k]) for k in ["time","symbol","side","entry","exit","qty","fee_in","fee_out","gross","net"])+"\n")

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! AI-bot (utan ORB) redo ✅", reply_markup=reply_kb())
    # Försök ladda modell om finns
    if os.path.exists(MODEL_PATH):
        try:
            MODEL.load(MODEL_PATH)
            await tg_app.bot.send_message(STATE.chat_id, "AI-modell inläst från disk.", reply_markup=reply_kb())
        except Exception as e:
            await tg_app.bot.send_message(STATE.chat_id, f"Kunde inte läsa modell: {e}", reply_markup=reply_kb())
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
    """Sätt TF-lista via meddelandet som följer, t.ex. '1m,3m,5m'."""
    STATE.chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    # /timeframe 1m,3m,5m
    parts = text.split(None, 1)
    if len(parts) == 2:
        raw = parts[1].replace(" ", "")
        lst = [p for p in raw.split(",") if p in TF_MAP]
        if lst:
            STATE.timeframes = lst
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(lst)}", reply_markup=reply_kb())
            return
    # annars toggla mellan några presets
    presets = [["1m"], ["1m","3m","5m"], ["3m","5m","15m"]]
    cur = STATE.timeframes
    idx = 0
    for i, p in enumerate(presets):
        if p == cur:
            idx = (i + 1) % len(presets)
            break
    STATE.timeframes = presets[idx]
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.timeframes)}", reply_markup=reply_kb())

async def cmd_th_up(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.th_up += 0.05
    await tg_app.bot.send_message(STATE.chat_id, f"th_up -> {STATE.th_up:+.2f}", reply_markup=reply_kb())

async def cmd_th_down(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.th_down -= 0.05
    await tg_app.bot.send_message(STATE.chat_id, f"th_down -> {STATE.th_down:+.2f}", reply_markup=reply_kb())

async def cmd_save_model(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    try:
        MODEL.save(MODEL_PATH)
        await tg_app.bot.send_message(STATE.chat_id, f"Modell sparad till {MODEL_PATH}", reply_markup=reply_kb())
    except Exception as e:
        await tg_app.bot.send_message(STATE.chat_id, f"Sparfel: {e}", reply_markup=reply_kb())

async def cmd_load_model(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    try:
        MODEL.load(MODEL_PATH)
        await tg_app.bot.send_message(STATE.chat_id, "Modell inläst.", reply_markup=reply_kb())
    except Exception as e:
        await tg_app.bot.send_message(STATE.chat_id, f"Läsfel: {e}", reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    if os.path.exists(TRADES_CSV):
        await tg_app.bot.send_document(STATE.chat_id, document=open(TRADES_CSV, "rb"), filename="trades_ai.csv")
    else:
        await tg_app.bot.send_message(STATE.chat_id, "Ingen tradefil ännu.", reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            # stäng på aktuellt entry (neutral panik)
            exit_px = st.pos.entry
            fee_out = exit_px * st.pos.qty * FEE_PER_SIDE
            if st.pos.side == "LONG":
                gross = (exit_px - st.pos.entry) * st.pos.qty
            else:
                gross = (st.pos.entry - exit_px) * st.pos.qty
            net = gross - st.pos.fee_in - fee_out
            st.realized_gross += gross
            st.realized_fees += (st.pos.fee_in + fee_out)
            closed.append(f"{s} {st.pos.side} net {net:+.4f}")
            st.pos = None
    msg = " | ".join(closed) if closed else "Inga positioner."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("th_up", cmd_th_up))
tg_app.add_handler(CommandHandler("th_down", cmd_th_down))
tg_app.add_handler(CommandHandler("save_model", cmd_save_model))
tg_app.add_handler(CommandHandler("load_model", cmd_load_model))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))

# ========= FASTAPI / WEBHOOK =========
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
    # spara modellen på väg ned
    try:
        MODEL.save(MODEL_PATH)
    except:
        pass
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    total_net = sum((s.realized_gross - s.realized_fees) for s in STATE.per_sym.values())
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "timeframes": STATE.timeframes,
        "pnl_total": round(total_net, 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
