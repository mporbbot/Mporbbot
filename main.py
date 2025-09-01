# main_v42.py
# ------------------------------------------------------------
# MP Bot – v42
# - Reply-keyboard
# - Mock ELLER KuCoin LIVE (spot) – växla med /start_live /start_mock
# - Grid/DCA + TP, risk-DD-stop
# - AI-score som kan sparas/laddas
# - Välj coins & timeframes via kommandon
# - CSV-export för trades + K4 (bilaga K4 avsnitt D, svenska regler)
# - FastAPI webhook för Render
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

# ----------- ENV ------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = (os.getenv("WEBHOOK_BASE", "") or "").rstrip("/")

KUCOIN_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

DEFAULT_SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
                   .replace(" ", "")).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))  # 0.1% mock

GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.6"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.25"))
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.05"))

DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "3.0"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))

AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "./data/ai_model.pkl")

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# ---------- STATE ----------
@dataclass
class TradeLeg:
    side: str  # LONG/SHORT
    price: float
    qty: float
    time: datetime
    live: bool = False  # markerar om det var livefill

@dataclass
class Position:
    side: str
    legs: List[TradeLeg] = field(default_factory=list)
    avg_price: float = 0.0
    target_price: float = 0.0
    safety_count: int = 0
    live: bool = False  # position via live-ordrar

    def qty_total(self) -> float:
        return sum(l.qty for l in self.legs)

@dataclass
class TradeLogRow:
    time: str
    symbol: str
    side: str
    avg_price: float
    exit_price: float
    gross: float
    fee_in: float
    fee_out: float
    net: float
    safety_legs: int
    live: bool

@dataclass
class K4Row:
    datum: str
    beteckning: str  # t.ex. BTC
    antal: float
    ersattning: float  # försäljningspris efter utg avgift
    anskaffning: float  # anskaffningsutgift inkl. avgifter
    vinst: float  # (+/-)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[TradeLogRow] = field(default_factory=list)
    k4_rows: List[K4Row] = field(default_factory=list)  # endast LONG spot loggas hit
    last_signal_ts: Optional[int] = None
    next_step_pct: float = GRID_STEP_MIN_PCT

@dataclass
class EngineState:
    running: bool = False
    mode_live: bool = False  # False=mock, True=live
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
        allow_shorts=False  # K4-stöd primärt för spot LONG
    ))
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# AI
AI_WEIGHTS: Dict[Tuple[str, str], List[float]] = {}
AI_LEARN_RATE = 0.05

# ------------- UI -------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/start_mock"), KeyboardButton("/start_live")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/ai_on"), KeyboardButton("/ai_off")],
        [KeyboardButton("/symbols"), KeyboardButton("/export_csv")],
        [KeyboardButton("/export_k4"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -------- DATA/INDICATORS ----
async def get_klines(symbol: str, tf: str, limit: int = 60):
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
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [50.0]*(period)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss else 999
        rsis.append(100 - (100/(1+rs)))
    return [50.0] + rsis

def features_from_candles(candles):
    closes = [c[-1] for c in candles]
    if len(closes) < 30:
        return {"mom": 0.0, "ema_fast": closes[-1] if closes else 0.0,
                "ema_slow": closes[-1] if closes else 0.0, "rsi": 50.0, "atrp": 0.2}
    ema_fast = ema(closes, 9)[-1]
    ema_slow = ema(closes, 21)[-1]
    mom = (closes[-1] - closes[-6]) / closes[-6] * 100.0 if closes[-6] else 0.0
    rsi_last = rsi(closes, 14)[-1]
    rng = [(c[2]-c[3]) / (c[-1] or 1) * 100.0 for c in candles[-20:]]
    atrp = sum(rng)/len(rng) if rng else 0.2
    return {"mom": mom, "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": rsi_last, "atrp": max(0.02, min(3.0, atrp))}

# -------- AI ----------
def _wkey(symbol: str, tf: str): return (symbol, tf)

def ai_score(symbol: str, tf: str, feats: Dict[str, float]) -> float:
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        AI_WEIGHTS[key] = [0.0, 0.7, 0.6, 0.1]  # bias, mom, ema_diff, rsi_dev
    w0, w_mom, w_ema, w_rsi = AI_WEIGHTS[key]
    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1) * 100.0
    rsi_dev = (feats["rsi"] - 50.0) / 50.0
    return max(-10, min(10, w0 + w_mom*feats["mom"] + w_ema*ema_diff + w_rsi*rsi_dev*100.0))

def ai_learn(symbol: str, tf: str, feats: Dict[str, float], pnl_net: float):
    key = _wkey(symbol, tf)
    if key not in AI_WEIGHTS:
        AI_WEIGHTS[key] = [0.0, 0.7, 0.6, 0.1]
    w = AI_WEIGHTS[key]
    ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1) * 100.0
    rsi_dev = (feats["rsi"] - 50.0) / 50.0
    grad = [1.0, feats["mom"], ema_diff, rsi_dev*100.0]
    lr = AI_LEARN_RATE * (1 if pnl_net >= 0 else -1)
    AI_WEIGHTS[key] = [w[i] + lr*grad[i] for i in range(4)]

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_ai() -> str:
    try:
        _ensure_dir(AI_MODEL_PATH)
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
    except FileNotFoundError:
        return "Ingen AI-fil hittades (spara först med /save_ai)."
    except Exception as e:
        return f"Kunde inte ladda AI: {e}"

# ------ TRADING CORE (mock + live KuCoin spot) ------
_kucoin_client = None

def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

def _have_live_creds() -> bool:
    return all([KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE])

def _live_client():
    global _kucoin_client
    if _kucoin_client is None and _have_live_creds():
        from kucoin.client import Client
        _kucoin_client = Client(KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE)
    return _kucoin_client

def _enter_leg_mock(sym: str, side: str, price: float, usd_size: float, st: SymState) -> TradeLeg:
    qty = usd_size / price if price > 0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc), live=False)
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0, live=False)
        st.next_step_pct = STATE.grid_cfg["step_min"]
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        total_qty = st.pos.qty_total() or 1.0
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / total_qty
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
    st.pos.target_price = (st.pos.avg_price * (1.0 + tp/100.0)) if side == "LONG" \
                          else (st.pos.avg_price * (1.0 - tp/100.0))
    return leg

def _enter_leg_live(sym: str, side: str, usd_size: float, st: SymState) -> Optional[TradeLeg]:
    # market order med funds i USDT
    c = _live_client()
    if c is None:
        return None
    try:
        from kucoin.client import Market
        m = Market(url="https://api.kucoin.com")
        ticker = m.get_ticker(sym.replace("-", ""))
        price = float(ticker["price"])
    except Exception:
        # fallback via klines
        loop = asyncio.get_event_loop()
        kl = loop.run_until_complete(get_klines(sym, STATE.tfs[0], limit=1))
        price = kl[-1][-1]

    try:
        # KuCoin: symbol typ "BTC-USDT"; market order: side "buy"/"sell"; funds=USDT-belopp
        resp = c.create_market_order(sym, side.lower(), funds=str(round(usd_size, 2)))
        # kvantitet approx via utfört pris
        qty = usd_size / price if price > 0 else 0.0
        leg = TradeLeg(side="LONG" if side.upper()=="BUY" else "SHORT",
                       price=price, qty=qty, time=datetime.now(timezone.utc), live=True)
        if st.pos is None:
            st.pos = Position(side=leg.side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0, live=True)
            st.next_step_pct = STATE.grid_cfg["step_min"]
        else:
            st.pos.legs.append(leg)
            st.pos.safety_count += 1
            total_qty = st.pos.qty_total() or 1.0
            st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / total_qty
        tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
        st.pos.target_price = (st.pos.avg_price * (1.0 + tp/100.0)) if st.pos.side == "LONG" \
                              else (st.pos.avg_price * (1.0 - tp/100.0))
        return leg
    except Exception:
        return None

def _exit_all(sym: str, price: float, st: SymState) -> float:
    if not st.pos: return 0.0
    gross = 0.0; fee_in = 0.0; fee_out = 0.0
    qty_tot = st.pos.qty_total()
    for leg in st.pos.legs:
        usd_in = leg.qty * leg.price
        fee_in += _fee(usd_in)
        gross += (leg.qty * (price - leg.price)) if st.pos.side == "LONG" else (leg.qty * (leg.price - price))
    usd_out = qty_tot * price
    fee_out = _fee(usd_out)
    net = gross - fee_in - fee_out

    # K4 (endast LONG spot): ersättning = usd_out - fee_out; anskaffning = sum(usd_in)+fee_in
    if st.pos.side == "LONG":
        ers = max(0.0, usd_out - fee_out)
        ansk = max(0.0, sum(l.qty*l.price for l in st.pos.legs) + fee_in)
        vinst = ers - ansk
        st.k4_rows.append(K4Row(
            datum=datetime.now(timezone.utc).date().isoformat(),
            beteckning=sym.split("-")[0],
            antal=round(qty_tot, 8),
            ersattning=round(ers, 6),
            anskaffning=round(ansk, 6),
            vinst=round(vinst, 6),
        ))

    st.realized_pnl_net += net
    st.trades_log.append(TradeLogRow(
        time=datetime.now(timezone.utc).isoformat(),
        symbol=sym,
        side=st.pos.side,
        avg_price=st.pos.avg_price,
        exit_price=price,
        gross=round(gross, 6),
        fee_in=round(fee_in, 6),
        fee_out=round(fee_out, 6),
        net=round(net, 6),
        safety_legs=st.pos.safety_count,
        live=st.pos.live
    ))
    st.pos = None
    st.next_step_pct = STATE.grid_cfg["step_min"]
    return net

# -------- DECISION / ENGINE --------
def score_vote(symbol: str, feats_per_tf: Dict[str, Dict[str, float]]) -> float:
    votes = 0.0
    for tf, feats in feats_per_tf.items():
        sc = ai_score(symbol, tf, feats) if STATE.ai_on else 0.0
        ema_diff = (feats["ema_fast"] - feats["ema_slow"]) / (feats["ema_slow"] or 1.0) * 100.0
        bias = (1.0 if ema_diff > 0 else -1.0) * min(1.0, abs(ema_diff)/1.5)
        votes += sc/10.0 + bias
    return votes

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.running:
                # öppna nya om plats
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.risk_cfg["max_pos"]:
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos:
                            continue
                        feats_per_tf = {}
                        skip = False
                        for tf in STATE.tfs:
                            try: kl = await get_klines(sym, tf, limit=60)
                            except Exception: skip = True; break
                            feats_per_tf[tf] = features_from_candles(kl)
                        if skip or not feats_per_tf: continue
                        score = score_vote(sym, feats_per_tf)
                        if score > 0.35:
                            st = STATE.per_sym[sym]
                            tf0 = STATE.tfs[0]
                            kl = await get_klines(sym, tf0, limit=1)
                            px = kl[-1][-1]
                            if STATE.mode_live:
                                leg = _enter_leg_live(sym, "BUY", STATE.position_size, st)
                            else:
                                leg = _enter_leg_mock(sym, "LONG", px, STATE.position_size, st)
                            if leg and STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {leg.price:.4f} | "
                                    f"TP {st.pos.target_price:.4f} | {'LIVE' if STATE.mode_live else 'MOCK'}")
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                                break

                # hantera öppna
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos: continue
                    tf0 = STATE.tfs[0]
                    try: kl = await get_klines(sym, tf0, limit=2)
                    except Exception: continue
                    price = kl[-1][-1]
                    avg = st.pos.avg_price
                    move_pct = (price-avg)/avg*100.0

                    # TP
                    if (st.pos.side == "LONG" and price >= st.pos.target_price):
                        net = _exit_all(sym, price, st)
                        if STATE.chat_id:
                            mark = "✅" if net >= 0 else "❌"
                            await app.bot.send_message(STATE.chat_id, f"🟤 EXIT {sym} @ {price:.4f} | Net {net:+.4f} {mark}")
                        if STATE.ai_on:
                            ai_learn(sym, tf0, features_from_candles(kl), net)
                        continue

                    # DCA
                    step = st.next_step_pct
                    if st.pos.side == "LONG" and move_pct <= -step and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        usd = STATE.position_size * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
                        if STATE.mode_live:
                            leg = _enter_leg_live(sym, "BUY", usd, st)
                        else:
                            leg = _enter_leg_mock(sym, "LONG", price, usd, st)
                        st.next_step_pct = max(STATE.grid_cfg["step_min"], st.next_step_pct * STATE.grid_cfg["step_mult"])
                        if leg and STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🧩 DCA {sym} @ {leg.price:.4f} (leg {st.pos.safety_count})")
                        continue

                    # DD-stop
                    if abs(move_pct) >= STATE.risk_cfg["dd"]:
                        net = _exit_all(sym, price, st)
                        if STATE.chat_id:
                            mark = "✅" if net >= 0 else "❌"
                            await app.bot.send_message(STATE.chat_id, f"⛔ STOP {sym} @ {price:.4f} | Net {net:+.4f} {mark}")
            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try: await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except: pass
            await asyncio.sleep(5)

# -------- TELEGRAM ----------
tg_app = Application.builder().token(BOT_TOKEN).build()

def _status() -> str:
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} → TP {st.pos.target_price:.4f} | legs {st.pos.safety_count} | {'LIVE' if st.pos.live else 'MOCK'}")
    g = STATE.grid_cfg; r = STATE.risk_cfg
    return "\n".join([
        f"Engine: {'ON ✅' if STATE.running else 'OFF ⛔️'} | Mode: {'LIVE' if STATE.mode_live else 'MOCK'}",
        f"AI: {'ON 🧠' if STATE.ai_on else 'OFF'}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} USDT | Fee per sida (mock): {STATE.fee_side:.4%}",
        (f"Grid: max_safety={g['max_safety']} step_mult={g['step_mult']} step_min%={g['step_min']} "
         f"size_mult={g['size_mult']} tp%={g['tp']} (+{g['tp_bonus']}%/safety)"),
        f"Risk: dd={r['dd']}% | max_pos={r['max_pos']} | shorts={'ON' if r['allow_shorts'] else 'OFF'}",
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ])

async def _send_status():
    if STATE.chat_id:
        await tg_app.bot.send_message(STATE.chat_id, _status(), reply_markup=reply_kb())

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v42 – redo ✅", reply_markup=reply_kb())
    await _send_status()

async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await _send_status()

async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.running = True
    await tg_app.bot.send_message(STATE.chat_id, "Engine: ON ✅", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.running = False
    await tg_app.bot.send_message(STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

async def cmd_start_mock(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.mode_live = False
    await tg_app.bot.send_message(STATE.chat_id, "Läge: MOCK", reply_markup=reply_kb())
    await _send_status()

async def cmd_start_live(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    if not _have_live_creds():
        await tg_app.bot.send_message(STATE.chat_id, "Saknar KuCoin API-nycklar i env – stannar i MOCK.", reply_markup=reply_kb()); return
    _live_client()  # init
    STATE.mode_live = True
    await tg_app.bot.send_message(STATE.chat_id, "Läge: LIVE (KuCoin spot)", reply_markup=reply_kb())
    await _send_status()

async def cmd_ai_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id; STATE.ai_on = True
    await tg_app.bot.send_message(STATE.chat_id, "AI: ON 🧠", reply_markup=reply_kb())

async def cmd_ai_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id; STATE.ai_on = False
    await tg_app.bot.send_message(STATE.chat_id, "AI: OFF", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip()
    parts = msg.split(" ", 1)
    if len(parts) == 2:
        tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.tfs)}", reply_markup=reply_kb()); return
    await tg_app.bot.send_message(STATE.chat_id, "Använd: /timeframe 1m,3m,5m,15m", reply_markup=reply_kb())

async def cmd_grid(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks) == 4 and toks[0] == "/grid" and toks[1] == "set":
        key, val = toks[2], toks[3]
        if key in STATE.grid_cfg:
            try:
                STATE.grid_cfg[key] = float(val); await tg_app.bot.send_message(STATE.chat_id, "Grid uppdaterad.", reply_markup=reply_kb())
            except: await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
    g = STATE.grid_cfg
    await tg_app.bot.send_message(STATE.chat_id,
        (f"Grid:\n  max_safety={g['max_safety']}\n  step_mult={g['step_mult']}\n  step_min%={g['step_min']}\n"
         f"  size_mult={g['size_mult']}\n  tp%={g['tp']} (+{g['tp_bonus']}%/safety)\nEx: /grid set step_mult 0.7"),
        reply_markup=reply_kb())

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks) == 4 and toks[0] == "/risk" and toks[1] == "set":
        key, val = toks[2], toks[3]
        if key in STATE.risk_cfg:
            try:
                if key == "max_pos": STATE.risk_cfg[key] = int(val)
                elif key == "allow_shorts": STATE.risk_cfg[key] = (val.lower() in ("1","true","on","yes"))
                else: STATE.risk_cfg[key] = float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb())
            except: await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
    r = STATE.risk_cfg
    await tg_app.bot.send_message(STATE.chat_id,
        (f"Risk:\n  dd={r['dd']}%\n  max_pos={r['max_pos']}\n  shorts={'ON' if r['allow_shorts'] else 'OFF'}\n"
         "Ex: /risk set dd 4.0 | /risk set max_pos 8"), reply_markup=reply_kb())

async def cmd_symbols(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks) >= 3 and toks[0] == "/symbols":
        sym = toks[2].upper()
        if toks[1] == "add":
            if sym not in STATE.symbols:
                STATE.symbols.append(sym); STATE.per_sym[sym] = SymState()
                await tg_app.bot.send_message(STATE.chat_id, f"La till: {sym}", reply_markup=reply_kb())
            else: await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns redan.", reply_markup=reply_kb())
            return
        if toks[1] == "remove":
            if sym in STATE.symbols:
                STATE.symbols = [s for s in STATE.symbols if s != sym]; STATE.per_sym.pop(sym, None)
                await tg_app.bot.send_message(STATE.chat_id, f"Tog bort: {sym}", reply_markup=reply_kb())
            else: await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns ej.", reply_markup=reply_kb())
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
    rows = [["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs","mode"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r.time, r.symbol, r.side, r.avg_price, r.exit_price, r.gross, r.fee_in, r.fee_out, r.net, r.safety_legs, "LIVE" if r.live else "MOCK"])
    if len(rows) == 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb()); return
    buf = io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="trades.csv", caption="Export trades (NET)")

async def cmd_export_k4(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    rows = [["Datum","Beteckning","Antal","Ersättning (SEK/USDT)","Anskaffningsutgift","Vinst/Förlust"]]
    any_row = False
    for s in STATE.symbols:
        for r in STATE.per_sym[s].k4_rows:
            rows.append([r.datum, r.beteckning, r.antal, r.ersattning, r.anskaffning, r.vinst])
            any_row = True
    if not any_row:
        await tg_app.bot.send_message(STATE.chat_id, "Inga K4-rader ännu (stängda LONG spot-affärer).", reply_markup=reply_kb()); return
    buf = io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="k4_export.csv", caption="K4 – Bilaga D (övriga tillgångar)")

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if not st.pos: continue
        tf0 = STATE.tfs[0]
        try: kl = await get_klines(s, tf0, limit=1); px = kl[-1][-1]
        except: continue
        net = _exit_all(s, px, st); closed.append(f"{s}:{net:+.4f}")
    await tg_app.bot.send_message(STATE.chat_id, "Panic: "+(" | ".join(closed) if closed else "Inga positioner."), reply_markup=reply_kb())

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        st = STATE.per_sym[s]; st.realized_pnl_net = 0.0; st.trades_log.clear(); st.k4_rows.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL & K4 återställda.", reply_markup=reply_kb())

async def cmd_save_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, save_ai(), reply_markup=reply_kb())

async def cmd_load_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, load_ai(), reply_markup=reply_kb())

# handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("start_mock", cmd_start_mock))
tg_app.add_handler(CommandHandler("start_live", cmd_start_live))
tg_app.add_handler(CommandHandler("ai_on", cmd_ai_on))
tg_app.add_handler(CommandHandler("ai_off", cmd_ai_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("export_k4", cmd_export_k4))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
tg_app.add_handler(CommandHandler("load_ai", cmd_load_ai))

# -------- FASTAPI / WEBHOOK --------
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
    return f"MP Bot v42 OK | engine={STATE.running} | mode={'LIVE' if STATE.mode_live else 'MOCK'} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {"ok": True, "engine_on": STATE.running, "mode_live": STATE.mode_live,
            "tfs": STATE.tfs, "symbols": STATE.symbols, "pnl_total": round(total, 6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
