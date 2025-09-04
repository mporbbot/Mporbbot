# main_v41_noai.py
# ------------------------------------------------------------
# MP Bot – v41 (NO-AI)
# - Reply-keyboard only (inga inline-knappar)
# - Mock-handel med avgifter per sida
# - Grid/DCA + Trailing Stop
# - Long & short
# - CSV-export, symbol-urval, timeframe-lista
# - FastAPI webhook (Render)
# ------------------------------------------------------------

import os
import io
import csv
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone

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
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))
MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))

# Grid/DCA standard
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.5"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))

# Trailing stop
TP_TRIGGER_PCT = float(os.getenv("TP_TRIGGER_PCT", "0.4"))      # aktivera trailing efter denna vinst %
TRAILING_GAP_PCT = float(os.getenv("TRAILING_GAP_PCT", "0.25")) # hur långt efter priset vi följer

# Risk
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "2.0"))

# KuCoin REST
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}

# -----------------------------
# STATE
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
    safety_count: int = 0
    high_water: float = 0.0   # för trailing stop

    def qty_total(self) -> float:
        return sum(l.qty for l in self.legs)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)
    next_step_pct: float = GRID_STEP_MIN_PCT

@dataclass
class EngineState:
    engine_on: bool = False
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

# -----------------------------
# UI (reply keyboard)
# -----------------------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/symbols"), KeyboardButton("/export_csv")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -----------------------------
# DATA
# -----------------------------
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

# -----------------------------
# TRADING HELPERS
# -----------------------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * FEE_PER_SIDE

def _enter_leg(sym: str, side: str, price: float, usd_size: float, st: SymState) -> TradeLeg:
    qty = usd_size / price if price > 0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, safety_count=0, high_water=price)
        st.next_step_pct = STATE.grid_cfg["step_min"]
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        total_qty = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total_qty or 1.0)
    return leg

def _exit_all(sym: str, price: float, st: SymState) -> float:
    if not st.pos: return 0.0
    gross = 0.0; fee_in = 0.0; fee_out = 0.0
    for leg in st.pos.legs:
        usd_in = leg.qty * leg.price
        fee_in += _fee(usd_in)
        if st.pos.side == "LONG":
            gross += leg.qty * (price - leg.price)
        else:
            gross += leg.qty * (leg.price - price)
    usd_out = st.pos.qty_total() * price
    fee_out = _fee(usd_out)
    net = gross - fee_in - fee_out
    st.realized_pnl_net += net
    st.trades_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "side": st.pos.side,
        "avg_price": st.pos.avg_price,
        "exit_price": price,
        "net": round(net, 6),
        "safety_legs": st.pos.safety_count
    })
    st.pos = None
    st.next_step_pct = STATE.grid_cfg["step_min"]
    return net

# -----------------------------
# ENGINE
# -----------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    tf0 = STATE.tfs[0] if STATE.tfs else "1m"
                    try:
                        kl = await get_klines(sym, tf0, limit=3)
                    except:
                        continue
                    last = kl[-1]
                    price = last[-1]

                    # ENTRY-exempel: enkel EMA-cross eller bias (kan byggas ut)
                    if not st.pos:
                        if price > kl[-1][1]:  # dummy: pris över öppning = long
                            _enter_leg(sym, "LONG", price, STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"🟢 ENTRY LONG {sym} @ {price:.4f}")
                        elif price < kl[-1][1] and STATE.risk_cfg["allow_shorts"]:
                            _enter_leg(sym, "SHORT", price, STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"🔻 ENTRY SHORT {sym} @ {price:.4f}")
                        continue

                    # hantera öppen position
                    avg = st.pos.avg_price
                    move_pct = (price-avg)/avg*100.0 if avg else 0.0

                    # trailing high-water
                    if st.pos.side == "LONG":
                        st.pos.high_water = max(st.pos.high_water, price)
                        if move_pct >= TP_TRIGGER_PCT:
                            trail_stop = st.pos.high_water * (1 - TRAILING_GAP_PCT/100.0)
                            if price <= trail_stop:
                                net = _exit_all(sym, price, st)
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL EXIT {sym} net {net:+.4f} USDT")
                    else:
                        st.pos.high_water = min(st.pos.high_water, price) if st.pos.high_water else price
                        if move_pct <= -TP_TRIGGER_PCT:
                            trail_stop = st.pos.high_water * (1 + TRAILING_GAP_PCT/100.0)
                            if price >= trail_stop:
                                net = _exit_all(sym, price, st)
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL EXIT {sym} net {net:+.4f} USDT")

                    # DCA
                    step = st.next_step_pct
                    if st.pos.side == "LONG" and move_pct <= -step and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        usd = STATE.position_size * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
                        _enter_leg(sym, "LONG", price, usd, st)
                        st.next_step_pct *= STATE.grid_cfg["step_mult"]
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🧩 DCA LONG {sym} @ {price:.4f}")
                    if st.pos.side == "SHORT" and move_pct >= step and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        usd = STATE.position_size * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
                        _enter_leg(sym, "SHORT", price, usd, st)
                        st.next_step_pct *= STATE.grid_cfg["step_mult"]
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🧩 DCA SHORT {sym} @ {price:.4f}")

                    # DD-stop
                    if abs(move_pct) >= STATE.risk_cfg["dd"]:
                        net = _exit_all(sym, price, st)
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"⛔ STOP {sym} net {net:+.4f} USDT")

            await asyncio.sleep(2)
        except Exception as e:
            await asyncio.sleep(5)

# -----------------------------
# TELEGRAM COMMANDS
# -----------------------------
tg_app = Application.builder().token(BOT_TOKEN).build()

def status_text() -> str:
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f}")
    return "\n".join([
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Tfs: {', '.join(STATE.tfs)}",
        f"PnL total: {total:+.4f} USDT",
        "Open: " + (", ".join(pos_lines) if pos_lines else "inga")
    ])

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v41-noAI ✅", reply_markup=reply_kb())
    await tg_app.bot.send_message(STATE.chat_id, status_text())

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", lambda u,c: tg_app.bot.send_message(u.effective_chat.id, status_text())))

# -----------------------------
# FASTAPI WEBHOOK
# -----------------------------
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
    return f"MP Bot v41-noAI OK | engine_on={STATE.engine_on} | pnl_total={total:+.4f}"

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
