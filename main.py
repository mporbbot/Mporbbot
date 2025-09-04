# main_v42.py
# ------------------------------------------------------------
# MP Bot – v42 (utan AI, större vinster)
# - Reply-keyboard (kommandoknappar)
# - Mock-handel med avgifter per sida
# - Grid/DCA + TP + TRAILING STOP (låter vinsterna löpa)
# - Long & short
# - CSV-export, symbol-urval, timeframe-lista
# - FastAPI webhook (Render)
# ------------------------------------------------------------

import os
import io
import csv
import math
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
MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))                            # max antal öppna symboler

# Grid/DCA standard
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))        # % mellan DCA-ben
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.5"))               # nästa dca-steg *= mult
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))               # storlek multipliceras per ben
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.25"))                    # target vinst [%] från snittpris
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.05"))            # +% per DCA-ben

# Risk (nöd-stopp om pris går X% emot snittpris)
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "2.0"))                     # lite lägre än v41 (stopp snabbare)

# Entry-trösklar (utan AI) – på EMA-diff i procent
ENTRY_LONG_THR = float(os.getenv("ENTRY_LONG_THR", "0.35"))              # > så köper long
ENTRY_SHORT_THR = float(os.getenv("ENTRY_SHORT_THR", "0.35"))            # > så säljer short (negativ diff)

# Trailing stop (låter vinsterna löpa)
TRAIL_ARM_PCT = float(os.getenv("TRAIL_ARM_PCT", "0.20"))                # aktivera trailing när flytt från avg >= %
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.60"))                        # avstånd från high/low i %
TRAIL_BE_RR = float(os.getenv("TRAIL_BE_RR", "0.30"))                    # sätt BE (break-even) när move i % >=

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
    # trailing
    tr_armed: bool = False
    tr_anchor: float = 0.0        # högsta high (long) / lägsta low (short) efter arming
    tr_stop: float = 0.0          # rörlig stop
    be_set: bool = False          # break-even låst?

    def qty_total(self) -> float:
        return sum(l.qty for l in self.legs)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)   # för CSV
    next_step_pct: float = GRID_STEP_MIN_PCT

@dataclass
class EngineState:
    engine_on: bool = False
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
    per_sym: Dict[str, SymState] = field(default_factory=dict)
    position_size: float = POSITION_SIZE_USDT
    fee_side: float = FEE_PER_SIDE
    entry_thr_long: float = ENTRY_LONG_THR
    entry_thr_short: float = ENTRY_SHORT_THR
    # grid/risk
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
    # trailing
    trail_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        arm=TRAIL_ARM_PCT,
        pct=TRAIL_PCT,
        be_rr=TRAIL_BE_RR
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
        [KeyboardButton("/trail"), KeyboardButton("/entry")],
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
        data = list(reversed(r.json()["data"]))  # nyast först -> vänd
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

def features_from_candles(candles: List[Tuple[int,float,float,float,float]]) -> Dict[str, float]:
    closes = [c[-1] for c in candles]
    if len(closes) < 30:
        return {"ema_fast": closes[-1], "ema_slow": closes[-1]}
    ema_fast = ema(closes, 9)[-1]
    ema_slow = ema(closes, 21)[-1]
    return {"ema_fast": ema_fast, "ema_slow": ema_slow}

# -----------------------------
# TRADING HELPERS (mock)
# -----------------------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * FEE_PER_SIDE

def _enter_leg(sym: str, side: str, price: float, usd_size: float, st: SymState) -> TradeLeg:
    qty = usd_size / price if price > 0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0)
        st.next_step_pct = STATE.grid_cfg["step_min"]
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        total_qty = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total_qty or 1.0)

    # Ny TP (används som “take profit om ingen trailing”)
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
    if side == "LONG":
        st.pos.target_price = st.pos.avg_price * (1.0 + tp/100.0)
    else:
        st.pos.target_price = st.pos.avg_price * (1.0 - tp/100.0)

    # Nollställ trailing signaler vid ny leg
    st.pos.tr_armed = False
    st.pos.tr_anchor = price
    st.pos.tr_stop = 0.0
    st.pos.be_set = False
    return leg

def _exit_all(sym: str, price: float, st: SymState) -> float:
    """Stänger hela positionen och returnerar NET PnL (avgifter dragna)."""
    if not st.pos: return 0.0
    gross = 0.0
    fee_in = 0.0
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
# ENTRY (utan AI) & TRAILING
# -----------------------------
def entry_signal(feats: Dict[str, float]) -> float:
    """>0 = longbias, <0 = shortbias. Bygger på EMA-diff i procent."""
    ema_fast = feats["ema_fast"]; ema_slow = feats["ema_slow"] or 1.0
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow * 100.0
    return ema_diff_pct

def maybe_arm_trailing(st: SymState, price: float):
    """Aktivera trailing när move från snitt >= arm%."""
    p = st.pos
    if not p or p.tr_armed:
        return
    move_pct = (price - p.avg_price) / p.avg_price * 100.0 if p.side == "LONG" else (p.avg_price - price) / p.avg_price * 100.0
    if move_pct >= STATE.trail_cfg["arm"]:
        p.tr_armed = True
        p.tr_anchor = price
        # direkt sätt en första stop nära BE om be_rr uppnådd
        if not p.be_set and move_pct >= STATE.trail_cfg["be_rr"]:
            p.be_set = True
            if p.side == "LONG":
                p.tr_stop = max(p.avg_price, price * (1.0 - STATE.trail_cfg["pct"]/100.0))
            else:
                p.tr_stop = min(p.avg_price, price * (1.0 + STATE.trail_cfg["pct"]/100.0))

def update_trailing(st: SymState, last_candle: Tuple[int,float,float,float,float]) -> Optional[float]:
    """
    Uppdaterar trailingstop och returnerar ev. exitpris om stop träffas.
    last_candle: (ts, open, high, low, close)
    """
    p = st.pos
    if not p or not p.tr_armed:
        return None
    _, o, h, l, c = last_candle
    pct = STATE.trail_cfg["pct"] / 100.0

    if p.side == "LONG":
        # höj anchor till högsta high
        p.tr_anchor = max(p.tr_anchor, h)
        # följ med upp: stop = anchor * (1 - pct)
        new_stop = p.tr_anchor * (1.0 - pct)
        # aldrig sänka stop
        p.tr_stop = max(p.tr_stop, new_stop) if p.tr_stop else new_stop
        # break-even-lås (en gång) om ännu ej satt
        if not p.be_set:
            move_pct = (p.tr_anchor - p.avg_price) / p.avg_price * 100.0
            if move_pct >= STATE.trail_cfg["be_rr"]:
                p.be_set = True
                p.tr_stop = max(p.tr_stop, p.avg_price)
        # träff?
        if l <= p.tr_stop:
            return p.tr_stop  # exit på stop

    else:  # SHORT
        p.tr_anchor = min(p.tr_anchor, l)
        new_stop = p.tr_anchor * (1.0 + pct)
        p.tr_stop = min(p.tr_stop, new_stop) if p.tr_stop else new_stop
        if not p.be_set:
            move_pct = (p.avg_price - p.tr_anchor) / p.avg_price * 100.0
            if move_pct >= STATE.trail_cfg["be_rr"]:
                p.be_set = True
                p.tr_stop = min(p.tr_stop, p.avg_price)
        if h >= p.tr_stop:
            return p.tr_stop

    return None

# -----------------------------
# ENGINE LOOP
# -----------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                # öppna nya om plats finns
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.risk_cfg["max_pos"]:
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos:
                            continue
                        # använd snabbaste TF för entry
                        tf0 = STATE.tfs[0] if STATE.tfs else "1m"
                        try:
                            kl = await get_klines(sym, tf0, limit=60)
                        except Exception:
                            continue
                        feats = features_from_candles(kl)
                        bias = entry_signal(feats)  # ema-diff %
                        last_price = kl[-1][-1]
                        # LONG?
                        if bias > STATE.entry_thr_long:
                            st = STATE.per_sym[sym]
                            _enter_leg(sym, "LONG", last_price, STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {last_price:.4f} | bias {bias:.2f}% | "
                                    f"TP {st.pos.target_price:.4f} | DD {STATE.risk_cfg['dd']:.2f}%"
                                )
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                                break
                        # SHORT?
                        elif bias < -STATE.entry_thr_short and STATE.risk_cfg["allow_shorts"]:
                            st = STATE.per_sym[sym]
                            _enter_leg(sym, "SHORT", last_price, STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔻 ENTRY {sym} SHORT @ {last_price:.4f} | bias {bias:.2f}% | "
                                    f"TP {st.pos.target_price:.4f} | DD {STATE.risk_cfg['dd']:.2f}%"
                                )
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                                break

                # hantera öppna positioner (TP/DCA/Stop/Trailing)
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos:
                        continue
                    tf0 = STATE.tfs[0] if STATE.tfs else "1m"
                    try:
                        kl = await get_klines(sym, tf0, limit=3)
                    except Exception:
                        continue
                    last = kl[-1]
                    price = last[-1]
                    avg = st.pos.avg_price
                    move_pct = (price-avg)/avg*100.0 if st.pos.side=="LONG" else (avg-price)/avg*100.0

                    # Arm trailing när move passerar arm-%, och uppdatera därefter
                    maybe_arm_trailing(st, price)
                    stop_hit = update_trailing(st, last)
                    if stop_hit is not None:
                        net = _exit_all(sym, stop_hit, st)
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🟤 EXIT-TRAIL {sym} @ {stop_hit:.4f} | Net: {net:+.4f} USDT"
                            )
                        continue

                    # Klassisk TP (om trailing inte triggar)
                    if (st.pos.side == "LONG" and price >= st.pos.target_price) or \
                       (st.pos.side == "SHORT" and price <= st.pos.target_price):
                        net = _exit_all(sym, price, st)
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🟤 EXIT-TP {sym} @ {price:.4f} | Net: {net:+.4f} USDT"
                            )
                        continue

                    # DCA om emot oss med 'next_step_pct'
                    need_dca = False
                    if st.pos.side == "LONG" and (price-avg)/avg*100.0 <= -st.next_step_pct:
                        need_dca = True
                    if st.pos.side == "SHORT" and (avg-price)/avg*100.0 <= -st.next_step_pct:
                        need_dca = True
                    if need_dca and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        usd = STATE.position_size * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
                        _enter_leg(sym, st.pos.side, price, usd, st)
                        st.next_step_pct = max(STATE.grid_cfg["step_min"],
                                               st.next_step_pct * STATE.grid_cfg["step_mult"])
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} | leg {st.pos.safety_count}"
                            )

                    # DD-stop (nöd)
                    if move_pct >= STATE.risk_cfg["dd"]:
                        net = _exit_all(sym, price, st)
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"⛔ STOP (DD) {sym} @ {price:.4f} | Net: {net:+.4f} USDT"
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
            tinfo = f"TRAIL {'ON' if st.pos.tr_armed else 'OFF'}"
            if st.pos.tr_armed and st.pos.tr_stop:
                tinfo += f" stop {st.pos.tr_stop:.4f}"
            pos_lines.append(
                f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} → TP {st.pos.target_price:.4f} | "
                f"legs {st.pos.safety_count} | {tinfo}"
            )
    grid = STATE.grid_cfg; risk = STATE.risk_cfg; tr = STATE.trail_cfg
    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} USDT | Fee/side: {STATE.fee_side:.4%}",
        (f"Entry thr: long>{STATE.entry_thr_long:.2f}% short>{STATE.entry_thr_short:.2f}% (EMA-diff)"),
        (f"Grid: max_safety={grid['max_safety']} step_mult={grid['step_mult']} "
         f"step_min%={grid['step_min']} size_mult={grid['size_mult']} tp%={grid['tp']} (+{grid['tp_bonus']}%/leg)"),
        f"Risk: dd={risk['dd']}% | max_pos={risk['max_pos']} | shorts={'ON' if risk['allow_shorts'] else 'OFF'}",
        f"Trail: arm={tr['arm']}% | trail={tr['pct']}% | BE@{tr['be_rr']}%",
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

async def send_status(chat_id: int):
    await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

# --- Commands ---
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v42 – redo ✅", reply_markup=reply_kb())
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
    msg = update.message.text.strip()
    parts = msg.split(" ", 1)
    if len(parts) == 2:
        tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.tfs)}",
                                          reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, "Använd: /timeframe 1m,3m,5m,15m", reply_markup=reply_kb())

async def cmd_grid(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
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
         f"  tp%={g['tp']} (+{g['tp_bonus']}%/leg)\n\n"
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
         "Ex: /risk set dd 1.8   | /risk set max_pos 8   | /risk set allow_shorts on"),
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
         "\nLägg till/ta bort:\n  /symbols add SOL-USDT\n  /symbols remove LINK-USDT"),
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

async def cmd_trail(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
    toks = text.split()
    if len(toks) == 5 and toks[0] == "/trail" and toks[1] == "set":
        # /trail set arm 0.2 pct 0.6   eller  /trail set pct 0.7 be 0.3
        try:
            i = 2
            while i < len(toks)-1:
                k, v = toks[i], float(toks[i+1])
                if k in ("arm", "pct", "be"):
                    key = "be_rr" if k == "be" else k
                    STATE.trail_cfg[key] = v
                i += 2
            await tg_app.bot.send_message(STATE.chat_id, "Trail uppdaterad.", reply_markup=reply_kb())
            return
        except:
            pass
    tr = STATE.trail_cfg
    await tg_app.bot.send_message(STATE.chat_id,
        (f"Trail:\n  arm={tr['arm']}%  pct={tr['pct']}%  BE@{tr['be_rr']}%\n"
         "Ex: /trail set arm 0.2 pct 0.6\nEx: /trail set pct 0.7 be 0.3"),
        reply_markup=reply_kb())

async def cmd_entry(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
    toks = text.split()
    # /entry 0.35    eller    /entry long 0.4 short 0.5
    try:
        if len(toks) == 2:
            v = float(toks[1])
            STATE.entry_thr_long = v
            STATE.entry_thr_short = v
            await tg_app.bot.send_message(STATE.chat_id, f"Entry thr satt till {v:.2f}% (long & short).",
                                          reply_markup=reply_kb())
            return
        if len(toks) == 5:
            # long X short Y
            if toks[1] == "long" and toks[3] == "short":
                STATE.entry_thr_long = float(toks[2])
                STATE.entry_thr_short = float(toks[4])
                await tg_app.bot.send_message(STATE.chat_id,
                    f"Entry thr: long>{STATE.entry_thr_long:.2f}% short>{STATE.entry_thr_short:.2f}%",
                    reply_markup=reply_kb())
                return
    except:
        pass
    await tg_app.bot.send_message(STATE.chat_id,
        "Använd: /entry 0.35  eller  /entry long 0.35 short 0.45", reply_markup=reply_kb())

# registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("trail", cmd_trail))
tg_app.add_handler(CommandHandler("entry", cmd_entry))

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
    return f"MP Bot v42 OK | engine_on={STATE.engine_on} | pnl_total={total:+.4f}"

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
