# main_v42a.py
# ------------------------------------------------------------
# MP Bot – v42a (no-AI, fler entries)
# - Reply-keyboard (inga inline-knappar)
# - Mer aggressiv entry: ANY-TF-läge + konfliktfilter
# - Grid/DCA + TP + Trailing stop (arm/step/BE)
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
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m,15m").replace(" ", "")).split(",")

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))       # per entry
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))                 # 0.1% mock
MAX_OPEN_POS = int(os.getenv("MAX_POS", "8"))

# Grid/DCA standard
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))        # % mellan DCA-ben
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.5"))               # nästa steg * mult
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))               # nästa kvant. * mult
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.25"))                    # target vinst [%] från genomsnittspris
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.05"))            # +% per DCA-ben

# Risk
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "2.0"))                     # hårt DD-stopp (sänkt)
ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "on").lower() in ("1","true","on","yes")

# Trailing
TRAIL_ARM_PCT = float(os.getenv("TRAIL_ARM_PCT", "0.20"))                # % i vinst innan trail "armas"
TRAIL_STEP_PCT = float(os.getenv("TRAIL_STEP_PCT", "0.60"))              # hur tajt följer den efter arm
BE_ARM_PCT = float(os.getenv("BE_ARM_PCT", "0.30"))                      # flytta SL till breakeven vid denna vinst

# Entry (nya)
ENTRY_LONG_THR = float(os.getenv("ENTRY_LONG_THR", "0.35"))              # krävd EMA-diff för long
ENTRY_SHORT_THR = float(os.getenv("ENTRY_SHORT_THR", "0.35"))            # krävd EMA-diff för short
ENTRY_MODE = os.getenv("ENTRY_MODE", "any_tf")                           # any_tf | first_tf
ENTRY_CONTRA_MARGIN = float(os.getenv("ENTRY_CONTRA_MARGIN", "0.10"))    # motsats-EMA diff som blockerar

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
    # trailing runtime
    tr_armed: bool = False
    tr_stop: Optional[float] = None  # prisnivå
    be_done: bool = False            # breakeven flyttad?

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
        allow_shorts=ALLOW_SHORTS
    ))
    trail_cfg: Dict[str, float] = field(default_factory=lambda: dict(
        arm=TRAIL_ARM_PCT,
        step=TRAIL_STEP_PCT,
        be=BE_ARM_PCT
    ))
    entry_cfg: Dict[str, object] = field(default_factory=lambda: dict(
        long_thr=ENTRY_LONG_THR,
        short_thr=ENTRY_SHORT_THR,
        mode=ENTRY_MODE,                # "any_tf" | "first_tf"
        margin=ENTRY_CONTRA_MARGIN
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
        [KeyboardButton("/trailing"), KeyboardButton("/entry")],
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

def features(candles: List[Tuple[int,float,float,float,float]]) -> Dict[str, float]:
    closes = [c[-1] for c in candles]
    if len(closes) < 22:
        return {"ema_fast": closes[-1], "ema_slow": closes[-1], "atrp": 0.2}
    ema_fast = ema(closes, 9)[-1]
    ema_slow = ema(closes, 21)[-1]
    # ATR% approx
    rng = [(c[2]-c[3]) / c[-1] * 100.0 if c[-1] else 0.0 for c in candles[-20:]]
    atrp = sum(rng)/len(rng) if rng else 0.2
    return {"ema_fast": ema_fast, "ema_slow": ema_slow, "atrp": max(0.02, min(3.0, atrp))}

def ema_diff_pct(f: Dict[str,float]) -> float:
    return (f["ema_fast"] - f["ema_slow"]) / (f["ema_slow"] or 1.0) * 100.0

# -----------------------------
# TRADING HELPERS (mock)
# -----------------------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

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

    # sätt initial TP (kan ändå trailas upp)
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
    if side == "LONG":
        st.pos.target_price = st.pos.avg_price * (1.0 + tp/100.0)
    else:
        st.pos.target_price = st.pos.avg_price * (1.0 - tp/100.0)

    return leg

def _exit_all(sym: str, price: float, st: SymState) -> float:
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
# ENTRY LOGIC (v42a)
# -----------------------------
def decide_entry(symbol: str, feats_per_tf: Dict[str, Dict[str,float]]) -> Optional[str]:
    """
    Returnerar 'LONG'/'SHORT'/None.
    any_tf: Om någon TF passerar sina trösklar triggar vi, så länge ingen TF är starkt emot (margin).
    first_tf: Som v41 (bara första TF), men med samma konfliktfilter (andra TF får stoppa om de är starkt emot).
    """
    tfs = STATE.tfs
    cfg = STATE.entry_cfg
    long_thr = float(cfg["long_thr"])
    short_thr = float(cfg["short_thr"])
    margin = float(cfg["margin"])
    mode = str(cfg["mode"])

    diffs = {tf: ema_diff_pct(feats_per_tf[tf]) for tf in tfs if tf in feats_per_tf}

    def ok_long():
        trigger = diffs.get(tfs[0], -999) >= long_thr if mode == "first_tf" else any(d >= long_thr for d in diffs.values())
        if not trigger:
            return False
        # blockera om någon TF är tydligt bearish
        return not any(d <= -margin for d in diffs.values())

    def ok_short():
        trigger = diffs.get(tfs[0], 999) <= -short_thr if mode == "first_tf" else any(d <= -short_thr for d in diffs.values())
        if not trigger:
            return False
        # blockera om någon TF är tydligt bullish
        return not any(d >= margin for d in diffs.values())

    if ok_long():
        return "LONG"
    if STATE.risk_cfg["allow_shorts"] and ok_short():
        return "SHORT"
    return None

# -----------------------------
# ENGINE LOOP
# -----------------------------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                # Öppna nya?
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.risk_cfg["max_pos"]:
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos:
                            continue
                        feats_per_tf = {}
                        skip = False
                        for tf in STATE.tfs:
                            try:
                                kl = await get_klines(sym, tf, limit=60)
                            except Exception:
                                skip = True; break
                            feats_per_tf[tf] = features(kl)
                        if skip or not feats_per_tf:
                            continue

                        side = decide_entry(sym, feats_per_tf)
                        if side:
                            # pris från snabbaste TF
                            tf0 = STATE.tfs[0]
                            try:
                                kl0 = await get_klines(sym, tf0, limit=1)
                                price = kl0[-1][-1]
                            except:
                                continue
                            st = STATE.per_sym[sym]
                            _enter_leg(sym, side, price, STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"{'🟢' if side=='LONG' else '🔻'} ENTRY {sym} {side} @ {price:.4f} "
                                    f"| TP {st.pos.target_price:.4f} | ddSL~{STATE.risk_cfg['dd']:.2f}%"
                                )
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                                break

                # Hantera öppna (TP, DCA, trail, DD)
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos:
                        continue
                    tf0 = STATE.tfs[0] if STATE.tfs else "1m"
                    try:
                        kl = await get_klines(sym, tf0, limit=3)
                    except:
                        continue
                    last = kl[-1]
                    price = last[-1]
                    avg = st.pos.avg_price
                    move_pct = (price-avg)/avg*100.0

                    # 1) Trailing (arm/BE/step)
                    tr = STATE.trail_cfg
                    if st.pos.side == "LONG":
                        if not st.pos.be_done and move_pct >= tr["be"]:
                            st.pos.be_done = True
                            # flytta implicit SL till avg (hanteras i DD/exit-logiken nedan)
                        if not st.pos.tr_armed and move_pct >= tr["arm"]:
                            st.pos.tr_armed = True
                            st.pos.tr_stop = price * (1.0 - tr["step"]/100.0)
                        if st.pos.tr_armed:
                            # höj tr_stop om priset går upp
                            new_stop = price * (1.0 - tr["step"]/100.0)
                            if st.pos.tr_stop is None or new_stop > st.pos.tr_stop:
                                st.pos.tr_stop = new_stop
                            # om trail slås
                            if st.pos.tr_stop and price <= st.pos.tr_stop:
                                net = _exit_all(sym, st.pos.tr_stop, st)
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL EXIT {sym} @ {st.pos.tr_stop:.4f} | Net: {net:+.4f} USDT")
                                continue
                    else:  # SHORT
                        if not st.pos.be_done and -move_pct >= tr["be"]:
                            st.pos.be_done = True
                        if not st.pos.tr_armed and -move_pct >= tr["arm"]:
                            st.pos.tr_armed = True
                            st.pos.tr_stop = price * (1.0 + tr["step"]/100.0)
                        if st.pos.tr_armed:
                            new_stop = price * (1.0 + tr["step"]/100.0)
                            if st.pos.tr_stop is None or new_stop < st.pos.tr_stop:
                                st.pos.tr_stop = new_stop
                            if st.pos.tr_stop and price >= st.pos.tr_stop:
                                net = _exit_all(sym, st.pos.tr_stop, st)
                                if STATE.chat_id:
                                    await app.bot.send_message(STATE.chat_id, f"🏁 TRAIL EXIT {sym} @ {st.pos.tr_stop:.4f} | Net: {net:+.4f} USDT")
                                continue

                    # 2) TP (statisk mål från grid)
                    if (st.pos.side == "LONG" and price >= st.pos.target_price) or \
                       (st.pos.side == "SHORT" and price <= st.pos.target_price):
                        net = _exit_all(sym, price, st)
                        if STATE.chat_id:
                            await app.bot.send_message(STATE.chat_id, f"🟤 EXIT {sym} @ {price:.4f} | Net: {net:+.4f} USDT")
                        continue

                    # 3) DCA
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
                                f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} | leg {st.pos.safety_count} | qty {st.pos.legs[-1].qty:.6f}"
                            )

                    # 4) DD/BE stopp
                    dd = STATE.risk_cfg["dd"]
                    if st.pos.side == "LONG":
                        # BE-skydd
                        if st.pos.be_done and price <= st.pos.avg_price:
                            net = _exit_all(sym, st.pos.avg_price, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"🧯 BE EXIT {sym} @ {st.pos.avg_price:.4f} | Net: {net:+.4f}")
                            continue
                        # DD-stopp
                        if move_pct <= -dd:
                            net = _exit_all(sym, price, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"⛔ STOP {sym} @ {price:.4f} | Net: {net:+.4f} USDT (DD)")
                            continue
                    else:
                        if st.pos.be_done and price >= st.pos.avg_price:
                            net = _exit_all(sym, st.pos.avg_price, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"🧯 BE EXIT {sym} @ {st.pos.avg_price:.4f} | Net: {net:+.4f}")
                            continue
                        if -move_pct <= -dd:  # dvs price - avg >= dd%
                            net = _exit_all(sym, price, st)
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id, f"⛔ STOP {sym} @ {price:.4f} | Net: {net:+.4f} USDT (DD)")
                            continue

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
    g = STATE.grid_cfg; r = STATE.risk_cfg; tr = STATE.trail_cfg; ec = STATE.entry_cfg
    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} USDT | Fee/side: {STATE.fee_side:.4%}",
        (f"Grid: max_safety={g['max_safety']} step_mult={g['step_mult']} "
         f"step_min%={g['step_min']} size_mult={g['size_mult']} "
         f"tp%={g['tp']} (+{g['tp_bonus']}%/leg)"),
        f"Risk: dd={r['dd']}% | max_pos={r['max_pos']} | shorts={'ON' if r['allow_shorts'] else 'OFF'}",
        f"Trail: arm={tr['arm']}% step={tr['step']}% BE={tr['be']}%",
        f"Entry: mode={ec['mode']} long_thr={ec['long_thr']}% short_thr={ec['short_thr']}% margin={ec['margin']}%",
        f"PnL total (NET): {total:+.4f} USDT",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

async def send_status(chat_id: int):
    await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v42a – redo ✅", reply_markup=reply_kb())
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

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    msg = update.message.text.strip()
    parts = msg.split(" ", 1)
    if len(parts) == 2:
        tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
        if tfs:
            STATE.tfs = tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes satta: {', '.join(STATE.tfs)}",
                                          reply_markup=reply_kb()); return
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
            await tg_app.bot.send_message(STATE.chat_id, f"Okänd grid-nyckel: {key}", reply_markup=reply_kb()); return
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
                    STATE.risk_cfg[key] = (val.lower() in ("1","true","on","yes"))
                else:
                    STATE.risk_cfg[key] = float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb())
            except:
                await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
        else:
            await tg_app.bot.send_message(STATE.chat_id, f"Okänd risk-nyckel: {key}", reply_markup=reply_kb()); return
    r = STATE.risk_cfg
    await tg_app.bot.send_message(STATE.chat_id,
        (f"Risk:\n  dd={r['dd']}%\n  max_pos={r['max_pos']}\n  shorts={'ON' if r['allow_shorts'] else 'OFF'}\n"
         "Ex: /risk set dd 3.0 | /risk set max_pos 10 | /risk set allow_shorts on"),
        reply_markup=reply_kb())

async def cmd_trailing(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = update.message.text.strip()
    toks = txt.split()
    if len(toks) == 5 and toks[0] == "/trailing" and toks[1] == "set":
        try:
            STATE.trail_cfg["arm"] = float(toks[2])
            STATE.trail_cfg["step"] = float(toks[3])
            STATE.trail_cfg["be"] = float(toks[4])
            await tg_app.bot.send_message(STATE.chat_id, "Trailing uppdaterad.", reply_markup=reply_kb())
            return
        except:
            await tg_app.bot.send_message(STATE.chat_id, "Fel format.", reply_markup=reply_kb()); return
    tr = STATE.trail_cfg
    await tg_app.bot.send_message(STATE.chat_id,
        (f"Trail:\n  arm={tr['arm']}%  step={tr['step']}%  BE={tr['be']}%\n"
         "Ex: /trailing set 0.25 0.50 0.30"),
        reply_markup=reply_kb())

async def cmd_entry(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = update.message.text.strip()
    toks = txt.split()
    ec = STATE.entry_cfg
    # /entry           -> visa
    # /entry 0.15      -> båda thr
    # /entry long 0.12 short 0.18
    # /entry mode any_tf | first_tf
    # /entry margin 0.10
    if len(toks) == 1:
        await tg_app.bot.send_message(STATE.chat_id,
            (f"Entry:\n  mode={ec['mode']}  long_thr={ec['long_thr']}%  short_thr={ec['short_thr']}%  margin={ec['margin']}%\n"
             "Ex:\n  /entry 0.15\n  /entry long 0.12 short 0.18\n  /entry mode any_tf\n  /entry margin 0.10"),
            reply_markup=reply_kb())
        return
    if len(toks) == 2:
        try:
            v = float(toks[1])
            ec["long_thr"] = v; ec["short_thr"] = v
            await tg_app.bot.send_message(STATE.chat_id, "Entry-trösklar uppdaterade (båda).", reply_markup=reply_kb())
            return
        except:
            pass
    # fler format
    try:
        if "mode" in toks:
            i = toks.index("mode")
            ec["mode"] = toks[i+1]
        if "margin" in toks:
            i = toks.index("margin")
            ec["margin"] = float(toks[i+1])
        if "long" in toks:
            i = toks.index("long")
            ec["long_thr"] = float(toks[i+1])
        if "short" in toks:
            i = toks.index("short")
            ec["short_thr"] = float(toks[i+1])
        await tg_app.bot.send_message(STATE.chat_id, "Entry-inställningar uppdaterade.", reply_markup=reply_kb())
    except Exception:
        await tg_app.bot.send_message(STATE.chat_id, "Fel format på /entry.", reply_markup=reply_kb())

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
    csv.writer(buf).writerows(rows)
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

# registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("trailing", cmd_trailing))
tg_app.add_handler(CommandHandler("entry", cmd_entry))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))

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
    return f"MP Bot v42a OK | engine_on={STATE.engine_on} | pnl_total={total:+.4f}"

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
