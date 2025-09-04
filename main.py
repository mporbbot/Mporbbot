# main_v41_trend.py
# ------------------------------------------------------------
# MP Bot – v41 (NO-AI + TREND/TRAIL)
# - Ingen AI (allt borttaget)
# - Reply-keyboard (inga inline-knappar)
# - Mock-handel med avgifter per sida
# - Grid/DCA + TP
# - Long & (valfritt) short
# - Trailing stop (trigger + trail + break-even)
# - CSV-export, symbol-urval, timeframe-lista
# - K4-export
# - FastAPI webhook (Render). Sätt WEBHOOK_BASE=https://<din-app>.onrender.com
# - Live-läge är förberett (KuCoin), men lämnar order-funktionerna no-op
#   om API-nycklar saknas.
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

# --------------- ENV -----------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

DEFAULT_SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
                   .replace(" ", "")).split(",")
DEFAULT_TFS = (os.getenv("TIMEFRAMES", "1m,3m,5m").replace(" ", "")).split(",")

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))          # 0.1%

MAX_OPEN_POS = int(os.getenv("MAX_POS", "5"))

# Grid/DCA
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.6"))
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))
GRID_TP_PCT = float(os.getenv("GRID_TP_PCT", "0.25"))
TP_SAFETY_BONUS = float(os.getenv("TP_SAFETY_BONUS", "0.05"))

# Risk
DD_STOP_PCT = float(os.getenv("DD_STOP_PCT", "2.0"))               # nöd-stopp mot genomsnittspris

# Trend/Trailing (nya)
TREND_ON = os.getenv("TREND_ON", "true").lower() in ("1", "true", "on", "yes")
TREND_TRIGGER = float(os.getenv("TREND_TRIGGER", "0.9"))           # % vinst innan trailing aktiveras
TREND_TRAIL = float(os.getenv("TREND_TRAIL", "0.25"))              # hur långt efter priset stoppen ligger
TREND_BE = float(os.getenv("TREND_BE", "0.2"))                     # sätt BE efter så här mycket vinst

# KuCoin klines
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# (valfritt) live-API (om du vill koppla senare)
KU_API_KEY = os.getenv("KU_API_KEY", "")
KU_API_SECRET = os.getenv("KU_API_SECRET", "")
KU_API_PASSPHRASE = os.getenv("KU_API_PASSPHRASE", "")

# --------------- STATE -----------------
@dataclass
class TradeLeg:
    side: str        # "LONG" | "SHORT"
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
    trailing_active: bool = False
    trail_stop: Optional[float] = None
    peak: Optional[float] = None   # högsta sedan entry (LONG)
    trough: Optional[float] = None # lägsta sedan entry (SHORT)

    def qty_total(self) -> float:
        return sum(l.qty for l in self.legs)

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_net: float = 0.0
    trades_log: List[Dict] = field(default_factory=list)
    last_signal_ts: Optional[int] = None
    next_step_pct: float = GRID_STEP_MIN_PCT

@dataclass
class EngineState:
    engine_on: bool = False
    mode: str = "MOCK"  # "MOCK" | "LIVE"
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
        allow_shorts=False
    ))
    trend_cfg: Dict[str, float | bool] = field(default_factory=lambda: dict(
        on=TREND_ON, trigger=TREND_TRIGGER, trail=TREND_TRAIL, be=TREND_BE
    ))
    chat_id: Optional[int] = None

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# --------------- UI -----------------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/start_mock"), KeyboardButton("/start_live")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/trend"), KeyboardButton("/symbols")],
        [KeyboardButton("/export_csv"), KeyboardButton("/export_k4")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# --------------- DATA -----------------
async def get_klines(symbol: str, tf: str, limit: int = 60):
    """[(ts_ms, open, high, low, close)] äldst->nyast."""
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
    v = series[0]
    for x in series:
        v = (x - v) * k + v
        out.append(v)
    return out

def features_from_candles(candles):
    closes = [c[-1] for c in candles]
    if len(closes) < 21:
        return {"ema_fast": closes[-1], "ema_slow": closes[-1]}
    return {"ema_fast": ema(closes, 9)[-1], "ema_slow": ema(closes, 21)[-1]}

# --------------- HELPERS ---------------
def _fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_side

def _enter_leg(sym: str, side: str, price: float, usd_size: float, st: SymState) -> TradeLeg:
    qty = usd_size / price if price > 0 else 0.0
    leg = TradeLeg(side=side, price=price, qty=qty, time=datetime.now(timezone.utc))
    if st.pos is None:
        st.pos = Position(side=side, legs=[leg], avg_price=price, target_price=0.0, safety_count=0)
        st.next_step_pct = STATE.grid_cfg["step_min"]
        # init trailing runtime
        st.pos.trailing_active = False
        st.pos.trail_stop = None
        st.pos.peak = price
        st.pos.trough = price
    else:
        st.pos.legs.append(leg)
        st.pos.safety_count += 1
        total_qty = st.pos.qty_total()
        st.pos.avg_price = sum(l.price*l.qty for l in st.pos.legs) / (total_qty or 1.0)

    # sätt initial TP (klassisk grid-TP)
    tp = STATE.grid_cfg["tp"] + st.pos.safety_count * STATE.grid_cfg["tp_bonus"]
    if side == "LONG":
        st.pos.target_price = st.pos.avg_price * (1.0 + tp/100.0)
    else:
        st.pos.target_price = st.pos.avg_price * (1.0 - tp/100.0)
    return leg

def _exit_all(sym: str, price: float, st: SymState, reason: str = "EXIT") -> float:
    if not st.pos:
        return 0.0
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
        "symbol": sym, "side": st.pos.side,
        "avg_price": st.pos.avg_price, "exit_price": price,
        "gross": round(gross, 6), "fee_in": round(fee_in, 6),
        "fee_out": round(fee_out, 6), "net": round(net, 6),
        "safety_legs": st.pos.safety_count, "reason": reason
    })
    st.pos = None
    st.next_step_pct = STATE.grid_cfg["step_min"]
    return net

# --------------- ENGINE -----------------
def _score_from_ema(ema_fast: float, ema_slow: float) -> float:
    if ema_slow == 0:
        return 0.0
    diff = (ema_fast - ema_slow) / ema_slow * 100.0
    # enkel bias
    return max(-1.0, min(1.0, diff / 1.5))

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                # öppna nya positioner om plats finns
                open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
                if len(open_syms) < STATE.risk_cfg["max_pos"]:
                    for sym in STATE.symbols:
                        if STATE.per_sym[sym].pos:
                            continue
                        feats = {}
                        skip = False
                        for tf in STATE.tfs:
                            try:
                                kl = await get_klines(sym, tf, limit=60)
                            except Exception:
                                skip = True
                                break
                            feats[tf] = features_from_candles(kl)
                        if skip or not feats:
                            continue
                        # röstning via EMA-kors
                        score = sum(_score_from_ema(v["ema_fast"], v["ema_slow"]) for v in feats.values())
                        # tröskel
                        if score > 0.7:
                            st = STATE.per_sym[sym]
                            last_px = (await get_klines(sym, STATE.tfs[0], limit=1))[-1][-1]
                            _enter_leg(sym, "LONG", last_px, STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {last_px:.4f} | "
                                    f"TP ~ {st.pos.target_price:.4f} | legs 0"
                                )
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                                break
                        elif score < -0.7 and STATE.risk_cfg["allow_shorts"]:
                            st = STATE.per_sym[sym]
                            last_px = (await get_klines(sym, STATE.tfs[0], limit=1))[-1][-1]
                            _enter_leg(sym, "SHORT", last_px, STATE.position_size, st)
                            if STATE.chat_id:
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔻 ENTRY {sym} SHORT @ {last_px:.4f} | "
                                    f"TP ~ {st.pos.target_price:.4f} | legs 0"
                                )
                            open_syms.append(sym)
                            if len(open_syms) >= STATE.risk_cfg["max_pos"]:
                                break

                # hantera öppna positioner
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    if not st.pos:
                        continue
                    tf0 = STATE.tfs[0] if STATE.tfs else "1m"
                    try:
                        kl = await get_klines(sym, tf0, limit=3)
                    except Exception:
                        continue
                    price = kl[-1][-1]
                    avg = st.pos.avg_price
                    move_pct = (price - avg) / (avg or 1.0) * 100.0

                    # uppdatera peak/trough
                    if st.pos.side == "LONG":
                        st.pos.peak = max(st.pos.peak or price, price)
                    else:
                        st.pos.trough = min(st.pos.trough or price, price)

                    # aktivera break-even och trailing när trigger nås
                    tr = STATE.trend_cfg
                    if tr["on"]:
                        if st.pos.side == "LONG":
                            # BE
                            if move_pct >= float(tr["be"]) and (st.pos.trail_stop is None):
                                st.pos.trail_stop = avg    # BE
                            # aktivera trailing
                            if move_pct >= float(tr["trigger"]):
                                st.pos.trailing_active = True
                            # följa efter
                            if st.pos.trailing_active:
                                follow = (st.pos.peak or price) * (1.0 - float(tr["trail"])/100.0)
                                st.pos.trail_stop = max(st.pos.trail_stop or follow, follow)
                        else:
                            if -move_pct >= float(tr["be"]) and (st.pos.trail_stop is None):
                                st.pos.trail_stop = avg
                            if -move_pct >= float(tr["trigger"]):
                                st.pos.trailing_active = True
                            if st.pos.trailing_active:
                                follow = (st.pos.trough or price) * (1.0 + float(tr["trail"])/100.0)
                                st.pos.trail_stop = min(st.pos.trail_stop or follow, follow)

                    # TP (klassisk)
                    hit_tp = (st.pos.side == "LONG" and price >= st.pos.target_price) or \
                             (st.pos.side == "SHORT" and price <= st.pos.target_price)
                    # Trailing stop
                    hit_trail = False
                    if st.pos.trail_stop is not None:
                        if st.pos.side == "LONG" and price <= st.pos.trail_stop:
                            hit_trail = True
                        if st.pos.side == "SHORT" and price >= st.pos.trail_stop:
                            hit_trail = True
                    # DD nödstopp
                    hit_dd = abs(move_pct) >= STATE.risk_cfg["dd"]

                    if hit_tp or hit_trail or hit_dd:
                        reason = "TP" if hit_tp else ("TRAIL" if hit_trail else "DD")
                        net = _exit_all(sym, price, st, reason=reason)
                        if STATE.chat_id:
                            icon = "🟤" if reason == "TP" else ("🟠" if reason == "TRAIL" else "⛔")
                            mark = "✅" if net >= 0 else "❌"
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"{icon} {reason} {sym} @ {price:.4f} | Net: {net:+.4f} USDT {mark}"
                            )
                        continue

                    # DCA om priset rör sig emot
                    step = st.next_step_pct
                    need_dca = False
                    if st.pos.side == "LONG" and move_pct <= -step:
                        need_dca = True
                    if st.pos.side == "SHORT" and move_pct >= step:
                        need_dca = True
                    if need_dca and st.pos.safety_count < int(STATE.grid_cfg["max_safety"]):
                        usd = STATE.position_size * (STATE.grid_cfg["size_mult"] ** st.pos.safety_count)
                        _enter_leg(sym, st.pos.side, price, usd, st)
                        st.next_step_pct = max(STATE.grid_cfg["step_min"], st.next_step_pct * STATE.grid_cfg["step_mult"])
                        if STATE.chat_id:
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🧩 DCA {sym} {st.pos.side} @ {price:.4f} (leg {st.pos.safety_count})"
                            )

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# --------------- TELEGRAM -----------------
tg_app = Application.builder().token(BOT_TOKEN).build()

def status_text() -> str:
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            tstop = f" | trail={st.pos.trail_stop:.4f}" if st.pos.trail_stop else ""
            pos_lines.append(
                f"{s}: {st.pos.side} avg {st.pos.avg_price:.4f} → TP {st.pos.target_price:.4f}"
                f" | legs {st.pos.safety_count}{tstop}"
            )
    g = STATE.grid_cfg; r = STATE.risk_cfg; tr = STATE.trend_cfg
    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'} | Mode: {STATE.mode}",
        f"Timeframes: {', '.join(STATE.tfs)}",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.position_size:.1f} USDT | Fee per sida (mock): {STATE.fee_side:.4%}",
        (f"Grid: max_safety={g['max_safety']} step_mult={g['step_mult']} step_min%={g['step_min']} "
         f"size_mult={g['size_mult']} tp%={g['tp']} (+{g['tp_bonus']}%/safety)"),
        (f"Trend: on={tr['on']} | trigger={tr['trigger']}% | trail={tr['trail']}% | be={tr['be']}%"),
        f"Risk: dd={r['dd']}% | max_pos={r['max_pos']} | shorts={'ON' if r['allow_shorts'] else 'OFF'}",
        f"PnL total (NET): {total:+.4f} USDT",
        "Open: " + (", ".join(pos_lines) if pos_lines else "inga")
    ]
    return "\n".join(lines)

async def send_status(chat_id: int):
    await tg_app.bot.send_message(chat_id, status_text(), reply_markup=reply_kb())

# ---- Commands
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "MP Bot v41 (NO-AI + TREND) – redo ✅", reply_markup=reply_kb())
    await send_status(STATE.chat_id)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    await send_status(STATE.chat_id)

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(STATE.chat_id, "Engine: ON ✅", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

async def cmd_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    STATE.mode = "MOCK"
    STATE.engine_on = True
    await tg_app.bot.send_message(STATE.chat_id, "Mode: MOCK | Engine ON ✅", reply_markup=reply_kb())

async def cmd_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    STATE.mode = "LIVE"
    STATE.engine_on = True
    await tg_app.bot.send_message(STATE.chat_id, "Mode: LIVE | Engine ON ✅", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    await tg_app.bot.send_message(STATE.chat_id, "Använd: /timeframe 1m,3m,5m", reply_markup=reply_kb())

async def cmd_grid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
    toks = text.split()
    if len(toks) == 4 and toks[1] == "set":
        key, val = toks[2], toks[3]
        if key in STATE.grid_cfg:
            try:
                STATE.grid_cfg[key] = float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Grid uppdaterad.", reply_markup=reply_kb())
            except:
                await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
        await tg_app.bot.send_message(STATE.chat_id, f"Okänd grid-nyckel: {key}", reply_markup=reply_kb())
        return
    g = STATE.grid_cfg
    await tg_app.bot.send_message(
        STATE.chat_id,
        (f"Grid:\n  max_safety={g['max_safety']}\n  step_mult={g['step_mult']}\n  step_min%={g['step_min']}\n"
         f"  size_mult={g['size_mult']}\n  tp%={g['tp']} (+{g['tp_bonus']}%/safety)\n\nEx: /grid set step_mult 0.6"),
        reply_markup=reply_kb()
    )

async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks) == 4 and toks[1] == "set":
        key, val = toks[2], toks[3]
        if key in STATE.risk_cfg:
            try:
                if key == "max_pos":
                    STATE.risk_cfg[key] = int(val)
                elif key == "allow_shorts":
                    STATE.risk_cfg[key] = (val.lower() in ("1","on","true","yes"))
                else:
                    STATE.risk_cfg[key] = float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb())
            except:
                await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
        await tg_app.bot.send_message(STATE.chat_id, f"Okänd risk-nyckel: {key}", reply_markup=reply_kb())
        return
    r = STATE.risk_cfg
    await tg_app.bot.send_message(
        STATE.chat_id,
        (f"Risk:\n  dd={r['dd']}%\n  max_pos={r['max_pos']}\n  shorts={'ON' if r['allow_shorts'] else 'OFF'}\n"
         "Ex: /risk set dd 2.5   | /risk set max_pos 8   | /risk set allow_shorts on"),
        reply_markup=reply_kb()
    )

async def cmd_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks) == 4 and toks[1] == "set":
        key, val = toks[2], toks[3]
        if key in STATE.trend_cfg:
            try:
                if key == "on":
                    STATE.trend_cfg["on"] = (val.lower() in ("1","true","on","yes"))
                else:
                    STATE.trend_cfg[key] = float(val)
                await tg_app.bot.send_message(STATE.chat_id, "Trend uppdaterad.", reply_markup=reply_kb())
            except:
                await tg_app.bot.send_message(STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb())
            return
        await tg_app.bot.send_message(STATE.chat_id, f"Okänd trend-nyckel: {key}", reply_markup=reply_kb())
        return
    tr = STATE.trend_cfg
    await tg_app.bot.send_message(
        STATE.chat_id,
        (f"Trend:\n  on={tr['on']}\n  trigger={tr['trigger']}%\n  trail={tr['trail']}%\n  be={tr['be']}%\n"
         "Ex: /trend set trigger 1.2  | /trend set trail 0.3  | /trend set be 0.2  | /trend set on off"),
        reply_markup=reply_kb()
    )

async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    toks = update.message.text.strip().split()
    if len(toks) >= 3 and toks[0] == "/symbols":
        action, sym = toks[1], toks[2].upper()
        if action == "add":
            if sym not in STATE.symbols:
                STATE.symbols.append(sym)
                STATE.per_sym[sym] = SymState()
                await tg_app.bot.send_message(STATE.chat_id, f"La till: {sym}", reply_markup=reply_kb())
            else:
                await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns redan.", reply_markup=reply_kb())
            return
        if action == "remove":
            if sym in STATE.symbols:
                STATE.symbols = [s for s in STATE.symbols if s != sym]
                STATE.per_sym.pop(sym, None)
                await tg_app.bot.send_message(STATE.chat_id, f"Tog bort: {sym}", reply_markup=reply_kb())
            else:
                await tg_app.bot.send_message(STATE.chat_id, f"{sym} finns ej.", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(
        STATE.chat_id,
        ("Symbols: " + ", ".join(STATE.symbols) +
         "\nLägg till/ta bort:\n  /symbols add LINK-USDT\n  /symbols remove LINK-USDT"),
        reply_markup=reply_kb()
    )

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    lines = [f"📈 PnL total (NET): {total:+.4f} USDT"]
    for s in STATE.symbols:
        lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl_net:+.4f} USDT")
    await tg_app.bot.send_message(STATE.chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    rows = [["time","symbol","side","avg_price","exit_price","gross","fee_in","fee_out","net","safety_legs","reason"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            rows.append([r[k] for k in rows[0]])
    if len(rows) == 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb())
        return
    buf = io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="trades.csv", caption="Export CSV")

async def cmd_export_k4(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enkel K4-lik CSV: datum, värdepapper, antal, försäljningspris, omkostnadsbelopp, vinst/förlust."""
    STATE.chat_id = update.effective_chat.id
    rows = [["date","instrument","qty","proceeds_usdt","cost_usdt","result_usdt"]]
    for s in STATE.symbols:
        for r in STATE.per_sym[s].trades_log:
            qty = 0.0
            # approx kvantitet = summa qty i sista positionen; vi har inte kvar efter stängning,
            # så skriv 0 och resultatet istället – Skatteverket bryr sig om belopp/resultat.
            proceeds = float(r["exit_price"])  # per enhet – här förenklat
            cost = float(r["avg_price"])
            result = float(r["net"])
            rows.append([r["time"][:10], s, f"{qty:.6f}", f"{proceeds:.6f}", f"{cost:.6f}", f"{result:.6f}"])
    if len(rows) == 1:
        await tg_app.bot.send_message(STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb())
        return
    buf = io.StringIO(); csv.writer(buf).writerows(rows); buf.seek(0)
    await tg_app.bot.send_document(STATE.chat_id, document=io.BytesIO(buf.getvalue().encode("utf-8")),
                                   filename="k4.csv", caption="K4-underlag (förenklat)")

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            try:
                px = (await get_klines(s, STATE.tfs[0], limit=1))[-1][-1]
            except:
                continue
            net = _exit_all(s, px, st, reason="PANIC")
            closed.append(f"{s}:{net:+.4f}")
    await tg_app.bot.send_message(STATE.chat_id, "Panic close: " + (" | ".join(closed) if closed else "Inga positioner."),
                                  reply_markup=reply_kb())

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl_net = 0.0
        STATE.per_sym[s].trades_log.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

# register handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("start_mock", cmd_start_mock))
tg_app.add_handler(CommandHandler("start_live", cmd_start_live))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("trend", cmd_trend))
tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("export_k4", cmd_export_k4))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))

# --------------- FASTAPI (Webhook) ---------------
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    # webhook
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    # starta bot + engine
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
    return f"MP Bot v41 (NO-AI+TREND) OK | engine_on={STATE.engine_on} | mode={STATE.mode} | pnl_total={total:+.4f}"

@app.get("/health", response_class=JSONResponse)
async def health():
    total = sum(STATE.per_sym[s].realized_pnl_net for s in STATE.symbols)
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "mode": STATE.mode,
        "tfs": STATE.tfs,
        "symbols": STATE.symbols,
        "pnl_total": round(total, 6),
        "trend": STATE.trend_cfg,
        "grid": STATE.grid_cfg,
        "risk": STATE.risk_cfg
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
