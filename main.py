# main_v40_fixgrid.py
# - Endast "knapparna där nere" (ReplyKeyboard)
# - Grid-parametrar / risk sparas till state.json och laddas vid start
# - Mock-handel med avgifter (fee per side) och enkel grid/TP-logik
# - Kommandon: /status /engine_on /engine_off /timeframe /grid /risk /pnl /export_csv /panic /reset_pnl

import os
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler

# ------------------ ENV ------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")
TF_ORDER = ["1m", "3m", "5m", "15m"]
TIMEFRAMES = [x.strip() for x in os.getenv("TIMEFRAMES", "1m,3m,5m").split(",") if x.strip()]

POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE_PCT", "0.1")) / 100.0  # 0.1% per sida
STATE_PATH = os.getenv("STATE_PATH", "state.json")

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# ------------------ STATE ------------------
@dataclass
class Trade:
    ts: float
    symbol: str
    side: str             # LONG/SHORT
    entry: float
    exit: float
    qty: float
    fee_in: float
    fee_out: float
    pnl_net: float

@dataclass
class Position:
    side: str             # LONG/SHORT
    entry: float
    qty: float
    stop: Optional[float] = None

@dataclass
class GridConfig:
    max_safety: int = 3           # max antal safety-buys/sells
    step_mult: float = 0.5        # multiplikator för steg (djupare -> större steg)
    step_min_pct: float = 0.30    # minsta steg i % mellan safety orders (0.30 = 0.30%)
    size_mult: float = 1.5        # storlekstrappa
    tp_pct: float = 0.25          # takeprofit procent av snittpris (+ extra 0.05% per safety)
    # OBS: alla dessa sparas/laddas

@dataclass
class RiskConfig:
    dd_pct: float = 3.0           # max drawdown innan paus
    atr_min_pct: float = 0.02     # filtrering: för låg ATR -> låt bli
    atr_max_pct: float = 3.0      # för hög ATR -> låt bli
    max_pos: int = 5              # max samtidiga symbolpositioner
    shorts_on: bool = True        # tillåt korta

@dataclass
class SymState:
    long_pos: Optional[Position] = None
    short_pos: Optional[Position] = None
    safety_count_long: int = 0
    safety_count_short: int = 0
    realized_pnl: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    last_prices: Dict[str, float] = field(default_factory=dict)  # tf->last close

@dataclass
class EngineState:
    engine_on: bool = False
    timeframes: List[str] = field(default_factory=lambda: TIMEFRAMES[:])
    symbols: List[str] = field(default_factory=lambda: SYMBOLS[:])
    grid: GridConfig = field(default_factory=GridConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    pos_size_usdt: float = POSITION_SIZE_USDT
    fee_per_side: float = FEE_PER_SIDE
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ---------- PERSISTENS ----------
def load_state():
    if not os.path.exists(STATE_PATH):
        return
    try:
        with open(STATE_PATH, "r") as f:
            raw = json.load(f)
        # ladda grid
        g = raw.get("grid", {})
        STATE.grid = GridConfig(
            max_safety=int(g.get("max_safety", STATE.grid.max_safety)),
            step_mult=float(g.get("step_mult", STATE.grid.step_mult)),
            step_min_pct=float(g.get("step_min_pct", STATE.grid.step_min_pct)),
            size_mult=float(g.get("size_mult", STATE.grid.size_mult)),
            tp_pct=float(g.get("tp_pct", STATE.grid.tp_pct)),
        )
        # ladda risk
        r = raw.get("risk", {})
        STATE.risk = RiskConfig(
            dd_pct=float(r.get("dd_pct", STATE.risk.dd_pct)),
            atr_min_pct=float(r.get("atr_min_pct", STATE.risk.atr_min_pct)),
            atr_max_pct=float(r.get("atr_max_pct", STATE.risk.atr_max_pct)),
            max_pos=int(r.get("max_pos", STATE.risk.max_pos)),
            shorts_on=bool(r.get("shorts_on", STATE.risk.shorts_on)),
        )
        # övrigt
        STATE.timeframes = raw.get("timeframes", STATE.timeframes)
        STATE.pos_size_usdt = float(raw.get("pos_size_usdt", STATE.pos_size_usdt))
        STATE.fee_per_side = float(raw.get("fee_per_side", STATE.fee_per_side))
    except Exception:
        pass

def save_state():
    data = {
        "grid": asdict(STATE.grid),
        "risk": asdict(STATE.risk),
        "timeframes": STATE.timeframes,
        "pos_size_usdt": STATE.pos_size_usdt,
        "fee_per_side": STATE.fee_per_side,
    }
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, STATE_PATH)

load_state()

# ---------- UI ----------
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/risk")],
        [KeyboardButton("/export_csv")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def fmt_grid() -> str:
    g = STATE.grid
    return (f"Grid:\n"
            f"  max_safety={g.max_safety}\n"
            f"  step_mult={g.step_mult}\n"
            f"  step_min%={g.step_min_pct}\n"
            f"  size_mult={g.size_mult}\n"
            f"  tp%={g.tp_pct} (+0.05%/safety)\n"
            f"Ex: /grid set step_mult 0.7")

def fmt_risk() -> str:
    r = STATE.risk
    return (f"Risk:\n"
            f"  dd%={r.dd_pct} | atr% {r.atr_min_pct}–{r.atr_max_pct}\n"
            f"  max_pos={r.max_pos} | shorts={'ON' if r.shorts_on else 'OFF'}\n"
            f"Ex: /risk set dd 5\n"
            f"    /risk set max_pos 8\n"
            f"    /risk set shorts on")

# ---------- MARKET HELP ----------
async def get_klines(symbol: str, tf: str, limit: int = 2) -> List[Tuple[int,float,float,float,float]]:
    """(t_ms, open, high, low, close) nyast först"""
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]
    out = []
    for row in data[:limit]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

def fee_for_notional(notional: float) -> float:
    return notional * STATE.fee_per_side

# ---------- ENGINE (väldigt enkel mock-grid) ----------
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    last_req: Dict[Tuple[str,str], float] = {}
    while True:
        try:
            if STATE.engine_on:
                # begränsa samtidiga positioner
                open_syms = 0
                for s in STATE.symbols:
                    st = STATE.per_sym[s]
                    if st.long_pos or st.short_pos:
                        open_syms += 1

                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]

                    # runda-robin över TFs men throttla så vi inte 429:ar
                    for tf in STATE.timeframes:
                        key = (sym, tf)
                        now = datetime.now().timestamp()
                        if now - last_req.get(key, 0) < 6:  # minst 6s mellan samma (sym,tf)
                            continue
                        last_req[key] = now

                        try:
                            kl = await get_klines(sym, tf, limit=2)
                        except Exception:
                            continue
                        if not kl:
                            continue
                        last_closed = kl[1]  # använd stängd candle
                        _, o, h, l, c = last_closed
                        st.last_prices[tf] = c

                        # enkel trendindikator (MA via två tidsramar)
                        prices = [p for p in st.last_prices.values()]
                        if len(prices) >= 2:
                            avg = sum(prices) / len(prices)
                        else:
                            avg = c

                        # köp/sälj-signal: avvikelse från "avg"
                        if open_syms < STATE.risk.max_pos:
                            dev_pct = (c - avg) / avg * 100.0 if avg != 0 else 0.0

                            # LONG-start om priset är X% under snittet
                            if dev_pct <= -STATE.grid.step_min_pct and st.long_pos is None:
                                entry = c
                                qty = STATE.pos_size_usdt / entry if entry > 0 else 0.0
                                st.long_pos = Position("LONG", entry=entry, qty=qty)
                                st.safety_count_long = 0
                                # stop sparas ej (griden jobbar med TP)
                                fee_in = fee_for_notional(qty * entry)
                                await app.bot.send_message(STATE.chat_id,
                                    f"🟢 ENTRY {sym} LONG @ {entry:.4f} | QTY {qty:.6f} | Avgift~ {fee_in:.4f} USDT")

                            # SHORT-start om priset är X% över snittet (om shorts tillåtet)
                            if STATE.risk.shorts_on and dev_pct >= STATE.grid.step_min_pct and st.short_pos is None:
                                entry = c
                                qty = STATE.pos_size_usdt / entry if entry > 0 else 0.0
                                st.short_pos = Position("SHORT", entry=entry, qty=qty)
                                st.safety_count_short = 0
                                fee_in = fee_for_notional(qty * entry)
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔴 ENTRY {sym} SHORT @ {entry:.4f} | QTY {qty:.6f} | Avgift~ {fee_in:.4f} USDT")

                        # LONG hantering: TP + safety buys
                        if st.long_pos:
                            avg_entry = st.long_pos.entry
                            qty = st.long_pos.qty
                            # TP: tp_pct + 0.05% per safety
                            tp_pct = STATE.grid.tp_pct + st.safety_count_long * 0.05
                            want = avg_entry * (1.0 + tp_pct / 100.0)
                            # Safety buy: om priset fallit ytterligare "step"
                            step_needed = STATE.grid.step_min_pct * ((STATE.grid.step_mult ** st.safety_count_long) if st.safety_count_long > 0 else 1.0)
                            need_price = avg_entry * (1.0 - step_needed / 100.0)

                            if c >= want:
                                # Sälj allt
                                fee_in = fee_for_notional(qty * avg_entry)
                                fee_out = fee_for_notional(qty * c)
                                gross = (c - avg_entry) * qty
                                net = gross - fee_in - fee_out
                                st.realized_pnl += net
                                st.trades.append(Trade(datetime.now(timezone.utc).timestamp(), sym, "LONG",
                                                       avg_entry, c, qty, fee_in, fee_out, net))
                                await app.bot.send_message(STATE.chat_id,
                                    f"🟤 EXIT {sym} @ {c:.4f} | Net: {net:+.4f} USDT "
                                    f"(avgifter in:{fee_in:.4f} ut:{fee_out:.4f})")
                                st.long_pos = None
                                st.safety_count_long = 0

                            elif c <= need_price and st.safety_count_long < STATE.grid.max_safety:
                                # Köp mer (safety)
                                add_qty = qty * (STATE.grid.size_mult ** (st.safety_count_long + 1) - 1.0)
                                notional = add_qty * c
                                if notional > 0:
                                    fee_in = fee_for_notional(notional)
                                    # uppdatera snitt
                                    new_qty = qty + add_qty
                                    new_avg = (avg_entry * qty + c * add_qty) / new_qty
                                    st.long_pos.qty = new_qty
                                    st.long_pos.entry = new_avg
                                    st.safety_count_long += 1
                                    await app.bot.send_message(STATE.chat_id,
                                        f"🔧 LONG safety {sym} #{st.safety_count_long} @ {c:.4f} | "
                                        f"ny avg {new_avg:.4f} | qty {new_qty:.6f} | avgift~ {fee_in:.4f}")

                        # SHORT hantering: TP + safety sells
                        if st.short_pos:
                            avg_entry = st.short_pos.entry
                            qty = st.short_pos.qty
                            tp_pct = STATE.grid.tp_pct + st.safety_count_short * 0.05
                            want = avg_entry * (1.0 - tp_pct / 100.0)  # profit när pris sjunker
                            step_needed = STATE.grid.step_min_pct * ((STATE.grid.step_mult ** st.safety_count_short) if st.safety_count_short > 0 else 1.0)
                            need_price = avg_entry * (1.0 + step_needed / 100.0)

                            if c <= want:
                                fee_in = fee_for_notional(qty * avg_entry)
                                fee_out = fee_for_notional(qty * c)
                                gross = (avg_entry - c) * qty
                                net = gross - fee_in - fee_out
                                st.realized_pnl += net
                                st.trades.append(Trade(datetime.now(timezone.utc).timestamp(), sym, "SHORT",
                                                       avg_entry, c, qty, fee_in, fee_out, net))
                                await app.bot.send_message(STATE.chat_id,
                                    f"⚪️ EXIT {sym} (SHORT) @ {c:.4f} | Net: {net:+.4f} USDT "
                                    f"(avgifter in:{fee_in:.4f} ut:{fee_out:.4f})")
                                st.short_pos = None
                                st.safety_count_short = 0

                            elif c >= need_price and st.safety_count_short < STATE.grid.max_safety:
                                add_qty = qty * (STATE.grid.size_mult ** (st.safety_count_short + 1) - 1.0)
                                notional = add_qty * c
                                if notional > 0:
                                    fee_in = fee_for_notional(notional)
                                    new_qty = qty + add_qty
                                    # “snitt” för short: vägda på entries
                                    new_avg = (avg_entry * qty + c * add_qty) / new_qty
                                    st.short_pos.qty = new_qty
                                    st.short_pos.entry = new_avg
                                    st.safety_count_short += 1
                                    await app.bot.send_message(STATE.chat_id,
                                        f"🔧 SHORT safety {sym} #{st.safety_count_short} @ {c:.4f} | "
                                        f"ny avg {new_avg:.4f} | qty {new_qty:.6f} | avgift~ {fee_in:.4f}")

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ------------------ TELEGRAM ------------------
tg_app = Application.builder().token(BOT_TOKEN).build()

def pnl_total() -> float:
    return sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)

async def send_status(chat_id: int):
    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Timeframes: {', '.join(STATE.timeframes)}    # ändra med /timeframe",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {STATE.pos_size_usdt:.1f} USDT | Fee/side: {STATE.fee_per_side*100:.4f}%",
        fmt_grid(),
        "Risk: dd={:.1f}% | atr% {:.2f}–{:.2f} | max_pos={} | shorts={}".format(
            STATE.risk.dd_pct, STATE.risk.atr_min_pct, STATE.risk.atr_max_pct,
            STATE.risk.max_pos, "ON" if STATE.risk.shorts_on else "OFF"),
        f"PnL total (NET): {pnl_total():+.4f} USDT",
    ]
    # positioner
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.long_pos:
            pos_lines.append(f"{s} LONG @ {st.long_pos.entry:.4f} | qty {st.long_pos.qty:.6f} "
                             f"(safety {st.safety_count_long})")
        if st.short_pos:
            pos_lines.append(f"{s} SHORT @ {st.short_pos.entry:.4f} | qty {st.short_pos.qty:.6f} "
                             f"(safety {st.safety_count_short})")
    lines.append("Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"))
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    lines = [f"📈 PnL total (NET): {pnl_total():+.4f} USDT"]
    for s in STATE.symbols:
        lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl:+.4f} USDT")
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! v40 fixgrid redo ✅", reply_markup=reply_kb())
    await send_status(STATE.chat_id)

async def cmd_status(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_status(STATE.chat_id)

async def cmd_engine_on(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = True
    save_state()
    await tg_app.bot.send_message(STATE.chat_id, "Engine: ON ✅", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.engine_on = False
    save_state()
    await tg_app.bot.send_message(STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    # cykla 1m->3m->5m->15m eller ta lista via nästa meddelande
    cur = ",".join(STATE.timeframes)
    try:
        i = TF_ORDER.index(STATE.timeframes[0])
        STATE.timeframes = [TF_ORDER[(i+1) % len(TF_ORDER)]]
    except Exception:
        STATE.timeframes = ["1m"]
    save_state()
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframes: {', '.join(STATE.timeframes)}", reply_markup=reply_kb())

async def cmd_grid(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
    # /grid set key value
    toks = text.split()
    if len(toks) == 4 and toks[1].lower() == "set":
        key, val = toks[2].lower(), toks[3]
        g = STATE.grid
        try:
            if key == "max_safety":
                g.max_safety = int(val)
            elif key == "step_mult":
                g.step_mult = float(val)
            elif key in ("step_min", "step_min%", "step_min_pct"):
                g.step_min_pct = float(val)
            elif key == "size_mult":
                g.size_mult = float(val)
            elif key in ("tp", "tp_pct"):
                g.tp_pct = float(val)
            else:
                await tg_app.bot.send_message(STATE.chat_id, "Okänd nyckel. Tillåtna: max_safety, step_mult, step_min, size_mult, tp", reply_markup=reply_kb())
                return
            save_state()
            await tg_app.bot.send_message(STATE.chat_id, "Grid uppdaterad.", reply_markup=reply_kb())
        except Exception as e:
            await tg_app.bot.send_message(STATE.chat_id, f"Fel: {e}", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, fmt_grid(), reply_markup=reply_kb())

async def cmd_risk(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    text = update.message.text.strip()
    toks = text.split()
    # /risk set dd 6
    if len(toks) == 4 and toks[1].lower() == "set":
        key, val = toks[2].lower(), toks[3]
        try:
            if key == "dd":
                STATE.risk.dd_pct = float(val)
            elif key == "max_pos":
                STATE.risk.max_pos = int(val)
            elif key == "shorts":
                STATE.risk.shorts_on = val.lower() in ("on", "true", "1", "yes", "ja")
            else:
                await tg_app.bot.send_message(STATE.chat_id, "Tillåtna: dd, max_pos, shorts(on/off)", reply_markup=reply_kb())
                return
            save_state()
            await tg_app.bot.send_message(STATE.chat_id, "Risk uppdaterad.", reply_markup=reply_kb())
        except Exception as e:
            await tg_app.bot.send_message(STATE.chat_id, f"Fel: {e}", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, fmt_risk(), reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        for side in ("long_pos", "short_pos"):
            pos = getattr(st, side)
            if pos:
                # stäng på “entry” (neutral)
                fee_in = fee_for_notional(pos.entry * pos.qty)
                fee_out = fee_for_notional(pos.entry * pos.qty)
                net = -fee_in - fee_out
                st.realized_pnl += net
                st.trades.append(Trade(datetime.now(timezone.utc).timestamp(), s,
                                       "LONG" if side == "long_pos" else "SHORT",
                                       pos.entry, pos.entry, pos.qty, fee_in, fee_out, net))
                setattr(st, side, None)
                closed.append(f"{s} {('LONG' if side=='long_pos' else 'SHORT')} {net:+.4f}")
    await tg_app.bot.send_message(STATE.chat_id, "Panic close: " + (", ".join(closed) if closed else "Inga positioner."), reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl = 0.0
        STATE.per_sym[s].trades.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    # skriv enkel CSV
    rows = ["ts_iso,symbol,side,entry,exit,qty,fee_in,fee_out,pnl_net"]
    for s in STATE.symbols:
        for t in STATE.per_sym[s].trades:
            ts_iso = datetime.fromtimestamp(t.ts, tz=timezone.utc).isoformat()
            rows.append(f"{ts_iso},{t.symbol},{t.side},{t.entry:.6f},{t.exit:.6f},{t.qty:.6f},{t.fee_in:.6f},{t.fee_out:.6f},{t.pnl_net:.6f}")
    path = "trades_export.csv"
    with open(path, "w") as f:
        f.write("\n".join(rows))
    await tg_app.bot.send_document(STATE.chat_id, document=open(path, "rb"), caption="Export klar.")

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("risk", cmd_risk))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))

# ------------- FASTAPI (webhook) -------------
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    # säkerställ att vi laddar state (om deploy efter ändringar)
    load_state()
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
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "timeframes": STATE.timeframes,
        "grid": asdict(STATE.grid),
        "risk": asdict(STATE.risk),
        "pos_size_usdt": STATE.pos_size_usdt,
        "fee_per_side": STATE.fee_per_side,
        "pnl_total": round(pnl_total(), 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
