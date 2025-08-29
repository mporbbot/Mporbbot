# main_v40_once.py
import os
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler

# ────────────────────────── ENV / SETTINGS ──────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")

# grid/avgifter
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PCT_PER_SIDE = float(os.getenv("FEE_PCT", "0.001"))  # 0.1% = 0.001

# timeframe
TIMEFRAME = os.getenv("TIMEFRAME", "1m")  # 1m/3m/5m/15m
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# grid-parametrar (lite snäll baseline)
GRID_MAX_SAFETY = int(os.getenv("GRID_MAX_SAFETY", "3"))   # max antal DCA
GRID_STEP_MIN_PCT = float(os.getenv("GRID_STEP_MIN_PCT", "0.15"))  # min % steg till nästa DCA
GRID_STEP_MULT = float(os.getenv("GRID_STEP_MULT", "0.5"))  # hur mycket stegen växer
GRID_SIZE_MULT = float(os.getenv("GRID_SIZE_MULT", "1.5"))  # DCA storleks-multiplikator
TP_PCT = float(os.getenv("TP_PCT", "0.25"))                 # take profit i %
ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "1") != "0"

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"

# ────────────────────────── STATE ──────────────────────────
@dataclass
class Leg:
    price: float
    qty: float

@dataclass
class Position:
    side: str                      # LONG eller SHORT
    legs: List[Leg] = field(default_factory=list)
    safety: int = 0                # hur många DCA som lagts
    sl: Optional[float] = None

    @property
    def qty(self) -> float:
        return sum(l.qty for l in self.legs)

    @property
    def avg(self) -> float:
        v = sum(l.price * l.qty for l in self.legs)
        q = self.qty
        return v / q if q > 0 else 0.0

@dataclass
class SymState:
    pos: Optional[Position] = None
    pnl_gross: float = 0.0
    fees_paid: float = 0.0
    # per-candle-låsning
    last_candle_ms: Optional[int] = None
    did_entry_this_candle: bool = False
    did_exit_this_candle: bool = False

@dataclass
class EngineState:
    engine_on: bool = True
    timeframe: str = TIMEFRAME
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ────────────────────────── UI (bara neder-knappar) ──────────────────────────
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/grid"), KeyboardButton("/export_csv")],
        [KeyboardButton("/panic"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ────────────────────────── DATA ──────────────────────────
async def get_klines(symbol: str, tf: str, limit: int = 3) -> List[Tuple[int,float,float,float,float]]:
    """Returnerar [(open_time_ms, open, high, low, close)], nyast först."""
    k_tf = TF_MAP.get(tf, "1min")
    params = {"symbol": symbol, "type": k_tf}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(KUCOIN_KLINES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]  # newest-first strings
    out = []
    for row in data[:limit]:
        t_ms = int(row[0]) * 1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        out.append((t_ms, o, h, l, c))
    return out

# ────────────────────────── GRID-LOGIK ──────────────────────────
def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

def next_safety_step_pct(safety: int) -> float:
    """Hur långt priset måste gå emot oss för nästa DCA."""
    return GRID_STEP_MIN_PCT * (1.0 + GRID_STEP_MULT * safety)

def dca_size_for_leg(base: float, safety: int) -> float:
    """Storlek (USDT) för DCA-leg."""
    return base * (GRID_SIZE_MULT ** safety)

def fee_cost(amount_usdt: float) -> float:
    return amount_usdt * FEE_PCT_PER_SIDE

# ────────────────────────── ENGINE ──────────────────────────
tg_app = Application.builder().token(BOT_TOKEN).build()

async def notify(chat_id: int, text: str):
    try:
        await tg_app.bot.send_message(chat_id, text)
    except:
        pass

async def engine_loop(app: Application):
    chat_id = None
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]

                    kl = await get_klines(sym, STATE.timeframe, limit=3)
                    if len(kl) < 2:
                        continue
                    # använd SENASTE STÄNGDA candle (inte pågående)
                    last_closed = kl[1]  # (t_ms, o,h,l,c)
                    t_ms, o, h, l, c = last_closed

                    # reset per-candle flaggor när ny candle kommit
                    if st.last_candle_ms != t_ms:
                        st.last_candle_ms = t_ms
                        st.did_entry_this_candle = False
                        st.did_exit_this_candle = False

                    # pris vi jobbar med:
                    price = c

                    # 1) ingen position → öppna grid “första leg” men bara EN gång per candle
                    if st.pos is None and not st.did_entry_this_candle:
                        # enkel trigger: om prisets 1-candle-change är större än step_min% öppna mot-riktning
                        prev = kl[2] if len(kl) > 2 else last_closed
                        prev_close = prev[4]
                        chg = pct(price, prev_close)

                        opened = False
                        # mean-reversion start: dump → LONG, pump → SHORT
                        if chg <= -GRID_STEP_MIN_PCT:
                            # LONG start
                            qty = POSITION_SIZE_USDT / price if price > 0 else 0.0
                            st.pos = Position(side="LONG", legs=[Leg(price=price, qty=qty)], safety=0)
                            st.fees_paid += fee_cost(POSITION_SIZE_USDT)
                            opened = True
                            if chat_id:
                                await notify(chat_id,
                                             f"🟢 ENTRY {sym} LONG @ {price:.4f} | QTY {qty:.6f} | Avgift~{fee_cost(POSITION_SIZE_USDT):.4f} USDT")
                        elif ALLOW_SHORTS and chg >= GRID_STEP_MIN_PCT:
                            # SHORT start
                            qty = POSITION_SIZE_USDT / price if price > 0 else 0.0
                            st.pos = Position(side="SHORT", legs=[Leg(price=price, qty=qty)], safety=0)
                            st.fees_paid += fee_cost(POSITION_SIZE_USDT)
                            opened = True
                            if chat_id:
                                await notify(chat_id,
                                             f"🔻 ENTRY {sym} SHORT @ {price:.4f} | QTY {qty:.6f} | Avgift~{fee_cost(POSITION_SIZE_USDT):.4f} USDT")

                        if opened:
                            st.did_entry_this_candle = True

                    # 2) har position → TP / DCA, men bara EN exit per candle
                    if st.pos:
                        pos = st.pos
                        avg = pos.avg

                        # Take Profit
                        target_up = avg * (1 + TP_PCT/100.0)
                        target_dn = avg * (1 - TP_PCT/100.0)

                        hit_tp = False
                        if pos.side == "LONG" and price >= target_up:
                            hit_tp = True
                        elif pos.side == "SHORT" and price <= target_dn:
                            hit_tp = True

                        if hit_tp and not st.did_exit_this_candle:
                            # räkna PnL och fees
                            notional_in = sum(l.price * l.qty for l in pos.legs)
                            notional_out = price * pos.qty
                            gross = (notional_out - notional_in) if pos.side == "LONG" else (notional_in - notional_out)
                            fee_in = notional_in * FEE_PCT_PER_SIDE
                            fee_out = notional_out * FEE_PCT_PER_SIDE
                            net = gross - fee_in - fee_out
                            st.pnl_gross += net
                            st.fees_paid += (fee_in + fee_out)
                            side_icon = "🟤" if pos.side == "LONG" else "⚪️"
                            if chat_id:
                                await notify(chat_id,
                                             f"{side_icon} EXIT {sym} @ {price:.4f} | Net: {net:+.4f} USDT "
                                             f"(avgifter in:{fee_in:.4f} ut:{fee_out:.4f})")
                            st.pos = None
                            st.did_exit_this_candle = True
                            continue  # inga fler åtgärder denna candle

                        # DCA (safety) – endast om vi inte redan har gjort ENTRY denna candle
                        if pos and pos.safety < GRID_MAX_SAFETY and not st.did_entry_this_candle:
                            step_need = next_safety_step_pct(pos.safety + 1)  # nästa steg
                            move = pct(price, avg)
                            need_down = -step_need   # LONG: pris under avg med X%
                            need_up = step_need      # SHORT: pris över avg med X%

                            dca = False
                            if pos.side == "LONG" and move <= need_down:
                                dca = True
                            elif pos.side == "SHORT" and move >= need_up:
                                dca = True

                            if dca:
                                usdt = dca_size_for_leg(POSITION_SIZE_USDT, pos.safety + 1)
                                qty = usdt / price if price > 0 else 0.0
                                pos.legs.append(Leg(price=price, qty=qty))
                                pos.safety += 1
                                st.fees_paid += fee_cost(usdt)
                                if chat_id:
                                    await notify(chat_id,
                                                 f"🧩 DCA {sym} {pos.side} @ {price:.4f} | leg {pos.safety} | QTY {qty:.6f}")
                                st.did_entry_this_candle = True  # räkna som en “entry-akt” denna candle

            await asyncio.sleep(2)
        except Exception as e:
            # fånga ev nätfel etc
            try:
                if chat_id:
                    await notify(chat_id, f"⚠️ Engine-fel: {e}")
            except:
                pass
            await asyncio.sleep(5)
        # uppdatera chat_id “lazily” efter att någon skrivit
        try:
            chat_id = getattr(engine_loop, "_chat_id", chat_id)
        except:
            pass

# ────────────────────────── TELEGRAM HANDLERS ──────────────────────────
async def send_status(chat_id: int):
    total_net = sum(STATE.per_sym[s].pnl_gross for s in STATE.symbols)
    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Timeframe: {STATE.timeframe}      # ändra med /timeframe",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Size: {POSITION_SIZE_USDT:.1f} USDT | Fee/side: {FEE_PCT_PER_SIDE*100:.4f}%",
        "",
        f"Grid: max_safety={GRID_MAX_SAFETY}",
        f"      step_mult={GRID_STEP_MULT}",
        f"      step_min%={GRID_STEP_MIN_PCT}",
        f"      size_mult={GRID_SIZE_MULT}",
        f"      tp%={TP_PCT}",
        "",
        f"PnL total (NET): {total_net:+.4f} USDT",
    ]
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        lines.append(f"• {s}: {st.pnl_gross:+.4f} USDT")
    # positioner
    pos_lines = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        if st.pos:
            pos_lines.append(f"{s} {st.pos.side} avg {st.pos.avg:.4f} qty {st.pos.qty:.6f} safety {st.pos.safety}")
    lines.append("Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"))

    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def cmd_start(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id  # för engine-notiser
    await tg_app.bot.send_message(engine_loop._chat_id, "Grid-bot v40 (once-per-candle) redo ✅", reply_markup=reply_kb())
    await send_status(engine_loop._chat_id)

async def cmd_status(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    await send_status(engine_loop._chat_id)

async def cmd_engine_on(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    STATE.engine_on = True
    await tg_app.bot.send_message(engine_loop._chat_id, "Engine: ON ✅", reply_markup=reply_kb())

async def cmd_engine_off(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    STATE.engine_on = False
    await tg_app.bot.send_message(engine_loop._chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    order = ["1m","3m","5m","15m","30m","1h"]
    i = order.index(STATE.timeframe) if STATE.timeframe in order else 0
    STATE.timeframe = order[(i+1) % len(order)]
    await tg_app.bot.send_message(engine_loop._chat_id, f"Timeframe satt till: {STATE.timeframe}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    await send_status(engine_loop._chat_id)

async def cmd_reset_pnl(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        st.pnl_gross = 0.0
        st.fees_paid = 0.0
    await tg_app.bot.send_message(engine_loop._chat_id, "PnL återställd.", reply_markup=reply_kb())

async def cmd_grid(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    msg = (f"Grid:\n"
           f"max_safety={GRID_MAX_SAFETY}\n"
           f"step_mult={GRID_STEP_MULT}\n"
           f"step_min%={GRID_STEP_MIN_PCT}\n"
           f"size_mult={GRID_SIZE_MULT}\n"
           f"tp%={TP_PCT}\n"
           f"Ex: /grid_set step_mult 0.7")
    await tg_app.bot.send_message(engine_loop._chat_id, msg, reply_markup=reply_kb())

async def cmd_grid_set(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    try:
        # förvänta: /grid_set param värde
        parts = update.message.text.strip().split()
        _, key, val = parts[0], parts[1], parts[2]
        global GRID_MAX_SAFETY, GRID_STEP_MULT, GRID_STEP_MIN_PCT, GRID_SIZE_MULT, TP_PCT
        if key == "max_safety":
            GRID_MAX_SAFETY = int(val)
        elif key == "step_mult":
            GRID_STEP_MULT = float(val)
        elif key == "step_min":
            GRID_STEP_MIN_PCT = float(val)
        elif key == "size_mult":
            GRID_SIZE_MULT = float(val)
        elif key == "tp":
            TP_PCT = float(val)
        await tg_app.bot.send_message(engine_loop._chat_id, "Grid uppdaterad.", reply_markup=reply_kb())
        await cmd_grid(update, _)
    except Exception as e:
        await tg_app.bot.send_message(engine_loop._chat_id, f"Fel: {e}", reply_markup=reply_kb())

async def cmd_export_csv(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    # Minimal stub – implementera riktig loggning till CSV om du vill
    await tg_app.bot.send_message(engine_loop._chat_id, "CSV-export kommer i en senare version.", reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    engine_loop._chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            price = st.pos.avg  # stäng nära medel (mock)
            notional_in = sum(l.price * l.qty for l in st.pos.legs)
            notional_out = price * st.pos.qty
            gross = (notional_out - notional_in) if st.pos.side == "LONG" else (notional_in - notional_out)
            fee_in = notional_in * FEE_PCT_PER_SIDE
            fee_out = notional_out * FEE_PCT_PER_SIDE
            net = gross - fee_in - fee_out
            st.pnl_gross += net
            st.fees_paid += (fee_in + fee_out)
            closed.append(f"{s} {net:+.4f}")
            st.pos = None
    msg = " | ".join(closed) if closed else "Inga positioner."
    await tg_app.bot.send_message(engine_loop._chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

# register
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("grid", cmd_grid))
tg_app.add_handler(CommandHandler("grid_set", cmd_grid_set))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("panic", cmd_panic))

# ────────────────────────── FASTAPI WEBHOOK ──────────────────────────
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

@app.get("/")
async def root():
    total_net = sum(STATE.per_sym[s].pnl_gross for s in STATE.symbols)
    return {"ok": True, "engine_on": STATE.engine_on, "tf": STATE.timeframe, "pnl_net": round(total_net, 6)}

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
