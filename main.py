# main_v36_ai_thbuttons.py
# Enkel AI-bot baserad på v36 med mock-avgifter + knappar för att justera th_up/th_down
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

# ================== ENV ==================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")
# flera TF stöds, separera med komma
TIMEFRAMES = [s.strip() for s in os.getenv("TIMEFRAMES", "3m,5m,15m").split(",") if s.strip()]
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))  # 0.1% per sida som default

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# ================== STATE ==================
@dataclass
class Position:
    side: str   # LONG eller SHORT
    entry: float
    stop: float
    qty: float

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_gross: float = 0.0
    realized_fees: float = 0.0
    trades: List[Tuple[str, float]] = field(default_factory=list)

@dataclass
class EngineState:
    engine_on: bool = False
    timeframes: List[str] = field(default_factory=lambda: TIMEFRAMES)
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    fee_per_side: float = FEE_PER_SIDE
    th_up: float = 0.35     # röstantal upp för köp
    th_down: float = -2.15  # röstantal ner för sälj
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ================== UI ==================
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],

        # --- THRESHOLD-UP KNAPPAR ---
        [KeyboardButton("/th_up +0.1"), KeyboardButton("/th_up -0.1")],
        [KeyboardButton("/th_up +0.5"), KeyboardButton("/th_up -0.5")],

        # --- THRESHOLD-DOWN KNAPPAR ---
        # OBS: th_down är negativ. "+0.1" gör den MINDRE negativ (närmare 0).
        [KeyboardButton("/th_down +0.1"), KeyboardButton("/th_down -0.1")],
        [KeyboardButton("/th_down +0.5"), KeyboardButton("/th_down -0.5")],

        [KeyboardButton("/save_model"), KeyboardButton("/load_model")],
        [KeyboardButton("/export_csv")],
        [KeyboardButton("/panic")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# ================== DATA ==================
async def get_klines(symbol: str, tf: str, limit: int = 60) -> List[Tuple[int, float, float, float, float]]:
    k_tf = TF_MAP.get(tf, "3min")
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
    return out  # nyast först

# ================== ENKEL "AI"-RÖSTNING ==================
def vote_signal(candles: List[Tuple[int, float, float, float, float]]) -> float:
    """
    Enkel heuristik: momentum/mean-reversion blandning.
    Returnerar ett rösttal: >0 köptryck, <0 säljtryck.
    """
    if len(candles) < 20:
        return 0.0
    # använd 20 senaste (nyast först i vår lista)
    closes = [c[4] for c in candles[:20]][::-1]  # äldst -> nyast
    ohlc = candles[:5]  # ett par senaste för wick/volatilitet (nyast först)
    # momentum: förändring senaste 3 & 8 candles
    def pct(a,b): 
        return 0.0 if a == 0 else (b-a)/a*100.0
    m3 = pct(closes[-4], closes[-1]) if len(closes)>=4 else 0.0
    m8 = pct(closes[-9], closes[-1]) if len(closes)>=9 else 0.0
    # wick-faktor på senaste stängda
    _, o,h,l,c = candles[1] if len(candles)>1 else (0,0,0,0,0)
    body = abs(c-o); range_ = max(1e-9, h-l)
    wick_bias = (h-c)/range_ - (o-l)/range_  # positiv om säljspikar, negativ om köpspikar
    score = 0.6*m3 + 0.4*m8 - 0.5*wick_bias
    return score

# ================== ENGINE ==================
def fee(amount_usdt: float) -> float:
    return amount_usdt * STATE.fee_per_side

async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    # hoppa om redan öppen position – här gör vi enkel variant: en position åt gången
                    if st.pos:
                        # trailing stop mot senaste låg på 3m TF om möjligt
                        try:
                            kl = await get_klines(sym, STATE.timeframes[0], limit=3)
                            if len(kl) >= 2:
                                last_closed = kl[1]
                                new_sl = max(st.pos.stop, last_closed[3]) if st.pos.side == "LONG" else min(st.pos.stop, last_closed[2])
                                st.pos.stop = new_sl
                        except Exception:
                            pass
                    else:
                        # RÖSTNING över TFs
                        votes = []
                        for tf in STATE.timeframes:
                            try:
                                kl = await get_klines(sym, tf, limit=60)
                                votes.append(vote_signal(kl))
                            except Exception:
                                pass
                        if not votes:
                            continue
                        vote_avg = sum(votes)/len(votes)

                        # LONG
                        if vote_avg >= STATE.th_up:
                            # entry vid senaste close på snabbaste TF
                            kl = await get_klines(sym, STATE.timeframes[0], limit=2)
                            if len(kl) >= 2:
                                px = kl[1][4]
                                qty = POSITION_SIZE_USDT/px if px>0 else 0.0
                                # stop under senaste stängda low
                                stop = kl[1][3]
                                st.pos = Position("LONG", entry=px, stop=stop, qty=qty)
                                fee_est = fee(POSITION_SIZE_USDT)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 ENTRY {sym} LONG @ {px:.4f} | SL {stop:.4f} | QTY {qty:.6f} | Avgift~ {fee_est:.4f} USDT",
                                        reply_markup=reply_kb()
                                    )
                        # SHORT
                        elif vote_avg <= STATE.th_down:
                            kl = await get_klines(sym, STATE.timeframes[0], limit=2)
                            if len(kl) >= 2:
                                px = kl[1][4]
                                qty = POSITION_SIZE_USDT/px if px>0 else 0.0
                                stop = kl[1][2]  # över senaste high
                                st.pos = Position("SHORT", entry=px, stop=stop, qty=qty)
                                fee_est = fee(POSITION_SIZE_USDT)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔴 ENTRY {sym} SHORT @ {px:.4f} | SL {stop:.4f} | QTY {qty:.6f} | Avgift~ {fee_est:.4f} USDT",
                                        reply_markup=reply_kb()
                                    )
                    # EXIT-logik (tick-simulerad på stängd candle)
                    if st.pos:
                        kl = await get_klines(sym, STATE.timeframes[0], limit=2)
                        if len(kl) >= 2:
                            last_closed = kl[1]
                            _, o,h,l,c = last_closed
                            exit_now = False
                            exit_px = c
                            if st.pos.side == "LONG" and l <= st.pos.stop:
                                exit_now = True
                                exit_px = st.pos.stop
                            elif st.pos.side == "SHORT" and h >= st.pos.stop:
                                exit_now = True
                                exit_px = st.pos.stop
                            if exit_now:
                                gross = (exit_px - st.pos.entry)*st.pos.qty if st.pos.side=="LONG" else (st.pos.entry - exit_px)*st.pos.qty
                                fees = fee(POSITION_SIZE_USDT) + fee(POSITION_SIZE_USDT + gross)
                                net = gross - fees
                                st.realized_gross += gross
                                st.realized_fees += fees
                                st.trades.append((f"{sym} {st.pos.side}", net))
                                if len(st.trades)>100:
                                    st.trades = st.trades[-100:]
                                if STATE.chat_id:
                                    mark = "✅" if net>=0 else "❌"
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"{'🔴' if st.pos.side=='LONG' else '🟢'} EXIT {sym} @ {exit_px:.4f} | "
                                        f"Net: {net:+.4f} USDT {mark}\n"
                                        f"(avgifter in:{STATE.fee_per_side:.4%} ut:{STATE.fee_per_side:.4%})",
                                        reply_markup=reply_kb()
                                    )
                                st.pos = None

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}", reply_markup=reply_kb())
                except:
                    pass
            await asyncio.sleep(5)

# ================== TELEGRAM ==================
tg_app = Application.builder().token(BOT_TOKEN).build()

def total_net() -> float:
    gross = sum(s.realized_gross for s in STATE.per_sym.values())
    fees  = sum(s.realized_fees for s in STATE.per_sym.values())
    return gross - fees

async def send_status(chat_id: int):
    sym_pnls = []
    for s in STATE.symbols:
        st = STATE.per_sym[s]
        sym_pnls.append(f"{s}:+{(st.realized_gross - st.realized_fees):.4f}")
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(f"{s} {st.pos.side} @ {st.pos.entry:.4f} | SL {st.pos.stop:.4f} | QTY {st.pos.qty:.6f}")

    lines = [
        f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
        f"Timeframes: {', '.join(STATE.timeframes)}    # ändra med /timeframe",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Position size: {POSITION_SIZE_USDT:.1f} USDT | Fee per sida: {STATE.fee_per_side:.4%}",
        f"Trösklar: th_up={STATE.th_up:+.2f}  th_down={STATE.th_down:+.2f}",
        f"PnL total: {total_net():+.4f} USDT",
        f"PnL per symbol: {', '.join(sym_pnls)}",
        "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
    ]
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! AI-bot med tröskelknappar redo ✅", reply_markup=reply_kb())
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
    # Ange t.ex. "3m,5m,15m"
    txt = (update.message.text or "").replace("/timeframe", "").strip()
    if txt:
        tfs = [s.strip() for s in txt.split(",") if s.strip()]
        if tfs:
            STATE.timeframes = tfs
            await tg_app.bot.send_message(STATE.chat_id, f"Timeframes uppdaterade: {', '.join(STATE.timeframes)}", reply_markup=reply_kb())
            return
    await tg_app.bot.send_message(STATE.chat_id, f"Nuvarande: {', '.join(STATE.timeframes)}\nSkicka som: /timeframe 3m,5m,15m", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_status(STATE.chat_id)

def _adj(value: float, delta: float, *, clamp: Optional[Tuple[float,float]] = None) -> float:
    out = value + delta
    if clamp:
        lo, hi = clamp
        out = max(lo, min(hi, out))
    return round(out, 4)

# th_up +/- via knappar
async def cmd_th_up(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = (update.message.text or "").replace("/th_up", "").strip()
    delta = 0.0
    if txt:
        try:
            delta = float(txt.replace("+", ""))
        except:
            pass
    STATE.th_up = _adj(STATE.th_up, delta, clamp=(-5.0, 5.0))
    await tg_app.bot.send_message(STATE.chat_id, f"th_up -> {STATE.th_up:+.2f}", reply_markup=reply_kb())

# th_down +/- via knappar (negativ tröskel)
async def cmd_th_down(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    txt = (update.message.text or "").replace("/th_down", "").strip()
    delta = 0.0
    if txt:
        try:
            delta = float(txt.replace("+", ""))
        except:
            pass
    STATE.th_down = _adj(STATE.th_down, delta, clamp=(-5.0, 0.0))
    await tg_app.bot.send_message(STATE.chat_id, f"th_down -> {STATE.th_down:+.2f}", reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            if st.pos.side == "LONG":
                gross = (st.pos.stop - st.pos.entry)*st.pos.qty
            else:
                gross = (st.pos.entry - st.pos.stop)*st.pos.qty
            fees = fee(POSITION_SIZE_USDT) + fee(max(0.0, POSITION_SIZE_USDT + gross))
            net = gross - fees
            st.realized_gross += gross
            st.realized_fees += fees
            st.trades.append((f"{s} {st.pos.side} (panic)", net))
            st.pos = None
            closed.append(f"{s}:{net:+.4f}")
    msg = " | ".join(closed) if closed else "Inga positioner öppna."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

# (Mock) spara/ladda – minimalistiskt
import json, os as _os
MODEL_PATH = _os.getenv("MODEL_PATH", "/tmp/ai_model.json")

async def cmd_save_model(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    data = {"th_up": STATE.th_up, "th_down": STATE.th_down}
    with open(MODEL_PATH, "w") as f:
        json.dump(data, f)
    await tg_app.bot.send_message(STATE.chat_id, f"Modell sparad till {MODEL_PATH}", reply_markup=reply_kb())

async def cmd_load_model(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    if not os.path.exists(MODEL_PATH):
        await tg_app.bot.send_message(STATE.chat_id, "Ingen sparad modell hittad.", reply_markup=reply_kb())
        return
    with open(MODEL_PATH, "r") as f:
        data = json.load(f)
    STATE.th_up = float(data.get("th_up", STATE.th_up))
    STATE.th_down = float(data.get("th_down", STATE.th_down))
    await tg_app.bot.send_message(STATE.chat_id, f"Modell laddad. th_up={STATE.th_up:+.2f}, th_down={STATE.th_down:+.2f}", reply_markup=reply_kb())

# Dummy export (kan byggas ut till riktig CSV över trades)
async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Export kommer – just nu endast placeholder.", reply_markup=reply_kb())

# Register
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("th_up", cmd_th_up))
tg_app.add_handler(CommandHandler("th_down", cmd_th_down))
tg_app.add_handler(CommandHandler("save_model", cmd_save_model))
tg_app.add_handler(CommandHandler("load_model", cmd_load_model))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
tg_app.add_handler(CommandHandler("panic", cmd_panic))

# ================== FASTAPI ==================
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
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tfs": STATE.timeframes,
        "th_up": STATE.th_up,
        "th_down": STATE.th_down,
        "pnl_total": round(total_net(), 6),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
