# main_v36_ai_persist.py
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

# ========== ENV ==========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")          # 1m,3m,5m,15m,30m,1h
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "30"))
# mock-avgift per sida (t.ex. 0.1% = 0.001). Både vid entry och exit.
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))

KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour"}

# === AI PERSIST FIL ===
AI_STATE_PATH = os.getenv("AI_STATE_PATH", "ai_state.json")
AUTO_SAVE_SEC = int(os.getenv("AI_AUTOSAVE_SEC", "120"))

# ========== STATE ==========
@dataclass
class Position:
    side: str           # "LONG" eller "SHORT"
    entry: float
    stop: Optional[float]
    qty: float          # beräknas från POSITION_SIZE_USDT/entry

@dataclass
class SymState:
    pos: Optional[Position] = None
    realized_pnl_gross: float = 0.0
    realized_fees: float = 0.0
    trades: List[Tuple[str, float, float]] = field(default_factory=list)  # (label, gross, fee)

@dataclass
class EngineState:
    engine_on: bool = False
    timeframe: str = TIMEFRAME
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)
    # -------- AI (beständigt) --------
    # enkel bandit: score per symbol & action
    ai: Dict[str, Dict[str, float]] = field(default_factory=dict)   # {"BTC-USDT":{"LONG":0,"SHORT":0,"HOLD":0}}
    epsilon: float = 0.2   # utforskning
    # meta
    version: str = "v36-ai-persist-1"

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()
    STATE.ai[s] = {"LONG": 0.0, "SHORT": 0.0, "HOLD": 0.0}

# ========== PERSIST (SAVE/LOAD) ==========
def load_ai_state():
    try:
        with open(AI_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # endast AI-del + ev. hyperparametrar
        if "ai" in data:
            for s in STATE.symbols:
                STATE.ai[s] = data["ai"].get(s, {"LONG":0.0, "SHORT":0.0, "HOLD":0.0})
        if "epsilon" in data:
            STATE.epsilon = float(data["epsilon"])
        # frivilligt: återställ även ackumulerad pnl/fees (kan kommenteras bort)
        if "per_sym" in data:
            for s, payload in data["per_sym"].items():
                if s in STATE.per_sym:
                    STATE.per_sym[s].realized_pnl_gross = float(payload.get("realized_pnl_gross", 0.0))
                    STATE.per_sym[s].realized_fees = float(payload.get("realized_fees", 0.0))
        print(f"[AI] Laddade AI-state från {AI_STATE_PATH}")
    except FileNotFoundError:
        print(f"[AI] Ingen tidigare AI-state ({AI_STATE_PATH}), startar färskt.")
    except Exception as e:
        print(f"[AI] Misslyckades ladda AI-state: {e}")

def save_ai_state():
    try:
        data = {
            "version": STATE.version,
            "epsilon": STATE.epsilon,
            "ai": STATE.ai,
            # spara lite resultat för översikt
            "per_sym": {s: {
                "realized_pnl_gross": st.realized_pnl_gross,
                "realized_fees": st.realized_fees
            } for s, st in STATE.per_sym.items()}
        }
        with open(AI_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # print("[AI] Sparade AI-state.")
    except Exception as e:
        print(f"[AI] Misslyckades spara AI-state: {e}")

async def autosave_loop():
    while True:
        await asyncio.sleep(AUTO_SAVE_SEC)
        save_ai_state()

# ========== UTIL ==========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/pnl")],
        [KeyboardButton("/reset_pnl"), KeyboardButton("/panic")],
        [KeyboardButton("/save_ai"), KeyboardButton("/load_ai")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

async def get_klines(symbol: str, tf: str, limit: int = 3) -> List[Tuple[int,float,float,float,float]]:
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

# ========== AI / POLICY ==========
def choose_action(symbol: str, last_closed: Tuple[int,float,float,float,float]) -> str:
    """
    Enkel bandit + lite momentum:
    - med sannolikhet epsilon: slump
    - annars action med högst score
    - vi kräver ett minimalt filter: LONG om close>open, SHORT om close<open; annars HOLD
    """
    import random
    _, o, h, l, c = last_closed

    # utforskning
    if random.random() < STATE.epsilon:
        # välj bland tillåtna enligt enkel filter
        allowed = []
        if c > o: allowed.append("LONG")
        if c < o: allowed.append("SHORT")
        allowed.append("HOLD")
        return random.choice(allowed)

    scores = STATE.ai.get(symbol, {"LONG":0.0, "SHORT":0.0, "HOLD":0.0}).copy()
    # filtera orimliga: om röd candlestick => LONG score -inf, om grön => SHORT -inf
    if c <= o: scores["LONG"] = -1e9
    if c >= o: scores["SHORT"] = -1e9
    # välj max
    return max(scores, key=scores.get)

def record_result(symbol: str, action: str, pnl_net: float):
    """
    Uppdatera score: enkel inkrement/dekrement.
    """
    if action not in ("LONG", "SHORT"):
        return
    cur = STATE.ai.setdefault(symbol, {"LONG":0.0, "SHORT":0.0, "HOLD":0.0})
    cur[action] = float(cur.get(action, 0.0) + pnl_net)

# ========== ENGINE ==========
async def engine_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            if STATE.engine_on:
                for sym in STATE.symbols:
                    st = STATE.per_sym[sym]
                    kl = await get_klines(sym, STATE.timeframe, limit=3)
                    if len(kl) < 2:
                        continue
                    last_closed, current = kl[1], kl[0]

                    # har vi öppen position? hantera exit via enkel trailing på förra low/high
                    if st.pos:
                        _, o2, h2, l2, c2 = last_closed
                        if st.pos.side == "LONG":
                            # trailing stop = föregående low
                            new_stop = l2 if st.pos.stop is None else max(st.pos.stop, l2)
                            st.pos.stop = new_stop
                            # stop hit på close-basis
                            if c2 < st.pos.stop:
                                exit_px = c2
                                gross = (exit_px - st.pos.entry) * st.pos.qty
                                fees = (st.pos.entry + exit_px) * st.pos.qty * FEE_RATE
                                st.realized_pnl_gross += gross
                                st.realized_fees += fees
                                st.trades.append((f"{sym} LONG", gross, fees))
                                record_result(sym, "LONG", gross - fees)
                                st.pos = None
                        elif st.pos.side == "SHORT":
                            new_stop = h2 if st.pos.stop is None else min(st.pos.stop, h2)
                            st.pos.stop = new_stop
                            if c2 > st.pos.stop:
                                exit_px = c2
                                gross = (st.pos.entry - exit_px) * st.pos.qty
                                fees = (st.pos.entry + exit_px) * st.pos.qty * FEE_RATE
                                st.realized_pnl_gross += gross
                                st.realized_fees += fees
                                st.trades.append((f"{sym} SHORT", gross, fees))
                                record_result(sym, "SHORT", gross - fees)
                                st.pos = None

                    # ingen position? välj action och ev. öppna
                    if st.pos is None:
                        action = choose_action(sym, last_closed)
                        _, o, h, l, c = last_closed
                        if action in ("LONG", "SHORT"):
                            entry_px = c
                            qty = POSITION_SIZE_USDT / max(entry_px, 1e-9)
                            if action == "LONG":
                                st.pos = Position("LONG", entry=entry_px, stop=l, qty=qty)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🟢 ENTRY LONG {sym} @ {entry_px:.6f} | SL={l:.6f} | QTY={qty:.6f}"
                                    )
                            else:
                                st.pos = Position("SHORT", entry=entry_px, stop=h, qty=qty)
                                if STATE.chat_id:
                                    await app.bot.send_message(
                                        STATE.chat_id,
                                        f"🔴 ENTRY SHORT {sym} @ {entry_px:.6f} | SL={h:.6f} | QTY={qty:.6f}"
                                    )

            await asyncio.sleep(2)
        except Exception as e:
            if STATE.chat_id:
                try:
                    await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
                except:
                    pass
            await asyncio.sleep(5)

# ========== TELEGRAM ==========
tg_app = Application.builder().token(BOT_TOKEN).build()

def pnl_summary_lines() -> List[str]:
    total_gross = sum(s.realized_pnl_gross for s in STATE.per_sym.values())
    total_fees = sum(s.realized_fees for s in STATE.per_sym.values())
    total_net = total_gross - total_fees
    lines = [
        f"PnL total: Net {total_net:+.4f}  Gross {total_gross:+.4f}  Fees {total_fees:.4f} USDT"
    ]
    parts = []
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        parts.append(f"{s}: Net {ss.realized_pnl_gross-ss.realized_fees:+.4f} (Gross {ss.realized_pnl_gross:+.4f}, Fees {ss.realized_fees:.4f})")
    lines.append("PnL per symbol: " + ", ".join(parts))
    return lines

async def send_status(chat_id: int):
    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Timeframe: {STATE.timeframe}      # eller 3m",
        f"Symbols: {', '.join(STATE.symbols)}",
        f"Position size: {POSITION_SIZE_USDT:.1f} USDT | Fee per sida: {FEE_RATE*100:.4f}%",
    ]
    lines += pnl_summary_lines()
    # positioner
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(f"{s}: {st.pos.side} @ {st.pos.entry:.6f} | SL {st.pos.stop if st.pos.stop is not None else '-':} | QTY {st.pos.qty:.6f}")
    lines.append("Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"))
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    lines = pnl_summary_lines()
    # senaste affärer
    last = []
    for s in STATE.symbols:
        for lbl, g, f in STATE.per_sym[s].trades[-5:]:
            last.append((lbl, g, f))
    if last:
        lines.append("\nSenaste affärer:")
        for lbl, g, f in last[-10:]:
            lines.append(f"  - {lbl}: Net {g-f:+.4f}  (Gross {g:+.4f}, Fees {f:.4f})")
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! AI-bot (persist) redo ✅", reply_markup=reply_kb())
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
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframe satt till: {STATE.timeframe}", reply_markup=reply_kb())

async def cmd_panic(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    closed = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            exit_px = st.pos.stop if st.pos.stop is not None else st.pos.entry
            if st.pos.side == "LONG":
                gross = (exit_px - st.pos.entry) * st.pos.qty
                fees = (st.pos.entry + exit_px) * st.pos.qty * FEE_RATE
                record_result(s, "LONG", gross - fees)
            else:
                gross = (st.pos.entry - exit_px) * st.pos.qty
                fees = (st.pos.entry + exit_px) * st.pos.qty * FEE_RATE
                record_result(s, "SHORT", gross - fees)
            st.realized_pnl_gross += gross
            st.realized_fees += fees
            st.trades.append((f"{s} {st.pos.side} (panic)", gross, fees))
            closed.append(f"{s} net {gross-fees:+.4f}")
            st.pos = None
    msg = " | ".join(closed) if closed else "Inga positioner öppna."
    await tg_app.bot.send_message(STATE.chat_id, f"Panic close: {msg}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_reset_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    for s in STATE.symbols:
        STATE.per_sym[s].realized_pnl_gross = 0.0
        STATE.per_sym[s].realized_fees = 0.0
        STATE.per_sym[s].trades.clear()
    await tg_app.bot.send_message(STATE.chat_id, "PnL återställd.", reply_markup=reply_kb())

async def cmd_save_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    save_ai_state()
    await tg_app.bot.send_message(STATE.chat_id, f"AI-state sparat till {AI_STATE_PATH}", reply_markup=reply_kb())

async def cmd_load_ai(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    load_ai_state()
    await tg_app.bot.send_message(STATE.chat_id, f"AI-state laddat från {AI_STATE_PATH}", reply_markup=reply_kb())

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("panic", cmd_panic))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
tg_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
tg_app.add_handler(CommandHandler("load_ai", cmd_load_ai))

# ========== FASTAPI WEBHOOK ==========
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    # Ladda AI från disk
    load_ai_state()
    # Telegram
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())
    # Engine + autosave
    asyncio.create_task(engine_loop(tg_app))
    asyncio.create_task(autosave_loop())

@app.on_event("shutdown")
async def on_shutdown():
    save_ai_state()
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    total_gross = sum(s.realized_pnl_gross for s in STATE.per_sym.values())
    total_fees = sum(s.realized_fees for s in STATE.per_sym.values())
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tf": STATE.timeframe,
        "pnl_total_gross": round(total_gross, 6),
        "pnl_total_fees": round(total_fees, 6),
        "ai_version": STATE.version,
        "epsilon": STATE.epsilon,
        "ai_state_path": AI_STATE_PATH,
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
