# main_v36_ai_k4.py
import os
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import csv
import pathlib

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler

# ========== ENV ==========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")

SYMBOLS = (os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
           .replace(" ", "")).split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")          # 1m,3m,5m,15m,30m,1h
ENTRYMODE = os.getenv("ENTRY_MODE", "close")      # close|tick
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "100"))  # mock-positionstorlek

# Handelsavgift (per sida) i basis points. 10 = 0.10% per köp och 0.10% per sälj.
FEE_BPS = float(os.getenv("FEE_BPS", "10"))
# Tidszon för daglig summering (K4/Skatteverket – svensk tid som default)
TIMEZONE = os.getenv("TIMEZONE", "Europe/Stockholm")

# Loggmappar/filnamn
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = DATA_DIR / "trade_log.csv"
DAILY_PNL = DATA_DIR / "daily_pnl.csv"

# KuCoin REST (candles)
KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
TF_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "1h": "1hour",
}

# Säkerställ CSV-huvuden finns
def _ensure_csv_headers():
    if not TRADE_LOG.exists():
        with TRADE_LOG.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ts_entry_utc", "ts_exit_utc", "date_local", "symbol", "side",
                "entry_price", "exit_price", "qty",
                "fee_entry_usdt", "fee_exit_usdt",
                "gross_pnl_usdt", "net_pnl_usdt", "reason"
            ])
    if not DAILY_PNL.exists():
        with DAILY_PNL.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date_local", "total_net_pnl_usdt", "trades"])

_ensure_csv_headers()

# ========== STATE ==========
@dataclass
class Position:
    side: str           # "LONG"
    entry: float
    stop: float
    qty: float          # beräknas från POSITION_SIZE_USDT/entry
    ts_entry: datetime

@dataclass
class SymState:
    # ORB-fält finns kvar om du använder din gamla strategi i AI:n – men de påverkar inte loggning/K4
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_candle_time: Optional[int] = None

    pos: Optional[Position] = None
    realized_pnl: float = 0.0      # USDT (netto)
    trades: List[Tuple[str, float]] = field(default_factory=list)  # ("symbol LONG", pnl)

@dataclass
class EngineState:
    engine_on: bool = True                 # starta på ON (ändra om du vill)
    entry_mode: str = ENTRYMODE
    timeframe: str = TIMEFRAME
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)
    chat_id: Optional[int] = None
    per_sym: Dict[str, SymState] = field(default_factory=dict)

STATE = EngineState()
for s in STATE.symbols:
    STATE.per_sym[s] = SymState()

# ========== UTIL ==========
def reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entrymode"), KeyboardButton("/timeframe")],
        [KeyboardButton("/pnl"), KeyboardButton("/export_csv")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def _fee_usdt(price: float, qty: float) -> float:
    """Avgift i USDT för en sida (köp eller sälj)."""
    return (FEE_BPS / 10_000.0) * price * qty

def _local_date(d: datetime) -> str:
    tz = ZoneInfo(TIMEZONE)
    return d.astimezone(tz).date().isoformat()

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

# ========== CSV LOGGNING ==========
def append_trade_csv(symbol: str, side: str,
                     entry_price: float, exit_price: float, qty: float,
                     fee_entry: float, fee_exit: float,
                     ts_entry: datetime, ts_exit: datetime,
                     reason: str):
    gross = (exit_price - entry_price) * qty
    net = gross - fee_entry - fee_exit
    date_local = _local_date(ts_exit)

    with TRADE_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            ts_entry.astimezone(timezone.utc).isoformat(),
            ts_exit.astimezone(timezone.utc).isoformat(),
            date_local,
            symbol, side, f"{entry_price:.10f}", f"{exit_price:.10f}", f"{qty:.8f}",
            f"{fee_entry:.6f}", f"{fee_exit:.6f}",
            f"{gross:.6f}", f"{net:.6f}", reason
        ])

    # Uppdatera daglig summering
    recompute_daily(date_local)

def recompute_daily(date_local: str):
    """Summera alla trades för date_local och skriv/uppdatera daily_pnl.csv."""
    # Läs alla trades för dagen
    total = 0.0
    trades = 0
    if TRADE_LOG.exists():
        with TRADE_LOG.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("date_local") == date_local:
                    total += float(row.get("net_pnl_usdt", "0") or 0)
                    trades += 1

    # Läs in befintlig daily för att uppdatera raden
    rows = []
    found = False
    if DAILY_PNL.exists():
        with DAILY_PNL.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if row["date_local"] == date_local:
                    rows.append({"date_local": date_local,
                                 "total_net_pnl_usdt": f"{total:.6f}",
                                 "trades": str(trades)})
                    found = True
                else:
                    rows.append(row)
    if not found:
        rows.append({"date_local": date_local,
                     "total_net_pnl_usdt": f"{total:.6f}",
                     "trades": str(trades)})

    # skriv tillbaka
    with DAILY_PNL.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date_local", "total_net_pnl_usdt", "trades"])
        w.writeheader()
        w.writerows(rows)

# ========== ENKEL AI/EXEMPEL-LOGIK ==========
# Obs: Detta är en enkel placeholder som tar trades för att visa loggningen.
# Du kan byta ut beslutsdelen mot din AI/strategi – loggningen triggas i exit.

def is_green(o,h,l,c): return c > o
def is_red(o,h,l,c):   return c < o

def next_candle_stop(cur_stop: float, prev_closed: Tuple[int,float,float,float,float]) -> float:
    _, o,h,l,c = prev_closed
    return max(cur_stop, l)

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
                    _, o1,h1,l1,c1 = last_closed
                    _, o0,h0,l0,c0 = current

                    # Enkel entry: om ingen position, gå LONG när senaste stängda var grön
                    if st.pos is None and is_green(o1,h1,l1,c1):
                        entry_px = c1 if STATE.entry_mode == "close" else max(c1, o0)
                        qty = (POSITION_SIZE_USDT / entry_px) if entry_px > 0 else 0.0
                        st.pos = Position("LONG", entry=entry_px, stop=l1, qty=qty, ts_entry=datetime.now(timezone.utc))
                        if STATE.chat_id:
                            fee_est = _fee_usdt(entry_px, qty)
                            await app.bot.send_message(
                                STATE.chat_id,
                                f"🟢 ENTRY {sym} LONG @ {entry_px:.4f} | "
                                f"SL {st.pos.stop:.4f} | QTY {qty:.6f} | Avgift~ {fee_est:.4f} USDT",
                                reply_markup=reply_kb()
                            )

                    # Trailing stop mot föregående stängda candlens low
                    if st.pos:
                        new_sl = next_candle_stop(st.pos.stop, last_closed)
                        if new_sl > st.pos.stop:
                            st.pos.stop = new_sl
                            if STATE.chat_id:
                                await app.bot.send_message(STATE.chat_id,
                                    f"🔧 SL flyttad {sym} -> {st.pos.stop:.4f}",
                                    reply_markup=reply_kb()
                                )

                        # Exit med close- eller tick-logik
                        exit_now = False
                        exit_px = c1  # default
                        if STATE.entry_mode == "close" and c1 < st.pos.stop:
                            exit_now = True
                            exit_px = c1
                        elif STATE.entry_mode == "tick" and l0 <= st.pos.stop:
                            exit_now = True
                            exit_px = st.pos.stop

                        if exit_now:
                            # Fees
                            fee_in = _fee_usdt(st.pos.entry, st.pos.qty)
                            fee_out = _fee_usdt(exit_px, st.pos.qty)
                            gross = (exit_px - st.pos.entry) * st.pos.qty
                            net = gross - fee_in - fee_out
                            st.realized_pnl += net
                            st.trades.append((f"{sym} LONG", net))
                            if len(st.trades) > 50:
                                st.trades = st.trades[-50:]

                            ts_exit = datetime.now(timezone.utc)
                            append_trade_csv(
                                symbol=sym, side="LONG",
                                entry_price=st.pos.entry, exit_price=exit_px,
                                qty=st.pos.qty,
                                fee_entry=fee_in, fee_exit=fee_out,
                                ts_entry=st.pos.ts_entry, ts_exit=ts_exit,
                                reason="STOP" if exit_px <= st.pos.stop else "RULE"
                            )

                            if STATE.chat_id:
                                sign = "✅" if net >= 0 else "❌"
                                await app.bot.send_message(
                                    STATE.chat_id,
                                    f"🔴 EXIT {sym} @ {exit_px:.4f} | "
                                    f"PnL: {net:+.4f} USDT {sign}\n"
                                    f"(avgifter in:{fee_in:.4f} ut:{fee_out:.4f})",
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

# ========== TELEGRAM ==========
tg_app = Application.builder().token(BOT_TOKEN).build()

def _pnl_summary_lines() -> List[str]:
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    lines = [f"📈 PnL total (netto): {total_pnl:+.4f} USDT"]
    for s in STATE.symbols:
        ss = STATE.per_sym[s]
        lines.append(f"• {s}: {ss.realized_pnl:+.4f} USDT")
    return lines

async def send_status(chat_id: int):
    pos_lines = []
    for s, st in STATE.per_sym.items():
        if st.pos:
            pos_lines.append(f"{s} LONG @ {st.pos.entry:.4f} | SL {st.pos.stop:.4f} | QTY {st.pos.qty:.6f}")
    sym_pnls = ", ".join([f"{s}:{STATE.per_sym[s].realized_pnl:+.2f}" for s in STATE.symbols])

    lines = [
        f"Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"Entry mode: {STATE.entry_mode}",
        f"Timeframe: {STATE.timeframe}",
        f"Symbols: {', '.join(STATE.symbols)}",
    ]
    lines += _pnl_summary_lines()
    lines.append("Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"))
    await tg_app.bot.send_message(chat_id, "\n".join(lines), reply_markup=reply_kb())

async def send_pnl(chat_id: int):
    await tg_app.bot.send_message(chat_id, "\n".join(_pnl_summary_lines()), reply_markup=reply_kb())

async def send_csvs(chat_id: int):
    files = []
    if TRADE_LOG.exists():
        files.append(("trade_log.csv", TRADE_LOG))
    if DAILY_PNL.exists():
        files.append(("daily_pnl.csv", DAILY_PNL))
    if not files:
        await tg_app.bot.send_message(chat_id, "Inga CSV-filer ännu.", reply_markup=reply_kb())
        return
    # Skicka en i taget (Telegram API)
    for name, path in files:
        with path.open("rb") as f:
            await tg_app.bot.send_document(chat_id, InputFile(f, filename=name))
    await tg_app.bot.send_message(chat_id, "✅ Export klar.", reply_markup=reply_kb())

# Kommandon
async def cmd_start(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await tg_app.bot.send_message(STATE.chat_id, "Hej! AI-bot (K4-logg) redo ✅", reply_markup=reply_kb())
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

async def cmd_entrymode(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    STATE.entry_mode = "tick" if STATE.entry_mode == "close" else "close"
    await tg_app.bot.send_message(STATE.chat_id, f"Entry mode: {STATE.entry_mode}", reply_markup=reply_kb())

async def cmd_timeframe(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    order = ["1m","3m","5m","15m","30m","1h"]
    i = order.index(STATE.timeframe) if STATE.timeframe in order else 0
    STATE.timeframe = order[(i+1) % len(order)]
    await tg_app.bot.send_message(STATE.chat_id, f"Timeframe: {STATE.timeframe}", reply_markup=reply_kb())

async def cmd_pnl(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_pnl(STATE.chat_id)

async def cmd_export_csv(update: Update, _):
    STATE.chat_id = update.effective_chat.id
    await send_csvs(STATE.chat_id)

# Registrera handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("status", cmd_status))
tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
tg_app.add_handler(CommandHandler("entrymode", cmd_entrymode))
tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))

# ========== FASTAPI WEBHOOK ==========
app = FastAPI()

class TgUpdate(BaseModel):
    update_id: Optional[int] = None

@app.on_event("startup")
async def on_startup():
    if WEBHOOK_BASE:
        await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    # starta telegram + engine
    asyncio.create_task(tg_app.initialize())
    asyncio.create_task(tg_app.start())
    asyncio.create_task(engine_loop(tg_app))

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.get("/")
async def root():
    total_pnl = sum(s.realized_pnl for s in STATE.per_sym.values())
    return {
        "ok": True,
        "engine_on": STATE.engine_on,
        "tf": STATE.timeframe,
        "mode": STATE.entry_mode,
        "fee_bps": FEE_BPS,
        "pnl_total_net": round(total_pnl, 6),
        "trade_log": str(TRADE_LOG),
        "daily_pnl": str(DAILY_PNL),
    }

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
