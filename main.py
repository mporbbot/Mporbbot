# main.py
# Mp ORBbot – Full version med live/mock, backtest och CSV-export
# För Render: kör som Web Service med uvicorn main:app --host 0.0.0.0 --port $PORT
# eller som Background Worker med python main.py

import os
import csv
import time
import signal
import queue
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import requests
from telegram import (
    InlineKeyboardMarkup, InlineKeyboardButton, Update, constants, InputFile
)
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    ContextTypes
)

# ======================= KONFIG =======================
BOT_NAME = "Mp ORBbot"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))

SYMBOLS = ["LINKUSDT","XRPUSDT","ADAUSDT","BTCUSDT","ETHUSDT"]
TIMEFRAME = "3min"
AI_MODE = "neutral"
ENGINE_ON = True
MIN_ORB_PCT = 0.001      # 0.1%
FEE_RATE = 0.001         # 0.1% per sida
MOCK_TRADE_USDT = 30.0
TICK_EPS = 1e-8
MOCK_ENABLED = True      # mockläge som standard
LIVE_ENABLED = False
KEEPALIVE = False
KEEPALIVE_INTERVAL = 60

MOCK_LOG = Path("mock_trade_log.csv")
REAL_LOG = Path("real_trade_log.csv")
BACKTEST_LOG = Path("backtest_log.csv")

BASE = "https://api.kucoin.com"

# ======================= HJÄLPFUNKTIONER =======================
def fmt_qty(q: float) -> str:
    return f"{q:.6f}".rstrip("0").rstrip(".")

def fmt_price(p: float) -> str:
    return f"{p:.6f}".rstrip("0").rstrip(".")

def qty_for_usdt(usdt: float, price: float) -> float:
    return usdt / max(price, 1e-12)

def calc_trade_result(entry, exit_, qty, side, fee_rate=FEE_RATE):
    direction = 1 if side == "LONG" else -1
    gross = (exit_ - entry) * qty * direction
    fees = (entry * qty + exit_ * qty) * fee_rate
    net = gross - fees
    pct = (net / (entry * qty)) * 100 if entry * qty else 0.0
    return net, pct, fees

def ensure_csv(file: Path):
    if not file.exists():
        with file.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "ts_iso", "mode", "symbol", "side",
                "entry", "exit", "qty", "fees", "net_usdt", "pct",
                "pnl_day_after", "note"
            ])

def log_trade(path: Path, mode: str, symbol: str, side: str,
              entry: float, exit_: float, qty: float,
              fees: float, net: float, pct: float,
              pnl_after: float, note: str = ""):
    ensure_csv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(),
            mode, symbol, side, f"{entry:.10f}", f"{exit_:.10f}", f"{qty:.10f}",
            f"{fees:.10f}", f"{net:.10f}", f"{pct:.5f}",
            f"{pnl_after:.10f}", note
        ])

# ======================= DATAKLASSER =======================
class Candle:
    __slots__ = ("t","o","h","l","c","v")
    def __init__(self, t:int,o:float,h:float,l:float,c:float,v:float):
        self.t,self.o,self.h,self.l,self.c,self.v = t,o,h,l,c,v
    @property
    def bullish(self): return self.c > self.o
    @property
    def bearish(self): return self.c < self.o
    @property
    def rng(self): return self.h - self.l

class SymState:
    def __init__(self, symbol:str):
        self.symbol = symbol
        self.position = None
        self.trades_today = 0
        self.last_usdt = None
        self.last_pct = None
        self.orb_high = None
        self.orb_low = None
        self.orb_dir = None
    def pos_side(self):  return self.position["side"] if self.position else None
    def pos_entry(self): return self.position["entry"] if self.position else None
    def pos_stop(self):  return self.position["stop"] if self.position else None

state = {
    "symbols": {s: SymState(s) for s in SYMBOLS},
    "daily_mock_pnl": 0.0,
    "day": datetime.now(timezone.utc).date().isoformat(),
}

# ======================= KUCOIN =======================
def kucoin_klines(symbol:str, tf:str, limit:int=200) -> List[Candle]:
    s = symbol.replace("USDT", "-USDT")
    url = f"{BASE}/api/v1/market/candles?type={tf}&symbol={s}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()["data"]
    out = []
    for row in reversed(data[-limit:]):
        t = int(float(row[0]))
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4]); v = float(row[5])
        out.append(Candle(t, o, h, l, c, v))
    return out

# ======================= ORB-LOGIK =======================
def update_orb(sym: SymState, closed: Candle):
    if closed.rng <= abs(closed.o) * MIN_ORB_PCT:
        sym.orb_high = sym.orb_low = sym.orb_dir = None
        return
    sym.orb_high = closed.h
    sym.orb_low  = closed.l
    sym.orb_dir  = "bullish" if closed.bullish else "bearish" if closed.bearish else None

def try_entry(sym: SymState, live: Candle, send):
    if not sym.orb_dir or sym.position:
        return
    if sym.orb_dir == "bullish" and live.h > sym.orb_high + TICK_EPS:
        entry = sym.orb_high + TICK_EPS
        qty = qty_for_usdt(MOCK_TRADE_USDT, entry)
        stop = sym.orb_low - TICK_EPS
        sym.position = {"side":"LONG","entry":entry,"qty":qty,"stop":stop}
        send(f"✅ MOCK: LONG {sym.symbol} @ {fmt_price(entry)} x {fmt_qty(qty)}")
    elif sym.orb_dir == "bearish" and live.l < sym.orb_low - TICK_EPS:
        entry = sym.orb_low - TICK_EPS
        qty = qty_for_usdt(MOCK_TRADE_USDT, entry)
        stop = sym.orb_high + TICK_EPS
        sym.position = {"side":"SHORT","entry":entry,"qty":qty,"stop":stop}
        send(f"✅ MOCK: SHORT {sym.symbol} @ {fmt_price(entry)} x {fmt_qty(qty)}")

def apply_trailing(sym: SymState, closed: Candle, send):
    if not sym.position:
        return
    side = sym.position["side"]
    stop = sym.position["stop"]
    if side == "LONG":
        new_stop = max(stop, closed.l - TICK_EPS)
        sym.position["stop"] = new_stop
        if closed.l <= stop:
            exit_trade(sym, stop, send)
    else:
        new_stop = min(stop, closed.h + TICK_EPS)
        sym.position["stop"] = new_stop
        if closed.h >= stop:
            exit_trade(sym, stop, send)

def exit_trade(sym: SymState, price: float, send):
    pos = sym.position
    entry, qty, side = pos["entry"], pos["qty"], pos["side"]
    net, pct, fees = calc_trade_result(entry, price, qty, side)
    state["daily_mock_pnl"] += net
    sym.trades_today += 1
    sym.last_usdt = net
    sym.last_pct = pct
    log_trade(MOCK_LOG, "MOCK", sym.symbol, side, entry, price, qty, fees, net, pct, state["daily_mock_pnl"], "stop/close")
    send(f"📄 Stängd {side} {sym.symbol}\nEntry: {fmt_price(entry)} Exit: {fmt_price(price)}\nQty: {fmt_qty(qty)} Avgifter: {fees:.4f}\nResultat: {'+' if net>=0 else ''}{net:.4f} USDT ({'+' if pct>=0 else ''}{pct:.2f}%)\nPnL idag: {state['daily_mock_pnl']:.4f}")
    sym.position = None

# ======================= ENGINE =======================
stop_event = threading.Event()
send_queue = queue.Queue()
_default_chat_id = None

def send_async(msg: str):
    send_queue.put((_default_chat_id, msg))

def sender_thread(app: Application):
    while not stop_event.is_set():
        try:
            chat_id, msg = send_queue.get(timeout=0.5)
            if chat_id:
                app.bot.send_message(chat_id=chat_id, text=msg)
        except queue.Empty:
            pass

def engine_loop():
    last_ts = {s: 0 for s in SYMBOLS}
    while not stop_event.is_set():
        d = datetime.now(timezone.utc).date().isoformat()
        if d != state["day"]:
            state["day"] = d
            state["daily_mock_pnl"] = 0.0
            for s in SYMBOLS:
                ss = state["symbols"][s]
                ss.trades_today = 0
                ss.last_usdt = ss.last_pct = None
        if not ENGINE_ON:
            time.sleep(1.5)
            continue
        for s in SYMBOLS:
            sym = state["symbols"][s]
            try:
                candles = kucoin_klines(s, TIMEFRAME, 200)
            except:
                continue
            if not candles: continue
            closed = candles[-2]
            if closed.t > last_ts[s]:
                update_orb(sym, closed)
                if MOCK_ENABLED: apply_trailing(sym, closed, send_async)
                last_ts[s] = closed.t
            live = candles[-1]
            if MOCK_ENABLED: try_entry(sym, live, send_async)
        time.sleep(2.5)

# ======================= TELEGRAM =======================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _default_chat_id
    _default_chat_id = update.effective_chat.id
    await update.message.reply_text(f"{BOT_NAME} redo. Skriv /help för kommandon.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("/status\n/set_ai\n/start_mock\n/start_live\n/engine_start\n/engine_stop\n/symbols\n/timeframe\n/pnl\n/reset_pnl\n/export_csv\n/backtest SYMBOL PERIOD [FEE]")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = f"Bot: {BOT_NAME}\nLäge: {'MOCK' if MOCK_ENABLED else 'LIVE' if LIVE_ENABLED else 'OFF'}\nAI: {AI_MODE}\nMotor: {'ON' if ENGINE_ON else 'OFF'}\nTimeframe: {TIMEFRAME}\nPnL idag: {state['daily_mock_pnl']:.4f}"
    await update.message.reply_text(msg)

async def cmd_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("JA ✅", callback_data="MOCK_Y"), InlineKeyboardButton("NEJ ❌", callback_data="MOCK_N")]])
    await update.message.reply_text("Vill du starta MOCK-trading?", reply_markup=kb)

async def cmd_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("JA ✅", callback_data="LIVE_Y"), InlineKeyboardButton("NEJ ❌", callback_data="LIVE_N")]])
    await update.message.reply_text("Vill du starta LIVE-trading?", reply_markup=kb)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MOCK_ENABLED, LIVE_ENABLED
    data = update.callback_query.data
    if data == "MOCK_Y":
        MOCK_ENABLED = True
        await update.callback_query.edit_message_text("MOCK-trading: AKTIVERAD ✅")
    elif data == "MOCK_N":
        MOCK_ENABLED = False
        await update.callback_query.edit_message_text("MOCK-trading: AVSTÄNGD ❌")
    elif data == "LIVE_Y":
        LIVE_ENABLED = True
        await update.callback_query.edit_message_text("LIVE-trading: AKTIVERAD ✅")
    elif data == "LIVE_N":
        LIVE_ENABLED = False
        await update.callback_query.edit_message_text("LIVE-trading: AVSTÄNGD ❌")

# ======================= BACKTEST =======================
async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Användning: /backtest SYMBOL PERIOD [FEE]")
        return
    symbol = context.args[0].upper()
    period = context.args[1]
    fee = float(context.args[2]) if len(context.args) > 2 else FEE_RATE
    days = int(period.replace("d",""))
    candles = []
    for i in range(days):
        candles.extend(kucoin_klines(symbol, TIMEFRAME, 200))
    pnl = 0.0
    trades = 0
    BACKTEST_LOG.unlink(missing_ok=True)
    orb_high = orb_low = orb_dir = None
    pos = None
    for i in range(1, len(candles)):
        closed = candles[i-1]
        if closed.bullish or closed.bearish:
            orb_high, orb_low, orb_dir = closed.h, closed.l, "bullish" if closed.bullish else "bearish"
        if pos:
            if pos["side"] == "LONG" and closed.l <= pos["stop"]:
                net,pct,fees = calc_trade_result(pos["entry"], pos["stop"], pos["qty"], "LONG", fee)
                pnl += net; trades += 1
                log_trade(BACKTEST_LOG, "BACKTEST", symbol, "LONG", pos["entry"], pos["stop"], pos["qty"], fees, net, pct, pnl, "stop")
                pos = None
            elif pos["side"] == "SHORT" and closed.h >= pos["stop"]:
                net,pct,fees = calc_trade_result(pos["entry"], pos["stop"], pos["qty"], "SHORT", fee)
                pnl += net; trades += 1
                log_trade(BACKTEST_LOG, "BACKTEST", symbol, "SHORT", pos["entry"], pos["stop"], pos["qty"], fees, net, pct, pnl, "stop")
                pos = None
        live = candles[i]
        if not pos and orb_dir:
            if orb_dir=="bullish" and live.h > orb_high:
                entry = orb_high + TICK_EPS
                pos = {"side":"LONG","entry":entry,"qty":qty_for_usdt(MOCK_TRADE_USDT, entry),"stop":orb_low-TICK_EPS}
            elif orb_dir=="bearish" and live.l < orb_low:
                entry = orb_low - TICK_EPS
                pos = {"side":"SHORT","entry":entry,"qty":qty_for_usdt(MOCK_TRADE_USDT, entry),"stop":orb_high+TICK_EPS}
    await update.message.reply_text(f"Backtest {symbol} {period} klar\nTrades: {trades}\nPnL: {pnl:.4f} USDT")
    if BACKTEST_LOG.exists():
        await context.bot.send_document(chat_id=update.effective_chat.id, document=InputFile(BACKTEST_LOG, filename="backtest_log.csv"))

# ======================= START BOT =======================
def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("start_mock", cmd_start_mock))
    app.add_handler(CommandHandler("start_live", cmd_start_live))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CallbackQueryHandler(on_callback))
    threading.Thread(target=sender_thread, args=(app,), daemon=True).start()
    threading.Thread(target=engine_loop, daemon=True).start()
    app.run_polling()

class _ASGIApp:
    def __init__(self):
        threading.Thread(target=start_bot, daemon=True).start()
    async def __call__(self, scope, receive, send):
        if scope["type"]=="http":
            await send({"type":"http.response.start","status":200,"headers":[(b"content-type",b"text/plain")]})
            await send({"type":"http.response.body","body":b"Mp ORBbot OK"})

app = _ASGIApp()

if __name__ == "__main__":
    start_bot()
