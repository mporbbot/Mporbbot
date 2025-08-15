# main.py – Mp ORBbot (Background Worker)
# Kör på Render med startkommandot:  python main.py

import os
import csv
import time
import queue
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import requests
from telegram import (
    InlineKeyboardMarkup, InlineKeyboardButton, Update, InputFile, constants
)
from telegram.ext import (
    ApplicationBuilder, Application, CommandHandler, CallbackQueryHandler, ContextTypes
)

# ============== KONFIG ==============
BOT_NAME = "Mp ORBbot"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")

SYMBOLS = [s.strip().upper() for s in os.getenv(
    "SYMBOLS", "LINKUSDT,XRPUSDT,ADAUSDT,BTCUSDT,ETHUSDT"
).split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "3min")  # 1min|3min|5min|15min
AI_MODE = os.getenv("AI_MODE", "neutral")   # neutral/aggressiv/försiktig (kan användas senare)
ENGINE_ON = os.getenv("ENGINE_ON", "true").lower() == "true"

MIN_ORB_PCT = float(os.getenv("MIN_ORB_PCT", "0.001"))  # 0.100%
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))        # 0.1%/sida
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30.0"))
TICK_EPS = float(os.getenv("TICK_EPS", "1e-8"))

MOCK_ENABLED = os.getenv("MOCK_ENABLED", "true").lower() == "true"
LIVE_ENABLED = os.getenv("LIVE_ENABLED", "false").lower() == "true"

# Loggar
MOCK_LOG = Path("mock_trade_log.csv")
REAL_LOG = Path("real_trade_log.csv")
BACKTEST_LOG = Path("backtest_log.csv")

BASE = "https://api.kucoin.com"

# ============== Hjälp ==============
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

# ============== Datastrukturer ==============
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
        self.position = None  # {side, entry, qty, stop}
        self.trades_today = 0
        self.last_usdt = None
        self.last_pct = None
        self.orb_high = None
        self.orb_low = None
        self.orb_dir = None   # bullish/bearish/None

state = {
    "symbols": {s: SymState(s) for s in SYMBOLS},
    "daily_mock_pnl": 0.0,
    "day": datetime.now(timezone.utc).date().isoformat(),
}

# ============== KuCoin ==============
def kucoin_klines(symbol:str, tf:str, limit:int=200) -> List[Candle]:
    s = symbol.replace("USDT", "-USDT")
    url = f"{BASE}/api/v1/market/candles?type={tf}&symbol={s}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()["data"]
    out = []
    for row in reversed(data[-limit:]):
        t = int(float(row[0])); o=float(row[1]); c=float(row[2]); h=float(row[3]); l=float(row[4]); v=float(row[5])
        out.append(Candle(t,o,h,l,c,v))
    return out

# ============== ORB-logik ==============
def update_orb(sym: SymState, closed: Candle):
    if closed.rng <= abs(closed.o) * MIN_ORB_PCT:
        sym.orb_high = sym.orb_low = sym.orb_dir = None
        return
    sym.orb_high = closed.h
    sym.orb_low  = closed.l
    sym.orb_dir  = "bullish" if closed.bullish else ("bearish" if closed.bearish else None)

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
    send(
        f"📄 Stängd {side} {sym.symbol}\n"
        f"Entry: {fmt_price(entry)}  Exit: {fmt_price(price)}\n"
        f"Qty: {fmt_qty(qty)}  Avgifter: {fees:.4f}\n"
        f"Resultat (MOCK): {'+' if net>=0 else ''}{net:.4f} USDT ({'+' if pct>=0 else ''}{pct:.2f}%)\n"
        f"PnL idag → MOCK: {state['daily_mock_pnl']:.4f}"
    )
    sym.position = None

# ============== Engine-trådar ==============
stop_event = threading.Event()
send_queue = queue.Queue()
_default_chat_id = None
_app: Application | None = None

def send_async(msg: str):
    send_queue.put((_default_chat_id, msg))

def sender_thread():
    # kör efter att Application skapats
    while not stop_event.is_set():
        try:
            chat_id, msg = send_queue.get(timeout=0.5)
            if chat_id and _app:
                _app.bot.send_message(chat_id=chat_id, text=msg)
        except queue.Empty:
            pass
        except Exception as e:
            print("Send error:", e)

def engine_loop():
    last_ts = {s: 0 for s in SYMBOLS}
    while not stop_event.is_set():
        # dagsreset
        d = datetime.now(timezone.utc).date().isoformat()
        if d != state["day"]:
            state["day"] = d
            state["daily_mock_pnl"] = 0.0
            for s in SYMBOLS:
                ss = state["symbols"][s]
                ss.trades_today = 0
                ss.last_usdt = ss.last_pct = None

        if not ENGINE_ON:
            time.sleep(1.5); continue

        for s in SYMBOLS:
            sym = state["symbols"][s]
            try:
                candles = kucoin_klines(s, TIMEFRAME, 200)
            except Exception as e:
                print("Kline error", s, e); time.sleep(0.3); continue
            if not candles: continue

            closed = candles[-2]              # senaste STÄNGDA
            if closed.t > last_ts[s]:
                update_orb(sym, closed)
                if MOCK_ENABLED:
                    apply_trailing(sym, closed, send_async)
                last_ts[s] = closed.t

            live = candles[-1]                # pågående ljus
            if MOCK_ENABLED:
                try_entry(sym, live, send_async)

        time.sleep(2.5)

# ============== Telegram-kommandon ==============
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _default_chat_id
    _default_chat_id = update.effective_chat.id
    await update.message.reply_text(f"{BOT_NAME} redo. Skriv /help för kommandon.")

def _help_text():
    return (
        "/status – visa status\n"
        "/set_ai <neutral|aggressiv|försiktig>\n"
        "/start_mock – starta MOCK (JA/NEJ)\n"
        "/start_live – starta LIVE (JA/NEJ)\n"
        "/symbols BTCUSDT,ETHUSDT,… – byt lista\n"
        "/timeframe 1min|3min|5min|15min – byt tidsram\n"
        "/pnl – visa dagens PnL\n"
        "/reset_pnl – nollställ dagens PnL\n"
        "/export_csv – skicka loggar\n"
        "/backtest SYMBOL PERIOD [FEE]\n"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_help_text())

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    header = (
        f"Bot: {BOT_NAME}\n"
        f"Läge: {'MOCK' if MOCK_ENABLED else ('LIVE' if LIVE_ENABLED else 'INAKTIV')}\n"
        f"AI: {AI_MODE} (min ORB {MIN_ORB_PCT*100:.4f}%)\n"
        f"Motor: {'ON' if ENGINE_ON else 'OFF'}\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"PnL idag → MOCK: {state['daily_mock_pnl']:.4f}"
    )
    await update.message.reply_text(header)
    for s in SYMBOLS:
        ss = state["symbols"][s]
        orb = f"ORB=Y ({ss.orb_dir})" if ss.orb_dir else "ORB=N"
        last_line = ""
        if ss.last_usdt is not None:
            su = ss.last_usdt; sp = ss.last_pct or 0.0
            last_line = f"\nSenaste affär: {'+' if su>=0 else ''}{su:.4f} USDT ({'+' if sp>=0 else ''}{sp:.2f}%)"
        msg = (
            f"{s}: pos={'Y' if ss.position else 'N'} side={ss.position['side'] if ss.position else None}\n"
            f"entry={fmt_price(ss.position['entry']) if ss.position else 'None'} "
            f"stop={fmt_price(ss.position['stop']) if ss.position else 'None'} "
            f"trades={ss.trades_today} {orb}{last_line}"
        )
        await update.message.reply_text(msg)

async def cmd_set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_MODE
    if not context.args:
        await update.message.reply_text("Användning: /set_ai <neutral|aggressiv|försiktig>")
        return
    val = context.args[0].lower()
    if val not in ("neutral", "aggressiv", "försiktig"):
        await update.message.reply_text("Ogiltigt läge.")
        return
    AI_MODE = val
    await update.message.reply_text(f"AI-läge satt till: {AI_MODE}")

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TIMEFRAME
    if not context.args:
        await update.message.reply_text(f"Aktuellt timeframe: {TIMEFRAME}")
        return
    tf = context.args[0].lower()
    if tf not in {"1min","3min","5min","15min"}:
        await update.message.reply_text("Ogiltigt timeframe.")
        return
    TIMEFRAME = tf
    await update.message.reply_text(f"Timeframe satt till: {TIMEFRAME}")

async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SYMBOLS, state
    if not context.args:
        await update.message.reply_text(f"Aktuella: {', '.join(SYMBOLS)}\nAnvändning: /symbols BTCUSDT,ETHUSDT,...")
        return
    lst = "".join(context.args).upper().replace(" ", "")
    parts = [p for p in lst.split(",") if p.endswith("USDT")]
    if not parts:
        await update.message.reply_text("Inga giltiga symboler.")
        return
    SYMBOLS = parts
    state["symbols"] = {s: SymState(s) for s in SYMBOLS}
    await update.message.reply_text(f"Symbols uppdaterade: {', '.join(SYMBOLS)}")

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Dagens PnL\nMOCK: {state['daily_mock_pnl']:.4f} USDT\nLIVE: 0.0000 USDT")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state["daily_mock_pnl"] = 0.0
    await update.message.reply_text("Dagens PnL nollställd.")

# JA/NEJ-knappar
async def cmd_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("JA ✅","CONFIRM_MOCK_Y"),
                                InlineKeyboardButton("NEJ ❌","CONFIRM_MOCK_N")]])
    await update.message.reply_text("Vill du starta MOCK-trading?", reply_markup=kb)

async def cmd_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("JA ✅","CONFIRM_LIVE_Y"),
                                InlineKeyboardButton("NEJ ❌","CONFIRM_LIVE_N")]])
    await update.message.reply_text("Vill du starta LIVE-trading?", reply_markup=kb)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MOCK_ENABLED, LIVE_ENABLED
    data = update.callback_query.data
    if data == "CONFIRM_MOCK_Y":
        MOCK_ENABLED = True
        await update.callback_query.edit_message_text("MOCK-trading: AKTIVERAD ✅")
    elif data == "CONFIRM_MOCK_N":
        MOCK_ENABLED = False
        await update.callback_query.edit_message_text("MOCK-trading: AVSTÄNGD ❌")
    elif data == "CONFIRM_LIVE_Y":
        LIVE_ENABLED = True
        await update.callback_query.edit_message_text("LIVE-trading: AKTIVERAD ✅ (ej implementerad)")
    elif data == "CONFIRM_LIVE_N":
        LIVE_ENABLED = False
        await update.callback_query.edit_message_text("LIVE-trading: AVSTÄNGD ❌")

# Export
async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sent = False
    for p in (MOCK_LOG, REAL_LOG, BACKTEST_LOG):
        if p.exists():
            await context.bot.send_document(chat_id=update.effective_chat.id, document=InputFile(p, filename=p.name))
            sent = True
    if not sent:
        await update.message.reply_text("Inga loggar ännu.")

# Backtest (enkel, samma ORB/entry/stop/trailing som live/mock – med avgift)
async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Användning: /backtest SYMBOL PERIOD[D] [FEE]\nEx: /backtest BTCUSDT 3D 0.001")
        return
    symbol = context.args[0].upper()
    days = int(context.args[1].rstrip("dD"))
    fee = float(context.args[2]) if len(context.args) > 2 else FEE_RATE

    BACKTEST_LOG.unlink(missing_ok=True)
    ensure_csv(BACKTEST_LOG)

    # Hämta tillräckligt med ljus (gör simpla multipla hämtningar)
    candles = []
    fetches = max(1, (days*24*60)//180)  # ungefär 200 candles per fetch på 3min ~ 10h
    for _ in range(fetches):
        candles.extend(kucoin_klines(symbol, TIMEFRAME, 200))
        time.sleep(0.2)
    if len(candles) < 3:
        await update.message.reply_text("Fick för lite data från KuCoin.")
        return

    pnl = 0.0
    trades = 0
    orb_high = orb_low = orb_dir = None
    pos = None

    def _trail_and_exit(_pos, c: Candle):
        nonlocal pnl, trades, pos
        if not _pos: return False
        if _pos["side"] == "LONG":
            _pos["stop"] = max(_pos["stop"], c.l - TICK_EPS)
            if c.l <= _pos["stop"]:
                net,pct,fees = calc_trade_result(_pos["entry"], _pos["stop"], _pos["qty"], "LONG", fee)
                pnl += net; trades += 1
                log_trade(BACKTEST_LOG, "BACKTEST", symbol, "LONG", _pos["entry"], _pos["stop"], _pos["qty"], fees, net, pct, pnl, "stop")
                pos = None
                return True
        else:
            _pos["stop"] = min(_pos["stop"], c.h + TICK_EPS)
            if c.h >= _pos["stop"]:
                net,pct,fees = calc_trade_result(_pos["entry"], _pos["stop"], _pos["qty"], "SHORT", fee)
                pnl += net; trades += 1
                log_trade(BACKTEST_LOG, "BACKTEST", symbol, "SHORT", _pos["entry"], _pos["stop"], _pos["qty"], fees, net, pct, pnl, "stop")
                pos = None
                return True
        return False

    for i in range(1, len(candles)):
        closed = candles[i-1]
        # ny ORB per grön/röd
        if closed.bullish or closed.bearish:
            orb_high, orb_low = closed.h, closed.l
            orb_dir = "bullish" if closed.bullish else "bearish"

        if pos:
            if _trail_and_exit(pos, closed):
                pass  # redan stängd
        live = candles[i]
        if (not pos) and orb_dir:
            if orb_dir=="bullish" and live.h > orb_high + TICK_EPS:
                entry = orb_high + TICK_EPS
                pos = {"side":"LONG","entry":entry,"qty":qty_for_usdt(MOCK_TRADE_USDT, entry),"stop":orb_low - TICK_EPS}
            elif orb_dir=="bearish" and live.l < orb_low - TICK_EPS:
                entry = orb_low - TICK_EPS
                pos = {"side":"SHORT","entry":entry,"qty":qty_for_usdt(MOCK_TRADE_USDT, entry),"stop":orb_high + TICK_EPS}

    await update.message.reply_text(f"Backtest {symbol} ~{days}d klar\nTrades: {trades}\nPnL (avgift {fee:.4%} per sida): {pnl:.4f} USDT")
    if BACKTEST_LOG.exists():
        await context.bot.send_document(chat_id=update.effective_chat.id, document=InputFile(BACKTEST_LOG, filename="backtest_log.csv"))

# ============== Start bot ==============
def start_bot():
    global _app
    _app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    _app.add_handler(CommandHandler("start", cmd_start))
    _app.add_handler(CommandHandler("help", cmd_help))
    _app.add_handler(CommandHandler("status", cmd_status))
    _app.add_handler(CommandHandler("set_ai", cmd_set_ai))
    _app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    _app.add_handler(CommandHandler("symbols", cmd_symbols))
    _app.add_handler(CommandHandler("pnl", cmd_pnl))
    _app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    _app.add_handler(CommandHandler("start_mock", cmd_start_mock))
    _app.add_handler(CommandHandler("start_live", cmd_start_live))
    _app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    _app.add_handler(CommandHandler("backtest", cmd_backtest))
    _app.add_handler(CallbackQueryHandler(on_callback))

    # Starta trådar EFTER att Application finns
    threading.Thread(target=sender_thread, daemon=True).start()
    threading.Thread(target=engine_loop, daemon=True).start()

    _app.run_polling(close_loop=False)

if __name__ == "__main__":
    start_bot()
