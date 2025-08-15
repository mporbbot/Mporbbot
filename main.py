# main.py – Mp ORBbot (Web Service / Free plan) – Spot LONG only, ORB per grön/röd, stop = senaste stängda low
# Start (Render Web Service): uvicorn main:app --host 0.0.0.0 --port $PORT

import os, csv, time, queue, threading
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

import requests
from fastapi import FastAPI
from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

# ============== KONFIG ==============
BOT_NAME = "Mp ORBbot"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

SYMBOLS = [s.strip().upper() for s in os.getenv(
    "SYMBOLS", "LINKUSDT,XRPUSDT,ADAUSDT,BTCUSDT,ETHUSDT"
).split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "3min")
ENGINE_ON = True

MIN_ORB_PCT = float(os.getenv("MIN_ORB_PCT", "0.001"))  # min range-filter
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))        # 0.1% per sida
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
TICK_EPS = float(os.getenv("TICK_EPS", "1e-8"))

MOCK_ENABLED = True
LIVE_ENABLED = False

# Live Spot (LONG only)
KU_API = os.getenv("KUCOIN_API_KEY")
KU_SEC = os.getenv("KUCOIN_API_SECRET")
KU_PWD = os.getenv("KUCOIN_API_PASSPHRASE")
LIVE_USDT_PER_TRADE = float(os.getenv("LIVE_USDT_PER_TRADE", "30"))
LIVE_ALLOW_SHORTS = False  # Spot stöder ej riktig short

# Keepalive
SELF_URL = os.getenv("SELF_URL", "")
_keepalive_on = False

# Loggar
MOCK_LOG = Path("mock_trade_log.csv")
REAL_LOG = Path("real_trade_log.csv")
BACKTEST_LOG = Path("backtest_log.csv")

BASE = "https://api.kucoin.com"

# ============== Hjälp ==============
def fmt_qty(q: float) -> str: return f"{q:.6f}".rstrip("0").rstrip(".")
def fmt_price(p: float) -> str: return f"{p:.6f}".rstrip("0").rstrip(".")
def qty_for_usdt(usdt: float, price: float) -> float: return usdt / max(price, 1e-12)

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
            datetime.now(timezone.utc).isoformat(), mode, symbol, side,
            f"{entry:.10f}", f"{exit_:.10f}", f"{qty:.10f}",
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
        self.position = None  # {side, entry, qty, stop, live}
        self.trades_today = 0
        self.last_usdt = None
        self.last_pct = None
        self.orb_high = None
        self.orb_low = None
        self.orb_dir = None   # bullish/bearish/None

state = {
    "symbols": {s: SymState(s) for s in SYMBOLS},
    "daily_pnl": 0.0,
    "day": datetime.now(timezone.utc).date().isoformat(),
}

# ============== KuCoin (klines + live orders) ==============
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

# ---- Live Spot via kucoin-python (init lazy för att inte krascha utan nycklar) ----
_kc = None
def kc():
    from kucoin.client import Client as KucoinClient
    global _kc
    if _kc is None:
        if not (KU_API and KU_SEC and KU_PWD):
            raise RuntimeError("Saknar KuCoin API-nycklar (KUCOIN_API_KEY/SECRET/PASSPHRASE).")
        _kc = KucoinClient(KU_API, KU_SEC, KU_PWD)
    return _kc

def market_buy_spot(symbol: str, usdt_funds: float) -> tuple[float,float]:
    """Returnerar (fill_price, qty) för köp. Fallback till entry om tomt."""
    s = symbol.replace("USDT","-USDT")
    o = kc().create_market_order(s, "buy", funds=str(usdt_funds))
    info = kc().get_order(o["orderId"])
    fill_price = float(info.get("dealPrice") or 0.0)
    qty = float(info.get("dealSize") or 0.0)
    return fill_price, qty

def market_sell_spot(symbol: str, base_size: float) -> float:
    """Returnerar fill_price för sälj (fallback: 0 om okänt direkt)."""
    s = symbol.replace("USDT","-USDT")
    o = kc().create_market_order(s, "sell", size=str(base_size))
    info = kc().get_order(o["orderId"])
    return float(info.get("dealPrice") or 0.0)

# ============== ORB-logik ==============
def update_orb(sym: SymState, closed: Candle):
    """Ny ORB på varje stängd grön/röd candle. Doji ignoreras. Range-filter med MIN_ORB_PCT."""
    if not (closed.bullish or closed.bearish):
        sym.orb_high = sym.orb_low = sym.orb_dir = None
        return
    if closed.rng <= abs(closed.o) * MIN_ORB_PCT:
        sym.orb_high = sym.orb_low = sym.orb_dir = None
        return
    sym.orb_high = closed.h
    sym.orb_low  = closed.l
    sym.orb_dir  = "bullish" if closed.bullish else "bearish"

def try_entry(sym: SymState, live: Candle, send):
    # Endast LONG (Spot). Kräver ny ORB och tom position.
    if not sym.orb_dir or sym.position:
        return
    if sym.orb_dir == "bullish" and live.h > sym.orb_high + TICK_EPS:
        entry = sym.orb_high + TICK_EPS
        stop  = sym.orb_low  - TICK_EPS
        if LIVE_ENABLED:
            try:
                fill_price, qty = market_buy_spot(sym.symbol, LIVE_USDT_PER_TRADE)
                if not qty:  # fallback om API inte hunnit räkna
                    qty = qty_for_usdt(LIVE_USDT_PER_TRADE, entry)
                price_use = fill_price or entry
                sym.position = {"side":"LONG","entry":price_use,"qty":qty,"stop":stop,"live":True}
                send(f"🟢 LIVE LONG {sym.symbol} @ {fmt_price(price_use)} x {fmt_qty(qty)}  SL: {fmt_price(stop)}")
            except Exception as e:
                send(f"⚠️ LIVE-köp misslyckades ({sym.symbol}): {e}")
                return
        else:
            qty = qty_for_usdt(MOCK_TRADE_USDT, entry)
            sym.position = {"side":"LONG","entry":entry,"qty":qty,"stop":stop,"live":False}
            send(f"✅ MOCK LONG {sym.symbol} @ {fmt_price(entry)} x {fmt_qty(qty)}  SL: {fmt_price(stop)}")
    # Bearish ignoreras (Spot har ej short)

def apply_trailing(sym: SymState, closed: Candle, send):
    """Stop följer ENDAST senaste stängda candlens low (aldrig nedåt)."""
    if not sym.position or sym.position["side"] != "LONG":
        return
    old = sym.position["stop"]
    new_stop = max(old, closed.l - TICK_EPS)
    sym.position["stop"] = new_stop
    # Om stängd candles low är <= stop => stäng på stop
    if closed.l <= new_stop:
        exit_trade(sym, new_stop, send)

def exit_trade(sym: SymState, price: float, send):
    pos = sym.position
    entry, qty, side, is_live = pos["entry"], pos["qty"], pos["side"], pos.get("live", False)
    if is_live and side == "LONG" and LIVE_ENABLED:
        try:
            fill = market_sell_spot(sym.symbol, qty) or price
            net, pct, fees = calc_trade_result(entry, fill, qty, side, FEE_RATE)
            state["daily_pnl"] += net
            sym.trades_today += 1
            sym.last_usdt, sym.last_pct = net, pct
            log_trade(REAL_LOG, "LIVE", sym.symbol, side, entry, fill, qty, fees, net, pct, state["daily_pnl"], "stop/close")
            send(f"📄 LIVE CLOSE {sym.symbol}  Entry {fmt_price(entry)}  Exit {fmt_price(fill)}  Qty {fmt_qty(qty)}\n"
                 f"Resultat: {'+' if net>=0 else ''}{net:.4f} USDT ({'+' if pct>=0 else ''}{pct:.2f}%)\n"
                 f"PnL idag: {state['daily_pnl']:.4f}")
        except Exception as e:
            send(f"⚠️ LIVE-sälj misslyckades ({sym.symbol}): {e}")
        finally:
            sym.position = None
        return
    # MOCK
    net, pct, fees = calc_trade_result(entry, price, qty, side)
    state["daily_pnl"] += net
    sym.trades_today += 1
    sym.last_usdt, sym.last_pct = net, pct
    log_trade(MOCK_LOG, "MOCK", sym.symbol, side, entry, price, qty, fees, net, pct, state["daily_pnl"], "stop/close")
    send(f"📄 MOCK CLOSE {sym.symbol}  Entry {fmt_price(entry)}  Exit {fmt_price(price)}  Qty {fmt_qty(qty)}\n"
         f"Resultat: {'+' if net>=0 else ''}{net:.4f} USDT ({'+' if pct>=0 else ''}{pct:.2f}%)\n"
         f"PnL idag: {state['daily_pnl']:.4f}")
    sym.position = None

# ============== Engine & Telegram ==============
stop_event = threading.Event()
send_queue = queue.Queue()
_default_chat_id = None
_app: Optional[Application] = None

def send_async(msg: str):
    send_queue.put((_default_chat_id, msg))

def sender_thread():
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
            state["daily_pnl"] = 0.0
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

            closed = candles[-2]  # senaste STÄNGDA
            if closed.t > last_ts[s]:
                update_orb(sym, closed)
                apply_trailing(sym, closed, send_async)  # flytta SL till closed low, ev. exit
                last_ts[s] = closed.t

            live = candles[-1]
            try_entry(sym, live, send_async)  # bara LONG

        time.sleep(2.5)

# Pending "JA" logik
pending_confirm = {}  # chat_id -> "mock_on"|"live_on"

# ---------- Telegram-kommandon ----------
def _help_text():
    return (
        "/status – visa status\n"
        "/start_mock – starta mock (svara JA)\n"
        "/start_live – starta LIVE (svara JA)\n"
        "/engine_start – starta motor\n"
        "/engine_stop – stoppa motor\n"
        "/symbols BTCUSDT,ETHUSDT,… – byt lista\n"
        "/timeframe 1min|3min|5min – byt tidsram\n"
        "/backtest [SYMBOL] [PERIOD[D]] [FEE]\n"
        "/export_csv – skicka loggar\n"
        "/pnl – visa dagens PnL\n"
        "/reset_pnl – nollställ dagens PnL\n"
        "/keepalive_on – håll Render vaken\n"
        "/keepalive_off – stäng keepalive"
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _default_chat_id
    _default_chat_id = update.effective_chat.id
    await update.message.reply_text(f"{BOT_NAME} redo. Skriv /help för kommandon.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_help_text())

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    header = (
        f"Bot: {BOT_NAME}\n"
        f"Läge: {'LIVE' if LIVE_ENABLED else 'MOCK'} (Spot LONG only)\n"
        f"Motor: {'ON' if ENGINE_ON else 'OFF'} | TF: {TIMEFRAME}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"PnL idag: {state['daily_pnl']:.4f}"
    )
    await update.message.reply_text(header)
    for s in SYMBOLS:
        ss = state["symbols"][s]
        orb = f"ORB=({ss.orb_dir}) H:{fmt_price(ss.orb_high) if ss.orb_high else None} L:{fmt_price(ss.orb_low) if ss.orb_low else None}"
        pos = ss.position
        msg = (
            f"{s}: {orb}\n"
            f"pos={'Y' if pos else 'N'}"
            + (f" side={pos['side']} entry={fmt_price(pos['entry'])} stop={fmt_price(pos['stop'])} qty={fmt_qty(pos['qty'])}" if pos else "")
        )
        if ss.last_usdt is not None:
            su = ss.last_usdt; sp = ss.last_pct or 0.0
            msg += f"\nSenaste: {'+' if su>=0 else ''}{su:.4f} USDT ({'+' if sp>=0 else ''}{sp:.2f}%)"
        await update.message.reply_text(msg)

async def cmd_engine_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON
    ENGINE_ON = True
    await update.message.reply_text("Motor: ON")

async def cmd_engine_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_ON
    ENGINE_ON = False
    await update.message.reply_text("Motor: OFF")

async def cmd_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending_confirm[update.effective_chat.id] = "mock_on"
    await update.message.reply_text("Vill du starta MOCK? Svara JA.")

async def cmd_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending_confirm[update.effective_chat.id] = "live_on"
    await update.message.reply_text("Vill du starta LIVE? Svara JA.")

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    txt = (update.message.text or "").strip().lower()
    if chat_id in pending_confirm and txt == "ja":
        mode = pending_confirm.pop(chat_id)
        global MOCK_ENABLED, LIVE_ENABLED
        if mode == "mock_on":
            MOCK_ENABLED, LIVE_ENABLED = True, False
            await update.message.reply_text("MOCK: AKTIVERAD")
        elif mode == "live_on":
            if not (KU_API and KU_SEC and KU_PWD):
                await update.message.reply_text("Saknar KuCoin API-nycklar (KUCOIN_API_KEY/SECRET/PASSPHRASE).")
            else:
                LIVE_ENABLED, MOCK_ENABLED = True, False
                await update.message.reply_text("LIVE: AKTIVERAD (Spot LONG only)")
        return

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

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Dagens PnL: {state['daily_pnl']:.4f} USDT")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state["daily_pnl"] = 0.0
    await update.message.reply_text("Dagens PnL nollställd.")

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sent = False
    for p in (MOCK_LOG, REAL_LOG, BACKTEST_LOG):
        if p.exists():
            await context.bot.send_document(chat_id=update.effective_chat.id, document=InputFile(p, filename=p.name))
            sent = True
    if not sent:
        await update.message.reply_text("Inga loggar ännu.")

# Enkel backtest (samma ORB/entry/stop, bara LONG)
async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Användning: /backtest SYMBOL PERIOD[D] [FEE]\nEx: /backtest BTCUSDT 3D 0.001")
        return
    symbol = context.args[0].upper()
    days = int(context.args[1].rstrip("dD"))
    fee = float(context.args[2]) if len(context.args) > 2 else FEE_RATE

    BACKTEST_LOG.unlink(missing_ok=True); ensure_csv(BACKTEST_LOG)

    # approx – hämta flera batchar à ~200 candles
    candles: List[Candle] = []
    fetches = max(1, (days*24*60)//180)
    for _ in range(fetches):
        candles.extend(kucoin_klines(symbol, TIMEFRAME, 200)); time.sleep(0.2)
    if len(candles) < 3:
        await update.message.reply_text("Fick för lite data från KuCoin.")
        return

    pnl = 0.0; trades = 0
    orb_h = orb_l = None
    pos = None

    def trail_and_maybe_exit(c: Candle):
        nonlocal pos, pnl, trades
        if not pos: return
        pos["stop"] = max(pos["stop"], c.l - TICK_EPS)
        if c.l <= pos["stop"]:
            net,pct,fees = calc_trade_result(pos["entry"], pos["stop"], pos["qty"], "LONG", fee)
            pnl += net; trades += 1
            log_trade(BACKTEST_LOG, "BACKTEST", symbol, "LONG", pos["entry"], pos["stop"], pos["qty"], fees, net, pct, pnl, "stop")
            pos = None

    for i in range(1, len(candles)):
        closed = candles[i-1]
        if closed.bullish or closed.bearish:
            if closed.rng > abs(closed.o)*MIN_ORB_PCT and closed.bullish:
                orb_h, orb_l = closed.h, closed.l
            else:
                orb_h = orb_l = None
        if pos: trail_and_maybe_exit(closed)
        live = candles[i]
        if (not pos) and orb_h is not None and live.h > orb_h + TICK_EPS:
            entry = orb_h + TICK_EPS
            pos = {"entry":entry,"qty":qty_for_usdt(MOCK_TRADE_USDT, entry),"stop":orb_l - TICK_EPS}

    await update.message.reply_text(f"Backtest {symbol} ~{days}d\nTrades: {trades}\nPnL (fee {fee:.4%}/sida): {pnl:.4f} USDT")
    if BACKTEST_LOG.exists():
        await context.bot.send_document(chat_id=update.effective_chat.id, document=InputFile(BACKTEST_LOG, filename="backtest_log.csv"))

async def cmd_keepalive_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _keepalive_on
    if not SELF_URL:
        await update.message.reply_text("Sätt SELF_URL i Environment först.")
        return
    _keepalive_on = True
    await update.message.reply_text("Keepalive: PÅ (pingar SELF_URL var 4:e minut)")

async def cmd_keepalive_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _keepalive_on
    _keepalive_on = False
    await update.message.reply_text("Keepalive: AV")

def keepalive_loop():
    while True:
        if _keepalive_on and SELF_URL:
            try: requests.get(SELF_URL, timeout=10)
            except Exception: pass
        time.sleep(240)

# ============== Start bot i bakgrund + FastAPI ==============
_app: Optional[Application] = None
bot_started = False

def start_bot_threaded():
    global _app, bot_started
    if bot_started: return
    bot_started = True
    _app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    _app.add_handler(CommandHandler("start", cmd_start))
    _app.add_handler(CommandHandler("help", cmd_help))
    _app.add_handler(CommandHandler("status", cmd_status))
    _app.add_handler(CommandHandler("engine_start", cmd_engine_start))
    _app.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    _app.add_handler(CommandHandler("start_mock", cmd_start_mock))
    _app.add_handler(CommandHandler("start_live", cmd_start_live))
    _app.add_handler(CommandHandler("symbols", cmd_symbols))
    _app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    _app.add_handler(CommandHandler("pnl", cmd_pnl))
    _app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    _app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    _app.add_handler(CommandHandler("backtest", cmd_backtest))
    _app.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    _app.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    threading.Thread(target=sender_thread, daemon=True).start()
    threading.Thread(target=engine_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=_app.run_polling, kwargs={"close_loop": False}, daemon=True).start()

app = FastAPI()

@app.on_event("startup")
def _on_startup():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Saknar TELEGRAM_TOKEN i Environment.")
    start_bot_threaded()

@app.get("/")
def root():
    return {"status": "ok", "bot": BOT_NAME, "mode": "web+polling", "spot": "LONG-only"}
