# main.py
# Mp ORBbot – med KuCoin-marknads-watchdog + auto-reconnect
# Kör: python main.py
# Miljövariabler (frivilligt för live): KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE
# Telegram: TOKEN kan hårdkodas nedan (som du bett om) eller via env TELEGRAM_TOKEN

import os
import asyncio
import json
import time
import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz

# Telegram (python-telegram-bot v20)
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# KuCoin (OBS: Market istället för Client i kucoin-python v1.x)
from kucoin.client import Market
from kucoin.ws_client import KucoinWsClient

# =========== KONFIG ===========
BOT_NAME = "Mp ORBbot"
DEFAULT_SYMBOLS = ["BTCUSDT","ETHUSDT","ADAUSDT","LINKUSDT","XRPUSDT"]
DEFAULT_TF_MIN = 1                              # 1m
MOCK_TRADE_USDT = 30.0                          # per mocktrade
FEE_RATE = 0.001                                # 0.1% default
AI_MODE_DEFAULT = "neutral"                     # neutral/aggressiv/försiktig
TRAIL_ENABLE = True
TRAIL_ARM_PCT = 0.009                           # 0.90%
TRAIL_STEP_PCT = 0.002                          # 0.20%
TRAIL_MIN_LOCK_PCT = 0.007                      # 0.70% min
ORB_MASTER_DEFAULT = True

# Watchdog
TICK_STALE_WARN_SEC = 60
TICK_STALE_STOP_ENGINE_SEC = 300

# Filer
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

# Telegram token (du gav denna; inkluderas här enligt din önskan)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")

# KuCoin API keys (behövs bara för live trading i framtiden)
KU_KEY = os.getenv("KUCOIN_KEY", "")
KU_SECRET = os.getenv("KUCOIN_SECRET", "")
KU_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE", "")

TZ = pytz.timezone("Europe/Stockholm")

# =========== STATE ===========
@dataclass
class SymbolState:
    pos_open: bool = False
    direction: Optional[str] = None  # 'long' or 'short'
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    orb_on: bool = True
    last_tick_ts: float = 0.0
    last_price: Optional[float] = None
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_window_active: bool = False
    trades_today: int = 0

@dataclass
class EngineState:
    running: bool = False
    mock_mode: bool = True
    ai_mode: str = AI_MODE_DEFAULT
    tf_min: int = DEFAULT_TF_MIN
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    day_pnl: float = 0.0
    keepalive_telegram: bool = True
    orb_master: bool = ORB_MASTER_DEFAULT
    last_telegram_heartbeat: float = time.time()
    trail_enable: bool = TRAIL_ENABLE
    fee_rate: float = FEE_RATE

engine = EngineState()
sym_state: Dict[str, SymbolState] = {s: SymbolState() for s in engine.symbols}

# KuCoin market (publik REST-klient) + WS
ku_market = Market(url='https://api.kucoin.com')
ws_client: Optional[KucoinWsClient] = None
ws_connected = asyncio.Event()

# =========== HJÄLPFUNKTIONER ===========
def now_ts() -> float:
    return time.time()

def fmt_usdt(x: float) -> str:
    return f"{x:.4f} USDT"

def stockholm_now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def ensure_csv_headers(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp","symbol","side","qty_usdt","price","pnl_usdt",
                "mode","ai_mode","fees","note"
            ])

async def log_trade(path: str, *, symbol: str, side: str, qty_usdt: float,
                    price: float, pnl: float, note: str=""):
    ensure_csv_headers(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            stockholm_now_str(), symbol, side, f"{qty_usdt:.8f}",
            f"{price:.8f}", f"{pnl:.8f}",
            "mock" if engine.mock_mode else "real",
            engine.ai_mode, f"{engine.fee_rate:.6f}", note
        ])

def apply_fees(gross_pnl: float, entry: float, exit: float, qty_usdt: float) -> float:
    fee = qty_usdt * engine.fee_rate * 2
    return gross_pnl - fee

def ai_filter_ok(symbol: str, price: float) -> bool:
    if engine.ai_mode == "aggressiv":
        return True
    if engine.ai_mode == "försiktig":
        st = sym_state[symbol]
        return st.last_price is not None and abs(price - st.last_price) / price > 0.0005
    return True

def reset_orb_window(symbol: str):
    st = sym_state[symbol]
    st.orb_high = None
    st.orb_low = None
    st.orb_window_active = False

def update_orb_window(symbol: str, price: float):
    st = sym_state[symbol]
    if not st.orb_window_active:
        st.orb_window_active = True
        st.orb_high = price
        st.orb_low = price
    else:
        st.orb_high = max(st.orb_high, price)
        st.orb_low = min(st.orb_low, price)

def trail_update(symbol: str, price: float):
    st = sym_state[symbol]
    if not (engine.trail_enable and st.pos_open and st.entry_price):
        return
    up = (price - st.entry_price) / st.entry_price
    if up >= TRAIL_ARM_PCT:
        target_lock = st.entry_price * (1 + max(TRAIL_MIN_LOCK_PCT, up - TRAIL_STEP_PCT))
        st.stop_price = max(st.stop_price or 0, target_lock)

async def close_position(symbol: str, price: float, why: str, tg: Optional[ContextTypes.DEFAULT_TYPE]=None):
    st = sym_state[symbol]
    if not st.pos_open or st.entry_price is None:
        return
    qty_usdt = MOCK_TRADE_USDT
    gross = (price - st.entry_price) / st.entry_price * qty_usdt
    pnl = apply_fees(gross, st.entry_price, price, qty_usdt)
    engine.day_pnl += pnl
    await log_trade(MOCK_LOG if engine.mock_mode else REAL_LOG,
                    symbol=symbol, side="sell", qty_usdt=qty_usdt,
                    price=price, pnl=pnl, note=why)
    st.pos_open = False
    st.entry_price = None
    st.stop_price = None
    st.direction = None
    if tg:
        await tg.bot.send_message(chat_id=engine_chat_id, text=f"🔔 {symbol} stängd @ {price:.4f} | PnL: {fmt_usdt(pnl)} ({why})")

async def maybe_open_orb(symbol: str, price: float, ctx: Optional[ContextTypes.DEFAULT_TYPE]=None):
    st = sym_state[symbol]
    if not engine.orb_master or not st.orb_on:
        return
    if st.pos_open:
        return
    if not st.orb_window_active or st.orb_high is None or st.orb_low is None:
        return
    if price > st.orb_high and ai_filter_ok(symbol, price):
        st.pos_open = True
        st.direction = "long"
        st.entry_price = price
        st.stop_price = st.orb_low
        st.trades_today += 1
        await log_trade(MOCK_LOG if engine.mock_mode else REAL_LOG,
                        symbol=symbol, side="buy", qty_usdt=MOCK_TRADE_USDT,
                        price=price, pnl=0.0, note="ORB breakout")
        if ctx:
            await ctx.bot.send_message(chat_id=engine_chat_id, text=f"✅ {symbol} ORB LONG @ {price:.4f} | SL {st.stop_price:.4f}")

# =========== TELEGRAM ===========
engine_chat_id: Optional[int] = None

def status_text() -> str:
    lines = []
    lines.append(f"Mode: {'mock' if engine.mock_mode else 'real'}   Engine: {'ON' if engine.running else 'OFF'}")
    lines.append(f"TF: {engine.tf_min}m   Symbols:\n{','.join(engine.symbols)}")
    lines.append(f"Entry: TICK   Trail: {'ON' if engine.trail_enable else 'OFF'} ({TRAIL_ARM_PCT*100:.2f}%/{TRAIL_STEP_PCT*100:.2f}% min {TRAIL_MIN_LOCK_PCT*100:.2f}%)")
    lines.append(f"Keepalive: {'ON' if engine.keepalive_telegram else 'OFF'}   DayPnL: {fmt_usdt(engine.day_pnl)}")
    lines.append(f"ORB master: {'ON' if engine.orb_master else 'OFF'}")
    for s in engine.symbols:
        st = sym_state[s]
        pos = "✅" if st.pos_open else "❌"
        stop = f"{st.stop_price:.4f}" if st.stop_price else "-"
        lines.append(f"{s}: pos={pos} stop={stop} | ORB: {'ON' if st.orb_on else 'OFF'}")
    return "\n".join(lines)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global engine_chat_id
    engine_chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"{BOT_NAME} redo.\n"
        "Skriv 'JA' för att starta motorn i MOCK-läge.\n"
        "Skriv 'LIVE' för att starta riktig trading (kräver API-nycklar)."
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start – initiera\n"
        "/stop – stoppa motorn (kommandon funkar ändå)\n"
        "/status – nuvarande status\n"
        "/pnl – dagens PnL\n"
        "/set_ai <aggressiv|neutral|försiktig>\n"
        "/backtest <symbol> <period> [fee]\n"
        "/export_csv – exportera CSV\n"
        "/orb_on | /orb_off – ORB för alla\n"
        "/mock_trade <SYMBOL> – manuell mocktrade\n"
    )

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine.running = False
    await update.message.reply_text("⏹️ Motorn stoppad. (Kommandon fungerar fortfarande)")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(status_text())

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"DayPnL: {fmt_usdt(engine.day_pnl)}")

async def cmd_set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /set_ai aggressiv|neutral|försiktig")
        return
    mode = context.args[0].lower()
    if mode not in ["aggressiv","neutral","försiktig"]:
        await update.message.reply_text("Ogiltigt AI-läge.")
        return
    engine.ai_mode = mode
    await update.message.reply_text(f"AI-läge satt till: {mode}")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for s in engine.symbols:
        sym_state[s].orb_on = True
    await update.message.reply_text("ORB: ON (för alla valda symboler)")

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for s in engine.symbols:
        sym_state[s].orb_on = False
    await update.message.reply_text("ORB: OFF (för alla valda symboler)")

async def cmd_mock_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Använd: /mock_trade SYMBOL")
        return
    symbol = context.args[0].upper()
    if symbol not in engine.symbols:
        await update.message.reply_text("Symbolen är inte aktiv.")
        return
    price = sym_state[symbol].last_price or 0.0
    sym_state[symbol].pos_open = True
    sym_state[symbol].direction = "long"
    sym_state[symbol].entry_price = price
    sym_state[symbol].stop_price = max(0.0, price*(1-0.01))
    await log_trade(MOCK_LOG, symbol=symbol, side="buy", qty_usdt=MOCK_TRADE_USDT, price=price, pnl=0.0, note="manual mock_trade")
    await update.message.reply_text(f"Mocktrade öppnad i {symbol} @ {price:.4f}")

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("CSV-filer finns lokalt i samma katalog:\n- mock_trade_log.csv\n- real_trade_log.csv")

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Backtest placeholder. Säg till om du vill att jag lägger in full KuCoin-historik här också.")

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip().upper()
    global engine_chat_id
    engine_chat_id = update.effective_chat.id

    if text == "JA":
        engine.running = True
        engine.mock_mode = True
        await update.message.reply_text("✅ Motorn startad i MOCK-läge.")
    elif text == "LIVE":
        engine.running = True
        engine.mock_mode = False
        await update.message.reply_text("⚠️ LIVE-läge begärt. Se till att KUCOIN API-nycklar finns satt.")
    else:
        await update.message.reply_text("Ok! Skriv /help för kommandon.")

# =========== KUCOIN STREAM + WATCHDOG ===========
async def _handle_tick(symbol: str, price: float):
    st = sym_state[symbol]
    st.last_price = price
    st.last_tick_ts = now_ts()
    update_orb_window(symbol, price)

    if engine.running:
        trail_update(symbol, price)
        if st.pos_open and st.stop_price and price <= st.stop_price:
            await close_position(symbol, price, "Stop-loss")
            reset_orb_window(symbol)
            return
        await maybe_open_orb(symbol, price)

async def _subscribe_ws(loop):
    async def deal_msg(msg):
        if "data" in msg and "price" in msg["data"]:
            topic = msg.get("topic", "")
            if ":" in topic:
                inst = topic.split(":")[1]  # 'BTC-USDT'
                symbol = inst.replace("-", "")
                try:
                    price = float(msg["data"]["price"])
                except:
                    return
                await _handle_tick(symbol, price)

    global ws_client
    try:
        markets = [s.replace("USDT","-USDT") for s in engine.symbols]
        ws_client = await KucoinWsClient.create(None, deal_msg, private=False)
        for m in markets:
            await ws_client.subscribe(f"/market/ticker:{m}")
        ws_connected.set()
    except Exception as e:
        ws_connected.clear()
        print("WS connect error:", e)

async def ws_manager():
    while True:
        if ws_client is None or not ws_connected.is_set():
            await _subscribe_ws(asyncio.get_running_loop())
        await asyncio.sleep(5)

async def rest_poll_fallback():
    """Pollar REST var 3s om WS inte är ansluten."""
    while True:
        if not ws_connected.is_set():
            try:
                for s in engine.symbols:
                    inst = s.replace("USDT","-USDT")
                    tick = ku_market.get_ticker(inst)
                    price = float(tick["price"])
                    await _handle_tick(s, price)
            except Exception as e:
                print("REST poll error:", e)
        await asyncio.sleep(3)

async def market_watchdog(context: ContextTypes.DEFAULT_TYPE):
    now = now_ts()
    stale_symbols = []
    very_stale = False
    for s in engine.symbols:
        st = sym_state[s]
        if now - st.last_tick_ts > TICK_STALE_WARN_SEC:
            stale_symbols.append(s)
        if now - st.last_tick_ts > TICK_STALE_STOP_ENGINE_SEC:
            very_stale = True
    if stale_symbols and engine_chat_id:
        await context.bot.send_message(chat_id=engine_chat_id,
            text=f"⚠️ Inga nya ticks för: {', '.join(stale_symbols)} – försöker reconnecta KuCoin WS...")
        try:
            global ws_client
            if ws_client:
                await ws_client.close()
            ws_connected.clear()
        except Exception:
            pass
    if very_stale and engine.running and engine_chat_id:
        engine.running = False
        await context.bot.send_message(chat_id=engine_chat_id,
            text="⛔ Marknadsdata saknas >5 min. Motorn pausad. Starta igen med 'JA' när ticks rullar.")

# =========== ENGINE LOOP ===========
async def engine_loop():
    while True:
        now = datetime.now(TZ)
        if now.hour == 0 and now.minute == 0:
            engine.day_pnl = 0.0
            for s in engine.symbols:
                sym_state[s].trades_today = 0
        await asyncio.sleep(30)

# =========== MAIN ===========
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("set_ai", cmd_set_ai))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("orb_on", cmd_orb_on))
    app.add_handler(CommandHandler("orb_off", cmd_orb_off))
    app.add_handler(CommandHandler("mock_trade", cmd_mock_trade))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.job_queue.run_repeating(market_watchdog, interval=10, first=15)

    loop = asyncio.get_running_loop()
    loop.create_task(ws_manager())
    loop.create_task(rest_poll_fallback())
    loop.create_task(engine_loop())

    print(f"{BOT_NAME} startar…")
    await app.initialize()
    await app.start()
    try:
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        while True:
            engine.last_telegram_heartbeat = now_ts()
            await asyncio.sleep(20)
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Avslutar…")
