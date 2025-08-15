# main.py
# Mp ORBbot – mini-ORB per candle. Entry direkt på break,
# initial stop under/över break-candle, trailing till föregående candle.
# Med inline JA/NEJ-knappar för start_mock/start_live. Mock-trading 30 USDT/affär.

import os
import time
import signal
import queue
import threading
from datetime import datetime, timezone
from typing import List, Optional

import requests
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

# ======================= KONFIG =======================
BOT_NAME = "Mp ORBbot"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))  # sätt ditt privata chat-id om du vill låsa

SYMBOLS = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAME = "3min"             # 1min / 3min / 5min / 15min
AI_MODE = "neutral"            # neutral/aggressiv/försiktig (hook; ingen filtrering här)
ENGINE_ON = True

MIN_ORB_PCT = 0.001            # min candlespann (0.100%) för att godkänna ORB
FEE_RATE = 0.001               # avgift per sida (0.1%)
MOCK_TRADE_USDT = 30.0         # storlek per mock-affär
TICK_EPS = 1e-8

MOCK_ENABLED = True            # mock på som standard
LIVE_ENABLED = False           # placeholder-flagga

KEEPALIVE = False              # skicka "typing" med intervall för att hålla Render vaken
KEEPALIVE_INTERVAL = 60

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
    fees = (entry * qty + exit_ * qty) * fee_rate   # köp + sälj
    net = gross - fees
    pct = (net / (entry * qty)) * 100 if entry * qty else 0.0
    return net, pct, fees

# ======================= TELEGRAM TEXTER =======================
def tg_open_msg(is_mock, side, symbol, price, qty):
    tag = "MOCK: " if is_mock else "LIVE: "
    return f"✅ {tag}{side} {symbol} @ {fmt_price(price)} x {fmt_qty(qty)}"

def tg_close_msg(side, symbol, entry, exit_, qty, fee_rate, daily_pnl_after, is_mock=True):
    net, pct, fees = calc_trade_result(entry, exit_, qty, side, fee_rate)
    trend = "📈" if net >= 0 else "📉"
    mode = "MOCK" if is_mock else "LIVE"
    lines = [
        f"📄 Stängd {side} {symbol}",
        f"Entry: {fmt_price(entry)}  Exit: {fmt_price(exit_)}",
        f"Qty: {fmt_qty(qty)}  Avgifter: {fees:.4f}",
        f"Resultat ({mode}): {'+' if net>=0 else ''}{net:.4f} USDT ({'+' if pct>=0 else ''}{pct:.2f}%) {trend}",
        f"PnL idag → {mode}: {daily_pnl_after:.4f}"
    ]
    return "\n".join(lines)

def tg_status_header():
    return (
        f"Bot: {BOT_NAME}\n"
        f"Läge: {'MOCK' if MOCK_ENABLED else ('LIVE' if LIVE_ENABLED else 'INAKTIV')}\n"
        f"AI: {AI_MODE} (min ORB {MIN_ORB_PCT*100:.4f}%)\n"
        f"Motor: {'ON' if ENGINE_ON else 'OFF'}\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"PnL idag → MOCK: {state['daily_mock_pnl']:.4f} | LIVE: 0.0000"
    )

def tg_status_block(ss):
    has_orb = "Y" if ss.get("orb_dir") else "N"
    orb_txt = f"ORB={has_orb} ({ss['orb_dir']})" if has_orb == "Y" else "ORB=N"
    last_line = ""
    if ss.get("last_usdt") is not None:
        lu = ss["last_usdt"]; lp = ss.get("last_pct", 0.0)
        last_line = f"\nSenaste affär: {'+' if lu>=0 else ''}{lu:.4f} USDT ({'+' if lp>=0 else ''}{lp:.2f}%)"
    return (
        f"{ss['symbol']}: pos={ss['pos']} side={ss['side']}\n"
        f"entry={fmt_price(ss['entry']) if ss.get('entry') else 'None'} "
        f"stop={fmt_price(ss['stop']) if ss.get('stop') else 'None'} "
        f"trades={ss.get('trades',0)} {orb_txt}"
        f"{last_line}"
    )

def help_text():
    return (
        "/status – visa status\n"
        "/set_ai <neutral|aggressiv|försiktig>\n"
        "/start_mock – starta MOCK (klicka JA/NEJ)\n"
        "/start_live – starta LIVE (klicka JA/NEJ)\n"
        "/engine_start – starta motor\n"
        "/engine_stop – stoppa motor\n"
        "/symbols BTCUSDT,ETHUSDT,… – byt lista\n"
        "/timeframe 1min|3min|5min|15min – byt tidsram\n"
        "/pnl – visa dagens PnL\n"
        "/reset_pnl – nollställ dagens PnL\n"
        "/keepalive_on – håll Render vaken\n"
        "/keepalive_off – stäng keepalive\n"
    )

# ======================= DATASTRUKTURER =======================
class Candle:
    __slots__ = ("t","o","h","l","c","v")
    def __init__(self, t:int,o:float,h:float,l:float,c:float,v:float):
        self.t,self.o,self.h,self.l,self.c,self.v = t,o,h,l,c,v
    @property
    def bullish(self): return self.c > self.o
    @property
    def bearish(self): return self.c < self.o
    @property
    def rng(self): return (self.h - self.l)

class SymState:
    def __init__(self, symbol:str):
        self.symbol = symbol
        self.position = None         # dict: {side, entry, qty, stop}
        self.trades_today = 0
        self.last_usdt = None
        self.last_pct = None
        self.orb_high = None
        self.orb_low = None
        self.orb_dir = None          # "bullish"/"bearish"/None
    def pos_side(self):  return self.position["side"] if self.position else None
    def pos_entry(self): return self.position["entry"] if self.position else None
    def pos_stop(self):  return self.position["stop"] if self.position else None

state = {
    "symbols": {s: SymState(s) for s in SYMBOLS},
    "daily_mock_pnl": 0.0,
    "day": datetime.now(timezone.utc).date().isoformat(),
}

# ======================= KUCOIN =======================
BASE = "https://api.kucoin.com"

def kucoin_klines(symbol:str, tf:str, limit:int=200) -> List[Candle]:
    # KuCoin-format: BTC-USDT
    s = symbol.replace("USDT", "-USDT")
    url = f"{BASE}/api/v1/market/candles?type={tf}&symbol={s}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()["data"]
    # data: [ [time, open, close, high, low, volume, turnover], ... ] (omvänd ordning)
    out = []
    for row in reversed(data[-limit:]):
        try:
            t = int(float(row[0]))  # KuCoin kan skicka som str, håll det robust
        except Exception:
            t = int(row[0])
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4]); v = float(row[5])
        out.append(Candle(t, o, h, l, c, v))
    return out

# ======================= ORB-LOGIK =======================
def update_orb_after_close(sym: SymState, closed: Candle):
    """Ny mini-ORB efter varje stängd candle (samma som innan – på varje grön/röd)."""
    if closed.rng <= max(1e-12, abs(closed.o) * MIN_ORB_PCT):
        sym.orb_high = sym.orb_low = sym.orb_dir = None
        return
    sym.orb_high = closed.h
    sym.orb_low  = closed.l
    sym.orb_dir  = "bullish" if closed.bullish else ("bearish" if closed.bearish else None)

def try_entry_on_break(sym: SymState, live: Candle, send):
    """Entry direkt när live-candlen bryter senaste ORB."""
    if not sym.orb_dir or sym.position:
        return
    # LONG om live high bryter ORB-high
    if sym.orb_dir == "bullish" and live.h > (sym.orb_high + TICK_EPS):
        entry = max(sym.orb_high + TICK_EPS, live.o) if live.o > sym.orb_high else sym.orb_high + TICK_EPS
        qty = qty_for_usdt(MOCK_TRADE_USDT, entry)
        stop = sym.orb_low - TICK_EPS       # under break-candlens low
        sym.position = {"side":"LONG","entry":entry,"qty":qty,"stop":stop}
        send(tg_open_msg(True, "LONG", sym.symbol, entry, qty))
    # SHORT om live low bryter ORB-low
    elif sym.orb_dir == "bearish" and live.l < (sym.orb_low - TICK_EPS):
        entry = min(sym.orb_low - TICK_EPS, live.o) if live.o < sym.orb_low else sym.orb_low - TICK_EPS
        qty = qty_for_usdt(MOCK_TRADE_USDT, entry)
        stop = sym.orb_high + TICK_EPS      # över break-candlens high
        sym.position = {"side":"SHORT","entry":entry,"qty":qty,"stop":stop}
        send(tg_open_msg(True, "SHORT", sym.symbol, entry, qty))

def apply_trailing_and_exit(sym: SymState, just_closed: Candle, send):
    """Efter varje stängning: flytta stop till föregående candle och exit om träffad."""
    if not sym.position:
        return
    side = sym.position["side"]
    stop = sym.position["stop"]

    if side == "LONG":
        # traila upp till föregående candles low
        new_stop = max(stop, just_closed.l - TICK_EPS)
        sym.position["stop"] = new_stop
        # om candlen redan gick under stop -> stäng på stop
        if just_closed.l <= stop:
            do_exit(sym, price=stop, send=send)
    else:
        # SHORT: traila ned till föregående candles high
        new_stop = min(stop, just_closed.h + TICK_EPS)
        sym.position["stop"] = new_stop
        if just_closed.h >= stop:
            do_exit(sym, price=stop, send=send)

def do_exit(sym: SymState, price: float, send):
    pos = sym.position
    if not pos:
        return
    entry, qty, side = pos["entry"], pos["qty"], pos["side"]
    net, pct, fees = calc_trade_result(entry, price, qty, side, FEE_RATE)
    state["daily_mock_pnl"] += net
    sym.trades_today += 1
    sym.last_usdt = net
    sym.last_pct = pct
    send(tg_close_msg(side, sym.symbol, entry, price, qty, FEE_RATE, state["daily_mock_pnl"], is_mock=True))
    sym.position = None

# ======================= TRÅDAR & QUEUE =======================
stop_event = threading.Event()
send_queue = queue.Queue()
_default_chat_id = None

def default_chat_id() -> int:
    return _default_chat_id or 0

def send_async(text: str, chat_id: Optional[int] = None):
    send_queue.put((chat_id or default_chat_id(), text))

def sender_thread(upd: Updater):
    bot = upd.bot
    while not stop_event.is_set():
        try:
            chat_id, text = send_queue.get(timeout=0.5)
            if CHAT_ID != 0 and chat_id != CHAT_ID:
                continue
            bot.send_message(chat_id=chat_id, text=text)
        except queue.Empty:
            pass
        except Exception as e:
            print("Send error:", e)

def keepalive_thread(upd: Updater):
    bot = upd.bot
    while not stop_event.is_set():
        try:
            if KEEPALIVE and default_chat_id() != 0:
                bot.send_chat_action(chat_id=default_chat_id(), action="typing")
        except Exception:
            pass
        time.sleep(KEEPALIVE_INTERVAL)

def engine_loop(upd: Updater):
    last_ts = {s: 0 for s in SYMBOLS}
    while not stop_event.is_set():
        try:
            # dagsrullning
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

            for s in list(state["symbols"].keys()):
                sym = state["symbols"][s]
                # hämta klines
                try:
                    candles = kucoin_klines(s, TIMEFRAME, 200)
                except Exception as e:
                    print(f"Kline error {s}:", e)
                    time.sleep(1)
                    continue
                if not candles:
                    continue

                closed = candles[-2]  # näst sista = senaste stängda
                if closed.t <= last_ts[s]:
                    # inget nytt stängt ljus; men försök entry på live-candlen ändå
                    live = candles[-1]
                    if MOCK_ENABLED:
                        try_entry_on_break(sym, live, send=lambda msg: send_async(msg))
                    continue

                # 1) uppdatera ORB och trailing/exit på stängd candle
                update_orb_after_close(sym, closed)
                if MOCK_ENABLED:
                    apply_trailing_and_exit(sym, closed, send=lambda msg: send_async(msg))
                last_ts[s] = closed.t

                # 2) entry om live-candle bryter ny ORB
                live = candles[-1]
                if MOCK_ENABLED:
                    try_entry_on_break(sym, live, send=lambda msg: send_async(msg))

            time.sleep(2.5)
        except Exception as e:
            print("Engine error:", e)
            time.sleep(3)

# ======================= TELEGRAM HANDLERS =======================
def cmd_start(update, context: CallbackContext):
    global _default_chat_id
    _default_chat_id = update.effective_chat.id
    update.message.reply_text(f"{BOT_NAME} redo. Skriv /help för kommandon.")

def cmd_help(update, context: CallbackContext):
    update.message.reply_text(help_text())

def cmd_status(update, context: CallbackContext):
    global _default_chat_id
    _default_chat_id = update.effective_chat.id
    update.message.reply_text(tg_status_header())
    for s in SYMBOLS:
        ss = state["symbols"][s]
        block = {
            "symbol": s,
            "pos": "Y" if ss.position else "N",
            "side": ss.pos_side(),
            "entry": ss.pos_entry(),
            "stop": ss.pos_stop(),
            "trades": ss.trades_today,
            "orb_dir": ss.orb_dir,
            "last_usdt": ss.last_usdt,
            "last_pct": ss.last_pct
        }
        update.message.reply_text(tg_status_block(block))

def cmd_set_ai(update, context: CallbackContext):
    global AI_MODE
    if not context.args:
        update.message.reply_text("Användning: /set_ai <neutral|aggressiv|försiktig>")
        return
    val = context.args[0].lower()
    if val not in ("neutral", "aggressiv", "försiktig"):
        update.message.reply_text("Ogiltigt läge. Välj neutral/aggressiv/försiktig.")
        return
    AI_MODE = val
    update.message.reply_text(f"AI-läge satt till: {AI_MODE}")

def cmd_symbols(update, context: CallbackContext):
    global SYMBOLS, state
    if not context.args:
        update.message.reply_text(f"Aktuella: {', '.join(SYMBOLS)}\nAnvändning: /symbols BTCUSDT,ETHUSDT,...")
        return
    lst = "".join(context.args).upper().replace(" ", "")
    parts = [p for p in lst.split(",") if p.endswith("USDT")]
    if not parts:
        update.message.reply_text("Inga giltiga symboler hittades (måste sluta på USDT).")
        return
    SYMBOLS = parts
    state["symbols"] = {s: SymState(s) for s in SYMBOLS}
    update.message.reply_text(f"Symbols uppdaterade: {', '.join(SYMBOLS)}")

def cmd_timeframe(update, context: CallbackContext):
    global TIMEFRAME
    allowed = {"1min","3min","5min","15min"}
    if not context.args:
        update.message.reply_text(f"Aktuellt timeframe: {TIMEFRAME}\nAnvändning: /timeframe 1min|3min|5min|15min")
        return
    tf = context.args[0].lower()
    if tf not in allowed:
        update.message.reply_text("Ogiltigt timeframe. Välj 1min, 3min, 5min eller 15min.")
        return
    TIMEFRAME = tf
    update.message.reply_text(f"Timeframe satt till: {TIMEFRAME}")

def cmd_pnl(update, context: CallbackContext):
    update.message.reply_text(f"Dagens PnL\nMOCK: {state['daily_mock_pnl']:.4f} USDT\nLIVE: 0.0000 USDT")

def cmd_reset_pnl(update, context: CallbackContext):
    state["daily_mock_pnl"] = 0.0
    update.message.reply_text("Dagens PnL nollställd.")

def cmd_engine_start(update, context: CallbackContext):
    global ENGINE_ON
    ENGINE_ON = True
    update.message.reply_text("Motor: ON")

def cmd_engine_stop(update, context: CallbackContext):
    global ENGINE_ON
    ENGINE_ON = False
    update.message.reply_text("Motor: OFF")

# ---- Inline JA/NEJ för start_mock/start_live ----
def cmd_start_mock(update, context: CallbackContext):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("JA ✅", callback_data="CONFIRM_MOCK_Y"),
         InlineKeyboardButton("NEJ ❌", callback_data="CONFIRM_MOCK_N")]
    ])
    update.message.reply_text("Vill du starta MOCK-trading?", reply_markup=kb)

def cmd_start_live(update, context: CallbackContext):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("JA ✅", callback_data="CONFIRM_LIVE_Y"),
         InlineKeyboardButton("NEJ ❌", callback_data="CONFIRM_LIVE_N")]
    ])
    update.message.reply_text("Vill du starta LIVE-trading?", reply_markup=kb)

def on_callback(update, context: CallbackContext):
    global MOCK_ENABLED, LIVE_ENABLED
    q = update.callback_query
    data = q.data
    if data == "CONFIRM_MOCK_Y":
        MOCK_ENABLED = True
        q.edit_message_text("MOCK-trading: AKTIVERAD ✅")
    elif data == "CONFIRM_MOCK_N":
        MOCK_ENABLED = False
        q.edit_message_text("MOCK-trading: AVSTÄNGD ❌")
    elif data == "CONFIRM_LIVE_Y":
        LIVE_ENABLED = True
        q.edit_message_text("LIVE-trading: AKTIVERAD ✅ (obs: ej implementerad i denna fil)")
    elif data == "CONFIRM_LIVE_N":
        LIVE_ENABLED = False
        q.edit_message_text("LIVE-trading: AVSTÄNGD ❌")

# Keepalive
def cmd_keepalive_on(update, context: CallbackContext):
    global KEEPALIVE
    KEEPALIVE = True
    update.message.reply_text("Keepalive: ON (skickar 'typing' var minut)")
def cmd_keepalive_off(update, context: CallbackContext):
    global KEEPALIVE
    KEEPALIVE = False
    update.message.reply_text("Keepalive: OFF")

# ======================= STYRNING =======================
def graceful_stop(signum, frame):
    stop_event.set()

# ======================= MAIN =======================
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Kommandon
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("set_ai", cmd_set_ai))
    dp.add_handler(CommandHandler("symbols", cmd_symbols))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("engine_start", cmd_engine_start))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dp.add_handler(CallbackQueryHandler(on_callback))

    # Trådar
    th_send = threading.Thread(target=sender_thread, args=(updater,), daemon=True)
    th_send.start()
    th_engine = threading.Thread(target=engine_loop, args=(updater,), daemon=True)
    th_engine.start()
    th_keep = threading.Thread(target=keepalive_thread, args=(updater,), daemon=True)
    th_keep.start()

    signal.signal(signal.SIGINT, graceful_stop)
    signal.signal(signal.SIGTERM, graceful_stop)

    updater.start_polling()
    updater.idle()

# ======================= ENGINE (loop) =======================
# (placerad här så hela filen finns samlad ovanför __main__)

if __name__ == "__main__":
    main()
