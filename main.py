# main.py
# Mp ORBbot – "mini-ORB på varje grön/röd candle", entry direkt på break,
# initial stop under/över break-candlen, trailing till föregående candle.
# Mock-trading (30 USDT/affär), Telegram-meddelanden i "första bilden"-stil.

import os
import time
import math
import json
import signal
import queue
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import requests
from telegram.ext import Updater, CommandHandler
from telegram import ParseMode

# =============== KONFIGURATION ===============
BOT_NAME = "Mp ORBbot"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))  # sätt till ditt privata chat id om du vill låsa

SYMBOLS = ["LINKUSDT","XRPUSDT","ADAUSDT","BTCUSDT","ETHUSDT"]
TIMEFRAME = "1min"  # KuCoin: "1min","3min","5min",...
ENGINE_ON = True
AI_MODE = "neutral"  # neutral/aggressiv/försiktig (hook ligger här om du vill filtrera)
MIN_ORB_PCT = 0.001  # 0.100% min range – använd om du vill filtrera för små candles

FEE_RATE = 0.001   # 0.1% per köp/sälj
MOCK_TRADE_USDT = 30.0
TICK_EPS = 1e-8

# =============== HJÄLPSAKER ===============
def now_utc_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def fmt_qty(q: float) -> str:
    return f"{q:.6f}".rstrip("0").rstrip(".")

def fmt_price(p: float) -> str:
    return f"{p:.6f}".rstrip("0").rstrip(".")

def calc_trade_result(entry, exit_, qty, side, fee_rate=FEE_RATE):
    direction = 1 if side == "LONG" else -1
    gross = (exit_ - entry) * qty * direction
    fees = (entry * qty + exit_ * qty) * fee_rate
    net = gross - fees
    pct = (net / (entry * qty)) * 100 if entry * qty else 0.0
    return net, pct, fees

def qty_for_usdt(usdt: float, price: float) -> float:
    return usdt / max(price, 1e-12)

# =============== TELEGRAM FORMAT ===============
def tg_open_msg(is_mock, side, symbol, price, qty):
    tag = "MOCK: " if is_mock else ""
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
        f"Läge: {'MOCK'}\n"
        f"AI: {AI_MODE} (min ORB {MIN_ORB_PCT*100:.4f}%)\n"
        f"Motor: {'ON' if ENGINE_ON else 'OFF'}\n"
        f"Timeframe: 1min\n"
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

# =============== DATA & STATE ===============
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
        self.position = None  # dict: {side, entry, qty, stop}
        self.trades_today = 0
        self.last_usdt = None
        self.last_pct = None
        self.orb_high = None
        self.orb_low = None
        self.orb_dir = None    # "bullish" / "bearish" / None
        self.prev_close_candle = None
        self.last_candle = None

    def pos_side(self):
        return self.position["side"] if self.position else None
    def pos_entry(self):
        return self.position["entry"] if self.position else None
    def pos_stop(self):
        return self.position["stop"] if self.position else None

state = {
    "symbols": {s: SymState(s) for s in SYMBOLS},
    "daily_mock_pnl": 0.0,
    "day": datetime.now(timezone.utc).date().isoformat(),
}

# =============== KUCOIN KLINES ===============
BASE = "https://api.kucoin.com"
def kucoin_klines(symbol:str, tf:str="1min", limit:int=200) -> List[Candle]:
    # KuCoin symbolformat: BTC-USDT
    s = symbol.replace("USDT","-USDT")
    url = f"{BASE}/api/v1/market/candles?type={tf}&symbol={s}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()["data"]
    # data: [ [time, open, close, high, low, volume, turnover], ... ] reverse chronological
    out = []
    for row in reversed(data[-limit:]):
        t = int(row[0])//1000
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4]); v = float(row[5])
        out.append(Candle(t,o,h,l,c,v))
    return out

# =============== ORB-LOGIK ===============
def update_orb_after_close(sym: SymState, closed: Candle):
    """Sätt nytt mini-ORB efter varje stängd candle, men filtrera pyttesmå ranges."""
    if closed.rng <= closed.o * MIN_ORB_PCT:
        sym.orb_high = sym.orb_low = sym.orb_dir = None
        return
    sym.orb_high = closed.h
    sym.orb_low  = closed.l
    sym.orb_dir  = "bullish" if closed.bullish else ("bearish" if closed.bearish else None)

def try_entry_on_break(sym: SymState, new_candle: Candle, send):
    """Entry direkt när den bryter senaste ORB."""
    if not sym.orb_dir: return
    if sym.position: return

    # Bullish ORB -> LONG när high bryts
    if sym.orb_dir == "bullish" and new_candle.h > (sym.orb_high + TICK_EPS):
        entry = max(sym.orb_high + TICK_EPS, new_candle.o) if new_candle.o > sym.orb_high else sym.orb_high + TICK_EPS
        qty = qty_for_usdt(MOCK_TRADE_USDT, entry)
        stop = sym.orb_low - TICK_EPS   # under break-candlens low (candlen som satte ORB)
        sym.position = {"side":"LONG","entry":entry,"qty":qty,"stop":stop}
        send(tg_open_msg(True, "LONG", sym.symbol, entry, qty))

    # Bearish ORB -> SHORT när low bryts
    if sym.orb_dir == "bearish" and new_candle.l < (sym.orb_low - TICK_EPS):
        entry = min(sym.orb_low - TICK_EPS, new_candle.o) if new_candle.o < sym.orb_low else sym.orb_low - TICK_EPS
        qty = qty_for_usdt(MOCK_TRADE_USDT, entry)
        stop = sym.orb_high + TICK_EPS  # över break-candlens high
        sym.position = {"side":"SHORT","entry":entry,"qty":qty,"stop":stop}
        send(tg_open_msg(True, "SHORT", sym.symbol, entry, qty))

def apply_trailing_and_exit(sym: SymState, closed: Candle, send):
    """Efter varje candle-stängning: flytta stop mot föregående candle och kolla exit."""
    if not sym.position: return

    side = sym.position["side"]
    stop = sym.position["stop"]

    # Trailing: till föregående candle (dvs den just stängda)
    if side == "LONG":
        new_stop = max(stop, closed.l - TICK_EPS)
        sym.position["stop"] = new_stop
        # Exit om stängda candlen redan punkterade stop (konservativt)
        if closed.l <= stop:
            do_exit(sym, price=stop, send=send)
    else:
        new_stop = min(stop, closed.h + TICK_EPS)
        sym.position["stop"] = new_stop
        if closed.h >= stop:
            do_exit(sym, price=stop, send=send)

def do_exit(sym: SymState, price: float, send):
    pos = sym.position
    if not pos: return
    entry, qty, side = pos["entry"], pos["qty"], pos["side"]
    net, pct, fees = calc_trade_result(entry, price, qty, side, FEE_RATE)
    state["daily_mock_pnl"] += net
    sym.trades_today += 1
    sym.last_usdt = net; sym.last_pct = pct
    send(tg_close_msg(side, sym.symbol, entry, price, qty, FEE_RATE, state["daily_mock_pnl"], is_mock=True))
    sym.position = None

# =============== LOOP & TELEGRAM ===============
stop_event = threading.Event()
send_queue = queue.Queue()

def sender_thread(upd:Updater):
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

def send_async(text: str, chat_id: Optional[int]=None):
    send_queue.put((chat_id or default_chat_id(), text))

_default_chat_id = None
def default_chat_id():
    return _default_chat_id or 0

def cmd_start(update, _):
    global _default_chat_id
    _default_chat_id = update.effective_chat.id
    update.message.reply_text("Mp ORBbot redo (MOCK). Skriv /status för läget.")

def cmd_status(update, _):
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

def graceful_stop(signum, frame):
    stop_event.set()

def day_rollover_if_needed():
    d = datetime.now(timezone.utc).date().isoformat()
    if d != state["day"]:
        state["day"] = d
        state["daily_mock_pnl"] = 0.0
        for s in SYMBOLS:
            st = state["symbols"][s]
            st.trades_today = 0
            st.last_usdt = st.last_pct = None

def engine_loop(upd:Updater):
    last_ts = {s:0 for s in SYMBOLS}
    while not stop_event.is_set():
        try:
            day_rollover_if_needed()
            for s in SYMBOLS:
                sym = state["symbols"][s]
                candles = kucoin_klines(s, "1min", 200)
                if not candles: continue
                closed = candles[-2]  # näst sista = stängd
                if closed.t <= last_ts[s]:
                    continue
                # 1) Ny stängd candle => uppdatera mini-ORB och trailing/exit på stängningen
                update_orb_after_close(sym, closed)
                apply_trailing_and_exit(sym, closed, send=lambda msg: send_async(msg))
                last_ts[s] = closed.t
                # 2) Försök entry på den FÄRSKA ORB när nästa candle bryter
                live = candles[-1]
                try_entry_on_break(sym, live, send=lambda msg: send_async(msg))
            time.sleep(3)
        except Exception as e:
            print("Engine error:", e)
            time.sleep(3)

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))

    th_send = threading.Thread(target=sender_thread, args=(updater,), daemon=True)
    th_send.start()

    th_engine = threading.Thread(target=engine_loop, args=(updater,), daemon=True)
    th_engine.start()

    signal.signal(signal.SIGINT, graceful_stop)
    signal.signal(signal.SIGTERM, graceful_stop)

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
