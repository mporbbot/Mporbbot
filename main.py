# main.py
import os, time, json, csv, threading, signal, math, hmac, hashlib, base64
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI

from telegram import (Bot, Update, InlineKeyboardButton,
                      InlineKeyboardMarkup, ParseMode)
from telegram.ext import (Updater, CommandHandler, CallbackQueryHandler,
                          CallbackContext)

# =========================
# ----- FastAPI (ping) ----
# =========================
app = FastAPI()
START_TS = datetime.utcnow()

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "mporbbot",
        "uptime_sec": int((datetime.utcnow() - START_TS).total_seconds()),
        "engine_on": STATE.engine_on if 'STATE' in globals() else False,
    }

# ==============================
# ----- Konfig & Tillstånd -----
# ==============================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
MOCK_TRADE_USDT = float(os.environ.get("MOCK_TRADE_USDT", "30"))

KU_KEY = os.environ.get("KUCOIN_API_KEY", "")
KU_SECRET = os.environ.get("KUCOIN_API_SECRET", "")
KU_PASS = os.environ.get("KUCOIN_API_PASSPHRASE", "")

TIMEFRAME = "1m"
ENTRY_MODE = "tick"      # "tick" eller "close"
TRAIL_ON = True
TRAIL_TRIG = 0.009       # 0.9%
TRAIL_DIST = 0.002       # 0.2%
TRAIL_MIN_LOCK = 0.007   # 0.7%

SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

DATA_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
TRADES_CSV = os.path.join(DATA_DIR, "trades.csv")
K4_CSV     = os.path.join(DATA_DIR, "k4.csv")

KU_BASE   = "https://api.kucoin.com"
KU_PUB    = KU_BASE + "/api/v1"
KU_PRIVATE= KU_BASE + "/api/v1"
TF_MAP = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min"}

def utc_ms(): return int(time.time() * 1000)
def now_iso(): return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

class Position:
    def __init__(self, symbol:str):
        self.symbol = symbol
        self.qty = 0.0
        self.entry = 0.0
        self.stop = None
        self.highest = 0.0
        self.orb_high = None
        self.orb_low  = None
        self.orb_candle_id = None
        self.last_candle_id = None

class GlobalState:
    def __init__(self):
        self.engine_on = False
        self.mock = True
        self.balance = 10000.0
        self.day_pnl = 0.0
        self.keepalive = True
        self.positions: Dict[str, Position] = {s:Position(s) for s in SYMBOLS}
        self.running = True

STATE = GlobalState()

def fmt_pct(x: float) -> str: return f"{x*100:.2f}%"

def write_trade(symbol, side, qty, price, pnl=0.0, mode="MOCK"):
    row = {"ts": now_iso(),"symbol": symbol,"side": side,
           "qty": f"{qty:.8f}","price": f"{price:.8f}","pnl": f"{pnl:.8f}","mode": mode}
    newfile = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if newfile: w.writeheader()
        w.writerow(row)
    if side.lower()=="sell":
        k4_new = not os.path.exists(K4_CSV)
        k4_fields = ["datum","beteckning","antal","forsaljningspris","omkostnadsbelopp","vinst_forlust"]
        with open(K4_CSV,"a",newline="") as f:
            w=csv.DictWriter(f, fieldnames=k4_fields)
            if k4_new: w.writeheader()
            w.writerow({
                "datum": now_iso()[:10],
                "beteckning": symbol,
                "antal": f"{qty:.8f}",
                "forsaljningspris": f"{price:.6f} USDT",
                "omkostnadsbelopp": "-",
                "vinst_forlust": f"{pnl:.6f} USDT"
            })

def ku_headers(method, path, body=""):
    ts = str(utc_ms())
    str_to_sign = ts + method.upper() + path + body
    signature = base64.b64encode(hmac.new(KU_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(KU_SECRET.encode(), KU_PASS.encode(), hashlib.sha256).digest()).decode()
    return {"KC-API-KEY": KU_KEY,"KC-API-SIGN": signature,"KC-API-TIMESTAMP": ts,
            "KC-API-PASSPHRASE": passphrase,"KC-API-KEY-VERSION": "2","Content-Type": "application/json"}

def ku_get_price(symbol:str) -> Optional[float]:
    try:
        r = requests.get(f"{KU_PUB}/market/orderbook/level1", params={"symbol":symbol})
        j = r.json()
        if j.get("code")=="200000":
            return float(j["data"]["price"])
    except Exception:
        return None
    return None

def ku_get_klines(symbol:str, tf:str, limit:int=3) -> List[dict]:
    t = TF_MAP.get(tf, "1min")
    try:
        r = requests.get(f"{KU_PUB}/market/candles", params={"type":t, "symbol":symbol})
        j = r.json()
        if j.get("code")=="200000":
            rows = []
            for raw in j["data"][:limit][::-1]:
                rows.append({"ts": int(raw[0]),"open": float(raw[1]),"close": float(raw[2]),
                             "high": float(raw[3]),"low": float(raw[4])})
            return rows
    except Exception:
        pass
    return []

def qty_from_usdt(symbol:str, usdt:float, price:float)->float:
    if price<=0: return 0.0
    return round(usdt/price, 6)

def place_live_order(symbol:str, side:str, qty:float, price:Optional[float]=None) -> bool:
    try:
        path = "/api/v1/orders"
        body = {"clientOid": str(utc_ms()),"side": side,"symbol": symbol,"type": "market"}
        if side.lower()=="sell":
            body["size"] = f"{qty}"
        else:
            body["funds"] = f"{qty*price:.6f}" if price else f"{MOCK_TRADE_USDT:.2f}"
        headers = ku_headers("POST", path, json.dumps(body))
        r = requests.post(KU_PRIVATE+path, headers=headers, data=json.dumps(body), timeout=10)
        return r.status_code==200
    except Exception:
        return False

# ================ ORB/handel =================
def new_green_candle_orb(prev:dict, new:dict)->bool:
    return new["close"]>new["open"]

class Position(Position): pass  # (behåller klassen, bara för tydlighet)

def update_orb_and_maybe_enter(sym: str, pos: Position, price: float, klines: List[dict]):
    if len(klines)<2: return
    prev, last = klines[-2], klines[-1]
    last_id = last["ts"]
    pos.last_candle_id = last_id

    if new_green_candle_orb(prev, last) and (pos.orb_candle_id != last_id):
        pos.orb_high = last["high"]
        pos.orb_low  = last["low"]
        pos.orb_candle_id = last_id

    if pos.qty<=0 and pos.orb_high and price>pos.orb_high:
        entry = price
        qty = qty_from_usdt(sym, MOCK_TRADE_USDT, entry)
        pos.qty = qty; pos.entry = entry; pos.stop = pos.orb_low; pos.highest = entry
        write_trade(sym, "buy", qty, entry, 0.0, "MOCK" if STATE.mock else "LIVE")
        bot_send(f"{sym}: ORB breakout BUY @ {entry:.4f} | SL {pos.stop:.4f}")
        if not STATE.mock: place_live_order(sym, "buy", qty, entry)

def update_trailing_and_exit(sym: str, pos: Position, price: float):
    if pos.qty<=0: return
    if price>pos.highest: pos.highest = price
    kl = ku_get_klines(sym, TIMEFRAME, limit=1)
    if kl:
        low_now = kl[-1]["low"]
        if pos.stop is None or low_now > pos.stop:
            pos.stop = low_now
    gain = (price/pos.entry)-1.0
    if TRAIL_ON and gain >= TRAIL_TRIG:
        pos.stop = max(pos.stop or 0.0, pos.highest * (1.0 - TRAIL_DIST),
                       pos.entry * (1.0 + TRAIL_MIN_LOCK))
    if pos.stop and price <= pos.stop:
        pnl = (price - pos.entry) * pos.qty
        STATE.day_pnl += pnl
        write_trade(sym, "sell", pos.qty, price, pnl, "MOCK" if STATE.mock else "LIVE")
        bot_send(f"{sym}: EXIT @ {price:.4f} | PnL {pnl:.4f} USDT | day {STATE.day_pnl:.4f}")
        if not STATE.mock: place_live_order(sym, "sell", pos.qty, price)
        pos.qty=0.0; pos.entry=0.0; pos.stop=None; pos.highest=0.0

# ================ Telegram ====================
bot: Bot = None
updater: Updater = None
CHAT_ID = None

def bot_send(text:str):
    try: bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception: pass

def cmd_help(update:Update, ctx:CallbackContext):
    txt = ("<b>Kommandon</b>\n"
           "/status\n"
           "/engine_start   /engine_stop\n"
           "/start_mock     /start_live\n"
           "/symbols BTCUSDT/ETHUSDT/ADAUSDT\n"
           "/timeframe  (välj i knappar)\n"
           "/entry_mode tick|close  (eller tryck — knappar visas)\n"
           "/trailing (knappar)\n"
           "/pnl   /reset_pnl   /export_csv   /export_k4\n"
           "/keepalive_on   /keepalive_off\n"
           "/panic")
    update.message.reply_text(txt, parse_mode=ParseMode.HTML)

def cmd_status(update:Update, ctx:CallbackContext):
    global CHAT_ID; CHAT_ID = update.effective_chat.id
    lines = [
        f"Mode: {'mock' if STATE.mock else 'live'}   Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"TF: {TIMEFRAME}   Symbols: {','.join(SYMBOLS)}",
        f"Trail: trig {fmt_pct(TRAIL_TRIG)} | avst {fmt_pct(TRAIL_DIST)} | min {fmt_pct(TRAIL_MIN_LOCK)}",
        f"Keepalive: {'ON' if STATE.keepalive else 'OFF'}",
        f"PnL -> {'MOCK' if STATE.mock else 'LIVE'} {STATE.day_pnl:.4f}",
    ]
    for s in SYMBOLS:
        p = STATE.positions.get(s)
        if p and p.qty>0: lines.append(f"{s}: pos=✅ qty={p.qty:.6f} stop={p.stop:.4f}")
        else:            lines.append(f"{s}: pos=❌ stop=-")
    update.message.reply_text("\n".join(lines))

def cmd_engine_start(u,c): STATE.engine_on=True;  u.message.reply_text("Engine: ON")
def cmd_engine_stop(u,c):  STATE.engine_on=False; u.message.reply_text("Engine: OFF")
def cmd_start_mock(u,c):   STATE.mock=True;      u.message.reply_text("MOCK-läge aktiverat.")
def cmd_start_live(u,c):   STATE.mock=False;     u.message.reply_text("LIVE-läge aktiverat (KuCoin).")

def cmd_symbols(update, ctx):
    global SYMBOLS
    if ctx.args:
        syms = "".join(ctx.args).upper().replace(" ","").split(",")
        SYMBOLS = [s for s in syms if s]
        STATE.positions = {s: Position(s) for s in SYMBOLS}
        update.message.reply_text(f"Symbols uppdaterade: {','.join(SYMBOLS)}")
    else:
        update.message.reply_text(f"Aktuella symbols: {','.join(SYMBOLS)}")

# ---- timeframe (knappar) ----
TF_OPTIONS = ["1m","3m","5m","15m"]
def set_timeframe(tf:str):  global TIMEFRAME; TIMEFRAME=tf
def cmd_timeframe(update, ctx):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("TF 1m", callback_data="tf|1m"),
         InlineKeyboardButton("TF 3m", callback_data="tf|3m")],
        [InlineKeyboardButton("TF 5m", callback_data="tf|5m"),
         InlineKeyboardButton("TF 15m",callback_data="tf|15m")],
    ])
    update.message.reply_text(f"Aktuellt TF: <b>{TIMEFRAME}</b>\nVälj ny tidsram:",
                              parse_mode=ParseMode.HTML, reply_markup=kb)

# ---- trailing (knappar) ----
def cmd_trailing(update, ctx):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Trail ON (0.9%/0.2%)", callback_data="trail|on"),
         InlineKeyboardButton("Trail OFF",           callback_data="trail|off")]
    ])
    update.message.reply_text(
        f"Trailing är {'ON' if TRAIL_ON else 'OFF'} – trig {fmt_pct(TRAIL_TRIG)} / dist {fmt_pct(TRAIL_DIST)} / min {fmt_pct(TRAIL_MIN_LOCK)}",
        reply_markup=kb
    )

# ---- entry_mode (NU MED KNAPPAR) ----
def cmd_entry_mode(update, ctx):
    global ENTRY_MODE
    if ctx.args and ctx.args[0] in ("tick","close"):
        ENTRY_MODE = ctx.args[0]
        update.message.reply_text(f"entry_mode satt till {ENTRY_MODE}")
    else:
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Entry: TICK",  callback_data="emode|tick"),
             InlineKeyboardButton("Entry: CLOSE", callback_data="emode|close")]
        ])
        update.message.reply_text(f"Aktuellt entry_mode: <b>{ENTRY_MODE}</b>\nVälj:",
                                  parse_mode=ParseMode.HTML, reply_markup=kb)

def on_callback(update:Update, ctx:CallbackContext):
    global TRAIL_ON, ENTRY_MODE
    q = update.callback_query
    data = q.data
    if data.startswith("tf|"):
        tf = data.split("|",1)[1]
        if tf in TF_OPTIONS:
            set_timeframe(tf)
            q.answer(f"Satt TF till {tf}")
            q.edit_message_text(f"Tidsram uppdaterad till <b>{tf}</b>.", parse_mode=ParseMode.HTML)
            return
    if data.startswith("trail|"):
        onoff = data.split("|",1)[1]
        TRAIL_ON = (onoff=="on")
        q.answer(f"Trailing {onoff.upper()}")
        q.edit_message_text(f"Trailing är nu {'ON' if TRAIL_ON else 'OFF'}.")
        return
    if data.startswith("emode|"):
        mode = data.split("|",1)[1]
        if mode in ("tick","close"):
            ENTRY_MODE = mode
            q.answer(f"entry_mode: {mode}")
            q.edit_message_text(f"entry_mode uppdaterad till <b>{mode}</b>.", parse_mode=ParseMode.HTML)
            return
    q.answer()

def cmd_pnl(u,c):        u.message.reply_text(f"Dagens PnL: {STATE.day_pnl:.4f} USDT")
def cmd_reset_pnl(u,c):  STATE.day_pnl = 0.0; u.message.reply_text("Dagens PnL nollställt.")

def cmd_export_csv(update, ctx):
    if os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"rb") as f:
            ctx.bot.send_document(update.effective_chat.id, f, filename="trades.csv", timeout=60)
    else:
        update.message.reply_text("Ingen trades.csv ännu.")

def cmd_export_k4(update, ctx):
    if os.path.exists(K4_CSV):
        with open(K4_CSV,"rb") as f:
            ctx.bot.send_document(update.effective_chat.id, f, filename="k4.csv", timeout=60)
    else:
        update.message.reply_text("Ingen k4.csv ännu.")

def cmd_keepalive_on(u,c):  STATE.keepalive=True;  u.message.reply_text("Keepalive: ON")
def cmd_keepalive_off(u,c): STATE.keepalive=False; u.message.reply_text("Keepalive: OFF")

def cmd_panic(update, ctx):
    for s, p in STATE.positions.items():
        if p.qty>0:
            price = ku_get_price(s) or p.entry
            pnl = (price - p.entry) * p.qty
            STATE.day_pnl += pnl
            write_trade(s, "sell", p.qty, price, pnl, "MOCK" if STATE.mock else "LIVE")
            p.qty=0; p.entry=0; p.stop=None; p.highest=0
    update.message.reply_text("PANIC: Alla positioner stängda, engine OFF.")
    STATE.engine_on = False

# =============== Trader-loop =================
def trader_loop():
    while STATE.running:
        if STATE.engine_on:
            for sym in SYMBOLS:
                p = STATE.positions.setdefault(sym, Position(sym))
                price = ku_get_price(sym)
                if not price: continue
                kl = ku_get_klines(sym, TIMEFRAME, limit=3)
                update_orb_and_maybe_enter(sym, p, price, kl)
                if p.qty>0:
                    update_trailing_and_exit(sym, p, price)
        if STATE.keepalive and int(time.time())%120==0:
            try: requests.get("http://127.0.0.1:10000/")
            except Exception: pass
        time.sleep(1 if ENTRY_MODE=="tick" else 5)

# =============== Start =======================
def start_bot():
    global bot, updater
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN saknas."); return
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    bot = updater.bot
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("engine_start", cmd_engine_start))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(CommandHandler("symbols", cmd_symbols))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe))
    dp.add_handler(CommandHandler("entry_mode", cmd_entry_mode))   # <-- knappar vid behov
    dp.add_handler(CommandHandler("trailing", cmd_trailing))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("export_csv", cmd_export_csv))
    dp.add_handler(CommandHandler("export_k4", cmd_export_k4))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dp.add_handler(CommandHandler("panic", cmd_panic))
    dp.add_handler(CallbackQueryHandler(on_callback, pattern="^(tf|trail|emode)\|"))

    updater.start_polling(drop_pending_updates=True)
    print("Telegram-boten kör.")

threading.Thread(target=start_bot, daemon=True).start()
threading.Thread(target=trader_loop, daemon=True).start()

def _shutdown(*_): STATE.running = False
signal.signal(signal.SIGTERM, _shutdown)
