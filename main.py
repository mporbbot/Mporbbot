import os, time, json, csv, threading, signal, hmac, base64, hashlib
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
APP_START = datetime.utcnow()

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "mporbbot",
        "uptime_sec": int((datetime.utcnow()-APP_START).total_seconds()),
        "engine_on": STATE.engine_on if "STATE" in globals() else False
    }

# ==============================
# ----- Konfig & Tillstånd -----
# ==============================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
MOCK_TRADE_USDT = float(os.environ.get("MOCK_TRADE_USDT", "30"))

KU_KEY = os.environ.get("KUCOIN_API_KEY", "")
KU_SECRET = os.environ.get("KUCOIN_API_SECRET", "")
KU_PASS = os.environ.get("KUCOIN_API_PASSPHRASE", "")

TIMEFRAME = "1m"                 # 1m / 3m / 5m / 15m
ENTRY_MODE = "close"             # "close" (default enligt din spec) eller "tick"

TRAIL_ON = True
TRAIL_TRIG = 0.009               # 0.9%
TRAIL_DIST = 0.002               # 0.2%
TRAIL_MIN_LOCK = 0.007           # 0.7%

SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

DATA_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
TRADES_CSV = os.path.join(DATA_DIR, "trades.csv")
K4_CSV     = os.path.join(DATA_DIR, "k4.csv")

KU_BASE = "https://api.kucoin.com"
KU_PUB  = KU_BASE + "/api/v1"
KU_PRIV = KU_BASE + "/api/v1"

TF_MAP = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min"}

def now_iso(): return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
def utc_ms():  return int(time.time()*1000)

# =========================
# -------- State ----------
# =========================
class Position:
    def __init__(self, symbol:str):
        self.symbol = symbol
        self.qty = 0.0
        self.entry = 0.0
        self.stop = None
        self.highest = 0.0

class ORB:
    def __init__(self):
        self.active = False
        self.used   = False    # exakt 1 trade per ORB
        self.high   = None
        self.low    = None
        self.start_ts = None

class GlobalState:
    def __init__(self):
        self.engine_on = False
        self.mock = True
        self.keepalive = True
        self.day_pnl = 0.0
        self.positions: Dict[str, Position] = {s: Position(s) for s in SYMBOLS}
        self.orb: Dict[str, ORB] = {s: ORB() for s in SYMBOLS}
        self.running = True

STATE = GlobalState()

# =========================
# ------- Helpers ---------
# =========================
def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def _kc_headers(method, path, body=""):
    ts = str(utc_ms())
    str_to_sign = ts + method.upper() + path + body
    signature = base64.b64encode(hmac.new(KU_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(KU_SECRET.encode(), KU_PASS.encode(), hashlib.sha256).digest()).decode()
    return {
        "KC-API-KEY": KU_KEY,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }

def ku_price(symbol:str)->Optional[float]:
    try:
        r = requests.get(f"{KU_PUB}/market/orderbook/level1", params={"symbol":symbol})
        j = r.json()
        if j.get("code")=="200000":
            return float(j["data"]["price"])
    except Exception:
        return None
    return None

def ku_klines(symbol:str, tf:str, limit:int=3)->List[dict]:
    t = TF_MAP.get(tf, "1min")
    try:
        r = requests.get(f"{KU_PUB}/market/candles", params={"type":t, "symbol":symbol})
        j = r.json()
        if j.get("code")=="200000":
            rows = []
            for raw in j["data"][:limit][::-1]:
                rows.append({
                    "ts": int(raw[0]),
                    "open": float(raw[1]),
                    "close": float(raw[2]),
                    "high": float(raw[3]),
                    "low": float(raw[4]),
                })
            return rows
    except Exception:
        pass
    return []

def qty_from_usdt(price: float, usdt: float)->float:
    if price<=0: return 0.0
    return round(usdt/price, 6)

def place_live(symbol:str, side:str, qty:float, price:Optional[float]=None) -> bool:
    try:
        path = "/api/v1/orders"
        body = {"clientOid": str(utc_ms()), "side": side, "symbol": symbol, "type": "market"}
        if side.lower() == "sell":
            body["size"] = f"{qty:.8f}"
        else:
            # market buy -> funds i USDT
            funds = (qty * (price or 0.0)) if price else MOCK_TRADE_USDT
            body["funds"] = f"{funds:.6f}"
        headers = _kc_headers("POST", path, json.dumps(body))
        r = requests.post(KU_PRIV+path, headers=headers, data=json.dumps(body), timeout=10)
        return r.status_code==200
    except Exception:
        return False

def write_trade(symbol, side, qty, price, pnl=0.0, mode="MOCK"):
    row = {
        "ts": now_iso(),
        "symbol": symbol,
        "side": side,
        "qty": f"{qty:.8f}",
        "price": f"{price:.8f}",
        "pnl": f"{pnl:.8f}",
        "mode": mode,
    }
    nf = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if nf: w.writeheader()
        w.writerow(row)
    if side.lower()=="sell":
        # Förenklad K4-rad (1 rad per avyttring, i USDT – omräkning till SEK får ske i efterhand)
        fields = ["datum","beteckning","antal","forsaljningspris","omkostnadsbelopp","vinst_forlust"]
        k4_new = not os.path.exists(K4_CSV)
        with open(K4_CSV,"a",newline="") as f:
            w=csv.DictWriter(f, fieldnames=fields)
            if k4_new: w.writeheader()
            w.writerow({
                "datum": now_iso()[:10],
                "beteckning": symbol,
                "antal": f"{qty:.8f}",
                "forsaljningspris": f"{price:.6f} USDT",
                "omkostnadsbelopp": "-",
                "vinst_forlust": f"{pnl:.6f} USDT"
            })

# =========================
# ------ ORB/Trading ------
# =========================
def reset_orb(sym:str):
    STATE.orb[sym] = ORB()

def maybe_arm_orb(sym:str, prev:dict, cur:dict):
    """Arm ORB när en GRÖN candle stänger direkt efter en RÖD candle."""
    if cur["close"] > cur["open"] and prev["close"] < prev["open"]:
        o = STATE.orb[sym]
        o.active = True
        o.used = False
        o.high = cur["high"]
        o.low  = cur["low"]
        o.start_ts = cur["ts"]

def maybe_invalidate_orb_on_red(sym:str, cur:dict):
    """Om ORB är aktiv (ej köpt ännu) och vi får en röd candle innan breakout → ogiltig."""
    o = STATE.orb[sym]
    if o.active and not o.used and (cur["close"] < cur["open"]):
        reset_orb(sym)

def try_entry(sym:str, price:float, prev:dict, cur:dict):
    """Köp vid breakout: default = stängning över ORB-high (ENTRY_MODE=close).
       Om ENTRY_MODE=tick → räcker med att priset passerar ORB-high intrabar."""
    o = STATE.orb[sym]
    p = STATE.positions[sym]
    if p.qty>0 or not o.active or o.used:  # redan i pos eller ej aktiv
        return
    do_buy = False
    if ENTRY_MODE == "close":
        do_buy = (cur["close"] > (o.high or 0))
    else:  # tick
        do_buy = (price is not None and price > (o.high or 0))
    if do_buy:
        entry = price if ENTRY_MODE=="tick" else cur["close"]
        qty = qty_from_usdt(entry, MOCK_TRADE_USDT)
        p.qty = qty
        p.entry = entry
        p.stop = o.low   # initial SL = ORB-low
        p.highest = entry
        o.used = True
        o.active = False   # 1 köp per ORB
        write_trade(sym, "buy", qty, entry, 0.0, "MOCK" if STATE.mock else "LIVE")
        _bot_send(f"{sym}: BUY @ {entry:.6f} | SL {p.stop:.6f}")
        if not STATE.mock:
            place_live(sym, "buy", qty, entry)

def on_candle_close(sym:str, cur:dict):
    """Uppdatera SL endast på stängd candle. Följ candlens low uppåt (aldrig ned)."""
    p = STATE.positions[sym]
    if p.qty>0:
        new_sl = max(p.stop or 0.0, cur["low"])
        if new_sl > (p.stop or 0.0):
            p.stop = new_sl

        # Trailing: trigga vid +0.9% → håll 0.2% under peak och minst +0.7% från entry
        gain = (cur["close"]/p.entry) - 1.0
        if TRAIL_ON and gain >= TRAIL_TRIG:
            p.highest = max(p.highest, cur["close"])
            trail_stop = p.highest * (1.0 - TRAIL_DIST)
            min_lock   = p.entry   * (1.0 + TRAIL_MIN_LOCK)
            p.stop = max(p.stop or 0.0, trail_stop, min_lock)

def maybe_exit(sym:str, price:float):
    p = STATE.positions[sym]
    if p.qty>0 and p.stop:
        if price <= p.stop:
            pnl = (price - p.entry) * p.qty
            STATE.day_pnl += pnl
            write_trade(sym, "sell", p.qty, price, pnl, "MOCK" if STATE.mock else "LIVE")
            _bot_send(f"{sym}: EXIT @ {price:.6f} | PnL {pnl:.4f} USDT | day {STATE.day_pnl:.4f}")
            if not STATE.mock:
                place_live(sym, "sell", p.qty, price)
            # nolla pos
            p.qty=0.0; p.entry=0.0; p.stop=None; p.highest=0.0
            # OBS: ORB startas igen när en ny röd→grön uppstår (se maybe_arm_orb)

# =========================
# ------ Telegram ----------
# =========================
bot: Bot = None
updater: Updater = None
CHAT_ID = None

def _bot_send(text:str):
    try:
        if CHAT_ID:
            bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception:
        pass

def cmd_help(update:Update, ctx:CallbackContext):
    txt = (
        "<b>Kommandon</b>\n"
        "/status\n"
        "/engine_start   /engine_stop\n"
        "/start_mock     /start_live\n"
        "/symbols BTCUSDT,ETHUSDT,ADAUSDT\n"
        "/timeframe  (välj i knappar)\n"
        "/entry_mode (välj i knappar)\n"
        "/trailing  (välj i knappar)\n"
        "/pnl   /reset_pnl   /export_csv   /export_k4\n"
        "/keepalive_on   /keepalive_off\n"
        "/panic"
    )
    update.message.reply_text(txt, parse_mode=ParseMode.HTML)

def cmd_status(update:Update, ctx:CallbackContext):
    global CHAT_ID
    CHAT_ID = update.effective_chat.id
    lines = [
        f"Mode: {'mock' if STATE.mock else 'live'}   Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"TF: {TIMEFRAME}   Symbols: {','.join(SYMBOLS)}",
        f"Entry: {ENTRY_MODE}   Trail: {'ON' if TRAIL_ON else 'OFF'} ({fmt_pct(TRAIL_TRIG)}/{fmt_pct(TRAIL_DIST)} min {fmt_pct(TRAIL_MIN_LOCK)})",
        f"Keepalive: {'ON' if STATE.keepalive else 'OFF'}   DayPnL: {STATE.day_pnl:.4f} USDT",
    ]
    for s in SYMBOLS:
        p = STATE.positions.get(s)
        o = STATE.orb.get(s)
        pos_txt = "✅" if (p and p.qty>0) else "❌"
        orb_txt = f"ORB: {'ON' if (o and o.active) else 'OFF'}"
        stop_txt = f"{p.stop:.6f}" if (p and p.stop) else "-"
        lines.append(f"{s}: pos={pos_txt} stop={stop_txt} | {orb_txt}")
    update.message.reply_text("\n".join(lines))

def cmd_engine_start(update, ctx): STATE.engine_on=True;  update.message.reply_text("Engine: ON")
def cmd_engine_stop(update, ctx):  STATE.engine_on=False; update.message.reply_text("Engine: OFF")
def cmd_start_mock(update, ctx):   STATE.mock=True;       update.message.reply_text("MOCK-läge: ON")
def cmd_start_live(update, ctx):   STATE.mock=False;      update.message.reply_text("LIVE-läge: ON (KuCoin)")

def cmd_symbols(update, ctx):
    global SYMBOLS
    if ctx.args:
        syms = "".join(ctx.args).upper().replace(" ","").split(",")
        syms = [s for s in syms if s]
        if not syms:
            update.message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,ADAUSDT")
            return
        SYMBOLS = syms
        # re-init states
        STATE.positions = {s: Position(s) for s in SYMBOLS}
        STATE.orb       = {s: ORB() for s in SYMBOLS}
        update.message.reply_text(f"Symbols uppdaterade: {','.join(SYMBOLS)}")
    else:
        update.message.reply_text(f"Aktuella symbols: {','.join(SYMBOLS)}")

# --- knappar efter kommandon ---
def cmd_timeframe(update, ctx):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("TF 1m", callback_data="tf|1m"),
         InlineKeyboardButton("TF 3m", callback_data="tf|3m")],
        [InlineKeyboardButton("TF 5m", callback_data="tf|5m"),
         InlineKeyboardButton("TF 15m",callback_data="tf|15m")],
    ])
    update.message.reply_text(f"Aktuellt TF: <b>{TIMEFRAME}</b>\nVälj ny tidsram:",
                              parse_mode=ParseMode.HTML, reply_markup=kb)

def cmd_entry_mode(update, ctx):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Entry: CLOSE (rekommenderad)", callback_data="emode|close")],
        [InlineKeyboardButton("Entry: TICK (aggressiv)",      callback_data="emode|tick")]
    ])
    update.message.reply_text(f"Aktuellt entry_mode: <b>{ENTRY_MODE}</b>\nVälj:",
                              parse_mode=ParseMode.HTML, reply_markup=kb)

def cmd_trailing(update, ctx):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Trail ON (0.9%/0.2%)", callback_data="trail|on"),
         InlineKeyboardButton("Trail OFF",            callback_data="trail|off")]
    ])
    update.message.reply_text(
        f"Trailing är {'ON' if TRAIL_ON else 'OFF'} – trig {fmt_pct(TRAIL_TRIG)} / dist {fmt_pct(TRAIL_DIST)} / min {fmt_pct(TRAIL_MIN_LOCK)}",
        reply_markup=kb
    )

def on_callback(update:Update, ctx:CallbackContext):
    global TIMEFRAME, ENTRY_MODE, TRAIL_ON
    q = update.callback_query
    data = q.data
    if data.startswith("tf|"):
        TIMEFRAME = data.split("|",1)[1]
        q.answer(f"Satt TF till {TIMEFRAME}")
        q.edit_message_text(f"Tidsram uppdaterad till <b>{TIMEFRAME}</b>.", parse_mode=ParseMode.HTML)
        return
    if data.startswith("emode|"):
        ENTRY_MODE = data.split("|",1)[1]
        q.answer(f"entry_mode: {ENTRY_MODE}")
        q.edit_message_text(f"entry_mode uppdaterad till <b>{ENTRY_MODE}</b>.", parse_mode=ParseMode.HTML)
        return
    if data.startswith("trail|"):
        onoff = data.split("|",1)[1]
        TRAIL_ON = (onoff=="on")
        q.answer(f"Trailing {onoff.upper()}")
        q.edit_message_text(f"Trailing är nu {'ON' if TRAIL_ON else 'OFF'}.")
        return
    q.answer()

def cmd_pnl(update, ctx):        update.message.reply_text(f"Dagens PnL: {STATE.day_pnl:.4f} USDT")
def cmd_reset_pnl(update, ctx):  STATE.day_pnl = 0.0; update.message.reply_text("Dagens PnL nollställt.")

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

def cmd_keepalive_on(update, ctx):  STATE.keepalive=True;  update.message.reply_text("Keepalive: ON")
def cmd_keepalive_off(update, ctx): STATE.keepalive=False; update.message.reply_text("Keepalive: OFF")

def cmd_panic(update, ctx):
    for s, p in STATE.positions.items():
        if p.qty>0:
            price = ku_price(s) or p.entry
            pnl = (price - p.entry) * p.qty
            STATE.day_pnl += pnl
            write_trade(s, "sell", p.qty, price, pnl, "MOCK" if STATE.mock else "LIVE")
            p.qty=0; p.entry=0; p.stop=None; p.highest=0
    STATE.engine_on = False
    update.message.reply_text("PANIC: alla positioner stängda. Engine OFF.")

# =========================
# ------ Trader-loop -------
# =========================
def trader_loop():
    while STATE.running:
        try:
            if STATE.engine_on:
                for sym in SYMBOLS:
                    # se till att state finns
                    if sym not in STATE.positions: STATE.positions[sym] = Position(sym)
                    if sym not in STATE.orb:       STATE.orb[sym] = ORB()

                    price = ku_price(sym)
                    if not price: 
                        continue

                    # hämta senaste 3 candles (för prev + cur)
                    kl = ku_klines(sym, TIMEFRAME, limit=3)
                    if len(kl) < 2:
                        continue
                    prev, cur = kl[-2], kl[-1]

                    # 1) Arm ORB när röd -> grön
                    maybe_arm_orb(sym, prev, cur)

                    # 2) Om ORB aktiv men ännu inget köp: invalidation om röd innan breakout
                    maybe_invalidate_orb_on_red(sym, cur)

                    # 3) Entry (1 köp per ORB)
                    try_entry(sym, price, prev, cur)

                    # 4) SL-följning + trailing på candle close
                    on_candle_close(sym, cur)

                    # 5) Exit på stop (tick-baserad jämförelse)
                    maybe_exit(sym, price)

            # keepalive varannan minut
            if STATE.keepalive and int(time.time()) % 120 == 0:
                try: requests.get("http://127.0.0.1:10000/")
                except Exception: pass

            time.sleep(1 if ENTRY_MODE=="tick" else 5)
        except Exception as e:
            # tyst felhantering för att inte döda loopen på Render
            # (kan kompletteras med /debug om vi vill)
            time.sleep(2)

# =========================
# ------ Starta allt -------
# =========================
def start_bot():
    global bot, updater
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN saknas.")
        return
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
    dp.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    dp.add_handler(CommandHandler("trailing", cmd_trailing))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("export_csv", cmd_export_csv))
    dp.add_handler(CommandHandler("export_k4", cmd_export_k4))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dp.add_handler(CommandHandler("panic", cmd_panic))
    dp.add_handler(CallbackQueryHandler(on_callback, pattern="^(tf|emode|trail)\|"))

    updater.start_polling(drop_pending_updates=True)
    print("Telegram-boten kör.")

threading.Thread(target=start_bot, daemon=True).start()
threading.Thread(target=trader_loop, daemon=True).start()

def _shutdown(*_):
    STATE.running = False
signal.signal(signal.SIGTERM, _shutdown)
