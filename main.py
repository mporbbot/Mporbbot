import os, hmac, hashlib, base64, time, json, threading, csv
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# ============== Konfig / global state ===================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
KU_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KU_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KU_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
MOCK_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))

# symbols skrivs som "BTCUSDT,ETHUSDT" i UI men KuCoin behöver "BTC-USDT"
def to_kucoin(sym: str) -> str:
    sym = sym.upper().replace("-", "")
    return f"{sym[:-4]}-{sym[-4:]}"

def from_kucoin(sym: str) -> str:
    return sym.replace("-", "")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
VALID_TIMEFRAMES = {"1m":"1min","3m":"3min","5m":"5min"}
DEFAULT_TF = "1m"

# Bot-state
state = {
    "mode": "mock",                 # mock | live
    "engine": False,                # motor på/av (tar candles)
    "symbols": DEFAULT_SYMBOLS[:],
    "tf": DEFAULT_TF,
    "entry_mode": "close",          # tick|close (vi använder close-logik)
    "keepalive": True,
    "trail_trigger": 0.009,         # 0.9%
    "trail_offset": 0.002,          # 0.2%
    "pnl_day_mock": 0.0,
    "pnl_day_live": 0.0,
}

# Per-symbol trade-state
# in_position, qty, entry, stop, trailing_active, orb_high, orb_low, last_orb_candle_ts
positions: Dict[str, Dict[str, Any]] = {}

TRADES_CSV = "/tmp/trades.csv"     # logg för export_csv
K4_CSV = "/tmp/k4.csv"             # logg för export_k4

# ============== KuCoin REST (minimal) ===================
KU_BASE = "https://api.kucoin.com"

def _ku_headers(method: str, path: str, body: str = "") -> Dict[str, str]:
    now_ms = str(int(time.time() * 1000))
    str_to_sign = now_ms + method.upper() + path + body
    sig = base64.b64encode(hmac.new(KU_API_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(KU_API_SECRET.encode(), KU_API_PASSPHRASE.encode(), hashlib.sha256).digest()).decode()
    return {
        "KC-API-KEY": KU_API_KEY,
        "KC-API-SIGN": sig,
        "KC-API-TIMESTAMP": now_ms,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }

def ku_public_get(path: str, params: Dict[str, str]) -> Any:
    r = requests.get(KU_BASE + path, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def ku_private_post(path: str, payload: Dict[str, Any]) -> Any:
    headers = _ku_headers("POST", path, json.dumps(payload))
    r = requests.post(KU_BASE + path, headers=headers, data=json.dumps(payload), timeout=10)
    r.raise_for_status()
    return r.json()

def get_ticker_price(sym: str) -> float:
    """Last price"""
    data = ku_public_get("/api/v1/market/orderbook/level1", {"symbol": to_kucoin(sym)})
    return float(data["data"]["price"])

def get_klines(sym: str, tf: str, limit: int = 3) -> List[List[str]]:
    """
    KuCoin candles: [time, open, close, high, low, volume, turnover]
    time is ms Unix, string
    """
    ktype = VALID_TIMEFRAMES.get(tf, "1min")
    res = ku_public_get("/api/v1/market/candles", {"type": ktype, "symbol": to_kucoin(sym)})
    # latest first; we want last N in correct order
    arr = res["data"][:limit][::-1]
    return arr

# ============== Tradinglogik ===================

def color(candle) -> str:
    o = float(candle[1]); c = float(candle[2])
    return "green" if c > o else ("red" if c < o else "doji")

def hi_lo(candle) -> Tuple[float, float]:
    return float(candle[3]), float(candle[4])

def ts_ms(candle) -> int:
    # candle[0] is like "1701402000" (s) according to KuCoin docs; sometimes ms.
    t = int(candle[0])
    if t < 10**12:  # seconds
        t *= 1000
    return t

def ensure_pos(sym: str):
    if sym not in positions:
        positions[sym] = {
            "in_position": False,
            "qty": 0.0,
            "entry": 0.0,
            "stop": 0.0,
            "trailing_active": False,
            "orb_high": None,
            "orb_low": None,
            "last_orb_ts": None,
        }

def calc_qty_for_mock(price: float) -> float:
    return round(MOCK_USDT / price, 6)

def place_live_market_buy(sym: str, funds_usdt: float) -> Tuple[bool, float, float]:
    """Return (ok, qty, price). KuCoin min-lots hanteras enkelt."""
    price = get_ticker_price(sym)
    qty = round(funds_usdt / price, 6)
    payload = {
        "clientOid": f"buy-{sym}-{int(time.time())}",
        "side": "buy",
        "symbol": to_kucoin(sym),
        "type": "market",
        "funds": str(funds_usdt)
    }
    try:
        ku_private_post("/api/v1/orders", payload)
        return True, qty, price
    except Exception as e:
        print("Live BUY error:", e)
        return False, 0.0, 0.0

def place_live_market_sell(sym: str, qty: float) -> Tuple[bool, float]:
    price = get_ticker_price(sym)
    payload = {
        "clientOid": f"sell-{sym}-{int(time.time())}",
        "side": "sell",
        "symbol": to_kucoin(sym),
        "type": "market",
        "size": str(qty)
    }
    try:
        ku_private_post("/api/v1/orders", payload)
        return True, price
    except Exception as e:
        print("Live SELL error:", e)
        return False, 0.0

def record_trade(ts: datetime, sym: str, side: str, qty: float, price: float, fee: float, mode: str, pnl: float=0.0):
    header = ["timestamp","symbol","side","qty","price","fee","mode","pnl"]
    exists = os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([ts.isoformat(), sym, side, qty, price, fee, mode, round(pnl,8)])

def record_k4(ts: datetime, sym: str, qty: float, buy_price: float, sell_price: float):
    """
    Enkel K4-rad (Avsnitt D – Övriga tillgångar). Formaten varierar; vi skapar en praktisk CSV:
    datum;värdepapper;antal;anskaffning;försäljning;resultat
    """
    exists = os.path.exists(K4_CSV)
    with open(K4_CSV, "a", newline="") as f:
        w = csv.writer(f, delimiter=';')
        if not exists:
            w.writerow(["datum","värdepapper","antal","anskaffning","försäljning","resultat"])
        amount = round(qty, 6)
        purchase = round(qty*buy_price, 2)
        sale = round(qty*sell_price, 2)
        result = round(sale - purchase, 2)
        w.writerow([ts.date().isoformat(), sym, amount, purchase, sale, result])

def try_open_long(sym: str, price: float, now: datetime):
    ensure_pos(sym)
    p = positions[sym]
    if p["in_position"] or p["orb_high"] is None or p["orb_low"] is None:
        return False
    if price <= p["orb_high"]:
        return False
    # open
    if state["mode"] == "mock":
        qty = calc_qty_for_mock(price)
        fee = 0.0
        p["qty"] = qty
        p["entry"] = price
        p["stop"] = p["orb_low"]
        p["trailing_active"] = False
        p["in_position"] = True
        record_trade(now, sym, "BUY", qty, price, fee, "mock")
        return True
    else:
        ok, qty, fill_price = place_live_market_buy(sym, MOCK_USDT)
        if ok and qty > 0:
            p["qty"] = qty
            p["entry"] = fill_price
            p["stop"] = p["orb_low"]
            p["trailing_active"] = False
            p["in_position"] = True
            record_trade(now, sym, "BUY", qty, fill_price, 0.0, "live")
            return True
    return False

def try_close_long(sym: str, price_for_sell: float, now: datetime, reason: str):
    ensure_pos(sym)
    p = positions[sym]
    if not p["in_position"]:
        return False
    entry = p["entry"]; qty = p["qty"]
    pnl = (price_for_sell - entry) * qty
    if state["mode"] == "mock":
        record_trade(now, sym, "SELL", qty, price_for_sell, 0.0, "mock", pnl)
    else:
        ok, fill_price = place_live_market_sell(sym, qty)
        if ok:
            price_for_sell = fill_price
            pnl = (price_for_sell - entry) * qty
            record_trade(now, sym, "SELL", qty, price_for_sell, 0.0, "live", pnl)
    # K4-rad
    record_k4(now, sym, qty, entry, price_for_sell)
    # uppdatera PnL
    if state["mode"] == "mock":
        state["pnl_day_mock"] += pnl
    else:
        state["pnl_day_live"] += pnl
    # nollställ
    p.update({"in_position": False, "qty":0.0, "entry":0.0, "stop":0.0, "trailing_active":False})

    return True

def process_symbol(sym: str):
    ensure_pos(sym)
    p = positions[sym]
    candles = get_klines(sym, state["tf"], limit=4)
    if len(candles) < 3:
        return

    # Ta de två senaste fulla candlarna
    c0, c1 = candles[-2], candles[-1]   # näst senaste, senaste (kan vara "nästan klar" men ok i 1m)
    # ORB-detektion: röd -> grön
    # Vi letar efter grön candle (c0) som föregås av röd (c-1). Om match: sätt ny ORB på c0.
    if len(candles) >= 3:
        cprev = candles[-3]
        if color(cprev) == "red" and color(c0) == "green":
            high, low = hi_lo(c0)
            p["orb_high"], p["orb_low"] = high, low
            p["last_orb_ts"] = ts_ms(c0)

    price = float(c1[2])  # senaste close
    now = datetime.now(timezone.utc)

    # Öppna?
    if try_open_long(sym, price, now):
        pass

    # Trailing logik och stop-flytt vid varje ny candle (använd low)
    last_low = float(c1[4])
    if p["in_position"]:
        # stop får bara gå upp
        if last_low > p["stop"]:
            p["stop"] = last_low

        # extra trailing aktiveras när +0.9% (orealiserat)
        if not p["trailing_active"]:
            if (price - p["entry"]) / p["entry"] >= state["trail_trigger"]:
                p["trailing_active"] = True

        if p["trailing_active"]:
            # håll stop 0.2% under nuvarande pris men aldrig under befintlig stop
            trailing_stop = price * (1 - state["trail_offset"])
            if trailing_stop > p["stop"]:
                p["stop"] = trailing_stop

        # Exit om low under stop
        if last_low <= p["stop"]:
            try_close_long(sym, p["stop"], now, reason="stop_hit")

# ============== Telegram-bot (v13) ===================
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

updater: Optional[Updater] = None

def fmt_status() -> str:
    lines = []
    lines.append(f"Mode: {state['mode']}   Engine: {'ON' if state['engine'] else 'OFF'}")
    lines.append(f"TF: {state['tf']}   Symbols: {','.join(state['symbols'])}")
    lines.append(f"Trail: trig {state['trail_trigger']*100:.2f}% | avst {state['trail_offset']*100:.2f}% | min {max(0,(state['trail_trigger']-state['trail_offset'])*100):.2f}%")
    lines.append(f"Keepalive: {'ON' if state['keepalive'] else 'OFF'}")
    lines.append(f"PnL → MOCK {state['pnl_day_mock']:.4f} | LIVE {state['pnl_day_live']:.4f}")
    for s in state["symbols"]:
        p = positions.get(s, {})
        pos = "✅" if p.get("in_position") else "❌"
        stop = p.get("stop")
        lines.append(f"{s}: pos={pos} stop={round(stop,6) if stop else '-'}")
    return "\n".join(lines)

def kb_start():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Start MOCK", callback_data="start_mock"),
         InlineKeyboardButton("Start LIVE", callback_data="start_live")],
        [InlineKeyboardButton("Engine ON", callback_data="engine_on"),
         InlineKeyboardButton("Engine OFF", callback_data="engine_off")],
        [InlineKeyboardButton("TF 1m", callback_data="tf_1m"),
         InlineKeyboardButton("TF 3m", callback_data="tf_3m"),
         InlineKeyboardButton("TF 5m", callback_data="tf_5m")],
        [InlineKeyboardButton("Trail +0.9%/0.2%", callback_data="trail_default"),
         InlineKeyboardButton("Trail OFF", callback_data="trail_off")],
    ])

def send_help(update: Update, ctx: CallbackContext):
    text = (
        "/status\n"
        "/engine_start   /engine_stop\n"
        "/start_mock     /start_live\n"
        f"/symbols {'/'.join(state['symbols'])}\n"
        "/timeframe\n"
        "/entry_mode tick|close\n"
        "/trailing (eller via knappar)\n"
        "/pnl   /reset_pnl   /export_csv   /export_k4\n"
        "/keepalive_on   /keepalive_off\n"
        "/panic"
    )
    update.message.reply_text(text, reply_markup=kb_start())

def cmd_status(update: Update, ctx: CallbackContext): update.message.reply_text(fmt_status(), reply_markup=kb_start())
def cmd_engine_start(update: Update, ctx: CallbackContext): state["engine"]=True; update.message.reply_text("Engine: ON")
def cmd_engine_stop(update: Update, ctx: CallbackContext): state["engine"]=False; update.message.reply_text("Engine: OFF")
def cmd_start_mock(update: Update, ctx: CallbackContext): state["mode"]="mock"; update.message.reply_text("Mode: MOCK (spot sim)")
def cmd_start_live(update: Update, ctx: CallbackContext): state["mode"]="live"; update.message.reply_text("Mode: LIVE (KuCoin spot)")

def cmd_symbols(update: Update, ctx: CallbackContext):
    if ctx.args:
        syms = "".join(ctx.args).upper().replace(" ", "")
        arr = [s for s in syms.split(",") if s]
        if arr: state["symbols"] = arr
    update.message.reply_text(f"Symbols: {','.join(state['symbols'])}")

def cmd_timeframe(update: Update, ctx: CallbackContext):
    if ctx.args and ctx.args[0] in VALID_TIMEFRAMES:
        state["tf"] = ctx.args[0]
    update.message.reply_text(f"TF: {state['tf']} (1m|3m|5m)")

def cmd_entry_mode(update: Update, ctx: CallbackContext):
    if ctx.args and ctx.args[0] in ("tick","close"):
        state["entry_mode"] = ctx.args[0]
    update.message.reply_text(f"entry_mode: {state['entry_mode']}")

def cmd_trailing(update: Update, ctx: CallbackContext):
    # /trailing 0.9 0.2
    if len(ctx.args) == 2:
        try:
            trig = float(ctx.args[0])/100.0 if float(ctx.args[0])>0.5 else float(ctx.args[0])
            off  = float(ctx.args[1])/100.0 if float(ctx.args[1])>0.5 else float(ctx.args[1])
            state["trail_trigger"] = trig
            state["trail_offset"] = off
        except: pass
    update.message.reply_text(f"Trail set: trig {state['trail_trigger']*100:.2f}% | avst {state['trail_offset']*100:.2f}%")

def cmd_pnl(update: Update, ctx: CallbackContext):
    update.message.reply_text(f"PnL → MOCK {state['pnl_day_mock']:.4f} | LIVE {state['pnl_day_live']:.4f}")

def cmd_reset_pnl(update: Update, ctx: CallbackContext):
    state["pnl_day_mock"]=0.0; state["pnl_day_live"]=0.0
    update.message.reply_text("Dagens PnL nollställd.")

def cmd_export_csv(update: Update, ctx: CallbackContext):
    if not os.path.exists(TRADES_CSV):
        update.message.reply_text("Ingen trades-logg ännu.")
        return
    with open(TRADES_CSV, "rb") as f:
        update.message.reply_document(f, filename="trades.csv")

def cmd_export_k4(update: Update, ctx: CallbackContext):
    if not os.path.exists(K4_CSV):
        update.message.reply_text("Ingen K4-logg ännu.")
        return
    with open(K4_CSV, "rb") as f:
        update.message.reply_document(f, filename="k4.csv")

def cmd_keepalive_on(update: Update, ctx: CallbackContext):
    state["keepalive"]=True; update.message.reply_text("Keepalive: ON")

def cmd_keepalive_off(update: Update, ctx: CallbackContext):
    state["keepalive"]=False; update.message.reply_text("Keepalive: OFF")

def cmd_panic(update: Update, ctx: CallbackContext):
    closed=0
    now = datetime.now(timezone.utc)
    for s,p in positions.items():
        if p.get("in_position"):
            price = get_ticker_price(s)
            if try_close_long(s, price, now, "panic"):
                closed+=1
    update.message.reply_text(f"Panic: stängt {closed} positioner.")

def on_button(update: Update, ctx: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data
    if data=="start_mock": state["mode"]="mock"; q.edit_message_text("Mode: MOCK", reply_markup=kb_start())
    elif data=="start_live": state["mode"]="live"; q.edit_message_text("Mode: LIVE", reply_markup=kb_start())
    elif data=="engine_on": state["engine"]=True; q.edit_message_text("Engine: ON", reply_markup=kb_start())
    elif data=="engine_off": state["engine"]=False; q.edit_message_text("Engine: OFF", reply_markup=kb_start())
    elif data=="tf_1m": state["tf"]="1m"; q.edit_message_text(f"TF: {state['tf']}", reply_markup=kb_start())
    elif data=="tf_3m": state["tf"]="3m"; q.edit_message_text(f"TF: {state['tf']}", reply_markup=kb_start())
    elif data=="tf_5m": state["tf"]="5m"; q.edit_message_text(f"TF: {state['tf']}", reply_markup=kb_start())
    elif data=="trail_default":
        state["trail_trigger"]=0.009; state["trail_offset"]=0.002
        q.edit_message_text("Trail: +0.9% trig / 0.2% offset", reply_markup=kb_start())
    elif data=="trail_off":
        state["trail_trigger"]=9.9; state["trail_offset"]=0.0
        q.edit_message_text("Trail: AV", reply_markup=kb_start())

def start_bot():
    global updater
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN saknas.")
        return
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("help", send_help))
    dp.add_handler(CommandHandler("start", send_help))
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
    dp.add_handler(CallbackQueryHandler(on_button))
    updater.start_polling(drop_pending_updates=True)
    print("Telegram-bot kör.")

# ============== Loopar: engine + keepalive ===================

def engine_loop():
    while True:
        try:
            if state["engine"]:
                for s in state["symbols"]:
                    try:
                        process_symbol(s)
                    except Exception as e:
                        print("process_symbol error", s, e)
            time.sleep(2)  # liten vila
        except Exception as e:
            print("engine_loop top error:", e)
            time.sleep(3)

def keepalive_loop():
    while True:
        try:
            if state["keepalive"]:
                url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("RENDER_URL")
                if url:
                    try:
                        requests.get(url + "/health", timeout=5)
                    except Exception:
                        pass
            time.sleep(120)  # varannan minut
        except Exception:
            time.sleep(120)

# ============== FastAPI app ===================
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "healthy"

# start trådar vid import (uvicorn)
def _bootstrap_once():
    t1 = threading.Thread(target=start_bot, daemon=True)
    t2 = threading.Thread(target=engine_loop, daemon=True)
    t3 = threading.Thread(target=keepalive_loop, daemon=True)
    t1.start(); t2.start(); t3.start()

_bootstrap_once()
