import os, sys, time, json, hmac, base64, hashlib, threading, queue, io, csv
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from telegram import (
    Bot, Update, InlineKeyboardMarkup, InlineKeyboardButton, ParseMode
)
from telegram.ext import (
    Updater, CallbackContext, CommandHandler, MessageHandler, Filters,
    CallbackQueryHandler, ConversationHandler
)

# =============== Konfiguration ===============
TZ = timezone.utc

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", os.getenv("PUBLIC_URL", "")).rstrip("/")
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.10"))  # courtage per sida i procent

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "").strip()
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "").strip()
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "").strip()

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
DEFAULT_TF = 1  # minuter
ENTRY_MODE = "close"  # "close" (rekommenderat) eller "tick"

# Trailing default (kan ändras i Telegram)
TRAIL = {"trig": 0.009, "offset": 0.002, "min": 0.007}  # 0.9% / 0.2% / min 0.7%

# =============== App & Telegram ===============
app = FastAPI()
bot = Bot(token=TELEGRAM_TOKEN)
updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
dispatcher = updater.dispatcher

EXPECT_TRAIL_INPUT = 9001

# =============== Engine-state ===============
ENGINE = {
    "enabled": False,
    "mode": "mock",             # "mock" eller "live"
    "symbols": DEFAULT_SYMBOLS[:],
    "tf": DEFAULT_TF,
    "keepalive": True,
}

STATE: Dict[str, Dict[str, Any]] = {}  # per symbol
for s in ENGINE["symbols"]:
    STATE[s] = {
        "position": None,       # dict: entry, qty, stop, high, entry_at
        "last_candle_id": None,
        "last_color": None,     # "red" / "green"
        "orb_high": None,
        "orb_low": None,
        "have_orb": False,
    }

PNL = {"day": datetime.now(TZ).date(), "mock": 0.0, "live": 0.0}
LOGQ: "queue.Queue[List[Any]]" = queue.Queue()
K4_LOG: List[Dict[str, Any]] = []  # varje avslutad affär → K4-rad

# =============== Hjälp ===============
def now_utc() -> datetime:
    return datetime.now(TZ)

def sym_to_kucoin(s: str) -> str:
    s = s.upper()
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s

def tf_to_kucoin(tf_min: int) -> str:
    m = {1:"1min",3:"3min",5:"5min",15:"15min",30:"30min",60:"1hour",240:"4hour"}
    return m.get(tf_min, "1min")

def http_get(url: str, params=None, headers=None, timeout=10) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def http_post(url: str, data=None, headers=None, timeout=12) -> Optional[dict]:
    try:
        r = requests.post(url, data=data, headers=headers, timeout=timeout)
        if r.status_code in (200, 201):
            return r.json()
        return None
    except Exception:
        return None

def kucoin_headers(method: str, endpoint: str, body: str="") -> Dict[str,str]:
    """
    V2-signering för KuCoin REST.
    """
    if not (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE):
        return {}
    ts = str(int(time.time() * 1000))
    sig_str = ts + method.upper() + endpoint + body
    sign = base64.b64encode(hmac.new(KUCOIN_API_SECRET.encode(), sig_str.encode(), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(KUCOIN_API_SECRET.encode(), KUCOIN_API_PASSPHRASE.encode(), hashlib.sha256).digest()).decode()
    return {
        "KC-API-KEY": KUCOIN_API_KEY,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }

def kline(symbol: str, tf_min: int, limit: int=3) -> Optional[List[Dict[str,Any]]]:
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"type": tf_to_kucoin(tf_min), "symbol": sym_to_kucoin(symbol)}
    js = http_get(url, params=params, timeout=8)
    if not js or js.get("code") != "200000":
        return None
    rows = js.get("data", [])
    out = []
    for r in rows[:max(limit,2)]:
        # format: [time, open, close, high, low, volume, turnover]
        t = int(float(r[0]))
        o = float(r[1]); c = float(r[2]); h = float(r[3]); l = float(r[4])
        out.append({"ts": t, "open": o, "close": c, "high": h, "low": l})
    out.sort(key=lambda x: x["ts"])
    return out[-limit:]

def last_price(symbol: str) -> Optional[float]:
    url = "https://api.kucoin.com/api/v1/market/orderbook/level1"
    js = http_get(url, params={"symbol": sym_to_kucoin(symbol)}, timeout=6)
    try:
        return float(js["data"]["price"]) if js and "data" in js else None
    except Exception:
        return None

def qty_from_usdt(symbol: str, usdt: float, price: float) -> float:
    if price <= 0: return 0.0
    return round(usdt / price, 6)

# =============== Live orders ===============
def live_market_buy(symbol: str, usdt: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Market BUY med 'funds' (USDT). Returnerar (fill_price, qty) approximativt.
    """
    try:
        endpoint = "/api/v1/orders"
        url = "https://api.kucoin.com" + endpoint
        body_dict = {
            "clientOid": f"mp_{int(time.time()*1000)}",
            "side": "buy",
            "symbol": sym_to_kucoin(symbol),
            "type": "market",
            "funds": f"{usdt:.2f}",
            "tradeType": "TRADE"
        }
        body = json.dumps(body_dict)
        headers = kucoin_headers("POST", endpoint, body)
        if not headers:
            return None, None
        js = http_post(url, data=body, headers=headers, timeout=12)
        # Vi approximera fill_price = senaste pris
        px = last_price(symbol) or 0.0
        qty = qty_from_usdt(symbol, usdt, px)
        return px, qty
    except Exception:
        return None, None

def live_market_sell(symbol: str, qty: float) -> Optional[float]:
    try:
        endpoint = "/api/v1/orders"
        url = "https://api.kucoin.com" + endpoint
        body_dict = {
            "clientOid": f"mp_{int(time.time()*1000)}",
            "side": "sell",
            "symbol": sym_to_kucoin(symbol),
            "type": "market",
            "size": f"{qty:.6f}",
            "tradeType": "TRADE"
        }
        body = json.dumps(body_dict)
        headers = kucoin_headers("POST", endpoint, body)
        if not headers:
            return None
        _ = http_post(url, data=body, headers=headers, timeout=12)
        px = last_price(symbol) or 0.0
        return px
    except Exception:
        return None

# =============== Strategi (ORB long-only) ===============
def candle_color(c: Dict[str,Any]) -> str:
    return "green" if c["close"] >= c["open"] else "red"

def new_candle_ready(ts: int, tf_min: int, last_id: Optional[int]) -> Tuple[bool, int]:
    tf_ms = tf_min * 60
    cid = (ts // tf_ms) * tf_ms
    return (last_id is None) or (cid > last_id), cid

def ensure_symbol_state(symbol: str):
    if symbol not in STATE:
        STATE[symbol] = {
            "position": None,
            "last_candle_id": None,
            "last_color": None,
            "orb_high": None,
            "orb_low": None,
            "have_orb": False,
        }

def set_orb(symbol: str, prev: Dict[str,Any], curr: Dict[str,Any]):
    STATE[symbol]["orb_high"] = max(prev["high"], curr["high"])
    STATE[symbol]["orb_low"]  = min(prev["low"],  curr["low"])
    STATE[symbol]["have_orb"] = True

def follow_next_candle_bottom(pos: Dict[str,Any], candle_low: float):
    if pos.get("stop") is None:
        pos["stop"] = candle_low
    else:
        pos["stop"] = max(pos["stop"], candle_low)

def update_trailing(pos: Dict[str,Any], high_price: float):
    entry = pos["entry"]
    pos["high"] = max(pos.get("high", entry), high_price)
    best_ret = (pos["high"] - entry) / entry
    if best_ret >= TRAIL["trig"]:
        locked = entry * (1.0 + TRAIL["min"]) if TRAIL["min"] > 0 else 0.0
        trail  = pos["high"] * (1.0 - TRAIL["offset"])
        pos["stop"] = max(pos.get("stop", 0.0), locked, trail)

def open_long(symbol: str, price_hint: float):
    if ENGINE["mode"] == "mock":
        px = price_hint or last_price(symbol) or 0.0
        qty = qty_from_usdt(symbol, MOCK_TRADE_USDT, px)
        STATE[symbol]["position"] = {"entry": px, "qty": qty, "stop": None, "high": px, "entry_at": now_utc()}
        LOGQ.put([now_utc().isoformat(), symbol, "MOCK-BUY", px, qty])
        return px, qty
    else:
        px = last_price(symbol) or 0.0
        fill_px, qty = live_market_buy(symbol, MOCK_TRADE_USDT)
        px_eff = fill_px or px
        STATE[symbol]["position"] = {"entry": px_eff, "qty": qty or 0.0, "stop": None, "high": px_eff, "entry_at": now_utc(), "live": True}
        LOGQ.put([now_utc().isoformat(), symbol, "LIVE-BUY", px_eff, qty or 0.0])
        return px_eff, qty or 0.0

def exit_position(symbol: str, exit_px: float):
    pos = STATE[symbol]["position"]
    if not pos: return
    qty = pos["qty"]
    if ENGINE["mode"] == "mock":
        pnl = (exit_px - pos["entry"]) * qty
        PNL["mock"] += pnl
        LOGQ.put([now_utc().isoformat(), symbol, "MOCK-SELL", exit_px, qty, pnl])
    else:
        live_px = live_market_sell(symbol, qty) or exit_px
        pnl = (live_px - pos["entry"]) * qty
        PNL["live"] += pnl
        LOGQ.put([now_utc().isoformat(), symbol, "LIVE-SELL", live_px, qty, pnl])
        exit_px = live_px

    # K4-rad
    K4_LOG.append({
        "symbol": symbol,
        "qty": qty,
        "buy_price": pos["entry"],
        "buy_at": pos["entry_at"].astimezone(timezone.utc),
        "sell_price": exit_px,
        "sell_at": now_utc().astimezone(timezone.utc),
    })
    STATE[symbol]["position"] = None

def engine_loop():
    while True:
        if ENGINE["enabled"]:
            try:
                for s in ENGINE["symbols"]:
                    ensure_symbol_state(s)
                    kl = kline(s, ENGINE["tf"], 3)
                    if not kl or len(kl) < 2:
                        continue
                    latest = kl[-1]          # pågående eller senast avslutade
                    prev   = kl[-2]          # säkert avslutad
                    ready, cid = new_candle_ready(latest["ts"], ENGINE["tf"], STATE[s]["last_candle_id"])
                    if ready:
                        # Ny candle har börjat → arbeta på föregående "prev" (sluten)
                        prev_col = candle_color(prev)
                        STATE[s]["last_color"] = prev_col if STATE[s]["last_color"] is None else STATE[s]["last_color"]
                        # Ny ORB vid röd -> grön
                        if STATE[s]["last_color"] == "red" and prev_col == "green":
                            set_orb(s, kl[-3] if len(kl)>=3 else prev, prev)
                        STATE[s]["last_color"] = prev_col

                        # Underhåll SL på föreg candle
                        if STATE[s]["position"]:
                            update_trailing(STATE[s]["position"], prev["high"])
                            follow_next_candle_bottom(STATE[s]["position"], prev["low"])
                            stop = STATE[s]["position"].get("stop")
                            if stop and prev["close"] <= stop:
                                exit_position(s, prev["close"])

                        # Entry på close över ORB-high (om valt)
                        if ENTRY_MODE == "close" and STATE[s]["have_orb"] and STATE[s]["position"] is None:
                            if prev["close"] > STATE[s]["orb_high"]:
                                entry_px, _ = open_long(s, prev["close"])
                                # initial SL = ORB-low
                                STATE[s]["position"]["stop"] = STATE[s]["orb_low"]

                        STATE[s]["last_candle_id"] = cid

                    # Entry på "tick" om valt
                    if ENTRY_MODE == "tick" and STATE[s]["have_orb"] and STATE[s]["position"] is None:
                        px = last_price(s)
                        if px and px > STATE[s]["orb_high"]:
                            entry_px, _ = open_long(s, px)
                            STATE[s]["position"]["stop"] = STATE[s]["orb_low"]

                    # Exit i realtid om pris <= stop
                    pos = STATE[s]["position"]
                    if pos:
                        px = last_price(s)
                        if px:
                            update_trailing(pos, max(pos.get("high", pos["entry"]), px))
                            if pos.get("stop") and px <= pos["stop"]:
                                exit_position(s, px)

            except Exception as e:
                LOGQ.put([now_utc().isoformat(), "ENGINE", "ERR", str(e)])
        time.sleep(2)

# =============== Keepalive (ping /health) ===============
def keepalive_loop():
    url = (PUBLIC_URL.rstrip("/") + "/health") if PUBLIC_URL else None
    while True:
        try:
            if ENGINE["keepalive"] and url:
                requests.get(url, timeout=5)
        except Exception:
            pass
        time.sleep(120)

# =============== Telegram UI ===============
def build_main_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 Starta MOCK", callback_data="start_mock"),
         InlineKeyboardButton("🟣 Starta LIVE", callback_data="start_live")],
        [InlineKeyboardButton("⏱ Timeframe", callback_data="tf_menu"),
         InlineKeyboardButton("🎯 Trailing", callback_data="trail_menu")],
        [InlineKeyboardButton("▶️ Engine ON", callback_data="engine_on"),
         InlineKeyboardButton("⏹ Engine OFF", callback_data="engine_off")],
    ])

def build_tf_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1m", callback_data="tf_1"),
         InlineKeyboardButton("3m", callback_data="tf_3"),
         InlineKeyboardButton("5m", callback_data="tf_5"),
         InlineKeyboardButton("15m", callback_data="tf_15")],
        [InlineKeyboardButton("⬅️ Tillbaka", callback_data="back_main")]
    ])

def build_trailing_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡ 0.9/0.2 (min 0.7)", callback_data="trail_preset:0.009,0.002,0.007")],
        [InlineKeyboardButton("⚡ 0.7/0.15 (min 0.5)", callback_data="trail_preset:0.007,0.0015,0.005")],
        [InlineKeyboardButton("⚡ 1.2/0.25 (min 0.9)", callback_data="trail_preset:0.012,0.0025,0.009")],
        [InlineKeyboardButton("✏️ Eget", callback_data="trail_custom"),
         InlineKeyboardButton("⬅️ Tillbaka", callback_data="back_main")],
    ])

def fmt_trail() -> str:
    return f"Trig {TRAIL['trig']*100:.2f}% | Avst {TRAIL['offset']*100:.2f}% | Min {TRAIL['min']*100:.2f}%"

def status_text() -> str:
    lines = []
    lines.append(f"Mode: *{ENGINE['mode']}*    Engine: *{'ON' if ENGINE['enabled'] else 'OFF'}*")
    lines.append(f"TF: *{ENGINE['tf']}m*    Entry: *{ENTRY_MODE.upper()}*")
    lines.append(f"Symbols: {', '.join(ENGINE['symbols'])}")
    if PNL["day"] != now_utc().date():
        PNL["day"] = now_utc().date(); PNL["mock"]=0.0; PNL["live"]=0.0
    lines.append(f"PnL → MOCK {PNL['mock']:.4f} | LIVE {PNL['live']:.4f}")
    lines.append(f"Trail: {fmt_trail()}")
    lines.append(f"Keepalive: {'ON' if ENGINE['keepalive'] else 'OFF'}")
    for s in ENGINE["symbols"]:
        pos = STATE.get(s, {}).get("position")
        if pos:
            stop_val = pos.get("stop")
            stop_str = "None" if stop_val is None else f"{stop_val:.6f}"
            refh = pos.get("high", pos["entry"])
            lines.append(f"{s}: pos=✅  refH={refh:.6f} stop={stop_str}")
        else:
            lines.append(f"{s}: pos=❌")
    return "\n".join(lines)

# -------- Kommandon --------
def start_cmd(update: Update, context: CallbackContext):
    update.message.reply_text("Välj:", reply_markup=build_main_kb())

def help_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(
        "/status\n"
        "/engine_start  /engine_stop\n"
        "/start_mock    /start_live\n"
        "/symbols BTCUSDT,ETHUSDT\n"
        "/timeframe\n"
        "/entry_mode tick|close\n"
        "/trailing (eller via knappar)\n"
        "/pnl  /reset_pnl  /export_csv  /export_k4\n"
        "/keepalive_on  /keepalive_off\n"
        "/panic"
    )

def status_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(status_text(), parse_mode=ParseMode.MARKDOWN, reply_markup=build_trailing_kb())
    update.message.reply_text("Huvudmeny:", reply_markup=build_main_kb())

def engine_start_cmd(update: Update, context: CallbackContext):
    ENGINE["enabled"] = True
    update.message.reply_text("Engine startad.")

def engine_stop_cmd(update: Update, context: CallbackContext):
    ENGINE["enabled"] = False
    update.message.reply_text("Engine stoppad.")

def pnl_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(f"PnL → MOCK {PNL['mock']:.4f} | LIVE {PNL['live']:.4f}")

def reset_pnl_cmd(update: Update, context: CallbackContext):
    PNL["day"] = now_utc().date(); PNL["mock"]=0.0; PNL["live"]=0.0
    update.message.reply_text("Dags-PnL nollställd.")

def export_csv_cmd(update: Update, context: CallbackContext):
    rows: List[List[Any]] = []
    try:
        while True:
            rows.append(LOGQ.get_nowait())
    except queue.Empty:
        pass
    if not rows:
        update.message.reply_text("Ingen logg att exportera.")
        return
    out = ["time,symbol,side,price,qty,pnl"]
    for r in rows:
        if len(r)==5:
            out.append(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},")
        else:
            out.append(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}")
    data = "\n".join(out).encode("utf-8")
    bio = io.BytesIO(data); bio.name="trade_log.csv"
    context.bot.send_document(chat_id=update.effective_chat.id, document=bio)

def export_k4_cmd(update: Update, context: CallbackContext):
    if not K4_LOG:
        update.message.reply_text("Inga avslutade affärer ännu.")
        return
    fieldnames = ["Beteckning","Antal","Försäljningspris","Omkostnadsbelopp","Vinst/Förlust","Datum"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames); w.writeheader()
    for tr in K4_LOG:
        # räkna courtage in/ut
        buy_gross = tr["buy_price"] * tr["qty"]
        buy_fee = buy_gross * (FEE_PCT/100.0)
        cost = buy_gross + buy_fee
        sell_gross = tr["sell_price"] * tr["qty"]
        sell_fee = sell_gross * (FEE_PCT/100.0)
        proceeds = sell_gross - sell_fee
        pnl = proceeds - cost
        w.writerow({
            "Beteckning": tr["symbol"],
            "Antal": f"{tr['qty']:.8f}",
            "Försäljningspris": f"{proceeds:.2f}",
            "Omkostnadsbelopp": f"{cost:.2f}",
            "Vinst/Förlust": f"{pnl:.2f}",
            "Datum": tr["sell_at"].strftime("%Y-%m-%d %H:%M:%S"),
        })
    data = buf.getvalue().encode("utf-8"); buf.close()
    bio = io.BytesIO(data); bio.name = f"K4_{now_utc().strftime('%Y%m%d_%H%M%S')}.csv"
    context.bot.send_document(chat_id=update.effective_chat.id, document=bio)

def symbols_cmd(update: Update, context: CallbackContext):
    if context.args:
        raw = " ".join(context.args).replace(" ", "")
        parts = [p for p in raw.split(",") if p]
        if parts:
            ENGINE["symbols"] = [p.upper() for p in parts]
            for s in ENGINE["symbols"]:
                ensure_symbol_state(s)
            update.message.reply_text(f"Symbols: {', '.join(ENGINE['symbols'])}")
            return
    update.message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,ADAUSDT")

def timeframe_cmd(update: Update, context: CallbackContext):
    update.message.reply_text("Välj tidsram:", reply_markup=build_tf_kb())

def entry_mode_cmd(update: Update, context: CallbackContext):
    global ENTRY_MODE
    if context.args and context.args[0].lower() in ("tick","close"):
        ENTRY_MODE = context.args[0].lower()
        update.message.reply_text(f"Entry-mode: {ENTRY_MODE.upper()}")
    else:
        update.message.reply_text(f"Nuvarande entry-mode: {ENTRY_MODE.upper()} (använd /entry_mode tick|close)")

def keepalive_on_cmd(update: Update, context: CallbackContext):
    ENGINE["keepalive"] = True
    update.message.reply_text("Keepalive: ON (ping varannan minut).")

def keepalive_off_cmd(update: Update, context: CallbackContext):
    ENGINE["keepalive"] = False
    update.message.reply_text("Keepalive: OFF.")

def panic_cmd(update: Update, context: CallbackContext):
    for s in ENGINE["symbols"]:
        if STATE[s]["position"]:
            # tvångs-exit på senaste pris
            px = last_price(s) or STATE[s]["position"]["entry"]
            exit_position(s, px)
        STATE[s]["position"] = None
    ENGINE["enabled"] = False
    update.message.reply_text("PANIC: alla positioner stängda, motor stoppad.")

# -------- Trailing UI --------
def trailing_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(
        f"Aktuell trailing:\n{fmt_trail()}\n\n"
        "Skriv tre procenttal (ex: `0.9, 0.2, 0.7`)\n"
        "• trigger %, avstånd %, min-lås %\n"
        "_Tips: 0.9,0.2,0.7 låser minst +0.7%._",
        parse_mode=ParseMode.MARKDOWN
    )
    return EXPECT_TRAIL_INPUT

def trailing_parse_msg(update: Update, context: CallbackContext):
    txt = (update.message.text or "").replace("%","").replace(" ","").lower()
    parts = [p for p in txt.replace(";",",").split(",") if p]
    if len(parts)!=3:
        update.message.reply_text("Kunde inte tolka. Ex: 0.9,0.2,0.7")
        return EXPECT_TRAIL_INPUT
    try:
        a,b,c = [float(x) for x in parts]
        # till decimal om användaren skrev i %
        a = a/100.0 if a>1 else a
        b = b/100.0 if b>1 else b
        c = c/100.0 if c>1 else c
        TRAIL["trig"], TRAIL["offset"], TRAIL["min"] = a,b,c
        update.message.reply_text(f"Ny trailing: {fmt_trail()}")
    except Exception:
        update.message.reply_text("Felaktiga värden.")
    return ConversationHandler.END

def trailing_cancel(update: Update, context: CallbackContext):
    update.message.reply_text("Avbrutet.")
    return ConversationHandler.END

def trailing_preset_cb(update: Update, context: CallbackContext):
    q = update.callback_query; q.answer()
    payload = q.data.split(":",1)[1]
    trig, off, mn = [float(x) for x in payload.split(",")]
    TRAIL["trig"], TRAIL["offset"], TRAIL["min"] = trig, off, mn
    q.edit_message_text(f"Preset vald ✅  {fmt_trail()}")

# -------- Callbacks (menyer) --------
def callbacks(update: Update, context: CallbackContext):
    q = update.callback_query; q.answer()
    data = q.data
    if data == "start_mock":
        ENGINE["mode"] = "mock"; q.edit_message_text("MOCK valt. Starta motorn med ▶️.")
    elif data == "start_live":
        ENGINE["mode"] = "live"; q.edit_message_text("LIVE valt. Starta motorn med ▶️.")
    elif data == "engine_on":
        if not ENGINE["enabled"]:
            ENGINE["enabled"] = True
            q.edit_message_text("Engine startad.")
        else:
            q.edit_message_text("Engine kör redan.")
    elif data == "engine_off":
        ENGINE["enabled"] = False; q.edit_message_text("Engine stoppad.")
    elif data == "tf_menu":
        q.edit_message_text("Välj tidsram:", reply_markup=build_tf_kb())
    elif data.startswith("tf_"):
        ENGINE["tf"] = int(data.split("_")[1]); q.edit_message_text(f"TF satt till {ENGINE['tf']}m.", reply_markup=build_main_kb())
    elif data == "trail_menu":
        q.edit_message_text("Välj trailing:", reply_markup=build_trailing_kb())
    elif data == "back_main":
        q.edit_message_text("Huvudmeny:", reply_markup=build_main_kb())
    elif data.startswith("trail_preset:"):
        trailing_preset_cb(update, context)

# =============== FastAPI endpoints ===============
@app.get("/", response_class=PlainTextResponse)
def root():
    return "ok"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

# =============== Start bot & loopar ===============
def start_services_once():
    if getattr(start_services_once, "started", False):
        return
    start_services_once.started = True
    # motor
    threading.Thread(target=engine_loop, daemon=True).start()
    # keepalive
    threading.Thread(target=keepalive_loop, daemon=True).start()
    # telegram polling
    updater.start_polling(drop_pending_updates=True)

# Handlers
dispatcher.add_handler(CommandHandler("start", start_cmd))
dispatcher.add_handler(CommandHandler("help", help_cmd))
dispatcher.add_handler(CommandHandler("status", status_cmd))
dispatcher.add_handler(CommandHandler("engine_start", engine_start_cmd))
dispatcher.add_handler(CommandHandler("engine_stop", engine_stop_cmd))
dispatcher.add_handler(CommandHandler("symbols", symbols_cmd, pass_args=True))
dispatcher.add_handler(CommandHandler("timeframe", timeframe_cmd))
dispatcher.add_handler(CommandHandler("entry_mode", entry_mode_cmd, pass_args=True))
dispatcher.add_handler(CommandHandler("pnl", pnl_cmd))
dispatcher.add_handler(CommandHandler("reset_pnl", reset_pnl_cmd))
dispatcher.add_handler(CommandHandler("export_csv", export_csv_cmd))
dispatcher.add_handler(CommandHandler("export_k4", export_k4_cmd))
dispatcher.add_handler(CommandHandler("keepalive_on", keepalive_on_cmd))
dispatcher.add_handler(CommandHandler("keepalive_off", keepalive_off_cmd))
dispatcher.add_handler(CommandHandler("panic", panic_cmd))

dispatcher.add_handler(CallbackQueryHandler(callbacks, pattern=r"^(start_|engine_|tf_|trail_|back_main)"))
dispatcher.add_handler(CallbackQueryHandler(trailing_preset_cb, pattern=r"^trail_preset:"))

conv_trailing = ConversationHandler(
    entry_points=[CommandHandler("trailing", trailing_cmd)],
    states={EXPECT_TRAIL_INPUT: [MessageHandler(Filters.text & ~Filters.command, trailing_parse_msg)]},
    fallbacks=[CommandHandler("cancel", trailing_cancel)],
    conversation_timeout=120,
)
dispatcher.add_handler(conv_trailing)

# start allt när uvicorn startar
start_services_once()

# lokal körning
if __name__ == "__main__":
    start_services_once()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8000")))
