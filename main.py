import os
import sys
import csv
import hmac
import time
import json
import math
import queue
import base64
import signal
import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ParseMode,
)
from telegram.ext import (
    Updater,
    CallbackContext,
    CommandHandler,
    CallbackQueryHandler,
    Filters,
)

# -------------------------
# Konfiguration & Globalt
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("mporbbot")

TZ = timezone.utc

def now_utc() -> datetime:
    return datetime.now(tz=TZ)

# Miljö
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
KUCOIN_KEY = os.getenv("KUCOIN_API_KEY", "").strip()
KUCOIN_SECRET = os.getenv("KUCOIN_API_SECRET", "").strip()
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "").strip()
PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")

MOCK_USDT = float(os.getenv("MOCK_TRADE_USDT", "30") or 30)

# Motorstatus
ENGINE: Dict[str, Any] = {
    "enabled": False,          # motor on/off
    "mode": "mock",            # mock|live
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "tf": 1,                   # minuter
    "ai": "aggressiv",
    "keepalive": True,
}

# Trail / SL-regler
TRAIL: Dict[str, float] = {
    "trig": 0.009,     # 0.9%
    "offset": 0.002,   # 0.2%
    "min": 0.0,        # minsta låsning (0.0 = av)
}

# PnL (dagligt)
PNL: Dict[str, Any] = {"day": now_utc().date(), "mock": 0.0, "live": 0.0}

# Marknads-/positions-state per symbol
STATE: Dict[str, Any] = {}
for s in ENGINE["symbols"]:
    STATE[s] = {
        "position": None,       # {entry, qty, high, stop}  (stop kan vara None)
        "last_candle_ts": None, # senast behandlade candle start-timestamp (sek)
        "last_high": None,      # föreg ljusets high
        "last_low": None,       # föreg ljusets low
    }

# Thread-säker kö för loggrader (till export)
LOGQ = queue.Queue()

# -------------------------
# Hjälp/utility
# -------------------------
def symbol_to_kucoin(s: str) -> str:
    """BTCUSDT -> BTC-USDT"""
    s = s.upper()
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USDT"
    # fallback
    if "-" in s:
        return s
    return s

def tf_to_kucoin(tf_min: int) -> str:
    # Kucoin typ: 1min, 3min, 5min, etc.
    return f"{tf_min}min"

def http_get(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Optional[dict]:
    try:
        r = requests.get(url, headers=headers or {}, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        log.warning("GET %s -> %s %s", url, r.status_code, r.text[:300])
    except Exception as e:
        log.warning("GET fail %s: %s", url, e)
    return None

def http_post(url: str, headers: Optional[Dict[str, str]] = None, data: Optional[dict] = None) -> Optional[dict]:
    try:
        r = requests.post(url, headers=headers or {}, json=data or {}, timeout=10)
        if r.status_code in (200, 201):
            return r.json()
        log.warning("POST %s -> %s %s", url, r.status_code, r.text[:300])
    except Exception as e:
        log.warning("POST fail %s: %s", url, e)
    return None

def kucoin_auth_headers(method: str, endpoint: str, body: str = "") -> Dict[str, str]:
    """
    Enkel V2 signering. Byggd för spot/market. Kör defensivt (om nycklar saknas -> tomma headers).
    """
    if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
        return {}
    ts = str(int(time.time() * 1000))
    str_to_sign = ts + method.upper() + endpoint + body
    sign = base64.b64encode(
        hmac.new(KUCOIN_SECRET.encode("utf-8"), str_to_sign.encode("utf-8"), hashlib.sha256).digest()
    ).decode()
    passphrase = base64.b64encode(
        hmac.new(KUCOIN_SECRET.encode("utf-8"), KUCOIN_PASSPHRASE.encode("utf-8"), hashlib.sha256).digest()
    ).decode()
    return {
        "KC-API-KEY": KUCOIN_KEY,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }

def get_latest_candles_kucoin(symbol: str, tf_min: int, limit: int = 2) -> Optional[List[Dict[str, Any]]]:
    """
    Hämtar senaste 2 candels via KuCoins publika endpoint.
    Returnerar lista med dictionaries: [{t, o, h, l, c}]
    """
    kc_symbol = symbol_to_kucoin(symbol)
    ktype = tf_to_kucoin(tf_min)
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"type": ktype, "symbol": kc_symbol}
    data = http_get(url, params=params)
    # KuCoin returnerar ["time","open","close","high","low","volume","turnover"]
    if not data or data.get("code") != "200000":
        return None
    items = data.get("data", [])[:limit]
    out = []
    for it in items:
        try:
            # Kucoin tid i unix millisek? Dokument säger str sek? Vi hanterar båda.
            t_raw = it[0]
            if isinstance(t_raw, str):
                tsec = int(float(t_raw))
            else:
                tsec = int(t_raw)
            o = float(it[1]); c = float(it[2]); h = float(it[3]); l = float(it[4])
            out.append({"t": tsec, "o": o, "h": h, "l": l, "c": c})
        except Exception:
            continue
    if not out:
        return None
    # KuCoin sorterar ofta senaste först; vi vill ha äldst->nyast
    out.sort(key=lambda x: x["t"])
    return out[-limit:]

def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b

# -------------------------
# Handelslogik (ORB)
# -------------------------
def handle_new_candle(symbol: str, o: float, h: float, l: float, c: float, tsec: int):
    st = STATE[symbol]

    # Spara "förgående ljus" – vi behöver high/low för breakout & SL-follow
    prev_high = st["last_high"]
    prev_low = st["last_low"]

    st["last_high"] = h
    st["last_low"] = l
    st["last_candle_ts"] = tsec

    pos = st["position"]

    # 1) Har vi position? Flytta SL till nya candlens low (trail på candle-low)
    if pos:
        new_stop = l
        stop_val = pos.get("stop")
        if (stop_val is None) or (new_stop > stop_val):
            pos["stop"] = new_stop

        # 2) Om orealiserad vinst ≥ trig → trailing offset
        entry = pos["entry"]
        if pct(c, entry) >= TRAIL["trig"]:
            # trailing stop till (högsta - offset%), baserat på high vi håller i pos
            pos["high"] = max(pos.get("high", entry), h)
            target = pos["high"] * (1.0 - TRAIL["offset"])
            if pos.get("stop") is None or target > pos["stop"]:
                pos["stop"] = target

        # 3) Stop hit? (Mock & Live hanteras senare i tick)
        return

    # 4) Ingen position → ORB long: Om föreg. candle var "grön" (c>o) och nytt pris bryter över föreg. high → köp
    if prev_high is not None and prev_low is not None:
        prev_green = (c > o)  # här går vi på aktuellt ljus; mild tolkning: med nästa tick säljs
        # stramare variant: använd föregående ljus färg, dvs c_prev>o_prev – men vi saknar c_prev/o_prev här
        # vi kör breakout på ny candle: close över prev_high
        if c > prev_high:
            # skapa position (handel sker i tick_scan, här registrerar vi intention)
            qty = 0.0  # sätts i mock/live exekvering
            STATE[symbol]["position"] = {
                "entry": c,
                "qty": qty,
                "high": c,
                "stop": prev_low,  # initial SL = föregående low
            }

# -------------------------
# Mock & Live exekvering
# -------------------------
def account_market_buy_live(symbol: str, usdt_amount: float) -> Optional[Dict[str, Any]]:
    """Marknadsköp spot via KuCoin. Returnerar order-info eller None vid fel."""
    try:
        kc_symbol = symbol_to_kucoin(symbol)
        endpoint = "/api/v1/orders"
        url = "https://api.kucoin.com" + endpoint
        client_oid = f"mporb_{int(time.time()*1000)}"
        payload = {
            "clientOid": client_oid,
            "side": "buy",
            "symbol": kc_symbol,
            "type": "market",
            "funds": str(round(usdt_amount, 2)),  # USDT-summa
            "tradeType": "TRADE",  # spot
        }
        headers = kucoin_auth_headers("POST", endpoint, json.dumps(payload))
        if not headers:
            return None
        data = http_post(url, headers=headers, data=payload)
        return data
    except Exception as e:
        log.warning("Live BUY fail %s: %s", symbol, e)
        return None

def account_market_sell_live(symbol: str, base_qty: float) -> Optional[Dict[str, Any]]:
    try:
        kc_symbol = symbol_to_kucoin(symbol)
        endpoint = "/api/v1/orders"
        url = "https://api.kucoin.com" + endpoint
        client_oid = f"mporb_{int(time.time()*1000)}"
        payload = {
            "clientOid": client_oid,
            "side": "sell",
            "symbol": kc_symbol,
            "type": "market",
            "size": str(base_qty),
            "tradeType": "TRADE",
        }
        headers = kucoin_auth_headers("POST", endpoint, json.dumps(payload))
        if not headers:
            return None
        data = http_post(url, headers=headers, data=payload)
        return data
    except Exception as e:
        log.warning("Live SELL fail %s: %s", symbol, e)
        return None

def price_tick(symbol: str) -> Optional[float]:
    """Senaste pris (close) via KuCoin 1-min candle."""
    cs = get_latest_candles_kucoin(symbol, ENGINE["tf"], limit=1)
    if not cs:
        return None
    return float(cs[-1]["c"])

def mock_fill_qty(symbol: str, price: float, usdt_amount: float) -> float:
    if price <= 0:
        return 0.0
    # enkel beräkning, ingen avgift
    return round(usdt_amount / price, 6)

def tick_scan():
    """
    Körs av job_queue var 10:e sekund:
    - Hämta 2 senaste candles
    - Om ny candle → kör handle_new_candle
    - Kontrollera SL/TP/Trail & fyll mock/live
    """
    if not ENGINE["enabled"]:
        return

    for s in ENGINE["symbols"]:
        try:
            cs = get_latest_candles_kucoin(s, ENGINE["tf"], limit=2)
            if not cs or len(cs) < 1:
                continue

            # om ny candle:
            st = STATE[s]
            latest = cs[-1]
            tsec = latest["t"]
            if st["last_candle_ts"] != tsec:
                # stäm av föreg candle (cs[-2]) och aktuellt
                o, h, l, c = latest["o"], latest["h"], latest["l"], latest["c"]
                handle_new_candle(s, o, h, l, c, tsec)

            # uppdatera marknadspris
            cprice = latest["c"]
            pos = st["position"]
            if not pos:
                continue

            # mock/l ive öppna/fylla köp om qty=0
            if pos["qty"] == 0.0:
                if ENGINE["mode"] == "mock":
                    qty = mock_fill_qty(s, pos["entry"], MOCK_USDT)
                    pos["qty"] = qty
                    LOGQ.put([now_utc().isoformat(), s, "MOCK-BUY", pos["entry"], qty])
                else:
                    # live
                    if KUCOIN_KEY:
                        res = account_market_buy_live(s, MOCK_USDT)
                        # vi "antar" fill på entry-priset för PnL-uppföljning
                        qty = mock_fill_qty(s, pos["entry"], MOCK_USDT)
                        pos["qty"] = qty
                        LOGQ.put([now_utc().isoformat(), s, "LIVE-BUY", pos["entry"], qty])
                    else:
                        log.info("No KUCOIN keys; live disabled, skipping buy.")
                        ENGINE["mode"] = "mock"

            # trail/stop – sälj när pris <= stop
            stop_val = pos.get("stop")
            if (stop_val is not None) and (cprice <= stop_val):
                # sälj
                if ENGINE["mode"] == "mock":
                    sell_p = cprice
                    pnl = (sell_p - pos["entry"]) * pos["qty"]
                    PNL["mock"] += pnl
                    LOGQ.put([now_utc().isoformat(), s, "MOCK-SELL", sell_p, pos["qty"], pnl])
                else:
                    sell_p = cprice
                    if KUCOIN_KEY:
                        _ = account_market_sell_live(s, pos["qty"])
                        pnl = (sell_p - pos["entry"]) * pos["qty"]
                        PNL["live"] += pnl
                        LOGQ.put([now_utc().isoformat(), s, "LIVE-SELL", sell_p, pos["qty"], pnl])
                    else:
                        log.info("No KUCOIN keys; live disabled, skipping sell.")
                st["position"] = None
                continue

            # uppdatera "high" för trail-beräkning
            pos["high"] = max(pos.get("high", pos["entry"]), cprice)

        except Exception as e:
            log.warning("tick_scan %s: %s", s, e)

# -------------------------
# Telegram UI
# -------------------------
def kb_start(kind: str) -> InlineKeyboardMarkup:
    # kind: mock|live
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ JA", callback_data=f"confirm_start:{kind}:yes"),
         InlineKeyboardButton("❌ NEJ", callback_data=f"confirm_start:{kind}:no")]
    ])

def kb_timeframe() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1m", callback_data="tf:1"),
         InlineKeyboardButton("3m", callback_data="tf:3"),
         InlineKeyboardButton("5m", callback_data="tf:5")],
    ])

def kb_trail() -> InlineKeyboardMarkup:
    # Snabbval
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Trig 0.9% / Avst 0.2%", callback_data="trail:0.009:0.002")],
        [InlineKeyboardButton("Trig 0.6% / Avst 0.2%", callback_data="trail:0.006:0.002")],
        [InlineKeyboardButton("Trig 1.2% / Avst 0.3%", callback_data="trail:0.012:0.003")],
    ])

def status_text() -> str:
    lines: List[str] = []
    lines.append(f"Mode: *{ENGINE['mode']}*    Engine: *{'ON' if ENGINE['enabled'] else 'OFF'}*")
    lines.append(f"AI: *{ENGINE['ai']}*    TF: *{ENGINE['tf']}m*")
    lines.append(f"Symbols: {', '.join(ENGINE['symbols'])}")
    if PNL["day"] != now_utc().date():
        PNL["day"] = now_utc().date()
        PNL["mock"] = 0.0
        PNL["live"] = 0.0
    lines.append(f"PnL → MOCK {PNL['mock']:.4f} | LIVE {PNL['live']:.4f}")
    lines.append(f"Trail: trig {TRAIL['trig']*100:.2f}% | avst {TRAIL['offset']*100:.2f}% | min {TRAIL['min']*100:.2f}%")
    lines.append(f"Keepalive: {'ON' if ENGINE['keepalive'] else 'OFF'}")
    for s in ENGINE["symbols"]:
        pos = STATE.get(s, {}).get("position")
        if pos:
            stop_val = pos.get("stop")
            if stop_val is None:
                stop_str = "None"
            else:
                stop_str = f"{stop_val:.6f}"
            lines.append(f"{s}: pos=✅  refH={pos.get('high', pos['entry']):.6f} stop={stop_str}")
        else:
            lines.append(f"{s}: pos=❌")
    return "\n".join(lines)

# --- Handlers ---
def cmd_help(update: Update, ctx: CallbackContext):
    txt = (
        "/status – visa status\n"
        "/start_mock – starta MOCK (tryck JA)\n"
        "/start_live – starta LIVE (tryck JA)\n"
        "/engine_start – starta motor\n"
        "/engine_stop – stoppa motor\n"
        "/timeframe – välj tidsram (knappar)\n"
        "/symbols BTCUSDT,ETHUSDT,ADAUSDT – byt lista\n"
        "/set_ai <neutral|aggressiv|försiktig> – välj AI-läge (enbart etikett)\n"
        "/trail – snabbval för trail (knappar)\n"
        "/pnl – visa dagens PnL\n"
        "/reset_pnl – nollställ dagens PnL\n"
        "/export_csv – skicka logg (dagens)\n"
        "/keepalive_on – håll Render vaken\n"
        "/keepalive_off – stäng keepalive\n"
        "/panic – stäng alla pos och stoppa\n"
    )
    update.message.reply_text(txt)

def cmd_status(update: Update, ctx: CallbackContext):
    update.message.reply_text(status_text(), parse_mode=ParseMode.MARKDOWN)

def cmd_timeframe(update: Update, ctx: CallbackContext):
    update.message.reply_text("Välj tidsram:", reply_markup=kb_timeframe())

def cb_timeframe(update: Update, ctx: CallbackContext):
    q = update.callback_query
    q.answer()
    try:
        _, val = q.data.split(":")
        ENGINE["tf"] = int(val)
        q.edit_message_text(f"Tidsram satt till {ENGINE['tf']}m")
    except Exception:
        q.edit_message_text("Fel vid byte av tidsram.")

def cmd_trail(update: Update, ctx: CallbackContext):
    update.message.reply_text("Välj trailing-inställning:", reply_markup=kb_trail())

def cb_trail(update: Update, ctx: CallbackContext):
    q = update.callback_query
    q.answer()
    try:
        _, trig, off = q.data.split(":")
        TRAIL["trig"] = float(trig)
        TRAIL["offset"] = float(off)
        q.edit_message_text(f"Trail uppdaterad: trig {TRAIL['trig']*100:.2f}% | avst {TRAIL['offset']*100:.2f}%")
    except Exception:
        q.edit_message_text("Fel vid trail-byte.")

def cmd_symbols(update: Update, ctx: CallbackContext):
    if not ctx.args:
        update.message.reply_text(f"Aktuella symboler: {', '.join(ENGINE['symbols'])}")
        return
    raw = " ".join(ctx.args).replace(" ", "")
    parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    if not parts:
        update.message.reply_text("Inga symboler angivna.")
        return
    ENGINE["symbols"] = parts
    for s in parts:
        if s not in STATE:
            STATE[s] = {"position": None, "last_candle_ts": None, "last_high": None, "last_low": None}
    update.message.reply_text(f"Symboler uppdaterade: {', '.join(ENGINE['symbols'])}")

def cmd_set_ai(update: Update, ctx: CallbackContext):
    if not ctx.args:
        update.message.reply_text(f"AI-läge: {ENGINE['ai']}")
        return
    ai = ctx.args[0].lower()
    ENGINE["ai"] = ai
    update.message.reply_text(f"AI-läge uppdaterat: {ai}")

def cmd_engine_start(update: Update, ctx: CallbackContext):
    ENGINE["enabled"] = True
    update.message.reply_text("Motorn startad.")

def cmd_engine_stop(update: Update, ctx: CallbackContext):
    ENGINE["enabled"] = False
    update.message.reply_text("Motorn stoppad.")

def cmd_start_mock(update: Update, ctx: CallbackContext):
    update.message.reply_text("Vill du starta MOCK-trading?", reply_markup=kb_start("mock"))

def cmd_start_live(update: Update, ctx: CallbackContext):
    update.message.reply_text("Vill du starta LIVE-trading?", reply_markup=kb_start("live"))

def cb_confirm_start(update: Update, ctx: CallbackContext):
    q = update.callback_query
    q.answer()
    try:
        _, kind, ans = q.data.split(":")
        if ans == "yes":
            ENGINE["mode"] = kind
            ENGINE["enabled"] = True
            q.edit_message_text(f"Startar {kind.upper()}-läge. Motorn ON.")
        else:
            q.edit_message_text("Avbrutet.")
    except Exception:
        q.edit_message_text("Fel vid start.")

def cmd_pnl(update: Update, ctx: CallbackContext):
    if PNL["day"] != now_utc().date():
        PNL["day"] = now_utc().date()
        PNL["mock"] = 0.0
        PNL["live"] = 0.0
    update.message.reply_text(f"PnL → MOCK {PNL['mock']:.4f} | LIVE {PNL['live']:.4f}")

def cmd_reset_pnl(update: Update, ctx: CallbackContext):
    PNL["day"] = now_utc().date()
    PNL["mock"] = 0.0
    PNL["live"] = 0.0
    update.message.reply_text("Dags-PnL nollställd.")

def cmd_export_csv(update: Update, ctx: CallbackContext):
    # Töm kö till minnes-CSV och skicka som text (enkelt för Render free)
    rows: List[List[Any]] = []
    try:
        while True:
            rows.append(LOGQ.get_nowait())
    except queue.Empty:
        pass
    if not rows:
        update.message.reply_text("Ingen logg för idag.")
        return
    # Skapa CSV text
    out = ["time,symbol,side,price,qty,pnl"]
    for r in rows:
        # r kan vara 5 eller 6 kolumner
        if len(r) == 5:
            out.append(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},")
        else:
            out.append(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}")
    txt = "\n".join(out)
    update.message.reply_document(document=("log.csv", txt.encode("utf-8")))

def cmd_keepalive_on(update: Update, ctx: CallbackContext):
    ENGINE["keepalive"] = True
    update.message.reply_text("Keepalive: ON")

def cmd_keepalive_off(update: Update, ctx: CallbackContext):
    ENGINE["keepalive"] = False
    update.message.reply_text("Keepalive: OFF")

def cmd_panic(update: Update, ctx: CallbackContext):
    # stäng mock pos och stoppa
    for s in ENGINE["symbols"]:
        st = STATE[s]
        pos = st["position"]
        if pos:
            c = price_tick(s) or pos["entry"]
            if ENGINE["mode"] == "mock":
                pnl = (c - pos["entry"]) * pos["qty"]
                PNL["mock"] += pnl
                LOGQ.put([now_utc().isoformat(), s, "MOCK-FORCE-SELL", c, pos["qty"], pnl])
            else:
                if KUCOIN_KEY:
                    _ = account_market_sell_live(s, pos["qty"])
                    pnl = (c - pos["entry"]) * pos["qty"]
                    PNL["live"] += pnl
                    LOGQ.put([now_utc().isoformat(), s, "LIVE-FORCE-SELL", c, pos["qty"], pnl])
        st["position"] = None
    ENGINE["enabled"] = False
    update.message.reply_text("PANIC: alla positioner stängda och motor stoppad.")

# -------------------------
# Keepalive / Ping
# -------------------------
def job_keepalive(context: CallbackContext):
    if not ENGINE["keepalive"]:
        return
    url = f"{PUBLIC_URL}/healthz" if PUBLIC_URL else "https://render.com"
    try:
        requests.get(url, timeout=5)
    except Exception:
        pass

def job_scan(context: CallbackContext):
    try:
        tick_scan()
    except Exception as e:
        log.warning("job_scan: %s", e)

# -------------------------
# FastAPI (web)
# -------------------------
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def index():
    return "mporbbot online"

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# -------------------------
# Start Telegram + Jobs
# -------------------------
def run_bot():
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN saknas.")
        return

    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Kommandon
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("engine_start", cmd_engine_start))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe))
    dp.add_handler(CommandHandler("trail", cmd_trail))
    dp.add_handler(CommandHandler("symbols", cmd_symbols, pass_args=True))
    dp.add_handler(CommandHandler("set_ai", cmd_set_ai, pass_args=True))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("export_csv", cmd_export_csv))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dp.add_handler(CommandHandler("panic", cmd_panic))

    # Callbacks
    dp.add_handler(CallbackQueryHandler(cb_confirm_start, pattern=r"^confirm_start:"))
    dp.add_handler(CallbackQueryHandler(cb_timeframe, pattern=r"^tf:"))
    dp.add_handler(CallbackQueryHandler(cb_trail, pattern=r"^trail:"))

    # Jobbar
    jq = updater.job_queue
    jq.run_repeating(job_scan, interval=10, first=5)
    jq.run_repeating(job_keepalive, interval=120, first=10)  # varannan minut

    # Kör polling i separat tråd så FastAPI kan svara
    threading.Thread(target=updater.start_polling, daemon=True).start()
    log.info("Telegram-bot startad.")

# -------------------------
# Uvicorn entrypoint
# -------------------------
# Gör så Render (uvicorn main:app) även startar Telegram-motorn.
run_once = False
@app.on_event("startup")
def on_startup():
    global run_once
    if not run_once:
        run_once = True
        run_bot()
        log.info("Startup klar.")

# Lokal körning:
if __name__ == "__main__":
    # För lokal test: uvicorn main:app --host 0.0.0.0 --port 8000
    run_bot()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
