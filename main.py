import os
import hmac
import hashlib
import base64
import time
import json
import csv
import threading
import queue
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from telegram import (
    Bot, Update, InlineKeyboardMarkup, InlineKeyboardButton, ParseMode
)
from telegram.ext import (
    Updater, CallbackContext, CommandHandler, MessageHandler,
    Filters, CallbackQueryHandler, ConversationHandler
)

# ========= Config & Globals =========
TZ = timezone.utc

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
DEFAULT_SYMBOLS = os.getenv("DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT").replace(" ", "")
DEFAULT_TF = int(os.getenv("DEFAULT_TF", "1"))  # minutes
KEEPALIVE_DEFAULT = os.getenv("KEEPALIVE", "1") == "1"
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")  # Render brukar sätta denna

app = FastAPI()

# Telegram
bot = Bot(token=TELEGRAM_TOKEN)
updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
dispatcher = updater.dispatcher

# States for conversation (custom trailing)
EXPECT_TRAIL_INPUT = 1001

# Engine States
ENGINE = {
    "enabled": False,     # motor på/av
    "mode": "mock",       # "mock" eller "live"
    "ai": "aggressiv",    # bara etikett till status
    "tf": DEFAULT_TF,     # timeframe i minuter
    "symbols": [s for s in DEFAULT_SYMBOLS.split(",") if s],
    "keepalive": KEEPALIVE_DEFAULT,
    "ping_thread": None,
    "ping_stop": threading.Event(),
}

# Per-symbol state
# position: None eller dict{entry, qty, stop, high, trail_active}
STATE: Dict[str, Dict] = {}

# PnL (dag)
PNL = {"mock": 0.0, "live": 0.0, "day": datetime.now(TZ).date()}

# Trailing defaults (trig/offset/min) – i decimal (0.009 = 0,9 %)
TRAIL = {"trig": 0.009, "offset": 0.002, "min": 0.007}

# Säkerhetskö
LOG_Q: "queue.Queue[str]" = queue.Queue()


# ========= Hjälpfunktioner =========
def now_utc() -> datetime:
    return datetime.now(TZ)


def kucoin_public(base: str = "https://api.kucoin.com") -> str:
    return base


def kucoin_spot_base() -> str:
    return "https://api.kucoin.com"


def sign_kucoin(method: str, path: str, body: str = "") -> Dict[str, str]:
    """
    Enkel KuCoin-signering för REST v1 (spot).
    """
    ts = str(int(time.time() * 1000))
    str_to_sign = ts + method.upper() + path + body
    sig = base64.b64encode(
        hmac.new(KUCOIN_API_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).digest()
    ).decode()
    passphrase = base64.b64encode(
        hmac.new(KUCOIN_API_SECRET.encode(), KUCOIN_API_PASSPHRASE.encode(), hashlib.sha256).digest()
    ).decode()

    return {
        "KC-API-KEY": KUCOIN_API_KEY,
        "KC-API-SIGN": sig,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }


def kline(symbol: str, tf_min: int, limit: int = 120) -> List[Dict]:
    """
    Hämta klines (candles) från KuCoins publika endpoint.
    tf_min stöds: 1,3,5,15, ... (KuCoin "type": 1min,3min,5min,15min etc)
    """
    tf_map = {
        1: "1min", 3: "3min", 5: "5min", 15: "15min",
        30: "30min", 60: "1hour", 240: "4hour"
    }
    tft = tf_map.get(tf_min, "1min")
    url = f"{kucoin_public()}/api/v1/market/candles"
    # KuCoin vill ha symbol som ADA-USDT (med bindestreck)
    sym = symbol.replace("USDT", "-USDT")
    params = {"type": tft, "symbol": sym}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    # data är senaste först; normalisera och vänd i tidsordning
    candles = []
    for row in data:
        # [time, open, close, high, low, volume, turnover]
        candles.append({
            "ts": int(row[0])*1000,
            "open": float(row[1]),
            "close": float(row[2]),
            "high": float(row[3]),
            "low": float(row[4]),
            "volume": float(row[5]),
        })
    candles.sort(key=lambda x: x["ts"])
    return candles[-limit:]


def last_price(symbol: str) -> float:
    url = f"{kucoin_public()}/api/v1/market/orderbook/level1"
    sym = symbol.replace("USDT", "-USDT")
    r = requests.get(url, params={"symbol": sym}, timeout=8)
    r.raise_for_status()
    return float(r.json()["data"]["price"])


def qty_from_usdt(symbol: str, usdt: float) -> float:
    px = last_price(symbol)
    if px <= 0:
        return 0.0
    # kucoin min size varierar; förenklad avrundning
    qty = round(usdt / px, 6)
    return max(qty, 0.000001)


def live_market_buy(symbol: str, usdt: float) -> Optional[Dict]:
    """
    Enkel marknadsköp på KuCoin (spot). Returnerar api-svar eller None vid fel.
    """
    try:
        qty = qty_from_usdt(symbol, usdt)
        path = "/api/v1/orders"
        url = kucoin_spot_base() + path
        body = json.dumps({
            "clientOid": f"mp_{int(time.time()*1000)}",
            "side": "buy",
            "symbol": symbol.replace("USDT", "-USDT"),
            "type": "market",
            "funds": f"{usdt:.2f}",  # KuCoin kräver 'funds' för market BUY
        })
        headers = sign_kucoin("POST", path, body)
        r = requests.post(url, headers=headers, data=body, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        LOG_Q.put(f"[LIVE] BUY error {symbol}: {e}")
        return None


def live_market_sell(symbol: str, qty: float) -> Optional[Dict]:
    try:
        path = "/api/v1/orders"
        url = kucoin_spot_base() + path
        body = json.dumps({
            "clientOid": f"mp_{int(time.time()*1000)}",
            "side": "sell",
            "symbol": symbol.replace("USDT", "-USDT"),
            "type": "market",
            "size": f"{qty:.6f}",
        })
        headers = sign_kucoin("POST", path, body)
        r = requests.post(url, headers=headers, data=body, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        LOG_Q.put(f"[LIVE] SELL error {symbol}: {e}")
        return None


def ensure_state(symbols: List[str]):
    for s in symbols:
        if s not in STATE:
            STATE[s] = {
                "position": None,     # dict: entry, qty, stop, high, qty_live?
                "last_candle_id": None,
            }


def new_candle_ready(ts_ms: int, tf_min: int, last_id: Optional[int]) -> Tuple[bool, int]:
    # Candle id som hel minuttid
    tf_ms = tf_min * 60_000
    cid = (ts_ms // tf_ms) * tf_ms
    return (last_id is None) or (cid > last_id), cid


def update_trailing(pos: Dict, high_price: float):
    """
    Uppdatera trailing baserat på globala TRAIL.
    pos innehåller: entry, qty, stop, high
    """
    entry = pos["entry"]
    # uppdatera high
    pos["high"] = max(pos.get("high", entry), high_price)
    best_ret = (pos["high"] - entry) / entry

    # Om vinst över trig – aktivera trailing
    if best_ret >= TRAIL["trig"]:
        # target att låsa minst 'min' (från entry)
        locked_price = entry * (1.0 + TRAIL["min"])
        # trailing-nivå: high - offset
        trail_price = pos["high"] * (1.0 - TRAIL["offset"])
        new_stop = max(pos.get("stop", 0.0), locked_price, trail_price)
        # aldrig över high, men det blir det inte med formeln
        pos["stop"] = new_stop


def follow_next_candle_bottom(pos: Dict, candle_low: float):
    """
    'Stoppen följer nästa candle botten' – uppdatera med candle_low om det höjer vår stop.
    """
    if pos.get("stop") is None:
        pos["stop"] = candle_low
    else:
        pos["stop"] = max(pos["stop"], candle_low)


def maybe_open_long(symbol: str, candles: List[Dict], mode: str):
    """
    Ny candle, om senaste (föregående close) var grön & bryter förra högsta – enter long.
    Enkel ORB-liknande för 1-5m.
    """
    if len(candles) < 3:
        return
    c0, c1 = candles[-2], candles[-1]  # c0=föregående komplett, c1=senaste pågående
    # grön candle och close över föregående high => ha breakout
    if c0["close"] > c0["open"] and c0["close"] >= c0["high"]:
        if STATE[symbol]["position"] is None:
            px = c0["close"]
            if mode == "mock":
                qty = round(MOCK_TRADE_USDT / px, 6)
                STATE[symbol]["position"] = {
                    "entry": px, "qty": qty, "stop": None, "high": px
                }
                LOG_Q.put(f"[MOCK] BUY {symbol} @ {px:.6f} qty={qty}")
            else:
                res = live_market_buy(symbol, MOCK_TRADE_USDT)
                if res:
                    qty = round(MOCK_TRADE_USDT / px, 6)  # förenklad
                    STATE[symbol]["position"] = {
                        "entry": px, "qty": qty, "stop": None, "high": px, "live": True
                    }
                    LOG_Q.put(f"[LIVE] BUY {symbol} @ ~{px:.6f} qty≈{qty}")


def maybe_exit(symbol: str, candle: Dict, mode: str):
    """
    Exit om close < stop. Stoppen uppdateras av trailing och candle-botten-följ.
    """
    pos = STATE[symbol]["position"]
    if not pos:
        return
    # uppdatera trailing (baserat på high)
    update_trailing(pos, candle["high"])
    # följ nästa candles botten
    follow_next_candle_bottom(pos, candle["low"])

    stop = pos.get("stop")
    if stop and candle["close"] <= stop:
        # sälj på close som approximation
        exit_px = candle["close"]
        pnl = (exit_px - pos["entry"]) * pos["qty"]
        PNL[ENGINE["mode"]] += pnl
        if mode == "mock":
            LOG_Q.put(f"[MOCK] SELL {symbol} @ {exit_px:.6f} pnl={pnl:.4f}")
        else:
            live_market_sell(symbol, pos["qty"])
            LOG_Q.put(f"[LIVE] SELL {symbol} @ ~{exit_px:.6f} pnl≈{pnl:.4f}")
        STATE[symbol]["position"] = None


def engine_loop():
    LOG_Q.put("[ENGINE] Startar motor")
    while ENGINE["enabled"]:
        start = time.time()
        try:
            ensure_state(ENGINE["symbols"])
            for s in ENGINE["symbols"]:
                kl = kline(s, ENGINE["tf"], 120)
                if not kl:
                    continue
                ready, cid = new_candle_ready(kl[-1]["ts"], ENGINE["tf"], STATE[s]["last_candle_id"])
                if ready:
                    # arbeta på föregående kompletta candle (kl[-2])
                    if len(kl) >= 2:
                        prev = kl[-2]
                        maybe_exit(s, prev, ENGINE["mode"])
                        maybe_open_long(s, kl, ENGINE["mode"])
                    STATE[s]["last_candle_id"] = cid
        except Exception as e:
            LOG_Q.put(f"[ENGINE] fel: {e}")

        # flush log till daglig csv
        flush_log_to_csv()

        # vilotid till nästa halv-minut
        spent = time.time() - start
        sleep_s = max(2.0, 10.0 - spent)
        time.sleep(sleep_s)

    LOG_Q.put("[ENGINE] Motor stoppad")


def flush_log_to_csv():
    """
    Skriv allt i LOG_Q till dagsfil (UTC).
    """
    fn = f"logs_{now_utc().strftime('%Y-%m-%d')}.csv"
    rows = []
    while True:
        try:
            rows.append(LOG_Q.get_nowait())
        except queue.Empty:
            break
    if not rows:
        return
    newfile = not os.path.exists(fn)
    with open(fn, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["ts_utc", "event"])
        for line in rows:
            w.writerow([now_utc().isoformat(), line])


# ========= Keepalive (ping varannan minut) =========
def ping_self_loop():
    while not ENGINE["ping_stop"].is_set():
        try:
            if ENGINE["keepalive"] and RENDER_URL:
                requests.get(RENDER_URL, timeout=5)
        except Exception:
            pass
        ENGINE["ping_stop"].wait(120)  # varannan minut


def ensure_ping_thread():
    if ENGINE["ping_thread"] and ENGINE["ping_thread"].is_alive():
        return
    ENGINE["ping_stop"].clear()
    t = threading.Thread(target=ping_self_loop, daemon=True)
    ENGINE["ping_thread"] = t
    t.start()


# ========= Telegram UI =========
def build_main_kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🟢 Starta MOCK", callback_data="start_mock"),
            InlineKeyboardButton("🟣 Starta LIVE", callback_data="start_live"),
        ],
        [
            InlineKeyboardButton("⏱ Timeframe", callback_data="tf_menu"),
            InlineKeyboardButton("🎯 Trailing", callback_data="trail_menu"),
        ],
        [
            InlineKeyboardButton("▶️ Engine ON", callback_data="engine_on"),
            InlineKeyboardButton("⏹ Engine OFF", callback_data="engine_off"),
        ],
    ])


def build_tf_kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("1m", callback_data="tf_1"),
            InlineKeyboardButton("3m", callback_data="tf_3"),
            InlineKeyboardButton("5m", callback_data="tf_5"),
            InlineKeyboardButton("15m", callback_data="tf_15"),
        ],
        [InlineKeyboardButton("⬅️ Tillbaka", callback_data="back_main")]
    ])


def build_trailing_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡ 0.9/0.2 (min 0.7)", callback_data="trail_preset:0.009,0.002,0.007")],
        [InlineKeyboardButton("⚡ 0.7/0.15 (min 0.5)", callback_data="trail_preset:0.007,0.0015,0.005")],
        [InlineKeyboardButton("⚡ 1.2/0.25 (min 0.9)", callback_data="trail_preset:0.012,0.0025,0.009")],
        [
            InlineKeyboardButton("✏️ Eget", callback_data="trail_custom"),
            InlineKeyboardButton("⬅️ Tillbaka", callback_data="back_main")
        ],
    ])


def status_text() -> str:
    lines = []
    lines.append(f"Mode: *{ENGINE['mode']}*    Engine: *{'ON' if ENGINE['enabled'] else 'OFF'}*")
    lines.append(f"AI: *{ENGINE['ai']}*    TF: *{ENGINE['tf']}m*")
    lines.append(f"Symbols: {', '.join(ENGINE['symbols'])}")
    # pnl reset per dag
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
            lines.append(f"{s}: pos=✅  refH={pos.get('high', pos['entry']):.6f} "
                         f"stop={pos.get('stop','None') if pos.get('stop') is None else f'{pos['stop']:.6f}'}")
        else:
            lines.append(f"{s}: pos=❌")
    return "\n".join(lines)


# ----- Commands -----
def start_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Välkommen! Välj nedan:",
        reply_markup=build_main_kb()
    )


def help_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(
        "/status – visa status\n"
        "/engine_start – starta motor\n"
        "/engine_stop – stoppa motor\n"
        "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
        "/timeframe – byt tidsram\n"
        "/set_ai <neutral|aggressiv|försiktig>\n"
        "/pnl – visa dagens PnL\n"
        "/reset_pnl – nollställ PnL\n"
        "/export_csv – skicka dagens logg\n"
        "/keepalive_on | /keepalive_off – Render keepalive\n"
        "/panic – stäng alla pos och stoppa\n"
        "/trailing – sätt trig/offset/min"
    )


def status_cmd(update: Update, context: CallbackContext):
    ensure_state(ENGINE["symbols"])
    update.message.reply_text(
        status_text(),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_trailing_kb()  # visar trailing-snabbval direkt
    )
    # plus huvudmeny
    update.message.reply_text("Huvudmeny:", reply_markup=build_main_kb())


def pnl_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(f"PnL (dag): MOCK {PNL['mock']:.4f} | LIVE {PNL['live']:.4f}")


def reset_pnl_cmd(update: Update, context: CallbackContext):
    PNL["mock"] = 0.0
    PNL["live"] = 0.0
    PNL["day"] = now_utc().date()
    update.message.reply_text("PnL återställd för idag.")


def export_csv_cmd(update: Update, context: CallbackContext):
    fn = f"logs_{now_utc().strftime('%Y-%m-%d')}.csv"
    try:
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                bot.send_document(chat_id=update.effective_chat.id, document=f)
        else:
            update.message.reply_text("Ingen logg för idag ännu.")
    except Exception as e:
        update.message.reply_text(f"Kunde inte skicka: {e}")


def symbols_cmd(update: Update, context: CallbackContext):
    if context.args:
        ENGINE["symbols"] = [s.strip().upper() for s in " ".join(context.args).replace(",", " ").split()]
        ensure_state(ENGINE["symbols"])
        update.message.reply_text(f"Symbols uppdaterade: {', '.join(ENGINE['symbols'])}")
    else:
        update.message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,ADAUSDT")


def set_ai_cmd(update: Update, context: CallbackContext):
    if context.args:
        ENGINE["ai"] = context.args[0]
        update.message.reply_text(f"AI satt till: {ENGINE['ai']}")
    else:
        update.message.reply_text("Använd: /set_ai aggressiv|neutral|försiktig")


def tf_open_cmd(update: Update, context: CallbackContext):
    update.message.reply_text("Välj tidsram:", reply_markup=build_tf_kb())


def engine_start_cmd(update: Update, context: CallbackContext):
    if not ENGINE["enabled"]:
        ENGINE["enabled"] = True
        threading.Thread(target=engine_loop, daemon=True).start()
        update.message.reply_text("Engine startad.")
    else:
        update.message.reply_text("Engine kör redan.")


def engine_stop_cmd(update: Update, context: CallbackContext):
    ENGINE["enabled"] = False
    update.message.reply_text("Engine stoppad.")


def keepalive_on_cmd(update: Update, context: CallbackContext):
    ENGINE["keepalive"] = True
    ensure_ping_thread()
    update.message.reply_text("Keepalive: ON (ping varannan minut).")


def keepalive_off_cmd(update: Update, context: CallbackContext):
    ENGINE["keepalive"] = False
    update.message.reply_text("Keepalive: OFF.")


def panic_cmd(update: Update, context: CallbackContext):
    for s in ENGINE["symbols"]:
        STATE[s]["position"] = None
    ENGINE["enabled"] = False
    update.message.reply_text("PANIC: alla positioner stängda (mock) och motor stoppad.")


# ----- Trailing -----
def trailing_cmd(update: Update, context: CallbackContext):
    t, o, m = TRAIL["trig"], TRAIL["offset"], TRAIL["min"]
    update.message.reply_text(
        f"Aktuell trailing:\n"
        f"• Trig: {t*100:.2f}%\n"
        f"• Offset: {o*100:.2f}%\n"
        f"• Min vinst: {m*100:.2f}%\n\n"
        f"Skriv tre procenttal (komma eller punkt), t.ex:\n"
        f"`0.9, 0.2, 0.7` (motsvarar trig 0.9%, offset 0.2%, min 0.7%)",
        parse_mode=ParseMode.MARKDOWN
    )
    return EXPECT_TRAIL_INPUT


def trailing_set_from_text(txt: str) -> Optional[Tuple[float, float, float]]:
    txt = txt.replace("%", "").replace(" ", "")
    if "," in txt:
        parts = txt.split(",")
    else:
        parts = txt.split(";")
    if len(parts) != 3:
        return None
    try:
        trig = float(parts[0]) / 100.0 if float(parts[0]) > 1.0 else float(parts[0])
        off = float(parts[1]) / 100.0 if float(parts[1]) > 1.0 else float(parts[1])
        mn  = float(parts[2]) / 100.0 if float(parts[2]) > 1.0 else float(parts[2])
        return trig, off, mn
    except Exception:
        return None


def trailing_conv_msg(update: Update, context: CallbackContext):
    res = trailing_set_from_text(update.message.text.strip())
    if not res:
        update.message.reply_text("Kunde inte tolka. Prova t.ex: 0.9,0.2,0.7")
        return EXPECT_TRAIL_INPUT
    TRAIL["trig"], TRAIL["offset"], TRAIL["min"] = res
    update.message.reply_text(
        f"Ny trailing satt:\n"
        f"Trig {TRAIL['trig']*100:.2f}% | Avst {TRAIL['offset']*100:.2f}% | Min {TRAIL['min']*100:.2f}%"
    )
    return ConversationHandler.END


def trailing_cancel(update: Update, context: CallbackContext):
    update.message.reply_text("Avbrutet.")
    return ConversationHandler.END


def trailing_preset_cb(update: Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    payload = q.data.split(":")[1]
    trig, off, mn = [float(x) for x in payload.split(",")]
    TRAIL["trig"], TRAIL["offset"], TRAIL["min"] = trig, off, mn
    q.edit_message_text(
        f"Preset vald ✅  Trig {trig*100:.2f}% | Avst {off*100:.2f}% | Min {mn*100:.2f}%"
    )


def trailing_open_cb(update: Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    update.effective_message.reply_text("Öppnar trailing-inställningar…")
    trailing_cmd(update, context)


# ----- Callback meny -----
def callbacks(update: Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data

    if data == "start_mock":
        ENGINE["mode"] = "mock"
        q.edit_message_text("Läge satt till MOCK. Starta motorn med ▶️.")
    elif data == "start_live":
        ENGINE["mode"] = "live"
        q.edit_message_text("Läge satt till LIVE. Starta motorn med ▶️.")
    elif data == "engine_on":
        if not ENGINE["enabled"]:
            ENGINE["enabled"] = True
            threading.Thread(target=engine_loop, daemon=True).start()
            q.edit_message_text("Engine startad.")
        else:
            q.edit_message_text("Engine kör redan.")
    elif data == "engine_off":
        ENGINE["enabled"] = False
        q.edit_message_text("Engine stoppad.")
    elif data == "tf_menu":
        q.edit_message_text("Välj tidsram:", reply_markup=build_tf_kb())
    elif data.startswith("tf_"):
        tf = int(data.split("_")[1])
        ENGINE["tf"] = tf
        q.edit_message_text(f"Timeframe satt till {tf}m.", reply_markup=build_main_kb())
    elif data == "trail_menu":
        q.edit_message_text("Välj trailing:", reply_markup=build_trailing_kb())
    elif data == "back_main":
        q.edit_message_text("Huvudmeny:", reply_markup=build_main_kb())
    elif data.startswith("trail_preset:"):
        trailing_preset_cb(update, context)
    elif data == "trail_custom":
        trailing_open_cb(update, context)


# ========= FastAPI endpoints =========
@app.get("/", response_class=PlainTextResponse)
def root():
    return "ok"


@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "pong"


# ========= Register handlers & start =========
dispatcher.add_handler(CommandHandler("start", start_cmd))
dispatcher.add_handler(CommandHandler("help", help_cmd))
dispatcher.add_handler(CommandHandler("status", status_cmd))
dispatcher.add_handler(CommandHandler("engine_start", engine_start_cmd))
dispatcher.add_handler(CommandHandler("engine_stop", engine_stop_cmd))
dispatcher.add_handler(CommandHandler("symbols", symbols_cmd))
dispatcher.add_handler(CommandHandler("set_ai", set_ai_cmd))
dispatcher.add_handler(CommandHandler("pnl", pnl_cmd))
dispatcher.add_handler(CommandHandler("reset_pnl", reset_pnl_cmd))
dispatcher.add_handler(CommandHandler("export_csv", export_csv_cmd))
dispatcher.add_handler(CommandHandler("timeframe", tf_open_cmd))
dispatcher.add_handler(CommandHandler("keepalive_on", keepalive_on_cmd))
dispatcher.add_handler(CommandHandler("keepalive_off", keepalive_off_cmd))
dispatcher.add_handler(CommandHandler("panic", panic_cmd))

# Trailing conversation
conv_trailing = ConversationHandler(
    entry_points=[CommandHandler("trailing", trailing_cmd)],
    states={EXPECT_TRAIL_INPUT: [MessageHandler(Filters.text & ~Filters.command, trailing_conv_msg)]},
    fallbacks=[CommandHandler("cancel", trailing_cancel)],
    conversation_timeout=120,
)
dispatcher.add_handler(conv_trailing)

# Callbacks
dispatcher.add_handler(CallbackQueryHandler(callbacks, pattern=r"^(start_|engine_|tf_|trail_|back_main)"))
dispatcher.add_handler(CallbackQueryHandler(trailing_preset_cb, pattern=r"^trail_preset:"))
dispatcher.add_handler(CallbackQueryHandler(trailing_open_cb, pattern=r"^trail_custom$"))

# Start bot (polling) i bakgrunden när uvicorn startar
def start_bot_once():
    if not getattr(start_bot_once, "started", False):
        updater.start_polling(drop_pending_updates=True)
        ensure_ping_thread()
        start_bot_once.started = True

start_bot_once()
