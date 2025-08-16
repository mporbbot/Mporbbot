# Mp ORBbot – FULL Web Service (Render Free) + Telegram (PTB 13.15)
# Long-only spot på KuCoin • ORB per ny candle • SL = entry-candlens low och flyttas upp till varje ny candles low
# Mock & Live • Skatteverket-CSV • Backtest • Panic • Keepalive • AUTHORIZED_USER_ID

import os, csv, json, time, hmac, base64, hashlib, threading, logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# -------------- Konfiguration via ENV --------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0") or "0")

KU_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KU_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KU_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KU_API_KEY_VERSION = "2"  # KuCoin v2-signering

# Strategi/parametrar
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,XRPUSDT,LINKUSDT")
DEFAULT_TIMEFRAME = os.getenv("TIMEFRAME", "3min")  # 1min / 3min / 5min
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_RATE", "0.001"))  # 0.1% per sida default
SELF_URL = os.getenv("SELF_URL", "")  # valfri keepalive-URL

# AI-lägen påverkar minsta kropp på referens-candle
AI_MODE = "neutral"
MIN_ORB_PCT = {"aggressiv": 0.0005, "neutral": 0.001, "försiktig": 0.002}
TICK_EPS = float(os.getenv("TICK_EPS", "1e-8"))

# -------------- Globala states --------------
UTC = timezone.utc
app = FastAPI(title="Mp ORBbot", version="1.0.0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mporbbot")

STATE = {
    "mode": "MOCK",  # "MOCK" eller "LIVE"
    "engine_on": False,
    "symbols": [s.strip().upper() for s in DEFAULT_SYMBOLS.split(",") if s.strip()],
    "timeframe": DEFAULT_TIMEFRAME,
    "pnl_day_mock": 0.0,
    "pnl_day_live": 0.0,
}

# Per symbol: ref (senast stängda candle), och pågående position
# {symbol: {"ref_ts": int, "ref_high": float, "ref_low": float,
#           "entry": float|None, "stop": float|None, "qty": float|0}}
SYMBOL_STATE: Dict[str, Dict] = {}

PENDING_CONFIRM = {"type": None}  # "MOCK" eller "LIVE"
KEEPALIVE = {"on": False, "last": 0.0}

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
MOCK_CSV = os.path.join(LOG_DIR, "mock_trade_log.csv")
REAL_CSV = os.path.join(LOG_DIR, "real_trade_log.csv")
ERROR_LOG = os.path.join(LOG_DIR, "error_log.txt")
POS_FILE = os.path.join(LOG_DIR, "positions.json")

last_telegram_ok = time.time()

# -------------- Hjälpare --------------
def now_iso() -> str:
    return datetime.now(UTC).isoformat()

def dash_pair(symbol: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    if "-" in symbol: return symbol
    if symbol.endswith("USDT"):
        return symbol[:-4] + "-USDT"
    return symbol

def restrict(update: Update) -> bool:
    return (AUTHORIZED_USER_ID == 0) or (update.effective_user and update.effective_user.id == AUTHORIZED_USER_ID)

def write_error(msg: str):
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{now_iso()} {msg}\n")
    except Exception:
        pass

def save_positions():
    try:
        with open(POS_FILE, "w", encoding="utf-8") as f:
            json.dump(SYMBOL_STATE, f)
    except Exception as e:
        write_error(f"save_positions: {e}")

def load_positions():
    global SYMBOL_STATE
    try:
        if os.path.exists(POS_FILE):
            with open(POS_FILE, "r", encoding="utf-8") as f:
                SYMBOL_STATE = json.load(f)
    except Exception as e:
        write_error(f"load_positions: {e}")

def ensure_symbol(symbol: str):
    if symbol not in SYMBOL_STATE:
        SYMBOL_STATE[symbol] = {
            "ref_ts": 0,
            "ref_high": None,
            "ref_low": None,
            "entry": None,
            "stop": None,
            "qty": 0.0,
        }

# -------------- KuCoin REST --------------
BASE = "https://api.kucoin.com"

def ku_get(url: str, params=None) -> dict:
    try:
        r = requests.get(url, params=params or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        write_error(f"ku_get {url} {e}")
        return {}

def ku_headers(method: str, endpoint: str, body_str: str = "") -> dict:
    ts = str(int(time.time() * 1000))
    prehash = ts + method.upper() + endpoint + body_str
    sign = base64.b64encode(hmac.new(KU_API_SECRET.encode(), prehash.encode(), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(KU_API_SECRET.encode(), KU_API_PASSPHRASE.encode(), hashlib.sha256).digest()).decode()
    return {
        "KC-API-KEY": KU_API_KEY,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": KU_API_KEY_VERSION,
        "Content-Type": "application/json",
    }

def ku_private(method: str, endpoint: str, body: Optional[dict] = None) -> dict:
    import json as _json
    body_str = _json.dumps(body) if body else ""
    headers = ku_headers(method, endpoint, body_str)
    url = BASE + endpoint
    try:
        if method.upper() == "POST":
            r = requests.post(url, headers=headers, data=body_str, timeout=15)
        else:
            r = requests.get(url, headers=headers, params=body or {}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        write_error(f"ku_private {method} {endpoint} {e} {r.text if 'r' in locals() else ''}")
        return {}

def ku_place_market_buy(symbol_dash: str, funds_usdt: float) -> bool:
    if not (KU_API_KEY and KU_API_SECRET and KU_API_PASSPHRASE): return False
    body = {
        "clientOid": str(int(time.time()*1000)),
        "side": "buy",
        "symbol": symbol_dash,
        "type": "market",
        "funds": f"{funds_usdt:.2f}",  # USDT
    }
    res = ku_private("POST", "/api/v1/orders", body)
    return res.get("code") == "200000"

def ku_place_market_sell(symbol_dash: str, size: float) -> bool:
    if not (KU_API_KEY and KU_API_SECRET and KU_API_PASSPHRASE): return False
    body = {
        "clientOid": str(int(time.time()*1000)),
        "side": "sell",
        "symbol": symbol_dash,
        "type": "market",
        "size": f"{max(size, 0):.6f}",
    }
    res = ku_private("POST", "/api/v1/orders", body)
    return res.get("code") == "200000"

def ku_last_price(symbol: str) -> float:
    sym = dash_pair(symbol)
    data = ku_get(f"{BASE}/api/v1/market/orderbook/level1", {"symbol": sym})
    try:
        return float(data["data"]["price"])
    except Exception:
        return 0.0

def ku_candles(symbol: str, timeframe: str, limit: int = 3) -> List[List]:
    # KuCoin returns: [time, open, close, high, low, volume, turnover]
    sym = dash_pair(symbol)
    data = ku_get(f"{BASE}/api/v1/market/candles", {"symbol": sym, "type": timeframe})
    arr = data.get("data", [])[:max(3, limit)]
    # sort by time ascending
    arr.sort(key=lambda x: int(x[0]))
    return arr

# -------------- CSV-logg --------------
def log_trade(live: bool, symbol: str, entry: float, exit_price: float, qty: float, stop_exit: bool, comment: str = ""):
    fees_in = entry * qty * FEE_PER_SIDE
    fees_out = exit_price * qty * FEE_PER_SIDE
    pnl = (exit_price - entry) * qty - fees_in - fees_out
    if live: STATE["pnl_day_live"] += pnl
    else:    STATE["pnl_day_mock"] += pnl

    row = {
        "timestamp_utc": now_iso(),
        "mode": "LIVE" if live else "MOCK",
        "symbol": symbol,
        "side": "LONG",
        "entry_price": f"{entry:.8f}",
        "exit_price": f"{exit_price:.8f}",
        "qty": f"{qty:.6f}",
        "fee_entry": f"{fees_in:.6f}",
        "fee_exit": f"{fees_out:.6f}",
        "pnl_usdt": f"{pnl:.6f}",
        "reason": "stop" if stop_exit else comment,
    }
    path = REAL_CSV if live else MOCK_CSV
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists: w.writeheader()
        w.writerow(row)

# -------------- ORB-strategi --------------
def update_ref(symbol: str, timeframe: str) -> Optional[int]:
    """
    Sätt referens till senast STÄNGDA candle om den har tillräcklig kropp (AI_MODE).
    Returnerar ref_ts eller None om inget nytt.
    """
    ensure_symbol(symbol)
    st = SYMBOL_STATE[symbol]
    arr = ku_candles(symbol, timeframe, limit=3)
    if len(arr) < 2: return None
    prev = arr[-2]  # senast stängd
    ref_ts = int(prev[0])
    if st["ref_ts"] == ref_ts:  # oförändrad
        return ref_ts

    # Kolla kroppsstorlek
    o = float(prev[1]); c = float(prev[2]); h = float(prev[3]); l = float(prev[4])
    body = abs(c - o)
    if body < max(o, 1e-9) * MIN_ORB_PCT.get(AI_MODE, 0.001):
        # för liten candle → hoppa över att uppdatera referens (ingen ORB)
        return None

    st["ref_ts"] = ref_ts
    st["ref_high"] = h
    st["ref_low"]  = l
    # Flytta SL på befintlig position till nya candle-low (uppåt)
    if st["entry"] is not None and st["stop"] is not None:
        st["stop"] = max(st["stop"], l - TICK_EPS)
    save_positions()
    return ref_ts

def step_symbol(symbol: str):
    st = SYMBOL_STATE.get(symbol) or ensure_symbol(symbol)
    tf = STATE["timeframe"]

    # Uppdatera referens
    update_ref(symbol, tf)

    # Hämta livepris
    px = ku_last_price(symbol)
    if px <= 0: return
    st = SYMBOL_STATE[symbol]
    ref_h = st["ref_high"]; ref_l = st["ref_low"]

    # ENTRY: bryter över referens-high, endast long
    if st["entry"] is None and ref_h and px > ref_h + TICK_EPS:
        entry = px
        stop = (ref_l or px) - TICK_EPS
        live = (STATE["mode"] == "LIVE") and (KU_API_KEY and KU_API_SECRET and KU_API_PASSPHRASE)
        qty = (MOCK_TRADE_USDT / entry)
        ok = True
        if live:
            ok = ku_place_market_buy(dash_pair(symbol), funds_usdt=MOCK_TRADE_USDT)
        if ok:
            st["entry"] = entry
            st["stop"] = stop
            st["qty"] = qty
            save_positions()
            try:
                updater.bot.send_message(AUTHORIZED_USER_ID, f"✅ ENTRY {symbol} {STATE['mode']}\n@ {entry:.6f} | SL {stop:.6f}")
            except Exception: pass
            log.info(f"ENTRY {symbol} {STATE['mode']} @ {entry:.6f} SL {stop:.6f}")

    # EXIT: pris <= stop
    if st["entry"] is not None and st["stop"] is not None and px <= st["stop"] + TICK_EPS:
        exit_px = px
        qty = st["qty"] or 0.0
        live = (STATE["mode"] == "LIVE") and (KU_API_KEY and KU_API_SECRET and KU_API_PASSPHRASE)
        ok = True
        if live and qty > 0:
            ok = ku_place_market_sell(dash_pair(symbol), size=qty)
        if ok:
            log_trade(live, symbol, st["entry"], exit_px, qty, stop_exit=True)
            try:
                pnl_show = STATE["pnl_day_live"] if live else STATE["pnl_day_mock"]
                updater.bot.send_message(AUTHORIZED_USER_ID, f"⏹ EXIT {symbol} {STATE['mode']}\n@ {exit_px:.6f}\nDagens PnL: {pnl_show:.4f} USDT")
            except Exception: pass
            st["entry"] = None; st["stop"] = None; st["qty"] = 0.0
            save_positions()

# -------------- Motor --------------
ENGINE_STOP = threading.Event()

def engine_loop():
    global last_telegram_ok
    log.info("Engine started")
    while not ENGINE_STOP.is_set():
        try:
            if STATE["engine_on"]:
                for s in STATE["symbols"]:
                    ensure_symbol(s)
                    step_symbol(s)
            # Fail-safe: om vi inte skickat/lyckats interagera med Telegram på 5 min → stoppa engine
            if time.time() - last_telegram_ok > 300:
                STATE["engine_on"] = False
                log.warning("Telegram inaktivitet >5min → stoppar motor (failsafe)")
            # Keepalive
            if KEEPALIVE["on"] and SELF_URL:
                now = time.time()
                if now - KEEPALIVE["last"] > 60:
                    KEEPALIVE["last"] = now
                    try: requests.get(SELF_URL, timeout=5)
                    except Exception: pass
        except Exception as e:
            write_error(f"engine_loop: {e}")
        time.sleep(2)
    log.info("Engine stopped")

def start_engine():
    STATE["engine_on"] = True

def stop_engine():
    STATE["engine_on"] = False

# -------------- Telegram --------------
updater: Optional[Updater] = None

def tg_send(txt: str):
    global last_telegram_ok
    if not updater: return
    try:
        updater.bot.send_message(AUTHORIZED_USER_ID, txt)
        last_telegram_ok = time.time()
    except Exception as e:
        write_error(f"tg_send: {e}")

def cmd_help(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    txt = (
        "*Mp ORBbot*\n"
        "/status – visa status\n"
        "/set_ai <neutral|aggressiv|försiktig>\n"
        "/start_mock – starta MOCK (kräver 'JA')\n"
        "/start_live – starta LIVE (kräver 'JA')\n"
        "/engine_start – starta motor\n"
        "/engine_stop – stoppa motor\n"
        "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
        "/timeframe 1min|3min|5min – byt timeframe\n"
        "/pnl – dagens PnL\n"
        "/reset_pnl – nollställ PnL idag\n"
        "/export_csv – skicka loggar\n"
        "/get_log – fel-logg\n"
        "/panic – sälj allt och stoppa\n"
        "/help – den här listan"
    )
    update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

def cmd_status(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    lines = []
    for s, st in SYMBOL_STATE.items():
        pos = "✔" if st.get("entry") else "–"
        lines.append(f"{s}: pos={pos} refH={st.get('ref_high')} refL={st.get('ref_low')} SL={st.get('stop')}")
    txt = (
        f"🤖 *Status*\n"
        f"Läge: *{STATE['mode']}*\nMotor: *{'På' if STATE['engine_on'] else 'Av'}*\n"
        f"AI: *{AI_MODE}*\nTF: *{STATE['timeframe']}*\n"
        f"Symbols: {', '.join(STATE['symbols'])}\n"
        f"PnL MOCK: {STATE['pnl_day_mock']:.4f} | LIVE: {STATE['pnl_day_live']:.4f}\n\n" +
        ("\n".join(lines) if lines else "(ingen symbolinfo)")
    )
    update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

def cmd_set_ai(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    global AI_MODE
    if not ctx.args: update.message.reply_text("Använd: /set_ai neutral|aggressiv|försiktig"); return
    val = ctx.args[0].strip().lower()
    if val not in MIN_ORB_PCT: update.message.reply_text("Endast neutral|aggressiv|försiktig"); return
    AI_MODE = val; update.message.reply_text(f"AI-läge: {AI_MODE}")

def cmd_symbols(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    if not ctx.args: update.message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,..."); return
    raw = " ".join(ctx.args).replace(" ", "")
    syms = [s for s in raw.split(",") if s]
    if not syms: update.message.reply_text("Inga symbols"); return
    STATE["symbols"] = [s.upper() for s in syms]
    for s in STATE["symbols"]: ensure_symbol(s)
    save_positions()
    update.message.reply_text("Symbols uppdaterade.")

def cmd_timeframe(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    if not ctx.args: update.message.reply_text("Använd: /timeframe 1min|3min|5min"); return
    tf = ctx.args[0].lower()
    if tf not in ("1min","3min","5min"): update.message.reply_text("Endast 1min|3min|5min"); return
    STATE["timeframe"] = tf; update.message.reply_text(f"Timeframe: {tf}")

def cmd_engine_start(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    start_engine(); update.message.reply_text("Engine: startad")

def cmd_engine_stop(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    stop_engine(); update.message.reply_text("Engine: stoppad")

def cmd_pnl(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    update.message.reply_text(f"PnL idag → MOCK: {STATE['pnl_day_mock']:.4f} | LIVE: {STATE['pnl_day_live']:.4f}")

def cmd_reset_pnl(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    STATE["pnl_day_mock"] = 0.0; STATE["pnl_day_live"] = 0.0
    update.message.reply_text("PnL nollställd.")

def cmd_export_csv(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    try:
        if os.path.exists(MOCK_CSV): update.message.reply_document(open(MOCK_CSV, "rb"))
        if os.path.exists(REAL_CSV): update.message.reply_document(open(REAL_CSV, "rb"))
    except Exception as e:
        write_error(f"export_csv: {e}")
        update.message.reply_text("Kunde inte skicka loggar.")

def cmd_get_log(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    if os.path.exists(ERROR_LOG):
        update.message.reply_document(open(ERROR_LOG, "rb"))
    else:
        update.message.reply_text("Ingen fel-logg ännu.")

def cmd_panic(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    # Stäng alla positioner till marknadspris och stoppa motor
    for s, st in list(SYMBOL_STATE.items()):
        if st.get("entry") is not None and (st.get("qty") or 0) > 0:
            px = ku_last_price(s)
            if px > 0:
                live = (STATE["mode"] == "LIVE") and (KU_API_KEY and KU_API_SECRET and KU_API_PASSPHRASE)
                if live: ku_place_market_sell(dash_pair(s), size=st["qty"])
                log_trade(live, s, st["entry"], px, st["qty"], stop_exit=False, comment="panic")
                st["entry"] = None; st["stop"] = None; st["qty"] = 0.0
    stop_engine()
    save_positions()
    update.message.reply_text("PANIC: Alla positioner stängda och engine stoppad.")

def cmd_start_mock(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    PENDING_CONFIRM["type"] = "MOCK"
    update.message.reply_text("Vill du starta MOCK? Svara JA")

def cmd_start_live(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    if not (KU_API_KEY and KU_API_SECRET and KU_API_PASSPHRASE):
        update.message.reply_text("LIVE kräver KuCoin API-nycklar i Render Environment.")
        return
    PENDING_CONFIRM["type"] = "LIVE"
    update.message.reply_text("Vill du starta LIVE? Svara JA")

def cmd_text(update: Update, ctx: CallbackContext):
    global last_telegram_ok
    last_telegram_ok = time.time()
    if not restrict(update): return
    txt = (update.message.text or "").strip().lower()
    if txt == "ja" and PENDING_CONFIRM["type"]:
        STATE["mode"] = PENDING_CONFIRM["type"]
        PENDING_CONFIRM["type"] = None
        update.message.reply_text(f"Läge: {STATE['mode']}")

def start_bot():
    global updater
    if not TELEGRAM_TOKEN:
        log.warning("Ingen TELEGRAM_TOKEN – Telegram-bot startas inte.")
        return
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("set_ai", cmd_set_ai, pass_args=True))
    dp.add_handler(CommandHandler("symbols", cmd_symbols, pass_args=True))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe, pass_args=True))
    dp.add_handler(CommandHandler("engine_start", cmd_engine_start))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("export_csv", cmd_export_csv))
    dp.add_handler(CommandHandler("get_log", cmd_get_log))
    dp.add_handler(CommandHandler("panic", cmd_panic))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, cmd_text))
    updater.start_polling(drop_pending_updates=True, timeout=20)
    log.info("Telegram-bot startad.")

# -------------- FastAPI endpoints --------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "Mp ORBbot up"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.on_event("startup")
def on_startup():
    load_positions()
    start_bot()
    threading.Thread(target=engine_loop, daemon=True).start()

# Lokal körning: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
