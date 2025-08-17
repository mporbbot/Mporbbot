import os
import time
import json
import math
import queue
import httpx
import logging
import threading
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from telegram import (
    Bot, Update, ReplyKeyboardMarkup, KeyboardButton, ParseMode,
)
from telegram.ext import (
    Updater, CommandHandler, CallbackContext, Filters,
)

# =============== Logging ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mporbbot")

# =============== Config ===================
TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0"))
PING_URL = os.getenv("PING_URL", "").strip()
MOCK_SIZE_USDT = float(os.getenv("MOCK_TRADE_USDT", "100") or 100)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]
TF = "1m"           # visning + ORB på 1m
ENTRY_MODE = "CLOSE"  # CLOSE eller TICK
TRAIL_PCT = 0.009     # 0.90%
TRAIL_MIN = 0.002     # 0.20%
TRAIL_ABS_MIN = 0.007 # min 0.70%

MODE = "mock"       # "mock" eller "live"
ENGINE_ON = True
KEEPALIVE_ON = True

# =============== State ====================
# per symbol: orb, pos och PnL
class SymState:
    def __init__(self, sym):
        self.sym = sym
        self.orb_on = True
        self.orb_high = None
        self.orb_low = None
        self.entry_price = None
        self.stop = None
        self.position = False
        self.qty = 0.0
        self.day_pnl = 0.0
        self.last_close_ts = 0  # unix sec för senaste candle close
        self.last_alert_side = None  # för att undvika spam

SYMS = {s: SymState(s) for s in SYMBOLS}

# =============== FastAPI ==================
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "mporbbot ok"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

# =============== Utils ====================
def ku_symbol(s):
    # KuCoin vill ha bindestreck: ADA-USDT
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s

def now_utc():
    return datetime.now(tz=timezone.utc)

def fmt(f):
    try:
        return f"{float(f):.4f}"
    except Exception:
        return str(f)

def reply_menu():
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("/start_mock"), KeyboardButton("/start_live")],
            [KeyboardButton("/engine_start"), KeyboardButton("/engine_stop")],
            [KeyboardButton("/symbols"), KeyboardButton("/timeframe")],
            [KeyboardButton("/entry_mode"), KeyboardButton("/trailing")],
            [KeyboardButton("/pnl"), KeyboardButton("/reset_pnl")],
            [KeyboardButton("/export_csv"), KeyboardButton("/export_k4")],
            [KeyboardButton("/keepalive_on"), KeyboardButton("/keepalive_off")],
            [KeyboardButton("/status")],
        ],
        resize_keyboard=True
    )

# =============== Price feed (KuCoin 1m) ===
# Vi använder KuCoins publika klines för mock/live-synk
# GET /api/v1/market/candles?type=1min&symbol=ADA-USDT
KU_BASE = "https://api.kucoin.com"
def fetch_ku_1m_candle(sym: str):
    symbol = ku_symbol(sym)
    url = f"{KU_BASE}/api/v1/market/candles"
    params = {"type": "1min", "symbol": symbol}
    with httpx.Client(timeout=10) as cli:
        r = cli.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        # KuCoin returnerar listor som strings:
        # [ time, open, close, high, low, volume, turnover ]
        # sorteras senaste först -> vi tar index 0
        if not data or data.get("code") != "200000":
            raise RuntimeError(f"Ku resp fail: {data}")
        arr = data["data"]
        if not arr:
            return None
        # senaste candle
        c = arr[0]
        ts = int(c[0])  # epoch sek
        o, c_, h, l = map(float, [c[1], c[2], c[3], c[4]])
        return {"ts": ts, "open": o, "close": c_, "high": h, "low": l}

# =============== ORB & trading logic ======
def ensure_orb(sym: str, cndl):
    st = SYMS[sym]
    # "ORB sätts av senaste röda candle" – förenklad tolkning:
    # Om candle är röd, uppdatera orb till den röda candlens high/low.
    if cndl["close"] < cndl["open"]:
        st.orb_high = cndl["high"]
        st.orb_low = cndl["low"]
        logger.info(f"[{sym}] ORB reset: high={st.orb_high:.6f} low={st.orb_low:.6f}")

def try_entry(sym: str, cndl):
    st = SYMS[sym]
    if not st.orb_on or st.position:
        return None
    if st.orb_high is None or st.orb_low is None:
        return None
    # Entry-regel: CLOSE bryter över orb_high
    cond = cndl["close"] > st.orb_high if ENTRY_MODE == "CLOSE" else (cndl["high"] > st.orb_high)
    if cond:
        st.position = True
        st.entry_price = cndl["close"]
        st.qty = (MOCK_SIZE_USDT / st.entry_price) if MODE == "mock" else 0.0
        st.stop = st.orb_low
        st.last_alert_side = "long"
        txt = f"{sym}: ORB breakout BUY @ {fmt(st.entry_price)} | SL {fmt(st.stop)}"
        return ("buy", txt)
    return None

def update_trailing(sym: str, cndl):
    st = SYMS[sym]
    if not st.position:
        return
    # Trailing följer varje candle (uppåt):
    # stop = max(nuvarande stop, close - max(close*TRAIL_PCT, TRAIL_MIN, entry*TRAIL_ABS_MIN))
    step1 = cndl["close"] * TRAIL_PCT
    step2 = cndl["close"] * TRAIL_MIN
    step3 = st.entry_price * TRAIL_ABS_MIN
    new_stop = cndl["close"] - max(step1, step2, step3)
    if new_stop > st.stop:
        st.stop = new_stop

def try_exit(sym: str, cndl):
    st = SYMS[sym]
    if not st.position:
        return None
    # Exit om close går under stop (CLOSE-läge) eller low bryter (TICK-läge)
    breached = cndl["close"] < st.stop if ENTRY_MODE == "CLOSE" else (cndl["low"] < st.stop)
    if breached:
        exit_px = cndl["close"]
        pnl = (exit_px - st.entry_price) * st.qty
        st.day_pnl += pnl
        st.position = False
        st.qty = 0.0
        st.last_alert_side = "flat"
        txt = f"{sym}: EXIT @ {fmt(exit_px)} | PnL {fmt(pnl)} USDT | day {fmt(st.day_pnl)}"
        # nollställ stop efter exit
        st.stop = None
        st.entry_price = None
        return ("sell", txt)
    return None

# =============== Telegram =================
bot: Bot = None
updater: Updater = None

def delete_webhook_if_any():
    if not TOKEN:
        return
    try:
        with httpx.Client(timeout=10) as cli:
            url = f"https://api.telegram.org/bot{TOKEN}/deleteWebhook"
            r = cli.get(url, params={"drop_pending_updates": True})
            logger.info(f"deleteWebhook -> {r.text}")
    except Exception as e:
        logger.warning(f"deleteWebhook fail: {e}")

def send_admin(text):
    try:
        if ADMIN_CHAT_ID:
            bot.send_message(chat_id=ADMIN_CHAT_ID, text=text)
    except Exception as e:
        logger.warning(f"admin send fail: {e}")

def cmd_start_mock(update: Update, ctx: CallbackContext):
    global MODE
    MODE = "mock"
    update.message.reply_text("Mode: MOCK", reply_markup=reply_menu())

def cmd_start_live(update: Update, ctx: CallbackContext):
    global MODE
    MODE = "live"
    update.message.reply_text("Mode: LIVE", reply_markup=reply_menu())

def cmd_engine_start(update: Update, ctx: CallbackContext):
    global ENGINE_ON
    ENGINE_ON = True
    update.message.reply_text("Engine: ON")

def cmd_engine_stop(update: Update, ctx: CallbackContext):
    global ENGINE_ON
    ENGINE_ON = False
    update.message.reply_text("Engine: OFF")

def cmd_symbols(update: Update, ctx: CallbackContext):
    update.message.reply_text("Symbols: " + ",".join(SYMBOLS))

def cmd_timeframe(update: Update, ctx: CallbackContext):
    update.message.reply_text("Timeframe satt till 1m")

def cmd_entry_mode(update: Update, ctx: CallbackContext):
    global ENTRY_MODE
    # visa val
    if ctx.args:
        v = ctx.args[0].strip().lower()
        if v in ("tick", "close"):
            ENTRY_MODE = v.upper()
            update.message.reply_text(f"✅ Entry mode set to: {ENTRY_MODE}")
            return
    update.message.reply_text("Använd: /entry_mode tick|close")

def cmd_trailing(update: Update, ctx: CallbackContext):
    update.message.reply_text(
        f"Trail: ON ({TRAIL_PCT*100:.2f}%/{TRAIL_MIN*100:.2f}% min {TRAIL_ABS_MIN*100:.2f}%)"
    )

def cmd_keepalive_on(update: Update, ctx: CallbackContext):
    global KEEPALIVE_ON
    KEEPALIVE_ON = True
    update.message.reply_text("Keepalive: ON")

def cmd_keepalive_off(update: Update, ctx: CallbackContext):
    global KEEPALIVE_ON
    KEEPALIVE_ON = False
    update.message.reply_text("Keepalive: OFF")

def cmd_orb_on(update: Update, ctx: CallbackContext):
    args = [a.strip().upper() for a in (ctx.args or [])]
    if not args:
        for s in SYMBOLS: SYMS[s].orb_on = True
        update.message.reply_text("ORB: ON (för alla valda symboler)")
    else:
        for s in args:
            if s in SYMS: SYMS[s].orb_on = True
        update.message.reply_text("OK")

def cmd_orb_off(update: Update, ctx: CallbackContext):
    args = [a.strip().upper() for a in (ctx.args or [])]
    if not args:
        for s in SYMBOLS: SYMS[s].orb_on = False
        update.message.reply_text("ORB: OFF (för alla valda symboler)")
    else:
        for s in args:
            if s in SYMS: SYMS[s].orb_on = False
        update.message.reply_text("OK")

def cmd_status(update: Update, ctx: CallbackContext):
    lines = []
    lines.append(f"Mode: {MODE}   Engine: {'ON' if ENGINE_ON else 'OFF'}")
    lines.append(f"TF: 1m   Symbols: {','.join(SYMBOLS)}")
    lines.append(f"Entry: {ENTRY_MODE}   Trail: ON ({TRAIL_PCT*100:.2f}%/{TRAIL_MIN*100:.2f}% min {TRAIL_ABS_MIN*100:.2f}%)")
    lines.append(f"Keepalive: {'ON' if KEEPALIVE_ON else 'OFF'}   DayPnL: {fmt(sum(st.day_pnl for st in SYMS.values()))} USDT")
    lines.append(f"ORB master: {'ON' if any(st.orb_on for st in SYMS.values()) else 'OFF'}")
    for s in SYMBOLS:
        st = SYMS[s]
        pos = "✅" if st.position else "❌"
        stop = "-" if st.stop is None else fmt(st.stop)
        orb = "ON" if st.orb_on else "OFF"
        lines.append(f"{s}: pos={pos} stop={stop} | ORB: {orb}")
    update.message.reply_text("\n".join(lines), reply_markup=reply_menu())

def cmd_pnl(update: Update, ctx: CallbackContext):
    d = sum(st.day_pnl for st in SYMS.values())
    update.message.reply_text(f"DayPnL: {fmt(d)} USDT")

def cmd_reset_pnl(update: Update, ctx: CallbackContext):
    for st in SYMS.values():
        st.day_pnl = 0.0
    update.message.reply_text("DayPnL nollställd.")

def cmd_help(update: Update, ctx: CallbackContext):
    update.message.reply_text(
        "/status\n"
        "/engine_start /engine_stop\n"
        "/start_mock /start_live\n"
        "/symbols BTCUSDT/ETHUSDT/ADAUSDT/LINKUSDT/XRPUSDT\n"
        "/timeframe\n"
        "/entry_mode tick|close\n"
        "/trailing\n"
        "/pnl /reset_pnl /export_csv /export_k4\n"
        "/keepalive_on /keepalive_off\n"
        "/orb_on [SYM...] /orb_off [SYM...]\n",
        reply_markup=reply_menu()
    )

def setup_telegram():
    global bot, updater
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN saknas.")
        return
    delete_webhook_if_any()
    updater = Updater(token=TOKEN, use_context=True)
    bot = updater.bot

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(CommandHandler("engine_start", cmd_engine_start))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("symbols", cmd_symbols))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe))
    dp.add_handler(CommandHandler("entry_mode", cmd_entry_mode, pass_args=True))
    dp.add_handler(CommandHandler("trailing", cmd_trailing))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dp.add_handler(CommandHandler("orb_on", cmd_orb_on, pass_args=True))
    dp.add_handler(CommandHandler("orb_off", cmd_orb_off, pass_args=True))
    dp.add_handler(CommandHandler("status", cmd_status))

    # Viktigt: bara start_polling, INTE idle() → inga signals → funkar i Render
    updater.start_polling(drop_pending_updates=True, timeout=20)
    logger.info("Telegram polling started")
    if ADMIN_CHAT_ID:
        send_admin("✅ Bot online och polling startad.")

# =============== Keepalive ping ============
def keepalive_loop():
    while True:
        try:
            if KEEPALIVE_ON and PING_URL:
                with httpx.Client(timeout=10) as cli:
                    cli.get(PING_URL)
        except Exception:
            pass
        time.sleep(60)

# =============== Engine loop ===============
def engine_loop():
    # Kör kontinuerligt, hämtar senaste 1m candle från KuCoin för varje symbol
    while True:
        try:
            if ENGINE_ON:
                for s in SYMBOLS:
                    st = SYMS[s]
                    cndl = fetch_ku_1m_candle(s)
                    if not cndl:
                        continue

                    # kör bara en gång per avslutad candle
                    if cndl["ts"] == st.last_close_ts:
                        continue
                    st.last_close_ts = cndl["ts"]

                    # ORB uppdatering
                    ensure_orb(s, cndl)

                    # Exit först (så vi inte återköper i samma candle)
                    exited = try_exit(s, cndl)
                    if exited:
                        _, txt = exited
                        try:
                            bot.send_message(chat_id=ADMIN_CHAT_ID or None, text=txt)
                        except Exception:
                            pass

                    # Trailing upp
                    update_trailing(s, cndl)

                    # Entry
                    ent = try_entry(s, cndl)
                    if ent:
                        _, txt = ent
                        try:
                            bot.send_message(chat_id=ADMIN_CHAT_ID or None, text=txt)
                        except Exception:
                            pass
            time.sleep(2)
        except Exception as e:
            logger.error(f"Engine fel: {e}")
            time.sleep(5)

# =============== Lifespan ==================
@app.on_event("startup")
def on_startup():
    # Telegram
    threading.Thread(target=setup_telegram, daemon=True).start()
    # Keepalive
    threading.Thread(target=keepalive_loop, daemon=True).start()
    # Engine
    threading.Thread(target=engine_loop, daemon=True).start()
