import os
import csv
import time
import json
import threading
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    ParseMode,
    Bot,
    InputMediaDocument,
)
from telegram.ext import (
    Updater,
    CallbackContext,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    Filters,
)

# =========================
# ====== KONFIG/ENV =======
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
PUBLIC_URL = os.getenv("PUBLIC_URL", "")  # t.ex. https://mporbbot.onrender.com

# Render Free stänger av efter inaktivitet: ping var X min om keepalive är på
KEEPALIVE_EVERY_MIN = 9

# =========================
# ====== GLOBAL STATE =====
# =========================
state = {
    "mode": "idle",                # "idle" | "mock" | "live"
    "timeframe_min": 3,            # 1,3,5,15,30,60
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "ai_mode": "neutral",          # "neutral" | "aggressiv" | "försiktig"
    "today_pnl_usdt": 0.0,
    "engine_running": False,
    "keepalive_on": False,
    "last_candle_low": {},         # per symbol (för trailing via candle-botten)
}

# lås för trådsäkerhet
state_lock = threading.Lock()

# =========================
# ====== LOGGNING =========
# =========================
LOG_DIR = "./logs"        # Render filsystem är flyktigt mellan deploys – funkar ändå daglig drift
os.makedirs(LOG_DIR, exist_ok=True)

def _today_str():
    return datetime.utcnow().strftime("%Y-%m-%d")

def log_file_path(day: Optional[str] = None) -> str:
    if day is None:
        day = _today_str()
    return os.path.join(LOG_DIR, f"trades_{day}.csv")

def ensure_log_header(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_utc", "mode", "symbol", "side", "qty", "price",
                "fee", "pnl_usdt", "note"
            ])

def append_trade_log(symbol: str, side: str, qty: float, price: float,
                     fee: float, pnl: float, note: str = ""):
    path = log_file_path()
    ensure_log_header(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.utcnow().isoformat(),
            state["mode"],
            symbol,
            side,
            f"{qty:.8f}",
            f"{price:.8f}",
            f"{fee:.8f}",
            f"{pnl:.8f}",
            note
        ])

def add_pnl(amount: float):
    with state_lock:
        state["today_pnl_usdt"] = float(state["today_pnl_usdt"]) + float(amount)

# =========================
# ====== KUCOIN STUBS =====
# =========================
KUCOIN_PUBLIC = "https://api.kucoin.com"

def kucoin_get_ticker(symbol: str) -> Optional[float]:
    """
    Hämta senaste pris (enkel public endpoint). Returnerar None vid fel.
    """
    try:
        r = requests.get(f"{KUCOIN_PUBLIC}/api/v1/market/orderbook/level1", params={"symbol": symbol})
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == "200000":
                return float(data["data"]["price"])
    except Exception:
        pass
    return None

# OBS: Riktiga LIVE-orders kräver signering (HMAC) – här lämnar vi bara mjuka placeholders.
def place_live_spot_long(symbol: str, usdt_amount: float) -> Optional[dict]:
    """
    Placeholder för riktig spot-order. Returnera dict med exekveringsdata eller None vid fel.
    """
    # TODO: Implementera signering + /api/v1/orders (KuCoin spot) om du vill köra skarpt.
    # Här simulerar vi att ordern exekverades till marknadspris:
    price = kucoin_get_ticker(symbol) or 0.0
    if price <= 0:
        return None
    qty = usdt_amount / price
    fee = usdt_amount * 0.001  # 0.1% förenklad avgift
    return {"symbol": symbol, "price": price, "qty": qty, "fee": fee}

# =========================
# ====== MOTOR (MOCK) =====
# =========================
def compute_stop_from_candle_low(symbol: str, current_low: float) -> float:
    """
    Stoppen = nästa candles botten (för long). Vi lagrar senaste low
    och flyttar upp stop om ny candle har högre low.
    """
    prev = state["last_candle_low"].get(symbol)
    if prev is None or current_low > prev:
        state["last_candle_low"][symbol] = current_low
    return state["last_candle_low"][symbol]

def mock_trade_cycle():
    """
    Enkel mock-slinga: hämtar pris då och då och gör *simulerade* affärer
    när ”ny candle” upptäcks (vi approximera med tidsintervall).
    """
    # Enkelt tidsbaserat candle-tick
    last_tick = time.time()
    candle_seconds = state["timeframe_min"] * 60

    open_positions = {}  # symbol -> {"entry": price, "qty": float, "stop": float}

    while state["engine_running"] and state["mode"] == "mock":
        now = time.time()
        if now - last_tick >= candle_seconds:
            last_tick = now
            # "Ny candle" – uppdatera stop till candle-low, och ”öppna”/”stäng” med enkel logik
            for sym in list(state["symbols"]):
                price = kucoin_get_ticker(sym)
                if not price:
                    continue

                # fake candle low/close (vi använder samma pris pga ingen OHLC källa)
                candle_low = price * 0.998  # lite under priset bara för demo
                stop = compute_stop_from_candle_low(sym, candle_low)

                pos = open_positions.get(sym)
                if pos is None:
                    # ”ORBs” riktning: vi antar grön candle => köp (förenklat)
                    # Köp för MOCK_TRADE_USDT
                    qty = MOCK_TRADE_USDT / price
                    fee = MOCK_TRADE_USDT * 0.001
                    open_positions[sym] = {"entry": price, "qty": qty, "stop": stop}
                    append_trade_log(sym, "BUY", qty, price, fee, pnl=0.0, note="MOCK OPEN")
                else:
                    # uppdatera stop om candle_low stiger
                    if stop > pos["stop"]:
                        pos["stop"] = stop

                    # om priset ”klipper” stoppen -> stäng
                    if price <= pos["stop"]:
                        entry = pos["entry"]
                        qty = pos["qty"]
                        fee = (entry * qty) * 0.001 + (price * qty) * 0.001
                        pnl = (price - entry) * qty - fee
                        add_pnl(pnl)
                        append_trade_log(sym, "SELL", qty, price, fee, pnl, note="MOCK STOP OUT")
                        del open_positions[sym]

        time.sleep(1.0)

def live_trade_cycle():
    """
    Skelett för LIVE: här borde du lägga riktig strategi + orderläggning.
    Kör long ONLY (spot).
    """
    last_tick = time.time()
    candle_seconds = state["timeframe_min"] * 60

    while state["engine_running"] and state["mode"] == "live":
        now = time.time()
        if now - last_tick >= candle_seconds:
            last_tick = now
            # EXEMPEL: Lägg ”testköp” i verkligheten – här bara stub.
            for sym in state["symbols"]:
                # HÄR: ersätt med riktig logik (ORBs mm.)
                info = place_live_spot_long(sym, usdt_amount=MOCK_TRADE_USDT)
                if info:
                    append_trade_log(sym, "BUY", info["qty"], info["price"], info["fee"], pnl=0.0, note="LIVE OPEN (stub)")
        time.sleep(1.0)

def start_engine(mode: str):
    with state_lock:
        state["mode"] = mode
        state["engine_running"] = True
    t = threading.Thread(target=mock_trade_cycle if mode == "mock" else live_trade_cycle, daemon=True)
    t.start()

def stop_engine():
    with state_lock:
        state["engine_running"] = False
        state["mode"] = "idle"

def set_timeframe(minutes: int):
    with state_lock:
        state["timeframe_min"] = int(minutes)

def apply_symbols(symbols: List[str]):
    with state_lock:
        state["symbols"] = symbols[:]

def apply_ai_mode(mode: str):
    with state_lock:
        state["ai_mode"] = mode

# =========================
# ====== TELEGRAM =========
# =========================
SYMBOL_OPTIONS = [
    "BTCUSDT","ETHUSDT","ADAUSDT","XRPUSDT","SOLUSDT",
    "LINKUSDT","MATICUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT"
]
AI_OPTIONS = [("Neutral","neutral"), ("Aggressiv","aggressiv"), ("Försiktig","försiktig")]

def keyboard_yes_cancel(prefix: str):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ JA", callback_data=f"{prefix}:yes"),
         InlineKeyboardButton("❌ Avbryt", callback_data=f"{prefix}:cancel")]
    ])

def timeframe_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1m", callback_data="tf:1"),
         InlineKeyboardButton("3m", callback_data="tf:3"),
         InlineKeyboardButton("5m", callback_data="tf:5")],
        [InlineKeyboardButton("15m", callback_data="tf:15"),
         InlineKeyboardButton("30m", callback_data="tf:30"),
         InlineKeyboardButton("60m", callback_data="tf:60")],
        [InlineKeyboardButton("❌ Avbryt", callback_data="tf:cancel")]
    ])

def symbols_keyboard(current: List[str]):
    buttons, row = [], []
    for i, s in enumerate(SYMBOL_OPTIONS, 1):
        mark = " ✅" if s in current else ""
        row.append(InlineKeyboardButton(f"{s}{mark}", callback_data=f"sym:toggle:{s}"))
        if i % 2 == 0:
            buttons.append(row); row=[]
    if row: buttons.append(row)
    buttons.append([
        InlineKeyboardButton("➕ Egen", callback_data="sym:add_custom"),
        InlineKeyboardButton("🗑 Rensa", callback_data="sym:clear"),
    ])
    buttons.append([
        InlineKeyboardButton("💾 Spara", callback_data="sym:save"),
        InlineKeyboardButton("❌ Avbryt", callback_data="sym:cancel"),
    ])
    return InlineKeyboardMarkup(buttons)

def ai_keyboard():
    rows = [[InlineKeyboardButton(txt, callback_data=f"ai:{val}")] for txt, val in AI_OPTIONS]
    rows.append([InlineKeyboardButton("❌ Avbryt", callback_data="ai:cancel")])
    return InlineKeyboardMarkup(rows)

# --- Commands ---
def cmd_start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hej! 🤖\n\n"
        "Kommandon:\n"
        "/help – visa meny\n"
        "/status – visa status\n"
        "/start_mock – starta MOCK (bekräfta)\n"
        "/start_live – starta LIVE (bekräfta)\n"
        "/engine_stop – stoppa motor\n"
        "/timeframe – välj tidsram\n"
        "/symbols – välj symboler\n"
        "/set_ai – välj AI-läge\n"
        "/pnl – visa dagens PnL\n"
        "/reset_pnl – nollställ dagens PnL\n"
        "/export_csv – exportera logg (dagens)\n"
        "/keepalive_on – håll Render vaken\n"
        "/keepalive_off – stäng keepalive\n"
    )

def cmd_help(update: Update, context: CallbackContext):
    cmd_start(update, context)

def cmd_status(update: Update, context: CallbackContext):
    with state_lock:
        msg = (
            f"MODE: {state['mode']}\n"
            f"TF: {state['timeframe_min']} min\n"
            f"Symboler: {', '.join(state['symbols'])}\n"
            f"AI: {state['ai_mode']}\n"
            f"PnL idag: {state['today_pnl_usdt']:.4f} USDT\n"
            f"Engine: {'RUNNING' if state['engine_running'] else 'STOPPED'}\n"
            f"Keepalive: {'ON' if state['keepalive_on'] else 'OFF'}"
        )
    update.message.reply_text(msg)

def cmd_start_mock(update: Update, context: CallbackContext):
    update.message.reply_text("Vill du starta MOCK-trading?", reply_markup=keyboard_yes_cancel("mock"))

def cmd_start_live(update: Update, context: CallbackContext):
    update.message.reply_text("Vill du starta LIVE-trading?", reply_markup=keyboard_yes_cancel("live"))

def cmd_engine_stop(update: Update, context: CallbackContext):
    stop_engine()
    update.message.reply_text("🛑 Motorn stoppad.")

def cmd_timeframe(update: Update, context: CallbackContext):
    update.message.reply_text(f"Välj tidsram (nu: {state['timeframe_min']} min):", reply_markup=timeframe_keyboard())

def cmd_symbols(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Välj symboler (tryck igen för att toggla).",
        reply_markup=symbols_keyboard(state["symbols"])
    )

def cmd_set_ai(update: Update, context: CallbackContext):
    update.message.reply_text(f"Välj AI-läge (nu: {state['ai_mode']}):", reply_markup=ai_keyboard())

def cmd_pnl(update: Update, context: CallbackContext):
    update.message.reply_text(f"Dagens PnL: {state['today_pnl_usdt']:.4f} USDT")

def cmd_reset_pnl(update: Update, context: CallbackContext):
    with state_lock:
        state["today_pnl_usdt"] = 0.0
    update.message.reply_text("Dagens PnL nollställt.")

def cmd_export_csv(update: Update, context: CallbackContext):
    path = log_file_path()
    ensure_log_header(path)
    try:
        with open(path, "rb") as f:
            update.message.reply_document(f, filename=os.path.basename(path), caption="Dagens logg 📄")
    except Exception as e:
        update.message.reply_text(f"Kunde inte skicka logg: {e}")

def cmd_keepalive_on(update: Update, context: CallbackContext):
    with state_lock:
        state["keepalive_on"] = True
    update.message.reply_text("Keepalive är PÅ.")
    schedule_keepalive(context)

def cmd_keepalive_off(update: Update, context: CallbackContext):
    with state_lock:
        state["keepalive_on"] = False
    update.message.reply_text("Keepalive är AV.")
    # Ta bort jobb
    for job in context.job_queue.get_jobs_by_name("keepalive"):
        job.schedule_removal()

# --- Callbacks ---
def on_callback(update: Update, context: CallbackContext):
    q = update.callback_query
    q.answer()
    data = q.data

    # START MOCK/LIVE
    if data.startswith("mock:"):
        choice = data.split(":")[1]
        if choice == "yes":
            start_engine("mock")
            q.edit_message_text("MOCK-trading startad ✅")
        else:
            q.edit_message_text("Avbrutet.")
        return
    if data.startswith("live:"):
        choice = data.split(":")[1]
        if choice == "yes":
            start_engine("live")
            q.edit_message_text("LIVE-trading startad ✅")
        else:
            q.edit_message_text("Avbrutet.")
        return

    # TIMEFRAME
    if data.startswith("tf:"):
        val = data.split(":")[1]
        if val == "cancel":
            q.edit_message_text("Avbrutet.")
        else:
            set_timeframe(int(val))
            q.edit_message_text(f"Tidsram satt till {val} min ✅")
        return

    # AI MODE
    if data.startswith("ai:"):
        val = data.split(":")[1]
        if val == "cancel":
            q.edit_message_text("Avbrutet.")
        else:
            apply_ai_mode(val)
            q.edit_message_text(f"AI-läge satt till {val} ✅")
        return

    # SYMBOLS
    if data.startswith("sym:"):
        parts = data.split(":")
        action = parts[1]
        tmp = context.user_data.get("tmp_symbols", state["symbols"][:])

        if action == "toggle":
            sym = parts[2]
            if sym in tmp:
                tmp.remove(sym)
            else:
                tmp.append(sym)
            context.user_data["tmp_symbols"] = tmp
            q.edit_message_text(
                "Välj symboler (tryck igen för att toggla).",
                reply_markup=symbols_keyboard(tmp)
            )
            return

        if action == "clear":
            context.user_data["tmp_symbols"] = []
            q.edit_message_text("Alla symboler rensade.", reply_markup=symbols_keyboard([]))
            return

        if action == "save":
            final_list = context.user_data.get("tmp_symbols", state["symbols"])
            if not final_list:
                q.edit_message_text("Du måste välja minst en symbol.")
                return
            apply_symbols(final_list)
            q.edit_message_text("Sparat! Aktiva symboler: " + ", ".join(final_list))
            return

        if action == "cancel":
            q.edit_message_text("Avbrutet.")
            return

        if action == "add_custom":
            context.user_data["await_custom_symbol"] = True
            q.edit_message_text("Skriv en symbol (t.ex. `BNBUSDT`).")
            return

def on_text(update: Update, context: CallbackContext):
    if context.user_data.get("await_custom_symbol"):
        sym = update.message.text.strip().upper().replace(" ", "")
        context.user_data["await_custom_symbol"] = False
        tmp = context.user_data.get("tmp_symbols", state["symbols"][:])
        if sym and sym not in tmp:
            tmp.append(sym)
            context.user_data["tmp_symbols"] = tmp
            update.message.reply_text(
                f"➕ Lagt till {sym}.",
                reply_markup=symbols_keyboard(tmp)
            )
        else:
            update.message.reply_text("Ingen ändring.", reply_markup=symbols_keyboard(tmp))

# ---- Keepalive job ----
def keepalive_tick(context: CallbackContext):
    with state_lock:
        on = state["keepalive_on"]
    if not on or not PUBLIC_URL:
        return
    try:
        requests.get(PUBLIC_URL, timeout=8)
    except Exception:
        pass

def schedule_keepalive(context: CallbackContext):
    for job in context.job_queue.get_jobs_by_name("keepalive"):
        job.schedule_removal()
    context.job_queue.run_repeating(keepalive_tick, interval=KEEPALIVE_EVERY_MIN*60, first=3, name="keepalive")

# --- Bot start/stop i bakgrund ---
updater: Optional[Updater] = None

def start_bot():
    global updater
    if not TELEGRAM_TOKEN:
        print("⚠️ TELEGRAM_TOKEN saknas – bot startas inte.")
        return
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # commands
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe))
    dp.add_handler(CommandHandler("symbols", cmd_symbols))
    dp.add_handler(CommandHandler("set_ai", cmd_set_ai))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("export_csv", cmd_export_csv))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))

    # callbacks & text
    dp.add_handler(CallbackQueryHandler(on_callback))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, on_text))

    # Start polling (egen tråd)
    updater.start_polling(drop_pending_updates=True)
    print("✅ Telegram-bot igång.")

def stop_bot():
    global updater
    if updater:
        updater.stop()
        updater.is_idle = False
        updater = None

# =========================
# ====== FASTAPI APP ======
# =========================
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "healthy"

@app.on_event("startup")
def on_startup():
    # starta telegrambotten
    threading.Thread(target=start_bot, daemon=True).start()
    print("🌐 FastAPI startad.")

@app.on_event("shutdown")
def on_shutdown():
    stop_bot()
    print("👋 Stänger ned.")
