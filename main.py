import os
import csv
import time
import json
import hmac
import base64
import hashlib
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# Telegram PTB 13.15
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    ParseMode,
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
# ====== ENV & KONFIG =====
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0") or "0")

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
KUCOIN_API_KEY_VER = os.getenv("KUCOIN_API_KEY_VERSION", "2")  # v2

MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
FEE_PER_SIDE = float(os.getenv("FEE_RATE", "0.001"))  # 0.10% per sida
PUBLIC_URL = os.getenv("PUBLIC_URL", "")  # din Render-URL (för keepalive)

DEFAULT_SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT").split(",") if s.strip()]
DEFAULT_TF = os.getenv("TIMEFRAME", "3min").lower()  # "1min","3min","5min","15min","30min","1hour"
VALID_TF = {"1min","3min","5min","15min","30min","1hour"}
TF_LABELS = {"1min":1,"3min":3,"5min":5,"15min":15,"30min":30,"1hour":60}
TF_FROM_MIN = {1:"1min",3:"3min",5:"5min",15:"15min",30:"30min",60:"1hour"}
TICK_EPS = float(os.getenv("TICK_EPS", "1e-8"))

# AI-filter: kräver minsta kropp på referens-candle
AI_MODE = "neutral"
MIN_ORB_PCT = {"aggressiv": 0.0005, "neutral": 0.001, "försiktig": 0.002}

# =========================
# ====== GLOBAL STATE =====
# =========================
UTC = timezone.utc
app = FastAPI(title="Mp ORBbot", version="1.0.0")

STATE = {
    "mode": "idle",                  # "idle" | "mock" | "live"
    "engine_on": False,
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": DEFAULT_TF if DEFAULT_TF in VALID_TF else "3min",
    "ai": AI_MODE,
    "pnl_day_mock": 0.0,
    "pnl_day_live": 0.0,
    "keepalive_on": False,
}

# per symbol
# {symbol: {"ref_ts": int, "ref_high": float, "ref_low": float,
#           "entry": float|None, "stop": float|None, "qty": float|0}}
SYMBOL_STATE: Dict[str, Dict] = {}
SYMBOL_LOCK = threading.Lock()

# engine tråd
ENGINE_STOP = threading.Event()
ENGINE_THREAD: Optional[threading.Thread] = None

# Telegram
updater: Optional[Updater] = None

# logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def today_csv_path() -> str:
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"trades_{day}.csv")

def write_trade(mode_live: bool, symbol: str, entry: float, exit_px: float, qty: float, reason: str):
    fee_in = entry * qty * FEE_PER_SIDE
    fee_out = exit_px * qty * FEE_PER_SIDE
    pnl = (exit_px - entry) * qty - fee_in - fee_out
    if mode_live: STATE["pnl_day_live"] += pnl
    else: STATE["pnl_day_mock"] += pnl

    path = today_csv_path()
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "timestamp_utc","mode","symbol","side","entry","exit","qty","fee_entry","fee_exit","pnl_usdt","reason"
            ])
        w.writerow([
            datetime.now(UTC).isoformat(),
            "LIVE" if mode_live else "MOCK",
            symbol,
            "LONG",
            f"{entry:.8f}",
            f"{exit_px:.8f}",
            f"{qty:.6f}",
            f"{fee_in:.6f}",
            f"{fee_out:.6f}",
            f"{pnl:.6f}",
            reason
        ])

# =========================
# ====== KUCOIN REST ======
# =========================
KU_BASE = "https://api.kucoin.com"

def sym_dash(symbol: str) -> str:
    return symbol if "-" in symbol else (symbol[:-4] + "-USDT" if symbol.endswith("USDT") else symbol)

def k_get(url: str, params=None) -> dict:
    try:
        r = requests.get(url, params=params or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def ku_level1_price(symbol: str) -> float:
    data = k_get(f"{KU_BASE}/api/v1/market/orderbook/level1", {"symbol": sym_dash(symbol)})
    try:
        return float(data["data"]["price"])
    except Exception:
        return 0.0

def ku_candles(symbol: str, tf: str, limit: int = 3) -> List[List]:
    # returns ascending by time: [time, open, close, high, low, volume, turnover]
    data = k_get(f"{KU_BASE}/api/v1/market/candles", {"symbol": sym_dash(symbol), "type": tf})
    arr = data.get("data", [])
    for row in arr:
        row[0] = int(row[0])  # time
        row[1] = float(row[1]); row[2] = float(row[2]); row[3] = float(row[3]); row[4] = float(row[4])
    arr.sort(key=lambda x: x[0])
    return arr[-limit:] if len(arr) > limit else arr

def ku_headers(method: str, endpoint: str, body_str: str = "", query_str: str = "") -> dict:
    ts = str(int(time.time() * 1000))
    prehash = ts + method.upper() + endpoint + query_str + body_str
    sign = base64.b64encode(hmac.new(KUCOIN_API_SECRET.encode(), prehash.encode(), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(KUCOIN_API_SECRET.encode(), KUCOIN_API_PASSPHRASE.encode(), hashlib.sha256).digest()).decode()
    return {
        "KC-API-KEY": KUCOIN_API_KEY,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": KUCOIN_API_KEY_VER,
        "Content-Type": "application/json",
    }

def ku_post(endpoint: str, body: dict) -> dict:
    if not (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE):
        return {}
    import json as _json
    body_str = _json.dumps(body)
    headers = ku_headers("POST", endpoint, body_str=body_str)
    url = KU_BASE + endpoint
    try:
        r = requests.post(url, headers=headers, data=body_str, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def ku_market_buy(symbol_dash: str, funds_usdt: float) -> bool:
    # Market buy with funds (USDT)
    body = {
        "clientOid": str(int(time.time()*1000)),
        "side": "buy",
        "symbol": symbol_dash,
        "type": "market",
        "funds": f"{funds_usdt:.2f}",
    }
    res = ku_post("/api/v1/orders", body)
    return res.get("code") == "200000"

def ku_market_sell(symbol_dash: str, size: float) -> bool:
    body = {
        "clientOid": str(int(time.time()*1000)),
        "side": "sell",
        "symbol": symbol_dash,
        "type": "market",
        "size": f"{max(size,0):.6f}",
    }
    res = ku_post("/api/v1/orders", body)
    return res.get("code") == "200000"

# =========================
# ===== ORB STRATEGI ======
# =========================
def ensure_symbol_state(symbol: str):
    with SYMBOL_LOCK:
        if symbol not in SYMBOL_STATE:
            SYMBOL_STATE[symbol] = {
                "ref_ts": 0,
                "ref_high": None,
                "ref_low": None,
                "entry": None,
                "stop": None,
                "qty": 0.0,
                "trades": 0,
            }

def update_reference(symbol: str, tf: str) -> Optional[int]:
    """
    Sätter referens till senast STÄNGD candle (näst sista efter sortering).
    AI-läge kräver minsta kroppsstorlek.
    """
    ensure_symbol_state(symbol)
    candles = ku_candles(symbol, tf, limit=3)
    if len(candles) < 2:
        return None
    # näst sista = senast stängda
    ref = candles[-2]
    ref_ts = int(ref[0]); o = float(ref[1]); c = float(ref[2]); h = float(ref[3]); l = float(ref[4])
    body = abs(c - o)
    if body < max(o, 1e-9) * MIN_ORB_PCT.get(STATE["ai"], 0.001):
        return None

    with SYMBOL_LOCK:
        st = SYMBOL_STATE[symbol]
        if st["ref_ts"] != ref_ts:
            st["ref_ts"] = ref_ts
            st["ref_high"] = h
            st["ref_low"] = l
            # Trailing: flytta stop upp till nya candle-low om pos finns
            if st["entry"] is not None and st["stop"] is not None:
                st["stop"] = max(st["stop"], l - TICK_EPS)
    return ref_ts

def engine_step_symbol(symbol: str):
    ensure_symbol_state(symbol)
    tf = STATE["timeframe"]
    update_reference(symbol, tf)

    price = ku_level1_price(symbol)
    if price <= 0:
        return

    with SYMBOL_LOCK:
        st = SYMBOL_STATE[symbol]
        ref_h = st["ref_high"]
        ref_l = st["ref_low"]

        # ENTRY: bryter över ref_high
        if st["entry"] is None and ref_h and price > ref_h + TICK_EPS:
            entry = price
            stop = (ref_l or price) - TICK_EPS
            qty = MOCK_TRADE_USDT / entry
            ok = True
            live = (STATE["mode"] == "live") and (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE)
            if live:
                ok = ku_market_buy(sym_dash(symbol), funds_usdt=MOCK_TRADE_USDT)
            if ok:
                st["entry"] = entry
                st["stop"] = stop
                st["qty"] = qty
                st["trades"] += 1
                try:
                    updater.bot.send_message(
                        AUTHORIZED_USER_ID or update_any_chat_id(),
                        f"✅ ENTRY {symbol} {STATE['mode'].upper()} @ {entry:.6f} | SL {stop:.6f}"
                    )
                except Exception:
                    pass

        # EXIT: träffar stop
        if st["entry"] is not None and st["stop"] is not None and price <= st["stop"] + TICK_EPS:
            exit_px = price
            qty = st["qty"]
            live = (STATE["mode"] == "live") and (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE)
            ok = True
            if live and qty > 0:
                ok = ku_market_sell(sym_dash(symbol), size=qty)
            if ok:
                write_trade(live, symbol, st["entry"], exit_px, qty, reason="stop")
                pnl_show = STATE["pnl_day_live"] if live else STATE["pnl_day_mock"]
                try:
                    updater.bot.send_message(
                        AUTHORIZED_USER_ID or update_any_chat_id(),
                        f"⏹ EXIT {symbol} {STATE['mode'].upper()} @ {exit_px:.6f}\nPnL idag: {pnl_show:.4f} USDT"
                    )
                except Exception:
                    pass
                st["entry"] = None; st["stop"] = None; st["qty"] = 0.0

def engine_loop():
    while not ENGINE_STOP.is_set():
        if STATE["engine_on"]:
            for s in STATE["symbols"]:
                try:
                    engine_step_symbol(s)
                except Exception:
                    pass
        # Keepalive
        if STATE["keepalive_on"] and PUBLIC_URL:
            try:
                requests.get(PUBLIC_URL + ("" if PUBLIC_URL.endswith("/") else "/"), timeout=5)
            except Exception:
                pass
        time.sleep(2)

def start_engine(mode: str):
    STATE["mode"] = mode.lower()
    STATE["engine_on"] = True

def stop_engine():
    STATE["engine_on"] = False
    STATE["mode"] = "idle"

# =========================
# ====== TELEGRAM UI ======
# =========================
SYMBOL_OPTIONS = [
    "BTCUSDT","ETHUSDT","ADAUSDT","XRPUSDT","SOLUSDT",
    "LINKUSDT","MATICUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT"
]
AI_CHOICES = [("Neutral","neutral"), ("Aggressiv","aggressiv"), ("Försiktig","försiktig")]

def restrict(update: Update) -> bool:
    if AUTHORIZED_USER_ID == 0:
        return True
    u = update.effective_user
    return (u and u.id == AUTHORIZED_USER_ID)

def yes_cancel(prefix: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ JA", callback_data=f"{prefix}:yes"),
         InlineKeyboardButton("❌ Avbryt", callback_data=f"{prefix}:cancel")]
    ])

def timeframe_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("1m", callback_data="tf:1"),
         InlineKeyboardButton("3m", callback_data="tf:3"),
         InlineKeyboardButton("5m", callback_data="tf:5")],
        [InlineKeyboardButton("15m", callback_data="tf:15"),
         InlineKeyboardButton("30m", callback_data="tf:30"),
         InlineKeyboardButton("60m", callback_data="tf:60")],
        [InlineKeyboardButton("❌ Avbryt", callback_data="tf:cancel")]
    ]
    return InlineKeyboardMarkup(rows)

def symbols_keyboard(current: List[str]) -> InlineKeyboardMarkup:
    buttons, row = [], []
    for i, s in enumerate(SYMBOL_OPTIONS, 1):
        mark = " ✅" if s in current else ""
        row.append(InlineKeyboardButton(f"{s}{mark}", callback_data=f"sym:toggle:{s}"))
        if i % 2 == 0:
            buttons.append(row); row=[]
    if row: buttons.append(row)
    buttons.append([InlineKeyboardButton("➕ Egen", callback_data="sym:add_custom"),
                    InlineKeyboardButton("🗑 Rensa", callback_data="sym:clear")])
    buttons.append([InlineKeyboardButton("💾 Spara", callback_data="sym:save"),
                    InlineKeyboardButton("❌ Avbryt", callback_data="sym:cancel")])
    return InlineKeyboardMarkup(buttons)

def ai_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(txt, callback_data=f"ai:{val}")] for txt, val in AI_CHOICES]
    rows.append([InlineKeyboardButton("❌ Avbryt", callback_data="ai:cancel")])
    return InlineKeyboardMarkup(rows)

# ----- Commands -----
def cmd_start(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    update.message.reply_text(
        "Hej! 🤖 Mp ORBbot\n\n"
        "/help – meny\n"
        "/status – status\n"
        "/start_mock – starta MOCK (knapp)\n"
        "/start_live – starta LIVE (knapp)\n"
        "/engine_start <mock|live> – starta med text\n"
        "/engine_stop – stoppa motor\n"
        "/timeframe – välj tidsram\n"
        "/symbols – välj symboler\n"
        "/set_ai – välj AI-läge\n"
        "/pnl – PnL idag\n"
        "/reset_pnl – nollställ PnL idag\n"
        "/export_csv – skicka logg (dagens)\n"
        "/keepalive_on, /keepalive_off – Render keepalive\n"
        "/panic – stäng alla pos och stoppa"
    )

def cmd_help(update: Update, ctx: CallbackContext):
    cmd_start(update, ctx)

def cmd_status(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    lines = []
    with SYMBOL_LOCK:
        for s, st in SYMBOL_STATE.items():
            pos = "✔" if st.get("entry") else "–"
            lines.append(f"{s}: pos={pos} refH={st.get('ref_high')} refL={st.get('ref_low')} SL={st.get('stop')}")
    txt = (
        f"Mode: *{STATE['mode']}*  Engine: *{'ON' if STATE['engine_on'] else 'OFF'}*\n"
        f"AI: *{STATE['ai']}*  TF: *{STATE['timeframe']}* ({TF_LABELS.get(STATE['timeframe'],3)}m)\n"
        f"Symbols: {', '.join(STATE['symbols'])}\n"
        f"PnL → MOCK {STATE['pnl_day_mock']:.4f} | LIVE {STATE['pnl_day_live']:.4f}\n\n" +
        ("\n".join(lines) if lines else "(ingen symbolinfo)")
    )
    update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

def cmd_pnl(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    update.message.reply_text(f"PnL idag → MOCK: {STATE['pnl_day_mock']:.4f} | LIVE: {STATE['pnl_day_live']:.4f}")

def cmd_reset_pnl(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    STATE["pnl_day_mock"] = 0.0; STATE["pnl_day_live"] = 0.0
    update.message.reply_text("PnL idag nollställt.")

def cmd_export_csv(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    path = today_csv_path()
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["timestamp_utc","mode","symbol","side","entry","exit","qty","fee_entry","fee_exit","pnl_usdt","reason"])
    try:
        with open(path, "rb") as f:
            update.message.reply_document(f, filename=os.path.basename(path), caption="Dagens logg 📄")
    except Exception as e:
        update.message.reply_text(f"Kunde inte skicka logg: {e}")

def cmd_keepalive_on(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    STATE["keepalive_on"] = True
    update.message.reply_text("Keepalive: ON")

def cmd_keepalive_off(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    STATE["keepalive_on"] = False
    update.message.reply_text("Keepalive: OFF")

def cmd_panic(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    # Stäng allt lokalt och stoppa engine
    with SYMBOL_LOCK:
        for s, st in SYMBOL_STATE.items():
            if st.get("entry") is not None and (st.get("qty") or 0) > 0:
                px = ku_level1_price(s)
                if px > 0:
                    write_trade(STATE["mode"]=="live", s, st["entry"], px, st["qty"], reason="panic")
                st["entry"] = None; st["stop"] = None; st["qty"] = 0.0
    stop_engine()
    update.message.reply_text("PANIC: Alla positioner stängda och motor stoppad.")

def cmd_start_mock(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    update.message.reply_text("Vill du starta MOCK-trading?", reply_markup=yes_cancel("mock"))

def cmd_start_live(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    if not (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE):
        update.message.reply_text("LIVE kräver KuCoin API-nycklar i Environment.")
        return
    update.message.reply_text("Vill du starta LIVE-trading?", reply_markup=yes_cancel("live"))

def cmd_engine_start(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    if not ctx.args:
        update.message.reply_text("Använd: /engine_start mock|live")
        return
    mode = ctx.args[0].lower()
    if mode not in ("mock","live"):
        update.message.reply_text("Endast 'mock' eller 'live'.")
        return
    start_engine(mode)
    update.message.reply_text(f"Engine: ON ({mode.upper()})")

def cmd_engine_stop(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    stop_engine()
    update.message.reply_text("Engine: OFF")

def cmd_timeframe(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    update.message.reply_text(f"Välj tidsram (nu: {STATE['timeframe']})", reply_markup=timeframe_keyboard())

def cmd_symbols(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    update.message.reply_text("Välj symboler (tryck igen för att toggla):", reply_markup=symbols_keyboard(STATE["symbols"]))

def cmd_set_ai(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    update.message.reply_text(f"Välj AI-läge (nu: {STATE['ai']}):", reply_markup=ai_keyboard())

# ----- Callbacks & text -----
def cb_all(update: Update, ctx: CallbackContext):
    if not restrict(update): 
        update.callback_query.answer()
        update.callback_query.edit_message_text("Åtkomst nekad.")
        return

    q = update.callback_query
    q.answer()
    data = q.data

    # Start mock/live
    if data.startswith("mock:"):
        if data.endswith(":yes"):
            start_engine("mock"); q.edit_message_text("MOCK startad ✅")
        else:
            q.edit_message_text("Avbrutet.")
        return
    if data.startswith("live:"):
        if data.endswith(":yes"):
            start_engine("live"); q.edit_message_text("LIVE startad ✅")
        else:
            q.edit_message_text("Avbrutet.")
        return

    # Timeframe
    if data.startswith("tf:"):
        val = data.split(":")[1]
        if val == "cancel":
            q.edit_message_text("Avbrutet.")
            return
        minutes = int(val)
        tf = TF_FROM_MIN.get(minutes, "3min")
        STATE["timeframe"] = tf
        q.edit_message_text(f"Tidsram satt till {tf} ✅")
        return

    # AI
    if data.startswith("ai:"):
        val = data.split(":")[1]
        if val == "cancel":
            q.edit_message_text("Avbrutet.")
            return
        if val in MIN_ORB_PCT:
            STATE["ai"] = val
            q.edit_message_text(f"AI-läge satt till {val} ✅")
        else:
            q.edit_message_text("Ogiltigt AI-läge.")
        return

    # Symbols
    if data.startswith("sym:"):
        parts = data.split(":")
        action = parts[1]
        tmp = ctx.user_data.get("tmp_symbols", STATE["symbols"][:])

        if action == "toggle":
            sym = parts[2]
            if sym in tmp: tmp.remove(sym)
            else: tmp.append(sym)
            ctx.user_data["tmp_symbols"] = tmp
            q.edit_message_text("Välj symboler:", reply_markup=symbols_keyboard(tmp))
            return
        if action == "clear":
            ctx.user_data["tmp_symbols"] = []
            q.edit_message_text("Alla symboler rensade.", reply_markup=symbols_keyboard([]))
            return
        if action == "save":
            final_list = ctx.user_data.get("tmp_symbols", STATE["symbols"])
            if not final_list:
                q.edit_message_text("Du måste välja minst en symbol.")
                return
            STATE["symbols"] = [s.upper() for s in final_list]
            for s in STATE["symbols"]: ensure_symbol_state(s)
            q.edit_message_text("Sparat! Aktiva: " + ", ".join(STATE["symbols"]))
            return
        if action == "cancel":
            q.edit_message_text("Avbrutet.")
            return
        if action == "add_custom":
            ctx.user_data["await_custom_symbol"] = True
            q.edit_message_text("Skriv en symbol (t.ex. BNBUSDT).")
            return

def on_text(update: Update, ctx: CallbackContext):
    if not restrict(update): return
    # custom symbol input
    if ctx.user_data.get("await_custom_symbol"):
        sym = (update.message.text or "").strip().upper().replace(" ", "")
        ctx.user_data["await_custom_symbol"] = False
        tmp = ctx.user_data.get("tmp_symbols", STATE["symbols"][:])
        if sym and sym not in tmp:
            tmp.append(sym); ctx.user_data["tmp_symbols"] = tmp
            update.message.reply_text(f"➕ Lagt till {sym}.", reply_markup=symbols_keyboard(tmp))
        else:
            update.message.reply_text("Ingen ändring.", reply_markup=symbols_keyboard(tmp))

# Hjälp för att kunna pusha meddelanden även utan satt admin
def update_any_chat_id() -> Optional[int]:
    # Endast om du inte satt AUTHORIZED_USER_ID – då returnerar vi None
    return AUTHORIZED_USER_ID if AUTHORIZED_USER_ID != 0 else None

# =========================
# ====== BOT START ========
# =========================
def start_bot():
    global updater, ENGINE_THREAD
    if not TELEGRAM_TOKEN:
        print("⚠️ TELEGRAM_TOKEN saknas – boten startas inte.")
        return
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # commands
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("start_mock", cmd_start_mock))
    dp.add_handler(CommandHandler("start_live", cmd_start_live))
    dp.add_handler(CommandHandler("engine_start", cmd_engine_start, pass_args=True))
    dp.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    dp.add_handler(CommandHandler("timeframe", cmd_timeframe))
    dp.add_handler(CommandHandler("symbols", cmd_symbols))
    dp.add_handler(CommandHandler("set_ai", cmd_set_ai))
    dp.add_handler(CommandHandler("pnl", cmd_pnl))
    dp.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    dp.add_handler(CommandHandler("export_csv", cmd_export_csv))
    dp.add_handler(CommandHandler("keepalive_on", cmd_keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", cmd_keepalive_off))
    dp.add_handler(CommandHandler("panic", cmd_panic))

    # callbacks & text
    dp.add_handler(CallbackQueryHandler(cb_all))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, on_text))

    # engine tråd
    ENGINE_STOP.clear()
    ENGINE_THREAD = threading.Thread(target=engine_loop, name="engine-loop", daemon=True)
    ENGINE_THREAD.start()

    updater.start_polling(drop_pending_updates=True)
    print("✅ Telegram-bot & Engine igång.")

def stop_bot():
    global updater
    STATE["engine_on"] = False
    ENGINE_STOP.set()
    if updater:
        try: updater.stop()
        except Exception: pass
        updater = None

# =========================
# ====== FASTAPI APP ======
# =========================
@app.get("/", response_class=PlainTextResponse)
def root():
    return "Mp ORBbot up"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.on_event("startup")
def on_startup():
    # initiera symbol state
    for s in STATE["symbols"]:
        ensure_symbol_state(s)
    # starta bot+engine
    threading.Thread(target=start_bot, daemon=True).start()

@app.on_event("shutdown")
def on_shutdown():
    stop_bot()

# Lokal körning
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
