# main.py — Mp ORBbot (2025-09-04)
# En-fil Telegram-bot för ORB-handel (mock/live), backtest, CSV-export, AI-lägen som kan sparas.
# Bygger på python-telegram-bot v20 (async).

import os, sys, time, hmac, json, math, hashlib, asyncio, logging, threading
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# ============================
# Konfiguration (säkra defaults)
# ============================
BOT_NAME = "Mp ORBbot"
SYMBOLS = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAME = "3m"                  # endast 3m stöds i denna fil
FEE_DEFAULT = 0.001               # 0.1% per sida
MOCK_TRADE_NOTIONAL = 15.0        # USDT (mock) — sänkt från 30
MAX_CONCURRENT_COINS = 2

# Handelstid (nya entries)
ENGINE_TIME_START_UTC = 7         # 07:00 UTC
ENGINE_TIME_END_UTC = 20          # 20:00 UTC

# AI/strategi-parametrar (går att spara/ladda)
AI_STATE_FILE = "ai_state.json"
AI_DEFAULT = {
    "mode": "neutral",            # 'aggressiv' | 'försiktig' | 'neutral'
    "ENTRY_MODE": "CLOSE",        # 'CLOSE' eller 'TICK'
    "be_trigger": 0.0025,         # +0.25% -> flytta SL till BE
    "trail_arm": 0.0045,          # +0.45% -> starta trailing
    "trail_dist": 0.0025,         # trailing-avstånd 0.25%
    "atr_mult_sl": 0.6,           # SL = max(ORB-extrem, atr_mult*ATR(14))
    "adx_min": 18,                # min ADX för entries
    "atrp_min": 0.002,            # ATR% (ATR/Close) >= 0.20%
    "ema_fast": 50,
    "ema_slow": 200,
    "dca_max_legs": 1,            # max 1 DCA-ben
    "dca_trigger": 0.0035,        # 0.35% emot -> DCA
    "dca_adx_min": 20,            # minst ADX för DCA
}

# Failsafe (tappar Telegram > 5 min => stäng motor)
FAILSAFE_MAX_NOUPDATES_SEC = 300
WATCHDOG_INTERVAL_SEC = 30

# Loggfiler
MOCK_LOG = "mock_trade_log.csv"   # endast mocktrades
REAL_LOG = "real_trade_log.csv"   # endast riktiga trades
BACKTEST_CSV = "backtest_results.csv"

# Miljö
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
KUCOIN_KEY = os.getenv("KUCOIN_KEY", "").strip()
KUCOIN_SECRET = os.getenv("KUCOIN_SECRET", "").strip()
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE", "").strip()

# ============================
# Logging
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("MpORB")

# ============================
# Hjälp
# ============================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def percent(a, b) -> float:
    if b == 0: return 0.0
    return (a - b) / b

def to_symbol_dash(sym: str) -> str:
    sym = sym.upper().replace("/", "").replace("-", "")
    return f"{sym[:-4]}-{sym[-4:]}" if sym.endswith("USDT") else sym

def within_entry_hours(ts: datetime) -> bool:
    hr = ts.hour
    if ENGINE_TIME_START_UTC <= ENGINE_TIME_END_UTC:
        return ENGINE_TIME_START_UTC <= hr < ENGINE_TIME_END_UTC
    # fallback om intervallet passerar midnatt
    return (hr >= ENGINE_TIME_START_UTC) or (hr < ENGINE_TIME_END_UTC)

def ensure_csv(path: str, columns: List[str]):
    if not os.path.exists(path):
        pd.DataFrame([], columns=columns).to_csv(path, index=False)

def load_ai_state() -> dict:
    if os.path.exists(AI_STATE_FILE):
        try:
            with open(AI_STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                return {**AI_DEFAULT, **d}
        except Exception as e:
            log.warning(f"Kunde inte läsa {AI_STATE_FILE}: {e}")
    return AI_DEFAULT.copy()

def save_ai_state(state: dict):
    try:
        with open(AI_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.error(f"Kunde inte spara {AI_STATE_FILE}: {e}")

# ============================
# Datatyper
# ============================
@dataclass
class Position:
    side: str               # 'long' eller 'short'
    qty: float
    avg_price: float
    entry_price: float
    sl: float               # aktuell stop
    orb_high: float
    orb_low: float
    entry_time: datetime
    ai_mode: str
    legs: int = 1
    trail_armed: bool = False
    max_fav: float = 0.0
    min_fav: float = 0.0

@dataclass
class SymState:
    last_color: Optional[str] = None  # 'green'/'red'
    orb_long: Optional[Tuple[float,float,datetime]] = None   # (high, low, time)
    orb_short: Optional[Tuple[float,float,datetime]] = None
    position: Optional[Position] = None
    loss_streak: int = 0
    cooldown_until: Optional[datetime] = None
    trades_today: int = 0
    realized_pnl: float = 0.0        # mock PnL summerat
    dca_done: bool = False

# ============================
# Global state
# ============================
AI = load_ai_state()
ENGINE_ON = False
LIVE_TRADING = False
LAST_USER_UPDATE = utcnow()
CONFIRM_PENDING: Optional[str] = None   # 'engine_on' | 'live_on' | 'engine_off' | 'panic'
SYMSTATES: Dict[str, SymState] = {s: SymState() for s in SYMBOLS}

# Säkerställ loggfiler
ensure_csv(MOCK_LOG, ["time","symbol","side","action","price","qty","fee","pnl","ai","note"])
ensure_csv(REAL_LOG, ["time","symbol","side","action","price","qty","fee","pnl","ai","note"])

# ============================
# KuCoin helpers
# ============================
KU_BASE = "https://api.kucoin.com"

def ku_public(path: str, params: dict=None) -> dict:
    r = requests.get(KU_BASE + path, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if str(data.get("code")) != "200000":
        raise RuntimeError(f"KuCoin API fel: {data}")
    return data["data"]

def ku_private(method: str, path: str, params: dict=None, body: dict=None) -> dict:
    if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
        raise RuntimeError("KuCoin API-nycklar saknas.")
    now = str(int(time.time() * 1000))
    body_str = json.dumps(body) if body else ""
    query = ""
    if params:
        # ordna parametrar deterministiskt
        query = "?" + "&".join(f"{k}={params[k]}" for k in sorted(params.keys()))
    pre = now + method.upper() + path + query + body_str
    sign = base64_encode(hmac_sha256(pre.encode(), KUCOIN_SECRET.encode()))
    passphrase = base64_encode(hmac_sha256(KUCOIN_PASSPHRASE.encode(), KUCOIN_SECRET.encode()))
    headers = {
        "KC-API-KEY": KUCOIN_KEY,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": now,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }
    url = KU_BASE + path
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, params=params, timeout=10)
    elif method.upper() == "POST":
        r = requests.post(url, headers=headers, params=params, json=body, timeout=10)
    else:
        raise ValueError("Endast GET/POST implementerat")
    r.raise_for_status()
    data = r.json()
    if str(data.get("code")) != "200000":
        raise RuntimeError(f"KuCoin privat API fel: {data}")
    return data["data"]

def hmac_sha256(msg: bytes, key: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()

def base64_encode(b: bytes) -> str:
    import base64
    return base64.b64encode(b).decode()

def fetch_klines(symbol: str, start: int, end: int, tf: str="3min") -> pd.DataFrame:
    # KuCoin klines: /api/v1/market/candles?type=3min&symbol=BTC-USDT&startAt=&endAt=
    params = {"type": tf, "symbol": to_symbol_dash(symbol), "startAt": start, "endAt": end}
    data = ku_public("/api/v1/market/candles", params)  # returns list of [time, open, close, high, low, vol, turnover]
    # KuCoin sorterar nyast först – vänd
    rows = []
    for r in reversed(data):
        ts = int(r[0])
        rows.append([datetime.fromtimestamp(ts, tz=timezone.utc),
                     float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
    df = pd.DataFrame(rows, columns=["time","open","close","high","low","volume"])
    return df

# ============================
# Indikatorer
# ============================
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int=14) -> pd.Series:
    return true_range(df).rolling(n).mean()

def adx(df: pd.DataFrame, n: int=14) -> pd.Series:
    # klassisk ADX-beräkning (förenklad)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_n = tr.rolling(n).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(n).sum() / atr_n
    minus_di = 100 * pd.Series(minus_dm).rolling(n).sum() / atr_n
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0.0) * 100
    return dx.rolling(n).mean().fillna(0.0)

# ============================
# ORB-detektor
# ============================
def detect_orb(prev_color: Optional[str], o: float, c: float, h: float, l: float) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]], str]:
    color = "green" if c >= o else "red"
    long_orb = None
    short_orb = None
    if prev_color == "red" and color == "green":
        long_orb = (h, l)
    if prev_color == "green" and color == "red":
        short_orb = (h, l)
    return long_orb, short_orb, color

# ============================
# PnL & logg
# ============================
def log_trade(mock: bool, symbol: str, side: str, action: str, price: float, qty: float, fee: float, pnl: float, ai: str, note: str=""):
    path = MOCK_LOG if mock else REAL_LOG
    row = {
        "time": utcnow().isoformat(),
        "symbol": symbol.upper(),
        "side": side,
        "action": action,
        "price": round(price, 8),
        "qty": round(qty, 8),
        "fee": round(fee, 8),
        "pnl": round(pnl, 8),
        "ai": ai,
        "note": note[:120]
    }
    try:
        df = pd.DataFrame([row])
        df.to_csv(path, mode="a", index=False, header=False)
    except Exception as e:
        log.error(f"Kunde inte skriva logg {path}: {e}")

# ============================
# Handelsmotor (pollande)
# ============================
async def engine_loop(app: Application):
    global ENGINE_ON
    while True:
        try:
            if ENGINE_ON:
                await process_symbols()
        except Exception as e:
            log.error(f"Engine fel: {e}")
        await asyncio.sleep(20)  # kör var ~20s

async def process_symbols():
    # Hämta senaste ~150 candles (drygt 7.5h)
    end = int(time.time())
    start = end - 150*180  # 150 * 3min
    active_positions = sum(1 for s in SYMBOLS if SYMSTATES[s].position)
    for sym in SYMBOLS:
        st = SYMSTATES[sym]
        # cooldown?
        if st.cooldown_until and utcnow() < st.cooldown_until:
            continue
        # hämta data
        try:
            df = fetch_klines(sym, start, end, tf="3min")
        except Exception as e:
            log.warning(f"Kunde inte hämta klines {sym}: {e}")
            continue
        if len(df) < 50:  # behöver indikatorer
            continue

        # indikatorer
        df["ema_fast"] = ema(df["close"], AI["ema_fast"])
        df["ema_slow"] = ema(df["close"], AI["ema_slow"])
        df["atr14"] = atr(df, 14)
        df["adx14"] = adx(df, 14)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        atrp = (last["atr14"] / last["close"]) if last["close"] else 0.0

        # ORB detektion
        long_orb, short_orb, color = detect_orb(st.last_color, last["open"], last["close"], last["high"], last["low"])
        st.last_color = color
        now = last["time"].to_pydatetime().replace(tzinfo=timezone.utc)

        if long_orb:
            st.orb_long = (last["high"], last["low"], now)
        if short_orb:
            st.orb_short = (last["high"], last["low"], now)

        pos = st.position
        price = float(last["close"])

        # Hantera existerande position (SL/BE/Trailing + DCA)
        if pos:
            await manage_position(sym, st, price, last, atrp)
        else:
            # Inga nya trades om för många öppna
            active_positions = sum(1 for s in SYMBOLS if SYMSTATES[s].position)
            if active_positions >= MAX_CONCURRENT_COINS:
                continue
            # Nya entries endast inom handelfönster
            if not within_entry_hours(utcnow()):
                continue
            # filter: ADX/ATR%/trend
            if last["adx14"] < AI["adx_min"] or atrp < AI["atrp_min"]:
                continue
            long_trend = (last["close"] > last["ema_fast"] > last["ema_slow"])
            short_trend = (last["close"] < last["ema_fast"] < last["ema_slow"])
            # ENTRY_MODE
            mode = AI["ENTRY_MODE"].upper()

            # Long entry?
            if st.orb_long and long_trend:
                orb_h, orb_l, _ = st.orb_long
                trig = (last["close"] > orb_h) if mode == "CLOSE" else (last["high"] > orb_h)
                if trig:
                    await open_position(sym, st, "long", price, orb_h, orb_l, last["atr14"])
                    st.orb_long = None  # konsumerad
                    continue
            # Short entry?
            if st.orb_short and short_trend:
                orb_h, orb_l, _ = st.orb_short
                trig = (last["close"] < orb_l) if mode == "CLOSE" else (last["low"] < orb_l)
                if trig:
                    await open_position(sym, st, "short", price, orb_h, orb_l, last["atr14"])
                    st.orb_short = None
                    continue

async def open_position(sym: str, st: SymState, side: str, price: float, orb_h: float, orb_l: float, atr14: float):
    global AI
    # qty baserat på notional
    qty = max(MOCK_TRADE_NOTIONAL / price, 1e-6)
    # initial SL: max(ORB-extrem, atr_mult * ATR)
    if side == "long":
        atr_stop = price - AI["atr_mult_sl"] * atr14
        sl = max(orb_l, atr_stop)
        st.position = Position(side="long", qty=qty, avg_price=price, entry_price=price,
                               sl=sl, orb_high=orb_h, orb_low=orb_l, entry_time=utcnow(),
                               ai_mode=AI["mode"], max_fav=price, min_fav=price)
    else:
        atr_stop = price + AI["atr_mult_sl"] * atr14
        sl = min(orb_h, atr_stop)
        st.position = Position(side="short", qty=qty, avg_price=price, entry_price=price,
                               sl=sl, orb_high=orb_h, orb_low=orb_l, entry_time=utcnow(),
                               ai_mode=AI["mode"], max_fav=price, min_fav=price)
    st.trades_today += 1
    fee = price * qty * FEE_DEFAULT
    log_trade(True, sym, side, "ENTRY", price, qty, fee, 0.0, AI["mode"], note=f"ORB {AI['ENTRY_MODE']}")
    log.info(f"{sym} ENTRY {side.upper()} @ {price:.6f} | SL {st.position.sl:.6f} | qty {qty:.6f}")

async def manage_position(sym: str, st: SymState, price: float, last_row: pd.Series, atrp: float):
    global AI
    pos = st.position
    if not pos:
        return
    # uppdatera fav
    pos.max_fav = max(pos.max_fav, price)
    pos.min_fav = min(pos.min_fav, price)
    # Break-even
    be_level = pos.entry_price * (1 + AI["be_trigger"]) if pos.side == "long" else pos.entry_price * (1 - AI["be_trigger"])
    if (pos.side == "long" and price >= be_level) or (pos.side == "short" and price <= be_level):
        # flytta SL till BE
        new_sl = pos.entry_price if pos.side == "long" else pos.entry_price
        if pos.side == "long":
            pos.sl = max(pos.sl, new_sl)
        else:
            pos.sl = min(pos.sl, new_sl)
    # Trailing arm
    arm_level = pos.entry_price * (1 + AI["trail_arm"]) if pos.side == "long" else pos.entry_price * (1 - AI["trail_arm"])
    if (pos.side == "long" and price >= arm_level) or (pos.side == "short" and price <= arm_level):
        pos.trail_armed = True
    # Trailing
    if pos.trail_armed:
        if pos.side == "long":
            trail_sl = price * (1 - AI["trail_dist"])
            pos.sl = max(pos.sl, trail_sl)
        else:
            trail_sl = price * (1 + AI["trail_dist"])
            pos.sl = min(pos.sl, trail_sl)

    # DCA (max 1 ben, ATR% tillräcklig och ADX >= dca_adx_min)
    if (not st.dca_done) and AI["dca_max_legs"] > 0 and last_row["adx14"] >= AI["dca_adx_min"] and atrp >= AI["atrp_min"]:
        if pos.side == "long" and price <= pos.entry_price * (1 - AI["dca_trigger"]):
            add_qty = max(MOCK_TRADE_NOTIONAL / price, 1e-6)
            pos.avg_price = (pos.avg_price * pos.qty + price * add_qty) / (pos.qty + add_qty)
            pos.qty += add_qty
            st.dca_done = True
            log_trade(True, sym, "long", "DCA", price, add_qty, price*add_qty*FEE_DEFAULT, 0.0, pos.ai_mode, note="DCA1")
            log.info(f"{sym} DCA LONG @ {price:.6f}")
        if pos.side == "short" and price >= pos.entry_price * (1 + AI["dca_trigger"]):
            add_qty = max(MOCK_TRADE_NOTIONAL / price, 1e-6)
            pos.avg_price = (pos.avg_price * pos.qty + price * add_qty) / (pos.qty + add_qty)
            pos.qty += add_qty
            st.dca_done = True
            log_trade(True, sym, "short", "DCA", price, add_qty, price*add_qty*FEE_DEFAULT, 0.0, pos.ai_mode, note="DCA1")
            log.info(f"{sym} DCA SHORT @ {price:.6f}")

    # Exit på SL (tick = använder high/low i senaste candle)
    hit = False
    exit_price = pos.sl
    if pos.side == "long":
        if last_row["low"] <= pos.sl: hit = True
    else:
        if last_row["high"] >= pos.sl: hit = True

    if hit:
        # PnL
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - exit_price) * pos.qty
        fees = (pos.entry_price * pos.qty + exit_price * pos.qty) * FEE_DEFAULT
        pnl -= fees
        st.realized_pnl += pnl
        # update loss streak
        if pnl < 0: st.loss_streak += 1
        else: st.loss_streak = 0
        # cooldown
        if st.loss_streak >= 2:
            st.cooldown_until = utcnow() + timedelta(hours=2)
        # logg
        log_trade(True, sym, pos.side, "EXIT", exit_price, pos.qty, fees, pnl, pos.ai_mode, note="SL/Trail")
        log.info(f"{sym} EXIT {pos.side.upper()} @ {exit_price:.6f} | pnl {pnl:.4f} USDT")
        # stäng position
        st.position = None
        st.dca_done = False

# ============================
# Telegram-kommandon
# ============================
def require_token():
    if not TELEGRAM_TOKEN:
        raise SystemExit("Saknar TELEGRAM_TOKEN. Sätt env och starta igen.")

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    await update.message.reply_text(f"Hej! {BOT_NAME} är redo.\nSkriv /help för kommandon.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    txt = (
        "Kommandon:\n"
        "/status – motor/AI/symboler/PNL\n"
        "/engine_on – starta motor (bekräfta med JA)\n"
        "/engine_off – stoppa motor\n"
        "/live_on – slå på live-trading (kräver JA + API-nycklar)\n"
        "/live_off – slå av live-trading\n"
        "/set_ai <aggressiv|försiktig|neutral>\n"
        "/save_ai – spara AI-params till fil\n"
        "/load_ai – ladda AI-params från fil\n"
        "/pnl – visa PnL per symbol (mock)\n"
        "/reset_pnl – nollställ PnL (minne)\n"
        "/symbols – lista coins\n"
        "/backtest <symbol|all> <period t.ex. 3d/24h> [fee]\n"
        "/export_csv – skicka senaste backtest CSV\n"
        "/panic – stänger alla mock-positioner nu (bekräfta JA)\n"
    )
    await update.message.reply_text(txt)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    lines = [
        f"Engine: {'ON' if ENGINE_ON else 'OFF'}",
        f"Live: {'ON' if LIVE_TRADING else 'OFF'}",
        f"AI-läge: {AI['mode']} | ENTRY: {AI['ENTRY_MODE']} | BE {AI['be_trigger']*100:.2f}% | Arm {AI['trail_arm']*100:.2f}% | Trail {AI['trail_dist']*100:.2f}%",
        f"Tidsfönster: {ENGINE_TIME_START_UTC:02d}:00–{ENGINE_TIME_END_UTC:02d}:00 UTC",
        f"Max samtidiga coins: {MAX_CONCURRENT_COINS}",
        "— PnL (mock, realiserad):"
    ]
    total = 0.0
    for s in SYMBOLS:
        p = SYMSTATES[s].realized_pnl
        total += p
        lines.append(f"• {s}: {p:+.4f} USDT")
    lines.append(f"Summa: {total:+.4f} USDT")
    await update.message.reply_text("\n".join(lines))

async def symbols_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    await update.message.reply_text("Aktiva coins:\n" + ", ".join(SYMBOLS))

async def set_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, AI
    LAST_USER_UPDATE = utcnow()
    if not context.args:
        await update.message.reply_text("Använd: /set_ai <aggressiv|försiktig|neutral>")
        return
    val = context.args[0].lower()
    if val not in ["aggressiv","försiktig","neutral"]:
        await update.message.reply_text("Ogiltigt läge.")
        return
    AI["mode"] = val
    # små auto-justeringar per läge
    if val == "aggressiv":
        AI["adx_min"] = 14; AI["atrp_min"] = 0.0015
    elif val == "försiktig":
        AI["adx_min"] = 20; AI["atrp_min"] = 0.0025
    else:
        AI["adx_min"] = 18; AI["atrp_min"] = 0.0020
    await update.message.reply_text(f"AI-läge satt till {val} ✅")

async def save_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, AI
    LAST_USER_UPDATE = utcnow()
    save_ai_state(AI)
    await update.message.reply_text("AI-parametrar sparade till ai_state.json ✅")

async def load_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, AI
    LAST_USER_UPDATE = utcnow()
    AI = load_ai_state()
    await update.message.reply_text(f"AI-parametrar laddade ✅\nLäge: {AI['mode']} | ENTRY: {AI['ENTRY_MODE']}")

async def pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    lines = ["📈 PnL (mock, realiserad):"]
    total = 0.0
    for s in SYMBOLS:
        p = SYMSTATES[s].realized_pnl
        total += p
        lines.append(f"• {s}: {p:+.4f} USDT")
    lines.append(f"Total: {total:+.4f} USDT")
    await update.message.reply_text("\n".join(lines))

async def reset_pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    for s in SYMBOLS:
        SYMSTATES[s].realized_pnl = 0.0
    await update.message.reply_text("PnL nollställt (minne) ✅")

async def engine_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, CONFIRM_PENDING
    LAST_USER_UPDATE = utcnow()
    CONFIRM_PENDING = "engine_on"
    await update.message.reply_text("Bekräfta med 'JA' för att starta motorn.")

async def engine_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, CONFIRM_PENDING
    LAST_USER_UPDATE = utcnow()
    CONFIRM_PENDING = "engine_off"
    await update.message.reply_text("Bekräfta med 'JA' för att stoppa motorn.")

async def live_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, CONFIRM_PENDING
    LAST_USER_UPDATE = utcnow()
    if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
        await update.message.reply_text("API-nycklar saknas i env. Sätt KUCOIN_KEY/SECRET/PASSPHRASE.")
        return
    CONFIRM_PENDING = "live_on"
    await update.message.reply_text("Detta aktiverar RIKTIG handel. Svara 'JA' för att bekräfta.")

async def live_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, LIVE_TRADING
    LAST_USER_UPDATE = utcnow()
    LIVE_TRADING = False
    await update.message.reply_text("Live-handel avstängd.")

async def panic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, CONFIRM_PENDING
    LAST_USER_UPDATE = utcnow()
    CONFIRM_PENDING = "panic"
    await update.message.reply_text("PANIK: stäng alla mock-positioner? Bekräfta med 'JA'.")

async def text_catch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # används för bekräftelser
    global LAST_USER_UPDATE, CONFIRM_PENDING, ENGINE_ON, LIVE_TRADING
    LAST_USER_UPDATE = utcnow()
    msg = (update.message.text or "").strip().upper()
    if msg == "JA" and CONFIRM_PENDING:
        if CONFIRM_PENDING == "engine_on":
            ENGINE_ON = True
            await update.message.reply_text("Motor STARTAD ✅")
        elif CONFIRM_PENDING == "engine_off":
            ENGINE_ON = False
            await update.message.reply_text("Motor STOPPAD ✅")
        elif CONFIRM_PENDING == "live_on":
            LIVE_TRADING = True
            await update.message.reply_text("Live-handel AKTIVERAD ✅ (OBS! Endast mock i denna fil tills order-API binds på riktigt).")
        elif CONFIRM_PENDING == "panic":
            # stäng alla mock-positioner till senaste pris (utan verklig marknad)
            for s in SYMBOLS:
                st = SYMSTATES[s]
                if st.position:
                    pos = st.position
                    price = pos.sl
                    if pos.side == "long":
                        pnl = (price - pos.entry_price) * pos.qty
                    else:
                        pnl = (pos.entry_price - price) * pos.qty
                    fees = (pos.entry_price * pos.qty + price * pos.qty) * FEE_DEFAULT
                    pnl -= fees
                    st.realized_pnl += pnl
                    log_trade(True, s, pos.side, "EXIT", price, pos.qty, fees, pnl, pos.ai_mode, note="PANIC")
                    st.position = None
                    st.dca_done = False
            await update.message.reply_text("Alla mock-positioner stängda ✅")
        CONFIRM_PENDING = None

# ===== Backtest =====
def parse_period(s: str) -> int:
    s = s.lower().strip()
    if s.endswith("d"):
        return int(float(s[:-1]) * 24 * 60 * 60)
    if s.endswith("h"):
        return int(float(s[:-1]) * 60 * 60)
    raise ValueError("Ange t.ex. 3d eller 24h")

def backtest_symbol(symbol: str, seconds: int, fee: float) -> Tuple[pd.DataFrame, dict]:
    end = int(time.time())
    start = end - seconds
    df = fetch_klines(symbol, start, end, tf="3min")
    if len(df) < 60:
        return df, {"trades":0,"winrate":0.0,"pnl":0.0}
    # indikatorer
    df["ema_fast"] = ema(df["close"], AI["ema_fast"])
    df["ema_slow"] = ema(df["close"], AI["ema_slow"])
    df["atr14"] = atr(df, 14)
    df["adx14"] = adx(df, 14)

    # backtest state
    pos = None
    realized = 0.0
    wins = 0
    total_trades = 0
    dca_done = False

    records = []

    def entry(side, px, orb_h, orb_l, t):
        nonlocal pos, total_trades
        qty = max(MOCK_TRADE_NOTIONAL / px, 1e-6)
        if side == "long":
            sl = max(orb_l, px - AI["atr_mult_sl"] * df.loc[t,"atr14"])
        else:
            sl = min(orb_h, px + AI["atr_mult_sl"] * df.loc[t,"atr14"])
        pos = {
            "side": side, "qty": qty, "avg": px, "entry": px, "sl": sl,
            "trail": False, "max": px, "min": px
        }
        total_trades += 1
        records.append([df.index.get_loc(t), df.loc[t,"time"], "ENTRY", side, px, qty, 0.0, 0.0])

    def exit_at(px, t, note="EXIT"):
        nonlocal pos, realized, wins
        if not pos: return
        if pos["side"] == "long":
            pnl = (px - pos["entry"]) * pos["qty"]
        else:
            pnl = (pos["entry"] - px) * pos["qty"]
        fees = (pos["entry"] * pos["qty"] + px * pos["qty"]) * fee
        pnl -= fees
        realized += pnl
        if pnl > 0: wins += 1
        records.append([df.index.get_loc(t), df.loc[t,"time"], note, pos["side"], px, pos["qty"], fees, pnl])
        pos = None

    prev_color = None
    orb_long = None
    orb_short = None

    for i in range(2, len(df)):
        o,c,h,l = df.loc[df.index[i],"open"], df.loc[df.index[i],"close"], df.loc[df.index[i],"high"], df.loc[df.index[i],"low"]
        t = df.index[i]
        # ORB
        long_orb, short_orb, color = detect_orb(prev_color, o,c,h,l)
        prev_color = color
        if long_orb: orb_long = (long_orb[0], long_orb[1])
        if short_orb: orb_short = (short_orb[0], short_orb[1])
        # indicators
        adx_ok = df.loc[t,"adx14"] >= AI["adx_min"]
        atrp = df.loc[t,"atr14"] / (c if c else 1.0)
        atr_ok = atrp >= AI["atrp_min"]
        long_trend = (c > df.loc[t,"ema_fast"] > df.loc[t,"ema_slow"])
        short_trend = (c < df.loc[t,"ema_fast"] < df.loc[t,"ema_slow"])
        if pos:
            # manage
            pos["max"] = max(pos["max"], c)
            pos["min"] = min(pos["min"], c)
            # BE
            be = pos["entry"] * (1 + AI["be_trigger"] if pos["side"]=="long" else 1 - AI["be_trigger"])
            if (pos["side"]=="long" and c>=be) or (pos["side"]=="short" and c<=be):
                pos["sl"] = pos["entry"]
            # Arm
            arm = pos["entry"] * (1 + AI["trail_arm"] if pos["side"]=="long" else 1 - AI["trail_arm"])
            if (pos["side"]=="long" and c>=arm) or (pos["side"]=="short" and c<=arm):
                pos["trail"] = True
            # Trailing
            if pos["trail"]:
                if pos["side"]=="long":
                    pos["sl"] = max(pos["sl"], c*(1-AI["trail_dist"]))
                else:
                    pos["sl"] = min(pos["sl"], c*(1+AI["trail_dist"]))
            # DCA (en gång)
            if (not dca_done) and df.loc[t,"adx14"]>=AI["dca_adx_min"] and atr_ok:
                if pos["side"]=="long" and c <= pos["entry"]*(1-AI["dca_trigger"]):
                    add = max(MOCK_TRADE_NOTIONAL / c, 1e-6)
                    pos["avg"] = (pos["avg"]*pos["qty"] + c*add) / (pos["qty"]+add)
                    pos["qty"] += add
                    dca_done = True
                    records.append([df.index.get_loc(t), df.loc[t,"time"], "DCA", "long", c, add, 0.0, 0.0])
                if pos["side"]=="short" and c >= pos["entry"]*(1+AI["dca_trigger"]):
                    add = max(MOCK_TRADE_NOTIONAL / c, 1e-6)
                    pos["avg"] = (pos["avg"]*pos["qty"] + c*add) / (pos["qty"]+add)
                    pos["qty"] += add
                    dca_done = True
                    records.append([df.index.get_loc(t), df.loc[t,"time"], "DCA", "short", c, add, 0.0, 0.0])
            # SL hit?
            if (pos["side"]=="long" and l <= pos["sl"]) or (pos["side"]=="short" and h >= pos["sl"]):
                exit_at(pos["sl"], t, note="SL/Trail")
                dca_done = False
        else:
            # nya entries bara om filters OK
            if not (adx_ok and atr_ok):
                continue
            entry_mode = AI["ENTRY_MODE"].upper()
            # long
            if orb_long and long_trend:
                trig = (c > orb_long[0]) if entry_mode=="CLOSE" else (h > orb_long[0])
                if trig:
                    entry("long", c, orb_long[0], orb_long[1], t)
                    orb_long = None
                    continue
            # short
            if orb_short and short_trend:
                trig = (c < orb_short[1]) if entry_mode=="CLOSE" else (l < orb_short[1])
                if trig:
                    entry("short", c, orb_short[0], orb_short[1], t)
                    orb_short = None
                    continue

    # Sammanställ
    out = pd.DataFrame(records, columns=["idx","time","action","side","price","qty","fee","pnl"])
    stats = {
        "trades": int(out[out["action"]=="ENTRY"].shape[0]),
        "winrate": (wins / max(1,total_trades)) * 100.0,
        "pnl": float(realized)
    }
    return out, stats

async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    if len(context.args) < 2:
        await update.message.reply_text("Använd: /backtest <symbol|all> <period t.ex. 3d/24h> [fee]\nEx: /backtest btcusdt 3d 0.001")
        return
    symbol = context.args[0].upper()
    period = context.args[1]
    fee = float(context.args[2]) if len(context.args) >= 3 else FEE_DEFAULT
    secs = parse_period(period)

    syms = SYMBOLS if symbol == "ALL" else [symbol]
    all_out = []
    summary_lines = []
    total_pnl = 0.0
    total_trades = 0

    for s in syms:
        try:
            df, stats = backtest_symbol(s, secs, fee)
        except Exception as e:
            await update.message.reply_text(f"{s}: backtest fel: {e}")
            continue
        df["symbol"] = s
        all_out.append(df)
        total_pnl += stats["pnl"]
        total_trades += stats["trades"]
        summary_lines.append(f"• {s}: trades {stats['trades']}, win {stats['winrate']:.1f}%, pnl {stats['pnl']:+.4f} USDT")

    if all_out:
        bt = pd.concat(all_out, ignore_index=True)
        try:
            bt.to_csv(BACKTEST_CSV, index=False)
            await update.message.reply_document(document=open(BACKTEST_CSV, "rb"), filename=BACKTEST_CSV)
        except Exception as e:
            await update.message.reply_text(f"Kunde inte spara/skicka CSV: {e}")

    summary_lines.append(f"Totalt: trades {total_trades}, PnL {total_pnl:+.4f} USDT")
    await update.message.reply_text("\n".join(summary_lines))

async def export_csv_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    if os.path.exists(BACKTEST_CSV):
        await update.message.reply_document(document=open(BACKTEST_CSV, "rb"), filename=BACKTEST_CSV)
    else:
        await update.message.reply_text("Ingen backtest CSV hittad ännu.")

# ============================
# Watchdog / failsafe
# ============================
async def watchdog_loop(app: Application):
    global ENGINE_ON
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL_SEC)
        delta = (utcnow() - LAST_USER_UPDATE).total_seconds()
        if ENGINE_ON and delta > FAILSAFE_MAX_NOUPDATES_SEC:
            ENGINE_ON = False
            log.warning("Failsafe: inga Telegram-uppdateringar, motor stoppad.")

# ============================
# Huvud
# ============================
async def main():
    require_token()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Kommandon
    app.add_handler(CommandHandler("start", hello))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("symbols", symbols_cmd))
    app.add_handler(CommandHandler("set_ai", set_ai_cmd))
    app.add_handler(CommandHandler("save_ai", save_ai_cmd))
    app.add_handler(CommandHandler("load_ai", load_ai_cmd))
    app.add_handler(CommandHandler("pnl", pnl_cmd))
    app.add_handler(CommandHandler("reset_pnl", reset_pnl_cmd))
    app.add_handler(CommandHandler("engine_on", engine_on_cmd))
    app.add_handler(CommandHandler("engine_off", engine_off_cmd))
    app.add_handler(CommandHandler("live_on", live_on_cmd))
    app.add_handler(CommandHandler("live_off", live_off_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("export_csv", export_csv_cmd))

    # Bekräftelser (JA)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_catch))

    # Bakgrundsloopar
    asyncio.create_task(engine_loop(app))
    asyncio.create_task(watchdog_loop(app))

    log.info(f"{BOT_NAME} startar… mock-läge, AI={AI['mode']}, ENTRY={AI['ENTRY_MODE']}")
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Avslutar…")
