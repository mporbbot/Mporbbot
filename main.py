# main.py — Mp ORBbot (Render/ASGI-ready, token-safe) — 2025-09-04
# Kör på Render med: uvicorn main:app --host 0.0.0.0 --port $PORT

import os, sys, time, json, math, hmac, hashlib, base64, asyncio, logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ======================
# Konfiguration
# ======================
BOT_NAME = "Mp ORBbot"
SYMBOLS = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAME = "3m"
FEE_DEFAULT = 0.001                # 0.1 % per sida
MOCK_TRADE_NOTIONAL = 15.0         # USDT per trade (mock)
MAX_CONCURRENT_COINS = 2
ENGINE_TIME_START_UTC = 7          # 07:00 UTC
ENGINE_TIME_END_UTC = 20           # 20:00 UTC

AI_STATE_FILE = "ai_state.json"
AI_DEFAULT = {
    "mode": "neutral",             # 'aggressiv' | 'försiktig' | 'neutral'
    "ENTRY_MODE": "CLOSE",         # 'CLOSE' | 'TICK'
    "be_trigger": 0.0025,          # +0.25 %
    "trail_arm": 0.0045,           # +0.45 %
    "trail_dist": 0.0025,          # 0.25 %
    "atr_mult_sl": 0.6,            # SL = max(ORB-extrem, mult*ATR14)
    "adx_min": 18,                 # min ADX(14) för entry
    "atrp_min": 0.002,             # ATR% (ATR/Close) >= 0.20 %
    "ema_fast": 50,
    "ema_slow": 200,
    "dca_max_legs": 1,
    "dca_trigger": 0.0035,         # 0.35 %
    "dca_adx_min": 20,
}

FAILSAFE_MAX_NOUPDATES_SEC = 300   # 5 min utan Telegram => motor OFF
WATCHDOG_INTERVAL_SEC = 30

MOCK_LOG = "mock_trade_log.csv"    # endast mocktrades
REAL_LOG = "real_trade_log.csv"    # endast riktiga trades
BACKTEST_CSV = "backtest_results.csv"

# Miljö/nycklar
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip() or os.getenv("BOT_TOKEN", "").strip()
KUCOIN_KEY = os.getenv("KUCOIN_KEY", "").strip()
KUCOIN_SECRET = os.getenv("KUCOIN_SECRET", "").strip()
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE", "").strip()

# ======================
# Logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("MpORB")

# ======================
# Hjälp / utils
# ======================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def ensure_csv(path: str, columns: List[str]):
    if not os.path.exists(path):
        pd.DataFrame([], columns=columns).to_csv(path, index=False)

def within_entry_hours(ts: datetime) -> bool:
    hr = ts.hour
    if ENGINE_TIME_START_UTC <= ENGINE_TIME_END_UTC:
        return ENGINE_TIME_START_UTC <= hr < ENGINE_TIME_END_UTC
    return hr >= ENGINE_TIME_START_UTC or hr < ENGINE_TIME_END_UTC

def to_symbol_dash(sym: str) -> str:
    s = sym.upper().replace("/", "").replace("-", "")
    return f"{s[:-4]}-{s[-4:]}" if s.endswith("USDT") else s

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

# ======================
# Indikatorer
# ======================
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = (df["high"] - df["low"]).abs()
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int=14) -> pd.Series:
    return true_range(df).rolling(n).mean()

def adx(df: pd.DataFrame, n: int=14) -> pd.Series:
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

# ======================
# KuCoin API
# ======================
KU_BASE = "https://api.kucoin.com"

def ku_public(path: str, params: dict=None) -> dict:
    r = requests.get(KU_BASE + path, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if str(data.get("code")) != "200000":
        raise RuntimeError(f"KuCoin fel: {data}")
    return data["data"]

def fetch_klines(symbol: str, start: int, end: int, tf: str="3min") -> pd.DataFrame:
    params = {"type": tf, "symbol": to_symbol_dash(symbol), "startAt": start, "endAt": end}
    data = ku_public("/api/v1/market/candles", params)  # [time, open, close, high, low, vol, turnover], nyast först
    rows = []
    for r in reversed(data):
        ts = int(r[0])
        rows.append([datetime.fromtimestamp(ts, tz=timezone.utc), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
    df = pd.DataFrame(rows, columns=["time","open","close","high","low","volume"])
    return df

# ======================
# ORB
# ======================
def detect_orb(prev_color: Optional[str], o: float, c: float, h: float, l: float) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]], str]:
    color = "green" if c >= o else "red"
    long_orb = (h, l) if prev_color == "red" and color == "green" else None
    short_orb = (h, l) if prev_color == "green" and color == "red" else None
    return long_orb, short_orb, color

# ======================
# State / typer
# ======================
@dataclass
class Position:
    side: str   # 'long' | 'short'
    qty: float
    avg_price: float
    entry_price: float
    sl: float
    orb_high: float
    orb_low: float
    entry_time: datetime
    ai_mode: str
    trail_armed: bool = False
    max_fav: float = 0.0
    min_fav: float = 0.0

@dataclass
class SymState:
    last_color: Optional[str] = None
    orb_long: Optional[Tuple[float,float,datetime]] = None
    orb_short: Optional[Tuple[float,float,datetime]] = None
    position: Optional[Position] = None
    loss_streak: int = 0
    cooldown_until: Optional[datetime] = None
    trades_today: int = 0
    realized_pnl: float = 0.0
    dca_done: bool = False

AI = load_ai_state()
ENGINE_ON = False
LIVE_TRADING = False
LAST_USER_UPDATE = utcnow()
CONFIRM_PENDING: Optional[str] = None      # 'engine_on' | 'engine_off' | 'live_on' | 'panic'
SYM: Dict[str, SymState] = {s: SymState() for s in SYMBOLS}

ensure_csv(MOCK_LOG, ["time","symbol","side","action","price","qty","fee","pnl","ai","note"])
ensure_csv(REAL_LOG, ["time","symbol","side","action","price","qty","fee","pnl","ai","note"])

# ======================
# Logg
# ======================
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
        pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=False)
    except Exception as e:
        log.error(f"Loggfel {path}: {e}")

# ======================
# Handelsmotor
# ======================
async def engine_loop():
    global ENGINE_ON
    while True:
        try:
            if ENGINE_ON:
                await process_symbols()
        except Exception as e:
            log.error(f"Engine fel: {e}")
        await asyncio.sleep(20)

async def process_symbols():
    end = int(time.time())
    start = end - 150 * 180     # 150 st 3-min candles
    for sym in SYMBOLS:
        st = SYM[sym]
        # cooldown?
        if st.cooldown_until and utcnow() < st.cooldown_until:
            continue
        # data
        try:
            df = fetch_klines(sym, start, end, tf="3min")
        except Exception as e:
            log.warning(f"Kunde inte hämta klines {sym}: {e}")
            continue
        if len(df) < 60:
            continue

        df["ema_fast"] = ema(df["close"], AI["ema_fast"])
        df["ema_slow"] = ema(df["close"], AI["ema_slow"])
        df["atr14"] = atr(df, 14)
        df["adx14"] = adx(df, 14)

        last = df.iloc[-1]
        atrp = (last["atr14"] / last["close"]) if last["close"] else 0.0

        long_orb, short_orb, color = detect_orb(st.last_color, last["open"], last["close"], last["high"], last["low"])
        st.last_color = color
        nowt = last["time"].to_pydatetime().replace(tzinfo=timezone.utc)

        if long_orb:
            st.orb_long = (last["high"], last["low"], nowt)
        if short_orb:
            st.orb_short = (last["high"], last["low"], nowt)

        pos = st.position
        price = float(last["close"])

        if pos:
            await manage_position(sym, st, price, last, atrp)
            continue

        # inga nya trades om för många öppna
        if sum(1 for s in SYMBOLS if SYM[s].position) >= MAX_CONCURRENT_COINS:
            continue
        # handla bara inom tid
        if not within_entry_hours(utcnow()):
            continue
        # filter
        if last["adx14"] < AI["adx_min"] or atrp < AI["atrp_min"]:
            continue
        long_trend = (last["close"] > last["ema_fast"] > last["ema_slow"])
        short_trend = (last["close"] < last["ema_fast"] < last["ema_slow"])
        entry_mode = AI["ENTRY_MODE"].upper()

        # Long entry?
        if st.orb_long and long_trend:
            orb_h, orb_l, _ = st.orb_long
            trig = (last["close"] > orb_h) if entry_mode == "CLOSE" else (last["high"] > orb_h)
            if trig:
                await open_position(sym, st, "long", price, orb_h, orb_l, last["atr14"])
                st.orb_long = None
                continue
        # Short entry?
        if st.orb_short and short_trend:
            orb_h, orb_l, _ = st.orb_short
            trig = (last["close"] < orb_l) if entry_mode == "CLOSE" else (last["low"] < orb_l)
            if trig:
                await open_position(sym, st, "short", price, orb_h, orb_l, last["atr14"])
                st.orb_short = None
                continue

async def open_position(sym: str, st: SymState, side: str, price: float, orb_h: float, orb_l: float, atr14: float):
    qty = max(MOCK_TRADE_NOTIONAL / price, 1e-6)
    if side == "long":
        sl = max(orb_l, price - AI["atr_mult_sl"] * atr14)
    else:
        sl = min(orb_h, price + AI["atr_mult_sl"] * atr14)
    st.position = Position(side=side, qty=qty, avg_price=price, entry_price=price,
                           sl=sl, orb_high=orb_h, orb_low=orb_l, entry_time=utcnow(),
                           ai_mode=AI["mode"], max_fav=price, min_fav=price)
    st.trades_today += 1
    log_trade(True, sym, side, "ENTRY", price, qty, price*qty*FEE_DEFAULT, 0.0, AI["mode"], note=f"ORB {AI['ENTRY_MODE']}")
    log.info(f"{sym} ENTRY {side} @ {price:.6f} | SL {sl:.6f}")

async def manage_position(sym: str, st: SymState, price: float, last_row: pd.Series, atrp: float):
    pos = st.position
    if not pos:
        return
    pos.max_fav = max(pos.max_fav, price)
    pos.min_fav = min(pos.min_fav, price)

    # Break-even
    be = pos.entry_price * (1 + AI["be_trigger"]) if pos.side == "long" else pos.entry_price * (1 - AI["be_trigger"])
    if (pos.side == "long" and price >= be) or (pos.side == "short" and price <= be):
        pos.sl = pos.entry_price

    # Trailing arm
    arm = pos.entry_price * (1 + AI["trail_arm"]) if pos.side == "long" else pos.entry_price * (1 - AI["trail_arm"])
    if (pos.side == "long" and price >= arm) or (pos.side == "short" and price <= arm):
        pos.trail_armed = True

    # Trailing
    if pos.trail_armed:
        if pos.side == "long":
            pos.sl = max(pos.sl, price * (1 - AI["trail_dist"]))
        else:
            pos.sl = min(pos.sl, price * (1 + AI["trail_dist"]))

    # DCA (max 1 ben, ADX & ATR% OK)
    if (not st.dca_done) and AI["dca_max_legs"] > 0 and last_row["adx14"] >= AI["dca_adx_min"] and atrp >= AI["atrp_min"]:
        if pos.side == "long" and price <= pos.entry_price * (1 - AI["dca_trigger"]):
            add = max(MOCK_TRADE_NOTIONAL / price, 1e-6)
            pos.avg_price = (pos.avg_price * pos.qty + price * add) / (pos.qty + add)
            pos.qty += add
            st.dca_done = True
            log_trade(True, sym, "long", "DCA", price, add, price*add*FEE_DEFAULT, 0.0, pos.ai_mode, note="DCA1")
            log.info(f"{sym} DCA LONG @ {price:.6f}")
        if pos.side == "short" and price >= pos.entry_price * (1 + AI["dca_trigger"]):
            add = max(MOCK_TRADE_NOTIONAL / price, 1e-6)
            pos.avg_price = (pos.avg_price * pos.qty + price * add) / (pos.qty + add)
            pos.qty += add
            st.dca_done = True
            log_trade(True, sym, "short", "DCA", price, add, price*add*FEE_DEFAULT, 0.0, pos.ai_mode, note="DCA1")
            log.info(f"{sym} DCA SHORT @ {price:.6f}")

    # SL / Trail exit
    hit = False
    exit_price = pos.sl
    if pos.side == "long" and last_row["low"] <= pos.sl:
        hit = True
    if pos.side == "short" and last_row["high"] >= pos.sl:
        hit = True

    if hit:
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - exit_price) * pos.qty
        fees = (pos.entry_price * pos.qty + exit_price * pos.qty) * FEE_DEFAULT
        pnl -= fees
        st.realized_pnl += pnl
        st.loss_streak = st.loss_streak + 1 if pnl < 0 else 0
        if st.loss_streak >= 2:
            st.cooldown_until = utcnow() + timedelta(hours=2)
        log_trade(True, sym, pos.side, "EXIT", exit_price, pos.qty, fees, pnl, pos.ai_mode, note="SL/Trail")
        log.info(f"{sym} EXIT {pos.side} @ {exit_price:.6f} | pnl {pnl:+.4f}")
        st.position = None
        st.dca_done = False

# ======================
# Backtest
# ======================
def parse_period(s: str) -> int:
    s = s.lower().strip()
    if s.endswith("d"): return int(float(s[:-1]) * 24 * 60 * 60)
    if s.endswith("h"): return int(float(s[:-1]) * 60 * 60)
    raise ValueError("Ange t.ex. 3d eller 24h")

def backtest_symbol(symbol: str, seconds: int, fee: float):
    end = int(time.time()); start = end - seconds
    df = fetch_klines(symbol, start, end, tf="3min")
    if len(df) < 60: return pd.DataFrame(), {"trades":0,"winrate":0.0,"pnl":0.0}

    df["ema_fast"] = ema(df["close"], AI["ema_fast"])
    df["ema_slow"] = ema(df["close"], AI["ema_slow"])
    df["atr14"] = atr(df, 14)
    df["adx14"] = adx(df, 14)

    prev_color=None; orb_long=None; orb_short=None
    pos=None; realized=0.0; wins=0; total_entries=0; dca_done=False; rows=[]

    def entry(side, px, orb_h, orb_l, i):
        nonlocal pos, total_entries
        qty = max(MOCK_TRADE_NOTIONAL / px, 1e-6)
        sl = max(orb_l, px - AI["atr_mult_sl"] * df.loc[i,"atr14"]) if side=="long" else min(orb_h, px + AI["atr_mult_sl"] * df.loc[i,"atr14"])
        pos = {"side":side,"qty":qty,"entry":px,"avg":px,"sl":sl,"trail":False}
        total_entries += 1
        rows.append([df.loc[i,"time"], "ENTRY", side, px, qty, 0.0, 0.0])

    def exit_at(px, i, note="EXIT"):
        nonlocal pos, realized, wins
        if not pos: return
        pnl = (px - pos["entry"]) * pos["qty"] if pos["side"]=="long" else (pos["entry"] - px) * pos["qty"]
        fees = (pos["entry"] * pos["qty"] + px * pos["qty"]) * fee
        pnl -= fees
        realized += pnl
        if pnl > 0: wins += 1
        rows.append([df.loc[i,"time"], note, pos["side"], px, pos["qty"], fees, pnl])
        pos=None

    for i in range(2, len(df)):
        o,c,h,l = df.loc[i,"open"], df.loc[i,"close"], df.loc[i,"high"], df.loc[i,"low"]
        lo, so, col = detect_orb(prev_color, o,c,h,l); prev_color=col
        if lo: orb_long=(lo[0], lo[1])
        if so: orb_short=(so[0], so[1])

        adx_ok = df.loc[i,"adx14"] >= AI["adx_min"]
        atrp = df.loc[i,"atr14"] / (c if c else 1.0)
        atr_ok = atrp >= AI["atrp_min"]
        long_trend = (c > df.loc[i,"ema_fast"] > df.loc[i,"ema_slow"])
        short_trend = (c < df.loc[i,"ema_fast"] < df.loc[i,"ema_slow"])
        entry_mode = AI["ENTRY_MODE"].upper()

        if pos:
            be = pos["entry"] * (1 + AI["be_trigger"] if pos["side"]=="long" else 1 - AI["be_trigger"])
            if (pos["side"]=="long" and c>=be) or (pos["side"]=="short" and c<=be): pos["sl"] = pos["entry"]
            arm = pos["entry"] * (1 + AI["trail_arm"] if pos["side"]=="long" else 1 - AI["trail_arm"])
            if (pos["side"]=="long" and c>=arm) or (pos["side"]=="short" and c<=arm): pos["trail"] = True
            if pos["trail"]:
                pos["sl"] = max(pos["sl"], c*(1-AI["trail_dist"])) if pos["side"]=="long" else min(pos["sl"], c*(1+AI["trail_dist"]))
            if (not dca_done) and df.loc[i,"adx14"]>=AI["dca_adx_min"] and atr_ok:
                if pos["side"]=="long" and c <= pos["entry"]*(1-AI["dca_trigger"]):
                    add = max(MOCK_TRADE_NOTIONAL / c, 1e-6); dca_done=True
                    rows.append([df.loc[i,"time"], "DCA", "long", c, add, 0.0, 0.0])
                if pos["side"]=="short" and c >= pos["entry"]*(1+AI["dca_trigger"]):
                    add = max(MOCK_TRADE_NOTIONAL / c, 1e-6); dca_done=True
                    rows.append([df.loc[i,"time"], "DCA", "short", c, add, 0.0, 0.0])
            if (pos["side"]=="long" and l<=pos["sl"]) or (pos["side"]=="short" and h>=pos["sl"]):
                exit_at(pos["sl"], i, note="SL/Trail"); dca_done=False
        else:
            if not (adx_ok and atr_ok): continue
            if orb_long and long_trend:
                trig = (c > orb_long[0]) if entry_mode=="CLOSE" else (h > orb_long[0])
                if trig: entry("long", c, orb_long[0], orb_long[1], i); orb_long=None; continue
            if orb_short and short_trend:
                trig = (c < orb_short[1]) if entry_mode=="CLOSE" else (l < orb_short[1])
                if trig: entry("short", c, orb_short[0], orb_short[1], i); orb_short=None; continue

    out = pd.DataFrame(rows, columns=["time","action","side","price","qty","fee","pnl"])
    trades = int((out["action"]=="ENTRY").sum()) if not out.empty else 0
    wins = out[out["pnl"] > 0].shape[0] if not out.empty else 0
    stats = {"trades": trades, "winrate": (wins/max(1,trades))*100.0, "pnl": float(out["pnl"].sum() if "pnl" in out else 0.0)}
    return out, stats

# ======================
# Telegram-kommandon
# ======================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    await update.message.reply_text(f"Hej! {BOT_NAME} är igång.\nSkriv /help för kommandon.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    txt = (
        "/status – visa motor/AI/PNL\n"
        "/engine_on – starta motor (svara JA)\n"
        "/engine_off – stoppa motor (svara JA)\n"
        "/live_on – aktivera live (JA + API-nycklar krävs)\n"
        "/live_off – stäng av live\n"
        "/set_ai <aggressiv|försiktig|neutral>\n"
        "/save_ai – spara AI-parametrar\n"
        "/load_ai – ladda AI-parametrar\n"
        "/pnl – visa PnL\n"
        "/reset_pnl – nollställ PnL (minne)\n"
        "/symbols – lista coins\n"
        "/backtest <symbol|all> <period t.ex. 3d/24h> [fee]\n"
        "/export_csv – skicka senaste backtest CSV\n"
        "/panic – stäng alla mock-positioner (svara JA)\n"
    )
    await update.message.reply_text(txt)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    lines = [
        f"Engine: {'ON' if ENGINE_ON else 'OFF'}",
        f"Live: {'ON' if LIVE_TRADING else 'OFF'}",
        f"AI: {AI['mode']} | ENTRY={AI['ENTRY_MODE']} | BE {AI['be_trigger']*100:.2f}% | Arm {AI['trail_arm']*100:.2f}% | Trail {AI['trail_dist']*100:.2f}%",
        f"Tidsfönster: {ENGINE_TIME_START_UTC:02d}:00–{ENGINE_TIME_END_UTC:02d}:00 UTC",
        f"Max samtidiga coins: {MAX_CONCURRENT_COINS}",
        "— PnL (mock, realiserad per coin):"
    ]
    total = 0.0
    for s in SYMBOLS:
        p = SYM[s].realized_pnl
        total += p
        lines.append(f"• {s}: {p:+.4f} USDT")
    lines.append(f"Summa: {total:+.4f} USDT")
    await update.message.reply_text("\n".join(lines))

async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    await update.message.reply_text("Aktiva coins: " + ", ".join(SYMBOLS))

async def cmd_set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, AI
    LAST_USER_UPDATE = utcnow()
    if not context.args:
        await update.message.reply_text("Använd: /set_ai <aggressiv|försiktig|neutral>")
        return
    val = context.args[0].lower()
    if val not in ["aggressiv","försiktig","neutral"]:
        await update.message.reply_text("Ogiltigt AI-läge.")
        return
    AI["mode"] = val
    if val == "aggressiv":
        AI["adx_min"] = 14; AI["atrp_min"] = 0.0015
    elif val == "försiktig":
        AI["adx_min"] = 20; AI["atrp_min"] = 0.0025
    else:
        AI["adx_min"] = 18; AI["atrp_min"] = 0.0020
    await update.message.reply_text(f"AI-läge satt till {val} ✅")

async def cmd_save_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    save_ai_state(AI)
    await update.message.reply_text("AI-parametrar sparade ✅")

async def cmd_load_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, AI
    LAST_USER_UPDATE = utcnow()
    AI = load_ai_state()
    await update.message.reply_text(f"AI laddad ✅  (mode={AI['mode']}, ENTRY={AI['ENTRY_MODE']})")

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    lines = ["📈 PnL (mock, realiserad):"]
    tot = 0.0
    for s in SYMBOLS:
        p = SYM[s].realized_pnl
        tot += p
        lines.append(f"• {s}: {p:+.4f} USDT")
    lines.append(f"Total: {tot:+.4f} USDT")
    await update.message.reply_text("\n".join(lines))

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    for s in SYMBOLS:
        SYM[s].realized_pnl = 0.0
    await update.message.reply_text("PnL nollställt ✅")

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, CONFIRM_PENDING
    LAST_USER_UPDATE = utcnow()
    CONFIRM_PENDING = "engine_on"
    await update.message.reply_text("Svara 'JA' för att starta motorn.")

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, CONFIRM_PENDING
    LAST_USER_UPDATE = utcnow()
    CONFIRM_PENDING = "engine_off"
    await update.message.reply_text("Svara 'JA' för att stoppa motorn.")

async def cmd_live_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, CONFIRM_PENDING
    LAST_USER_UPDATE = utcnow()
    if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
        await update.message.reply_text("Saknar KUCOIN_KEY/SECRET/PASSPHRASE i env.")
        return
    CONFIRM_PENDING = "live_on"
    await update.message.reply_text("Aktivera RIKTIG handel? Svara 'JA' för att bekräfta.")

async def cmd_live_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, LIVE_TRADING
    LAST_USER_UPDATE = utcnow()
    LIVE_TRADING = False
    await update.message.reply_text("Live-handel avstängd.")

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE, CONFIRM_PENDING
    LAST_USER_UPDATE = utcnow()
    CONFIRM_PENDING = "panic"
    await update.message.reply_text("PANIK: stäng alla mock-positioner? Svara 'JA'.")

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    if len(context.args) < 2:
        await update.message.reply_text("Använd: /backtest <symbol|all> <period t.ex. 3d/24h> [fee]")
        return
    symbol = context.args[0].upper()
    period = context.args[1]
    fee = float(context.args[2]) if len(context.args) >= 3 else FEE_DEFAULT
    secs = parse_period(period)
    syms = SYMBOLS if symbol == "ALL" else [symbol]
    all_frames = []
    summary = []
    total_pnl = 0.0
    total_trades = 0
    for s in syms:
        try:
            bt, stats = backtest_symbol(s, secs, fee)
        except Exception as e:
            await update.message.reply_text(f"{s}: fel i backtest: {e}")
            continue
        if not bt.empty:
            bt["symbol"] = s
            all_frames.append(bt)
        total_pnl += stats["pnl"]; total_trades += stats["trades"]
        summary.append(f"• {s}: trades {stats['trades']}, win {stats['winrate']:.1f}%, pnl {stats['pnl']:+.4f} USDT")
    if all_frames:
        out = pd.concat(all_frames, ignore_index=True)
        try:
            out.to_csv(BACKTEST_CSV, index=False)
            await update.message.reply_document(open(BACKTEST_CSV,"rb"), filename=BACKTEST_CSV)
        except Exception as e:
            await update.message.reply_text(f"Kunde inte spara/skicka CSV: {e}")
    summary.append(f"Totalt: trades {total_trades}, PnL {total_pnl:+.4f} USDT")
    await update.message.reply_text("\n".join(summary))

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_USER_UPDATE
    LAST_USER_UPDATE = utcnow()
    if os.path.exists(BACKTEST_CSV):
        await update.message.reply_document(open(BACKTEST_CSV,"rb"), filename=BACKTEST_CSV)
    else:
        await update.message.reply_text("Ingen backtest CSV hittad.")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            await update.message.reply_text("Live-handel AKTIVERAD ✅ (orders inte kopplade i denna fil).")
        elif CONFIRM_PENDING == "panic":
            for s in SYMBOLS:
                st = SYM[s]
                if st.position:
                    pos = st.position
                    px = pos.sl
                    pnl = (px - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - px) * pos.qty
                    fees = (pos.entry_price * pos.qty + px * pos.qty) * FEE_DEFAULT
                    pnl -= fees
                    st.realized_pnl += pnl
                    log_trade(True, s, pos.side, "EXIT", px, pos.qty, fees, pnl, pos.ai_mode, note="PANIC")
                    st.position = None
                    st.dca_done = False
            await update.message.reply_text("Alla mock-positioner stängda ✅")
        CONFIRM_PENDING = None

# ======================
# Watchdog / failsafe
# ======================
async def watchdog_loop():
    global ENGINE_ON
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL_SEC)
        if ENGINE_ON and (utcnow() - LAST_USER_UPDATE).total_seconds() > FAILSAFE_MAX_NOUPDATES_SEC:
            ENGINE_ON = False
            log.warning("Failsafe: inga Telegram-uppdateringar, motor stoppad.")

# ======================
# Telegram-bot init (token-safe)
# ======================
telegram_app: Optional[Application] = None
bg_tasks: List[asyncio.Task] = []

async def start_telegram_bot_if_token():
    """Starta Telegram endast om TELEGRAM_TOKEN finns; annars logga varning och kör vidare."""
    global TELEGRAM_TOKEN, telegram_app, bg_tasks
    if not TELEGRAM_TOKEN:
        log.warning("Ingen TELEGRAM_TOKEN satt — startar ENDAST FastAPI (Telegram inaktiverad).")
        return
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()

    telegram_app.add_handler(CommandHandler("start", cmd_start))
    telegram_app.add_handler(CommandHandler("help", cmd_help))
    telegram_app.add_handler(CommandHandler("status", cmd_status))
    telegram_app.add_handler(CommandHandler("symbols", cmd_symbols))
    telegram_app.add_handler(CommandHandler("set_ai", cmd_set_ai))
    telegram_app.add_handler(CommandHandler("save_ai", cmd_save_ai))
    telegram_app.add_handler(CommandHandler("load_ai", cmd_load_ai))
    telegram_app.add_handler(CommandHandler("pnl", cmd_pnl))
    telegram_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    telegram_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    telegram_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    telegram_app.add_handler(CommandHandler("live_on", cmd_live_on))
    telegram_app.add_handler(CommandHandler("live_off", cmd_live_off))
    telegram_app.add_handler(CommandHandler("backtest", cmd_backtest))
    telegram_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()

    bg_tasks.append(asyncio.create_task(engine_loop()))
    bg_tasks.append(asyncio.create_task(watchdog_loop()))
    log.info(f"{BOT_NAME} startad — mock-läge, AI={AI['mode']}, ENTRY={AI['ENTRY_MODE']}")

async def stop_telegram_bot_if_running():
    global telegram_app, bg_tasks
    for t in bg_tasks:
        t.cancel()
    bg_tasks.clear()
    if telegram_app:
        await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()
        telegram_app = None
    log.info("Telegram-bot stoppad.")

# ======================
# FastAPI app (ASGI)
# ======================
app = FastAPI(title=BOT_NAME)

@app.get("/")
async def root():
    return {
        "ok": True,
        "name": BOT_NAME,
        "engine_on": ENGINE_ON,
        "live": LIVE_TRADING,
        "ai": {"mode": AI.get("mode"), "entry": AI.get("ENTRY_MODE")},
        "telegram_enabled": bool(TELEGRAM_TOKEN)
    }

@app.get("/metrics")
async def metrics():
    total = sum(SYM[s].realized_pnl for s in SYMBOLS)
    open_pos = {s: (SYM[s].position.side if SYM[s].position else None) for s in SYMBOLS}
    return {
        "pnl_total": round(total, 6),
        "open_positions": open_pos,
        "cooldowns": {s: (SYM[s].cooldown_until.isoformat() if SYM[s].cooldown_until else None) for s in SYMBOLS}
    }

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(start_telegram_bot_if_token())

@app.on_event("shutdown")
async def on_shutdown():
    await stop_telegram_bot_if_running()

# ======================
# __main__ (lokal körning)
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
