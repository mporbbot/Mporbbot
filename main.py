# main.py
# Mp ORBbot - REAL session ORB (15m), LONG only
# Mock-live uses real KuCoin market data (paper trades).
# Live trading is built-in (KuCoin), but only activates if you set KUCOIN_* env vars
# and run /trade_mode live.
#
# Env required:
#   TELEGRAM_TOKEN="xxxx"
#
# Optional for LIVE trading:
#   KUCOIN_KEY="..."
#   KUCOIN_SECRET="..."
#   KUCOIN_PASSPHRASE="..."
#
# Requirements:
#   pip install python-telegram-bot==20.7 requests

import os
import json
import csv
import time
import hmac
import base64
import hashlib
import asyncio
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone

import requests
from zoneinfo import ZoneInfo

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# -----------------------------
# Files & constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MOCK_LOG_PATH = os.path.join(DATA_DIR, "mock_trade_log.csv")
REAL_LOG_PATH = os.path.join(DATA_DIR, "real_trade_log.csv")

# Default coins (KuCoin symbols use DASH format in endpoints: BTC-USDT)
DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

# Session definitions (the "usual" setup)
UTC = timezone.utc
NY_TZ = ZoneInfo("America/New_York")

SESSIONS = [
    # name, tz, start_hour, start_minute, duration_minutes
    ("ASIA", UTC, 0, 0, 15),   # 00:00-00:15 UTC
    ("EU",   UTC, 7, 0, 15),   # 07:00-07:15 UTC
    ("NY",   NY_TZ, 9, 30, 15) # 09:30-09:45 New York (DST aware)
]

# Trading loop
POLL_SECONDS = 10

# Paper trade size
DEFAULT_MOCK_USDT = 30.0

# Fees (used for mock accounting; live fees vary)
DEFAULT_FEE_RATE = 0.001  # 0.1%

# Entry modes
ENTRY_MODES = ["break", "retest"]  # long only

# ORB/exit settings defaults
TRAIL_TRIGGER_PCT_DEFAULT = 0.20   # start trailing after +0.20%
TRAIL_DIST_PCT_DEFAULT = 0.30      # trail distance 0.30%
RETEST_BAND_PCT_DEFAULT = 0.10     # price allowed around ORB-high for retest (±0.10%)

# Safety: 1 trade per session per symbol (keeps it "classic ORB" & avoids overtrading)
MAX_TRADES_PER_SESSION_DEFAULT = 1


# -----------------------------
# Helpers
# -----------------------------
def now_utc() -> datetime:
    return datetime.now(tz=UTC)

def ts_ms() -> int:
    return int(time.time() * 1000)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def parse_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ".").strip())
    except Exception:
        return None

def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


# -----------------------------
# Config / State
# -----------------------------
@dataclass
class BotConfig:
    engine_on: bool = False

    # Trading mode: "mock" = paper trades with real KuCoin data, "live" = real KuCoin orders
    trade_mode: str = "mock"

    # Entry mode
    entry_mode: str = "break"  # break | retest

    # Confirmation buffer above ORB-high (in %). This is your "threshold".
    # Example: 0.90 => need price >= ORB_high * (1 + 0.009)
    threshold_pct: float = 0.30

    # Risk just for display right now; live sizing uses fixed funds unless you extend it
    risk_pct: float = 1.0

    # Exit settings
    trail_trigger_pct: float = TRAIL_TRIGGER_PCT_DEFAULT
    trail_dist_pct: float = TRAIL_DIST_PCT_DEFAULT
    retest_band_pct: float = RETEST_BAND_PCT_DEFAULT

    # ORB session control
    orb_minutes: int = 15
    max_trades_per_session: int = MAX_TRADES_PER_SESSION_DEFAULT

    fee_rate: float = DEFAULT_FEE_RATE
    active_coins: List[str] = None

    # Stats
    trades: int = 0
    total_net_pnl: float = 0.0

    # Notifications
    notify_chat_id: int = 0
    notify_trades: bool = True

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["active_coins"] is None:
            d["active_coins"] = DEFAULT_COINS.copy()
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "BotConfig":
        cfg = BotConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if cfg.active_coins is None:
            cfg.active_coins = DEFAULT_COINS.copy()

        if cfg.trade_mode not in ("mock", "live"):
            cfg.trade_mode = "mock"
        if cfg.entry_mode not in ENTRY_MODES:
            cfg.entry_mode = "break"

        cfg.threshold_pct = float(clamp(float(cfg.threshold_pct), 0.05, 2.0))
        cfg.trail_trigger_pct = float(clamp(float(cfg.trail_trigger_pct), 0.05, 5.0))
        cfg.trail_dist_pct = float(clamp(float(cfg.trail_dist_pct), 0.05, 5.0))
        cfg.retest_band_pct = float(clamp(float(cfg.retest_band_pct), 0.01, 1.0))
        cfg.max_trades_per_session = int(clamp(int(cfg.max_trades_per_session), 1, 5))
        cfg.orb_minutes = int(clamp(int(cfg.orb_minutes), 5, 30))
        try:
            cfg.notify_chat_id = int(cfg.notify_chat_id or 0)
        except Exception:
            cfg.notify_chat_id = 0
        cfg.notify_trades = bool(cfg.notify_trades) if cfg.notify_trades is not None else True
        return cfg


CFG: BotConfig = None

def load_config() -> BotConfig:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return BotConfig.from_json(json.load(f))
        except Exception:
            pass
    return BotConfig(active_coins=DEFAULT_COINS.copy())

def save_config() -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CFG.to_json(), f, indent=2, ensure_ascii=False)


# -----------------------------
# KuCoin client (public + private)
# -----------------------------
class KuCoinClient:
    def __init__(self):
        self.base = "https://api.kucoin.com"
        self.key = os.environ.get("KUCOIN_KEY", "").strip()
        self.secret = os.environ.get("KUCOIN_SECRET", "").strip()
        self.passphrase = os.environ.get("KUCOIN_PASSPHRASE", "").strip()

    def has_keys(self) -> bool:
        return bool(self.key and self.secret and self.passphrase)

    # ---- Public
    def get_price(self, symbol: str) -> float:
        # GET /api/v1/market/orderbook/level1?symbol=BTC-USDT
        r = requests.get(f"{self.base}/api/v1/market/orderbook/level1", params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        data = r.json()
        price = float(data["data"]["price"])
        return price

    def get_klines_1m(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict[str, float]]:
        """
        KuCoin candles endpoint returns list: [ time, open, close, high, low, volume, turnover ]
        type: 1min
        time in seconds (string)
        """
        params = {
            "symbol": symbol,
            "type": "1min",
            "startAt": int(start_ts),
            "endAt": int(end_ts),
        }
        r = requests.get(f"{self.base}/api/v1/market/candles", params=params, timeout=10)
        r.raise_for_status()
        raw = r.json().get("data", [])
        out = []
        for row in raw:
            # NOTE: KuCoin returns newest first. We'll normalize.
            t = int(row[0])
            o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
            out.append({"t": t, "o": o, "c": c, "h": h, "l": l})
        out.sort(key=lambda x: x["t"])
        return out

    # ---- Private helpers
    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        now = str(ts_ms())
        str_to_sign = now + method.upper() + path + (body or "")
        signature = base64.b64encode(hmac.new(self.secret.encode(), str_to_sign.encode(), hashlib.sha256).digest()).decode()

        passphrase = base64.b64encode(hmac.new(self.secret.encode(), self.passphrase.encode(), hashlib.sha256).digest()).decode()

        return {
            "KC-API-KEY": self.key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": now,
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    # ---- Private (LIVE)
    def place_market_buy_funds(self, symbol: str, funds_usdt: float) -> str:
        # POST /api/v1/orders
        path = "/api/v1/orders"
        payload = {
            "clientOid": str(ts_ms()),
            "side": "buy",
            "symbol": symbol,
            "type": "market",
            "funds": str(funds_usdt),
        }
        body = json.dumps(payload)
        headers = self._headers("POST", path, body)
        r = requests.post(self.base + path, data=body, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["data"]["orderId"]

    def place_market_sell_size(self, symbol: str, size: float) -> str:
        path = "/api/v1/orders"
        payload = {
            "clientOid": str(ts_ms()),
            "side": "sell",
            "symbol": symbol,
            "type": "market",
            "size": str(size),
        }
        body = json.dumps(payload)
        headers = self._headers("POST", path, body)
        r = requests.post(self.base + path, data=body, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["data"]["orderId"]


KC = KuCoinClient()


# -----------------------------
# Trading state per symbol
# -----------------------------
@dataclass
class OrbState:
    session_name: str = ""
    session_start_utc: int = 0
    orb_start_utc: int = 0
    orb_end_utc: int = 0
    orb_high: float = 0.0
    orb_low: float = 0.0
    orb_ready: bool = False

    # Entry mode tracking
    broke_above: bool = False
    retest_touched: bool = False

    # Trades per session
    trades_this_session: int = 0

@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    entry_ts: int
    stop_price: float
    trail_active: bool = False
    trail_stop: float = 0.0
    peak_price: float = 0.0
    live_order_id: str = ""


ORB: Dict[str, OrbState] = {}
POS: Dict[str, Position] = {}

def init_states():
    for sym in CFG.active_coins:
        ORB.setdefault(sym, OrbState())


# -----------------------------
# UI / keyboards
# -----------------------------
def main_menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/threshold"), KeyboardButton("/entry_mode")],
        [KeyboardButton("/trade_mode"), KeyboardButton("/notify")],
        [KeyboardButton("/close_all"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def entry_mode_buttons(current: str) -> InlineKeyboardMarkup:
    row = []
    for m in ENTRY_MODES:
        label = m + (" ✅" if m == current else "")
        row.append(InlineKeyboardButton(label, callback_data=f"em:{m}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="em:close")]])

def trade_mode_buttons(current: str) -> InlineKeyboardMarkup:
    row = []
    for m in ["mock", "live"]:
        label = m + (" ✅" if m == current else "")
        row.append(InlineKeyboardButton(label, callback_data=f"tm:{m}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="tm:close")]])

def threshold_buttons(current: float) -> InlineKeyboardMarkup:
    vals = []
    v = 0.05
    while v <= 2.0 + 1e-9:
        vals.append(round(v, 2))
        v += 0.05
    rows, row = [], []
    for x in vals:
        label = f"{x:.2f}%" + (" ✅" if abs(x - current) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"thr:{x:.2f}"))
        if len(row) == 4:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("Stäng", callback_data="thr:close")])
    return InlineKeyboardMarkup(rows)


# -----------------------------
# Notifications + logging
# -----------------------------
async def notify_trade(app: Application, text: str):
    try:
        if not CFG.notify_trades or not CFG.notify_chat_id:
            return
        await app.bot.send_message(chat_id=CFG.notify_chat_id, text=text)
    except Exception:
        pass

def log_trade(is_mock: bool, symbol: str, side: str, qty: float,
              entry_price: float, exit_price: float,
              gross_pnl: float, fee_paid: float, net_pnl: float, reason: str):
    path = MOCK_LOG_PATH if is_mock else REAL_LOG_PATH
    ensure_csv_header(
        path,
        ["timestamp", "symbol", "side", "qty", "entry_price", "exit_price", "gross_pnl", "fee_paid", "net_pnl", "reason"],
    )
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            int(time.time()), symbol, side, f"{qty:.8f}",
            f"{entry_price:.8f}", f"{exit_price:.8f}",
            f"{gross_pnl:.8f}", f"{fee_paid:.8f}", f"{net_pnl:.8f}", reason
        ])


# -----------------------------
# Session / ORB calculations
# -----------------------------
def session_start_dt_for_day(session_name: str, tz: ZoneInfo, hour: int, minute: int, ref_utc: datetime) -> datetime:
    # Build a datetime in that timezone for "today" relative to that timezone, then convert to UTC
    local = ref_utc.astimezone(tz)
    local_start = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # if local time already passed significantly, keep today's; if we're before, still today
    # ORB refresh should happen when we cross it
    return local_start.astimezone(UTC)

def get_current_session(ref_utc: datetime) -> Dict[str, Any]:
    """
    Returns dict:
      {name, start_utc_dt, orb_start_utc, orb_end_utc, next_session_start_utc_dt}
    Based on the three session starts.
    """
    starts = []
    for name, tz, h, m, orb_min in SESSIONS:
        sdt = session_start_dt_for_day(name, tz, h, m, ref_utc)
        starts.append((name, sdt, orb_min))

    # handle ordering + if a start is in the future relative to now, it still belongs to today;
    # for previous sessions, it may be earlier today. If we're before ASIA(00:00 UTC), the "current session"
    # is NY from yesterday—simplify by also adding yesterday's starts.
    starts2 = []
    for name, tz, h, m, orb_min in SESSIONS:
        sdt_today = session_start_dt_for_day(name, tz, h, m, ref_utc)
        sdt_yday = (ref_utc - timedelta(days=1)).astimezone(tz).replace(hour=h, minute=m, second=0, microsecond=0).astimezone(UTC)
        starts2.append((name, sdt_yday, orb_min))
        starts2.append((name, sdt_today, orb_min))

    starts2.sort(key=lambda x: x[1])
    # pick latest start <= now
    current = None
    for name, sdt, orb_min in starts2:
        if sdt <= ref_utc:
            current = (name, sdt, orb_min)
        else:
            break

    # next session start = first start > now from today list
    future = [x for x in starts2 if x[1] > ref_utc]
    next_start = future[0][1] if future else (starts2[0][1] + timedelta(days=1))

    name, sdt, orb_min = current
    orb_start = sdt
    orb_end = sdt + timedelta(minutes=orb_min)
    return {
        "name": name,
        "start_utc_dt": sdt,
        "orb_start_utc": int(orb_start.timestamp()),
        "orb_end_utc": int(orb_end.timestamp()),
        "next_start_utc_dt": next_start,
        "orb_minutes": orb_min
    }


async def compute_orb_if_ready(symbol: str, ref_utc: datetime) -> None:
    """
    Ensure ORB state matches current session. If ORB window ended and we have data, compute H/L.
    """
    st = ORB[symbol]
    sess = get_current_session(ref_utc)

    # Session changed?
    if st.session_name != sess["name"] or st.session_start_utc != int(sess["start_utc_dt"].timestamp()):
        st.session_name = sess["name"]
        st.session_start_utc = int(sess["start_utc_dt"].timestamp())
        st.orb_start_utc = sess["orb_start_utc"]
        st.orb_end_utc = sess["orb_end_utc"]
        st.orb_high = 0.0
        st.orb_low = 0.0
        st.orb_ready = False
        st.broke_above = False
        st.retest_touched = False
        st.trades_this_session = 0

    # Wait until ORB window has ended to "lock" the range
    if st.orb_ready:
        return
    if int(ref_utc.timestamp()) < st.orb_end_utc:
        return

    # Pull 1m candles for the ORB window
    try:
        candles = await asyncio.to_thread(KC.get_klines_1m, symbol, st.orb_start_utc, st.orb_end_utc)
        # need at least ~15 candles (but KuCoin can return fewer near boundaries). Use what we have if >= 10.
        if len(candles) < max(10, CFG.orb_minutes - 5):
            return

        hi = max(c["h"] for c in candles)
        lo = min(c["l"] for c in candles)
        st.orb_high = float(hi)
        st.orb_low = float(lo)
        st.orb_ready = True
    except Exception:
        return


# -----------------------------
# Trading logic (LONG only)
# -----------------------------
def entry_break_condition(price: float, orb_high: float, threshold_pct: float) -> bool:
    return price >= orb_high * (1.0 + threshold_pct / 100.0)

def entry_retest_condition(st: OrbState, price: float, orb_high: float, threshold_pct: float, band_pct: float) -> bool:
    # Step 1: detect initial break
    if not st.broke_above and entry_break_condition(price, orb_high, threshold_pct):
        st.broke_above = True

    if not st.broke_above:
        return False

    # Step 2: detect retest touch near ORB-high
    upper = orb_high * (1.0 + band_pct / 100.0)
    lower = orb_high * (1.0 - band_pct / 100.0)
    if not st.retest_touched and (lower <= price <= upper):
        st.retest_touched = True

    if not st.retest_touched:
        return False

    # Step 3: confirm break again
    return entry_break_condition(price, orb_high, threshold_pct)

def calc_fees(entry_price: float, exit_price: float, qty: float, fee_rate: float) -> float:
    # Approx: fee on entry + exit
    return (entry_price * qty + exit_price * qty) * fee_rate

async def open_long(app: Application, symbol: str, price: float, orb_low: float) -> None:
    if symbol in POS:
        return
    if ORB[symbol].trades_this_session >= CFG.max_trades_per_session:
        return

    # mock sizing: fixed funds
    funds = DEFAULT_MOCK_USDT
    qty = funds / price

    if CFG.trade_mode == "live":
        # Require keys
        if not KC.has_keys():
            # fallback to mock if user turned on live without keys
            CFG.trade_mode = "mock"
            save_config()
            await notify_trade(app, "⚠️ KUCOIN keys saknas. Växlar till mock.")
        else:
            # place market buy with funds
            try:
                order_id = await asyncio.to_thread(KC.place_market_buy_funds, symbol, funds)
                # We still track entry at current price (practical approximation)
                POS[symbol] = Position(
                    symbol=symbol,
                    entry_price=price,
                    qty=qty,
                    entry_ts=int(time.time()),
                    stop_price=orb_low,
                    trail_active=False,
                    trail_stop=0.0,
                    peak_price=price,
                    live_order_id=order_id,
                )
                ORB[symbol].trades_this_session += 1
                await notify_trade(app, f"🟢 ENTRY (LIVE) {symbol} LONG @ {price:.4f} | ORB L={orb_low:.4f}")
                return
            except Exception as e:
                CFG.trade_mode = "mock"
                save_config()
                await notify_trade(app, f"⚠️ Live BUY misslyckades, fallback mock. ({type(e).__name__})")

    # MOCK entry
    POS[symbol] = Position(
        symbol=symbol,
        entry_price=price,
        qty=qty,
        entry_ts=int(time.time()),
        stop_price=orb_low,
        trail_active=False,
        trail_stop=0.0,
        peak_price=price,
    )
    ORB[symbol].trades_this_session += 1
    await notify_trade(app, f"🟢 ENTRY {symbol} LONG @ {price:.4f} | ORB H={ORB[symbol].orb_high:.4f} L={orb_low:.4f}")

async def close_long(app: Application, symbol: str, price: float, reason: str) -> None:
    if symbol not in POS:
        return
    p = POS.pop(symbol)

    gross = (price - p.entry_price) * p.qty
    fee = calc_fees(p.entry_price, price, p.qty, CFG.fee_rate)
    net = gross - fee

    CFG.trades += 1
    CFG.total_net_pnl += net
    save_config()

    # Live: attempt market sell
    if CFG.trade_mode == "live" and KC.has_keys():
        try:
            await asyncio.to_thread(KC.place_market_sell_size, symbol, p.qty)
        except Exception:
            # We still record and notify; live sell failed is serious but we won’t crash bot
            await notify_trade(app, f"⚠️ Live SELL misslyckades för {symbol}.")

    log_trade(
        is_mock=(CFG.trade_mode == "mock"),
        symbol=symbol,
        side="LONG",
        qty=p.qty,
        entry_price=p.entry_price,
        exit_price=price,
        gross_pnl=gross,
        fee_paid=fee,
        net_pnl=net,
        reason=reason
    )

    emoji = "🎯" if net >= 0 else "🟥"
    await notify_trade(app, f"{emoji} EXIT {symbol} @ {price:.4f} | Net {net:+.4f} USDT | {reason}")

def update_trailing(pos: Position, price: float) -> None:
    if price > pos.peak_price:
        pos.peak_price = price
    if not pos.trail_active:
        # activate after trigger
        if price >= pos.entry_price * (1.0 + CFG.trail_trigger_pct / 100.0):
            pos.trail_active = True
            pos.trail_stop = price * (1.0 - CFG.trail_dist_pct / 100.0)
    else:
        # move trail with new peaks
        new_trail = pos.peak_price * (1.0 - CFG.trail_dist_pct / 100.0)
        if new_trail > pos.trail_stop:
            pos.trail_stop = new_trail


# -----------------------------
# Engine loop
# -----------------------------
async def engine_background_loop(app: Application):
    init_states()
    while True:
        try:
            if CFG.engine_on:
                ref = now_utc()
                # compute ORB per symbol if ready
                for sym in CFG.active_coins:
                    await compute_orb_if_ready(sym, ref)

                # trading decisions
                for sym in CFG.active_coins:
                    st = ORB[sym]
                    if not st.orb_ready:
                        continue

                    price = await asyncio.to_thread(KC.get_price, sym)

                    # manage open positions first
                    if sym in POS:
                        pos = POS[sym]
                        update_trailing(pos, price)

                        # stop loss at ORB low
                        if price <= pos.stop_price:
                            await close_long(app, sym, price, reason="STOP_ORB_LOW")
                            continue

                        # trailing stop
                        if pos.trail_active and price <= pos.trail_stop:
                            await close_long(app, sym, price, reason="TRAIL_STOP")
                            continue

                        continue  # no new entries if already in position

                    # entry (LONG only)
                    if st.trades_this_session >= CFG.max_trades_per_session:
                        continue

                    if CFG.entry_mode == "break":
                        if entry_break_condition(price, st.orb_high, CFG.threshold_pct):
                            await open_long(app, sym, price, st.orb_low)
                    else:
                        if entry_retest_condition(st, price, st.orb_high, CFG.threshold_pct, CFG.retest_band_pct):
                            await open_long(app, sym, price, st.orb_low)

        except Exception:
            # Never crash the bot due to engine exceptions
            pass

        await asyncio.sleep(POLL_SECONDS)


# -----------------------------
# Telegram commands
# -----------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # save chat id for notifications
    CFG.notify_chat_id = int(update.effective_chat.id)
    save_config()

    await update.message.reply_text(
        "Mp ORBbot (REAL ORB) är igång.\n"
        "• Session ORB 15m: ASIA 00:00 UTC | EU 07:00 UTC | NY 09:30 NY\n"
        "• LONG only\n\n"
        "Kommandon:\n"
        "• /engine_on  /engine_off\n"
        "• /entry_mode  (break/retest)\n"
        "• /threshold 0.9\n"
        "• /trade_mode  (mock/live)\n"
        "• /status\n",
        reply_markup=main_menu_keyboard()
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # show current session + ORB per first symbol
    ref = now_utc()
    sess = get_current_session(ref)
    lines = [
        f"ENGINE: {'ON' if CFG.engine_on else 'OFF'}",
        f"Trade mode: {CFG.trade_mode} (live keys: {'OK' if KC.has_keys() else 'missing'})",
        f"Entry mode: {CFG.entry_mode}",
        f"Threshold: {CFG.threshold_pct:.2f}%",
        f"Trail trig/dist: {CFG.trail_trigger_pct:.2f}% / {CFG.trail_dist_pct:.2f}%",
        f"Retest band: ±{CFG.retest_band_pct:.2f}%",
        f"Trades: {CFG.trades}",
        f"Total NET PnL: {CFG.total_net_pnl:.4f} USDT",
        f"Session: {sess['name']} | ORB window ends (UTC): {datetime.fromtimestamp(sess['orb_end_utc'], tz=UTC).strftime('%H:%M')}",
        f"Coins: {CFG.active_coins}",
        f"Open positions: {list(POS.keys()) if POS else 'none'}",
        f"Notify: {'ON' if CFG.notify_trades else 'OFF'}",
    ]

    # include ORB state for first two symbols for quick debugging
    for sym in CFG.active_coins[:2]:
        st = ORB.get(sym, OrbState())
        if st.orb_ready:
            lines.append(f"{sym} ORB: H={st.orb_high:.4f} L={st.orb_low:.4f} | trades/session={st.trades_this_session}")
        else:
            lines.append(f"{sym} ORB: building... (ready={st.orb_ready})")

    await update.message.reply_text("\n".join(lines), reply_markup=main_menu_keyboard())

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Total NET PnL: {CFG.total_net_pnl:.4f} USDT\nTrades: {CFG.trades}")

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CFG.engine_on = True
    save_config()
    await update.message.reply_text("✅ ENGINE ON")

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CFG.engine_on = False
    save_config()
    await update.message.reply_text("⛔ ENGINE OFF")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CFG.total_net_pnl = 0.0
    CFG.trades = 0
    save_config()
    await update.message.reply_text("✅ PnL & trade-counter nollställda.")

async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not POS:
        await update.message.reply_text("Inga öppna positioner.")
        return
    # close at current market price
    total = 0.0
    for sym in list(POS.keys()):
        try:
            price = await asyncio.to_thread(KC.get_price, sym)
            p = POS[sym]
            gross = (price - p.entry_price) * p.qty
            fee = calc_fees(p.entry_price, price, p.qty, CFG.fee_rate)
            net = gross - fee
            total += net
            await close_long(context.application, sym, price, reason="CLOSE_ALL")
        except Exception:
            pass
    await update.message.reply_text(f"✅ Close all skickad. (Net approx: {total:+.4f} USDT)")

async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if not args:
        await update.message.reply_text(f"notify_trades: {CFG.notify_trades}\nAnvänd: /notify on eller /notify off")
        return
    v = args[0].lower().strip()
    if v in ("on", "1", "true", "yes"):
        CFG.notify_trades = True
        CFG.notify_chat_id = int(update.effective_chat.id)
        save_config()
        await update.message.reply_text("✅ Notiser PÅ.")
        return
    if v in ("off", "0", "false", "no"):
        CFG.notify_trades = False
        save_config()
        await update.message.reply_text("⛔ Notiser AV.")
        return
    await update.message.reply_text("Ogiltigt. Använd: /notify on eller /notify off")

async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        v = parse_float(args[0])
        if v is None:
            await update.message.reply_text("Ogiltigt värde. Ex: /threshold 0.9")
            return
        CFG.threshold_pct = float(round(clamp(v, 0.05, 2.0), 4))
        save_config()
        await update.message.reply_text(f"✅ Threshold satt till: {CFG.threshold_pct:.2f}%")
        return
    await update.message.reply_text(
        f"Threshold nu: {CFG.threshold_pct:.2f}%\nVälj eller skriv /threshold 0.9",
        reply_markup=threshold_buttons(CFG.threshold_pct)
    )

async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        m = args[0].lower().strip()
        if m not in ENTRY_MODES:
            await update.message.reply_text("Ogiltigt. Använd: /entry_mode break eller /entry_mode retest")
            return
        CFG.entry_mode = m
        save_config()
        await update.message.reply_text(f"✅ Entry mode: {CFG.entry_mode}")
        return
    await update.message.reply_text(
        f"Entry mode nu: {CFG.entry_mode}\nVälj:",
        reply_markup=entry_mode_buttons(CFG.entry_mode)
    )

async def cmd_trade_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        m = args[0].lower().strip()
        if m not in ("mock", "live"):
            await update.message.reply_text("Ogiltigt. Använd: /trade_mode mock eller /trade_mode live")
            return
        if m == "live" and not KC.has_keys():
            await update.message.reply_text("⚠️ KUCOIN_KEY/SECRET/PASSPHRASE saknas. Sätt dem först, annars kör mock.")
            return
        CFG.trade_mode = m
        save_config()
        await update.message.reply_text(f"✅ Trade mode: {CFG.trade_mode}")
        return
    await update.message.reply_text(
        f"Trade mode nu: {CFG.trade_mode}\nVälj:",
        reply_markup=trade_mode_buttons(CFG.trade_mode)
    )


# -----------------------------
# Callbacks
# -----------------------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()

    if data.startswith("em:"):
        payload = data.split("em:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        if payload in ENTRY_MODES:
            CFG.entry_mode = payload
            save_config()
            await q.edit_message_text(f"✅ Entry mode: {CFG.entry_mode}")
            return

    if data.startswith("tm:"):
        payload = data.split("tm:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        if payload in ("mock", "live"):
            if payload == "live" and not KC.has_keys():
                await q.edit_message_text("⚠️ KUCOIN keys saknas. Kan inte slå på live.")
                return
            CFG.trade_mode = payload
            save_config()
            await q.edit_message_text(f"✅ Trade mode: {CFG.trade_mode}")
            return

    if data.startswith("thr:"):
        payload = data.split("thr:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        v = parse_float(payload)
        if v is None:
            await q.edit_message_text("Ogiltigt threshold.")
            return
        CFG.threshold_pct = float(round(clamp(v, 0.05, 2.0), 4))
        save_config()
        await q.edit_message_text(f"✅ Threshold satt till: {CFG.threshold_pct:.2f}%")
        return

    await q.edit_message_text("Ok.")


# -----------------------------
# Boot
# -----------------------------
def build_app() -> Application:
    token = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Sätt TELEGRAM_TOKEN i environment innan du startar boten.")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("notify", cmd_notify))
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    app.add_handler(CommandHandler("trade_mode", cmd_trade_mode))

    app.add_handler(CallbackQueryHandler(on_callback))

    return app

async def post_init(app: Application):
    # start trading loop (no job_queue)
    app.create_task(engine_background_loop(app))

def main():
    global CFG
    CFG = load_config()
    if CFG.active_coins is None:
        CFG.active_coins = DEFAULT_COINS.copy()
    save_config()

    ensure_csv_header(
        MOCK_LOG_PATH,
        ["timestamp", "symbol", "side", "qty", "entry_price", "exit_price", "gross_pnl", "fee_paid", "net_pnl", "reason"],
    )
    ensure_csv_header(
        REAL_LOG_PATH,
        ["timestamp", "symbol", "side", "qty", "entry_price", "exit_price", "gross_pnl", "fee_paid", "net_pnl", "reason"],
    )

    init_states()

    app = build_app()
    app.post_init = post_init
    print("Mp ORBbot running (REAL ORB)...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
