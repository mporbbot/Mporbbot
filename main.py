# main.py
# Mp ORBbot - Strategy: VWAP Dip Reversion (1m) with 5m Uptrend Filter
# LONG only, high winrate focus + "let winners run" via Partial TP + Trailing Runner.
#
# CHANGE (minimal): After PARTIAL_TP -> move SL to breakeven + buffer on remaining qty.
# This prevents a profitable partial trade from turning into a net loser due to runner SL.
#
# Telegram: keeps existing flow, adds /partial buttons (enable + fraction + partialTP).
# Token NOT hardcoded. Uses TELEGRAM_TOKEN or TELEGRAM_BOT_TOKEN env var.
#
# Commands:
#   /start /status /pnl /reset_pnl /engine_on /engine_off /close_all
#   /entry (BREAK/RETEST)
#   /threshold (dip-band % under VWAP)
#   /stake /tp /sl /coins /trade_mode /notify /export_csv
#   /partial

import os
import json
import csv
import time
import hmac
import base64
import hashlib
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import requests
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
# Paths & constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MOCK_LOG_PATH = os.path.join(DATA_DIR, "mock_trade_log.csv")
REAL_LOG_PATH = os.path.join(DATA_DIR, "real_trade_log.csv")

DEFAULT_COINS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "ADA-USDT", "LINK-USDT"]

UTC = timezone.utc

POLL_SECONDS = 6  # faster loop for 1m strategy
DEFAULT_FEE_RATE = 0.001  # 0.1% per side

# Strategy TFs
TF_SIGNAL = "1min"
TF_TREND = "5min"

# Trend filter EMAs on 5m
EMA_FAST = 50
EMA_SLOW = 200

# VWAP lookback on 1m (rolling intraday-ish)
VWAP_LOOKBACK = 240  # 240 minutes (~4h)

# /threshold now defines DIP band under VWAP (in %), e.g. 0.30% under VWAP
DIP_BAND_PCT_DEFAULT = 0.30

TP_PCT_DEFAULT = 0.70
SL_PCT_DEFAULT = 0.50

COOLDOWN_SECONDS_DEFAULT = 20
MAX_POSITIONS_DEFAULT = 5
STAKE_USDT_DEFAULT = 30.0

# Trailing runner (applies after partial)
TRAIL_ACTIVATE_PCT_DEFAULT = 0.40
TRAIL_DISTANCE_PCT_DEFAULT = 0.30

# Partial TP defaults
PARTIAL_ENABLED_DEFAULT = True
PARTIAL_TP_PCT_DEFAULT = 0.40
PARTIAL_FRACTION_DEFAULT = 0.50

# IMPORTANT: Breakeven buffer after partial (covers fees/slip). Minimal change.
PARTIAL_BE_BUFFER_PCT_DEFAULT = 0.15  # move SL to entry*(1+0.15%) after partial

# MOCK realism defaults
MOCK_USE_BID_ASK_DEFAULT = True
MOCK_SLIPPAGE_PCT_DEFAULT = 0.02  # 0.02% friction on top

ENTRY_MODES = ["break", "retest"]

# -----------------------------
# Helpers
# -----------------------------
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

def iso_now() -> str:
    return datetime.now(tz=UTC).isoformat()

def safe_symbol(sym: str) -> str:
    s = sym.upper().replace("_", "-").replace("/", "-")
    if "-" not in s and s.endswith("USDT"):
        s = s[:-4] + "-USDT"
    return s

def ema_last(values: List[float], period: int) -> Optional[float]:
    if len(values) < period + 5:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def rolling_vwap(candles: List[Dict[str, float]]) -> Optional[float]:
    if not candles:
        return None
    pv = 0.0
    vv = 0.0
    for c in candles:
        tp = (c["h"] + c["l"] + c["c"]) / 3.0
        v = float(c.get("v", 0.0))
        if v <= 0:
            continue
        pv += tp * v
        vv += v
    if vv <= 0:
        return None
    return pv / vv

# -----------------------------
# Config
# -----------------------------
@dataclass
class BotConfig:
    engine_on: bool = False
    trade_mode: str = "mock"      # mock | live
    entry_mode: str = "break"     # break | retest

    threshold_pct: float = DIP_BAND_PCT_DEFAULT
    fee_rate: float = DEFAULT_FEE_RATE
    active_coins: List[str] = None

    tp_pct: float = TP_PCT_DEFAULT
    sl_pct: float = SL_PCT_DEFAULT
    cooldown_seconds: int = COOLDOWN_SECONDS_DEFAULT
    max_positions: int = MAX_POSITIONS_DEFAULT
    stake_usdt: float = STAKE_USDT_DEFAULT

    partial_enabled: bool = PARTIAL_ENABLED_DEFAULT
    partial_tp_pct: float = PARTIAL_TP_PCT_DEFAULT
    partial_fraction: float = PARTIAL_FRACTION_DEFAULT
    trail_activate_pct: float = TRAIL_ACTIVATE_PCT_DEFAULT
    trail_distance_pct: float = TRAIL_DISTANCE_PCT_DEFAULT

    # NEW (minimal): BE buffer used after partial
    partial_be_buffer_pct: float = PARTIAL_BE_BUFFER_PCT_DEFAULT

    mock_use_bid_ask: bool = MOCK_USE_BID_ASK_DEFAULT
    mock_slippage_pct: float = MOCK_SLIPPAGE_PCT_DEFAULT

    trades: int = 0
    total_net_pnl: float = 0.0

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

        cfg.trade_mode = cfg.trade_mode if cfg.trade_mode in ("mock", "live") else "mock"
        cfg.entry_mode = cfg.entry_mode if cfg.entry_mode in ENTRY_MODES else "break"

        cfg.threshold_pct = float(clamp(float(cfg.threshold_pct), 0.05, 2.0))
        cfg.tp_pct = float(clamp(float(cfg.tp_pct), 0.20, 2.00))
        cfg.sl_pct = float(clamp(float(cfg.sl_pct), 0.10, 3.00))
        cfg.cooldown_seconds = int(clamp(int(cfg.cooldown_seconds), 0, 600))
        cfg.max_positions = int(clamp(int(cfg.max_positions), 1, 20))
        cfg.stake_usdt = float(clamp(float(cfg.stake_usdt), 5.0, 500.0))

        cfg.partial_enabled = bool(cfg.partial_enabled)
        cfg.partial_tp_pct = float(clamp(float(cfg.partial_tp_pct), 0.10, 2.00))
        cfg.partial_fraction = float(clamp(float(cfg.partial_fraction), 0.10, 0.90))
        cfg.trail_activate_pct = float(clamp(float(cfg.trail_activate_pct), 0.10, 3.00))
        cfg.trail_distance_pct = float(clamp(float(cfg.trail_distance_pct), 0.05, 3.00))

        # NEW (minimal): validate BE buffer
        try:
            cfg.partial_be_buffer_pct = float(cfg.partial_be_buffer_pct)
        except Exception:
            cfg.partial_be_buffer_pct = PARTIAL_BE_BUFFER_PCT_DEFAULT
        cfg.partial_be_buffer_pct = float(clamp(cfg.partial_be_buffer_pct, 0.00, 1.00))

        cfg.mock_use_bid_ask = bool(cfg.mock_use_bid_ask) if cfg.mock_use_bid_ask is not None else True
        cfg.mock_slippage_pct = float(clamp(float(cfg.mock_slippage_pct), 0.0, 0.50))

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
# KuCoin client
# -----------------------------
class KuCoinClient:
    def __init__(self):
        self.base = "https://api.kucoin.com"
        self.key = os.environ.get("KUCOIN_KEY", "").strip()
        self.secret = os.environ.get("KUCOIN_SECRET", "").strip()
        self.passphrase = os.environ.get("KUCOIN_PASSPHRASE", "").strip()

    def refresh_keys(self):
        self.key = os.environ.get("KUCOIN_KEY", "").strip()
        self.secret = os.environ.get("KUCOIN_SECRET", "").strip()
        self.passphrase = os.environ.get("KUCOIN_PASSPHRASE", "").strip()

    def has_keys(self) -> bool:
        return bool(self.key and self.secret and self.passphrase)

    def get_level1(self, symbol: str) -> Dict[str, float]:
        r = requests.get(
            f"{self.base}/api/v1/market/orderbook/level1",
            params={"symbol": symbol},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()["data"]
        last = float(data.get("price") or 0.0)
        bid = float(data.get("bestBid") or last or 0.0)
        ask = float(data.get("bestAsk") or last or 0.0)
        if bid <= 0:
            bid = last
        if ask <= 0:
            ask = last
        if last <= 0:
            last = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)
        return {"price": last, "bid": bid, "ask": ask}

    def get_klines(self, symbol: str, ktype: str, start_ts: int, end_ts: int) -> List[Dict[str, float]]:
        params = {"symbol": symbol, "type": ktype, "startAt": int(start_ts), "endAt": int(end_ts)}
        r = requests.get(f"{self.base}/api/v1/market/candles", params=params, timeout=12)
        r.raise_for_status()
        raw = r.json().get("data", [])
        out = []
        for row in raw:
            t = int(row[0])
            o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
            v = float(row[5])
            out.append({"t": t, "o": o, "c": c, "h": h, "l": l, "v": v})
        out.sort(key=lambda x: x["t"])
        return out

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        now = str(ts_ms())
        str_to_sign = now + method.upper() + path + (body or "")
        signature = base64.b64encode(
            hmac.new(self.secret.encode(), str_to_sign.encode(), hashlib.sha256).digest()
        ).decode()

        passphrase = base64.b64encode(
            hmac.new(self.secret.encode(), self.passphrase.encode(), hashlib.sha256).digest()
        ).decode()

        return {
            "KC-API-KEY": self.key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": now,
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    def place_market_buy_funds(self, symbol: str, funds_usdt: float) -> str:
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
        r = requests.post(self.base + path, data=body, headers=headers, timeout=12)
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
        r = requests.post(self.base + path, data=body, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        return data["data"]["orderId"]

KC = KuCoinClient()

# -----------------------------
# Strategy state
# -----------------------------
@dataclass
class Position:
    symbol: str
    entry_price: float
    qty_total: float
    qty_open: float
    entry_ts: int
    stake_usdt: float

    tp_price: float
    sl_price: float

    partial_done: bool = False
    partial_price: float = 0.0
    runner_peak: float = 0.0
    runner_trailing_active: bool = False
    runner_trail_stop: float = 0.0

    live_buy_order_id: str = ""
    live_sell_order_id: str = ""

POS: Dict[str, Position] = {}
COOLDOWN_UNTIL: Dict[str, int] = {}
LAST_1M_T: Dict[str, int] = {}
TREND_CACHE_OK: Dict[str, bool] = {}
TREND_CACHE_T: Dict[str, int] = {}

DIP_ARMED: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# UI / Telegram keyboards
# -----------------------------
def main_menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/entry"), KeyboardButton("/threshold")],
        [KeyboardButton("/partial"), KeyboardButton("/stake")],
        [KeyboardButton("/tp"), KeyboardButton("/sl")],
        [KeyboardButton("/coins"), KeyboardButton("/trade_mode")],
        [KeyboardButton("/notify"), KeyboardButton("/export_csv")],
        [KeyboardButton("/close_all"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def threshold_buttons(current: float) -> InlineKeyboardMarkup:
    options = [0.15, 0.20, 0.30, 0.40, 0.50, 0.70]
    row = []
    for x in options:
        label = f"{x:.2f}%" + (" ✅" if abs(current - x) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"thr:{x:.2f}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="thr:close")]])

def entry_buttons(current: str) -> InlineKeyboardMarkup:
    row = []
    for m in ["break", "retest"]:
        label = m.upper() + (" ✅" if current == m else "")
        row.append(InlineKeyboardButton(label, callback_data=f"em:{m}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="em:close")]])

def stake_buttons(current: float) -> InlineKeyboardMarkup:
    options = [10, 20, 30, 50, 100]
    row = []
    for x in options:
        label = f"{x} USDT" + (" ✅" if abs(current - float(x)) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"stk:{x}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="stk:close")]])

def tp_buttons(current: float) -> InlineKeyboardMarkup:
    options = [0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
    row = []
    for x in options:
        label = f"{x:.2f}%" + (" ✅" if abs(current - x) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"tp:{x:.2f}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="tp:close")]])

def sl_buttons(current: float) -> InlineKeyboardMarkup:
    options = [0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
    row = []
    for x in options:
        label = f"{x:.2f}%" + (" ✅" if abs(current - x) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"sl:{x:.2f}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="sl:close")]])

def coins_buttons(current_list: List[str]) -> InlineKeyboardMarkup:
    n_cur = len(current_list or [])
    row = [
        InlineKeyboardButton("2 coins (BTC,ETH)" + (" ✅" if n_cur == 2 else ""), callback_data="coinsn:2"),
        InlineKeyboardButton("3 coins (+XRP)" + (" ✅" if n_cur == 3 else ""), callback_data="coinsn:3"),
        InlineKeyboardButton("5 coins (default)" + (" ✅" if n_cur == 5 else ""), callback_data="coinsn:5"),
    ]
    rows = [row]
    rows.append([InlineKeyboardButton("Visa nuvarande", callback_data="coins:show")])
    rows.append([InlineKeyboardButton("Stäng", callback_data="coins:close")])
    return InlineKeyboardMarkup(rows)

def trade_mode_buttons(current: str) -> InlineKeyboardMarkup:
    row = []
    for m in ["mock", "live"]:
        label = m + (" ✅" if m == current else "")
        row.append(InlineKeyboardButton(label, callback_data=f"tm:{m}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="tm:close")]])

def partial_buttons() -> InlineKeyboardMarkup:
    en_label = "PARTIAL: ON ✅" if CFG.partial_enabled else "PARTIAL: OFF ✅"
    en_btn = InlineKeyboardButton(en_label, callback_data="pt:toggle")

    frac_opts = [0.30, 0.50, 0.70]
    frac_row = []
    for f in frac_opts:
        label = f"{int(f*100)}%" + (" ✅" if abs(CFG.partial_fraction - f) < 1e-9 else "")
        frac_row.append(InlineKeyboardButton(label, callback_data=f"pt:frac:{f:.2f}"))

    tp_opts = [0.30, 0.40, 0.50, 0.60]
    tp_row = []
    for x in tp_opts:
        label = f"{x:.2f}%" + (" ✅" if abs(CFG.partial_tp_pct - x) < 1e-9 else "")
        tp_row.append(InlineKeyboardButton(label, callback_data=f"pt:tp:{x:.2f}"))

    trail_opts = [0.25, 0.30, 0.35]
    trail_row = []
    for x in trail_opts:
        label = f"trail {x:.2f}%" + (" ✅" if abs(CFG.trail_distance_pct - x) < 1e-9 else "")
        trail_row.append(InlineKeyboardButton(label, callback_data=f"pt:trail:{x:.2f}"))

    return InlineKeyboardMarkup([
        [en_btn],
        frac_row,
        tp_row,
        trail_row,
        [InlineKeyboardButton("Stäng", callback_data="pt:close")]
    ])

# -----------------------------
# Notifications + Logging
# -----------------------------
async def notify_trade(app: Application, text: str):
    try:
        if not CFG.notify_trades or not CFG.notify_chat_id:
            return
        await app.bot.send_message(chat_id=CFG.notify_chat_id, text=text)
    except Exception:
        pass

def calc_fees(entry_price: float, exit_price: float, qty: float, fee_rate: float) -> float:
    return (entry_price * qty + exit_price * qty) * fee_rate

def log_trade(
    is_mock: bool,
    exchange: str,
    symbol: str,
    side: str,
    qty: float,
    stake_usdt: float,
    entry_price: float,
    exit_price: float,
    gross_pnl: float,
    fee_paid: float,
    net_pnl: float,
    reason: str,
    buy_order_id: str = "",
    sell_order_id: str = "",
):
    path = MOCK_LOG_PATH if is_mock else REAL_LOG_PATH
    ensure_csv_header(
        path,
        [
            "timestamp_unix",
            "timestamp_iso_utc",
            "exchange",
            "trade_mode",
            "symbol",
            "side",
            "qty",
            "stake_usdt",
            "entry_price",
            "exit_price",
            "gross_pnl_usdt",
            "fee_paid_usdt",
            "net_pnl_usdt",
            "reason",
            "buy_order_id",
            "sell_order_id",
        ],
    )
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                int(time.time()),
                iso_now(),
                exchange,
                "mock" if is_mock else "live",
                symbol,
                side,
                f"{qty:.12f}",
                f"{stake_usdt:.4f}",
                f"{entry_price:.12f}",
                f"{exit_price:.12f}",
                f"{gross_pnl:.8f}",
                f"{fee_paid:.8f}",
                f"{net_pnl:.8f}",
                reason,
                buy_order_id,
                sell_order_id,
            ]
        )

# -----------------------------
# Pricing helpers (mock realism)
# -----------------------------
def mock_entry_price(level1: Dict[str, float]) -> float:
    slip = CFG.mock_slippage_pct / 100.0
    if CFG.mock_use_bid_ask:
        p = float(level1.get("ask", level1.get("price", 0.0)))
    else:
        p = float(level1.get("price", 0.0))
    if p <= 0:
        p = float(level1.get("price", 0.0)) or 0.0
    return p * (1.0 + slip)

def mock_exit_price(level1: Dict[str, float]) -> float:
    slip = CFG.mock_slippage_pct / 100.0
    if CFG.mock_use_bid_ask:
        p = float(level1.get("bid", level1.get("price", 0.0)))
    else:
        p = float(level1.get("price", 0.0))
    if p <= 0:
        p = float(level1.get("price", 0.0)) or 0.0
    return p * (1.0 - slip)

# -----------------------------
# Strategy core
# -----------------------------
def in_cooldown(symbol: str) -> bool:
    return int(time.time()) < int(COOLDOWN_UNTIL.get(symbol, 0))

def can_open_new_position() -> bool:
    return len(POS) < int(CFG.max_positions)

async def open_long(app: Application, symbol: str, reason: str):
    if symbol in POS or in_cooldown(symbol) or not can_open_new_position():
        return

    funds = float(CFG.stake_usdt)
    lvl1 = await asyncio.to_thread(KC.get_level1, symbol)

    buy_id = ""
    if CFG.trade_mode == "live":
        KC.refresh_keys()
        if not KC.has_keys():
            CFG.trade_mode = "mock"
            save_config()
            await notify_trade(app, "⚠️ KUCOIN keys saknas. Växlar till mock.")
        else:
            try:
                buy_id = await asyncio.to_thread(KC.place_market_buy_funds, symbol, funds)
            except Exception as e:
                CFG.trade_mode = "mock"
                save_config()
                await notify_trade(app, f"⚠️ Live BUY misslyckades, fallback mock. ({type(e).__name__})")
                buy_id = ""

    entry_price = mock_entry_price(lvl1) if CFG.trade_mode == "mock" else (
        float(lvl1.get("price", 0.0)) or float(lvl1.get("ask", 0.0)) or float(lvl1.get("bid", 0.0))
    )

    qty = funds / entry_price if entry_price > 0 else 0.0
    tp_price = entry_price * (1.0 + CFG.tp_pct / 100.0)
    sl_price = entry_price * (1.0 - CFG.sl_pct / 100.0)

    POS[symbol] = Position(
        symbol=symbol,
        entry_price=entry_price,
        qty_total=qty,
        qty_open=qty,
        entry_ts=int(time.time()),
        stake_usdt=funds,
        tp_price=tp_price,
        sl_price=sl_price,
        partial_done=False,
        partial_price=entry_price * (1.0 + CFG.partial_tp_pct / 100.0),
        runner_peak=entry_price,
        runner_trailing_active=False,
        runner_trail_stop=0.0,
        live_buy_order_id=buy_id,
        live_sell_order_id="",
    )

    DIP_ARMED.pop(symbol, None)

    tag = "LIVE" if (CFG.trade_mode == "live" and buy_id) else "MOCK"
    extra = f" | mock(bid/ask={CFG.mock_use_bid_ask}, slip={CFG.mock_slippage_pct:.2f}%)" if CFG.trade_mode == "mock" else ""
    await notify_trade(
        app,
        f"🟢 ENTRY ({tag}) {symbol} @ {entry_price:.6f} | TP={tp_price:.6f} SL={sl_price:.6f} "
        f"| dip={CFG.threshold_pct:.2f}% | entry={CFG.entry_mode.upper()} | {reason}{extra}"
    )

async def _close_qty(app: Application, symbol: str, qty_to_close: float, reason: str, count_trade_if_flat: bool):
    if symbol not in POS:
        return
    p = POS[symbol]
    qty_to_close = min(qty_to_close, p.qty_open)
    if qty_to_close <= 0:
        return

    lvl1 = await asyncio.to_thread(KC.get_level1, symbol)

    sell_id = ""
    if CFG.trade_mode == "live":
        KC.refresh_keys()
        if KC.has_keys():
            try:
                sell_id = await asyncio.to_thread(KC.place_market_sell_size, symbol, qty_to_close)
            except Exception:
                await notify_trade(app, f"⚠️ Live SELL misslyckades för {symbol}.")

    exit_price = mock_exit_price(lvl1) if CFG.trade_mode == "mock" else (
        float(lvl1.get("price", 0.0)) or float(lvl1.get("bid", 0.0)) or float(lvl1.get("ask", 0.0))
    )

    gross = (exit_price - p.entry_price) * qty_to_close
    fee = calc_fees(p.entry_price, exit_price, qty_to_close, CFG.fee_rate)
    net = gross - fee

    log_trade(
        is_mock=(CFG.trade_mode == "mock"),
        exchange="KuCoin",
        symbol=symbol,
        side="LONG",
        qty=qty_to_close,
        stake_usdt=p.stake_usdt,
        entry_price=p.entry_price,
        exit_price=exit_price,
        gross_pnl=gross,
        fee_paid=fee,
        net_pnl=net,
        reason=reason,
        buy_order_id=p.live_buy_order_id,
        sell_order_id=sell_id,
    )

    CFG.total_net_pnl += net

    p.qty_open -= qty_to_close
    if p.qty_open <= 1e-12:
        POS.pop(symbol, None)
        COOLDOWN_UNTIL[symbol] = int(time.time()) + int(CFG.cooldown_seconds)
        if count_trade_if_flat:
            CFG.trades += 1
        save_config()

    emoji = "🎯" if net >= 0 else "🟥"
    extra = f" | mock(bid/ask={CFG.mock_use_bid_ask}, slip={CFG.mock_slippage_pct:.2f}%)" if CFG.trade_mode == "mock" else ""
    await notify_trade(app, f"{emoji} EXIT {symbol} qty={qty_to_close:.6f} @ {exit_price:.6f} | Net {net:+.4f} USDT | {reason}{extra}")

async def close_all(app: Application, reason: str = "CLOSE_ALL"):
    for sym in list(POS.keys()):
        p = POS.get(sym)
        if not p:
            continue
        await _close_qty(app, sym, p.qty_open, reason=reason, count_trade_if_flat=True)

async def manage_position(app: Application, symbol: str):
    if symbol not in POS:
        return
    p = POS[symbol]

    lvl1 = await asyncio.to_thread(KC.get_level1, symbol)
    px = mock_exit_price(lvl1) if CFG.trade_mode == "mock" else (
        float(lvl1.get("price", 0.0)) or float(lvl1.get("bid", 0.0)) or float(lvl1.get("ask", 0.0))
    )

    # Hard SL for entire remaining position
    if px <= p.sl_price:
        await _close_qty(app, symbol, p.qty_open, reason="SL", count_trade_if_flat=True)
        return

    # Full TP if partial disabled
    if px >= p.tp_price and (not CFG.partial_enabled):
        await _close_qty(app, symbol, p.qty_open, reason="TP", count_trade_if_flat=True)
        return

    # Partial + runner trail
    if CFG.partial_enabled and (not p.partial_done):
        if px >= p.partial_price:
            qty_part = p.qty_total * CFG.partial_fraction
            qty_part = min(qty_part, p.qty_open)
            if qty_part > 0:
                p.partial_done = True

                # Close partial
                await _close_qty(app, symbol, qty_part, reason=f"PARTIAL_TP(+{CFG.partial_tp_pct:.2f}%)", count_trade_if_flat=False)

                # If still open after partial, apply breakeven stop to remaining qty (MINIMAL CHANGE)
                if symbol in POS:
                    p2 = POS[symbol]
                    be = p2.entry_price * (1.0 + (CFG.partial_be_buffer_pct / 100.0))
                    if be > p2.sl_price:
                        p2.sl_price = be
                        await notify_trade(app, f"🛡️ {symbol} SL flyttad till BE+{CFG.partial_be_buffer_pct:.2f}% = {p2.sl_price:.6f} (efter partial)")

                    # init runner
                    p2.runner_peak = px
                    p2.runner_trailing_active = True
                    p2.runner_trail_stop = px * (1.0 - CFG.trail_distance_pct / 100.0)
            return

    # Runner management (after partial)
    if CFG.partial_enabled and p.partial_done and p.qty_open > 0:
        if px > p.runner_peak:
            p.runner_peak = px

        if not p.runner_trailing_active:
            if px >= p.entry_price * (1.0 + CFG.trail_activate_pct / 100.0):
                p.runner_trailing_active = True
                p.runner_trail_stop = p.runner_peak * (1.0 - CFG.trail_distance_pct / 100.0)
        else:
            new_stop = p.runner_peak * (1.0 - CFG.trail_distance_pct / 100.0)
            if new_stop > p.runner_trail_stop:
                p.runner_trail_stop = new_stop
            if px <= p.runner_trail_stop:
                await _close_qty(app, symbol, p.qty_open, reason="RUNNER_TRAIL", count_trade_if_flat=True)
                return

# -----------------------------
# Trend filter (cached per 5m candle)
# -----------------------------
async def trend_ok_5m(symbol: str) -> bool:
    now = int(time.time())
    last_t = TREND_CACHE_T.get(symbol, 0)
    if now - last_t < 60:
        return TREND_CACHE_OK.get(symbol, False)

    end = int(time.time())
    start = end - 60 * 60 * 40
    candles = await asyncio.to_thread(KC.get_klines, symbol, TF_TREND, start, end)
    if len(candles) < (EMA_SLOW + 20):
        TREND_CACHE_OK[symbol] = False
        TREND_CACHE_T[symbol] = now
        return False

    closes = [c["c"] for c in candles]
    ema50 = ema_last(closes, EMA_FAST)
    ema200 = ema_last(closes, EMA_SLOW)
    if ema50 is None or ema200 is None:
        TREND_CACHE_OK[symbol] = False
        TREND_CACHE_T[symbol] = now
        return False

    ema50_prev = ema_last(closes[:-3], EMA_FAST) if len(closes) > (EMA_FAST + 10) else None
    ok = (ema50 > ema200) and (ema50_prev is not None and ema50 >= ema50_prev)

    TREND_CACHE_OK[symbol] = bool(ok)
    TREND_CACHE_T[symbol] = now
    return bool(ok)

# -----------------------------
# Entry logic (VWAP dip + confirm)
# -----------------------------
async def evaluate_symbol(app: Application, symbol: str):
    if symbol in POS:
        await manage_position(app, symbol)
        return

    if in_cooldown(symbol) or not can_open_new_position():
        return

    if not await trend_ok_5m(symbol):
        DIP_ARMED.pop(symbol, None)
        return

    end = int(time.time())
    start = end - 60 * (VWAP_LOOKBACK + 30)
    candles = await asyncio.to_thread(KC.get_klines, symbol, TF_SIGNAL, start, end)
    if len(candles) < (VWAP_LOOKBACK + 10):
        return

    last_t = candles[-1]["t"]
    if LAST_1M_T.get(symbol) == last_t:
        return
    LAST_1M_T[symbol] = last_t

    look = candles[-VWAP_LOOKBACK:]
    vwap = rolling_vwap(look)
    if vwap is None:
        return

    cur = candles[-1]
    prev = candles[-2]

    dip_band = CFG.threshold_pct / 100.0
    dip_level = vwap * (1.0 - dip_band)

    if cur["l"] <= dip_level:
        DIP_ARMED[symbol] = {"armed_ts": int(time.time())}

    if symbol not in DIP_ARMED:
        return

    is_green = cur["c"] > cur["o"]
    if not is_green:
        return

    if CFG.entry_mode == "break":
        if cur["c"] >= dip_level:
            await open_long(app, symbol, reason=f"DIP->GREEN rebound (low<=VWAP-{CFG.threshold_pct:.2f}%)")
    else:
        if cur["c"] >= vwap and prev["c"] < vwap:
            await open_long(app, symbol, reason=f"DIP->RECLAIM VWAP (retest confirm)")

# -----------------------------
# Engine loop
# -----------------------------
async def engine_background_loop(app: Application):
    while True:
        try:
            if CFG.engine_on:
                for sym in list(CFG.active_coins or []):
                    try:
                        await evaluate_symbol(app, sym)
                    except Exception:
                        continue
        except Exception:
            pass
        await asyncio.sleep(POLL_SECONDS)

# -----------------------------
# Telegram commands
# -----------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CFG.notify_chat_id = int(update.effective_chat.id)
    CFG.notify_trades = True
    save_config()
    await update.message.reply_text(
        "Mp ORBbot är igång.\n"
        "Strategi: VWAP Dip Reversion (1m) + 5m Uptrend-filter (EMA50>EMA200).\n"
        "LONG only.\n"
        "• /threshold = dip % under VWAP\n"
        "• /entry = BREAK (snabb) eller RETEST (strikt)\n"
        "• /partial = partial + runner trail\n",
        reply_markup=main_menu_keyboard(),
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    KC.refresh_keys()
    lines = [
        f"ENGINE: {'ON' if CFG.engine_on else 'OFF'}",
        "Strategy: VWAP Dip Reversion (1m) + Trendfilter (5m EMA50>EMA200)",
        f"Entry mode: {CFG.entry_mode.upper()}",
        f"Trade mode: {CFG.trade_mode} (live keys: {'OK' if KC.has_keys() else 'missing'})",
        f"Dip band (/threshold): {CFG.threshold_pct:.2f}% under VWAP",
        f"Stake per trade: {CFG.stake_usdt:.2f} USDT",
        f"TP/SL: {CFG.tp_pct:.2f}% / {CFG.sl_pct:.2f}%",
        f"Partial: {'ON' if CFG.partial_enabled else 'OFF'} | partialTP {CFG.partial_tp_pct:.2f}% | fraction {int(CFG.partial_fraction*100)}%",
        f"BE buffer after partial: +{CFG.partial_be_buffer_pct:.2f}%",
        f"Runner trail: activate {CFG.trail_activate_pct:.2f}% | dist {CFG.trail_distance_pct:.2f}%",
        f"Cooldown: {CFG.cooldown_seconds}s",
        f"Max positions: {CFG.max_positions}",
        f"Trades (closed): {CFG.trades}",
        f"Total NET PnL: {CFG.total_net_pnl:.4f} USDT",
        f"Coins ({len(CFG.active_coins)}): {CFG.active_coins}",
        f"Open positions: {list(POS.keys()) if POS else 'none'}",
        f"Dip armed: {list(DIP_ARMED.keys()) if DIP_ARMED else 'none'}",
        f"Notify: {'ON' if CFG.notify_trades else 'OFF'}",
        f"Mock spread (bid/ask): {'ON' if CFG.mock_use_bid_ask else 'OFF'}",
        f"Mock slippage: {CFG.mock_slippage_pct:.2f}%",
    ]
    await update.message.reply_text("\n".join(lines), reply_markup=main_menu_keyboard())

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Total NET PnL: {CFG.total_net_pnl:.4f} USDT\nTrades (closed): {CFG.trades}")

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CFG.engine_on = True
    CFG.notify_chat_id = int(update.effective_chat.id)
    CFG.notify_trades = True
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
    await close_all(context.application, reason="CLOSE_ALL")
    await update.message.reply_text("✅ Close all skickad.")

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

async def cmd_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        m = args[0].lower().strip()
        if m not in ENTRY_MODES:
            await update.message.reply_text("Ogiltigt. Använd: /entry break eller /entry retest")
            return
        CFG.entry_mode = m
        save_config()
        await update.message.reply_text(f"✅ Entry mode satt till: {CFG.entry_mode.upper()}")
        return
    await update.message.reply_text(
        f"Entry mode nu: {CFG.entry_mode.upper()}\nVälj:",
        reply_markup=entry_buttons(CFG.entry_mode),
    )

async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        v = parse_float(args[0])
        if v is None:
            await update.message.reply_text("Ogiltigt värde. Ex: /threshold 0.3")
            return
        CFG.threshold_pct = float(round(clamp(v, 0.05, 2.0), 4))
        save_config()
        await update.message.reply_text(f"✅ Dip-band satt till: {CFG.threshold_pct:.2f}% under VWAP")
        return
    await update.message.reply_text(
        f"Dip-band nu: {CFG.threshold_pct:.2f}% under VWAP\nVälj:",
        reply_markup=threshold_buttons(CFG.threshold_pct),
    )

async def cmd_partial(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Partial nu: {'ON' if CFG.partial_enabled else 'OFF'}\n"
        f"Partial TP: {CFG.partial_tp_pct:.2f}% | Fraction: {int(CFG.partial_fraction*100)}%\n"
        f"Runner trail dist: {CFG.trail_distance_pct:.2f}%\n"
        f"BE buffer after partial: +{CFG.partial_be_buffer_pct:.2f}%",
        reply_markup=partial_buttons(),
    )

async def cmd_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        v = parse_float(args[0])
        if v is None:
            await update.message.reply_text("Ogiltigt värde. Ex: /stake 30")
            return
        CFG.stake_usdt = float(round(clamp(v, 5.0, 500.0), 4))
        save_config()
        await update.message.reply_text(f"✅ Stake per trade satt till: {CFG.stake_usdt:.2f} USDT")
        return
    await update.message.reply_text(
        f"Stake per trade nu: {CFG.stake_usdt:.2f} USDT\nVälj:",
        reply_markup=stake_buttons(CFG.stake_usdt),
    )

async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        v = parse_float(args[0])
        if v is None:
            await update.message.reply_text("Ogiltigt. Ex: /tp 0.7")
            return
        CFG.tp_pct = float(round(clamp(v, 0.20, 2.00), 4))
        save_config()
        await update.message.reply_text(f"✅ TP satt till: {CFG.tp_pct:.2f}%")
        return
    await update.message.reply_text(
        f"TP nu: {CFG.tp_pct:.2f}%\nVälj:",
        reply_markup=tp_buttons(CFG.tp_pct),
    )

async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        v = parse_float(args[0])
        if v is None:
            await update.message.reply_text("Ogiltigt. Ex: /sl 0.5")
            return
        CFG.sl_pct = float(round(clamp(v, 0.10, 3.00), 4))
        save_config()
        await update.message.reply_text(f"✅ SL satt till: {CFG.sl_pct:.2f}%")
        return
    await update.message.reply_text(
        f"SL nu: {CFG.sl_pct:.2f}%\nVälj:",
        reply_markup=sl_buttons(CFG.sl_pct),
    )

async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        n = None
        try:
            n = int(args[0])
        except Exception:
            n = None

        if n is not None:
            n = int(clamp(n, 1, 20))
            CFG.active_coins = DEFAULT_COINS[:n]
            save_config()
            await update.message.reply_text(f"✅ Aktiva coins satt till första {n}: {CFG.active_coins}")
            return

        coins = [safe_symbol(a) for a in args]
        uniq = []
        for c in coins:
            if c.endswith("-USDT") and c not in uniq:
                uniq.append(c)
        if not uniq:
            await update.message.reply_text("Ogiltig lista. Ex: /coins BTC-USDT ETH-USDT XRP-USDT")
            return
        CFG.active_coins = uniq
        save_config()
        await update.message.reply_text(f"✅ Aktiva coins uppdaterade ({len(uniq)}): {uniq}")
        return

    await update.message.reply_text(
        f"Aktiva coins ({len(CFG.active_coins)}): {CFG.active_coins}\nVälj:",
        reply_markup=coins_buttons(CFG.active_coins),
    )

async def cmd_trade_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        m = args[0].lower().strip()
        if m not in ("mock", "live"):
            await update.message.reply_text("Ogiltigt. Använd: /trade_mode mock eller /trade_mode live")
            return
        if m == "live":
            KC.refresh_keys()
            if not KC.has_keys():
                await update.message.reply_text("⚠️ KUCOIN_KEY/SECRET/PASSPHRASE saknas. Sätt dem först, annars kör mock.")
                return
        CFG.trade_mode = m
        save_config()
        await update.message.reply_text(f"✅ Trade mode: {CFG.trade_mode}")
        return
    await update.message.reply_text(
        f"Trade mode nu: {CFG.trade_mode}\nVälj:",
        reply_markup=trade_mode_buttons(CFG.trade_mode),
    )

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_csv_header(MOCK_LOG_PATH, ["timestamp_unix"])
    ensure_csv_header(REAL_LOG_PATH, ["timestamp_unix"])

    await update.message.reply_text(
        "📄 Exporterar CSV-loggar.\n"
        "• mock_trade_log.csv = paper trades\n"
        "• real_trade_log.csv = live trades\n"
        "Partial exits loggas också (PARTIAL_TP / RUNNER_TRAIL).\n"
    )

    try:
        if os.path.exists(MOCK_LOG_PATH) and os.path.getsize(MOCK_LOG_PATH) > 0:
            await context.bot.send_document(chat_id=update.effective_chat.id, document=open(MOCK_LOG_PATH, "rb"))
    except Exception:
        await update.message.reply_text("⚠️ Kunde inte skicka mock_trade_log.csv")

    try:
        if os.path.exists(REAL_LOG_PATH) and os.path.getsize(REAL_LOG_PATH) > 0:
            await context.bot.send_document(chat_id=update.effective_chat.id, document=open(REAL_LOG_PATH, "rb"))
    except Exception:
        await update.message.reply_text("⚠️ Kunde inte skicka real_trade_log.csv")

# -----------------------------
# Callbacks
# -----------------------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = (q.data or "").strip()
    await q.answer()

    if data.startswith("em:"):
        payload = data.split("em:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        if payload not in ENTRY_MODES:
            await q.edit_message_text("Ogiltigt val.")
            return
        CFG.entry_mode = payload
        save_config()
        await q.edit_message_text(f"✅ Entry mode satt till: {CFG.entry_mode.upper()}")
        return

    if data.startswith("thr:"):
        payload = data.split("thr:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        v = parse_float(payload)
        if v is None:
            await q.edit_message_text("Ogiltigt värde.")
            return
        CFG.threshold_pct = float(round(clamp(v, 0.05, 2.0), 4))
        save_config()
        await q.edit_message_text(f"✅ Dip-band satt till: {CFG.threshold_pct:.2f}% under VWAP")
        return

    if data.startswith("stk:"):
        payload = data.split("stk:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        v = parse_float(payload)
        if v is None:
            await q.edit_message_text("Ogiltigt stake.")
            return
        CFG.stake_usdt = float(round(clamp(v, 5.0, 500.0), 4))
        save_config()
        await q.edit_message_text(f"✅ Stake per trade satt till: {CFG.stake_usdt:.2f} USDT")
        return

    if data.startswith("tp:"):
        payload = data.split("tp:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        v = parse_float(payload)
        if v is None:
            await q.edit_message_text("Ogiltigt TP.")
            return
        CFG.tp_pct = float(round(clamp(v, 0.20, 2.00), 4))
        save_config()
        await q.edit_message_text(f"✅ TP satt till: {CFG.tp_pct:.2f}%")
        return

    if data.startswith("sl:"):
        payload = data.split("sl:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        v = parse_float(payload)
        if v is None:
            await q.edit_message_text("Ogiltigt SL.")
            return
        CFG.sl_pct = float(round(clamp(v, 0.10, 3.00), 4))
        save_config()
        await q.edit_message_text(f"✅ SL satt till: {CFG.sl_pct:.2f}%")
        return

    if data.startswith("coinsn:"):
        payload = data.split("coinsn:", 1)[1]
        try:
            n = int(payload)
        except Exception:
            await q.edit_message_text("Ogiltigt val.")
            return
        n = int(clamp(n, 1, 20))
        CFG.active_coins = DEFAULT_COINS[:n]
        save_config()
        await q.edit_message_text(f"✅ Aktiva coins satt till första {n}: {CFG.active_coins}")
        return

    if data == "coins:show":
        await q.edit_message_text(f"Aktiva coins ({len(CFG.active_coins)}): {CFG.active_coins}")
        return

    if data == "coins:close":
        await q.edit_message_text("✅ Stängt.")
        return

    if data.startswith("tm:"):
        payload = data.split("tm:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        if payload in ("mock", "live"):
            if payload == "live":
                KC.refresh_keys()
                if not KC.has_keys():
                    await q.edit_message_text("⚠️ KUCOIN keys saknas. Kan inte slå på live.")
                    return
            CFG.trade_mode = payload
            save_config()
            await q.edit_message_text(f"✅ Trade mode: {CFG.trade_mode}")
            return

    if data.startswith("pt:"):
        if data == "pt:close":
            await q.edit_message_text("✅ Stängt.")
            return
        if data == "pt:toggle":
            CFG.partial_enabled = not CFG.partial_enabled
            save_config()
            await q.edit_message_text(
                f"✅ Partial är nu: {'ON' if CFG.partial_enabled else 'OFF'}",
                reply_markup=partial_buttons()
            )
            return
        if data.startswith("pt:frac:"):
            v = parse_float(data.split("pt:frac:", 1)[1])
            if v is None:
                await q.edit_message_text("Ogiltigt.")
                return
            CFG.partial_fraction = float(clamp(v, 0.10, 0.90))
            save_config()
            await q.edit_message_text("✅ Uppdaterat.", reply_markup=partial_buttons())
            return
        if data.startswith("pt:tp:"):
            v = parse_float(data.split("pt:tp:", 1)[1])
            if v is None:
                await q.edit_message_text("Ogiltigt.")
                return
            CFG.partial_tp_pct = float(clamp(v, 0.10, 2.00))
            save_config()
            await q.edit_message_text("✅ Uppdaterat.", reply_markup=partial_buttons())
            return
        if data.startswith("pt:trail:"):
            v = parse_float(data.split("pt:trail:", 1)[1])
            if v is None:
                await q.edit_message_text("Ogiltigt.")
                return
            CFG.trail_distance_pct = float(clamp(v, 0.05, 3.00))
            save_config()
            await q.edit_message_text("✅ Uppdaterat.", reply_markup=partial_buttons())
            return

    await q.edit_message_text("Ok.")

# -----------------------------
# Boot
# -----------------------------
def build_app() -> Application:
    token = (os.environ.get("TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Sätt TELEGRAM_TOKEN (eller TELEGRAM_BOT_TOKEN) i environment innan du startar boten.")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))

    app.add_handler(CommandHandler("notify", cmd_notify))
    app.add_handler(CommandHandler("entry", cmd_entry))
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("partial", cmd_partial))

    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("sl", cmd_sl))
    app.add_handler(CommandHandler("coins", cmd_coins))
    app.add_handler(CommandHandler("trade_mode", cmd_trade_mode))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))

    app.add_handler(CallbackQueryHandler(on_callback))
    return app

async def post_init(app: Application):
    app.create_task(engine_background_loop(app))

def main():
    global CFG
    CFG = load_config()
    if CFG.active_coins is None:
        CFG.active_coins = DEFAULT_COINS.copy()
    save_config()

    ensure_csv_header(MOCK_LOG_PATH, ["timestamp_unix"])
    ensure_csv_header(REAL_LOG_PATH, ["timestamp_unix"])

    app = build_app()
    app.post_init = post_init
    print("Mp ORBbot running (VWAP Dip Reversion 1m + Partial Runner + BE after partial)...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
