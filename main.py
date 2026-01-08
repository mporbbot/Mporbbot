# main.py
# Mp ORBbot - Strategy: Inventory-first Spread Market Maker (KuCoin, LONG-only inventory)
#
# NEW (minimal change): Telegram-selectable exit mode:
#   /exit  -> LIMIT / HYBRID / MARKET
#
# Exit modes:
#   LIMIT  = limit buy @ bid, limit sell @ ask (postOnly)
#   HYBRID = limit buy @ bid, after fill: try limit sell @ ask briefly, else market sell
#   MARKET = limit buy @ bid, after fill: market sell immediately (debug/stress)
#
# Controls:
#   /threshold = MIN SPREAD % required to quote (e.g. 0.35)
#   /tp        = PROFIT TARGET % (e.g. 0.15)
#   /sl        = PANIC STOP % (max loss from entry before market exit, e.g. 0.60)
#   /stake     = stake USDT per cycle (change anytime)
#   /coins     = active symbols
#   /trade_mode = mock|live
#
# Token NOT hardcoded. Uses TELEGRAM_TOKEN or TELEGRAM_BOT_TOKEN env var.
# KuCoin keys: KUCOIN_KEY KUCOIN_SECRET KUCOIN_PASSPHRASE (for live)
#
# pip install python-telegram-bot==20.7 requests

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

POLL_SECONDS = 3

DEFAULT_FEE_RATE = 0.001  # 0.1% per side (conservative)

# Strategy defaults
MIN_SPREAD_PCT_DEFAULT = 0.35     # /threshold
PROFIT_TARGET_PCT_DEFAULT = 0.15  # /tp
PANIC_STOP_PCT_DEFAULT = 0.60     # /sl

STAKE_USDT_DEFAULT = 30.0
MAX_POSITIONS_DEFAULT = 10
COOLDOWN_SECONDS_DEFAULT = 2

INVENTORY_CAP_USDT_DEFAULT = 60.0  # per coin cap
ORDER_TIMEOUT_SEC_DEFAULT = 20
REQUOTE_SEC_DEFAULT = 5

POST_ONLY = True

# Exit modes
EXIT_MODES = ["limit", "hybrid", "market"]
EXIT_MODE_DEFAULT = "limit"  # keep previous behavior by default
HYBRID_EXIT_WAIT_SEC_DEFAULT = 8   # wait this long for limit sell, then market sell

# -----------------------------
# Helpers
# -----------------------------
def ts_ms() -> int:
    return int(time.time() * 1000)

def iso_now() -> str:
    return datetime.now(tz=UTC).isoformat()

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def parse_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ".").strip())
    except Exception:
        return None

def safe_symbol(sym: str) -> str:
    s = sym.upper().replace("_", "-").replace("/", "-")
    if "-" not in s and s.endswith("USDT"):
        s = s[:-4] + "-USDT"
    return s

def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

# -----------------------------
# Config
# -----------------------------
@dataclass
class BotConfig:
    engine_on: bool = False
    trade_mode: str = "mock"   # mock | live

    # Strategy knobs (re-using existing commands)
    threshold_pct: float = MIN_SPREAD_PCT_DEFAULT   # min spread %
    tp_pct: float = PROFIT_TARGET_PCT_DEFAULT       # profit target %
    sl_pct: float = PANIC_STOP_PCT_DEFAULT          # panic stop %

    stake_usdt: float = STAKE_USDT_DEFAULT
    fee_rate: float = DEFAULT_FEE_RATE
    active_coins: List[str] = None

    max_positions: int = MAX_POSITIONS_DEFAULT
    cooldown_seconds: int = COOLDOWN_SECONDS_DEFAULT

    # Cap/timeout (config.json)
    inventory_cap_usdt: float = INVENTORY_CAP_USDT_DEFAULT
    order_timeout_sec: int = ORDER_TIMEOUT_SEC_DEFAULT
    requote_sec: int = REQUOTE_SEC_DEFAULT

    # NEW: exit mode (telegram)
    exit_mode: str = EXIT_MODE_DEFAULT
    hybrid_exit_wait_sec: int = HYBRID_EXIT_WAIT_SEC_DEFAULT

    # Stats
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

        cfg.threshold_pct = float(clamp(float(cfg.threshold_pct), 0.05, 5.0))
        cfg.tp_pct = float(clamp(float(cfg.tp_pct), 0.02, 2.0))
        cfg.sl_pct = float(clamp(float(cfg.sl_pct), 0.10, 5.0))
        cfg.stake_usdt = float(clamp(float(cfg.stake_usdt), 5.0, 500.0))

        cfg.max_positions = int(clamp(int(cfg.max_positions), 1, 50))
        cfg.cooldown_seconds = int(clamp(int(cfg.cooldown_seconds), 0, 60))

        cfg.inventory_cap_usdt = float(clamp(float(cfg.inventory_cap_usdt), 10.0, 2000.0))
        cfg.order_timeout_sec = int(clamp(int(cfg.order_timeout_sec), 3, 300))
        cfg.requote_sec = int(clamp(int(cfg.requote_sec), 1, 60))

        # NEW
        cfg.exit_mode = (cfg.exit_mode or EXIT_MODE_DEFAULT).lower().strip()
        if cfg.exit_mode not in EXIT_MODES:
            cfg.exit_mode = EXIT_MODE_DEFAULT
        cfg.hybrid_exit_wait_sec = int(clamp(int(cfg.hybrid_exit_wait_sec), 1, 60))

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

    def place_limit(self, symbol: str, side: str, price: float, size: float, post_only: bool = True) -> str:
        path = "/api/v1/orders"
        payload = {
            "clientOid": str(ts_ms()),
            "side": side,
            "symbol": symbol,
            "type": "limit",
            "price": f"{price:.10f}",
            "size": f"{size:.10f}",
        }
        if post_only:
            payload["postOnly"] = True

        body = json.dumps(payload)
        headers = self._headers("POST", path, body)
        r = requests.post(self.base + path, data=body, headers=headers, timeout=12)
        r.raise_for_status()
        return r.json()["data"]["orderId"]

    def cancel_order(self, order_id: str) -> None:
        path = f"/api/v1/orders/{order_id}"
        headers = self._headers("DELETE", path, "")
        r = requests.delete(self.base + path, headers=headers, timeout=12)
        if r.status_code >= 400:
            try:
                _ = r.json()
            except Exception:
                pass

    def get_order(self, order_id: str) -> Dict[str, Any]:
        path = f"/api/v1/orders/{order_id}"
        headers = self._headers("GET", path, "")
        r = requests.get(self.base + path, headers=headers, timeout=12)
        r.raise_for_status()
        return r.json()["data"]

KC = KuCoinClient()

# -----------------------------
# Logging
# -----------------------------
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
# Telegram UI
# -----------------------------
def main_menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/threshold"), KeyboardButton("/stake")],
        [KeyboardButton("/tp"), KeyboardButton("/sl")],
        [KeyboardButton("/exit"), KeyboardButton("/coins")],
        [KeyboardButton("/trade_mode"), KeyboardButton("/notify")],
        [KeyboardButton("/export_csv"), KeyboardButton("/close_all")],
        [KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def threshold_buttons(current: float) -> InlineKeyboardMarkup:
    options = [0.20, 0.25, 0.35, 0.50, 0.70, 1.00]
    row = []
    for x in options:
        label = f"{x:.2f}%" + (" ✅" if abs(current - x) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"thr:{x:.2f}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="thr:close")]])

def stake_buttons(current: float) -> InlineKeyboardMarkup:
    options = [10, 20, 30, 50, 100]
    row = []
    for x in options:
        label = f"{x} USDT" + (" ✅" if abs(current - float(x)) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"stk:{x}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="stk:close")]])

def tp_buttons(current: float) -> InlineKeyboardMarkup:
    options = [0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
    row = []
    for x in options:
        label = f"{x:.2f}%" + (" ✅" if abs(current - x) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"tp:{x:.2f}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="tp:close")]])

def sl_buttons(current: float) -> InlineKeyboardMarkup:
    options = [0.40, 0.60, 0.80, 1.00, 1.50, 2.00]
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

def exit_mode_buttons(current: str) -> InlineKeyboardMarkup:
    # buttons for LIMIT / HYBRID / MARKET
    row = []
    for m in EXIT_MODES:
        label = m.upper() + (" ✅" if m == current else "")
        row.append(InlineKeyboardButton(label, callback_data=f"ex:{m}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="ex:close")]])

# -----------------------------
# Notifications
# -----------------------------
async def notify(app: Application, text: str):
    try:
        if not CFG.notify_trades or not CFG.notify_chat_id:
            return
        await app.bot.send_message(chat_id=CFG.notify_chat_id, text=text)
    except Exception:
        pass

# -----------------------------
# Strategy state
# -----------------------------
@dataclass
class MMState:
    symbol: str

    buy_order_id: str = ""
    buy_ts: int = 0
    buy_price: float = 0.0
    buy_size: float = 0.0

    sell_order_id: str = ""
    sell_ts: int = 0
    sell_price: float = 0.0
    sell_size: float = 0.0

    inv_qty: float = 0.0
    inv_entry_price: float = 0.0
    inv_entry_ts: int = 0

    cooldown_until: int = 0

MM: Dict[str, MMState] = {}

def now_s() -> int:
    return int(time.time())

def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 0.0
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0
    return ((ask - bid) / mid) * 100.0

def inventory_value_usdt(lvl: Dict[str, float], st: MMState) -> float:
    bid = float(lvl.get("bid", 0.0))
    return max(0.0, st.inv_qty) * bid

# -----------------------------
# Live market sell (panic/hybrid fallback)
# -----------------------------
def kucoin_market_sell(symbol: str, size: float, client: KuCoinClient) -> str:
    path = "/api/v1/orders"
    payload = {
        "clientOid": str(ts_ms()),
        "side": "sell",
        "symbol": symbol,
        "type": "market",
        "size": str(size),
    }
    body = json.dumps(payload)
    headers = client._headers("POST", path, body)
    r = requests.post(client.base + path, data=body, headers=headers, timeout=12)
    r.raise_for_status()
    return r.json()["data"]["orderId"]

# -----------------------------
# Core logic
# -----------------------------
async def cancel_if_stale(app: Application, st: MMState):
    if st.buy_order_id and (now_s() - st.buy_ts) >= CFG.order_timeout_sec:
        oid = st.buy_order_id
        st.buy_order_id = ""
        await asyncio.to_thread(KC.cancel_order, oid)
        await notify(app, f"⏱️ {st.symbol} cancel BUY (timeout) {oid}")

    if st.sell_order_id and (now_s() - st.sell_ts) >= CFG.order_timeout_sec:
        oid = st.sell_order_id
        st.sell_order_id = ""
        await asyncio.to_thread(KC.cancel_order, oid)
        await notify(app, f"⏱️ {st.symbol} cancel SELL (timeout) {oid}")

async def check_fill_buy(app: Application, st: MMState) -> bool:
    if not st.buy_order_id:
        return False
    try:
        od = await asyncio.to_thread(KC.get_order, st.buy_order_id)
    except Exception:
        return False

    is_active = bool(od.get("isActive", True))
    deal_size = float(od.get("dealSize") or 0.0)
    deal_funds = float(od.get("dealFunds") or 0.0)

    if (not is_active) and deal_size > 0:
        avg = (deal_funds / deal_size) if deal_size > 0 else st.buy_price
        st.inv_qty += deal_size
        st.inv_entry_price = avg
        st.inv_entry_ts = now_s()

        await notify(app, f"✅ BUY filled {st.symbol} qty={deal_size:.6f} avg={avg:.6f}")

        st.buy_order_id = ""
        st.buy_price = 0.0
        st.buy_size = 0.0
        return True

    return False

async def check_fill_sell(app: Application, st: MMState) -> bool:
    if not st.sell_order_id:
        return False
    try:
        od = await asyncio.to_thread(KC.get_order, st.sell_order_id)
    except Exception:
        return False

    is_active = bool(od.get("isActive", True))
    deal_size = float(od.get("dealSize") or 0.0)
    deal_funds = float(od.get("dealFunds") or 0.0)

    if (not is_active) and deal_size > 0:
        avg = (deal_funds / deal_size) if deal_size > 0 else st.sell_price
        qty = min(st.inv_qty, deal_size)

        entry = st.inv_entry_price
        gross = (avg - entry) * qty
        fee = calc_fees(entry, avg, qty, CFG.fee_rate)
        net = gross - fee

        log_trade(
            is_mock=(CFG.trade_mode == "mock"),
            exchange="KuCoin",
            symbol=st.symbol,
            side="MM_LONG",
            qty=qty,
            stake_usdt=CFG.stake_usdt,
            entry_price=entry,
            exit_price=avg,
            gross_pnl=gross,
            fee_paid=fee,
            net_pnl=net,
            reason=f"SPREAD_CYCLE_{CFG.exit_mode.upper()}",
            buy_order_id="",
            sell_order_id=st.sell_order_id,
        )

        CFG.total_net_pnl += net
        CFG.trades += 1
        save_config()

        await notify(app, f"🎯 SELL filled {st.symbol} qty={qty:.6f} avg={avg:.6f} | NET {net:+.4f} USDT")

        st.inv_qty = max(0.0, st.inv_qty - qty)
        if st.inv_qty <= 1e-12:
            st.inv_qty = 0.0
            st.inv_entry_price = 0.0
            st.inv_entry_ts = 0

        st.sell_order_id = ""
        st.sell_price = 0.0
        st.sell_size = 0.0

        st.cooldown_until = now_s() + CFG.cooldown_seconds
        return True

    return False

async def place_buy_if_ok(app: Application, st: MMState, lvl: Dict[str, float]):
    if st.inv_qty > 0:
        return
    if st.buy_order_id or st.sell_order_id:
        return
    if now_s() < st.cooldown_until:
        return

    bid = float(lvl["bid"]); ask = float(lvl["ask"])
    spr = spread_pct(bid, ask)
    if spr < CFG.threshold_pct:
        return

    # Inventory cap check
    cap = float(CFG.inventory_cap_usdt)
    if cap > 0:
        inv_val = inventory_value_usdt(lvl, st)
        if inv_val >= cap:
            return

    funds = float(CFG.stake_usdt)
    if bid <= 0:
        return
    size = funds / bid
    if size <= 0:
        return

    if CFG.trade_mode == "mock":
        st.inv_qty = size
        st.inv_entry_price = bid
        st.inv_entry_ts = now_s()
        await notify(app, f"🟢 MOCK BUY {st.symbol} qty={size:.6f} @ bid={bid:.6f} | spread={spr:.2f}%")
        return

    try:
        oid = await asyncio.to_thread(KC.place_limit, st.symbol, "buy", bid, size, POST_ONLY)
        st.buy_order_id = oid
        st.buy_ts = now_s()
        st.buy_price = bid
        st.buy_size = size
        await notify(app, f"🟢 BUY order {st.symbol} price={bid:.6f} size={size:.6f} | spread={spr:.2f}%")
    except Exception as e:
        await notify(app, f"⚠️ BUY place failed {st.symbol} ({type(e).__name__})")

async def _log_and_clear_inventory(app: Application, st: MMState, exit_price: float, reason: str, is_live: bool, sell_oid: str = ""):
    entry = st.inv_entry_price
    qty = st.inv_qty
    if qty <= 0 or entry <= 0 or exit_price <= 0:
        st.inv_qty = 0.0
        st.inv_entry_price = 0.0
        st.inv_entry_ts = 0
        return

    gross = (exit_price - entry) * qty
    fee = calc_fees(entry, exit_price, qty, CFG.fee_rate)
    net = gross - fee

    log_trade(
        is_mock=(not is_live),
        exchange="KuCoin",
        symbol=st.symbol,
        side="MM_LONG",
        qty=qty,
        stake_usdt=CFG.stake_usdt,
        entry_price=entry,
        exit_price=exit_price,
        gross_pnl=gross,
        fee_paid=fee,
        net_pnl=net,
        reason=reason,
        buy_order_id="",
        sell_order_id=sell_oid,
    )
    CFG.total_net_pnl += net
    CFG.trades += 1
    save_config()

    emoji = "🎯" if net >= 0 else "🟥"
    await notify(app, f"{emoji} EXIT {st.symbol} qty={qty:.6f} @ {exit_price:.6f} | NET {net:+.4f} USDT | {reason}")

    st.inv_qty = 0.0
    st.inv_entry_price = 0.0
    st.inv_entry_ts = 0
    st.cooldown_until = now_s() + max(5, CFG.cooldown_seconds)

async def place_sell_if_have_inventory(app: Application, st: MMState, lvl: Dict[str, float]):
    if st.inv_qty <= 0:
        return

    bid = float(lvl["bid"]); ask = float(lvl["ask"])
    entry = st.inv_entry_price
    if entry <= 0:
        return

    # Panic stop
    panic = float(CFG.sl_pct) / 100.0
    if bid > 0 and bid <= entry * (1.0 - panic):
        # cancel any open sell order first
        if st.sell_order_id and CFG.trade_mode == "live":
            oid = st.sell_order_id
            st.sell_order_id = ""
            await asyncio.to_thread(KC.cancel_order, oid)

        if CFG.trade_mode == "mock":
            await _log_and_clear_inventory(app, st, bid, "PANIC_EXIT_MOCK", is_live=False)
            return

        # live market exit
        try:
            oid = await asyncio.to_thread(kucoin_market_sell, st.symbol, st.inv_qty, KC)
            await notify(app, f"🟥 PANIC EXIT (LIVE) {st.symbol} market sell qty={st.inv_qty:.6f} order={oid}")
        except Exception as e:
            await notify(app, f"⚠️ PANIC market sell failed {st.symbol} ({type(e).__name__})")
            return

        # approximate logging at bid
        await _log_and_clear_inventory(app, st, bid, "PANIC_EXIT_LIVE_APPROX", is_live=True)
        return

    # Exit mode: MARKET (immediate)
    if CFG.exit_mode == "market":
        if st.sell_order_id and CFG.trade_mode == "live":
            oid = st.sell_order_id
            st.sell_order_id = ""
            await asyncio.to_thread(KC.cancel_order, oid)

        if CFG.trade_mode == "mock":
            # simulate market sell at bid
            if bid > 0:
                await _log_and_clear_inventory(app, st, bid, "MARKET_EXIT_MOCK", is_live=False)
            return

        try:
            oid = await asyncio.to_thread(kucoin_market_sell, st.symbol, st.inv_qty, KC)
            await notify(app, f"🔵 MARKET EXIT (LIVE) {st.symbol} qty={st.inv_qty:.6f} order={oid}")
            # approximate at bid
            if bid > 0:
                await _log_and_clear_inventory(app, st, bid, "MARKET_EXIT_LIVE_APPROX", is_live=True, sell_oid=oid)
        except Exception as e:
            await notify(app, f"⚠️ MARKET EXIT failed {st.symbol} ({type(e).__name__})")
        return

    # Profit target gate (only used when trying to exit with limit)
    target = float(CFG.tp_pct) / 100.0
    target_px = entry * (1.0 + target)

    # If ask doesn't meet target, wait
    if ask <= 0 or ask < target_px:
        return

    # If already have a sell order, HYBRID may decide to fallback after wait
    if st.sell_order_id:
        if CFG.trade_mode == "live" and CFG.exit_mode == "hybrid":
            # If sell has been waiting longer than hybrid_exit_wait_sec -> cancel & market sell
            if (now_s() - st.sell_ts) >= int(CFG.hybrid_exit_wait_sec):
                oid = st.sell_order_id
                st.sell_order_id = ""
                await asyncio.to_thread(KC.cancel_order, oid)
                await notify(app, f"🔁 HYBRID fallback: cancel SELL {st.symbol} {oid} -> market sell")

                try:
                    moid = await asyncio.to_thread(kucoin_market_sell, st.symbol, st.inv_qty, KC)
                    await notify(app, f"🔵 HYBRID MARKET EXIT (LIVE) {st.symbol} qty={st.inv_qty:.6f} order={moid}")
                    # approximate at bid
                    if bid > 0:
                        await _log_and_clear_inventory(app, st, bid, "HYBRID_FALLBACK_MARKET_LIVE_APPROX", is_live=True, sell_oid=moid)
                except Exception as e:
                    await notify(app, f"⚠️ HYBRID market sell failed {st.symbol} ({type(e).__name__})")
        return

    # Place sell order (LIMIT or HYBRID initial step)
    price = ask
    size = st.inv_qty

    if CFG.trade_mode == "mock":
        # mock: immediate fill at ask
        await _log_and_clear_inventory(app, st, ask, f"SPREAD_CYCLE_{CFG.exit_mode.upper()}_MOCK", is_live=False)
        return

    try:
        oid = await asyncio.to_thread(KC.place_limit, st.symbol, "sell", price, size, POST_ONLY)
        st.sell_order_id = oid
        st.sell_ts = now_s()
        st.sell_price = price
        st.sell_size = size
        await notify(app, f"🎯 SELL order {st.symbol} price={price:.6f} size={size:.6f} ({CFG.exit_mode.upper()})")
    except Exception as e:
        await notify(app, f"⚠️ SELL place failed {st.symbol} ({type(e).__name__})")

async def requote_stale_orders(app: Application, st: MMState, lvl: Dict[str, float]):
    # Only affects limit orders
    if st.buy_order_id and (now_s() - st.buy_ts) >= CFG.requote_sec:
        new_bid = float(lvl["bid"])
        if new_bid > 0 and st.buy_price > 0 and abs(new_bid - st.buy_price) / st.buy_price > 0.0002:
            oid = st.buy_order_id
            st.buy_order_id = ""
            await asyncio.to_thread(KC.cancel_order, oid)
            await notify(app, f"🔁 Requote BUY {st.symbol} cancel {oid}")
            try:
                oid2 = await asyncio.to_thread(KC.place_limit, st.symbol, "buy", new_bid, st.buy_size, POST_ONLY)
                st.buy_order_id = oid2
                st.buy_ts = now_s()
                st.buy_price = new_bid
                await notify(app, f"🟢 BUY re-placed {st.symbol} @ {new_bid:.6f}")
            except Exception as e:
                await notify(app, f"⚠️ BUY requote failed {st.symbol} ({type(e).__name__})")

    if st.sell_order_id and (now_s() - st.sell_ts) >= CFG.requote_sec and st.inv_qty > 0:
        # in HYBRID we still allow requote while waiting
        new_ask = float(lvl["ask"])
        entry = st.inv_entry_price
        target = float(CFG.tp_pct) / 100.0
        if new_ask >= entry * (1.0 + target) and new_ask > 0 and st.sell_price > 0:
            if abs(new_ask - st.sell_price) / st.sell_price > 0.0002:
                oid = st.sell_order_id
                st.sell_order_id = ""
                await asyncio.to_thread(KC.cancel_order, oid)
                await notify(app, f"🔁 Requote SELL {st.symbol} cancel {oid}")
                try:
                    oid2 = await asyncio.to_thread(KC.place_limit, st.symbol, "sell", new_ask, st.inv_qty, POST_ONLY)
                    st.sell_order_id = oid2
                    st.sell_ts = now_s()
                    st.sell_price = new_ask
                    st.sell_size = st.inv_qty
                    await notify(app, f"🎯 SELL re-placed {st.symbol} @ {new_ask:.6f}")
                except Exception as e:
                    await notify(app, f"⚠️ SELL requote failed {st.symbol} ({type(e).__name__})")

async def evaluate_symbol(app: Application, symbol: str):
    st = MM.get(symbol)
    if not st:
        st = MMState(symbol=symbol)
        MM[symbol] = st

    if CFG.trade_mode == "live":
        KC.refresh_keys()
        if not KC.has_keys():
            CFG.trade_mode = "mock"
            save_config()
            await notify(app, "⚠️ KUCOIN keys saknas. Växlar till mock.")
            return

    try:
        lvl = await asyncio.to_thread(KC.get_level1, symbol)
    except Exception:
        return

    if CFG.trade_mode == "live":
        await cancel_if_stale(app, st)
        await check_fill_buy(app, st)
        await check_fill_sell(app, st)

    await place_sell_if_have_inventory(app, st, lvl)

    if CFG.trade_mode == "live":
        await requote_stale_orders(app, st, lvl)

    await place_buy_if_ok(app, st, lvl)

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
        "Strategi: Spread Market Maker (limit på bid -> exit enligt /exit).\n"
        "Styrning:\n"
        "• /threshold = min spread % för att lägga buy\n"
        "• /tp = target %\n"
        "• /sl = panic stop %\n"
        "• /stake = USDT per cycle\n"
        "• /exit = LIMIT / HYBRID / MARKET\n",
        reply_markup=main_menu_keyboard(),
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    KC.refresh_keys()
    open_inv = [s for s, st in MM.items() if st.inv_qty > 0]
    pending = [s for s, st in MM.items() if st.buy_order_id or st.sell_order_id]
    lines = [
        f"ENGINE: {'ON' if CFG.engine_on else 'OFF'}",
        "Strategy: Spread Market Maker (inventory-first)",
        f"Trade mode: {CFG.trade_mode} (live keys: {'OK' if KC.has_keys() else 'missing'})",
        f"Exit mode: {CFG.exit_mode.upper()} (hybrid wait: {CFG.hybrid_exit_wait_sec}s)",
        f"Min spread (/threshold): {CFG.threshold_pct:.2f}%",
        f"Target (/tp): {CFG.tp_pct:.2f}%",
        f"Panic stop (/sl): {CFG.sl_pct:.2f}%",
        f"Stake per cycle: {CFG.stake_usdt:.2f} USDT",
        f"Inventory cap / coin: {CFG.inventory_cap_usdt:.2f} USDT",
        f"Order timeout: {CFG.order_timeout_sec}s | requote: {CFG.requote_sec}s | loop: {POLL_SECONDS}s",
        f"Trades (closed cycles): {CFG.trades}",
        f"Total NET PnL: {CFG.total_net_pnl:.4f} USDT",
        f"Coins ({len(CFG.active_coins)}): {CFG.active_coins}",
        f"Inventory open: {open_inv if open_inv else 'none'}",
        f"Pending orders: {pending if pending else 'none'}",
        f"Notify: {'ON' if CFG.notify_trades else 'OFF'}",
    ]
    await update.message.reply_text("\n".join(lines), reply_markup=main_menu_keyboard())

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Total NET PnL: {CFG.total_net_pnl:.4f} USDT\nClosed cycles: {CFG.trades}")

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
    # cancel orders + close inventory
    app = context.application
    KC.refresh_keys()
    for sym, st in list(MM.items()):
        try:
            lvl = await asyncio.to_thread(KC.get_level1, sym)
        except Exception:
            lvl = {"bid": 0.0, "ask": 0.0, "price": 0.0}

        if st.buy_order_id and CFG.trade_mode == "live":
            oid = st.buy_order_id
            st.buy_order_id = ""
            await asyncio.to_thread(KC.cancel_order, oid)

        if st.sell_order_id and CFG.trade_mode == "live":
            oid = st.sell_order_id
            st.sell_order_id = ""
            await asyncio.to_thread(KC.cancel_order, oid)

        if st.inv_qty > 0:
            bid = float(lvl.get("bid", 0.0)) or float(lvl.get("price", 0.0)) or 0.0
            if CFG.trade_mode == "mock":
                if bid > 0:
                    await _log_and_clear_inventory(app, st, bid, "FORCE_CLOSE_MOCK", is_live=False)
            else:
                if KC.has_keys():
                    try:
                        oid = await asyncio.to_thread(kucoin_market_sell, sym, st.inv_qty, KC)
                        await notify(app, f"🔵 FORCE MARKET SELL {sym} order={oid}")
                    except Exception:
                        pass
                if bid > 0:
                    await _log_and_clear_inventory(app, st, bid, "FORCE_CLOSE_LIVE_APPROX", is_live=True)

    await update.message.reply_text("✅ Close all: orders avbrutna + inventory stängd.")

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
            await update.message.reply_text("Ogiltigt värde. Ex: /threshold 0.35")
            return
        CFG.threshold_pct = float(round(clamp(v, 0.05, 5.0), 4))
        save_config()
        await update.message.reply_text(f"✅ Min spread satt till: {CFG.threshold_pct:.2f}%")
        return
    await update.message.reply_text(
        f"Min spread nu: {CFG.threshold_pct:.2f}%\nVälj:",
        reply_markup=threshold_buttons(CFG.threshold_pct),
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
        await update.message.reply_text(f"✅ Stake per cycle satt till: {CFG.stake_usdt:.2f} USDT")
        return
    await update.message.reply_text(
        f"Stake per cycle nu: {CFG.stake_usdt:.2f} USDT\nVälj:",
        reply_markup=stake_buttons(CFG.stake_usdt),
    )

async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        v = parse_float(args[0])
        if v is None:
            await update.message.reply_text("Ogiltigt. Ex: /tp 0.15")
            return
        CFG.tp_pct = float(round(clamp(v, 0.02, 2.00), 4))
        save_config()
        await update.message.reply_text(f"✅ Target profit satt till: {CFG.tp_pct:.2f}%")
        return
    await update.message.reply_text(
        f"Target profit nu: {CFG.tp_pct:.2f}%\nVälj:",
        reply_markup=tp_buttons(CFG.tp_pct),
    )

async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        v = parse_float(args[0])
        if v is None:
            await update.message.reply_text("Ogiltigt. Ex: /sl 0.6")
            return
        CFG.sl_pct = float(round(clamp(v, 0.10, 5.00), 4))
        save_config()
        await update.message.reply_text(f"✅ Panic stop satt till: {CFG.sl_pct:.2f}%")
        return
    await update.message.reply_text(
        f"Panic stop nu: {CFG.sl_pct:.2f}%\nVälj:",
        reply_markup=sl_buttons(CFG.sl_pct),
    )

async def cmd_exit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if hasattr(context, "args") else []
    if args:
        m = args[0].lower().strip()
        if m not in EXIT_MODES:
            await update.message.reply_text("Ogiltigt. Använd: /exit limit | hybrid | market")
            return
        CFG.exit_mode = m
        save_config()
        await update.message.reply_text(f"✅ Exit mode satt till: {CFG.exit_mode.upper()}")
        return

    await update.message.reply_text(
        f"Exit mode nu: {CFG.exit_mode.upper()}\nVälj:",
        reply_markup=exit_mode_buttons(CFG.exit_mode),
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
                await update.message.reply_text("⚠️ KUCOIN_KEY/SECRET/PASSPHRASE saknas. Sätt dem först.")
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

    await update.message.reply_text("📄 Exporterar CSV-loggar.\n• mock_trade_log.csv\n• real_trade_log.csv\n")

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

    if data.startswith("thr:"):
        payload = data.split("thr:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        v = parse_float(payload)
        if v is None:
            await q.edit_message_text("Ogiltigt värde.")
            return
        CFG.threshold_pct = float(round(clamp(v, 0.05, 5.0), 4))
        save_config()
        await q.edit_message_text(f"✅ Min spread satt till: {CFG.threshold_pct:.2f}%")
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
        await q.edit_message_text(f"✅ Stake per cycle satt till: {CFG.stake_usdt:.2f} USDT")
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
        CFG.tp_pct = float(round(clamp(v, 0.02, 2.00), 4))
        save_config()
        await q.edit_message_text(f"✅ Target profit satt till: {CFG.tp_pct:.2f}%")
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
        CFG.sl_pct = float(round(clamp(v, 0.10, 5.00), 4))
        save_config()
        await q.edit_message_text(f"✅ Panic stop satt till: {CFG.sl_pct:.2f}%")
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
        if payload == "live":
            KC.refresh_keys()
            if not KC.has_keys():
                await q.edit_message_text("⚠️ KUCOIN keys saknas. Kan inte slå på live.")
                return
        if payload in ("mock", "live"):
            CFG.trade_mode = payload
            save_config()
            await q.edit_message_text(f"✅ Trade mode: {CFG.trade_mode}")
            return

    if data.startswith("ex:"):
        payload = data.split("ex:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        payload = payload.lower().strip()
        if payload not in EXIT_MODES:
            await q.edit_message_text("Ogiltigt val.")
            return
        CFG.exit_mode = payload
        save_config()
        await q.edit_message_text(f"✅ Exit mode satt till: {CFG.exit_mode.upper()}")
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
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("stake", cmd_stake))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("sl", cmd_sl))
    app.add_handler(CommandHandler("exit", cmd_exit))
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
    print("Mp ORBbot running (Spread MM + /exit)...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
