# main.py
# Mp ORBbot - Strategy 1: VWAP Pullback + EMA trend, LONG only
# Telegram: stable (no job_queue). Adds:
#  - Stake: /stake + button menu
#  - Coins: /coins + button menu
#  - Notify reliability: /engine_on binds notify_chat_id automatically
#  - /status shows notify_chat_id
#
# Env required:
#   TELEGRAM_TOKEN="xxxx"
#   (Fallback supported) TELEGRAM_BOT_TOKEN="xxxx"
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

POLL_SECONDS = 10
DEFAULT_FEE_RATE = 0.001  # 0.1%

# Strategy defaults
EMA_FAST = 20
EMA_SLOW = 50
VWAP_LOOKBACK_MIN_DEFAULT = 180
TP_PCT_DEFAULT = 0.70
SL_PCT_DEFAULT = 0.35
PULLBACK_TOL_PCT_DEFAULT = 0.20  # /threshold controls this
COOLDOWN_SECONDS_DEFAULT = 180
MAX_POSITIONS_DEFAULT = 5
STAKE_USDT_DEFAULT = 30.0

ENTRY_MODES = ["break", "retest"]  # kept for compatibility/UI only


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


def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period + 5:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def safe_symbol(sym: str) -> str:
    s = sym.upper().replace("_", "-").replace("/", "-")
    if "-" not in s and s.endswith("USDT"):
        s = s[:-4] + "-USDT"
    return s


# -----------------------------
# Config
# -----------------------------
@dataclass
class BotConfig:
    engine_on: bool = False
    trade_mode: str = "mock"  # mock | live
    entry_mode: str = "retest"  # UI only
    threshold_pct: float = PULLBACK_TOL_PCT_DEFAULT  # pullback tolerance to EMA20
    fee_rate: float = DEFAULT_FEE_RATE
    active_coins: List[str] = None

    # Strategy params
    tp_pct: float = TP_PCT_DEFAULT
    sl_pct: float = SL_PCT_DEFAULT
    vwap_lookback_min: int = VWAP_LOOKBACK_MIN_DEFAULT
    cooldown_seconds: int = COOLDOWN_SECONDS_DEFAULT
    max_positions: int = MAX_POSITIONS_DEFAULT
    stake_usdt: float = STAKE_USDT_DEFAULT

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

        cfg.trade_mode = cfg.trade_mode if cfg.trade_mode in ("mock", "live") else "mock"
        cfg.entry_mode = cfg.entry_mode if cfg.entry_mode in ENTRY_MODES else "retest"

        cfg.threshold_pct = float(clamp(float(cfg.threshold_pct), 0.05, 2.0))
        cfg.tp_pct = float(clamp(float(cfg.tp_pct), 0.20, 2.0))
        cfg.sl_pct = float(clamp(float(cfg.sl_pct), 0.10, 2.0))
        cfg.vwap_lookback_min = int(clamp(int(cfg.vwap_lookback_min), 60, 720))
        cfg.cooldown_seconds = int(clamp(int(cfg.cooldown_seconds), 30, 1800))
        cfg.max_positions = int(clamp(int(cfg.max_positions), 1, 10))
        cfg.stake_usdt = float(clamp(float(cfg.stake_usdt), 5.0, 500.0))

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

    def get_price(self, symbol: str) -> float:
        r = requests.get(
            f"{self.base}/api/v1/market/orderbook/level1",
            params={"symbol": symbol},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return float(data["data"]["price"])

    def get_klines_1m(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict[str, float]]:
        params = {"symbol": symbol, "type": "1min", "startAt": int(start_ts), "endAt": int(end_ts)}
        r = requests.get(f"{self.base}/api/v1/market/candles", params=params, timeout=12)
        r.raise_for_status()
        raw = r.json().get("data", [])
        out = []
        for row in raw:
            # [ time, open, close, high, low, volume, turnover ] (strings)
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
    qty: float
    entry_ts: int
    tp_price: float
    sl_price: float
    stake_usdt: float
    live_buy_order_id: str = ""
    live_sell_order_id: str = ""


POS: Dict[str, Position] = {}
LAST_CANDLE_T: Dict[str, int] = {}
COOLDOWN_UNTIL: Dict[str, int] = {}


# -----------------------------
# UI / Telegram keyboards
# -----------------------------
def main_menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/threshold"), KeyboardButton("/stake")],
        [KeyboardButton("/coins"), KeyboardButton("/trade_mode")],
        [KeyboardButton("/notify"), KeyboardButton("/export_csv")],
        [KeyboardButton("/close_all"), KeyboardButton("/reset_pnl")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


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


def stake_buttons(current: float) -> InlineKeyboardMarkup:
    options = [10, 20, 30, 50, 100]
    row = []
    rows = []
    for x in options:
        label = f"{x} USDT" + (" ✅" if abs(current - float(x)) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"stk:{x}"))
    rows.append(row)
    rows.append([InlineKeyboardButton("Stäng", callback_data="stk:close")])
    return InlineKeyboardMarkup(rows)


def coins_buttons(current_list: List[str]) -> InlineKeyboardMarkup:
    # presets: 2,3,5 from DEFAULT_COINS
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
# Strategy math
# -----------------------------
def compute_vwap(candles: List[Dict[str, float]]) -> Optional[float]:
    vol_sum = 0.0
    pv_sum = 0.0
    for c in candles:
        v = float(c["v"])
        if v <= 0:
            continue
        tp = (c["h"] + c["l"] + c["c"]) / 3.0
        pv_sum += tp * v
        vol_sum += v
    if vol_sum <= 0:
        return None
    return pv_sum / vol_sum


def in_cooldown(symbol: str) -> bool:
    return int(time.time()) < int(COOLDOWN_UNTIL.get(symbol, 0))


def can_open_new_position() -> bool:
    return len(POS) < int(CFG.max_positions)


async def open_long(app: Application, symbol: str, price: float, reason: str):
    if symbol in POS or in_cooldown(symbol) or not can_open_new_position():
        return

    funds = float(CFG.stake_usdt)
    qty = funds / price

    tp = price * (1.0 + CFG.tp_pct / 100.0)
    sl = price * (1.0 - CFG.sl_pct / 100.0)

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

    POS[symbol] = Position(
        symbol=symbol,
        entry_price=price,
        qty=qty,
        entry_ts=int(time.time()),
        tp_price=tp,
        sl_price=sl,
        stake_usdt=funds,
        live_buy_order_id=buy_id,
        live_sell_order_id="",
    )

    tag = "LIVE" if (CFG.trade_mode == "live" and buy_id) else "MOCK"
    await notify_trade(app, f"🟢 ENTRY ({tag}) {symbol} @ {price:.6f} | TP={tp:.6f} SL={sl:.6f} | stake={funds:.2f} | {reason}")


async def close_long(app: Application, symbol: str, price: float, reason: str):
    if symbol not in POS:
        return
    p = POS.pop(symbol)

    gross = (price - p.entry_price) * p.qty
    fee = calc_fees(p.entry_price, price, p.qty, CFG.fee_rate)
    net = gross - fee

    CFG.trades += 1
    CFG.total_net_pnl += net
    save_config()

    COOLDOWN_UNTIL[symbol] = int(time.time()) + int(CFG.cooldown_seconds)

    sell_id = ""
    if CFG.trade_mode == "live":
        KC.refresh_keys()
        if KC.has_keys():
            try:
                sell_id = await asyncio.to_thread(KC.place_market_sell_size, symbol, p.qty)
            except Exception:
                await notify_trade(app, f"⚠️ Live SELL misslyckades för {symbol}.")

    log_trade(
        is_mock=(CFG.trade_mode == "mock"),
        exchange="KuCoin",
        symbol=symbol,
        side="LONG",
        qty=p.qty,
        stake_usdt=p.stake_usdt,
        entry_price=p.entry_price,
        exit_price=price,
        gross_pnl=gross,
        fee_paid=fee,
        net_pnl=net,
        reason=reason,
        buy_order_id=p.live_buy_order_id,
        sell_order_id=sell_id,
    )

    emoji = "🎯" if net >= 0 else "🟥"
    await notify_trade(app, f"{emoji} EXIT {symbol} @ {price:.6f} | Net {net:+.4f} USDT | {reason}")


async def evaluate_symbol(app: Application, symbol: str):
    end = int(time.time())
    start = end - int(CFG.vwap_lookback_min) * 60
    candles = await asyncio.to_thread(KC.get_klines_1m, symbol, start, end)
    if len(candles) < max(EMA_SLOW + 10, 80):
        return

    last_t = candles[-1]["t"]
    if LAST_CANDLE_T.get(symbol) == last_t:
        return
    LAST_CANDLE_T[symbol] = last_t

    closes = [c["c"] for c in candles]
    opens = [c["o"] for c in candles]
    lows = [c["l"] for c in candles]

    ema20 = ema(closes, EMA_FAST)
    ema50 = ema(closes, EMA_SLOW)
    if ema20 is None or ema50 is None:
        return

    vwap = compute_vwap(candles)
    if vwap is None:
        return

    last_close = closes[-1]
    last_open = opens[-1]
    last_low = lows[-1]

    # manage open position first
    if symbol in POS:
        price = await asyncio.to_thread(KC.get_price, symbol)
        if price >= POS[symbol].tp_price:
            await close_long(app, symbol, price, reason="TP")
        elif price <= POS[symbol].sl_price:
            await close_long(app, symbol, price, reason="SL")
        return

    # entry checks
    if in_cooldown(symbol) or not can_open_new_position():
        return

    if not (ema20 > ema50):
        return
    if not (last_close > vwap):
        return

    tol = CFG.threshold_pct / 100.0
    pullback_ok = last_low <= ema20 * (1.0 + tol)
    bullish_ok = (last_close > last_open) and (last_close > ema20)

    if pullback_ok and bullish_ok:
        price = await asyncio.to_thread(KC.get_price, symbol)
        reason = f"VWAP+EMA Pullback | EMA20={ema20:.6f} EMA50={ema50:.6f} VWAP={vwap:.6f}"
        await open_long(app, symbol, price, reason=reason)


# -----------------------------
# Engine loop
# -----------------------------
async def engine_background_loop(app: Application):
    while True:
        try:
            if CFG.engine_on:
                coins = list(CFG.active_coins or [])
                for sym in coins:
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
    # Always bind notify to this chat on /start
    CFG.notify_chat_id = int(update.effective_chat.id)
    CFG.notify_trades = True
    save_config()
    await update.message.reply_text(
        "Mp ORBbot (VWAP Pullback) är igång.\n"
        "LONG only | EMA20>EMA50 & Close>VWAP & Pullback->Bullish reclaim.\n\n"
        "Tips:\n"
        "• /engine_on binder notiser till denna chat\n"
        "• /stake (knappval)\n"
        "• /coins (knappval)\n",
        reply_markup=main_menu_keyboard(),
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    KC.refresh_keys()
    lines = [
        f"ENGINE: {'ON' if CFG.engine_on else 'OFF'}",
        f"Trade mode: {CFG.trade_mode} (live keys: {'OK' if KC.has_keys() else 'missing'})",
        f"Pullback tol (/threshold): {CFG.threshold_pct:.2f}%",
        f"Stake per trade: {CFG.stake_usdt:.2f} USDT",
        f"TP/SL: {CFG.tp_pct:.2f}% / {CFG.sl_pct:.2f}%",
        f"VWAP lookback: {CFG.vwap_lookback_min} min",
        f"Cooldown: {CFG.cooldown_seconds}s",
        f"Max positions: {CFG.max_positions}",
        f"Trades: {CFG.trades}",
        f"Total NET PnL: {CFG.total_net_pnl:.4f} USDT",
        f"Coins ({len(CFG.active_coins)}): {CFG.active_coins}",
        f"Open positions: {list(POS.keys()) if POS else 'none'}",
        f"Notify: {'ON' if CFG.notify_trades else 'OFF'}",
        f"Notify chat id: {CFG.notify_chat_id}",
    ]
    await update.message.reply_text("\n".join(lines), reply_markup=main_menu_keyboard())


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Total NET PnL: {CFG.total_net_pnl:.4f} USDT\nTrades: {CFG.trades}")


async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CFG.engine_on = True
    # Notify-fix: bind to this chat every time
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
    total = 0.0
    for sym in list(POS.keys()):
        try:
            price = await asyncio.to_thread(KC.get_price, sym)
            p = POS.get(sym)
            if p:
                gross = (price - p.entry_price) * p.qty
                fee = calc_fees(p.entry_price, price, p.qty, CFG.fee_rate)
                total += (gross - fee)
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
            await update.message.reply_text("Ogiltigt värde. Ex: /threshold 0.2")
            return
        CFG.threshold_pct = float(round(clamp(v, 0.05, 2.0), 4))
        save_config()
        await update.message.reply_text(f"✅ Pullback tolerans satt till: {CFG.threshold_pct:.2f}%")
        return
    await update.message.reply_text(
        f"Pullback tolerans (/threshold) nu: {CFG.threshold_pct:.2f}%\nVälj:",
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
        await update.message.reply_text(f"✅ Stake per trade satt till: {CFG.stake_usdt:.2f} USDT")
        return
    await update.message.reply_text(
        f"Stake per trade nu: {CFG.stake_usdt:.2f} USDT\nVälj:",
        reply_markup=stake_buttons(CFG.stake_usdt),
    )


async def cmd_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /coins 5 => first 5 default
    /coins BTC-USDT ETH-USDT ... => explicit list
    /coins => shows menu
    """
    args = context.args if hasattr(context, "args") else []
    if args:
        # number?
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

        # explicit list
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
        "Allt i USDT."
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
    data = q.data or ""
    await q.answer()

    # threshold
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
        await q.edit_message_text(f"✅ Pullback tolerans satt till: {CFG.threshold_pct:.2f}%")
        return

    # stake
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

    # coins preset
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

    # trade mode
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

    await q.edit_message_text("Ok.")


# -----------------------------
# Boot
# -----------------------------
def build_app() -> Application:
    # Support both names (so deployments don't break)
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
    print("Mp ORBbot running (VWAP Pullback)...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
