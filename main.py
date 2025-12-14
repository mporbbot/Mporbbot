# main.py
# Mp ORBbot - Telegram control + settings + threshold fix (free input + buttons)
# Note: Trading/KuCoin integration is intentionally stubbed/mocked here.
# Run:
#   pip install python-telegram-bot==20.7
#   export TELEGRAM_BOT_TOKEN="xxx"
#   python main.py

import os
import json
import csv
import time
import math
import random
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

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

# Threshold controls
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.05

# Risk controls
RISK_MIN = 0.1
RISK_MAX = 5.0

# Timeframes
ALLOWED_TF = ["1m", "3m", "5m", "15m"]

# AI modes
ALLOWED_MODES = ["aggressive", "neutral", "cautious"]

# Mock engine tick seconds (fast)
ENGINE_TICK_SECONDS = 3

# Default coins
DEFAULT_COINS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]

# Mock trade size (as per your memory)
DEFAULT_MOCK_USDT = 30.0

# Fee (0.1% default)
DEFAULT_FEE_RATE = 0.001


# -----------------------------
# Helpers
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def parse_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ".").strip())
    except Exception:
        return None


def now_ts() -> int:
    return int(time.time())


def fmt2(x: float) -> str:
    return f"{x:.4f}"


def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


# -----------------------------
# Config / State
# -----------------------------
@dataclass
class BotConfig:
    engine_on: bool = False
    ai_mode: str = "neutral"
    timeframe: str = "3m"
    risk_pct: float = 1.0
    threshold: float = 0.30
    trail_trigger_pct: float = 0.50
    trail_dist_pct: float = 0.25
    fee_rate: float = DEFAULT_FEE_RATE
    mode: str = "mock"  # "mock" or "real"
    active_coins: List[str] = None

    # Stats
    trades: int = 0
    total_net_pnl: float = 0.0

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

        # sanitize
        if cfg.ai_mode not in ALLOWED_MODES:
            cfg.ai_mode = "neutral"
        if cfg.timeframe not in ALLOWED_TF:
            cfg.timeframe = "3m"
        cfg.risk_pct = float(clamp(float(cfg.risk_pct), RISK_MIN, RISK_MAX))
        cfg.threshold = float(clamp(float(cfg.threshold), THRESHOLD_MIN, THRESHOLD_MAX))
        cfg.trail_trigger_pct = float(clamp(float(cfg.trail_trigger_pct), 0.1, 5.0))
        cfg.trail_dist_pct = float(clamp(float(cfg.trail_dist_pct), 0.05, 5.0))
        cfg.fee_rate = float(clamp(float(cfg.fee_rate), 0.0, 0.01))
        if cfg.mode not in ("mock", "real"):
            cfg.mode = "mock"
        return cfg


CFG: BotConfig = None

# Simple in-memory positions (mock)
# symbol -> (entry_price, qty, entry_ts)
POSITIONS: Dict[str, Tuple[float, float, int]] = {}

# Mock prices
MOCK_PRICE: Dict[str, float] = {s: random.uniform(0.5, 1.5) for s in DEFAULT_COINS}
# Make BTC/ETH look bigger
MOCK_PRICE["BTCUSDT"] = 65000.0
MOCK_PRICE["ETHUSDT"] = 3500.0
MOCK_PRICE["XRPUSDT"] = 0.60
MOCK_PRICE["ADAUSDT"] = 0.45
MOCK_PRICE["LINKUSDT"] = 14.0


def load_config() -> BotConfig:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BotConfig.from_json(data)
        except Exception:
            pass
    return BotConfig(active_coins=DEFAULT_COINS.copy())


def save_config() -> None:
    global CFG
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CFG.to_json(), f, indent=2, ensure_ascii=False)


# -----------------------------
# Keyboards
# -----------------------------
def main_menu_keyboard() -> ReplyKeyboardMarkup:
    # Simple quick buttons (not mandatory)
    rows = [
        [KeyboardButton("/status"), KeyboardButton("/pnl")],
        [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
        [KeyboardButton("/timeframe"), KeyboardButton("/threshold")],
        [KeyboardButton("/risk"), KeyboardButton("/export_csv")],
        [KeyboardButton("/close_all"), KeyboardButton("/reset_pnl")],
        [KeyboardButton("/mode"), KeyboardButton("/test_buy")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def threshold_buttons(current: float) -> InlineKeyboardMarkup:
    vals = []
    v = THRESHOLD_MIN
    while v <= THRESHOLD_MAX + 1e-9:
        vals.append(round(v, 2))
        v += THRESHOLD_STEP

    rows = []
    row = []
    for x in vals:
        label = f"{x:.2f}" + (" ✅" if abs(x - current) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"thr:{x:.2f}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    rows.append([InlineKeyboardButton("Stäng", callback_data="thr:close")])
    return InlineKeyboardMarkup(rows)


def timeframe_buttons(current: str) -> InlineKeyboardMarkup:
    rows = []
    row = []
    for tf in ALLOWED_TF:
        label = tf + (" ✅" if tf == current else "")
        row.append(InlineKeyboardButton(label, callback_data=f"tf:{tf}"))
    rows.append(row)
    rows.append([InlineKeyboardButton("Stäng", callback_data="tf:close")])
    return InlineKeyboardMarkup(rows)


def mode_buttons(current: str) -> InlineKeyboardMarkup:
    rows = []
    row = []
    for m in ALLOWED_MODES:
        label = m + (" ✅" if m == current else "")
        row.append(InlineKeyboardButton(label, callback_data=f"aim:{m}"))
    rows.append(row)
    rows.append([InlineKeyboardButton("Stäng", callback_data="aim:close")])
    return InlineKeyboardMarkup(rows)


def risk_buttons(current: float) -> InlineKeyboardMarkup:
    # Common presets
    presets = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    rows = []
    row = []
    for r in presets:
        label = f"{r:.1f}%" + (" ✅" if abs(current - r) < 1e-9 else "")
        row.append(InlineKeyboardButton(label, callback_data=f"risk:{r:.1f}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("Stäng", callback_data="risk:close")])
    return InlineKeyboardMarkup(rows)


# -----------------------------
# Logging trades
# -----------------------------
def log_trade(
    is_mock: bool,
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    gross_pnl: float,
    fee_paid: float,
    net_pnl: float,
    reason: str,
):
    path = MOCK_LOG_PATH if is_mock else REAL_LOG_PATH
    ensure_csv_header(
        path,
        [
            "timestamp",
            "symbol",
            "side",
            "qty",
            "entry_price",
            "exit_price",
            "gross_pnl",
            "fee_paid",
            "net_pnl",
            "reason",
        ],
    )
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                now_ts(),
                symbol,
                side,
                f"{qty:.8f}",
                f"{entry_price:.8f}",
                f"{exit_price:.8f}",
                f"{gross_pnl:.8f}",
                f"{fee_paid:.8f}",
                f"{net_pnl:.8f}",
                reason,
            ]
        )


# -----------------------------
# Mock engine (for now)
# -----------------------------
def mock_next_price(symbol: str) -> float:
    # Random walk with symbol-specific volatility
    p = MOCK_PRICE.get(symbol, 1.0)
    vol = 0.0008
    if symbol in ("BTCUSDT", "ETHUSDT"):
        vol = 0.0012
    if symbol in ("XRPUSDT", "ADAUSDT", "LINKUSDT"):
        vol = 0.0020

    drift = 0.0
    shock = random.gauss(0, vol)
    p = max(1e-8, p * (1.0 + drift + shock))
    MOCK_PRICE[symbol] = p
    return p


def should_enter_trade(symbol: str, cfg: BotConfig) -> bool:
    """
    Mock "signal probability" shaped by threshold + ai_mode:
    - Higher threshold => fewer entries
    - aggressive => more entries
    - cautious => fewer entries
    """
    base = 0.10  # base per tick entry chance
    # mode effect
    if cfg.ai_mode == "aggressive":
        base *= 1.30
    elif cfg.ai_mode == "cautious":
        base *= 0.70
    else:
        base *= 1.00

    # threshold effect: map threshold [0.05..0.90] -> multiplier ~ [1.4..0.2]
    t = clamp(cfg.threshold, THRESHOLD_MIN, THRESHOLD_MAX)
    mult = 1.4 - (t - THRESHOLD_MIN) * (1.2 / (THRESHOLD_MAX - THRESHOLD_MIN))
    prob = clamp(base * mult, 0.01, 0.25)

    return random.random() < prob


def position_size_usdt(cfg: BotConfig) -> float:
    # In mock mode, keep it fixed 30 USDT as requested.
    return DEFAULT_MOCK_USDT if cfg.mode == "mock" else DEFAULT_MOCK_USDT


def open_position(symbol: str, cfg: BotConfig) -> None:
    if symbol in POSITIONS:
        return
    price = MOCK_PRICE[symbol]
    usdt = position_size_usdt(cfg)
    qty = usdt / price
    POSITIONS[symbol] = (price, qty, now_ts())


def maybe_close_position(symbol: str, cfg: BotConfig) -> Optional[float]:
    """
    Close logic mock:
    - Use trailing-ish: if price moved up enough, random chance to close
    - Else random stop-out chance
    """
    if symbol not in POSITIONS:
        return None

    entry, qty, ets = POSITIONS[symbol]
    price = MOCK_PRICE[symbol]
    change = (price - entry) / entry * 100.0

    # When up >= trail_trigger => more likely close profit
    if change >= cfg.trail_trigger_pct:
        if random.random() < 0.35:
            return price

    # If down a bit => sometimes stop
    if change <= -max(0.2, cfg.risk_pct * 0.6):
        if random.random() < 0.50:
            return price

    # Occasional close to realize
    if random.random() < 0.05:
        return price

    return None


def close_position(symbol: str, exit_price: float, cfg: BotConfig, reason: str) -> float:
    entry, qty, ets = POSITIONS.pop(symbol)
    gross = (exit_price - entry) * qty
    # fee on entry + exit (approx)
    fee = (entry * qty + exit_price * qty) * cfg.fee_rate
    net = gross - fee

    cfg.trades += 1
    cfg.total_net_pnl += net
    save_config()

    log_trade(
        is_mock=(cfg.mode == "mock"),
        symbol=symbol,
        side="LONG",
        qty=qty,
        entry_price=entry,
        exit_price=exit_price,
        gross_pnl=gross,
        fee_paid=fee,
        net_pnl=net,
        reason=reason,
    )

    return net


async def engine_loop(context: ContextTypes.DEFAULT_TYPE):
    global CFG
    if not CFG.engine_on:
        return

    # update prices
    for sym in CFG.active_coins:
        mock_next_price(sym)

    # entries
    for sym in CFG.active_coins:
        if sym not in POSITIONS and should_enter_trade(sym, CFG):
            open_position(sym, CFG)

    # exits
    for sym in list(POSITIONS.keys()):
        exit_price = maybe_close_position(sym, CFG)
        if exit_price is not None:
            close_position(sym, exit_price, CFG, reason="mock_exit")


# -----------------------------
# Telegram commands
# -----------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Mp ORBbot är igång.\n\n"
        "Tips:\n"
        "• /threshold 0.6 (nu funkar fri input)\n"
        "• /mode (AI-läge)\n"
        "• /engine_on /engine_off\n"
        "• /status\n",
        reply_markup=main_menu_keyboard(),
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    msg = (
        f"ENGINE: {'ON' if CFG.engine_on else 'OFF'}\n"
        f"Aktiva coins: {CFG.active_coins}\n"
        f"Trades: {CFG.trades}\n"
        f"Total NETTO-PNL: {fmt2(CFG.total_net_pnl)} USDT\n"
        f"AI-läge: {CFG.ai_mode}\n"
        f"Timeframe: {CFG.timeframe}\n"
        f"Risk: {CFG.risk_pct}%\n"
        f"Threshold: {CFG.threshold}\n"
        f"Trail trig: {CFG.trail_trigger_pct:.2f}%\n"
        f"Trail dist: {CFG.trail_dist_pct:.2f}%\n"
        f"Mode: {CFG.mode}\n"
        f"Öppna positioner: {list(POSITIONS.keys()) if POSITIONS else 'Inga'}"
    )
    await update.message.reply_text(msg, reply_markup=main_menu_keyboard())


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    await update.message.reply_text(
        f"Total NETTO-PNL: {fmt2(CFG.total_net_pnl)} USDT\nTrades: {CFG.trades}"
    )


async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    CFG.engine_on = True
    save_config()
    await update.message.reply_text("✅ ENGINE ON")


async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    CFG.engine_on = False
    save_config()
    await update.message.reply_text("⛔ ENGINE OFF")


async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    CFG.total_net_pnl = 0.0
    CFG.trades = 0
    save_config()
    await update.message.reply_text("✅ PnL och trade-counter nollställda.")


async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    if not POSITIONS:
        await update.message.reply_text("Inga öppna positioner.")
        return

    # close all at current mock price
    total = 0.0
    for sym in list(POSITIONS.keys()):
        exit_price = MOCK_PRICE.get(sym, 1.0)
        total += close_position(sym, exit_price, CFG, reason="close_all")

    await update.message.reply_text(f"✅ Stängde alla positioner. Netto: {fmt2(total)} USDT")


async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # just tells where files are (telegram file send can be added later)
    await update.message.reply_text(
        "CSV-export är redo lokalt på servern:\n"
        f"• Mock: {MOCK_LOG_PATH}\n"
        f"• Real: {REAL_LOG_PATH}\n\n"
        "Om du vill att boten ska skicka filerna direkt i Telegram, säg till så lägger jag in det."
    )


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    # This is AI mode selection (aggressive/neutral/cautious)
    await update.message.reply_text(
        f"AI-läge nu: {CFG.ai_mode}\nVälj:",
        reply_markup=mode_buttons(CFG.ai_mode),
    )


async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    await update.message.reply_text(
        f"Timeframe nu: {CFG.timeframe}\nVälj:",
        reply_markup=timeframe_buttons(CFG.timeframe),
    )


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    args = context.args if hasattr(context, "args") else []
    if args:
        val = parse_float(args[0])
        if val is None:
            await update.message.reply_text("Ogiltigt värde. Ex: /risk 0.7")
            return
        val = clamp(val, RISK_MIN, RISK_MAX)
        CFG.risk_pct = float(round(val, 4))
        save_config()
        await update.message.reply_text(f"✅ Risk satt till: {CFG.risk_pct}%")
        return

    await update.message.reply_text(
        f"Risk nu: {CFG.risk_pct}%\nVälj eller skriv /risk 0.7",
        reply_markup=risk_buttons(CFG.risk_pct),
    )


async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    FIXED:
    - /threshold 0.6 sets threshold
    - /threshold shows button menu 0.05..0.90
    """
    global CFG
    args = context.args if hasattr(context, "args") else []
    if args:
        val = parse_float(args[0])
        if val is None:
            await update.message.reply_text("Ogiltigt värde. Ex: /threshold 0.6")
            return

        val = clamp(val, THRESHOLD_MIN, THRESHOLD_MAX)
        CFG.threshold = float(round(val, 4))
        save_config()
        await update.message.reply_text(
            f"✅ Threshold satt till: {CFG.threshold}\n"
            f"(Min {THRESHOLD_MIN}, Max {THRESHOLD_MAX})"
        )
        return

    await update.message.reply_text(
        f"Threshold nu: {CFG.threshold}\nVälj eller skriv /threshold 0.6",
        reply_markup=threshold_buttons(CFG.threshold),
    )


async def cmd_test_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Manual mock trade simulation:
    - opens and closes quickly to test logging + pnl
    Usage:
      /test_buy BTCUSDT
    """
    global CFG
    args = context.args if hasattr(context, "args") else []
    sym = (args[0].upper() if args else "BTCUSDT")
    if sym not in CFG.active_coins:
        await update.message.reply_text(f"{sym} är inte i aktiva coins just nu. Aktiva: {CFG.active_coins}")
        return

    # open
    mock_next_price(sym)
    if sym not in POSITIONS:
        open_position(sym, CFG)
    # close after tiny move
    mock_next_price(sym)
    exit_price = MOCK_PRICE[sym]
    net = close_position(sym, exit_price, CFG, reason="test_buy")
    await update.message.reply_text(f"✅ Test-trade klar i {sym}. Netto: {fmt2(net)} USDT")


# -----------------------------
# Callback handler
# -----------------------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    q = update.callback_query
    data = q.data or ""
    await q.answer()

    # threshold buttons
    if data.startswith("thr:"):
        payload = data.split("thr:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        val = parse_float(payload)
        if val is None:
            await q.edit_message_text("Ogiltigt threshold.")
            return
        val = clamp(val, THRESHOLD_MIN, THRESHOLD_MAX)
        CFG.threshold = float(round(val, 4))
        save_config()
        await q.edit_message_text(f"✅ Threshold satt till: {CFG.threshold}")
        return

    # timeframe buttons
    if data.startswith("tf:"):
        payload = data.split("tf:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        if payload not in ALLOWED_TF:
            await q.edit_message_text("Ogiltigt timeframe.")
            return
        CFG.timeframe = payload
        save_config()
        await q.edit_message_text(f"✅ Timeframe satt till: {CFG.timeframe}")
        return

    # ai mode buttons
    if data.startswith("aim:"):
        payload = data.split("aim:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        if payload not in ALLOWED_MODES:
            await q.edit_message_text("Ogiltigt AI-läge.")
            return
        CFG.ai_mode = payload
        save_config()
        await q.edit_message_text(f"✅ AI-läge satt till: {CFG.ai_mode}")
        return

    # risk buttons
    if data.startswith("risk:"):
        payload = data.split("risk:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        val = parse_float(payload)
        if val is None:
            await q.edit_message_text("Ogiltigt risk-värde.")
            return
        val = clamp(val, RISK_MIN, RISK_MAX)
        CFG.risk_pct = float(round(val, 4))
        save_config()
        await q.edit_message_text(f"✅ Risk satt till: {CFG.risk_pct}%")
        return

    await q.edit_message_text("Ok.")


# -----------------------------
# Boot
# -----------------------------
def build_app() -> Application:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Sätt TELEGRAM_BOT_TOKEN i environment innan du startar boten.")

    app = Application.builder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("test_buy", cmd_test_buy))

    # Callbacks
    app.add_handler(CallbackQueryHandler(on_callback))

    # Engine job
    app.job_queue.run_repeating(engine_loop, interval=ENGINE_TICK_SECONDS, first=ENGINE_TICK_SECONDS)

    return app


def main():
    global CFG
    CFG = load_config()
    if CFG.active_coins is None:
        CFG.active_coins = DEFAULT_COINS.copy()
    save_config()

    # Ensure csv headers
    ensure_csv_header(
        MOCK_LOG_PATH,
        ["timestamp", "symbol", "side", "qty", "entry_price", "exit_price", "gross_pnl", "fee_paid", "net_pnl", "reason"],
    )
    ensure_csv_header(
        REAL_LOG_PATH,
        ["timestamp", "symbol", "side", "qty", "entry_price", "exit_price", "gross_pnl", "fee_paid", "net_pnl", "reason"],
    )

    app = build_app()
    print("Mp ORBbot running...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
