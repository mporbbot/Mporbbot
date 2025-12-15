# main.py
# Mp ORBbot - STABLE Telegram bot with ENTRY/EXIT notifications
# IMPORTANT:
#   - Uses TELEGRAM_TOKEN (DigitalOcean env var)
#   - Does NOT use job_queue (avoids NoneType.run_repeating crash)
#   - Adds trade notifications ONLY (ENTRY/EXIT) + /notify on|off
#
# Requirements:
#   pip install python-telegram-bot==20.7
#
# Env:
#   TELEGRAM_TOKEN="xxxxx"

import os
import json
import csv
import time
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

DEFAULT_COINS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
ALLOWED_TF = ["1m", "3m", "5m", "15m"]
ALLOWED_MODES = ["aggressive", "neutral", "cautious"]

# Threshold controls
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.05

# Risk controls
RISK_MIN = 0.1
RISK_MAX = 5.0

# Mock engine tick seconds
ENGINE_TICK_SECONDS = 3

# Mock trade size
DEFAULT_MOCK_USDT = 30.0

# Fee (0.1%)
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


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def pretty_symbol(sym: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    s = sym.upper().replace("-", "")
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s


def fmt_price(sym: str, p: float) -> str:
    # Keep it readable like your screenshot
    if sym.upper().startswith(("BTC", "ETH")):
        return f"{p:.4f}" if p < 1000 else f"{p:.4f}".rstrip("0").rstrip(".")
    # alts often small
    if p < 1:
        return f"{p:.4f}"
    return f"{p:.4f}".rstrip("0").rstrip(".")


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

    # NEW: notifications
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

        # NEW: safe defaults
        try:
            cfg.notify_chat_id = int(cfg.notify_chat_id or 0)
        except Exception:
            cfg.notify_chat_id = 0
        cfg.notify_trades = bool(cfg.notify_trades) if cfg.notify_trades is not None else True

        return cfg


CFG: BotConfig = None

# Positions: symbol -> (entry_price, qty, entry_ts)
POSITIONS: Dict[str, Tuple[float, float, int]] = {}

# Mock prices
MOCK_PRICE: Dict[str, float] = {
    "BTCUSDT": 65000.0,
    "ETHUSDT": 3500.0,
    "XRPUSDT": 0.60,
    "ADAUSDT": 0.45,
    "LINKUSDT": 14.0,
}


def load_config() -> BotConfig:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return BotConfig.from_json(json.load(f))
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

    rows, row = [], []
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
    row = []
    for tf in ALLOWED_TF:
        label = tf + (" ✅" if tf == current else "")
        row.append(InlineKeyboardButton(label, callback_data=f"tf:{tf}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="tf:close")]])


def mode_buttons(current: str) -> InlineKeyboardMarkup:
    row = []
    for m in ALLOWED_MODES:
        label = m + (" ✅" if m == current else "")
        row.append(InlineKeyboardButton(label, callback_data=f"aim:{m}"))
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("Stäng", callback_data="aim:close")]])


def risk_buttons(current: float) -> InlineKeyboardMarkup:
    presets = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    rows, row = [], []
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
# Notifications (NEW)
# -----------------------------
async def notify_trade(app: Application, text: str):
    # Sends trade feed messages like your screenshot
    try:
        if not CFG.notify_trades:
            return
        if not CFG.notify_chat_id:
            return
        await app.bot.send_message(chat_id=CFG.notify_chat_id, text=text)
    except Exception:
        # Never crash the bot because of notifications
        pass


# -----------------------------
# Logging
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
# Mock engine
# -----------------------------
def mock_next_price(symbol: str) -> float:
    p = MOCK_PRICE.get(symbol, 1.0)
    vol = 0.0012 if symbol in ("BTCUSDT", "ETHUSDT") else 0.0020
    p = max(1e-8, p * (1.0 + random.gauss(0, vol)))
    MOCK_PRICE[symbol] = p
    return p


def should_enter_trade(cfg: BotConfig) -> bool:
    base = 0.10
    if cfg.ai_mode == "aggressive":
        base *= 1.30
    elif cfg.ai_mode == "cautious":
        base *= 0.70

    t = clamp(cfg.threshold, THRESHOLD_MIN, THRESHOLD_MAX)
    mult = 1.4 - (t - THRESHOLD_MIN) * (1.2 / (THRESHOLD_MAX - THRESHOLD_MIN))
    prob = clamp(base * mult, 0.01, 0.25)
    return random.random() < prob


def open_position(symbol: str, cfg: BotConfig) -> None:
    if symbol in POSITIONS:
        return
    price = MOCK_PRICE[symbol]
    qty = DEFAULT_MOCK_USDT / price
    POSITIONS[symbol] = (price, qty, now_ts())


def maybe_close_position(symbol: str, cfg: BotConfig) -> Optional[float]:
    entry, qty, ets = POSITIONS[symbol]
    price = MOCK_PRICE[symbol]
    change = (price - entry) / entry * 100.0

    if change >= cfg.trail_trigger_pct and random.random() < 0.35:
        return price
    if change <= -max(0.2, cfg.risk_pct * 0.6) and random.random() < 0.50:
        return price
    if random.random() < 0.05:
        return price
    return None


def close_position(symbol: str, exit_price: float, cfg: BotConfig, reason: str) -> float:
    entry, qty, ets = POSITIONS.pop(symbol)
    gross = (exit_price - entry) * qty
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


async def engine_background_loop(app: Application):
    """Runs forever; does nothing unless engine_on = True."""
    global CFG
    while True:
        try:
            if CFG and CFG.engine_on:
                for sym in CFG.active_coins:
                    mock_next_price(sym)

                # entries
                for sym in CFG.active_coins:
                    if sym not in POSITIONS and should_enter_trade(CFG):
                        # open
                        open_position(sym, CFG)

                        # NEW: notify ENTRY (as in your screenshot)
                        entry_price = POSITIONS[sym][0]
                        # keep a "score" look (mock score from threshold/mode)
                        score = 1.0 + max(0.0, (0.9 - CFG.threshold)) * (1.0 if CFG.ai_mode != "cautious" else 0.7)
                        text = f"🟢 MOMO ENTRY {pretty_symbol(sym)} LONG @ {fmt_price(sym, entry_price)} | score={score:.2f}"
                        asyncio.create_task(notify_trade(app, text))

                # exits
                for sym in list(POSITIONS.keys()):
                    exit_price = maybe_close_position(sym, CFG)
                    if exit_price is not None:
                        net = close_position(sym, exit_price, CFG, reason="mock_exit")

                        # NEW: notify EXIT (as in your screenshot)
                        # choose label like "TP EXIT" even though it's mock exit logic
                        text = f"🎯 TP EXIT {pretty_symbol(sym)} @ {fmt_price(sym, exit_price)} | Net {net:+.4f} USDT"
                        asyncio.create_task(notify_trade(app, text))

        except Exception:
            # never crash bot from engine
            pass

        await asyncio.sleep(ENGINE_TICK_SECONDS)


# -----------------------------
# Telegram commands
# -----------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # NEW: save chat id automatically for notifications
    global CFG
    try:
        CFG.notify_chat_id = int(update.effective_chat.id)
        save_config()
    except Exception:
        pass

    await update.message.reply_text(
        "Mp ORBbot är igång.\n\n"
        "✅ Du kommer nu få ENTRY/EXIT-notiser i denna chat.\n"
        "• /notify off (stäng av)\n"
        "• /notify on  (slå på)\n"
        "• /threshold 0.6\n"
        "• /status\n",
        reply_markup=main_menu_keyboard(),
    )


async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    args = context.args if hasattr(context, "args") else []
    if not args:
        await update.message.reply_text(f"notify_trades: {CFG.notify_trades}\nAnvänd: /notify on eller /notify off")
        return

    v = args[0].lower().strip()
    if v in ("on", "yes", "1", "true"):
        CFG.notify_trades = True
        # also set chat id to this chat
        try:
            CFG.notify_chat_id = int(update.effective_chat.id)
        except Exception:
            pass
        save_config()
        await update.message.reply_text("✅ Notiser PÅ (ENTRY/EXIT).")
        return

    if v in ("off", "no", "0", "false"):
        CFG.notify_trades = False
        save_config()
        await update.message.reply_text("⛔ Notiser AV.")
        return

    await update.message.reply_text("Ogiltigt. Använd: /notify on eller /notify off")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    msg = (
        f"ENGINE: {'ON' if CFG.engine_on else 'OFF'}\n"
        f"Aktiva coins: {CFG.active_coins}\n"
        f"Trades: {CFG.trades}\n"
        f"Total NETTO-PNL: {fmt4(CFG.total_net_pnl)} USDT\n"
        f"AI-läge: {CFG.ai_mode}\n"
        f"Timeframe: {CFG.timeframe}\n"
        f"Risk: {CFG.risk_pct}%\n"
        f"Threshold: {CFG.threshold}\n"
        f"Trail trig: {CFG.trail_trigger_pct:.2f}%\n"
        f"Trail dist: {CFG.trail_dist_pct:.2f}%\n"
        f"Mode: {CFG.mode}\n"
        f"Notiser: {'ON' if CFG.notify_trades else 'OFF'}\n"
        f"Öppna positioner: {list(POSITIONS.keys()) if POSITIONS else 'Inga'}"
    )
    await update.message.reply_text(msg, reply_markup=main_menu_keyboard())


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    await update.message.reply_text(f"Total NETTO-PNL: {fmt4(CFG.total_net_pnl)} USDT\nTrades: {CFG.trades}")


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

    total = 0.0
    for sym in list(POSITIONS.keys()):
        total += close_position(sym, MOCK_PRICE.get(sym, 1.0), CFG, reason="close_all")

    await update.message.reply_text(f"✅ Stängde alla positioner. Netto: {fmt4(total)} USDT")


async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "CSV-loggar finns på servern:\n"
        f"• Mock: {MOCK_LOG_PATH}\n"
        f"• Real: {REAL_LOG_PATH}"
    )


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
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
        CFG.risk_pct = float(round(clamp(val, RISK_MIN, RISK_MAX), 4))
        save_config()
        await update.message.reply_text(f"✅ Risk satt till: {CFG.risk_pct}%")
        return

    await update.message.reply_text(
        f"Risk nu: {CFG.risk_pct}%\nVälj eller skriv /risk 0.7",
        reply_markup=risk_buttons(CFG.risk_pct),
    )


async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    args = context.args if hasattr(context, "args") else []
    if args:
        val = parse_float(args[0])
        if val is None:
            await update.message.reply_text("Ogiltigt värde. Ex: /threshold 0.6")
            return
        CFG.threshold = float(round(clamp(val, THRESHOLD_MIN, THRESHOLD_MAX), 4))
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
    global CFG
    args = context.args if hasattr(context, "args") else []
    sym = (args[0].upper() if args else "BTCUSDT")
    if sym not in CFG.active_coins:
        await update.message.reply_text(f"{sym} är inte aktiv. Aktiva: {CFG.active_coins}")
        return

    # quick open+close for testing logging
    mock_next_price(sym)
    if sym not in POSITIONS:
        open_position(sym, CFG)
        entry_price = POSITIONS[sym][0]
        asyncio.create_task(
            notify_trade(
                context.application,
                f"🟢 MOMO ENTRY {pretty_symbol(sym)} LONG @ {fmt_price(sym, entry_price)} | score=1.00",
            )
        )

    mock_next_price(sym)
    net = close_position(sym, MOCK_PRICE[sym], CFG, reason="test_buy")
    asyncio.create_task(
        notify_trade(
            context.application,
            f"🎯 TP EXIT {pretty_symbol(sym)} @ {fmt_price(sym, MOCK_PRICE[sym])} | Net {net:+.4f} USDT",
        )
    )
    await update.message.reply_text(f"✅ Test-trade klar i {sym}. Netto: {fmt4(net)} USDT")


# -----------------------------
# Callbacks
# -----------------------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CFG
    q = update.callback_query
    data = q.data or ""
    await q.answer()

    if data.startswith("thr:"):
        payload = data.split("thr:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        val = parse_float(payload)
        if val is None:
            await q.edit_message_text("Ogiltigt threshold.")
            return
        CFG.threshold = float(round(clamp(val, THRESHOLD_MIN, THRESHOLD_MAX), 4))
        save_config()
        await q.edit_message_text(f"✅ Threshold satt till: {CFG.threshold}")
        return

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

    if data.startswith("risk:"):
        payload = data.split("risk:", 1)[1]
        if payload == "close":
            await q.edit_message_text("✅ Stängt.")
            return
        val = parse_float(payload)
        if val is None:
            await q.edit_message_text("Ogiltigt risk-värde.")
            return
        CFG.risk_pct = float(round(clamp(val, RISK_MIN, RISK_MAX), 4))
        save_config()
        await q.edit_message_text(f"✅ Risk satt till: {CFG.risk_pct}%")
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

    # commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("notify", cmd_notify))  # NEW
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

    # callbacks
    app.add_handler(CallbackQueryHandler(on_callback))

    return app


async def post_init(app: Application):
    # Start background engine loop safely (no job_queue)
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

    app = build_app()
    app.post_init = post_init
    print("Mp ORBbot running...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
