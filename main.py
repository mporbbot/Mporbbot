# main.py
# Mp ORBbot – mock-version med "min" ORB-logik inbyggd.
# - ORB = första gröna candle efter röd
# - ENTRY_MODE: CLOSE / TICK (styr via /entry_mode)
# - SL = ORB-låg
# - Trailing stop efter +0.1 %
# - AI-läge: neutral / aggressive / cautious
# - /test_buy kör en ORB-simulering på fejkade 3m-candles
# - Kommandon: /status, /pnl, /engine_on, /engine_off, /mode, /timeframe,
#   /threshold, /risk, /reset_pnl, /close_all, /export_csv, /test_buy, /entry_mode

import csv
import logging
import random
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ----------------------------------------------------------------------
# TELEGRAM TOKEN (från Environment Variable)
# ----------------------------------------------------------------------

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not BOT_TOKEN or ":" not in BOT_TOKEN:
    raise ValueError(
        f"❌ Ogiltig TELEGRAM_TOKEN i Environment Variables: '{BOT_TOKEN}'.\n"
        "Lägg in korrekt token i DigitalOcean → Environment Variables."
    )

# ----------------------------------------------------------------------
# KONFIG
# ----------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

MOCK_TRADE_FILE = LOG_DIR / "mock_trade_log.csv"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TRADE_SIZE_USDT = 30.0
DEFAULT_FEE_RATE = 0.001  # 0.1% fee per köp/sälj

# ORB / trailing-parametrar
TRAIL_TRIGGER = 0.001    # +0.1 %
TRAIL_DISTANCE = 0.002   # 0.2 %

# ----------------------------------------------------------------------
# DATAMODELLER
# ----------------------------------------------------------------------

@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def color(self) -> str:
        if self.close > self.open:
            return "green"
        if self.close < self.open:
            return "red"
        return "doji"


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fees: float
    ai_mode: str
    entry_mode: str
    timeframe: str
    opened_at: str
    closed_at: str
    trade_type: str = "mock"


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    qty: float
    stop_loss: float
    best_price: float
    opened_at: str
    ai_mode: str
    entry_mode: str
    timeframe: str
    is_mock: bool = True
    trailing_active: bool = False


@dataclass
class SymbolState:
    symbol: str
    orb_active: bool = False
    orb_high: float = 0.0
    orb_low: float = 0.0
    last_candle_color: str = ""
    position: Optional[Position] = None


class BotState:
    def __init__(self) -> None:
        self.engine_on: bool = False
        self.ai_mode: str = "neutral"
        self.entry_mode: str = "CLOSE"  # CLOSE / TICK
        self.timeframe: str = "3m"
        self.threshold: float = 0.1      # kan användas senare i filter
        self.risk_percent: float = 1.0
        self.trade_size_usdt: float = DEFAULT_TRADE_SIZE_USDT

        self.mock_trades: List[Trade] = []
        self.mock_pnl: float = 0.0

    def reset_pnl(self) -> None:
        self.mock_trades.clear()
        self.mock_pnl = 0.0


STATE = BotState()

# ----------------------------------------------------------------------
# HJÄLPARE
# ----------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# LOGGNING
# ----------------------------------------------------------------------

def ensure_mock_log_header() -> None:
    if MOCK_TRADE_FILE.exists():
        return
    with MOCK_TRADE_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "symbol",
            "side",
            "entry_price",
            "exit_price",
            "qty",
            "pnl",
            "fees",
            "ai_mode",
            "entry_mode",
            "timeframe",
            "opened_at",
            "closed_at",
            "trade_type",
        ])


async def log_and_notify_trade(update: Update, context: ContextTypes.DEFAULT_TYPE, trade: Trade) -> None:
    ensure_mock_log_header()

    # skriv CSV
    with MOCK_TRADE_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            trade.symbol,
            trade.side,
            trade.entry_price,
            trade.exit_price,
            trade.qty,
            trade.pnl,
            trade.fees,
            trade.ai_mode,
            trade.entry_mode,
            trade.timeframe,
            trade.opened_at,
            trade.closed_at,
            trade.trade_type,
        ])

    # telegram-meddelande
    side_emoji = "🟢 LONG" if trade.side == "LONG" else "🔴 SHORT"

    msg = (
        f"📊 *Mock-ORB trade avslutad*\n"
        f"{side_emoji}\n\n"
        f"Symbol: *{trade.symbol}*\n"
        f"Entry: `{trade.entry_price}`\n"
        f"Exit: `{trade.exit_price}`\n\n"
        f"💰 PnL: `{trade.pnl:.4f}` USDT\n"
        f"💸 Avgifter: `{trade.fees:.4f}` USDT\n\n"
        f"🤖 AI-läge: `{trade.ai_mode}`\n"
        f"🎯 Entry-mode: `{trade.entry_mode}`\n"
        f"⏱ Timeframe: `{trade.timeframe}`\n\n"
        f"Öppnad: `{trade.opened_at}`\n"
        f"Stängd: `{trade.closed_at}`"
    )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode="Markdown"
    )

# ----------------------------------------------------------------------
# ORB-LOGIK (min design)
# ----------------------------------------------------------------------

def ai_filter(ai: str, c: Candle) -> bool:
    # enkel filter som använder candle-body
    body = abs(c.close - c.open)
    rng = max(c.high - c.low, 1e-8)
    ratio = body / rng

    if ai == "neutral":
        return True
    if ai == "aggressive":
        return ratio > 0.05  # släpper igenom de flesta
    if ai == "cautious":
        if c.color != "green":
            return False
        if ratio < 0.5:
            return False
        # kräver close i övre delen
        if c.close < c.low + 0.7 * (c.high - c.low):
            return False
        return True
    return True


def maybe_start_orb(state: SymbolState, c: Candle) -> None:
    # ORB = första gröna efter röd, ingen position
    if state.position:
        return
    if state.last_candle_color == "red" and c.color == "green":
        state.orb_active = True
        state.orb_high = c.high
        state.orb_low = c.low


def check_orb_entry(state: SymbolState, c: Candle) -> Optional[Position]:
    if not state.orb_active:
        return None
    if state.position:
        return None
    if not ai_filter(STATE.ai_mode, c):
        return None

    entry_price: Optional[float] = None

    if STATE.entry_mode == "CLOSE":
        if c.close > state.orb_high:
            entry_price = c.close
    elif STATE.entry_mode == "TICK":
        if c.high > state.orb_high:
            entry_price = state.orb_high

    if entry_price is None:
        return None

    qty = STATE.trade_size_usdt / entry_price

    pos = Position(
        symbol=state.symbol,
        side="LONG",
        entry_price=entry_price,
        qty=qty,
        stop_loss=state.orb_low,
        best_price=entry_price,
        opened_at=now_iso(),
        ai_mode=STATE.ai_mode,
        entry_mode=STATE.entry_mode,
        timeframe=STATE.timeframe,
    )

    # ORB är förbrukad när vi går in
    state.orb_active = False
    return pos


def check_orb_exit(pos: Position, c: Candle) -> Optional[float]:
    # Stop-loss
    if c.low <= pos.stop_loss:
        return pos.stop_loss

    # Trailing start ( +0.1 % )
    if not pos.trailing_active:
        if c.high >= pos.entry_price * (1 + TRAIL_TRIGGER):
            pos.trailing_active = True

    # Trailing stop
    if pos.trailing_active:
        if c.high > pos.best_price:
            pos.best_price = c.high
        trail_stop = pos.best_price * (1 - TRAIL_DISTANCE)
        if c.close <= trail_stop:
            return trail_stop

    return None


def process_candle_orb(state: SymbolState, c: Candle) -> List[Trade]:
    closed: List[Trade] = []

    # kolla exit först
    if state.position:
        pos = state.position
        ex = check_orb_exit(pos, c)
        if ex is not None:
            notional_entry = pos.entry_price * pos.qty
            notional_exit = ex * pos.qty
            gross_pnl = (ex - pos.entry_price) * pos.qty
            fees = (notional_entry + notional_exit) * DEFAULT_FEE_RATE
            pnl = gross_pnl - fees

            t = Trade(
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=ex,
                qty=pos.qty,
                pnl=pnl,
                fees=fees,
                ai_mode=pos.ai_mode,
                entry_mode=pos.entry_mode,
                timeframe=pos.timeframe,
                opened_at=pos.opened_at,
                closed_at=now_iso(),
                trade_type="mock",
            )
            closed.append(t)
            state.position = None

    # definiera ORB (röd -> grön)
    maybe_start_orb(state, c)

    # kolla entry
    if state.position is None:
        pos = check_orb_entry(state, c)
        if pos:
            state.position = pos

    state.last_candle_color = c.color
    return closed

# ----------------------------------------------------------------------
# MOCKDATA – FEJK 3M-CANDLES
# ----------------------------------------------------------------------

def generate_fake_candles(n: int = 200, base_price: float = 100.0) -> List[Candle]:
    candles: List[Candle] = []
    price = base_price
    ts = int(datetime.now(timezone.utc).timestamp()) * 1000

    for _ in range(n):
        ts += 3 * 60 * 1000  # 3 minuter
        drift = random.uniform(-0.005, 0.005)  # ±0.5 %
        new_price = price * (1 + drift)
        high = max(price, new_price) * (1 + random.uniform(0, 0.002))
        low = min(price, new_price) * (1 - random.uniform(0, 0.002))
        open_ = price
        close = new_price
        vol = random.uniform(10, 100)
        candles.append(Candle(ts=ts, open=open_, high=high, low=low, close=close, volume=vol))
        price = new_price

    return candles

# ----------------------------------------------------------------------
# ORB-MOCKTRADE VIA /test_buy
# ----------------------------------------------------------------------

def simulate_mock_trade_orb(symbol: str) -> Trade:
    """
    Kör min ORB-logik på fejkade 3m-candles och returnerar
    sista stängda trade (eller en fallback om ingen trade triggar).
    """
    candles = generate_fake_candles()
    state = SymbolState(symbol=symbol)
    closed_all: List[Trade] = []

    for c in candles:
        closed = process_candle_orb(state, c)
        closed_all.extend(closed)

    if closed_all:
        return closed_all[-1]

    # Fallback om ingen ORB trade triggar (ska vara ovanligt)
    price = candles[-1].close
    direction = "LONG"
    entry = price
    exit_ = price * random.uniform(0.99, 1.01)
    qty = STATE.trade_size_usdt / entry
    pnl_gross = (exit_ - entry) * qty
    fee = (entry * qty + exit_ * qty) * DEFAULT_FEE_RATE
    pnl = pnl_gross - fee
    now = now_iso()
    return Trade(
        symbol=symbol,
        side=direction,
        entry_price=round(entry, 4),
        exit_price=round(exit_, 4),
        qty=round(qty, 6),
        pnl=round(pnl, 4),
        fees=round(fee, 4),
        ai_mode=STATE.ai_mode,
        entry_mode=STATE.entry_mode,
        timeframe=STATE.timeframe,
        opened_at=now,
        closed_at=now,
    )

# ----------------------------------------------------------------------
# KOMMANDON
# ----------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hej! 👋 Detta är Mp ORBbot (mock) med inbyggd ORB-logik.\n\n"
        "ORB:\n"
        " • Första gröna efter röd = ORB\n"
        " • ENTRY_MODE: CLOSE/TICK (styr med /entry_mode)\n"
        " • SL = ORB-låg\n"
        " • Trailing efter +0.1%\n\n"
        "Testa med /test_buy (valfritt: /test_buy BTCUSDT)."
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"📊 *Status*\n\n"
        f"Engine: `{'ON' if STATE.engine_on else 'OFF'}`\n"
        f"AI-läge: `{STATE.ai_mode}`\n"
        f"Entry-mode: `{STATE.entry_mode}`\n"
        f"Timeframe: `{STATE.timeframe}`\n"
        f"Threshold: `{STATE.threshold}`\n"
        f"Risk: `{STATE.risk_percent}%`\n"
        f"Trade-size: `{STATE.trade_size_usdt}` USDT\n\n"
        f"Mock-trades: `{len(STATE.mock_trades)}`\n"
        f"Total PnL: `{STATE.mock_pnl:.4f}` USDT"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📈 Avslutade mock-trades: {len(STATE.mock_trades)}\n"
        f"Total PnL: {STATE.mock_pnl:.4f} USDT"
    )


async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = True
    await update.message.reply_text("▶️ Engine flagga ON (används för framtida live).")


async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = False
    await update.message.reply_text("⛔ Engine flagga OFF.")


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Nuvarande AI-läge: {STATE.ai_mode}")
        return

    STATE.ai_mode = context.args[0].lower()
    await update.message.reply_text(f"AI-läge satt till: {STATE.ai_mode}")


async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Nuvarande timeframe: {STATE.timeframe}")
        return

    STATE.timeframe = context.args[0]
    await update.message.reply_text(f"Timeframe satt till {STATE.timeframe}.")


async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Nuvarande threshold: {STATE.threshold}")
        return
    try:
        value = float(context.args[0])
    except ValueError:
        await update.message.reply_text("Fel format. Använd t.ex.: /threshold 0.2")
        return
    STATE.threshold = value
    await update.message.reply_text(f"Threshold satt till {STATE.threshold}")


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Nuvarande risk: {STATE.risk_percent} %")
        return
    try:
        value = float(context.args[0])
    except ValueError:
        await update.message.reply_text("Fel format. Använd t.ex.: /risk 1.5")
        return
    STATE.risk_percent = value
    await update.message.reply_text(f"Risk satt till {STATE.risk_percent} %")


async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /entry_mode CLOSE
    /entry_mode TICK
    eller /entry_mode för att se nuvarande.
    """
    if not context.args:
        await update.message.reply_text(
            f"Nuvarande ENTRY_MODE: {STATE.entry_mode}\n"
            "Använd: /entry_mode CLOSE eller /entry_mode TICK"
        )
        return

    mode = context.args[0].upper()
    if mode not in ["CLOSE", "TICK"]:
        await update.message.reply_text("Ogiltigt läge. Använd: CLOSE eller TICK.")
        return

    STATE.entry_mode = mode
    await update.message.reply_text(f"ENTRY_MODE satt till: {STATE.entry_mode}")


async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.reset_pnl()
    await update.message.reply_text("PnL återställd.")


async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Inga öppna mock-positioner att stänga (allt är candle-simulerat).")


async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MOCK_TRADE_FILE.exists():
        await update.message.reply_text("Ingen mock_trade_log.csv ännu.")
        return

    await update.message.reply_document(
        document=InputFile(MOCK_TRADE_FILE.open("rb"), filename="mock_trade_log.csv"),
        caption="mock_trade_log.csv"
    )


async def cmd_test_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = DEFAULT_SYMBOL
    if context.args:
        symbol = context.args[0].upper()

    trade = simulate_mock_trade_orb(symbol)
    STATE.mock_trades.append(trade)
    STATE.mock_pnl += trade.pnl

    await log_and_notify_trade(update, context, trade)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # REGISTER COMMANDS (matchar knapparna i din meny)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("test_buy", cmd_test_buy))

    app.run_polling()


if __name__ == "__main__":
    main()
