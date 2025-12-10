# main.py
# Mp ORBbot – stabil mock-version (ingen job_queue, fungerar på DigitalOcean)
# - Alla kommandon enligt din meny
# - Mocktrades + logg
# - Entry/Exit-meddelanden
# - Token via TELEGRAM_TOKEN environment variable

import csv
import logging
import random
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

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

# ----------------------------------------------------------------------
# DATAKLASSER + STATE
# ----------------------------------------------------------------------

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


class BotState:
    def __init__(self) -> None:
        self.engine_on: bool = False
        self.ai_mode: str = "neutral"
        self.entry_mode: str = "retest"
        self.timeframe: str = "3m"
        self.threshold: float = 0.1
        self.risk_percent: float = 1.0
        self.trade_size_usdt: float = DEFAULT_TRADE_SIZE_USDT

        self.mock_trades: List[Trade] = []
        self.mock_pnl: float = 0.0

    def reset_pnl(self) -> None:
        self.mock_trades.clear()
        self.mock_pnl = 0.0


STATE = BotState()

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
        f"📊 *Mock-trade avslutad*\n"
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
# MOCKTRADE
# ----------------------------------------------------------------------

def simulate_mock_trade(symbol: str) -> Trade:
    price = random.uniform(80, 120)
    direction = random.choice(["LONG", "SHORT"])

    if direction == "LONG":
        entry = price
        exit_ = price * random.uniform(0.98, 1.02)
    else:
        exit_ = price
        entry = price * random.uniform(0.98, 1.02)

    qty = STATE.trade_size_usdt / entry

    if direction == "LONG":
        pnl_gross = (exit_ - entry) * qty
    else:
        pnl_gross = (entry - exit_) * qty

    fee = (entry * qty + exit_ * qty) * DEFAULT_FEE_RATE
    pnl = pnl_gross - fee

    now = datetime.utcnow().isoformat()

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
        "Hej! 👋 Detta är Mp ORBbot (mock)."
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
        f"📈 Avslutade trades: {len(STATE.mock_trades)}\n"
        f"Total PnL: {STATE.mock_pnl:.4f} USDT"
    )


async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = True
    await update.message.reply_text("▶️ Engine flagga ON.")


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


async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.reset_pnl()
    await update.message.reply_text("PnL återställd.")


async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Inga öppna mock-positioner att stänga.")


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

    trade = simulate_mock_trade(symbol)
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

    # REGISTER COMMANDS
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("test_buy", cmd_test_buy))

    app.run_polling()


if __name__ == "__main__":
    main()
