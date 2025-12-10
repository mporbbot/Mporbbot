# main.py
# Mp ORBbot – stabil mock-version med:
#  - Telegram-kommandon enligt din meny (/status, /pnl, /engine_on, /engine_off, osv)
#  - Mocktrades via /test_buy (30 USDT styck)
#  - Loggning till logs/mock_trade_log.csv
#  - Entry & Exit skickas i Telegram för varje trade

import csv
import logging
import random
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
# KONFIGURATION
# ----------------------------------------------------------------------

BOT_TOKEN = "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

MOCK_TRADE_FILE = LOG_DIR / "mock_trade_log.csv"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TRADE_SIZE_USDT = 30.0        # varje mocktrade använder 30 USDT
DEFAULT_FEE_RATE = 0.001              # 0.1 % per köp/sälj

# ----------------------------------------------------------------------
# DATAKLASSER & STATE
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
    trade_type: str = "mock"  # mock / real


class BotState:
    def __init__(self) -> None:
        self.engine_on: bool = False
        self.ai_mode: str = "neutral"     # neutral / aggressive / cautious etc
        self.entry_mode: str = "retest"   # t.ex. tick / close / retest
        self.timeframe: str = "3m"
        self.threshold: float = 0.1       # ex. 0.1 %
        self.risk_percent: float = 1.0
        self.trade_size_usdt: float = DEFAULT_TRADE_SIZE_USDT

        self.mock_trades: List[Trade] = []
        self.mock_pnl: float = 0.0

    def reset_pnl(self) -> None:
        self.mock_trades.clear()
        self.mock_pnl = 0.0


STATE = BotState()

# ----------------------------------------------------------------------
# LOGGNING & TRADE-MEDDELANDEN
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


async def log_and_notify_trade(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    trade: Trade,
) -> None:
    """
    Loggar trade till CSV och skickar Telegram-meddelande.
    Anropas ALLTID när en mocktrade avslutas.
    """

    ensure_mock_log_header()

    # --- Logga till CSV ---
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

    # --- Telegram-meddelande ---
    chat_id = update.effective_chat.id
    side_emoji = "🟢 LONG" if trade.side.upper() == "LONG" else "🔴 SHORT"
    trade_type_prefix = "📊 Mock-trade"

    text = (
        f"{trade_type_prefix} avslutad\n"
        f"{side_emoji}\n"
        f"Symbol: *{trade.symbol}*\n\n"
        f"➡️ Entry: `{trade.entry_price}`\n"
        f"⬅️ Exit: `{trade.exit_price}`\n\n"
        f"📉 PnL: `{trade.pnl:.4f}` USDT\n"
        f"💸 Avgifter: `{trade.fees:.4f}` USDT\n\n"
        f"🤖 AI-läge: `{trade.ai_mode}`\n"
        f"🎯 Entry-mode: `{trade.entry_mode}`\n"
        f"⏱ Timeframe: `{trade.timeframe}`\n\n"
        f"🕒 Öppnad: `{trade.opened_at}`\n"
        f"🕤 Stängd: `{trade.closed_at}`"
    )

    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode="Markdown",
    )

# ----------------------------------------------------------------------
# MOCK ENGINE – EN ENKEL TEST-TRADE
# ----------------------------------------------------------------------

def simulate_mock_trade(symbol: str) -> Trade:
    """
    En enkel mocktrade:
    - Använder fixed trade-size i USDT (30 USDT som standard)
    - Skapar ett påhittat entry/exit-pris
    - Beräknar PnL + avgifter
    """

    # Simulerat pris kring 100
    base_price = random.uniform(80, 120)
    direction = random.choice(["LONG", "SHORT"])

    if direction == "LONG":
        entry_price = base_price
        exit_price = base_price * random.uniform(0.98, 1.02)
    else:
        exit_price = base_price
        entry_price = base_price * random.uniform(0.98, 1.02)

    qty = STATE.trade_size_usdt / entry_price

    if direction == "LONG":
        gross_pnl = (exit_price - entry_price) * qty
    else:
        gross_pnl = (entry_price - exit_price) * qty

    notional_entry = entry_price * qty
    notional_exit = exit_price * qty
    fees = (notional_entry + notional_exit) * DEFAULT_FEE_RATE

    net_pnl = gross_pnl - fees

    opened_at = datetime.utcnow().isoformat()
    closed_at = datetime.utcnow().isoformat()

    trade = Trade(
        symbol=symbol.upper(),
        side=direction,
        entry_price=round(entry_price, 4),
        exit_price=round(exit_price, 4),
        qty=round(qty, 6),
        pnl=round(net_pnl, 4),
        fees=round(fees, 4),
        ai_mode=STATE.ai_mode,
        entry_mode=STATE.entry_mode,
        timeframe=STATE.timeframe,
        opened_at=opened_at,
        closed_at=closed_at,
        trade_type="mock",
    )

    return trade

# ----------------------------------------------------------------------
# KOMMANDON (matchar din meny)
# ----------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Hej! 👋\n"
        "Det här är Mp ORBbot (mock-version).\n\n"
        "Kommandon:\n"
        "• /status – visa läge\n"
        "• /pnl – visa mock-PnL\n"
        "• /engine_on – slå på engine-flagga (för framtida live)\n"
        "• /engine_off – slå av engine-flagga\n"
        "• /timeframe – visa/sätt timeframe\n"
        "• /threshold – visa/sätt threshold\n"
        "• /risk – visa/sätt risk %\n"
        "• /mode – visa/sätt AI-läge\n"
        "• /test_buy – skapa en mocktrade (30 USDT)\n"
        "• /export_csv – hämta mock_trade_log.csv\n"
        "• /reset_pnl – nollställ PnL\n"
        "• /close_all – (mock) stäng alla positioner\n"
    )
    await update.message.reply_text(text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📊 *Status*\n\n"
        f"Engine: `{'ON' if STATE.engine_on else 'OFF'}`\n"
        f"AI-läge: `{STATE.ai_mode}`\n"
        f"Entry-mode: `{STATE.entry_mode}`\n"
        f"Timeframe: `{STATE.timeframe}`\n"
        f"Threshold: `{STATE.threshold}`\n"
        f"Risk: `{STATE.risk_percent}%`\n"
        f"Trade-size: `{STATE.trade_size_usdt} USDT`\n\n"
        f"Mock-trades: `{len(STATE.mock_trades)}`\n"
        f"Mock-PnL: `{STATE.mock_pnl:.4f}` USDT"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📈 *Mock-PnL*\n"
        f"Avslutade trades: {len(STATE.mock_trades)}\n"
        f"Total PnL: {STATE.mock_pnl:.4f} USDT"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.engine_on = True
    await update.message.reply_text("▶️ Engine ON (flagga satt – används i framtida live-version).")


async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.engine_on = False
    await update.message.reply_text("⛔ Engine OFF.")


async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        STATE.timeframe = context.args[0]
        await update.message.reply_text(f"⏱ Timeframe satt till {STATE.timeframe}.")
    else:
        await update.message.reply_text(f"Nuvarande timeframe: {STATE.timeframe}")


async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        try:
            value = float(context.args[0])
            STATE.threshold = value
            await update.message.reply_text(f"Threshold satt till {STATE.threshold}.")
        except ValueError:
            await update.message.reply_text("Använd: /threshold 0.1")
    else:
        await update.message.reply_text(f"Nuvarande threshold: {STATE.threshold}")


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        try:
            value = float(context.args[0])
            STATE.risk_percent = value
            await update.message.reply_text(f"Risk satt till {STATE.risk_percent} %.")
        except ValueError:
            await update.message.reply_text("Använd: /risk 1.0")
    else:
        await update.message.reply_text(f"Nuvarande risk: {STATE.risk_percent} %")


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            f"Nuvarande AI-läge: {STATE.ai_mode}\n"
            f"Använd: /mode neutral | aggressive | cautious"
        )
        return

    mode = context.args[0].lower()
    STATE.ai_mode = mode
    await update.message.reply_text(f"🤖 AI-läge satt till: {STATE.ai_mode}")


async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.reset_pnl()
    await update.message.reply_text("PnL och mock-tradelista återställd.")


async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # I denna mock-version finns inga riktiga öppna trades.
    await update.message.reply_text("Inga öppna mock-positioner att stänga (allt är redan avslutat).")


async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not MOCK_TRADE_FILE.exists():
        await update.message.reply_text("Ingen mock_trade_log.csv hittades ännu.")
        return

    await update.message.reply_document(
        document=InputFile(MOCK_TRADE_FILE.open("rb"), filename="mock_trade_log.csv"),
        caption="Här är din mock_trade_log.csv 📄",
    )


async def cmd_test_buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Skapar en mocktrade, loggar den, uppdaterar PnL och skickar meddelande
    med ENTRY & EXIT.
    """
    symbol = DEFAULT_SYMBOL
    if context.args:
        symbol = context.args[0].upper()

    trade = simulate_mock_trade(symbol)
    STATE.mock_trades.append(trade)
    STATE.mock_pnl += trade.pnl

    await log_and_notify_trade(update, context, trade)

# ----------------------------------------------------------------------
# HUVUDFUNKTION (ingen asyncio.run, ingen JobQueue)
# ----------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Kommandon (matchar din meny)
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("pnl", cmd_pnl))
    application.add_handler(CommandHandler("engine_on", cmd_engine_on))
    application.add_handler(CommandHandler("engine_off", cmd_engine_off))
    application.add_handler(CommandHandler("timeframe", cmd_timeframe))
    application.add_handler(CommandHandler("threshold", cmd_threshold))
    application.add_handler(CommandHandler("risk", cmd_risk))
    application.add_handler(CommandHandler("export_csv", cmd_export_csv))
    application.add_handler(CommandHandler("close_all", cmd_close_all))
    application.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    application.add_handler(CommandHandler("mode", cmd_mode))
    application.add_handler(CommandHandler("test_buy", cmd_test_buy))

    # Telegram sköter sin egen event loop
    application.run_polling()


if __name__ == "__main__":
    main()
