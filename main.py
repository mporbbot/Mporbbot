# =======================
# Mp ORBbot – Ny stabil main.py
# =======================

import asyncio
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

# ----------------------------------------------------------------------
# KONFIGURATION
# ----------------------------------------------------------------------

BOT_TOKEN = "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "LINKUSDT", "XRPUSDT", "ADAUSDT"]
TIMEFRAME = "3m"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

MOCK_LOG_FILE = LOG_DIR / "mock_trade_log.csv"
REAL_LOG_FILE = LOG_DIR / "real_trade_log.csv"

FEE_RATE = 0.001            # 0.1 %
DEFAULT_TRADE_SIZE = 30.0   # mock trade size
TRAIL_TRIGGER = 0.001       # 0.1%
TRAIL_DISTANCE = 0.002      # 0.2%

WAIT_MOCK_CONFIRM, WAIT_LIVE_CONFIRM = range(2)

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
    def color(self):
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
    is_mock: bool


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
    is_mock: bool
    trailing_active: bool = False


@dataclass
class SymbolState:
    symbol: str
    orb_active: bool = False
    orb_high: float = 0
    orb_low: float = 0
    last_candle_color: str = ""
    position: Optional[Position] = None


class BotState:
    def __init__(self):
        self.engine_running = False
        self.mock_mode = True
        self.ai_mode = "neutral"
        self.entry_mode = "CLOSE"
        self.trade_size_usdt = DEFAULT_TRADE_SIZE
        self.symbol_states: Dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in SYMBOLS
        }
        self.mock_trades: List[Trade] = []
        self.real_trades: List[Trade] = []

    @property
    def mock_pnl(self):
        return sum(t.pnl for t in self.mock_trades)

    @property
    def real_pnl(self):
        return sum(t.pnl for t in self.real_trades)


STATE = BotState()

# ----------------------------------------------------------------------
# HJÄLPFUNKTIONER
# ----------------------------------------------------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def ensure_log_header(path: Path):
    if path.exists():
        return
    with path.open("w", newline="") as f:
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
            "type",
        ])


async def log_and_notify_trade(update: Update, context: ContextTypes.DEFAULT_TYPE, trade: Trade):
    log_file = MOCK_LOG_FILE if trade.is_mock else REAL_LOG_FILE
    ensure_log_header(log_file)

    with log_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now_iso(),
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
            "mock" if trade.is_mock else "real",
        ])

    chat_id = update.effective_chat.id
    msg = (
        f"📊 Trade avslutad\n"
        f"🟢 LONG {trade.symbol}\n\n"
        f"Entry: `{trade.entry_price}`\n"
        f"Exit: `{trade.exit_price}`\n"
        f"PnL: `{trade.pnl:.4f}`\n"
        f"Avgifter: `{trade.fees:.4f}`\n\n"
        f"AI: `{trade.ai_mode}`\n"
        f"Entry-mode: `{trade.entry_mode}`\n"
        f"TF: `{trade.timeframe}`"
    )

    await context.bot.send_message(chat_id, msg, parse_mode="Markdown")

# ----------------------------------------------------------------------
# ORB-LOGIK
# ----------------------------------------------------------------------

def maybe_start_orb(state: SymbolState, candle: Candle):
    if state.position:
        return
    if state.last_candle_color == "red" and candle.color == "green":
        state.orb_active = True
        state.orb_high = candle.high
        state.orb_low = candle.low


def ai_filter(ai: str, c: Candle):
    if ai == "neutral":
        return True
    body = abs(c.close - c.open)
    rng = max(c.high - c.low, 1e-8)
    ratio = body / rng
    if ai == "aggressive":
        return ratio > 0.05
    if ai == "cautious":
        return c.color == "green" and ratio > 0.5
    return True


def check_entry(state: SymbolState, c: Candle):
    if not state.orb_active:
        return None
    if state.position:
        return None
    if not ai_filter(STATE.ai_mode, c):
        return None

    entry = None
    if STATE.entry_mode == "CLOSE" and c.close > state.orb_high:
        entry = c.close
    elif STATE.entry_mode == "TICK" and c.high > state.orb_high:
        entry = state.orb_high

    if not entry:
        return None

    qty = STATE.trade_size_usdt / entry

    pos = Position(
        symbol=state.symbol,
        side="LONG",
        entry_price=entry,
        qty=qty,
        stop_loss=state.orb_low,
        best_price=entry,
        opened_at=now_iso(),
        ai_mode=STATE.ai_mode,
        entry_mode=STATE.entry_mode,
        timeframe=TIMEFRAME,
        is_mock=STATE.mock_mode,
    )

    state.orb_active = False
    return pos


def check_exit(pos: Position, c: Candle):
    # Stop-loss
    if c.low <= pos.stop_loss:
        return pos.stop_loss

    # Trailing trigger
    if not pos.trailing_active:
        if c.high >= pos.entry_price * (1 + TRAIL_TRIGGER):
            pos.trailing_active = True

    if pos.trailing_active:
        pos.best_price = max(pos.best_price, c.high)
        trail_stop = pos.best_price * (1 - TRAIL_DISTANCE)
        if c.close <= trail_stop:
            return trail_stop

    return None


def process_candle(state: SymbolState, c: Candle):
    results = []

    # EXIT
    if state.position:
        ex = check_exit(state.position, c)
        if ex:
            pos = state.position
            gross = (ex - pos.entry_price) * pos.qty
            fees = (pos.entry_price * pos.qty + ex * pos.qty) * FEE_RATE
            pnl = gross - fees

            trade = Trade(
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
                is_mock=pos.is_mock,
            )

            results.append(trade)
            state.position = None

    # ORB
    maybe_start_orb(state, c)

    # ENTRY
    if not state.position:
        pos = check_entry(state, c)
        if pos:
            state.position = pos

    state.last_candle_color = c.color
    return results

# ----------------------------------------------------------------------
# TESTDATA / MOCK
# ----------------------------------------------------------------------

def fake_candles(n=120, base=100):
    import random
    out = []
    p = base
    ts = int(datetime.now(timezone.utc).timestamp()) * 1000
    for _ in range(n):
        ts += 180000
        drift = p * (1 + (random.uniform(-0.005, 0.005)))
        high = max(p, drift) * 1.001
        low = min(p, drift) * 0.999
        out.append(Candle(ts, p, high, low, drift, 10))
        p = drift
    return out

# ----------------------------------------------------------------------
# TELEGRAM – KOMMANDON
# ----------------------------------------------------------------------

async def cmd_start(update: Update, ctx):
    await update.message.reply_text(
        "Mp ORBbot\nNy ORB-logik aktiv.\n\n"
        "/start_mock\n/start_live\n/mock_trade BTCUSDT\n/status\n/pnl\n/export_csv"
    )

async def cmd_status(update: Update, ctx):
    msg = (
        f"Engine: {STATE.engine_running}\n"
        f"Mock: {STATE.mock_mode}\n"
        f"AI: {STATE.ai_mode}\n"
        f"Entry: {STATE.entry_mode}\n"
        f"Mock-PnL: {STATE.mock_pnl:.4f}"
    )
    await update.message.reply_text(msg)

async def cmd_set_ai(update: Update, ctx):
    if not ctx.args:
        return await update.message.reply_text("Använd: /set_ai neutral|aggressive|cautious")
    m = ctx.args[0].lower()
    if m not in ["neutral", "aggressive", "cautious"]:
        return await update.message.reply_text("Ogiltigt läge.")
    STATE.ai_mode = m
    await update.message.reply_text(f"AI-läge satt: {m}")

async def cmd_set_entry_mode(update: Update, ctx):
    if not ctx.args:
        return await update.message.reply_text("Använd: /set_entry_mode CLOSE|TICK")
    m = ctx.args[0].upper()
    if m not in ["CLOSE", "TICK"]:
        return await update.message.reply_text("Ogiltigt mode.")
    STATE.entry_mode = m
    await update.message.reply_text(f"Entry-mode: {m}")

# ---- START MOCK (JA-bekräftelse) ----

async def cmd_start_mock(update: Update, ctx):
    STATE.mock_mode = True
    await update.message.reply_text("Starta mock? Skriv JA eller /cancel")
    return WAIT_MOCK_CONFIRM

async def confirm_mock(update: Update, ctx):
    if update.message.text.upper() == "JA":
        STATE.engine_running = True
        await update.message.reply_text("Mock-engine på.")
    else:
        await update.message.reply_text("Avbrutet.")
    return ConversationHandler.END

# ---- START LIVE ----

async def cmd_start_live(update: Update, ctx):
    STATE.mock_mode = False
    await update.message.reply_text("Starta live-läge? (inga riktiga orders) – Skriv JA")
    return WAIT_LIVE_CONFIRM

async def confirm_live(update: Update, ctx):
    if update.message.text.upper() == "JA":
        STATE.engine_running = True
        await update.message.reply_text("Live-engine på.")
    else:
        await update.message.reply_text("Avbrutet.")
    return ConversationHandler.END

async def cmd_cancel(update: Update, ctx):
    await update.message.reply_text("Avbrutet.")
    return ConversationHandler.END

async def cmd_stop(update: Update, ctx):
    STATE.engine_running = False
    await update.message.reply_text("Engine stoppad.")

async def cmd_pnl(update: Update, ctx):
    await update.message.reply_text(f"Mock-PnL: {STATE.mock_pnl:.4f}")

async def cmd_export_csv(update: Update, ctx):
    if not MOCK_LOG_FILE.exists():
        return await update.message.reply_text("Ingen mock_trade_log.csv än.")
    await update.message.reply_document(
        InputFile(MOCK_LOG_FILE.open("rb"), filename="mock_trade_log.csv")
    )

async def cmd_mock_trade(update: Update, ctx):
    symbol = ctx.args[0].upper() if ctx.args else "BTCUSDT"
    candles = fake_candles()
    state = SymbolState(symbol=symbol)
    closed = []
    for c in candles:
        trades = process_candle(state, c)
        closed.extend(trades)
    if not closed:
        return await update.message.reply_text("Inga mock trades.")
    t = closed[-1]
    STATE.mock_trades.append(t)
    await log_and_notify_trade(update, ctx, t)
    await update.message.reply_text(
        f"Mock klar. PnL: {t.pnl:.4f}  Total: {STATE.mock_pnl:.4f}"
    )

# ----------------------------------------------------------------------
# ENGINE – EGEN ASYNCIO LOOP
# ----------------------------------------------------------------------

async def engine_loop():
    # Här kommer riktig KuCoin-data senare
    if not STATE.engine_running:
        return
    # Just nu gör vi inget live-data här
    pass

async def engine_background():
    while True:
        try:
            await engine_loop()
        except Exception as e:
            logging.error(f"Engine error: {e}")
        await asyncio.sleep(10)

# ----------------------------------------------------------------------
# MAIN – VIKTIGT: INGEN asyncio.run()!!!
# DigitalOcean kräver loop.run_forever()
# ----------------------------------------------------------------------

async def main():
    logging.basicConfig(level=logging.INFO)

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("set_ai", cmd_set_ai))
    app.add_handler(CommandHandler("set_entry_mode", cmd_set_entry_mode))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("mock_trade", cmd_mock_trade))

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start_mock", cmd_start_mock),
            CommandHandler("start_live", cmd_start_live),
        ],
        states={
            WAIT_MOCK_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, confirm_mock)],
            WAIT_LIVE_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, confirm_live)],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )
    app.add_handler(conv)

    # Starta engine i bakgrunden
    asyncio.create_task(engine_background())

    await app.run_polling()

# ---- Starta på DigitalOcean ----
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
