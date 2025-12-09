# main.py
# Mp ORBbot – ny ORB-logik, mock-läge, redo för live.
# Fokus:
#  - ORB = första gröna candle efter röd
#  - ENTRY_MODE: CLOSE eller TICK
#  - Stop-loss = ORB-låg
#  - Trailing stop efter +0.1 %
#  - AI-lägen: neutral / aggressive / cautious
#  - Telegram-kommandon + mocktrades med ENTRY & EXIT-logg

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
    ContextTypes,
)

# ---------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------

BOT_TOKEN = "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "LINKUSDT", "XRPUSDT", "ADAUSDT"]
TIMEFRAME = "3m"  # ORB-timeframe

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
MOCK_LOG_FILE = LOG_DIR / "mock_trade_log.csv"
REAL_LOG_FILE = LOG_DIR / "real_trade_log.csv"

# Mock / risk / trailing
DEFAULT_MOCK_TRADE_SIZE_USDT = 30.0
FEE_RATE = 0.001          # 0.1% per köp/sälj (simulerat)
TRAIL_TRIGGER = 0.001     # +0.1% från entry
TRAIL_DISTANCE = 0.002    # 0.2% från high

# ---------------------------------------------------
# DATAMODELLER
# ---------------------------------------------------

@dataclass
class Candle:
    ts: int          # epoch ms
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def color(self) -> str:
        if self.close > self.open:
            return "green"
        elif self.close < self.open:
            return "red"
        return "doji"


@dataclass
class Trade:
    symbol: str
    side: str      # LONG (vi kör bara long-ORB här)
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
    orb_high: float = 0.0
    orb_low: float = 0.0
    last_candle_color: str = "none"
    position: Optional[Position] = None


class BotState:
    def __init__(self) -> None:
        self.mock_mode: bool = True           # start i mock-läge
        self.engine_running: bool = False
        self.ai_mode: str = "neutral"         # neutral / aggressive / cautious
        self.entry_mode: str = "CLOSE"        # CLOSE / TICK
        self.timeframe: str = TIMEFRAME
        self.trade_size_usdt: float = DEFAULT_MOCK_TRADE_SIZE_USDT

        self.symbol_states: Dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in SYMBOLS
        }

        self.mock_trades: List[Trade] = []
        self.real_trades: List[Trade] = []

    @property
    def total_mock_pnl(self) -> float:
        return sum(t.pnl for t in self.mock_trades)

    @property
    def total_real_pnl(self) -> float:
        return sum(t.pnl for t in self.real_trades)


STATE = BotState()

# ---------------------------------------------------
# HJÄLPARE
# ---------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_log_header(path: Path, is_mock: bool) -> None:
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


async def log_and_notify_trade(
    update: Optional[Update],
    context: ContextTypes.DEFAULT_TYPE,
    trade: Trade,
) -> None:
    """Logga mock/live-trade + skicka Telegrammeddelande med ENTRY & EXIT."""
    log_file = MOCK_LOG_FILE if trade.is_mock else REAL_LOG_FILE
    ensure_log_header(log_file, is_mock=trade.is_mock)

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

    chat_id = None
    if update and update.effective_chat:
        chat_id = update.effective_chat.id

    if chat_id is None:
        return

    side_emoji = "🟢 LONG"
    prefix = "📊 Mock-trade" if trade.is_mock else "💰 Live-trade"

    txt = (
        f"{prefix} avslutad\n"
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

    await context.bot.send_message(chat_id=chat_id, text=txt, parse_mode="Markdown")


# ---------------------------------------------------
# ORB-LOGIK (min design)
# ---------------------------------------------------

def maybe_start_orb(state: SymbolState, candle: Candle) -> None:
    """
    ORB = första gröna candle efter en röd.
    Vi skapar en ORB bara om:
      - förra candle var röd
      - nuvarande är grön
      - vi har ingen öppen position
    """
    if state.position is not None:
        return
    if state.last_candle_color == "red" and candle.color == "green":
        state.orb_active = True
        state.orb_high = candle.high
        state.orb_low = candle.low


def ai_filter_passes(ai_mode: str, candle: Candle) -> bool:
    """
    Enkel AI-filter placeholder:
    - neutral: alltid OK
    - aggressive: tillåter även små bodies (mer trades)
    - cautious: kräver tydlig trend (green + close nära high)
    """
    body = abs(candle.close - candle.open)
    range_ = candle.high - candle.low if candle.high > candle.low else 1e-8
    body_ratio = body / range_

    if ai_mode == "neutral":
        return True
    if ai_mode == "aggressive":
        return body_ratio > 0.1  # nästan allt OK
    if ai_mode == "cautious":
        # kräver stor grön candle och stängning i övre delen
        if candle.color != "green":
            return False
        if body_ratio < 0.5:
            return False
        if candle.close < candle.low + 0.7 * range_:
            return False
        return True
    return True


def check_entry(
    state: SymbolState,
    candle: Candle,
    ai_mode: str,
    entry_mode: str,
    trade_size_usdt: float,
    is_mock: bool,
) -> Optional[Position]:
    """
    Returnerar Position om vi ska gå in long, annars None.
    - CLOSE: entry vid candle.close om close > ORB-high
    - TICK : entry vid ORB-high om high > ORB-high (approx tick-baserad)
    """
    if not state.orb_active:
        return None
    if state.position is not None:
        return None
    if not ai_filter_passes(ai_mode, candle):
        return None

    entry_price: Optional[float] = None

    if entry_mode == "CLOSE":
        if candle.close > state.orb_high:
            entry_price = candle.close
    elif entry_mode == "TICK":
        if candle.high > state.orb_high:
            entry_price = state.orb_high

    if entry_price is None:
        return None

    qty = trade_size_usdt / entry_price
    pos = Position(
        symbol=state.symbol,
        side="LONG",
        entry_price=entry_price,
        qty=qty,
        stop_loss=state.orb_low,
        best_price=entry_price,
        opened_at=now_iso(),
        ai_mode=ai_mode,
        entry_mode=entry_mode,
        timeframe=TIMEFRAME,
        is_mock=is_mock,
    )

    # ORB konsumerad
    state.orb_active = False
    return pos


def check_exit(
    pos: Position,
    candle: Candle,
) -> Optional[float]:
    """
    Enkelt exit-scenario:
    - Stop-loss om low < stop_loss
    - Trailing stop:
        * aktiveras när pris rört sig +0.1% från entry
        * stop = best_price * (1 - TRAIL_DISTANCE)
        * exit om close < trail_stop
    Returnerar exit_price eller None.
    """
    # Uppdatera best_price
    if candle.high > pos.best_price:
        pos.best_price = candle.high

    # Stop-loss först
    if candle.low <= pos.stop_loss:
        return pos.stop_loss

    # Trailing start
    if not pos.trailing_active:
        if candle.high >= pos.entry_price * (1 + TRAIL_TRIGGER):
            pos.trailing_active = True

    if pos.trailing_active:
        trail_stop = pos.best_price * (1 - TRAIL_DISTANCE)
        if candle.close <= trail_stop:
            return trail_stop

    return None


def process_candle_for_symbol(
    state: SymbolState,
    candle: Candle,
    ai_mode: str,
    entry_mode: str,
    trade_size_usdt: float,
    is_mock: bool,
) -> List[Trade]:
    """
    Kärn-ORB-logik: uppdaterar ORB & position baserat på en ny candle.
    Returnerar lista med stängda trades (0 eller 1 här).
    """
    closed_trades: List[Trade] = []

    # 1) eventuella exits först
    if state.position:
        pos = state.position
        exit_price = check_exit(pos, candle)
        if exit_price is not None:
            notional_entry = pos.entry_price * pos.qty
            notional_exit = exit_price * pos.qty
            gross_pnl = (exit_price - pos.entry_price) * pos.qty
            fees = (notional_entry + notional_exit) * FEE_RATE
            pnl = gross_pnl - fees

            trade = Trade(
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
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
            closed_trades.append(trade)
            state.position = None

    # 2) ORB-definition
    maybe_start_orb(state, candle)

    # 3) entry
    if state.position is None:
        pos = check_entry(
            state,
            candle,
            ai_mode=ai_mode,
            entry_mode=entry_mode,
            trade_size_usdt=trade_size_usdt,
            is_mock=is_mock,
        )
        if pos is not None:
            state.position = pos

    state.last_candle_color = candle.color
    return closed_trades


# ---------------------------------------------------
# MOCKDATA FÖR TEST (utan KuCoin)
# ---------------------------------------------------

def generate_fake_candles(n: int, base_price: float = 100.0) -> List[Candle]:
    """
    En enkel faker för /mock_trade_orb så att du kan se hur
    ORB-logiken beter sig utan riktig marknadsdata.
    """
    import random
    candles: List[Candle] = []
    price = base_price
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    for i in range(n):
        ts += 3 * 60 * 1000  # 3m
        drift = random.uniform(-0.005, 0.005)  # +/-0.5 %
        new_price = price * (1 + drift)
        high = max(price, new_price) * (1 + random.uniform(0, 0.002))
        low = min(price, new_price) * (1 - random.uniform(0, 0.002))
        open_ = price
        close = new_price
        vol = random.uniform(10, 100)
        candles.append(Candle(ts=ts, open=open_, high=high, low=low, close=close, volume=vol))
        price = new_price

    return candles


# ---------------------------------------------------
# LIVE-HOOKS (KuCoin) – FÖRBEREDDA, EJ IFRÅNKÖRDA ÄN
# ---------------------------------------------------

async def place_live_order(symbol: str, side: str, qty: float) -> None:
    """
    Här kopplar du på riktig KuCoin-order sen.
    Just nu är det bara en placeholder.
    """
    # TODO: implementera KuCoin-order här (signering, REST-anrop etc.).
    # För säkerhet kör vi ingen riktig order i den här versionen.
    logging.info(f"[LIVE-PLACE-ORDER] {symbol} {side} qty={qty}")


async def close_live_position(symbol: str) -> None:
    """
    Här stänger du riktig KuCoin-position senare.
    """
    logging.info(f"[LIVE-CLOSE-POSITION] {symbol}")


# ---------------------------------------------------
# TELEGRAM-KOMMANDON
# ---------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    txt = (
        "Hej! 👋\n"
        "Det här är Mp ORBbot med *ny ORB-logik*.\n\n"
        "ORB:\n"
        "• Första gröna candle efter en röd = ORB-fönster\n"
        "• Entry CLOSE: close > ORB-high\n"
        "• Entry TICK: high > ORB-high (entry vid ORB-high)\n"
        "• Stop-loss: ORB-low\n"
        "• Trailing: efter +0.1%, stop flyttas upp\n\n"
        "Kommandon:\n"
        "• /status – visa läge\n"
        "• /set_ai neutral|aggressive|cautious\n"
        "• /set_entry_mode CLOSE|TICK\n"
        "• /start_mock – starta mock-engine (med bekräftelse)\n"
        "• /start_live – förberett för live (ingen riktig order än)\n"
        "• /stop – stoppa engine\n"
        "• /mock_trade SYMBOL – kör en ORB-simulering på fake-data\n"
        "• /pnl – visa mock-PnL\n"
        "• /export_csv – exportera mock_trade_log.csv\n"
    )
    await update.message.reply_text(txt, parse_mode="Markdown")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    txt = (
        "📊 *Status*\n\n"
        f"Engine: `{'ON' if STATE.engine_running else 'OFF'}`\n"
        f"Mock-mode: `{STATE.mock_mode}`\n"
        f"AI-läge: `{STATE.ai_mode}`\n"
        f"Entry-mode: `{STATE.entry_mode}`\n"
        f"Timeframe: `{STATE.timeframe}`\n"
        f"Trade-size (mock): `{STATE.trade_size_usdt} USDT`\n\n"
        f"Mock-trades: `{len(STATE.mock_trades)}`\n"
        f"Mock-PnL: `{STATE.total_mock_pnl:.4f}` USDT\n"
        f"Real-trades: `{len(STATE.real_trades)}`\n"
        f"Real-PnL: `{STATE.total_real_pnl:.4f}` USDT"
    )
    await update.message.reply_text(txt, parse_mode="Markdown")


async def cmd_set_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            f"Nuvarande AI-läge: `{STATE.ai_mode}`\n"
            "Använd: /set_ai neutral|aggressive|cautious",
            parse_mode="Markdown",
        )
        return
    mode = context.args[0].lower()
    if mode not in ["neutral", "aggressive", "cautious"]:
        await update.message.reply_text("Ogiltigt AI-läge. Välj neutral/aggressive/cautious.")
        return
    STATE.ai_mode = mode
    await update.message.reply_text(f"🤖 AI-läge satt till `{STATE.ai_mode}`", parse_mode="Markdown")


async def cmd_set_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            f"Nuvarande ENTRY_MODE: `{STATE.entry_mode}`\n"
            "Använd: /set_entry_mode CLOSE|TICK",
            parse_mode="Markdown",
        )
        return
    mode = context.args[0].upper()
    if mode not in ["CLOSE", "TICK"]:
        await update.message.reply_text("Ogiltigt läge. Använd CLOSE eller TICK.")
        return
    STATE.entry_mode = mode
    await update.message.reply_text(f"🎯 ENTRY_MODE satt till `{STATE.entry_mode}`", parse_mode="Markdown")


async def cmd_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.mock_mode = True
    await update.message.reply_text(
        "Du är på väg att starta *mock-trading*.\n"
        "Skriv `JA` för att bekräfta.",
        parse_mode="Markdown",
    )

    def check_reply(u: Update) -> bool:
        return u.effective_chat.id == update.effective_chat.id

    try:
        reply = await context.application.bot.wait_for_message(
            timeout=30, chat_id=update.effective_chat.id
        )
    except Exception:
        reply = None

    if not reply or reply.text.strip().upper() != "JA":
        await update.message.reply_text("Mock-trading avbruten.")
        return

    STATE.engine_running = True
    await update.message.reply_text("▶️ Engine startad i *mock-läge*.", parse_mode="Markdown")


async def cmd_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.mock_mode = False
    await update.message.reply_text(
        "⚠️ Du är på väg att aktivera *live-läge* (just nu skickas inga riktiga KuCoin-orders).\n"
        "Skriv `JA` för att bekräfta.",
        parse_mode="Markdown",
    )

    try:
        reply = await context.application.bot.wait_for_message(
            timeout=30, chat_id=update.effective_chat.id
        )
    except Exception:
        reply = None

    if not reply or reply.text.strip().upper() != "JA":
        await update.message.reply_text("Live-läge avbrutet.")
        return

    STATE.engine_running = True
    await update.message.reply_text(
        "▶️ Engine startad i *live-läge* (men utan riktiga KuCoin-orders i denna version).",
        parse_mode="Markdown",
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.engine_running = False
    await update.message.reply_text("⛔ Engine stoppad.")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    txt = (
        "📈 *PnL*\n\n"
        f"Mock-trades: {len(STATE.mock_trades)}\n"
        f"Mock-PnL: {STATE.total_mock_pnl:.4f} USDT\n\n"
        f"Real-trades: {len(STATE.real_trades)}\n"
        f"Real-PnL: {STATE.total_real_pnl:.4f} USDT"
    )
    await update.message.reply_text(txt, parse_mode="Markdown")


async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not MOCK_LOG_FILE.exists():
        await update.message.reply_text("Ingen mock_trade_log.csv hittades ännu.")
        return
    await update.message.reply_document(
        document=InputFile(MOCK_LOG_FILE.open("rb"), filename="mock_trade_log.csv"),
        caption="Här är mock_trade_log.csv 📄",
    )


async def cmd_mock_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Enkel ORB-simulering på fejk-candles för ETT symbol.
    Bra för att se att ORB-logiken + ENTRY/EXIT-notiser funkar.
    """
    symbol = SYMBOLS[0]
    if context.args:
        s = context.args[0].upper()
        if s in SYMBOLS:
            symbol = s

    candles = generate_fake_candles(150)
    sym_state = SymbolState(symbol=symbol)
    closed: List[Trade] = []

    for c in candles:
        trades = process_candle_for_symbol(
            sym_state,
            c,
            ai_mode=STATE.ai_mode,
            entry_mode=STATE.entry_mode,
            trade_size_usdt=STATE.trade_size_usdt,
            is_mock=True,
        )
        closed.extend(trades)

    if not closed:
        await update.message.reply_text(
            f"Inga trades genererades för {symbol} i denna ORB-simulering."
        )
        return

    # ta sista trade och logga/notifiera (alla loggas till mock-lista)
    for t in closed:
        STATE.mock_trades.append(t)

    last_trade = closed[-1]
    await log_and_notify_trade(update, context, last_trade)

    txt = (
        f"✅ ORB-mock på {symbol} klar.\n"
        f"Antal trades: {len(closed)}\n"
        f"Senaste trade PnL: {last_trade.pnl:.4f} USDT\n"
        f"Total mock-PnL: {STATE.total_mock_pnl:.4f} USDT"
    )
    await update.message.reply_text(txt)


# ---------------------------------------------------
# ENGINE-SCHEDULER (READY FÖR RIKTIG DATA SEN)
# ---------------------------------------------------

async def engine_loop(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Här ska vi i framtiden:
      - hämta senaste 3m-candles från KuCoin
      - mata dem in i process_candle_for_symbol för varje symbol
      - vid stängda trades: logga + notifiera
    I den här versionen lämnar vi den tom för att inte köra fejk-live.
    """
    if not STATE.engine_running:
        return
    # TODO: implementera riktig datafeed här
    # t.ex.:
    # for symbol in SYMBOLS:
    #    candle = fetch_latest_candle_from_kucoin(symbol, TIMEFRAME)
    #    trades = process_candle_for_symbol(...)
    #    för varje trade -> log_and_notify_trade + ev. place_live_order/close_live_position
    pass


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

async def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("set_ai", cmd_set_ai))
    app.add_handler(CommandHandler("set_entry_mode", cmd_set_entry_mode))
    app.add_handler(CommandHandler("start_mock", cmd_start_mock))
    app.add_handler(CommandHandler("start_live", cmd_start_live))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("mock_trade", cmd_mock_trade))

    # engine-loop var 10:e sekund (redo för riktig data sen)
    app.job_queue.run_repeating(engine_loop, interval=10, first=10)

    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
