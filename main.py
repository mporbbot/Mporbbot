import csv
import logging
import os
import json
import asyncio
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict

from telegram import (
    Update,
    InputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ============================================================
# TELEGRAM-TOKEN (hårdkodad)
# ============================================================

BOT_TOKEN = "8265069090:AAHVB72TmPDrZP3U9jU_fEzXwdCnoY5tcSI"

if not BOT_TOKEN or ":" not in BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN i main.py är ogiltig. Kontrollera strängen.")

# Chat-id där engine ska skicka trades (sparas när du pratar med boten)
ENGINE_CHAT_ID: Optional[int] = None

# ============================================================
# KONFIG
# ============================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
MOCK_TRADE_FILE = LOG_DIR / "mock_trade_log.csv"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]
DEFAULT_TRADE_SIZE_USDT = 30.0
DEFAULT_FEE_RATE = 0.001       # 0.1 %
SLIPPAGE_RATE = 0.0005         # 0.05 %

TRAIL_TRIGGER = 0.001          # +0.1 %
TRAIL_DISTANCE = 0.002         # -0.2 %

KUCOIN_URL = "https://api.kucoin.com"

# ============================================================
# DATAMODELLER
# ============================================================

@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    vol: float

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
    pnl: float          # NETTO efter avgifter
    fees: float
    opened_at: str
    closed_at: str


@dataclass
class Position:
    symbol: str
    entry_price: float      # inkl. slippage
    qty: float
    stop_loss: float        # rå nivå (ORB-low)
    best_price: float       # används för trailing
    opened_at: str
    trailing_active: bool = False


class SymbolState:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.orb_ready: bool = False
        self.orb_high: float = 0.0
        self.orb_low: float = 0.0
        self.last_color: str = ""
        self.position: Optional[Position] = None
        self.last_candle_time: int = 0


class BotState:
    def __init__(self) -> None:
        self.engine_on: bool = False
        self.symbol_active: Dict[str, bool] = {s: False for s in SYMBOLS}

        self.entry_mode: str = "CLOSE"   # CLOSE / TICK
        self.ai_mode: str = "neutral"    # neutral/aggressive/cautious
        self.timeframe: str = "3min"

        self.trade_size: float = DEFAULT_TRADE_SIZE_USDT
        self.threshold: float = 0.1
        self.risk_percent: float = 1.0

        self.mock_trades: List[Trade] = []
        self.mock_pnl: float = 0.0       # netto

    def reset_pnl(self) -> None:
        self.mock_trades.clear()
        self.mock_pnl = 0.0


STATE = BotState()
SYMBOL_STATES: Dict[str, SymbolState] = {s: SymbolState(s) for s in SYMBOLS}

# ============================================================
# HJÄLPFUNKTIONER
# ============================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def kucoin_pair(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s


# ============================================================
# CSV-LOGG
# ============================================================

def ensure_log_header() -> None:
    if MOCK_TRADE_FILE.exists():
        return
    with MOCK_TRADE_FILE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "symbol", "side",
            "entry_price", "exit_price", "qty",
            "pnl_net", "fees", "opened_at", "closed_at",
        ])


def save_trade(t: Trade) -> None:
    ensure_log_header()
    with MOCK_TRADE_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.utcnow().isoformat(),
            t.symbol,
            t.side,
            t.entry_price,
            t.exit_price,
            t.qty,
            t.pnl,
            t.fees,
            t.opened_at,
            t.closed_at,
        ])


# ============================================================
# KUCOIN – CANDLES
# ============================================================

def fetch_kucoin_candle(symbol: str) -> Optional[Candle]:
    """Hämta senaste 3m-candle för engine."""
    pair = kucoin_pair(symbol)
    url = f"{KUCOIN_URL}/api/v1/market/candles?type=3min&symbol={pair}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        logging.exception("KuCoin-fel för %s", symbol)
        return None

    if "data" not in data or not data["data"]:
        return None

    row = data["data"][0]   # senaste candle

    try:
        ts = int(float(row[0])) * 1000
        o = float(row[1])
        c = float(row[2])
        h = float(row[3])
        l = float(row[4])
        v = float(row[5])
    except (ValueError, IndexError, TypeError):
        return None

    return Candle(ts=ts, open=o, high=h, low=l, close=c, vol=v)


def fetch_kucoin_candles(symbol: str, limit: int = 200) -> List[Candle]:
    """Hämta historik för /test_buy."""
    pair = kucoin_pair(symbol)
    url = f"{KUCOIN_URL}/api/v1/market/candles?type=3min&symbol={pair}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        logging.exception("KuCoin-fel (historik) för %s", symbol)
        return []

    if "data" not in data or not data["data"]:
        return []

    rows = list(reversed(data["data"]))[:limit]

    candles: List[Candle] = []
    for row in rows:
        try:
            ts = int(float(row[0])) * 1000
            o = float(row[1])
            c = float(row[2])
            h = float(row[3])
            l = float(row[4])
            v = float(row[5])
        except (ValueError, IndexError, TypeError):
            continue
        candles.append(Candle(ts=ts, open=o, high=h, low=l, close=c, vol=v))

    return candles


# ============================================================
# AI-FILTER
# ============================================================

def ai_pass(c: Candle) -> bool:
    body = abs(c.close - c.open)
    rng = max(c.high - c.low, 1e-8)
    r = body / rng

    if STATE.ai_mode == "neutral":
        return True
    if STATE.ai_mode == "aggressive":
        return r > 0.03
    if STATE.ai_mode == "cautious":
        if c.color != "green":
            return False
        if r < 0.5:
            return False
        if c.close < c.low + 0.7 * (c.high - c.low):
            return False
        return True

    return True


# ============================================================
# ORB-LOGIK
# ============================================================

def orb_detect(s: SymbolState, c: Candle) -> None:
    if s.position:
        return
    if s.last_color == "red" and c.color == "green":
        s.orb_ready = True
        s.orb_high = c.high
        s.orb_low = c.low


def orb_entry(s: SymbolState, c: Candle) -> Optional[Position]:
    if not s.orb_ready or s.position:
        return None
    if not ai_pass(c):
        return None

    raw: Optional[float] = None
    if STATE.entry_mode == "CLOSE":
        if c.close > s.orb_high:
            raw = c.close
    else:  # TICK
        if c.high > s.orb_high:
            raw = s.orb_high

    if raw is None:
        return None

    entry = raw * (1 + SLIPPAGE_RATE)
    qty = STATE.trade_size / entry

    pos = Position(
        symbol=s.symbol,
        entry_price=entry,
        qty=qty,
        stop_loss=s.orb_low,
        best_price=raw,
        opened_at=now_iso(),
    )

    s.orb_ready = False
    return pos


def orb_exit(pos: Position, c: Candle) -> Optional[float]:
    # Stop-loss
    if c.low <= pos.stop_loss:
        return pos.stop_loss * (1 - SLIPPAGE_RATE)

    # Trailing start
    if not pos.trailing_active:
        if c.high >= pos.best_price * (1 + TRAIL_TRIGGER):
            pos.trailing_active = True

    # Trailing stop
    if pos.trailing_active:
        if c.high > pos.best_price:
            pos.best_price = c.high

        tstop = pos.best_price * (1 - TRAIL_DISTANCE)
        if c.close <= tstop:
            return tstop * (1 - SLIPPAGE_RATE)

    return None


def process_candle(s: SymbolState, c: Candle) -> List[Trade]:
    trades: List[Trade] = []

    # EXIT
    if s.position:
        ex = orb_exit(s.position, c)
        if ex is not None:
            pos = s.position
            s.position = None

            notional_entry = pos.entry_price * pos.qty
            notional_exit = ex * pos.qty
            gross = notional_exit - notional_entry
            fees = (notional_entry + notional_exit) * DEFAULT_FEE_RATE
            pnl = gross - fees

            t = Trade(
                symbol=s.symbol,
                side="LONG",
                entry_price=round(pos.entry_price, 4),
                exit_price=round(ex, 4),
                qty=round(pos.qty, 6),
                pnl=round(pnl, 4),
                fees=round(fees, 4),
                opened_at=pos.opened_at,
                closed_at=now_iso(),
            )
            trades.append(t)

    # ORB start
    orb_detect(s, c)

    # ENTRY
    if not s.position:
        pos = orb_entry(s, c)
        if pos:
            s.position = pos

    s.last_candle_time = c.ts
    s.last_color = c.color
    return trades


# ============================================================
# ENGINE – EN TICK (ERSÄTTER JOBQUEUE)
# ============================================================

async def engine_tick(app: Application) -> None:
    global ENGINE_CHAT_ID

    if not STATE.engine_on or not ENGINE_CHAT_ID:
        return

    trades_to_send: List[Trade] = []

    for sym in SYMBOLS:
        if not STATE.symbol_active.get(sym, False):
            continue

        st = SYMBOL_STATES[sym]
        c = fetch_kucoin_candle(sym)
        if not c or c.ts <= st.last_candle_time:
            continue

        closed = process_candle(st, c)
        trades_to_send.extend(closed)

    bot = app.bot
    for t in trades_to_send:
        save_trade(t)
        STATE.mock_trades.append(t)
        STATE.mock_pnl += t.pnl

        msg = (
            f"📊 *Auto ORB trade*\n"
            f"Symbol: {t.symbol}\n"
            f"Entry: {t.entry_price}\n"
            f"Exit: {t.exit_price}\n"
            f"Qty: {t.qty}\n"
            f"PNL: {t.pnl} USDT\n"
            f"Fees: {t.fees} USDT\n"
            f"Opened: {t.opened_at}\n"
            f"Closed: {t.closed_at}"
        )

        await bot.send_message(
            chat_id=ENGINE_CHAT_ID,
            text=msg,
            parse_mode="Markdown",
        )


async def engine_runner(app: Application) -> None:
    """Bakgrundsloop som ersätter JobQueue."""
    while True:
        try:
            await engine_tick(app)
        except Exception:
            logging.exception("Engine loop error")
        await asyncio.sleep(10)  # kolla var 10:e sekund efter ny candle


# ============================================================
# TELEGRAM UI – SYMBOLMENY
# ============================================================

def symbol_menu() -> InlineKeyboardMarkup:
    rows = []
    for s in SYMBOLS:
        active = STATE.symbol_active[s]
        label = f"{s}: {'ON' if active else 'OFF'}"
        rows.append([InlineKeyboardButton(label, callback_data=f"toggle_{s}")])

    rows.append([
        InlineKeyboardButton("▶ ENGINE ON", callback_data="engine_on"),
        InlineKeyboardButton("⛔ ENGINE OFF", callback_data="engine_off"),
    ])

    return InlineKeyboardMarkup(rows)


async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    ENGINE_CHAT_ID = update.effective_chat.id
    await update.message.reply_text("Välj symboler:", reply_markup=symbol_menu())


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    q = update.callback_query
    data = q.data
    ENGINE_CHAT_ID = q.message.chat.id

    if data.startswith("toggle_"):
        sym = data.split("_", 1)[1]
        STATE.symbol_active[sym] = not STATE.symbol_active[sym]
        await q.edit_message_text("Välj symboler:", reply_markup=symbol_menu())
        return

    if data == "engine_on":
        STATE.engine_on = True
        await q.edit_message_text("ENGINE ON", reply_markup=symbol_menu())
        return

    if data == "engine_off":
        STATE.engine_on = False
        await q.edit_message_text("ENGINE OFF", reply_markup=symbol_menu())
        return


# ============================================================
# KOMMANDON
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    ENGINE_CHAT_ID = update.effective_chat.id
    await update.message.reply_text(
        "Mp ORBbot 🚀\n"
        "• /symbols – välj coins\n"
        "• /status – engine & PnL\n"
        "• /mode – AI-läge\n"
        "• /entry_mode – CLOSE/TICK\n"
        "• /test_buy – simulera ORB på en symbol\n"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = [s for s in SYMBOLS if STATE.symbol_active[s]]
    await update.message.reply_text(
        f"ENGINE: {'ON' if STATE.engine_on else 'OFF'}\n"
        f"Aktiva coins: {active}\n"
        f"Trades: {len(STATE.mock_trades)}\n"
        f"PNL (netto): {STATE.mock_pnl:.4f} USDT\n"
        f"AI-läge: {STATE.ai_mode}\n"
        f"ENTRY_MODE: {STATE.entry_mode}\n"
        f"Risk: {STATE.risk_percent}%\n"
        f"Threshold: {STATE.threshold}"
    )


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Total NETTO-PNL: {STATE.mock_pnl:.4f} USDT"
    )


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MOCK_TRADE_FILE.exists():
        await update.message.reply_text("Ingen loggfil ännu.")
        return

    await update.message.reply_document(
        InputFile(MOCK_TRADE_FILE.open("rb"), filename="mock_trade_log.csv")
    )


async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"ENTRY_MODE: {STATE.entry_mode}")
        return

    mode = context.args[0].upper()
    if mode in ["CLOSE", "TICK"]:
        STATE.entry_mode = mode
        await update.message.reply_text(f"ENTRY_MODE satt till {mode}")
    else:
        await update.message.reply_text("Välj CLOSE eller TICK.")


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"AI-läge: {STATE.ai_mode}")
        return

    mode = context.args[0].lower()
    if mode in ["neutral", "aggressive", "cautious"]:
        STATE.ai_mode = mode
        await update.message.reply_text(f"AI-läge satt till {mode}")
    else:
        await update.message.reply_text("Välj neutral / aggressive / cautious.")


async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    ENGINE_CHAT_ID = update.effective_chat.id
    STATE.engine_on = True
    await update.message.reply_text("ENGINE flagga satt till ON (kör på aktiva coins).")


async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = False
    await update.message.reply_text("ENGINE flagga satt till OFF.")


async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            f"Timeframe (lagrad): {STATE.timeframe} – KuCoin-endpoint är ändå 3min."
        )
        return

    STATE.timeframe = context.args[0]
    await update.message.reply_text(
        f"Timeframe sparad som: {STATE.timeframe} (engine använder fortfarande 3min just nu)."
    )


async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Threshold: {STATE.threshold}")
        return

    try:
        v = float(context.args[0])
    except ValueError:
        await update.message.reply_text("Använd t.ex. /threshold 0.2")
        return

    STATE.threshold = v
    await update.message.reply_text(f"Threshold satt till {STATE.threshold}")


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Risk: {STATE.risk_percent}%")
        return

    try:
        v = float(context.args[0])
    except ValueError:
        await update.message.reply_text("Använd t.ex. /risk 1.5")
        return

    STATE.risk_percent = v
    await update.message.reply_text(f"Risk satt till {STATE.risk_percent}%")


async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for s in SYMBOLS:
        SYMBOL_STATES[s].position = None
    await update.message.reply_text("Alla mock-positioner nollställda (ingen extra trade loggad).")


async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.reset_pnl()
    await update.message.reply_text("Mock-PnL återställd.")


async def cmd_test_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else "BTCUSDT"

    candles = fetch_kucoin_candles(symbol, limit=200)
    if not candles:
        await update.message.reply_text(f"Kunde inte hämta candles för {symbol}.")
        return

    local_state = SymbolState(symbol)
    closed_all: List[Trade] = []

    for c in candles:
        closed = process_candle(local_state, c)
        closed_all.extend(closed)

    if not closed_all:
        await update.message.reply_text(f"Inga ORB-trades triggades i test för {symbol}.")
        return

    for t in closed_all:
        save_trade(t)
        STATE.mock_trades.append(t)
        STATE.mock_pnl += t.pnl

    last = closed_all[-1]
    msg = (
        f"📊 *Test ORB trade ({symbol})*\n"
        f"Trades: {len(closed_all)}\n"
        f"Senaste entry: {last.entry_price}\n"
        f"Senaste exit: {last.exit_price}\n"
        f"Senaste PNL: {last.pnl} USDT\n"
        f"Total NETTO-PNL: {STATE.mock_pnl:.4f} USDT"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


# ============================================================
# START-HOOK – STARTA ENGINELOOPEN
# ============================================================

async def on_startup(app: Application):
    app.create_task(engine_runner(app))


# ============================================================
# MAIN
# ============================================================

def main():
    logging.basicConfig(level=logging.INFO)

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(on_startup)
        .build()
    )

    # Kommandon
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("symbols", cmd_symbols))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("export_csv", cmd_export))
    app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("test_buy", cmd_test_buy))

    # Knapp-hanterare
    app.add_handler(CallbackQueryHandler(button_handler))

    app.run_polling()


if __name__ == "__main__":
    main()
