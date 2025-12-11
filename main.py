# ============================
# main.py – BLOCK 1/2
# ============================

import csv
import logging
import random
import os
import json
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
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ============================================================
# TELEGRAM-TOKEN – HÅRDKODAD (INTE ENV)
# ============================================================

# Din bottoken – byt till din riktiga om du skapar ny hos BotFather
BOT_TOKEN = "8265069090:AAHVB72TmäPDrZP3U9jU_fEzXwdCnoY5tcSI"

if not BOT_TOKEN or ":" not in BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN i main.py är ogiltig. Kontrollera strängen.")

# Chat-id där engine ska skicka meddelanden (sätts första gången du skriver till boten)
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
        self.ai_mode: str = "neutral"
        self.timeframe: str = "3min"

        self.trade_size: float = DEFAULT_TRADE_SIZE_USDT
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
# KUCOIN – HÄMTA SENASTE CANDLE
# ============================================================

def fetch_kucoin_candle(symbol: str) -> Optional[Candle]:
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

    row = data["data"][0]  # senaste candle

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
    if not s.orb_ready:
        return None
    if s.position:
        return None
    if not ai_pass(c):
        return None

    raw = None
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

    if pos.trailing_active:
        if c.high > pos.best_price:
            pos.best_price = c.high

        tstop = pos.best_price * (1 - TRAIL_DISTANCE)
        if c.close <= tstop:
            return tstop * (1 - SLIPPAGE_RATE)

    return None

# ============================================================
# ENGINE – PROCESSA EN CANDLE
# ============================================================

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
# ============================
# main.py – BLOCK 2A/2
# ============================

# ============================================================
# ENGINE-LOOP (KÖRS AV JOB_QUEUE)
# ============================================================

async def engine_loop(context: ContextTypes.DEFAULT_TYPE):
    if not STATE.engine_on:
        return

    global ENGINE_CHAT_ID
    trades_to_send: List[Trade] = []

    for sym in SYMBOLS:
        if not STATE.symbol_active.get(sym, False):
            continue

        st = SYMBOL_STATES[sym]
        c = fetch_kucoin_candle(sym)
        if not c:
            continue

        # Kör endast på NY candle
        if c.ts <= st.last_candle_time:
            continue

        closed = process_candle(st, c)
        if closed:
            trades_to_send.extend(closed)

    # Ingen chat ännu → hoppa över Telegram-sändning
    if not ENGINE_CHAT_ID:
        return

    # Skicka trades
    bot = context.bot
    for t in trades_to_send:
        save_trade(t)
        STATE.mock_trades.append(t)
        STATE.mock_pnl += t.pnl

        msg = (
            f"📊 *Auto ORB Trade*\n"
            f"Symbol: {t.symbol}\n"
            f"Entry: {t.entry_price}\n"
            f"Exit: {t.exit_price}\n"
            f"Qty: {t.qty}\n"
            f"PNL: {t.pnl} USDT\n"
            f"Fees: {t.fees}\n"
            f"Opened: {t.opened_at}\n"
            f"Closed: {t.closed_at}"
        )

        await bot.send_message(
            chat_id=ENGINE_CHAT_ID,
            text=msg,
            parse_mode="Markdown"
        )

# ============================================================
# SYMBOL-MENY (ON/OFF)
# ============================================================

def symbol_menu():
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
    ENGINE_CHAT_ID = update.effective_chat.id  # spara chatt-ID permanent
    await update.message.reply_text("Välj symboler:", reply_markup=symbol_menu())


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    q = update.callback_query
    data = q.data
    ENGINE_CHAT_ID = q.message.chat.id

    # Toggle-symbol
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
# ============================
# main.py – BLOCK 2B/2 (SLUTET)
# ============================

# ============================================================
# KOMMANDON
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    ENGINE_CHAT_ID = update.effective_chat.id
    await update.message.reply_text(
        "Mp ORBbot 🔥\n"
        "Använd /symbols för att välja vilka coins som ska handlas.\n"
        "Engine körs automatiskt när du slår på den."
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = [s for s in SYMBOLS if STATE.symbol_active[s]]
    await update.message.reply_text(
        f"ENGINE: {'ON' if STATE.engine_on else 'OFF'}\n"
        f"Aktiva coins: {active}\n"
        f"Mock-trades: {len(STATE.mock_trades)}\n"
        f"PNL: {STATE.mock_pnl:.4f} USDT"
    )

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Total NETTO-PNL: {STATE.mock_pnl:.4f} USDT"
    )

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MOCK_TRADE_FILE.exists():
        await update.message.reply_text("Ingen logg ännu.")
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


# ============================================================
# MAIN
# ============================================================

def main():
    logging.basicConfig(level=logging.INFO)

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Kommandon
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("symbols", cmd_symbols))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    app.add_handler(CommandHandler("export_csv", cmd_export))

    # Knapp-hantering
    app.add_handler(CallbackQueryHandler(button_handler))

    # Engine-loop – körs var 10:e sekund och letar efter NY candle
    app.job_queue.run_repeating(engine_loop, interval=10, first=5)

    app.run_polling()

if __name__ == "__main__":
    main()
