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

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ============================================================
# TOKEN (ENV) – SÄKERT
# ============================================================
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
if not BOT_TOKEN or ":" not in BOT_TOKEN:
    raise ValueError(
        "TELEGRAM_TOKEN saknas/är fel i Environment Variables. "
        "Lägg in hela token (måste innehålla ':')."
    )

ENGINE_CHAT_ID: Optional[int] = None

# ============================================================
# KONFIG
# ============================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
MOCK_TRADE_FILE = LOG_DIR / "mock_trade_log.csv"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]

DEFAULT_TRADE_SIZE_USDT = 30.0
DEFAULT_FEE_RATE = 0.001     # 0.1%
SLIPPAGE_RATE = 0.0005       # 0.05%

KUCOIN_URL = "https://api.kucoin.com"

# ============================================================
# MODELLER
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
    pnl: float
    fees: float
    opened_at: str
    closed_at: str


@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    stop_loss: float
    best_price: float
    opened_at: str
    trailing_active: bool = False


class SymbolState:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.orb_ready = False
        self.orb_high = 0.0
        self.orb_low = 0.0
        self.last_color = ""
        self.position: Optional[Position] = None
        self.last_candle_time = 0


class BotState:
    def __init__(self) -> None:
        self.engine_on: bool = False
        self.symbol_active: Dict[str, bool] = {s: False for s in SYMBOLS}

        self.entry_mode: str = "CLOSE"   # CLOSE/TICK
        self.ai_mode: str = "neutral"    # neutral/aggressive/cautious
        self.timeframe: str = "3m"       # visning/lagring

        # knapparna ska styra dessa:
        self.threshold: float = 0.10
        self.risk_percent: float = 1.0

        # trailing (det du vill styra)
        self.trail_trigger: float = 0.003    # 0.3%
        self.trail_distance: float = 0.0015  # 0.15%

        self.trade_size: float = DEFAULT_TRADE_SIZE_USDT

        self.mock_trades: List[Trade] = []
        self.mock_pnl: float = 0.0

    def reset_pnl(self) -> None:
        self.mock_trades.clear()
        self.mock_pnl = 0.0


STATE = BotState()
SYMBOL_STATES: Dict[str, SymbolState] = {s: SymbolState(s) for s in SYMBOLS}

# ============================================================
# HELPERS
# ============================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def kucoin_pair(symbol: str) -> str:
    s = symbol.upper()
    return s[:-4] + "-USDT" if s.endswith("USDT") else s

# ============================================================
# CSV LOGG
# ============================================================
def ensure_log_header() -> None:
    if MOCK_TRADE_FILE.exists():
        return
    with MOCK_TRADE_FILE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "symbol", "side",
            "entry_price", "exit_price", "qty",
            "pnl_net", "fees",
            "opened_at", "closed_at",
        ])

def save_trade(t: Trade) -> None:
    ensure_log_header()
    with MOCK_TRADE_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.utcnow().isoformat(),
            t.symbol, t.side,
            t.entry_price, t.exit_price,
            t.qty, t.pnl, t.fees,
            t.opened_at, t.closed_at
        ])

# ============================================================
# KUCOIN – SENASTE 3m CANDLE
# ============================================================
def fetch_kucoin_candle(symbol: str) -> Optional[Candle]:
    pair = kucoin_pair(symbol)
    url = f"{KUCOIN_URL}/api/v1/market/candles?type=3min&symbol={pair}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        logging.exception("KuCoin fel %s", symbol)
        return None

    rows = data.get("data") or []
    if not rows:
        return None

    r = rows[0]  # senaste
    try:
        ts = int(float(r[0])) * 1000
        o = float(r[1]); c = float(r[2]); h = float(r[3]); l = float(r[4]); v = float(r[5])
    except Exception:
        return None

    return Candle(ts, o, h, l, c, v)

# ============================================================
# AI FILTER
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
# ORB LOGIK (min)
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

    s.orb_ready = False
    return Position(
        symbol=s.symbol,
        entry_price=entry,
        qty=qty,
        stop_loss=s.orb_low,
        best_price=raw,
        opened_at=now_iso(),
    )

def orb_exit(pos: Position, c: Candle) -> Optional[float]:
    # stop-loss
    if c.low <= pos.stop_loss:
        return pos.stop_loss * (1 - SLIPPAGE_RATE)

    # trailing start
    if not pos.trailing_active:
        if c.high >= pos.best_price * (1 + STATE.trail_trigger):
            pos.trailing_active = True

    # trailing stop
    if pos.trailing_active:
        if c.high > pos.best_price:
            pos.best_price = c.high
        tstop = pos.best_price * (1 - STATE.trail_distance)
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

            trades.append(Trade(
                symbol=s.symbol,
                side="LONG",
                entry_price=round(pos.entry_price, 4),
                exit_price=round(ex, 4),
                qty=round(pos.qty, 6),
                pnl=round(pnl, 4),
                fees=round(fees, 4),
                opened_at=pos.opened_at,
                closed_at=now_iso(),
            ))

    # ORB detect
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
# ENGINE (jobqueue-fri)
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

        trades_to_send.extend(process_candle(st, c))

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
            f"Opened:\n{t.opened_at}\n"
            f"Closed:\n{t.closed_at}"
        )
        await app.bot.send_message(chat_id=ENGINE_CHAT_ID, text=msg, parse_mode="Markdown")

async def engine_runner(app: Application) -> None:
    while True:
        try:
            await engine_tick(app)
        except Exception:
            logging.exception("Engine loop error")
        await asyncio.sleep(10)

async def on_startup(app: Application):
    app.create_task(engine_runner(app))

# ============================================================
# INLINE MENYS (för knapparna du ringat)
# ============================================================
def menu_risk() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1%", callback_data="set_risk_1"),
         InlineKeyboardButton("2%", callback_data="set_risk_2"),
         InlineKeyboardButton("3%", callback_data="set_risk_3"),
         InlineKeyboardButton("5%", callback_data="set_risk_5")],
    ])

def menu_threshold() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("0.05", callback_data="set_thr_0.05"),
         InlineKeyboardButton("0.10", callback_data="set_thr_0.10"),
         InlineKeyboardButton("0.20", callback_data="set_thr_0.20"),
         InlineKeyboardButton("0.30", callback_data="set_thr_0.30")],
    ])

def menu_mode() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("neutral", callback_data="set_mode_neutral"),
         InlineKeyboardButton("aggressive", callback_data="set_mode_aggressive"),
         InlineKeyboardButton("cautious", callback_data="set_mode_cautious")],
    ])

def menu_timeframe() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("3m", callback_data="set_tf_3m"),
         InlineKeyboardButton("5m", callback_data="set_tf_5m"),
         InlineKeyboardButton("15m", callback_data="set_tf_15m")],
        [InlineKeyboardButton("⚠️ Engine kör 3m (just nu)", callback_data="noop")],
    ])

def menu_trail() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Trig 0.2%", callback_data="set_trig_0.002"),
         InlineKeyboardButton("Trig 0.3%", callback_data="set_trig_0.003"),
         InlineKeyboardButton("Trig 0.5%", callback_data="set_trig_0.005")],
        [InlineKeyboardButton("Dist 0.10%", callback_data="set_dist_0.0010"),
         InlineKeyboardButton("Dist 0.15%", callback_data="set_dist_0.0015"),
         InlineKeyboardButton("Dist 0.25%", callback_data="set_dist_0.0025")],
        [InlineKeyboardButton("✅ Rekommenderad (0.3 / 0.15)", callback_data="set_trail_rec")],
    ])

def symbol_menu() -> InlineKeyboardMarkup:
    rows = []
    for s in SYMBOLS:
        active = STATE.symbol_active[s]
        rows.append([InlineKeyboardButton(f"{s}: {'ON' if active else 'OFF'}", callback_data=f"toggle_{s}")])
    rows.append([
        InlineKeyboardButton("▶ ENGINE ON", callback_data="engine_on"),
        InlineKeyboardButton("⛔ ENGINE OFF", callback_data="engine_off"),
    ])
    return InlineKeyboardMarkup(rows)

# ============================================================
# KOMMANDON (kopplade till knapparna där nere)
# ============================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    ENGINE_CHAT_ID = update.effective_chat.id
    await update.message.reply_text("Mp ORBbot igång ✅\nTryck /symbols för coin-menyn.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = [s for s in SYMBOLS if STATE.symbol_active[s]]
    await update.message.reply_text(
        f"ENGINE: {'ON' if STATE.engine_on else 'OFF'}\n"
        f"Aktiva coins: {active}\n"
        f"Trades: {len(STATE.mock_trades)}\n"
        f"Total NETTO-PNL: {STATE.mock_pnl:.4f} USDT\n"
        f"AI-läge: {STATE.ai_mode}\n"
        f"Timeframe: {STATE.timeframe}\n"
        f"Risk: {STATE.risk_percent}%\n"
        f"Threshold: {STATE.threshold}\n"
        f"Trail trig: {STATE.trail_trigger*100:.2f}%\n"
        f"Trail dist: {STATE.trail_distance*100:.2f}%"
    )

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Total NETTO-PNL: {STATE.mock_pnl:.4f} USDT")

async def cmd_engine_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    ENGINE_CHAT_ID = update.effective_chat.id
    STATE.engine_on = True
    await update.message.reply_text("ENGINE ON ✅")

async def cmd_engine_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.engine_on = False
    await update.message.reply_text("ENGINE OFF ⛔")

async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    ENGINE_CHAT_ID = update.effective_chat.id
    await update.message.reply_text("Välj coins:", reply_markup=symbol_menu())

async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Risk nu: {STATE.risk_percent}%", reply_markup=menu_risk())

async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Threshold nu: {STATE.threshold}", reply_markup=menu_threshold())

async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"AI-läge nu: {STATE.ai_mode}", reply_markup=menu_mode())

async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Timeframe nu: {STATE.timeframe}", reply_markup=menu_timeframe())

async def cmd_trail(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Trailing:\n"
        f"Trigger: {STATE.trail_trigger*100:.2f}%\n"
        f"Distance: {STATE.trail_distance*100:.2f}%",
        reply_markup=menu_trail()
    )
async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MOCK_TRADE_FILE.exists():
        await update.message.reply_text("Ingen loggfil ännu.")
        return
    await update.message.reply_document(InputFile(MOCK_TRADE_FILE.open("rb"), filename="mock_trade_log.csv"))

async def cmd_close_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for s in SYMBOLS:
        SYMBOL_STATES[s].position = None
    await update.message.reply_text("Alla mock-positioner nollställda.")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.reset_pnl()
    await update.message.reply_text("PnL nollställd.")

async def cmd_test_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Testbuy är kvar som kommando men förenklad i denna build. (Vill du ha full test-backtest igen säger du till.)")

# ============================================================
# CALLBACK HANDLER (alla inline-knappar)
# ============================================================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ENGINE_CHAT_ID
    q = update.callback_query
    data = q.data
    ENGINE_CHAT_ID = q.message.chat.id

    # no-op
    if data == "noop":
        await q.answer()
        return

    # symbols toggle
    if data.startswith("toggle_"):
        sym = data.split("_", 1)[1]
        STATE.symbol_active[sym] = not STATE.symbol_active[sym]
        await q.edit_message_text("Välj coins:", reply_markup=symbol_menu())
        return

    if data == "engine_on":
        STATE.engine_on = True
        await q.edit_message_text("ENGINE ON ✅", reply_markup=symbol_menu())
        return

    if data == "engine_off":
        STATE.engine_on = False
        await q.edit_message_text("ENGINE OFF ⛔", reply_markup=symbol_menu())
        return

    # risk
    if data.startswith("set_risk_"):
        v = float(data.split("_")[-1])
        STATE.risk_percent = v
        await q.edit_message_text(f"✅ Risk satt till {STATE.risk_percent}%", reply_markup=menu_risk())
        return

    # threshold
    if data.startswith("set_thr_"):
        v = float(data.split("_")[-1])
        STATE.threshold = v
        await q.edit_message_text(f"✅ Threshold satt till {STATE.threshold}", reply_markup=menu_threshold())
        return

    # mode
    if data.startswith("set_mode_"):
        m = data.split("_", 2)[-1]
        STATE.ai_mode = m
        await q.edit_message_text(f"✅ AI-läge: {STATE.ai_mode}", reply_markup=menu_mode())
        return

    # timeframe (lagras)
    if data.startswith("set_tf_"):
        tf = data.split("_", 2)[-1]
        STATE.timeframe = tf
        await q.edit_message_text(f"✅ Timeframe sparad: {STATE.timeframe}", reply_markup=menu_timeframe())
        return

    # trailing recommended
    if data == "set_trail_rec":
        STATE.trail_trigger = 0.003
        STATE.trail_distance = 0.0015
        await q.edit_message_text(
            f"✅ Trailing satt\nTrigger: {STATE.trail_trigger*100:.2f}%\nDistance: {STATE.trail_distance*100:.2f}%",
            reply_markup=menu_trail()
        )
        return

    # trailing trigger
    if data.startswith("set_trig_"):
        v = float(data.split("_")[-1])
        STATE.trail_trigger = v
        await q.edit_message_text(
            f"✅ Trigger satt: {STATE.trail_trigger*100:.2f}%\nDistance: {STATE.trail_distance*100:.2f}%",
            reply_markup=menu_trail()
        )
        return

    # trailing distance
    if data.startswith("set_dist_"):
        v = float(data.split("_")[-1])
        STATE.trail_distance = v
        await q.edit_message_text(
            f"✅ Distance satt: {STATE.trail_distance*100:.2f}%\nTrigger: {STATE.trail_trigger*100:.2f}%",
            reply_markup=menu_trail()
        )
        return

    await q.answer()

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

    # Kommandon (dessa matchar knapparna där nere)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("engine_on", cmd_engine_on))
    app.add_handler(CommandHandler("engine_off", cmd_engine_off))
    app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    app.add_handler(CommandHandler("close_all", cmd_close_all))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("test_buy", cmd_test_buy))

    # Extra (coin-meny + trailing-meny)
    app.add_handler(CommandHandler("symbols", cmd_symbols))
    app.add_handler(CommandHandler("trail", cmd_trail))

    # Inline buttons
    app.add_handler(CallbackQueryHandler(button_handler))

    app.run_polling()

if __name__ == "__main__":
    main()
