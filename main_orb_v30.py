# main_orb_v30.py
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import io

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from telegram import Update, Bot, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# --------------------
# Konfiguration
# --------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
owner_env = os.getenv("OWNER_CHAT_ID", "").strip()
OWNER_CHAT_ID = int(owner_env) if owner_env.isdigit() else 5397586616  # Ditt ID

# Webhook URL (Render public URL + path)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://mporbbot.onrender.com").rstrip("/")
WEBHOOK_PATH = f"/webhook/{TELEGRAM_TOKEN}"
WEBHOOK_URL = f"{PUBLIC_BASE_URL}{WEBHOOK_PATH}"

KUCOIN_BASE = "https://api.kucoin.com"
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "").strip()
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "").strip()
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "").strip()

DEFAULT_SYMBOLS = ["BTC-USDT"]          # Du kan ändra via /symbols
DEFAULT_TIMEFRAME = "1min"              # 1min eller 3min (KuCoin: '1min'/'3min')
DEFAULT_ENTRY_MODE = "close"            # 'close' eller 'tick'
DEFAULT_TRADE_USDT = float(os.getenv("TRADE_USDT", "25"))  # orderstorlek i USDT

# --------------------
# Logging
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s orb_v30: %(message)s"
)
log = logging.getLogger("orb_v30")

# --------------------
# Datamodeller
# --------------------
@dataclass
class Candle:
    ts: int  # ms epoch
    open: float
    close: float
    high: float
    low: float

    @property
    def is_green(self) -> bool:
        return self.close > self.open

    @property
    def is_red(self) -> bool:
        return self.close < self.open


@dataclass
class Trade:
    symbol: str
    side: str           # "LONG" (vi kör long-logik)
    qty: float
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    @property
    def closed(self) -> bool:
        return self.exit_time is not None and self.exit_price is not None

    @property
    def pnl_abs(self) -> Optional[float]:
        if not self.closed:
            return None
        return (self.exit_price - self.entry_price) * self.qty

    def k4_row(self) -> Optional[dict]:
        if not self.closed:
            return None
        # K4 avsnitt D – Övriga tillgångar (krypto)
        # Försäljningspris = exit_value, Omkostnadsbelopp = entry_value
        exit_value = self.exit_price * self.qty
        entry_value = self.entry_price * self.qty
        diff = exit_value - entry_value
        return {
            "Datum försäljning": self.exit_time.date().isoformat(),
            "Tillgång": self.symbol.replace("-USDT", ""),
            "Antal": f"{self.qty:.8f}",
            "Försäljningspris (SEK/USDT)": f"{exit_value:.2f}",
            "Omkostnadsbelopp (SEK/USDT)": f"{entry_value:.2f}",
            "Vinst/Förlust (SEK/USDT)": f"{diff:.2f}",
        }


@dataclass
class Position:
    in_position: bool = False
    entry_price: float = 0.0
    qty: float = 0.0
    stop: float = 0.0
    entry_time: Optional[datetime] = None
    entry_mode: str = DEFAULT_ENTRY_MODE  # 'close' eller 'tick'
    # ORB
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_ts: Optional[int] = None  # ts för ORB-candlen

    active_trade: Optional[Trade] = None


@dataclass
class SymbolState:
    symbol: str
    position: Position = field(default_factory=Position)
    last_candle_ts: Optional[int] = None  # senast behandlad candle ts (ms)
    closed_trades: List[Trade] = field(default_factory=list)
    task: Optional[asyncio.Task] = None


@dataclass
class EngineConfig:
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    timeframe: str = DEFAULT_TIMEFRAME  # '1min'/'3min'
    entry_mode: str = DEFAULT_ENTRY_MODE
    trade_usdt: float = DEFAULT_TRADE_USDT
    engine_on: bool = False
    live_trading: bool = False


# --------------------
# Globalt tillstånd
# --------------------
engine = EngineConfig()
states: Dict[str, SymbolState] = {}

# Telegram app & FastAPI
app = FastAPI()
tg_app: Optional[Application] = None
tg_bot: Optional[Bot] = None

# --------------------
# KuCoin Helpers
# --------------------
async def kucoin_get_candles(symbol: str, tf: str) -> List[Candle]:
    """
    Hämtar candles. KuCoin returnerar senaste först.
    Vi returnerar i stigande tid (äldst -> nyast).
    """
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    params = {"type": tf, "symbol": symbol}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
    # Format: [time, open, close, high, low, volume, turnover]
    candles: List[Candle] = []
    for row in data:
        try:
            ts = int(row[0]) * 1000  # KuCoin time = seconds, convert to ms
            o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
            candles.append(Candle(ts=ts, open=o, close=c, high=h, low=l))
        except Exception:
            continue
    candles.reverse()  # äldst -> nyast
    return candles


async def kucoin_get_last_price(symbol: str) -> Optional[float]:
    url = f"{KUCOIN_BASE}/api/v1/market/orderbook/level1"
    params = {"symbol": symbol}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data")
    if not data:
        return None
    return float(data["price"])


# --------------------
# ORB-Logik
# --------------------
def find_latest_orb(candles: List[Candle]) -> Optional[Candle]:
    """
    ORB = första GRÖNA candle EFTER en RÖD.
    Vi letar igenom listan (äldst->nyast) och returnerar den SENASTE förekomsten.
    """
    last_orb: Optional[Candle] = None
    for i in range(1, len(candles)):
        prev_c = candles[i-1]
        cur_c = candles[i]
        if prev_c.is_red and cur_c.is_green:
            last_orb = cur_c
    return last_orb


# --------------------
# Order & Risk
# --------------------
def calc_qty(notional_usdt: float, price: float) -> float:
    if price <= 0:
        return 0.0
    qty = notional_usdt / price
    # avrunda till rimligt antal decimals
    return round(qty, 6)


async def place_live_market_buy(symbol: str, qty: float) -> Tuple[bool, Optional[str]]:
    """
    Här skulle riktiga KuCoin-orders placeras (spot market buy).
    För enkelhet kör vi en "mock" bekräftelse om nycklar saknas.
    """
    if not (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE):
        return False, "KuCoin API-nycklar saknas – kan inte lägga live-order."
    # TODO: Implementera riktig order via kucoin-python om så önskas.
    # Denna funktion returnerar mock-ok tills riktiga anrop kopplas på.
    return True, "Live BUY (mock-kvitto)."


async def place_live_market_sell(symbol: str, qty: float) -> Tuple[bool, Optional[str]]:
    if not (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE):
        return False, "KuCoin API-nycklar saknas – kan inte lägga live-order."
    # TODO: Implementera riktig order via kucoin-python om så önskas.
    return True, "Live SELL (mock-kvitto)."


# --------------------
# Motor per symbol
# --------------------
async def run_symbol_loop(state: SymbolState):
    symbol = state.symbol
    log.info(f"[{symbol}] loop startad. tf={engine.timeframe} entry={engine.entry_mode}")
    try:
        while engine.engine_on:
            # 1) Hämta candles
            candles = await kucoin_get_candles(symbol, engine.timeframe)
            if not candles:
                await asyncio.sleep(2)
                continue

            last_closed = candles[-1]     # senaste stängda candle (KuCoin ger bara stängda)
            prev_closed = candles[-2] if len(candles) >= 2 else None

            # Uppdatera ORB om vi inte har en eller om en ny sekvens röd->grön uppstått
            orb_c = find_latest_orb(candles)
            pos = state.position
            if (not pos.orb_ts) or (orb_c and orb_c.ts != pos.orb_ts and not pos.in_position):
                if orb_c:
                    pos.orb_ts = orb_c.ts
                    pos.orb_high = orb_c.high
                    pos.orb_low = orb_c.low
                    log.info(f"[{symbol}] Ny ORB: ts={orb_c.ts} high={pos.orb_high} low={pos.orb_low}")

            # ENTRY
            if (not pos.in_position) and pos.orb_high and pos.orb_low:
                if engine.entry_mode == "close":
                    # köp om senaste stängda candle stängde över ORB-high
                    if last_closed.close > pos.orb_high:
                        price = last_closed.close
                        qty = calc_qty(engine.trade_usdt, price)
                        await open_long(state, price, qty, reason="Close över ORB-high")
                else:  # "tick"
                    tick = await kucoin_get_last_price(symbol)
                    if tick and tick > pos.orb_high:
                        qty = calc_qty(engine.trade_usdt, tick)
                        await open_long(state, tick, qty, reason="Tick över ORB-high")

            # TRAILING STOP (vid varje NY stängd candle flyttas stop till max(stop, candle.low))
            if pos.in_position and pos.entry_time:
                # flytta stop uppåt på varje ny candle
                if state.last_candle_ts is None or last_closed.ts != state.last_candle_ts:
                    new_stop = max(pos.stop, last_closed.low)
                    if new_stop > pos.stop:
                        pos.stop = new_stop
                        log.info(f"[{symbol}] Trailing stop flyttad till {pos.stop:.6f}")

            # EXIT på stop (tick-baserad kontroll)
            if pos.in_position:
                tick = await kucoin_get_last_price(symbol)
                if tick is not None and tick <= pos.stop:
                    await close_long(state, tick, reason="STOP HIT")

            # uppdatera senast behandlad candle ts
            state.last_candle_ts = last_closed.ts

            # loop-takt
            await asyncio.sleep(2 if engine.entry_mode == "tick" else 5)

    except asyncio.CancelledError:
        log.info(f"[{symbol}] loop avbruten.")
        return
    except Exception as e:
        log.exception(f"[{symbol}] loop fel: {e}")


async def open_long(state: SymbolState, price: float, qty: float, reason: str):
    symbol = state.symbol
    pos = state.position
    if qty <= 0:
        return
    # Live?
    if engine.live_trading:
        ok, msg = await place_live_market_buy(symbol, qty)
        if not ok:
            await safe_notify(f"❌ {symbol} LIVE BUY misslyckades: {msg}")
            return
    # Mock alltid
    pos.in_position = True
    pos.entry_price = price
    pos.qty = qty
    pos.stop = pos.orb_low if pos.orb_low else price * 0.99  # fallback
    pos.entry_time = datetime.now(timezone.utc)
    pos.entry_mode = engine.entry_mode
    pos.active_trade = Trade(
        symbol=symbol, side="LONG", qty=qty,
        entry_time=pos.entry_time, entry_price=price
    )
    await safe_notify(
        f"✅ {symbol} ENTRY LONG @ {price:.4f} qty={qty} stop={pos.stop:.4f}\n"
        f"reason={reason} mode={pos.entry_mode}"
    )


async def close_long(state: SymbolState, price: float, reason: str):
    symbol = state.symbol
    pos = state.position
    if not pos.in_position or not pos.active_trade:
        return
    qty = pos.qty
    if qty <= 0:
        qty = 0.0

    if engine.live_trading:
        ok, msg = await place_live_market_sell(symbol, qty)
        if not ok:
            await safe_notify(f"❌ {symbol} LIVE SELL misslyckades: {msg}")
            return

    # Mock stängning
    trade = pos.active_trade
    trade.exit_time = datetime.now(timezone.utc)
    trade.exit_price = price
    state.closed_trades.append(trade)

    pnl_abs = trade.pnl_abs or 0.0
    await safe_notify(
        f"🏁 {symbol} EXIT LONG @ {price:.4f} pnl={pnl_abs:.4f} ({reason})"
    )

    # Nollställ position – behåll ORB tills en NY röd->grön hittas
    pos.in_position = False
    pos.entry_price = 0.0
    pos.qty = 0.0
    pos.stop = 0.0
    pos.entry_time = None
    pos.active_trade = None
    # OBS: pos.orb_high/orb_low/orb_ts behålls tills ny detekteras


async def safe_notify(text: str):
    try:
        if tg_bot:
            await tg_bot.send_message(chat_id=OWNER_CHAT_ID, text=text)
    except Exception:
        log.exception("Telegram notify misslyckades.")


# --------------------
# Telegram-kommandon
# --------------------
def owner_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user and update.effective_user.id != OWNER_CHAT_ID:
            return
        return await func(update, context)
    return wrapper


@owner_only
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hej! ORB-bot v30 ✅\n"
        "ORB: första gröna efter röd. Entry: /entry close|tick. Stop: ORB-low. Trailing: candle lows.\n"
        "Använd /help för fler kommandon."
    )


@owner_only
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/engine on|off – starta/stoppa motorn\n"
        "/status – visa läget\n"
        "/symbols BTC-USDT ETH-USDT … – sätt symboler\n"
        "/timeframe 1min|3min – sätt timeframe\n"
        "/entry close|tick – entry-metod\n"
        "/size 25 – orderstorlek i USDT\n"
        "/live on|off – live-order (kräver API-nycklar)\n"
        "/k4 [år] – exportera K4 CSV (stängda trades)\n"
        "/panic – stäng positioner (mock) & stoppa engine"
    )


@owner_only
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [
        f"Engine: {'ON' if engine.engine_on else 'OFF'}",
        f"Symbols: {', '.join(engine.symbols)}",
        f"Timeframe: {engine.timeframe}",
        f"Entry mode: {engine.entry_mode}",
        f"Trade size: {engine.trade_usdt} USDT",
        f"Live trading: {'ON' if engine.live_trading else 'OFF'}",
    ]
    for sym in engine.symbols:
        st = states.get(sym)
        if not st:
            continue
        p = st.position
        if p.in_position:
            lines.append(
                f"{sym}: IN position entry={p.entry_price:.4f} qty={p.qty} stop={p.stop:.4f} "
                f"ORB(H/L)={p.orb_high:.4f}/{p.orb_low:.4f}"
            )
        else:
            if p.orb_high and p.orb_low:
                lines.append(f"{sym}: FLAT ORB(H/L)={p.orb_high:.4f}/{p.orb_low:.4f}")
            else:
                lines.append(f"{sym}: FLAT (ingen ORB än)")
        lines.append(f"{sym}: Closed trades: {len(st.closed_trades)}")
    await update.message.reply_text("\n".join(lines))


@owner_only
async def cmd_engine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Använd: /engine on eller /engine off")
    mode = context.args[0].lower()
    if mode == "on":
        if engine.engine_on:
            return await update.message.reply_text("Engine är redan ON.")
        engine.engine_on = True
        # starta tasks
        for sym in engine.symbols:
            st = states.setdefault(sym, SymbolState(symbol=sym))
            if st.task and not st.task.done():
                continue
            st.task = asyncio.create_task(run_symbol_loop(st))
        await update.message.reply_text("Engine startad ✅")
    elif mode == "off":
        if not engine.engine_on:
            return await update.message.reply_text("Engine är redan OFF.")
        engine.engine_on = False
        # stoppa tasks
        for st in states.values():
            if st.task and not st.task.done():
                st.task.cancel()
        await update.message.reply_text("Engine stoppad ⛔")
    else:
        await update.message.reply_text("Använd: /engine on eller /engine off")


@owner_only
async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Använd: /symbols BTC-USDT ETH-USDT …")
    engine.symbols = [s.upper() for s in context.args]
    # stoppa gamla tasks om på
    if engine.engine_on:
        for st in states.values():
            if st.task and not st.task.done():
                st.task.cancel()
    # rensa / init
    states.clear()
    for sym in engine.symbols:
        states[sym] = SymbolState(symbol=sym)
    # starta igen om engine_on
    if engine.engine_on:
        for sym in engine.symbols:
            states[sym].task = asyncio.create_task(run_symbol_loop(states[sym]))
    await update.message.reply_text(f"Symbols uppdaterade: {', '.join(engine.symbols)}")


@owner_only
async def cmd_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Använd: /timeframe 1min|3min")
    tf = context.args[0].lower()
    if tf not in ("1min", "3min"):
        return await update.message.reply_text("Endast 1min eller 3min är stöd.")
    engine.timeframe = tf
    await update.message.reply_text(f"Timeframe satt till {tf}")


@owner_only
async def cmd_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Använd: /entry close|tick")
    mode = context.args[0].lower()
    if mode not in ("close", "tick"):
        return await update.message.reply_text("Endast 'close' eller 'tick'.")
    engine.entry_mode = mode
    await update.message.reply_text(f"Entry mode satt till {mode}")


@owner_only
async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Använd: /size 25 (USDT per trade)")
    try:
        val = float(context.args[0])
        if val <= 0:
            raise ValueError()
        engine.trade_usdt = val
        await update.message.reply_text(f"Trade size satt till {val} USDT")
    except Exception:
        await update.message.reply_text("Ogiltigt värde. Ex: /size 25")


@owner_only
async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Använd: /live on|off")
    mode = context.args[0].lower()
    if mode == "on":
        if not (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE):
            return await update.message.reply_text("Saknar KuCoin API-nycklar. Lägg in env vars först.")
        engine.live_trading = True
        await update.message.reply_text("Live trading: ON ⚠️ (marknadsorder).")
    elif mode == "off":
        engine.live_trading = False
        await update.message.reply_text("Live trading: OFF.")
    else:
        await update.message.reply_text("Använd: /live on eller /live off")


@owner_only
async def cmd_k4(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Samla alla stängda trades och exportera CSV
    year: Optional[int] = None
    if context.args:
        try:
            year = int(context.args[0])
        except Exception:
            pass

    rows: List[dict] = []
    for st in states.values():
        for tr in st.closed_trades:
            if not tr.closed:
                continue
            if year and tr.exit_time.year != year:
                continue
            r = tr.k4_row()
            if r:
                rows.append(r)

    if not rows:
        return await update.message.reply_text("Inga stängda trades att exportera.")

    # Skapa CSV i minnet
    output = io.StringIO()
    fieldnames = ["Datum försäljning", "Tillgång", "Antal",
                  "Försäljningspris (SEK/USDT)", "Omkostnadsbelopp (SEK/USDT)", "Vinst/Förlust (SEK/USDT)"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    output.seek(0)

    # Skicka som dokument
    data = io.BytesIO(output.read().encode("utf-8"))
    data.name = f"k4_export{'_'+str(year) if year else ''}.csv"
    await tg_bot.send_document(chat_id=OWNER_CHAT_ID, document=InputFile(data))
    await update.message.reply_text("K4-CSV skickad ✅")


@owner_only
async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Stäng mock-positioner och stoppa engine
    for st in states.values():
        pos = st.position
        if pos.in_position:
            last = await kucoin_get_last_price(st.symbol) or pos.entry_price
            await close_long(st, last, reason="PANIC CLOSE")
    if engine.engine_on:
        engine.engine_on = False
        for st in states.values():
            if st.task and not st.task.done():
                st.task.cancel()
    await update.message.reply_text("PANIC: stängt positioner & stoppat engine.")


# --------------------
# FastAPI + Telegram webhook
# --------------------
@app.on_event("startup")
async def on_startup():
    global tg_app, tg_bot
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN saknas! Lägg till env var.")
    tg_app = Application.builder().token(TELEGRAM_TOKEN).build()
    tg_bot = tg_app.bot

    # Handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("help", cmd_help))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(CommandHandler("engine", cmd_engine))
    tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
    tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
    tg_app.add_handler(CommandHandler("entry", cmd_entry))
    tg_app.add_handler(CommandHandler("size", cmd_size))
    tg_app.add_handler(CommandHandler("live", cmd_live))
    tg_app.add_handler(CommandHandler("k4", cmd_k4))
    tg_app.add_handler(CommandHandler("panic", cmd_panic))
    tg_app.add_handler(MessageHandler(filters.COMMAND, cmd_help))  # fånga okända

    # Sätt webhook
    try:
        me = await tg_bot.get_me()
        log.info(f"Startar… Telegram-bot: @{me.username} (id={me.id})")
        await tg_bot.set_webhook(WEBHOOK_URL)
        log.info(f"Webhook satt: {WEBHOOK_URL}")
    except Exception as e:
        log.exception(f"Kunde inte sätta webhook: {e}")

    # Initiera states
    for sym in engine.symbols:
        states[sym] = SymbolState(symbol=sym)

    await safe_notify("🔧 ORB v30 startad. Använd /engine on för att börja.")


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if not tg_app or not tg_bot:
        return JSONResponse({"ok": False})
    data = await request.json()
    update = Update.de_json(data, tg_bot)
    await tg_app.process_update(update)
    return JSONResponse({"ok": True})


@app.get("/")
async def root():
    return PlainTextResponse("OK v30")


# ------------- lokalkörning -------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_orb_v30:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
