# main_v24.py
import os
import time
import json
import queue
import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

# Telegram (PTB v20.x)
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# Konfiguration & logger
# =========================

APP_NAME = "orb_v24"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s " + APP_NAME + ": %(message)s",
)
log = logging.getLogger(APP_NAME)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL = os.getenv("BASE_URL", "https://mporbbot.onrender.com").rstrip("/")

# Kör dessa symboler (visningsformat). KuCoin kräver bindestreck (fixas i helpern).
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]

# Default state
STATE: Dict = {
    "orb_on": True,
    "entry_mode": "both",        # "long" | "short" | "both"
    "trailing": True,            # trailing stop aktiv
    "timeframe": "1m",           # "1m"|"3m"|"5m"|"15m"
    "risk_usdt": 30.0,           # per mocktrade
    "open_positions": {},        # symbol -> position dict
    "realized_pnl_usdt": 0.0,    # ackumulerad PnL (mock)
    "outbox": queue.Queue(),     # telegram utgående meddelanden
    "chat_id": None,             # senast kända chat_id (sätts via /start)
}

# KuCoin REST (public klines & tickers)
KUCOIN_SPOT_API = "https://api.kucoin.com/api/v1"

# =========================
# FastAPI
# =========================
app = FastAPI(title="ORB Bot v24")


@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK - " + APP_NAME


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.get("/state", response_class=JSONResponse)
def get_state():
    s = dict(STATE)
    s["open_positions"] = list(STATE["open_positions"].keys())
    return s


# =========================
# Hjälpfunktioner
# =========================

def now_ts() -> int:
    return int(time.time())


def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.1f}"
    if p >= 1:
        return f"{p:.2f}"
    return f"{p:.4f}"


def fmt_usdt(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f} USDT"


def minute_tf_label(tf: str) -> str:
    return {"1m": "1", "3m": "3", "5m": "5", "15m": "15"}.get(tf, "1")


def to_kucoin_symbol(s: str) -> str:
    """
    KuCoin kräver bindestreck, t.ex. BTC-USDT.
    Om s redan har '-', returneras oförändrat.
    """
    if "-" in s:
        return s
    if s.endswith("USDT"):
        return f"{s[:-4]}-USDT"
    # fallback, försök sätta bindestreck innan sista 4 tecken
    if len(s) > 4:
        return f"{s[:-4]}-{s[-4:]}"
    return s


def tg_send(msg: Dict):
    """Stoppa meddelande i outbox så TG-tråden skickar."""
    try:
        STATE["outbox"].put_nowait(msg)
    except Exception as e:
        log.warning(f"[TG] Outbox queue full? {e}")


def build_menu_kb() -> InlineKeyboardMarkup:
    btns: List[List[InlineKeyboardButton]] = []
    # Tidsram
    btns.append([
        InlineKeyboardButton("TF 1m", callback_data="tf:1m"),
        InlineKeyboardButton("TF 3m", callback_data="tf:3m"),
        InlineKeyboardButton("TF 5m", callback_data="tf:5m"),
        InlineKeyboardButton("TF 15m", callback_data="tf:15m"),
    ])
    # Entry mode
    btns.append([
        InlineKeyboardButton("Entry: Long", callback_data="entry:long"),
        InlineKeyboardButton("Entry: Short", callback_data="entry:short"),
        InlineKeyboardButton("Entry: Both", callback_data="entry:both"),
    ])
    # ORB on/off & trailing
    btns.append([
        InlineKeyboardButton("ORB ON" if STATE["orb_on"] else "ORB OFF", callback_data="orb:toggle"),
        InlineKeyboardButton("Trailing ON" if STATE["trailing"] else "Trailing OFF", callback_data="trail:toggle"),
    ])
    # PnL & panic
    btns.append([
        InlineKeyboardButton("PNL", callback_data="pnl:show"),
        InlineKeyboardButton("Reset PNL", callback_data="pnl:reset"),
        InlineKeyboardButton("PANIC", callback_data="panic:all"),
    ])
    return InlineKeyboardMarkup(btns)


def make_buy_card(symbol: str, side: str, entry_price: float, qty: float) -> str:
    tf = STATE["timeframe"]
    return (
        f"🟢 <b>BUY {symbol}</b>\n"
        f"<b>Side:</b> {side.upper()} | <b>TF:</b> {minute_tf_label(tf)}m\n"
        f"<b>Entry:</b> {fmt_price(entry_price)} | <b>Qty:</b> {qty:.6f}\n"
        f"<b>Strategy:</b> ORB + TrailingStop\n"
    )


def make_sell_card(symbol: str, exit_price: float, qty: float, trade_pnl: float, reason: str) -> str:
    return (
        f"🔴 <b>SELL {symbol}</b>\n"
        f"<b>Exit:</b> {fmt_price(exit_price)} | <b>Qty:</b> {qty:.6f}\n"
        f"<b>Reason:</b> {reason}\n"
        f"<b>Trade PnL:</b> {fmt_usdt(trade_pnl)}\n"
    )


# =========================
# KuCoin helpers
# =========================

async def kucoin_klines(symbol: str, tf: str, limit: int = 50) -> List[List]:
    """
    Hämtar klines från KuCoin.
    KuCoin tf-map: 1min/3min/5min/15min
    Returnerar sorterat äldst->nyast.
    """
    tf_map = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}
    k_tf = tf_map.get(tf, "1min")
    k_sym = to_kucoin_symbol(symbol)
    url = f"{KUCOIN_SPOT_API}/market/candles?type={k_tf}&symbol={k_sym}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        kl = data["data"][:limit]
    # KuCoin ger nyast->äldst, sortera tvärtom
    kl_sorted = sorted(kl, key=lambda x: int(x[0]))
    return kl_sorted


async def kucoin_ticker(symbol: str) -> Optional[float]:
    k_sym = to_kucoin_symbol(symbol)
    url = f"{KUCOIN_SPOT_API}/market/orderbook/level1?symbol={k_sym}"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "200000":
                return None
            price = float(data["data"]["price"])
            return price
    except Exception as e:
        log.warning(f"KuCoin ticker fail {symbol}: {e}")
        return None


# =========================
# ORB/Tradinglogik
# =========================

def compute_orb(candles: List[List]) -> Optional[Tuple[float, float]]:
    """
    Enkel ORB: använder första (äldsta) candlen i setet som bas.
    KuCoin candle: [time, open, close, high, low, volume, turnover] (alla str)
    """
    if not candles:
        return None
    first = candles[0]
    high = float(first[3])
    low = float(first[4])
    return (high, low)


def qty_for_trade(symbol: str, price: float) -> float:
    """Qty = risk_usdt / price  (mocktrade)"""
    usdt = STATE["risk_usdt"]
    if price <= 0:
        return 0.0
    return max(0.0, usdt / price)


def candle_trailing_stop_long(current_stop: float, last_low: float) -> float:
    """För long: stop ska aldrig sänkas; trailas till max(prev, last_low)"""
    return max(current_stop, last_low)


def candle_trailing_stop_short(current_stop: float, last_high: float) -> float:
    """För short: stop ska aldrig höjas; trailas till min(prev, last_high)"""
    return min(current_stop, last_high)


async def try_entries_and_manage(symbol: str):
    """
    Körs i loop: hämtar klines, beräknar ORB och hanterar entry/exit/stop.
    """
    tf = STATE["timeframe"]
    try:
        kl = await kucoin_klines(symbol, tf, limit=60)
    except Exception as e:
        log.warning(f"{symbol} klines error: {e}")
        return

    if not kl or len(kl) < 3:
        return

    # ORB
    orb = compute_orb(kl)
    if not orb:
        return
    orb_h, orb_l = orb

    # senaste och näst senaste candle
    last = kl[-1]
    prev = kl[-2]
    last_high = float(last[3])
    last_low = float(last[4])
    last_close = float(last[2])
    prev_high = float(prev[3])
    prev_low = float(prev[4])

    price = last_close

    pos = STATE["open_positions"].get(symbol)

    # Exit-hantering om position finns
    if pos:
        side = pos["side"]  # "long" / "short"
        qty = pos["qty"]
        entry = pos["entry"]
        stop = pos["stop"]
        reason_exit = None

        if STATE["trailing"]:
            if side == "long":
                stop = candle_trailing_stop_long(stop, prev_low)
            else:
                stop = candle_trailing_stop_short(stop, prev_high)
            pos["stop"] = stop

        if side == "long":
            if price <= stop:
                reason_exit = "STOP HIT"
        else:
            if price >= stop:
                reason_exit = "STOP HIT"

        if reason_exit:
            exit_price = price
            trade_pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
            STATE["realized_pnl_usdt"] += trade_pnl
            del STATE["open_positions"][symbol]

            tg_send({
                "type": "text",
                "text": make_sell_card(symbol, exit_price, qty, trade_pnl, reason_exit),
                "parse_mode": "HTML",
            })
            log.info(f"{symbol} EXIT {side} @ {fmt_price(exit_price)} pnl={trade_pnl:.2f} ({reason_exit})")
            return

    # Entry-hantering
    if STATE["orb_on"] and pos is None:
        want_long = STATE["entry_mode"] in ("both", "long")
        want_short = STATE["entry_mode"] in ("both", "short")

        # Long break
        if want_long and last_high > orb_h and last_close > orb_h:
            qty = qty_for_trade(symbol, price)
            if qty > 0:
                stop = orb_l  # direkt vid köp
                STATE["open_positions"][symbol] = {
                    "side": "long",
                    "entry": price,
                    "qty": qty,
                    "stop": stop,
                    "t_open": now_ts(),
                }
                tg_send({
                    "type": "text",
                    "text": make_buy_card(symbol, "long", price, qty),
                    "parse_mode": "HTML",
                })
                log.info(f"{symbol} ENTRY LONG @ {fmt_price(price)} stop={fmt_price(stop)} qty={qty:.6f}")

        # Short break
        elif want_short and last_low < orb_l and last_close < orb_l:
            qty = qty_for_trade(symbol, price)
            if qty > 0:
                stop = orb_h  # direkt vid sälj
                STATE["open_positions"][symbol] = {
                    "side": "short",
                    "entry": price,
                    "qty": qty,
                    "stop": stop,
                    "t_open": now_ts(),
                }
                tg_send({
                    "type": "text",
                    "text": make_buy_card(symbol, "short", price, qty),
                    "parse_mode": "HTML",
                })
                log.info(f"{symbol} ENTRY SHORT @ {fmt_price(price)} stop={fmt_price(stop)} qty={qty:.6f}")


# =========================
# Telegram-handlers
# =========================

def _remember_chat(update: Update):
    chat = update.effective_chat
    if chat:
        STATE["chat_id"] = chat.id

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    await update.message.reply_text(
        "Hej! ORB-bot v24 uppe. Använd knapparna eller kommandon.\n"
        "/menu /entry_mode /trailing /orb_on /orb_off /pnl /reset_pnl /panic\n"
        "/tf1 /tf3 /tf5 /tf15\n"
        f"(chat_id: {STATE['chat_id']})",
        reply_markup=build_menu_kb(),
    )

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    await update.message.reply_text(f"chat_id = {STATE['chat_id']}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    await update.message.reply_text("Meny:", reply_markup=build_menu_kb())

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    t = (
        f"Realized PnL: {fmt_usdt(STATE['realized_pnl_usdt'])}\n"
        f"Open positions: {len(STATE['open_positions'])}"
    )
    await update.message.reply_text(t)

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    STATE["realized_pnl_usdt"] = 0.0
    await update.message.reply_text("PnL nollställd.")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    STATE["orb_on"] = True
    await update.message.reply_text("ORB: ON", reply_markup=build_menu_kb())

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    STATE["orb_on"] = False
    await update.message.reply_text("ORB: OFF", reply_markup=build_menu_kb())

async def cmd_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    STATE["trailing"] = not STATE["trailing"]
    await update.message.reply_text(
        f"Trailing: {'ON' if STATE['trailing'] else 'OFF'}",
        reply_markup=build_menu_kb(),
    )

async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    modes = ["long", "short", "both"]
    cur = STATE["entry_mode"]
    nxt = modes[(modes.index(cur) + 1) % len(modes)]
    STATE["entry_mode"] = nxt
    await update.message.reply_text(
        f"Entry mode: {nxt.upper()}",
        reply_markup=build_menu_kb(),
    )

async def cmd_tf_generic(update: Update, context: ContextTypes.DEFAULT_TYPE, tf: str = "1m"):
    _remember_chat(update)
    STATE["timeframe"] = tf
    await update.message.reply_text(
        f"Tidsram satt till {minute_tf_label(tf)}m",
        reply_markup=build_menu_kb(),
    )

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _remember_chat(update)
    closed = 0
    for symbol, pos in list(STATE["open_positions"].items()):
        price = await kucoin_ticker(symbol) or pos["entry"]
        side = pos["side"]
        qty = pos["qty"]
        entry = pos["entry"]
        trade_pnl = (price - entry) * qty if side == "long" else (entry - price) * qty
        STATE["realized_pnl_usdt"] += trade_pnl
        del STATE["open_positions"][symbol]
        tg_send({
            "type": "text",
            "text": make_sell_card(symbol, price, qty, trade_pnl, "PANIC"),
            "parse_mode": "HTML",
        })
        closed += 1
    await update.message.reply_text(f"Panic: stängde {closed} positioner.")

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    _remember_chat(update)
    await q.answer()

    data = q.data or ""
    if data.startswith("tf:"):
        tf = data.split(":")[1]
        STATE["timeframe"] = tf
        await q.edit_message_reply_markup(build_menu_kb())
        await q.message.reply_text(f"TF = {minute_tf_label(tf)}m")
        return
    if data.startswith("entry:"):
        mode = data.split(":")[1]
        STATE["entry_mode"] = mode
        await q.edit_message_reply_markup(build_menu_kb())
        await q.message.reply_text(f"Entry mode = {mode.upper()}")
        return
    if data == "orb:toggle":
        STATE["orb_on"] = not STATE["orb_on"]
        await q.edit_message_reply_markup(build_menu_kb())
        await q.message.reply_text(f"ORB = {'ON' if STATE['orb_on'] else 'OFF'}")
        return
    if data == "trail:toggle":
        STATE["trailing"] = not STATE["trailing"]
        await q.edit_message_reply_markup(build_menu_kb())
        await q.message.reply_text(f"Trailing = {'ON' if STATE['trailing'] else 'OFF'}")
        return
    if data == "pnl:show":
        await cmd_pnl(update, context)
        return
    if data == "pnl:reset":
        await cmd_reset_pnl(update, context)
        return
    if data == "panic:all":
        await cmd_panic(update, context)
        return


# =========================
# Telegram worker (thread)
# =========================

async def tg_outbox_pumper(app: Application):
    """Skickar alla meddelanden som ligger i outbox-kön."""
    while True:
        try:
            item = STATE["outbox"].get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.3)
            continue

        try:
            if item["type"] == "text":
                chat_id = item.get("chat_id") or STATE.get("chat_id")
                if not chat_id:
                    # Ingen känd chat ännu – användaren måste skicka /start först
                    continue
                await app.bot.send_message(
                    chat_id=chat_id,
                    text=item["text"],
                    parse_mode=item.get("parse_mode"),
                    reply_markup=item.get("reply_markup"),
                    disable_web_page_preview=True,
                )
        except Exception as e:
            log.warning(f"[TG] send fail: {e}")


def start_telegram_in_thread():
    if not TELEGRAM_BOT_TOKEN:
        log.error("[TG] Ingen TELEGRAM_BOT_TOKEN – hoppar över Telegram.")
        return

    def _runner():
        # Egen event loop i thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _amain():
            app = (
                ApplicationBuilder()
                .token(TELEGRAM_BOT_TOKEN)
                .build()
            )

            # Commands
            app.add_handler(CommandHandler("start", cmd_start))
            app.add_handler(CommandHandler("id", cmd_id))
            app.add_handler(CommandHandler("menu", cmd_menu))
            app.add_handler(CommandHandler("pnl", cmd_pnl))
            app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
            app.add_handler(CommandHandler("orb_on", cmd_orb_on))
            app.add_handler(CommandHandler("orb_off", cmd_orb_off))
            app.add_handler(CommandHandler("trailing", cmd_trailing))
            app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
            app.add_handler(CommandHandler("panic", cmd_panic))
            app.add_handler(CommandHandler("tf1", lambda u, c: cmd_tf_generic(u, c, "1m")))
            app.add_handler(CommandHandler("tf3", lambda u, c: cmd_tf_generic(u, c, "3m")))
            app.add_handler(CommandHandler("tf5", lambda u, c: cmd_tf_generic(u, c, "5m")))
            app.add_handler(CommandHandler("tf15", lambda u, c: cmd_tf_generic(u, c, "15m")))
            # Buttons
            app.add_handler(CallbackQueryHandler(on_button))

            # Kör outbox-pumper parallellt
            loop.create_task(tg_outbox_pumper(app))

            # Kör polling utan signal-hantering (vi är i thread)
            await app.run_polling(drop_pending_updates=True, stop_signals=None)

        try:
            loop.run_until_complete(_amain())
        except Exception as e:
            log.error(f"[TG] thread run error: {e}")
        finally:
            try:
                loop.stop()
                loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_runner, name="telegram", daemon=True)
    t.start()


# =========================
# Trading worker
# =========================

async def trade_loop():
    """Huvudloopen som går igenom alla symboler."""
    while True:
        try:
            for symbol in SYMBOLS:
                await try_entries_and_manage(symbol)
            await asyncio.sleep(3.0)  # frekvens
        except Exception as e:
            log.warning(f"trade_loop fel: {e}")
            await asyncio.sleep(2.0)


# =========================
# Startup (FastAPI event)
# =========================

@app.on_event("startup")
async def on_startup():
    log.info("Startup: kicking workers …")
    # Starta TG
    if TELEGRAM_BOT_TOKEN:
        start_telegram_in_thread()
    else:
        log.error("[TG] TELEGRAM_BOT_TOKEN saknas – bot startas inte.")

    # Starta tradingloop i bakgrund
    loop = asyncio.get_event_loop()
    loop.create_task(trade_loop())
