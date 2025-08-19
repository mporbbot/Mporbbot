# main_v25.py
import os
import time
import json
import math
import queue
import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

# Telegram (python-telegram-bot v20.x)
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
    MessageHandler,
    filters,
)

# =========================
# Konfiguration & logger
# =========================

APP_NAME = "orb_v25"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s " + APP_NAME + ": %(message)s",
)
log = logging.getLogger(APP_NAME)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL = os.getenv("BASE_URL", "https://mporbbot.onrender.com").rstrip("/")

# Kör KuCoin (Binance 451-blockad undviks)
KUCOIN_SPOT_API = "https://api.kucoin.com/api/v1"

# Symbols
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]

# Globalt state
STATE: Dict = {
    "orb_on": True,
    "entry_mode": "both",        # "long" | "short" | "both"
    "trailing": True,            # trailing stop aktiv
    "timeframe": "1m",           # "1m"|"3m"|"5m"|"15m"
    "risk_usdt": 30.0,           # per mocktrade
    "open_positions": {},        # symbol -> position dict
    "realized_pnl_usdt": 0.0,    # summerad mock-PnL
    "outbox": queue.Queue(),     # telegram utbox
}

# Senast kända chat_id sparas efter /start
LAST_CHAT = {"id": None}

def _mask(s: str) -> str:
    if not s:
        return "(empty)"
    if len(s) <= 10:
        return s[:3] + "…"
    return s[:6] + "…" + s[-4:]

# =========================
# FastAPI
# =========================
app = FastAPI(title="ORB Bot v25")


@app.get("/", response_class=PlainTextResponse)
def root():
    return f"OK - {APP_NAME}"


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.get("/state", response_class=JSONResponse)
def get_state():
    s = dict(STATE)
    s["open_positions"] = list(STATE["open_positions"].keys())
    s["orb_on"] = bool(STATE["orb_on"])
    s["entry_mode"] = STATE["entry_mode"]
    s["trailing"] = bool(STATE["trailing"])
    s["timeframe"] = STATE["timeframe"]
    s["last_chat_id"] = LAST_CHAT["id"]
    return s


# =========================
# Hjälpfunktioner (format)
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


def tg_send(msg: Dict):
    """Lägg meddelande i outbox – TG-tråden skickar."""
    try:
        STATE["outbox"].put_nowait(msg)
    except Exception as e:
        log.warning(f"[TG] Outbox queue full? {e}")


def build_menu_kb() -> InlineKeyboardMarkup:
    btns: List[List[InlineKeyboardButton]] = []

    btns.append([
        InlineKeyboardButton("TF 1m", callback_data="tf:1m"),
        InlineKeyboardButton("TF 3m", callback_data="tf:3m"),
        InlineKeyboardButton("TF 5m", callback_data="tf:5m"),
        InlineKeyboardButton("TF 15m", callback_data="tf:15m"),
    ])
    btns.append([
        InlineKeyboardButton("Entry: Long", callback_data="entry:long"),
        InlineKeyboardButton("Entry: Short", callback_data="entry:short"),
        InlineKeyboardButton("Entry: Both", callback_data="entry:both"),
    ])
    btns.append([
        InlineKeyboardButton("ORB ON" if STATE["orb_on"] else "ORB OFF", callback_data="orb:toggle"),
        InlineKeyboardButton("Trailing ON" if STATE["trailing"] else "Trailing OFF", callback_data="trail:toggle"),
    ])
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
    Hämtar klines från KuCoin. tf-map: 1min/3min/5min/15min
    Returnerad lista sorteras äldst->nyast.
    """
    tf_map = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}
    k_tf = tf_map.get(tf, "1min")
    url = f"{KUCOIN_SPOT_API}/market/candles?type={k_tf}&symbol={symbol}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {data}")
        kl = data["data"][:limit]
    # Format: [time, open, close, high, low, volume, turnover]
    kl_sorted = sorted(kl, key=lambda x: int(x[0]))
    return kl_sorted


async def kucoin_ticker(symbol: str) -> Optional[float]:
    url = f"{KUCOIN_SPOT_API}/market/orderbook/level1?symbol={symbol}"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "200000":
                return None
            return float(data["data"]["price"])
    except Exception as e:
        log.warning(f"KuCoin ticker fail {symbol}: {e}")
        return None


# =========================
# ORB/Tradinglogik (mock)
# =========================

def compute_orb(candles: List[List]) -> Optional[Tuple[float, float]]:
    if not candles:
        return None
    first = candles[0]  # första candle som ORB-bas (förenklad)
    high = float(first[3])
    low = float(first[4])
    return high, low


def qty_for_trade(price: float, usdt: float) -> float:
    return 0.0 if price <= 0 else max(0.0, usdt / price)


def candle_trailing_stop_long(current_stop: float, last_low: float) -> float:
    return max(current_stop, last_low)


def candle_trailing_stop_short(current_stop: float, last_high: float) -> float:
    return min(current_stop, last_high)


async def try_entries_and_manage(symbol: str):
    tf = STATE["timeframe"]
    try:
        kl = await kucoin_klines(symbol, tf, limit=60)
    except Exception as e:
        log.warning(f"{symbol} klines error: {e}")
        return

    if not kl or len(kl) < 3:
        return

    orb = compute_orb(kl)
    if not orb:
        return
    orb_h, orb_l = orb

    last = kl[-1]
    prev = kl[-2]
    last_high = float(last[3])
    last_low = float(last[4])
    last_close = float(last[2])
    prev_high = float(prev[3])
    prev_low = float(prev[4])

    price = last_close
    pos = STATE["open_positions"].get(symbol)

    # Exit om position finns
    if pos:
        side = pos["side"]
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

        if side == "long" and price <= stop:
            reason_exit = "STOP HIT"
        elif side == "short" and price >= stop:
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

    # Entry
    if STATE["orb_on"] and pos is None:
        want_long = STATE["entry_mode"] in ("both", "long")
        want_short = STATE["entry_mode"] in ("both", "short")

        if want_long and last_high > orb_h and last_close > orb_h:
            qty = qty_for_trade(price, STATE["risk_usdt"])
            if qty > 0:
                stop = orb_l
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

        elif want_short and last_low < orb_l and last_close < orb_l:
            qty = qty_for_trade(price, STATE["risk_usdt"])
            if qty > 0:
                stop = orb_h
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

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Meny:", reply_markup=build_menu_kb())


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = f"Realized PnL: {fmt_usdt(STATE['realized_pnl_usdt'])}\n" \
        f"Open positions: {len(STATE['open_positions'])}"
    await update.message.reply_text(t)


async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["realized_pnl_usdt"] = 0.0
    await update.message.reply_text("PnL nollställd.")


async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["orb_on"] = True
    await update.message.reply_text("ORB: ON", reply_markup=build_menu_kb())


async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["orb_on"] = False
    await update.message.reply_text("ORB: OFF", reply_markup=build_menu_kb())


async def cmd_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["trailing"] = not STATE["trailing"]
    await update.message.reply_text(f"Trailing: {'ON' if STATE['trailing'] else 'OFF'}",
                                    reply_markup=build_menu_kb())


async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    modes = ["long", "short", "both"]
    cur = STATE["entry_mode"]
    nxt = modes[(modes.index(cur) + 1) % len(modes)]
    STATE["entry_mode"] = nxt
    await update.message.reply_text(f"Entry mode: {nxt.upper()}",
                                    reply_markup=build_menu_kb())


async def cmd_tf_generic(update: Update, context: ContextTypes.DEFAULT_TYPE, tf: str = "1m"):
    STATE["timeframe"] = tf
    await update.message.reply_text(f"Tidsram satt till {minute_tf_label(tf)}m",
                                    reply_markup=build_menu_kb())


async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await cmd_pnl(update, context); return
    if data == "pnl:reset":
        await cmd_reset_pnl(update, context); return
    if data == "panic:all":
        await cmd_panic(update, context); return


# =========================
# Telegram worker (thread)
# =========================

async def tg_outbox_pumper(app: Application):
    """Skicka allt som ligger i outbox-kön."""
    while True:
        try:
            item = STATE["outbox"].get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.3)
            continue

        try:
            if item["type"] == "text":
                chat_id = item.get("chat_id") or LAST_CHAT["id"]
                if not chat_id:
                    # Användaren har inte kört /start ännu
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _amain():
            app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

            # --- Bas-commando för att registrera chat_id
            async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
                LAST_CHAT["id"] = update.effective_chat.id
                await update.message.reply_text(
                    "Hej! ORB-bot v25 online ✅\n"
                    "Testa /ping eller öppna /menu."
                )
                # skicka meny direkt
                await cmd_menu(update, context)

            async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text("pong 🟢")

            async def fallback(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text(
                    "Jag är online ✅ (/ping, /menu, /pnl, /orb_on, /orb_off, /tf1, /tf3, /tf5, /tf15)"
                )

            # — Dina ordinarie handlers —
            app.add_handler(CommandHandler("start", cmd_start))
            app.add_handler(CommandHandler("ping", cmd_ping))
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
            app.add_handler(CallbackQueryHandler(on_button))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback))

            # starta outbox-pumpern
            loop.create_task(tg_outbox_pumper(app))

            # logga getMe för att bekräfta token/bot
            me = await app.bot.get_me()
            log.info(f"[TG] getMe: @{me.username} id={me.id}")

            # kör polling utan signal-hantering (vi är i thread)
            await app.run_polling(drop_pending_updates=True, stop_signals=None)

        try:
            log.info(f"[TG] token: {_mask(TELEGRAM_BOT_TOKEN)}")
            loop.run_until_complete(_amain())
        except Exception as e:
            log.error(f"[TG] thread run error: {e}")
        finally:
            try:
                loop.stop()
                loop.close()
            except Exception:
                pass

    threading.Thread(target=_runner, name="telegram", daemon=True).start()


# =========================
# Trading worker
# =========================

async def trade_loop():
    while True:
        try:
            for symbol in SYMBOLS:
                await try_entries_and_manage(symbol)
            await asyncio.sleep(3.0)
        except Exception as e:
            log.warning(f"trade_loop fel: {e}")
            await asyncio.sleep(2.0)


# =========================
# Startup (FastAPI event)
# =========================

@app.on_event("startup")
async def on_startup():
    log.info("Startup: kicking workers …")
    log.info(f"[TG] token (masked): {_mask(TELEGRAM_BOT_TOKEN)}")
    if TELEGRAM_BOT_TOKEN:
        start_telegram_in_thread()
    else:
        log.error("[TG] TELEGRAM_BOT_TOKEN saknas – bot startas inte.")

    asyncio.get_event_loop().create_task(trade_loop())
