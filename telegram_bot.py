# telegram_bot.py
# PTB v20+ kompatibel Telegram-bot för ORB
# - Körs i bakgrundstråd
# - Inline-/reply keyboard med dina kommandon
# - Skicka BUY/SELL-signaler i snygg layout
# - Skicka statusöversikt i samma stil som din bild
# - Enkla "hooks" (callbacks) som main/engine kan koppla in

from __future__ import annotations

import os
import asyncio
import logging
import threading
from datetime import datetime
from typing import Callable, Dict, Optional, Any, List

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
)

log = logging.getLogger("orb_v26.TG")

# === Globalt state som main kan använda ===
STATE: Dict[str, Any] = {
    "started": False,
    "app": None,          # telegram.ext.Application
    "loop": None,         # asyncio.AbstractEventLoop (i TG-tråden)
    "chat_id": None,      # int
    "lock": threading.Lock(),
    "ready_evt": threading.Event(),
    # Callbacks som main kan sätta för att styra engine:
    "cb": {
        "engine_start": lambda: "OK",
        "engine_stop": lambda: "OK",
        "start_mock":  lambda: "OK",
        "start_live":  lambda: "OK",
        "entry_mode":  lambda: "OK",
        "toggle_trailing": lambda: "OK",
        "pnl": lambda: "0.0",
        "reset_pnl": lambda: "OK",
        "orb_on": lambda: "OK",
        "orb_off": lambda: "OK",
        "panic": lambda: "OK",
        # Ska returnera antingen färdig text (str) eller ett dict med nycklar (se build_status_text)
        "get_status": lambda: "Ingen status tillgänglig.",
    },
}

# === Layout / keyboard ===

def _menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["/status"],
        ["/engine_start", "/engine_stop"],
        ["/start_mock", "/start_live"],
        ["/entry_mode", "/trailing"],
        ["/pnl", "/reset_pnl"],
        ["/orb_on", "/orb_off"],
        ["/panic"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# === Utils ===

MDV2_SPECIALS = r"_*[]()~`>#+-=|{}.!"

def _escape_mdv2(s: str) -> str:
    return "".join("\\" + c if c in MDV2_SPECIALS else c for c in s)

def _run_coro(coro: asyncio.Future) -> None:
    """Schemalägg ett async anrop i TG-trådens loop."""
    loop = STATE.get("loop")
    if loop is None or loop.is_closed():
        log.warning("[TG] loop saknas/är stängd; kan inte skicka meddelande.")
        return
    asyncio.run_coroutine_threadsafe(coro, loop)

async def _reply(update: Update, text: str, *, mdv2: bool = True) -> None:
    if mdv2:
        text = _escape_mdv2(text)
    await update.effective_chat.send_message(
        text=text,
        parse_mode=ParseMode.MARKDOWN_V2 if mdv2 else None,
        reply_markup=_menu_keyboard()
    )

def _fmt_side(is_buy: bool) -> str:
    return "🟢 *BUY*" if is_buy else "🔴 *SELL*"

def _fmt_pair(symbol: str) -> str:
    # För konsekvent stil: t.ex. "XRPUSDT"
    return _escape_mdv2(symbol.upper())

def _fmt_num(v: float, digits: int = 4) -> str:
    try:
        return f"{v:.{digits}f}"
    except Exception:
        return str(v)


# === Publika funktioner som main/engine kan använda ===

def send_text(text: str) -> None:
    """Skicka ett enkelt textmeddelande till senast /start:ade chatten."""
    chat_id = STATE.get("chat_id")
    app = STATE.get("app")
    if not chat_id or not app:
        log.info("[TG] chat_id saknas – text ej skickad.")
        return
    _run_coro(app.bot.send_message(chat_id=chat_id, text=_escape_mdv2(text), parse_mode=ParseMode.MARKDOWN_V2))

def send_signal(
    *,
    symbol: str,
    side: str,                 # "BUY" eller "SELL"
    entry: float,
    stop: float,
    size: float,
    reason: str,
    ts_utc: Optional[datetime] = None
) -> None:
    """Skicka signal i layout som på din screenshot."""
    chat_id = STATE.get("chat_id")
    app = STATE.get("app")
    if not chat_id or not app:
        log.info("[TG] chat_id saknas – signal ej skickad.")
        return

    is_buy = side.upper() == "BUY"
    head = f"{_fmt_side(is_buy)} {_fmt_pair(symbol)}"
    tstamp = ts_utc or datetime.utcnow()
    # %-diff mellan entry och stop
    try:
        pct = (stop / entry - 1.0) * 100.0
    except Exception:
        pct = 0.0

    body = (
        f"{head}\n"
        f"time={tstamp.strftime('%H:%M:%S')} UTC\n"
        f"entry={_fmt_num(entry, 4)}  stop={_fmt_num(stop, 4)} ({_fmt_num(pct, 2)}\\%)\n"
        f"size={_fmt_num(size, 6)}\n"
        f"reason={_escape_mdv2(reason)}"
    )
    _run_coro(app.bot.send_message(
        chat_id=chat_id,
        text=body,
        parse_mode=ParseMode.MARKDOWN_V2
    ))

def send_status(status: Any) -> None:
    """Skicka status; tar str eller dict. Dict byggs till samma layout som din bild."""
    chat_id = STATE.get("chat_id")
    app = STATE.get("app")
    if not chat_id or not app:
        log.info("[TG] chat_id saknas – status ej skickad.")
        return

    if isinstance(status, str):
        text = status
    else:
        text = build_status_text(status)

    _run_coro(app.bot.send_message(
        chat_id=chat_id,
        text=_escape_mdv2(text),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=_menu_keyboard()
    ))

def build_status_text(s: Dict[str, Any]) -> str:
    """
    Bygger statuslayouten (som på bilden) från ett dict.
    Förväntade nycklar (exempel):
        {
          "mode": "mock" / "live",
          "engine": True/False,
          "tf": "1m",
          "symbols": ["BTCUSDT","ETHUSDT",...],
          "entry": "TICK" or "CLOSE",
          "trailing": {"on": True, "up": 0.90, "down": 0.20, "min": 0.70},
          "keepalive": True,
          "day_pnl": -0.4407,
          "orb_master": True,
          "markets": {
             "BTCUSDT": {"pos": True, "stop": 117551.8, "orb": True},
             ...
          }
        }
    """
    parts: List[str] = []
    mode = s.get("mode", "-")
    engine = "ON" if s.get("engine") else "OFF"
    tf = s.get("tf", "-")
    symbols = ",".join(s.get("symbols", []))
    entry = s.get("entry", "-")

    tr = s.get("trailing", {})
    if isinstance(tr, dict) and tr.get("on"):
        trail_txt = f"ON ({_fmt_num(tr.get('up', 0),2)}\\%/{_fmt_num(tr.get('down',0),2)}\\% min {_fmt_num(tr.get('min',0),2)}\\%)"
    else:
        trail_txt = "OFF"

    keepalive = "ON" if s.get("keepalive") else "OFF"
    day_pnl = s.get("day_pnl", 0.0)
    orb_master = "ON" if s.get("orb_master") else "OFF"

    header = (
        f"Mode: {mode}   Engine: {engine}\n"
        f"TF: {tf}   Symbols:\n"
        f"{symbols}\n"
        f"Entry: {entry}   Trail: {trail_txt}\n"
        f"Keepalive: {keepalive}   DayPnL: {_fmt_num(day_pnl,4)} USDT\n"
        f"ORB master: {orb_master}"
    )
    parts.append(header)

    markets = s.get("markets", {})
    for sym, info in markets.items():
        pos = "✅" if info.get("pos") else "❌"
        stop = info.get("stop", "-")
        orb = "ON" if info.get("orb") else "OFF"
        parts.append(f"{sym}: pos={pos} stop={_fmt_num(stop,4)} | ORB: {orb}")

    return "\n".join(parts)


# === Kommandon ===

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with STATE["lock"]:
        STATE["chat_id"] = update.effective_chat.id

    welcome = (
        "Hej! Du är nu kopplad till ORB-boten.\n"
        "Jag skickar signaler till denna chatt.\n\n"
        "Kommandon: /menu /id"
    )
    await _reply(update, welcome)

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(update, "Meny")

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(update, f"chat_id = {update.effective_chat.id}", mdv2=False)

# Följande kommandon mappar till callbacks i STATE["cb"]:

async def _call_cb(update: Update, key: str, ok_text: str) -> None:
    cb = STATE["cb"].get(key)
    try:
        res = cb() if cb else "N/A"
    except Exception as e:
        res = f"Fel: {e}"
    await _reply(update, f"{ok_text}\n{res}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cb = STATE["cb"].get("get_status")
    data = None
    try:
        data = cb() if cb else "Ingen status."
    except Exception as e:
        data = f"Fel vid status: {e}"
    # Skicka:
    if isinstance(data, str):
        await _reply(update, data)
    else:
        await _reply(update, build_status_text(data))

async def cmd_engine_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "engine_start", "Engine START")

async def cmd_engine_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "engine_stop", "Engine STOP")

async def cmd_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "start_mock", "Mode = MOCK")

async def cmd_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "start_live", "Mode = LIVE")

async def cmd_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "entry_mode", "Entry mode togglad")

async def cmd_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "toggle_trailing", "Trailing togglad")

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "pnl", "PnL")

async def cmd_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "reset_pnl", "DayPnL nollställd")

async def cmd_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "orb_on", "ORB: ON")

async def cmd_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "orb_off", "ORB: OFF")

async def cmd_panic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _call_cb(update, "panic", "PANIC skickad")

def set_callback(name: str, func: Callable[[], Any]) -> None:
    """Main kan registrera sina funktioner här."""
    STATE["cb"][name] = func

# === Start / Stop i bakgrundstråd ===

def _add_handlers(app):
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("id", cmd_id))

    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("engine_start", cmd_engine_start))
    app.add_handler(CommandHandler("engine_stop", cmd_engine_stop))
    app.add_handler(CommandHandler("start_mock", cmd_start_mock))
    app.add_handler(CommandHandler("start_live", cmd_start_live))
    app.add_handler(CommandHandler("entry_mode", cmd_entry_mode))
    app.add_handler(CommandHandler("trailing", cmd_trailing))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
    app.add_handler(CommandHandler("orb_on", cmd_orb_on))
    app.add_handler(CommandHandler("orb_off", cmd_orb_off))
    app.add_handler(CommandHandler("panic", cmd_panic))

def _thread_target(token: str):
    """Kör Telegram-appen i egen asyncio-loop i denna tråd."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        app = ApplicationBuilder().token(token).build()
        _add_handlers(app)

        STATE["app"] = app
        STATE["loop"] = loop

        async def run():
            await app.initialize()
            await app.start()
            # PTB v20: använd updater.start_polling (ingen .idle)
            await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            log.info("[TG] started.")
            STATE["started"] = True
            STATE["ready_evt"].set()
            # Kör tills någon stoppar loop/event
            await asyncio.Event().wait()

        loop.run_until_complete(run())

    except Exception as e:
        log.exception(f"[TG] thread exception: {e}")
    finally:
        try:
            if STATE.get("app"):
                loop = STATE.get("loop")
                async def _stop():
                    try:
                        await STATE["app"].updater.stop()
                    except Exception:
                        pass
                    await STATE["app"].stop()
                    await STATE["app"].shutdown()
                if loop and not loop.is_closed():
                    loop.run_until_complete(_stop())
        except Exception:
            pass

def start_in_thread() -> None:
    """Starta boten. Läser token från TELEGRAM_BOT_TOKEN."""
    with STATE["lock"]:
        if STATE["started"]:
            return
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN saknas i env.")
        t = threading.Thread(target=_thread_target, args=(token,), daemon=True)
        t.start()

def set_chat_id(chat_id: int) -> None:
    """Om du vill hårdkoda chat_id från main (valfritt)."""
    with STATE["lock"]:
        STATE["chat_id"] = chat_id
