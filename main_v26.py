import os
import inspect
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# ---------------------------------------------------------
# Loggning
# ---------------------------------------------------------
log = logging.getLogger("orb_v26")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------
# Globalt state (endast outbox för Telegram)
# ---------------------------------------------------------
STATE = {
    "outbox": None,   # sätts i startup till en queue som telegram_bot kan använda
}

try:
    import asyncio
    STATE["outbox"] = asyncio.Queue()
except Exception:
    # Fallback om något udda händer innan event loop finns
    STATE["outbox"] = None

# ---------------------------------------------------------
# FastAPI-app
# ---------------------------------------------------------
app = FastAPI(title="MporbBot", version="v26")

@app.get("/", response_class=PlainTextResponse)
async def root() -> str:
    return "OK"

@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "healthy"

# ---------------------------------------------------------
# Hjälpare för att starta/stoppa Telegram-modulen
# ---------------------------------------------------------
_TELEGRAM_STARTED = False
_telegram_mod = None

async def _start_telegram_if_configured() -> None:
    """Starta telegram_bot om token finns och modulens API finns."""
    global _TELEGRAM_STARTED, _telegram_mod

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        log.error("[TG] TELEGRAM_BOT_TOKEN saknas – Telegram startas inte.")
        return

    # Flagga för att kunna slå av via env om man vill
    if os.getenv("DISABLE_TELEGRAM", "").lower() in {"1", "true", "yes"}:
        log.info("[TG] Inaktiverad via DISABLE_TELEGRAM.")
        return

    try:
        import telegram_bot  # din separata modul
        _telegram_mod = telegram_bot

        # Ge modulen outbox-kön om funktionen finns
        if hasattr(telegram_bot, "set_outbox") and STATE["outbox"] is not None:
            try:
                telegram_bot.set_outbox(STATE["outbox"])
            except Exception as e:
                log.warning(f"[TG] set_outbox fel: {e}")

        # Hämta start-funktionen
        if not hasattr(telegram_bot, "start_in_thread"):
            log.error("[TG] Hittar inte start_in_thread i telegram_bot.")
            return

        start_fn = telegram_bot.start_in_thread

        # Kör oavsett om den är async eller synkron
        if inspect.iscoroutinefunction(start_fn):
            await start_fn()
        else:
            start_fn()

        _TELEGRAM_STARTED = True
        log.info("[TG] started.")
    except Exception as e:
        log.error(f"[TG] startup error: {e}")

async def _stop_telegram_if_started() -> None:
    """Försök stoppa telegram_bot om modulen exponerar stop-funktion."""
    global _TELEGRAM_STARTED, _telegram_mod
    if not _TELEGRAM_STARTED or _telegram_mod is None:
        return
    try:
        if hasattr(_telegram_mod, "stop"):
            stop_fn = getattr(_telegram_mod, "stop")
            if inspect.iscoroutinefunction(stop_fn):
                await stop_fn()
            else:
                stop_fn()
        log.info("[TG] stopped.")
    except Exception as e:
        log.warning(f"[TG] stop warning: {e}")
    finally:
        _TELEGRAM_STARTED = False
        _telegram_mod = None

# ---------------------------------------------------------
# FastAPI lifecycle hooks
# ---------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    log.info("Startup: init …")
    await _start_telegram_if_configured()
    log.info("Startup done.")

@app.on_event("shutdown")
async def on_shutdown() -> None:
    log.info("Application is stopping. This might take a moment.")
    await _stop_telegram_if_started()
