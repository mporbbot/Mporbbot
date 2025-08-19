# main_v26.py
# Minimal FastAPI + ORB-mockmotor integrerad med telegram_bot.py (PTB v20)
from __future__ import annotations

import os
import time
import math
import threading
import logging
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI
from dotenv import load_dotenv

import telegram_bot

load_dotenv()  # valfritt: läs .env lokalt / på Render

log = logging.getLogger("orb_v26")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# =========================
#    Enkel mock-motor
# =========================
class ORBEngine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        # Statusfält (matchar telegram_bot.build_status_text)
        self.mode = "mock"  # "mock" eller "live"
        self.engine_on = False
        self.tf = "1m"
        self.symbols: List[str] = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "XRPUSDT"]
        self.entry = "TICK"
        self.trailing = {"on": True, "up": 0.90, "down": 0.20, "min": 0.70}
        self.keepalive = True
        self.day_pnl = -0.4407
        self.orb_master = True
        self.markets: Dict[str, Dict[str, Any]] = {
            "BTCUSDT": {"pos": True, "stop": 117551.8, "orb": True},
            "ETHUSDT": {"pos": True, "stop": 4463.1600, "orb": True},
            "ADAUSDT": {"pos": True, "stop": 0.9509, "orb": True},
            "LINKUSDT": {"pos": True, "stop": 25.4533, "orb": True},
            "XRPUSDT": {"pos": True, "stop": 3.0899, "orb": True},
        }

    # ----- Publika API:n som binds till TG-kommandon -----

    def start(self) -> str:
        with self.lock:
            if self.engine_on:
                return "Engine redan ON"
            self.engine_on = True
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        return "Engine startad"

    def stop(self) -> str:
        with self.lock:
            if not self.engine_on:
                return "Engine redan OFF"
            self.engine_on = False
            self._stop_evt.set()
        return "Engine stoppad"

    def start_mock(self) -> str:
        self.mode = "mock"
        return "Mode = MOCK"

    def start_live(self) -> str:
        self.mode = "live"
        return "Mode = LIVE"

    def toggle_entry_mode(self) -> str:
        self.entry = "CLOSE" if self.entry == "TICK" else "TICK"
        return f"Entry mode = {self.entry}"

    def toggle_trailing(self) -> str:
        self.trailing["on"] = not self.trailing.get("on", False)
        return f"Trailing = {'ON' if self.trailing['on'] else 'OFF'}"

    def pnl(self) -> str:
        return f"DayPnL: {self.day_pnl:.4f} USDT"

    def reset_pnl(self) -> str:
        self.day_pnl = 0.0
        return "DayPnL nollställd"

    def orb_on(self) -> str:
        self.orb_master = True
        return "ORB: ON"

    def orb_off(self) -> str:
        self.orb_master = False
        return "ORB: OFF"

    def panic(self) -> str:
        # Tömmer positioner i denna mock (sätter pos=False)
        for m in self.markets.values():
            m["pos"] = False
        return "PANIC: alla positioner stängda (mock)"

    def status_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "engine": self.engine_on,
            "tf": self.tf,
            "symbols": self.symbols,
            "entry": self.entry,
            "trailing": self.trailing,
            "keepalive": self.keepalive,
            "day_pnl": self.day_pnl,
            "orb_master": self.orb_master,
            "markets": self.markets,
        }

    # ----- Intern loop -----
    def _run_loop(self) -> None:
        log.info("Engine-loop startar …")
        # Enkel mock: flytta stop-nivåerna lite grann och skicka ibland signaler
        t0 = time.time()
        while not self._stop_evt.is_set():
            try:
                phase = time.time() - t0
                # Låt stopparna “vobbla” svagt
                for sym, info in self.markets.items():
                    base = info["stop"]
                    wobble = 1.0 + 0.0005 * math.sin(phase / 6.0)
                    info["stop"] = base * wobble

                # Exempel: sänd en signal då och då
                if int(phase) % 60 == 5 and self.orb_master:
                    telegram_bot.send_signal(
                        symbol="XRPUSDT",
                        side="BUY",
                        entry=2.9985,
                        stop=2.9905,
                        size=33.349563,
                        reason="CloseBreak",
                        ts_utc=datetime.utcnow(),
                    )
            except Exception as e:
                log.exception(f"Engine-loop fel: {e}")

            time.sleep(1.0)

        log.info("Engine-loop stoppad.")

# Global motor
ENGINE = ORBEngine()

# =========================
#      FastAPI-app
# =========================
app = FastAPI()

# ---- Knyt TG-kommandon till motor-funktioner ----
telegram_bot.set_callback("engine_start", ENGINE.start)
telegram_bot.set_callback("engine_stop", ENGINE.stop)
telegram_bot.set_callback("start_mock", ENGINE.start_mock)
telegram_bot.set_callback("start_live", ENGINE.start_live)
telegram_bot.set_callback("entry_mode", ENGINE.toggle_entry_mode)
telegram_bot.set_callback("toggle_trailing", ENGINE.toggle_trailing)
telegram_bot.set_callback("pnl", ENGINE.pnl)
telegram_bot.set_callback("reset_pnl", ENGINE.reset_pnl)
telegram_bot.set_callback("orb_on", ENGINE.orb_on)
telegram_bot.set_callback("orb_off", ENGINE.orb_off)
telegram_bot.set_callback("panic", ENGINE.panic)
telegram_bot.set_callback("get_status", ENGINE.status_dict)

# ---- Lifecycle ----
@app.on_event("startup")
def on_startup():
    log.info("Startup: init …")
    # Starta Telegram-boten – läser TELEGRAM_BOT_TOKEN från env internt
    telegram_bot.start_in_thread()
    log.info("[TG] start initierad.")
    log.info("Startup done.")

@app.on_event("shutdown")
def on_shutdown():
    try:
        ENGINE.stop()
    except Exception:
        pass
    log.info("Shutdown klar.")

# ---- Endpoints (enkla test) ----
@app.get("/")
def root():
    return {"ok": True, "service": "mporbbot"}

@app.get("/tg/test")
def tg_test():
    telegram_bot.send_text("Hej från API:et \\- test ✅")
    return {"sent": True}

@app.get("/tg/signal")
def tg_signal():
    telegram_bot.send_signal(
        symbol="ADAUSDT",
        side="BUY",
        entry=0.9096,
        stop=0.9067,
        size=109.938434,
        reason="CloseBreak",
        ts_utc=datetime.utcnow(),
    )
    return {"sent": True}

@app.get("/tg/status")
def tg_status():
    telegram_bot.send_status(ENGINE.status_dict())
    return {"sent": True}
