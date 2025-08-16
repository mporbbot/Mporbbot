import os
import time
import hmac
import json
import base64
import hashlib
import logging
import threading
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from uvicorn import Config, Server

# Telegram v13
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ParseMode,
)
from telegram.ext import (
    Updater,
    CallbackContext,
    CommandHandler,
    CallbackQueryHandler,
    Filters,
    MessageHandler,
)

# ==========
# Logging
# ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mporbbot")

# ==========
# FastAPI (health/keepalive)
# ==========
app = FastAPI()

class HealthResp(BaseModel):
    ok: bool
    ts: str

@app.get("/")
def root():
    return {"ok": True, "msg": "Mporbbot online"}

@app.get("/health", response_model=HealthResp)
def health():
    return HealthResp(ok=True, ts=dt.datetime.utcnow().isoformat())

# ==========
# Config & State
# ==========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
KUCOIN_KEY = os.getenv("KUCOIN_API_KEY", "").strip()
KUCOIN_SECRET = os.getenv("KUCOIN_API_SECRET", "").strip()
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "").strip()
MOCK_TRADE_USDT = float(os.getenv("MOCK_TRADE_USDT", "30"))
KEEPALIVE_URL = os.getenv("KEEPALIVE_URL", "").strip()

if not TELEGRAM_TOKEN:
    log.warning("TELEGRAM_TOKEN saknas — botten startar ändå men Telegram fungerar inte.")

KU_PUBLIC = "https://api.kucoin.com"

TIMEFRAME_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
}

ENTRY_MODES = ("tick", "close")  # när vi triggar logiken: varje pris-tick, eller när en candle stänger

@dataclass
class Trade:
    symbol: str
    side: str  # "buy" / "sell"
    qty: float
    price: float
    ts: dt.datetime
    mode: str  # "mock" / "live"
    fee: float = 0.0
    pnl: float = 0.0  # realiserad på exit
    note: str = ""

@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    stop_loss: float
    entry_candle_low: float
    highest: float = 0.0
    opened_at: dt.datetime = field(default_factory=lambda: dt.datetime.utcnow())

@dataclass
class SymbolState:
    symbol: str
    in_position: bool = False
    pos: Optional[Position] = None
    last_candle_close_time: Optional[int] = None  # unix ms
    last_color: Optional[str] = None  # "red" / "green"
    # PnL (mock & live)
    pnl_mock: float = 0.0
    pnl_live: float = 0.0

@dataclass
class EngineConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "ADAUSDT"])
    timeframe: str = "1m"  # 1m|3m|5m
    entry_mode: str = "close"  # "tick" or "close"
    trail_trigger: float = 0.0  # ex 0.009 (0.9%)
    trail_offset: float = 0.0   # ex 0.002 (0.2%)
    min_lock: float = 0.0       # min “låst” vinst
    mode: str = "mock"          # "mock" eller "live"
    engine_on: bool = False
    keepalive_on: bool = False

class Engine:
    def __init__(self):
        self.cfg = EngineConfig()
        self.state: Dict[str, SymbolState] = {s: SymbolState(s) for s in self.cfg.symbols}
        self.trades: List[Trade] = []  # alla exekverade trades (mock+live)
        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._price_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None

    # ------------- KuCoin helpers -------------
    @staticmethod
    def _kc_headers(method: str, endpoint: str, body: dict = None) -> Dict[str, str]:
        now = int(time.time() * 1000)
        body_str = json.dumps(body) if body else ""
        str_to_sign = f"{now}{method}{endpoint}{body_str}"
        sign = base64.b64encode(
            hmac.new(KUCOIN_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).digest()
        ).decode()
        passphrase = base64.b64encode(
            hmac.new(KUCOIN_SECRET.encode(), KUCOIN_PASSPHRASE.encode(), hashlib.sha256).digest()
        ).decode()
        return {
            "KC-API-KEY": KUCOIN_KEY,
            "KC-API-SIGN": sign,
            "KC-API-TIMESTAMP": str(now),
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    def kc_get(self, endpoint: str, params: dict = None):
        url = KU_PUBLIC + endpoint
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {d}")
        return d["data"]

    def kc_private(self, method: str, endpoint: str, body: dict = None):
        url = KU_PUBLIC + endpoint
        headers = self._kc_headers(method, endpoint, body)
        r = requests.request(method, url, data=json.dumps(body) if body else None, headers=headers, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {d}")
        return d["data"]

    def ku_symbol(self, s: str) -> str:
        # "ADAUSDT" -> "ADA-USDT"
        if "-" in s:
            return s.upper()
        base = s[:-4]
        quote = s[-4:]
        return f"{base}-{quote}"

    # ------------- Market Data -------------
    def last_price(self, symbol: str) -> float:
        ku = self.ku_symbol(symbol)
        data = self.kc_get("/api/v1/market/orderbook/level1", {"symbol": ku})
        return float(data["price"])

    def klines(self, symbol: str, timeframe: str, limit: int = 3) -> List[Tuple[int,float,float,float,float]]:
        """
        Returnerar lista av (ts_ms, open, close, high, low) för senaste 'limit' candles.
        """
        ku = self.ku_symbol(symbol)
        t = TIMEFRAME_MAP.get(timeframe, "1min")
        data = self.kc_get("/api/v1/market/candles", {"type": t, "symbol": ku})
        # KuCoin candles: [time, open, close, high, low, volume, turnover]
        out = []
        for row in data[:limit]:
            ts_sec = int(row[0])
            o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
            out.append((ts_sec*1000, o, c, h, l))
        out.sort(key=lambda x: x[0])  # äldst -> nyast
        return out

    # ------------- Trading (mock + live) -------------
    def _mock_buy(self, s: SymbolState, usdt: float, price: float):
        qty = round(usdt / price, 6)
        s.in_position = True
        pos = Position(symbol=s.symbol, entry_price=price, qty=qty,
                       stop_loss=price, entry_candle_low=price, highest=price)
        s.pos = pos
        self.trades.append(Trade(symbol=s.symbol, side="buy", qty=qty, price=price, ts=dt.datetime.utcnow(), mode="mock"))
        log.info(f"[MOCK] BUY {s.symbol} qty={qty} @ {price:.6f}")

    def _mock_sell(self, s: SymbolState, price: float, reason: str):
        if not s.in_position or not s.pos:
            return
        pnl = (price - s.pos.entry_price) * s.pos.qty
        s.pnl_mock += pnl
        self.trades.append(Trade(symbol=s.symbol, side="sell", qty=s.pos.qty, price=price,
                                 ts=dt.datetime.utcnow(), mode="mock", pnl=pnl, note=reason))
        log.info(f"[MOCK] SELL {s.symbol} qty={s.pos.qty} @ {price:.6f}  pnl={pnl:.6f} ({reason})")
        s.pos = None
        s.in_position = False

    def _live_buy(self, s: SymbolState, usdt: float, price_hint: float):
        # Market buy: specify "funds" in quote (USDT)
        if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
            raise RuntimeError("KuCoin API-nycklar saknas.")
        ku = self.ku_symbol(s.symbol)
        order = {
            "clientOid": f"buy-{s.symbol}-{int(time.time()*1000)}",
            "side": "buy",
            "symbol": ku,
            "type": "market",
            "funds": str(round(usdt, 2)),
        }
        res = self.kc_private("POST", "/api/v1/orders", order)
        # Vi antar fill nära hint-priset – uppdatera pos med faktisk qty/price via ticker
        px = self.last_price(s.symbol) if price_hint <= 0 else price_hint
        qty = round(usdt / px, 6)
        s.in_position = True
        s.pos = Position(symbol=s.symbol, entry_price=px, qty=qty,
                         stop_loss=px, entry_candle_low=px, highest=px)
        self.trades.append(Trade(symbol=s.symbol, side="buy", qty=qty, price=px,
                                 ts=dt.datetime.utcnow(), mode="live"))
        log.info(f"[LIVE] BUY {s.symbol} funds={usdt} est_qty={qty} @ ~{px:.6f} (orderId={res.get('orderId')})")

    def _live_sell(self, s: SymbolState, price_hint: float, reason: str):
        if not s.in_position or not s.pos:
            return
        if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
            raise RuntimeError("KuCoin API-nycklar saknas.")
        ku = self.ku_symbol(s.symbol)
        # Market sell: specify size (base)
        order = {
            "clientOid": f"sell-{s.symbol}-{int(time.time()*1000)}",
            "side": "sell",
            "symbol": ku,
            "type": "market",
            "size": str(s.pos.qty),
        }
        res = self.kc_private("POST", "/api/v1/orders", order)
        px = self.last_price(s.symbol) if price_hint <= 0 else price_hint
        pnl = (px - s.pos.entry_price) * s.pos.qty
        s.pnl_live += pnl
        self.trades.append(Trade(symbol=s.symbol, side="sell", qty=s.pos.qty, price=px,
                                 ts=dt.datetime.utcnow(), mode="live", pnl=pnl, note=reason))
        log.info(f"[LIVE] SELL {s.symbol} size={s.pos.qty} @ ~{px:.6f} pnl={pnl:.6f} (orderId={res.get('orderId')})")
        s.pos = None
        s.in_position = False

    # ------------- Strategy (ORB long-only) -------------
    def _update_trailing_sl(self, s: SymbolState, last_close: float, last_low: float):
        """
        1) Bas: SL följer varje ny candle-low upp (men aldrig ned).
        2) Extra skydd: om vinsten >= trail_trigger, lås in vinst med trail_offset
           (t.ex. trig=0.9%, offset=0.2% => minst 0.7% låst).
        """
        if not s.in_position or not s.pos:
            return
        pos = s.pos

        # Följa candle-low:
        new_sl = max(pos.stop_loss, last_low)
        pos.stop_loss = new_sl

        # Profit-based trailing (valfri)
        if self.cfg.trail_trigger > 0 and self.cfg.trail_offset > 0:
            up = (last_close / pos.entry_price) - 1.0
            if up >= self.cfg.trail_trigger:
                # sätt SL till last_close * (1 - offset), men aldrig lägre än befintlig
                prof_sl = last_close * (1.0 - self.cfg.trail_offset)
                pos.stop_loss = max(pos.stop_loss, prof_sl)

        pos.highest = max(pos.highest, last_close)

    def _maybe_enter(self, s: SymbolState, last_o: float, last_c: float, last_h: float, last_l: float):
        # Vi går LONG när senaste stängda candle är grön och föregående var röd (switch).
        # Entry sker på "close"-mode direkt på sista close (simulerat) eller vid tick om entry_mode="tick".
        if s.in_position:
            return

        # Behöver förra färgen:
        if s.last_color != "red":
            return

        # Senaste stängda är grön?
        if last_c > last_o:
            price = last_c
            if self.cfg.mode == "mock":
                self._mock_buy(s, MOCK_TRADE_USDT, price)
            else:
                self._live_buy(s, MOCK_TRADE_USDT, price)
            # Direkt stop placeras vid entry-candle low:
            if s.pos:
                s.pos.stop_loss = last_l
                s.pos.entry_candle_low = last_l

    def _maybe_exit(self, s: SymbolState, last_close: float):
        if not s.in_position or not s.pos:
            return
        pos = s.pos
        # Exit om priset passerar stop_loss
        if last_close <= pos.stop_loss:
            reason = f"SL hit @ {pos.stop_loss:.6f}"
            if self.cfg.mode == "mock":
                self._mock_sell(s, pos.stop_loss, reason)
            else:
                self._live_sell(s, pos.stop_loss, reason)

    def _update_symbol(self, s: SymbolState):
        try:
            kl = self.klines(s.symbol, self.cfg.timeframe, limit=3)  # få minst två senaste stängda
            if len(kl) < 2:
                return
            # näst sista = senast stängda candle
            prev_ts, prev_o, prev_c, prev_h, prev_l = kl[-2]
            last_ts, last_o, last_c, last_h, last_l = kl[-1]

            # färg på senast stängda:
            prev_color = "green" if prev_c > prev_o else "red" if prev_c < prev_o else "doji"

            # uppdatera trailing och ev. exit baserat på senast stängda candle:
            if s.in_position:
                self._update_trailing_sl(s, prev_c, prev_l)
                self._maybe_exit(s, prev_c)

            # ORB-entry (red -> green switch):
            if self.cfg.entry_mode == "close":
                if s.last_color == "red" and prev_color == "green":
                    self._maybe_enter(s, prev_o, prev_c, prev_h, prev_l)

            # Spara färg för nästa varv
            s.last_color = prev_color
            s.last_candle_close_time = prev_ts
        except Exception as e:
            log.exception(f"update_symbol({s.symbol}) error: {e}")

    # ------------- Threads -------------
    def _price_loop(self):
        log.info("Engine price loop started.")
        while not self._stop_evt.is_set():
            if self.cfg.engine_on:
                with self._lock:
                    for sym in self.cfg.symbols:
                        self._update_symbol(self.state[sym])
            # poll-frekvens ~ 2 sekunder för 1m klines är lagom
            time.sleep(2)
        log.info("Engine price loop stopped.")

    def _keepalive_loop(self):
        log.info("Keepalive loop started.")
        while not self._stop_evt.is_set():
            if self.cfg.keepalive_on:
                try:
                    if KEEPALIVE_URL:
                        requests.get(KEEPALIVE_URL, timeout=5)
                    else:
                        # Pinga FastAPI lokalt
                        requests.get("http://127.0.0.1:10000/health", timeout=5)
                except Exception as e:
                    log.debug(f"keepalive ping misslyckades: {e}")
            time.sleep(120)  # varannan minut
        log.info("Keepalive loop stopped.")

    def start(self):
        self._stop_evt.clear()
        if not self._price_thread or not self._price_thread.is_alive():
            self._price_thread = threading.Thread(target=self._price_loop, daemon=True)
            self._price_thread.start()
        if not self._keepalive_thread or not self._keepalive_thread.is_alive():
            self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
            self._keepalive_thread.start()

    def stop(self):
        self._stop_evt.set()

ENGINE = Engine()
ENGINE.start()

# ==========
# Telegram bot
# ==========
HELP_TEXT = (
    "/status\n"
    "/engine_start  /engine_stop\n"
    "/start_mock    /start_live\n"
    "/symbols BTCUSDT,ETHUSDT,ADAUSDT\n"
    "/timeframe\n"
    "/entry_mode tick|close\n"
    "/trailing  (knappar för att sätta/av)\n"
    "/pnl   /reset_pnl  /export_csv  /export_k4\n"
    "/keepalive_on  /keepalive_off\n"
    "/panic\n"
)

def fmt_status() -> str:
    e = ENGINE
    lines = []
    lines.append(f"Mode: <b>{e.cfg.mode}</b>    Engine: <b>{'ON' if e.cfg.engine_on else 'OFF'}</b>")
    lines.append(f"TF: <b>{e.cfg.timeframe}</b>   Entry: <b>{e.cfg.entry_mode}</b>")
    lines.append(
        f"Trail: trig <b>{e.cfg.trail_trigger*100:.2f}%</b> | avst <b>{e.cfg.trail_offset*100:.2f}%</b> | min <b>{e.cfg.min_lock*100:.2f}%</b>"
    )
    lines.append(f"Keepalive: <b>{'ON' if e.cfg.keepalive_on else 'OFF'}</b>")
    lines.append(f"Symbols: {', '.join(e.cfg.symbols)}")
    # PnL summering
    pnlm = sum(s.pnl_mock for s in e.state.values())
    pnll = sum(s.pnl_live for s in e.state.values())
    lines.append(f"PnL → MOCK <b>{pnlm:.4f}</b> | LIVE <b>{pnll:.4f}</b>")
    for s in e.state.values():
        if s.in_position and s.pos:
            lines.append(
                f"{s.symbol}: pos=✅  entry={s.pos.entry_price:.6f}  SL={s.pos.stop_loss:.6f}"
            )
        else:
            lines.append(f"{s.symbol}: pos=❌")
    return "\n".join(lines)

def start_cmd(update: Update, ctx: CallbackContext):
    update.message.reply_text("Mporbbot klar. Använd /help för kommandon.")

def help_cmd(update: Update, ctx: CallbackContext):
    update.message.reply_text(HELP_TEXT)

def status_cmd(update: Update, ctx: CallbackContext):
    update.message.reply_text(fmt_status(), parse_mode=ParseMode.HTML)

def engine_start_cmd(update: Update, ctx: CallbackContext):
    ENGINE.cfg.engine_on = True
    update.message.reply_text("Engine: ON")

def engine_stop_cmd(update: Update, ctx: CallbackContext):
    ENGINE.cfg.engine_on = False
    update.message.reply_text("Engine: OFF")

def start_mock_cmd(update: Update, ctx: CallbackContext):
    ENGINE.cfg.mode = "mock"
    update.message.reply_text("Mock-läge: AKTIVERAT")

def start_live_cmd(update: Update, ctx: CallbackContext):
    ENGINE.cfg.mode = "live"
    update.message.reply_text("LIVE-läge: AKTIVERAT (KuCoin)")

def symbols_cmd(update: Update, ctx: CallbackContext):
    if ctx.args:
        syms = "".join(ctx.args).replace(" ", "")
        new = [s.strip().upper() for s in syms.split(",") if s.strip()]
        ENGINE.cfg.symbols = new
        ENGINE.state = {s: ENGINE.state.get(s, SymbolState(s)) for s in new}
        update.message.reply_text(f"Symbols uppdaterade: {', '.join(new)}")
    else:
        update.message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,ADAUSDT")

def timeframe_buttons():
    kb = [
        [InlineKeyboardButton("1m", callback_data="tf:1m"),
         InlineKeyboardButton("3m", callback_data="tf:3m"),
         InlineKeyboardButton("5m", callback_data="tf:5m")]
    ]
    return InlineKeyboardMarkup(kb)

def timeframe_cmd(update: Update, ctx: CallbackContext):
    update.message.reply_text("Välj tidsram:", reply_markup=timeframe_buttons())

def entry_mode_cmd(update: Update, ctx: CallbackContext):
    kb = [[
        InlineKeyboardButton("tick", callback_data="em:tick"),
        InlineKeyboardButton("close", callback_data="em:close"),
    ]]
    update.message.reply_text("Välj entry_mode:", reply_markup=InlineKeyboardMarkup(kb))

def trailing_cmd(update: Update, ctx: CallbackContext):
    kb = [
        [
            InlineKeyboardButton("Av (0%)", callback_data="tr:0:0"),
            InlineKeyboardButton("0.9% / 0.2%", callback_data="tr:0.009:0.002"),
            InlineKeyboardButton("1.5% / 0.5%", callback_data="tr:0.015:0.005"),
        ]
    ]
    update.message.reply_text("Välj trailing:", reply_markup=InlineKeyboardMarkup(kb))

def pnl_cmd(update: Update, ctx: CallbackContext):
    e = ENGINE
    pnlm = sum(s.pnl_mock for s in e.state.values())
    pnll = sum(s.pnl_live for s in e.state.values())
    lines = [f"PnL MOCK: {pnlm:.6f}", f"PnL LIVE: {pnll:.6f}"]
    update.message.reply_text("\n".join(lines))

def reset_pnl_cmd(update: Update, ctx: CallbackContext):
    for s in ENGINE.state.values():
        s.pnl_mock = 0.0
        s.pnl_live = 0.0
    update.message.reply_text("Dagens PnL nollställd.")

def export_csv_cmd(update: Update, ctx: CallbackContext):
    # Dagens trades i CSV
    today = dt.datetime.utcnow().date()
    rows = ["ts,symbol,mode,side,qty,price,pnl,note"]
    for t in ENGINE.trades:
        if t.ts.date() == today:
            rows.append(
                f"{t.ts.isoformat()},{t.symbol},{t.mode},{t.side},{t.qty:.6f},{t.price:.6f},{t.pnl:.6f},{t.note}"
            )
    csv = "\n".join(rows) if len(rows) > 1 else "ts,symbol,mode,side,qty,price,pnl,note"
    update.message.reply_document(("trades_today.csv", csv.encode("utf-8")))

def export_k4_cmd(update: Update, ctx: CallbackContext):
    """
    Enkel K4-liknande export: varje realiserad affär (sell) som rad.
    Kolumner (exempel): Datum, Typ, Beteckning, Antal, Försäljningspris, Anskaffningsutgift, Vinst/Förlust
    """
    rows = ["Datum,Typ,Beteckning,Antal,Försäljningspris,Anskaffningsutgift,Vinst/Förlust,Notering"]
    for t in ENGINE.trades:
        if t.side != "sell":
            continue
        # hitta entry för denna SELL (enkel matchning: närmast tidigare BUY samma symbol)
        entry = None
        for tt in reversed(ENGINE.trades):
            if tt.symbol == t.symbol and tt.side == "buy" and tt.ts <= t.ts:
                entry = tt
                break
        qty = t.qty
        sell_total = t.price * qty
        buy_total = (entry.price * qty) if entry else 0.0
        pnl = sell_total - buy_total
        rows.append(
            f"{t.ts.date().isoformat()},Krypto,{t.symbol},{qty:.6f},{sell_total:.6f},{buy_total:.6f},{pnl:.6f},{t.note}"
        )
    csv = "\n".join(rows)
    update.message.reply_document(("k4_export.csv", csv.encode("utf-8")))

def keepalive_on_cmd(update: Update, ctx: CallbackContext):
    ENGINE.cfg.keepalive_on = True
    update.message.reply_text("Keepalive: ON (ping varannan minut)")

def keepalive_off_cmd(update: Update, ctx: CallbackContext):
    ENGINE.cfg.keepalive_on = False
    update.message.reply_text("Keepalive: OFF")

def panic_cmd(update: Update, ctx: CallbackContext):
    # stäng alla positioner
    for s in ENGINE.state.values():
        if s.in_position and s.pos:
            px = ENGINE.last_price(s.symbol)
            if ENGINE.cfg.mode == "mock":
                ENGINE._mock_sell(s, px, "panic")
            else:
                ENGINE._live_sell(s, px, "panic")
    ENGINE.cfg.engine_on = False
    update.message.reply_text("PANIC: Alla positioner stängda. Engine OFF.")

def button_cb(update: Update, ctx: CallbackContext):
    query = update.callback_query
    data = query.data or ""
    try:
        if data.startswith("tf:"):
            tf = data.split(":")[1]
            ENGINE.cfg.timeframe = tf
            query.answer(f"Timeframe = {tf}")
            query.edit_message_text(f"Timeframe satt: {tf}")
        elif data.startswith("em:"):
            em = data.split(":")[1]
            ENGINE.cfg.entry_mode = em
            query.answer(f"Entry mode = {em}")
            query.edit_message_text(f"Entry mode satt: {em}")
        elif data.startswith("tr:"):
            _, trig, off = data.split(":")
            ENGINE.cfg.trail_trigger = float(trig)
            ENGINE.cfg.trail_offset = float(off)
            ENGINE.cfg.min_lock = max(0.0, ENGINE.cfg.trail_trigger - ENGINE.cfg.trail_offset)
            query.answer("Trailing uppdaterad")
            query.edit_message_text(
                f"Trailing satt: trig={ENGINE.cfg.trail_trigger*100:.2f}%  "
                f"avst={ENGINE.cfg.trail_offset*100:.2f}%  "
                f"min={ENGINE.cfg.min_lock*100:.2f}%"
            )
    except Exception as e:
        query.answer("Fel")
        log.exception(e)

def unknown_cmd(update: Update, ctx: CallbackContext):
    update.message.reply_text("Okänt kommando. /help")

def build_updater() -> Updater:
    upd = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = upd.dispatcher

    dp.add_handler(CommandHandler("start", start_cmd))
    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(CommandHandler("status", status_cmd))

    dp.add_handler(CommandHandler("engine_start", engine_start_cmd))
    dp.add_handler(CommandHandler("engine_stop", engine_stop_cmd))

    dp.add_handler(CommandHandler("start_mock", start_mock_cmd))
    dp.add_handler(CommandHandler("start_live", start_live_cmd))

    dp.add_handler(CommandHandler("symbols", symbols_cmd, pass_args=True))
    dp.add_handler(CommandHandler("timeframe", timeframe_cmd))
    dp.add_handler(CommandHandler("entry_mode", entry_mode_cmd))
    dp.add_handler(CommandHandler("trailing", trailing_cmd))

    dp.add_handler(CommandHandler("pnl", pnl_cmd))
    dp.add_handler(CommandHandler("reset_pnl", reset_pnl_cmd))
    dp.add_handler(CommandHandler("export_csv", export_csv_cmd))
    dp.add_handler(CommandHandler("export_k4", export_k4_cmd))

    dp.add_handler(CommandHandler("keepalive_on", keepalive_on_cmd))
    dp.add_handler(CommandHandler("keepalive_off", keepalive_off_cmd))

    dp.add_handler(CommandHandler("panic", panic_cmd))

    dp.add_handler(CallbackQueryHandler(button_cb))
    dp.add_handler(MessageHandler(Filters.command, unknown_cmd))
    return upd

# Starta Telegram i bakgrunden när modulen laddas
def _start_telegram():
    if not TELEGRAM_TOKEN:
        log.warning("Ingen TELEGRAM_TOKEN — hoppar över Telegram.")
        return
    upd = build_updater()
    # kör polling i egen tråd så FastAPI kan köra samtidigt
    threading.Thread(target=upd.start_polling, daemon=True).start()
    log.info("Telegram polling startad.")

_start_telegram()

# ==========
# Uvicorn self-run (lokalt)
# ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    server = Server(Config(app, host="0.0.0.0", port=port, log_level="info"))
    server.run()
