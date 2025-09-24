# main.py
# Mp ORBbot — v2.2 (Render Web Service-kompatibel)
# - FastAPI app (app) med /health för Render health checks
# - Telegram-bot via python-telegram-bot (polling i bakgrund)
# - ORB-regel: ny ORB vid röd→grön (LONG) / grön→röd (SHORT)
# - Entry mode standard: TICK (kan ändras till CLOSE)
# - Exit: SL 1.5%, TP 0.5%, Trailing: trigger +0.4% → lås +0.2%
# - AI-filter + EMA200 för trendriktning (LONG endast över EMA200, SHORT endast under)
# - Max 1 DCA-ben
# - Mock-läge och AI=neutral som standard
# - Backtest inkluderar avgifter och exporterar CSV
# - Loggar: mock_trade_log.csv och real_trade_log.csv (endast mock används i denna version)

import os
import csv
import time
import math
import json
import enum
import asyncio
import logging
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import requests
import pandas as pd
import numpy as np
import pytz

from fastapi import FastAPI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes
)

# ========== BAS-KONFIG ==========
SYMBOLS_DEFAULT = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]
TIMEZONE = "Europe/Stockholm"

# Risk & exits
TP_PCT = 0.005          # +0.5%
SL_PCT = 0.015          # -1.5%
TRAIL_TRIGGER = 0.004   # +0.4%
TRAIL_LOCK = 0.002      # lås in +0.2% efter trigger

# DCA
ENABLE_DCA = True
MAX_DCA_LEGS = 1
DCA_TRIGGER_PCT = -0.005    # ett ben på -0.5% (LONG; speglas för SHORT)
DCA_SIZE_MULTIPLIER = 1.0    # samma storlek som initial

# ORB/ENTRY
class EntryMode(enum.Enum):
    TICK = "TICK"
    CLOSE = "CLOSE"

ENTRY_MODE_DEFAULT = EntryMode.TICK

# AI-lägen
AI_LEVELS = ["aggressiv", "neutral", "försiktig"]
AI_DEFAULT = "neutral"

# Mock trade konfig
MOCK_TRADE_SIZE_USDT = 30.0  # enligt din preferens

# Avgifter
DEFAULT_FEE = 0.001  # 0.1% per köp/sälj

# Filer
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

# Telegram token
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("mp-orbbot")


# ========== HJÄLPFUNKTIONER ==========
def now_ts() -> dt.datetime:
    return dt.datetime.now(pytz.timezone(TIMEZONE))

def pct_change(from_price: float, to_price: float) -> float:
    return (to_price - from_price) / from_price if from_price != 0 else 0.0

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def ensure_csv_headers(path: str, headers: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def write_csv(path: str, row: List):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ========== KUCOIN PUBLIC DATA ==========
class KuCoinPublic:
    BASE = "https://api.kucoin.com"

    @staticmethod
    def get_kline(symbol: str, granularity_sec: int, start_ts: Optional[int]=None, end_ts: Optional[int]=None) -> List[List]:
        """
        Returnerar listor: [time, open, close, high, low, volume, turnover]
        KuCoin granularity i sekunder; 3-min = 180
        OBS: KuCoin returnerar reverse chronological.
        """
        url = f"{KuCoinPublic.BASE}/api/v1/market/candles"
        params = {
            "type": KuCoinPublic._gran_to_str(granularity_sec),
            "symbol": symbol
        }
        if start_ts: params["startAt"] = int(start_ts)
        if end_ts: params["endAt"] = int(end_ts)
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "200000":
            raise RuntimeError(f"KuCoin API error: {data}")
        return data["data"]

    @staticmethod
    def _gran_to_str(gran: int) -> str:
        if gran == 180: return "3min"
        if gran == 60: return "1min"
        if gran == 300: return "5min"
        raise ValueError("Unsupported granularity")

def klines_to_df(rows: List[List]) -> pd.DataFrame:
    cols = ["time", "open", "close", "high", "low", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "close", "high", "low", "volume", "turnover"]:
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(TIMEZONE)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ========== STRATEGI & TRADER ==========
@dataclass
class Trade:
    symbol: str
    side: str  # "LONG" / "SHORT"
    entry_price: float
    size_usdt: float
    qty: float
    entry_time: dt.datetime
    fee_rate: float
    tp_pct: float = TP_PCT
    sl_pct: float = SL_PCT
    trail_trigger: float = TRAIL_TRIGGER
    trail_lock: float = TRAIL_LOCK
    dca_legs_done: int = 0
    max_dca: int = MAX_DCA_LEGS
    dca_trigger_pct: float = DCA_TRIGGER_PCT
    dca_mult: float = DCA_SIZE_MULTIPLIER
    is_open: bool = True
    trail_active: bool = False
    trail_floor_price: Optional[float] = None  # LONG
    trail_ceiling_price: Optional[float] = None  # SHORT

    def current_rr(self, price: float) -> float:
        if self.side == "LONG":
            return pct_change(self.entry_price, price)
        else:
            return pct_change(price, self.entry_price)

    def check_exits_and_manage(self, price: float, now: dt.datetime) -> Tuple[bool, Optional[float], str]:
        """
        Returnerar: (closed, exit_price, reason)
        """
        move = self.current_rr(price)

        # TP
        if move >= self.tp_pct:
            self.is_open = False
            return True, price, "TP"

        # Trailing: aktivering
        if not self.trail_active and move >= self.trail_trigger:
            if self.side == "LONG":
                self.trail_floor_price = self.entry_price * (1 + self.trail_lock)
            else:
                self.trail_ceiling_price = self.entry_price * (1 - self.trail_lock)
            self.trail_active = True

        # Trailing: uppföljning
        if self.trail_active:
            if self.side == "LONG":
                locked = self.entry_price * (1 + self.trail_lock)
                if price > (self.trail_floor_price or 0):
                    self.trail_floor_price = max(self.trail_floor_price or locked, locked)
                if price <= (self.trail_floor_price or price):
                    self.is_open = False
                    return True, price, "TRAIL"
            else:
                locked = self.entry_price * (1 - self.trail_lock)
                if price < (self.trail_ceiling_price or 1e12):
                    self.trail_ceiling_price = min(self.trail_ceiling_price or locked, locked)
                if price >= (self.trail_ceiling_price or price):
                    self.is_open = False
                    return True, price, "TRAIL"

        # SL
        if move <= -self.sl_pct:
            self.is_open = False
            return True, price, "SL"

        # DCA (1 max)
        if ENABLE_DCA and self.dca_legs_done < self.max_dca:
            if self.side == "LONG" and move <= self.dca_trigger_pct:
                self._apply_dca(price)
                return False, None, "DCA"
            if self.side == "SHORT" and (-move) <= self.dca_trigger_pct:
                self._apply_dca(price)
                return False, None, "DCA"

        return False, None, ""

    def _apply_dca(self, price: float):
        add_usdt = self.size_usdt * self.dca_mult
        add_qty = add_usdt / price
        if self.side == "LONG":
            total_cost = self.qty * self.entry_price + add_qty * price
            self.qty += add_qty
            self.size_usdt += add_usdt
            self.entry_price = total_cost / self.qty
        else:
            total_value = self.qty * self.entry_price + add_qty * price
            self.qty += add_qty
            self.size_usdt += add_usdt
            self.entry_price = total_value / self.qty
        self.dca_legs_done += 1


# ========== AI-FILTER & ORB ==========
def ai_filter_trend(df: pd.DataFrame, ai_level: str) -> pd.Series:
    """
    trend: 1 (bull), -1 (bear), 0 (osäkert)
    EMA200 + marginal per AI-läge.
    """
    close = df["close"]
    ema200 = ema(close, 200)
    margin = {"aggressiv": 0.000, "neutral": 0.001, "försiktig": 0.002}.get(ai_level, 0.001)
    bull = (close >= ema200 * (1 + margin)).astype(int)
    bear = (close <= ema200 * (1 - margin)).astype(int) * -1
    trend = bull + bear
    trend.replace(2, 1, inplace=True)
    return trend

def detect_orb_windows(df: pd.DataFrame) -> pd.DataFrame:
    o, c, h, l = df["open"].values, df["close"].values, df["high"].values, df["low"].values
    n = len(df)
    orb_high = np.full(n, np.nan)
    orb_low  = np.full(n, np.nan)
    orb_side = np.array([""] * n, dtype=object)
    armed = False
    side = ""

    for i in range(1, n):
        prev_red_to_green = (c[i-1] < o[i-1]) and (c[i] > o[i])   # LONG
        prev_green_to_red = (c[i-1] > o[i-1]) and (c[i] < o[i])   # SHORT

        if prev_red_to_green:
            orb_high[i] = h[i]
            orb_low[i] = l[i]
            orb_side[i] = "LONG"
            armed = True
            side = "LONG"
        elif prev_green_to_red:
            orb_high[i] = h[i]
            orb_low[i] = l[i]
            orb_side[i] = "SHORT"
            armed = True
            side = "SHORT"
        else:
            if armed:
                orb_high[i] = orb_high[i-1]
                orb_low[i] = orb_low[i-1]
                orb_side[i] = side
            else:
                orb_side[i] = ""

    df = df.copy()
    df["orb_high"] = orb_high
    df["orb_low"] = orb_low
    df["orb_side"] = orb_side
    return df


# ========== BACKTEST ==========
class Backtester:
    def __init__(self, fee_rate: float = DEFAULT_FEE, ai_level: str = AI_DEFAULT, entry_mode: EntryMode = ENTRY_MODE_DEFAULT):
        self.fee_rate = fee_rate
        self.ai_level = ai_level
        self.entry_mode = entry_mode

    def run(self, df: pd.DataFrame, symbol: str) -> Dict:
        df = df.copy()
        df["trend"] = ai_filter_trend(df, self.ai_level)
        df = detect_orb_windows(df)

        trades: List[Dict] = []
        open_trade: Optional[Trade] = None

        for i in range(1, len(df)):
            row = df.iloc[i]

            # Hantera exit för ev. öppen trade på candle-close (konservativ simulering)
            if open_trade:
                closed, exit_price, reason = open_trade.check_exits_and_manage(row["close"], row["time"])
                if closed:
                    gross = (exit_price - open_trade.entry_price) * open_trade.qty if open_trade.side == "LONG" else (open_trade.entry_price - exit_price) * open_trade.qty
                    fees = open_trade.size_usdt * open_trade.fee_rate + open_trade.size_usdt * (1 + open_trade.current_rr(exit_price)) * open_trade.fee_rate
                    pnl = gross - fees
                    trades.append({
                        "entry_time": open_trade.entry_time,
                        "exit_time": row["time"],
                        "side": open_trade.side,
                        "entry": float(open_trade.entry_price),
                        "exit": float(exit_price),
                        "pnl": float(pnl),
                        "reason": reason,
                        "legs": open_trade.dca_legs_done + 1
                    })
                    open_trade = None

            # Entry
            if open_trade is None and isinstance(row["orb_high"], float) and not math.isnan(row["orb_high"]):
                side = row["orb_side"]
                trend = row["trend"]
                # Trendfilter
                if side == "LONG" and trend != 1:
                    continue
                if side == "SHORT" and trend != -1:
                    continue

                # Trigger
                if self.entry_mode == EntryMode.TICK:
                    trigger = (row["high"] >= row["orb_high"]) if side == "LONG" else (row["low"] <= row["orb_low"])
                else:
                    trigger = (row["close"] >= row["orb_high"]) if side == "LONG" else (row["close"] <= row["orb_low"])

                if trigger:
                    entry_price = row["orb_high"] if side == "LONG" else row["orb_low"]
                    qty = MOCK_TRADE_SIZE_USDT / entry_price
                    open_trade = Trade(
                        symbol=symbol, side=side, entry_price=entry_price, size_usdt=MOCK_TRADE_SIZE_USDT,
                        qty=qty, entry_time=row["time"], fee_rate=self.fee_rate
                    )

        pnl_total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = sum(1 for t in trades if t["pnl"] <= 0)
        return {
            "symbol": symbol,
            "trades": trades,
            "summary": {"n_trades": len(trades), "wins": wins, "losses": losses, "pnl_total": pnl_total}
        }


# ========== STATE & LOGGNING ==========
@dataclass
class BotState:
    ai_level: str = AI_DEFAULT
    entry_mode: EntryMode = ENTRY_MODE_DEFAULT
    mock_mode: bool = True
    running: bool = False
    active_symbol: Optional[str] = None
    last_trade: Optional[Dict] = None
    todays_pnl: float = 0.0
    trades_today: int = 0

class TradeLogger:
    def __init__(self):
        ensure_csv_headers(MOCK_LOG, ["time", "symbol", "side", "entry", "exit", "pnl_usdt", "reason", "fees", "qty", "legs"])
        ensure_csv_headers(REAL_LOG, ["time", "symbol", "side", "entry", "exit", "pnl_usdt", "reason", "fees", "qty", "legs"])

    def log(self, trade: Trade, exit_price: float, reason: str, mock: bool, now: dt.datetime):
        gross = (exit_price - trade.entry_price) * trade.qty if trade.side == "LONG" else (trade.entry_price - exit_price) * trade.qty
        fees = trade.size_usdt * trade.fee_rate + trade.size_usdt * (1 + trade.current_rr(exit_price)) * trade.fee_rate
        pnl = gross - fees
        row = [now.isoformat(), trade.symbol, trade.side, round(trade.entry_price, 8), round(exit_price, 8), round(pnl, 4), reason, round(fees, 4), round(trade.qty, 8), trade.dca_legs_done]
        write_csv(MOCK_LOG if mock else REAL_LOG, row)


# ========== HUVUDBOT ==========
class MpORBbot:
    def __init__(self):
        self.state = BotState()
        self.logger = TradeLogger()

    # ===== Telegram-kommandon =====
    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self.state.running = True
        await update.message.reply_text("✅ Boten är igång (mock-läge). /status för info.")

    async def cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self.state.running = False
        await update.message.reply_text("⏹️ Boten stoppad. Kommandon funkar, men inga trades körs.")

    async def cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        last = self.state.last_trade or {}
        txt = (
            f"🤖 Mp ORBbot\n"
            f"• Running: {self.state.running}\n"
            f"• Mock mode: {self.state.mock_mode}\n"
            f"• AI-läge: {self.state.ai_level}\n"
            f"• Entry mode: {self.state.entry_mode.value}\n"
            f"• Aktivt coin: {self.state.active_symbol or '-'}\n"
            f"• Dagens PnL: {round(self.state.todays_pnl, 4)} USDT\n"
            f"• Antal trades idag: {self.state.trades_today}\n"
            f"• Senaste trade: {json.dumps(last, default=str) if last else '-'}"
        )
        await update.message.reply_text(txt)

    async def cmd_set_ai(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not ctx.args:
            await update.message.reply_text(f"Användning: /set_ai <{'|'.join(AI_LEVELS)}>  (nu: {self.state.ai_level})")
            return
        level = ctx.args[0].lower()
        if level not in AI_LEVELS:
            await update.message.reply_text(f"Ogiltigt AI-läge. Välj: {', '.join(AI_LEVELS)}")
            return
        self.state.ai_level = level
        await update.message.reply_text(f"✅ AI-läge satt till: {level}")

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "/start – starta boten\n"
            "/stop – stoppa boten\n"
            "/status – visa status\n"
            f"/set_ai <{'|'.join(AI_LEVELS)}> – sätt AI-filter\n"
            "/backtest <symbol|ALL> <period> [fee] – ex: /backtest btcusdt 3d 0.001\n"
            "/export_csv – skicka loggar\n"
            "/mock_trade <symbol> – simulera snabb trade\n"
            "/help – den här hjälpen\n\n"
            "Parametrar: TP=0.5%, SL=1.5%, Trailing: +0.4%→+0.2%, max 1 DCA."
        )

    async def cmd_export_csv(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        sent_any = False
        if os.path.exists(MOCK_LOG):
            await update.message.reply_document(document=open(MOCK_LOG, "rb"), filename=MOCK_LOG)
            sent_any = True
        if os.path.exists(REAL_LOG):
            await update.message.reply_document(document=open(REAL_LOG, "rb"), filename=REAL_LOG)
            sent_any = True
        if not sent_any:
            await update.message.reply_text("Inga loggar funna ännu.")

    async def cmd_mock_trade(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not ctx.args:
            await update.message.reply_text("Användning: /mock_trade <symbol>")
            return
        symbol = ctx.args[0].upper()
        price = self._fetch_last_price(symbol)
        if price is None:
            await update.message.reply_text(f"Kunde inte hämta pris för {symbol}.")
            return

        side = "LONG"
        trade = Trade(
            symbol=symbol, side=side, entry_price=price, size_usdt=MOCK_TRADE_SIZE_USDT,
            qty=MOCK_TRADE_SIZE_USDT / price, entry_time=now_ts(), fee_rate=DEFAULT_FEE
        )
        # Simulera liten +0.6% rörelse → TP borde slå
        exit_price = price * 1.006
        closed, _, _ = trade.check_exits_and_manage(exit_price, now_ts())
        self.logger.log(trade, exit_price, "MOCK", True, now_ts())
        self.state.last_trade = {"symbol": symbol, "side": side, "entry": price, "exit": exit_price, "reason": "MOCK"}
        await update.message.reply_text(f"✅ Mock trade {symbol}: in {round(price,6)} → ut {round(exit_price,6)}")

    async def cmd_backtest(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if len(ctx.args) < 2:
            await update.message.reply_text("Användning: /backtest <symbol|ALL> <period> [fee]. Ex: /backtest btcusdt 3d 0.001")
            return
        symbol = ctx.args[0].upper()
        period = ctx.args[1].lower()
        try:
            fee = float(ctx.args[2]) if len(ctx.args) >= 3 else DEFAULT_FEE
        except ValueError:
            fee = DEFAULT_FEE

        symbols = SYMBOLS_DEFAULT if symbol == "ALL" else [symbol]

        result_texts = []
        for sym in symbols:
            try:
                df = self._fetch_history(sym, period)
                bt = Backtester(fee_rate=fee, ai_level=self.state.ai_level, entry_mode=self.state.entry_mode)
                res = bt.run(df, sym)
                s = res["summary"]
                result_texts.append(f"{sym}: trades={s['n_trades']}, wins={s['wins']}, losses={s['losses']}, pnl={round(s['pnl_total'],4)} USDT")
                # Exportera trades direkt som CSV
                out_path = f"backtest_{sym}_{period}.csv"
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["entry_time","exit_time","side","entry","exit","pnl","reason","legs"])
                    for t in res["trades"]:
                        w.writerow([
                            t["entry_time"], t["exit_time"], t["side"],
                            round(t["entry"],8), round(t["exit"],8),
                            round(t["pnl"],6), t["reason"], t["legs"]
                        ])
                await update.message.reply_document(document=open(out_path, "rb"), filename=os.path.basename(out_path))
            except Exception as e:
                logger.exception("Backtest error")
                result_texts.append(f"{sym}: fel – {e}")

        await update.message.reply_text("📊 Backtest klart:\n" + "\n".join(result_texts))

    # ===== Interna =====
    def _fetch_last_price(self, symbol: str) -> Optional[float]:
        try:
            rows = KuCoinPublic.get_kline(symbol, 180)
            df = klines_to_df(rows)
            return float(df["close"].iloc[-1])
        except Exception as e:
            logger.error(f"Fetch last price failed: {e}")
            return None

    def _period_to_seconds(self, s: str) -> int:
        unit = s[-1]
        val = int(s[:-1])
        if unit == "m": return val * 60
        if unit == "h": return val * 3600
        if unit == "d": return val * 86400
        if unit == "w": return val * 604800
        raise ValueError("Ogiltig period. Använd m/h/d/w.")

    def _fetch_history(self, symbol: str, period: str) -> pd.DataFrame:
        seconds = self._period_to_seconds(period)
        end = int(time.time())
        start = end - seconds
        rows = KuCoinPublic.get_kline(symbol, 180, start_ts=start, end_ts=end)
        df = klines_to_df(rows)
        return df


# ========== FASTAPI-APP (för Render) ==========
app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}


# ========== TELEGRAM-BOT START (i bakgrund) ==========
_bot_instance: Optional[MpORBbot] = None
_application = None

async def run_bot():
    global _bot_instance, _application
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN saknas – sätt den i Render Environment.")
        return

    _bot_instance = MpORBbot()
    _application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    _application.add_handler(CommandHandler("start", _bot_instance.cmd_start))
    _application.add_handler(CommandHandler("stop", _bot_instance.cmd_stop))
    _application.add_handler(CommandHandler("status", _bot_instance.cmd_status))
    _application.add_handler(CommandHandler("set_ai", _bot_instance.cmd_set_ai))
    _application.add_handler(CommandHandler("help", _bot_instance.cmd_help))
    _application.add_handler(CommandHandler("export_csv", _bot_instance.cmd_export_csv))
    _application.add_handler(CommandHandler("mock_trade", _bot_instance.cmd_mock_trade))
    _application.add_handler(CommandHandler("backtest", _bot_instance.cmd_backtest))

    logger.info("Mp ORBbot startar polling...")
    await _application.run_polling(close_loop=False)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(run_bot())
    logger.info("FastAPI startup klar – botpolling körs i bakgrunden.")

@app.on_event("shutdown")
async def on_shutdown():
    # python-telegram-bot stänger själv på process-exit; här kan vi lägga extra städning vid behov.
    logger.info("FastAPI shutdown – avslutar.")
