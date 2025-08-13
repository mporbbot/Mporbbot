# main.py — Mp ORBbot (full version)
# - ORB vid riktningsändring (dynamisk box)
# - Entry på close > ORB-high*(1+buffer)
# - Fast TP +0.12%, Break-even efter +0.10% (säkra avgifter), ATR-trailing
# - Trendfilter EMA20>EMA50 (kan stängas av), min ORB-range
# - Mock/Live, CSV-loggar (gross/net/fees/entry/exit), PnL idag
# - Telegram-kommandon enligt /help

from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ================== Konfig via ENV (med vettiga defaults) ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()

DEFAULT_SYMBOLS = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]

FEE_RATE           = float(os.getenv("FEE_RATE", "0.001"))  # 0.1% per sida
MOCK_USDT          = float(os.getenv("MOCK_USDT", "30"))    # positionstorlek (USDT) för mock
TIMEFRAME_DEFAULT  = os.getenv("TIMEFRAME", "3min")

# ORB/Trend/ATR
ORB_BUFFER     = float(os.getenv("ORB_BUFFER", "0.0005"))    # +0.05% över ORB-high
MIN_ORB_RANGE  = float(os.getenv("MIN_ORB_RANGE", "0.0015")) # min 0.15% range
USE_EMA_TREND  = os.getenv("USE_EMA_TREND", "1") == "1"      # EMA20 > EMA50
ATR_STOP_MULT  = float(os.getenv("ATR_STOP_MULT", "1.0"))
ATR_TRAIL_MULT = float(os.getenv("ATR_TRAIL_MULT", "0.5"))

# Holding/exit-skydd
MIN_HOLD_BARS       = int(os.getenv("MIN_HOLD_BARS", "1"))     # vänta minst 1 stängd bar
TP_STATIC_PCT       = float(os.getenv("TP_STATIC_PCT", "0.0012"))   # +0.12%
BREAKEVEN_AFTER_PCT = float(os.getenv("BREAKEVEN_AFTER_PCT", "0.001")) # +0.10%
BREAKEVEN_BUFFER    = float(os.getenv("BREAKEVEN_BUFFER", "0.00005"))  # +0.005%

# Loggar
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"
LOG_FIELDS = [
    "timestamp","mode","symbol","side","price","qty",
    "fee_rate","fee_cost","gross_pnl","net_pnl",
    "entry_price","exit_price","note"
]

KU_PUBLIC = "https://api.kucoin.com"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ================== Globalt state ==================
state: Dict[str, Any] = {
    "active": False,             # motor på/av
    "mode": "mock",              # "mock" | "live"
    "ai_mode": "neutral",        # neutral vid uppstart
    "pending_confirm": None,     # {"type":"mock"|"live"}
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": TIMEFRAME_DEFAULT,
    "engine_task": None,
    "per_symbol": {},            # symbol -> ORBState
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
}

# ================== Telegram utils ==================
TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None

async def tg_send_text(chat_id: int|str, text: str):
    if not TG_BASE: return
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(f"{TG_BASE}/sendMessage", json={"chat_id": chat_id, "text": text})

async def tg_send_document(chat_id: int|str, file_path: str, caption: str=""):
    if not TG_BASE: return
    if not os.path.exists(file_path):
        await tg_send_text(chat_id, f"Filen saknas: {file_path}")
        return
    async with httpx.AsyncClient(timeout=60) as client:
        with open(file_path, "rb") as f:
            files = {"document": (os.path.basename(file_path), f)}
            data = {"chat_id": chat_id, "caption": caption}
            await client.post(f"{TG_BASE}/sendDocument", data=data, files=files)

def is_admin(chat_id: Any) -> bool:
    return str(chat_id) == str(ADMIN_CHAT_ID) if ADMIN_CHAT_ID else True

# ================== AI-filter ==================
def ai_allows(ai_mode: str, signal_strength: float) -> bool:
    m = (ai_mode or "neutral").lower()
    if m == "aggressiv": return signal_strength >= 0.2
    if m in ("försiktig","forsiktig"): return signal_strength >= 0.7
    return signal_strength >= 0.5  # neutral

# ================== Loggning & PnL ==================
def write_log_row(path: str, row: Dict[str, Any]):
    newfile = not os.path.exists(path)
    for k in LOG_FIELDS:
        row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})

def compute_pnl(entry_price: float, exit_price: float, qty: float, fee_rate: float):
    gross = (exit_price - entry_price) * qty
    fees  = (entry_price + exit_price) * qty * fee_rate  # båda håll
    net   = gross - fees
    return gross, fees, net

async def mock_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "mock", "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": "mock",
    }
    if extra: row.update(extra)
    write_log_row(MOCK_LOG, row)
    await tg_send_text(chat_id, f"✅ Mock: {side.upper()} {symbol} @ {price:.6f} x {qty:g} (avg ~ {fee_cost:.8f})")

async def live_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "live", "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": "live (stub)",
    }
    if extra: row.update(extra)
    write_log_row(REAL_LOG, row)
    await tg_send_text(chat_id, f"🟢 LIVE (loggad): {side.upper()} {symbol} @ {price:.6f} x {qty:g} (avg ~ {fee_cost:.8f})")

# ================== KuCoin publika candles ==================
def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"):
        return f"{usdt_sym[:-4]}-USDT"
    return usdt_sym.replace("/", "-").upper()

async def fetch_candles(symbol: str, timeframe: str="3min", limit: int=120) -> pd.DataFrame:
    ksym = to_kucoin_symbol(symbol)
    url = f"{KU_PUBLIC}/api/v1/market/candles"
    params = {"type": timeframe, "symbol": ksym}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
    cols = ["ts","open","close","high","low","volume","turnover"]
    rows = []
    for row in data:
        vals = row[:7] if len(row)>=7 else row+[None]
        rows.append(vals)
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    for c in ["open","close","high","low","volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("ts").reset_index(drop=True).tail(limit)

# ================== EMA & ATR ==================
def ema(series: pd.Series, period: int) -> pd.Series:
    s = pd.Series(series, dtype="float64")
    return s.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ================== ORB State ==================
@dataclass
class ORBState:
    direction: str = "flat"    # "up"|"down"|"flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    side: str = ""             # "long"
    entry_price: float = 0.0
    entry_qty: float = 0.0
    trail_stop: float = 0.0
    last_price: float = 0.0
    bars_in_trade: int = 0
    last_bar_ts: Optional[pd.Timestamp] = None
    peak_price: float = 0.0

def detect_dir(prev_close: float, close: float) -> str:
    if close > prev_close: return "up"
    if close < prev_close: return "down"
    return "flat"

def reset_orb(st: ORBState, hi: float, lo: float):
    st.orb_high, st.orb_low = float(hi), float(lo)
    st.in_position = False
    st.side = ""
    st.entry_price = 0.0
    st.entry_qty = 0.0
    st.trail_stop = 0.0
    st.bars_in_trade = 0
    st.peak_price = 0.0

# ================== PnL-notis ==================
async def notify_closed_trade(chat_id: int, symbol: str, entry_price: float, exit_price: float, qty: float):
    gross, fees, net = compute_pnl(entry_price, exit_price, qty, FEE_RATE)
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
    emo = "📈" if net >= 0 else "📉"
    txt = (
        f"{emo} Stängd LONG {symbol}\n"
        f"Entry: {entry_price:.6f}  Exit: {exit_price:.6f}\n"
        f"Qty: {qty:g}\n"
        f"Gross: {gross:.6f}\n"
        f"Avg: {fees:.6f}\n"
        f"Netto: {net:.6f}\n"
        f"PnL idag: {state['pnl_today']:.6f}"
    )
    await tg_send_text(chat_id, txt)

# ================== Motor ==================
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, timeframe, limit=100)
    if df.empty or len(df) < 55:
        return

    last = df.iloc[-1]
    if st.last_bar_ts is not None and pd.Timestamp(last["ts"]) == st.last_bar_ts:
        return  # redan agerat på denna bar

    prev = df.iloc[-2]
    st.last_bar_ts = pd.Timestamp(last["ts"])
    st.last_price = float(last["close"])

    # Riktningsdetektion baserat på föregående->senaste stängning
    last_dir = detect_dir(prev["close"], last["close"])

    # Ny ORB vid riktningsändring → box från senast stängda bar
    if last_dir != st.direction and last_dir in ("up","down"):
        reset_orb(st, prev["high"], prev["low"])
        st.direction = last_dir

    # Indikatorer
    df_close = df["close"].astype(float)
    ema20 = float(ema(df_close, 20).iloc[-1])
    ema50 = float(ema(df_close, 50).iloc[-1])
    atr14 = float(atr(df, 14).iloc[-1])
    if atr14 <= 0:
        return

    # Min ORB-range
    orb_range_ok = True
    if st.orb_high > 0 and st.orb_low > 0 and st.orb_low > 0:
        rng = (st.orb_high / st.orb_low) - 1.0
        orb_range_ok = rng >= MIN_ORB_RANGE

    # ENTRY (endast long enligt din spec)
    entry_ok = (
        (not st.in_position)
        and st.direction == "up"
        and st.orb_high > 0
        and float(last["close"]) > st.orb_high * (1.0 + ORB_BUFFER)
        and (not USE_EMA_TREND or ema20 > ema50)
        and orb_range_ok
    )
    if entry_ok and ai_allows(state["ai_mode"], 0.6):
        st.in_position = True
        st.side = "long"
        st.entry_price = float(last["close"])  # entry på close
        st.entry_qty = max(MOCK_USDT / st.entry_price, 0.0001)
        st.trail_stop = max(st.entry_price - ATR_STOP_MULT*atr14, st.orb_low)
        st.peak_price = st.entry_price
        st.bars_in_trade = 0

        if state["mode"] == "mock":
            await mock_trade(chat_id, symbol, "buy", st.entry_price, st.entry_qty)
        else:
            await live_trade(chat_id, symbol, "buy", st.entry_price, st.entry_qty)
        return

    # EXIT-logik för LONG
    if st.in_position and st.side == "long":
        st.bars_in_trade += 1

        last_close = float(last["close"])
        last_high  = float(last["high"])

        # Peak-uppdatering
        st.peak_price = max(st.peak_price or last_close, last_high, last_close)

        # ATR-trailing (på close)
        proposed_trail = last_close - ATR_TRAIL_MULT * atr14
        st.trail_stop = max(st.trail_stop or proposed_trail, proposed_trail)

        # --- Fast TP +0.12% ---
        tp_level = st.entry_price * (1.0 + TP_STATIC_PCT)
        if last_close >= tp_level:
            exit_price = last_close
            qty = st.entry_qty
            gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
            extra = {"entry_price": st.entry_price, "exit_price": exit_price,
                     "gross_pnl": round(gross, 8), "net_pnl": round(net, 8)}
            if state["mode"] == "mock":
                await mock_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            else:
                await live_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            await notify_closed_trade(chat_id, symbol, st.entry_price, exit_price, qty)
            reset_orb(st, st.orb_high, st.orb_low)  # behåll box tills riktning byts
            return

        # Grace period: låt minst 1 bar stänga innan vi kan exita
        if st.bars_in_trade < MIN_HOLD_BARS:
            return

        # --- Break-even-skydd efter +0.10% ---
        runup = (st.peak_price / st.entry_price) - 1.0
        if runup >= BREAKEVEN_AFTER_PCT:
            be_level = st.entry_price * (1.0 + 2*FEE_RATE + BREAKEVEN_BUFFER)
            st.trail_stop = max(st.trail_stop, be_level)

        # Exit på trail / hård ATR / ORB-low (alla på CLOSE)
        hard_stop = st.entry_price - ATR_STOP_MULT * atr14
        exit_ok = (
            (st.trail_stop and last_close < st.trail_stop) or
            (last_close < hard_stop) or
            (st.orb_low and last_close < st.orb_low)
        )

        if exit_ok:
            exit_price = last_close
            qty = st.entry_qty
            gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
            extra = {"entry_price": st.entry_price, "exit_price": exit_price,
                     "gross_pnl": round(gross, 8), "net_pnl": round(net, 8)}
            if state["mode"] == "mock":
                await mock_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            else:
                await live_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            await notify_closed_trade(chat_id, symbol, st.entry_price, exit_price, qty)
            reset_orb(st, st.orb_high, st.orb_low)
            return

async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(20)  # loopintervall
        except asyncio.CancelledError:
            break
        except Exception as e:
            await tg_send_text(chat_id, f"Motorfel: {e}")
            await asyncio.sleep(5)

async def start_engine(chat_id: int):
    if state["engine_task"] and not state["engine_task"].done():
        await tg_send_text(chat_id, "Motorn kör redan.")
        return
    state["active"] = True
    loop = asyncio.get_event_loop()
    state["engine_task"] = loop.create_task(run_engine(chat_id))
    await tg_send_text(chat_id, f"🔄 Motor startad {utcnow_iso()} (tf={state['timeframe']}, symbols={', '.join(state['symbols'])}).")

async def stop_engine(chat_id: int):
    state["active"] = False
    task = state.get("engine_task")
    if task and not task.done():
        task.cancel()
    await tg_send_text(chat_id, "⏹️ Motor stoppad.")

# ================== Keepalive ==================
async def keepalive_loop(url: str, chat_id: int|str):
    while state.get("keepalive"):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(url)
        except Exception:
            pass
        await asyncio.sleep(12 * 60)  # var 12:e minut

# ================== Backtest (dummy) ==================
async def run_backtest(chat_id: int, symbol: str="BTCUSDT", period: str="3d", fee: float|None=None, out_csv: str="backtest_result.csv"):
    import random
    random.seed(42)
    fee_rate = FEE_RATE if fee is None else float(fee)
    rows = []
    for _ in range(40):
        gross = random.uniform(-0.004, 0.006)     # -0.4%..+0.6%
        net = gross - 2 * fee_rate
        rows.append({
            "timestamp": utcnow_iso(),
            "symbol": symbol.upper(),
            "gross_return": round(gross, 6),
            "fee_rate_each_side": fee_rate,
            "net_return": round(net, 6)
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    await tg_send_text(chat_id, f"Backtest klart: {symbol} ({period}) avgift {fee_rate:.4f}")
    return out_csv

# ================== Telegram-kommandon ==================
async def handle_update(data: Dict[str, Any]):
    msg = data.get("message") or data.get("edited_message") or {}
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}
    if not is_admin(chat_id):
        await tg_send_text(chat_id, "⛔️ Du är inte behörig.")
        return {"ok": True}

    # Pending confirm
    if state["pending_confirm"]:
        if text.lower() == "ja":
            mode = state["pending_confirm"]["type"]
            state["mode"] = mode
            state["pending_confirm"] = None
            await tg_send_text(chat_id, f"✅ {mode.upper()} aktiverad.")
            await start_engine(chat_id)
        else:
            await tg_send_text(chat_id, "Avbrutet.")
            state["pending_confirm"] = None
        return {"ok": True}

    if text.startswith("/help"):
        await tg_send_text(chat_id,
            "/status – visa status\n"
            "/set_ai <neutral|aggressiv|försiktig>\n"
            "/start_mock – starta mock (svara JA)\n"
            "/start_live – starta LIVE (svara JA)\n"
            "/engine_start – starta motor\n"
            "/engine_stop – stoppa motor\n"
            "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
            "/timeframe 1min|3min – byt tidsram\n"
            "/backtest [SYMBOL] [PERIOD] [FEE]\n"
            "/export_csv – skicka loggar\n"
            "/mock_trade SYMBOL SIDE PRICE QTY – manuell mock\n"
            "/pnl – visa dagens PnL\n"
            "/reset_pnl – nollställ dagens PnL\n"
            "/keepalive_on – håll Render vaken\n"
            "/keepalive_off – stäng keepalive")
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}\n"
            f"AI: {state['ai_mode']}\n"
            f"Avgift (per sida): {FEE_RATE:.4f}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Min range: {MIN_ORB_RANGE:.4%}\n"
            f"Trendfilter: {'på' if USE_EMA_TREND else 'av'}  ATR stop x{ATR_STOP_MULT}  trail x{ATR_TRAIL_MULT}\n"
            f"TP: +{TP_STATIC_PCT*100:.2f}%  BE efter +{BREAKEVEN_AFTER_PCT*100:.2f}%")
        return {"ok": True}

    if text.startswith("/set_ai"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send_text(chat_id, "Använd: /set_ai neutral|aggressiv|försiktig")
            return {"ok": True}
        nm = parts[1].strip().lower()
        if nm == "forsiktig": nm = "försiktig"
        if nm not in {"neutral","aggressiv","försiktig"}:
            await tg_send_text(chat_id, "Ogiltigt AI-läge.")
            return {"ok": True}
        state["ai_mode"] = nm
        await tg_send_text(chat_id, f"AI-läge satt till: {nm}")
        return {"ok": True}

    if text.startswith("/start_mock"):
        state["pending_confirm"] = {"type": "mock"}
        await tg_send_text(chat_id, "Vill du starta MOCK-trading? Svara JA.")
        return {"ok": True}

    if text.startswith("/start_live"):
        state["pending_confirm"] = {"type": "live"}
        await tg_send_text(chat_id, "⚠️ Vill du starta LIVE-trading? Svara JA.")
        return {"ok": True}

    if text.startswith("/engine_start"):
        await start_engine(chat_id);  return {"ok": True}

    if text.startswith("/engine_stop") or text.startswith("/stop"):
        await stop_engine(chat_id);   return {"ok": True}

    if text.startswith("/symbols"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send_text(chat_id, f"Aktuell lista: {', '.join(state['symbols'])}")
            return {"ok": True}
        raw = parts[1].replace(" ", "")
        syms = [s for s in raw.split(",") if s]
        state["symbols"] = [s.upper() for s in syms]
        await tg_send_text(chat_id, f"Symbols uppdaterade: {', '.join(state['symbols'])}")
        return {"ok": True}

    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts) > 1 else "3min"
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram satt till: {tf}")
        return {"ok": True}

    if text.startswith("/backtest"):
        parts = text.split()
        symbol = parts[1] if len(parts) > 1 else "BTCUSDT"
        period = parts[2] if len(parts) > 2 else "3d"
        fee = float(parts[3]) if len(parts) > 3 else None
        path = await run_backtest(chat_id, symbol=symbol.upper(), period=period, fee=fee)
        await tg_send_document(chat_id, path, caption=f"Backtest {symbol} ({period})")
        return {"ok": True}

    if text.startswith("/export_csv"):
        sent = False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p)
                sent = True
        if not sent:
            await tg_send_text(chat_id, "Inga loggar hittades ännu.")
        return {"ok": True}

    if text.startswith("/mock_trade"):
        parts = text.split()
        if len(parts) < 5:
            await tg_send_text(chat_id, "Använd: /mock_trade SYMBOL SIDE PRICE QTY")
            return {"ok": True}
        _, symbol, side, price, qty = parts[:5]
        await mock_trade(chat_id, symbol, side, float(price), float(qty))
        return {"ok": True}

    if text.startswith("/pnl"):
        await tg_send_text(chat_id, f"Ackumulerad PnL idag: {state.get('pnl_today', 0.0):.6f}")
        return {"ok": True}

    if text.startswith("/reset_pnl"):
        state["pnl_today"] = 0.0
        await tg_send_text(chat_id, "PnL för idag har nollställts.")
        return {"ok": True}

    if text.startswith("/keepalive_on"):
        public_url = f"https://{os.getenv('RENDER_EXTERNAL_URL','')}/"
        if not public_url or public_url == "https:///":
            public_url = "https://mporbbot.onrender.com/"
        if state.get("keepalive"):
            await tg_send_text(chat_id, "Keepalive är redan på.")
            return {"ok": True}
        state["keepalive"] = True
        loop = asyncio.get_event_loop()
        state["keepalive_task"] = loop.create_task(keepalive_loop(public_url, chat_id))
        await tg_send_text(chat_id, f"Keepalive PÅ – pingar {public_url} var 12:e minut.")
        return {"ok": True}

    if text.startswith("/keepalive_off"):
        state["keepalive"] = False
        task = state.get("keepalive_task")
        if task and not task.done():
            task.cancel()
        await tg_send_text(chat_id, "Keepalive AV.")
        return {"ok": True}

    return {"ok": True}"

# ================== FastAPI ==================
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {
        "message": "Mp ORBbot is live!",
        "webhook_path": WEBHOOK_PATH,
        "params": {
            "ORB_BUFFER": ORB_BUFFER, "MIN_ORB_RANGE": MIN_ORB_RANGE,
            "USE_EMA_TREND": USE_EMA_TREND,
            "ATR_STOP_MULT": ATR_STOP_MULT, "ATR_TRAIL_MULT": ATR_TRAIL_MULT,
            "TP_STATIC_PCT": TP_STATIC_PCT, "BREAKEVEN_AFTER_PCT": BREAKEVEN_AFTER_PCT
        }
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
