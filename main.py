# main.py — Mp ORBbot (auto-keepalive + autostart)
# Nytt i denna version:
# - Keepalive ping var 4:e minut (för Render Free)
# - Keepalive startar automatiskt vid app-start
# - Autostarta motorn när appen vaknar (styrt av env AUTOSTART=1, default 1)
# Övrigt: ORB (buffer+min-range), EMA20/50, ATR stop/trail, candle-filter,
# mock/live loggar med PnL, Telegram-kommandon, KuCoin-priser.

from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ================== Miljö & Defaults ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()  # valfritt
FEE_RATE           = float(os.getenv("FEE_RATE", "0.001"))   # 0.1%/sida
DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]
DEFAULT_TIMEFRAME  = os.getenv("DEFAULT_TIMEFRAME", "3min")

# ORB/Trend/ATR
ORB_BUFFER     = float(os.getenv("ORB_BUFFER", "0.0005"))     # 0.05% över high/under low
MIN_ORB_RANGE  = float(os.getenv("MIN_ORB_RANGE", "0.0015"))  # 0.15% min box
USE_EMA_TREND  = os.getenv("USE_EMA_TREND", "1") == "1"
ATR_STOP_MULT  = float(os.getenv("ATR_STOP_MULT", "1.0"))
ATR_TRAIL_MULT = float(os.getenv("ATR_TRAIL_MULT", "0.5"))

# Candle-filter
DOJI_BODY_TO_RANGE_MAX = float(os.getenv("DOJI_BODY_TO_RANGE_MAX", "0.1"))
USE_CANDLE_FILTERS     = os.getenv("USE_CANDLE_FILTERS", "1") == "1"

# ORB vid start
INITIAL_ORB_ON_START   = os.getenv("INITIAL_ORB_ON_START", "1") == "1"

# Mock sizing
MOCK_USDT_PER_TRADE    = float(os.getenv("MOCK_USDT", "30"))

# Engine-loop
ENGINE_LOOP_SEC        = int(os.getenv("ENGINE_LOOP_SEC", "12"))

# Keepalive & Autostart
KEEPALIVE_SEC          = int(os.getenv("KEEPALIVE_SEC", "240"))  # 4 min
AUTOSTART              = os.getenv("AUTOSTART", "1") == "1"      # default PÅ

# Loggfiler
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"
LOG_FIELDS = [
    "timestamp","mode","symbol","side","price","qty",
    "fee_rate","fee_cost","gross_pnl","net_pnl",
    "entry_price","exit_price","note"
]

KU_PUBLIC = "https://api.kucoin.com"
def utcnow_iso() -> str: return datetime.now(timezone.utc).isoformat()

# ================== State ==================
@dataclass
class ORBState:
    direction: str = "flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    side: str = ""
    entry_price: float = 0.0
    entry_qty: float = 0.0
    trail_stop: float = 0.0
    last_price: float = 0.0

state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",
    "ai_mode": "neutral",
    "pending_confirm": None,
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": DEFAULT_TIMEFRAME,
    "engine_task": None,
    "per_symbol": {},
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
}

# ================== Telegram ==================
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

# ================== Indikatorer & PnL ==================
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

def write_log_row(path: str, row: Dict[str, Any]):
    newfile = not os.path.exists(path)
    for k in LOG_FIELDS:
        row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})

def compute_pnl(entry: float, exit_: float, qty: float, fee_rate: float) -> Tuple[float,float,float]:
    gross = (exit_ - entry) * qty
    fees  = (entry + exit_) * qty * fee_rate
    net   = gross - fees
    return gross, fees, net

# ================== Candle-mönster ==================
def is_doji(o, h, l, c, max_r=DOJI_BODY_TO_RANGE_MAX):
    rng = max(h-l, 1e-12); body = abs(c-o)
    return body <= max_r * rng

def is_bull_engulf(po, pc, o, c): return (pc < po) and (c > o) and (c >= po) and (o <= pc)
def is_bear_engulf(po, pc, o, c): return (pc > po) and (c < o) and (c <= po) and (o >= pc)

def is_hammer(o,h,l,c):
    body = abs(c-o); low_sh = min(o,c)-l; up_sh = h-max(o,c); rng=max(h-l,1e-12)
    return (low_sh > 2*body) and (up_sh < body) and (body/rng < 0.6)

def is_shooting_star(o,h,l,c):
    body = abs(c-o); up_sh = h-max(o,c); low_sh = min(o,c)-l; rng=max(h-l,1e-12)
    return (up_sh > 2*body) and (low_sh < body) and (body/rng < 0.6)

def candle_filter_long(prev, last):
    po, pc, oo, cc = prev["open"], prev["close"], last["open"], last["close"]
    ph, pl, oh, ol = prev["high"], prev["low"], last["high"], last["low"]
    return is_bull_engulf(po, pc, oo, cc) or is_hammer(oo, oh, ol, cc) or is_doji(oo, oh, ol, cc)

def candle_filter_short(prev, last):
    po, pc, oo, cc = prev["open"], prev["close"], last["open"], last["close"]
    ph, pl, oh, ol = prev["high"], prev["low"], last["high"], last["low"]
    return is_bear_engulf(po, pc, oo, cc) or is_shooting_star(oo, oh, ol, cc) or is_doji(oo, oh, ol, cc)

# ================== KuCoin candles ==================
def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"): return f"{usdt_sym[:-4]}-USDT"
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
    df = pd.DataFrame([row[:7] for row in data], columns=cols)
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    for c in ["open","close","high","low","volume"]: df[c] = df[c].astype(float)
    return df.sort_values("ts").reset_index(drop=True).tail(limit)

# ================== Hjälp ==================
def ai_allows(ai_mode: str, signal_strength: float) -> bool:
    m = (ai_mode or "neutral").lower()
    if m == "aggressiv": return signal_strength >= 0.2
    if m in ("försiktig","forsiktig"): return signal_strength >= 0.7
    return signal_strength >= 0.5

def detect_dir(prev_close: float, close: float) -> str:
    if close > prev_close: return "up"
    if close < prev_close: return "down"
    return "flat"

def reset_orb(st: ORBState, hi: float, lo: float):
    st.orb_high, st.orb_low = float(hi), float(lo)
    st.in_position = False; st.side = ""
    st.entry_price = 0.0; st.entry_qty = 0.0; st.trail_stop = 0.0

async def notify_closed_trade(chat_id, symbol, entry_price, exit_price, qty):
    gross, fees, net = compute_pnl(entry_price, exit_price, qty, FEE_RATE)
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
    emo = "📈" if net >= 0 else "📉"
    await tg_send_text(
        chat_id,
        f"{emo} Stängd trade {symbol}\n"
        f"Entry: {entry_price:.6f}  Exit: {exit_price:.6f}\n"
        f"Qty: {qty:g}\nGross: {gross:.6f}  Avgifter: {fees:.6f}\n"
        f"Netto: {net:.6f}\nPnL idag: {state['pnl_today']:.6f}"
    )

# ================== Order/logg ==================
async def mock_trade(chat_id, symbol, side, price, qty, extra: Optional[Dict[str,Any]]=None):
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

async def live_trade(chat_id, symbol, side, price, qty, extra: Optional[Dict[str,Any]]=None):
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

# ================== Motor ==================
def mock_qty_for(price: float) -> float:
    return round(MOCK_USDT_PER_TRADE / price, 6) if price > 0 else 0.0

async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, timeframe, limit=90)
    if df.empty or len(df) < 55: return

    prev, last = df.iloc[-2], df.iloc[-1]
    st.last_price = float(last["close"])

    # Riktningsändring
    last_dir = detect_dir(prev["close"], last["close"])
    if last_dir != st.direction and last_dir in ("up","down"):
        reset_orb(st, last["high"], last["low"])
        st.direction = last_dir

    # Initial ORB
    if INITIAL_ORB_ON_START and st.orb_high == 0 and st.orb_low == 0:
        reset_orb(st, last["high"], last["low"])
        if st.direction == "flat": st.direction = last_dir

    # Indikatorer
    ema20 = float(ema(df["close"].astype(float), 20).iloc[-1])
    ema50 = float(ema(df["close"].astype(float), 50).iloc[-1])
    atr14 = float(atr(df, 14).iloc[-1])

    # Candle-filter
    long_ok = short_ok = True
    if USE_CANDLE_FILTERS:
        long_ok  = candle_filter_long(prev, last)
        short_ok = candle_filter_short(prev, last)

    # Min range
    orb_ok = True
    if st.orb_high > 0 and st.orb_low > 0 and st.orb_low > 0:
        orb_ok = (st.orb_high / st.orb_low - 1.0) >= MIN_ORB_RANGE

    close, high, low = float(last["close"]), float(last["high"]), float(last["low"])

    # Entry long
    entry_long = (
        (not st.in_position) and st.orb_high > 0 and
        close > st.orb_high * (1.0 + ORB_BUFFER) and
        st.direction == "up" and orb_ok and
        (not USE_EMA_TREND or ema20 > ema50) and
        long_ok and ai_allows(state["ai_mode"], 0.6)
    )

    # Entry short (mock)
    entry_short = (
        (not st.in_position) and st.orb_low > 0 and
        close < st.orb_low * (1.0 - ORB_BUFFER) and
        st.direction == "down" and orb_ok and
        (not USE_EMA_TREND or ema20 < ema50) and
        short_ok and ai_allows(state["ai_mode"], 0.6)
    )

    if entry_long:
        st.in_position, st.side = True, "long"
        st.entry_price, st.entry_qty = close, mock_qty_for(close)
        st.trail_stop = max(st.entry_price - ATR_STOP_MULT * atr14, st.orb_low)
        if state["mode"] == "mock":
            await mock_trade(chat_id, symbol, "buy", st.entry_price, st.entry_qty)
        else:
            await live_trade(chat_id, symbol, "buy", st.entry_price, st.entry_qty)

    if entry_short and state["mode"] == "mock":
        st.in_position, st.side = True, "short"
        st.entry_price, st.entry_qty = close, mock_qty_for(close)
        st.trail_stop = min(st.entry_price + ATR_STOP_MULT * atr14,
                            st.orb_high if st.orb_high>0 else st.entry_price*1.02)
        await mock_trade(chat_id, symbol, "sell", st.entry_price, st.entry_qty, extra={"note":"mock short"})

    # Trailing
    if st.in_position and st.side == "long":
        st.trail_stop = max(st.trail_stop, close - ATR_TRAIL_MULT * atr14)
    if st.in_position and st.side == "short":
        new_tr = close + ATR_TRAIL_MULT * atr14
        st.trail_stop = min(st.trail_stop if st.trail_stop>0 else new_tr, new_tr)

    # Exit
    exit_ok = False; exit_price = close
    if st.in_position:
        if st.side == "long":
            if (st.trail_stop and low < st.trail_stop) or (low < st.orb_low) or (low < st.entry_price - ATR_STOP_MULT*atr14):
                exit_ok = True
        else:  # short
            if (st.trail_stop and high > st.trail_stop) or (high > st.orb_high) or (high > st.entry_price + ATR_STOP_MULT*atr14):
                exit_ok = True

    if exit_ok:
        qty = st.entry_qty or 0.0
        if qty>0:
            if st.side == "long":
                gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
                extra = {"entry_price": st.entry_price, "exit_price": exit_price,
                         "gross_pnl": round(gross,8), "net_pnl": round(net,8)}
                if state["mode"] == "mock":
                    await mock_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
                else:
                    await live_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
                await notify_closed_trade(chat_id, symbol, st.entry_price, exit_price, qty)
            else:
                gross = (st.entry_price - exit_price) * qty
                fees  = (st.entry_price + exit_price) * qty * FEE_RATE
                net   = gross - fees
                extra = {"entry_price": st.entry_price, "exit_price": exit_price,
                         "gross_pnl": round(gross,8), "net_pnl": round(net,8), "note":"mock short exit"}
                await mock_trade(chat_id, symbol, "buy", exit_price, qty, extra=extra)
                state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
                emo = "📈" if net >= 0 else "📉"
                await tg_send_text(chat_id,
                    f"{emo} Stängd short {symbol}\nEntry: {st.entry_price:.6f}  Exit: {exit_price:.6f}\n"
                    f"Qty: {qty:g}\nGross: {gross:.6f}  Avgifter: {fees:.6f}\nNetto: {net:.6f}\n"
                    f"PnL idag: {state['pnl_today']:.6f}")
        st.in_position=False; st.side=""; st.entry_price=0.0; st.entry_qty=0.0; st.trail_stop=0.0

async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks: await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(ENGINE_LOOP_SEC)
        except asyncio.CancelledError:
            break
        except Exception as e:
            await tg_send_text(chat_id, f"Motorfel: {e}")
            await asyncio.sleep(5)

async def start_engine(chat_id: int|str):
    if state["engine_task"] and not state["engine_task"].done():
        await tg_send_text(chat_id, "Motorn kör redan."); return
    state["active"] = True
    loop = asyncio.get_event_loop()
    state["engine_task"] = loop.create_task(run_engine(chat_id))
    await tg_send_text(chat_id, f"🔄 Motor startad {utcnow_iso()} (tf={state['timeframe']}, symbols={', '.join(state['symbols'])}).")

async def stop_engine(chat_id: int|str):
    state["active"] = False
    t = state.get("engine_task")
    if t and not t.done(): t.cancel()
    await tg_send_text(chat_id, "⏹️ Motor stoppad.")

# ================== Keepalive ==================
async def keepalive_loop(url: str, chat_id: int|str):
    while state.get("keepalive"):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(url)
        except Exception:
            pass
        await asyncio.sleep(KEEPALIVE_SEC)  # 4 minuter default

# ================== Backtest (dummy) ==================
async def run_backtest(chat_id: int, symbol: str="BTCUSDT", period: str="3d", fee: float|None=None, out_csv: str="backtest_result.csv"):
    import random; random.seed(42)
    fee_rate = FEE_RATE if fee is None else float(fee)
    rows = []
    for _ in range(40):
        gross = random.uniform(-0.004, 0.006)  # -0.4%..+0.6%
        net = gross - 2*fee_rate
        rows.append({"timestamp": utcnow_iso(),"symbol":symbol.upper(),
                     "gross_return":round(gross,6),"fee_rate_each_side":fee_rate,
                     "net_return":round(net,6)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    await tg_send_text(chat_id, f"Backtest klart: {symbol} ({period}) avgift {fee_rate:.4f}")
    return out_csv

# ================== Kommandon ==================
async def handle_update(data: Dict[str, Any]):
    msg = data.get("message") or data.get("edited_message") or {}
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()
    if not chat_id or not text: return {"ok": True}
    if not is_admin(chat_id):
        await tg_send_text(chat_id, "⛔️ Du är inte behörig."); return {"ok": True}

    # Bekräftelser
    if state["pending_confirm"]:
        if text.lower() == "ja":
            mode = state["pending_confirm"]["type"]
            state["mode"] = mode; state["pending_confirm"] = None
            await tg_send_text(chat_id, f"✅ {mode.upper()} aktiverad."); await start_engine(chat_id)
        else:
            await tg_send_text(chat_id, "Avbrutet."); state["pending_confirm"] = None
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
            "/timeframe 1min|3min|5min|15min|...\n"
            "/backtest [SYMBOL] [PERIOD] [FEE]\n"
            "/export_csv – skicka loggar\n"
            "/mock_trade SYMBOL SIDE PRICE QTY – manuell mock\n"
            "/pnl – visa dagens PnL\n"
            "/reset_pnl – nollställ dagens PnL\n"
            "/keepalive_on – håll vaken\n"
            "/keepalive_off – stäng keepalive")
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}  AI: {state['ai_mode']}\n"
            f"Avgift/sida: {FEE_RATE:.4f}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}  Mock USDT: {MOCK_USDT_PER_TRADE:.2f}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Min range: {MIN_ORB_RANGE:.4%}\n"
            f"Trendfilter: {'på' if USE_EMA_TREND else 'av'}  ATR stop x{ATR_STOP_MULT}  trail x{ATR_TRAIL_MULT}\n"
            f"CandleFilter: {'på' if USE_CANDLE_FILTERS else 'av'}  Keepalive: {'på' if state.get('keepalive') else 'av'} ({KEEPALIVE_SEC}s)")
        return {"ok": True}

    if text.startswith("/set_ai"):
        parts = text.split(maxsplit=1)
        if len(parts)<2: await tg_send_text(chat_id,"Använd: /set_ai neutral|aggressiv|försiktig"); return {"ok": True}
        nm = parts[1].strip().lower(); nm = "försiktig" if nm=="forsiktig" else nm
        if nm not in {"neutral","aggressiv","försiktig"}: await tg_send_text(chat_id,"Ogiltigt AI-läge."); return {"ok": True}
        state["ai_mode"] = nm; await tg_send_text(chat_id,f"AI-läge satt till: {nm}"); return {"ok": True}

    if text.startswith("/start_mock"):
        state["pending_confirm"] = {"type":"mock"}
        await tg_send_text(chat_id,"Vill du starta MOCK-trading? Svara JA."); return {"ok": True}

    if text.startswith("/start_live"):
        state["pending_confirm"] = {"type":"live"}
        await tg_send_text(chat_id,"⚠️ Vill du starta LIVE-trading? Svara JA."); return {"ok": True}

    if text.startswith("/engine_start"): await start_engine(chat_id); return {"ok": True}
    if text.startswith("/engine_stop") or text.startswith("/stop"): await stop_engine(chat_id); return {"ok": True}

    if text.startswith("/symbols"):
        parts = text.split(maxsplit=1)
        if len(parts)<2: await tg_send_text(chat_id,f"Aktuell: {', '.join(state['symbols'])}"); return {"ok": True}
        syms=[s for s in parts[1].replace(" ","").split(",") if s]
        state["symbols"]=[s.upper() for s in syms]
        await tg_send_text(chat_id,f"Symbols uppdaterade: {', '.join(state['symbols'])}"); return {"ok": True}

    if text.startswith("/timeframe"):
        tf = (text.split()[1] if len(text.split())>1 else "3min").lower()
        state["timeframe"]=tf; await tg_send_text(chat_id,f"Tidsram satt till: {tf}"); return {"ok": True}

    if text.startswith("/backtest"):
        parts=text.split(); symbol=parts[1] if len(parts)>1 else "BTCUSDT"
        period=parts[2] if len(parts)>2 else "3d"; fee=float(parts[3]) if len(parts)>3 else None
        path=await run_backtest(chat_id,symbol=symbol.upper(),period=period,fee=fee)
        await tg_send_document(chat_id,path,caption=f"Backtest {symbol} ({period})"); return {"ok": True}

    if text.startswith("/export_csv"):
        sent=False
        for p in (MOCK_LOG,REAL_LOG):
            if os.path.exists(p): await tg_send_document(chat_id,p,caption=p); sent=True
        if not sent: await tg_send_text(chat_id,"Inga loggar hittades ännu.")
        return {"ok": True}

    if text.startswith("/mock_trade"):
        parts=text.split()
        if len(parts)<5: await tg_send_text(chat_id,"Använd: /mock_trade SYMBOL SIDE PRICE QTY"); return {"ok": True}
        _,sym,side,price,qty=parts[:5]; await mock_trade(chat_id,sym,side,float(price),float(qty)); return {"ok": True}

    if text.startswith("/pnl"): await tg_send_text(chat_id,f"Ack PnL idag: {state.get('pnl_today',0.0):.6f}"); return {"ok": True}
    if text.startswith("/reset_pnl"): state["pnl_today"]=0.0; await tg_send_text(chat_id,"PnL nollställt."); return {"ok": True}

    if text.startswith("/keepalive_on"):
        public_url = f"https://{os.getenv('RENDER_EXTERNAL_URL','')}/"
        if not public_url or public_url == "https:///": public_url = "https://mporbbot.onrender.com/"
        if state.get("keepalive"): await tg_send_text(chat_id,"Keepalive är redan på."); return {"ok": True}
        state["keepalive"]=True; loop=asyncio.get_event_loop()
        state["keepalive_task"]=loop.create_task(keepalive_loop(public_url, chat_id))
        await tg_send_text(chat_id, f"Keepalive PÅ – pingar {public_url} var {KEEPALIVE_SEC//60}:e minut."); return {"ok": True}

    if text.startswith("/keepalive_off"):
        state["keepalive"]=False
        t=state.get("keepalive_task")
        if t and not t.done(): t.cancel()
        await tg_send_text(chat_id,"Keepalive AV."); return {"ok": True}

    return {"ok": True}

# ================== FastAPI ==================
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {
        "message": "Mp ORBbot is live!",
        "webhook_path": WEBHOOK_PATH,
        "params": {
            "mode": state["mode"], "ai_mode": state["ai_mode"],
            "symbols": state["symbols"], "timeframe": state["timeframe"],
            "ORB_BUFFER": ORB_BUFFER, "MIN_ORB_RANGE": MIN_ORB_RANGE,
            "USE_EMA_TREND": USE_EMA_TREND,
            "ATR_STOP_MULT": ATR_STOP_MULT, "ATR_TRAIL_MULT": ATR_TRAIL_MULT,
            "MOCK_USDT_PER_TRADE": MOCK_USDT_PER_TRADE,
            "KEEPALIVE_SEC": KEEPALIVE_SEC, "AUTOSTART": AUTOSTART
        }
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)

# Auto-keepalive + ev. autostart när appen vaknar
@app.on_event("startup")
async def on_startup():
    public_url = f"https://{os.getenv('RENDER_EXTERNAL_URL','')}/"
    if not public_url or public_url == "https:///":
        public_url = "https://mporbbot.onrender.com/"
    # Starta keepalive direkt
    if not state.get("keepalive"):
        state["keepalive"] = True
        loop = asyncio.get_event_loop()
        state["keepalive_task"] = loop.create_task(keepalive_loop(public_url, ADMIN_CHAT_ID or ""))

    # Autostarta motorn (mock) om aktiverat
    if AUTOSTART and not state["active"]:
        if state["mode"] not in ("mock","live"):
            state["mode"] = "mock"
        await start_engine(ADMIN_CHAT_ID or "")

# Lokal körning
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
