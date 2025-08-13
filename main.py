# main.py — Mp ORBbot (full)
# FastAPI Telegram-webhook + ORB "första candle" logik (long & mock-short),
# ATR-baserad trailstop, intrabar-entry, mock/live-logging & PnL, keepalive,
# och ett gäng Telegram-kommandon.

from __future__ import annotations
import os, csv, asyncio, json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ========================= Miljö/inställningar =========================
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID          = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()   # tomt = tillåt alla
FEE_RATE               = float(os.getenv("FEE_RATE", "0.001"))             # 0.1% per sida
DEFAULT_SYMBOLS        = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]
DEFAULT_TIMEFRAME      = os.getenv("DEFAULT_TIMEFRAME", "3min")

# ORB/entry/exit parametrar
ORB_BUFFER             = float(os.getenv("ORB_BUFFER", "0.0005"))   # 0.05% (fallback)
ORB_SEED_BUFFER        = float(os.getenv("ORB_SEED_BUFFER", "0.0002")) # 0.02% för seed
MIN_ORB_RANGE          = float(os.getenv("MIN_ORB_RANGE", "0.0005")) # 0.05% min range
USE_EMA_TREND          = os.getenv("USE_EMA_TREND", "0") == "1"     # valfritt trendfilter
ATR_STOP_MULT          = float(os.getenv("ATR_STOP_MULT", "1.0"))
ATR_TRAIL_MULT         = float(os.getenv("ATR_TRAIL_MULT", "0.5"))
INITIAL_ORB_ON_START   = os.getenv("INITIAL_ORB_ON_START", "1") == "1"

# Motor
ENGINE_LOOP_SEC        = int(os.getenv("ENGINE_LOOP_SEC", "8"))
MOCK_USDT              = float(os.getenv("MOCK_USDT", "30"))        # per trade (mock)

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

# ========================= Globalt state =========================
state: Dict[str, Any] = {
    "active": False,              # motor kör
    "mode": "mock",               # mock | live
    "ai_mode": "neutral",         # neutral vid start
    "pending_confirm": None,      # {"type":"mock"|"live"}
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": DEFAULT_TIMEFRAME,
    "engine_task": None,
    "per_symbol": {},             # symbol -> ORBState
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
}

# ========================= Telegram utils =========================
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

# ========================= Hjälpare =========================
def ai_allows(ai_mode: str, strength: float) -> bool:
    m = (ai_mode or "neutral").lower()
    if m == "aggressiv": return strength >= 0.2
    if m in ("försiktig", "forsiktig"): return strength >= 0.7
    return strength >= 0.5

def write_log_row(path: str, row: Dict[str, Any]):
    newfile = not os.path.exists(path)
    for k in LOG_FIELDS:
        row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})

def compute_pnl_long(entry: float, exit: float, qty: float, fee_rate: float):
    gross = (exit - entry) * qty
    fees  = (entry + exit) * qty * fee_rate
    return gross, fees, gross - fees

def compute_pnl_short(entry: float, exit: float, qty: float, fee_rate: float):
    gross = (entry - exit) * qty
    fees  = (entry + exit) * qty * fee_rate
    return gross, fees, gross - fees

def mock_qty_for(price: float) -> float:
    # enkel storleksberäkning för mock: köp för MOCK_USDT
    if price <= 0: return 0.0
    return round(MOCK_USDT / price, 6)

def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"):
        return f"{usdt_sym[:-4]}-USDT"
    return usdt_sym.replace("/", "-").upper()

# ========================= Data/indikatorer =========================
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
    df = df.sort_values("ts").reset_index(drop=True)
    return df.tail(limit)

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

# ========================= ORB-state =========================
@dataclass
class ORBState:
    direction: str = "flat"    # "up"|"down"|"flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    side: str = ""             # "long"|"short"
    entry_price: float = 0.0
    entry_qty: float = 0.0
    trail_stop: float = 0.0
    last_price: float = 0.0
    seed_set: bool = False
    seed_color: str = ""       # "green"|"red"

def candle_color(row) -> str:
    return "green" if float(row["close"]) >= float(row["open"]) else "red"

def detect_dir(prev_close: float, close: float) -> str:
    if close > prev_close: return "up"
    if close < prev_close: return "down"
    return "flat"

# ========================= Notifier för stängd affär =========================
async def notify_close_long(chat_id: int, symbol: str, entry: float, exit: float, qty: float):
    g, f, n = compute_pnl_long(entry, exit, qty, FEE_RATE)
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + n
    emo = "📈" if n >= 0 else "📉"
    await tg_send_text(chat_id,
        f"{emo} Stängd LONG {symbol}\nEntry: {entry:.6f}  Exit: {exit:.6f}\n"
        f"Qty: {qty:g}\nGross: {g:.6f}  Avg: {f:.6f}\nNetto: {n:.6f}\nPnL idag: {state['pnl_today']:.6f}")

async def notify_close_short(chat_id: int, symbol: str, entry: float, exit: float, qty: float):
    g, f, n = compute_pnl_short(entry, exit, qty, FEE_RATE)
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + n
    emo = "📈" if n >= 0 else "📉"
    await tg_send_text(chat_id,
        f"{emo} Stängd SHORT {symbol}\nEntry: {entry:.6f}  Exit: {exit:.6f}\n"
        f"Qty: {qty:g}\nGross: {g:.6f}  Avg: {f:.6f}\nNetto: {n:.6f}\nPnL idag: {state['pnl_today']:.6f}")

# ========================= Trade-logg helpers =========================
async def mock_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str,Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "mock", "symbol": symbol.upper(),
        "side": side, "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost,8),
        "gross_pnl":"", "net_pnl":"", "entry_price":"", "exit_price":"",
        "note":"mock"
    }
    if extra: row.update(extra)
    write_log_row(MOCK_LOG, row)

async def live_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str,Any]]=None):
    # Stub: logga bara
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "live", "symbol": symbol.upper(),
        "side": side, "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost,8),
        "gross_pnl":"", "net_pnl":"", "entry_price":"", "exit_price":"",
        "note":"live (stub)"
    }
    if extra: row.update(extra)
    write_log_row(REAL_LOG, row)

# ========================= Motor (”första candle”-ORB) =========================
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    st: ORBState = state["per_symbol"].setdefault(symbol, ORBState())

    df = await fetch_candles(symbol, timeframe, limit=90)
    if df.empty or len(df) < 55:
        return

    prev, last = df.iloc[-2], df.iloc[-1]
    st.last_price = float(last["close"])

    last_col = candle_color(last)
    last_dir = detect_dir(float(prev["close"]), float(last["close"]))

    # Riktning bytt → reset seed/box/position (behåll riktning)
    if last_dir != st.direction and last_dir in ("up","down"):
        st.direction = last_dir
        st.seed_set = False
        st.seed_color = ""
        st.orb_high = 0.0
        st.orb_low  = 0.0
        st.in_position = False
        st.side = ""
        st.entry_price = 0.0
        st.entry_qty = 0.0
        st.trail_stop = 0.0

    # Initial seed vid start om parameter tillåter
    if INITIAL_ORB_ON_START and st.direction in ("up","down") and not st.seed_set:
        # använd *senaste* ljus som seed om det matchar riktningens färg
        want = "green" if st.direction == "up" else "red"
        if last_col == want:
            st.seed_set = True
            st.seed_color = want
            st.orb_high = float(last["high"])
            st.orb_low  = float(last["low"])

    # Om vi saknar seed och den här candelns färg matchar riktning → sätt seed
    if not st.seed_set and st.direction in ("up","down"):
        if st.direction == "up" and last_col == "green":
            st.seed_set = True; st.seed_color="green"
            st.orb_high=float(last["high"]); st.orb_low=float(last["low"])
        elif st.direction == "down" and last_col == "red":
            st.seed_set = True; st.seed_color="red"
            st.orb_high=float(last["high"]); st.orb_low=float(last["low"])

    # Indikatorer
    ema20 = float(ema(df["close"], 20).iloc[-1])
    ema50 = float(ema(df["close"], 50).iloc[-1])
    atr14 = float(atr(df, 14).iloc[-1])

    # Min range
    orb_ok = True
    if st.orb_high > 0 and st.orb_low > 0 and st.orb_low > 0:
        orb_ok = (st.orb_high / st.orb_low - 1.0) >= MIN_ORB_RANGE

    close = float(last["close"])
    high  = float(last["high"])
    low   = float(last["low"])
    seed_buf = max(ORB_SEED_BUFFER, ORB_BUFFER)

    # ENTRY — intrabar break
    entry_long = (
        (not st.in_position) and st.seed_set and st.seed_color == "green" and
        st.direction == "up" and orb_ok and
        high > st.orb_high * (1.0 + seed_buf) and
        (not USE_EMA_TREND or ema20 > ema50) and
        ai_allows(state["ai_mode"], 0.6)
    )
    entry_short = (
        (not st.in_position) and st.seed_set and st.seed_color == "red" and
        st.direction == "down" and orb_ok and
        low < st.orb_low * (1.0 - seed_buf) and
        (not USE_EMA_TREND or ema20 < ema50) and
        ai_allows(state["ai_mode"], 0.6)
    )

    if entry_long:
        st.in_position, st.side = True, "long"
        st.entry_price = close
        st.entry_qty = max(mock_qty_for(close), 0.000001)
        st.trail_stop = max(st.entry_price - ATR_STOP_MULT * atr14, st.orb_low)
        if state["mode"] == "mock":
            await mock_trade(chat_id, symbol, "buy", st.entry_price, st.entry_qty, extra={"note":"seed-long"})
        else:
            await live_trade(chat_id, symbol, "buy", st.entry_price, st.entry_qty)

    if entry_short and state["mode"] == "mock":  # short endast i mock
        st.in_position, st.side = True, "short"
        st.entry_price = close
        st.entry_qty = max(mock_qty_for(close), 0.000001)
        st.trail_stop = min(st.entry_price + ATR_STOP_MULT * atr14,
                            st.orb_high if st.orb_high>0 else st.entry_price*1.02)
        await mock_trade(chat_id, symbol, "sell", st.entry_price, st.entry_qty, extra={"note":"seed-short"})

    # TRAIL uppdatering
    if st.in_position and st.side == "long":
        st.trail_stop = max(st.trail_stop, close - ATR_TRAIL_MULT * atr14)
    if st.in_position and st.side == "short":
        st.trail_stop = min(st.trail_stop if st.trail_stop>0 else close + ATR_TRAIL_MULT*atr14,
                            close + ATR_TRAIL_MULT * atr14)

    # EXIT – första motsatt färg eller trail
    exit_now = False
    exit_price = close
    if st.in_position:
        if st.side == "long" and last_col == "red":  exit_now = True
        if st.side == "short" and last_col == "green": exit_now = True
        if st.side == "long" and st.trail_stop and low < st.trail_stop: exit_now = True
        if st.side == "short" and st.trail_stop and high > st.trail_stop: exit_now = True

    if exit_now and st.in_position:
        qty = st.entry_qty or 0.0
        if st.side == "long":
            g,f,n = compute_pnl_long(st.entry_price, exit_price, qty, FEE_RATE)
            extra = {"entry_price":st.entry_price,"exit_price":exit_price,
                     "gross_pnl":round(g,8),"net_pnl":round(n,8)}
            if state["mode"] == "mock":
                await mock_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            else:
                await live_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            await notify_close_long(chat_id, symbol, st.entry_price, exit_price, qty)
        else:
            g,f,n = compute_pnl_short(st.entry_price, exit_price, qty, FEE_RATE)
            extra = {"entry_price":st.entry_price,"exit_price":exit_price,
                     "gross_pnl":round(g,8),"net_pnl":round(n,8),"note":"seed-short exit"}
            await mock_trade(chat_id, symbol, "buy", exit_price, qty, extra=extra)
            await notify_close_short(chat_id, symbol, st.entry_price, exit_price, qty)

        st.in_position = False
        st.side = ""
        st.entry_price = 0.0
        st.entry_qty = 0.0
        st.trail_stop = 0.0
        # seed/box behålls tills riktningen byter

# ========================= Motor-loop =========================
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
            await asyncio.sleep(3)

async def start_engine(chat_id: int):
    if state["engine_task"] and not state["engine_task"].done():
        await tg_send_text(chat_id, "Motorn kör redan."); return
    state["active"] = True
    loop = asyncio.get_event_loop()
    state["engine_task"] = loop.create_task(run_engine(chat_id))
    await tg_send_text(chat_id, f"🔄 Motor startad {utcnow_iso()} (tf={state['timeframe']}, symbols={', '.join(state['symbols'])}).")

async def stop_engine(chat_id: int):
    state["active"] = False
    t = state.get("engine_task")
    if t and not t.done(): t.cancel()
    await tg_send_text(chat_id, "⏹️ Motor stoppad.")

# ========================= Keepalive =========================
async def keepalive_loop(url: str):
    while state.get("keepalive"):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.get(url)
        except Exception:
            pass
        await asyncio.sleep(12*60)

# ========================= Backtest (dummy) =========================
async def run_backtest(chat_id: int, symbol: str="BTCUSDT", period: str="3d", fee: float|None=None, out_csv: str="backtest_result.csv"):
    import random
    random.seed(42)
    fee_rate = FEE_RATE if fee is None else float(fee)
    rows = []
    for _ in range(50):
        gross = random.uniform(-0.004, 0.006)
        net = gross - 2*fee_rate
        rows.append({"timestamp": utcnow_iso(), "symbol": symbol.upper(), "gross_return": round(gross,6), "net_return": round(net,6), "fee_each_side": fee_rate})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    await tg_send_text(chat_id, f"Backtest klart {symbol} ({period})")
    return out_csv

# ========================= Telegram-kommandon =========================
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

    # Bekräftelser
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

    # ---- Kommandon ----
    if text.startswith("/help"):
        await tg_send_text(chat_id,
            "/status\n"
            "/set_ai <neutral|aggressiv|försiktig>\n"
            "/start_mock (svara JA)\n"
            "/start_live (svara JA)\n"
            "/engine_start  /engine_stop\n"
            "/symbols BTCUSDT,ETHUSDT,...\n"
            "/timeframe 1min|3min|5min|15min\n"
            "/backtest [SYMBOL] [PERIOD] [FEE]\n"
            "/export_csv  /mock_trade SYMBOL SIDE PRICE QTY\n"
            "/pnl  /reset_pnl\n"
            "/keepalive_on  /keepalive_off")
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}\nAI: {state['ai_mode']}\n"
            f"Avgift per sida: {FEE_RATE:.4f}\n"
            f"Symbols: {', '.join(state['symbols'])}\nTF: {state['timeframe']}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Seed buffer: {ORB_SEED_BUFFER:.4%}\n"
            f"Min range: {MIN_ORB_RANGE:.4%}  Trendfilter: {'på' if USE_EMA_TREND else 'av'}\n"
            f"ATR stop x{ATR_STOP_MULT} trail x{ATR_TRAIL_MULT}\n"
            f"Mock USDT: {MOCK_USDT:.2f}")
        return {"ok": True}

    if text.startswith("/set_ai"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send_text(chat_id, "Använd: /set_ai neutral|aggressiv|försiktig"); return {"ok": True}
        nm = parts[1].strip().lower()
        if nm == "forsiktig": nm = "försiktig"
        if nm not in {"neutral","aggressiv","försiktig"}:
            await tg_send_text(chat_id, "Ogiltigt AI-läge."); return {"ok": True}
        state["ai_mode"] = nm; await tg_send_text(chat_id, f"AI-läge: {nm}"); return {"ok": True}

    if text.startswith("/start_mock"):
        state["pending_confirm"] = {"type":"mock"}
        await tg_send_text(chat_id, "Starta MOCK? Svara JA."); return {"ok": True}

    if text.startswith("/start_live"):
        state["pending_confirm"] = {"type":"live"}
        await tg_send_text(chat_id, "⚠️ Starta LIVE? Svara JA."); return {"ok": True}

    if text.startswith("/engine_start"):
        await start_engine(chat_id); return {"ok": True}

    if text.startswith("/engine_stop") or text.startswith("/stop"):
        await stop_engine(chat_id); return {"ok": True}

    if text.startswith("/symbols"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send_text(chat_id, f"Aktuell lista: {', '.join(state['symbols'])}"); return {"ok": True}
        syms = [s for s in parts[1].replace(" ","").split(",") if s]
        state["symbols"] = [s.upper() for s in syms]
        await tg_send_text(chat_id, f"Symbols uppdaterade: {', '.join(state['symbols'])}")
        return {"ok": True}

    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts)>1 else "3min"
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram satt till: {tf}")
        return {"ok": True}

    if text.startswith("/backtest"):
        parts = text.split()
        symbol = parts[1] if len(parts)>1 else "BTCUSDT"
        period = parts[2] if len(parts)>2 else "3d"
        fee = float(parts[3]) if len(parts)>3 else None
        path = await run_backtest(chat_id, symbol.upper(), period, fee)
        await tg_send_document(chat_id, path, caption=f"Backtest {symbol} {period}")
        return {"ok": True}

    if text.startswith("/export_csv"):
        sent=False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p); sent=True
        if not sent: await tg_send_text(chat_id, "Inga loggar ännu.")
        return {"ok": True}

    if text.startswith("/mock_trade"):
        parts = text.split()
        if len(parts) < 5:
            await tg_send_text(chat_id, "Använd: /mock_trade SYMBOL SIDE PRICE QTY"); return {"ok": True}
        _, sym, side, price, qty = parts[:5]
        await mock_trade(chat_id, sym.upper(), side.lower(), float(price), float(qty))
        return {"ok": True}

    if text.startswith("/pnl"):
        await tg_send_text(chat_id, f"Ack. PnL idag: {state.get('pnl_today',0.0):.6f}")
        return {"ok": True}

    if text.startswith("/reset_pnl"):
        state["pnl_today"] = 0.0; await tg_send_text(chat_id, "PnL nollställt."); return {"ok": True}

    if text.startswith("/keepalive_on"):
        public_url = f"https://{os.getenv('RENDER_EXTERNAL_URL','')}/"
        if public_url in {"https:///",""}: public_url = "https://mporbbot.onrender.com/"
        if state.get("keepalive"): await tg_send_text(chat_id, "Keepalive är redan på."); return {"ok": True}
        state["keepalive"] = True
        loop = asyncio.get_event_loop()
        state["keepalive_task"] = loop.create_task(keepalive_loop(public_url))
        await tg_send_text(chat_id, f"Keepalive PÅ – pingar {public_url} var 12:e minut.")
        return {"ok": True}

    if text.startswith("/keepalive_off"):
        state["keepalive"] = False
        t = state.get("keepalive_task")
        if t and not t.done(): t.cancel()
        await tg_send_text(chat_id, "Keepalive AV.")
        return {"ok": True}

    return {"ok": True}

# ========================= FastAPI =========================
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {
        "message": "Mp ORBbot is live!",
        "webhook_path": WEBHOOK_PATH,
        "params": {
            "fee_rate": FEE_RATE,
            "symbols": state["symbols"],
            "timeframe": state["timeframe"],
            "orb_buffer": ORB_BUFFER,
            "seed_buffer": ORB_SEED_BUFFER,
            "min_orb_range": MIN_ORB_RANGE,
            "ema_trend": USE_EMA_TREND,
            "atr_stop": ATR_STOP_MULT,
            "atr_trail": ATR_TRAIL_MULT,
            "loop_sec": ENGINE_LOOP_SEC,
            "mock_usdt": MOCK_USDT
        }
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
