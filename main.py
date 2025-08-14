# main.py — Mp ORBbot (ORB + entry på close + SL = föregående candle close)
# - ORB på första candle som byter färg (grön<->röd)
# - LONG entry: första candle som STÄNGER över ORB-high
# - Dynamisk SL: flyttas upp till föregående candle CLOSE vid varje stängning
# - Mock/live-läge, CSV-loggar, Telegram-kommandon, FastAPI-webhook

from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ============== Miljö & defaults ==============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()

FEE_RATE           = float(os.getenv("FEE_RATE", "0.001"))  # 0.10% per sida
DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]
DEFAULT_TIMEFRAME  = os.getenv("DEFAULT_TIMEFRAME", "1min").strip()
MOCK_USDT          = float(os.getenv("MOCK_USDT", "30"))
MIN_ORB_PCT        = 0.0010  # min 0.10% ORB-range (filtrera brus)

# CSV
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

# ============== Globalt state ==============
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",
    "ai_mode": "neutral",
    "pending_confirm": None,
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": DEFAULT_TIMEFRAME,
    "engine_task": None,
    "per_symbol": {},      # symbol -> ORBState
    "pnl_today": 0.0,
}

# ============== Telegram ==============
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

# ============== Logg & PnL ==============
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
    fees  = (entry_price + exit_price) * qty * fee_rate
    net   = gross - fees
    return gross, fees, net

async def log_trade(mode: str, symbol: str, side: str, price: float, qty: float, extra: Dict[str,Any]|None=None):
    path = MOCK_LOG if mode=="mock" else REAL_LOG
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": mode, "symbol": symbol.upper(),
        "side": side, "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8)
    }
    if extra: row.update(extra)
    write_log_row(path, row)

# ============== Datahämtning ==============
def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"):
        return f"{usdt_sym[:-4]}-USDT"
    return usdt_sym.replace("/", "-").upper()

async def fetch_candles(symbol: str, timeframe: str="1min", limit: int=200) -> pd.DataFrame:
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

# ============== ORB & position ==============
@dataclass
class ORBState:
    last_color: str = ""                # "green" | "red"
    orb_set_at: Optional[pd.Timestamp] = None
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    entry_price: float = 0.0
    entry_qty: float = 0.0
    dyn_sl: float = 0.0                 # dynamisk SL (föregående candle close)
    last_closed_close: float = 0.0      # senaste stängda candle close (för att flytta SL)

def candle_color(row: pd.Series) -> str:
    if row["close"] > row["open"]: return "green"
    if row["close"] < row["open"]: return "red"
    return "doji"

def reset_orb(st: ORBState, candle: pd.Series):
    st.orb_set_at = candle["ts"]
    st.orb_high = float(candle["high"])
    st.orb_low  = float(candle["low"])

# ============== Motor ==============
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, timeframe, limit=240)
    if df.empty or len(df) < 5:
        return

    # Arbeta på senaste fönstret
    view = df.iloc[-60:].copy()
    for i in range(1, len(view)):
        prev = view.iloc[i-1]
        cur  = view.iloc[i]

        col_prev = candle_color(prev)
        col_cur  = candle_color(cur)

        # initial färg
        if st.last_color == "" and col_prev in ("green","red"):
            st.last_color = col_prev

        # Ny ORB vid första färgbytet
        if col_cur in ("green","red") and st.last_color in ("green","red") and col_cur != st.last_color:
            reset_orb(st, cur)
            st.last_color = col_cur
        else:
            if col_cur in ("green","red"):
                st.last_color = col_cur

        # ORB-range filter
        if st.orb_high > 0 and st.orb_low > 0 and st.orb_low > 0:
            orb_ok = (st.orb_high / st.orb_low - 1.0) >= MIN_ORB_PCT
        else:
            orb_ok = False

        # Spara senaste STÄNGDA candle close (för SL-flytt)
        st.last_closed_close = float(prev["close"])

        # ENTRY: första candle som STÄNGER över ORB-high (efter att ORB sattes)
        if (not st.in_position) and orb_ok and st.orb_set_at is not None:
            if cur["ts"] > st.orb_set_at and cur["close"] > st.orb_high:
                entry_price = float(cur["close"])
                qty = round(MOCK_USDT / entry_price, 8)
                st.in_position = True
                st.entry_price = entry_price
                st.entry_qty = qty
                # initiera dynamisk SL med entry-close (kan även vara prev close om du vill)
                st.dyn_sl = st.last_closed_close if st.last_closed_close > 0 else entry_price
                await log_trade(state["mode"], symbol, "buy", entry_price, qty, {"note": "entry_close_breakout"})
                await tg_send_text(chat_id, f"✅ {state['mode'].upper()}: BUY {symbol} @ {entry_price:.6f} x {qty:g}")
                continue

        # Hantera öppen position (dynamisk SL)
        if st.in_position:
            price_low   = float(cur["low"])
            price_close = float(cur["close"])

            # 1) Flytta upp SL till föregående candle CLOSE om den är högre än nuvarande SL
            if st.last_closed_close > st.dyn_sl:
                st.dyn_sl = st.last_closed_close

            # 2) Exit om vi bryter SL intrabar (low <= SL)
            if price_low <= st.dyn_sl:
                exit_price = st.dyn_sl  # anta fill vid SL
                qty = st.entry_qty or 0.0
                gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
                await log_trade(state["mode"], symbol, "sell", exit_price, qty, {
                    "entry_price": st.entry_price, "exit_price": exit_price,
                    "gross_pnl": round(gross,8), "net_pnl": round(net,8),
                    "note": "dyn_close_SL"
                })
                state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
                emo = "📈" if net >= 0 else "📉"
                await tg_send_text(chat_id,
                    f"{emo} Stängd LONG {symbol}\n"
                    f"Entry: {st.entry_price:.6f}  Exit: {exit_price:.6f}\n"
                    f"Qty: {qty:g}\nPnL idag: {state['pnl_today']:.6f}"
                )
                # nollställ
                st.in_position = False
                st.entry_price = st.entry_qty = 0.0
                st.dyn_sl = 0.0
                continue

# Huvudloop
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(10)  # kontroll varje 10:e sekund (hämtar 1m/3m candles ändå)
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

# ============== Kommandon ==============
async def handle_update(data: Dict[str, Any]):
    msg = data.get("message") or data.get("edited_message") or {}
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id or not text: return {"ok": True}
    if not is_admin(chat_id):
        await tg_send_text(chat_id, "⛔️ Du är inte behörig.")
        return {"ok": True}

    # pending JA
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

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"Min ORB: {MIN_ORB_PCT:.4%}"
        )
        return {"ok": True}

    if text.startswith("/start_mock"):
        state["pending_confirm"] = {"type": "mock"}
        await tg_send_text(chat_id, "Starta MOCK? Svara JA.")
        return {"ok": True}

    if text.startswith("/start_live"):
        state["pending_confirm"] = {"type": "live"}
        await tg_send_text(chat_id, "⚠️ Starta LIVE? Svara JA.")
        return {"ok": True}

    if text.startswith("/engine_start"):
        await start_engine(chat_id);  return {"ok": True}

    if text.startswith("/engine_stop"):
        await stop_engine(chat_id);   return {"ok": True}

    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts) > 1 else state["timeframe"]
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram satt till: {tf}")
        return {"ok": True}

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

    if text.startswith("/export_csv"):
        sent = False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p)
                sent = True
        if not sent:
            await tg_send_text(chat_id, "Inga loggar ännu.")
        return {"ok": True}

    if text.startswith("/pnl"):
        await tg_send_text(chat_id, f"PnL idag: {state.get('pnl_today',0.0):.6f}")
        return {"ok": True}

    if text.startswith("/reset_pnl"):
        state["pnl_today"] = 0.0
        await tg_send_text(chat_id, "PnL nollställt.")
        return {"ok": True}

    return {"ok": True}

# ============== FastAPI webhook ==============
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {
        "message": "Mp ORBbot is live!",
        "webhook_path": WEBHOOK_PATH,
        "params": {"MIN_ORB_PCT": MIN_ORB_PCT}
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
