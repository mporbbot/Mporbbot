# main.py — Mp ORBbot (en-fil)
# ORB skapas på första färgbyte-candle (röd<->grön). Entry kräver close-brott.
# Regler:
#  - LONG: close > ORB-high  | SHORT: close < ORB-low
#  - TP: +0.12% | SL: -0.05%
#  - Break-even/Trail: vid +0.10% flytta stop >= entry ± avgifter och traila
#  - Mock default, live = logg-stub
#  - CSV-loggar: mock_trade_log.csv / real_trade_log.csv
# Kommandon:
#  /status, /set_ai, /start_mock, /start_live, /engine_start, /engine_stop,
#  /symbols, /timeframe, /export_csv, /pnl, /reset_pnl, /keepalive_on, /keepalive_off

from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ---------- Miljö & standard ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()
FEE_RATE           = float(os.getenv("FEE_RATE", "0.001"))  # 0.1% per sida
DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]
MOCK_USDT          = float(os.getenv("MOCK_USDT", "30"))

# Risk/TP/SL/trail
TP_PCT      = 0.0012  # +0.12%
SL_PCT      = 0.0005  # -0.05%
BE_TRIGGER  = 0.0010  # +0.10%
TRAIL_STEP  = 0.0002  # 0.02% steg efter BE

MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"
KU_PUBLIC = "https://api.kucoin.com"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------- Telegram ----------
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

# ---------- Logg & PnL ----------
LOG_FIELDS = [
    "timestamp","mode","symbol","side","price","qty",
    "fee_rate","fee_cost","gross_pnl","net_pnl",
    "entry_price","exit_price","note"
]

def write_log_row(path: str, row: Dict[str, Any]):
    newfile = not os.path.exists(path)
    for k in LOG_FIELDS:
        row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})

def compute_pnl(entry: float, exit_: float, qty: float, fee_rate: float, side: str):
    # PnL sign baserat på side
    if side.lower() == "long":
        gross = (exit_ - entry) * qty
    else:  # short
        gross = (entry - exit_) * qty
    fees  = (entry + exit_) * qty * fee_rate
    net   = gross - fees
    return gross, fees, net

# ---------- KuCoin candlesticks ----------
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
    # KuCoin: [time, open, close, high, low, volume, turnover]
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

# ---------- ORB State ----------
@dataclass
class ORBState:
    last_color: str = ""        # "green" | "red" | ""
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    side: str = ""              # "long" | "short"
    entry_price: float = 0.0
    entry_qty: float = 0.0
    stop_price: float = 0.0     # aktiv SL (inkl. BE/trail)
    tp_price: float = 0.0

def candle_color(o: float, c: float) -> str:
    if c > o: return "green"
    if c < o: return "red"
    return "doji"

def reset_orb_on_change(st: ORBState, hi: float, lo: float, new_color: str):
    st.orb_high = float(hi)
    st.orb_low  = float(lo)
    st.last_color = new_color
    # Obs: position behålls/döms separat (vi nollställer ej position här)

# ---------- Globalt kör-state ----------
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",         # "mock" | "live"
    "ai_mode": "neutral",   # påverkar inte reglerna här, kvar för kompat.
    "pending_confirm": None,
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": "3min",
    "engine_task": None,
    "per_symbol": {},       # symbol -> ORBState
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
}

# ---------- Order-stubs ----------
def calc_qty(usdt: float, price: float) -> float:
    if price <= 0: return 0.0
    q = usdt / price
    return round(max(q, 0.0001), 8)

async def log_trade(chat_id: int, mode: str, symbol: str, side: str, price: float, qty: float, note: str="", extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": mode, "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": note
    }
    if extra: row.update(extra)
    path = MOCK_LOG if mode == "mock" else REAL_LOG
    write_log_row(path, row)
    await tg_send_text(chat_id, f"{'✅' if mode=='mock' else '🟢'} {mode.upper()}: {side.upper()} {symbol} @ {price:.6f} x {qty:g}")

async def close_and_log(chat_id: int, st: ORBState, symbol: str, exit_price: float):
    if not st.in_position: return
    side = st.side
    qty  = st.entry_qty or 0.0
    gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE, side)
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
    extra = {
        "entry_price": st.entry_price,
        "exit_price": exit_price,
        "gross_pnl": round(gross, 8),
        "net_pnl": round(net, 8),
    }
    mode = state["mode"]
    await log_trade(chat_id, mode, symbol, "sell" if side=="long" else "buy", exit_price, qty, note="close", extra=extra)
    emo = "📈" if net >= 0 else "📉"
    await tg_send_text(chat_id,
        f"{emo} Stängd {side.upper()} {symbol}\n"
        f"Entry: {st.entry_price:.6f}  Exit: {exit_price:.6f}\n"
        f"Qty: {qty:g}\nGross: {gross:.6f}  Avg: {fees:.6f}\n"
        f"Netto: {net:.6f}\nPnL idag: {state['pnl_today']:.6f}")
    # Nollställ position
    st.in_position = False
    st.side = ""
    st.entry_price = 0.0
    st.entry_qty = 0.0
    st.stop_price = 0.0
    st.tp_price = 0.0

# ---------- Kärnlogik per symbol ----------
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, timeframe, limit=60)
    if df.empty or len(df) < 3:  # behöver minst 3 för färgbyte + bekräftelse
        return

    # Senaste 2–3 candles
    prev = df.iloc[-2]
    last = df.iloc[-1]

    prev_col = candle_color(prev["open"], prev["close"])
    last_col = candle_color(last["open"], last["close"])

    # 1) Starta NY ORB direkt vid första färgbyte-candle
    #    Dvs när prev_col != last_col (ignorera doji som bytar-candle)
    if last_col in ("green","red") and prev_col != last_col:
        reset_orb_on_change(st, last["high"], last["low"], last_col)

    # 2) Entry-regler (kräver att vi har en giltig ORB-box)
    if not st.in_position and st.orb_high > 0 and st.orb_low > 0:
        # Enbart CLOSE-breakouts
        if last["close"] > st.orb_high:
            # LONG-entry
            qty = calc_qty(MOCK_USDT, float(last["close"]))
            st.in_position = True
            st.side = "long"
            st.entry_price = float(last["close"])
            st.entry_qty = qty
            st.tp_price = st.entry_price * (1.0 + TP_PCT)
            st.stop_price = st.entry_price * (1.0 - SL_PCT)
            await log_trade(chat_id, state["mode"], symbol, "buy", st.entry_price, qty, note="entry_long")
        elif last["close"] < st.orb_low:
            # SHORT-entry
            qty = calc_qty(MOCK_USDT, float(last["close"]))
            st.in_position = True
            st.side = "short"
            st.entry_price = float(last["close"])
            st.entry_qty = qty
            st.tp_price = st.entry_price * (1.0 - TP_PCT)
            st.stop_price = st.entry_price * (1.0 + SL_PCT)
            await log_trade(chat_id, state["mode"], symbol, "sell", st.entry_price, qty, note="entry_short")

    # 3) Hantera öppen position: BE/Trail/TP/SL
    if st.in_position:
        price = float(last["close"])
        # Break-even trigger
        if st.side == "long":
            # Uppmätt avkastning i %
            up = (price / st.entry_price) - 1.0
            if up >= BE_TRIGGER:
                be_floor = st.entry_price * (1 + 2 * FEE_RATE)  # skydda mot bägge avgifter
                st.stop_price = max(st.stop_price, be_floor)
                # traila upp med steg
                target_trail = max(st.stop_price, price * (1.0 - TRAIL_STEP))
                st.stop_price = max(st.stop_price, target_trail)
            # Exitvillkor
            if price >= st.tp_price:
                await close_and_log(chat_id, st, symbol, price); return
            if price <= st.stop_price:
                await close_and_log(chat_id, st, symbol, price); return
        else:  # short
            down = 1.0 - (price / st.entry_price)
            if down >= BE_TRIGGER:
                be_cap = st.entry_price * (1 - 2 * FEE_RATE)  # skydda short mot avgifter
                st.stop_price = min(st.stop_price, be_cap) if st.stop_price else be_cap
                # trail ner
                target_trail = min(st.stop_price, price * (1.0 + TRAIL_STEP)) if st.stop_price else price * (1.0 + TRAIL_STEP)
                st.stop_price = min(st.stop_price, target_trail) if st.stop_price else target_trail
            # Exitvillkor
            if price <= st.tp_price:
                await close_and_log(chat_id, st, symbol, price); return
            if price >= st.stop_price and st.stop_price > 0:
                await close_and_log(chat_id, st, symbol, price); return

    # 4) Uppdatera senaste färg (för init)
    if st.last_color == "" and last_col in ("green","red"):
        st.last_color = last_col

# ---------- Motor ----------
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(10)  # tätt intervall för fler setuper
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

# ---------- Keepalive ----------
async def keepalive_loop(url: str, chat_id: int|str):
    while state.get("keepalive"):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(url)
        except Exception:
            pass
        await asyncio.sleep(12 * 60)

# ---------- Backtest (dummy, kvar för kompat) ----------
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

# ---------- Telegram-kommandon ----------
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
            "/timeframe 1min|3min|5min – byt tidsram\n"
            "/backtest [SYMBOL] [PERIOD] [FEE]\n"
            "/export_csv – skicka loggar\n"
            "/pnl – visa dagens PnL\n"
            "/reset_pnl – nollställ dagens PnL\n"
            "/keepalive_on – håll Render vaken\n"
            "/keepalive_off – stäng keepalive")
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}   AI: {state['ai_mode']}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}\n"
            f"Avgift (per sida): {FEE_RATE:.4f}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"TP: {TP_PCT*100:.3f}%  SL: {SL_PCT*100:.3f}%  BE: {BE_TRIGGER*100:.3f}% Trail step: {TRAIL_STEP*100:.3f}%")
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

    if text.startswith("/export_csv"):
        sent = False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p)
                sent = True
        if not sent:
            await tg_send_text(chat_id, "Inga loggar hittades ännu.")
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
        if task and not task.done(): task.cancel()
        await tg_send_text(chat_id, "Keepalive AV.")
        return {"ok": True}

    return {"ok": True}

# ---------- FastAPI ----------
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {
        "message": "Mp ORBbot is live!",
        "webhook_path": WEBHOOK_PATH,
        "params": {
            "TP_PCT": TP_PCT, "SL_PCT": SL_PCT,
            "BE_TRIGGER": BE_TRIGGER, "TRAIL_STEP": TRAIL_STEP,
            "FEE_RATE": FEE_RATE
        }
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
