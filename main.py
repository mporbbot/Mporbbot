# main.py — Mp ORBbot (kompakt/robust, single-file)
# FastAPI webhook + ORB (1m/3m) med:
# - Entry: stängning över ORB-high (eller under ORB-low) med buffert
# - SL/TP i procent, Break-even och Trail efter BE
# - Mock/live (live stub), separata CSV-loggar + PnL
# - Kommandon: /status, /set_ai, /start_mock, /start_live, /engine_start, /engine_stop,
#               /symbols, /timeframe, /export_csv, /pnl, /reset_pnl, /risk,
#               /keepalive_on, /keepalive_off
# Kräver env: TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID (valfritt), RENDER_EXTERNAL_URL (Render sätter)

from __future__ import annotations
import os, csv, asyncio, math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# -------------------- Miljö & standardvärden --------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()
FEE_RATE           = float(os.getenv("FEE_RATE", "0.001"))        # 0.1% per sida
DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]

# ORB/indika­torer
ORB_BUFFER    = float(os.getenv("ORB_BUFFER", "0.0005"))  # 0.05% buffert
MIN_ORB_RANGE = float(os.getenv("MIN_ORB_RANGE", "0.001"))# minst 0.10% range
USE_EMA_TREND = os.getenv("USE_EMA_TREND", "1") == "1"    # EMA20 > EMA50 krävs?

# Risk/exit standard
TP_PCT     = float(os.getenv("TP_PCT", "0.0012"))     # +0.12%
SL_PCT     = float(os.getenv("SL_PCT", "0.0005"))     # -0.05%
BE_TRIGGER = float(os.getenv("BE_TRIGGER", "0.0010")) # +0.10% -> flytta SL till BE
TRAIL_STEP = float(os.getenv("TRAIL_STEP", "0.0002")) # trailsteg 0.02% efter BE

# Positionstorlek (mock)
MOCK_USDT_PER_TRADE = float(os.getenv("MOCK_USDT", "30"))

MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

KU_PUBLIC = "https://api.kucoin.com"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -------------------- Globalt state --------------------
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",           # "mock" | "live"
    "ai_mode": "neutral",     # "neutral" | "aggressiv" | "försiktig"
    "pending_confirm": None,  # {"type": "mock"|"live"}
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": "1min",
    "engine_task": None,
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
    "risk": {                 # inga global — allt här
        "TP_PCT": TP_PCT,
        "SL_PCT": SL_PCT,
        "BE_TRIGGER": BE_TRIGGER,
        "TRAIL_STEP": TRAIL_STEP,
    },
    "per_symbol": {},         # symbol -> ORBState
}

# -------------------- Telegram utils --------------------
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

# -------------------- Hjälpfunktioner --------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    s = pd.Series(series, dtype="float64")
    return s.ewm(span=period, adjust=False).mean()

def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"):
        return f"{usdt_sym[:-4]}-USDT"
    return usdt_sym.replace("/", "-").upper()

async def fetch_candles(symbol: str, timeframe: str="1min", limit: int=120) -> pd.DataFrame:
    """Hämtar OHLC från KuCoin publika API. Returnerar kronologisk DataFrame."""
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
        rows.append(row[:7])
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    for c in ["open","close","high","low","volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df.tail(limit)

# -------------------- Loggning & PnL --------------------
LOG_FIELDS = [
    "timestamp","mode","symbol","side","price","qty",
    "fee_rate","fee_cost","gross_pnl","net_pnl","entry_price","exit_price","note"
]

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

# -------------------- ORB state --------------------
@dataclass
class ORBState:
    direction: str = "flat"   # "up"|"down"|"flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_pos: bool = False
    side: str = ""            # "long"|"short"
    entry_price: float = 0.0
    entry_qty: float = 0.0
    be_level: Optional[float] = None  # break-even pris
    trail_level: Optional[float] = None
    last_close: float = 0.0
    last_low: float = 0.0
    last_high: float = 0.0

def detect_dir(prev_close: float, close: float) -> str:
    if close > prev_close: return "up"
    if close < prev_close: return "down"
    return "flat"

def reset_orb(st: ORBState, hi: float, lo: float):
    st.orb_high, st.orb_low = float(hi), float(lo)
    st.in_pos = False
    st.side = ""
    st.entry_price = 0.0
    st.entry_qty = 0.0
    st.be_level = None
    st.trail_level = None

# -------------------- Orderhelpers --------------------
def qty_for_usdt(usdt: float, price: float) -> float:
    if price <= 0: return 0.0
    q = usdt / price
    # runda till 6 decimaler (safe för de flesta spot-par)
    return round(q, 6)

async def do_trade(mode: str, chat_id: int|str, symbol: str, side: str,
                   price: float, qty: float, extra: Optional[dict]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": mode, "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": mode
    }
    if extra: row.update(extra)
    log_path = MOCK_LOG if mode == "mock" else REAL_LOG
    write_log_row(log_path, row)
    prefix = "✅ Mock" if mode == "mock" else "🟢 LIVE"
    await tg_send_text(chat_id, f"{prefix}: {side.upper()} {symbol} @ {price:.6f} x {qty:g} (avg ~ {fee_cost:.6f})")

async def notify_close(chat_id: int|str, symbol: str, entry: float, exitp: float, qty: float):
    gross, fees, net = compute_pnl(entry, exitp, qty, FEE_RATE)
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
    emo = "📈" if net >= 0 else "📉"
    txt = (
        f"{emo} Stängd LONG {symbol}\n"
        f"Entry: {entry:.6f}  Exit: {exitp:.6f}\n"
        f"Qty: {qty:g}\n"
        f"Gross: {gross:.6f}  Avg: {fees:.6f}\n"
        f"Netto: {net:.6f}\n"
        f"PnL idag: {state['pnl_today']:.6f}"
    )
    await tg_send_text(chat_id, txt)

# -------------------- ORB-logik (entry/exit) --------------------
def ai_allows(ai_mode: str, signal_strength: float) -> bool:
    m = (ai_mode or "neutral").lower()
    if m == "aggressiv": return signal_strength >= 0.2
    if m in ("försiktig", "forsiktig"): return signal_strength >= 0.7
    return signal_strength >= 0.5

async def process_symbol(chat_id: int|str, symbol: str, tf: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, tf, limit=120)
    if df.empty or len(df) < 55:
        return

    # Senaste två stängningar
    prev, last = df.iloc[-2], df.iloc[-1]
    st.last_close = float(last["close"])
    st.last_low   = float(last["low"])
    st.last_high  = float(last["high"])

    # Trendfilter (valfritt)
    if USE_EMA_TREND:
        ema20 = float(ema(df["close"], 20).iloc[-1])
        ema50 = float(ema(df["close"], 50).iloc[-1])
        trend_ok = ema20 > ema50
    else:
        trend_ok = True

    # Ny ORB när riktning byter mellan föregående & senaste stängning
    last_dir = detect_dir(prev["close"], last["close"])
    if last_dir != st.direction and last_dir in ("up", "down"):
        reset_orb(st, last["high"], last["low"])
        st.direction = last_dir

    # Säkerställ rimlig ORB-range
    orb_range_ok = True
    if st.orb_high > 0 and st.orb_low > 0 and st.orb_low > 0:
        rng = (st.orb_high / st.orb_low) - 1.0
        orb_range_ok = rng >= MIN_ORB_RANGE

    # --- ENTRY (LONG): candle måste STÄNGA över orb_high*(1+buffer) ---
    entry_ok = (
        not st.in_pos
        and st.direction == "up"
        and st.orb_high > 0
        and last["close"] > st.orb_high * (1.0 + ORB_BUFFER)
        and trend_ok and orb_range_ok
    )

    if entry_ok and ai_allows(state["ai_mode"], 0.6):
        # beräkna qty från mock-usdt
        qty = qty_for_usdt(MOCK_USDT_PER_TRADE, float(last["close"]))
        if qty > 0:
            st.in_pos = True
            st.side = "long"
            st.entry_price = float(last["close"])  # entry på close
            st.entry_qty = qty
            st.be_level = st.entry_price * (1.0 + state["risk"]["BE_TRIGGER"])
            st.trail_level = None
            await do_trade(state["mode"], chat_id, symbol, "buy", st.entry_price, qty)

    # --- EXIT-LOGIK ---
    if st.in_pos and st.side == "long":
        cur = st.last_close
        # TP
        tp_price = st.entry_price * (1.0 + state["risk"]["TP_PCT"])
        if cur >= tp_price:
            # ta vinst
            exitp = cur
            qty   = st.entry_qty
            gross, fees, net = compute_pnl(st.entry_price, exitp, qty, FEE_RATE)
            extra = {"entry_price": st.entry_price, "exit_price": exitp,
                     "gross_pnl": round(gross, 8), "net_pnl": round(net, 8)}
            await do_trade(state["mode"], chat_id, symbol, "sell", exitp, qty, extra=extra)
            await notify_close(chat_id, symbol, st.entry_price, exitp, qty)
            st.in_pos = False; st.side = ""; st.entry_price = 0.0; st.entry_qty = 0.0
            st.be_level = None; st.trail_level = None
            return

        # BE/TRAIL aktivering
        if st.be_level and cur >= st.be_level:
            # flytta SL till BE och aktivera trail
            if st.trail_level is None:
                st.trail_level = st.entry_price  # börja vid break-even
            # uppdatera trail om priset stiger med step
            needed = st.trail_level * (1.0 + state["risk"]["TRAIL_STEP"])
            if cur >= needed:
                # flytta trail upp men ALDRIG under entry (BE)
                st.trail_level = max(st.trail_level * (1.0 + state["risk"]["TRAIL_STEP"]),
                                     st.entry_price)

        # SL — fast stop om pris faller SL_PCT
        sl_price = st.entry_price * (1.0 - state["risk"]["SL_PCT"])
        # Trail-stop (när aktiverad): stäng om close < trail_level
        hit_trail = (st.trail_level is not None) and (cur < st.trail_level)

        if cur <= sl_price or hit_trail:
            exitp = cur
            qty   = st.entry_qty
            gross, fees, net = compute_pnl(st.entry_price, exitp, qty, FEE_RATE)
            extra = {"entry_price": st.entry_price, "exit_price": exitp,
                     "gross_pnl": round(gross, 8), "net_pnl": round(net, 8),
                     "note": "SL" if cur <= sl_price else "TRAIL"}
            await do_trade(state["mode"], chat_id, symbol, "sell", exitp, qty, extra=extra)
            await notify_close(chat_id, symbol, st.entry_price, exitp, qty)
            st.in_pos = False; st.side = ""; st.entry_price = 0.0; st.entry_qty = 0.0
            st.be_level = None; st.trail_level = None
            return

# -------------------- Motor --------------------
async def run_engine(chat_id: int|str):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(12)  # loopintervall (sek)
        except asyncio.CancelledError:
            break
        except Exception as e:
            await tg_send_text(chat_id, f"Motorfel: {e}")
            await asyncio.sleep(5)

async def start_engine(chat_id: int|str):
    if state["engine_task"] and not state["engine_task"].done():
        await tg_send_text(chat_id, "Motorn kör redan.")
        return
    state["active"] = True
    loop = asyncio.get_event_loop()
    state["engine_task"] = loop.create_task(run_engine(chat_id))
    await tg_send_text(chat_id, f"🔄 Motor startad {utcnow_iso()} (tf={state['timeframe']}, symbols={', '.join(state['symbols'])}).")

async def stop_engine(chat_id: int|str):
    state["active"] = False
    task = state.get("engine_task")
    if task and not task.done():
        task.cancel()
    await tg_send_text(chat_id, "⏹️ Motor stoppad.")

# -------------------- Keepalive --------------------
async def keepalive_loop(url: str, chat_id: int|str):
    while state.get("keepalive"):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(url)
        except Exception:
            pass
        await asyncio.sleep(12 * 60)

# -------------------- Backtest (enkel dummy) --------------------
async def run_backtest(chat_id: int, symbol: str="BTCUSDT", period: str="3d", fee: float|None=None, out_csv: str="backtest_result.csv"):
    import random
    random.seed(42)
    fee_rate = FEE_RATE if fee is None else float(fee)
    rows = []
    for _ in range(40):
        gross = random.uniform(-0.004, 0.006)
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

# -------------------- Telegram-kommandon --------------------
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

    # Pending bekräftelser
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

    # Hjälp
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
            "/risk tp=0.0012 sl=0.0005 be=0.0010 trail=0.0002 – uppdatera risk\n"
            "/export_csv – skicka loggar\n"
            "/pnl – visa dagens PnL\n"
            "/reset_pnl – nollställ PnL\n"
            "/keepalive_on | /keepalive_off")
        return {"ok": True}

    # Status
    if text.startswith("/status"):
        r = state["risk"]
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}\nAI: {state['ai_mode']}\n"
            f"Avgift (per sida): {FEE_RATE:.4f}\n"
            f"Symbols: {', '.join(state['symbols'])}\nTF: {state['timeframe']}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Min range: {MIN_ORB_RANGE:.4%}\n"
            f"Risk → TP:{r['TP_PCT']:.4%}  SL:{r['SL_PCT']:.4%}  "
            f"BE:{r['BE_TRIGGER']:.4%}  Trail step:{r['TRAIL_STEP']:.4%}\n"
            f"Mock USDT: {MOCK_USDT_PER_TRADE:.2f}")
        return {"ok": True}

    # AI-läge
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

    # Start/stop
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

    # Byt symboler
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

    # Byt timeframe
    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts) > 1 else "1min"
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram satt till: {tf}")
        return {"ok": True}

    # Risk-parametrar
    if text.startswith("/risk"):
        # ex: /risk tp=0.0012 sl=0.0005 be=0.0010 trail=0.0002
        tokens = text.split()
        kv = {}
        for t in tokens[1:]:
            if "=" in t:
                k,v = t.split("=",1)
                kv[k.strip().lower()] = float(v.strip())
        r = state["risk"]
        r["TP_PCT"] = kv.get("tp", r["TP_PCT"])
        r["SL_PCT"] = kv.get("sl", r["SL_PCT"])
        r["BE_TRIGGER"] = kv.get("be", r["BE_TRIGGER"])
        r["TRAIL_STEP"] = kv.get("trail", r["TRAIL_STEP"])
        await tg_send_text(chat_id, f"Ny risk: TP:{r['TP_PCT']:.4%} SL:{r['SL_PCT']:.4%} BE:{r['BE_TRIGGER']:.4%} Trail:{r['TRAIL_STEP']:.4%}")
        return {"ok": True}

    # Export & PnL
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

    # Keepalive
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

    return {"ok": True}

# -------------------- FastAPI --------------------
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    r = state["risk"]
    return {
        "message": "Mp ORBbot is live!",
        "webhook_path": WEBHOOK_PATH,
        "params": {
            "ORB_BUFFER": ORB_BUFFER, "MIN_ORB_RANGE": MIN_ORB_RANGE,
            "USE_EMA_TREND": USE_EMA_TREND,
            "TP_PCT": r["TP_PCT"], "SL_PCT": r["SL_PCT"],
            "BE_TRIGGER": r["BE_TRIGGER"], "TRAIL_STEP": r["TRAIL_STEP"],
        },
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
