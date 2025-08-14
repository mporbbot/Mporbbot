# main.py — Mp ORBbot (full single-file version)
# FastAPI webhook + Telegram + ORB-strategi.
# - ORB skapas direkt vid start och varje riktningsändring
# - Entry: först när en CANDLE STÄNGER över ORB-high*(1+buffer)
# - Fast SL = -0.05%, Fast TP = +0.12%
# - Breakeven/trail när priset passerar +0.10% (skydd mot avgifter)
# - AI-lägen (neutral/aggressiv/försiktig) filtrerar signalstyrka
# - Mock & Live (live är stub – loggar till real_csv)
# - Robust KuCoin-fetch (retries, keep-alive klient)
# - CSV-loggar + PnL och /export_csv, /pnl, /reset_pnl
# - Keepalive för Render Free

from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import httpx
from fastapi import FastAPI, Request

# ========================= Konfig via ENV =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()  # tomt = alla
DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]
DEFAULT_TIMEFRAME  = os.getenv("DEFAULT_TIMEFRAME", "3min")

# Handelsparametrar
FEE_RATE        = float(os.getenv("FEE_RATE", "0.001"))     # 0.1% per sida
MOCK_USDT       = float(os.getenv("MOCK_USDT", "30"))       # belopp per mock-köp
ORB_BUFFER      = float(os.getenv("ORB_BUFFER", "0.0005"))  # +0.05% över ORB-high
MIN_ORB_RANGE   = float(os.getenv("MIN_ORB_RANGE","0.0015"))# 0.15% min box
USE_EMA_TREND   = os.getenv("USE_EMA_TREND","1") == "1"     # EMA20>EMA50 filter

# Fasta risk-parametrar enligt dina önskemål
SL_PCT          = float(os.getenv("SL_PCT", "0.0005"))      # 0.05% stoploss
TP_PCT          = float(os.getenv("TP_PCT", "0.0012"))      # 0.12% take profit
BE_TRIGGER_PCT  = float(os.getenv("BE_TRIGGER_PCT","0.001"))# +0.10% aktiverar BE+trail
TRAIL_STEP_PCT  = float(os.getenv("TRAIL_STEP_PCT","0.0003"))# 0.03% steg vid fortsatt uppgång

# Motor
LOOP_SECONDS    = float(os.getenv("LOOP_SECONDS", "6"))     # loopintervall
KU_PUBLIC       = "https://api.kucoin.com"

# Loggar
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"
LOG_FIELDS = [
    "timestamp","mode","symbol","side","price","qty",
    "fee_rate","fee_cost","gross_pnl","net_pnl",
    "entry_price","exit_price","note"
]

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ========================= Globalt state =========================
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",                 # mock | live
    "ai_mode": "neutral",           # neutral vid uppstart
    "pending_confirm": None,        # {"type":"mock"|"live"}
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": DEFAULT_TIMEFRAME,
    "engine_task": None,
    "per_symbol": {},               # symbol -> ORBState
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

# ========================= AI-filter =========================
def ai_allows(ai_mode: str, signal_strength: float) -> bool:
    m = (ai_mode or "neutral").lower()
    if m == "aggressiv": return signal_strength >= 0.2
    if m in ("försiktig","forsiktig"): return signal_strength >= 0.7
    return signal_strength >= 0.5

# ========================= CSV & PnL =========================
def write_log_row(path: str, row: Dict[str, Any]):
    newfile = not os.path.exists(path)
    for k in LOG_FIELDS: row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})

def compute_pnl(entry: float, exit: float, qty: float, fee_rate: float):
    gross = (exit - entry) * qty
    fees  = (entry + exit) * qty * fee_rate
    net   = gross - fees
    return gross, fees, net

async def mock_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "mock", "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost,8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": "mock",
    }
    if extra: row.update(extra)
    write_log_row(MOCK_LOG, row)

async def live_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "live", "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost,8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": "live (stub)",
    }
    if extra: row.update(extra)
    write_log_row(REAL_LOG, row)

async def notify_closed_trade(chat_id: int, symbol: str, entry: float, exit: float, qty: float):
    gross, fees, net = compute_pnl(entry, exit, qty, FEE_RATE)
    state["pnl_today"] = float(state.get("pnl_today",0.0)) + net
    emo = "📈" if net >= 0 else "📉"
    await tg_send_text(
        chat_id,
        f"{emo} Stängd LONG {symbol}\n"
        f"Entry: {entry:.6f}  Exit: {exit:.6f}\n"
        f"Qty: {qty:g}\nGross: {gross:.6f}  Avg: {fees:.6f}\n"
        f"Netto: {net:.6f}\nPnL idag: {state['pnl_today']:.6f}"
    )

# ========================= KuCoin fetch (robust) =========================
def to_ksym(usdt_sym: str) -> str:
    s = usdt_sym.upper()
    if s.endswith("USDT"): return f"{s[:-4]}-USDT"
    return s.replace("/", "-").upper()

HTTP_CLIENT: httpx.AsyncClient | None = None
async def get_http_client() -> httpx.AsyncClient:
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        HTTP_CLIENT = httpx.AsyncClient(
            timeout=10,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return HTTP_CLIENT

async def candles(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame:
    url = f"{KU_PUBLIC}/api/v1/market/candles"
    params = {"symbol": to_ksym(symbol), "type": timeframe}
    client = await get_http_client()

    for attempt in range(4):  # retries
        try:
            r = await client.get(url, params=params)
            if r.status_code in (429,500,502,503,504):
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            data = r.json().get("data", [])
            cols = ["ts","open","close","high","low","volume","turnover"]
            rows = [(row[:7] if len(row)>=7 else row+[None]) for row in data]
            df = pd.DataFrame(rows, columns=cols)
            if df.empty: return df
            df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
            for c in ["open","close","high","low","volume"]:
                df[c] = df[c].astype(float)
            df = df.sort_values("ts").reset_index(drop=True)
            return df.tail(limit)
        except Exception:
            await asyncio.sleep(0.4 * (2 ** attempt))

    return pd.DataFrame(columns=["ts","open","close","high","low","volume","turnover"])

def ema(series: pd.Series, period: int) -> pd.Series:
    s = pd.Series(series, dtype="float64")
    return s.ewm(span=period, adjust=False).mean()

# ========================= ORB-state =========================
@dataclass
class ORBState:
    direction: str = "flat"      # up|down|flat (vi använder bara long)
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    entry_price: float = 0.0
    qty: float = 0.0
    sl: float = 0.0              # fast stoploss nivå
    tp: float = 0.0              # fast take profit nivå
    trail_on: bool = False
    trail_level: float = 0.0
    last_close: float = 0.0

def detect_dir(prev_close: float, close: float) -> str:
    if close > prev_close: return "up"
    if close < prev_close: return "down"
    return "flat"

def reset_orb(st: ORBState, hi: float, lo: float):
    st.orb_high, st.orb_low = float(hi), float(lo)
    st.in_position = False
    st.entry_price = 0.0
    st.qty = 0.0
    st.sl = 0.0
    st.tp = 0.0
    st.trail_on = False
    st.trail_level = 0.0

# ========================= Motor per symbol =========================
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await candles(symbol, timeframe, limit=90)
    if df.empty or len(df) < 55:
        return

    # indikatorer
    close = df["close"].astype(float)
    ema20 = float(ema(close, 20).iloc[-1])
    ema50 = float(ema(close, 50).iloc[-1])

    prev, last = df.iloc[-2], df.iloc[-1]
    st.last_close = float(last["close"])

    last_dir = detect_dir(prev["close"], last["close"])

    # Initiera ORB vid start om saknas
    if st.orb_high == 0.0 or st.orb_low == 0.0:
        reset_orb(st, last["high"], last["low"])
        st.direction = last_dir

    # Ny ORB vid riktningsändring
    if last_dir != st.direction and last_dir in ("up","down"):
        reset_orb(st, last["high"], last["low"])
        st.direction = last_dir

    # Minsta ORB-range
    orb_ok = True
    if st.orb_high > 0 and st.orb_low > 0 and st.orb_low > 0:
        rng = (st.orb_high / st.orb_low) - 1.0
        orb_ok = rng >= MIN_ORB_RANGE

    # ENTRY: candle måste STÄNGA över ORB-high*(1+buffer) + trendfilter
    entry_ok = (
        (not st.in_position)
        and st.direction == "up"
        and st.orb_high > 0
        and float(last["close"]) > st.orb_high * (1.0 + ORB_BUFFER)
        and (not USE_EMA_TREND or ema20 > ema50)
        and orb_ok
    )

    if entry_ok and ai_allows(state["ai_mode"], 0.6):
        # qty enligt MOCK_USDT
        px = st.last_close
        qty = max(MOCK_USDT / px, 0.0001)

        st.in_position = True
        st.entry_price = px
        st.qty = qty
        st.sl = st.entry_price * (1.0 - SL_PCT)
        st.tp = st.entry_price * (1.0 + TP_PCT)
        st.trail_on = False
        st.trail_level = 0.0

        if state["mode"] == "mock":
            await mock_trade(chat_id, symbol, "buy", px, qty)
        else:
            await live_trade(chat_id, symbol, "buy", px, qty)

    # Hantera exit/TP/SL/trail
    if st.in_position:
        px = st.last_close
        # aktivera breakeven/trail när +0.10%
        if not st.trail_on and px >= st.entry_price * (1.0 + BE_TRIGGER_PCT):
            st.trail_on = True
            # BE på entry + avgift per sida så du inte går back
            st.sl = st.entry_price * (1.0 + FEE_RATE)

        # uppdatera trailing om aktiv och priset gör nya steg
        if st.trail_on:
            desired = px * (1.0 - TRAIL_STEP_PCT)
            st.sl = max(st.sl, desired)

        # exits:
        hit_tp = px >= st.tp
        hit_sl = px <= st.sl
        # hård exit om ljusets low bryter ORB-low (failsafe)
        low_break = float(last["low"]) < st.orb_low

        if hit_tp or hit_sl or low_break:
            exit_price = px
            qty = st.qty
            gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
            extra = {
                "entry_price": st.entry_price,
                "exit_price": exit_price,
                "gross_pnl": round(gross,8),
                "net_pnl": round(net,8),
            }
            if state["mode"] == "mock":
                await mock_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            else:
                await live_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            await notify_closed_trade(chat_id, symbol, st.entry_price, exit_price, qty)

            # nollställ position
            st.in_position = False
            st.entry_price = 0.0
            st.qty = 0.0
            st.sl = 0.0
            st.tp = 0.0
            st.trail_on = False
            st.trail_level = 0.0

# ========================= Motor-loop =========================
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(LOOP_SECONDS)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(2)

async def start_engine(chat_id: int):
    if state["engine_task"] and not state["engine_task"].done():
        await tg_send_text(chat_id, "Motorn kör redan.")
        return
    state["active"] = True
    state["engine_task"] = asyncio.get_event_loop().create_task(run_engine(chat_id))
    await tg_send_text(chat_id, f"🔄 Motor startad {utcnow_iso()} (tf={state['timeframe']}, symbols={', '.join(state['symbols'])}).")

async def stop_engine(chat_id: int):
    state["active"] = False
    task = state.get("engine_task")
    if task and not task.done():
        task.cancel()
    await tg_send_text(chat_id, "⏹️ Motor stoppad.")

# ========================= Keepalive =========================
async def keepalive_loop(url: str, chat_id: int|str):
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
    rows=[]
    for _ in range(40):
        gross = random.uniform(-0.004, 0.006)
        net = gross - 2*fee_rate
        rows.append({
            "timestamp": utcnow_iso(),
            "symbol": symbol.upper(),
            "gross_return": round(gross,6),
            "fee_rate_each_side": fee_rate,
            "net_return": round(net,6)
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    await tg_send_text(chat_id, f"Backtest klart: {symbol} ({period}) avgift {fee_rate:.4f}")
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

    # bekräftelser
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

    # hjälp
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
            "/keepalive_on / keepalive_off")
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}\nAI: {state['ai_mode']}\n"
            f"Avgift (per sida): {FEE_RATE:.4f}\n"
            f"Symbols: {', '.join(state['symbols'])}\nTF: {state['timeframe']}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Min range: {MIN_ORB_RANGE:.4%}\n"
            f"Trendfilter: {'på' if USE_EMA_TREND else 'av'}\n"
            f"SL: {SL_PCT:.4%}  TP: {TP_PCT:.4%}  BE-tröskel: {BE_TRIGGER_PCT:.4%}\n"
            f"Mock USDT: {MOCK_USDT:.2f}"
        )
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
        await tg_send_text(chat_id, "Vill du starta MOCK? Svara JA.")
        return {"ok": True}

    if text.startswith("/start_live"):
        state["pending_confirm"] = {"type": "live"}
        await tg_send_text(chat_id, "⚠️ Vill du starta LIVE? Svara JA.")
        return {"ok": True}

    if text.startswith("/engine_start"):
        await start_engine(chat_id); return {"ok": True}

    if text.startswith("/engine_stop") or text.startswith("/stop"):
        await stop_engine(chat_id); return {"ok": True}

    if text.startswith("/symbols"):
        parts = text.split(maxsplit=1)
        if len(parts)<2:
            await tg_send_text(chat_id, f"Aktuell lista: {', '.join(state['symbols'])}")
            return {"ok": True}
        raw = parts[1].replace(" ", "")
        syms = [s for s in raw.split(",") if s]
        state["symbols"] = [s.upper() for s in syms]
        await tg_send_text(chat_id, f"Symbols uppdaterade: {', '.join(state['symbols'])}")
        return {"ok": True}

    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts)>1 else DEFAULT_TIMEFRAME
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram satt till: {tf}")
        return {"ok": True}

    if text.startswith("/backtest"):
        parts = text.split()
        symbol = parts[1] if len(parts)>1 else "BTCUSDT"
        period = parts[2] if len(parts)>2 else "3d"
        fee = float(parts[3]) if len(parts)>3 else None
        path = await run_backtest(chat_id, symbol.upper(), period, fee)
        await tg_send_document(chat_id, path, caption=f"Backtest {symbol} ({period})")
        return {"ok": True}

    if text.startswith("/export_csv"):
        sent=False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p)
                sent=True
        if not sent: await tg_send_text(chat_id, "Inga loggar ännu.")
        return {"ok": True}

    if text.startswith("/pnl"):
        await tg_send_text(chat_id, f"Ackumulerad PnL idag: {state.get('pnl_today',0.0):.6f}")
        return {"ok": True}

    if text.startswith("/reset_pnl"):
        state["pnl_today"] = 0.0
        await tg_send_text(chat_id, "PnL nollställd.")
        return {"ok": True}

    if text.startswith("/keepalive_on"):
        public_url = f"https://{os.getenv('RENDER_EXTERNAL_URL','')}/"
        if not public_url or public_url == "https:///":
            public_url = "https://mporbbot.onrender.com/"
        if state.get("keepalive"):
            await tg_send_text(chat_id, "Keepalive redan PÅ.")
            return {"ok": True}
        state["keepalive"]=True
        state["keepalive_task"]=asyncio.get_event_loop().create_task(keepalive_loop(public_url, chat_id))
        await tg_send_text(chat_id, f"Keepalive PÅ – pingar {public_url} var 12:e minut.")
        return {"ok": True}

    if text.startswith("/keepalive_off"):
        state["keepalive"]=False
        t=state.get("keepalive_task")
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
            "SL_PCT": SL_PCT, "TP_PCT": TP_PCT,
            "BE_TRIGGER_PCT": BE_TRIGGER_PCT, "TRAIL_STEP_PCT": TRAIL_STEP_PCT,
            "ORB_BUFFER": ORB_BUFFER, "MIN_ORB_RANGE": MIN_ORB_RANGE,
            "USE_EMA_TREND": USE_EMA_TREND, "LOOP_SECONDS": LOOP_SECONDS
        }
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
