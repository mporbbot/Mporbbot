# main.py — Mp ORBbot (en-fil)
# Entry: candle CLOSE > ORB-high*(1+buffer)
# Exit: på CLOSE (wick ignoreras om inte EXIT_ON_WICK=1)
# Risk: TP/SL/BE/Trail via env eller /risk-kommando
# Flera telegramkommandon inkl. /exitwicks och /failsafe

from __future__ import annotations
import os, csv, asyncio, math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ================== Miljö & standardvärden ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()

# Avgift per sida (0.001 = 0.10%)
FEE_RATE           = float(os.getenv("FEE_RATE", "0.001"))

# ORB & filter
ORB_BUFFER     = float(os.getenv("ORB_BUFFER", "0.0002"))   # 0.02% över ORB-high
MIN_ORB_RANGE  = float(os.getenv("MIN_ORB_RANGE", "0.0008"))# min 0.08% range
USE_EMA_TREND  = os.getenv("USE_EMA_TREND", "0") == "1"     # av default

# Risk (procent som decimaler)
TP_PCT          = float(os.getenv("TP_PCT", "0.0040"))      # 0.40%
SL_PCT          = float(os.getenv("SL_PCT", "0.0015"))      # 0.15%
BE_TRIGGER_PCT  = float(os.getenv("BE_TRIGGER_PCT", "0.0020")) # 0.20%
TRAIL_STEP_PCT  = float(os.getenv("TRAIL_STEP_PCT", "0.0010")) # 0.10%

EXIT_ON_WICK    = os.getenv("EXIT_ON_WICK", "0") == "1"     # standard: av
FAILSAFE_ORB    = os.getenv("FAILSAFE_ORB", "0") == "1"     # standard: av

# Mock-belopp per trade (USDT)
MOCK_USDT_PER_TRADE = float(os.getenv("MOCK_USDT", "30.0"))

DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]

MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

KU_PUBLIC = "https://api.kucoin.com"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ================== Globalt state ==================
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",                 # "mock" | "live"
    "ai_mode": "neutral",
    "pending_confirm": None,        # {"type":"mock"|"live"}
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": "1min",            # 1min för fler trades
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

# ================== Hjälp-funktioner ==================
def write_log_row(path: str, row: Dict[str, Any]):
    fields = [
        "timestamp","mode","symbol","side","price","qty",
        "fee_rate","fee_cost","gross_pnl","net_pnl","entry_price","exit_price","note"
    ]
    newfile = not os.path.exists(path)
    for k in fields: row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})

def compute_pnl(entry_price: float, exit_price: float, qty: float, fee_rate: float):
    gross = (exit_price - entry_price) * qty
    fees  = (entry_price + exit_price) * qty * fee_rate
    net   = gross - fees
    return gross, fees, net

def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"):
        return f"{usdt_sym[:-4]}-USDT"
    return usdt_sym.replace("/", "-").upper()

async def fetch_candles(symbol: str, timeframe: str="1min", limit: int=90) -> pd.DataFrame:
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

# ================== ORB State ==================
@dataclass
class ORBState:
    direction: str = "flat"    # "up"|"down"|"flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    entry_price: float = 0.0
    qty: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    be_armed: bool = False
    last_closed_close: float = 0.0

def reset_orb(st: ORBState, hi: float, lo: float):
    st.orb_high, st.orb_low = float(hi), float(lo)
    st.in_position = False
    st.entry_price = 0.0
    st.qty = 0.0
    st.tp = 0.0
    st.sl = 0.0
    st.be_armed = False

def detect_dir(prev_close: float, close: float) -> str:
    if close > prev_close: return "up"
    if close < prev_close: return "down"
    return "flat"

# ================== Handel (mock/live) ==================
async def mock_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "mock", "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "note": "mock",
    }
    if extra: row.update(extra)
    write_log_row(MOCK_LOG, row)

async def live_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "live", "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "note": "live (stub)",
    }
    if extra: row.update(extra)
    write_log_row(REAL_LOG, row)

async def notify_closed(chat_id: int, symbol: str, entry: float, exitp: float, qty: float):
    gross, fees, net = compute_pnl(entry, exitp, qty, FEE_RATE)
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
    emo = "📈" if net >= 0 else "📉"
    await tg_send_text(chat_id,
        f"{emo} Stängd LONG {symbol}\n"
        f"Entry: {entry:.6f}  Exit: {exitp:.6f}\n"
        f"Qty: {qty:.5f}\n"
        f"Gross: {gross:.6f}  Avg: {fees:.6f}\n"
        f"Netto: {net:.6f}\n"
        f"PnL idag: {state['pnl_today']:.6f}"
    )

# ================== Motor: en symbol ==================
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, timeframe, limit=120)
    if df.empty or len(df) < 25:
        return

    # Vi jobbar med senast STÄNGDA candlen
    prev, last = df.iloc[-2], df.iloc[-1]
    st.last_closed_close = float(last["close"])
    last_dir = detect_dir(prev["close"], last["close"])

    # Ny ORB när riktning ändras
    if last_dir != st.direction and last_dir in ("up", "down"):
        reset_orb(st, last["high"], last["low"])
        st.direction = last_dir

    # Min ORB-range
    orb_ok = True
    if st.orb_high > 0 and st.orb_low > 0 and st.orb_low > 0:
        rng = (st.orb_high / st.orb_low) - 1.0
        orb_ok = rng >= MIN_ORB_RANGE

    # Trendfilter (valfritt)
    trend_ok = True
    if USE_EMA_TREND:
        e20 = float(ema(df["close"], 20).iloc[-1])
        e50 = float(ema(df["close"], 50).iloc[-1])
        trend_ok = e20 > e50

    # ENTRY: CLOSE över ORB-high*(1+buffer)
    if (not st.in_position) and st.direction == "up" and st.orb_high > 0 and orb_ok and trend_ok:
        trigger = st.orb_high * (1.0 + ORB_BUFFER)
        if float(last["close"]) > trigger and float(prev["close"]) <= trigger:
            # storlek (mock): USDT / pris
            px = float(last["close"])
            qty = max(MOCK_USDT_PER_TRADE / px, 0.00001)
            # initial tp/sl
            tp = px * (1.0 + TP_PCT)
            sl = px * (1.0 - SL_PCT)
            # arm BE?
            st.in_position = True
            st.entry_price = px
            st.qty = qty
            st.tp = tp
            st.sl = sl
            st.be_armed = False

            if state["mode"] == "mock":
                await mock_trade(chat_id, symbol, "buy", px, qty)
            else:
                await live_trade(chat_id, symbol, "buy", px, qty)
            await tg_send_text(chat_id, f"✅ ENTRY {symbol} long @ {px:.6f} (tp {tp:.6f} / sl {sl:.6f})")

    # Hantera aktiv position
    if st.in_position:
        close_px = float(last["close"])
        low_px   = float(last["low"])

        # Arma/move BE + trail
        up_pct = (close_px / st.entry_price) - 1.0
        if up_pct >= BE_TRIGGER_PCT:
            # flytta SL till BE + fees (tur & retur ~ 2*FEE_RATE)
            be_plus_fees = st.entry_price * (1.0 + (2*FEE_RATE))
            st.sl = max(st.sl, be_plus_fees)  # skydda mot minus
            st.be_armed = True
            # trailing: höj SL stegvis bakom pris
            target_trail = close_px * (1.0 - TRAIL_STEP_PCT)
            st.sl = max(st.sl, target_trail)

        # TP/SL/logik – jämför CLOSE (eller LOW om EXIT_ON_WICK)
        test_px_tp = close_px
        test_px_sl = low_px if EXIT_ON_WICK else close_px

        hit_tp = test_px_tp >= st.tp
        hit_sl = test_px_sl <= st.sl

        # Failsafe på ORB-low?
        low_fail = False
        if FAILSAFE_ORB and st.orb_low > 0:
            fail_test = low_px if EXIT_ON_WICK else close_px
            low_fail = fail_test < st.orb_low

        if hit_tp or hit_sl or low_fail:
            exit_price = close_px
            qty = st.qty
            gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
            extra = {
                "entry_price": st.entry_price,
                "exit_price": exit_price,
                "gross_pnl": round(gross, 8),
                "net_pnl": round(net, 8),
            }
            if state["mode"] == "mock":
                await mock_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            else:
                await live_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            await notify_closed(chat_id, symbol, st.entry_price, exit_price, qty)

            # Nollställ
            reset_orb(st, st.orb_high, st.orb_low)
            st.direction = last_dir  # behåll aktuell riktning

# ================== Motor-loop ==================
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(15)  # snabbare loop
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
    t = state.get("engine_task")
    if t and not t.done():
        t.cancel()
    await tg_send_text(chat_id, "⏹️ Motor stoppad.")

# ================== Keepalive ==================
async def keepalive_loop(url: str, chat_id: int|str):
    while state.get("keepalive"):
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                await client.get(url)
        except Exception:
            pass
        await asyncio.sleep(12 * 60)

# ================== Kommandon ==================
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

    # väntar på JA/NEJ
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
            "/status\n"
            "/set_ai <neutral|aggressiv|försiktig>\n"
            "/start_mock  (svara JA)\n"
            "/start_live  (svara JA)\n"
            "/engine_start | /engine_stop\n"
            "/symbols BTCUSDT,ETHUSDT,...\n"
            "/timeframe 1min|3min|5min\n"
            "/risk tp 0.004 sl 0.0015 be 0.002 trail 0.001\n"
            "/exitwicks on|off   (exit på wick eller close)\n"
            "/failsafe on|off    (nöd-exit på ORB-low)\n"
            "/export_csv\n"
            "/pnl | /reset_pnl\n"
            "/keepalive_on | /keepalive_off"
        )
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}   AI: {state['ai_mode']}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}\n"
            f"Avgift (per sida): {FEE_RATE:.4f}\n"
            f"TP: {TP_PCT:.4%}  SL: {SL_PCT:.4%}\n"
            f"BE: {BE_TRIGGER_PCT:.4%}  Trail: {TRAIL_STEP_PCT:.4%}\n"
            f"ExitWick: {'på' if EXIT_ON_WICK else 'av'}  FailsafeORB: {'på' if FAILSAFE_ORB else 'av'}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Min range: {MIN_ORB_RANGE:.4%}\n"
            f"Mock USDT: {MOCK_USDT_PER_TRADE:.2f}\n"
            f"PnL idag: {state['pnl_today']:.6f}"
        )
        return {"ok": True}

    if text.startswith("/set_ai"):
        parts = text.split(maxsplit=1)
        if len(parts)<2:
            await tg_send_text(chat_id,"Använd: /set_ai neutral|aggressiv|försiktig"); return {"ok":True}
        nm = parts[1].strip().lower()
        if nm == "forsiktig": nm = "försiktig"
        if nm not in {"neutral","aggressiv","försiktig"}:
            await tg_send_text(chat_id,"Ogiltigt AI-läge."); return {"ok":True}
        state["ai_mode"] = nm
        await tg_send_text(chat_id, f"AI-läge: {nm}")
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
        await start_engine(chat_id); return {"ok": True}
    if text.startswith("/engine_stop") or text.startswith("/stop"):
        await stop_engine(chat_id); return {"ok": True}

    if text.startswith("/symbols"):
        parts = text.split(maxsplit=1)
        if len(parts)<2:
            await tg_send_text(chat_id, f"Aktuell: {', '.join(state['symbols'])}"); return {"ok":True}
        raw = parts[1].replace(" ","")
        syms = [s for s in raw.split(",") if s]
        state["symbols"] = [s.upper() for s in syms]
        await tg_send_text(chat_id, f"Symbols uppdaterade: {', '.join(state['symbols'])}")
        return {"ok": True}

    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts)>1 else "1min"
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram: {tf}")
        return {"ok": True}

    if text.startswith("/risk"):
        # /risk tp 0.004 sl 0.0015 be 0.002 trail 0.001
        global TP_PCT, SL_PCT, BE_TRIGGER_PCT, TRAIL_STEP_PCT
        tokens = text.lower().split()
        try:
            for i in range(1,len(tokens),2):
                k=tokens[i]; v=float(tokens[i+1])
                if k=="tp": TP_PCT=v
                elif k=="sl": SL_PCT=v
                elif k=="be": BE_TRIGGER_PCT=v
                elif k in ("trail","tr"): TRAIL_STEP_PCT=v
            await tg_send_text(chat_id, f"Risk uppdaterad — TP {TP_PCT:.4%}, SL {SL_PCT:.4%}, BE {BE_TRIGGER_PCT:.4%}, Trail {TRAIL_STEP_PCT:.4%}")
        except Exception:
            await tg_send_text(chat_id,"Exempel: /risk tp 0.004 sl 0.0015 be 0.002 trail 0.001")
        return {"ok": True}

    if text.startswith("/exitwicks"):
        global EXIT_ON_WICK
        EXIT_ON_WICK = ("on" in text.lower() or "1" in text.lower())
        await tg_send_text(chat_id, f"Exit på wick: {'på' if EXIT_ON_WICK else 'av'}")
        return {"ok": True}

    if text.startswith("/failsafe"):
        global FAILSAFE_ORB
        FAILSAFE_ORB = ("on" in text.lower() or "1" in text.lower())
        await tg_send_text(chat_id, f"Failsafe ORB-low: {'på' if FAILSAFE_ORB else 'av'}")
        return {"ok": True}

    if text.startswith("/export_csv"):
        sent=False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p); sent=True
        if not sent: await tg_send_text(chat_id,"Inga loggar hittades.")
        return {"ok": True}

    if text.startswith("/pnl"):
        await tg_send_text(chat_id, f"Ack PnL idag: {state['pnl_today']:.6f}")
        return {"ok": True}

    if text.startswith("/reset_pnl"):
        state["pnl_today"]=0.0
        await tg_send_text(chat_id,"PnL nollställt.")
        return {"ok": True}

    if text.startswith("/keepalive_on"):
        public_url = f"https://{os.getenv('RENDER_EXTERNAL_URL','')}/"
        if not public_url or public_url == "https:///":
            public_url = "https://mporbbot.onrender.com/"
        if state.get("keepalive"):
            await tg_send_text(chat_id,"Keepalive redan på."); return {"ok":True}
        state["keepalive"]=True
        loop = asyncio.get_event_loop()
        state["keepalive_task"]=loop.create_task(keepalive_loop(public_url, chat_id))
        await tg_send_text(chat_id, f"Keepalive PÅ – pingar {public_url} var 12:e minut.")
        return {"ok": True}

    if text.startswith("/keepalive_off"):
        state["keepalive"]=False
        t=state.get("keepalive_task")
        if t and not t.done(): t.cancel()
        await tg_send_text(chat_id,"Keepalive AV.")
        return {"ok": True}

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
            "FEE_RATE": FEE_RATE,
            "TP_PCT": TP_PCT, "SL_PCT": SL_PCT,
            "BE_TRIGGER_PCT": BE_TRIGGER_PCT, "TRAIL_STEP_PCT": TRAIL_STEP_PCT,
            "ORB_BUFFER": ORB_BUFFER, "MIN_ORB_RANGE": MIN_ORB_RANGE,
            "EXIT_ON_WICK": EXIT_ON_WICK, "FAILSAFE_ORB": FAILSAFE_ORB
        }
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
