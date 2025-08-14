# main.py — Mp ORBbot (stabil en-fil)
# - ORB skapas när riktningen byter
# - ENTRY: stängd candle CLOSE > ORB-high*(1+buffer)
# - EXIT: TP/SL på close (wick ignoreras), BE + trailing efter vinst
# - Mock/live-loggar (separata CSV) med gross/fees/net
# - Kommandon via Telegram (se /help)

from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ================== Miljövariabler (med vettiga default) ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()

FEE_RATE   = float(os.getenv("FEE_RATE", "0.001"))      # 0.10% per sida
TP_PCT     = float(os.getenv("TP_PCT", "0.0012"))       # +0.12%
SL_PCT     = float(os.getenv("SL_PCT", "0.0005"))       # -0.05%
BE_TRIGGER = float(os.getenv("BE_TRIGGER_PCT", "0.001"))# +0.10% -> BE
TRAIL_STEP = float(os.getenv("TRAIL_STEP_PCT", "0.0005")) # 0.05% trailsteg

ORB_BUFFER    = float(os.getenv("ORB_BUFFER", "0.0000"))   # extra buffert
MIN_ORB_RANGE = float(os.getenv("MIN_ORB_RANGE", "0.0005"))# min 0.05% range

MOCK_USDT     = float(os.getenv("MOCK_USDT", "30.0"))      # ~30 USDT per trade
LOOP_SECONDS  = float(os.getenv("LOOP_SECONDS", "12"))     # motor loop-sleep

DEFAULT_SYMBOLS = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]

MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

KU_PUBLIC = "https://api.kucoin.com"
TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ================== Globalt state ==================
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",           # "mock" | "live"
    "ai_mode": "neutral",
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": "1min",
    "engine_task": None,
    "per_symbol": {},         # symbol -> ORBState
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
}

# ================== Telegram helpers ==================
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

# ================== CSV & PnL ==================
TRADE_FIELDS = [
    "timestamp","mode","symbol","side","price","qty",
    "fee_rate","fee_cost","gross_pnl","net_pnl","entry_price","exit_price","note"
]

def write_log_row(path: str, row: Dict[str, Any]):
    newfile = not os.path.exists(path)
    for k in TRADE_FIELDS: row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRADE_FIELDS})

def compute_pnl(entry: float, exitp: float, qty: float, fee_rate: float):
    gross = (exitp - entry) * qty
    fees  = (entry + exitp) * qty * fee_rate
    net   = gross - fees
    return gross, fees, net

# ================== Data ==================
def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"):
        return f"{usdt_sym[:-4]}-USDT"
    return usdt_sym.replace("/", "-").upper()

async def fetch_candles(symbol: str, timeframe: str="1min", limit: int=120) -> pd.DataFrame:
    url = f"{KU_PUBLIC}/api/v1/market/candles"
    params = {"type": timeframe, "symbol": to_kucoin_symbol(symbol)}
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

# ================== ORB ==================
@dataclass
class ORBState:
    direction: str = "flat"    # "up"|"down"|"flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    entry_price: float = 0.0
    qty: float = 0.0
    stop: float = 0.0
    take: float = 0.0
    be_moved: bool = False
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
    st.stop = 0.0
    st.take = 0.0
    st.be_moved = False

# ================== Handel ==================
async def log_trade(mode: str, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    path = MOCK_LOG if mode == "mock" else REAL_LOG
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": mode, "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "", "note": ""
    }
    if extra: row.update(extra)
    write_log_row(path, row)

async def enter_long(chat_id: int, st: ORBState, symbol: str, price: float):
    qty = max(MOCK_USDT / price, 0.00001)
    st.in_position = True
    st.entry_price = price
    st.qty = qty
    st.stop = price * (1 - SL_PCT)
    st.take = price * (1 + TP_PCT)
    st.be_moved = False
    await log_trade(state["mode"], symbol, "buy", price, qty, {"note": "entry"})
    await tg_send_text(chat_id, f"🟢 LONG {symbol} @ {price:.6f} | TP {st.take:.6f}  SL {st.stop:.6f}")

async def exit_long(chat_id: int, st: ORBState, symbol: str, price: float, note: str):
    gross, fees, net = compute_pnl(st.entry_price, price, st.qty, FEE_RATE)
    await log_trade(state["mode"], symbol, "sell", price, st.qty, {
        "entry_price": st.entry_price, "exit_price": price,
        "gross_pnl": round(gross, 8), "net_pnl": round(net, 8),
        "note": note
    })
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
    emo = "📈" if net >= 0 else "📉"
    await tg_send_text(chat_id,
        f"{emo} Stängd LONG {symbol}\n"
        f"Entry: {st.entry_price:.6f}  Exit: {price:.6f}\n"
        f"Qty: {st.qty:g}\nGross: {gross:.6f}  Avg: {fees:.6f}\n"
        f"Netto: {net:.6f}\nPnL idag: {state['pnl_today']:.6f}"
    )
    st.in_position = False
    st.entry_price = 0.0
    st.qty = 0.0
    st.stop = 0.0
    st.take = 0.0
    st.be_moved = False

# ================== Motor (per symbol) ==================
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, timeframe, limit=90)
    if df.empty or len(df) < 10:
        return

    # Jobba på SENAST STÄNGDA candle (last)
    prev, last = df.iloc[-2], df.iloc[-1]
    st.last_close = float(last["close"])
    last_dir = detect_dir(prev["close"], last["close"])

    # Ny ORB vid riktning
    if last_dir != st.direction and last_dir in ("up","down"):
        reset_orb(st, last["high"], last["low"])
        st.direction = last_dir

    # Min ORB-range
    if st.orb_low > 0 and st.orb_high > 0 and st.orb_low > 0:
        rng = (st.orb_high / st.orb_low) - 1.0
        if rng < MIN_ORB_RANGE:
            pass  # för liten box – vi väntar, men låter entry-logik avgöra

    # ENTRY: CLOSE över ORB-high*(1+buffer) och riktning upp
    trigger = st.orb_high * (1.0 + ORB_BUFFER) if st.orb_high > 0 else None
    if (not st.in_position) and st.direction == "up" and trigger and (last["close"] > trigger) and (prev["close"] <= trigger):
        await enter_long(chat_id, st, symbol, float(last["close"]))

    # EXIT: TP/SL på close-logik + BE + trail
    if st.in_position:
        entry = st.entry_price
        high  = float(last["high"])
        low   = float(last["low"])
        close = float(last["close"])

        # BE när vi varit +BE_TRIGGER
        if (not st.be_moved) and (high >= entry * (1 + BE_TRIGGER)):
            # vill du inkludera avgifter i BE: entry*(1+2*FEE_RATE)
            st.stop = max(st.stop, entry)  # BE
            st.be_moved = True

        # Trail efter BE
        if st.be_moved:
            target_trail = close * (1 - TRAIL_STEP)
            if target_trail > st.stop:
                st.stop = target_trail

        # Prioritera TP
        if close >= st.take:
            await exit_long(chat_id, st, symbol, st.take, "take-profit")
            return

        # SL
        if close <= st.stop:
            await exit_long(chat_id, st, symbol, st.stop, "stop-loss")
            return

# ================== Motor-loop ==================
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks: await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(LOOP_SECONDS)
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
    state["engine_task"] = asyncio.create_task(run_engine(chat_id))
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
            async with httpx.AsyncClient(timeout=8) as c:
                await c.get(url)
        except Exception:
            pass
        await asyncio.sleep(12*60)

# ================== Telegram-kommandon ==================
async def handle_update(data: Dict[str, Any]):
    msg = data.get("message") or data.get("edited_message") or {}
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}
    if not is_admin(chat_id):
        await tg_send_text(chat_id, "⛔️ Inte behörig.")
        return {"ok": True}

    if text.startswith("/help"):
        await tg_send_text(chat_id,
            "/status\n"
            "/engine_start | /engine_stop\n"
            "/set_ai <neutral|aggressiv|försiktig>\n"
            "/timeframe 1min|3min|5min\n"
            "/symbols BTCUSDT,ETHUSDT,...\n"
            "/risk tp 0.0012 sl 0.0005 be 0.001 trail 0.0005\n"
            "/export_csv\n"
            "/pnl | /reset_pnl\n"
            "/keepalive_on | /keepalive_off"
        ); return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}  AI: {state['ai_mode']}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}\n"
            f"TP: {TP_PCT:.4%}  SL: {SL_PCT:.4%}\n"
            f"BE: {BE_TRIGGER:.4%}  Trail: {TRAIL_STEP:.4%}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Min range: {MIN_ORB_RANGE:.4%}\n"
            f"Mock USDT: {MOCK_USDT:.2f}\n"
            f"PnL idag: {state['pnl_today']:.6f}"
        ); return {"ok": True}

    if text.startswith("/engine_start"):
        await start_engine(chat_id); return {"ok": True}

    if text.startswith("/engine_stop"):
        await stop_engine(chat_id); return {"ok": True}

    if text.startswith("/set_ai"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send_text(chat_id, "Använd: /set_ai neutral|aggressiv|försiktig")
        else:
            nm = parts[1].strip().lower()
            if nm == "forsiktig": nm = "försiktig"
            if nm in {"neutral","aggressiv","försiktig"}:
                state["ai_mode"] = nm
                await tg_send_text(chat_id, f"AI-läge: {nm}")
            else:
                await tg_send_text(chat_id, "Ogiltigt AI-läge.")
        return {"ok": True}

    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts) > 1 else "1min"
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram: {tf}")
        return {"ok": True}

    if text.startswith("/symbols"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send_text(chat_id, f"Aktuell: {', '.join(state['symbols'])}")
        else:
            syms = [s.strip().upper() for s in parts[1].split(",") if s.strip()]
            state["symbols"] = syms
            await tg_send_text(chat_id, f"Symbols uppdaterade: {', '.join(syms)}")
        return {"ok": True}

    if text.startswith("/risk"):
        # /risk tp 0.0012 sl 0.0005 be 0.001 trail 0.0005
        global TP_PCT, SL_PCT, BE_TRIGGER, TRAIL_STEP
        parts = text.lower().split()
        try:
            if "tp" in parts:    TP_PCT     = float(parts[parts.index("tp")+1])
            if "sl" in parts:    SL_PCT     = float(parts[parts.index("sl")+1])
            if "be" in parts:    BE_TRIGGER = float(parts[parts.index("be")+1])
            if "trail" in parts: TRAIL_STEP = float(parts[parts.index("trail")+1])
            await tg_send_text(chat_id, f"Risk: TP {TP_PCT:.4%} SL {SL_PCT:.4%} BE {BE_TRIGGER:.4%} Trail {TRAIL_STEP:.4%}")
        except Exception:
            await tg_send_text(chat_id, "Ex: /risk tp 0.0012 sl 0.0005 be 0.001 trail 0.0005")
        return {"ok": True}

    if text.startswith("/export_csv"):
        sent = False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p); sent = True
        if not sent:
            await tg_send_text(chat_id, "Inga loggar ännu.")
        return {"ok": True}

    if text.startswith("/pnl"):
        await tg_send_text(chat_id, f"PnL idag: {state.get('pnl_today', 0.0):.6f}")
        return {"ok": True}

    if text.startswith("/reset_pnl"):
        state["pnl_today"] = 0.0
        await tg_send_text(chat_id, "PnL nollställd.")
        return {"ok": True}

    if text.startswith("/keepalive_on"):
        public_url = f"https://{os.getenv('RENDER_EXTERNAL_URL','')}/"
        if not public_url or public_url == "https:///": public_url = "https://mporbbot.onrender.com/"
        if state.get("keepalive"):
            await tg_send_text(chat_id, "Keepalive redan på."); return {"ok": True}
        state["keepalive"] = True
        async def loop():
            while state["keepalive"]:
                try:
                    async with httpx.AsyncClient(timeout=8) as c:
                        await c.get(public_url)
                except Exception: pass
                await asyncio.sleep(12*60)
        state["keepalive_task"] = asyncio.create_task(loop())
        await tg_send_text(chat_id, f"Keepalive PÅ – pingar {public_url}")
        return {"ok": True}

    if text.startswith("/keepalive_off"):
        state["keepalive"] = False
        t = state.get("keepalive_task")
        if t and not t.done(): t.cancel()
        await tg_send_text(chat_id, "Keepalive AV.")
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
            "TP": TP_PCT, "SL": SL_PCT, "BE_TRIGGER": BE_TRIGGER, "TRAIL_STEP": TRAIL_STEP,
            "ORB_BUFFER": ORB_BUFFER, "MIN_ORB_RANGE": MIN_ORB_RANGE,
            "FEE_RATE": FEE_RATE, "MOCK_USDT": MOCK_USDT
        }
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
