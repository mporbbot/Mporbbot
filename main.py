# main.py — Mp ORBbot (kompakt & stabil)
# - ORB vid riktningsskifte
# - Entry: CLOSE > ORB-high (inte wick)
# - Exit: TP/SL, break-even efter +0.10%, trailing efter BE
# - KuCoin publika candles, FastAPI Telegram-webhook
# - Mock/live-loggar + PnL

from __future__ import annotations
import os, csv, asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ================== Miljö & standarder ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()  # t.ex. 5397586616
TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None

DEFAULT_SYMBOLS = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]

# Avgifter & risk
FEE_RATE   = float(os.getenv("FEE_RATE", "0.001"))   # 0.10% per sida
TP_PCT     = float(os.getenv("TP_PCT", "0.0012"))    # +0.12%
SL_PCT     = float(os.getenv("SL_PCT", "0.0005"))    # -0.05%
BE_TRIGGER = float(os.getenv("BE_TRIGGER_PCT", "0.0010"))  # +0.10% -> flytta stop till BE
TRAIL_STEP = float(os.getenv("TRAIL_STEP_PCT", "0.0005"))  # trail steg 0.05%

# ORB-regler
ORB_BUFFER   = float(os.getenv("ORB_BUFFER", "0.0"))   # extra buffert över high (0 = av)
MIN_ORB_RANGE = float(os.getenv("MIN_ORB_RANGE", "0.0005"))  # min 0.05%

MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ================== Tillstånd ==================
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",            # "mock" | "live"
    "ai_mode": "neutral",      # neutral vid start
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": "1min",       # fler trades
    "engine_task": None,
    "per_symbol": {},          # symbol -> ORBState
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
}

# ================== Telegram ==================
async def tg_send_text(chat_id: int|str, text: str):
    if not TG_BASE: return
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(f"{TG_BASE}/sendMessage", json={"chat_id": chat_id, "text": text})

async def tg_send_document(chat_id: int|str, file_path: str, caption: str=""):
    if not TG_BASE or not os.path.exists(file_path): 
        if TG_BASE: await tg_send_text(chat_id, f"Filen saknas: {file_path}")
        return
    async with httpx.AsyncClient(timeout=60) as client:
        with open(file_path, "rb") as f:
            files = {"document": (os.path.basename(file_path), f)}
            data = {"chat_id": chat_id, "caption": caption}
            await client.post(f"{TG_BASE}/sendDocument", data=data, files=files)

def is_admin(chat_id: Any) -> bool:
    return str(chat_id) == str(ADMIN_CHAT_ID) if ADMIN_CHAT_ID else True

# ================== Hjälp-funktioner ==================
def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"):
        return f"{usdt_sym[:-4]}-USDT"
    return usdt_sym.replace("/", "-").upper()

async def fetch_candles(symbol: str, timeframe: str="1min", limit: int=120) -> pd.DataFrame:
    url = "https://api.kucoin.com/api/v1/market/candles"
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

def write_log_row(path: str, row: Dict[str, Any], header: List[str]):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

def compute_pnl(entry: float, exit: float, qty: float, fee_rate: float):
    gross = (exit - entry) * qty
    fees  = (entry + exit) * qty * fee_rate
    net   = gross - fees
    return gross, fees, net

# ================== Modell ==================
@dataclass
class ORBState:
    direction: str = "flat"        # "up"|"down"|"flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    entry_price: float = 0.0
    qty: float = 0.0
    stop: float = 0.0
    take: float = 0.0
    be_moved: bool = False         # stop flyttad till BE?
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
TRADE_HEADER = ["timestamp","mode","symbol","side","price","qty",
                "fee_rate","fee_cost","gross_pnl","net_pnl","entry_price","exit_price","note"]

async def log_trade(mode: str, symbol: str, side: str, price: float, qty: float,
                    extra: Optional[Dict[str, Any]]=None):
    path = MOCK_LOG if mode=="mock" else REAL_LOG
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": mode, "symbol": symbol,
        "side": side, "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": ""
    }
    if extra: row.update(extra)
    write_log_row(path, row, TRADE_HEADER)

async def enter_long(chat_id: int, st: ORBState, symbol: str, price: float):
    st.in_position = True
    st.entry_price = price
    st.qty = round(30.0 / price, 6)  # ~30 USDT per trade (ändra här om du vill)
    st.stop = price * (1 - SL_PCT)
    st.take = price * (1 + TP_PCT)
    st.be_moved = False
    await log_trade(state["mode"], symbol, "buy", price, st.qty, {"note":"entry"})
    await tg_send_text(chat_id, f"🟢 LONG {symbol} @ {price:.6f} | TP {st.take:.6f} SL {st.stop:.6f}")

async def exit_long(chat_id: int, st: ORBState, symbol: str, price: float, note: str):
    gross, fees, net = compute_pnl(st.entry_price, price, st.qty, FEE_RATE)
    await log_trade(state["mode"], symbol, "sell", price, st.qty, {
        "entry_price": st.entry_price, "exit_price": price,
        "gross_pnl": round(gross, 8), "net_pnl": round(net, 8),
        "note": note
    })
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
    emo = "📈" if net >= 0 else "📉"
    await tg_send_text(
        chat_id,
        f"{emo} Stängd LONG {symbol}\nEntry: {st.entry_price:.6f}  Exit: {price:.6f}\n"
        f"Qty: {st.qty:g}\nGross: {gross:.6f}  Avg: {fees:.6f}\nNetto: {net:.6f}\n"
        f"PnL idag: {state['pnl_today']:.6f}"
    )
    st.in_position = False
    st.entry_price = 0.0
    st.qty = 0.0
    st.stop = 0.0
    st.take = 0.0
    st.be_moved = False

# ================== Motorn ==================
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, timeframe, limit=80)
    if df.empty or len(df) < 10:
        return

    prev, last = df.iloc[-2], df.iloc[-1]
    st.last_close = float(last["close"])

    # 1) Uppdatera riktning och ev. skapa ny ORB
    last_dir = detect_dir(prev["close"], last["close"])
    if last_dir != st.direction and last_dir in ("up","down"):
        reset_orb(st, last["high"], last["low"])
        st.direction = last_dir

    # 2) Minsta range-krav
    if st.orb_low > 0 and st.orb_high > 0:
        rng = (st.orb_high / st.orb_low) - 1.0 if st.orb_low else 0.0
        if rng < MIN_ORB_RANGE:
            # för liten box -> vänta
            pass

    # 3) ENTRY: endast om candle CLOSE > ORB-high*(1+buffer)
    entry_ok = (
        (not st.in_position)
        and st.orb_high > 0
        and last["close"] > st.orb_high * (1.0 + ORB_BUFFER)
        and st.direction == "up"
    )
    if entry_ok:
        await enter_long(chat_id, st, symbol, float(last["close"]))

    # 4) EXIT: TP/SL intrabar via high/low + BE & trail
    if st.in_position:
        entry = st.entry_price
        high = float(last["high"])
        low  = float(last["low"])
        close = float(last["close"])

        # flytta stop till BE vid +BE_TRIGGER
        if not st.be_moved and (high >= entry * (1 + BE_TRIGGER)):
            st.stop = entry  # BE (grovt – vill du inkludera avgift: entry * (1 + 2*FEE_RATE))
            st.be_moved = True

        # trailing efter BE: höj stop i steg under nuvarande pris
        if st.be_moved:
            target_trail = close * (1 - TRAIL_STEP)
            if target_trail > st.stop:
                st.stop = target_trail

        # TP först
        if high >= st.take:
            await exit_long(chat_id, st, symbol, st.take, "take-profit")
            return

        # SL
        if low <= st.stop:
            await exit_long(chat_id, st, symbol, st.stop, "stop-loss")
            return

async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks: await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(15)  # loop-intervall
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

# ================== Kommandon ==================
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

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}  AI: {state['ai_mode']}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}\n"
            f"TP: {TP_PCT:.4%}  SL: {SL_PCT:.4%}\n"
            f"BE: {BE_TRIGGER:.4%}  Trail steg: {TRAIL_STEP:.4%}\n"
            f"PnL idag: {state['pnl_today']:.6f}")
        return {"ok": True}

    if text.startswith("/engine_start"):
        await start_engine(chat_id); return {"ok": True}

    if text.startswith("/engine_stop"):
        await stop_engine(chat_id); return {"ok": True}

    if text.startswith("/set_ai"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send_text(chat_id, "Använd: /set_ai neutral|aggressiv|försiktig")
            return {"ok": True}
        nm = parts[1].strip().lower()
        if nm == "forsiktig": nm = "försiktig"
        if nm not in {"neutral","aggressiv","försiktig"}:
            await tg_send_text(chat_id, "Ogiltigt AI-läge."); return {"ok": True}
        state["ai_mode"] = nm
        await tg_send_text(chat_id, f"AI-läge: {nm}"); return {"ok": True}

    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts) > 1 else "1min"
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram: {tf}")
        return {"ok": True}

    if text.startswith("/symbols"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send_text(chat_id, f"Aktuell lista: {', '.join(state['symbols'])}")
            return {"ok": True}
        syms = [s.strip().upper() for s in parts[1].split(",") if s.strip()]
        state["symbols"] = syms
        await tg_send_text(chat_id, f"Symbols uppdaterade: {', '.join(syms)}")
        return {"ok": True}

    if text.startswith("/risk"):
        # /risk tp 0.0012 sl 0.0005 be 0.0010 trail 0.0005
        global TP_PCT, SL_PCT, BE_TRIGGER, TRAIL_STEP
        parts = text.lower().split()
        try:
            if "tp" in parts:    TP_PCT = float(parts[parts.index("tp")+1])
            if "sl" in parts:    SL_PCT = float(parts[parts.index("sl")+1])
            if "be" in parts:    BE_TRIGGER = float(parts[parts.index("be")+1])
            if "trail" in parts: TRAIL_STEP = float(parts[parts.index("trail")+1])
            await tg_send_text(chat_id, f"Risk uppdaterad: TP {TP_PCT:.4%}, SL {SL_PCT:.4%}, BE {BE_TRIGGER:.4%}, Trail {TRAIL_STEP:.4%}")
        except Exception:
            await tg_send_text(chat_id, "Fel format. Ex: /risk tp 0.0012 sl 0.0005 be 0.0010 trail 0.0005")
        return {"ok": True}

    if text.startswith("/export_csv"):
        sent = False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p); sent = True
        if not sent: await tg_send_text(chat_id, "Inga loggar än.")
        return {"ok": True}

    if text.startswith("/pnl"):
        await tg_send_text(chat_id, f"PnL idag: {state.get('pnl_today',0.0):.6f}")
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
                    async with httpx.AsyncClient(timeout=10) as c: await c.get(public_url)
                except Exception: pass
                await asyncio.sleep(12*60)
        state["keepalive_task"] = asyncio.get_event_loop().create_task(loop())
        await tg_send_text(chat_id, f"Keepalive PÅ – ping {public_url}")
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
            "ORB_BUFFER": ORB_BUFFER, "MIN_ORB_RANGE": MIN_ORB_RANGE
        }
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
