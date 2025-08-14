# main.py — Mp ORBbot (stabil: per-candle, ingen masshandel vid start)
# - ORB skapas på första färgbyte (grön<->röd) i STÄNGD candle
# - LONG entry när STÄNGD candle stänger över ORB-high
# - SL = föregående stängda candle close (flyttas upp varje close)
# - Exit om pågående candle's low <= SL (intrabar)
# - Bearbetar EN stängd candle per loop (ingen historikspamm)
# - Mock/live, CSV-loggar, Telegramkontroller, FastAPI webhook

from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ---------- Miljö ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()
DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]
DEFAULT_TIMEFRAME  = os.getenv("DEFAULT_TIMEFRAME", "1min").strip()

FEE_RATE  = float(os.getenv("FEE_RATE", "0.001"))  # 0.1% per sida
MOCK_USDT = float(os.getenv("MOCK_USDT", "30"))
MIN_ORB_PCT = 0.0010  # min 0.10% range för att undvika brus

MOCK_LOG, REAL_LOG = "mock_trade_log.csv", "real_trade_log.csv"
LOG_FIELDS = ["timestamp","mode","symbol","side","price","qty",
              "fee_rate","fee_cost","gross_pnl","net_pnl","entry_price","exit_price","note"]

KU_PUBLIC = "https://api.kucoin.com"

def utcnow_iso(): return datetime.now(timezone.utc).isoformat()

# ---------- State ----------
@dataclass
class ORBState:
    last_processed_ts: Optional[pd.Timestamp] = None  # senaste bearbetade STÄNGDA candle
    warmed_up: bool = False                          # första nya candle passerad
    last_color: str = ""                             # "green"/"red"
    orb_high: float = 0.0
    orb_low: float = 0.0
    orb_set_ts: Optional[pd.Timestamp] = None
    armed: bool = False                              # ORB redo för entry
    in_position: bool = False
    entry_price: float = 0.0
    entry_qty: float = 0.0
    dyn_sl: float = 0.0                              # dynamisk SL (föregående close)

state: Dict[str, Any] = {
    "active": False, "mode": "mock", "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": DEFAULT_TIMEFRAME, "engine_task": None, "per_symbol": {},
    "pnl_today": 0.0,
}

# ---------- Telegram ----------
TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None
async def tg_send_text(chat_id, text):
    if not TG_BASE: return
    async with httpx.AsyncClient(timeout=20) as c:
        await c.post(f"{TG_BASE}/sendMessage", json={"chat_id": chat_id, "text": text})
async def tg_send_document(chat_id, path, caption=""):
    if not TG_BASE: return
    if not os.path.exists(path):
        await tg_send_text(chat_id, f"Filen saknas: {path}"); return
    async with httpx.AsyncClient(timeout=60) as c:
        with open(path,"rb") as f:
            files={"document":(os.path.basename(path),f)}
            await c.post(f"{TG_BASE}/sendDocument", data={"chat_id":chat_id,"caption":caption}, files=files)
def is_admin(chat_id): return str(chat_id)==str(ADMIN_CHAT_ID) if ADMIN_CHAT_ID else True

# ---------- Logg & PnL ----------
def write_log_row(path, row):
    new = not os.path.exists(path)
    for k in LOG_FIELDS: row.setdefault(k,"")
    with open(path,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new: w.writeheader()
        w.writerow({k:row.get(k,"") for k in LOG_FIELDS})
def compute_pnl(entry, exit, qty, fee_rate):
    gross = (exit-entry)*qty
    fees = (entry+exit)*qty*fee_rate
    return gross, fees, gross-fees
async def log_trade(mode, symbol, side, price, qty, extra=None):
    fee = price*qty*FEE_RATE
    row = {"timestamp":utcnow_iso(),"mode":mode,"symbol":symbol,"side":side,
           "price":price,"qty":qty,"fee_rate":FEE_RATE,"fee_cost":round(fee,8)}
    if extra: row.update(extra)
    write_log_row(MOCK_LOG if mode=="mock" else REAL_LOG, row)

# ---------- Data ----------
def to_ksym(sym: str) -> str:
    sym=sym.upper()
    return f"{sym[:-4]}-USDT" if sym.endswith("USDT") else sym.replace("/","-")
async def fetch_candles(symbol, timeframe="1min", limit=120) -> pd.DataFrame:
    url=f"{KU_PUBLIC}/api/v1/market/candles"; params={"type":timeframe,"symbol":to_ksym(symbol)}
    async with httpx.AsyncClient(timeout=20) as c:
        r=await c.get(url, params=params); r.raise_for_status()
        data=r.json().get("data",[])
    cols=["ts","open","close","high","low","volume","turnover"]
    df=pd.DataFrame([row[:7] for row in data], columns=cols)
    if df.empty: return df
    df["ts"]=pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    for c in ["open","close","high","low","volume"]: df[c]=df[c].astype(float)
    return df.sort_values("ts").reset_index(drop=True)

def color(row)->str: return "green" if row["close"]>row["open"] else ("red" if row["close"]<row["open"] else "doji")

# ---------- Motor per symbol ----------
async def process_symbol(chat_id: int, symbol: str, tf: str):
    if symbol not in state["per_symbol"]: state["per_symbol"][symbol]=ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, tf, limit=120)
    if df.empty or len(df)<3: return

    # stängd candle = [-2], pågående = [-1]
    prev_closed = df.iloc[-3]  # för SL vid entry och SL-flytt
    closed      = df.iloc[-2]
    forming     = df.iloc[-1]

    # Värma upp: första gången vi ser en stängd candle, spara ts och gå ur.
    if st.last_processed_ts is None:
        st.last_processed_ts = closed["ts"]
        st.last_color = color(closed)
        return

    # Kör bara om ny stängd candle kommit
    if closed["ts"] == st.last_processed_ts:
        # men hantera SL på den pågående candle för exit (utan att flytta SL)
        if st.in_position and forming["low"] <= st.dyn_sl:
            exit_price = st.dyn_sl
            qty = st.entry_qty
            gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
            await log_trade(state["mode"], symbol, "sell", exit_price, qty, {
                "entry_price": st.entry_price, "exit_price": exit_price,
                "gross_pnl": round(gross,8), "net_pnl": round(net,8), "note":"intrabar_SL"
            })
            state["pnl_today"] += net
            await tg_send_text(chat_id, f"📉 Stängd LONG {symbol}\nEntry: {st.entry_price:.6f}  Exit: {exit_price:.6f}\nQty: {qty:g}\nPnL idag: {state['pnl_today']:.6f}")
            st.in_position=False; st.entry_price=st.entry_qty=0.0; st.dyn_sl=0.0
        return

    # ======= NY stängd candle har kommit =======
    st.last_processed_ts = closed["ts"]

    # 1) Kolla färgbyte -> sätt ORB på den NYSS stängda candle
    col_prev = color(prev_closed)
    col_cur  = color(closed)

    # init last_color om tom
    if st.last_color=="" and col_prev in ("green","red"):
        st.last_color = col_prev

    if col_cur in ("green","red") and st.last_color in ("green","red") and col_cur != st.last_color:
        st.orb_high = float(closed["high"])
        st.orb_low  = float(closed["low"])
        st.orb_set_ts = closed["ts"]
        # rangefilter
        st.armed = (st.orb_low>0 and (st.orb_high/st.orb_low - 1.0) >= MIN_ORB_PCT)
    st.last_color = col_cur if col_cur in ("green","red") else st.last_color

    # 2) Flytta SL (till föregående stängda close) om i position
    if st.in_position:
        new_sl = float(closed["close"])  # “föregående” inför nästa candle
        if new_sl > st.dyn_sl: st.dyn_sl = new_sl

    # 3) ENTRY: stängd candle stänger över ORB-high (och ORB är beväpnad)
    if (not st.in_position) and st.armed and st.orb_high>0 and closed["close"] > st.orb_high:
        # förhindra tokstart vid allra första handeln efter start
        if not st.warmed_up:
            st.warmed_up = True  # hoppa första möjliga efter uppstart
        else:
            entry_price = float(closed["close"])
            qty = round(MOCK_USDT/entry_price, 8)
            st.in_position=True; st.entry_price=entry_price; st.entry_qty=qty
            st.dyn_sl = float(prev_closed["close"])  # SL start = föregående close
            await log_trade(state["mode"], symbol, "buy", entry_price, qty, {"note":"close>ORB_high"})
            await tg_send_text(chat_id, f"✅ {state['mode'].upper()}: BUY {symbol} @ {entry_price:.6f} x {qty:g}")

    # 4) Direkt efter close: om pågående candle redan punkterat SL -> exit
    if st.in_position and forming["low"] <= st.dyn_sl:
        exit_price = st.dyn_sl
        qty = st.entry_qty
        gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
        await log_trade(state["mode"], symbol, "sell", exit_price, qty, {
            "entry_price": st.entry_price, "exit_price": exit_price,
            "gross_pnl": round(gross,8), "net_pnl": round(net,8), "note":"intrabar_SL"
        })
        state["pnl_today"] += net
        await tg_send_text(chat_id, f"📉 Stängd LONG {symbol}\nEntry: {st.entry_price:.6f}  Exit: {exit_price:.6f}\nQty: {qty:g}\nPnL idag: {state['pnl_today']:.6f}")
        st.in_position=False; st.entry_price=st.entry_qty=0.0; st.dyn_sl=0.0

# ---------- Loop ----------
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks=[process_symbol(chat_id,s,state["timeframe"]) for s in state["symbols"]]
            if tasks: await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(3)  # snabb men lätt
        except asyncio.CancelledError:
            break
        except Exception as e:
            await tg_send_text(chat_id, f"Motorfel: {e}")
            await asyncio.sleep(5)

async def start_engine(chat_id:int):
    if state["engine_task"] and not state["engine_task"].done():
        await tg_send_text(chat_id,"Motorn kör redan."); return
    state["active"]=True
    state["engine_task"]=asyncio.get_event_loop().create_task(run_engine(chat_id))
    await tg_send_text(chat_id, f"🔄 Motor startad {utcnow_iso()} (tf={state['timeframe']}, symbols={', '.join(state['symbols'])}).")

async def stop_engine(chat_id:int):
    state["active"]=False
    t=state.get("engine_task")
    if t and not t.done(): t.cancel()
    await tg_send_text(chat_id,"⏹️ Motor stoppad.")

# ---------- Kommandon ----------
async def handle_update(data: Dict[str,Any]):
    msg = data.get("message") or data.get("edited_message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    text = (msg.get("text") or "").strip()
    if not chat_id or not text: return {"ok":True}
    if not is_admin(chat_id):
        await tg_send_text(chat_id,"⛔️ Du är inte behörig."); return {"ok":True}

    if text.lower()=="ja" and state.get("pending_confirm"):
        mode=state["pending_confirm"]["type"]; state["mode"]=mode; state["pending_confirm"]=None
        await tg_send_text(chat_id, f"✅ {mode.upper()} aktiverad."); await start_engine(chat_id); return {"ok":True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}\nSymbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}\nPnL idag: {state['pnl_today']:.6f}\nMin ORB: {MIN_ORB_PCT:.4%}"
        ); return {"ok":True}

    if text.startswith("/start_mock"):
        state["pending_confirm"]={"type":"mock"}; await tg_send_text(chat_id,"Starta MOCK? Svara JA."); return {"ok":True}
    if text.startswith("/start_live"):
        state["pending_confirm"]={"type":"live"}; await tg_send_text(chat_id,"⚠️ Starta LIVE? Svara JA."); return {"ok":True}
    if text.startswith("/engine_start"): await start_engine(chat_id); return {"ok":True}
    if text.startswith("/engine_stop"):  await stop_engine(chat_id);  return {"ok":True}

    if text.startswith("/timeframe"):
        parts = text.split()
        if len(parts)>1: state["timeframe"]=parts[1]
        await tg_send_text(chat_id, f"Tidsram satt till: {state['timeframe']}"); return {"ok":True}

    if text.startswith("/symbols"):
        parts=text.split(maxsplit=1)
        if len(parts)>1:
            syms=[s for s in parts[1].replace(" ","").split(",") if s]
            state["symbols"]=[s.upper() for s in syms]
        await tg_send_text(chat_id, f"Symbols: {', '.join(state['symbols'])}"); return {"ok":True}

    if text.startswith("/export_csv"):
        sent=False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p): await tg_send_document(chat_id,p,caption=p); sent=True
        if not sent: await tg_send_text(chat_id,"Inga loggar ännu.")
        return {"ok":True}

    if text.startswith("/pnl"): await tg_send_text(chat_id, f"PnL idag: {state['pnl_today']:.6f}"); return {"ok":True}
    if text.startswith("/reset_pnl"): state["pnl_today"]=0.0; await tg_send_text(chat_id,"PnL nollställt."); return {"ok":True}
    return {"ok":True}

# ---------- FastAPI ----------
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"
@app.get("/")
async def root(): return {"message":"Mp ORBbot is live!","webhook_path":WEBHOOK_PATH}
@app.post(WEBHOOK_PATH)
async def webhook(req: Request): return await handle_update(await req.json())
