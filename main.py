# main.py — Mp ORBbot (färgbyte-ORB, TP/SL/BE/Trail, mock/live stub)
from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ───── Miljö ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()

FEE_RATE     = float(os.getenv("FEE_RATE", "0.001"))            # 0.1%/sida
DEFAULT_TF   = os.getenv("DEFAULT_TF", "1min")
DEFAULT_SYMS = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]

# Risk/exit
TP_PCT     = float(os.getenv("TP_PCT", "0.0012"))   # +0.12%
SL_PCT     = float(os.getenv("SL_PCT", "0.0005"))   # -0.05%
BE_TRIGGER = float(os.getenv("BE_TRIGGER", "0.0010")) # +0.10% aktiverar BE
TRAIL_STEP = float(os.getenv("TRAIL_STEP", "0.0005"))  # 0.05% trail

MOCK_USDT  = float(os.getenv("MOCK_USDT", "30"))
MOCK_LOG   = "mock_trade_log.csv"
REAL_LOG   = "real_trade_log.csv"
KU_PUBLIC  = "https://api.kucoin.com"

# ───── Hjälpare ───────────────────────────────────────────────────────────────
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def candle_color(o: float, c: float) -> str:
    return "green" if c > o else ("red" if c < o else "doji")

def to_kucoin_symbol(usdt_sym: str) -> str:
    s = usdt_sym.upper()
    return f"{s[:-4]}-USDT" if s.endswith("USDT") else s.replace("/","-")

def compute_qty_from_usdt(price: float, usdt: float) -> float:
    if price <= 0: return 0.0
    return float(f"{usdt/price:.6f}")

def compute_pnl_long(entry: float, exit: float, qty: float):
    gross = (exit - entry) * qty
    fees  = (entry + exit) * qty * FEE_RATE
    return gross, fees, gross - fees

# ───── State ──────────────────────────────────────────────────────────────────
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",
    "ai_mode": "neutral",          # kvar för kompatibilitet
    "pending_confirm": None,       # {"type":"mock"|"live"}
    "symbols": DEFAULT_SYMS[:],
    "timeframe": DEFAULT_TF,
    "engine_task": None,
    "per_symbol": {},              # symbol -> ORBState
    "pnl_today": 0.0,
}

@dataclass
class ORBState:
    last_color: str = ""           # "green"|"red"
    orb_high: float = 0.0
    orb_low: float  = 0.0
    orb_from: str = ""             # färgen på bytar-candlen
    orb_ts: pd.Timestamp | None = None
    in_position: bool = False
    side: str = ""
    entry_price: float = 0.0
    entry_qty: float = 0.0
    highest: float = 0.0
    trail_level: float = 0.0

# ───── Telegram ───────────────────────────────────────────────────────────────
TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None

async def tg_send_text(chat_id: int|str, text: str):
    if not TG_BASE: return
    async with httpx.AsyncClient(timeout=20) as c:
        await c.post(f"{TG_BASE}/sendMessage", json={"chat_id": chat_id, "text": text})

async def tg_send_document(chat_id: int|str, path: str, caption: str=""):
    if not TG_BASE: return
    if not os.path.exists(path):
        await tg_send_text(chat_id, f"Filen saknas: {path}")
        return
    async with httpx.AsyncClient(timeout=60) as c:
        with open(path, "rb") as f:
            files = {"document": (os.path.basename(path), f)}
            data = {"chat_id": chat_id, "caption": caption}
            await c.post(f"{TG_BASE}/sendDocument", data=data, files=files)

def is_admin(chat_id: Any) -> bool:
    return str(chat_id) == str(ADMIN_CHAT_ID) if ADMIN_CHAT_ID else True

# ───── Loggning ───────────────────────────────────────────────────────────────
LOG_FIELDS = ["timestamp","mode","symbol","side","price","qty","fee_rate","fee_cost",
              "gross_pnl","net_pnl","entry_price","exit_price","note"]

def write_log(path: str, row: Dict[str, Any]):
    new = not os.path.exists(path)
    for k in LOG_FIELDS: row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new: w.writeheader()
        w.writerow({k: row.get(k,"") for k in LOG_FIELDS})

async def mock_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str,Any]]=None):
    fee = price*qty*FEE_RATE
    row = {"timestamp": utcnow_iso(), "mode":"mock", "symbol":symbol, "side":side, "price":price,
           "qty":qty, "fee_rate":FEE_RATE, "fee_cost":round(fee,8), "note":"mock"}
    if extra: row.update(extra)
    write_log(MOCK_LOG, row)
    await tg_send_text(chat_id, f"✅ MOCK: {side.upper()} {symbol} @ {price:.6f} x {qty}")

async def live_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str,Any]]=None):
    # Stub – byt mot riktig order när du kopplar in API
    fee = price*qty*FEE_RATE
    row = {"timestamp": utcnow_iso(), "mode":"live", "symbol":symbol, "side":side, "price":price,
           "qty":qty, "fee_rate":FEE_RATE, "fee_cost":round(fee,8), "note":"live (stub)"}
    if extra: row.update(extra)
    write_log(REAL_LOG, row)
    await tg_send_text(chat_id, f"🟢 LIVE-logg: {side.upper()} {symbol} @ {price:.6f} x {qty}")

async def notify_closed(chat_id: int, symbol: str, entry: float, exit: float, qty: float, note: str):
    g,f,n = compute_pnl_long(entry, exit, qty)
    state["pnl_today"] = float(state.get("pnl_today",0.0)) + n
    emo = "📈" if n>=0 else "📉"
    await tg_send_text(chat_id,
        f"{emo} Stängd LONG {symbol}\nEntry: {entry:.6f}  Exit: {exit:.6f}\n"
        f"Qty: {qty:g}\nGross: {g:.6f}  Avg: {f:.6f}\nNetto: {n:.6f}\nPnL idag: {state['pnl_today']:.6f} ({note})"
    )

# ───── Datahämtning ───────────────────────────────────────────────────────────
async def fetch_candles(symbol: str, timeframe: str="1min", limit: int=200) -> pd.DataFrame:
    url = f"{KU_PUBLIC}/api/v1/market/candles"
    params = {"type": timeframe, "symbol": to_kucoin_symbol(symbol)}
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
    cols = ["ts","open","close","high","low","volume","turnover"]
    rows = [row[:7] for row in data]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    for c_ in ["open","close","high","low","volume"]: df[c_] = df[c_].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df.tail(limit)

# ───── ORB-logik ──────────────────────────────────────────────────────────────
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    st: ORBState = state["per_symbol"].setdefault(symbol, ORBState())
    df = await fetch_candles(symbol, timeframe, 200)
    if df.empty or len(df)<5: return

    last, prev = df.iloc[-1], df.iloc[-2]
    last_color = candle_color(last["open"], last["close"])

    # Sätt ORB enbart vid färgbyte (på bytar-candlen)
    if last_color in ("green","red"):
        prev_color = candle_color(prev["open"], prev["close"])
        if st.last_color and last_color != st.last_color:
            st.orb_high, st.orb_low = float(last["high"]), float(last["low"])
            st.orb_from, st.orb_ts = last_color, last["ts"]
            st.in_position = False; st.side=""; st.entry_price=0.0; st.entry_qty=0.0
            st.highest=0.0; st.trail_level=0.0
        st.last_color = last_color

    if st.orb_from not in ("green","red") or st.orb_high==0 or st.orb_low==0:
        return

    price = float(last["close"])

    # ENTRY (bara LONG): stängning över ORB high efter bytar-candlen
    if (not st.in_position) and st.orb_from=="green" and price>st.orb_high and (st.orb_ts is None or last["ts"]>st.orb_ts):
        st.in_position=True; st.side="long"; st.entry_price=price
        st.entry_qty=compute_qty_from_usdt(price, MOCK_USDT)
        st.highest=price; st.trail_level=0.0
        extra={"note":"ORB long"}
        if state["mode"]=="mock": await mock_trade(chat_id, symbol, "buy", price, st.entry_qty, extra)
        else: await live_trade(chat_id, symbol, "buy", price, st.entry_qty, extra)

    # EXIT/Trail
    if st.in_position and st.side=="long":
        if price>st.highest: st.highest=price
        # aktivera BE vid trigger
        if st.trail_level==0.0 and price>=st.entry_price*(1+BE_TRIGGER):
            st.trail_level=st.entry_price
        elif st.trail_level>0.0:
            cand = st.highest*(1-TRAIL_STEP)
            if cand>st.trail_level: st.trail_level=cand

        reason=""; do_exit=False
        if price>=st.entry_price*(1+TP_PCT): do_exit=True; reason="TP"
        elif price<=st.entry_price*(1-SL_PCT): do_exit=True; reason="SL"
        elif st.trail_level>0 and price<=st.trail_level: do_exit=True; reason="Trail/BE"

        if do_exit:
            exit_price=price; qty=st.entry_qty or compute_qty_from_usdt(st.entry_price, MOCK_USDT)
            g,f,n = compute_pnl_long(st.entry_price, exit_price, qty)
            extra={"entry_price":st.entry_price,"exit_price":exit_price,"gross_pnl":round(g,8),"net_pnl":round(n,8),"note":reason}
            if state["mode"]=="mock": await mock_trade(chat_id, symbol, "sell", exit_price, qty, extra)
            else: await live_trade(chat_id, symbol, "sell", exit_price, qty, extra)
            await notify_closed(chat_id, symbol, st.entry_price, exit_price, qty, reason)
            st.in_position=False; st.side=""; st.entry_price=0.0; st.entry_qty=0.0; st.highest=0.0; st.trail_level=0.0

# ───── Motor ──────────────────────────────────────────────────────────────────
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            await asyncio.gather(*[process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]], return_exceptions=True)
        except Exception as e:
            await tg_send_text(chat_id, f"Motorfel: {e}")
        await asyncio.sleep(6)  # ~ var sjätte sekund

async def start_engine(chat_id: int):
    if state["engine_task"] and not state["engine_task"].done():
        await tg_send_text(chat_id, "Motorn kör redan."); return
    state["active"]=True
    state["engine_task"]=asyncio.get_event_loop().create_task(run_engine(chat_id))
    await tg_send_text(chat_id, f"🔄 Motor startad {utcnow_iso()} (tf={state['timeframe']}, symbols={', '.join(state['symbols'])}).")

async def stop_engine(chat_id: int):
    state["active"]=False
    t=state.get("engine_task")
    if t and not t.done(): t.cancel()
    await tg_send_text(chat_id, "⏹️ Motor stoppad.")

# ───── Kommandon ──────────────────────────────────────────────────────────────
HELP = (
    "/status – visa status\n"
    "/set_ai <neutral|aggressiv|försiktig>\n"
    "/start_mock – starta mock (svara JA)\n"
    "/start_live – starta LIVE (svara JA)\n"
    "/engine_start – starta motor\n"
    "/engine_stop – stoppa motor\n"
    "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
    "/timeframe 1min – byt tidsram\n"
    "/export_csv – skicka loggar\n"
    "/pnl – visa dagens PnL\n"
    "/reset_pnl – nollställ PnL"
)

async def handle_update(data: Dict[str, Any]):
    msg = data.get("message") or data.get("edited_message") or {}
    chat = msg.get("chat",{})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()
    if not chat_id or not text: return {"ok": True}
    if not is_admin(chat_id):
        await tg_send_text(chat_id, "⛔️ Inte behörig."); return {"ok": True}

    # Pending confirm
    if state["pending_confirm"]:
        if text.lower()=="ja":
            state["mode"]=state["pending_confirm"]["type"]
            state["pending_confirm"]=None
            await tg_send_text(chat_id, f"✅ {state['mode'].upper()} aktiverad.")
            await start_engine(chat_id)
        else:
            await tg_send_text(chat_id,"Avbrutet."); state["pending_confirm"]=None
        return {"ok": True}

    if text.startswith("/help"):
        await tg_send_text(chat_id, HELP); return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            "Status: {s}\nLäge: {m}\nAI: {ai}\nAvgift (per sida): {fee:.4f}\n"
            "Symbols: {sym}\nTF: {tf}\nPnL idag: {pnl:.6f}\n"
            "TP: {tp:.4%}  SL: {sl:.4%}  BE: {be:.4%}  Trail: {tr:.4%}\n"
            "Mock USDT: {mu:.2f}".format(
                s="AKTIV" if state["active"] else "INAKTIV",
                m=state["mode"], ai=state["ai_mode"], fee=FEE_RATE,
                sym=", ".join(state["symbols"]), tf=state["timeframe"],
                pnl=state["pnl_today"], tp=TP_PCT, sl=SL_PCT, be=BE_TRIGGER, tr=TRAIL_STEP, mu=MOCK_USDT
            )
        ); return {"ok": True}

    if text.startswith("/set_ai"):
        mode = (text.split(maxsplit=1)[1] if len(text.split())>1 else "neutral").lower()
        if mode=="forsiktig": mode="försiktig"
        if mode not in {"neutral","aggressiv","försiktig"}:
            await tg_send_text(chat_id,"Använd: /set_ai neutral|aggressiv|försiktig"); return {"ok": True}
        state["ai_mode"]=mode; await tg_send_text(chat_id,f"AI-läge: {mode}"); return {"ok": True}

    if text.startswith("/start_mock"):
        state["pending_confirm"]={"type":"mock"}
        await tg_send_text(chat_id,"Starta MOCK? Svara JA."); return {"ok": True}

    if text.startswith("/start_live"):
        state["pending_confirm"]={"type":"live"}
        await tg_send_text(chat_id,"⚠️ Starta LIVE? Svara JA."); return {"ok": True}

    if text.startswith("/engine_start"):
        await start_engine(chat_id); return {"ok": True}

    if text.startswith("/engine_stop") or text.startswith("/stop"):
        await stop_engine(chat_id); return {"ok": True}

    if text.startswith("/symbols"):
        parts=text.split(maxsplit=1)
        if len(parts)<2:
            await tg_send_text(chat_id, f"Aktuella: {', '.join(state['symbols'])}")
        else:
            syms=[s for s in parts[1].replace(" ","").split(",") if s]
            state["symbols"]=[s.upper() for s in syms]
            await tg_send_text(chat_id, f"Symbols uppdaterade: {', '.join(state['symbols'])}")
        return {"ok": True}

    if text.startswith("/timeframe"):
        tf=(text.split()[1] if len(text.split())>1 else "1min")
        state["timeframe"]=tf; await tg_send_text(chat_id,f"Tidsram: {tf}"); return {"ok": True}

    if text.startswith("/export_csv"):
        sent=False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p): await tg_send_document(chat_id,p,caption=p); sent=True
        if not sent: await tg_send_text(chat_id,"Inga loggar ännu.")
        return {"ok": True}

    if text.startswith("/pnl"):
        await tg_send_text(chat_id, f"PnL idag: {state['pnl_today']:.6f}"); return {"ok": True}

    if text.startswith("/reset_pnl"):
        state["pnl_today"]=0.0; await tg_send_text(chat_id,"PnL nollställd."); return {"ok": True}

    return {"ok": True}

# ───── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {"message":"Mp ORBbot live", "webhook_path": WEBHOOK_PATH}

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
