# ======================== main.py (Mp ORBbot — close-confirmed ORB) ========================
from __future__ import annotations
import os, csv, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ---------------------------- Miljö / konfig ----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()  # valfritt lås till dig

# Avgift per sida (0.001 = 0.1%). Tur/retur ≈ 2*FEE_RATE
FEE_RATE           = float(os.getenv("FEE_RATE", "0.001"))

# ORB
ORB_BUFFER         = float(os.getenv("ORB_BUFFER", "0.0008"))    # 0.08% över ORB-high (för long)
MIN_ORB_RANGE      = float(os.getenv("MIN_ORB_RANGE", "0.0020")) # min 0.20% range för att undvika micro-boxar

# Risk/TP/Trail
STOP_PCT           = float(os.getenv("STOP_PCT", "0.0012"))      # 0.12 % hård SL från entry
TP_PCT             = float(os.getenv("TP_PCT", "0.0012"))        # +0.12 % fast TP
ARM_BE_AT_PCT      = float(os.getenv("ARM_BE_AT_PCT", "0.0010")) # +0.10 % aktivera BE+trail
TRAIL_GAP_PCT      = float(os.getenv("TRAIL_GAP_PCT", "0.0006")) # trail = highest_close*(1-TRAIL_GAP_PCT)
BE_FEE_BUFFER      = float(os.getenv("BE_FEE_BUFFER", "0.00005"))# BE läggs över avgifter med liten buffert

# Stabilitet
MIN_HOLD_BARS      = int(os.getenv("MIN_HOLD_BARS", "1"))        # kräver 1 stängd bar efter entry innan exit
COOLDOWN_BARS      = int(os.getenv("COOLDOWN_BARS", "1"))        # skippar N barer efter exit

# Övrigt
DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]
DEFAULT_TF         = os.getenv("DEFAULT_TIMEFRAME", "1min")
LOOP_SECONDS       = int(os.getenv("LOOP_SECONDS", "5"))
MOCK_QTY_USDT      = float(os.getenv("MOCK_QTY_USDT", "30"))

MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"
LOG_FIELDS = ["timestamp","mode","symbol","side","price","qty","fee_rate","fee_cost",
              "gross_pnl","net_pnl","entry_price","exit_price","note"]

KU_PUBLIC = "https://api.kucoin.com"
def utcnow_iso() -> str: return datetime.now(timezone.utc).isoformat()

# ---------------------------- Tillstånd ----------------------------
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",         # mock | live
    "ai_mode": "neutral",
    "pending_confirm": None,
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": DEFAULT_TF,
    "engine_task": None,
    "per_symbol": {},       # symbol -> ORBState
    "pnl_today": 0.0,
}

# ---------------------------- Telegram utils ----------------------------
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

# ---------------------------- CSV / PnL ----------------------------
def write_log_row(path: str, row: Dict[str, Any]):
    new = not os.path.exists(path)
    for k in LOG_FIELDS: row.setdefault(k, "")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})

def pnl(entry_price: float, exit_price: float, qty: float, fee_rate: float):
    gross = (exit_price - entry_price) * qty
    fees  = (entry_price + exit_price) * qty * fee_rate
    net   = gross - fees
    return gross, fees, net

async def mock_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str,Any]]=None):
    fee_cost = price*qty*FEE_RATE
    row = {"timestamp": utcnow_iso(),"mode":"mock","symbol":symbol,"side":side,"price":price,"qty":qty,
           "fee_rate":FEE_RATE,"fee_cost":round(fee_cost,8),
           "gross_pnl":"","net_pnl":"","entry_price":"","exit_price":"","note":"mock"}
    if extra: row.update(extra)
    write_log_row(MOCK_LOG, row)
    await tg_send_text(chat_id, f"✅ Mock {side.upper()} {symbol} @ {price:.6f} x {qty:g}")

async def live_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str,Any]]=None):
    fee_cost = price*qty*FEE_RATE
    row = {"timestamp": utcnow_iso(),"mode":"live","symbol":symbol,"side":side,"price":price,"qty":qty,
           "fee_rate":FEE_RATE,"fee_cost":round(fee_cost,8),
           "gross_pnl":"","net_pnl":"","entry_price":"","exit_price":"","note":"live-stub"}
    if extra: row.update(extra)
    write_log_row(REAL_LOG, row)
    await tg_send_text(chat_id, f"🟢 LIVE (loggad) {side.upper()} {symbol} @ {price:.6f} x {qty:g}")

# ---------------------------- Data/Klines ----------------------------
def to_ksym(usdt: str) -> str:
    s = usdt.upper()
    if s.endswith("USDT"): return f"{s[:-4]}-USDT"
    return s.replace("/", "-").upper()

async def candles(symbol: str, timeframe: str, limit: int=120) -> pd.DataFrame:
    url = f"{KU_PUBLIC}/api/v1/market/candles"
    params = {"symbol": to_ksym(symbol), "type": timeframe}
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
    cols = ["ts","open","close","high","low","volume","turnover"]
    rows = []
    for row in data:
        rows.append(row[:7] if len(row)>=7 else row+[None])
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    for c in ["open","close","high","low","volume"]: df[c] = df[c].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df.tail(limit)

# ---------------------------- ORB-state ----------------------------
@dataclass
class ORBState:
    direction: str = "flat"    # up | down | flat
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_pos: bool = False
    entry_price: float = 0.0
    qty: float = 0.0
    highest_close: float = 0.0
    armed_be: bool = False
    bars_since_entry: int = 0
    cooldown: int = 0

def detect_dir(prev_close: float, close: float) -> str:
    if close > prev_close: return "up"
    if close < prev_close: return "down"
    return "flat"

def reset_orb(st: ORBState, hi: float, lo: float):
    st.orb_high, st.orb_low = float(hi), float(lo)
    st.in_pos = False
    st.entry_price = 0.0
    st.qty = 0.0
    st.highest_close = 0.0
    st.armed_be = False
    st.bars_since_entry = 0

# ---------------------------- Motorlogik ----------------------------
async def process_symbol(chat_id: int, symbol: str, tf: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await candles(symbol, tf, limit=90)
    if df.empty or len(df) < 5: return

    # Vi jobbar på SENAST STÄNGDA candle:
    last_closed = df.iloc[-2]
    prev_closed = df.iloc[-3]
    cur         = df.iloc[-1]   # pågående (används inte för entry/exit)

    # Ny riktning?
    last_dir = detect_dir(prev_closed["close"], last_closed["close"])

    # Cooldown
    if st.cooldown > 0:
        st.cooldown -= 1

    # Ny ORB om riktningen ändras
    if last_dir != st.direction and last_dir in ("up","down"):
        reset_orb(st, last_closed["high"], last_closed["low"])
        st.direction = last_dir

    # Range-krav
    orb_ok = (st.orb_high > 0 and st.orb_low > 0 and st.orb_low > 0
              and (st.orb_high/st.orb_low - 1.0) >= MIN_ORB_RANGE)

    # ================= Entry (bara LONG) =================
    entry_ok = (
        not st.in_pos
        and st.cooldown == 0
        and st.direction == "up"
        and orb_ok
        and last_closed["close"] > st.orb_high * (1.0 + ORB_BUFFER)  # CLOSE över ORB-high
    )

    if entry_ok:
        st.in_pos = True
        st.entry_price = float(last_closed["close"])
        st.qty = round(MOCK_QTY_USDT / st.entry_price, 6)
        st.highest_close = st.entry_price
        st.armed_be = False
        st.bars_since_entry = 0

        if state["mode"] == "mock":
            await mock_trade(chat_id, symbol, "buy", st.entry_price, st.qty)
        else:
            await live_trade(chat_id, symbol, "buy", st.entry_price, st.qty)

    # ================= Hantera position =================
    if st.in_pos:
        st.bars_since_entry += 1
        st.highest_close = max(st.highest_close, float(last_closed["close"]))

        # armera break-even + trail när +0.10% passerats
        up_pct = st.highest_close / st.entry_price - 1.0
        if not st.armed_be and up_pct >= ARM_BE_AT_PCT:
            st.armed_be = True

        # Beräkna nivåer
        tp_price   = st.entry_price * (1.0 + TP_PCT)
        sl_hard    = st.entry_price * (1.0 - STOP_PCT)

        # BE/Trail stop om armerad
        if st.armed_be:
            be_price = st.entry_price * (1.0 + (2*FEE_RATE) + BE_FEE_BUFFER)
            trail_price = st.highest_close * (1.0 - TRAIL_GAP_PCT)
            sl_working = max(be_price, trail_price)
        else:
            sl_working = sl_hard

        close_price = float(last_closed["close"])

        # Exit-regler (gäller först efter MIN_HOLD_BARS stängningar)
        can_exit = st.bars_since_entry >= MIN_HOLD_BARS

        should_tp  = can_exit and (close_price >= tp_price)
        back_in_box= can_exit and (close_price < st.orb_high)  # close tillbaka under ORB-high
        hit_sl     = can_exit and (close_price <= sl_working)

        if should_tp or hit_sl or back_in_box:
            exit_price = close_price
            qty = st.qty
            gross, fees, net = pnl(st.entry_price, exit_price, qty, FEE_RATE)
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

            state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
            emo = "📈" if net >= 0 else "📉"
            await tg_send_text(chat_id,
                f"{emo} Stängd LONG {symbol}\n"
                f"Entry: {st.entry_price:.6f}  Exit: {exit_price:.6f}\n"
                f"Qty: {qty:g}\nGross: {gross:.6f}  Avg: {(fees):.6f}\nNetto: {net:.6f}\n"
                f"PnL idag: {state['pnl_today']:.6f}")

            # reset + cooldown
            st.in_pos = False
            st.entry_price = 0.0
            st.qty = 0.0
            st.highest_close = 0.0
            st.armed_be = False
            st.bars_since_entry = 0
            st.cooldown = COOLDOWN_BARS

# ---------------------------- Motorloop ----------------------------
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks: await asyncio.gather(*tasks)
            await asyncio.sleep(LOOP_SECONDS)
        except asyncio.CancelledError:
            break
        except Exception as e:
            await tg_send_text(chat_id, f"Motorfel: {e}")
            await asyncio.sleep(2)

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
    if t and not t.done(): t.cancel()
    await tg_send_text(chat_id, "⏹️ Motor stoppad.")

# ---------------------------- Kommandon ----------------------------
async def handle_update(data: Dict[str, Any]):
    msg = data.get("message") or data.get("edited_message") or {}
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id or not text: return {"ok": True}
    if not is_admin(chat_id):
        await tg_send_text(chat_id, "⛔️ Du är inte behörig.")
        return {"ok": True}

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
            "/start_mock – starta mock (svara JA)\n"
            "/start_live – starta LIVE (svara JA)\n"
            "/engine_start – starta motorn\n"
            "/engine_stop – stoppa motorn\n"
            "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
            "/timeframe 1min|3min – byt tidsram\n"
            "/export_csv – skicka loggar")
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}  AI: {state['ai_mode']}\n"
            f"Symbols: {', '.join(state['symbols'])}\nTF: {state['timeframe']}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Min range: {MIN_ORB_RANGE:.4%}\n"
            f"SL: {STOP_PCT:.4%}  TP: {TP_PCT:.4%}  BE@{ARM_BE_AT_PCT:.4%}  Trail gap: {TRAIL_GAP_PCT:.4%}\n"
            f"Mock USDT: {MOCK_QTY_USDT:.2f}")
        return {"ok": True}

    if text.startswith("/start_mock"):
        state["pending_confirm"] = {"type":"mock"}
        await tg_send_text(chat_id, "Vill du starta MOCK-trading? Svara JA.")
        return {"ok": True}

    if text.startswith("/start_live"):
        state["pending_confirm"] = {"type":"live"}
        await tg_send_text(chat_id, "⚠️ Vill du starta LIVE-trading? Svara JA.")
        return {"ok": True}

    if text.startswith("/engine_start"):
        await start_engine(chat_id); return {"ok": True}

    if text.startswith("/engine_stop"):
        await stop_engine(chat_id);  return {"ok": True}

    if text.startswith("/symbols"):
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            await tg_send_text(chat_id, f"Aktuell lista: {', '.join(state['symbols'])}")
        else:
            raw = parts[1].replace(" ", "")
            syms = [s for s in raw.split(",") if s]
            state["symbols"] = [s.upper() for s in syms]
            await tg_send_text(chat_id, f"Symbols uppdaterade: {', '.join(state['symbols'])}")
        return {"ok": True}

    if text.startswith("/timeframe"):
        parts = text.split()
        tf = parts[1] if len(parts) > 1 else DEFAULT_TF
        state["timeframe"] = tf
        await tg_send_text(chat_id, f"Tidsram satt till: {tf}")
        return {"ok": True}

    if text.startswith("/export_csv"):
        sent = False
        for p in (MOCK_LOG, REAL_LOG):
            if os.path.exists(p):
                await tg_send_document(chat_id, p, caption=p); sent = True
        if not sent: await tg_send_text(chat_id, "Inga loggar ännu.")
        return {"ok": True}

    return {"ok": True}

# ---------------------------- FastAPI ----------------------------
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {"message":"Mp ORBbot is live!","webhook_path":WEBHOOK_PATH,
            "timeframe":state["timeframe"],"symbols":state["symbols"]}

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
# =========================================================================================
