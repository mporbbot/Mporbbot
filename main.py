# ========================= main.py (Mp ORBbot — clean full) =========================
from __future__ import annotations
import os, csv, asyncio, math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ------------------------- Konfig / miljö -------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()  # valfri: lås botten till dig

# Avgift per sida (0.001 = 0.1%). Total tur & retur ≈ 2*FEE_RATE.
FEE_RATE           = float(os.getenv("FEE_RATE", "0.001"))

# ORB-parametrar
ORB_BUFFER         = float(os.getenv("ORB_BUFFER", "0.0004"))   # 0.04% buffert utöver ORB-linje
MIN_ORB_RANGE      = float(os.getenv("MIN_ORB_RANGE", "0.0008"))# min 0.08% range för att gälla

# TP & trailing
TP_FIXED_PCT       = float(os.getenv("TP_FIXED_PCT", "0.0012")) # +0.12% fast take-profit
TRAIL_ARM_PCT      = float(os.getenv("TRAIL_ARM_PCT", "0.0010"))# aktivera trailing om >= +0.10%
TRAIL_GAP_PCT      = float(os.getenv("TRAIL_GAP_PCT", "0.0006"))# följ priset med 0.06% slack
TRAIL_BE_EXTRA     = float(os.getenv("TRAIL_BE_EXTRA", "0.00005")) # extra buffert vid BE

# Övrigt
DEFAULT_SYMBOLS    = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]
DEFAULT_TF         = os.getenv("DEFAULT_TIMEFRAME", "3min")     # 1min/3min/5min osv
LOOP_SECONDS       = int(os.getenv("LOOP_SECONDS", "10"))       # motor-intervall
MOCK_QTY_USDT      = float(os.getenv("MOCK_QTY_USDT", "30"))    # mock: ungefärlig USDT per trade

MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"
LOG_FIELDS = ["timestamp","mode","symbol","side","price","qty","fee_rate","fee_cost",
              "gross_pnl","net_pnl","entry_price","exit_price","note"]

KU_PUBLIC = "https://api.kucoin.com"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ------------------------- Globalt tillstånd -------------------------
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",                 # "mock" | "live"
    "ai_mode": "neutral",
    "pending_confirm": None,        # {"type":"mock"|"live"}
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": DEFAULT_TF,
    "engine_task": None,
    "per_symbol": {},               # symbol -> ORBState
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
}

# ------------------------- Telegram utils -------------------------
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

# ------------------------- CSV & PnL -------------------------
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
    fees  = (entry_price + exit_price) * qty * fee_rate  # båda håll
    net   = gross - fees
    return gross, fees, net

# ------------------------- KuCoin publika candles -------------------------
def to_kucoin_symbol(usdt_sym: str) -> str:
    usdt_sym = usdt_sym.upper()
    if usdt_sym.endswith("USDT"):
        return f"{usdt_sym[:-4]}-USDT"
    return usdt_sym.replace("/", "-").upper()

async def fetch_candles(symbol: str, timeframe: str="3min", limit: int=80) -> pd.DataFrame:
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

# ------------------------- ORB‐logik -------------------------
@dataclass
class ORBState:
    direction: str = "flat"      # "up"|"down"|"flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    side: str = ""               # "long"|"short"
    entry_price: float = 0.0
    entry_qty: float = 0.0
    stop: float = 0.0            # aktuell stop (kan trailas)
    trail_armed: bool = False    # om vi har flyttat till BE och ska traila vidare

def detect_dir(prev_close: float, close: float) -> str:
    if close > prev_close: return "up"
    if close < prev_close: return "down"
    return "flat"

def reset_orb(st: ORBState, hi: float, lo: float):
    st.orb_high, st.orb_low = float(hi), float(lo)
    st.in_position = False
    st.side = ""
    st.entry_price = 0.0
    st.entry_qty = 0.0
    st.stop = 0.0
    st.trail_armed = False

# ------------------------- Trade helpers -------------------------
def qty_from_usdt(price: float, usdt_amount: float) -> float:
    if price <= 0: return 0.0
    # Runda till 6 decimaler – duger för mock/logg.
    return round(usdt_amount / price, 6)

async def log_trade(chat_id: int, mode: str, symbol: str, side: str,
                    price: float, qty: float, note: str = "",
                    entry_price: float = 0.0, exit_price: float = 0.0):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": mode, "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "", "note": note,
    }
    if entry_price and exit_price:
        gross, fees, net = compute_pnl(entry_price, exit_price, qty, FEE_RATE)
        row["gross_pnl"] = round(gross, 8)
        row["net_pnl"]   = round(net, 8)
        row["entry_price"] = entry_price
        row["exit_price"]  = exit_price
    path = MOCK_LOG if mode == "mock" else REAL_LOG
    write_log_row(path, row)

    if entry_price and exit_price:
        emo = "📈" if row["net_pnl"] >= 0 else "📉"
        await tg_send_text(chat_id,
            f"{emo} Stängd {side.upper()} {symbol}\n"
            f"Entry: {entry_price:.6f}  Exit: {exit_price:.6f}\n"
            f"Qty: {qty:g}\nGross: {row['gross_pnl']:.6f}\n"
            f"Avg: {row['fee_cost']:.6f}  Netto: {row['net_pnl']:.6f}")
        state["pnl_today"] = float(state.get("pnl_today", 0.0)) + row["net_pnl"]
    else:
        await tg_send_text(chat_id, f"✅ {mode.upper()} {side.upper()} {symbol} @ {price} x {qty}  (avg ~ {fee_cost:.8f})")

# ------------------------- Motor per symbol -------------------------
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()
    st: ORBState = state["per_symbol"][symbol]

    df = await fetch_candles(symbol, timeframe, limit=60)
    if df.empty or len(df) < 3:
        return

    prev, last = df.iloc[-2], df.iloc[-1]
    last_close = float(last["close"])
    last_high  = float(last["high"])
    last_low   = float(last["low"])

    # 1) Ny ORB vid riktningsändring (enkel definition)
    curr_dir = detect_dir(prev["close"], last["close"])
    if curr_dir != st.direction and curr_dir in ("up","down"):
        reset_orb(st, last["high"], last["low"])
        st.direction = curr_dir

    # 2) Om ingen ORB än, sätt den till senaste candle high/low
    if st.orb_high == 0.0 or st.orb_low == 0.0:
        reset_orb(st, last["high"], last["low"])

    # 3) Entrylogik (long/short)
    orb_range_ok = (st.orb_low > 0 and (st.orb_high / st.orb_low - 1.0) >= MIN_ORB_RANGE)

    # LONG-entry: bryt över ORB-high * (1+buffer)
    long_trigger = (not st.in_position and
                    last_close > st.orb_high * (1.0 + ORB_BUFFER) and
                    st.direction == "up" and orb_range_ok)

    # SHORT-entry: bryt under ORB-low * (1-buffer)
    short_trigger = (not st.in_position and
                     last_close < st.orb_low * (1.0 - ORB_BUFFER) and
                     st.direction == "down" and orb_range_ok)

    if long_trigger or short_trigger:
        st.in_position = True
        st.side = "long" if long_trigger else "short"
        st.entry_price = last_close
        st.entry_qty   = qty_from_usdt(st.entry_price, MOCK_QTY_USDT)
        # Initial stop: motsatt ORB-linje
        st.stop = st.orb_low if st.side == "long" else st.orb_high
        st.trail_armed = False

        mode = state["mode"]
        await log_trade(chat_id, mode, symbol, "buy" if st.side=="long" else "sell",
                        st.entry_price, st.entry_qty, note="entry")

    # 4) Exit-logik om i position
    if st.in_position:
        # a) Fast take-profit
        if st.side == "long":
            if last_high >= st.entry_price * (1.0 + TP_FIXED_PCT):
                mode = state["mode"]
                await log_trade(chat_id, mode, symbol, "sell", last_close, st.entry_qty,
                                note="tp_fixed",
                                entry_price=st.entry_price, exit_price=last_close)
                st.in_position = False; st.side=""; st.entry_price=0.0; st.entry_qty=0.0; st.stop=0.0; st.trail_armed=False
                return
        else:  # short
            if last_low <= st.entry_price * (1.0 - TP_FIXED_PCT):
                mode = state["mode"]
                await log_trade(chat_id, mode, symbol, "buy", last_close, st.entry_qty,
                                note="tp_fixed",
                                entry_price=st.entry_price, exit_price=last_close)
                st.in_position = False; st.side=""; st.entry_price=0.0; st.entry_qty=0.0; st.stop=0.0; st.trail_armed=False
                return

        # b) Trailing: när orealiserad vinst ≥ TRAIL_ARM_PCT flytta till break-even + avgifter
        #    och traila därefter med TRAIL_GAP_PCT
        if st.side == "long":
            profit_pct = (last_close / st.entry_price) - 1.0
            if (not st.trail_armed) and profit_pct >= TRAIL_ARM_PCT:
                be_plus_fees = st.entry_price * (1 + 2*FEE_RATE + TRAIL_BE_EXTRA)
                st.stop = max(st.stop, be_plus_fees)
                st.trail_armed = True
            if st.trail_armed:
                # följ priset men lämna TRAIL_GAP_PCT
                st.stop = max(st.stop, last_close * (1.0 - TRAIL_GAP_PCT))
        else:
            profit_pct = 1.0 - (last_close / st.entry_price)
            if (not st.trail_armed) and profit_pct >= TRAIL_ARM_PCT:
                be_plus_fees = st.entry_price * (1 - (2*FEE_RATE + TRAIL_BE_EXTRA))
                st.stop = min(st.stop, be_plus_fees)
                st.trail_armed = True
            if st.trail_armed:
                st.stop = min(st.stop, last_close * (1.0 + TRAIL_GAP_PCT))

        # c) Stop-utlösning (inkl. “vänd direkt till orb-linjen”)
        stop_hit = (last_low <= st.stop) if st.side=="long" else (last_high >= st.stop)
        orb_flip = (last_low <= st.orb_low) if st.side=="long" else (last_high >= st.orb_high)
        if stop_hit or orb_flip:
            mode = state["mode"]
            side_out = "sell" if st.side=="long" else "buy"
            await log_trade(chat_id, mode, symbol, side_out, last_close, st.entry_qty,
                            note="stop" if stop_hit else "orb_flip",
                            entry_price=st.entry_price, exit_price=last_close)
            st.in_position = False; st.side=""; st.entry_price=0.0; st.entry_qty=0.0; st.stop=0.0; st.trail_armed=False

# ------------------------- Motorloop -------------------------
async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(LOOP_SECONDS)
        except asyncio.CancelledError:
            break
        except Exception as e:
            await tg_send_text(chat_id, f"Motorfel: {e}")
            await asyncio.sleep(3)

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

# ------------------------- Keepalive -------------------------
async def keepalive_loop(url: str, chat_id: int|str):
    while state.get("keepalive"):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(url)
        except Exception:
            pass
        await asyncio.sleep(12 * 60)

# ------------------------- Telegram / kommandon -------------------------
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

    # Bekräftelser för start_mock / start_live
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
            "/start_live – starta live (svara JA)\n"
            "/engine_start – starta motor\n"
            "/engine_stop – stoppa motor\n"
            "/symbols BTCUSDT,ETHUSDT,... – byt lista\n"
            "/timeframe 1min|3min|5min – byt TF\n"
            "/export_csv – skicka loggar\n"
            "/pnl – visa dagens PnL\n"
            "/reset_pnl – nollställ PnL idag\n"
            "/keepalive_on | /keepalive_off")
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}   AI: {state['ai_mode']}\n"
            f"Avgift per sida: {FEE_RATE:.4f}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}   Loop: {LOOP_SECONDS}s\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"ORB buffer: {ORB_BUFFER:.4%}  Min range: {MIN_ORB_RANGE:.4%}\n"
            f"TP: {TP_FIXED_PCT:.4%}  Trail arm: {TRAIL_ARM_PCT:.4%}  Gap: {TRAIL_GAP_PCT:.4%}\n"
            f"Mock USDT: {MOCK_QTY_USDT:.2f}")
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
        tf = parts[1] if len(parts) > 1 else DEFAULT_TF
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
        if task and not task.done():
            task.cancel()
        await tg_send_text(chat_id, "Keepalive AV.")
        return {"ok": True}

    return {"ok": True}

# ------------------------- FastAPI -------------------------
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {
        "message": "Mp ORBbot is live!",
        "webhook_path": WEBHOOK_PATH,
        "symbols": state["symbols"],
        "timeframe": state["timeframe"],
    }

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
# ========================= end main.py =========================
