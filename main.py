# main.py — Mp ORBbot (improved entries/exits to reduce chop)
# Nyheter:
# - Close-baserad breakout av ORB-high
# - Trendfilter (MA20 > prev MA20 och pris > MA20)
# - ATR%-filter (0.2%–2.0% default)
# - Volymfilter (vol > volSMA10)
# - Cooldown 2 candles efter exit
# - Risk: initial SL max 0.3%, BE vid +0.15%, trail 0.10% vid +0.30%, TP +0.60%
#
# Miljövariabler (Render → Environment) för finjustering:
#   TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID
#   FEE_RATE                (default 0.001 = 0.1%/sida)
#   DEFAULT_SYMBOLS         (t.ex. "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT")
#   ATR_MIN_PCT             (default 0.002 = 0.2%)
#   ATR_MAX_PCT             (default 0.02  = 2.0%)
#   MAX_LOSS                (default 0.003 = 0.30% initial SL)
#   BE_TRIGGER              (default 0.0015 = +0.15% flytta SL till BE)
#   TRAIL_TRIGGER           (default 0.003  = +0.30% börja traila)
#   TRAIL_GAP               (default 0.001  = 0.10% avstånd)
#   TAKE_PROFIT             (default 0.006  = +0.60% hård TP)
#
from __future__ import annotations
import os, csv, asyncio, math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Request

# ======= Settings =======
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID      = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()

FEE_RATE     = float(os.getenv("FEE_RATE", "0.001"))
ATR_MIN_PCT  = float(os.getenv("ATR_MIN_PCT", "0.002"))   # 0.2%
ATR_MAX_PCT  = float(os.getenv("ATR_MAX_PCT", "0.02"))    # 2.0%
MAX_LOSS     = float(os.getenv("MAX_LOSS", "0.003"))      # 0.30%
BE_TRIGGER   = float(os.getenv("BE_TRIGGER", "0.0015"))   # +0.15%
TRAIL_TRIGGER= float(os.getenv("TRAIL_TRIGGER", "0.003")) # +0.30%
TRAIL_GAP    = float(os.getenv("TRAIL_GAP", "0.001"))     # 0.10%
TAKE_PROFIT  = float(os.getenv("TAKE_PROFIT", "0.006"))   # +0.60%

DEFAULT_SYMBOLS = [s.strip().upper() for s in os.getenv(
    "DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT,LINKUSDT,XRPUSDT"
).split(",") if s.strip()]

MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

LOG_FIELDS = [
    "timestamp","mode","symbol","side","price","qty",
    "fee_rate","fee_cost","gross_pnl","net_pnl",
    "entry_price","exit_price","note"
]

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ======= Global state =======
state: Dict[str, Any] = {
    "active": False,
    "mode": "mock",
    "ai_mode": "neutral",
    "pending_confirm": None,
    "symbols": DEFAULT_SYMBOLS[:],
    "timeframe": "3min",
    "engine_task": None,
    "per_symbol": {},
    "pnl_today": 0.0,
    "keepalive": False,
    "keepalive_task": None,
}

# ======= Telegram =======
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

# ======= AI-läge =======
def ai_allows(ai_mode: str, signal_strength: float) -> bool:
    m = (ai_mode or "neutral").lower()
    if m == "aggressiv": return signal_strength >= 0.2
    if m in ("försiktig","forsiktig"): return signal_strength >= 0.7
    return signal_strength >= 0.5  # neutral

# ======= Loggar & PnL =======
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

async def mock_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "mock", "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": "mock",
    }
    if extra: row.update(extra)
    write_log_row(MOCK_LOG, row)
    await tg_send_text(chat_id, f"✅ Mock: {side.upper()} {symbol} @ {price} x {qty} (avg ~ {fee_cost:.8f})")

async def live_trade(chat_id: int, symbol: str, side: str, price: float, qty: float, extra: Optional[Dict[str, Any]]=None):
    fee_cost = price * qty * FEE_RATE
    row = {
        "timestamp": utcnow_iso(), "mode": "live", "symbol": symbol.upper(),
        "side": side.lower(), "price": price, "qty": qty,
        "fee_rate": FEE_RATE, "fee_cost": round(fee_cost, 8),
        "gross_pnl": "", "net_pnl": "", "entry_price": "", "exit_price": "",
        "note": "live (stub)",
    }
    if extra: row.update(extra)
    write_log_row(REAL_LOG, row)
    await tg_send_text(chat_id, f"🟢 LIVE (loggad): {side.upper()} {symbol} @ {price} x {qty} (avg ~ {fee_cost:.8f})")

# ======= KuCoin public candles =======
KU_PUBLIC = "https://api.kucoin.com"

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
    cols = ["ts","open","close","high","low","volume","turnover"]
    rows = []
    for row in data:
        vals = row[:7] if len(row)>=7 else row+[None]
        rows.append(vals)
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    for c in ["open","close","high","low","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df.tail(limit)

# ======= Hjälpindikatorer =======
def ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def atr(df: pd.DataFrame, n: int=14) -> pd.Series:
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

# ======= ORB =======
@dataclass
class ORBState:
    direction: str = "flat"    # "up"|"down"|"flat"
    orb_high: float = 0.0
    orb_low: float = 0.0
    in_position: bool = False
    side: str = ""
    entry_price: float = 0.0
    entry_qty: float = 0.0
    trail_stop: float = 0.0
    last_price: float = 0.0
    be_set: bool = False
    cooldown: int = 0          # antal candles att vänta före ny entry

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
    st.trail_stop = 0.0
    st.be_set = False
    # cooldown minskar bara i process-loop

def apply_trailing(st: ORBState, last_close: float):
    # break-even
    if not st.be_set and st.entry_price > 0 and last_close >= st.entry_price * (1 + BE_TRIGGER):
        st.trail_stop = max(st.trail_stop, st.entry_price)  # BE
        st.be_set = True
    # trailing efter trigger
    if st.be_set and last_close >= st.entry_price * (1 + TRAIL_TRIGGER):
        new_trail = last_close * (1 - TRAIL_GAP)
        st.trail_stop = max(st.trail_stop, new_trail)

def enter_long_ok(st: ORBState, last_close: float, last_high: float, orb_high: float, trend_ok: bool, vol_ok: bool, atr_ok: bool, cooldown_ok: bool) -> bool:
    # close-baserad breakout + filter
    if st.in_position or not cooldown_ok: return False
    if last_close <= orb_high: return False
    if not (trend_ok and vol_ok and atr_ok): return False
    return True

def stop_out_ok(st: ORBState, last_low: float) -> bool:
    if not st.in_position: return False
    # hård TP
    if st.entry_price and st.last_price >= st.entry_price * (1 + TAKE_PROFIT):
        return True
    # trail eller orb-low eller hård max-loss
    max_loss_price = st.entry_price * (1 - MAX_LOSS) if st.entry_price else 0.0
    if st.trail_stop and last_low < st.trail_stop: return True
    if last_low < st.orb_low: return True
    if max_loss_price and last_low < max_loss_price: return True
    return False

async def ensure_sym_state(symbol: str):
    if symbol not in state["per_symbol"]:
        state["per_symbol"][symbol] = ORBState()

# ======= PnL-notis =======
async def notify_closed_trade(chat_id: int, symbol: str, entry_price: float, exit_price: float, qty: float):
    gross, fees, net = compute_pnl(entry_price, exit_price, qty, FEE_RATE)
    state["pnl_today"] = float(state.get("pnl_today", 0.0)) + net
    emo = "📈" if net >= 0 else "📉"
    txt = (
        f"{emo} Stängd trade {symbol}\n"
        f"Entry: {entry_price:.6f}  Exit: {exit_price:.6f}\n"
        f"Qty: {qty:g}\n"
        f"Gross: {gross:.6f}\n"
        f"Avgifter: {fees:.6f}\n"
        f"Netto: {net:.6f}\n"
        f"PnL idag: {state['pnl_today']:.6f}"
    )
    await tg_send_text(chat_id, txt)

# ======= Motor =======
async def process_symbol(chat_id: int, symbol: str, timeframe: str):
    await ensure_sym_state(symbol)
    st: ORBState = state["per_symbol"][symbol]
    df = await fetch_candles(symbol, timeframe, limit=80)
    if df.empty or len(df) < 21:  # behöver minst 21 för MA20-slope
        return

    # indikatorer
    df["ma20"] = ma(df["close"], 20)
    df["atr14"] = atr(df, 14)
    df["atr_pct"] = df["atr14"] / df["close"]
    df["vol_sma10"] = ma(df["volume"], 10)

    prev, last = df.iloc[-2], df.iloc[-1]
    last_dir = detect_dir(prev["close"], last["close"])

    # ny ORB vid riktningsändring
    if last_dir != st.direction and last_dir in ("up","down"):
        reset_orb(st, last["high"], last["low"])
        st.direction = last_dir

    st.last_price = float(last["close"])

    # trendfilter: pris över MA20 och MA20 stiger
    trend_ok = (not math.isnan(last["ma20"])) and last["close"] > last["ma20"] and last["ma20"] > prev["ma20"]
    # volymfilter: vol > vol_sma10
    vol_ok   = (not math.isnan(last["vol_sma10"])) and last["volume"] >= last["vol_sma10"]
    # ATR%-filter
    atr_ok   = (not math.isnan(last["atr_pct"])) and (ATR_MIN_PCT <= last["atr_pct"] <= ATR_MAX_PCT)
    # cooldown
    cooldown_ok = st.cooldown <= 0

    # trailing uppdatering
    apply_trailing(st, st.last_price)

    # enkel AI-signal
    sig_strength = 0.6 if trend_ok else 0.4

    # ENTRY (close breakout av orb_high + filter)
    if enter_long_ok(st, float(last["close"]), float(last["high"]), st.orb_high, trend_ok, vol_ok, atr_ok, cooldown_ok) and ai_allows(state["ai_mode"], sig_strength):
        st.in_position = True
        st.side = "long"
        st.entry_price = st.last_price
        st.entry_qty = 0.001  # demo
        # initial SL: min(orb_low, entry*(1-MAX_LOSS))
        hard_sl = st.entry_price * (1 - MAX_LOSS)
        st.trail_stop = min(max(hard_sl, 0.0), st.entry_price)  # start under entry; blir BE/trail senare
        # logga BUY
        if state["mode"] == "mock":
            await mock_trade(chat_id, symbol, "buy", st.entry_price, st.entry_qty)
        else:
            await live_trade(chat_id, symbol, "buy", st.entry_price, st.entry_qty)

    # EXIT villkor
    if stop_out_ok(st, float(last["low"])):
        if st.in_position:
            exit_price = st.last_price
            qty = st.entry_qty or 0.001
            gross, fees, net = compute_pnl(st.entry_price, exit_price, qty, FEE_RATE)
            extra = {"entry_price": st.entry_price, "exit_price": exit_price,
                     "gross_pnl": round(gross, 8), "net_pnl": round(net, 8)}
            if state["mode"] == "mock":
                await mock_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            else:
                await live_trade(chat_id, symbol, "sell", exit_price, qty, extra=extra)
            await notify_closed_trade(chat_id, symbol, st.entry_price, exit_price, qty)
        # reset + cooldown
        st.in_position, st.side = False, ""
        st.entry_price, st.entry_qty, st.trail_stop = 0.0, 0.0, 0.0
        st.be_set = False
        st.cooldown = 2  # vänta 2 candles

    # ticka ner cooldown
    if st.cooldown > 0:
        st.cooldown -= 1

async def run_engine(chat_id: int):
    while state["active"]:
        try:
            tasks = [process_symbol(chat_id, s, state["timeframe"]) for s in state["symbols"]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(15)  # lite tätare
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

# ======= Keepalive =======
async def keepalive_loop(url: str, chat_id: int|str):
    while state.get("keepalive"):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(url)
        except Exception:
            pass
        await asyncio.sleep(12 * 60)

# ======= Backtest (dummy) =======
async def run_backtest(chat_id: int, symbol: str="BTCUSDT", period: str="3d", fee: float|None=None, out_csv: str="backtest_result.csv"):
    import random
    random.seed(42)
    fee_rate = FEE_RATE if fee is None else float(fee)
    rows = []
    for _ in range(60):
        gross = random.uniform(-0.003, 0.008)     # något bättre edge
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

# ======= Telegram-kommandon =======
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
            "/timeframe 3min – byt tidsram\n"
            "/backtest [SYMBOL] [PERIOD] [FEE]\n"
            "/export_csv – skicka loggar\n"
            "/mock_trade SYMBOL SIDE PRICE QTY – manuell mock\n"
            "/pnl – visa dagens PnL\n"
            "/reset_pnl – nollställ dagens PnL\n"
            "/keepalive_on – håll Render vaken\n"
            "/keepalive_off – stäng keepalive")
        return {"ok": True}

    if text.startswith("/status"):
        await tg_send_text(chat_id,
            f"Status: {'AKTIV' if state['active'] else 'INAKTIV'}\n"
            f"Läge: {state['mode']}\n"
            f"AI: {state['ai_mode']}\n"
            f"Avgift (per sida): {FEE_RATE:.4f}\n"
            f"Symbols: {', '.join(state['symbols'])}\n"
            f"TF: {state['timeframe']}\n"
            f"PnL idag: {state['pnl_today']:.6f}\n"
            f"Filters: ATR% {ATR_MIN_PCT*100:.2f}–{ATR_MAX_PCT*100:.2f}, BE {BE_TRIGGER*100:.2f}%, Trail {TRAIL_GAP*100:.2f}% efter {TRAIL_TRIGGER*100:.2f}%, TP {TAKE_PROFIT*100:.2f}%")
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

    if text.startswith("/backtest"):
        parts = text.split()
        symbol = parts[1] if len(parts) > 1 else "BTCUSDT"
        period = parts[2] if len(parts) > 2 else "3d"
        fee = float(parts[3]) if len(parts) > 3 else None
        path = await run_backtest(chat_id, symbol=symbol.upper(), period=period, fee=fee)
        await tg_send_document(chat_id, path, caption=f"Backtest {symbol} ({period})")
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

    if text.startswith("/mock_trade"):
        parts = text.split()
        if len(parts) < 5:
            await tg_send_text(chat_id, "Använd: /mock_trade SYMBOL SIDE PRICE QTY")
            return {"ok": True}
        _, symbol, side, price, qty = parts[:5]
        await mock_trade(chat_id, symbol, side, float(price), float(qty))
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

# ======= FastAPI =======
app = FastAPI()
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else "/webhook-not-set"

@app.get("/")
async def root():
    return {"message": "Mp ORBbot is live!", "webhook_path": WEBHOOK_PATH}

@app.post(WEBHOOK_PATH)
async def webhook(req: Request):
    data = await req.json()
    return await handle_update(data)
