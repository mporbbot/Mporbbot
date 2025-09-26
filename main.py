# main.py
# -----------------------------------------------------------
# Mp ORBbot – Komplett FastAPI-baserad Telegram-webhook
# - Webhook utan token i URL: verifiering via hemlig header
# - Kommandon: /help /status /set_ai /set_entry /backtest /export_csv
#              /mock_trade /start /start_live /stop
# - ORB-strategi (3-min): ORB = första gröna efter röd; Entry CLOSE/TICK
# - SL = ORB-lägsta (LONG), re-arm varje röd->grön; trailing efter +0.1 %
# - KuCoin historik (publik endpoint) + fallback syntetisk data
# - CSV-loggar: mock_trade_log.csv, real_trade_log.csv
# -----------------------------------------------------------

import os
import io
import csv
import math
import time
import json
import uuid
import asyncio
import logging
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Header, Response
from fastapi.responses import JSONResponse

# ---------------------------
# Konfiguration
# ---------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")  # Sätt i Render
TG_SECRET = os.getenv("TG_SECRET", "")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

if not BOT_TOKEN:
    raise RuntimeError("Saknar env var BOT_TOKEN. Sätt BOT_TOKEN i Render.")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mp-orbbot")

# Standardinställningar
DEFAULT_AI_MODE = "neutral"  # 'aggressiv' | 'försiktig' | 'neutral'
DEFAULT_ENTRY_MODE = "CLOSE" # 'CLOSE' | 'TICK'
DEFAULT_IS_MOCK = True       # Botten startar i mock-läge
TRAIL_ACTIVATE_PCT = 0.001   # +0.1% aktiverar trailing
TRAIL_GAP_PCT = 0.001        # trailar 0.1% under högsta pris efter aktivering

# Dina fem valda symboler (KuCoin-format och Telegram/kommandon hantering)
DEFAULT_SYMBOLS = ["LINKUSDT", "XRPUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT"]

# CSV-filer
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

# ---------------------------
# States
# ---------------------------

@dataclass
class SessionState:
    ai_mode: str = DEFAULT_AI_MODE
    entry_mode: str = DEFAULT_ENTRY_MODE
    mock_mode: bool = DEFAULT_IS_MOCK
    trading_enabled: bool = False  # aktiveras via /start eller /start_live (med JA)
    awaiting_confirmation: Optional[str] = None  # "start_mock" | "start_live" | None
    last_trade: Optional[str] = None
    today_pnl: float = 0.0
    trades_today: int = 0
    last_pnl_reset_date: str = field(default_factory=lambda: dt.date.today().isoformat())

# Per-chat state (enkelt i minnet)
CHAT_STATE: Dict[int, SessionState] = {}

def get_state(chat_id: int) -> SessionState:
    st = CHAT_STATE.get(chat_id)
    if not st:
        st = SessionState()
        CHAT_STATE[chat_id] = st
    # reset daglig statistik vid datumbyte
    today = dt.date.today().isoformat()
    if st.last_pnl_reset_date != today:
        st.last_pnl_reset_date = today
        st.today_pnl = 0.0
        st.trades_today = 0
    return st

# ---------------------------
# Telegram helpers
# ---------------------------

async def tg_send_message(chat_id: int | str, text: str, reply_to_message_id: Optional[int] = None) -> None:
    payload = {"chat_id": chat_id, "text": text}
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{API_BASE}/sendMessage", json=payload)
        if r.status_code != 200:
            log.error("sendMessage fel: %s", r.text)

async def tg_send_document(chat_id: int | str, filename: str, content: bytes, caption: Optional[str] = None) -> None:
    files = {"document": (filename, content)}
    data = {"chat_id": str(chat_id)}
    if caption:
        data["caption"] = caption
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{API_BASE}/sendDocument", data=data, files=files)
        if r.status_code != 200:
            log.error("sendDocument fel: %s", r.text)

def parse_command(text: str) -> tuple[str, list[str]]:
    parts = (text or "").strip().split()
    if not parts:
        return "", []
    return parts[0].lower(), parts[1:]

# ---------------------------
# CSV loggning
# ---------------------------

def ensure_csv(path: str, headers: List[str]) -> None:
    exists = os.path.exists(path)
    if not exists:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_trade(path: str, row: List[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ---------------------------
# KuCoin OHLCV (publik)
# ---------------------------

async def fetch_kucoin_ohlcv(symbol: str, period_spec: str) -> List[Dict[str, Any]]:
    """
    Hämtar 3-min candles från KuCoin publik endpoint: /api/v1/market/candles
    KuCoin symbolformat är typ 'BTC-USDT'. Våra kommandon ger 'BTCUSDT' -> konvertera.
    Returnerar lista av dict: {ts, o, h, l, c, v}
    """
    # Konvertera
    sym = symbol.upper()
    if sym.endswith("USDT"):
        sym = sym[:-4] + "-USDT"
    # period_spec t.ex. '3d', '1d', '12h'
    now = int(time.time())
    seconds = parse_period_to_seconds(period_spec)
    start_at = now - seconds
    params = {
        "symbol": sym,
        "type": "3min",
        "startAt": start_at,
        "endAt": now
    }
    url = "https://api.kucoin.com/api/v1/market/candles"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            # KuCoin returnerar listor i omvänd ordning (nyast först): [ time, open, close, high, low, volume, turnover ]
            raw = data.get("data", [])
            out: List[Dict[str, Any]] = []
            for item in reversed(raw):  # gör dem stigande i tiden
                t, o, c, h, l, v, _turnover = item
                out.append({
                    "ts": int(t),
                    "o": float(o),
                    "h": float(h),
                    "l": float(l),
                    "c": float(c),
                    "v": float(v),
                })
            return out
    except Exception as e:
        log.warning("KuCoin fetch misslyckades (%s) – använder syntetisk data.", e)
        return synth_data_3min(seconds // 180)

def parse_period_to_seconds(spec: str) -> int:
    spec = spec.strip().lower()
    if spec.endswith("d"):
        return int(spec[:-1]) * 86400
    if spec.endswith("h"):
        return int(spec[:-1]) * 3600
    if spec.endswith("m"):
        return int(spec[:-1]) * 60
    # default dagar
    return int(spec) * 86400

def synth_data_3min(n_candles: int) -> List[Dict[str, Any]]:
    """Syntetiskt 3-min OHLCV om KuCoin inte kan nås."""
    out = []
    base = 100.0
    ts = int(time.time()) - n_candles * 180
    for i in range(n_candles):
        drift = math.sin(i / 50.0) * 0.2
        noise = (math.sin(i / 7.0) + math.cos(i / 9.0)) * 0.05
        close = max(1e-6, base + drift + noise)
        high = close * (1 + 0.002 + abs(math.sin(i / 13.0))*0.001)
        low = close * (1 - 0.002 - abs(math.cos(i / 17.0))*0.001)
        open_ = (close + low) / 2
        vol = 100 + 20 * math.sin(i / 5.0)
        out.append({"ts": ts, "o": open_, "h": high, "l": low, "c": close, "v": vol})
        base = close
        ts += 180
    return out

# ---------------------------
# ORB-backtest
# ---------------------------

@dataclass
class TradeResult:
    symbol: str
    entry_time: int
    entry_px: float
    exit_time: int
    exit_px: float
    pnl_abs: float
    pnl_pct: float
    fee_cost: float

def is_green(cndl: Dict[str, float]) -> bool:
    return cndl["c"] > cndl["o"]

def is_red(cndl: Dict[str, float]) -> bool:
    return cndl["c"] < cndl["o"]

def run_orb_backtest(ohlcv: List[Dict[str, Any]], entry_mode: str, fee_rate: float = 0.001) -> List[TradeResult]:
    """
    ORB: första gröna efter röd definierar ORB-range med den gröna candlens high/low.
    LONG-entry:
      - CLOSE: vid candle close som stänger över ORB-high
      - TICK: om högsta pris (h) bryter ORB-high
    Initial SL: ORB-low
    Re-arm: varje gång vi ser röd -> grön, skapas ny ORB (om inte i position)
    Trailing: aktiveras efter +0.1% vinst; stop = max(stop, max_price*(1-TRAIL_GAP_PCT))
    Antag en "1 enhet" position för procent-PnL; fee appliceras på både köp och sälj.
    """
    results: List[TradeResult] = []
    in_pos = False
    orb_high = None
    orb_low = None
    entry_px = None
    entry_ts = None
    stop_px = None
    trailing_active = False
    max_price = None

    prev = None
    for i, c in enumerate(ohlcv):
        if prev and (is_red(prev) and is_green(c)) and not in_pos:
            # definiera ny ORB
            orb_high = c["h"]
            orb_low = c["l"]

        # Entrylogik
        if not in_pos and orb_high is not None and orb_low is not None:
            if entry_mode.upper() == "CLOSE":
                # inträde om stängning över orb_high
                if c["c"] > orb_high:
                    in_pos = True
                    entry_px = c["c"]
                    entry_ts = c["ts"]
                    stop_px = orb_low
                    trailing_active = False
                    max_price = entry_px
            else:  # TICK
                if c["h"] > orb_high:
                    in_pos = True
                    # anta inträde på orb_high som "break"
                    entry_px = orb_high
                    entry_ts = c["ts"]
                    stop_px = orb_low
                    trailing_active = False
                    max_price = entry_px

        if in_pos:
            # uppdatera trailing
            max_price = max(max_price, c["h"])
            if not trailing_active and max_price >= entry_px * (1 + TRAIL_ACTIVATE_PCT):
                trailing_active = True
            if trailing_active:
                trail_stop = max_price * (1 - TRAIL_GAP_PCT)
                stop_px = max(stop_px, trail_stop)

            # kontrollera exit på stop
            if c["l"] <= stop_px:
                exit_px = stop_px
                exit_ts = c["ts"]
                # räkna PnL för 1 enhet minus fees på köp & sälj
                gross = exit_px - entry_px
                fees = (entry_px + exit_px) * fee_rate
                pnl_abs = gross - fees
                pnl_pct = pnl_abs / entry_px
                results.append(TradeResult(
                    symbol="",
                    entry_time=entry_ts,
                    entry_px=entry_px,
                    exit_time=exit_ts,
                    exit_px=exit_px,
                    pnl_abs=pnl_abs,
                    pnl_pct=pnl_pct,
                    fee_cost=fees
                ))
                in_pos = False
                orb_high = None
                orb_low = None
                entry_px = None
                entry_ts = None
                stop_px = None
                trailing_active = False
                max_price = None
        prev = c

    # stäng eventuell öppen position på sista close
    if in_pos and entry_px is not None:
        last = ohlcv[-1]
        exit_px = last["c"]
        exit_ts = last["ts"]
        gross = exit_px - entry_px
        fees = (entry_px + exit_px) * fee_rate
        pnl_abs = gross - fees
        pnl_pct = pnl_abs / entry_px
        results.append(TradeResult(
            symbol="",
            entry_time=entry_ts,
            entry_px=entry_px,
            exit_time=exit_ts,
            exit_px=exit_px,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
            fee_cost=fees
        ))
    return results

def results_to_csv_bytes(symbol: str, results: List[TradeResult]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["symbol", "entry_time", "entry_iso", "entry_px", "exit_time", "exit_iso", "exit_px", "pnl_abs", "pnl_pct", "fee_cost"])
    for r in results:
        writer.writerow([
            symbol,
            r.entry_time,
            dt.datetime.utcfromtimestamp(r.entry_time).isoformat(),
            f"{r.entry_px:.8f}",
            r.exit_time,
            dt.datetime.utcfromtimestamp(r.exit_time).isoformat(),
            f"{r.exit_px:.8f}",
            f"{r.pnl_abs:.8f}",
            f"{r.pnl_pct:.6f}",
            f"{r.fee_cost:.8f}",
        ])
    return buf.getvalue().encode("utf-8")

# ---------------------------
# Kommandon
# ---------------------------

async def cmd_help(chat_id: int, mid: int) -> None:
    await tg_send_message(chat_id,
        "📖 Kommandon:\n"
        "/start – starta MOCK-trading (kräver bekräftelse: JA)\n"
        "/start_live – starta LIVE-trading (kräver bekräftelse: JA)\n"
        "/stop – stoppa trading (kommandon funkar ändå)\n"
        "/status – visa läge, PnL och trades\n"
        "/set_ai <aggressiv|försiktig|neutral>\n"
        "/set_entry <CLOSE|TICK> – entry-logik för ORB\n"
        "/backtest <SYMBOL|all> <TID> [avgift] – ex: /backtest btcusdt 3d 0.001\n"
        "/export_csv – skicka mock_trade_log.csv och real_trade_log.csv (om finns)\n"
        "/mock_trade <SYMBOL> – logga en mocktrade (30 USDT)"
    , reply_to_message_id=mid)

async def cmd_status(chat_id: int, mid: int) -> None:
    st = get_state(chat_id)
    await tg_send_message(chat_id,
        "📊 Status:\n"
        f"AI-läge: {st.ai_mode}\n"
        f"Entry-läge: {st.entry_mode}\n"
        f"Mock-läge: {'PÅ' if st.mock_mode else 'AV'}\n"
        f"Trading: {'AKTIV' if st.trading_enabled else 'STOPPAD'}\n"
        f"Senaste trade: {st.last_trade or '–'}\n"
        f"Dagens PnL: {st.today_pnl:.2f} USDT\n"
        f"Antal trades idag: {st.trades_today}"
    , reply_to_message_id=mid)

async def cmd_set_ai(chat_id: int, mid: int, args: List[str]) -> None:
    st = get_state(chat_id)
    if not args:
        await tg_send_message(chat_id, "Användning: /set_ai <aggressiv|försiktig|neutral>", reply_to_message_id=mid); return
    mode = args[0].lower()
    if mode not in {"aggressiv", "försiktig", "neutral"}:
        await tg_send_message(chat_id, "Ogiltigt läge. Välj: aggressiv, försiktig eller neutral.", reply_to_message_id=mid); return
    st.ai_mode = mode
    await tg_send_message(chat_id, f"✅ AI-läge satt till: {mode}", reply_to_message_id=mid)

async def cmd_set_entry(chat_id: int, mid: int, args: List[str]) -> None:
    st = get_state(chat_id)
    if not args:
        await tg_send_message(chat_id, "Användning: /set_entry <CLOSE|TICK>", reply_to_message_id=mid); return
    mode = args[0].upper()
    if mode not in {"CLOSE", "TICK"}:
        await tg_send_message(chat_id, "Ogiltigt läge. Välj CLOSE eller TICK.", reply_to_message_id=mid); return
    st.entry_mode = mode
    await tg_send_message(chat_id, f"✅ ENTRY_MODE satt till: {mode}", reply_to_message_id=mid)

async def cmd_start(chat_id: int, mid: int) -> None:
    st = get_state(chat_id)
    st.awaiting_confirmation = "start_mock"
    await tg_send_message(chat_id, "⚠️ Bekräfta att starta MOCK-trading genom att svara exakt: JA", reply_to_message_id=mid)

async def cmd_start_live(chat_id: int, mid: int) -> None:
    st = get_state(chat_id)
    st.awaiting_confirmation = "start_live"
    await tg_send_message(chat_id, "⚠️ Bekräfta att starta LIVE-trading genom att svara exakt: JA", reply_to_message_id=mid)

async def cmd_stop(chat_id: int, mid: int) -> None:
    st = get_state(chat_id)
    st.trading_enabled = False
    await tg_send_message(chat_id, "⏹️ Trading stoppad. Kommandon fungerar fortfarande.", reply_to_message_id=mid)

async def handle_confirmation(chat_id: int, mid: int, text: str) -> bool:
    st = get_state(chat_id)
    if st.awaiting_confirmation and text.strip().upper() == "JA":
        if st.awaiting_confirmation == "start_mock":
            st.mock_mode = True
            st.trading_enabled = True
            st.awaiting_confirmation = None
            await tg_send_message(chat_id, "✅ MOCK-trading AKTIV.", reply_to_message_id=mid)
            return True
        if st.awaiting_confirmation == "start_live":
            st.mock_mode = False
            st.trading_enabled = True
            st.awaiting_confirmation = None
            await tg_send_message(chat_id, "✅ LIVE-trading AKTIV. Var försiktig!", reply_to_message_id=mid)
            return True
    return False

async def cmd_mock_trade(chat_id: int, mid: int, args: List[str]) -> None:
    st = get_state(chat_id)
    if not args:
        await tg_send_message(chat_id, "Användning: /mock_trade <SYMBOL>", reply_to_message_id=mid); return
    symbol = args[0].upper()
    qty_usdt = 30.0  # enligt din spec
    price = 1.0      # placeholder (ingen marknadsdata i detta kommando)
    ensure_csv(MOCK_LOG, ["ts","chat_id","symbol","side","qty_usdt","price","note"])
    append_trade(MOCK_LOG, [int(time.time()), chat_id, symbol, "BUY", qty_usdt, price, "manual mock_trade"])
    st.last_trade = f"MOCK BUY {symbol} {qty_usdt} USDT"
    st.trades_today += 1
    await tg_send_message(chat_id, f"🧪 Mocktrade loggad: {st.last_trade}", reply_to_message_id=mid)

async def cmd_export_csv(chat_id: int, mid: int) -> None:
    sent_any = False
    if os.path.exists(MOCK_LOG):
        with open(MOCK_LOG, "rb") as f:
            await tg_send_document(chat_id, "mock_trade_log.csv", f.read(), caption="📄 mock_trade_log.csv")
            sent_any = True
    if os.path.exists(REAL_LOG):
        with open(REAL_LOG, "rb") as f:
            await tg_send_document(chat_id, "real_trade_log.csv", f.read(), caption="📄 real_trade_log.csv")
            sent_any = True
    if not sent_any:
        await tg_send_message(chat_id, "Inga CSV-loggar hittades ännu.", reply_to_message_id=mid)

async def cmd_backtest(chat_id: int, mid: int, args: List[str]) -> None:
    st = get_state(chat_id)
    if len(args) < 2:
        await tg_send_message(chat_id, "Användning: /backtest <SYMBOL|all> <TID> [avgift]\nEx: /backtest btcusdt 3d 0.001", reply_to_message_id=mid); return
    target = args[0].upper()
    period = args[1].lower()
    fee = float(args[2]) if len(args) >= 3 else 0.001

    symbols = DEFAULT_SYMBOLS if target == "ALL" else [target]
    await tg_send_message(chat_id, f"⏱️ Kör backtest {', '.join(symbols)} {period} fee={fee:.4f} …", reply_to_message_id=mid)

    total_pnl = 0.0
    combined_zip_buf = io.BytesIO()
    # enkel "zip" via flera dokumentuppladdningar – en fil per symbol
    for sym in symbols:
        ohlcv = await fetch_kucoin_ohlcv(sym, period)
        res = run_orb_backtest(ohlcv, st.entry_mode, fee)
        # sätt symbol i result
        for r in res:
            r.symbol = sym
        csv_bytes = results_to_csv_bytes(sym, res)
        fn = f"backtest_{sym}_{period}.csv"
        await tg_send_document(chat_id, fn, csv_bytes, caption=f"✅ Backtest {sym} klart.")
        # summera PnL (antag 1 enhet)
        total_pnl += sum(r.pnl_abs for r in res)

    await tg_send_message(chat_id, f"📈 Summerad PnL (≈1 enhet vardera): {total_pnl:.6f}", reply_to_message_id=mid)

# ---------------------------
# FastAPI-app
# ---------------------------

app = FastAPI(title="Mp ORBbot")

@app.get("/")
async def health():
    return {"ok": True, "service": "mp-orbbot", "webhook": "/webhook"}

@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)
):
    # 1) Verifiera hemlig header
    if not TG_SECRET:
        log.warning("TG_SECRET saknas – avvisar inkommande.")
        return Response(status_code=500)
    if x_telegram_bot_api_secret_token != TG_SECRET:
        log.warning("Felaktig secret header – 401.")
        return Response(status_code=401)

    # 2) Läs update
    try:
        update: Dict[str, Any] = await request.json()
    except Exception as e:
        log.error("Kunde inte läsa JSON: %s", e)
        return Response(status_code=400)

    message = update.get("message") or update.get("edited_message")
    if not message:
        return JSONResponse({"status": "ignored"})

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    text = message.get("text") or ""

    if not chat_id or not message_id:
        return JSONResponse({"status": "ignored"})

    st = get_state(chat_id)

    # 3) Bekräftelsehantering (JA)
    if st.awaiting_confirmation:
        handled = await handle_confirmation(chat_id, message_id, text)
        if handled:
            return JSONResponse({"status": "ok"})

    # 4) Kommandoparsning
    cmd, args = parse_command(text)

    try:
        if cmd in ("/help",):
            await cmd_help(chat_id, message_id)

        elif cmd in ("/status",):
            await cmd_status(chat_id, message_id)

        elif cmd in ("/set_ai",):
            await cmd_set_ai(chat_id, message_id, args)

        elif cmd in ("/set_entry",):
            await cmd_set_entry(chat_id, message_id, args)

        elif cmd in ("/start", "/start@Mp_ORBbot"):
            await cmd_start(chat_id, message_id)

        elif cmd in ("/start_live",):
            await cmd_start_live(chat_id, message_id)

        elif cmd in ("/stop", "/stop@Mp_ORBbot"):
            await cmd_stop(chat_id, message_id)

        elif cmd in ("/mock_trade",):
            await cmd_mock_trade(chat_id, message_id, args)

        elif cmd in ("/export_csv",):
            await cmd_export_csv(chat_id, message_id)

        elif cmd in ("/backtest",):
            await cmd_backtest(chat_id, message_id, args)

        else:
            # Vanligt meddelande – håll enkel
            await tg_send_message(chat_id, "🤖 Skriv /help för kommandon.", reply_to_message_id=message_id)

    except Exception as e:
        log.exception("Fel i kommandorouter: %s", e)
        await tg_send_message(chat_id, "❌ Ett fel uppstod när kommandot kördes.", reply_to_message_id=message_id)

    return JSONResponse({"status": "ok"})

# ---------------------------
# Lokal körning
# ---------------------------
# uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
