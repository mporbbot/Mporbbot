import os
import asyncio
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone
from io import BytesIO
import csv

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse

from telegram import Update, Bot, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

# ------------------------------------------------------------
# Konfiguration via miljövariabler (med säkra defaults)
# ------------------------------------------------------------
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "5397586616") or "5397586616")
SYMBOLS_ENV = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").strip()
TF = os.getenv("TF", "1min").strip()            # '1min' eller '3min'
ENTRY_MODE = os.getenv("ENTRY_MODE", "tick").strip().lower()  # 'tick' eller 'close'
LIVE = int(os.getenv("LIVE", "0") or "0")       # 0=mock, 1=live (förberett)
ORDER_SIZE_USDT = float(os.getenv("ORDER_SIZE_USDT", "25") or "25")

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "").strip()
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "").strip()
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "").strip()

# Render ger PORT i miljön; uvicorn använder den via CLI
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()  # kan lämnas blank (detekteras av Render)
# ------------------------------------------------------------

LOG = logging.getLogger("orb_v33")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

KUCOIN_BASE = "https://api.kucoin.com"
# KuCoin candle-typ strängar: '1min', '3min', '5min' etc.
VALID_TF = {"1min", "3min"}

def to_kucoin_symbol(symbol: str) -> str:
    s = symbol.upper().replace("-", "").replace("/", "")
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USDT"
    return symbol.replace("/", "-")

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ------------------------------------------------------------
# ORB/Trading-tillstånd per symbol
# ------------------------------------------------------------
class Position:
    def __init__(self, entry_price: float, qty: float, entry_mode: str, orb_high: float, orb_low: float):
        self.side = "LONG"
        self.entry_price = entry_price
        self.qty = qty
        self.entry_time = now_utc()
        self.stop = orb_low  # initial stop vid ORB-low
        self.entry_mode = entry_mode
        self.orb_high = orb_high
        self.orb_low = orb_low

class SymbolState:
    def __init__(self, symbol: str):
        self.symbol = symbol  # KuCoin-format 'BTC-USDT'
        self.prev_closed = None   # (ts:int, open:float, close:float, high:float, low:float)
        self.last_closed = None
        self.forming = None
        # ORB (första gröna candle efter en röd)
        self.orb = None  # dict: {'ts':int,'open':float,'close':float,'high':float,'low':float}
        # Position (LONG)
        self.pos: Optional[Position] = None
        # Statistik
        self.closed_pnls: List[float] = []
        # För att undvika dubbla entries på samma forming-candle i tick-läge
        self._last_tick_entry_ts: Optional[int] = None

    def reset_orb(self):
        self.orb = None
        self._last_tick_entry_ts = None

class BotState:
    def __init__(self):
        self.engine_on: bool = True
        self.live: bool = LIVE == 1
        self.entry_mode: str = ENTRY_MODE if ENTRY_MODE in ("tick", "close") else "tick"
        self.tf: str = TF if TF in VALID_TF else "1min"
        self.symbols: List[str] = [to_kucoin_symbol(s.strip()) for s in SYMBOLS_ENV.split(",") if s.strip()]
        self.per_symbol: Dict[str, SymbolState] = {s: SymbolState(s) for s in self.symbols}
        # CSV paths
        self.csv_paths: Dict[str, str] = {s: f"/tmp/trades_{s.replace('-', '')}.csv" for s in self.symbols}

STATE = BotState()

# ------------------------------------------------------------
# Telegram Application (webhook-läge)
# ------------------------------------------------------------
if not TG_TOKEN:
    LOG.warning("TELEGRAM_BOT_TOKEN saknas! Endast health-endpoint kommer fungera.")

app = FastAPI()
tg_app: Optional[Application] = None
WEBHOOK_PATH: Optional[str] = None

# ------------------------------------------------------------
# Hjälpfunktioner
# ------------------------------------------------------------
async def send_owner(text: str, html: bool = False):
    """Skicka meddelande till Owner."""
    if tg_app is None:
        LOG.info(f"[TG] (ingen tg_app) -> {text}")
        return
    try:
        if html:
            await tg_app.bot.send_message(chat_id=OWNER_CHAT_ID, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        else:
            await tg_app.bot.send_message(chat_id=OWNER_CHAT_ID, text=text, disable_web_page_preview=True)
    except Exception as e:
        LOG.error(f"Misslyckades skicka till owner: {e}")

def fmt_price(p: float) -> str:
    if p >= 100:
        return f"{p:,.1f}"
    if p >= 1:
        return f"{p:,.2f}"
    return f"{p:,.4f}"

def row_to_candle(row: List[str]) -> Tuple[int, float, float, float, float]:
    # KuCoin return: [time, open, close, high, low, volume, turnover]
    ts = int(row[0]) * 1000  # sek → ms
    o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
    return ts, o, c, h, l

async def kucoin_get_candles(symbol: str, tf: str, limit: int = 3) -> List[Tuple[int,float,float,float,float]]:
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    params = {"type": tf, "symbol": symbol, "limit": str(limit)}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    items = data.get("data") or []
    # Senaste först → vänd till stigande tid
    candles = [row_to_candle(x) for x in items][::-1]
    return candles

def is_red(candle) -> bool:
    _, o, c, _, _ = candle
    return c < o

def is_green(candle) -> bool:
    _, o, c, _, _ = candle
    return c > o

def write_trade_csv(symbol: str, fields: Dict):
    path = STATE.csv_paths.get(symbol)
    if not path:
        path = f"/tmp/trades_{symbol.replace('-', '')}.csv"
        STATE.csv_paths[symbol] = path
    exists = False
    try:
        with open(path, "r") as _:
            exists = True
    except FileNotFoundError:
        exists = False
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp","symbol","event","side","entry_mode",
            "price","stop","qty","pnl","note"
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(fields)

def k4_rows_from_csv(symbol: str) -> List[Dict]:
    """Bygger K4-liknande rader från symbolens trade-CSV (bara EXIT-rader)."""
    rows = []
    path = STATE.csv_paths.get(symbol)
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("event") == "EXIT":
                # Antag omkostnadsbelopp = qty * entry avg-price finns ej här.
                # Vi stoppar in 'note' som "entry_price;exit_price"
                note = r.get("note", "")
                entry_price = None
                exit_price = None
                try:
                    parts = note.split(";")
                    for p in parts:
                        if p.startswith("entry="):
                            entry_price = float(p.split("=",1)[1])
                        if p.startswith("exit="):
                            exit_price = float(p.split("=",1)[1])
                except Exception:
                    pass
                qty = float(r.get("qty") or 0.0)
                if entry_price is None or exit_price is None:
                    # fallback: använd r['price'] som exit_price
                    exit_price = float(r.get("price") or 0.0)
                    # och anta omkostnad saknas → 0 (inte korrekt men csv blir komplett)
                    entry_price = 0.0
                fors = qty * exit_price
                omk = qty * entry_price
                vinst = fors - omk
                rows.append({
                    "Datum": r.get("timestamp",""),
                    "Värdepapper": symbol,
                    "Antal": f"{qty}",
                    "Försäljningspris": f"{fors:.2f}",
                    "Omkostnadsbelopp": f"{omk:.2f}",
                    "Vinst/Förlust": f"{vinst:.2f}",
                })
    return rows

# ------------------------------------------------------------
# Tradinglogik
# ------------------------------------------------------------
async def process_symbol(symbol: str):
    st = STATE.per_symbol[symbol]

    try:
        candles = await kucoin_get_candles(symbol, STATE.tf, limit=4)
        if len(candles) < 3:
            return
        # Vi använder:
        # - prev_closed = näst-näst-sista (= -3) (stängd)
        # - last_closed = näst-sista (= -2) (stängd)
        # - forming     = sista (= -1)     (pågående)
        prev_closed = candles[-3]
        last_closed = candles[-2]
        forming = candles[-1]

        st.prev_closed = prev_closed
        st.last_closed = last_closed
        st.forming = forming

        # 1) Hitta ORB: första GRÖNA efter en RÖD → den GRÖNA är ORBen
        if st.pos is None and st.orb is None:
            if is_red(prev_closed) and is_green(last_closed):
                ts, o, c, h, l = last_closed
                st.orb = {"ts": ts, "open": o, "close": c, "high": h, "low": l}
                await send_owner(f"🟩 <b>{symbol}</b> ORB funnen: green efter red\nHigh={fmt_price(h)} Low={fmt_price(l)}", html=True)
                write_trade_csv(symbol, {
                    "timestamp": now_utc().isoformat(),
                    "symbol": symbol,
                    "event": "ORB",
                    "side": "",
                    "entry_mode": STATE.entry_mode,
                    "price": f"{h:.8f}",
                    "stop": f"{l:.8f}",
                    "qty": "",
                    "pnl": "",
                    "note": "orb_found"
                })

        # Ingen handel om engine OFF
        if not STATE.engine_on:
            return

        # 2) Entry-regler
        if st.pos is None and st.orb is not None:
            orb_h = st.orb["high"]
            orb_l = st.orb["low"]
            orb_ts = st.orb["ts"]

            if STATE.entry_mode == "close":
                # Köp om en stängd candle STÄNGER över ORB-high
                ts, o, c, h, l = last_closed
                if ts > orb_ts and c > orb_h:
                    # mock-qty baserat på ORDER_SIZE_USDT
                    qty = ORDER_SIZE_USDT / c if c > 0 else 0.0
                    st.pos = Position(entry_price=c, qty=qty, entry_mode="close", orb_high=orb_h, orb_low=orb_l)
                    await send_owner(f"✅ <b>{symbol}</b> ENTRY (close) @ {fmt_price(c)}\nStop start = {fmt_price(orb_l)}", html=True)
                    write_trade_csv(symbol, {
                        "timestamp": now_utc().isoformat(),
                        "symbol": symbol,
                        "event": "ENTRY",
                        "side": "LONG",
                        "entry_mode": "close",
                        "price": f"{c:.8f}",
                        "stop": f"{orb_l:.8f}",
                        "qty": f"{qty:.8f}",
                        "pnl": "",
                        "note": f"orb_high={orb_h};orb_low={orb_l}"
                    })
            else:
                # tick-läge: om FORMING-candlens HIGH bryter över ORB-high (inom samma minut)
                ts_f, o_f, c_f, h_f, l_f = forming
                if ts_f >= orb_ts and h_f >= orb_h:
                    if st._last_tick_entry_ts != ts_f:
                        # simulera fill vid orb_h (brytpunkten)
                        fill = float(orb_h)
                        qty = ORDER_SIZE_USDT / fill if fill > 0 else 0.0
                        st.pos = Position(entry_price=fill, qty=qty, entry_mode="tick", orb_high=orb_h, orb_low=orb_l)
                        st._last_tick_entry_ts = ts_f
                        await send_owner(f"✅ <b>{symbol}</b> ENTRY (tick) bryt @ {fmt_price(fill)}\nStop start = {fmt_price(orb_l)}", html=True)
                        write_trade_csv(symbol, {
                            "timestamp": now_utc().isoformat(),
                            "symbol": symbol,
                            "event": "ENTRY",
                            "side": "LONG",
                            "entry_mode": "tick",
                            "price": f"{fill:.8f}",
                            "stop": f"{orb_l:.8f}",
                            "qty": f"{qty:.8f}",
                            "pnl": "",
                            "note": f"orb_high={orb_h};orb_low={orb_l}"
                        })

        # 3) Trailing stop + Exit
        if st.pos is not None:
            # Flytta stop till low på varje stängd candle (ratchet uppåt)
            ts_l, o_l, c_l, h_l, l_l = last_closed
            new_stop = max(st.pos.stop, l_l)
            if new_stop > st.pos.stop:
                st.pos.stop = new_stop
                await send_owner(f"🔧 <b>{symbol}</b> Trail stop → {fmt_price(new_stop)}", html=True)
                write_trade_csv(symbol, {
                    "timestamp": now_utc().isoformat(),
                    "symbol": symbol,
                    "event": "TRAIL",
                    "side": "LONG",
                    "entry_mode": st.pos.entry_mode,
                    "price": f"{c_l:.8f}",
                    "stop": f"{st.pos.stop:.8f}",
                    "qty": f"{st.pos.qty:.8f}",
                    "pnl": "",
                    "note": "trail_to_last_low"
                })

            # Exit om FORMING candle (tick) går under stop
            ts_f, o_f, c_f, h_f, l_f = forming
            if l_f <= st.pos.stop:
                exit_price = st.pos.stop  # antag fill på stop
                pnl = (exit_price - st.pos.entry_price) * st.pos.qty
                st.closed_pnls.append(pnl)
                await send_owner(
                    f"🛑 <b>{symbol}</b> EXIT @ {fmt_price(exit_price)} (Stop hit)\n"
                    f"PNL: {pnl:.2f} USDT",
                    html=True
                )
                write_trade_csv(symbol, {
                    "timestamp": now_utc().isoformat(),
                    "symbol": symbol,
                    "event": "EXIT",
                    "side": "LONG",
                    "entry_mode": st.pos.entry_mode,
                    "price": f"{exit_price:.8f}",
                    "stop": f"{st.pos.stop:.8f}",
                    "qty": f"{st.pos.qty:.8f}",
                    "pnl": f"{pnl:.8f}",
                    "note": f"entry={st.pos.entry_price};exit={exit_price}"
                })
                # Efter exit: ny ORB kräver ny röd→grön sekvens → nolla ORB
                st.pos = None
                st.reset_orb()

    except Exception as e:
        LOG.exception(f"Fel i process_symbol({symbol}): {e}")

async def engine_loop():
    await asyncio.sleep(1.0)
    await send_owner("🤖 ORB-engine startad.")
    while True:
        try:
            tasks = [process_symbol(sym) for sym in STATE.symbols]
            await asyncio.gather(*tasks)
        except Exception as e:
            LOG.exception(f"Engine loop fel: {e}")
        await asyncio.sleep(2.0)

# ------------------------------------------------------------
# Telegram handlers
# ------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    await update.effective_message.reply_text(
        "Mporb-bot ✅\n\n"
        "Kommandon:\n"
        "/status – visa läge\n"
        "/entry tick|close – välj entry-typ\n"
        "/engine on|off – start/stoppa motor\n"
        "/live 0|1 – mock eller live (kräver KuCoin-nycklar för 1)\n"
        "/symbols BTCUSDT,ETHUSDT,… – sätt symboler\n"
        "/tf 1min|3min – timeframe\n"
        "/export_csv – skicka trade-CSV\n"
        "/k4 – skicka K4-CSV (samlad)\n"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    lines = [
        f"🔧 Engine: {'ON' if STATE.engine_on else 'OFF'}",
        f"📡 Live: {'ON' if STATE.live else 'OFF (mock)'}",
        f"⏱️ TF: {STATE.tf}",
        f"🎯 Entry: {STATE.entry_mode}",
        f"🎛️ Symbols: {', '.join(STATE.symbols)}",
        "",
    ]
    for s, st in STATE.per_symbol.items():
        orb = st.orb
        pos = st.pos
        if orb:
            lines.append(f"• {s} ORB: H={fmt_price(orb['high'])} L={fmt_price(orb['low'])}")
        else:
            lines.append(f"• {s} ORB: –")
        if pos:
            lines.append(f"   LONG @ {fmt_price(pos.entry_price)} stop={fmt_price(pos.stop)} mode={pos.entry_mode}")
        else:
            lines.append(f"   Position: –")
        if st.closed_pnls:
            lines.append(f"   Realized PNL: {sum(st.closed_pnls):.2f} USDT")
    await update.effective_message.reply_text("\n".join(lines))

async def cmd_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    if not context.args:
        await update.effective_message.reply_text("Använd: /entry tick eller /entry close")
        return
    m = context.args[0].lower()
    if m not in ("tick", "close"):
        await update.effective_message.reply_text("Endast 'tick' eller 'close' är giltigt.")
        return
    STATE.entry_mode = m
    await update.effective_message.reply_text(f"Entry-mode satt till: {m}")

async def cmd_engine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    if not context.args:
        await update.effective_message.reply_text("Använd: /engine on eller /engine off")
        return
    v = context.args[0].lower()
    if v == "on":
        STATE.engine_on = True
    elif v == "off":
        STATE.engine_on = False
    else:
        await update.effective_message.reply_text("Använd: /engine on | off")
        return
    await update.effective_message.reply_text(f"Engine: {'ON' if STATE.engine_on else 'OFF'}")

async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    if not context.args:
        await update.effective_message.reply_text("Använd: /live 0 eller /live 1")
        return
    try:
        v = int(context.args[0])
    except:
        await update.effective_message.reply_text("Använd: /live 0 eller /live 1")
        return
    if v == 1 and not (KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE):
        await update.effective_message.reply_text("Kan inte sätta LIVE=1 utan KuCoin API-nycklar i miljön.")
        return
    STATE.live = (v == 1)
    await update.effective_message.reply_text(f"Live-läge: {'ON' if STATE.live else 'OFF (mock)'}")

async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    txt = " ".join(context.args)
    if not txt:
        await update.effective_message.reply_text("Använd: /symbols BTCUSDT,ETHUSDT,LINKUSDT …")
        return
    raw_syms = [to_kucoin_symbol(s.strip()) for s in txt.split(",") if s.strip()]
    if not raw_syms:
        await update.effective_message.reply_text("Inga symboler tolkades.")
        return
    STATE.symbols = raw_syms
    STATE.per_symbol = {s: SymbolState(s) for s in STATE.symbols}
    STATE.csv_paths = {s: f"/tmp/trades_{s.replace('-', '')}.csv" for s in STATE.symbols}
    await update.effective_message.reply_text(f"Symbols satta: {', '.join(STATE.symbols)}")

async def cmd_tf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    if not context.args:
        await update.effective_message.reply_text("Använd: /tf 1min eller /tf 3min")
        return
    v = context.args[0].lower()
    if v not in VALID_TF:
        await update.effective_message.reply_text("Endast 1min eller 3min gäller här.")
        return
    STATE.tf = v
    await update.effective_message.reply_text(f"Timeframe satt till: {v}")

async def cmd_export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    for sym, path in STATE.csv_paths.items():
        if not os.path.exists(path):
            await update.effective_message.reply_text(f"{sym}: Ingen CSV ännu.")
            continue
        try:
            with open(path, "rb") as f:
                await update.effective_message.reply_document(document=InputFile(f, filename=os.path.basename(path)),
                                                             caption=f"{sym} trades CSV")
        except Exception as e:
            await update.effective_message.reply_text(f"{sym}: misslyckades skicka CSV: {e}")

async def cmd_k4(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.id != OWNER_CHAT_ID:
        return
    # Sätt ihop alla EXIT-rader till en K4-liknande CSV
    rows = []
    for sym in STATE.symbols:
        rows.extend(k4_rows_from_csv(sym))
    if not rows:
        await update.effective_message.reply_text("Inga avslut hittade för K4.")
        return
    # Bygg CSV i minnet
    buf = BytesIO()
    fieldnames = ["Datum","Värdepapper","Antal","Försäljningspris","Omkostnadsbelopp","Vinst/Förlust"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    buf.seek(0)
    await update.effective_message.reply_document(document=InputFile(buf, filename="k4_export.csv"),
                                                  caption="K4-CSV (sammanställning)")

# ------------------------------------------------------------
# FastAPI lifecycle: starta Telegram + engine
# ------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    global tg_app, WEBHOOK_PATH
    if not TG_TOKEN:
        LOG.warning("Ingen TELEGRAM_BOT_TOKEN – hoppar över Telegram-init.")
        return

    tg_app = ApplicationBuilder().token(TG_TOKEN).build()
    # Registrera handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("status", cmd_status))
    tg_app.add_handler(CommandHandler("entry", cmd_entry))
    tg_app.add_handler(CommandHandler("engine", cmd_engine))
    tg_app.add_handler(CommandHandler("live", cmd_live))
    tg_app.add_handler(CommandHandler("symbols", cmd_symbols))
    tg_app.add_handler(CommandHandler("tf", cmd_tf))
    tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
    tg_app.add_handler(CommandHandler("k4", cmd_k4))

    await tg_app.initialize()

    # Webhook path = /webhook/<token>
    WEBHOOK_PATH = f"/webhook/{TG_TOKEN}"
    # Render public URL är din tjänsts URL
    # Sätt webhook
    webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'mporbbot.onrender.com')}{WEBHOOK_PATH}"
    await tg_app.bot.set_webhook(url=webhook_url)
    await tg_app.start()
    LOG.info(f"Webhook satt: {webhook_url}")

    # Starta motor
    asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def on_shutdown():
    if tg_app:
        try:
            await tg_app.stop()
            await tg_app.shutdown()
        except Exception:
            pass

# ------------------------------------------------------------
# Webhook endpoint
# ------------------------------------------------------------
@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != TG_TOKEN:
        return Response(status_code=403)
    if tg_app is None:
        return Response(status_code=503)

    data = await request.json()
    try:
        update = Update.de_json(data=data, bot=tg_app.bot)
    except Exception as e:
        LOG.error(f"Kunde inte parse Update: {e}  payload={data}")
        return Response(status_code=200)

    # Lägg i PTB:s kö så den processas i rätt loop
    await tg_app.update_queue.put(update)
    return Response(status_code=200)

# ------------------------------------------------------------
# Övriga endpoints
# ------------------------------------------------------------
@app.get("/")
async def root():
    return PlainTextResponse("OK - Mporb ORB-bot v33")

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "engine_on": STATE.engine_on,
        "live": STATE.live,
        "entry_mode": STATE.entry_mode,
        "tf": STATE.tf,
        "symbols": STATE.symbols
    })
