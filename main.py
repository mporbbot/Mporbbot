# main.py
import os, time, json, hmac, base64, hashlib, asyncio, logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton, ParseMode, InputMediaDocument
)
from telegram.ext import (
    Updater, CallbackContext, CommandHandler, CallbackQueryHandler
)

# ---------- Helpers: Decimal med snygg avrundning ----------
def d(x): return Decimal(str(x))
def rnd(x, n=4): return d(x).quantize(Decimal(10) ** -n, rounding=ROUND_HALF_UP)

# ---------- Miljö ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TRADE_USDT     = d(os.getenv("MOCK_TRADE_USDT", "30"))
FEE_RATE       = d(os.getenv("FEE_RATE", "0.001"))  # per sida
PING_URL       = os.getenv("PING_URL", "")
ENTRY_MODE     = "close"  # "tick" eller "close" (vi använder kline close)
TIMEFRAME      = "1min"   # "1min","3min","5min"
SYMBOLS        = ["ADAUSDT","ETHUSDT","BTCUSDT"]  # går att ändra i /symbols
KEEPALIVE      = True
ENGINE_ON      = True
MODE           = "mock"   # "mock" eller "live"

# Trail-profil (kan knäppas om i UI)
TRIGGER_PCT = d("0.0090")  # 0.90%
OFFSET_PCT  = d("0.0020")  # 0.20%
MIN_LOCK    = d("0.0070")  # 0.70% min vinst när triggat

# ---------- Logg ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

# ---------- Data-strukturer ----------
class OrbState:
    def __init__(self):
        self.orb_high: Optional[Decimal] = None
        self.orb_low: Optional[Decimal] = None
        self.last_candle_color: Optional[str] = None  # "red"/"green"
        self.stop: Optional[Decimal] = None
        self.in_position: bool = False
        self.entry: Optional[Decimal] = None
        self.qty: Optional[Decimal] = None
        self.trail_armed: bool = False

symbol_state: Dict[str, OrbState] = {}
for s in SYMBOLS: symbol_state[s]=OrbState()

# PnL/mock
class MockPos:
    def __init__(self, symbol:str, entry:Decimal, qty:Decimal, fee_rate:Decimal):
        self.symbol=symbol; self.entry=entry; self.qty=qty; self.fee_rate=fee_rate

mock_positions: Dict[str, MockPos] = {}
daily_pnl_usdt: Decimal = d("0")
trade_log: List[Dict] = []   # för CSV/K4

# ---------- KuCoin publika klines ----------
# KuCoin kline endpoint: https://api.kucoin.com/api/v1/market/candles?type=1min&symbol=BTC-USDT
KU_PUBLIC = "https://api.kucoin.com"

def ku_symbol(sym:str)->str:
    # ADAUSDT -> ADA-USDT
    if sym.endswith("USDT"): return sym[:-4] + "-USDT"
    return sym

def fetch_last_kline(symbol:str, tf:str)->Optional[dict]:
    tmap = {"1m":"1min","3m":"3min","5m":"5min", "1min":"1min","3min":"3min","5min":"5min"}
    typ = tmap.get(tf, "1min")
    url=f"{KU_PUBLIC}/api/v1/market/candles"
    try:
        r=requests.get(url, params={"type":typ, "symbol":ku_symbol(symbol)}, timeout=10)
        r.raise_for_status()
        # API returnerar lista: [time, open, close, high, low, volume, turnover]
        arr=r.json().get("data",[])
        if not arr: return None
        k=arr[0]
        return {
            "ts": int(k[0]),
            "open": d(k[1]),
            "close": d(k[2]),
            "high": d(k[3]),
            "low": d(k[4]),
        }
    except Exception as e:
        log.warning(f"kline error {symbol}: {e}")
        return None

# ---------- Mock PnL ----------
def mock_open(symbol:str, entry_price:Decimal):
    if symbol in mock_positions: return
    qty = (TRADE_USDT / entry_price)
    mock_positions[symbol]=MockPos(symbol, entry_price, qty, FEE_RATE)

def mock_close(symbol:str, exit_price:Decimal):
    global daily_pnl_usdt
    pos=mock_positions.get(symbol)
    if not pos: return d("0")
    realized = (exit_price - pos.entry) * pos.qty
    fee = (pos.entry + exit_price) * pos.qty * pos.fee_rate
    pnl = realized - fee
    daily_pnl_usdt += pnl
    # trade logg
    trade_log.append({
        "ts": int(time.time()),
        "symbol": symbol,
        "side": "SELL", # vi kör bara long
        "entry": str(rnd(pos.entry,6)),
        "exit":  str(rnd(exit_price,6)),
        "qty":   str(rnd(pos.qty,8)),
        "pnl":   str(rnd(pnl,6))
    })
    del mock_positions[symbol]
    return pnl

# ---------- ORB/Engine ----------
def color_of_candle(open_p:Decimal, close_p:Decimal)->str:
    return "green" if close_p>=open_p else "red"

async def process_symbol(symbol:str, kline:dict, bot_say):
    st=symbol_state.setdefault(symbol, OrbState())
    o, c, h, l = kline["open"], kline["close"], kline["high"], kline["low"]
    col=color_of_candle(o,c)

    # 1) Ny ORB när red->green & första gröna
    if st.last_candle_color=="red" and col=="green":
        st.orb_high=h; st.orb_low=l; st.trail_armed=False
        if not st.in_position:
            st.stop = l  # första stop = low på första gröna
    st.last_candle_color=col

    # 2) Bryt upp över ORB-high => köp (bara long)
    if (not st.in_position) and st.orb_high and c>st.orb_high:
        st.in_position=True
        st.entry=c
        # mock open
        if MODE=="mock":
            mock_open(symbol, c)
        await bot_say(f"{symbol}: ORB breakout BUY @ {rnd(c)} | SL {rnd(st.stop)}")

    # 3) När ny candle blir klar: traila stop till nya candlens low om högre
    if st.in_position and st.stop and l>st.stop:
        st.stop = l

    # 4) Extra skyddstrail: om latent vinst ≥ 0.90%, säkra 0.70% via offset 0.20%
    if st.in_position and st.entry:
        move = (c - st.entry) / st.entry
        if move >= TRIGGER_PCT and st.orb_high:
            # dynamisk: ny stop = max(nuvarande stop, high*(1-offset)) men min 0.7% över entry
            protective = (h * (d("1") - OFFSET_PCT))
            min_lock  = st.entry * (d("1") + MIN_LOCK)
            new_stop = max(st.stop or l, protective, min_lock)
            if new_stop > (st.stop or d("0")):
                st.stop = new_stop

    # 5) Exit om priset går under stop
    if st.in_position and st.stop and c <= st.stop:
        exit_p = st.stop
        pnl = d("0")
        if MODE=="mock":
            pnl = mock_close(symbol, exit_p)
        st.in_position=False
        st.entry=None; st.qty=None; st.stop=None; st.orb_high=None; st.orb_low=None; st.trail_armed=False
        await bot_say(f"{symbol}: EXIT @ {rnd(exit_p)} | PnL {rnd(pnl)} USDT | day {rnd(daily_pnl_usdt)}")

# ---------- Telegram ----------
updater: Optional[Updater] = None

def kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Start MOCK", callback_data="start_mock"),
         InlineKeyboardButton("Start LIVE", callback_data="start_live")],
        [InlineKeyboardButton("Engine ON", callback_data="eng_on"),
         InlineKeyboardButton("Engine OFF", callback_data="eng_off")],
        [InlineKeyboardButton("TF 1m", callback_data="tf_1m"),
         InlineKeyboardButton("TF 3m", callback_data="tf_3m"),
         InlineKeyboardButton("TF 5m", callback_data="tf_5m")],
        [InlineKeyboardButton("Trail +0.9%/0.2%", callback_data="trail_on"),
         InlineKeyboardButton("Trail OFF", callback_data="trail_off")]
    ])

async def bot_send(text:str):
    if not updater: return
    try:
        updater.bot.send_message(chat_id=chat_id_holder["id"], text=text)
    except Exception:
        pass

chat_id_holder={"id": None}

def fmt_status()->str:
    lines=[]
    lines.append(f"Mode: {MODE}   Engine: {'ON' if ENGINE_ON else 'OFF'}")
    tf_disp = {"1min":"1m","3min":"3m","5min":"5m"}.get(TIMEFRAME, TIMEFRAME)
    lines.append(f"TF: {tf_disp}    Symbols: {','.join(SYMBOLS)}")
    lines.append(f"Trail: trig {rnd(TRIGGER_PCT*100,2)}% | avst {rnd(OFFSET_PCT*100,2)}% | min {rnd(MIN_LOCK*100,2)}%")
    lines.append(f"Keepalive: {'ON' if KEEPALIVE else 'OFF'}")
    lines.append(f"PnL → MOCK {rnd(daily_pnl_usdt)} | LIVE 0.0000")
    for s in SYMBOLS:
        st=symbol_state.setdefault(s,OrbState())
        pos = "✅" if st.in_position else "❌"
        stop = rnd(st.stop) if st.stop else "-"
        lines.append(f"{s}: pos={pos}  stop={stop}")
    return "\n".join(lines)

def start(update:Update, ctx:CallbackContext):
    chat_id_holder["id"]=update.effective_chat.id
    update.message.reply_text(fmt_status(), reply_markup=kb())

def status_cmd(update:Update, ctx:CallbackContext):
    chat_id_holder["id"]=update.effective_chat.id
    update.message.reply_text(fmt_status(), reply_markup=kb())

def engine_start(update:Update, ctx:CallbackContext):
    global ENGINE_ON
    ENGINE_ON=True
    update.message.reply_text("Engine: ON", reply_markup=kb())

def engine_stop(update:Update, ctx:CallbackContext):
    global ENGINE_ON
    ENGINE_ON=False
    update.message.reply_text("Engine: OFF", reply_markup=kb())

def start_mock(update:Update, ctx:CallbackContext):
    global MODE
    MODE="mock"
    update.message.reply_text("Mock-läge: ON", reply_markup=kb())

def start_live(update:Update, ctx:CallbackContext):
    global MODE
    MODE="live"
    update.message.reply_text("Live-läge: ON (OBS! Endast krokar, mock är källan till PnL)", reply_markup=kb())

def symbols_cmd(update:Update, ctx:CallbackContext):
    global SYMBOLS, symbol_state
    chat_id_holder["id"]=update.effective_chat.id
    if ctx.args:
        new = " ".join(ctx.args).replace(","," ").split()
        SYMBOLS=[s.upper() for s in new]
        symbol_state={s:OrbState() for s in SYMBOLS}
        update.message.reply_text(f"Symbols uppdaterade: {','.join(SYMBOLS)}", reply_markup=kb())
    else:
        update.message.reply_text(f"Aktuella: {','.join(SYMBOLS)}\nSkicka: /symbols BTCUSDT,ETHUSDT,ADAUSDT", reply_markup=kb())

def timeframe_cmd(update:Update, ctx:CallbackContext):
    global TIMEFRAME
    chat_id_holder["id"]=update.effective_chat.id
    if ctx.args:
        m=ctx.args[0].lower()
        TIMEFRAME="1min" if m in ["1m","1","1min"] else "3min" if m in ["3m","3","3min"] else "5min"
    update.message.reply_text(f"Tidsram → {TIMEFRAME}", reply_markup=kb())

def entry_mode_cmd(update:Update, ctx:CallbackContext):
    global ENTRY_MODE
    chat_id_holder["id"]=update.effective_chat.id
    if ctx.args and ctx.args[0] in ["tick","close"]:
        ENTRY_MODE=ctx.args[0]
    update.message.reply_text(f"Entry mode: {ENTRY_MODE}", reply_markup=kb())

def trailing_cmd(update:Update, ctx:CallbackContext):
    global TRIGGER_PCT, OFFSET_PCT, MIN_LOCK
    chat_id_holder["id"]=update.effective_chat.id
    if len(ctx.args)==2:
        TRIGGER_PCT = d(ctx.args[0])
        OFFSET_PCT  = d(ctx.args[1])
        MIN_LOCK    = TRIGGER_PCT - OFFSET_PCT
    update.message.reply_text(f"Trail satt: trig {rnd(TRIGGER_PCT*100,2)}% | avst {rnd(OFFSET_PCT*100,2)}% | min {rnd(MIN_LOCK*100,2)}%", reply_markup=kb())

def pnl_cmd(update:Update, ctx:CallbackContext):
    update.message.reply_text(f"PnL → MOCK {rnd(daily_pnl_usdt)} | LIVE 0.0000", reply_markup=kb())

def reset_pnl(update:Update, ctx:CallbackContext):
    global daily_pnl_usdt, trade_log
    daily_pnl_usdt = d("0")
    trade_log=[]
    update.message.reply_text("Dagens PnL återställd.", reply_markup=kb())

def export_csv(update:Update, ctx:CallbackContext):
    # Enkel CSV (ts,symbol,entry,exit,qty,pnl)
    header="timestamp,symbol,entry,exit,qty,pnl\n"
    rows=[f"{t['ts']},{t['symbol']},{t['entry']},{t['exit']},{t['qty']},{t['pnl']}" for t in trade_log]
    blob=header+"\n".join(rows)
    update.message.reply_document(document=blob.encode("utf-8"), filename="trades.csv")

def export_k4(update:Update, ctx:CallbackContext):
    # K4 (mycket förenklad, en rad per affär)
    # Kolumner: Typ;Beteckning;Antal;Anskaffningsutgift;Försäljningspris;Vinst/Förlust;Datum
    # Vi antar spot och beskriver som "Krypto spot"
    header="Typ;Beteckning;Antal;Anskaffningsutgift;Försäljningspris;Vinst/Förlust;Datum\n"
    lines=[]
    for t in trade_log:
        qty = Decimal(t["qty"])
        buy = Decimal(t["entry"])*qty
        sell= Decimal(t["exit"])*qty
        v   = sell - buy
        dt  = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        lines.append(f"Övrig tillgång;{t['symbol']};{rnd(qty,8)};{rnd(buy,2)};{rnd(sell,2)};{rnd(v,2)};{dt}")
    blob=header+"\n".join(lines)
    update.message.reply_document(document=blob.encode("utf-8"), filename="k4.csv")

def keepalive_on(update:Update, ctx:CallbackContext):
    global KEEPALIVE
    KEEPALIVE=True
    update.message.reply_text("Keepalive: ON", reply_markup=kb())

def keepalive_off(update:Update, ctx:CallbackContext):
    global KEEPALIVE
    KEEPALIVE=False
    update.message.reply_text("Keepalive: OFF", reply_markup=kb())

def panic(update:Update, ctx:CallbackContext):
    # Stäng alla mock-positioner på senaste stop (eller close)
    closed=0
    for s in list(mock_positions.keys()):
        # Försök stänga på senaste close
        k=fetch_last_kline(s, TIMEFRAME) or {}
        px=k.get("close", d("0"))
        mock_close(s, px)
        closed+=1
    for s,st in symbol_state.items():
        st.in_position=False; st.entry=None; st.stop=None; st.orb_high=None; st.orb_low=None; st.trail_armed=False
    update.message.reply_text(f"Panic: stängde {closed} mock-positioner.", reply_markup=kb())

def on_cb(update:Update, ctx:CallbackContext):
    q=update.callback_query
    q.answer()
    data=q.data
    global TIMEFRAME, TRIGGER_PCT, OFFSET_PCT, MIN_LOCK, ENGINE_ON, MODE
    if data=="start_mock":
        MODE="mock"; q.edit_message_text(fmt_status(), reply_markup=kb())
    elif data=="start_live":
        MODE="live"; q.edit_message_text(fmt_status(), reply_markup=kb())
    elif data=="eng_on":
        ENGINE_ON=True; q.edit_message_text(fmt_status(), reply_markup=kb())
    elif data=="eng_off":
        ENGINE_ON=False; q.edit_message_text(fmt_status(), reply_markup=kb())
    elif data=="tf_1m":
        TIMEFRAME="1min"; q.edit_message_text(fmt_status(), reply_markup=kb())
    elif data=="tf_3m":
        TIMEFRAME="3min"; q.edit_message_text(fmt_status(), reply_markup=kb())
    elif data=="tf_5m":
        TIMEFRAME="5min"; q.edit_message_text(fmt_status(), reply_markup=kb())
    elif data=="trail_on":
        TRIGGER_PCT=d("0.0090"); OFFSET_PCT=d("0.0020"); MIN_LOCK=d("0.0070")
        q.edit_message_text(fmt_status(), reply_markup=kb())
    elif data=="trail_off":
        TRIGGER_PCT=d("9"); OFFSET_PCT=d("0")  # i praktiken aldrig trigga
        MIN_LOCK=d("0")
        q.edit_message_text(fmt_status(), reply_markup=kb())

# ---------- Engine loop ----------
async def engine_loop():
    await asyncio.sleep(3)
    while True:
        if ENGINE_ON and SYMBOLS:
            for s in SYMBOLS:
                k=fetch_last_kline(s, TIMEFRAME)
                if k:
                    await process_symbol(s, k, bot_send)
                await asyncio.sleep(0.3)
        await asyncio.sleep(2)

# ---------- Keepalive ----------
async def keepalive_loop(port_env=""):
    await asyncio.sleep(5)
    while True:
        if KEEPALIVE:
            try:
                url = PING_URL or f"http://127.0.0.1:{os.getenv('PORT', '10000')}/health"
                requests.get(url, timeout=5)
            except Exception:
                pass
        await asyncio.sleep(120)  # varannan minut

# ---------- FastAPI ----------
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root(): return "OK"

@app.get("/health", response_class=PlainTextResponse)
def health(): return "OK"

# ---------- Start PTB + bakgrundsloopar ----------
def start_bot():
    global updater
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN saknas.")
        return
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp=updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("status", status_cmd))
    dp.add_handler(CommandHandler("engine_start", engine_start))
    dp.add_handler(CommandHandler("engine_stop", engine_stop))
    dp.add_handler(CommandHandler("start_mock", start_mock))
    dp.add_handler(CommandHandler("start_live", start_live))
    dp.add_handler(CommandHandler("symbols", symbols_cmd, pass_args=True))
    dp.add_handler(CommandHandler("timeframe", timeframe_cmd, pass_args=True))
    dp.add_handler(CommandHandler("entry_mode", entry_mode_cmd, pass_args=True))
    dp.add_handler(CommandHandler("trailing", trailing_cmd, pass_args=True))
    dp.add_handler(CommandHandler("pnl", pnl_cmd))
    dp.add_handler(CommandHandler("reset_pnl", reset_pnl))
    dp.add_handler(CommandHandler("export_csv", export_csv))
    dp.add_handler(CommandHandler("export_k4", export_k4))
    dp.add_handler(CommandHandler("keepalive_on", keepalive_on))
    dp.add_handler(CommandHandler("keepalive_off", keepalive_off))
    dp.add_handler(CommandHandler("panic", panic))
    dp.add_handler(CallbackQueryHandler(on_cb))

    updater.start_polling(drop_pending_updates=True)
    log.info("Telegram-bot startad.")

# Kör bot + loops när appen startar på Render
@app.on_event("startup")
async def on_startup():
    loop=asyncio.get_event_loop()
    loop.create_task(asyncio.to_thread(start_bot))
    loop.create_task(engine_loop())
    loop.create_task(keepalive_loop())
