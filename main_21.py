# main_21.py — PTB v20 + BUY/SELL-kort, TF 1/3/5/15, webhook-kill, FastAPI
# Robust polling + debugendpoints för Telegram-kontakt.

import os, csv, time, threading, asyncio, traceback, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from queue import Queue

import requests
from fastapi import FastAPI
from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("orb_v21")

# =================== Konfig ===================
SYMBOLS = ["BTCUSDT","ETHUSDT","ADAUSDT","LINKUSDT","XRPUSDT"]
TRADE_SIZE_USDT = 30.0
FEE_RATE = 0.001
AI_DEFAULT = "neutral"
DOJI_FILTER = True
DOJI_BODY_PCT = 0.10
BREAKOUT_BUFFER = 0.001           # +0.1% över ORB-high
DEFAULT_TF_MIN = 3
POLL_FAST = 10                    # sek (1m)
POLL_SLOW = 20                    # sek (3/5/15m)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

LOG_DIR = "logs"
MOCK_LOG = os.path.join(LOG_DIR, "mock_trade_log.csv")
REAL_LOG = os.path.join(LOG_DIR, "real_trade_log.csv")
ACTIVITY_LOG = os.path.join(LOG_DIR, "activity_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

def _ensure_csv(p, header):
    if not os.path.exists(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

_ensure_csv(MOCK_LOG, ["time_utc","symbol","side","qty","price","fee","ai_mode","reason"])
_ensure_csv(REAL_LOG, ["time_utc","symbol","side","qty","price","fee","ai_mode","reason"])
_ensure_csv(ACTIVITY_LOG, ["time_utc","symbol","event","details"])

def log_activity(symbol, event, details):
    with open(ACTIVITY_LOG,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(),symbol,event,details])
    LOG.info(f"{symbol} {event}: {details}")

def candle_color(o,c): return "green" if c>=o else "red"
def is_doji(o,h,l,c,body_pct=DOJI_BODY_PCT):
    rng=max(h-l,1e-12); body=abs(c-o); return (body/rng)<=body_pct
def ku_symbol(s): s=s.upper(); return s.replace("USDT","-USDT") if "USDT" in s and "-" not in s else s

# =================== Typer & State ===================
@dataclass
class Candle: ts:int; o:float; h:float; l:float; c:float; v:float
@dataclass
class ORB: start_ts:int; high:float; low:float; base:Candle
@dataclass
class Position:
    side:str; entry:float; qty:float; orb_at_entry:ORB
    stop: Optional[float]=None
    active: bool = True

@dataclass
class BotState:
    trading_enabled: bool = False
    mode: str = "mock"                    # mock | live
    ai_mode: str = AI_DEFAULT
    epoch: int = 0

    use_orb: bool = True
    use_closebreak: bool = True
    trailing_enabled: bool = True
    pnl_offset: float = 0.0

    tf_min: int = DEFAULT_TF_MIN

    positions: Dict[str, Optional[Position]] = field(default_factory=dict)
    orbs: Dict[str, Optional[ORB]] = field(default_factory=dict)
    prev_seen_color: Dict[str, Optional[str]] = field(default_factory=dict)
    shift_armed: Dict[str, bool] = field(default_factory=dict)
    last_candle_ts: Dict[str, Optional[int]] = field(default_factory=dict)
    last_closed_low: Dict[str, Optional[float]] = field(default_factory=dict)

    chat_id: Optional[int] = None  # Telegram mottagare

    def reset_symbol(self, s:str):
        self.positions[s]=None; self.orbs[s]=None
        self.prev_seen_color[s]=None; self.shift_armed[s]=False
        self.last_candle_ts[s]=None; self.last_closed_low[s]=None

STATE = BotState()
for _s in SYMBOLS: STATE.reset_symbol(_s)

# Notifieringskö (trådsäker)
OUTBOX: "Queue[str]" = Queue()

# =================== Order/PnL ===================
def kucoin_min_qty(symbol, price): return round(5.0/max(price,1e-9),6)

def log_trade(mock, symbol, side, qty, price, fee, ai_mode, reason):
    path = MOCK_LOG if mock else REAL_LOG
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(),symbol,side,f"{qty:.8f}",f"{price:.8f}",f"{fee:.8f}",ai_mode,reason])

def todays_pnl_from_log(path):
    today=datetime.utcnow().date(); pnl=0.0
    if not os.path.exists(path): return 0.0
    with open(path,"r",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for r in reader:
            t=datetime.fromisoformat(r["time_utc"])
            if t.date()!=today: continue
            side=r["side"].lower(); price=float(r["price"]); qty=float(r["qty"]); fee=float(r["fee"])
            amt=price*qty
            if side=="buy": pnl -= (amt+fee)
            elif side=="sell": pnl += (amt-fee)
    return pnl

def current_pnl_with_offset():
    return todays_pnl_from_log(MOCK_LOG if STATE.mode=="mock" else REAL_LOG) - STATE.pnl_offset

def enqueue_buy_card(symbol: str, entry: float, stop: Optional[float], qty: float, reason: str):
    pct_str = "-"
    if stop is not None and entry>0:
        pct = (stop/entry - 1.0)*100.0
        pct_str = f"({pct:.2f}%)"
    text = (
        f"🟢 <b>BUY {symbol}</b>\n"
        f"time={datetime.utcnow().strftime('%H:%M:%S')} UTC\n"
        f"entry={entry:.6f}  stop={('-' if stop is None else f'{stop:.6f}')} {pct_str}\n"
        f"size={qty:.6f}\n"
        f"reason={reason}"
    )
    OUTBOX.put(text)

def enqueue_sell_card(symbol: str, exit_price: float, entry: float, qty: float, reason: str, pnl_usdt: float):
    pct_str = "-"
    if entry > 0:
        pct = (exit_price/entry - 1.0)*100.0
        pct_str = f"({pct:.2f}%)"
    text = (
        f"🔴 <b>SELL {symbol}</b>\n"
        f"time={datetime.utcnow().strftime('%H:%M:%S')} UTC\n"
        f"exit={exit_price:.6f}  entry={entry:.6f} {pct_str}\n"
        f"size={qty:.6f}   PnL={pnl_usdt:.4f} USDT\n"
        f"reason={reason}"
    )
    OUTBOX.put(text)

def place_order(symbol, side, price, usdt_amount, reason):
    if not STATE.trading_enabled:
        log_activity(symbol,"ORDER_BLOCKED",reason); return None

    qty = max(usdt_amount/max(price,1e-9),0.0)
    if qty < kucoin_min_qty(symbol, price):
        log_activity(symbol,"ORDER_SKIPPED",f"qty too small {qty:.8f}"); return None

    fee = usdt_amount*FEE_RATE
    log_trade(STATE.mode=="mock",symbol,side,qty,price,fee,STATE.ai_mode,reason)

    if side.lower()=="buy":
        current_orb = STATE.orbs.get(symbol)
        init_stop = current_orb.low if current_orb else None
        STATE.positions[symbol] = Position("long", price, qty, current_orb, stop=init_stop, active=True)
        log_activity(symbol, "POSITION_OPEN", f"entry={price} qty={qty:.8f} stop={init_stop if init_stop is not None else '-'}")
        if STATE.chat_id is not None:
            enqueue_buy_card(symbol, price, init_stop, qty, reason)
    elif side.lower()=="sell" and STATE.positions.get(symbol):
        pos = STATE.positions[symbol]
        pnl = (price - pos.entry) * pos.qty - fee
        STATE.positions[symbol] = None
        log_activity(symbol, "POSITION_CLOSE", f"close={price} pnl={pnl:.8f} reason={reason}")
        if STATE.chat_id is not None:
            enqueue_sell_card(symbol, price, pos.entry, pos.qty, reason, pnl)
    return True

# =================== ORB & logik ===================
def start_new_orb(symbol, base:Candle):
    STATE.orbs[symbol]=ORB(base.ts, base.h, base.l, base)
    log_activity(symbol,"NEW_ORB",f"H={base.h} L={base.l} ts={base.ts}")

def ai_allows(x):
    if STATE.ai_mode=="aggressiv": return x>=0.2
    if STATE.ai_mode=="neutral":   return x>=0.5
    if STATE.ai_mode=="försiktig": return x>=0.75
    return True

def try_breakouts(symbol, c:Candle):
    orb=STATE.orbs.get(symbol)
    if not orb: return
    if DOJI_FILTER and is_doji(c.o,c.h,c.l,c.c):
        log_activity(symbol,"SKIP_DOJI",f"ts={c.ts}"); return

    trig = orb.high*(1+BREAKOUT_BUFFER)
    if STATE.use_orb and c.h>=trig and ai_allows(0.6):
        place_order(symbol,"buy",max(trig,c.o),TRADE_SIZE_USDT,"ORB_LONG_BREAK"); return
    if STATE.use_closebreak and c.o<orb.high and c.c>=trig and ai_allows(0.6):
        place_order(symbol,"buy",c.c,TRADE_SIZE_USDT,"CloseBreak")

def update_stop_trailing(symbol: str, prev_closed_low: Optional[float], curr: Candle):
    if not STATE.trailing_enabled: return
    pos = STATE.positions.get(symbol)
    if not pos or not pos.active: return

    if pos.stop is None and pos.orb_at_entry:
        pos.stop = pos.orb_at_entry.low
    if prev_closed_low is not None and (pos.stop is None or prev_closed_low > pos.stop):
        pos.stop = prev_closed_low
        log_activity(symbol, "STOP_RAISE", f"new_stop={pos.stop:.6f}")
    if pos.stop is not None and curr.l <= pos.stop:
        place_order(symbol, "sell", pos.stop, pos.qty * pos.entry, "HARD_STOP_HIT")

def process_candle(symbol: str, prev: Optional[Candle], curr: Candle):
    if STATE.orbs.get(symbol) is None:
        start_new_orb(symbol, curr)
    prev_seen = STATE.prev_seen_color.get(symbol)
    curr_col = candle_color(curr.o, curr.c)
    if prev is not None and prev_seen is not None:
        prev_col_actual = candle_color(prev.o, prev.c)
        if (prev_col_actual != prev_seen) and not STATE.shift_armed.get(symbol,False):
            STATE.shift_armed[symbol]=True
            log_activity(symbol,"SHIFT_ARMED",f"prev_ts={prev.ts} {prev_seen}->{prev_col_actual}")
        if STATE.shift_armed.get(symbol,False):
            start_new_orb(symbol, curr)
            STATE.shift_armed[symbol]=False
    STATE.prev_seen_color[symbol]=curr_col
    try_breakouts(symbol, curr)
    update_stop_trailing(symbol, STATE.last_closed_low.get(symbol), curr)
    STATE.last_closed_low[symbol] = curr.l

# =================== Data (KuCoin REST) ===================
TF_MAP = {1:"1min", 3:"3min", 5:"5min", 15:"15min"}

def fetch_kucoin(symbol: str, tf_min: int, limit: int = 60)->List[Candle]:
    tf = TF_MAP.get(tf_min, "3min")
    url=f"{KUCOIN_BASE_URL}/api/v1/market/candles"
    params={"type":tf,"symbol":ku_symbol(symbol)}
    r=requests.get(url,params=params,timeout=10); r.raise_for_status()
    arr=r.json().get("data",[])
    out=[]
    for it in reversed(arr[-limit:]):   # äldst→nyast
        t_end=int(float(it[0])); o=float(it[1]); c=float(it[2]); h=float(it[3]); l=float(it[4]); v=float(it[5])
        ts=(t_end - tf_min*60)*1000
        out.append(Candle(ts,o,h,l,c,v))
    return out

def poll_loop():
    while True:
        try:
            tf = STATE.tf_min
            for s in SYMBOLS:
                candles=fetch_kucoin(s, tf, 60)
                last=STATE.last_candle_ts.get(s)
                prev=None
                for c in candles:
                    if last is None or c.ts>last:
                        process_candle(s, prev, c)
                        STATE.last_candle_ts[s]=c.ts
                    prev=c
            time.sleep(POLL_FAST if tf==1 else POLL_SLOW)
        except Exception as e:
            log_activity("-", "POLL_ERROR", str(e)); time.sleep(5)

# =================== Telegram (PTB v20) ===================
HELP = (
    "/engine_start – slå på handel\n"
    "/engine_stop – stäng av handel\n"
    "/start_mock – mockläge på\n"
    "/start_live – liveläge på\n"
    "/status – kompakt status\n"
    "/entry_mode [orb|closebreak|both]\n"
    "/trailing [on|off]\n"
    "/pnl – dagens PnL\n"
    "/reset_pnl – nollställ PnL (offset)\n"
    "/orb_on  /orb_off\n"
    "/tf [1|3|5|15] – byt timeframe\n"
    "/panic – stäng alla & slå av"
)

async def set_commands(app):
    cmds = [
        BotCommand("status","visa status"),
        BotCommand("engine_start","starta motorn"),
        BotCommand("engine_stop","stoppa motorn"),
        BotCommand("start_mock","mockläge på"),
        BotCommand("start_live","liveläge på"),
        BotCommand("entry_mode","byt entry-läge"),
        BotCommand("trailing","på/av trailing"),
        BotCommand("pnl","visa dagens PnL"),
        BotCommand("reset_pnl","nollställ PnL"),
        BotCommand("orb_on","sätt ORB på"),
        BotCommand("orb_off","sätt ORB av"),
        BotCommand("tf","byt timeframe (1/3/5/15)"),
        BotCommand("panic","stäng alla & stäng av"),
    ]
    await app.bot.set_my_commands(cmds)

def fmt_status_compact() -> str:
    lines=[f"Mode: {STATE.mode}   Engine: {'ON' if STATE.trading_enabled else 'OFF'}",
           f"TF: {STATE.tf_min}m   Symbols: {','.join(SYMBOLS)}",
           f"Entry: {'ORB' if STATE.use_orb else ''}{'+' if STATE.use_orb and STATE.use_closebreak else ''}{'CloseBreak' if STATE.use_closebreak else ''}",
           f"Trail: {'ON' if STATE.trailing_enabled else 'OFF'}   DayPnL: {current_pnl_with_offset():.4f} USDT"]
    for s in SYMBOLS:
        pos = STATE.positions.get(s); orb = STATE.orbs.get(s)
        pos_str = "pos=❌" if not pos else (f"pos=✅ stop={pos.stop:.4f}" if pos.stop is not None else "pos=✅")
        lines.append(f"{s}: {pos_str} | ORB: {'ON' if orb else '-'}")
    return "\n".join(lines)

# --- Egen outbox-worker (ingen JobQueue behövs) ---
async def outbox_worker(app):
    LOG.info("Outbox worker startad")
    while True:
        try:
            if STATE.chat_id is not None and not OUTBOX.empty():
                while not OUTBOX.empty():
                    text = OUTBOX.get()
                    try:
                        await app.bot.send_message(chat_id=STATE.chat_id, text=text, parse_mode=ParseMode.HTML)
                    except Exception:
                        traceback.print_exc()
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(1.0)
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(1.0)

# ---- Handlers ----
async def h_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    LOG.info(f"/start från chat_id={STATE.chat_id}")
    await update.message.reply_text("Mp ORBbot v21 igång.\n" + HELP)

async def h_engine_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.trading_enabled=True
    await update.message.reply_text("✅ Engine: ON")

async def h_engine_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.trading_enabled=False; STATE.epoch+=1
    await update.message.reply_text("🛑 Engine: OFF – alla signaler blockeras.")
    log_activity("-", "ENGINE_OFF", "manual")

async def h_start_mock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.mode="mock"; STATE.trading_enabled=True
    await update.message.reply_text("✅ MOCK-läge: ON")

async def h_start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.mode="live"; STATE.trading_enabled=True
    await update.message.reply_text("⚠️ LIVE-läge: ON")

async def h_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(fmt_status_compact())

async def h_entry_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        arg=context.args[0].lower()
        if arg=="orb": STATE.use_orb=True; STATE.use_closebreak=False
        elif arg=="closebreak": STATE.use_orb=False; STATE.use_closebreak=True
        elif arg=="both": STATE.use_orb=True; STATE.use_closebreak=True
        else: await update.message.reply_text("Använd: /entry_mode [orb|closebreak|both]"); return
        await update.message.reply_text(f"Entry mode: ORB={'ON' if STATE.use_orb else 'OFF'}, CloseBreak={'ON' if STATE.use_closebreak else 'OFF'}")
    else:
        await update.message.reply_text(f"Aktuell: ORB={'ON' if STATE.use_orb else 'OFF'}, CloseBreak={'ON' if STATE.use_closebreak else 'OFF'}")

async def h_trailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        v=context.args[0].lower()
        if v in ("on","off"): STATE.trailing_enabled=(v=="on")
        else: await update.message.reply_text("Använd: /trailing [on|off]"); return
    else:
        STATE.trailing_enabled=not STATE.trailing_enabled
    await update.message.reply_text(f"Trailing: {'ON' if STATE.trailing_enabled else 'OFF'}")

async def h_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Dagens PnL ({STATE.mode.upper()}): {current_pnl_with_offset():.2f} USDT")

async def h_reset_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.pnl_offset = todays_pnl_from_log(MOCK_LOG if STATE.mode=="mock" else REAL_LOG)
    await update.message.reply_text("PnL nollställd (offset satt).")

async def h_orb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.use_orb=True; await update.message.reply_text("ORB: ON")

async def h_orb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.use_orb=False; await update.message.reply_text("ORB: OFF")

async def h_tf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Aktuell TF: {STATE.tf_min}m. Använd /tf 1|3|5|15"); return
    try:
        v=int(context.args[0])
        if v not in (1,3,5,15): raise ValueError()
        STATE.tf_min=v
        await update.message.reply_text(f"Timeframe satt till {v}m.")
    except Exception:
        await update.message.reply_text("Ogiltigt. Använd /tf 1|3|5|15")

async def h_panic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    closed=0
    for s,pos in list(STATE.positions.items()):
        if pos:
            place_order(s,"sell",pos.stop if pos.stop is not None else pos.entry, pos.qty*pos.entry, "PANIC_CLOSE")
            closed+=1
    STATE.trading_enabled=False
    await update.message.reply_text(f"PANIC – stängde {closed} pos. Engine OFF.")

async def h_echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id

# =================== Körning (PTB i tråd + poll-loop) ===================
_application = None

def _run_bot_and_poll():
    async def _amain():
        global _application
        _application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        # Info/logg + rensa webhook
        me = await _application.bot.get_me()
        LOG.info(f"[TG] getMe: @{me.username} id={me.id}")
        info = await _application.bot.get_webhook_info()
        LOG.info(f"[TG] webhook url='{info.url}' pending={info.pending_update_count}")
        await _application.bot.delete_webhook(drop_pending_updates=True)
        LOG.info("[TG] webhook deleted (drop_pending_updates=True)")

        _application.add_handler(CommandHandler("start", h_start))
        _application.add_handler(CommandHandler("engine_start", h_engine_start))
        _application.add_handler(CommandHandler("engine_stop", h_engine_stop))
        _application.add_handler(CommandHandler("start_mock", h_start_mock))
        _application.add_handler(CommandHandler("start_live", h_start_live))
        _application.add_handler(CommandHandler("status", h_status))
        _application.add_handler(CommandHandler("entry_mode", h_entry_mode))
        _application.add_handler(CommandHandler("trailing", h_trailing))
        _application.add_handler(CommandHandler("pnl", h_pnl))
        _application.add_handler(CommandHandler("reset_pnl", h_reset_pnl))
        _application.add_handler(CommandHandler("orb_on", h_orb_on))
        _application.add_handler(CommandHandler("orb_off", h_orb_off))
        _application.add_handler(CommandHandler("tf", h_tf))
        _application.add_handler(CommandHandler("panic", h_panic))
        _application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), h_echo))

        await set_commands(_application)

        # Outbox-worker
        asyncio.create_task(outbox_worker(_application))

        # Starta REST-poll i separat tråd
        def poll_thread():
            try:
                LOG.info("REST poll loop startar")
                poll_loop()
            except Exception:
                traceback.print_exc()
        threading.Thread(target=poll_thread, daemon=True).start()

        LOG.info("PTB run_polling startar ...")
        await _application.run_polling(drop_pending_updates=True)

    asyncio.run(_amain())

# =================== FastAPI (Render) ===================
app = FastAPI()
_bot_thread: Optional[threading.Thread] = None

@app.on_event("startup")
def startup():
    global _bot_thread
    if _bot_thread is None:
        _bot_thread = threading.Thread(target=_run_bot_and_poll, daemon=True)
        _bot_thread.start()
        log_activity("-", "TG_STARTED", "PTB v20 polling + REST poll loop")

@app.get("/healthz")
def healthz():
    return {
        "status":"ok","engine":STATE.trading_enabled,"mode":STATE.mode,
        "ai":STATE.ai_mode,"tf_min":STATE.tf_min,
        "trail":STATE.trailing_enabled,"orb":STATE.use_orb,"closebreak":STATE.use_closebreak,
        "chat_id": STATE.chat_id
    }

@app.get("/")
def root():
    return {"name":"Mp ORBbot","version":"v21","note":"TF 1/3/5/15, BUY/SELL-kort, webhook-kill, no JobQueue"}

# ---------- Debug endpoints för Telegram-nät/Token ----------
TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

@app.get("/tg_me")
def tg_me():
    try:
        me = requests.get(f"{TG_API}/getMe", timeout=10).json()
        info = requests.get(f"{TG_API}/getWebhookInfo", timeout=10).json()
        return {"getMe": me, "webhookInfo": info}
    except Exception as e:
        return {"error": str(e)}

@app.get("/tg_delete_webhook")
def tg_delete_webhook():
    try:
        r = requests.get(f"{TG_API}/deleteWebhook?drop_pending_updates=true", timeout=10).json()
        return r
    except Exception as e:
        return {"error": str(e)}

@app.get("/tg_test")
def tg_test(chat_id: int, text: str = "hello from v21"):
    try:
        r = requests.post(f"{TG_API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=10).json()
        return r
    except Exception as e:
        return {"error": str(e)}
