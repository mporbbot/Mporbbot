# main_19.py — Mp ORBbot (PTB 13.15) med hård ORB-stop som trailas upp candle för candle
# - Entry: ORB-long + CloseBreak (valbara)
# - Stop: sätts på ORB-low direkt vid entry; flyttas UPP till varje färdig candles low om den är högre (aldrig ner)
# - Exit: sälj när priset når stop
# - FastAPI /healthz för Render; Telegram-kommandon fungerar

import os, csv, time, threading, traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import requests
from fastapi import FastAPI
from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters

# ---------------- Konfig ----------------
SYMBOLS = ["LINKUSDT","XRPUSDT","ADAUSDT","BTCUSDT","ETHUSDT"]
TRADE_SIZE_USDT = 30.0
FEE_RATE = 0.001
AI_DEFAULT = "neutral"        # aggressiv | neutral | försiktig
DOJI_FILTER = True
DOJI_BODY_PCT = 0.10          # <10% av range => Doji
BREAKOUT_BUFFER = 0.001       # +0.1% över ORB-high
POLL_INTERVAL_SEC = 20

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8079688612:AAGM-6vTQ6R_ZSdfnQWD0LCqcmS7_zk46Us")
KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

LOG_DIR = "logs"
MOCK_LOG = os.path.join(LOG_DIR, "mock_trade_log.csv")
REAL_LOG = os.path.join(LOG_DIR, "real_trade_log.csv")
ACTIVITY_LOG = os.path.join(LOG_DIR, "activity_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

def _ensure_csv(p, header):
    if not os.path.exists(p):
        with open(p,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(header)

_ensure_csv(MOCK_LOG, ["time_utc","symbol","side","qty","price","fee","ai_mode","reason"])
_ensure_csv(REAL_LOG, ["time_utc","symbol","side","qty","price","fee","ai_mode","reason"])
_ensure_csv(ACTIVITY_LOG, ["time_utc","symbol","event","details"])

def log_activity(symbol, event, details):
    with open(ACTIVITY_LOG,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(),symbol,event,details])

def candle_color(o,c): return "green" if c>=o else "red"
def is_doji(o,h,l,c,body_pct=DOJI_BODY_PCT):
    rng=max(h-l,1e-12); body=abs(c-o); return (body/rng)<=body_pct
def ku_symbol(s): s=s.upper(); return s.replace("USDT","-USDT") if "USDT" in s and "-" not in s else s

# ---------------- Typer & State ----------------
@dataclass
class Candle:
    ts:int; o:float; h:float; l:float; c:float; v:float

@dataclass
class ORB:
    start_ts:int; high:float; low:float; base:Candle

@dataclass
class Position:
    side:str; entry:float; qty:float; orb_at_entry:ORB
    stop: Optional[float]=None            # vår hårda stop (trailas upp)
    active: bool = True

@dataclass
class BotState:
    trading_enabled: bool = False
    mode: str = "mock"                    # mock | live
    ai_mode: str = AI_DEFAULT
    epoch: int = 0

    # strategiflaggor & pnl
    use_orb: bool = True
    use_closebreak: bool = True
    trailing_enabled: bool = True         # om false → stop uppdateras ej
    pnl_offset: float = 0.0

    positions: Dict[str, Optional[Position]] = field(default_factory=dict)
    orbs: Dict[str, Optional[ORB]] = field(default_factory=dict)
    prev_seen_color: Dict[str, Optional[str]] = field(default_factory=dict)
    shift_armed: Dict[str, bool] = field(default_factory=dict)
    last_candle_ts: Dict[str, Optional[int]] = field(default_factory=dict)
    last_closed_low: Dict[str, Optional[float]] = field(default_factory=dict)  # för "föregående candle low"

    def reset_symbol(self, s:str):
        self.positions[s]=None; self.orbs[s]=None
        self.prev_seen_color[s]=None; self.shift_armed[s]=False
        self.last_candle_ts[s]=None; self.last_closed_low[s]=None

STATE = BotState()
for s in SYMBOLS: STATE.reset_symbol(s)

# ---------------- Order/PnL ----------------
def kucoin_min_qty(symbol, price):
    return round(5.0/max(price,1e-9),6)

def log_trade(mock, symbol, side, qty, price, fee, ai_mode, reason):
    path = MOCK_LOG if mock else REAL_LOG
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(),symbol,side,f"{qty:.8f}",f"{price:.8f}",f"{fee:.8f}",ai_mode,reason])

def place_order(symbol, side, price, usdt_amount, reason):
    # central guard
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
    elif side.lower()=="sell" and STATE.positions.get(symbol):
        pos = STATE.positions[symbol]
        pnl = (price - pos.entry) * pos.qty - fee
        STATE.positions[symbol] = None
        log_activity(symbol, "POSITION_CLOSE", f"close={price} pnl={pnl:.8f} reason={reason}")
    return True

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

# ---------------- ORB & logik ----------------
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

    # ORB-long
    if STATE.use_orb and c.h>=trig and ai_allows(0.6):
        place_order(symbol,"buy",max(trig,c.o),TRADE_SIZE_USDT,"ORB_LONG_BREAK"); return

    # CloseBreak
    if STATE.use_closebreak and c.o<orb.high and c.c>=trig and ai_allows(0.6):
        place_order(symbol,"buy",c.c,TRADE_SIZE_USDT,"CloseBreak")

def update_stop_trailing(symbol: str, prev_closed_low: Optional[float], curr: Candle):
    """Höj hård stop till föregående candles low (eller ORB-low initialt). Sälj om träffad."""
    if not STATE.trailing_enabled: return
    pos = STATE.positions.get(symbol)
    if not pos or not pos.active: return

    # initial säkerhet
    if pos.stop is None and pos.orb_at_entry:
        pos.stop = pos.orb_at_entry.low

    # 1) Höj stop till föregående candle low om den är högre än nuvarande stop
    if prev_closed_low is not None and (pos.stop is None or prev_closed_low > pos.stop):
        pos.stop = prev_closed_low
        log_activity(symbol, "STOP_RAISE", f"new_stop={pos.stop:.6f}")

    # 2) Stop-hit: om nuvarande candle (som byggs) går ned till stop → stäng
    if pos.stop is not None and curr.l <= pos.stop:
        place_order(symbol, "sell", pos.stop, pos.qty * pos.entry, "HARD_STOP_HIT")

def process_candle(symbol: str, prev: Optional[Candle], curr: Candle):
    # Sätt ORB om saknas
    if STATE.orbs.get(symbol) is None:
        start_new_orb(symbol, curr)

    # Färgskifte ⇒ ny ORB på första candlen efter skiftet
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

    # Entries + trailing stop
    try_breakouts(symbol, curr)
    update_stop_trailing(symbol, STATE.last_closed_low.get(symbol), curr)

    # Spara low på denna färdiga candle som "föregående" för nästa varv
    STATE.last_closed_low[symbol] = curr.l

# ---------------- Data (KuCoin REST 3m) ----------------
def fetch_kucoin_3m(symbol, limit=50)->List[Candle]:
    url=f"{KUCOIN_BASE_URL}/api/v1/market/candles"
    params={"type":"3min","symbol":ku_symbol(symbol)}
    r=requests.get(url,params=params,timeout=10); r.raise_for_status()
    arr=r.json().get("data",[])
    out=[]
    for it in reversed(arr[-limit:]):   # äldst → nyast
        t_end=int(float(it[0])); o=float(it[1]); c=float(it[2]); h=float(it[3]); l=float(it[4]); v=float(it[5])
        ts=(t_end-180)*1000
        out.append(Candle(ts,o,h,l,c,v))
    return out

def poll_loop():
    while True:
        try:
            for s in SYMBOLS:
                candles=fetch_kucoin_3m(s,50)
                last=STATE.last_candle_ts.get(s)
                prev=None
                for c in candles:
                    if last is None or c.ts>last:
                        process_candle(s, prev, c)
                        STATE.last_candle_ts[s]=c.ts
                    prev=c
            time.sleep(POLL_INTERVAL_SEC)
        except Exception as e:
            log_activity("-", "POLL_ERROR", str(e)); time.sleep(5)

# ---------------- Telegram-kommandon ----------------
HELP = (
    "/engine_start – slå på handel\n"
    "/engine_stop – stäng av handel\n"
    "/start_mock – byt till MOCK och slå på handel\n"
    "/start_live – byt till LIVE och slå på handel\n"
    "/status – visa läge, positioner, ORB och stop\n"
    "/entry_mode [orb|closebreak|both]\n"
    "/trailing [on|off]  (utan argument: toggle)\n"
    "/pnl – dagens PnL\n"
    "/reset_pnl – nollställer visad PnL (offset)\n"
    "/orb_on  /orb_off\n"
    "/panic – stäng alla positioner och slå av engine\n"
)

def tg_start(u:Update,c:CallbackContext): u.message.reply_text("Mp ORBbot igång.\n"+HELP)

def tg_engine_start(u:Update,c:CallbackContext):
    STATE.trading_enabled=True; u.message.reply_text("✅ Engine: ON")

def tg_engine_stop(u:Update,c:CallbackContext):
    STATE.trading_enabled=False; STATE.epoch+=1
    u.message.reply_text("🛑 Engine: OFF – alla signaler blockeras.")
    log_activity("-", "ENGINE_OFF", "manual")

def tg_start_mock(u:Update,c:CallbackContext):
    STATE.mode="mock"; STATE.trading_enabled=True; u.message.reply_text("✅ MOCK-läge: ON")

def tg_start_live(u:Update,c:CallbackContext):
    STATE.mode="live"; STATE.trading_enabled=True; u.message.reply_text("⚠️ LIVE-läge: ON")

def tg_status(u:Update,c:CallbackContext):
    lines=[f"Engine: {'ON' if STATE.trading_enabled else 'OFF'}",
           f"Läge: {STATE.mode.upper()}",
           f"AI: {STATE.ai_mode}",
           f"Trailing: {'ON' if STATE.trailing_enabled else 'OFF'}",
           f"EntryModes: ORB={'ON' if STATE.use_orb else 'OFF'}, CloseBreak={'ON' if STATE.use_closebreak else 'OFF'}"]
    for s in SYMBOLS:
        orb=STATE.orbs.get(s); pos=STATE.positions.get(s)
        lines.append(f"{s} ORB: {'-' if not orb else f'H={orb.high:.6f} L={orb.low:.6f}'}")
        if not pos:
            lines.append("  Pos: -")
        else:
            lines.append(f"  Pos: long qty={pos.qty:.6f} entry={pos.entry:.6f} stop={pos.stop}")
    u.message.reply_text("\n".join(lines))

def tg_entry_mode(u:Update,c:CallbackContext):
    if c.args:
        arg=c.args[0].lower()
        if arg=="orb": STATE.use_orb=True; STATE.use_closebreak=False
        elif arg=="closebreak": STATE.use_orb=False; STATE.use_closebreak=True
        elif arg=="both": STATE.use_orb=True; STATE.use_closebreak=True
        else: u.message.reply_text("Använd: /entry_mode [orb|closebreak|both]"); return
        u.message.reply_text(f"Entry mode: ORB={'ON' if STATE.use_orb else 'OFF'}, CloseBreak={'ON' if STATE.use_closebreak else 'OFF'}")
    else:
        u.message.reply_text(f"Aktuell: ORB={'ON' if STATE.use_orb else 'OFF'}, CloseBreak={'ON' if STATE.use_closebreak else 'OFF'}")

def tg_trailing(u:Update,c:CallbackContext):
    if c.args:
        v=c.args[0].lower()
        if v in ("on","off"): STATE.trailing_enabled=(v=="on")
        else: u.message.reply_text("Använd: /trailing [on|off]"); return
    else:
        STATE.trailing_enabled=not STATE.trailing_enabled
    u.message.reply_text(f"Trailing (stop-raise): {'ON' if STATE.trailing_enabled else 'OFF'}")

def todays_pnl_from_log_safe():
    return todays_pnl_from_log(MOCK_LOG if STATE.mode=="mock" else REAL_LOG)

def tg_pnl(u:Update,c:CallbackContext):
    u.message.reply_text(f"Dagens PnL ({STATE.mode.upper()}): {current_pnl_with_offset():.2f} USDT")

def tg_reset_pnl(u:Update,c:CallbackContext):
    STATE.pnl_offset = todays_pnl_from_log_safe()
    u.message.reply_text("PnL nollställd (offset satt).")

def tg_orb_on(u:Update,c:CallbackContext): STATE.use_orb=True; u.message.reply_text("ORB: ON")
def tg_orb_off(u:Update,c:CallbackContext): STATE.use_orb=False; u.message.reply_text("ORB: OFF")

def tg_panic(u:Update,c:CallbackContext):
    closed=0
    for s,pos in list(STATE.positions.items()):
        if pos:
            place_order(s,"sell",pos.stop if pos.stop is not None else pos.entry, pos.qty*pos.entry, "PANIC_CLOSE")
            closed+=1
    STATE.trading_enabled=False
    u.message.reply_text(f"PANIC – stängde {closed} pos. Engine OFF.")

class TelegramRunner:
    def __init__(self):
        self.updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
        dp=self.updater.dispatcher
        dp.add_handler(CommandHandler("start", tg_start))
        dp.add_handler(CommandHandler("engine_start", tg_engine_start))
        dp.add_handler(CommandHandler("engine_stop", tg_engine_stop))
        dp.add_handler(CommandHandler("start_mock", tg_start_mock))
        dp.add_handler(CommandHandler("start_live", tg_start_live))
        dp.add_handler(CommandHandler("status", tg_status))
        dp.add_handler(CommandHandler("entry_mode", tg_entry_mode, pass_args=True))
        dp.add_handler(CommandHandler("trailing", tg_trailing, pass_args=True))
        dp.add_handler(CommandHandler("pnl", tg_pnl))
        dp.add_handler(CommandHandler("reset_pnl", tg_reset_pnl))
        dp.add_handler(CommandHandler("orb_on", tg_orb_on))
        dp.add_handler(CommandHandler("orb_off", tg_orb_off))
        dp.add_handler(CommandHandler("panic", tg_panic))
        dp.add_handler(MessageHandler(Filters.text & (~Filters.command), lambda u,c: None))

    def run(self):
        self.updater.start_polling(drop_pending_updates=True)
        self.updater.idle()

# ---------------- Poll & FastAPI ----------------
def poll_thread_fn():
    try: poll_loop()
    except Exception: traceback.print_exc()

app = FastAPI()
_tg_thread=None; _poll_thread=None; _runner=None

@app.on_event("startup")
def startup():
    global _tg_thread,_poll_thread,_runner
    if _runner is None:
        _runner=TelegramRunner()
        _tg_thread=threading.Thread(target=_runner.run,daemon=True); _tg_thread.start()
        log_activity("-", "TG_STARTED", "polling")
    if _poll_thread is None:
        _poll_thread=threading.Thread(target=poll_thread_fn,daemon=True); _poll_thread.start()
        log_activity("-", "POLL_STARTED", "kucoin REST 3m")

@app.get("/healthz")
def healthz():
    return {
        "status":"ok","engine":STATE.trading_enabled,"mode":STATE.mode,
        "ai":STATE.ai_mode,"trailing":STATE.trailing_enabled,
        "orb":STATE.use_orb,"closebreak":STATE.use_closebreak
    }

@app.get("/")
def root():
    return {"name":"Mp ORBbot","msg":"Service up","note":"Stop = ORB-low, trailas till föregående candle-low"}
