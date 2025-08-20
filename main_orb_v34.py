# main_orb_v34.py
import os, asyncio, logging, time
from typing import Dict, Optional, Literal
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("orb_v34")

# === ENV ===
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "5397586616") or "5397586616")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "")
if not BOT_TOKEN: raise RuntimeError("Saknar BOT_TOKEN")
if not WEBHOOK_BASE: raise RuntimeError("Saknar WEBHOOK_BASE (ex: https://<din-app>.onrender.com)")

# === STATE (in-memory) ===
class State(BaseModel):
    engine_on: bool = False
    entry_mode: Literal["close","tick"] = "close"   # starta i close-läge
    timeframe: Literal["1m","3m","5m","15m"] = "3m"
    allow_shorts: bool = False
    # ORB-tracking per symbol
    pos_side: Optional[Literal["LONG","SHORT"]] = None
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    in_orb_phase: bool = False
    last_candle_ts: Optional[int] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None

STATE: Dict[str, State] = {}
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").replace(" ", "").split(",")

# === Mock prisfeed (byt mot din realtid) ===
import random
_prices: Dict[str, float] = {s: 100.0 + random.random()*10 for s in SYMBOLS}
def get_tick_price(symbol: str) -> float:
    # Byt mot riktig ticker (KuCoin/Binance)
    # Här simuleras lite rörelse:
    _prices[symbol] += random.uniform(-0.2, 0.2)
    return round(_prices[symbol], 2)

# === Hjälp: klaviatur ===
def main_menu_kb(st: State) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Engine ON" if not st.engine_on else "Engine OFF", callback_data="toggle_engine")],
        [
            InlineKeyboardButton("Entry: CLOSE", callback_data="entry_close"),
            InlineKeyboardButton("Entry: TICK",  callback_data="entry_tick"),
        ],
        [
            InlineKeyboardButton(f"TF: {st.timeframe}", callback_data="noop"),
            InlineKeyboardButton("1m", callback_data="tf_1m"),
            InlineKeyboardButton("3m", callback_data="tf_3m"),
            InlineKeyboardButton("5m", callback_data="tf_5m"),
            InlineKeyboardButton("15m", callback_data="tf_15m"),
        ],
        [
            InlineKeyboardButton(f"Shorts: {'ON' if st.allow_shorts else 'OFF'}", callback_data="toggle_shorts")
        ],
    ])

# === Telegram setup ===
app = FastAPI()
tg_app: Application

class TGUpdate(BaseModel):
    update_id: int | None = None
    message: dict | None = None
    edited_message: dict | None = None
    callback_query: dict | None = None

@app.on_event("startup")
async def on_startup():
    global tg_app
    log.info("Initierar Telegram Application...")
    tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Handlers
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("menu", cmd_menu))
    tg_app.add_handler(CallbackQueryHandler(on_button))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    await tg_app.initialize()
    await tg_app.start()
    await tg_app.bot.set_webhook(f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}")
    log.info("Webhook satt: %s/webhook/%s", WEBHOOK_BASE, BOT_TOKEN)

    # Init state per symbol
    for s in SYMBOLS:
        STATE[s] = State()

    # Start “engine”-task
    asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()

@app.get("/")
async def root():
    return {"ok": True, "service": "orb_v34"}

@app.post(f"/webhook/{BOT_TOKEN}")
async def webhook(req: Request):
    data = await req.json()
    upd = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(upd)
    return {"ok": True}

# === Kommandon ===
async def cmd_start(update: Update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    await update.message.reply_text(
        "Mp ORBbot v34 — endast LONG + SHORT enligt dina regler.\n"
        "Startläge: Entry=CLOSE, Shorts=OFF, TF=3m, Engine=OFF.\n"
        "Använd /menu för knappar.",
    )

async def cmd_menu(update: Update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    # visa första symbolens state i UI (alla delar samma knappar)
    st = STATE[SYMBOLS[0]]
    await update.message.reply_text("Kontrollpanel:", reply_markup=main_menu_kb(st))

async def on_text(update: Update, context):
    if update.effective_chat.id != OWNER_CHAT_ID:
        return
    await update.message.reply_text("Använd /menu för knappar.")

# === Knappar ===
async def on_button(update: Update, context):
    if update.effective_user.id != OWNER_CHAT_ID:
        await update.callback_query.answer("Endast ägaren kan ändra.")
        return
    q = update.callback_query
    data = q.data
    # Alla symboler delar samma settings (enkelt): kopiera till alla states
    st0 = STATE[SYMBOLS[0]]

    if data == "toggle_engine":
        new_val = not st0.engine_on
        for s in SYMBOLS:
            STATE[s].engine_on = new_val
        await q.answer("Engine " + ("ON" if new_val else "OFF"))
        await q.edit_message_reply_markup(reply_markup=main_menu_kb(STATE[SYMBOLS[0]]))
    elif data == "entry_close":
        for s in SYMBOLS: STATE[s].entry_mode = "close"
        await q.answer("Entry = CLOSE")
        await q.edit_message_reply_markup(reply_markup=main_menu_kb(STATE[SYMBOLS[0]]))
    elif data == "entry_tick":
        for s in SYMBOLS: STATE[s].entry_mode = "tick"
        await q.answer("Entry = TICK")
        await q.edit_message_reply_markup(reply_markup=main_menu_kb(STATE[SYMBOLS[0]]))
    elif data.startswith("tf_"):
        tf = data.split("_")[1]
        for s in SYMBOLS: STATE[s].timeframe = tf
        await q.answer(f"Timeframe = {tf}")
        await q.edit_message_reply_markup(reply_markup=main_menu_kb(STATE[SYMBOLS[0]]))
    elif data == "toggle_shorts":
        new_val = not st0.allow_shorts
        for s in SYMBOLS: STATE[s].allow_shorts = new_val
        await q.answer("Shorts " + ("ON" if new_val else "OFF"))
        await q.edit_message_reply_markup(reply_markup=main_menu_kb(STATE[SYMBOLS[0]]))
    else:
        await q.answer("OK")

# === ORB/Entry/Stop (din specifikation) ===
# - För LONG: ORB = första gröna candle efter en röd; entry när priset bryter/stänger över ORB-high
# - För SHORT: ORB = första röda candle efter en grön; entry när priset bryter/stänger under ORB-low
# - Stop sätts på motsatt ORB-nivå
# - Stop trailas till varje ny candle (för LONG: stop = förra candlens low, men aldrig nedåt; för SHORT: stop = förra candlens high, men aldrig uppåt)

class Candle(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float

_last_candle: Dict[str, Candle] = {}
_prev_candle: Dict[str, Candle] = {}

def new_candle(symbol: str) -> Candle:
    # mocka fram candle från tick (ersätt med riktig aggregator per timeframe)
    now = int(time.time())
    o = get_tick_price(symbol)
    h = o + random.uniform(0.0, 0.6)
    l = o - random.uniform(0.0, 0.6)
    c = round(random.choice([h, l, (h+l)/2]), 2)  # stäng slumpat inom HL
    return Candle(ts=now, open=o, high=round(h,2), low=round(l,2), close=c)

def is_green(c: Candle) -> bool: return c.close > c.open
def is_red(c: Candle) -> bool:   return c.close < c.open

def timeframe_seconds(tf: str) -> int:
    return {"1m":60, "3m":180, "5m":300, "15m":900}[tf]

async def engine_loop():
    # Enkel loop som “skapar” en ny candle per timeframe och utvärderar reglerna
    while True:
        for symbol in SYMBOLS:
            st = STATE[symbol]
            secs = timeframe_seconds(st.timeframe)
            # skapa ny candle per TF
            c = new_candle(symbol)
            _prev_candle[symbol] = _last_candle.get(symbol, c)
            _last_candle[symbol] = c

            if st.engine_on:
                await evaluate(symbol, st, _prev_candle[symbol], c)

        await asyncio.sleep(4)  # snabbare än TF bara för demo

async def evaluate(symbol: str, st: State, prev: Candle, cur: Candle):
    # 1) Om ingen position: bygg ORB och försök trigga entry
    if st.pos_side is None:
        # om vi inte är i ORB-fas: leta start-candle
        if not st.in_orb_phase:
            # LONG-ORB om vi ser en grön candle efter en röd
            if is_red(prev) and is_green(cur):
                st.in_orb_phase = True
                st.orb_high = max(cur.open, cur.close)  # grönens "kropp"-high (du ville själva candle som ORB)
                st.orb_low  = min(cur.open, cur.close)
                st.last_candle_ts = cur.ts
                await notify(f"{symbol}: Startar LONG-ORB. ORB [{st.orb_low} .. {st.orb_high}]")
            # SHORT-ORB om shorts är på och röd efter grön
            elif st.allow_shorts and is_green(prev) and is_red(cur):
                st.in_orb_phase = True
                st.orb_high = max(cur.open, cur.close)
                st.orb_low  = min(cur.open, cur.close)
                st.last_candle_ts = cur.ts
                await notify(f"{symbol}: Startar SHORT-ORB. ORB [{st.orb_low} .. {st.orb_high}]")
        else:
            # Vi är i ORB-fas: kolla breakouts beroende på entry_mode
            px_tick = get_tick_price(symbol)
            px_close = cur.close
            go_long = go_short = False

            if st.entry_mode == "tick":
                if px_tick > (st.orb_high or 9e9): go_long = True
                if st.allow_shorts and px_tick < (st.orb_low or -9e9): go_short = True
            else:  # close
                # vänta tills candle stänger över/under — kör bara när cur är “ny”
                # här förenklat: vi använder cur.close direkt
                if px_close > (st.orb_high or 9e9): go_long = True
                if st.allow_shorts and px_close < (st.orb_low or -9e9): go_short = True

            if go_long:
                st.pos_side = "LONG"
                st.entry_price = px_tick if st.entry_mode=="tick" else px_close
                st.stop_price = st.orb_low
                await notify(f"{symbol} LONG @ {st.entry_price} | SL {st.stop_price} (ORB-high break)")
            elif go_short:
                st.pos_side = "SHORT"
                st.entry_price = px_tick if st.entry_mode=="tick" else px_close
                st.stop_price = st.orb_high
                await notify(f"{symbol} SHORT @ {st.entry_price} | SL {st.stop_price} (ORB-low break)")
    else:
        # 2) Om vi HAR position: traila stop för varje ny candle, och kolla stopp
        if st.pos_side == "LONG":
            # traila uppåt till föregående candlens low (aldrig nedåt)
            new_sl = max(st.stop_price or -9e9, (prev.low))
            if new_sl > (st.stop_price or -9e9):
                st.stop_price = round(new_sl, 2)
                await notify(f"{symbol} LONG trail -> SL {st.stop_price}")

            # stopp?
            last_px = get_tick_price(symbol)
            if last_px <= (st.stop_price or -9e9):
                await notify(f"{symbol} LONG stopped @ {st.stop_price}")
                # reset
                STATE[symbol] = State(
                    engine_on=st.engine_on,
                    entry_mode=st.entry_mode,
                    timeframe=st.timeframe,
                    allow_shorts=st.allow_shorts
                )

        elif st.pos_side == "SHORT":
            # traila nedåt till föregående candlens high (aldrig uppåt)
            new_sl = min(st.stop_price or 9e9, (prev.high))
            if (st.stop_price is None) or (new_sl < st.stop_price):
                st.stop_price = round(new_sl, 2)
                await notify(f"{symbol} SHORT trail -> SL {st.stop_price}")

            last_px = get_tick_price(symbol)
            if last_px >= (st.stop_price or 9e9):
                await notify(f"{symbol} SHORT stopped @ {st.stop_price}")
                STATE[symbol] = State(
                    engine_on=st.engine_on,
                    entry_mode=st.entry_mode,
                    timeframe=st.timeframe,
                    allow_shorts=st.allow_shorts
                )

async def notify(text: str):
    try:
        await tg_app.bot.send_message(chat_id=OWNER_CHAT_ID, text=text)
    except Exception as e:
        log.warning("Notify fail: %s", e)
