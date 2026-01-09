import os
import time
import json
import csv
import hmac
import base64
import hashlib
import random
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import aiohttp
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# =========================================================
# ENV
# =========================================================
def get_telegram_token() -> str:
    # Accept both names to avoid deployment mismatch
    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Sätt TELEGRAM_TOKEN (eller TELEGRAM_BOT_TOKEN) i environment.")
    return token


KUCOIN_KEY = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_SECRET = os.getenv("KUCOIN_API_SECRET", "")
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

# =========================================================
# FILES
# =========================================================
MOCK_LOG = "mock_trade_log.csv"
REAL_LOG = "real_trade_log.csv"

TRADE_HEADERS = [
    "timestamp_unix",
    "timestamp_iso",
    "exchange",
    "mode",
    "symbol",
    "strategy",
    "side",
    "qty_base",
    "stake_usdt",
    "entry_price",
    "exit_price",
    "gross_pnl_usdt",
    "fees_usdt",
    "net_pnl_usdt",
    "reason",
    "notes",
]

def ensure_csv(path: str):
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRADE_HEADERS)

ensure_csv(MOCK_LOG)
ensure_csv(REAL_LOG)

# =========================================================
# TELEGRAM MENU (ReplyKeyboard)
# =========================================================
MENU = ReplyKeyboardMarkup(
    [
        ["/status", "/pnl"],
        ["/engine_on", "/engine_off"],
        ["/threshold", "/stake"],
        ["/tp", "/sl"],
        ["/coins", "/trade_mode"],
        ["/notify", "/export_csv"],
        ["/close_all", "/reset_pnl"],
        ["/safe_live", "/help"],
    ],
    resize_keyboard=True
)

# =========================================================
# UTILS
# =========================================================
def now_ts() -> float:
    return time.time()

def iso_now(ts: Optional[int] = None) -> str:
    if ts is None:
        ts = int(now_ts())
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))

def fmt(x: float, n: int = 6) -> str:
    return f"{x:.{n}f}"

def pct(x: float) -> str:
    return f"{x:.2f}%"

# =========================================================
# KUCOIN CLIENT (market data + optional live orders)
# =========================================================
class KuCoinClient:
    BASE = "https://api.kucoin.com"

    def __init__(self, session: aiohttp.ClientSession):
        self.s = session

    async def get_level1(self, symbol: str) -> Optional[Dict]:
        url = f"{self.BASE}/api/v1/market/orderbook/level1"
        try:
            async with self.s.get(url, params={"symbol": symbol}, timeout=10) as r:
                js = await r.json()
                if js.get("code") != "200000":
                    return None
                return js.get("data")
        except Exception:
            return None

    async def get_server_time(self) -> int:
        url = f"{self.BASE}/api/v1/timestamp"
        async with self.s.get(url, timeout=10) as r:
            js = await r.json()
            return int(js["data"])

    def live_ready(self) -> bool:
        return bool(KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE)

    def _sign(self, ts_ms: int, method: str, endpoint: str, body: str = "") -> Dict[str, str]:
        str_to_sign = f"{ts_ms}{method.upper()}{endpoint}{body}"
        signature = base64.b64encode(
            hmac.new(KUCOIN_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).digest()
        ).decode()

        passphrase = base64.b64encode(
            hmac.new(KUCOIN_SECRET.encode(), KUCOIN_PASSPHRASE.encode(), hashlib.sha256).digest()
        ).decode()

        return {
            "KC-API-KEY": KUCOIN_KEY,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": str(ts_ms),
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    async def place_limit(self, symbol: str, side: str, price: float, size: float, post_only: bool = True) -> Optional[str]:
        if not self.live_ready():
            return None
        endpoint = "/api/v1/orders"
        url = f"{self.BASE}{endpoint}"
        ts = await self.get_server_time()

        body_dict = {
            "clientOid": f"mpmm-{int(time.time()*1000)}-{random.randint(1000,9999)}",
            "side": side,
            "symbol": symbol,
            "type": "limit",
            "price": fmt(price, 8),
            "size": fmt(size, 8),
            "postOnly": post_only,
        }
        body = json.dumps(body_dict)
        headers = self._sign(ts, "POST", endpoint, body)

        try:
            async with self.s.post(url, data=body, headers=headers, timeout=15) as r:
                js = await r.json()
                if js.get("code") != "200000":
                    return None
                return js["data"]["orderId"]
        except Exception:
            return None

    async def get_order(self, order_id: str) -> Optional[Dict]:
        if not self.live_ready():
            return None
        endpoint = f"/api/v1/orders/{order_id}"
        url = f"{self.BASE}{endpoint}"
        ts = await self.get_server_time()
        headers = self._sign(ts, "GET", endpoint, "")
        try:
            async with self.s.get(url, headers=headers, timeout=15) as r:
                js = await r.json()
                if js.get("code") != "200000":
                    return None
                return js.get("data")
        except Exception:
            return None

# =========================================================
# BOT STATE
# =========================================================
DEFAULT_COINS = ["LINK-USDT", "ADA-USDT", "XRP-USDT"]

@dataclass
class Settings:
    engine_on: bool = False
    trade_mode: str = "mock"   # mock | live
    coins: List[str] = field(default_factory=lambda: DEFAULT_COINS.copy())
    notify: bool = True

    # Spread MM params
    min_spread: float = 0.20   # % (threshold)
    target: float = 0.15       # % (tp)
    panic: float = 0.60        # % (sl)
    stake: float = 30.0        # USDT per cycle
    inv_cap: float = 60.0      # max inventory value per coin in USDT

    order_timeout: float = 20.0
    requote: float = 5.0
    loop: float = 3.0
    coin_cooldown: float = 30.0

    # Mock realism & fees
    maker_fee: float = 0.001   # 0.10%
    taker_fee: float = 0.001   # 0.10%
    mock_slip: float = 0.0002  # 0.02%

    fill_base: float = 0.08
    fill_spread_boost: float = 0.30
    fill_price_band: float = 0.15  # %


SAFE_LIVE_PRESET = {
    "min_spread": 0.25,
    "target": 0.20,
    "panic": 0.60,
    "stake": 5.0,
    "inv_cap": 10.0,
    "order_timeout": 25.0,
    "requote": 8.0,
    "loop": 5.0,
    "coin_cooldown": 45.0,
}

@dataclass
class OrderState:
    side: str                 # buy | sell
    price: float
    size: float
    placed_ts: float
    last_requote_ts: float
    order_id: Optional[str] = None
    filled: bool = False
    fill_price: Optional[float] = None

@dataclass
class CoinState:
    in_cycle: bool = False
    last_cycle_close_ts: float = 0.0
    inventory_base: float = 0.0

    buy: Optional[OrderState] = None
    sell: Optional[OrderState] = None

    entry_price: Optional[float] = None
    entry_size: Optional[float] = None

    cycles_closed: int = 0
    net_pnl: float = 0.0

@dataclass
class BotState:
    chat_id: Optional[int] = None
    settings: Settings = field(default_factory=Settings)
    coins: Dict[str, CoinState] = field(default_factory=dict)

    total_cycles: int = 0
    total_net_pnl: float = 0.0

    pending_live_confirm: bool = False

STATE = BotState()

# =========================================================
# LOGGING
# =========================================================
def log_trade(mode: str, symbol: str, side: str, qty: float, stake: float,
              entry: float, exitp: float, gross: float, fees: float, net: float,
              reason: str, notes: str):

    path = MOCK_LOG if mode == "mock" else REAL_LOG
    ts = int(now_ts())
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            ts,
            iso_now(ts),
            "KuCoin",
            mode,
            symbol,
            "Spread Market Maker",
            side,
            fmt(qty, 12),
            fmt(stake, 2),
            fmt(entry, 12),
            fmt(exitp, 12),
            fmt(gross, 8),
            fmt(fees, 8),
            fmt(net, 8),
            reason,
            notes,
        ])

# =========================================================
# MOCK FILL REALISM
# =========================================================
def mock_fee(notional_usdt: float, maker: bool, s: Settings) -> float:
    return notional_usdt * (s.maker_fee if maker else s.taker_fee)

def fill_chance(spread_pct: float, s: Settings) -> float:
    boost = min(s.fill_spread_boost, (spread_pct / max(s.min_spread, 0.01)) * (s.fill_spread_boost / 2.0))
    p = s.fill_base + boost
    return max(0.01, min(0.95, p))

def should_fill_limit(order: OrderState, mid: float, spread_pct: float, s: Settings) -> bool:
    # Needs both: price condition + probability
    band = s.fill_price_band / 100.0
    p = fill_chance(spread_pct, s)

    if order.side == "buy":
        if mid <= order.price:
            p = min(0.98, p + 0.35)
        elif mid <= order.price * (1 + band):
            p = min(0.98, p + 0.10)
        else:
            p = max(0.01, p * 0.25)
    else:
        if mid >= order.price:
            p = min(0.98, p + 0.35)
        elif mid >= order.price * (1 - band):
            p = min(0.98, p + 0.10)
        else:
            p = max(0.01, p * 0.25)

    return random.random() < p

# =========================================================
# MARKET DATA
# =========================================================
async def get_quotes(kc: KuCoinClient, symbol: str) -> Optional[Tuple[float, float, float, float]]:
    d = await kc.get_level1(symbol)
    if not d:
        return None
    try:
        bid = float(d["bestBid"])
        ask = float(d["bestAsk"])
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(d.get("price") or 0)
        spread_pct = ((ask - bid) / mid) * 100.0 if mid > 0 else 0.0
        return bid, ask, mid, spread_pct
    except Exception:
        return None

# =========================================================
# TELEGRAM SEND (stable)
# =========================================================
APP: Optional[Application] = None

async def tg_send(text: str):
    if not APP or STATE.chat_id is None:
        return
    try:
        await APP.bot.send_message(chat_id=STATE.chat_id, text=text, reply_markup=MENU)
    except Exception:
        pass

# =========================================================
# STRATEGY CORE
# =========================================================
async def maybe_start_cycle(kc: KuCoinClient, symbol: str, cs: CoinState):
    s = STATE.settings
    t = now_ts()

    if cs.last_cycle_close_ts and (t - cs.last_cycle_close_ts) < s.coin_cooldown:
        return
    if cs.in_cycle:
        return

    quotes = await get_quotes(kc, symbol)
    if not quotes:
        return
    bid, ask, mid, spread_pct = quotes

    if spread_pct < s.min_spread:
        return

    inv_value = cs.inventory_base * mid
    if inv_value >= s.inv_cap:
        return

    stake = s.stake
    if stake <= 0:
        return
    qty = stake / bid if bid > 0 else 0.0
    if qty <= 0:
        return

    cs.in_cycle = True
    cs.buy = OrderState(side="buy", price=bid, size=qty, placed_ts=t, last_requote_ts=t)
    cs.sell = None
    cs.entry_price = None
    cs.entry_size = None

    if s.notify:
        await tg_send(f"🟢 PLACE BUY {symbol} @ {fmt(bid,4)} | spread={pct(spread_pct)} | stake={fmt(stake,2)}")

async def process_coin(kc: KuCoinClient, symbol: str, cs: CoinState):
    s = STATE.settings
    t = now_ts()

    quotes = await get_quotes(kc, symbol)
    if not quotes:
        return
    bid, ask, mid, spread_pct = quotes

    # if not in cycle, maybe start
    if not cs.in_cycle:
        await maybe_start_cycle(kc, symbol, cs)
        return

    # BUY stage
    if cs.buy and not cs.buy.filled:
        bo = cs.buy

        # requote
        if (t - bo.last_requote_ts) >= s.requote:
            bo.price = bid
            bo.last_requote_ts = t
            if s.notify:
                await tg_send(f"🔁 REQUOTE BUY {symbol} -> {fmt(bid,4)} | spread={pct(spread_pct)}")

        # timeout cancels cycle
        if (t - bo.placed_ts) > s.order_timeout:
            cs.in_cycle = False
            cs.buy = None
            cs.sell = None
            cs.last_cycle_close_ts = t
            if s.notify:
                await tg_send(f"⚪ BUY TIMEOUT {symbol} (no fill)")
            return

        if s.trade_mode == "mock":
            if should_fill_limit(bo, mid, spread_pct, s):
                bo.filled = True
                bo.fill_price = bo.price
        else:
            # LIVE: place & check
            if bo.order_id is None:
                oid = await kc.place_limit(symbol, "buy", bo.price, bo.size, post_only=True)
                bo.order_id = oid
                if oid and s.notify:
                    await tg_send(f"🟩 LIVE BUY SENT {symbol} oid={oid[-6:]}")
            else:
                od = await kc.get_order(bo.order_id)
                if od and od.get("isActive") is False:
                    deal_size = float(od.get("dealSize", "0") or 0)
                    deal_funds = float(od.get("dealFunds", "0") or 0)
                    if deal_size > 0 and deal_funds > 0:
                        bo.filled = True
                        bo.fill_price = deal_funds / deal_size

        # if filled -> create SELL
        if bo.filled and cs.entry_price is None:
            entry_price = bo.fill_price or bo.price
            qty = bo.size

            cs.entry_price = entry_price
            cs.entry_size = qty
            cs.inventory_base += qty

            target_price = entry_price * (1 + s.target / 100.0)
            cs.sell = OrderState(side="sell", price=target_price, size=qty, placed_ts=t, last_requote_ts=t)

            if s.notify:
                await tg_send(f"✅ BUY FILLED {symbol} @ {fmt(entry_price,4)} | SELL @ {fmt(target_price,4)}")

        return

    # SELL stage
    if cs.sell and cs.entry_price is not None and cs.entry_size is not None:
        so = cs.sell
        entry_price = cs.entry_price
        qty = cs.entry_size

        # Panic stop
        panic_level = entry_price * (1 - s.panic / 100.0)
        if mid <= panic_level:
            exit_price = bid * (1 - (s.mock_slip if s.trade_mode == "mock" else 0.0))
            gross = (exit_price - entry_price) * qty

            buy_notional = entry_price * qty
            sell_notional = exit_price * qty
            fees = mock_fee(buy_notional, maker=True, s=s) + mock_fee(sell_notional, maker=False, s=s)
            net = gross - fees

            cs.inventory_base -= qty
            cs.in_cycle = False
            cs.buy = None
            cs.sell = None
            cs.last_cycle_close_ts = t

            cs.cycles_closed += 1
            cs.net_pnl += net
            STATE.total_cycles += 1
            STATE.total_net_pnl += net

            log_trade(s.trade_mode, symbol, "MM_LONG", qty, s.stake, entry_price, exit_price, gross, fees, net,
                      "PANIC_STOP", f"mid={fmt(mid,4)} panic={fmt(panic_level,4)}")

            if s.notify:
                await tg_send(f"🟥 EXIT {symbol} @ {fmt(exit_price,4)} | Net {fmt(net,4)} USDT | PANIC_STOP")
            return

        # SELL timeout -> keep waiting but "refresh" timer (no aggressive behavior)
        if (t - so.placed_ts) > s.order_timeout:
            so.placed_ts = t
            if s.notify:
                await tg_send(f"⏳ SELL STILL WAIT {symbol} @ {fmt(so.price,4)}")

        # Fill SELL
        filled = False
        fill_price_val = None

        if s.trade_mode == "mock":
            if should_fill_limit(so, mid, spread_pct, s):
                filled = True
                fill_price_val = so.price
        else:
            if so.order_id is None:
                oid = await kc.place_limit(symbol, "sell", so.price, so.size, post_only=True)
                so.order_id = oid
                if oid and s.notify:
                    await tg_send(f"🟩 LIVE SELL SENT {symbol} oid={oid[-6:]}")
            else:
                od = await kc.get_order(so.order_id)
                if od and od.get("isActive") is False:
                    deal_size = float(od.get("dealSize", "0") or 0)
                    deal_funds = float(od.get("dealFunds", "0") or 0)
                    if deal_size > 0 and deal_funds > 0:
                        filled = True
                        fill_price_val = deal_funds / deal_size

        if filled and fill_price_val:
            exit_price = fill_price_val
            gross = (exit_price - entry_price) * qty

            buy_notional = entry_price * qty
            sell_notional = exit_price * qty
            fees = mock_fee(buy_notional, maker=True, s=s) + mock_fee(sell_notional, maker=True, s=s)
            net = gross - fees

            cs.inventory_base -= qty
            cs.in_cycle = False
            cs.buy = None
            cs.sell = None
            cs.last_cycle_close_ts = t

            cs.cycles_closed += 1
            cs.net_pnl += net
            STATE.total_cycles += 1
            STATE.total_net_pnl += net

            log_trade(s.trade_mode, symbol, "MM_LONG", qty, s.stake, entry_price, exit_price, gross, fees, net,
                      "TARGET_HIT", f"spread={pct(spread_pct)}")

            if s.notify:
                await tg_send(f"🎯 EXIT {symbol} @ {fmt(exit_price,4)} | Net {fmt(net,4)} USDT | TARGET")
            return

# =========================================================
# ENGINE LOOP (stable background task)
# =========================================================
async def engine_task():
    async with aiohttp.ClientSession() as session:
        kc = KuCoinClient(session)

        # init coin states
        for c in STATE.settings.coins:
            STATE.coins.setdefault(c, CoinState())

        while True:
            try:
                if STATE.settings.engine_on and STATE.chat_id is not None:
                    # ensure state for coins
                    for c in STATE.settings.coins:
                        STATE.coins.setdefault(c, CoinState())

                    # process
                    for c in list(STATE.settings.coins):
                        cs = STATE.coins.get(c)
                        if cs:
                            await process_coin(kc, c, cs)
            except Exception:
                # keep loop alive
                pass

            await asyncio.sleep(max(0.5, STATE.settings.loop))

# =========================================================
# COMMANDS
# =========================================================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    STATE.chat_id = update.effective_chat.id
    await update.message.reply_text("Mporbbot online ✅", reply_markup=MENU)

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Kommandon:\n"
        "/engine_on /engine_off\n"
        "/status /pnl\n"
        "/threshold <min spread %>\n"
        "/tp <target %>\n"
        "/sl <panic %>\n"
        "/stake <USDT>\n"
        "/coins <A B C...>\n"
        "/trade_mode mock|live (live kräver JA)\n"
        "/safe_live (preset)\n"
        "/export_csv\n"
        "/close_all /reset_pnl\n"
        "/notify\n"
    )
    await update.message.reply_text(txt, reply_markup=MENU)

async def engine_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    STATE.settings.engine_on = True
    await update.message.reply_text("✅ ENGINE ON", reply_markup=MENU)

async def engine_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    STATE.settings.engine_on = False
    await update.message.reply_text("⛔ ENGINE OFF", reply_markup=MENU)

async def notify_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    STATE.settings.notify = not STATE.settings.notify
    await update.message.reply_text(f"Notify: {'ON' if STATE.settings.notify else 'OFF'}", reply_markup=MENU)

async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = STATE.settings
    lines = []
    lines.append(f"ENGINE: {'ON' if s.engine_on else 'OFF'}")
    lines.append("Strategy: Spread Market Maker (realistic mock + safe live)")
    lines.append(f"Trade mode: {s.trade_mode} (live keys: {'OK' if (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE) else 'missing'})")
    lines.append(f"Min spread (/threshold): {pct(s.min_spread)}")
    lines.append(f"Target (/tp): {pct(s.target)}")
    lines.append(f"Panic stop (/sl): {pct(s.panic)}")
    lines.append(f"Stake per cycle: {fmt(s.stake,2)} USDT")
    lines.append(f"Inventory cap / coin: {fmt(s.inv_cap,2)} USDT")
    lines.append(f"Order timeout: {int(s.order_timeout)}s | requote: {int(s.requote)}s | loop: {int(s.loop)}s | cooldown/coin: {int(s.coin_cooldown)}s")
    lines.append(f"Trades (closed cycles): {STATE.total_cycles}")
    lines.append(f"Total NET PnL: {fmt(STATE.total_net_pnl,4)} USDT")
    lines.append(f"Coins ({len(s.coins)}): {s.coins}")

    for c in s.coins:
        cs = STATE.coins.get(c, CoinState())
        stage = "idle"
        if cs.in_cycle and cs.buy and not cs.buy.filled:
            stage = f"BUY waiting @{fmt(cs.buy.price,4)}"
        elif cs.in_cycle and cs.sell:
            stage = f"SELL waiting @{fmt(cs.sell.price,4)}"
        lines.append(f"{c}: {stage} | inv={fmt(cs.inventory_base,6)} | cycles={cs.cycles_closed} | net={fmt(cs.net_pnl,4)}")

    await update.message.reply_text("\n".join(lines), reply_markup=MENU)

async def pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Total NET PnL: {fmt(STATE.total_net_pnl,4)} USDT\nTrades: {STATE.total_cycles}",
        reply_markup=MENU
    )

async def reset_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    STATE.total_net_pnl = 0.0
    STATE.total_cycles = 0
    for cs in STATE.coins.values():
        cs.net_pnl = 0.0
        cs.cycles_closed = 0
    await update.message.reply_text("✅ PnL reset.", reply_markup=MENU)

async def close_all(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Safe: we only stop cycles; we do not force live market closes here.
    for cs in STATE.coins.values():
        cs.in_cycle = False
        cs.buy = None
        cs.sell = None
        cs.entry_price = None
        cs.entry_size = None
    await update.message.reply_text("✅ Close-all: cycles stoppade.", reply_markup=MENU)

async def export_csv(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        if os.path.exists(MOCK_LOG):
            await ctx.bot.send_document(chat_id=update.effective_chat.id, document=open(MOCK_LOG, "rb"))
        if os.path.exists(REAL_LOG):
            await ctx.bot.send_document(chat_id=update.effective_chat.id, document=open(REAL_LOG, "rb"))
    except Exception:
        await update.message.reply_text("❌ Kunde inte skicka CSV.", reply_markup=MENU)

def _parse_float_arg(args: List[str]) -> Optional[float]:
    if not args:
        return None
    try:
        return float(args[0])
    except Exception:
        return None

async def threshold(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    v = _parse_float_arg(ctx.args)
    if v is None:
        await update.message.reply_text(f"Min spread nu: {pct(STATE.settings.min_spread)}\nEx: /threshold 0.25", reply_markup=MENU)
        return
    STATE.settings.min_spread = max(0.01, min(10.0, v))
    await update.message.reply_text(f"✅ Min spread nu: {pct(STATE.settings.min_spread)}", reply_markup=MENU)

async def tp(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    v = _parse_float_arg(ctx.args)
    if v is None:
        await update.message.reply_text(f"Target nu: {pct(STATE.settings.target)}\nEx: /tp 0.25", reply_markup=MENU)
        return
    STATE.settings.target = max(0.01, min(10.0, v))
    await update.message.reply_text(f"✅ Target nu: {pct(STATE.settings.target)}", reply_markup=MENU)

async def sl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    v = _parse_float_arg(ctx.args)
    if v is None:
        await update.message.reply_text(f"Panic stop nu: {pct(STATE.settings.panic)}\nEx: /sl 0.80", reply_markup=MENU)
        return
    STATE.settings.panic = max(0.05, min(25.0, v))
    await update.message.reply_text(f"✅ Panic stop nu: {pct(STATE.settings.panic)}", reply_markup=MENU)

async def stake(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    v = _parse_float_arg(ctx.args)
    if v is None:
        await update.message.reply_text(f"Stake nu: {fmt(STATE.settings.stake,2)} USDT\nEx: /stake 30", reply_markup=MENU)
        return
    STATE.settings.stake = max(1.0, min(100000.0, v))
    await update.message.reply_text(f"✅ Stake nu: {fmt(STATE.settings.stake,2)} USDT", reply_markup=MENU)

async def coins(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text(f"Coins ({len(STATE.settings.coins)}): {STATE.settings.coins}\nEx: /coins LINK-USDT ADA-USDT XRP-USDT", reply_markup=MENU)
        return

    cleaned = []
    for raw in ctx.args:
        c = raw.strip().upper().replace("_", "-")
        if "-" not in c and c.endswith("USDT"):
            c = c[:-4] + "-USDT"
        if c.endswith("-USDT"):
            cleaned.append(c)

    if not cleaned:
        await update.message.reply_text("❌ Inga giltiga coins. Ex: /coins LINK-USDT ADA-USDT", reply_markup=MENU)
        return

    STATE.settings.coins = cleaned[:30]
    for c in STATE.settings.coins:
        STATE.coins.setdefault(c, CoinState())

    await update.message.reply_text(f"✅ Coins uppdaterade ({len(STATE.settings.coins)}): {STATE.settings.coins}", reply_markup=MENU)

async def safe_live(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    for k, v in SAFE_LIVE_PRESET.items():
        setattr(STATE.settings, k, v)
    await update.message.reply_text(
        "✅ Safe-live preset satt.\n"
        f"min_spread={pct(STATE.settings.min_spread)} target={pct(STATE.settings.target)} panic={pct(STATE.settings.panic)} stake={fmt(STATE.settings.stake,2)}",
        reply_markup=MENU
    )

async def trade_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Ex: /trade_mode mock  eller  /trade_mode live", reply_markup=MENU)
        return
    m = ctx.args[0].strip().lower()
    if m not in ("mock", "live"):
        await update.message.reply_text("❌ Välj mock eller live.", reply_markup=MENU)
        return

    if m == "mock":
        STATE.settings.trade_mode = "mock"
        STATE.pending_live_confirm = False
        await update.message.reply_text("✅ Trade mode: mock", reply_markup=MENU)
        return

    # live requested
    if not (KUCOIN_KEY and KUCOIN_SECRET and KUCOIN_PASSPHRASE):
        await update.message.reply_text("❌ Live keys saknas (KUCOIN_API_KEY/SECRET/PASSPHRASE).", reply_markup=MENU)
        return

    STATE.pending_live_confirm = True
    await update.message.reply_text("⚠️ Skriv JA för att aktivera LIVE trading. (Annars fortsätter mock)", reply_markup=MENU)

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if not STATE.pending_live_confirm:
        return
    txt = update.message.text.strip().upper()
    if txt == "JA":
        STATE.settings.trade_mode = "live"
        STATE.pending_live_confirm = False
        await update.message.reply_text("✅ LIVE trading aktiverad.", reply_markup=MENU)
    else:
        STATE.pending_live_confirm = False
        await update.message.reply_text("OK. Fortsätter i mock.", reply_markup=MENU)

# =========================================================
# STARTUP HOOK (start engine task safely)
# =========================================================
async def post_init(app: Application):
    global APP
    APP = app
    app.create_task(engine_task())

# =========================================================
# MAIN
# =========================================================
def build_app() -> Application:
    token = get_telegram_token()
    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("engine_on", engine_on))
    app.add_handler(CommandHandler("engine_off", engine_off))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("pnl", pnl))
    app.add_handler(CommandHandler("reset_pnl", reset_pnl))
    app.add_handler(CommandHandler("close_all", close_all))
    app.add_handler(CommandHandler("export_csv", export_csv))
    app.add_handler(CommandHandler("notify", notify_cmd))

    app.add_handler(CommandHandler("threshold", threshold))
    app.add_handler(CommandHandler("tp", tp))
    app.add_handler(CommandHandler("sl", sl))
    app.add_handler(CommandHandler("stake", stake))
    app.add_handler(CommandHandler("coins", coins))
    app.add_handler(CommandHandler("safe_live", safe_live))
    app.add_handler(CommandHandler("trade_mode", trade_mode))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app

if __name__ == "__main__":
    application = build_app()
    application.run_polling()
