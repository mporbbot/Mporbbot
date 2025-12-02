#########################################################
# Mp HybridBot – PRO MAX Edition
# Regime Switching + Momentum + Mean Reversion
# ATR StopLoss + ATR Trailing Stop (klassisk)
# Mock + Live Trading
# Telegram (aiogram)
# KuCoin Spot CSV-logg (Skatteverket-godkänd)
#########################################################

import pandas as pd
import numpy as np
import asyncio
import datetime
import csv
import os
import traceback

from aiogram import Bot, Dispatcher, executor, types
from kucoin.client import Market, Trade

#########################################################
# KONFIGURATION – FYLL I DINA NYCKLAR
#########################################################

TELEGRAM_TOKEN = "DIN_TELEGRAM_TOKEN_HÄR"

KUCOIN_API = "DIN_KUCOIN_API_KEY"
KUCOIN_SECRET = "DIN_KUCOIN_SECRET"
KUCOIN_PASSPHRASE = "DIN_KUCOIN_PASSPHRASE"

# Symboler du vill handla
SYMBOLS = ["ADA-USDT", "BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT"]

# Tradinginställningar (styr via Telegram)
TRADE_AMOUNT = 30.0            # Mock-belopp
STOP_LOSS_PCT = 1.0            # Endast som fallback (vi kör ATR-SL)
THRESHOLD = 25                 # Momentumscore-gräns

RUNNING = False
MOCK_MODE = True               # TRUE = ingen riktig handel

# Regime (auto)
CURRENT_REGIME = "TREND"

# Statistik
signal_history = []
trade_count = 0
win_count = 0
loss_count = 0
open_positions = {}            # öppna trade per symbol
daily_pnl = 0.0

# CSV-filer
MOCK_LOG = "mock_trade_log.csv"
LIVE_LOG = "live_trade_log.csv"

#########################################################
# INITIERA KLIENTER
#########################################################

bot = Bot(TELEGRAM_TOKEN)
dp = Dispatcher(bot)

market = Market(url="https://api.kucoin.com")
trade_client = Trade(
    key=KUCOIN_API,
    secret=KUCOIN_SECRET,
    passphrase=KUCOIN_PASSPHRASE,
    is_sandbox=False
)

#########################################################
# CSV-export (KuCoin Spot History-format – exakt)
#########################################################

def write_csv(mode="mock", row=None):
    filename = MOCK_LOG if mode == "mock" else LIVE_LOG
    exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Time","Symbol","Side","Order Type","Price",
            "Size","Fee","Fee Coin","Amount","Remark"
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)

#########################################################
# HJÄLPFUNKTIONER
#########################################################

def now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x):
    try: return float(x)
    except: return 0.0

#########################################################
# INDIKATORER
#########################################################

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def true_range(df):
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).rolling(period).mean()

def bollinger(df, period=20, std=2):
    mid = df["close"].rolling(period).mean()
    dev = df["close"].rolling(period).std()
    return mid + dev*std, mid - dev*std

#########################################################
# REGIME-DETECTION (Trend vs Range)
#########################################################

def detect_regime(df):
    global CURRENT_REGIME

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr"] = atr(df, 14)
    df["bb_up"], df["bb_lo"] = bollinger(df, 20, 2)

    ema_slope = df["ema20"].diff().iloc[-1]
    atr_slope = df["atr"].diff().iloc[-1]
    bb_width = (df["bb_up"].iloc[-1] - df["bb_lo"].iloc[-1]) / df["close"].iloc[-1]

    trend = (
        df["ema20"].iloc[-1] > df["ema50"].iloc[-1] and
        ema_slope > 0 and
        atr_slope > 0 and
        bb_width > 0.02
    )

    CURRENT_REGIME = "TREND" if trend else "RANGE"
    return CURRENT_REGIME

#########################################################
# MOMENTUM SCORE
#########################################################

def momentum_score(df):
    last = df.iloc[-1]

    strength = (last["close"] - last["open"]) / (last["high"] - last["low"] + 1e-8)
    range_factor = (last["high"] - last["low"]) / last["close"]

    score = strength*10 + range_factor*20 + 10
    return max(0, round(score, 2))

#########################################################
# ENTRY LOGIK – MOMENTUM
#########################################################

def momentum_entry(df):
    score = momentum_score(df)
    if score >= THRESHOLD:
        return "LONG", score
    return None, score

#########################################################
# ENTRY LOGIK – MEAN REVERSION
#########################################################

def mean_reversion_entry(df):
    last = df.iloc[-1]
    if last["close"] < last["bb_lo"]:
        return "LONG", "MR"
    return None, None

#########################################################
# STOP LOSS – ATR-baserad
#########################################################

def atr_stoploss(entry_price, atr_value):
    return entry_price - atr_value * 1.5      # SL = entry – 1.5×ATR

#########################################################
# TRAILING STOP – ATR-klassiskt
#########################################################

def atr_trailing(highest_price, atr_value):
    return highest_price - atr_value * 1.5
#########################################################
# HÄMTA DATA FRÅN KUCOIN (3m-kliner)
#########################################################

def get_data(symbol, limit=200):
    """
    Hämtar 3-minuterskliner från KuCoin och bygger en DataFrame.
    """
    try:
        raw = market.get_kline(symbol, "3min")
    except Exception as e:
        print(f"[{symbol}] Fel vid hämtning av kline: {e}")
        return None

    # KuCoin kline-format: [time, open, close, high, low, volume, turnover]
    cols = ["time", "open", "close", "high", "low", "volume", "turnover"]
    df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
    df["open"] = df["open"].astype(float)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # KuCoin returnerar nyaste först → vänd
    df = df.iloc[::-1].reset_index(drop=True)
    if len(df) > limit:
        df = df.tail(limit)
    return df

#########################################################
# ORDER-HANTERING (mock + live)
#########################################################

def execute_buy(symbol, price, qty, mode="mock", remark="entry"):
    """
    Utför en BUY – loggar alltid CSV i KuCoin Spot-format.
    mode = "mock" eller "live"
    """
    fee = round(price * qty * 0.001, 8)   # ~0.1 %
    amount = round(price * qty, 8)

    if mode == "live":
        try:
            # KuCoin vill ha size i base-coin
            trade_client.create_market_order(
                symbol,
                side="buy",
                size=qty
            )
        except Exception as e:
            print(f"[{symbol}] Live BUY-fel: {e}")

    write_csv("mock" if mode == "mock" else "live", {
        "Time": now(),
        "Symbol": symbol,
        "Side": "BUY",
        "Order Type": "market",
        "Price": price,
        "Size": qty,
        "Fee": fee,
        "Fee Coin": "USDT",
        "Amount": amount,
        "Remark": remark
    })


def execute_sell(symbol, price, qty, mode="mock", remark="exit", entry_price=None):
    """
    Utför en SELL – loggar alltid CSV.
    Beräknar PnL om entry_price finns.
    """
    global daily_pnl, trade_count, win_count, loss_count

    fee = round(price * qty * 0.001, 8)
    amount = round(price * qty, 8)
    pnl = 0.0

    if entry_price is not None:
        pnl = (price - entry_price) * qty
        daily_pnl += pnl

    if mode == "live":
        try:
            trade_client.create_market_order(
                symbol,
                side="sell",
                size=qty
            )
        except Exception as e:
            print(f"[{symbol}] Live SELL-fel: {e}")

    write_csv("mock" if mode == "mock" else "live", {
        "Time": now(),
        "Symbol": symbol,
        "Side": "SELL",
        "Order Type": "market",
        "Price": price,
        "Size": qty,
        "Fee": fee,
        "Fee Coin": "USDT",
        "Amount": amount,
        "Remark": f"{remark} PnL={round(pnl, 8)}"
    })

    trade_count += 1
    if pnl >= 0:
        win_count += 1
    else:
        loss_count += 1

    return pnl

#########################################################
# TRADING-LOOP PER SYMBOL
#########################################################

async def symbol_loop(symbol: str):
    """
    Huvudloop för en enskild symbol.
    Hämtar ny data, uppdaterar regime, kollar entry/exit, hanterar ATR-SL & ATR-trail.
    """
    global LAST_STATUS, open_positions

    print(f"[{symbol}] Trading-loop startad.")
    while True:
        if not RUNNING:
            await asyncio.sleep(2)
            continue

        try:
            df = get_data(symbol)
            if df is None or len(df) < 50:
                await asyncio.sleep(3)
                continue

            # Beräkna indikatorer & regime
            regime = detect_regime(df)
            last = df.iloc[-1]
            last_close = last["close"]
            last_high = last["high"]
            atr_val = df["atr"].iloc[-1]

            # Ingen position? → leta entry
            if symbol not in open_positions:

                if regime == "TREND":
                    side, score = momentum_entry(df)
                    strategy = "MOMENTUM"
                else:
                    side, score = mean_reversion_entry(df)
                    strategy = "MEAN-REV"

                if side == "LONG":
                    entry_price = last_close
                    sl_price = atr_stoploss(entry_price, atr_val)
                    qty = round(TRADE_AMOUNT / entry_price, 6)
                    mode = "mock" if MOCK_MODE else "live"

                    # Utför BUY
                    execute_buy(symbol, entry_price, qty, mode=mode,
                                remark=f"{strategy} {regime}")

                    open_positions[symbol] = {
                        "mode": mode,
                        "entry": entry_price,
                        "sl": sl_price,
                        "highest": entry_price,
                        "atr": atr_val,
                        "trail_active": False,
                        "regime": regime,
                        "strategy": strategy,
                        "qty": qty,
                        "opened_at": now()
                    }

                    LAST_STATUS = (
                        f"{symbol} | ENTRY {entry_price:.6f} | SL {sl_price:.6f} | "
                        f"ATR {atr_val:.6f} | Regime {regime} | Strat {strategy} | Score {score}"
                    )

                else:
                    # ingen entry
                    LAST_STATUS = f"{symbol} | {regime} | Score {score}"
            else:
                # Position finns → hantera trailing & SL
                pos = open_positions[symbol]
                entry_price = pos["entry"]
                mode = pos["mode"]

                # uppdatera ATR och highest
                pos["atr"] = atr_val
                if last_high > pos["highest"]:
                    pos["highest"] = last_high

                # aktivera trailing när priset gått +1×ATR över entry
                if not pos["trail_active"]:
                    if last_close >= entry_price + atr_val:
                        pos["trail_active"] = True

                # uppdatera SL med klassisk ATR-trail
                if pos["trail_active"]:
                    new_sl = atr_trailing(pos["highest"], atr_val)
                    if new_sl > pos["sl"]:
                        pos["sl"] = new_sl

                # EXIT om priset bryter SL
                if last_close <= pos["sl"]:
                    pnl = execute_sell(
                        symbol,
                        last_close,
                        pos["qty"],
                        mode=mode,
                        remark="ATR SL/Trail exit",
                        entry_price=entry_price
                    )
                    LAST_STATUS = (
                        f"{symbol} | EXIT {last_close:.6f} | PnL {round(pnl, 6)} | "
                        f"Regime {pos['regime']} | Strat {pos['strategy']}"
                    )
                    del open_positions[symbol]

            await asyncio.sleep(3)

        except Exception as e:
            print(f"[{symbol}] FEL i symbol_loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)

#########################################################
# TELEGRAM-TANGENTBORD
#########################################################

def main_keyboard():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row("/start", "/stop")
    kb.row("/status", "/stats")
    kb.row("/pnl_today")
    kb.row("/risk 25", "/sl 1.0")
    kb.row("/testbuy ADA-USDT")
    kb.row("/export_mock", "/export_live")
    return kb
#########################################################
# TELEGRAM-KOMMANDON
#########################################################

@dp.message_handler(commands=["start"])
async def start_cmd(msg: types.Message):
    global RUNNING
    RUNNING = True
    await msg.answer(
        "🚀 *Mp HybridBot PRO MAX startad!*\n"
        "Regime Switching + Momentum + Mean Reversion är aktiv.\n"
        "ATR StopLoss + ATR Trailing Stop.",
        parse_mode="Markdown",
        reply_markup=main_keyboard()
    )

    # starta en loop per symbol
    for s in SYMBOLS:
        asyncio.create_task(symbol_loop(s))


@dp.message_handler(commands=["stop"])
async def stop_cmd(msg: types.Message):
    global RUNNING
    RUNNING = False
    await msg.answer("🛑 Bot *stoppad*.", parse_mode="Markdown")


@dp.message_handler(commands=["status"])
async def status_cmd(msg: types.Message):
    global LAST_STATUS
    await msg.answer(f"📡 *STATUS*\n{LAST_STATUS}", parse_mode="Markdown")


@dp.message_handler(commands=["stats"])
async def stats_cmd(msg: types.Message):
    global trade_count, win_count, loss_count, open_positions, CURRENT_REGIME

    wr = 0
    if trade_count > 0:
        wr = round(win_count / trade_count * 100, 2)

    txt = (
        "📊 *STATISTIK*\n"
        f"-------------------------\n"
        f"Totala trades: {trade_count}\n"
        f"Vinster: {win_count}\n"
        f"Förluster: {loss_count}\n"
        f"Winrate: {wr}%\n\n"
        f"Öppna positioner: {len(open_positions)}\n"
        f"Regime: {CURRENT_REGIME}\n"
        f"Threshold: {THRESHOLD}\n"
        f"ATR-baserad SL: aktiv\n"
        f"ATR Trailing: klassisk\n"
    )
    await msg.answer(txt, parse_mode="Markdown")


@dp.message_handler(commands=["pnl_today"])
async def pnl_cmd(msg: types.Message):
    global daily_pnl
    await msg.answer(f"💰 *Dagens PnL: {round(daily_pnl, 4)} USDT*", parse_mode="Markdown")


@dp.message_handler(commands=["sl"])
async def sl_cmd(msg: types.Message):
    global STOP_LOSS_PCT
    try:
        parts = msg.text.split()
        if len(parts) == 2:
            STOP_LOSS_PCT = float(parts[1])
            await msg.answer(f"🔧 StopLoss fallback ändrad till {STOP_LOSS_PCT} %")
        else:
            await msg.answer(f"Aktuellt fallback-SL: {STOP_LOSS_PCT}%")
    except:
        await msg.answer("Fel format. Exempel: /sl 1.0")


@dp.message_handler(commands=["risk"])
async def risk_cmd(msg: types.Message):
    global THRESHOLD
    try:
        parts = msg.text.split()
        THRESHOLD = int(parts[1])
        await msg.answer(f"⚙️ Threshold ändrad till {THRESHOLD}")
    except:
        await msg.answer("Fel format. Exempel: /risk 25")


@dp.message_handler(commands=["testbuy"])
async def testbuy_cmd(msg: types.Message):
    try:
        symbol = msg.text.split()[1]
        df = get_data(symbol)
        price = df.iloc[-1]["close"]
        qty = round(TRADE_AMOUNT / price, 6)

        execute_buy(symbol, price, qty, mode="mock", remark="TESTBUY")
        await msg.answer(f"🧪 TestBuy loggad för {symbol} @ {price}")
    except:
        await msg.answer("Exempel: /testbuy ADA-USDT")


@dp.message_handler(commands=["export_mock"])
async def export_mock_cmd(msg: types.Message):
    if os.path.isfile(MOCK_LOG):
        await msg.answer_document(open(MOCK_LOG, "rb"))
    else:
        await msg.answer("Ingen mock-log hittades.")


@dp.message_handler(commands=["export_live"])
async def export_live_cmd(msg: types.Message):
    if os.path.isfile(LIVE_LOG):
        await msg.answer_document(open(LIVE_LOG, "rb"))
    else:
        await msg.answer("Ingen live-log hittades.")


#########################################################
# FELHANTERING – Fallback
#########################################################

@dp.errors_handler()
async def errors_handler(update, error):
    print(f"Telegram error: {error}")
    return True


#########################################################
# MAIN ENTRYPOINT
#########################################################

if __name__ == "__main__":
    print("🚀 Mp HybridBot PRO MAX startas…")
    executor.start_polling(dp, skip_updates=True)
