#########################################################
# Mp HybridBot – PRO MAX (python-telegram-bot version)
# Regime Switching + Momentum + Mean Reversion
# ATR StopLoss + ATR Trailing Stop (klassisk)
# Mock + Live trading
# KuCoin Spot CSV-logg (Spot History-format)
#########################################################

import pandas as pd
import numpy as np
import datetime
import csv
import os
import traceback
import threading
import time

from kucoin.client import Market, Trade

from telegram import Bot, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext

#########################################################
# KONFIGURATION – FYLL I DINA NYCKLAR
#########################################################

TELEGRAM_TOKEN = "DIN_TELEGRAM_TOKEN_HÄR"

KUCOIN_API = "DIN_KUCOIN_API_KEY"
KUCOIN_SECRET = "DIN_KUCOIN_SECRET"
KUCOIN_PASSPHRASE = "DIN_KUCOIN_PASSPHRASE"

# Symboler du vill handla
SYMBOLS = ["ADA-USDT", "BTC-USDT", "ETH-USDT", "LINK-USDT", "XRP-USDT"]

# Tradinginställningar (kan styras från Telegram)
TRADE_AMOUNT = 30.0          # Mock-belopp (USDT)
STOP_LOSS_PCT = 1.0          # Fallback-SL i %, ATR är primär
THRESHOLD = 25               # Momentum-score threshold
MOCK_MODE = True             # True = inga riktiga orders

RUNNING = False              # styrs via /start och /stop
CURRENT_REGIME = "TREND"

# Statistik / status
LAST_STATUS = "Ingen signal ännu."
trade_count = 0
win_count = 0
loss_count = 0
daily_pnl = 0.0

# Öppna positioner per symbol
# { symbol: {entry, sl, highest, atr, trail_active, regime, strategy, qty, mode, opened_at} }
open_positions = {}

# CSV-filer (Skatteverket / KuCoin-format)
MOCK_LOG = "mock_trade_log.csv"
LIVE_LOG = "live_trade_log.csv"

#########################################################
# INITIERA KUCOIN-KLIENTER
#########################################################

market = Market(url="https://api.kucoin.com")
trade_client = Trade(
    key=KUCOIN_API,
    secret=KUCOIN_SECRET,
    passphrase=KUCOIN_PASSPHRASE,
    is_sandbox=False
)

#########################################################
# CSV-export (KuCoin Spot History-format – A)
#########################################################

def write_csv(mode="mock", row=None):
    """
    mode: 'mock' eller 'live'
    row: dict med nycklar:
      Time, Symbol, Side, Order Type, Price,
      Size, Fee, Fee Coin, Amount, Remark
    """
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
# HJÄLPFUNKTIONER & INDIKATORER
#########################################################

def now_str():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

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
# REGIME-DETECTION (TREND vs RANGE)
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
# MOMENTUM-SCORE & ENTRY
#########################################################

def momentum_score(df):
    last = df.iloc[-1]
    strength = (last["close"] - last["open"]) / (last["high"] - last["low"] + 1e-8)
    range_factor = (last["high"] - last["low"]) / last["close"]
    score = strength*10 + range_factor*20 + 10
    return max(0, round(score, 2))

def momentum_entry(df):
    score = momentum_score(df)
    if score >= THRESHOLD:
        return "LONG", score
    return None, score

#########################################################
# MEAN-REVERSION ENTRY
#########################################################

def mean_reversion_entry(df):
    last = df.iloc[-1]
    if last["close"] < last["bb_lo"]:
        return "LONG", "MR"
    return None, None

#########################################################
# ATR-BASERAD STOPLOSS & TRAILING
#########################################################

def atr_stoploss(entry_price, atr_value):
    # SL = entry - 1.5 × ATR
    return entry_price - atr_value * 1.5

def atr_trailing(highest_price, atr_value):
    # trailing-SL = highest - 1.5 × ATR
    return highest_price - atr_value * 1.5

#########################################################
# HÄMTA 3m-DATA FRÅN KUCOIN
#########################################################

def get_data(symbol, limit=200):
    try:
        raw = market.get_kline(symbol, "3min")
    except Exception as e:
        print(f"[{symbol}] Fel vid kline-hämtning: {e}")
        return None

    cols = ["time","open","close","high","low","volume","turnover"]
    df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
    df["open"] = df["open"].astype(float)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df.iloc[::-1].reset_index(drop=True)
    if len(df) > limit:
        df = df.tail(limit)
    return df
#########################################################
# ORDER-HANTERING (mock + live)
#########################################################

def execute_buy(symbol, price, qty, mode="mock", remark="entry"):
    fee = round(price * qty * 0.001, 8)     # ~0.1 % fee
    amount = round(price * qty, 8)

    # LIVE-ORDER (om aktiverad)
    if mode == "live":
        try:
            trade_client.create_market_order(
                symbol,
                side="buy",
                size=qty
            )
        except Exception as e:
            print(f"[{symbol}] Live BUY-fel: {e}")

    # CSV-export
    write_csv("mock" if mode=="mock" else "live", {
        "Time": now_str(),
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

    write_csv("mock" if mode=="mock" else "live", {
        "Time": now_str(),
        "Symbol": symbol,
        "Side": "SELL",
        "Order Type": "market",
        "Price": price,
        "Size": qty,
        "Fee": fee,
        "Fee Coin": "USDT",
        "Amount": amount,
        "Remark": f"{remark} PnL={round(pnl,8)}"
    })

    trade_count += 1
    if pnl >= 0:
        win_count += 1
    else:
        loss_count += 1

    return pnl

#########################################################
# TRADING-LOOP (KÖRS I EGEN THREAD PER SYMBOL)
#########################################################

def trading_loop(symbol):
    global LAST_STATUS, open_positions

    print(f"[{symbol}] Trading-thread startad.")

    while True:
        if not RUNNING:
            time.sleep(2)
            continue

        try:
            df = get_data(symbol)
            if df is None or len(df) < 50:
                time.sleep(3)
                continue

            regime = detect_regime(df)
            last = df.iloc[-1]
            last_close = last["close"]
            last_high = last["high"]
            atr_val = df["atr"].iloc[-1]

            # ======================
            # ENTRY LOGIK
            # ======================
            if symbol not in open_positions:

                if regime == "TREND":
                    side, score = momentum_entry(df)
                    strategy = "MOMENTUM"
                else:
                    side, score = mean_reversion_entry(df)
                    strategy = "MEANREV"

                if side == "LONG":
                    entry_price = last_close
                    qty = round(TRADE_AMOUNT / entry_price, 6)

                    sl_price = atr_stoploss(entry_price, atr_val)
                    mode = "mock" if MOCK_MODE else "live"

                    execute_buy(
                        symbol,
                        entry_price,
                        qty,
                        mode=mode,
                        remark=f"{strategy} {regime}"
                    )

                    open_positions[symbol] = {
                        "mode": mode,
                        "entry": entry_price,
                        "sl": sl_price,
                        "highest": entry_price,
                        "atr": atr_val,
                        "trail_active": False,
                        "strategy": strategy,
                        "regime": regime,
                        "qty": qty,
                        "opened_at": now_str()
                    }

                    LAST_STATUS = (
                        f"{symbol} | ENTRY {entry_price:.6f} | SL {sl_price:.6f} | "
                        f"ATR {atr_val:.6f} | Regime {regime} | Strat {strategy} | Score {score}"
                    )
                else:
                    LAST_STATUS = f"{symbol} | {regime} | Score {score}"

            # ======================
            # EXIT / TRAILING
            # ======================
            else:
                pos = open_positions[symbol]
                entry_price = pos["entry"]
                qty = pos["qty"]
                mode = pos["mode"]

                # uppdatera ATR + highest
                pos["atr"] = atr_val
                if last_high > pos["highest"]:
                    pos["highest"] = last_high

                # aktivera trailing när price >= entry + 1×ATR
                if not pos["trail_active"]:
                    if last_close >= entry_price + atr_val:
                        pos["trail_active"] = True

                # uppdatera trailing SL
                if pos["trail_active"]:
                    new_sl = atr_trailing(pos["highest"], atr_val)
                    if new_sl > pos["sl"]:
                        pos["sl"] = new_sl

                # EXIT om pris <= SL
                if last_close <= pos["sl"]:
                    pnl = execute_sell(
                        symbol,
                        last_close,
                        qty,
                        mode=mode,
                        remark="ATR Trail Exit",
                        entry_price=entry_price
                    )

                    LAST_STATUS = (
                        f"{symbol} | EXIT {last_close:.6f} | PnL {round(pnl,6)} | "
                        f"{pos['strategy']} | {pos['regime']}"
                    )

                    del open_positions[symbol]

            time.sleep(3)

        except Exception as e:
            print(f"[{symbol}] FEL i trading_loop: {e}")
            traceback.print_exc()
            time.sleep(5)

#########################################################
# TELEGRAM-TANGENTBORD
#########################################################

def main_keyboard():
    return ReplyKeyboardMarkup([
        ["/start", "/stop"],
        ["/status", "/stats"],
        ["/pnl_today"],
        ["/risk 25", "/sl 1.0"],
        ["/testbuy ADA-USDT"],
        ["/export_mock", "/export_live"]
    ],
    resize_keyboard=True)
#########################################################
# TELEGRAM-KOMMANDON
#########################################################

def cmd_start(update, context: CallbackContext):
    global RUNNING

    if RUNNING:
        update.message.reply_text("Bot är redan igång ✅", reply_markup=main_keyboard())
        return

    RUNNING = True
    update.message.reply_text(
        "🚀 Mp HybridBot PRO MAX *startad!*\n"
        "Regime Switching + Momentum + Mean Reversion är aktiv.\n"
        "ATR StopLoss + ATR Trailing Stop (klassisk).",
        parse_mode="Markdown",
        reply_markup=main_keyboard()
    )

    # starta en trading-thread per symbol
    for s in SYMBOLS:
        t = threading.Thread(target=trading_loop, args=(s,), daemon=True)
        t.start()


def cmd_stop(update, context: CallbackContext):
    global RUNNING
    RUNNING = False
    update.message.reply_text("🛑 Bot *stoppad*.", parse_mode="Markdown")


def cmd_status(update, context: CallbackContext):
    global LAST_STATUS
    update.message.reply_text(f"📡 *STATUS*\n{LAST_STATUS}", parse_mode="Markdown")


def cmd_stats(update, context: CallbackContext):
    global trade_count, win_count, loss_count, open_positions, CURRENT_REGIME, THRESHOLD

    wr = 0.0
    if trade_count > 0:
        wr = round(win_count / trade_count * 100, 2)

    txt = (
        "📊 *STATISTIK*\n"
        "-------------------------\n"
        f"Totala trades: {trade_count}\n"
        f"Vinster: {win_count}\n"
        f"Förluster: {loss_count}\n"
        f"Winrate: {wr}%\n\n"
        f"Öppna positioner: {len(open_positions)}\n"
        f"Aktuell regime: {CURRENT_REGIME}\n"
        f"Threshold (score): {THRESHOLD}\n"
        f"Fallback SL% (om ATR ej finns): {STOP_LOSS_PCT}%\n"
        f"Mock mode: {'ON' if MOCK_MODE else 'OFF'}\n"
    )
    update.message.reply_text(txt, parse_mode="Markdown")


def cmd_pnl_today(update, context: CallbackContext):
    global daily_pnl
    update.message.reply_text(
        f"💰 *Dagens PnL:* {round(daily_pnl, 4)} USDT",
        parse_mode="Markdown"
    )


def cmd_sl(update, context: CallbackContext):
    global STOP_LOSS_PCT
    msg = update.message.text.split()
    if len(msg) == 2:
        try:
            val = float(msg[1])
            STOP_LOSS_PCT = val
            update.message.reply_text(f"🔧 Fallback-StopLoss ändrad till {STOP_LOSS_PCT} %")
        except:
            update.message.reply_text("Fel format. Exempel: /sl 1.0")
    else:
        update.message.reply_text(f"Aktuellt fallback-SL: {STOP_LOSS_PCT} %")


def cmd_risk(update, context: CallbackContext):
    global THRESHOLD
    msg = update.message.text.split()
    if len(msg) == 2:
        try:
            val = int(msg[1])
            THRESHOLD = val
            update.message.reply_text(f"⚙️ Threshold ändrad till {THRESHOLD}")
        except:
            update.message.reply_text("Fel format. Exempel: /risk 25")
    else:
        update.message.reply_text(f"Aktuell threshold: {THRESHOLD}")


def cmd_testbuy(update, context: CallbackContext):
    try:
        msg = update.message.text.split()
        if len(msg) != 2:
            update.message.reply_text("Exempel: /testbuy ADA-USDT")
            return

        symbol = msg[1]
        df = get_data(symbol)
        if df is None or len(df) == 0:
            update.message.reply_text(f"Kunde inte hämta data för {symbol}.")
            return

        price = df.iloc[-1]["close"]
        qty = round(TRADE_AMOUNT / price, 6)

        execute_buy(symbol, price, qty, mode="mock", remark="TESTBUY")
        update.message.reply_text(f"🧪 TestBuy loggad för {symbol} @ {price}")
    except Exception as e:
        print("Fel i /testbuy:", e)
        traceback.print_exc()
        update.message.reply_text("Fel vid /testbuy. Kolla loggen.")


def cmd_export_mock(update, context: CallbackContext):
    if os.path.isfile(MOCK_LOG):
        with open(MOCK_LOG, "rb") as f:
            update.message.reply_document(f, filename=MOCK_LOG)
    else:
        update.message.reply_text("Ingen mock-logfil hittades ännu.")


def cmd_export_live(update, context: CallbackContext):
    if os.path.isfile(LIVE_LOG):
        with open(LIVE_LOG, "rb") as f:
            update.message.reply_document(f, filename=LIVE_LOG)
    else:
        update.message.reply_text("Ingen live-logfil hittades ännu.")


#########################################################
# MAIN / STARTUP
#########################################################

def main():
    print("🚀 Startar Mp HybridBot PRO MAX (python-telegram-bot)…")

    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Kommandon
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("stop", cmd_stop))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("stats", cmd_stats))
    dp.add_handler(CommandHandler("pnl_today", cmd_pnl_today))
    dp.add_handler(CommandHandler("sl", cmd_sl))
    dp.add_handler(CommandHandler("risk", cmd_risk))
    dp.add_handler(CommandHandler("testbuy", cmd_testbuy))
    dp.add_handler(CommandHandler("export_mock", cmd_export_mock))
    dp.add_handler(CommandHandler("export_live", cmd_export_live))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
