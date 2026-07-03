import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# TELEGRAM
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

CHAT_ID_FILE = "chat_id.txt"

# =========================
# MODE
# =========================

MODE = "mock"

# =========================
# EXCHANGE
# =========================

EXCHANGE_ID = "kucoin"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "LINK/USDT",
]

# =========================
# TRADING
# =========================

TRADE_SIZE_USDT = 30.0
MAX_OPEN_TRADES = 5
FEE_RATE = 0.001

POLL_SECONDS = 60

# =========================
# ORB
# =========================

ORB_TIMEFRAME = "15m"
ENTRY_TIMEFRAME = "5m"
TREND_TIMEFRAME = "1h"

# 07:00 UTC = 09:00 svensk sommartid
ORB_HOUR_UTC = 7
ORB_MINUTE_UTC = 0

BREAKOUT_BUFFER = 0.0003

MIN_ORB_PERCENT = 0.001
MAX_ORB_PERCENT = 0.04

# =========================
# ENTRY MODES
# =========================

USE_RETEST_ENTRY = True
USE_MOMENTUM_ENTRY = False
USE_PULLBACK_ENTRY = False

# =========================
# FILTERS
# =========================

USE_TREND_FILTER = True
USE_VOLUME_FILTER = False

VOLUME_MULTIPLIER = 1.0

# =========================
# RISK
# =========================

RISK_REWARD = 2.0
TRAIL_PERCENT = 0.006
COOLDOWN_MINUTES = 20

USE_TRAILING_STOP = True

# =========================
# FILES
# =========================

MOCK_LOG = "mock_trade_log.csv"
SIGNAL_LOG = "signal_log.csv"
