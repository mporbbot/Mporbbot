import time
import ccxt


class ExchangeClient:
    def __init__(self):
        self.exchange = ccxt.kucoin({
            "enableRateLimit": True
        })

    def _retry(self, func, retries=3, base_delay=3):
        last_error = None

        for attempt in range(retries):
            try:
                return func()

            except (
                ccxt.RateLimitExceeded,
                ccxt.RequestTimeout,
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
            ) as e:
                last_error = e

                delay = base_delay * (attempt + 1)

                print(
                    f"KuCoin tillfälligt fel "
                    f"(försök {attempt + 1}/{retries}): {e}"
                )

                if attempt < retries - 1:
                    time.sleep(delay)

            except Exception as e:
                last_error = e

                print(
                    f"KuCoin oväntat fel "
                    f"(försök {attempt + 1}/{retries}): {e}"
                )

                if attempt < retries - 1:
                    time.sleep(base_delay * (attempt + 1))

        print(f"KuCoin gav upp efter {retries} försök: {last_error}")

        return None

    def fetch_candles(self, symbol, timeframe, limit=200):
        def request():
            return self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit
            )

        candles = self._retry(request)

        if not candles:
            return []

        return [
            {
                "time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in candles
        ]

    def fetch_price(self, symbol):
        def request():
            return self.exchange.fetch_ticker(symbol)

        ticker = self._retry(request)

        if not ticker:
            return None

        price = ticker.get("last")

        if price is None:
            return None

        return float(price)
