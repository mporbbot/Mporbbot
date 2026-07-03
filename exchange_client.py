import ccxt


class ExchangeClient:
    def __init__(self):
        self.exchange = ccxt.kucoin({
            "enableRateLimit": True
        })

    def fetch_candles(self, symbol, timeframe, limit=200):
        candles = self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

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
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])
