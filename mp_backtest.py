import asyncio

async def run_backtest(symbol: str, days: str) -> str:
    # Dummydata tills riktig KuCoin-data är klar
    await asyncio.sleep(1)
    return f"📊 Backtest klart för {symbol.upper()} ({days})\nTrades: 12\nPnL: +4.3%"