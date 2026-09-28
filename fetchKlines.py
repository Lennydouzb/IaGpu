import pandas as pd
import requests
import argparse
import argcomplete
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def importKlines(symbol: str, interval: str, limit: int, depth: int = 4, history_lags: int = 10):
    # 1. Fetch Binance
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(response.json(), columns=columns)
    df = df.apply(pd.to_numeric)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    #technical indicators
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['sma_14'] = df['close'].rolling(window=14).mean()
    df['volatility_14'] = df['close'].rolling(window=14).std()

    #before candles value (kind of memory)
    for i in range(1, history_lags + 1):
        df[f'close_T-{i}'] = df['close'].shift(i)
        df[f'volume_T-{i}'] = df['volume'].shift(i)
        df[f'rsi_T-{i}'] = df['rsi_14'].shift(i)

    df = df.dropna().reset_index(drop=True)

        
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--symbol",
            type = str,
            default= "BTCUSDT", 
            choices=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"], 
            help="Binance symbol to fetch (BTCUSDT , SOLUSDT)"
            )
    parser.add_argument(
            "--interval",
            type = str,
            default= "1h",
            choices=["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"], 
            help="Interval of the candles ()"
            )
    parser.add_argument(
            "--limit",
            type = int,
            default= 20000,
            help="Max candle to fetch"
            )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    df = importKlines(args.symbol, args.interval, args.limit)
    df.to_parquet(args.symbol + str(args.interval)+ str(args.limit) + ".parquet")


if __name__ == "__main__":
    main()
