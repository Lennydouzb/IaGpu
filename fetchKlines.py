import pandas as pd
import requests
import argparse
import argcomplete

def importKlines(symbol: str, interval, limit: int):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol":symbol, "interval":interval, "limit":limit}
    response = requests.get(url, params = params)
    response.raise_for_status()
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    df = pd.DataFrame(response.json(), columns=columns)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
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
