import pandas as pd
import requests
import argparse
import argcomplete
import time

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_binance_klines(symbol: str, interval: str, limit: int):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_end_time = None
    remaining = limit
    
    
    while remaining > 0:
        fetch_amount = min(remaining, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": fetch_amount}
        if current_end_time:
            params["endTime"] = current_end_time
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
            
        all_data = data + all_data
        remaining -= len(data)
        
        current_end_time = data[0][0] - 1
        time.sleep(0.1)
        
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(all_data[-limit:], columns=columns)
    df = df.apply(pd.to_numeric)
    return df[['timestamp', 'close_time', 'open', 'high', 'low', 'close', 'volume']]

def fetch_binance_klines_range(symbol: str, interval: str, start_time: int, end_time: int):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start_time = start_time
    
    while current_start_time <= end_time:
        params = {
            "symbol": symbol, 
            "interval": interval, 
            "limit": 1000, 
            "startTime": int(current_start_time),
            "endTime": int(end_time)
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        current_start_time = data[-1][6] + 1
        time.sleep(0.1)
        
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(all_data, columns=columns)
    df = df.apply(pd.to_numeric)
    return df[['timestamp', 'close_time', 'open', 'high', 'low', 'close', 'volume']]

def importKlines(symbol: str, interval: str, limit: int, history_lags: int, enable_long_tendency: bool = False, enable_micro_tendency: bool = False):
    micro_intervals_map = {
        "1M": ["1w", "1d"], "1w": ["1d", "4h"], "3d": ["1d", "4h"], "1d": ["4h", "1h"],
        "12h": ["4h", "1h"], "8h": ["4h", "1h"], "6h": ["1h", "15m"], "4h": ["1h", "15m"],
        "2h": ["15m", "5m"], "1h": ["15m", "5m"], "30m": ["5m", "1m"], "15m": ["5m", "1m"],
        "5m": ["1m"], "3m": ["1m"], "1m": ["1s"], "1s": []
    }
    macro_intervals_map = {
        "1s": ["1m"], "1m": ["5m", "15m", "1h"], "3m": ["15m", "1h"], "5m": ["15m", "1h", "4h"],
        "15m": ["1h", "4h", "1d"], "30m": ["1h", "4h", "1d"], "1h": ["4h", "1d", "1w"], "2h": ["4h", "1d"],
        "4h": ["1d", "1w"], "6h": ["1d", "1w"], "8h": ["1d", "1w"], "12h": ["1d", "1w"],
        "1d": ["1w", "1M"], "3d": ["1w", "1M"], "1w": ["1M"], "1M": []
    }

    df = fetch_binance_klines(symbol, interval, limit)
    
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['sma_14'] = df['close'].rolling(window=14).mean()
    df['volatility_14'] = df['close'].rolling(window=14).std()

    if enable_micro_tendency:
        micro_list = micro_intervals_map.get(interval, [])
        for micro_int in micro_list:
            start_ts = df['timestamp'].iloc[0]
            end_ts = df['close_time'].iloc[-1]
            
            df_micro = fetch_binance_klines_range(symbol, micro_int, start_ts, end_ts)
            if df_micro.empty:
                continue
            
            df_micro[f'{micro_int}_rsi_14'] = compute_rsi(df_micro['close'], 14)
            df_micro[f'{micro_int}_volatility_14'] = df_micro['close'].rolling(window=14).std()
            
            df_micro = df_micro.rename(columns={'close': f'{micro_int}_close'})
            df_micro = df_micro.dropna(subset=[f'{micro_int}_rsi_14'])
            
            df_main_times = df[['close_time']].rename(columns={'close_time': 'main_close_time'})
            
            merged = pd.merge_asof(
                df_micro.sort_values('close_time'), 
                df_main_times.sort_values('main_close_time'), 
                left_on='close_time', 
                right_on='main_close_time',
                direction='forward'
            )
            
            merged['micro_idx'] = merged.groupby('main_close_time').cumcount(ascending=False)
            
            pivot_df = merged.pivot(index='main_close_time', columns='micro_idx', values=[f'{micro_int}_close', f'{micro_int}_rsi_14', f'{micro_int}_volatility_14'])
            pivot_df.columns = [f"{col[0]}_minus_{col[1]}" for col in pivot_df.columns]
            
            df = pd.merge(df, pivot_df, left_on='close_time', right_index=True, how='left')

    if enable_long_tendency:
        macro_list = macro_intervals_map.get(interval, [])
        for macro_int in macro_list:
            df_macro = fetch_binance_klines(symbol, macro_int, limit)
            if df_macro.empty:
                continue
                
            df_macro[f'{macro_int}_rsi_14'] = compute_rsi(df_macro['close'], 14)
            df_macro[f'{macro_int}_sma_14'] = df_macro['close'].rolling(window=14).mean()
            df_macro[f'{macro_int}_volatility_14'] = df_macro['close'].rolling(window=14).std()
            
            df_macro = df_macro.rename(columns={'close': f'{macro_int}_close'})
            df_macro = df_macro.dropna(subset=[f'{macro_int}_rsi_14'])
            
            merged_macro = pd.merge_asof(
                df[['close_time']].sort_values('close_time'),
                df_macro[['close_time', f'{macro_int}_close', f'{macro_int}_rsi_14', f'{macro_int}_sma_14', f'{macro_int}_volatility_14']].sort_values('close_time'),
                on='close_time',
                direction='backward'
            )
            df = pd.merge(df, merged_macro, on='close_time', how='left')

    new_cols = {}
    cols_to_shift = [c for c in df.columns if c not in ['timestamp', 'close_time']]
    
    for i in range(1, history_lags + 1):
        for col in cols_to_shift:
            new_cols[f'{col}_T-{i}'] = df[col].shift(i)

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=['close_time'])
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--symbol",
            type = str,
            default= "BTCUSDT", 
            help="Binance symbol to fetch (BTCUSDT , SOLUSDT)"
            )
    parser.add_argument(
            "--interval",
            type = str,
            default= "1h",
            choices=["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"], 
            help="Interval of the candles"
            )
    parser.add_argument(
            "--limit",
            type = int,
            default= 2000,
            help="Max candle to fetch"
            )
    parser.add_argument(
            "--lag",
            type = int,
            default= 10,
            help="How much candle in the past each candle need to know"
            )
    parser.add_argument(
            "--enable-long-tendency",
            action="store_true",
            help="Fetch macro tendency data (e.g. 15m, 1h, 4h contexts for 5m candles)"
            )
    parser.add_argument(
            "--enable-micro-tendency",
            action="store_true",
            help="Fetch micro tendency data (e.g. 1m context for 5m candles)"
            )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    df = importKlines(args.symbol, args.interval, args.limit, args.lag, args.enable_long_tendency, args.enable_micro_tendency)
    
    suffix = ""
    if args.enable_long_tendency: suffix += "_macro"
    if args.enable_micro_tendency: suffix += "_micro"
    output_name = f"{args.symbol}{args.interval}{args.limit}{suffix}.parquet"
    
    df.to_parquet(output_name)

if __name__ == "__main__":
    main()
