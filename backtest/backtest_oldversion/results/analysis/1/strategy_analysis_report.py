# strategy_analysis_report.py

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, MACD
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

# Load the original OHLCV data (replace with your own file if needed)
df = pd.read_csv("ETHFIUSDT_5m_kline_export.csv")
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

# Ensure numeric types
df[['open', 'high', 'low', 'close', 'volume']] = df[
    ['open', 'high', 'low', 'close', 'volume']
].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# --- Indicator Calculations ---
df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
df['macd'] = MACD(close=df['close']).macd_diff()
df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
df['bb_lower'] = BollingerBands(close=df['close']).bollinger_lband()
df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()

# --- Example Strategies ---
strategies = [
    {"name": "RSI<30 and MACD<0", "logic": lambda row: row['rsi'] < 30 and row['macd'] < 0},
    {"name": "Price<EMA20 and EMA20<EMA50", "logic": lambda row: row['close'] < row['ema_20'] < row['ema_50']},
    {"name": "OBV decreasing and MFI<30", "logic": lambda row: row['obv'] < row['obv'].shift(1) and row['mfi'] < 30},
    {"name": "Price < BB Lower and ATR rising", "logic": lambda row: row['close'] < row['bb_lower'] and row['atr'] > row['atr'].rolling(10).mean()},
]

results = []
for strat in strategies:
    df['signal'] = df.apply(strat['logic'], axis=1)
    trades = df[df['signal']].copy()
    trades['gain_pct'] = (df['close'].shift(-1) - df['close']) / df['close'] * 100
    trades['win'] = trades['gain_pct'] > 0

    total_signals = len(trades)
    win_rate = trades['win'].mean() * 100 if total_signals > 0 else 0
    avg_gain = trades['gain_pct'].mean() if total_signals > 0 else 0

    results.append({
        "Strategy": strat['name'],
        "Signals": total_signals,
        "Win Rate %": round(win_rate, 2),
        "Avg Gain %": round(avg_gain, 2)
    })

# Save results to CSV
result_df = pd.DataFrame(results)
result_df.to_csv("strategy_test_results.csv", index=False)
print("✅ Strategy test completed. Results saved to strategy_test_results.csv")
