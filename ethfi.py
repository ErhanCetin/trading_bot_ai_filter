import pandas as pd
import numpy as np
import ta
from ta.trend import ADXIndicator


# Load data
df = pd.read_csv('ETHFIUSDT_5m_kline_export.csv')

# Preprocess data
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df.set_index('open_time', inplace=True)
df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})

# Advanced indicator calculations
df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
df['ema_100'] = ta.trend.ema_indicator(df['close'], window=100)
df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
df['rsi'] = ta.momentum.rsi(df['close'], window=14)
df['stoch_rsi'] = ta.momentum.stochrsi(df['close'], window=14)
df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
df['macd_diff'] = ta.trend.macd_diff(df['close'])
df['tsi'] = ta.momentum.tsi(df['close'])
df['supertrend'] = ta.trend.stc(df['close'], window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3)
df['bollinger_b'] = ta.volatility.bollinger_pband(df['close'], window=20, window_dev=2)
adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
df['adx'] = adx.adx()
df['dmi_plus'] = adx.adx_pos()
df['dmi_minus'] = adx.adx_neg()
df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
df['vwap_dev'] = (df['close'] - (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()) / df['close']

# Signal scoring mechanism
latest = df.iloc[-1]

score = 0
reasons = []

# Trend direction by EMA ribbon
if latest['close'] > latest['ema_20'] > latest['ema_50'] > latest['ema_100'] > latest['ema_200']:
    score += 2
    reasons.append('Strong bullish EMA ribbon')
elif latest['close'] < latest['ema_20'] < latest['ema_50'] < latest['ema_100'] < latest['ema_200']:
    score -= 2
    reasons.append('Strong bearish EMA ribbon')

# Momentum confirmation
if latest['rsi'] > 55 and latest['stoch_rsi'] > 0.8 and latest['tsi'] > 0:
    score += 1.5
    reasons.append('Momentum bullish (RSI, StochRSI, TSI)')
elif latest['rsi'] < 45 and latest['stoch_rsi'] < 0.2 and latest['tsi'] < 0:
    score -= 1.5
    reasons.append('Momentum bearish (RSI, StochRSI, TSI)')

# Volume confirmation
if latest['obv'] > df['obv'].rolling(window=50).mean().iloc[-1] and latest['vwap_dev'] > 0:
    score += 1
    reasons.append('Volume supports bullish bias')
elif latest['obv'] < df['obv'].rolling(window=50).mean().iloc[-1] and latest['vwap_dev'] < 0:
    score -= 1
    reasons.append('Volume supports bearish bias')

# ADX and DMI
if latest['adx'] > 20:
    if latest['dmi_plus'] > latest['dmi_minus']:
        score += 1
        reasons.append('ADX strong, DMI bullish')
    elif latest['dmi_plus'] < latest['dmi_minus']:
        score -= 1
        reasons.append('ADX strong, DMI bearish')

# Volatility check for possible squeeze
if latest['bollinger_b'] < 0.05:
    reasons.append('Volatility squeeze detected')

# Final decision logic
if score >= 4:
    signal = 'STRONG LONG'
elif score >= 2:
    signal = 'LONG'
elif score <= -4:
    signal = 'STRONG SHORT'
elif score <= -2:
    signal = 'SHORT'
else:
    signal = 'NO CLEAR SIGNAL'

# Reporting
print(f'🚨 SIGNAL: {signal} (Score: {score})')
print('📋 Reasons:')
for reason in reasons:
    print('-', reason)

print('\n📊 Latest Indicators Snapshot:')
print(latest[['close', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'rsi', 'stoch_rsi', 'mfi', 'macd_diff', 'tsi', 'supertrend', 'bollinger_b', 'adx', 'dmi_plus', 'dmi_minus', 'obv', 'vwap_dev']].round(4))
