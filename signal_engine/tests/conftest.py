"""
Pytest yapılandırmaları ve ortak test fixture'ları.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile

@pytest.fixture
def sample_price_data():
    """Test için örnek fiyat verisi oluştur."""
    # Örnek fiyat verileri oluştur
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    price_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(102, 5, 100),
        'low': np.random.normal(98, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000, 200, 100)
    }, index=dates)
    price_data['open_time'] = dates
    
    # High > Low olacak şekilde düzelt
    price_data['high'] = price_data[['high', 'low']].max(axis=1) + 1
    price_data['low'] = price_data[['high', 'low']].min(axis=1)
    
    return price_data

@pytest.fixture
def sample_indicator_data(sample_price_data):
    """Test için örnek indikatör verisi oluştur."""
    # İndikatör verisi oluştur
    indicator_data = sample_price_data.copy()
    
    # RSI
    indicator_data['rsi_14'] = np.clip(np.random.normal(50, 15, 100), 0, 100)
    
    # Trend göstergeleri
    indicator_data['adx'] = np.random.normal(25, 10, 100)
    indicator_data['di_pos'] = np.random.normal(25, 10, 100)
    indicator_data['di_neg'] = np.random.normal(25, 10, 100)
    
    # MACD
    indicator_data['macd_line'] = np.random.normal(0, 1, 100)
    indicator_data['macd_signal'] = np.random.normal(0, 1, 100)
    indicator_data['macd_histogram'] = indicator_data['macd_line'] - indicator_data['macd_signal']
    
    # EMA
    indicator_data['ema_20'] = indicator_data['close'].rolling(window=20).mean()
    indicator_data['ema_50'] = indicator_data['close'].rolling(window=50).mean()
    
    # Bollinger Bands
    indicator_data['bollinger_middle'] = indicator_data['close'].rolling(window=20).mean()
    indicator_data['bollinger_std'] = indicator_data['close'].rolling(window=20).std()
    indicator_data['bollinger_upper'] = indicator_data['bollinger_middle'] + 2 * indicator_data['bollinger_std']
    indicator_data['bollinger_lower'] = indicator_data['bollinger_middle'] - 2 * indicator_data['bollinger_std']
    
    # Market Regime
    regimes = ['strong_uptrend', 'weak_uptrend', 'ranging', 'weak_downtrend', 'strong_downtrend']
    indicator_data['market_regime'] = np.random.choice(regimes, 100)
    
    return indicator_data

@pytest.fixture
def temp_model_dir():
    """Test için geçici model dizini oluştur."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir