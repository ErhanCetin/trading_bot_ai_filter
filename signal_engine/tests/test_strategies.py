"""
Stratejiler modülü için test sınıfları.
"""
import unittest
import pandas as pd
import numpy as np
from signal_engine.indicators import registry as indicator_registry
from signal_engine.strategies import registry as strategy_registry

class TestBaseStrategyFunctions(unittest.TestCase):
    """Temel strateji fonksiyonlarını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek fiyat verileri oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.price_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(102, 5, 100),
            'low': np.random.normal(98, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # High > Low olacak şekilde düzelt
        self.price_data['high'] = self.price_data[['high', 'low']].max(axis=1) + 1
        self.price_data['low'] = self.price_data[['high', 'low']].min(axis=1)
        
        # İndikatör hesaplamaları
        ema_indicator = indicator_registry.create_indicator("ema", {"periods": [9, 21]})
        rsi_indicator = indicator_registry.create_indicator("rsi", {"periods": [14]})
        macd_indicator = indicator_registry.create_indicator("macd")
        adx_indicator = indicator_registry.create_indicator("adx", {"window": 14})
        
        # İndikatörleri uygula
        self.indicator_data = self.price_data.copy()
        self.indicator_data = ema_indicator.calculate(self.indicator_data)
        self.indicator_data = rsi_indicator.calculate(self.indicator_data)
        self.indicator_data = macd_indicator.calculate(self.indicator_data)
        
        # Simüle edilmiş ADX verileri (adx_indicator kullanmak yerine)
        self.indicator_data['adx'] = np.random.normal(25, 10, 100)
        self.indicator_data['di_pos'] = np.random.normal(25, 10, 100)
        self.indicator_data['di_neg'] = np.random.normal(25, 10, 100)
    
    def test_strategy_registry(self):
        """Strateji registry'sini test et."""
        # Tüm stratejileri al
        all_strategies = strategy_registry.get_all_strategies()
        
        # En az bir strateji olmalı
        self.assertGreater(len(all_strategies), 0, "Hiç strateji bulunmuyor")
        
        # Trend stratejilerini al
        trend_strategies = strategy_registry.get_strategies_by_category("trend")
        
        # En az bir trend stratejisi olmalı
        self.assertGreater(len(trend_strategies), 0, "Hiç trend stratejisi bulunmuyor")

class TestTrendStrategies(unittest.TestCase):
    """Trend stratejilerini test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek fiyat verileri oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.price_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(102, 5, 100),
            'low': np.random.normal(98, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # High > Low olacak şekilde düzelt
        self.price_data['high'] = self.price_data[['high', 'low']].max(axis=1) + 1
        self.price_data['low'] = self.price_data[['high', 'low']].min(axis=1)
        
        # İndikatörleri hesapla
        self.indicator_data = self.price_data.copy()
        
        # RSI
        rsi_values = np.random.normal(50, 15, 100)
        self.indicator_data['rsi_14'] = np.clip(rsi_values, 0, 100)
        
        # Trend göstergeleri
        self.indicator_data['adx'] = np.random.normal(25, 10, 100)
        self.indicator_data['di_pos'] = np.random.normal(25, 10, 100)
        self.indicator_data['di_neg'] = np.random.normal(25, 10, 100)
        
        # MACD
        self.indicator_data['macd_line'] = np.random.normal(0, 1, 100)
        self.indicator_data['macd_signal'] = np.random.normal(0, 1, 100)
        self.indicator_data['macd_histogram'] = self.indicator_data['macd_line'] - self.indicator_data['macd_signal']
        
        # EMA
        self.indicator_data['ema_20'] = self.indicator_data['close'].rolling(window=20).mean()
        self.indicator_data['ema_50'] = self.indicator_data['close'].rolling(window=50).mean()
        
        # Market Regime
        regimes = ['strong_uptrend', 'weak_uptrend', 'ranging', 'weak_downtrend', 'strong_downtrend']
        self.indicator_data['market_regime'] = np.random.choice(regimes, 100)
    
    def test_trend_following_strategy(self):
        """Trend Following stratejisini test et."""
        # Strateji oluştur
        trend_strategy = strategy_registry.create_strategy("trend_following", {"adx_threshold": 20})
        
        # Doğru sınıf tipini kontrol et
        self.assertIsNotNone(trend_strategy, "Trend Following stratejisi oluşturulamadı")
        
        # Stratejiyi uygula
        signals = trend_strategy.generate_signals(self.indicator_data)
        
        # Sinyal üretildi mi kontrol et
        self.assertIsInstance(signals, pd.Series, "Sinyaller bir pandas Series değil")
        self.assertEqual(len(signals), len(self.indicator_data), "Sinyal sayısı veri sayısına eşit değil")
        
        # Sinyaller -1, 0, 1 değerlerinden oluşmalı
        unique_values = signals.unique()
        for val in unique_values:
            self.assertIn(val, [-1, 0, 1], f"Geçersiz sinyal değeri: {val}")

class TestBreakoutStrategies(unittest.TestCase):
    """Kırılma stratejilerini test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek fiyat verileri oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.price_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(102, 5, 100),
            'low': np.random.normal(98, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # High > Low olacak şekilde düzelt
        self.price_data['high'] = self.price_data[['high', 'low']].max(axis=1) + 1
        self.price_data['low'] = self.price_data[['high', 'low']].min(axis=1)
        
        # İndikatörleri hesapla
        self.indicator_data = self.price_data.copy()
        
        # Volatilite göstergeleri
        self.indicator_data['atr'] = np.random.normal(2, 0.5, 100)
        self.indicator_data['bollinger_upper'] = self.indicator_data['close'] + 2 * np.random.normal(2, 0.5, 100)
        self.indicator_data['bollinger_lower'] = self.indicator_data['close'] - 2 * np.random.normal(2, 0.5, 100)
        self.indicator_data['bollinger_middle'] = self.indicator_data['close']
        
        # Volume göstergeleri
        self.indicator_data['volume_ma'] = self.indicator_data['volume'].rolling(window=20).mean()
    
    def test_volatility_breakout_strategy(self):
        """Volatility Breakout stratejisini test et."""
        # Strateji oluştur
        volatility_strategy = strategy_registry.create_strategy("volatility_breakout", 
                                                              {"atr_multiplier": 2.0, "lookback_period": 14})
        
        # Doğru sınıf tipini kontrol et
        self.assertIsNotNone(volatility_strategy, "Volatility Breakout stratejisi oluşturulamadı")
        
        # Stratejiyi uygula
        signals = volatility_strategy.generate_signals(self.indicator_data)
        
        # Sinyal üretildi mi kontrol et
        self.assertIsInstance(signals, pd.Series, "Sinyaller bir pandas Series değil")
        self.assertEqual(len(signals), len(self.indicator_data), "Sinyal sayısı veri sayısına eşit değil")
        
        # Sinyaller -1, 0, 1 değerlerinden oluşmalı
        unique_values = signals.unique()
        for val in unique_values:
            self.assertIn(val, [-1, 0, 1], f"Geçersiz sinyal değeri: {val}")


if __name__ == '__main__':
    unittest.main()