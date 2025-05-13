"""
İndikatörler modülü için test sınıfları.
"""
import unittest
import pandas as pd
import numpy as np
from signal_engine.indicators import registry

class TestBaseIndicators(unittest.TestCase):
    """Temel indikatörleri test eden sınıf."""
    
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
        self.price_data['open_time'] = dates
        
        # High > Low olacak şekilde düzelt
        self.price_data['high'] = self.price_data[['high', 'low']].max(axis=1) + 1
        self.price_data['low'] = self.price_data[['high', 'low']].min(axis=1)
        
    def test_ema_indicator(self):
        """EMA indikatörünü test et."""
        # Indikatör oluştur
        ema_indicator = registry.create_indicator("ema", {"periods": [9, 21]})
        
        # Doğru sınıf tipini kontrol et
        self.assertIsNotNone(ema_indicator, "EMA indikatörü oluşturulamadı")
        
        # İndikatörü hesapla
        result_df = ema_indicator.calculate(self.price_data)
        
        # Sonuçları kontrol et
        self.assertIn("ema_9", result_df.columns, "EMA-9 sütunu eksik")
        self.assertIn("ema_21", result_df.columns, "EMA-21 sütunu eksik")
        
        # NaN değerleri atlayarak ilk değerleri kontrol et
        self.assertFalse(result_df["ema_9"].iloc[-1].isna(), "EMA-9 değerleri hesaplanamadı")
        self.assertFalse(result_df["ema_21"].iloc[-1].isna(), "EMA-21 değerleri hesaplanamadı")
    
    def test_rsi_indicator(self):
        """RSI indikatörünü test et."""
        # Indikatör oluştur
        rsi_indicator = registry.create_indicator("rsi", {"periods": [14]})
        
        # Doğru sınıf tipini kontrol et
        self.assertIsNotNone(rsi_indicator, "RSI indikatörü oluşturulamadı")
        
        # İndikatörü hesapla
        result_df = rsi_indicator.calculate(self.price_data)
        
        # Sonuçları kontrol et
        self.assertIn("rsi_14", result_df.columns, "RSI-14 sütunu eksik")
        
        # NaN değerleri atlayarak son değerleri kontrol et
        self.assertFalse(result_df["rsi_14"].iloc[-1].isna(), "RSI-14 değerleri hesaplanamadı")
        
        # RSI değerleri 0-100 arasında olmalı
        valid_values = result_df["rsi_14"].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in valid_values), "RSI değerleri 0-100 arasında değil")

class TestAdvancedIndicators(unittest.TestCase):
    """Gelişmiş indikatörleri test eden sınıf."""
    
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
        self.price_data['open_time'] = dates
        
        # High > Low olacak şekilde düzelt
        self.price_data['high'] = self.price_data[['high', 'low']].max(axis=1) + 1
        self.price_data['low'] = self.price_data[['high', 'low']].min(axis=1)
    
    def test_supertrend_indicator(self):
        """Supertrend indikatörünü test et."""
        # Indikatör oluştur
        supertrend_indicator = registry.create_indicator("supertrend", {"atr_period": 10, "atr_multiplier": 3.0})
        
        # Doğru sınıf tipini kontrol et
        self.assertIsNotNone(supertrend_indicator, "Supertrend indikatörü oluşturulamadı")
        
        # İndikatörü hesapla
        result_df = supertrend_indicator.calculate(self.price_data)
        
        # Sonuçları kontrol et
        self.assertIn("supertrend", result_df.columns, "Supertrend sütunu eksik")
        self.assertIn("supertrend_direction", result_df.columns, "Supertrend yön sütunu eksik")
        
        # NaN değerleri atlayarak son değerleri kontrol et
        self.assertFalse(result_df["supertrend"].iloc[-1].isna(), "Supertrend değerleri hesaplanamadı")
        
        # Supertrend yön değerleri True/False olmalı
        valid_directions = result_df["supertrend_direction"].dropna()
        self.assertTrue(all(isinstance(val, bool) for val in valid_directions), 
                        "Supertrend yön değerleri True/False değil")

class TestStatisticalIndicators(unittest.TestCase):
    """İstatistiksel indikatörleri test eden sınıf."""
    
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
    
    def test_zscore_indicator(self):
        """Z-Score indikatörünü test et."""
        # Indikatör oluştur
        zscore_indicator = registry.create_indicator("zscore", {"window": 20, "apply_to": ["close"]})
        
        # Doğru sınıf tipini kontrol et
        self.assertIsNotNone(zscore_indicator, "Z-Score indikatörü oluşturulamadı")
        
        # İndikatörü hesapla
        result_df = zscore_indicator.calculate(self.price_data)
        
        # Sonuçları kontrol et
        self.assertIn("close_zscore", result_df.columns, "close_zscore sütunu eksik")
        
        # NaN değerleri atlayarak son değerleri kontrol et (ilk 20 satır NaN olmalı)
        self.assertFalse(result_df["close_zscore"].iloc[-1].isna(), "Z-Score değerleri hesaplanamadı")
        
        # Z-Score değerleri genellikle -3 ile 3 arasında olmalı (kesin değil, ama aykırı değerler nadir olmalı)
        valid_zscores = result_df["close_zscore"].dropna()
        extreme_values = valid_zscores[(valid_zscores > 3) | (valid_zscores < -3)]
        self.assertLess(len(extreme_values) / len(valid_zscores), 0.1, 
                       "Z-Score değerlerinde çok fazla aykırı değer var")


if __name__ == '__main__':
    unittest.main()