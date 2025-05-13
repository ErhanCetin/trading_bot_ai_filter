"""
Sinyal Gücü Hesaplama modülü için test sınıfları.
"""
import unittest
import pandas as pd
import numpy as np
from signal_engine.strength import registry

class TestBaseStrengthFunctions(unittest.TestCase):
    """Temel güç hesaplama fonksiyonlarını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'close': np.random.normal(100, 5, 100),
            'rsi_14': np.clip(np.random.normal(50, 15, 100), 0, 100),
            'adx': np.random.normal(25, 10, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'market_regime': np.random.choice(
                ['strong_uptrend', 'weak_uptrend', 'ranging', 'weak_downtrend', 'strong_downtrend'], 
                100
            ),
            'trend_strength': np.random.randint(0, 100, 100),
            'ema_alignment': np.random.uniform(-1, 1, 100)
        }, index=dates)
        
        # Sinyal serisi oluştur
        self.signals = pd.Series(np.random.choice([-1, 0, 1], 100, p=[0.2, 0.6, 0.2]), index=dates)
    
    def test_strength_registry(self):
        """Güç hesaplama registry'sini test et."""
        # Tüm hesaplayıcıları al
        all_calculators = registry.get_all_calculators()
        
        # En az bir hesaplayıcı olmalı
        self.assertGreater(len(all_calculators), 0, "Hiç güç hesaplayıcı bulunmuyor")
        
        # Bağlam duyarlı hesaplayıcıları al
        context_calculators = registry.get_calculators_by_category("context")
        
        # En az bir bağlam duyarlı hesaplayıcı olmalı
        self.assertGreater(len(context_calculators), 0, "Hiç bağlam duyarlı hesaplayıcı bulunmuyor")

class TestContextStrength(unittest.TestCase):
    """Bağlam duyarlı güç hesaplayıcılarını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'close': np.random.normal(100, 5, 100),
            'market_regime': np.random.choice(
                ['strong_uptrend', 'weak_uptrend', 'ranging', 'weak_downtrend', 'strong_downtrend'], 
                100
            ),
            'regime_strength': np.random.randint(0, 100, 100),
            'volatility_regime': np.random.choice(['low', 'normal', 'high'], 100),
            'volatility_percentile': np.random.randint(0, 100, 100),
            'trend_health': np.random.randint(0, 100, 100)
        }, index=dates)
        
        # Sinyal serisi oluştur (0=no signal, 1=long, -1=short)
        self.signals = pd.Series(np.random.choice([-1, 0, 1], 100, p=[0.2, 0.6, 0.2]), index=dates)
    
    def test_market_context_strength(self):
        """Market Context güç hesaplayıcısını test et."""
        # Hesaplayıcı oluştur
        context_calculator = registry.create_calculator("market_context_strength")
        
        # Doğru sınıf tipini kontrol et
        self.assertIsNotNone(context_calculator, "Market Context güç hesaplayıcısı oluşturulamadı")
        
        # Güç değerlerini hesapla
        strength_values = context_calculator.calculate(self.test_df, self.signals)
        
        # Sonuçları kontrol et
        self.assertIsInstance(strength_values, pd.Series, "Güç değerleri bir pandas Series değil")
        self.assertEqual(len(strength_values), len(self.test_df), "Güç değerleri sayısı veri sayısına eşit değil")
        
        # Güç değerleri 0-100 arasında olmalı
        self.assertTrue(all(0 <= val <= 100 for val in strength_values if not pd.isna(val)), 
                       "Güç değerleri 0-100 arasında değil")
        
        # Sinyal olmayan noktalarda güç değeri 0 olmalı
        for i, signal in enumerate(self.signals):
            if signal == 0:
                self.assertEqual(strength_values.iloc[i], 0, 
                                "Sinyal olmayan noktada güç değeri 0 değil")

class TestPredictiveStrength(unittest.TestCase):
    """Tahmin bazlı güç hesaplayıcılarını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(102, 5, 100),
            'low': np.random.normal(98, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'market_regime': np.random.choice(
                ['strong_uptrend', 'weak_uptrend', 'ranging', 'weak_downtrend', 'strong_downtrend'], 
                100
            ),
            'atr': np.random.normal(2, 0.5, 100),
            'nearest_support': np.random.normal(95, 3, 100),
            'nearest_resistance': np.random.normal(105, 3, 100),
            'bollinger_upper': np.random.normal(110, 5, 100),
            'bollinger_lower': np.random.normal(90, 5, 100)
        }, index=dates)
        
        # High > Low olacak şekilde düzelt
        self.test_df['high'] = self.test_df[['high', 'low']].max(axis=1) + 1
        self.test_df['low'] = self.test_df[['high', 'low']].min(axis=1)
        
        # Sinyal serisi oluştur
        self.signals = pd.Series(np.random.choice([-1, 0, 1], 100, p=[0.2, 0.6, 0.2]), index=dates)
    
    def test_risk_reward_strength(self):
        """Risk/Reward güç hesaplayıcısını test et."""
        # Hesaplayıcı oluştur
        rr_calculator = registry.create_calculator("risk_reward_strength")
        
        # Doğru sınıf tipini kontrol et
        self.assertIsNotNone(rr_calculator, "Risk/Reward güç hesaplayıcısı oluşturulamadı")
        
        # Güç değerlerini hesapla
        strength_values = rr_calculator.calculate(self.test_df, self.signals)
        
        # Sonuçları kontrol et
        self.assertIsInstance(strength_values, pd.Series, "Güç değerleri bir pandas Series değil")
        self.assertEqual(len(strength_values), len(self.test_df), "Güç değerleri sayısı veri sayısına eşit değil")
        
        # Güç değerleri 0-100 arasında olmalı
        self.assertTrue(all(0 <= val <= 100 for val in strength_values if not pd.isna(val)), 
                       "Güç değerleri 0-100 arasında değil")
        
        # Sinyal olmayan noktalarda güç değeri 0 olmalı
        for i, signal in enumerate(self.signals):
            if signal == 0:
                self.assertEqual(strength_values.iloc[i], 0, 
                                "Sinyal olmayan noktada güç değeri 0 değil")


if __name__ == '__main__':
    unittest.main()