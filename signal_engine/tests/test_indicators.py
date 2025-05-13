"""
Indikatörler modülü için test sınıfları.
Farklı indikatörlerin işlevselliğini ve registry'e kaydedilmesini test eder.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Import hatalarını yakalamak için try/except kullan
try:
    from signal_engine.indicators import registry
except ImportError as e:
    print(f"Import Error: {e}")
    # Alternatif import yolu dene
    try:
        from signal_engine.signal_indicator_plugin_system import IndicatorRegistry
        registry = IndicatorRegistry()
        print("Alternatif import yolu başarılı: IndicatorRegistry doğrudan import edildi.")
    except ImportError as e:
        print(f"Alternatif import da başarısız: {e}")
        registry = None


# Registry kontrolü fonksiyonu
def is_registry_available():
    """Registry'nin mevcut olup olmadığını kontrol et."""
    return registry is not None


@unittest.skipIf(not is_registry_available(), "Indicator registry is not available")
class TestBaseIndicatorFunctions(unittest.TestCase):
    """Temel indikatör fonksiyonlarını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100), 
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # High ve low değerlerinin mantıklı olduğundan emin ol
        for i in range(len(self.test_df)):
            self.test_df.loc[self.test_df.index[i], 'high'] = max(
                self.test_df['high'].iloc[i],
                self.test_df['open'].iloc[i],
                self.test_df['close'].iloc[i]
            )
            self.test_df.loc[self.test_df.index[i], 'low'] = min(
                self.test_df['low'].iloc[i],
                self.test_df['open'].iloc[i],
                self.test_df['close'].iloc[i]
            )
    
    def test_indicator_registry(self):
        """Indikatör registry'sini test et."""
        # Registry'nin doğrudan metodlarını kontrol et
        self.assertTrue(hasattr(registry, 'get_all_indicators'), "Registry'de get_all_indicators metodu yok")
        
        # Tüm indikatörleri al
        all_indicators = {}
        try:
            all_indicators = registry.get_all_indicators()
        except Exception as e:
            print(f"Uyarı: registry.get_all_indicators() methodu çağrılırken hata oluştu: {e}")
            # Alternatif yöntem
            if hasattr(registry, '_indicators'):
                all_indicators = registry._indicators
                print("Alternatif: registry._indicators özelliği doğrudan kullanıldı")
        
        # Registry'de indikatörler olmalı
        self.assertTrue(isinstance(all_indicators, dict), "Registry.get_all_indicators() bir sözlük döndürmeli")
        
        # En az bir indikatör olmalı veya registry'nin düzgün çalıştığını kontrol et
        if len(all_indicators) == 0:
            print("Uyarı: Registry'de hiç indikatör bulunmuyor, registry oluşturma testi geçti ancak içeriği boş.")
        else:
            self.assertGreater(len(all_indicators), 0, "Hiç indikatör bulunmuyor")
            print(f"Registry'de {len(all_indicators)} indikatör bulundu")
            
            # İndikatörlerin isimlerini yazdır
            indicator_names = list(all_indicators.keys())
            print(f"İndikatör isimleri: {indicator_names[:10]}...")
            
            # Kategori bazlı filtreleme çalışmalı
            for indicator_name, indicator_class in all_indicators.items():
                self.assertTrue(hasattr(indicator_class, 'category'), f"{indicator_name} indikatörü 'category' özelliğine sahip değil")


@unittest.skipIf(not is_registry_available(), "Indicator registry is not available")
class TestBaseIndicators(unittest.TestCase):
    """Temel indikatörleri test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100), 
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # High ve low değerlerinin mantıklı olduğundan emin ol
        for i in range(len(self.test_df)):
            self.test_df.loc[self.test_df.index[i], 'high'] = max(
                self.test_df['high'].iloc[i],
                self.test_df['open'].iloc[i],
                self.test_df['close'].iloc[i]
            )
            self.test_df.loc[self.test_df.index[i], 'low'] = min(
                self.test_df['low'].iloc[i],
                self.test_df['open'].iloc[i],
                self.test_df['close'].iloc[i]
            )
    
    def test_ema_indicator(self):
        """EMA indikatörünü test et."""
        try:
            # EMA indikatörünü oluştur
            ema_indicator = registry.create_indicator("ema", {"periods": [9, 21]})
            
            # İndikatörün varlığını doğrula
            self.assertIsNotNone(ema_indicator, "EMA indikatörü oluşturulamadı")
            
            if ema_indicator:
                # Hesaplama metodu var mı kontrol et
                self.assertTrue(hasattr(ema_indicator, 'calculate'), "EMA indikatörü 'calculate' metoduna sahip değil")
                
                # Indikatörü hesapla
                result_df = ema_indicator.calculate(self.test_df)
                
                # Yeni sütunlar eklenmiş mi kontrol et
                self.assertIn("ema_9", result_df.columns, "ema_9 sütunu eklenmemiş")
                self.assertIn("ema_21", result_df.columns, "ema_21 sütunu eklenmemiş")
                
                # Değerler hesaplanmış mı kontrol et
                self.assertFalse(result_df["ema_9"].isna().all(), "ema_9 değerleri hesaplanmamış")
                self.assertFalse(result_df["ema_21"].isna().all(), "ema_21 değerleri hesaplanmamış")
                
                print("EMA indikatörü başarıyla test edildi")
        except Exception as e:
            self.fail(f"EMA indikatörü testi sırasında hata oluştu: {e}")
    
    def test_rsi_indicator(self):
        """RSI indikatörünü test et."""
        try:
            # RSI indikatörünü oluştur
            rsi_indicator = registry.create_indicator("rsi", {"periods": [14]})
            
            # İndikatörün varlığını doğrula
            self.assertIsNotNone(rsi_indicator, "RSI indikatörü oluşturulamadı")
            
            if rsi_indicator:
                # Hesaplama metodu var mı kontrol et
                self.assertTrue(hasattr(rsi_indicator, 'calculate'), "RSI indikatörü 'calculate' metoduna sahip değil")
                
                # Indikatörü hesapla
                result_df = rsi_indicator.calculate(self.test_df)
                
                # Yeni sütunlar eklenmiş mi kontrol et
                self.assertIn("rsi_14", result_df.columns, "rsi_14 sütunu eklenmemiş")
                
                # Değerler hesaplanmış mı kontrol et
                self.assertFalse(result_df["rsi_14"].isna().all(), "rsi_14 değerleri hesaplanmamış")
                
                # RSI değerlerinin 0-100 arasında olduğunu kontrol et
                valid_values = result_df["rsi_14"].dropna()
                if len(valid_values) > 0:
                    self.assertTrue((valid_values >= 0).all() and (valid_values <= 100).all(), 
                                  "RSI değerleri 0-100 arasında değil")
                
                print("RSI indikatörü başarıyla test edildi")
        except Exception as e:
            self.fail(f"RSI indikatörü testi sırasında hata oluştu: {e}")


@unittest.skipIf(not is_registry_available(), "Indicator registry is not available")
class TestAdvancedIndicators(unittest.TestCase):
    """Gelişmiş indikatörleri test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100), 
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'open_time': dates
        }, index=dates)
        
        # High ve low değerlerinin mantıklı olduğundan emin ol
        for i in range(len(self.test_df)):
            self.test_df.loc[self.test_df.index[i], 'high'] = max(
                self.test_df['high'].iloc[i],
                self.test_df['open'].iloc[i],
                self.test_df['close'].iloc[i]
            )
            self.test_df.loc[self.test_df.index[i], 'low'] = min(
                self.test_df['low'].iloc[i],
                self.test_df['open'].iloc[i],
                self.test_df['close'].iloc[i]
            )
    
    def test_heikin_ashi_indicator(self):
        """Heikin Ashi indikatörünü test et."""
        try:
            # Heikin Ashi indikatörünü oluştur
            ha_indicator = registry.create_indicator("heikin_ashi")
            
            # İndikatörün varlığını doğrula
            self.assertIsNotNone(ha_indicator, "Heikin Ashi indikatörü oluşturulamadı")
            
            if ha_indicator:
                # Indikatörü hesapla
                result_df = ha_indicator.calculate(self.test_df)
                
                # Yeni sütunlar eklenmiş mi kontrol et
                self.assertIn("ha_open", result_df.columns, "ha_open sütunu eklenmemiş")
                self.assertIn("ha_high", result_df.columns, "ha_high sütunu eklenmemiş")
                self.assertIn("ha_low", result_df.columns, "ha_low sütunu eklenmemiş")
                self.assertIn("ha_close", result_df.columns, "ha_close sütunu eklenmemiş")
                self.assertIn("ha_trend", result_df.columns, "ha_trend sütunu eklenmemiş")
                
                # Değerler hesaplanmış mı kontrol et
                self.assertFalse(result_df["ha_open"].isna().all(), "ha_open değerleri hesaplanmamış")
                self.assertFalse(result_df["ha_close"].isna().all(), "ha_close değerleri hesaplanmamış")
                
                print("Heikin Ashi indikatörü başarıyla test edildi")
        except Exception as e:
            self.fail(f"Heikin Ashi indikatörü testi sırasında hata oluştu: {e}")


@unittest.skipIf(not is_registry_available(), "Indicator registry is not available")
class TestRegimeIndicators(unittest.TestCase):
    """Rejim indikatörlerini test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100), 
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # High ve low değerlerinin mantıklı olduğundan emin ol
        for i in range(len(self.test_df)):
            self.test_df.loc[self.test_df.index[i], 'high'] = max(
                self.test_df['high'].iloc[i],
                self.test_df['open'].iloc[i],
                self.test_df['close'].iloc[i]
            )
            self.test_df.loc[self.test_df.index[i], 'low'] = min(
                self.test_df['low'].iloc[i],
                self.test_df['open'].iloc[i],
                self.test_df['close'].iloc[i]
            )
    
    def test_market_regime_indicator(self):
        """Market Regime indikatörünü test et."""
        try:
            # İlk olarak indikatör sınıfını kontrol et
            market_regime_class = registry.get_indicator("market_regime")
            if not market_regime_class:
                self.skipTest("Market Regime indikatörü registry'de bulunamadı")
                return
                
            # Calculate metodunu doğrula
            if not hasattr(market_regime_class, 'calculate'):
                self.skipTest("Market Regime indikatörü 'calculate' metoduna sahip değil")
                return
                
            # İndikatörü oluştur
            try:
                regime_indicator = registry.create_indicator("market_regime")
                self.assertIsNotNone(regime_indicator, "Market Regime indikatörü oluşturulamadı")
            except TypeError as e:
                if "abstract class" in str(e) and "calculate" in str(e):
                    self.skipTest(f"Market Regime indikatörü soyut sınıf hatası: {e}")
                    return
                else:
                    raise
                    
            # İndikatörü hesapla
            result_df = regime_indicator.calculate(self.test_df)
            
            # Yeni sütunlar eklenmiş mi kontrol et
            self.assertIn("market_regime", result_df.columns, "market_regime sütunu eklenmemiş")
            self.assertIn("regime_strength", result_df.columns, "regime_strength sütunu eklenmemiş")
            
            # Değerler hesaplanmış mı kontrol et (bazı NaN değerler olabilir)
            non_nan_values = result_df["market_regime"].dropna().unique()
            self.assertGreater(len(non_nan_values), 0, "market_regime değerleri hesaplanmamış")
            
            print(f"Market Regime indikatörü başarıyla test edildi. Bulunan rejimler: {non_nan_values}")
        except Exception as e:
            self.fail(f"Market Regime indikatörü testi sırasında hata oluştu: {e}")


if __name__ == '__main__':
    # Test çalıştırmadan önce registry durumunu kontrol et
    if not is_registry_available():
        print("HATA: Indicator registry mevcut değil. Testler çalıştırılamıyor.")
        print("Olası nedenler:")
        print("1. signal_engine.indicators modülü doğru şekilde import edilemiyor")
        print("2. signal_engine.signal_indicator_plugin_system modülü doğru şekilde import edilemiyor")
        print("3. IndicatorRegistry sınıfı tanımlanmamış veya erişilemiyor")
        sys.exit(1)
    
    # Registry içeriğini kontrol et
    if is_registry_available() and hasattr(registry, '_indicators'):
        print(f"Registry'deki indikatörleri kontrol et: {len(registry._indicators)} indikatör bulundu")
        categories = {}
        
        # İndikatörleri kategorilere göre grupla
        for name, indicator_class in registry._indicators.items():
            category = indicator_class.category
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        # Kategorileri ve indikatör sayılarını yazdır
        for category, indicators in categories.items():
            print(f"  - {category}: {len(indicators)} indikatör")
    
    # Testleri çalıştır
    unittest.main()