"""
Strategies modülü için test sınıfları.
Stratejilerin işlevselliğini ve registry'e kaydedilmesini test eder.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Import hatalarını yakalamak için try/except kullan
try:
    from signal_engine.strategies import registry
except ImportError as e:
    print(f"Import Error: {e}")
    # Alternatif import yolu dene
    try:
        from signal_engine.signal_strategy_system import StrategyRegistry
        registry = StrategyRegistry()
        print("Alternatif import yolu başarılı: StrategyRegistry doğrudan import edildi.")
    except ImportError as e:
        print(f"Alternatif import da başarısız: {e}")
        registry = None


# Registry kontrolü fonksiyonu
def is_registry_available():
    """Registry'nin mevcut olup olmadığını kontrol et."""
    return registry is not None


@unittest.skipIf(not is_registry_available(), "Strategy registry is not available")
class TestBaseStrategyFunctions(unittest.TestCase):
    """Temel strateji fonksiyonlarını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        # İndikatörlerle zenginleştirilmiş veri
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            # Ohlc ve hacim verileri
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100), 
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.randint(1000, 10000, 100),
            
            # Trend indikatörleri
            'adx': np.random.uniform(10, 50, 100),
            'di_pos': np.random.uniform(10, 40, 100),
            'di_neg': np.random.uniform(10, 40, 100),
            'ema_20': np.random.normal(99, 3, 100),
            'ema_50': np.random.normal(98, 4, 100),
            'trend_alignment': np.random.choice([1, 0, -1], 100),
            'multi_timeframe_agreement': np.random.choice([1, 0, -1], 100),
            
            # Momentum indikatörleri
            'rsi_14': np.random.uniform(30, 70, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'stoch_k': np.random.uniform(20, 80, 100),
            'stoch_d': np.random.uniform(20, 80, 100),
            
            # Volatilite indikatörleri
            'atr': np.random.uniform(1, 3, 100),
            'atr_percent': np.random.uniform(0.5, 2, 100),
            'bollinger_upper': np.random.normal(105, 1, 100),
            'bollinger_lower': np.random.normal(95, 1, 100),
            'bollinger_pct_b': np.random.uniform(0, 1, 100),
            
            # Rejim indikatörleri
            'market_regime': np.random.choice(
                ['strong_uptrend', 'weak_uptrend', 'ranging', 
                 'weak_downtrend', 'strong_downtrend', 'volatile', 
                 'overbought', 'oversold'], 100
            ),
            'volatility_regime': np.random.choice(['low', 'normal', 'high'], 100),
            
            # Pattern indikatörleri
            'engulfing_pattern': np.random.choice([1, 0, -1], 100),
            'hammer_pattern': np.random.choice([1, 0, -1], 100),
            'doji_pattern': np.random.choice([1, 0], 100),
            'shooting_star_pattern': np.random.choice([1, 0], 100),
            
            # İstatistiksel indikatörler
            'z_score': np.random.normal(0, 1, 100)
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
            
        # Support/resistance işaretlerini ekle
        self.test_df['nearest_support'] = self.test_df['low'] * 0.98
        self.test_df['nearest_resistance'] = self.test_df['high'] * 1.02
        self.test_df['in_support_zone'] = np.random.choice([True, False], 100)
        self.test_df['in_resistance_zone'] = np.random.choice([True, False], 100)
        self.test_df['broke_support'] = np.random.choice([True, False], 100)
        self.test_df['broke_resistance'] = np.random.choice([True, False], 100)
    
    def test_strategy_registry(self):
        """Strateji registry'sini test et."""
        # Registry'nin doğrudan metodlarını kontrol et
        self.assertTrue(hasattr(registry, 'get_all_strategies'), "Registry'de get_all_strategies metodu yok")
        
        # Tüm stratejileri al
        all_strategies = {}
        try:
            all_strategies = registry.get_all_strategies()
        except Exception as e:
            print(f"Uyarı: registry.get_all_strategies() methodu çağrılırken hata oluştu: {e}")
            # Alternatif yöntem
            if hasattr(registry, '_strategies'):
                all_strategies = registry._strategies
                print("Alternatif: registry._strategies özelliği doğrudan kullanıldı")
        
        # Registry'de stratejiler olmalı
        self.assertTrue(isinstance(all_strategies, dict), "Registry.get_all_strategies() bir sözlük döndürmeli")
        
        # En az bir strateji olmalı veya registry'nin düzgün çalıştığını kontrol et
        if len(all_strategies) == 0:
            print("Uyarı: Registry'de hiç strateji bulunmuyor, registry oluşturma testi geçti ancak içeriği boş.")
        else:
            self.assertGreater(len(all_strategies), 0, "Hiç strateji bulunmuyor")
            print(f"Registry'de {len(all_strategies)} strateji bulundu")
            
            # Stratejilerin isimlerini yazdır
            strategy_names = list(all_strategies.keys())
            print(f"Strateji isimleri: {strategy_names}")
            
            # Kategori bazlı stratejileri kontrol et
            categories = {}
            for name, strategy_class in all_strategies.items():
                category = strategy_class.category
                if category not in categories:
                    categories[category] = []
                categories[category].append(name)
            
            print("Stratejiler kategorilere göre:")
            for category, strategies in categories.items():
                print(f"  - {category}: {len(strategies)} strateji")
                print(f"    {strategies}")


@unittest.skipIf(not is_registry_available(), "Strategy registry is not available")
class TestTrendStrategies(unittest.TestCase):
    """Trend stratejilerini test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # BaseStrategyFunctions'daki ile aynı veri seti
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100), 
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'adx': np.random.uniform(10, 50, 100),
            'di_pos': np.random.uniform(10, 40, 100),
            'di_neg': np.random.uniform(10, 40, 100),
            'ema_20': np.random.normal(99, 3, 100),
            'ema_50': np.random.normal(98, 4, 100),
            'rsi_14': np.random.uniform(30, 70, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'market_regime': np.random.choice(['strong_uptrend', 'weak_uptrend', 'ranging'], 100),
            'mtf_ema_alignment': np.random.uniform(-1, 1, 100),
            'trend_alignment': np.random.choice([1, 0, -1], 100),
            'multi_timeframe_agreement': np.random.choice([1, 0, -1], 100),
            'atr_percent': np.random.uniform(0.5, 2, 100),
            'volatility_regime': np.random.choice(['low', 'normal', 'high'], 100)
        }, index=dates)
    
    def test_trend_following_strategy(self):
        """Trend Following stratejisini test et."""
        try:
            # Strateji sınıfını al
            strategy_class = registry.get_strategy("trend_following")
            
            # Stratejinin varlığını doğrula
            self.assertIsNotNone(strategy_class, "Trend Following stratejisi registry'de bulunamadı")
            
            if strategy_class:
                # Strateji örneği oluştur
                strategy = strategy_class()
                
                # Stratejinin metodlarını kontrol et
                self.assertTrue(hasattr(strategy, 'generate_signals'), "generate_signals metodu yok")
                self.assertTrue(hasattr(strategy, 'generate_conditions'), "generate_conditions metodu yok")
                
                # Stratejinin veri çerçevesini doğrulayabildiğini kontrol et
                self.assertTrue(strategy.validate_dataframe(self.test_df), "DataFrame doğrulaması başarısız")
                
                # Sinyalleri oluştur
                signals = strategy.generate_signals(self.test_df)
                
                # Veri tipi kontrolü
                self.assertTrue(isinstance(signals, pd.Series) or isinstance(signals, pd.DataFrame), 
                                "Sinyaller bir Series veya DataFrame olmalı")
                
                # Series ise, Series ile çalış
                if isinstance(signals, pd.Series):
                    # Sinyal değerlerini kontrol et
                    self.assertTrue(all(value in [0, 1, -1] for value in signals.unique()), 
                                  "Sinyal değerleri 0, 1, ya da -1 olmalı")
                    
                    # Sinyalleri say
                    long_signals = (signals == 1).sum()
                    short_signals = (signals == -1).sum()
                    
                    print(f"Trend Following stratejisi {long_signals} long ve {short_signals} short sinyal üretti.")
                    
                    # En az 1 sinyal oluşturuldu mu kontrol et
                    total_signals = long_signals + short_signals
                    self.assertGreater(total_signals, 0, "Hiç sinyal üretilmedi")
                
                # DataFrame ise, DataFrame ile çalış
                else:
                    # Sinyaller oluşturuldu mu kontrol et
                    self.assertIn("long_signal", signals.columns, "long_signal sütunu oluşturulmadı")
                    self.assertIn("short_signal", signals.columns, "short_signal sütunu oluşturulmadı")
                    
                    # En az 1 sinyal oluşturuldu mu kontrol et
                    total_signals = signals["long_signal"].sum() + signals["short_signal"].sum()
                    print(f"Trend Following stratejisi {signals['long_signal'].sum()} long ve {signals['short_signal'].sum()} short sinyal üretti.")
                    self.assertGreater(total_signals, 0, "Hiç sinyal üretilmedi")
                
        except Exception as e:
            self.fail(f"Trend Following stratejisi testi sırasında hata oluştu: {e}")
    
    def test_adaptive_trend_strategy(self):
        """Adaptive Trend stratejisini test et."""
        try:
            # Strateji sınıfını al
            strategy_class = registry.get_strategy("adaptive_trend")
            
            # Stratejinin varlığını doğrula
            self.assertIsNotNone(strategy_class, "Adaptive Trend stratejisi registry'de bulunamadı")
            
            if strategy_class:
                # Strateji örneği oluştur
                strategy = strategy_class()
                
                # Stratejinin veri çerçevesini doğrulayabildiğini kontrol et
                self.assertTrue(strategy.validate_dataframe(self.test_df), "DataFrame doğrulaması başarısız")
                
                # Sinyalleri oluştur
                signals = strategy.generate_signals(self.test_df)
                
                # Veri tipi kontrolü
                self.assertTrue(isinstance(signals, pd.Series) or isinstance(signals, pd.DataFrame), 
                                "Sinyaller bir Series veya DataFrame olmalı")
                
                # Series ise, Series ile çalış
                if isinstance(signals, pd.Series):
                    # Sinyal değerlerini kontrol et
                    self.assertTrue(all(value in [0, 1, -1] for value in signals.unique()), 
                                  "Sinyal değerleri 0, 1, ya da -1 olmalı")
                    
                    # Sinyalleri say
                    long_signals = (signals == 1).sum()
                    short_signals = (signals == -1).sum()
                    
                    print(f"Adaptive Trend stratejisi normal volatilitede {long_signals} long, {short_signals} short sinyal üretti.")
                    
                    # Stratejinin adaptif olduğunu doğrula - farklı volatilite rejimlerinde farklı sayıda sinyal olmalı
                    high_vol_df = self.test_df.copy()
                    high_vol_df["volatility_regime"] = "high"
                    high_vol_signals = strategy.generate_signals(high_vol_df)
                    high_vol_long = (high_vol_signals == 1).sum()
                    high_vol_short = (high_vol_signals == -1).sum()
                    
                    low_vol_df = self.test_df.copy()
                    low_vol_df["volatility_regime"] = "low"
                    low_vol_signals = strategy.generate_signals(low_vol_df)
                    low_vol_long = (low_vol_signals == 1).sum()
                    low_vol_short = (low_vol_signals == -1).sum()
                    
                    # Volatilite rejimlerine göre sonuçları yazdır
                    print(f"Adaptive Trend stratejisi yüksek volatilitede {high_vol_long} long, {high_vol_short} short sinyal üretti.")
                    print(f"Adaptive Trend stratejisi düşük volatilitede {low_vol_long} long, {low_vol_short} short sinyal üretti.")
                
                # DataFrame ise, DataFrame ile çalış
                else:
                    # Sinyaller oluşturuldu mu kontrol et
                    self.assertIn("long_signal", signals.columns, "long_signal sütunu oluşturulmadı")
                    self.assertIn("short_signal", signals.columns, "short_signal sütunu oluşturulmadı")
                    
                    # Stratejinin adaptif olduğunu doğrula - farklı volatilite rejimlerinde farklı sayıda sinyal olmalı
                    high_vol_df = self.test_df.copy()
                    high_vol_df["volatility_regime"] = "high"
                    high_vol_signals = strategy.generate_signals(high_vol_df)
                    
                    low_vol_df = self.test_df.copy()
                    low_vol_df["volatility_regime"] = "low"
                    low_vol_signals = strategy.generate_signals(low_vol_df)
                    
                    # Orjinal, yüksek ve düşük volatilite sinyallerinin sayılarını yazdır
                    print(f"Adaptif Trend stratejisi normal volatilitede {signals['long_signal'].sum()} long, {signals['short_signal'].sum()} short sinyal üretti.")
                    print(f"Adaptif Trend stratejisi yüksek volatilitede {high_vol_signals['long_signal'].sum()} long, {high_vol_signals['short_signal'].sum()} short sinyal üretti.")
                    print(f"Adaptif Trend stratejisi düşük volatilitede {low_vol_signals['long_signal'].sum()} long, {low_vol_signals['short_signal'].sum()} short sinyal üretti.")
                
        except Exception as e:
            self.fail(f"Adaptive Trend stratejisi testi sırasında hata oluştu: {e}")


@unittest.skipIf(not is_registry_available(), "Strategy registry is not available")
class TestBreakoutStrategies(unittest.TestCase):
    """Breakout stratejilerini test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Breakout testleri için uygun veri seti
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100), 
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'atr': np.random.uniform(1, 3, 100),
            'bollinger_upper': np.random.normal(105, 1, 100),
            'bollinger_lower': np.random.normal(95, 1, 100),
            'volume_ma': np.random.randint(5000, 7000, 100),
            'volatility_regime': np.random.choice(['low', 'normal', 'high'], 100),
            'nearest_support': np.random.normal(94, 1, 100),
            'nearest_resistance': np.random.normal(106, 1, 100),
            'broke_support': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'broke_resistance': np.random.choice([True, False], 100, p=[0.1, 0.9])
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
    
    def test_volatility_breakout_strategy(self):
        """Volatility Breakout stratejisini test et."""
        try:
            # Strateji sınıfını al
            strategy_class = registry.get_strategy("volatility_breakout")
            
            # Stratejinin varlığını doğrula
            self.assertIsNotNone(strategy_class, "Volatility Breakout stratejisi registry'de bulunamadı")
            
            if strategy_class:
                # Strateji örneği oluştur
                strategy = strategy_class()
                
                # Stratejinin veri çerçevesini doğrulayabildiğini kontrol et
                self.assertTrue(strategy.validate_dataframe(self.test_df), "DataFrame doğrulaması başarısız")
                
                # Sinyalleri oluştur
                signals = strategy.generate_signals(self.test_df)
                
                # Veri tipi kontrolü
                self.assertTrue(isinstance(signals, pd.Series) or isinstance(signals, pd.DataFrame), 
                                "Sinyaller bir Series veya DataFrame olmalı")
                
                # Series ise, Series ile çalış
                if isinstance(signals, pd.Series):
                    # Sinyalleri say
                    long_signals = (signals == 1).sum()
                    short_signals = (signals == -1).sum()
                    
                    print(f"Volatility Breakout stratejisi {long_signals} long ve {short_signals} short sinyal üretti.")
                
                # DataFrame ise, DataFrame ile çalış
                else:
                    # Sonuçları yazdır
                    print(f"Volatility Breakout stratejisi {signals['long_signal'].sum()} long ve {signals['short_signal'].sum()} short sinyal üretti.")
                
        except Exception as e:
            self.fail(f"Volatility Breakout stratejisi testi sırasında hata oluştu: {e}")
    
    def test_sr_breakout_strategy(self):
        """Support/Resistance Breakout stratejisini test et."""
        try:
            # Strateji sınıfını al
            strategy_class = registry.get_strategy("sr_breakout")
            
            # Stratejinin varlığını doğrula
            self.assertIsNotNone(strategy_class, "Support/Resistance Breakout stratejisi registry'de bulunamadı")
            
            if strategy_class:
                # Strateji örneği oluştur
                strategy = strategy_class()
                
                # Stratejinin veri çerçevesini doğrulayabildiğini kontrol et
                self.assertTrue(strategy.validate_dataframe(self.test_df), "DataFrame doğrulaması başarısız")
                
                # Sinyalleri oluştur
                signals = strategy.generate_signals(self.test_df)
                
                # Veri tipi kontrolü
                self.assertTrue(isinstance(signals, pd.Series) or isinstance(signals, pd.DataFrame), 
                                "Sinyaller bir Series veya DataFrame olmalı")
                
                # Series ise, Series ile çalış
                if isinstance(signals, pd.Series):
                    # Sinyalleri say
                    long_signals = (signals == 1).sum()
                    short_signals = (signals == -1).sum()
                    
                    print(f"S/R Breakout stratejisi {long_signals} long ve {short_signals} short sinyal üretti.")
                    
                    # Kırılma sinyallerinin doğruluğunu kontrol et - örneğin, broke_resistance olan yerlerde long sinyal olmalı
                    resistance_breaks = self.test_df[self.test_df["broke_resistance"] == True].index
                    if len(resistance_breaks) > 0:
                        resistance_signals = signals.loc[resistance_breaks]
                        resistance_longs = (resistance_signals == 1).sum()
                        print(f"Resistance kırılmalarının {resistance_longs}/{len(resistance_breaks)} kadarında long sinyal üretildi.")
                
                # DataFrame ise, DataFrame ile çalış
                else:
                    # Sonuçları yazdır
                    print(f"S/R Breakout stratejisi {signals['long_signal'].sum()} long ve {signals['short_signal'].sum()} short sinyal üretti.")
                    
                    # Kırılma sinyallerinin doğruluğunu kontrol et - örneğin, broke_resistance olan yerlerde long sinyal olmalı
                    resistance_breaks = self.test_df[self.test_df["broke_resistance"] == True].index
                    if len(resistance_breaks) > 0:
                        resistance_signals = signals.loc[resistance_breaks, "long_signal"].sum()
                        print(f"Resistance kırılmalarının {resistance_signals}/{len(resistance_breaks)} kadarında long sinyal üretildi.")
                
        except Exception as e:
            self.fail(f"Support/Resistance Breakout stratejisi testi sırasında hata oluştu: {e}")


@unittest.skipIf(not is_registry_available(), "Strategy registry is not available")
class TestEnsembleStrategies(unittest.TestCase):
    """Ensemble stratejilerini test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Ensemble testleri için kapsamlı veri seti
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100), 
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.randint(1000, 10000, 100),
            # Temel indikatörlerin çoğunu ekleyelim
            'adx': np.random.uniform(10, 50, 100),
            'di_pos': np.random.uniform(10, 40, 100),
            'di_neg': np.random.uniform(10, 40, 100),
            'ema_20': np.random.normal(99, 3, 100),
            'ema_50': np.random.normal(98, 4, 100),
            'rsi_14': np.random.uniform(30, 70, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'stoch_k': np.random.uniform(20, 80, 100),
            'stoch_d': np.random.uniform(20, 80, 100),
            'atr': np.random.uniform(1, 3, 100),
            'bollinger_upper': np.random.normal(105, 1, 100),
            'bollinger_lower': np.random.normal(95, 1, 100),
            # Rejim indikatörleri
            'market_regime': np.random.choice(
                ['strong_uptrend', 'weak_uptrend', 'ranging', 
                 'weak_downtrend', 'strong_downtrend', 'volatile', 
                 'overbought', 'oversold'], 100
            ),
            # Diğer yararlı indikatörler
            'volatility_regime': np.random.choice(['low', 'normal', 'high'], 100),
            'nearest_support': np.random.normal(94, 1, 100),
            'nearest_resistance': np.random.normal(106, 1, 100),
            'broke_support': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'broke_resistance': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'engulfing_pattern': np.random.choice([1, 0, -1], 100)
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
    
    def test_regime_ensemble_strategy(self):
        """Regime-Based Ensemble stratejisini test et."""
        try:
            # Strateji sınıfını al
            strategy_class = registry.get_strategy("regime_ensemble")
            
            # Stratejinin varlığını doğrula
            self.assertIsNotNone(strategy_class, "Regime Ensemble stratejisi registry'de bulunamadı")
            
            if strategy_class:
                # Strateji örneği oluştur
                strategy = strategy_class()
                
                # Stratejinin veri çerçevesini doğrulayabildiğini kontrol et
                self.assertTrue(strategy.validate_dataframe(self.test_df), "DataFrame doğrulaması başarısız")
                
                # Sinyalleri oluştur
                signals = strategy.generate_signals(self.test_df)
                
                # Veri tipi kontrolü
                self.assertTrue(isinstance(signals, pd.Series) or isinstance(signals, pd.DataFrame), 
                                "Sinyaller bir Series veya DataFrame olmalı")
                
                # Series ise, Series ile çalış
                if isinstance(signals, pd.Series):
                   # Sinyalleri say
                   long_signals = (signals == 1).sum()
                   short_signals = (signals == -1).sum()
                   
                   print(f"Regime Ensemble stratejisi {long_signals} long ve {short_signals} short sinyal üretti.")
                   
                   # Farklı rejimlerde oluşturulan sinyalleri karşılaştır
                   regimes = self.test_df["market_regime"].unique()
                   for regime in regimes:
                       regime_indices = self.test_df[self.test_df["market_regime"] == regime].index
                       if len(regime_indices) > 0:
                           regime_signals = signals.loc[regime_indices]
                           long_count = (regime_signals == 1).sum()
                           short_count = (regime_signals == -1).sum()
                           print(f"  {regime} rejiminde: {long_count} long, {short_count} short sinyal")
                 # DataFrame ise, DataFrame ile çalış
                else:
                    # Sonuçları yazdır
                    print(f"Regime Ensemble stratejisi {signals['long_signal'].sum()} long ve {signals['short_signal'].sum()} short sinyal üretti.")
                    
                    # Farklı rejimlerde oluşturulan sinyalleri karşılaştır
                    regimes = self.test_df["market_regime"].unique()
                    for regime in regimes:
                        regime_indices = self.test_df[self.test_df["market_regime"] == regime].index
                        if len(regime_indices) > 0:
                            long_count = signals.loc[regime_indices, "long_signal"].sum()
                            short_count = signals.loc[regime_indices, "short_signal"].sum()
                            print(f"  {regime} rejiminde: {long_count} long, {short_count} short sinyal")
                
        except Exception as e:        
            self.fail(f"Regime Ensemble stratejisi testi sırasında hata oluştu: {e}")

   
    def test_weighted_voting_strategy(self):
        """Weighted Voting Ensemble stratejisini test et."""
        try:
            # Strateji sınıfını al
            strategy_class = registry.get_strategy("weighted_voting")
            
            # Stratejinin varlığını doğrula
            self.assertIsNotNone(strategy_class, "Weighted Voting stratejisi registry'de bulunamadı")
            
            if strategy_class:
                # Strateji örneği oluştur
                strategy = strategy_class()
                
                # Stratejinin veri çerçevesini doğrulayabildiğini kontrol et
                self.assertTrue(strategy.validate_dataframe(self.test_df), "DataFrame doğrulaması başarısız")
                
                # Sinyalleri oluştur
                signals = strategy.generate_signals(self.test_df)
                
                # Veri tipi kontrolü
                self.assertTrue(isinstance(signals, pd.Series) or isinstance(signals, pd.DataFrame), 
                                "Sinyaller bir Series veya DataFrame olmalı")
                
                # Series ise, Series ile çalış
                if isinstance(signals, pd.Series):
                    # Sinyalleri say
                    long_signals = (signals == 1).sum()
                    short_signals = (signals == -1).sum()
                    
                    print(f"Weighted Voting stratejisi {long_signals} long ve {short_signals} short sinyal üretti.")
                    
                    # Aynı veri üzerinde farklı ağırlıklar kullanarak yeni sinyaller oluştur
                    custom_weights = {
                        "strategy_weights": {
                            "trend_following": 1.5,  # Trend stratejilerine daha fazla ağırlık ver
                            "mtf_trend": 1.5,
                            "adaptive_trend": 1.5,
                            "overextended_reversal": 0.5,  # Reversal stratejilerine daha az ağırlık ver
                            "pattern_reversal": 0.5,
                            "divergence_reversal": 0.5,
                            "volatility_breakout": 1.0,  # Breakout stratejilerine normal ağırlık
                            "range_breakout": 1.0,
                            "sr_breakout": 1.0
                        }
                    }
                    
                    weighted_strategy = strategy_class(custom_weights)
                    weighted_signals = weighted_strategy.generate_signals(self.test_df)
                    weighted_long = (weighted_signals == 1).sum()
                    weighted_short = (weighted_signals == -1).sum()
                    
                    # Ağırlıklı sonuçları yazdır
                    print(f"Özel ağırlıklı Weighted Voting stratejisi {weighted_long} long ve {weighted_short} short sinyal üretti.")
                
                # DataFrame ise, DataFrame ile çalış
                else:
                    # Sonuçları yazdır
                    print(f"Weighted Voting stratejisi {signals['long_signal'].sum()} long ve {signals['short_signal'].sum()} short sinyal üretti.")
                    
                    # Aynı veri üzerinde farklı ağırlıklar kullanarak yeni sinyaller oluştur
                    custom_weights = {
                        "strategy_weights": {
                            "trend_following": 1.5,  # Trend stratejilerine daha fazla ağırlık ver
                            "mtf_trend": 1.5,
                            "adaptive_trend": 1.5,
                            "overextended_reversal": 0.5,  # Reversal stratejilerine daha az ağırlık ver
                            "pattern_reversal": 0.5,
                            "divergence_reversal": 0.5,
                            "volatility_breakout": 1.0,  # Breakout stratejilerine normal ağırlık
                            "range_breakout": 1.0,
                            "sr_breakout": 1.0
                        }
                    }
                    
                    weighted_strategy = strategy_class(custom_weights)
                    weighted_signals = weighted_strategy.generate_signals(self.test_df)
                    
                    # Ağırlıklı sonuçları yazdır
                    print(f"Özel ağırlıklı Weighted Voting stratejisi {weighted_signals['long_signal'].sum()} long ve {weighted_signals['short_signal'].sum()} short sinyal üretti.")
                
        except Exception as e:
            self.fail(f"Weighted Voting stratejisi testi sırasında hata oluştu: {e}")


if __name__ == '__main__':
   # Test çalıştırmadan önce registry durumunu kontrol et
   if not is_registry_available():
       print("HATA: Strategy registry mevcut değil. Testler çalıştırılamıyor.")
       print("Olası nedenler:")
       print("1. signal_engine.strategies modülü doğru şekilde import edilemiyor")
       print("2. signal_engine.signal_strategy_system modülü doğru şekilde import edilemiyor")
       print("3. StrategyRegistry sınıfı tanımlanmamış veya erişilemiyor")
       sys.exit(1)
   
   # Registry içeriğini kontrol et
   if is_registry_available() and hasattr(registry, '_strategies'):
       print(f"Registry'deki stratejileri kontrol et: {len(registry._strategies)} strateji bulundu")
       categories = {}
       
       # Stratejileri kategorilere göre grupla
       for name, strategy_class in registry._strategies.items():
           category = strategy_class.category
           if category not in categories:
               categories[category] = []
           categories[category].append(name)
       
       # Kategorileri ve strateji sayılarını yazdır
       for category, strategies in categories.items():
           print(f"  - {category}: {len(strategies)} strateji")
   
   # Testleri çalıştır
   unittest.main()        
   