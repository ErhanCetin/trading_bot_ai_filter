Modüler Yapıda Tamamen Yeni Kaliteli Sinyal Üretim Sistemi
1. Ana Plugin Mimarisi Yapısı
İndikatör Modülleri (signal_engine/indicators/)

__init__.py - Registry ve indikatör importları
base_indicators.py - Temel indikatörler (RSI, EMA, vb.)
advanced_indicators.py - Gelişmiş indikatörler (Adaptive RSI, Multi-timeframe EMA)
feature_indicators.py - Özellik mühendisliği indikatörleri
regime_indicators.py - Piyasa rejimi indikatörleri
statistical_indicators.py - İstatistiksel anormallik indikatörleri

Strateji Modülleri (signal_engine/strategies/)

__init__.py - Registry ve strateji importları
base_strategy.py - Temel strateji sınıfı
trend_strategy.py - Trend stratejileri
reversal_strategy.py - Geri dönüş stratejileri
breakout_strategy.py - Kırılma stratejileri
ensemble_strategy.py - Ensemble stratejisi

Filtre Modülleri (signal_engine/filters/)

__init__.py - Registry ve filtre importları
base_filter.py - Temel filtre sınıfı
regime_filters.py - Rejim bazlı filtreler
statistical_filters.py - İstatistiksel filtreler
ml_filters.py - Makine öğrenimi tabanlı filtreler

Sinyal Gücü Modülleri (signal_engine/strength/)

__init__.py - Registry ve güç hesaplayıcı importları
base_strength.py - Temel güç hesaplayıcı
predictive_strength.py - Tahmin bazlı güç hesaplayıcı
context_strength.py - Bağlam duyarlı güç hesaplayıcı

ML Altyapısı (signal_engine/ml/)

__init__.py
model_trainer.py - Model eğitim altyapısı
feature_selector.py - Özellik seçme araçları
predictors.py - Tahmin modelleri
utils.py - Yardımcı fonksiyonlar

2. Temel Kod Yapısı
İndikatör Örneği
pythonfrom signal_engine.base import BaseIndicator

class AdaptiveRSIIndicator(BaseIndicator):
    """Volatiliteye adapte olan RSI indikatörü"""
    
    name = "adaptive_rsi"
    display_name = "Adaptive RSI"
    description = "RSI that adapts to market volatility"
    category = "momentum"
    
    default_params = {
        "base_period": 14,
        "volatility_window": 100
    }
    
    requires_columns = ["close", "high", "low"]
    output_columns = ["adaptive_rsi", "adaptive_rsi_period"]
    
    def calculate(self, df):
        # Temel kodlar...
        return df
Strateji Örneği
pythonfrom signal_engine.base import BaseStrategy

class TrendCaptureStrategy(BaseStrategy):
    """Güçlü trendleri yakalayan strateji"""
    
    name = "trend_capture"
    display_name = "Trend Capture Strategy"
    description = "Identifies and captures strong market trends"
    category = "trend"
    
    default_params = {
        "min_adx": 25,
        "confirmation_period": 3
    }
    
    required_indicators = ["adx", "ema_fast", "ema_slow", "adaptive_rsi"]
    
    def generate_conditions(self, df, row, i):
        # Koşul mantığı...
        return {"long": long_conditions, "short": short_conditions}
Filtre Örneği
pythonfrom signal_engine.base import BaseFilter

class MarketRegimeFilter(BaseFilter):
    """Piyasa rejimine göre filtreleyen kural"""
    
    name = "market_regime_filter"
    display_name = "Market Regime Filter"
    description = "Filters signals based on market regime"
    category = "context"
    
    default_params = {
        "trend_threshold": 25
    }
    
    required_indicators = ["market_regime", "adx", "atr_ratio"]
    
    def apply_filter(self, df):
        # Filtreleme mantığı...
        return filtered_df
3. Signal Engine Ana Sınıfı
pythonclass SignalEngine:
    """Ana sinyal motoru sınıfı"""
    
    def __init__(self, indicator_registry, strategy_registry, filter_registry, strength_registry):
        self.indicator_registry = indicator_registry
        self.strategy_registry = strategy_registry
        self.filter_registry = filter_registry
        self.strength_registry = strength_registry
        
        # Varsayılan kurulumları yap
        self.setup_defaults()
        
    def setup_defaults(self):
        # Varsayılan indikatörler, stratejiler, filtreler
        self.setup_indicators()
        self.setup_strategies()
        self.setup_filters()
        self.setup_strength_calculators()
        
    def process_data(self, df):
        """Veriyi işle ve sinyaller üret"""
        # İndikatörleri hesapla
        df = self.calculate_indicators(df)
        
        # Sinyalleri üret
        df = self.generate_signals(df)
        
        # Sinyal gücünü hesapla
        df = self.calculate_signal_strength(df)
        
        # Filtreleri uygula
        df = self.apply_filters(df)
        
        return df
4. Uygulama Adımları

Önce temel yapıyı oluştur (klasörler, boş dosyalar)
Temel sınıfları implement et (BaseIndicator, BaseStrategy, vb.)
Gelişmiş indikatörleri geliştir
Stratejileri geliştir
Filtreleme mekanizmalarını ekle
ML altyapısını kur
Tüm sistemi entegre et
Kapsamlı testler yap

5. İlk Öncelikli Modüller

MarketRegimeIndicator - Piyasa koşullarını tanımlar, tüm stratejilerin temelidir
FeatureEngineeringIndicator - Zengin özellikleri hesaplar
EnsembleStrategy - Tüm stratejileri birleştiren merkezi strateji
AdaptiveFilters - Piyasa koşullarına göre sinyal filtreleme

6. Test ve Doğrulama

Her modül için birim testler
Tüm sistem için entegrasyon testleri
Farklı piyasa koşullarında backtest
Performans metriklerinin izlenmesi (başarı oranı, kazanç/kayıp, vb.)

Modüler yapıyı koruyarak tamamen yeni bir sistem geliştirmek için bu plan kullanılabilir. Her modül bağımsız olarak geliştirilebilir ve test edilebilir, aynı zamanda tüm sistem bir araya geldiğinde kaliteli sinyaller üretecektir.
İlk adım olarak temel yapıyı oluşturup temel modülleri implement etmeye başlayabiliriz. Mevcut kodları ihtiyaç duyduğumuz kadar koruyarak, gerekirse tamamen yeni bir yaklaşım getirebiliriz.RetryClaude can make mistakes. Please double-check responses.