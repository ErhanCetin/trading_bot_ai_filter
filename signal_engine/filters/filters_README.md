## filters_README.md

```markdown
# Filtreler Modülü Dokümantasyonu

## Genel Bakış
Filtreler modülü, stratejilerden gelen ham sinyalleri piyasa koşulları, tarihsel performans, istatistiksel anormallikler ve makine öğrenimi öngörülerine göre filtreleyerek sinyal kalitesini artıran bileşenleri içerir.

## Dosya Yapısı
- `__init__.py`: Registry ve filtre importları
- `base_filter.py`: Temel filtre sınıfı (AdvancedFilterRule)
- `regime_filters.py`: Piyasa rejimine dayalı filtreler
- `statistical_filters.py`: İstatistiksel filtreler
- `ml_filters.py`: Makine öğrenimi tabanlı filtreler
- `adaptive_filters.py`: Adaptif filtreler
- `ensemble_filters.py`: Ensemble filtreler

## Kullanım
```python
from signal_engine.filters import registry

# Tüm filtreleri görüntüle
all_filters = registry.get_all_filters()

# Kategori bazlı filtreleri görüntüle
ml_filters = registry.get_filters_by_category("ml")

# Bir filtre oluştur
regime_filter = registry.create_filter("market_regime_filter", {"allowed_regimes": {"long": ["strong_uptrend"]}})

# Filtreyi uygula
filtered_signals_df = regime_filter.apply_to_dataframe(signals_df)


Yeni Filtre Ekleme

İlgili kategorideki dosyayı seçin (örneğin, rejim filtreleri için regime_filters.py)
AdvancedFilterRule sınıfından türeyen yeni bir sınıf oluşturun
Gerekli özellikleri doldurun: name, display_name, description, category, default_params, required_indicators
check_rule metodunu uygulayın
İsteğe bağlı olarak, prepare_filter ve post_filter metodlarını geçersiz kılın
__init__.py dosyasında filtreyi import edin ve registry'ye kaydedin

Mevcut Filtreler
Rejim Filtreleri

MarketRegimeFilter: Piyasa rejimine göre sinyalleri filtreler
VolatilityRegimeFilter: Volatilite rejimine göre sinyalleri filtreler
TrendStrengthFilter: Trend gücüne göre sinyalleri filtreler

İstatistiksel Filtreler

ZScoreExtremeFilter: Aşırı Z-skorlara göre sinyalleri filtreler
OutlierDetectionFilter: İstatistiksel aykırı değerlere göre sinyalleri filtreler
HistoricalVolatilityFilter: Tarihsel volatilite seviyelerine göre sinyalleri filtreler

ML Filtreleri

ProbabilisticSignalFilter: Olasılık tabanlı ML tahminlerine göre sinyalleri filtreler
PatternRecognitionFilter: Desen tanıma modellerine göre sinyalleri filtreler
PerformanceClassifierFilter: Tarihsel performans sınıflandırmasına göre sinyalleri filtreler

Adaptif Filtreler

DynamicThresholdFilter: Dinamik eşik değerleri kullanan filtre
ContextAwareFilter: Piyasa bağlamını dikkate alan filtre
MarketCycleFilter: Piyasa döngülerine dayalı filtre

Ensemble Filtreler

VotingEnsembleFilter: Birden çok filtreyi oylama mekanizmasıyla birleştiren filtre
SequentialFilterChain: Filtreleri sıralı zincir halinde uygulayan filtre
WeightedMetaFilter: Adaptif ağırlıklarla sinyal ve strateji kalitesini değerlendiren meta filtre