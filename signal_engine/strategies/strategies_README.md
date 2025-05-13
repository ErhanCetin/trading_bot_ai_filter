## strategies_README.md

```markdown
# Stratejiler Modülü Dokümantasyonu

## Genel Bakış
Stratejiler modülü, farklı piyasa koşullarında alım-satım sinyalleri üreten bileşenleri içerir. Bu modül, trend takip, geri dönüş, kırılma ve ensemble stratejilerini kapsar.

## Dosya Yapısı
- `__init__.py`: Registry ve strateji importları
- `base_strategy.py`: Temel strateji sınıfı
- `trend_strategy.py`: Trend takip stratejileri
- `reversal_strategy.py`: Geri dönüş stratejileri
- `breakout_strategy.py`: Kırılma stratejileri
- `ensemble_strategy.py`: Ensemble stratejileri

## Kullanım
```python
from signal_engine.strategies import registry

# Tüm stratejileri görüntüle
all_strategies = registry.get_all_strategies()

# Kategori bazlı stratejileri görüntüle
trend_strategies = registry.get_strategies_by_category("trend")

# Bir strateji oluştur
trend_strategy = registry.create_strategy("trend_following", {"adx_threshold": 25})

# Stratejiyi uygula ve sinyalleri oluştur
signals = trend_strategy.generate_signals(indicator_data)



Yeni Strateji Ekleme

İlgili kategorideki dosyayı seçin (örneğin, trend stratejileri için trend_strategy.py)
BaseStrategy sınıfından türeyen yeni bir sınıf oluşturun
Gerekli özellikleri doldurun: name, display_name, description, category, default_params, required_indicators, optional_indicators
generate_conditions metodunu uygulayın
__init__.py dosyasında stratejiyi import edin ve registry'ye kaydedin

Mevcut Stratejiler
Trend Stratejileri

TrendFollowingStrategy: Temel trend takip stratejisi
MultiTimeframeTrendStrategy: Çoklu zaman dilimli trend stratejisi
AdaptiveTrendStrategy: Adaptif trend stratejisi

Geri Dönüş Stratejileri

OverextendedReversalStrategy: Aşırı uzamış piyasalarda geri dönüş stratejisi
PatternReversalStrategy: Mum formasyonlarına dayalı geri dönüş stratejisi
DivergenceReversalStrategy: Uyumsuzluklara dayalı geri dönüş stratejisi

Kırılma Stratejileri

VolatilityBreakoutStrategy: Volatilite bazlı kırılma stratejisi
RangeBreakoutStrategy: Fiyat aralığı kırılma stratejisi
SupportResistanceBreakoutStrategy: Destek ve direnç kırılma stratejisi

Ensemble Stratejileri

RegimeBasedEnsembleStrategy: Piyasa rejimine dayalı ensemble stratejisi
WeightedVotingEnsembleStrategy: Ağırlıklı oylama ensemble stratejisi
AdaptiveEnsembleStrategy: Adaptif ensemble stratejisi (geçmiş performansa göre)