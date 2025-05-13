# İndikatörler Modülü Dokümantasyonu

## Genel Bakış
İndikatörler modülü, teknik analiz göstergelerini hesaplayan ve veri çerçevesine ekleyen bileşenleri içerir. Bu modül, temel, gelişmiş, özellik mühendisliği, rejim ve istatistiksel indikatörleri kapsar.

## Dosya Yapısı
- `__init__.py`: Registry ve indikatör importları
- `base_indicators.py`: Temel indikatörler (RSI, EMA, SMA, MACD, Bollinger Bands, ATR, Stochastic)
- `advanced_indicators.py`: Gelişmiş indikatörler (Adaptive RSI, MultitimeframeEMA, HeikinAshi, Supertrend, Ichimoku)
- `feature_indicators.py`: Özellik mühendisliği indikatörleri (PriceAction, VolumePrice, MomentumFeature, SupportResistance)
- `regime_indicators.py`: Piyasa rejimi indikatörleri (MarketRegime, VolatilityRegime, TrendStrength)
- `statistical_indicators.py`: İstatistiksel anormallik indikatörleri (ZScore, KeltnerChannel, StandardDeviation, LinearRegression)

## Kullanım
```python
from signal_engine.indicators import registry

# Tüm indikatörleri görüntüle
all_indicators = registry.get_all_indicators()

# Kategori bazlı indikatörleri görüntüle
trend_indicators = registry.get_indicators_by_category("trend")

# Bir indikatör oluştur
ema_indicator = registry.create_indicator("ema", {"periods": [9, 21, 50]})

# İndikatörü hesapla
result_df = ema_indicator.calculate(price_data)





Yeni İndikatör Ekleme

İlgili kategorideki dosyayı seçin (örneğin, trend indikatörleri için base_indicators.py)
BaseIndicator sınıfından türeyen yeni bir sınıf oluşturun
Gerekli özellikleri doldurun: name, display_name, description, category, default_params, requires_columns, output_columns
calculate metodunu uygulayın
__init__.py dosyasında indikatörü import edin ve registry'ye kaydedin

Mevcut İndikatörler
Temel İndikatörler

EMAIndicator: Üstel Hareketli Ortalama
SMAIndicator: Basit Hareketli Ortalama
RSIIndicator: Göreceli Güç Endeksi
MACDIndicator: Hareketli Ortalama Yakınsama/Iraksama
BollingerBandsIndicator: Bollinger Bantları
ATRIndicator: Ortalama Gerçek Aralık
StochasticIndicator: Stokastik Osilatör

Gelişmiş İndikatörler

AdaptiveRSIIndicator: Adaptif RSI (volatiliteye göre ayarlanan)
MultitimeframeEMAIndicator: Çoklu zaman dilimli EMA
HeikinAshiIndicator: Heikin Ashi mumları
SupertrendIndicator: Supertrend göstergesi
IchimokuIndicator: Ichimoku Bulutu



Özellik Mühendisliği İndikatörleri

PriceActionIndicator: Fiyat hareketi analizi
VolumePriceIndicator: Hacim-fiyat ilişkisi analizi
MomentumFeatureIndicator: Momentum bazlı özellikler
SupportResistanceIndicator: Destek ve direnç seviyeleri

Rejim İndikatörleri

MarketRegimeIndicator: Piyasa rejimi analizi
VolatilityRegimeIndicator: Volatilite rejimi analizi
TrendStrengthIndicator: Trend gücü analizi

İstatistiksel İndikatörler

ZScoreIndicator: Z-Skoru hesaplama
KeltnerChannelIndicator: Keltner Kanalı
StandardDeviationIndicator: Standart sapma bazlı göstergeler
LinearRegressionIndicator: Doğrusal regresyon analizi