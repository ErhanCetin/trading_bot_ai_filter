# Gelişmiş Ticaret Sinyali Üretme Motoru

Bu proje, Python tabanlı, modüler ve esnek bir ticaret sinyali üretme motorudur. Finansal piyasalarda alım/satım fırsatlarını belirlemek için çeşitli teknik analiz stratejilerini kullanır. Sistem, trend takip, piyasa dönüşleri, kırılmalar ve bu stratejilerin akıllıca birleştirildiği topluluk (ensemble) yöntemlerini destekler.

## Özellikler

* **Modüler Tasarım**: Her strateji bağımsız bir modül olarak geliştirilmiştir, bu da sistemi bakımı kolay ve genişletilebilir kılar.
* **Geniş Strateji Kütüphanesi**: Trend, tersine dönüş, kırılma ve topluluk (ensemble) gibi çeşitli kategorilerde önceden tanımlanmış birçok strateji sunar.
* **Esnek Parametrelendirme**: Tüm stratejiler, varsayılan değerlere sahip ve kullanıcı tarafından kolayca özelleştirilebilen parametrelerle gelir.
* **Gösterge Bağımlılığı**: Her strateji, çalışması için gerekli (`required_indicators`) ve isteğe bağlı (`optional_indicators`) teknik göstergeleri açıkça belirtir.
* **Akıllı Sinyal Onaylama**: Bir sinyalin üretilmesi için tüm koşulların katı bir şekilde karşılanması yerine, esnek bir onaylama mantığı (minimum teyit sayısı ve güven eşiği) kullanılır.
* **Gelişmiş Topluluk (Ensemble) Stratejileri**:
    * `WeightedVotingEnsembleStrategy`: Farklı stratejilerden gelen sinyalleri, önceden tanımlanmış ağırlıklara göre birleştirir.
    * `RegimeBasedEnsembleStrategy`: Mevcut piyasa rejimine (örneğin, trendli, yatay) göre en uygun stratejileri dinamik olarak seçer veya ağırlıklarını ayarlar.
    * `AdaptiveEnsembleStrategy`: Alt stratejilerin ağırlıklarını, gözlemlenen en son performanslarına göre otomatik olarak uyarlar.
* **Merkezi Strateji Yönetimi**: `StrategyRegistry` ile stratejiler merkezi olarak kaydedilir ve yönetilir. `StrategyManager` ise bu stratejileri kullanarak sinyal üretimini koordine eder.
* **Detaylı Günlükleme**: Sistem genelinde önemli olaylar ve hatalar için günlükleme (logging) mekanizmaları mevcuttur.

## Sistem Mimarisi

Sistem temel olarak şu bileşenlerden oluşur:

1.  **`BaseStrategy` (`signal_strategy_system.py`)**: Tüm stratejilerin miras aldığı temel soyut sınıftır. Stratejilerin ortak arayüzünü ve temel işlevselliğini tanımlar.
2.  **Strateji Sınıfları** (örneğin, `TrendFollowingStrategy` `trend_strategy.py` içinde): Belirli bir ticaret mantığını uygulayan somut strateji sınıflarıdır. `generate_conditions` metodu ile alım/satım koşullarını tanımlarlar.
3.  **`StrategyRegistry` (`signal_strategy_system.py`)**: Kullanılabilir tüm strateji sınıflarını kaydeder ve bunlara erişim sağlar.
4.  **`StrategyManager` (`signal_strategy_system.py`)**: `StrategyRegistry`'deki stratejileri kullanarak, sağlanan piyasa verileri (DataFrame) üzerinde sinyal üretme sürecini yönetir. Birden fazla stratejinin sinyallerini birleştirebilir.
5.  **Strateji Paketleri (`strategies/`)**: Stratejiler, mantıksal olarak kategorilere ayrılmış (`trend`, `reversal`, `breakout`, `ensemble`) ayrı dosyalarda bulunur. `strategies/__init__.py` dosyası, bu stratejileri otomatik olarak `StrategyRegistry`'ye kaydeder.

## Kurulum

```bash
# Projeyi klonlayın (eğer bir Git deposu varsa)
# git clone <repository_url>
# cd <project_directory>

# Gerekli bağımlılıkları yükleyin
pip install pandas numpy



Kullanım Örneği
Aşağıda, sistemi kullanarak nasıl sinyal üretebileceğinize dair temel bir örnek verilmiştir:

import pandas as pd
from signal_engine.signal_strategy_system import StrategyManager # signal_engine.strategies.__init__ içindeki registry'yi kullanır
from signal_engine.strategies import registry # Kayıtlı stratejilere erişim

# 1. StrategyManager'ı Başlatma
# Registry'yi StrategyManager'a iletmek en iyi pratiktir
strategy_mgr = StrategyManager(registry=registry)

# 2. Kullanılacak Stratejileri Ekleme
# Örnek: Trend Takip Stratejisi ve Adaptif Trend Stratejisi
strategy_mgr.add_strategy(
    strategy_name="trend_following",
    params={"adx_threshold": 20, "confirmation_count": 2}, # Örnek parametreler
    weight=0.6 # Topluluk içindeki ağırlığı
)
strategy_mgr.add_strategy(
    strategy_name="adaptive_trend",
    params={"adx_max_threshold": 35},
    weight=0.4
)

# 3. Piyasa Verilerini Hazırlama (Örnek DataFrame)
# DataFrame'iniz, stratejilerin gerektirdiği tüm gösterge sütunlarını içermelidir.
# Örneğin: 'close', 'adx', 'di_pos', 'di_neg', 'rsi_14', 'macd_line', 'ema_20', 'ema_50', 'atr_percent' vb.
data = {
    'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
    'close': [100, 101, 102, 101, 103],
    'adx': [22, 25, 28, 23, 26],
    'di_pos': [20, 22, 25, 18, 24],
    'di_neg': [15, 13, 10, 16, 12],
    'rsi_14': [55, 60, 65, 50, 62],
    'macd_line': [0.1, 0.3, 0.5, 0.2, 0.4],
    'ema_20': [99, 100, 101, 100.5, 101.5],
    'ema_50': [98, 98.5, 99, 99.5, 100],
    'atr_percent': [1.0, 1.1, 1.2, 1.0, 1.3]
    # ... diğer gerekli göstergeler ...
}
market_df = pd.DataFrame(data)
market_df.set_index('timestamp', inplace=True)

# Stratejilerin gerektirdiği tüm sütunların DataFrame'de olduğundan emin olun.
# Örneğin, TrendFollowingStrategy için:
# required = ["close", "adx", "di_pos", "di_neg", "rsi_14", "macd_line", "ema_20", "ema_50"]
# Eksik sütunlar varsa, bunları hesaplamanız veya sağlamanız gerekir.

# 4. Sinyalleri Üretme
# `generate_signals` metodu, eklenen tüm stratejileri kullanarak sinyal üretir.
# Eğer strategy_names ve params argümanları verilmezse, add_strategy ile eklenenleri kullanır.
signal_df = strategy_mgr.generate_signals(market_df)

# 5. Sonuçları İnceleme
print(signal_df[['close', 'long_signal', 'short_signal', 'strategy_name', 'signal_strength']])




Not: Yukarıdaki market_df örneği basitleştirilmiştir. Gerçek kullanımda, stratejilerin required_indicators listesinde belirtilen tüm göstergelerin DataFrame'de mevcut ve doğru şekilde hesaplanmış olması gerekir.

Strateji Kütüphanesi
Aşağıda sistemde mevcut olan ana strateji kategorileri ve bazı örnekler listelenmiştir:

Trend Takip Stratejileri (trend_strategy.py)
TrendFollowingStrategy: ADX, RSI, MACD ve EMA'lar gibi birden fazla göstergeyi kullanarak trendleri belirler ve takip eder.
MultiTimeframeTrendStrategy: Farklı zaman dilimlerinde hizalanmış trendleri arar.
AdaptiveTrendStrategy: ADX ve RSI eşikleri gibi parametrelerini mevcut piyasa volatilitesine göre dinamik olarak ayarlar.
Tersine Dönüş Stratejileri (reversal_strategy.py)
OverextendedReversalStrategy: RSI, Bollinger Bantları ve ardışık mumlar gibi göstergelerle piyasanın aşırı alım/satım durumlarında potansiyel tersine dönüşleri hedefler.
PatternReversalStrategy: Yutan mum, çekiç, kayan yıldız ve doji gibi mum formasyonlarına dayalı tersine dönüş sinyalleri üretir.
DivergenceReversalStrategy: Fiyat hareketleri ile RSI, MACD gibi osilatörler arasındaki uyumsuzlukları (divergence) tespit ederek tersine dönüş arar.
Kırılma (Breakout) Stratejileri (breakout_strategy.py)
VolatilityBreakoutStrategy: ATR (Average True Range) veya Bollinger Bantları ile tanımlanan volatilite kanallarından dışarı doğru olan fiyat kırılmalarını belirler.
RangeBreakoutStrategy: Belirli bir süre boyunca oluşan dar fiyat aralıklarından (consolidation) yukarı veya aşağı yönlü kırılmaları hedefler.
SupportResistanceBreakoutStrategy: Otomatik veya manuel olarak belirlenen önemli destek ve direnç seviyelerinden gerçekleşen kırılmaları yakalamaya çalışır.
Topluluk (Ensemble) Stratejileri (ensemble_strategy.py)
WeightedVotingEnsembleStrategy: Birden fazla alt stratejiden gelen sinyalleri, her stratejiye atanmış özel ağırlıklara göre birleştirerek nihai bir karar üretir.
RegimeBasedEnsembleStrategy: Mevcut piyasa rejimini (örneğin, güçlü trend, zayıf trend, yatay, volatil) tespit eder ve bu rejime en uygun alt stratejilere daha fazla ağırlık verir veya sadece onları aktive eder.
AdaptiveEnsembleStrategy: Alt stratejilerin ağırlıklarını, belirli bir geçmişe bakarak en son gözlemlenen kârlılık veya sinyal doğruluğu gibi performans metriklerine göre dinamik olarak günceller.
Geliştirmeye Katkıda Bulunma
Katkılarınızı bekliyoruz! Lütfen aşağıdaki adımları izleyin:

Bu depoyu fork'layın.
Yeni bir özellik dalı oluşturun (git checkout -b yeni-ozellik).
Değişikliklerinizi commit'leyin (git commit -am 'Yeni bir özellik eklendi').
Dalınızı push'layın (git push origin yeni-ozellik).
Bir Pull Request oluşturun.
Lütfen kodlama standartlarına uyun ve değişikliklerinizi test edin.

Gelecek Planları ve Potansiyel İyileştirmeler
Performans Optimizasyonu: Büyük veri setleri ve çok sayıda strateji için sinyal üretme hızının artırılması.
Gelişmiş Parametre Optimizasyonu Araçları: Strateji parametrelerinin en iyi değerlerini bulmak için genetik algoritmalar veya grid search gibi yöntemlerin entegrasyonu.
Kapsamlı Geriye Dönük Test (Backtesting) Modülü: Stratejilerin geçmiş performanslarını detaylı bir şekilde analiz etmek için entegre bir backtesting aracı.
Risk Yönetimi Katmanı: Pozisyon boyutlandırma, stop-loss/take-profit mekanizmaları gibi risk yönetimi kurallarının uygulanabilmesi.
Canlı Ticaret Entegrasyonları: Popüler broker API'leri ile entegrasyon için altyapı.
Daha Fazla Strateji ve Gösterge: Topluluk tarafından talep edilen yeni stratejilerin ve göstergelerin eklenmesi.
Kullanıcı Arayüzü (UI): Strateji yapılandırması, backtesting ve sonuçların görselleştirilmesi için bir web arayüzü veya masaüstü uygulaması.