Python Dosyalarının Analizi ve Yorumu
Bu belge, sağlanan Python dosyalarının her birinin amacını ve temel işlevlerini açıklamaktadır. Dosyalar, bir ticaret sinyal motoru için çeşitli teknik göstergeler oluşturmaya, yönetmeye ve bir eklenti sistemi üzerinden çalıştırmaya yönelik bir kütüphanenin parçasıdır.

1. signal_indicator_plugin_system.py (Gösterge Eklenti Sistemi Temeli)

Amaç: Bu dosya, tüm gösterge (indikatör) sisteminin temel altyapısını tanımlar. Göstergelerin standart bir şekilde oluşturulmasını, kaydedilmesini ve yönetilmesini sağlayan çekirdek sınıfları içerir. Bir eklenti (plugin) mimarisinin temelini oluşturur.

Anahtar İşlevler ve Sınıflar:

BaseIndicator (ABC): Tüm gösterge sınıfları için soyut bir temel sınıftır (Abstract Base Class). Her göstergenin uyması gereken standart bir arayüzü (örneğin, name, description, default_params, calculate metodu) tanımlar. Bu, yeni göstergelerin sisteme kolayca entegre edilebilmesini sağlar.

IndicatorRegistry: Gösterge sınıflarını kaydetmek ve yönetmek için merkezi bir kayıt defteri (registry) görevi görür. Göstergeleri isimleriyle kaydeder ve istendiğinde bu isimleri kullanarak gösterge nesneleri oluşturur (create_indicator). Ayrıca göstergeleri kategoriye göre listeleme gibi yardımcı işlevler sunar.

IndicatorManager: IndicatorRegistry'yi kullanarak bir veya daha fazla göstergeyi bir veri çerçevesi (DataFrame) üzerinde hesaplamak için bir yönetici sınıfıdır. Belirli göstergelerin parametreleriyle birlikte hesaplanmasını koordine eder.

Yorum: Bu dosya, kütüphanenin en kritik parçalarından biridir. Tanımladığı sınıflar sayesinde göstergeler modüler, genişletilebilir ve tutarlı bir yapıda geliştirilebilir. Eklenti sistemi, yeni göstergelerin mevcut sistemi değiştirmeden kolayca eklenmesine olanak tanır.

2. __init__.py (Paket Başlangıç Noktası)

Amaç: Bu dosya, indicators adlı Python paketinin başlangıç noktasıdır. signal_indicator_plugin_system.py dosyasından IndicatorRegistry sınıfını kullanarak bir kayıt defteri nesnesi oluşturur. Ardından, diğer modüllerde (base_indicators.py, advanced_indicators.py vb.) tanımlanan tüm gösterge sınıflarını içe aktarır ve bu merkezi IndicatorRegistry'ye kaydeder.

Anahtar İşlevler:

signal_indicator_plugin_system modülünden IndicatorRegistry'yi içe aktarır.

Bir IndicatorRegistry nesnesi (registry) oluşturur.

base_indicators, advanced_indicators, feature_indicators, regime_indicators ve statistical_indicators modüllerinden tüm gösterge sınıflarını içe aktarır.

common_calculations modülünden ADXCalculator gibi yardımcı sınıfları içe aktarır.

Tüm bu gösterge sınıflarını oluşturulan registry nesnesine kaydeder.

registry nesnesini dışa aktararak paketin dışından erişilebilir hale getirir.

Yorum: Paket yapısının ve gösterge yönetiminin merkezidir. signal_indicator_plugin_system.py'de tanımlanan kayıt mekanizmasını kullanarak tüm göstergeleri kullanıma hazır hale getirir. Yeni bir gösterge eklendiğinde, bu dosyada ilgili içe aktarma ve kayıt işleminin yapılması gerekir.

3. base_indicators.py (Temel Göstergeler)

Amaç: Temel ve yaygın olarak kullanılan teknik göstergeleri içerir.

Anahtar İşlevler (Örnekler):

EMAIndicator: Üstel Hareketli Ortalama (EMA) hesaplar.

SMAIndicator: Basit Hareketli Ortalama (SMA) hesaplar.

RSIIndicator: Göreceli Güç Endeksi (RSI) hesaplar.

MACDIndicator: Hareketli Ortalama Yakınsama/Iraksama (MACD) hesaplar.

BollingerBandsIndicator: Bollinger Bantlarını hesaplar.

ATRIndicator: Ortalama Gerçek Aralık (ATR) hesaplar.

StochasticIndicator: Stokastik Osilatör hesaplar.

Yorum: Teknik analizin temel yapı taşlarını oluşturan göstergeler burada tanımlanmıştır. Her bir gösterge, signal_indicator_plugin_system.py dosyasındaki BaseIndicator sınıfından miras alır ve calculate metodunu kendi hesaplama mantığıyla uygular.

4. advanced_indicators.py (Gelişmiş Göstergeler)

Amaç: Temel göstergelerin üzerine inşa edilen veya daha karmaşık hesaplamalar içeren gelişmiş teknik göstergeleri barındırır.

Anahtar İşlevler (Örnekler):

AdaptiveRSIIndicator: Piyasa volatilitesine göre periyodunu ayarlayan Adaptif RSI hesaplar.

MultitimeframeEMAIndicator: Farklı zaman dilimlerindeki EMA değerlerini birleştirerek analiz yapar.

HeikinAshiIndicator: Heikin Ashi mum grafiklerini hesaplar.

SupertrendIndicator: Trend takibi için Supertrend göstergesini hesaplar.

IchimokuIndicator: Ichimoku Bulutu göstergesinin bileşenlerini (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span) hesaplar.

Yorum: Daha sofistike ticaret stratejileri ve analizler için kullanılabilecek göstergelerdir. BaseIndicator yapısına uygun olarak geliştirilmişlerdir.

5. feature_indicators.py (Özellik Mühendisliği Göstergeleri)

Amaç: Fiyat ve hacim verilerinden özellik mühendisliği (feature engineering) yaparak yeni anlamlı göstergeler veya özellikler türetir.

Anahtar İşlevler (Örnekler):

PriceActionIndicator: Mum çubuğu formasyonları (örneğin, engulfing, doji, hammer) ve fiyat hareketi özelliklerini (örneğin, gövde boyutu, gölge uzunlukları) analiz eder.

VolumePriceIndicator: Hacim ve fiyat arasındaki ilişkiyi analiz eden özellikler üretir.

MomentumFeatureIndicator: Momentum tabanlı çeşitli özellikleri hesaplar.

SupportResistanceIndicator: Destek ve direnç seviyelerini belirlemeye yönelik özellikler üretir (pivot noktaları, dokunma sayıları vb.).

Yorum: Makine öğrenimi modelleri için girdi olarak kullanılabilecek veya kural tabanlı sistemlerde karar verme süreçlerini iyileştirebilecek karmaşık özellikler oluşturur.

6. regime_indicators.py (Rejim Göstergeleri)

Amaç: Piyasanın mevcut durumunu (rejimini) belirlemeye yardımcı olan göstergeler içerir. Örneğin, piyasanın trendli mi, yatay mı, yoksa volatil mi olduğunu anlamaya çalışır.

Anahtar İşlevler (Örnekler):

MarketRegimeIndicator: Çeşitli göstergeleri (ADX, Bollinger Bantları, RSI vb.) kullanarak piyasa rejimini (güçlü yükseliş trendi, zayıf düşüş trendi, yatay, volatil vb.) tanımlar.

VolatilityRegimeIndicator: Piyasadaki volatilite seviyesini belirler.

TrendStrengthIndicator: Mevcut trendin gücünü ve sağlığını analiz eder. ADX, EMA kesişimleri gibi metrikleri kullanabilir.

Yorum: Farklı piyasa koşullarına göre stratejileri adapte etmek için kritik öneme sahip göstergelerdir.

7. statistical_indicators.py (İstatistiksel Göstergeler)

Amaç: Fiyat hareketlerini analiz etmek için istatistiksel yöntemler kullanan göstergeler içerir.

Anahtar İşlevler (Örnekler):

ZScoreIndicator: Belirli bir metriğin (kapanış fiyatı, hacim vb.) ortalamadan kaç standart sapma uzakta olduğunu gösteren Z-Skorunu hesaplar.

KeltnerChannelIndicator: Keltner Kanallarını hesaplar.

StandardDeviationIndicator: Fiyatların standart sapmasını hesaplar.

LinearRegressionIndicator: Belirli bir periyottaki fiyatlara lineer regresyon çizgisi uygular ve eğim, kesişim, sapma gibi değerleri hesaplar.

Yorum: Aşırılıkları, ortalamaya dönüş potansiyelini veya trendin istatistiksel gücünü belirlemek için kullanılır.

8. common_calculations.py (Ortak Hesaplamalar)

Amaç: Birden fazla gösterge tarafından kullanılabilecek ortak hesaplama mantıklarını içerir. Bu, kod tekrarını önler ve tutarlılığı artırır.

Anahtar İşlevler:

ADXCalculator: Ortalama Yönsel Endeks (ADX) ve ilgili değerleri (+DI, -DI) hesaplamak için bir yardımcı sınıf içerir. Bu sınıf da BaseIndicator'dan miras alır ve registry'ye kaydedilir, böylece diğer göstergeler tarafından bir bağımlılık olarak kullanılabilir veya doğrudan çağrılabilir.

Yorum: Kodun modülerliğini ve bakımını kolaylaştıran önemli bir dosyadır. ADX gibi sık kullanılan hesaplamalar burada merkezileştirilmiştir.

9. indicators_test.py (Gösterge Testleri)

Amaç: indicators modülündeki göstergelerin doğru çalışıp çalışmadığını test etmek için bir test betiği (script) içerir.

Anahtar İşlevler:

Örnek fiyat verisi oluşturur (generate_sample_data).

Belirli bir göstergeyi veya bir kategoriye ait tüm göstergeleri test eder.

Test sonuçlarını (örneğin, hesaplanan gösterge değerlerini içeren DataFrame) görselleştirmek için grafikler çizer (plot_indicator_results).

Kullanıcıya interaktif bir menü sunarak hangi testlerin çalıştırılacağını seçme imkanı tanır.

IndicatorManager ve registry kullanarak göstergeleri dinamik olarak yükler ve çalıştırır.

Yorum: Göstergelerin geliştirilmesi ve bakımı sırasında hataları erken tespit etmek ve doğruluğu sağlamak için hayati bir araçtır. Yeni göstergeler eklendikçe veya mevcutlar güncellendikçe bu testlerin de güncellenmesi önemlidir.

10. indicators_README.md (Dokümantasyon)

Amaç: indicators modülünün dokümantasyonunu içerir. Modülün genel yapısını, kullanımını ve mevcut göstergelerin bir listesini sunar.

Anahtar İçerik:

Modüle genel bakış.

Dosya yapısının açıklaması.

Göstergelerin nasıl kullanılacağına dair örnek kodlar.

Yeni bir göstergenin nasıl ekleneceğine dair talimatlar.

Kategorilere ayrılmış mevcut göstergelerin listesi (Temel, Gelişmiş, Özellik Mühendisliği, Rejim, İstatistiksel).

Yorum: Modülü anlamak ve kullanmak isteyen geliştiriciler için önemli bir kaynaktır. İyi bir dokümantasyon, modülün benimsenmesini ve doğru kullanılmasını kolaylaştırır.

Genel Yorum:

Sağlanan dosyalar, son derece modüler, genişletilebilir ve iyi yapılandırılmış bir teknik gösterge kütüphanesi ve eklenti sistemi oluşturma çabasını göstermektedir. signal_indicator_plugin_system.py dosyasında tanımlanan BaseIndicator, IndicatorRegistry ve IndicatorManager sınıfları, bu sistemin temelini oluşturur. Bu temel, tüm göstergelerin standart bir arayüze sahip olmasını, merkezi bir yerden yönetilmesini ve kolayca sisteme entegre edilmesini sağlar.

Her gösterge türü için ayrı dosyaların (base_indicators.py, advanced_indicators.py vb.) olması, kodun organize olmasına ve bakımının kolaylaşmasına yardımcı olur. common_calculations.py ortak mantığı merkezileştirirken, indicators_test.py ve indicators_README.md dosyaları projenin kalitesini, güvenilirliğini ve kullanılabilirliğini artıran kritik unsurlardır.

Bu yapı, karmaşık ticaret stratejileri geliştirmek, finansal veriler üzerinde derinlemesine analizler yapmak ve algoritmik ticaret sistemleri için sağlam, esnek ve yönetilebilir bir gösterge altyapısı sunar.


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