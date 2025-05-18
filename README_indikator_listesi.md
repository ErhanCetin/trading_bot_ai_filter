



Kapsamlı İndikatör Listesi (Kategoriler, Açıklamalar ve Kullanım Nedenleri)
1. Trend İndikatörleri
Bu indikatörler piyasanın genel yönünü ve trendin gücünü ölçmek için kullanılır.

EMA (Exponential Moving Average)

Açıklama: Fiyat hareketlerini düzleştiren, ancak son fiyatlara daha fazla ağırlık veren ortalama.
Parametreler:

EMA_FAST: 20 periyotluk hızlı EMA
EMA_SLOW: 50 periyotluk yavaş EMA
EMA: [9, 21, 50, 200] periyotluk çoklu EMA'lar


Kullanım Nedeni: Kısa ve uzun vadeli trend yönünü belirlemek, destek/direnç seviyeleri oluşturmak ve çaprazlamaları (golden cross/death cross) tespit etmek için kullanılır. Farklı periyotlarda EMA kullanmak, farklı zaman dilimlerindeki trend değişikliklerini yakalamaya yardımcı olur.


SMA (Simple Moving Average)

Açıklama: Belirli bir dönem için basit aritmetik ortalama.
Parametreler:

SMA: [10, 20, 50, 200] periyotluk SMA'lar


Kullanım Nedeni: EMA'ya göre daha az reaktiftir ve "gürültüyü" daha iyi filtreler. Uzun vadeli trend analizinde, özellikle 200 günlük hareketli ortalama, önemli destek/direnç seviyeleri oluşturur. Kurumsal alıcıların izlediği seviyeleri belirlemede kullanılır.


MACD (Moving Average Convergence Divergence)

Açıklama: İki farklı hareketli ortalamanın birbirine yakınsama ve ıraksamasını ölçen momentum indikatörü.
Parametreler:

fast_period: 12 (Hızlı ortalama periyodu)
slow_period: 26 (Yavaş ortalama periyodu)
signal_period: 9 (Sinyal çizgisi periyodu)


Kullanım Nedeni: Momentum değişikliklerini, trend dönüşlerini ve alım-satım sinyallerini belirlemede kullanılır. Histogram sıfır çizgisini geçtiğinde veya MACD sinyal çizgisini kestiğinde sinyal üretir. Ayrıca uyumsuzluklar (divergence) piyasa dönüşlerinin habercisi olabilir.


ADX (Average Directional Index)

Açıklama: Trendin gücünü ölçen gösterge, yönü belirtmez.
Parametreler:

ADX: 25 (Eşik değeri)


Kullanım Nedeni: Piyasanın trend mi yoksa yatay mı hareket ettiğini belirlemek için kullanılır. ADX > 25 ise güçlü trend, < 20 ise zayıf trend veya yatay piyasa olduğunu gösterir. DI+ ve DI- ile birlikte trend yönü de belirlenebilir.


Supertrend

Açıklama: Trend yönünü ve giriş/çıkış noktalarını belirlemede kullanılan, ATR tabanlı bir indikatör.
Parametreler:

period: 10 (ATR periyodu)
multiplier: 3 (ATR çarpanı)


Kullanım Nedeni: Trend yönünü tek bir çizgiyle gösterir, stop-loss seviyeleri oluşturur ve trend değişikliklerinde otomatik olarak yön değiştirir. Supertrend çizginin renk değişimi, alım-satım sinyali olarak kullanılabilir.



2. Momentum İndikatörleri
Bu indikatörler fiyat hareketinin hızını ve gücünü ölçer.

RSI (Relative Strength Index)

Açıklama: Fiyat değişimlerinin gücünü ve hızını ölçen, 0-100 arasında değerler alan osilatör.
Parametreler:

RSI: [7, 14, 21] (Farklı periyotlar için)


Kullanım Nedeni: Aşırı alım (>70) ve aşırı satım (<30) durumlarını tespit etmek, fiyat-RSI uyumsuzluklarını (divergence) belirlemek ve momentum değişikliklerini yakalamak için kullanılır. Farklı periyotlar, farklı zaman çerçevelerindeki momentum değişimlerini gösterir.


CCI (Commodity Channel Index)

Açıklama: Fiyatın, hareketli ortalama etrafındaki hareketini ölçen osilatör.
Parametreler:

CCI: 20 (Periyot)


Kullanım Nedeni: Aşırı alım/satım durumlarını tespit etmek (±100 seviyeleri) ve trend dönüş noktalarını belirlemek için kullanılır. RSI'dan farklı olarak, CCI sınırsız değerler alabilir ve büyük fiyat hareketlerinde daha duyarlıdır.


Stochastic

Açıklama: Fiyatın, belirli bir dönemdeki en yüksek-en düşük aralığındaki konumunu gösteren osilatör.
Parametreler:

window: 14 (Ana periyot)
smooth_window: 3 (K çizgisi yumuşatma)
d_window: 3 (D çizgisi periyodu)


Kullanım Nedeni: Aşırı alım/satım durumlarını tespit etmek (80/20 seviyeleri), K ve D çizgilerinin kesişimlerinden alım-satım sinyalleri üretmek ve momentum değişikliklerini erken yakalamak için kullanılır.


Momentum Features

Açıklama: Momentum bazlı özellikler ve türevler üreten kapsamlı indikatör.
Parametreler:

lookback_periods: [3, 5, 10, 20, 50] (Geri bakış periyotları)


Kullanım Nedeni: Fiyat momentumunun hızını, ivmesini ve değişim oranını ölçmek, trend gücünü değerlendirmek ve çoklu zaman dilimlerinde momentum değişikliklerini izlemek için kullanılır.



3. Volatilite İndikatörleri
Bu indikatörler piyasa volatilitesini ve fiyat dalgalanmalarını ölçer.

ATR (Average True Range)

Açıklama: Piyasa volatilitesini ölçen indikatör.
Parametreler:

ATR: 14 (Periyot)


Kullanım Nedeni: Stop-loss seviyeleri belirlemek, pozisyon büyüklüğünü volatiliteye göre ayarlamak ve volatilite kırılmalarını tespit etmek için kullanılır. Yüksek ATR değerleri yüksek volatiliteyi, düşük değerler düşük volatiliteyi gösterir.


Bollinger Bands

Açıklama: Fiyatın hareketli ortalama etrafındaki standart sapmasını kullanarak volatilite bantları oluşturur.
Parametreler:

window: 20 (Ortalama periyodu)
window_dev: 2 (Standart sapma çarpanı)


Kullanım Nedeni: Fiyatın ortalamadan sapmasını ölçmek, volatilite daralma/genişleme dönemlerini tespit etmek ve potansiyel fiyat hedeflerini belirlemek için kullanılır. Fiyat alt banda dokunduğunda alım, üst banda dokunduğunda satım sinyali olabilir.


Keltner Channel

Açıklama: ATR tabanlı volatilite bantları, Bollinger Bands'e benzer ancak standart sapma yerine ATR kullanır.
Parametreler:

ema_window: 20 (EMA periyodu)
atr_window: 10 (ATR periyodu)
atr_multiplier: 2 (ATR çarpanı)


Kullanım Nedeni: Bollinger Bands'e göre daha az yanlış sinyal üretir. Trend yönünü belirlemek, olası destek/direnç seviyelerini tespit etmek ve fiyatın kanal dışına çıkmasıyla breakout/breakdown sinyalleri üretmek için kullanılır.



4. Volatilite ve Rejim İndikatörleri
Bu indikatörler piyasa koşullarını ve rejimlerini sınıflandırmak için kullanılır.

Volatility Regime

Açıklama: Piyasanın volatilite rejimini (yüksek, normal, düşük) belirleyen gösterge.
Parametreler:

VOLATILITY_REGIME: true


Kullanım Nedeni: Farklı piyasa koşullarına göre strateji parametrelerini dinamik olarak ayarlamak, yüksek volatilite dönemlerinde risk yönetimini sıkılaştırmak ve düşük volatilite dönemlerinde daha agresif pozisyonlar almak için kullanılır.


Market Regime

Açıklama: Piyasa rejimini (güçlü yükseliş, zayıf yükseliş, yatay, zayıf düşüş, güçlü düşüş) belirleyen gösterge.
Parametreler:

MARKET_REGIME: true


Kullanım Nedeni: Trend takip, geri dönüş veya breakout stratejileri arasında geçiş yapmak, piyasa koşullarına göre alım-satım kararlarını optimize etmek ve farklı rejimlerde farklı stratejiler uygulamak için kullanılır.


Trend Strength

Açıklama: Mevcut trendin gücünü ve sağlığını ölçen gösterge.
Parametreler:

TREND_STRENGTH: true


Kullanım Nedeni: Trend takip stratejilerinin ne zaman uygulanması gerektiğini belirlemek, trend gücüne göre pozisyon büyüklüğünü ayarlamak ve zayıf trendlerden kaçınmak için kullanılır.



5. İstatistiksel ve İleri Düzey İndikatörler
Bu indikatörler daha karmaşık analiz yöntemleri kullanarak piyasa koşullarını değerlendirir.

ZScore

Açıklama: Fiyat veya başka bir metriğin ortalamadan standart sapma cinsinden uzaklığını ölçer.
Parametreler:

window: 100 (Geri bakış periyodu)


Kullanım Nedeni: İstatistiksel anormallik tespiti, aşırılıkların belirlenmesi ve ortalamaya dönüş stratejileri için kullanılır. Z-Skoru ±2'yi aştığında potansiyel aşırı alım/satım durumları oluşabilir.


Standard Deviation

Açıklama: Fiyat değişkenliğini standart sapma ile ölçen gösterge.
Parametreler:

windows: [5, 20, 50] (Farklı periyotlar)


Kullanım Nedeni: Volatilite analizinde, risk yönetiminde ve volatilite rejimlerinin belirlenmesinde kullanılır. Farklı zaman dilimlerindeki volatilite karşılaştırması yapılabilir.


Linear Regression

Açıklama: Fiyatların doğrusal eğilimini ve trendin eğimini hesaplayan gösterge.
Parametreler:

windows: [20, 50, 100] (Regresyon periyotları)


Kullanım Nedeni: Fiyat trendlerinin yönünü ve gücünü ölçmek, fiyat hedeflerini belirlemek ve fiyatın regresyon kanalından sapmalarını tespit etmek için kullanılır.



6. Özel ve Gelişmiş İndikatörler
Bu indikatörler standart göstergelerin ötesinde, daha özelleştirilmiş analiz metotları sunar.

MTF_EMA (Multi-Timeframe EMA)

Açıklama: Farklı zaman dilimlerindeki EMA değerlerini ve bunların ilişkilerini analiz eden gösterge.
Parametreler:

period: 20 (EMA periyodu)
timeframes: [1, 4, 12, 24] (Çarpan faktörleri)


Kullanım Nedeni: Farklı zaman dilimlerindeki trend uyumunu değerlendirmek, çoklu zaman dilimi alım-satım stratejileri oluşturmak ve daha güvenilir trendin yönünü belirlemek için kullanılır.


Adaptive RSI

Açıklama: Piyasa volatilitesine göre periyodunu otomatik ayarlayan RSI versiyonu.
Parametreler:

base_period: 14 (Temel periyot)
min_period: 5 (Minimum periyot)
max_period: 30 (Maksimum periyot)


Kullanım Nedeni: Farklı volatilite koşullarında daha doğru RSI değerleri elde etmek, yüksek volatilitede daha kısa periyotlar kullanarak hızlı tepki vermek ve düşük volatilitede daha uzun periyotlar kullanarak gürültüyü azaltmak için kullanılır.


Heikin Ashi

Açıklama: Trend analizi için standart mum grafiklerinin filtrelenmiş versiyonu.
Parametreler:

HEIKIN_ASHI: true


Kullanım Nedeni: Piyasa gürültüsünü azaltmak, trend değişikliklerini daha net görmek ve trend devam ettiği sürece aynı renkte mumlar üreterek trend analizi yapmayı kolaylaştırmak için kullanılır.


Ichimoku

Açıklama: Japon tarzı, çoklu gösterge sistemini tek bir grafikte birleştiren kapsamlı bir teknik analiz aracı.
Parametreler:

tenkan_period: 9 (Tenkan-sen periyodu)
kijun_period: 26 (Kijun-sen periyodu)
senkou_b_period: 52 (Senkou Span B periyodu)
displacement: 26 (İleri kaydırma periyodu)


Kullanım Nedeni: Trend yönünü, momentumunu, destek/direnç seviyelerini ve sinyal üretimini tek bir göstergede birleştirmek için kullanılır. Kumo (bulut) fiyat için dinamik destek/direnç bölgeleri oluşturur.


Support Resistance

Açıklama: Önemli destek ve direnç seviyelerini otomatik olarak tespit eden gösterge.
Parametreler:

SUPPORT_RESISTANCE: true


Kullanım Nedeni: Alım-satım kararları için kritik fiyat seviyelerini belirlemek, fiyatın bu seviyelere yaklaşmasına veya bu seviyeleri kırmasına göre stratejiler geliştirmek ve stop-loss/take-profit hedefleri oluşturmak için kullanılır.


Price Action

Açıklama: Mum formasyonlarını ve fiyat hareketleri desenlerini analiz eden gösterge.
Parametreler:

PRICE_ACTION: true


Kullanım Nedeni: Engulfing, hammer, shooting star gibi önemli mum formasyonlarını tespit etmek, bu desenlerden alım-satım sinyalleri üretmek ve fiyat hareketi karakteristiklerini incelemek için kullanılır.


Volume Price

Açıklama: Fiyat-hacim ilişkisini analiz eden gösterge.
Parametreler:

VOLUME_PRICE: true


Kullanım Nedeni: Fiyat hareketlerinin hacim tarafından doğrulanıp doğrulanmadığını kontrol etmek, hacim ve fiyat uyumsuzluklarını tespit etmek ve hacim bazlı alım-satım stratejileri geliştirmek için kullanılır.


OBV (On-Balance Volume)

Açıklama: Fiyat değişimlerine göre kümülatif hacim akışını ölçen gösterge.
Parametreler:

OBV: true


Kullanım Nedeni: Fiyat hareketlerinin arkasındaki hacim baskısını ölçmek, fiyat-OBV uyumsuzluklarını belirlemek ve olası trend dönüşlerini önceden tespit etmek için kullanılır.



Her bir indikatör, trading stratejinizin farklı bir yönünü güçlendirmek, risk yönetimini iyileştirmek veya belirli piyasa koşullarında daha kesin sinyaller üretmek için kullanılabilir. Bu kapsamlı liste, çeşitli piyasa koşullarına adapte olabilen sağlam bir trading sistemi geliştirmenize yardımcı olacaktır.