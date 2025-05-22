test_strategies.py Kullanım Kılavuzu
Bu belge, test_strategies.py dosyasını nasıl kullanacağınızı açıklar.
Kurulum ve Gereksinimler
test_strategies.py dosyasını kullanabilmek için aşağıdaki dosyaların mevcut olması gerekir:

debug_indicator_list.py - İndikatörleri görselleştirmek için yardımcı modül
signal_engine paketi ve ilgili alt modülleri

Kullanım
Komut Satırından Çalıştırma
bashpython test_strategies.py
Bu komut, interaktif bir menü başlatır ve stratejileri test etmenizi sağlar.
Python Kodunuzda İçe Aktarma
pythonfrom test_strategies import test_strategy, test_ensemble_strategy, display_available_strategies

# Kullanılabilir stratejileri görüntüle
display_available_strategies()

# Tek bir strateji test et
test_strategy("trend_following")

# Ensemble strateji test et
test_ensemble_strategy("regime_ensemble")
Örnek Kullanım Senaryoları
1. Yeni Strateji Geliştirme ve Test Etme
Yeni bir strateji geliştirirken, stratejinizin doğru çalıştığını test etmek için:
python# Stratejinizi oluşturun ve registry'ye kaydedin
# Sonra test edin
test_strategy("your_new_strategy")
2. Veri Oluşturma ve Hazırlama
Test için veri oluşturmak istiyorsanız:
pythonfrom test_strategies import generate_sample_data, prepare_data_for_strategy

# Örnek veri oluştur
df = generate_sample_data(size=500, add_patterns=True)

# Strateji için veriyi hazırla
indicator_df = prepare_data_for_strategy(df, "trend_following")
3. Market Regime Sorunlarını Düzeltme
Market regime değerlerini kontrol etmek ve düzeltmek için:
pythonfrom test_strategies import check_market_regime_values

df = prepare_data_for_strategy(df, "regime_ensemble")
df = check_market_regime_values(df, fix_missing=True)
4. Strateji Sinyallerini Görselleştirme
Strateji sinyallerini görselleştirmek için:
pythonfrom test_strategies import visualize_strategy_signals, test_strategy

# Strateji çalıştır ve sonuçları al
result_df = test_strategy("supertrend")

# Sinyalleri görselleştir
visualize_strategy_signals(result_df, "supertrend")
Sorun Giderme
ModuleNotFoundError
Eğer "ModuleNotFoundError: No module named 'signal_engine'" hatası alıyorsanız:

Modül yolunun doğru ayarlandığından emin olun
signal_engine paketinin doğru konumda olduğunu kontrol edin

pythonimport sys
sys.path.append('/path/to/your/project')  # signal_engine modülünün bulunduğu dizin
IndicatorNotFoundError
Eğer indikatörler bulunamıyorsa:

İndikatörlerin doğru şekilde kaydedildiğinden emin olun
indicator_registry'nin doğru şekilde oluşturulduğunu kontrol edin

Diğer Hatalar
Diğer hatalar için lütfen hata mesajını dikkatle okuyun ve ilgili dosyaları kontrol edin. Genellikle hata mesajı, sorunun nereden kaynaklandığını belirtir.


TestTrendStrategies: Trend takip stratejilerini test eder. Farklı piyasa koşullarında trend stratejilerinin nasıl davrandığını ve strateji parametrelerinin doğru şekilde ayarlanıp ayarlanmadığını kontrol eder.
TestReversalStrategies: Geri dönüş stratejilerini test eder. Aşırı alım/satım koşullarında, mum formasyonlarında ve uyumsuzluk durumlarında stratejilerin beklendiği gibi sinyal üretip üretmediğini kontrol eder.
TestBreakoutStrategies: Kırılma stratejilerini test eder. Volatilite kırılmaları, fiyat aralığı kırılmaları ve destek/direnç kırılmalarında stratejilerin beklendiği gibi sinyal üretip üretmediğini kontrol eder.
TestEnsembleStrategies: Ensemble stratejilerini test eder. Farklı piyasa rejimlerinde, ağırlıklı oylama senaryolarında ve adaptif stratejilerde ensemble stratejilerinin beklendiği gibi çalışıp çalışmadığını kontrol eder. Bu sınıf, diğer stratejileri mock yöntemiyle taklit eder.

Her test, strateji sınıflarının inşasını, parametre yönetimini ve sinyal üretimini kontrol eder. Ayrıca, farklı piyasa koşullarında stratejilerin doğru sinyal ürettiğinden emin olur.
Bu testler, stratejilerin beklendiği gibi çalıştığından emin olmak için bir temel sağlar ve gelecekteki değişikliklerin mevcut davranışı bozmadığından emin olmak için kullanılabilir.RetryClaude can make mistakes. Please double-check responses.