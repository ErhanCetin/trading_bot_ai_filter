## strength_README.md

```markdown
# Sinyal Gücü Hesaplama Modülü Dokümantasyonu

## Genel Bakış
Sinyal Gücü Hesaplama modülü, stratejilerden ve filtrelerden gelen sinyallere güç değeri atayan, sinyallerin güvenilirliği ve potansiyel karlılığını ölçen bileşenleri içerir.

## Dosya Yapısı
- `__init__.py`: Registry ve güç hesaplayıcı importları
- `base_strength.py`: Temel güç hesaplayıcı sınıfı
- `predictive_strength.py`: Tahmin bazlı güç hesaplayıcılar
- `context_strength.py`: Bağlam duyarlı güç hesaplayıcılar

## Kullanım
```python
from signal_engine.strength import registry

# Tüm güç hesaplayıcıları görüntüle
all_calculators = registry.get_all_calculators()

# Kategori bazlı güç hesaplayıcıları görüntüle
predictive_calculators = registry.get_calculators_by_category("predictive")

# Bir güç hesaplayıcı oluştur
strength_calculator = registry.create_calculator("market_context_strength", {"regime_weights": {"trending": 0.8}})

# Güç değerlerini hesapla
strength_values = strength_calculator.calculate(indicator_data, signals)


Yeni Güç Hesaplayıcı Ekleme

İlgili kategorideki dosyayı seçin (örneğin, tahmin bazlı hesaplayıcılar için predictive_strength.py)
BaseStrengthCalculator sınıfından türeyen yeni bir sınıf oluşturun
Gerekli özellikleri doldurun: name, display_name, description, category, default_params, required_indicators
calculate metodunu uygulayın
__init__.py dosyasında hesaplayıcıyı import edin ve registry'ye kaydedin

Mevcut Güç Hesaplayıcılar
Tahmin Bazlı Hesaplayıcılar

ProbabilisticStrengthCalculator: Tarihsel başarı olasılığına dayalı güç hesaplama
RiskRewardStrengthCalculator: Risk/ödül oranına dayalı güç hesaplama
MLPredictiveStrengthCalculator: ML tahmin modellerine dayalı güç hesaplama

Bağlam Duyarlı Hesaplayıcılar

MarketContextStrengthCalculator: Piyasa bağlamı ve rejimine dayalı güç hesaplama
IndicatorConfirmationStrengthCalculator: İndikatör onaylarına dayalı güç hesaplama
MultiTimeframeStrengthCalculator: Çoklu zaman dilimi uyumuna dayalı güç hesaplama

Güç Değerlerinin Kullanımı
Sinyal gücü değerleri, 0-100 arasında olup, aşağıdaki şekilde yorumlanabilir:

80-100: Çok güçlü sinyal - Yüksek güven ve başarı olasılığı
60-80: Güçlü sinyal - İyi güven ve pozitif beklenen değer
40-60: Orta güçte sinyal - Makul güven ama ek teyit gerekebilir
20-40: Zayıf sinyal - Düşük güven, dikkatli yaklaşılmalı
0-20: Çok zayıf sinyal - Düşük başarı olasılığı, muhtemelen kaçınılmalı

Güç değerleri şu amaçlarla kullanılabilir:

Pozisyon boyutu belirleme
Sinyal filtreleme (belirli bir eşik değerin altındaki sinyalleri eleme)
Çoklu sinyal arasında önceliklendirme yapma
Farklı stratejilerin performans değerlendirmesi

