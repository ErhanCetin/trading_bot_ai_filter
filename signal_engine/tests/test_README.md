Kullanım
Bu test yapısını kullanmak için aşağıdaki adımları izleyebilirsiniz:

tests dizini oluşturun ve yukarıdaki dosyaları uygun konumlara yerleştirin.
Test bağımlılıklarını kurun:

bashpip install pytest pytest-cov

Testleri çalıştırın:

bash# Tüm testleri çalıştırmak için:
python run_tests.py

# Pytest kullanarak çalıştırmak için:
pytest

# Belirli bir test modülünü çalıştırmak için:
pytest tests/test_indicators.py

# Kod kapsama analizi ile çalıştırmak için:
pytest --cov=signal_engine