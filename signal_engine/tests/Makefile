#Opsiyonel
.PHONY: test test-unit test-integration test-coverage clean

# Tüm testleri çalıştır
test:
	python run_tests.py

# Ünite testlerini çalıştır
test-unit:
	pytest tests -m "unit" -v

# Entegrasyon testlerini çalıştır
test-integration:
	pytest tests -m "integration" -v

# Kod kapsama analizi ile testleri çalıştır
test-coverage:
	pytest tests --cov=signal_engine --cov-report=html --cov-report=term

# Tüm geçici dosyaları temizle
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete