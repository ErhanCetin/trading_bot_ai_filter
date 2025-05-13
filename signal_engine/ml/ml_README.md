## ml_README.md

```markdown
# Makine Öğrenmesi (ML) Modülü Dokümantasyonu

## Genel Bakış
ML modülü, sinyal üretim sistemine makine öğrenmesi yetenekleri ekler. Bu modül, model eğitimi, özellik seçimi, tahmin ve performans değerlendirme araçları sağlar.

## Dosya Yapısı
- `__init__.py`: ML modülünün ana giriş noktası
- `model_trainer.py`: Model eğitim altyapısı
- `feature_selector.py`: Özellik seçme araçları
- `predictors.py`: Sinyal ve güç tahmin modelleri
- `utils.py`: Yardımcı fonksiyonlar

## Kullanım
```python
from signal_engine.ml import ModelTrainer, FeatureSelector, SignalPredictor
from signal_engine.ml import prepare_data, evaluate_model, save_model

# Özellik seçimi
feature_selector = FeatureSelector()
selected_features = feature_selector.select_features(indicator_data, features, "signal")

# Model eğitimi
trainer = ModelTrainer()
model, metrics = trainer.train_signal_classifier(indicator_data, selected_features, "signal", "random_forest")

# Model kaydı
model_path = save_model(model, "signal_classifier")

# Tahmin modeli
predictor = SignalPredictor(model_path)
predicted_signals = predictor.predict(new_data)



Temel Bileşenler
ModelTrainer
ML modellerinin eğitimi için kullanılan sınıf. Sınıflandırma ve regresyon modelleri eğitebilir.
python# Sinyal sınıflandırıcı eğitimi
model, metrics = trainer.train_signal_classifier(
    df=indicator_data,
    features=selected_features,
    target_column="signal",
    model_name="random_forest",
    grid_search=True
)

# Güç tahmini regresyonu eğitimi
model, metrics = trainer.train_strength_regressor(
    df=indicator_data,
    features=selected_features,
    target_column="signal_strength",
    model_name="xgboost_regressor"
)
FeatureSelector
Özellik seçimi için kullanılan sınıf. Çeşitli yöntemlerle en önemli özellikleri belirlemeye yardımcı olur.
python# Özellik seçimi
selector = FeatureSelector()
selected_features = selector.select_features(
    df=indicator_data,
    features=all_features,
    target_column="signal",
    methods=["variance_threshold", "select_k_best", "feature_importance"]
)

# Korelasyon matrisi görselleştirme
selector.plot_correlation_matrix(indicator_data, selected_features)
SignalPredictor
Eğitilmiş modelleri kullanarak sinyal tahminleri yapan sınıf.
python# Sinyal tahmini
predictor = SignalPredictor(
    model_path="models/signal_classifier.joblib",
    config={
        "features": selected_features,
        "probability_threshold": 0.7
    }
)
predicted_signals = predictor.predict(new_data)
StrengthPredictor
Eğitilmiş modelleri kullanarak sinyal gücü tahminleri yapan sınıf.
python# Güç tahmini
strength_predictor = StrengthPredictor(
    model_path="models/strength_regressor.joblib",
    config={
        "features": selected_features,
        "min_strength": 0,
        "max_strength": 100
    }
)
predicted_strength = strength_predictor.predict(new_data, signals)
Desteklenen Model Tipleri
Sınıflandırıcılar

RandomForestClassifier
GradientBoostingClassifier
XGBClassifier
LogisticRegression

Regresyoncular

RandomForestRegressor
XGBRegressor

Tipik ML İş Akışı

Özellik (feature) hazırlama
Eğitim/test verisi ayırma
Özellik seçimi
Model eğitimi
Model değerlendirme
Model kaydetme
Tahmin modeli oluşturma
Yeni verilerle tahmin yapma