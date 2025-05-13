"""
ML modülü için test sınıfları.
"""
import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from signal_engine.ml import ModelTrainer, FeatureSelector, SignalPredictor, StrengthPredictor
from signal_engine.ml import prepare_data, evaluate_model, save_model

class TestMLUtils(unittest.TestCase):
    """ML yardımcı fonksiyonlarını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        np.random.seed(42)  # Tekrarlanabilirlik için seed belirle
        self.features = ['rsi_14', 'adx', 'macd_line', 'bollinger_width']
        self.target = 'signal'
        
        # Veri oluştur
        self.df = pd.DataFrame({
            'rsi_14': np.clip(np.random.normal(50, 15, 100), 0, 100),
            'adx': np.random.normal(25, 10, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'bollinger_width': np.random.normal(0.05, 0.02, 100),
            'signal': np.random.choice([0, 1, 2], 100, p=[0.6, 0.2, 0.2])  # 0=no signal, 1=long, 2=short
        })
    
    def test_prepare_data(self):
        """prepare_data fonksiyonunu test et."""
        # Veriyi hazırla
        X, y, X_train, X_test, y_train, y_test = prepare_data(
            self.df, self.features, self.target, test_size=0.2, random_state=42
        )
        
        # Sonuçları kontrol et
        self.assertEqual(X.shape[1], len(self.features), "X sütun sayısı feature sayısına eşit değil")
        self.assertEqual(len(y), len(self.df), "y uzunluğu df satır sayısına eşit değil")
        self.assertEqual(len(X_train) + len(X_test), len(X), "Train ve test veri sayısı toplamı doğru değil")
        self.assertEqual(len(y_train) + len(y_test), len(y), "Train ve test hedef sayısı toplamı doğru değil")
    
    def test_evaluate_model(self):
        """evaluate_model fonksiyonunu test et."""
        # Veriyi hazırla
        X, y, X_train, X_test, y_train, y_test = prepare_data(
            self.df, self.features, self.target, test_size=0.2, random_state=42
        )
        
        # Basit bir model eğit (dummy model)
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        
        # Modeli değerlendir
        metrics = evaluate_model(model, X_test, y_test, is_classifier=True)
        
        # Sonuçları kontrol et
        self.assertIn('accuracy', metrics, "Accuracy metriği eksik")
        self.assertIn('precision', metrics, "Precision metriği eksik")
        self.assertIn('recall', metrics, "Recall metriği eksik")
        self.assertIn('f1_score', metrics, "F1 score metriği eksik")
    
    def test_save_model(self):
        """save_model fonksiyonunu test et."""
        # Basit bir model oluştur
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="most_frequent")
        model.fit(self.df[self.features], self.df[self.target])
        
        # Geçici dizin oluştur
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Modeli kaydet
            model_path, _ = save_model(model, "test_model", model_dir=tmp_dir)
            
            # Kaydedilen dosyayı kontrol et
            self.assertTrue(os.path.exists(model_path), "Model dosyası oluşturulmadı")
            self.assertEqual(os.path.basename(model_path), "test_model.joblib", 
                            "Model dosya adı doğru değil")

class TestModelTrainer(unittest.TestCase):
    """ModelTrainer sınıfını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        np.random.seed(42)  # Tekrarlanabilirlik için seed belirle
        self.features = ['rsi_14', 'adx', 'macd_line', 'bollinger_width']
        
        # Sınıflandırma verisi
        self.classification_df = pd.DataFrame({
            'rsi_14': np.clip(np.random.normal(50, 15, 100), 0, 100),
            'adx': np.random.normal(25, 10, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'bollinger_width': np.random.normal(0.05, 0.02, 100),
            'signal': np.random.choice([0, 1, 2], 100, p=[0.6, 0.2, 0.2])  # 0=no signal, 1=long, 2=short
        })
        
        # Regresyon verisi
        self.regression_df = pd.DataFrame({
            'rsi_14': np.clip(np.random.normal(50, 15, 100), 0, 100),
            'adx': np.random.normal(25, 10, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'bollinger_width': np.random.normal(0.05, 0.02, 100),
            'signal_strength': np.random.uniform(0, 100, 100)  # 0-100 arası güç değeri
        })
        
        # ModelTrainer örneği oluştur
        self.trainer = ModelTrainer()
    
    def test_train_signal_classifier(self):
        """train_signal_classifier metodunu test et."""
        # Modeli eğit
        model, metrics = self.trainer.train_signal_classifier(
            self.classification_df, 
            self.features, 
            "signal", 
            model_name="random_forest",
            grid_search=False  # Test için hızlı olsun
        )
        
        # Sonuçları kontrol et
        self.assertIsNotNone(model, "Model eğitilemedi")
        self.assertIn('accuracy', metrics, "Accuracy metriği eksik")
        self.assertIn('precision', metrics, "Precision metriği eksik")
        self.assertIn('recall', metrics, "Recall metriği eksik")
        self.assertIn('f1_score', metrics, "F1 score metriği eksik")
    
    def test_train_strength_regressor(self):
        """train_strength_regressor metodunu test et."""
        # Modeli eğit
        model, metrics = self.trainer.train_strength_regressor(
            self.regression_df, 
            self.features, 
            "signal_strength", 
            model_name="random_forest",
            grid_search=False  # Test için hızlı olsun
        )
        
        # Sonuçları kontrol et
        self.assertIsNotNone(model, "Model eğitilemedi")
        self.assertIn('mean_squared_error', metrics, "MSE metriği eksik")
        self.assertIn('root_mean_squared_error', metrics, "RMSE metriği eksik")
        self.assertIn('r2_score', metrics, "R2 score metriği eksik")

class TestFeatureSelector(unittest.TestCase):
    """FeatureSelector sınıfını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        np.random.seed(42)  # Tekrarlanabilirlik için seed belirle
        
        # Özellikler ve hedef
        self.all_features = [
            'rsi_14', 'adx', 'macd_line', 'bollinger_width', 
            'ema_20', 'ema_50', 'atr', 'stoch_k', 'stoch_d'
        ]
        self.target = 'signal'
        
        # Veri oluştur
        self.df = pd.DataFrame()
        
        # RSI gibi özellikler
        self.df['rsi_14'] = np.clip(np.random.normal(50, 15, 100), 0, 100)
        
        # Trend göstergeleri
        self.df['adx'] = np.random.normal(25, 10, 100)
        self.df['macd_line'] = np.random.normal(0, 1, 100)
        
        # Volatilite göstergeleri
        self.df['bollinger_width'] = np.abs(np.random.normal(0.05, 0.02, 100))
        self.df['atr'] = np.abs(np.random.normal(2, 0.5, 100))
        
        # Hareketli ortalamalar
        self.df['ema_20'] = 100 + np.cumsum(np.random.normal(0, 1, 100))
        self.df['ema_50'] = 100 + np.cumsum(np.random.normal(0, 0.8, 100))
        
        # Stokastik
        self.df['stoch_k'] = np.clip(np.random.normal(50, 20, 100), 0, 100)
        self.df['stoch_d'] = np.clip(np.random.normal(50, 15, 100), 0, 100)
        
        # Hedef değişken
        self.df['signal'] = np.random.choice([0, 1, 2], 100, p=[0.6, 0.2, 0.2])
        
        # FeatureSelector örneği oluştur
        self.selector = FeatureSelector()
    
    def test_select_features(self):
        """select_features metodunu test et."""
        # Özellik seçimi yap
        selected_features = self.selector.select_features(
            self.df, 
            self.all_features, 
            self.target,
            methods=['variance_threshold', 'select_k_best', 'feature_importance']
        )
        
        # Sonuçları kontrol et
        self.assertIsInstance(selected_features, list, "Seçilen özellikler bir liste değil")
        self.assertGreater(len(selected_features), 0, "Hiç özellik seçilmemiş")
        self.assertLessEqual(len(selected_features), len(self.all_features), 
                           "Seçilen özellik sayısı toplam özellik sayısından fazla")
        
        # Seçilen tüm özellikler orijinal listedeki özellikler olmalı
        for feature in selected_features:
            self.assertIn(feature, self.all_features, f"{feature} orijinal özellik listesinde değil")

class TestPredictors(unittest.TestCase):
    """SignalPredictor ve StrengthPredictor sınıflarını test eden sınıf."""
    
    def setUp(self):
        """Test verilerini hazırla."""
        # Örnek veri çerçevesi oluştur
        np.random.seed(42)  # Tekrarlanabilirlik için seed belirle
        self.features = ['rsi_14', 'adx', 'macd_line', 'bollinger_width']
        
        # Eğitim verisi
        self.train_df = pd.DataFrame({
            'rsi_14': np.clip(np.random.normal(50, 15, 100), 0, 100),
            'adx': np.random.normal(25, 10, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'bollinger_width': np.random.normal(0.05, 0.02, 100),
            'signal': np.random.choice([0, 1, 2], 100, p=[0.6, 0.2, 0.2]),
            'signal_strength': np.random.uniform(0, 100, 100)
        })
        
        # Test verisi
        self.test_df = pd.DataFrame({
            'rsi_14': np.clip(np.random.normal(50, 15, 20), 0, 100),
            'adx': np.random.normal(25, 10, 20),
            'macd_line': np.random.normal(0, 1, 20),
            'bollinger_width': np.random.normal(0.05, 0.02, 20)
        })
        
        # Test sinyalleri
        self.test_signals = pd.Series(np.random.choice([0, 1, -1], 20, p=[0.6, 0.2, 0.2]))
        
        # Modelleri eğit ve kaydet
        self.model_dir = tempfile.mkdtemp()
        
        # Sınıflandırıcı model
        trainer = ModelTrainer()
        classifier, _ = trainer.train_signal_classifier(
            self.train_df, self.features, "signal", model_name="random_forest", grid_search=False
        )
        self.classifier_path, _ = save_model(classifier, "signal_classifier", model_dir=self.model_dir)
        
        # Regresyon model
        regressor, _ = trainer.train_strength_regressor(
            self.train_df, self.features, "signal_strength", model_name="random_forest", grid_search=False
        )
        self.regressor_path, _ = save_model(regressor, "strength_regressor", model_dir=self.model_dir)
    
    def tearDown(self):
        """Geçici dosyaları temizle."""
        import shutil
        try:
            shutil.rmtree(self.model_dir)
        except:
            pass
    
    def test_signal_predictor(self):
        """SignalPredictor sınıfını test et."""
        # Predictor oluştur
        predictor = SignalPredictor(
            model_path=self.classifier_path,
            config={
                'features': self.features,
                'probability_threshold': 0.5
            }
        )
        
        # Sinyalleri tahmin et
        predicted_signals = predictor.predict(self.test_df)
        
        # Sonuçları kontrol et
        self.assertIsInstance(predicted_signals, pd.Series, "Tahmin edilen sinyaller bir pandas Series değil")
        self.assertEqual(len(predicted_signals), len(self.test_df), 
                        "Tahmin edilen sinyal sayısı test veri sayısına eşit değil")
        
        # Tahmin edilen değerler 0, 1, -1 olmalı
        for val in predicted_signals.unique():
            self.assertIn(val, [0, 1, -1], f"Geçersiz sinyal değeri: {val}")
    
    def test_strength_predictor(self):
        """StrengthPredictor sınıfını test et."""
        # Predictor oluştur
        predictor = StrengthPredictor(
            model_path=self.regressor_path,
            config={
                'features': self.features,
                'min_strength': 0,
                'max_strength': 100
            }
        )
        
        # Güç değerlerini tahmin et
        predicted_strength = predictor.predict(self.test_df, self.test_signals)
        
        # Sonuçları kontrol et
        self.assertIsInstance(predicted_strength, pd.Series, "Tahmin edilen güç değerleri bir pandas Series değil")
        self.assertEqual(len(predicted_strength), len(self.test_df), 
                        "Tahmin edilen güç değeri sayısı test veri sayısına eşit değil")
        
        # Güç değerleri 0-100 arasında olmalı
        self.assertTrue(all(0 <= val <= 100 for val in predicted_strength if not pd.isna(val)), 
                       "Güç değerleri 0-100 arasında değil")
        
        # Sinyal olmayan noktalarda güç değeri 0 olmalı
        for i, signal in enumerate(self.test_signals):
            if signal == 0:
                self.assertEqual(predicted_strength.iloc[i], 0, 
                                "Sinyal olmayan noktada güç değeri 0 değil")


if __name__ == '__main__':
    unittest.main()