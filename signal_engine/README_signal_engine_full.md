# Signal Engine Trading Bot

## 🎯 Genel Bakış

Signal Engine, modüler ve esnek bir trading bot sistemidir. Teknik indikatörleri hesaplama, trading stratejileri geliştirme, sinyal filtreleme ve güç değerlendirmesi yaparak profesyonel seviyede trading sinyalleri üretir.

### 🚀 Temel Özellikler

- **Modüler Mimari**: Her bileşen bağımsız olarak çalışır ve kolayca genişletilebilir
- **Plugin Sistemi**: Yeni indikatörler, stratejiler ve filtreler kolayca eklenebilir
- **ML Entegrasyonu**: Makine öğrenmesi modelleriyle sinyal kalitesi artırılabilir
- **Kapsamlı Filtreleme**: Çoklu filtreleme sistemiyle yanlış sinyaller elimine edilir
- **Risk Yönetimi**: Dinamik pozisyon boyutlandırma ve risk kontrolü
- **Performans İzleme**: Detaylı raporlama ve analiz araçları

## 📁 Proje Yapısı

```
signal_engine/
├── __init__.py                    # Ana modül giriş noktası
├── signal_manager.py              # Ana yönetici sınıfı
├── signal_indicator_plugin_system.py  # İndikatör sistemi
├── signal_strategy_system.py      # Strateji sistemi
├── signal_filter_system.py       # Filtreleme sistemi
├── signal_strength_system.py     # Güç hesaplama sistemi
├── signal_ml_system.py           # ML entegrasyon sistemi
│
├── indicators/                    # İndikatör modülleri
│   ├── __init__.py
│   ├── base_indicators.py        # Temel indikatörler (RSI, EMA, MACD vs.)
│   ├── advanced_indicators.py   # Gelişmiş indikatörler
│   ├── feature_indicators.py    # Özellik mühendisliği
│   ├── regime_indicators.py     # Piyasa rejimi indikatörleri
│   ├── statistical_indicators.py # İstatistiksel indikatörler
│   └── common_calculations.py   # Ortak hesaplamalar
│
├── strategies/                   # Trading stratejileri
│   ├── __init__.py
│   ├── trend_strategy.py        # Trend takip stratejileri
│   ├── reversal_strategy.py     # Geri dönüş stratejileri
│   ├── breakout_strategy.py     # Kırılma stratejileri
│   └── ensemble_strategy.py     # Ensemble stratejiler
│
├── filters/                     # Sinyal filtreleri
│   ├── __init__.py
│   ├── regime_filters.py        # Piyasa rejimi filtreleri
│   ├── statistical_filters.py   # İstatistiksel filtreler
│   ├── ml_filters.py           # ML tabanlı filtreler
│   ├── adaptive_filters.py     # Adaptif filtreler
│   └── ensemble_filters.py     # Ensemble filtreler
│
├── strength/                    # Sinyal gücü hesaplayıcıları
│   ├── __init__.py
│   ├── base_strength.py        # Temel güç hesaplama
│   ├── predictive_strength.py  # Tahmin bazlı güç hesaplama
│   └── context_strength.py     # Bağlam duyarlı güç hesaplama
│
└── ml/                         # Makine öğrenmesi modülü
    ├── __init__.py
    ├── model_trainer.py        # Model eğitim sistemi
    ├── feature_selector.py     # Özellik seçimi
    ├── predictors.py          # Tahmin modelleri
    └── utils.py               # ML yardımcı fonksiyonları
```

## 🛠️ Kurulum

### Gereksinimler

```bash
pip install pandas numpy scikit-learn xgboost ta matplotlib seaborn joblib scipy
```

### Temel Kurulum

```python
# Signal Engine'i projenize ekleyin
from signal_engine import SignalManager

# Ana yönetici oluşturun
signal_manager = SignalManager()
```

## 🚀 Hızlı Başlangıç

### 1. Temel Kullanım

```python
import pandas as pd
from signal_engine import SignalManager

# Fiyat verilerini yükleyin (OHLCV formatında)
price_data = pd.read_csv('your_price_data.csv')

# Signal Manager oluşturun
manager = SignalManager()

# Basit bir yapılandırma
config = {
    'indicators': ['ema', 'rsi', 'macd', 'bollinger'],
    'strategies': ['trend_following', 'overextended_reversal'],
    'filters': ['market_regime_filter', 'volatility_regime_filter'],
    'strength_calculators': ['market_context_strength', 'probabilistic_strength']
}

# Sinyalleri işleyin
results = manager.process_data(price_data, config)

# Sonuçları görüntüleyin
print("Üretilen Sinyaller:", results['filtered_signals'].sum())
print("Ortalama Sinyal Gücü:", results['signal_strength'].mean())
```

### 2. Manuel Bileşen Kullanımı

```python
from signal_engine.indicators import registry as indicator_registry
from signal_engine.strategies import registry as strategy_registry
from signal_engine.filters import registry as filter_registry

# İndikatör hesaplama
ema_indicator = indicator_registry.create_indicator("ema", {"periods": [20, 50, 200]})
indicator_data = ema_indicator.calculate(price_data)

# Strateji sinyali üretme
trend_strategy = strategy_registry.create_strategy("trend_following")
signals = trend_strategy.generate_signals(indicator_data)

# Sinyal filtreleme
regime_filter = filter_registry.create_filter("market_regime_filter")
filtered_signals = regime_filter.apply(indicator_data, signals)
```

## 📊 İndikatör Sistemi

### Mevcut İndikatörler

#### Temel İndikatörler
- **EMA/SMA**: Hareketli ortalamalar
- **RSI**: Göreceli güç endeksi
- **MACD**: Hareketli ortalama yakınsama/ıraksama
- **Bollinger Bands**: Volatilite bantları
- **ATR**: Ortalama gerçek aralık
- **Stochastic**: Stokastik osilatör

#### Gelişmiş İndikatörler
- **Adaptive RSI**: Volatiliteye göre ayarlanan RSI
- **Multi-timeframe EMA**: Çoklu zaman dilimli EMA
- **Heikin Ashi**: Düzgünleştirilmiş mumlar
- **Supertrend**: Trend takip indikatörü
- **Ichimoku**: Ichimoku bulutu

#### Rejim İndikatörleri
- **Market Regime**: Piyasa durum analizi
- **Volatility Regime**: Volatilite seviye analizi
- **Trend Strength**: Trend gücü ölçümü

### Yeni İndikatör Ekleme

```python
from signal_engine.signal_indicator_plugin_system import BaseIndicator

class CustomIndicator(BaseIndicator):
    name = "custom_indicator"
    display_name = "Custom Indicator"
    description = "Özel indikatör açıklaması"
    category = "custom"
    
    default_params = {
        "period": 14,
        "threshold": 0.5
    }
    
    requires_columns = ["close"]
    output_columns = ["custom_value", "custom_signal"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        period = self.params.get("period", 14)
        
        # Özel hesaplama mantığınız
        result_df["custom_value"] = df["close"].rolling(period).mean()
        result_df["custom_signal"] = (result_df["custom_value"] > df["close"]).astype(int)
        
        return result_df

# İndikatörü kaydedin
from signal_engine.indicators import registry
registry.register(CustomIndicator)
```

## 📈 Strateji Sistemi

### Mevcut Stratejiler

#### Trend Stratejileri
- **Trend Following**: Temel trend takip
- **Multi-timeframe Trend**: Çoklu zaman dilimli trend
- **Adaptive Trend**: Adaptif trend stratejisi

#### Geri Dönüş Stratejileri
- **Overextended Reversal**: Aşırı uzama geri dönüşü
- **Pattern Reversal**: Mum formasyonu geri dönüşü
- **Divergence Reversal**: Uyumsuzluk geri dönüşü

#### Kırılma Stratejileri
- **Volatility Breakout**: Volatilite kırılması
- **Range Breakout**: Aralık kırılması
- **Support/Resistance Breakout**: Destek/direnç kırılması

### Yeni Strateji Ekleme

```python
from signal_engine.signal_strategy_system import BaseStrategy

class CustomStrategy(BaseStrategy):
    name = "custom_strategy"
    display_name = "Custom Strategy"
    description = "Özel strateji açıklaması"
    category = "custom"
    
    default_params = {
        "entry_threshold": 0.7,
        "exit_threshold": 0.3
    }
    
    required_indicators = ["rsi_14", "ema_20"]
    
    def generate_conditions(self, df, row, i):
        long_conditions = []
        short_conditions = []
        
        # Özel sinyal mantığınız
        if row["rsi_14"] < 30 and row["close"] > row["ema_20"]:
            long_conditions.append(True)
        
        if row["rsi_14"] > 70 and row["close"] < row["ema_20"]:
            short_conditions.append(True)
        
        return {"long": long_conditions, "short": short_conditions}
```

## 🎯 Filtreleme Sistemi

### Mevcut Filtreler

#### Rejim Filtreleri
- **Market Regime Filter**: Piyasa rejimine göre filtreleme
- **Volatility Regime Filter**: Volatilite rejimine göre filtreleme
- **Trend Strength Filter**: Trend gücüne göre filtreleme

#### İstatistiksel Filtreler
- **Z-Score Extreme Filter**: Aşırı Z-skoru filtreleme
- **Outlier Detection Filter**: Aykırı değer filtreleme
- **Historical Volatility Filter**: Tarihsel volatilite filtreleme

#### ML Filtreleri
- **Probabilistic Signal Filter**: Olasılık tabanlı filtreleme
- **Pattern Recognition Filter**: Desen tanıma filtreleme
- **Performance Classifier Filter**: Performans sınıflandırma filtreleme

### Filtre Kullanımı

```python
from signal_engine.signal_filter_system import FilterManager

# Filter manager oluşturun
filter_manager = FilterManager()

# Filtreleri ekleyin
filter_manager.add_rule("market_regime_filter", {
    "allowed_regimes": {
        "long": ["strong_uptrend", "weak_uptrend"],
        "short": ["strong_downtrend", "weak_downtrend"]
    }
})

# Filtreyi uygulayın
filtered_data = filter_manager.filter_signals(signals_df, ["market_regime_filter"])
```

## 💪 Sinyal Gücü Sistemi

### Güç Hesaplayıcıları

#### Tahmin Bazlı
- **Probabilistic Strength**: Tarihsel başarı olasılığı
- **Risk-Reward Strength**: Risk/ödül oranı analizi
- **ML Predictive Strength**: ML tahmin modelleri

#### Bağlam Duyarlı
- **Market Context Strength**: Piyasa bağlamı analizi
- **Indicator Confirmation Strength**: İndikatör onayları
- **Multi-timeframe Strength**: Çoklu zaman dilimi uyumu

### Güç Değerlendirmesi

```python
from signal_engine.signal_strength_system import StrengthManager

# Strength manager oluşturun
strength_manager = StrengthManager()

# Güç hesaplayıcıları ekleyin
strength_manager.add_calculator("market_context_strength", {"weight": 1.0})
strength_manager.add_calculator("probabilistic_strength", {"weight": 1.5})

# Güç değerlerini hesaplayın
strength_values = strength_manager.calculate_strength(
    indicator_data, 
    signals_df, 
    ["market_context_strength", "probabilistic_strength"]
)
```

## 🤖 Makine Öğrenmesi Entegrasyonu

### ML Bileşenleri

- **Model Trainer**: Otomatik model eğitimi
- **Feature Selector**: Özellik seçimi ve optimizasyonu
- **Signal Predictor**: Sinyal tahmin modeli
- **Strength Predictor**: Güç tahmin modeli

### ML Modeli Eğitimi

```python
from signal_engine.ml import ModelTrainer, FeatureSelector

# Özellik seçimi
feature_selector = FeatureSelector()
selected_features = feature_selector.select_features(
    df=indicator_data,
    features=all_feature_columns,
    target_column="future_return",
    methods=["variance_threshold", "feature_importance"]
)

# Model eğitimi
trainer = ModelTrainer()
model, metrics = trainer.train_signal_classifier(
    df=indicator_data,
    features=selected_features,
    target_column="signal_class",
    model_name="random_forest",
    grid_search=True
)

print("Model Performansı:", metrics)
```

### ML Tahminleri

```python
from signal_engine.ml.predictors import SignalPredictor

# Eğitilmiş modeli yükleyin
predictor = SignalPredictor(
    model_path="models/signal_classifier.joblib",
    config={
        "features": selected_features,
        "probability_threshold": 0.7
    }
)

# Tahminleri yapın
predicted_signals = predictor.predict(new_data)
signal_probabilities = predictor.predict_proba(new_data)
```

## 📊 Performans İzleme

### Temel Metrikler

```python
# Sinyal performansı
def calculate_signal_performance(signals, prices, holding_periods=20):
    results = []
    
    for i in range(len(signals)):
        if signals.iloc[i] != 0:
            entry_price = prices.iloc[i]
            
            # Forward return hesaplama
            if i + holding_periods < len(prices):
                exit_price = prices.iloc[i + holding_periods]
                
                if signals.iloc[i] > 0:  # Long
                    return_pct = (exit_price / entry_price - 1) * 100
                else:  # Short
                    return_pct = (1 - exit_price / entry_price) * 100
                
                results.append(return_pct)
    
    return {
        "total_signals": len(results),
        "win_rate": sum(1 for r in results if r > 0) / len(results) * 100,
        "avg_return": sum(results) / len(results),
        "max_return": max(results),
        "min_return": min(results)
    }

# Kullanım
performance = calculate_signal_performance(filtered_signals, df["close"])
print("Performans Raporu:", performance)
```

### Backtesting

```python
def simple_backtest(signals, prices, initial_capital=10000):
    capital = initial_capital
    positions = []
    
    for i in range(len(signals)):
        if signals.iloc[i] != 0 and i < len(prices) - 1:
            entry_price = prices.iloc[i]
            exit_price = prices.iloc[i + 1]  # Basit 1-bar tutma
            
            if signals.iloc[i] > 0:  # Long
                return_pct = (exit_price / entry_price - 1)
            else:  # Short
                return_pct = (1 - exit_price / entry_price)
            
            capital *= (1 + return_pct)
            positions.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": return_pct * 100,
                "capital": capital
            })
    
    total_return = (capital / initial_capital - 1) * 100
    
    return {
        "final_capital": capital,
        "total_return": total_return,
        "total_trades": len(positions),
        "positions": positions
    }
```

## ⚙️ Konfigürasyon

### Yapılandırma Dosyası Örneği

```python
# config.py
SIGNAL_CONFIG = {
    # İndikatör ayarları
    "indicators": {
        "ema": {"periods": [9, 21, 50, 200]},
        "rsi": {"periods": [14, 21]},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "bollinger": {"window": 20, "window_dev": 2.0},
        "market_regime": {"lookback_window": 50, "adx_threshold": 25}
    },
    
    # Strateji ayarları
    "strategies": {
        "trend_following": {
            "adx_threshold": 25,
            "confirmation_count": 3
        },
        "overextended_reversal": {
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "consecutive_candles": 3
        }
    },
    
    # Filtre ayarları
    "filters": {
        "market_regime_filter": {
            "regime_signal_map": {
                "strong_uptrend": {"long": True, "short": False},
                "strong_downtrend": {"long": False, "short": True}
            }
        },
        "volatility_regime_filter": {
            "high_volatility_filter": {"min_strength": 7}
        }
    },
    
    # Güç hesaplama ayarları
    "strength_calculators": {
        "market_context_strength": {"weight": 1.0},
        "probabilistic_strength": {"weight": 1.5, "lookback_window": 100}
    },
    
    # ML ayarları
    "ml_config": {
        "use_ml": True,
        "signal_model": "signal_classifier_v1",
        "strength_model": "strength_regressor_v1",
        "features": ["rsi_14", "adx", "macd_line", "market_regime_encoded"]
    }
}
```

## 🔧 Optimizasyon ve İyileştirme

### Parametre Optimizasyonu

```python
from itertools import product

def optimize_strategy_parameters(price_data, param_ranges):
    best_params = None
    best_performance = -float('inf')
    
    # Parametre kombinasyonları
    param_combinations = list(product(*param_ranges.values()))
    
    for combo in param_combinations:
        # Parametreleri hazırla
        current_params = dict(zip(param_ranges.keys(), combo))
        
        # Stratejiyi test et
        # ... test kodu ...
        
        # Performansı değerlendir
        performance_score = calculate_performance_score(results)
        
        if performance_score > best_performance:
            best_performance = performance_score
            best_params = current_params
    
    return best_params, best_performance

# Örnek kullanım
param_ranges = {
    "rsi_overbought": [70, 75, 80],
    "rsi_oversold": [20, 25, 30],
    "adx_threshold": [20, 25, 30]
}

best_params, score = optimize_strategy_parameters(price_data, param_ranges)
```

### Çoklu Market Testi

```python
def test_multiple_markets(market_data_dict, config):
    results = {}
    
    for market_name, data in market_data_dict.items():
        print(f"Testing {market_name}...")
        
        # Her market için test
        market_results = manager.process_data(data, config)
        
        # Performans hesapla
        performance = calculate_signal_performance(
            market_results['filtered_signals'], 
            data['close']
        )
        
        results[market_name] = performance
    
    return results

# Kullanım
markets = {
    "BTCUSD": btc_data,
    "ETHUSD": eth_data,
    "EURUSD": eur_data
}

multi_market_results = test_multiple_markets(markets, SIGNAL_CONFIG)
```

## 🚨 Hata Yönetimi ve Debug

### Logging Kurulumu

```python
import logging

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Signal Engine modüllerinde otomatik loglama aktif
```

### Yaygın Sorunlar ve Çözümler

#### 1. İndikatör Hesaplama Hataları
```python
# Problem: NaN değerleri
# Çözüm: Veri temizleme
def clean_price_data(df):
    # NaN değerleri temizle
    df = df.dropna()
    
    # Negatif fiyatları kontrol et
    for col in ['open', 'high', 'low', 'close']:
        df = df[df[col] > 0]
    
    # OHLC tutarlılığını kontrol et
    df = df[
        (df['high'] >= df['low']) & 
        (df['high'] >= df['open']) & 
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) & 
        (df['low'] <= df['close'])
    ]
    
    return df
```

#### 2. Bellek Kullanımı Optimizasyonu
```python
# Büyük veri setleri için chunk processing
def process_large_dataset(file_path, chunk_size=10000):
    results = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Her chunk'ı işle
        chunk_results = manager.process_data(chunk, config)
        results.append(chunk_results)
        
        # Belleği temizle
        del chunk_results
        import gc
        gc.collect()
    
    return pd.concat(results, ignore_index=True)
```

## 📈 Gerçek Zamanlı Trading

### WebSocket Entegrasyonu

```python
import websocket
import json
import threading

class RealTimeSignalEngine:
    def __init__(self, signal_manager, config):
        self.signal_manager = signal_manager
        self.config = config
        self.price_buffer = []
        self.buffer_size = 1000
    
    def on_message(self, ws, message):
        # Gelen price data'yı işle
        data = json.loads(message)
        
        # Buffer'a ekle
        self.price_buffer.append(data)
        
        # Buffer boyutunu kontrol et
        if len(self.price_buffer) > self.buffer_size:
            self.price_buffer = self.price_buffer[-self.buffer_size:]
        
        # Yeterli veri varsa sinyal üret
        if len(self.price_buffer) >= 200:  # Minimum required history
            df = pd.DataFrame(self.price_buffer)
            
            # Sinyalleri işle
            results = self.signal_manager.process_data(df, self.config)
            
            # Son sinyali kontrol et
            latest_signal = results['filtered_signals'].iloc[-1]
            latest_strength = results['signal_strength'].iloc[-1]
            
            if latest_signal != 0:
                self.handle_signal(latest_signal, latest_strength, data)
    
    def handle_signal(self, signal, strength, price_data):
        print(f"🚨 Yeni Sinyal: {signal}, Güç: {strength}")
        print(f"💰 Fiyat: {price_data['close']}")
        
        # Burada trading API'nizle işlem yapabilirsiniz
        # place_order(signal, strength, price_data)
    
    def start(self, websocket_url):
        ws = websocket.WebSocketApp(
            websocket_url,
            on_message=self.on_message
        )
        ws.run_forever()

# Kullanım
real_time_engine = RealTimeSignalEngine(signal_manager, SIGNAL_CONFIG)
real_time_engine.start("wss://stream.binance.com:9443/ws/btcusdt@kline_1m")
```

## 🔐 Güvenlik ve Risk Yönetimi

### Position Sizing

```python
def calculate_position_size(account_balance, signal_strength, risk_per_trade=0.02):
    """
    Hesap büyüklüğü ve sinyal gücüne göre pozisyon boyutu hesapla
    
    Args:
        account_balance: Hesap bakiyesi
        signal_strength: Sinyal gücü (0-100)
        risk_per_trade: İşlem başına risk yüzdesi (default: %2)
    """
    # Base risk
    base_risk = account_balance * risk_per_trade
    
    # Sinyal gücüne göre ayarlama
    strength_multiplier = signal_strength / 100  # 0-1 arası
    
    # Final position size
    position_size = base_risk * strength_multiplier
    
    return min(position_size, account_balance * 0.1)  # Max %10 risk

# Kullanım
account_balance = 10000
signal_strength = 75
position = calculate_position_size(account_balance, signal_strength)
print(f"💰 Pozisyon Büyüklüğü: ${position:.2f}")
```

### Stop Loss ve Take Profit

```python
def calculate_stop_take_levels(entry_price, signal_direction, atr_value, 
                             stop_multiplier=2.0, take_multiplier=3.0):
    """
    ATR bazlı stop loss ve take profit seviyeleri hesapla
    """
    if signal_direction > 0:  # Long
        stop_loss = entry_price - (atr_value * stop_multiplier)
        take_profit = entry_price + (atr_value * take_multiplier)
    else:  # Short
        stop_loss = entry_price + (atr_value * stop_multiplier)
        take_profit = entry_price - (atr_value * take_multiplier)
    
    return stop_loss, take_profit

# Kullanım
entry = 50000
direction = 1  # Long
atr = 500
stop, take = calculate_stop_take_levels(entry, direction, atr)
print(f"📉 Stop Loss: ${stop:.2f}")
print(f"📈 Take Profit: ${take:.2f}")
```

## 📊 Reporting ve Analytics

### Performans Dashboard

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_dashboard(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Equity Curve
    axes[0,0].plot(results_df.index, results_df['cumulative_return'])
    axes[0,0].set_title('Equity Curve')
    axes[0,0].set_ylabel('Cumulative Return (%)')
    
    # 2. Signal Distribution
    signal_counts = results_df['signal'].value_counts()
    axes[0,1].bar(['No Signal', 'Long', 'Short'], 
                  [signal_counts.get(0, 0), signal_counts.get(1, 0), signal_counts.get(-1, 0)])
    axes[0,1].set_title('Signal Distribution')
    
    # 3. Strength Distribution
    axes[1,0].hist(results_df['signal_strength'], bins=20, alpha=0.7)
    axes[1,0].set_title('Signal Strength Distribution')
    axes[1,0].set_xlabel('Strength Value')
    
    # 4. Monthly Returns Heatmap
    monthly_returns = results_df['monthly_return'].unstack()
    sns.heatmap(monthly_returns, annot=True, cmap='RdYlGn', center=0, ax=axes[1,1])
    axes[1,1].set_title('Monthly Returns Heatmap')
    
    plt.tight_layout()
    plt.show()

# Kullanım
# create_performance_dashboard(backtest_results)
```

## 🎓 Gelişmiş Örnekler

### Multi-Asset Portfolio

```python
class MultiAssetSignalEngine:
    def __init__(self):
        self.signal_manager = SignalManager()
        self.assets = {}
        self.correlations = None
    
    def add_asset(self, symbol, data, weight=1.0):
        self.assets[symbol] = {
            'data': data,
            'weight': weight,
            'signals': None,
            'strength': None
        }
    
    def generate_portfolio_signals(self, config):
        # Her asset için sinyal üret
        for symbol, asset in self.assets.items():
            results = self.signal_manager.process_data(asset['data'], config)
            asset['signals'] = results['filtered_signals']
            asset['strength'] = results['signal_strength']
        
        # Korelasyon bazlı sinyal ayarlaması
        self.adjust_signals_for_correlation()
        
        return self.assets
    
    def adjust_signals_for_correlation(self):
        """Yüksek korelasyonlu assetlerde sinyal çakışmalarını ayarla"""
        symbols = list(self.assets.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                # İki asset arasındaki korelasyonu kontrol et
                correlation = self.calculate_correlation(symbol1, symbol2)
                
                if abs(correlation) > 0.8:  # Yüksek korelasyon
                    # Daha güçlü sinyali tut, diğerini zayıflaştır
                    self.resolve_signal_conflict(symbol1, symbol2)
    
    def calculate_correlation(self, symbol1, symbol2):
        data1 = self.assets[symbol1]['data']['close']
        data2 = self.assets[symbol2]['data']['close']
        return data1.corr(data2)
    
    def resolve_signal_conflict(self, symbol1, symbol2):
        """Çakışan sinyalleri çöz"""
        asset1 = self.assets[symbol1]
        asset2 = self.assets[symbol2]
        
        # Son sinyalleri karşılaştır
        last_strength1 = asset1['strength'].iloc[-1] if len(asset1['strength']) > 0 else 0
        last_strength2 = asset2['strength'].iloc[-1] if len(asset2['strength']) > 0 else 0
        
        if last_strength1 > last_strength2:
            # Asset2'nin gücünü azalt
            asset2['strength'] *= 0.7
        else:
            # Asset1'in gücünü azalt
            asset1['strength'] *= 0.7

# Kullanım
portfolio_engine = MultiAssetSignalEngine()
portfolio_engine.add_asset("BTCUSD", btc_data, weight=0.4)
portfolio_engine.add_asset("ETHUSD", eth_data, weight=0.3)
portfolio_engine.add_asset("ADAUSD", ada_data, weight=0.3)

portfolio_signals = portfolio_engine.generate_portfolio_signals(SIGNAL_CONFIG)
```

### Adaptive Strategy Selection

```python
class AdaptiveStrategySelector:
    def __init__(self, signal_manager):
        self.signal_manager = signal_manager
        self.strategy_performance = {}
        self.lookback_window = 100
        
    def track_strategy_performance(self, strategy_name, signals, prices):
        """Strateji performansını izle"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        # Son sinyallerin performansını hesapla
        for i in range(len(signals)):
            if signals.iloc[i] != 0 and i < len(prices) - 20:
                entry_price = prices.iloc[i]
                exit_price = prices.iloc[i + 20]  # 20 bar sonra çık
                
                if signals.iloc[i] > 0:  # Long
                    return_pct = (exit_price / entry_price - 1) * 100
                else:  # Short
                    return_pct = (1 - exit_price / entry_price) * 100
                
                self.strategy_performance[strategy_name].append({
                    'timestamp': i,
                    'return': return_pct,
                    'win': return_pct > 0
                })
        
        # Sadece son lookback_window kadar tutma kayıt tut
        if len(self.strategy_performance[strategy_name]) > self.lookback_window:
            self.strategy_performance[strategy_name] = \
                self.strategy_performance[strategy_name][-self.lookback_window:]
    
    def get_best_strategies(self, top_n=3):
        """En iyi performans gösteren stratejileri getir"""
        strategy_scores = {}
        
        for strategy, performance in self.strategy_performance.items():
            if len(performance) >= 10:  # Minimum 10 trade
                win_rate = sum(1 for p in performance if p['win']) / len(performance)
                avg_return = sum(p['return'] for p in performance) / len(performance)
                
                # Kombinasyon skoru: win_rate * 0.6 + normalized_avg_return * 0.4
                score = win_rate * 0.6 + min(max(avg_return / 5, -1), 1) * 0.4
                strategy_scores[strategy] = score
        
        # En iyi top_n stratejiyi döndür
        best_strategies = sorted(strategy_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        return [strategy for strategy, _ in best_strategies]
    
    def adaptive_signal_generation(self, price_data, base_config):
        """Adaptif strateji seçimi ile sinyal üretimi"""
        # En iyi stratejileri seç
        best_strategies = self.get_best_strategies()
        
        if not best_strategies:
            # Performans verisi yoksa tüm stratejileri kullan
            best_strategies = base_config.get('strategies', [])
        
        # Konfigürasyonu güncelle
        adaptive_config = base_config.copy()
        adaptive_config['strategies'] = best_strategies
        
        # Sinyalleri üret
        results = self.signal_manager.process_data(price_data, adaptive_config)
        
        # Performansı güncelle
        for strategy in best_strategies:
            self.track_strategy_performance(
                strategy, 
                results['filtered_signals'], 
                price_data['close']
            )
        
        return results

# Kullanım
adaptive_selector = AdaptiveStrategySelector(signal_manager)
adaptive_results = adaptive_selector.adaptive_signal_generation(price_data, SIGNAL_CONFIG)
```

## 🔄 Sürekli İyileştirme Döngüsü

### Auto-Retraining Pipeline

```python
import schedule
import time
from datetime import datetime, timedelta

class AutoRetrainingPipeline:
    def __init__(self, signal_manager, ml_manager):
        self.signal_manager = signal_manager
        self.ml_manager = ml_manager
        self.last_training_date = None
        self.training_interval_days = 7
        
    def should_retrain(self):
        """Yeniden eğitim gerekli mi kontrol et"""
        if self.last_training_date is None:
            return True
        
        days_since_training = (datetime.now() - self.last_training_date).days
        return days_since_training >= self.training_interval_days
    
    def collect_new_data(self):
        """Yeni veri topla"""
        # Burada external API'den yeni veri çekme mantığı
        # Örnek: exchange API, data provider vs.
        pass
    
    def retrain_models(self, new_data):
        """Modelleri yeniden eğit"""
        try:
            # Feature engineering
            from signal_engine.ml import FeatureSelector
            
            feature_selector = FeatureSelector()
            
            # Yeni features seç
            all_features = [col for col in new_data.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            selected_features = feature_selector.select_features(
                df=new_data,
                features=all_features,
                target_column="future_return_class",
                methods=["variance_threshold", "feature_importance"]
            )
            
            # Signal classifier'ı yeniden eğit
            signal_training_config = {
                "model_type": "signal",
                "features": selected_features,
                "target": "signal_class",
                "algorithm": "random_forest",
                "grid_search": True
            }
            
            signal_results = self.ml_manager.train_model(new_data, signal_training_config)
            
            # Strength regressor'ı yeniden eğit
            strength_training_config = {
                "model_type": "strength",
                "features": selected_features,
                "target": "signal_strength",
                "algorithm": "xgboost_regressor",
                "grid_search": True
            }
            
            strength_results = self.ml_manager.train_model(new_data, strength_training_config)
            
            # Eğitim tarihini güncelle
            self.last_training_date = datetime.now()
            
            return {
                "signal_model": signal_results,
                "strength_model": strength_results,
                "retrain_date": self.last_training_date
            }
            
        except Exception as e:
            print(f"❌ Model retraining failed: {e}")
            return None
    
    def evaluate_model_performance(self, old_results, new_results):
        """Yeni model performansını değerlendir"""
        # Basit performans karşılaştırması
        old_accuracy = old_results.get('metrics', {}).get('accuracy', 0)
        new_accuracy = new_results.get('metrics', {}).get('accuracy', 0)
        
        improvement = new_accuracy - old_accuracy
        
        return {
            "improved": improvement > 0.02,  # %2'den fazla iyileşme
            "improvement": improvement,
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy
        }
    
    def scheduled_retrain(self):
        """Zamanlanmış yeniden eğitim"""
        if self.should_retrain():
            print("🔄 Starting scheduled model retraining...")
            
            # Yeni veri topla
            new_data = self.collect_new_data()
            
            if new_data is not None:
                # Modelleri yeniden eğit
                results = self.retrain_models(new_data)
                
                if results:
                    print("✅ Model retraining completed successfully")
                    print(f"📊 Signal Model Accuracy: {results['signal_model']['metrics'].get('accuracy', 'N/A')}")
                    print(f"📊 Strength Model R²: {results['strength_model']['metrics'].get('r2_score', 'N/A')}")
                else:
                    print("❌ Model retraining failed")
            else:
                print("❌ Failed to collect new data")
        else:
            print("⏳ Retraining not needed yet")

# Zamanlanmış otomatik eğitim
def setup_auto_retrain_schedule(pipeline):
    # Her hafta Pazar günü saat 02:00'da çalış
    schedule.every().sunday.at("02:00").do(pipeline.scheduled_retrain)
    
    # Scheduler döngüsü
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Her saat kontrol et

# Kullanım
auto_pipeline = AutoRetrainingPipeline(signal_manager, ml_manager)
# setup_auto_retrain_schedule(auto_pipeline)  # Background thread'de çalıştır
```

## 📱 Web Dashboard Entegrasyonu

### Flask-based Dashboard

```python
from flask import Flask, render_template, jsonify, request
import json

app = Flask(__name__)

class TradingDashboard:
    def __init__(self, signal_manager):
        self.signal_manager = signal_manager
        self.latest_results = None
        self.performance_history = []
    
    def update_data(self, new_data, config):
        """Yeni veri ile dashboard'u güncelle"""
        self.latest_results = self.signal_manager.process_data(new_data, config)
        
        # Performans geçmişine ekle
        self.performance_history.append({
            'timestamp': new_data.index[-1],
            'total_signals': len(self.latest_results['filtered_signals']),
            'avg_strength': self.latest_results['signal_strength'].mean()
        })
        
        # Son 100 kaydı tut
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

dashboard = TradingDashboard(signal_manager)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/signals')
def get_signals():
    if dashboard.latest_results is None:
        return jsonify({'error': 'No data available'})
    
    return jsonify({
        'signals': dashboard.latest_results['filtered_signals'].tolist(),
        'strength': dashboard.latest_results['signal_strength'].tolist(),
        'timestamp': dashboard.latest_results['indicator_data'].index.tolist()
    })

@app.route('/api/performance')
def get_performance():
    return jsonify(dashboard.performance_history)

@app.route('/api/update', methods=['POST'])
def update_data():
    try:
        # Yeni veri al
        data = request.json
        new_price_data = pd.DataFrame(data['price_data'])
        config = data['config']
        
        # Dashboard'u güncelle
        dashboard.update_data(new_price_data, config)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Dashboard HTML Template

```html
<!-- templates/dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Trading Signal Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container-fluid">
        <h1 class="mt-3">🚀 Signal Engine Dashboard</h1>
        
        <div class="row mt-4">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">📊 Real-time Stats</h5>
                        <p id="total-signals" class="h4">Loading...</p>
                        <p id="avg-strength" class="text-muted">Average Strength: -</p>
                        <p id="last-update" class="text-muted">Last Update: -</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-9">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">📈 Signal Chart</h5>
                        <div id="signal-chart"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">💪 Strength Distribution</h5>
                        <div id="strength-chart"></div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">📊 Performance History</h5>
                        <div id="performance-chart"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard JavaScript kodu
        function updateDashboard() {
            // Sinyal verilerini al
            $.get('/api/signals', function(data) {
                if (data.error) {
                    console.error('Error:', data.error);
                    return;
                }
                
                // İstatistikleri güncelle
                const totalSignals = data.signals.filter(s => s !== 0).length;
                const avgStrength = data.strength.reduce((a, b) => a + b, 0) / data.strength.length;
                
                $('#total-signals').text(totalSignals + ' Active Signals');
                $('#avg-strength').text(`Average Strength: ${avgStrength.toFixed(1)}`);
                $('#last-update').text(`Last Update: ${new Date().toLocaleTimeString()}`);
                
                // Grafikleri güncelle
                updateSignalChart(data);
                updateStrengthChart(data);
            });
            
            // Performans verilerini al
            $.get('/api/performance', function(data) {
                updatePerformanceChart(data);
            });
        }
        
        function updateSignalChart(data) {
            const trace = {
                x: data.timestamp,
                y: data.signals,
                type: 'scatter',
                mode: 'markers',
                marker: {
                    color: data.signals.map(s => s > 0 ? 'green' : s < 0 ? 'red' : 'gray'),
                    size: data.strength.map(s => Math.max(5, s/10))
                },
                name: 'Signals'
            };
            
            const layout = {
                title: 'Trading Signals Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Signal Type' }
            };
            
            Plotly.newPlot('signal-chart', [trace], layout);
        }
        
        function updateStrengthChart(data) {
            const trace = {
                x: data.strength,
                type: 'histogram',
                nbinsx: 20,
                marker: { color: 'blue', opacity: 0.7 }
            };
            
            const layout = {
                title: 'Signal Strength Distribution',
                xaxis: { title: 'Strength Value' },
                yaxis: { title: 'Frequency' }
            };
            
            Plotly.newPlot('strength-chart', [trace], layout);
        }
        
        function updatePerformanceChart(data) {
            const trace = {
                x: data.map(d => d.timestamp),
                y: data.map(d => d.avg_strength),
                type: 'scatter',
                mode: 'lines',
                line: { color: 'purple' },
                name: 'Avg Strength'
            };
            
            const layout = {
                title: 'Performance History',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Average Strength' }
            };
            
            Plotly.newPlot('performance-chart', [trace], layout);
        }
        
        // Her 5 saniyede bir güncelle
        setInterval(updateDashboard, 5000);
        
        // İlk yükleme
        updateDashboard();
    </script>
</body>
</html>
```

## 🔧 Troubleshooting ve SSS

### Sık Karşılaşılan Sorular

**Q: İndikatörler NaN değer döndürüyor?**
```python
# A: Veri temizleme ve validation ekleyin
def validate_price_data(df):
    # Gerekli sütunları kontrol et
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # NaN değerleri kontrol et
    if df[required_cols].isnull().any().any():
        print("⚠️ Warning: NaN values found, cleaning data...")
        df = df.dropna(subset=required_cols)
    
    # Minimum veri kontrolü
    if len(df) < 100:
        raise ValueError("Insufficient data: need at least 100 rows")
    
    return df
```

**Q: Stratejiler sinyal üretmiyor?**
```python
# A: Debug modunu aktif edin
import logging
logging.getLogger('signal_engine').setLevel(logging.DEBUG)

# Strateji validasyonunu kontrol edin
def debug_strategy(strategy, df):
    print(f"🔍 Debugging strategy: {strategy.name}")
    print(f"📊 Required indicators: {strategy.required_indicators}")
    print(f"📈 Available columns: {list(df.columns)}")
    
    # Eksik indikatörleri kontrol et
    missing = [col for col in strategy.required_indicators if col not in df.columns]
    if missing:
        print(f"❌ Missing indicators: {missing}")
    else:
        print("✅ All required indicators available")
    
    # Test sinyali üret
    try:
        signals = strategy.generate_signals(df)
        signal_count = (signals != 0).sum()
        print(f"📈 Generated {signal_count} signals")
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
```

**Q: Performans nasıl optimize edilir?**
```python
# A: Profiling ve optimizasyon
import cProfile
import time

def profile_signal_generation(manager, data, config):
    """Sinyal üretim sürecini profile et"""
    
    def timed_process():
        return manager.process_data(data, config)
    
    # Timing
    start_time = time.time()
    results = timed_process()
    end_time = time.time()
    
    print(f"⏱️ Total processing time: {end_time - start_time:.2f} seconds")
    print(f"📊 Processed {len(data)} rows")
    print(f"⚡ Processing rate: {len(data)/(end_time - start_time):.0f} rows/second")
    
    return results

# Bellek kullanımını izle
import psutil
import os

def monitor_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"💾 Memory usage: {memory_mb:.1f} MB")
    return memory_mb
```

## 📚 Kaynaklar ve Referanslar

### Faydalı Linkler
- **Technical Analysis Library**: [TA-Lib Documentation](https://ta-lib.org/)
- **Machine Learning**: [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- **XGBoost**: [XGBoost Documentation](https://xgboost.readthedocs.io/)
- **Pandas**: [Pandas Documentation](https://pandas.pydata.org/docs/)

### Önerilen Okumalar
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Algorithmic Trading" - Stefan Jansen
- "Evidence-Based Technical Analysis" - David Aronson

### Community ve Destek
- **GitHub Issues**: Sorun bildirimleri için
- **Discord Community**: Real-time destek ve tartışma
- **Wiki**: Detaylı dokümantasyon

## 🤝 Katkıda Bulunma

### Geliştirme Süreci
1. **Fork** edin
2. **Feature branch** oluşturun (`git checkout -b feature/amazing-feature`)
3. **Commit** edin (`git commit -m 'Add amazing feature'`)
4. **Push** edin (`git push origin feature/amazing-feature`)
5. **Pull Request** açın

### Kod Standartları
```python
# Type hints kullanın
def calculate_signals(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    pass

# Docstring'leri unutmayın
def custom_indicator(data: pd.DataFrame) -> pd.DataFrame:
    """
    Custom indicator calculation.
    
    Args:
        data: Price data with OHLCV columns
        
    Returns:
        DataFrame with indicator values added
        
    Raises:
        ValueError: If required columns are missing
    """
    pass

# Unit testler yazın
def test_custom_indicator():
    # Test data
    test_data = create_test_data()
    
    # Test execution
    result = custom_indicator(test_data)
    
    # Assertions
    assert "custom_value" in result.columns
    assert not result["custom_value"].isnull().any()
```

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## ⚠️ Yasal Uyarı

Bu yazılım eğitim ve araştırma amaçlıdır. Gerçek para ile trading yapmadan önce:

- **Dikkatli test edin**: Paper trading ile başlayın
- **Risk yönetimi**: Kaybetmeyi göze alabileceğiniz miktarla başlayın
- **Profesyonel danışmanlık**: Mali müşavirinize danışın
- **Yasal sorumluluk**: Kullanım sonucu oluşan kayıplardan geliştirici sorumlu değildir

---

## 🎯 Son Sözler

Signal Engine Trading Bot, modüler yapısı ve genişletilebilir mimarisiyle profesyonel trading sistemleri geliştirmek için güçlü bir temel sağlar. Sürekli geliştirme ve topluluk katkılarıyla daha da güçlenmeye devam edecektir.

**Happy trading! 🚀📈**

---

*Son güncelleme: 2024*  
*Versiyon: 0.1.0*