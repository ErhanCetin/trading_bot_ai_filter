  def check_market_regime_values(df: pd.DataFrame, fix_missing: bool = True) -> pd.DataFrame:   
    """
    Stratejileri test etmek için kullanılan yardımcı modül
    """
    """
    Market regime değerlerini kontrol eder ve sorunları düzeltir
    
    Args:
        df: İndikatörlerin hesaplandığı DataFrame
        fix_missing: Eksik değerleri düzeltme
        
    Returns:
        Düzeltilmiş DataFrame
    """
    result_df = df.copy()
    
    # Market regime var mı kontrol et
    if "market_regime" not in result_df.columns:
        print("❌ market_regime sütunu bulunamadı!")
        
        if fix_missing:
            print("➕ market_regime sütunu ekleniyor (tüm değerler 'unknown')...")
            result_df["market_regime"] = "unknown"
        
        return result_df
    
    # İstatistikleri yazdır
    total_rows = len(result_df)
    null_count = result_df["market_regime"].isnull().sum()
    none_count = sum(1 for x in result_df["market_regime"] if x is None)
    
    valid_values = result_df["market_regime"].dropna().unique()
    
    print(f"\n>> Market Regime Özeti:")
    print(f"   Toplam satır: {total_rows}")
    print(f"   Null değer sayısı: {null_count}")
    print(f"   None değer sayısı: {none_count}")
    print(f"   Geçerli değerler: {valid_values}")
    
    # Değer dağılımını göster
    value_counts = result_df["market_regime"].value_counts()
    print("\n>> Değer dağılımı:")
    for value, count in value_counts.items():
        print(f"   {value}: {count} satır ({count/total_rows*100:.1f}%)")
    
    # Eksik değerleri düzelt
    if fix_missing and (null_count > 0 or none_count > 0):
        print("\n>> Eksik değerler düzeltiliyor...")
        
        # NaN değerlerini "unknown" ile doldur
        result_df["market_regime"] = result_df["market_regime"].fillna("unknown")
        
        # None değerlerini "unknown" ile doldur
        result_df.loc[result_df["market_regime"].isnull(), "market_regime"] = "unknown"
        
        # Sonucu göster
        after_fix = result_df["market_regime"].value_counts()
        print("\n>> Düzeltme sonrası:")
        print(f"   'unknown' değer sayısı: {after_fix.get('unknown', 0)}")
    
    return result_dfimport os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import logging
import os
import sys

# Modül yolunu ayarla
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# İndikatör helpers'dan fonksiyonları içe aktar
from debug_indicator_list import debug_indicators

# Signal Engine bileşenlerini içe aktar
try:
    from signal_engine.strategies import registry as strategy_registry
    from signal_engine.signal_strategy_system import StrategyManager
    from signal_engine.indicators import registry as indicator_registry
    from signal_engine.signal_indicator_plugin_system import IndicatorManager
except ImportError as e:
    print(f"Gerekli modüller içe aktarılamadı: {e}")
    print("Lütfen signal_engine paketinin yolunu kontrol edin.")
    sys.exit(1)

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(size: int = 500, add_patterns: bool = True) -> pd.DataFrame:
    """
    Test için örnek fiyat verisi oluşturur, isteğe bağlı olarak belirli desenler ekleyebilir
    
    Args:
        size: Oluşturulacak örnek veri noktası sayısı
        add_patterns: Trend dönüşler, yükseliş/düşüş dalgaları ve volatilite artışları gibi desenler ekle
        
    Returns:
        Örnek fiyat verisi DataFrame'i
    """
    # Günlük data için tarih aralığı oluştur
    index = pd.date_range(start='2020-01-01', periods=size, freq='D')
    
    # Yapay fiyat verisi oluştur - basit random walk
    np.random.seed(42)  # Tekrarlanabilirlik için
    
    # Başlangıç değerleri
    base_price = 100.0
    volatility = 0.02
    
    # Random walk ile fiyat simülasyonu
    changes = np.random.normal(0, volatility, size)
    price_data = base_price * (1 + np.cumsum(changes))
    
    if add_patterns:
        # Daha gerçekçi fiyat desenleri oluştur
        
        # Uzun vadeli trend bileşeni (yükseliş, düşüş, tekrar yükseliş)
        long_trend = np.concatenate([
            np.linspace(0, 0.5, size // 3),            # Yükseliş trendi
            np.linspace(0.5, -0.2, size // 3),          # Düşüş trendi
            np.linspace(-0.2, 0.3, size - 2*(size // 3))  # Tekrar yükseliş
        ])
        
        # Orta vadeli trend bileşeni (dalgalar)
        medium_trend = 0.2 * np.sin(np.linspace(0, 4*np.pi, size))
        
        # Kısa vadeli dalgalanmalar
        short_trend = 0.1 * np.sin(np.linspace(0, 20*np.pi, size))
        
        # Volatilite değişimi (bazı bölgelerde yüksek volatilite)
        # Volatilite çarpanı oluştur (1 civarında, bazı bölgelerde yükselir)
        volatility_factor = np.ones(size)
        
        # Yüksek volatilite bölgeleri ekle (örneğin, 3 bölge)
        high_vol_regions = [
            (size // 5, size // 4),       # İlk yüksek volatilite bölgesi
            (size // 2, size // 2 + size // 6),  # İkinci bölge
            (3 * size // 4, 3 * size // 4 + size // 5)   # Üçüncü bölge
        ]
        
        for start, end in high_vol_regions:
            # Bu bölgede volatiliteyi 2-3 kat artır
            volatility_multiplier = np.linspace(1, 3, end - start)  # Kademeli artış
            volatility_factor[start:end] = volatility_multiplier
        
        # Ani fiyat değişimleri (örneğin haberler sonrası)
        shock_points = [size // 6, size // 2 - size // 10, 3 * size // 4 + size // 15]
        for point in shock_points:
            if point < size:
                # %5 ile %10 arasında ani değişimler
                price_data[point] *= (1 + np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.1))
        
        # Tüm bileşenleri birleştir
        price_data = price_data + (price_data * long_trend) + (price_data * medium_trend) + (price_data * short_trend)
        
        # Her gün için volatiliteyi ayarla
        for i in range(1, size):
            price_data[i] = price_data[i-1] * (1 + np.random.normal(0, volatility * volatility_factor[i]))
    else:
        # Basit trend ve volatilite ekle
        trend = np.linspace(0, 0.5, size)  # Yükseliş trendi
        volatility_factor = np.sin(np.linspace(0, 15, size)) * 0.1 + 1  # Volatilite dalgalanması
        price_data = price_data + trend + price_data * volatility_factor
    
    # DataFrame oluştur
    df = pd.DataFrame(index=index)
    df['close'] = price_data
    
    # High, Low, Open değerleri oluştur
    df['high'] = df['close'] * (1 + np.random.uniform(0.001, 0.02, size))
    df['low'] = df['close'] * (1 - np.random.uniform(0.001, 0.02, size))
    df['open'] = df['close'].shift(1)
    df.loc[df.index[0], 'open'] = df['low'].iloc[0]  # İlk değer için open değeri
    
    # Günlükleri ayarla - OHLC gerçek verilere daha yakın olsun
    for i in range(size):
        # Fiyat aralığını belirle
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        close_price = df['close'].iloc[i]
        
        # Open fiyatını ayarla (ilk satır için zaten ayarladık)
        if i > 0:
            prev_close = df['close'].iloc[i-1]
            # Open değerini, önceki kapanış ile şimdiki kapanış arasında rastgele bir nokta olarak belirle
            min_open = min(prev_close, close_price)
            max_open = max(prev_close, close_price)
            df.loc[df.index[i], 'open'] = min_open + np.random.random() * (max_open - min_open)
        
        # High/Low değerlerini ayarla
        max_price = max(df['open'].iloc[i], close_price)
        min_price = min(df['open'].iloc[i], close_price)
        
        # High ve Low, Open ve Close'un dışında olsun
        df.loc[df.index[i], 'high'] = max_price + np.random.uniform(0.001, 0.01) * max_price
        df.loc[df.index[i], 'low'] = min_price - np.random.uniform(0.001, 0.01) * min_price
    
    # Volume değerleri oluştur - fiyat değişimi ile ilişkili olsun
    base_volume = 1000000
    price_changes = np.abs(df['close'].pct_change().fillna(0))
    df['volume'] = base_volume * (1 + 3 * price_changes) * (1 + np.sin(np.linspace(0, 15, size)) * 0.5)
    
    # Hacim spike'ları ekle
    volume_spikes = [size // 7, size // 3, 2 * size // 3, 5 * size // 6]
    for spike in volume_spikes:
        if spike < size:
            df.loc[df.index[spike], 'volume'] *= np.random.uniform(3, 8)  # 3x-8x hacim spike'ı
    
    # open_time (millisaniye cinsinden timestamp) ekle - indikatörler için gerekli olabilir
    df['open_time'] = df.index.astype(np.int64) // 10**6
    
    # Symbol ve interval ekle
    df['symbol'] = 'BTCUSDT'
    df['interval'] = '1d'
    
    return df


def display_available_strategies():
    """
    Kullanılabilir tüm stratejileri ve bilgilerini görüntüler
    """
    print("\n=== KULLANILABILIR STRATEJİLER ===")
    
    all_strategies = strategy_registry.get_all_strategies()
    categories = {}
    
    # Stratejileri kategorilere ayır
    for name, strategy_class in all_strategies.items():
        if strategy_class.category not in categories:
            categories[strategy_class.category] = []
        
        categories[strategy_class.category].append({
            'name': name,
            'display_name': strategy_class.display_name,
            'description': strategy_class.description,
            'default_params': strategy_class.default_params,
            'required_indicators': strategy_class.required_indicators,
            'optional_indicators': getattr(strategy_class, 'optional_indicators', [])
        })
    
    # Kategorilere göre yazdır
    for category, strategies in categories.items():
        print(f"\n== {category.upper()} STRATEJİLERİ ==")
        for strat in strategies:
            print(f"- {strat['display_name']} ({strat['name']})")
            print(f"  Açıklama: {strat['description']}")
            print(f"  Gerekli indikatörler: {strat['required_indicators']}")
            print(f"  Opsiyonel indikatörler: {strat['optional_indicators']}")
            print(f"  Varsayılan parametreler:")
            for param, value in strat['default_params'].items():
                print(f"    - {param}: {value}")
            print("")


def prepare_data_for_strategy(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """
    Belirli bir strateji için gerekli indikatörleri hesaplar ve veriyi hazırlar
    
    Args:
        df: Ham fiyat verisi DataFrame
        strategy_name: Strateji adı
        
    Returns:
        İndikatörler hesaplanmış DataFrame
    """
    # Strateji sınıfını al
    strategy_class = strategy_registry.get_strategy(strategy_name)
    if not strategy_class:
        logger.error(f"Strateji bulunamadı: {strategy_name}")
        return df
    
    # Gerekli ve opsiyonel indikatörleri al
    required_indicators = strategy_class.required_indicators
    optional_indicators = getattr(strategy_class, 'optional_indicators', [])
    
    # İndikatör adı -> sınıf adı eşleştirmesi için bazı yaygın indikatörleri tanımla
    indicator_mappings = {
        "adx": "trend_strength",
        "di_pos": "trend_strength",
        "di_neg": "trend_strength",
        "rsi_14": "rsi",
        "ema_20": "ema",
        "ema_50": "ema",
        "ema_200": "ema",
        "sma_20": "sma",
        "sma_50": "sma",
        "macd_line": "macd",
        "macd_signal": "macd",
        "macd_histogram": "macd",
        "bollinger_upper": "bollinger",
        "bollinger_lower": "bollinger",
        "bollinger_width": "bollinger",
        "atr": "atr",
        "supertrend": "supertrend",
        "market_regime": "market_regime",
        "volatility_regime": "volatility_regime"
    }
    
    # Strateji için gerekli indikatörleri belirle
    needed_indicators = {}
    
    # Gerekli indikatörler ekle
    for indicator_name in required_indicators:
        # İndikatör adından sınıf adı belirle
        indicator_class = indicator_mappings.get(indicator_name, indicator_name)
        if indicator_class not in needed_indicators:
            needed_indicators[indicator_class] = {}
    
    # Opsiyonel indikatörler ekle
    for indicator_name in optional_indicators:
        indicator_class = indicator_mappings.get(indicator_name, indicator_name)
        if indicator_class not in needed_indicators:
            needed_indicators[indicator_class] = {}
    
    # Bazı özel indikatörler için parametreler belirle
    special_params = {
        "ema": {"periods": [9, 20, 50, 200]},
        "sma": {"periods": [10, 20, 50, 200]},
        "rsi": {"periods": [7, 14, 21]},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "bollinger": {"window": 20, "window_dev": 2.0},
        "atr": {"window": 14},
        "market_regime": {"lookback_window": 50, "adx_threshold": 25},
        "volatility_regime": {"lookback_window": 50}
    }
    
    # Özel parametreleri ekle
    for indicator_name, params in special_params.items():
        if indicator_name in needed_indicators:
            needed_indicators[indicator_name] = params
    
    # İndikatör yöneticisi oluştur
    indicator_manager = IndicatorManager(indicator_registry)
    
    # İhtiyaç duyulan tüm indikatörleri ekle
    for indicator_name, params in needed_indicators.items():
        indicator_manager.add_indicator(indicator_name, params)
    
    # Rejim indikatörlerinin bağımlılık sorunları için özel işlem
    if ("market_regime" in needed_indicators or 
        "trend_strength" in needed_indicators or 
        "adx" in needed_indicators):
        
        # Önce ADX hesaplayıcıları ekle
        result_df = indicator_manager.calculate_indicators(df, ["trend_strength"])
        
        # Sonra market_regime ekle
        if "market_regime" in needed_indicators:
            result_df = indicator_manager.calculate_indicators(result_df, ["market_regime"])
        
        # Sonra diğerleri
        other_indicators = [ind for ind in needed_indicators.keys() 
                          if ind not in ["trend_strength", "market_regime"]]
        
        if other_indicators:
            result_df = indicator_manager.calculate_indicators(result_df, other_indicators)
    else:
        # Normal hesaplama
        result_df = indicator_manager.calculate_indicators(df)
    
    return result_df


def test_strategy(strategy_name: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Belirli bir stratejiyi test eder ve sonuçları döndürür
    
    Args:
        strategy_name: Test edilecek strateji adı
        params: Strateji parametreleri
        
    Returns:
        Sinyal sütunları eklenmiş DataFrame
    """
    # Örnek veri oluştur
    df = generate_sample_data()
    
    # Strateji için veriyi hazırla
    print(f"\n>> {strategy_name} için veri hazırlanıyor...")
    indicator_df = prepare_data_for_strategy(df, strategy_name)
    
    # İlk birkaç satırı kontrol et - indikatörler doğru hesaplanmış mı
    print(f"\n>> İndikatör değerleri kontrol ediliyor...")
    debug_indicators(indicator_df, output_type="screen", sample_rows=3)
    
    # Strateji yöneticisi oluştur
    strategy_manager = StrategyManager(strategy_registry)
    
    # Stratejiyi ekle
    strategy_manager.add_strategy(strategy_name, params)
    
    try:
        # Stratejiyi çalıştır ve sinyaller oluştur
        print(f"\n>> {strategy_name} stratejisi çalıştırılıyor...")
        result_df = strategy_manager.generate_signals(indicator_df)
        
        # Sinyal özetini göster
        long_signals = result_df["long_signal"].sum()
        short_signals = result_df["short_signal"].sum()
        
        print(f"\n>> Sinyal Özeti:")
        print(f"   Long sinyaller: {long_signals}")
        print(f"   Short sinyaller: {short_signals}")
        print(f"   Toplam sinyal: {long_signals + short_signals}")
        print(f"   Veri noktası sayısı: {len(result_df)}")
        
        # Sinyalleri görselleştir
        visualize_strategy_signals(result_df, strategy_name)
        
        return result_df
    
    except Exception as e:
        print(f"\n❌ Strateji çalıştırılırken hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return indicator_df


def visualize_strategy_signals(df: pd.DataFrame, strategy_name: str):
    """
    Strateji sinyallerini görselleştirir
    
    Args:
        df: Sinyal sütunları içeren DataFrame
        strategy_name: Strateji adı
    """
    plt.figure(figsize=(14, 8))
    
    # Fiyat grafiği
    plt.subplot(211)
    plt.plot(df.index, df['close'], label='Kapanış Fiyatı', color='black')
    
    # Long ve short sinyalleri göster
    long_signals = df[df['long_signal']]
    short_signals = df[df['short_signal']]
    
    plt.scatter(long_signals.index, long_signals['close'], marker='^', color='green', s=100, label='Long Sinyal')
    plt.scatter(short_signals.index, short_signals['close'], marker='v', color='red', s=100, label='Short Sinyal')
    
    plt.title(f'{strategy_name} Stratejisi Sinyalleri')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    
    # Sinyal yoğunluğu grafiği
    plt.subplot(212)
    
    # Pencere içindeki sinyal sayılarını hesapla (5 günlük pencere)
    window_size = 5
    df['long_density'] = df['long_signal'].rolling(window=window_size).sum()
    df['short_density'] = df['short_signal'].rolling(window=window_size).sum()
    
    plt.plot(df.index, df['long_density'], label='Long Sinyal Yoğunluğu', color='green')
    plt.plot(df.index, df['short_density'], label='Short Sinyal Yoğunluğu', color='red')
    
    plt.ylabel('Sinyal Sayısı')
    plt.xlabel('Tarih')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_ensemble_strategy(strategy_name: str):
    """
    Ensemble stratejilerin test edilmesi için özel test fonksiyonu
    
    Args:
        strategy_name: Test edilecek ensemble strateji adı
    """
    # Örnek veri oluştur
    df = generate_sample_data()
    
    # Tüm indikatörleri hesapla (ensemble strateji tüm stratejileri kullanabilir)
    print(f"\n>> Ensemble strateji için tüm indikatörler hesaplanıyor...")
    
    # İndikatör yöneticisi oluştur
    indicator_manager = IndicatorManager(indicator_registry)
    
    # İlk olarak market_regime ve trend_strength için gerekli ADX indikatörlerini hesapla
    print("\n>> Önce ADX ve trend indikatörleri hesaplanıyor...")
    indicator_df = indicator_manager.calculate_indicators(df, ["trend_strength"])
    
    # Sonra rejim indikatörlerini hesapla
    print("\n>> Rejim indikatörleri hesaplanıyor...")
    indicator_df = indicator_manager.calculate_indicators(indicator_df, ["market_regime", "volatility_regime"])
    
    # Diğer gerekli indikatörleri ekle
    basic_indicators = [
        "ema", "sma", "rsi", "macd", "bollinger", "atr", "stochastic",
        "adaptive_rsi", "mtf_ema", "heikin_ashi", "supertrend", "ichimoku",
        "price_action", "volume_price", "momentum_features", "support_resistance"
    ]
    
    print("\n>> Diğer indikatörler hesaplanıyor...")
    for indicator_name in basic_indicators:
        indicator_manager.add_indicator(indicator_name)
    
    indicator_df = indicator_manager.calculate_indicators(indicator_df)
    
    # İndikatörleri kontrol et
    print("\n>> Hesaplanan indikatörler kontrol ediliyor...")
    debug_indicators(indicator_df, output_type="screen", sample_rows=2,
                    all_categories=False, specific_categories=["Regime"])
    
    # Market regime değerlerini kontrol et ve düzelt
    indicator_df = check_market_regime_values(indicator_df, fix_missing=True)
    
    # Strateji yöneticisi oluştur
    strategy_manager = StrategyManager(strategy_registry)
    
    # Stratejiyi ekle
    strategy_manager.add_strategy(strategy_name)
    
    try:
        # Stratejiyi çalıştır ve sinyaller oluştur
        print(f"\n>> {strategy_name} stratejisi çalıştırılıyor...")
        result_df = strategy_manager.generate_signals(indicator_df)
        
        # Sinyal özetini göster
        long_signals = result_df["long_signal"].sum()
        short_signals = result_df["short_signal"].sum()
        
        print(f"\n>> Sinyal Özeti:")
        print(f"   Long sinyaller: {long_signals}")
        print(f"   Short sinyaller: {short_signals}")
        print(f"   Toplam sinyal: {long_signals + short_signals}")
        print(f"   Veri noktası sayısı: {len(result_df)}")
        
        # Sinyalleri görselleştir
        visualize_strategy_signals(result_df, strategy_name)
        
        return result_df
    
    except Exception as e:
        print(f"\n❌ Strateji çalıştırılırken hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return indicator_df


def run_all_tests():
    """
    Tüm test fonksiyonlarını çalıştırır
    """
    print("=== STRATEJİLER TEST PROGRAMI ===")
    print(f"Registry'de {len(strategy_registry.get_all_strategies())} strateji bulundu.")
    
    # Kullanılabilir stratejileri görüntüle
    display_available_strategies()
    
    # Test menüsü
    while True:
        print("\n=== TEST MENÜSÜ ===")
        print("1. Tek strateji testi")
        print("2. Kategori bazlı test")
        print("3. Ensemble strateji testi")
        print("0. Çıkış")
        
        choice = input("\nSeçiminiz: ")
        
        if choice == "1":
            strategy_name = input("Strateji adı: ")
            if strategy_name in strategy_registry.get_all_strategies():
                param_input = input("Özel parametreler (JSON formatında, boş bırakabilirsiniz): ")
                params = {}
                if param_input.strip():
                    import json
                    try:
                        params = json.loads(param_input)
                    except:
                        print("Hatalı JSON formatı. Varsayılan parametreler kullanılacak.")
                
                if "ensemble" in strategy_registry.get_strategy(strategy_name).category:
                    test_ensemble_strategy(strategy_name)
                else:
                    test_strategy(strategy_name, params)
            else:
                print(f"❌ '{strategy_name}' adında bir strateji bulunamadı.")
        
        elif choice == "2":
            print("\nKategoriler:")
            categories = set(strat.category for strat in strategy_registry.get_all_strategies().values())
            for i, category in enumerate(categories, 1):
                print(f"{i}. {category}")
            
            cat_choice = input("\nTest edilecek kategori numarası: ")
            try:
                category = list(categories)[int(cat_choice)-1]
                print(f"\n>> {category.upper()} KATEGORİSİNDEKİ STRATEJİLER TESTİ")
                
                # Kategorideki stratejileri al
                strats = strategy_registry.get_strategies_by_category(category)
                
                for name in strats:
                    print(f"\n===== {name} Stratejisi Testi =====")
                    
                    # Ensemble için özel test
                    if "ensemble" in category:
                        test_ensemble_strategy(name)
                    else:
                        test_strategy(name)
            except (ValueError, IndexError):
                print("❌ Geçersiz kategori seçimi.")
        
        elif choice == "3":
            print("\nEnsemble Stratejiler:")
            ensemble_strats = strategy_registry.get_strategies_by_category("ensemble")
            for i, name in enumerate(ensemble_strats, 1):
                strat_class = strategy_registry.get_strategy(name)
                print(f"{i}. {strat_class.display_name} ({name})")
            
            if not ensemble_strats:
                print("❌ Hiç ensemble strateji bulunamadı.")
                continue
                
            es_choice = input("\nTest edilecek ensemble strateji numarası: ")
            try:
                strat_name = list(ensemble_strats.keys())[int(es_choice)-1]
                test_ensemble_strategy(strat_name)
            except (ValueError, IndexError):
                print("❌ Geçersiz strateji seçimi.")
        
        elif choice == "0":
            print("Test programı sonlandırılıyor...")
            break
        
        else:
            print("❌ Geçersiz seçim. Lütfen tekrar deneyin.")