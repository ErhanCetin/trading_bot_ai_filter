"""
İndikatörler modülünü test etmek için test scripti
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Optional, Union

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modül yolunu ayarla (signal_engine ve indikatör modüllerine erişim için)
# Geliştirme ortamınıza göre ayarlamanız gerekebilir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# İndikatörleri ve registry'yi içe aktar
from signal_engine.indicators import registry
from signal_engine.signal_indicator_plugin_system import IndicatorManager


def generate_sample_data(size: int = 1000, add_patterns: bool = True) -> pd.DataFrame:
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
    
    return df


def display_available_indicators():
    """
    Kullanılabilir tüm indikatörleri ve bilgilerini görüntüler
    """
    print("\n=== KULLANILABILIR İNDİKATÖRLER ===")
    
    all_indicators = registry.get_all_indicators()
    categories = {}
    
    # İndikatörleri kategorilere ayır
    for name, indicator_class in all_indicators.items():
        if indicator_class.category not in categories:
            categories[indicator_class.category] = []
        
        categories[indicator_class.category].append({
            'name': name,
            'display_name': indicator_class.display_name,
            'description': indicator_class.description,
            'default_params': indicator_class.default_params,
            'requires': indicator_class.requires_columns,
            'outputs': indicator_class.output_columns if hasattr(indicator_class, 'output_columns') else []
        })
    
    # Kategorilere göre yazdır
    for category, indicators in categories.items():
        print(f"\n== {category.upper()} İNDİKATÖRLERİ ==")
        for ind in indicators:
            print(f"- {ind['display_name']} ({ind['name']})")
            print(f"  Açıklama: {ind['description']}")
            print(f"  Gerekli sütunlar: {ind['requires']}")
            print(f"  Varsayılan parametreler:")
            for param, value in ind['default_params'].items():
                print(f"    - {param}: {value}")
            print("")


def test_indicator(indicator_name: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Belirli bir indikatörü test eder ve sonuçları döndürür
    
    Args:
        indicator_name: Test edilecek indikatör adı
        params: İndikatör parametreleri
        
    Returns:
        İndikatör hesaplanmış DataFrame
    """
    # Örnek veri oluştur
    df = generate_sample_data()
    
    # İndikatör yöneticisi oluştur
    manager = IndicatorManager(registry)
    
    # İndikatörü ekle
    manager.add_indicator(indicator_name, params)
    
    try:
        # İndikatörü hesapla
        result_df = manager.calculate_indicators(df)
        
        # Başarılı hesaplama mesajı
        indicator_class = registry.get_indicator(indicator_name)
        print(f"\n✅ {indicator_class.display_name} ({indicator_name}) başarıyla hesaplandı")
        print(f"   Parametreler: {params if params else indicator_class.default_params}")
        
        # Çıktı sütunlarını kontrol et
        indicator = registry.create_indicator(indicator_name, params)
        output_columns = indicator.output_columns
        
        # Eğer çıktı sütunları boşsa, dinamik olarak hesaplanıyordur
        # Bu durumda, result_df'te df'ten farklı olan sütunları bul
        if not output_columns:
            output_columns = [col for col in result_df.columns if col not in df.columns]
            print(f"   Dinamik çıktı sütunları: {output_columns}")
        else:
            print(f"   Çıktı sütunları: {output_columns}")
        
        # Çıktı özeti
        print("\n   Çıktı Özeti:")
        for col in output_columns:
            if col in result_df.columns:
                data = result_df[col].dropna()
                if len(data) > 0:
                    print(f"     - {col}: Min={data.min():.4f}, Max={data.max():.4f}, Ortalama={data.mean():.4f}")
                else:
                    print(f"     - {col}: Tüm değerler NaN")
            else:
                print(f"     - {col}: Sütun bulunamadı")
        
        return result_df
    
    except Exception as e:
        print(f"\n❌ {indicator_name} hesaplanırken hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return df  # Orijinal veriyi döndür


def plot_indicator_results(df: pd.DataFrame, indicator_name: str):
    """
    İndikatör sonuçlarını grafiksel olarak gösterir
    
    Args:
        df: İndikatör hesaplanmış DataFrame
        indicator_name: İndikatör adı
    """
    # İndikatör sınıfını al
    indicator_class = registry.get_indicator(indicator_name)
    if not indicator_class:
        print(f"❌ {indicator_name} indikatörü bulunamadı.")
        return
    
    # İndikatör çıktı sütunlarını al
    indicator = registry.create_indicator(indicator_name)
    output_columns = indicator.output_columns
    
    # Çıktı sütunları boşsa, df'teki fazladan sütunları bul
    if not output_columns:
        sample_df = generate_sample_data(size=10)  # Küçük bir örnek
        output_columns = [col for col in df.columns if col not in sample_df.columns]
    
    # Çıktı sütunu yoksa
    if not output_columns:
        print(f"❌ {indicator_name} için çıktı sütunları bulunamadı.")
        return
    
    # Grafik oluştur
    plt.figure(figsize=(14, 8))
    
    # Temel fiyat grafiği
    ax1 = plt.subplot(211)
    ax1.plot(df.index, df['close'], label='Kapanış Fiyatı', color='blue')
    ax1.set_title(f"{indicator_class.display_name} Testi")
    ax1.set_ylabel('Fiyat')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # İndikatör grafiği (düşük satır sayısı için)
    if len(output_columns) <= 4:
        ax2 = plt.subplot(212, sharex=ax1)
        
        # İndikatör kategorisine göre çizim stili seç
        if indicator_class.category in ['momentum', 'volatility']:
            # Oscillator tipi göstergeler
            for col in output_columns:
                if col in df.columns:
                    ax2.plot(df.index, df[col], label=col)
            
            # RSI-benzeri göstergeler için aşırı alım/satım bölgeleri
            if indicator_class.category == 'momentum':
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.3)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.3)
                ax2.set_ylim(0, 100)
        
        elif indicator_class.category == 'trend':
            # Trend göstergeleri için çizgiler
            for col in output_columns:
                if col in df.columns:
                    ax2.plot(df.index, df[col], label=col)
        
        else:
            # Diğer tüm göstergeler
            for col in output_columns:
                if col in df.columns:
                    ax2.plot(df.index, df[col], label=col)
        
        ax2.set_ylabel('İndikatör Değeri')
        ax2.legend(loc='upper left')
        ax2.grid(True)
    
    # Çok fazla çıktı sütunu varsa ayrı grafikler oluştur
    else:
        # Çıktı sütunlarını mantıklı gruplara ayır
        subplot_count = min(len(output_columns), 4)  # En fazla 4 alt grafik
        
        for i, col in enumerate(output_columns[:subplot_count]):
            if col in df.columns:
                ax = plt.subplot(2 + subplot_count, 1, i + 2, sharex=ax1)
                ax.plot(df.index, df[col], label=col, color=f'C{i}')
                ax.set_ylabel(col)
                ax.legend(loc='upper left')
                ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_indicators_by_category(category: str, sample_size: int = 500):
    """
    Belirli bir kategorideki tüm indikatörleri test eder
    
    Args:
        category: İndikatör kategorisi
        sample_size: Test için örnek veri noktası sayısı
    """
    print(f"\n=== {category.upper()} KATEGORİSİNDEKİ İNDİKATÖRLER TESTİ ===")
    
    # Kategorideki indikatörleri al
    indicators = registry.get_indicators_by_category(category)
    if not indicators:
        print(f"❌ '{category}' kategorisinde indikatör bulunamadı.")
        return
    
    # Her indikatörü test et
    results = {}
    for name, indicator_class in indicators.items():
        print(f"\n>> {indicator_class.display_name} ({name}) testi başlatılıyor...")
        
        # İndikatörü test et
        result_df = test_indicator(name)
        
        # Sonuçları sakla
        results[name] = result_df
        
        # Grafiği göster
        print(f"\n>> {indicator_class.display_name} grafiği oluşturuluyor...")
        plot_indicator_results(result_df, name)
    
    return results


def showcase_combination_indicators():
    """
    Birden fazla indikatörün birlikte kullanımını gösterir
    """
    print("\n=== BİRDEN FAZLA İNDİKATÖRÜN BİRLİKTE KULLANIMI ===")
    
    # Örnek veri oluştur
    df = generate_sample_data()
    
    # İndikatör yöneticisi oluştur
    manager = IndicatorManager(registry)
    
    # Birkaç popüler indikatörü ekle
    manager.add_indicator("ema", {"periods": [20, 50, 200]})
    manager.add_indicator("rsi", {"periods": [14]})
    manager.add_indicator("bollinger", {"window": 20, "window_dev": 2.0})
    manager.add_indicator("supertrend", {"atr_period": 10, "atr_multiplier": 3.0})
    
    try:
        # İndikatörleri hesapla
        result_df = manager.calculate_indicators(df)
        
        # Grafik oluştur
        plt.figure(figsize=(14, 12))
        
        # Fiyat grafiği
        ax1 = plt.subplot(411)
        ax1.plot(df.index, result_df['close'], label='Kapanış', color='black')
        
        # EMA çizgileri
        ax1.plot(df.index, result_df['ema_20'], label='EMA 20', color='blue')
        ax1.plot(df.index, result_df['ema_50'], label='EMA 50', color='orange')
        ax1.plot(df.index, result_df['ema_200'], label='EMA 200', color='red')
        
        # Bollinger Bantları
        ax1.plot(df.index, result_df['bollinger_upper'], label='BB Üst', color='green', linestyle='--')
        ax1.plot(df.index, result_df['bollinger_lower'], label='BB Alt', color='green', linestyle='--')
        
        ax1.set_title('Fiyat ve Trend İndikatörleri')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # RSI grafiği
        ax2 = plt.subplot(412, sharex=ax1)
        ax2.plot(df.index, result_df['rsi_14'], label='RSI 14', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_ylim(0, 100)
        ax2.set_title('RSI İndikatörü')
        ax2.grid(True)
        
        # Supertrend grafiği
        ax3 = plt.subplot(413, sharex=ax1)
        ax3.plot(df.index, result_df['close'], label='Fiyat', color='black', alpha=0.5)
        ax3.plot(df.index, result_df['supertrend'], label='Supertrend', color='blue')
        ax3.set_title('Supertrend İndikatörü')
        ax3.legend()
        ax3.grid(True)
        
        # Strateji sinyalleri (basit örnek)
        ax4 = plt.subplot(414, sharex=ax1)
        
        # Basit alım-satım sinyalleri oluştur (örnek amaçlı)
        # RSI aşırı alım/satım + EMA çapraz + Bollinger dokunuş
        buy_signals = ((result_df['rsi_14'] < 30) & 
                       (result_df['close'] > result_df['ema_50']) & 
                       (result_df['close'] < result_df['bollinger_lower'] * 1.02))
        
        sell_signals = ((result_df['rsi_14'] > 70) & 
                        (result_df['close'] < result_df['ema_50']) & 
                        (result_df['close'] > result_df['bollinger_upper'] * 0.98))
        
        # Sinyalleri göster
        ax4.plot(df.index, result_df['close'], label='Fiyat', color='black', alpha=0.5)
        ax4.scatter(df.index[buy_signals], result_df.loc[buy_signals, 'close'], 
                   color='green', label='Alım Sinyali', marker='^', s=100)
        ax4.scatter(df.index[sell_signals], result_df.loc[sell_signals, 'close'], 
                   color='red', label='Satım Sinyali', marker='v', s=100)
        
        ax4.set_title('Strateji Sinyalleri')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return result_df
    
    except Exception as e:
        print(f"\n❌ Kombinasyon indikatörleri hesaplanırken hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return df


def run_all_tests():
    """
    Tüm test fonksiyonlarını çalıştırır
    """
    print("=== İNDİKATÖRLER TEST PROGRAMI ===")
    print(f"Registry'de {len(registry.get_all_indicators())} indikatör bulundu.")
    
    # Kullanılabilir indikatörleri görüntüle
    display_available_indicators()
    
    # Test menüsü
    while True:
        print("\n=== TEST MENÜSÜ ===")
        print("1. Tek indikatör testi")
        print("2. Kategori bazlı test")
        print("3. Kombinasyon indikatörleri testi")
        print("0. Çıkış")
        
        choice = input("\nSeçiminiz: ")
        
        if choice == "1":
            indicator_name = input("İndikatör adı: ")
            if indicator_name in registry.get_all_indicators():
                result_df = test_indicator(indicator_name)
                plot_indicator_results(result_df, indicator_name)
            else:
                print(f"❌ '{indicator_name}' adında bir indikatör bulunamadı.")
        
        elif choice == "2":
            print("\nKategoriler:")
            categories = set(ind.category for ind in registry.get_all_indicators().values())
            for i, category in enumerate(categories, 1):
                print(f"{i}. {category}")
            
            cat_choice = input("\nTest edilecek kategori numarası: ")
            try:
                category = list(categories)[int(cat_choice)-1]
                test_indicators_by_category(category)
            except (ValueError, IndexError):
                print("❌ Geçersiz kategori seçimi.")
        
        elif choice == "3":
            showcase_combination_indicators()
        
        elif choice == "0":
            print("Test programı sonlandırılıyor...")
            break
        
        else:
            print("❌ Geçersiz seçim. Lütfen tekrar deneyin.")


# Test programını çalıştır
if __name__ == "__main__":
    run_all_tests()
