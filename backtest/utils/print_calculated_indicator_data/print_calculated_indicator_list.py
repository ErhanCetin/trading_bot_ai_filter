"""
İndikatörleri kategorilere göre yazdırmak için yardımcı fonksiyonlar
"""
import pandas as pd
import os
from typing import List, Dict, Any, Optional, Union

def debug_indicators(df: pd.DataFrame, 
                    output_type: str = "screen", 
                    output_file: str = None,
                    sample_rows: int = 5,
                    all_categories: bool = True,
                    specific_categories: List[str] = None) -> None:
    """
    İndikatörleri kategorilere göre yazdırır veya CSV dosyasına kaydeder.
    
    Args:
        df: İndikatörlerin bulunduğu DataFrame
        output_type: Çıktı tipi, "screen" veya "csv" olabilir
        output_file: CSV çıktısı için dosya adı (None ise otomatik oluşturulur)
        sample_rows: Ekrana yazdırırken gösterilecek örnek satır sayısı
        all_categories: Tüm kategori gruplarını göster/dışa aktar
        specific_categories: Sadece belirli kategorileri göster/dışa aktar
    """
    # Temel fiyat sütunlarını belirle - bunlar indikatör değil
    base_columns = ['id', 'symbol', 'interval', 'open_time', 'open', 'high', 'low', 'close', 'volume', 
                   'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 
                   'taker_buy_quote_volume', 'open_time_dt']
    
    # İndikatör kategorilerini ve eşleşen sütunları tanımla
    indicator_categories = {
        "Trend": [col for col in df.columns if any(x in col.lower() for x in 
                ['ema', 'sma', 'trend', 'supertrend', 'ichimoku'])],
        
        "Momentum": [col for col in df.columns if any(x in col.lower() for x in 
                   ['rsi', 'macd', 'stoch', 'momentum', 'cci', 'mfi'])],
        
        "Volatility": [col for col in df.columns if any(x in col.lower() for x in 
                     ['bollinger', 'atr', 'volatility', 'std', 'keltner'])],
        
        "Price Action": [col for col in df.columns if any(x in col.lower() for x in 
                       ['body', 'shadow', 'pattern', 'hammer', 'doji', 'engulfing', 'ha_'])],
        
        "Volume": [col for col in df.columns if any(x in col.lower() for x in 
                 ['volume', 'obv', 'price_volume', 'positive_volume', 'negative_volume'])],
        
        "Regime": [col for col in df.columns if any(x in col.lower() for x in 
                 ['regime', 'market_regime', 'volatility_regime'])],
        
        "Statistical": [col for col in df.columns if any(x in col.lower() for x in 
                      ['zscore', 'reg_', 'linear', 'percentile', 'deviation'])],
        
        "Support/Resistance": [col for col in df.columns if any(x in col.lower() for x in 
                            ['support', 'resistance', 'pivot'])]
    }
    
    # Kategorileri filtrele (eğer specific_categories belirtilmişse)
    if not all_categories and specific_categories:
        indicator_categories = {k: v for k, v in indicator_categories.items() 
                              if k in specific_categories}
    
    # Hiçbir kategoriye girmeyen indikatörleri bul
    all_assigned = []
    for indicators in indicator_categories.values():
        all_assigned.extend(indicators)
    
    other_indicators = [col for col in df.columns 
                      if col not in base_columns and col not in all_assigned]
    
    if other_indicators:
        indicator_categories["Other"] = other_indicators
    
    # CSV çıktısı için dosya adını belirle
    if output_type.lower() == "csv" and not output_file:
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "unknown"
        interval = df['interval'].iloc[0] if 'interval' in df.columns else "unknown"
        output_file = f"{symbol}_{interval}_indicators.csv"
    
    # Çıktı tipine göre işlem yap
    if output_type.lower() == "screen":
        _print_indicators_to_screen(df, indicator_categories, sample_rows)
    elif output_type.lower() == "csv":
        _save_indicators_to_csv(df, indicator_categories, output_file)
    else:
        print(f"Hata: Geçersiz output_type: {output_type}. 'screen' veya 'csv' olmalıdır.")


def _print_indicators_to_screen(df: pd.DataFrame, 
                               indicator_categories: Dict[str, List[str]], 
                               sample_rows: int = 5) -> None:
    """
    İndikatörleri kategorilere göre ekrana yazdırır.
    
    Args:
        df: İndikatörlerin bulunduğu DataFrame
        indicator_categories: Kategori adları ve sütun listelerini içeren sözlük
        sample_rows: Gösterilecek örnek satır sayısı
    """
    # Toplam indikatör sayısını hesapla
    total_indicators = sum(len(indicators) for indicators in indicator_categories.values())
    print(f"\n=== İNDİKATÖR DEĞERLERİ (Toplam {total_indicators} indikatör) ===\n")
    
    # Her kategori için yazdır
    for category, indicators in indicator_categories.items():
        if not indicators:
            continue
        
        print(f"\n== {category} İndikatörleri (toplam {len(indicators)}) ==")
        
        # Bu kategoriden en fazla 10 indikatör göster
        display_indicators = indicators[:10]
        
        try:
            # Sadece sayısal sütunları yazdır (hata önleme)
            numeric_indicators = [ind for ind in display_indicators 
                                if pd.api.types.is_numeric_dtype(df[ind])]
            
            # Tarih sütunu varsa ekle
            time_col = 'open_time' if 'open_time' in df.columns else None
            
            # Bir seferde en fazla 5 sütun yazdır (okunabilirlik için)
            for i in range(0, len(numeric_indicators), 5):
                show_cols = numeric_indicators[i:i+5]
                if time_col:
                    show_cols = [time_col] + show_cols
                
                print(f"\nGrup {i//5 + 1}:")
                print(df[show_cols].head(sample_rows))
            
            # Kalan indikatörleri listele
            if len(indicators) > 10:
                print(f"\n...ve {len(indicators) - 10} indikatör daha:")
                print(", ".join(indicators[10:]))
                
        except Exception as e:
            print(f"Hata: {e}")
    
    print("\n=== İNDİKATÖR ÖZET BİLGİLERİ ===\n")
    
    # Her kategori için özet istatistikler göster
    for category, indicators in indicator_categories.items():
        if not indicators:
            continue
        
        print(f"\n== {category} İndikatörleri Özet İstatistikleri ==")
        
        # İlk 5 indikatör için istatistik göster
        for col in indicators[:5]:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats = df[col].describe()
                    non_null = df[col].count()
                    null_pct = (len(df) - non_null) / len(df) * 100 if len(df) > 0 else 0
                    
                    print(f"{col}:")
                    print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Ort: {stats['mean']:.4f}")
                    print(f"  25%: {stats['25%']:.4f}, 50%: {stats['50%']:.4f}, 75%: {stats['75%']:.4f}")
                    print(f"  Null: {null_pct:.1f}% ({len(df) - non_null}/{len(df)})")
                else:
                    print(f"{col}: Sayısal olmayan değerler içeriyor")
            except Exception as e:
                print(f"{col}: İstatistik hesaplanamadı - {e}")


def _save_indicators_to_csv(df: pd.DataFrame, 
                          indicator_categories: Dict[str, List[str]], 
                          output_file: str) -> None:
    """
    İndikatörleri kategorilere göre CSV dosyalarına kaydeder.
    
    Args:
        df: İndikatörlerin bulunduğu DataFrame
        indicator_categories: Kategori adları ve sütun listelerini içeren sözlük
        output_file: Ana CSV dosyasının adı
    """
    # Çıktı dizini ve temel dosya adı
       # Mevcut çalışma dizinini kullan veya belirtilen dizini al
    if os.path.dirname(output_file):
        output_dir = os.path.dirname(output_file)
    else:
        # Mevcut çalışma dizinini (script'in çalıştığı dizini) al
        output_dir = os.path.dirname(os.path.abspath(__file__))
    base_filename = os.path.splitext(os.path.basename(output_file))[0]
    
    # Ana CSV dosyasını oluştur (tüm indikatörler)
    all_indicators = []
    for indicators in indicator_categories.values():
        all_indicators.extend(indicators)
    
    # Temel sütunlar (hep aynı kalsın)
    base_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    base_cols = [col for col in base_cols if col in df.columns]
    
    # Ana dosyayı kaydet
    df[base_cols + all_indicators].to_csv(output_file, index=False)
    print(f"Tüm indikatörler '{output_file}' dosyasına kaydedildi.")
    
    # Her kategori için ayrı CSV dosyası oluştur
    for category, indicators in indicator_categories.items():
        if not indicators:
            continue
        
        category_file = os.path.join(output_dir, f"{base_filename}_{category.lower().replace('/', '_')}.csv")
        df[base_cols + indicators].to_csv(category_file, index=False)
        print(f"{category} indikatörleri '{category_file}' dosyasına kaydedildi.")
    
    # İndikatör istatistikleri dosyası oluştur
    stats_data = []
    for category, indicators in indicator_categories.items():
        for col in indicators:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    non_null = df[col].count()
                    null_pct = (len(df) - non_null) / len(df) * 100 if len(df) > 0 else 0
                    
                    stats_dict = {
                        "category": category,
                        "indicator": col,
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "std": df[col].std(),
                        "null_percent": null_pct
                    }
                    stats_data.append(stats_dict)
            except:
                # İstatistik hesaplanamayan sütunları atla
                pass
    
    # İstatistik dosyasını kaydet
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_file = os.path.join(output_dir, f"{base_filename}_statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"İndikatör istatistikleri '{stats_file}' dosyasına kaydedildi.")


# Kullanım örneği
if __name__ == "__main__":
    # Test için örnek veri oluştur
    import numpy as np
    
    data = {
        'symbol': ['BTCUSDT'] * 100,
        'interval': ['1h'] * 100,
        'open_time': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'open': np.random.randn(100) * 10 + 100,
        'high': np.random.randn(100) * 10 + 105,
        'low': np.random.randn(100) * 10 + 95,
        'close': np.random.randn(100) * 10 + 102,
        'volume': np.random.randn(100) * 1000 + 5000,
        
        # Örnek indikatörler
        'ema_9': np.random.randn(100) * 5 + 100,
        'ema_21': np.random.randn(100) * 5 + 101,
        'rsi_14': np.random.randn(100) * 10 + 50,
        'macd_line': np.random.randn(100) * 2,
        'bollinger_upper': np.random.randn(100) * 5 + 110,
        'volume_ma': np.random.randn(100) * 500 + 5000,
        'market_regime': np.random.choice(['uptrend', 'downtrend', 'ranging'], 100),
        'body_size': np.random.randn(100) * 2 + 3,
    }
    
    test_df = pd.DataFrame(data)
    
    # Test: Ekrana yazdır
    debug_indicators(test_df, output_type="screen", sample_rows=3)
    
    # Test: CSV dosyasına kaydet
    debug_indicators(test_df, output_type="csv", output_file="test_indicators.csv")

        # Sadece belirli kategorileri göster
    debug_indicators(
        test_df, 
        all_categories=False, 
        specific_categories=["Trend", "Momentum", "Volatility"]
    )

    # Sadece Volatilite ve Rejim indikatörlerini göster
    debug_indicators(
        test_df,
        output_type="screen",
        all_categories=False,
        specific_categories=["Volatility", "Regime"]

        # Varsayılan dosya adıyla CSV kaydet
    debug_indicators(test_df, output_type="csv")

    # Özel dosya adıyla CSV kaydet
    debug_indicators(test_df, output_type="csv", output_file="indikatör_analizi.csv")

    # Belirli dizine kaydet
    debug_indicators(test_df, output_type="csv", output_file="/home/user/analysis/btc_indicators.csv")

    # Özel satır sayısıyla istatistik için (CSV'ye kaydederken örnek satır sayısı kullanılmaz)
    debug_indicators(test_df, output_type="csv", sample_rows=10)    
    # Sadece Trend ve Momentum indikatörlerini CSV'ye kaydet
    debug_indicators(
        test_df,
        output_type="csv",
        output_file="trend_momentum_analysis.csv",
        all_categories=False,
        specific_categories=["Trend", "Momentum"]
)
)