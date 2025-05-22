"""
Signal verileri debug modülü
"""
import pandas as pd
import os
from typing import List, Dict, Any, Optional, Union

def debug_signals(df: pd.DataFrame, 
                  output_type: str = "screen", 
                  output_file: str = None,
                  sample_rows: int = 5,
                  all_signal_types: bool = True,
                  specific_signal_types: List[str] = None) -> None:
    """
    Strateji sinyallerini kategorilere göre yazdırır veya CSV dosyasına kaydeder.
    
    Args:
        df: Sinyallerin bulunduğu DataFrame
        output_type: Çıktı tipi, "screen" veya "csv" olabilir
        output_file: CSV çıktısı için dosya adı (None ise otomatik oluşturulur)
        sample_rows: Ekrana yazdırırken gösterilecek örnek satır sayısı
        all_signal_types: Tüm sinyal tiplerini göster/dışa aktar
        specific_signal_types: Sadece belirli sinyal tiplerini göster/dışa aktar
    """
    # Temel fiyat sütunlarını belirle - bunlar sinyal değil
    base_columns = ['id', 'symbol', 'interval', 'open_time', 'open', 'high', 'low', 'close', 'volume', 
                   'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 
                   'taker_buy_quote_volume', 'open_time_dt']
    
    # Sinyal kategorilerini ve eşleşen sütunları tanımla
    signal_categories = {
        "Main Signals": [col for col in df.columns if any(x in col.lower() for x in 
                        ['long_signal', 'short_signal', 'strategy_name'])],
        
        "Trend Signals": [col for col in df.columns if any(x in col.lower() for x in 
                         ['trend_following', 'mtf_trend', 'adaptive_trend', 'trend_signal'])],
        
        "Reversal Signals": [col for col in df.columns if any(x in col.lower() for x in 
                            ['reversal', 'overextended', 'pattern_reversal', 'divergence'])],
        
        "Breakout Signals": [col for col in df.columns if any(x in col.lower() for x in 
                             ['breakout', 'volatility_breakout', 'range_breakout', 'sr_breakout'])],
        
        "Ensemble Signals": [col for col in df.columns if any(x in col.lower() for x in 
                            ['ensemble', 'weighted_voting', 'regime_ensemble', 'adaptive_ensemble'])],
        
        "Signal Strengths": [col for col in df.columns if any(x in col.lower() for x in 
                            ['signal_strength', 'strength', 'confidence'])],
        
        "Signal Filters": [col for col in df.columns if any(x in col.lower() for x in 
                          ['filter', 'signal_passed', 'validation'])]
    }
    
    # Kategorileri filtrele (eğer specific_signal_types belirtilmişse)
    if not all_signal_types and specific_signal_types:
        signal_categories = {k: v for k, v in signal_categories.items() 
                            if k in specific_signal_types}
    
    # Hiçbir kategoriye girmeyen sinyalleri bul
    all_assigned = []
    for signals in signal_categories.values():
        all_assigned.extend(signals)
    
    other_signals = [col for col in df.columns 
                    if col not in base_columns and col not in all_assigned and 
                    ('signal' in col.lower() or 'strategy' in col.lower())]
    
    if other_signals:
        signal_categories["Other Signals"] = other_signals
    
    # CSV çıktısı için dosya adını belirle
    if output_type.lower() == "csv" and not output_file:
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "unknown"
        interval = df['interval'].iloc[0] if 'interval' in df.columns else "unknown"
        output_file = f"{symbol}_{interval}_signals.csv"
    
    # Çıktı tipine göre işlem yap
    if output_type.lower() == "screen":
        _print_signals_to_screen(df, signal_categories, sample_rows)
    elif output_type.lower() == "csv":
        _save_signals_to_csv(df, signal_categories, output_file)
    else:
        print(f"Hata: Geçersiz output_type: {output_type}. 'screen' veya 'csv' olmalıdır.")


def _print_signals_to_screen(df: pd.DataFrame, 
                            signal_categories: Dict[str, List[str]], 
                            sample_rows: int = 5) -> None:
    """
    Sinyalleri kategorilere göre ekrana yazdırır.
    
    Args:
        df: Sinyallerin bulunduğu DataFrame
        signal_categories: Kategori adları ve sütun listelerini içeren sözlük
        sample_rows: Gösterilecek örnek satır sayısı
    """
    # Toplam sinyal sayısını hesapla
    total_signals = sum(len(signals) for signals in signal_categories.values())
    print(f"\n=== SİNYAL DEĞERLERİ (Toplam {total_signals} sinyal sütunu) ===\n")
    
    # Long ve short sinyal sayısını raporla
    if 'long_signal' in df.columns and 'short_signal' in df.columns:
        long_count = df['long_signal'].sum()
        short_count = df['short_signal'].sum()
        total_count = long_count + short_count
        print(f"Toplam {total_count} sinyal bulundu: {long_count} LONG, {short_count} SHORT")
    
    # Ana sinyalleri göster
    if 'long_signal' in df.columns or 'short_signal' in df.columns:
        print("\n== Sinyal İçeren Satırlar ==")
        signal_rows = df[(df.get('long_signal', False) | df.get('short_signal', False))]
        
        # Gösterilecek sütunları belirle
        display_cols = ['open_time'] if 'open_time' in df.columns else []
        display_cols.extend(['close', 'long_signal', 'short_signal', 'strategy_name'])
        display_cols.extend([col for col in ['signal_strength', 'signal_passed_filter'] 
                           if col in df.columns])
        
        # Mevcut sütunları filtrele
        display_cols = [col for col in display_cols if col in df.columns]
        
        # İlk birkaç sinyal satırını göster
        if not signal_rows.empty:
            print(signal_rows[display_cols].head(sample_rows))
            if len(signal_rows) > sample_rows:
                print(f"...ve {len(signal_rows) - sample_rows} sinyal daha.")
        else:
            print("Sinyal içeren satır bulunamadı.")
    
    # Her kategori için yazdır
    for category, signals in signal_categories.items():
        if not signals:
            continue
        
        print(f"\n== {category} (toplam {len(signals)}) ==")
        
        # Bu kategoriden en fazla 10 sinyal göster
        display_signals = signals[:10]
        
        try:
            # Tarih sütunu varsa ekle
            time_col = 'open_time' if 'open_time' in df.columns else None
            
            # Boolean ve kategorik sinyalleri sayısal hale getir
            display_df = df.copy()
            for col in display_signals:
                if pd.api.types.is_bool_dtype(display_df[col]):
                    display_df[col] = display_df[col].astype(int)
            
            # Bir seferde en fazla 5 sütun yazdır (okunabilirlik için)
            for i in range(0, len(display_signals), 5):
                show_cols = display_signals[i:i+5]
                if time_col:
                    show_cols = [time_col] + show_cols
                
                print(f"\nGrup {i//5 + 1}:")
                print(display_df[show_cols].head(sample_rows))
            
            # Kalan sinyalleri listele
            if len(signals) > 10:
                print(f"\n...ve {len(signals) - 10} sinyal sütunu daha:")
                print(", ".join(signals[10:]))
                
        except Exception as e:
            print(f"Hata: {e}")
    
    print("\n=== SİNYAL ÖZET BİLGİLERİ ===\n")
    
    # Boolean sinyal sütunları için özet istatistikler göster
    boolean_signals = [col for col in df.columns 
                     if pd.api.types.is_bool_dtype(df[col]) and col in 
                     sum(signal_categories.values(), [])]
    
    if boolean_signals:
        print("\n== Boolean Sinyal İstatistikleri ==")
        for col in boolean_signals:
            try:
                true_count = df[col].sum()
                total_count = len(df)
                true_pct = (true_count / total_count) * 100 if total_count > 0 else 0
                
                print(f"{col}:")
                print(f"  True sayısı: {true_count}/{total_count} ({true_pct:.2f}%)")
            except Exception as e:
                print(f"{col}: İstatistik hesaplanamadı - {e}")
    
    # Stratejilere göre sinyal dağılımı
    if 'strategy_name' in df.columns and ('long_signal' in df.columns or 'short_signal' in df.columns):
        print("\n== Stratejilere Göre Sinyal Dağılımı ==")
        try:
            # Stratejilere göre sinyal sayıları
            if 'long_signal' in df.columns:
                long_by_strategy = df[df['long_signal']]['strategy_name'].value_counts()
                if not long_by_strategy.empty:
                    print("LONG Sinyaller:")
                    for strategy, count in long_by_strategy.items():
                        print(f"  {strategy}: {count}")
            
            if 'short_signal' in df.columns:
                short_by_strategy = df[df['short_signal']]['strategy_name'].value_counts()
                if not short_by_strategy.empty:
                    print("SHORT Sinyaller:")
                    for strategy, count in short_by_strategy.items():
                        print(f"  {strategy}: {count}")
        except Exception as e:
            print(f"Stratejilere göre dağılım hesaplanamadı: {e}")


def _save_signals_to_csv(df: pd.DataFrame, 
                        signal_categories: Dict[str, List[str]], 
                        output_file: str) -> None:
    """
    Sinyalleri kategorilere göre CSV dosyalarına kaydeder.
    
    Args:
        df: Sinyallerin bulunduğu DataFrame
        signal_categories: Kategori adları ve sütun listelerini içeren sözlük
        output_file: Ana CSV dosyasının adı
    """
    # Çıktı dizini ve temel dosya adı
    if os.path.dirname(output_file):
        output_dir = os.path.dirname(output_file)
    else:
        # Mevcut çalışma dizinini (script'in çalıştığı dizini) al
        output_dir = os.path.dirname(os.path.abspath(__file__))
    base_filename = os.path.splitext(os.path.basename(output_file))[0]
    
    # Ana CSV dosyasını oluştur (tüm sinyaller)
    all_signals = []
    for signals in signal_categories.values():
        all_signals.extend(signals)
    
    # Temel sütunlar (hep aynı kalsın)
    base_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    base_cols = [col for col in base_cols if col in df.columns]
    
    # Ana dosyayı kaydet
    df[base_cols + all_signals].to_csv(output_file, index=False)
    print(f"Tüm sinyal sütunları '{output_file}' dosyasına kaydedildi.")
    
    # Her kategori için ayrı CSV dosyası oluştur
    for category, signals in signal_categories.items():
        if not signals:
            continue
        
        category_file = os.path.join(output_dir, f"{base_filename}_{category.lower().replace('/', '_').replace(' ', '_')}.csv")
        df[base_cols + signals].to_csv(category_file, index=False)
        print(f"{category} sinyalleri '{category_file}' dosyasına kaydedildi.")
    
    # Sadece sinyal içeren satırları ayrı dosyaya kaydet
    if 'long_signal' in df.columns or 'short_signal' in df.columns:
        signal_rows = df[(df.get('long_signal', False) | df.get('short_signal', False))]
        if not signal_rows.empty:
            signal_file = os.path.join(output_dir, f"{base_filename}_only_signals.csv")
            signal_rows.to_csv(signal_file, index=False)
            print(f"Sadece sinyal içeren satırlar '{signal_file}' dosyasına kaydedildi.")
    
    # Stratejilere göre sinyal istatistikleri dosyası oluştur
    if 'strategy_name' in df.columns and ('long_signal' in df.columns or 'short_signal' in df.columns):
        try:
            # Stratejilere göre sinyal sayıları
            stats_data = []
            
            strategies = df['strategy_name'].dropna().unique()
            for strategy in strategies:
                strategy_df = df[df['strategy_name'] == strategy]
                long_count = strategy_df.get('long_signal', pd.Series([False])).sum()
                short_count = strategy_df.get('short_signal', pd.Series([False])).sum()
                total_count = long_count + short_count
                
                stats_dict = {
                    "strategy": strategy,
                    "long_signals": long_count,
                    "short_signals": short_count,
                    "total_signals": total_count,
                    "percent_of_all_signals": 0  # Daha sonra hesaplanacak
                }
                stats_data.append(stats_dict)
            
            # Yüzdeleri hesapla
            if stats_data:
                total_all_signals = sum(item["total_signals"] for item in stats_data)
                if total_all_signals > 0:
                    for item in stats_data:
                        item["percent_of_all_signals"] = (item["total_signals"] / total_all_signals) * 100
                
                # İstatistik dosyasını kaydet
                stats_df = pd.DataFrame(stats_data)
                stats_file = os.path.join(output_dir, f"{base_filename}_signal_statistics.csv")
                stats_df.to_csv(stats_file, index=False)
                print(f"Stratejilere göre sinyal istatistikleri '{stats_file}' dosyasına kaydedildi.")
        except Exception as e:
            print(f"Sinyal istatistikleri dosyası oluşturulamadı: {e}")