"""
Sinyal gücü debug modülü
"""
import pandas as pd
import os
from typing import List, Dict, Any, Optional, Union

def debug_strength_values(df: pd.DataFrame, 
                         output_type: str = "screen", 
                         output_file: str = None,
                         sample_rows: int = 5) -> None:
    """
    Sinyal gücü değerlerini yazdırır veya CSV dosyasına kaydeder.
    
    Args:
        df: Sinyal gücü değerlerinin bulunduğu DataFrame
        output_type: Çıktı tipi, "screen" veya "csv" olabilir
        output_file: CSV çıktısı için dosya adı (None ise otomatik oluşturulur)
        sample_rows: Ekrana yazdırırken gösterilecek örnek satır sayısı
    """
    # Temel sütunları belirle
    base_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    base_columns = [col for col in base_columns if col in df.columns]
    
    # Sinyal sütunlarını bul
    signal_columns = ['long_signal', 'short_signal', 'strategy_name']
    signal_columns = [col for col in signal_columns if col in df.columns]
    
    # Güç sütunlarını bul
    strength_columns = [col for col in df.columns if 'strength' in col.lower() or 'gücü' in col.lower()]
    
    # Sinyal ve güç olan satırları filtrele
    signal_rows = df[(df.get('long_signal', False) | df.get('short_signal', False))]
    
    # CSV çıktısı için dosya adını belirle
    if output_type.lower() == "csv" and not output_file:
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "unknown"
        interval = df['interval'].iloc[0] if 'interval' in df.columns else "unknown"
        output_file = f"{symbol}_{interval}_strength_values.csv"
    
    # Ekrana yazdır
    if output_type.lower() == "screen":
        print("\n=== SİNYAL GÜCÜ DEĞERLERİ ===\n")
        
        # Toplam sinyal sayısı ve güç değerleri
        if 'signal_strength' in df.columns:
            total_signals = signal_rows.shape[0]
            avg_strength = signal_rows['signal_strength'].mean() if total_signals > 0 else 0
            print(f"Toplam {total_signals} sinyal, ortalama güç değeri: {avg_strength:.2f}/100")
            
            # Güç değer aralıklarını göster
            print("\nSinyal Gücü Dağılımı:")
            strength_bins = [(0, 20, "Çok zayıf"), (20, 40, "Zayıf"), (40, 60, "Orta"), 
                            (60, 80, "Güçlü"), (80, 100, "Çok güçlü")]
            
            for low, high, label in strength_bins:
                count = signal_rows[(signal_rows['signal_strength'] >= low) & 
                                    (signal_rows['signal_strength'] < high)].shape[0]
                pct = (count / total_signals * 100) if total_signals > 0 else 0
                print(f"  {label} ({low}-{high}): {count} sinyal ({pct:.1f}%)")
        
        # Güç hesaplayıcı sütunlarını göster
        other_strength_cols = [col for col in strength_columns if col != 'signal_strength']
        if other_strength_cols:
            print("\nDiğer Güç Değeri Sütunları:")
            for col in other_strength_cols:
                print(f"  - {col}")
        
        # Örnek sinyal satırlarını göster
        if not signal_rows.empty:
            print(f"\nÖrnek Sinyal Satırları (Toplam {signal_rows.shape[0]} sinyal):")
            display_cols = base_columns + signal_columns + strength_columns
            display_cols = [col for col in display_cols if col in df.columns]
            print(signal_rows[display_cols].head(sample_rows))
            
            if signal_rows.shape[0] > sample_rows:
                print(f"...ve {signal_rows.shape[0] - sample_rows} sinyal daha.")
        else:
            print("\nSinyal içeren satır bulunamadı.")
    
    # CSV dosyasına kaydet
    elif output_type.lower() == "csv":
        # Çıktı dizini kontrolü
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else os.path.dirname(os.path.abspath(__file__))
        base_filename = os.path.splitext(os.path.basename(output_file))[0]
        
        # Ana CSV dosyası - tüm veriyi içerir
        save_cols = base_columns + signal_columns + strength_columns
        save_cols = [col for col in save_cols if col in df.columns]
        df[save_cols].to_csv(output_file, index=False)
        print(f"Tüm sinyal gücü verileri '{output_file}' dosyasına kaydedildi.")
        
        # Sadece sinyal içeren satırları kaydet
        if not signal_rows.empty:
            signal_file = os.path.join(output_dir, f"{base_filename}_only_signals.csv")
            signal_rows[save_cols].to_csv(signal_file, index=False)
            print(f"Sadece sinyal içeren satırlar '{signal_file}' dosyasına kaydedildi.")
        
        # Stratejilere göre güç değerleri
        if 'strategy_name' in df.columns and 'signal_strength' in df.columns and not signal_rows.empty:
            # Stratejilere göre güç istatistikleri
            stats_data = []
            strategy_groups = signal_rows.groupby('strategy_name')
            
            for strategy, group in strategy_groups:
                stats_dict = {
                    "strategy": strategy,
                    "signal_count": len(group),
                    "avg_strength": group['signal_strength'].mean(),
                    "min_strength": group['signal_strength'].min(),
                    "max_strength": group['signal_strength'].max()
                }
                stats_data.append(stats_dict)
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_file = os.path.join(output_dir, f"{base_filename}_strength_by_strategy.csv")
                stats_df.to_csv(stats_file, index=False)
                print(f"Stratejilere göre güç istatistikleri '{stats_file}' dosyasına kaydedildi.")
    else:
        print(f"Hata: Geçersiz output_type: {output_type}. 'screen' veya 'csv' olmalıdır.")