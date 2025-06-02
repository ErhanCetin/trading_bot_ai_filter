import sys
import os
import pandas as pd
import csv
import json
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_environment
load_environment()

from data.binance_fetch_historical import (
    fetch_kline,
    fetch_funding_rate,
    fetch_open_interest,
    fetch_long_short_ratio,
)
from db.writer import (
    insert_kline,
    insert_funding_rate,
    insert_open_interest,
    insert_long_short_ratio,
)

# Configuration settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
CONFIG_FILE = os.path.join(project_root, "backtest", "config", "config_combinations.csv")
DAYS = 2  # Number of days to fetch
DELAY_BETWEEN_REQUESTS = 1  # Seconds to wait between API calls to avoid rate limiting

def load_trading_configs(csv_file_path):
    """
    Load trading configurations from CSV file
    Returns list of unique (symbol, interval) pairs
    """
    configs = []
    unique_pairs = set()
    
    try:
        print(f"📂 CSV dosyası okunuyor: {os.path.abspath(csv_file_path)}")
        
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            # First, read a few lines to debug
            file.seek(0)
            first_lines = [file.readline().strip() for _ in range(3)]
            print(f"🔍 İlk 3 satır:")
            for i, line in enumerate(first_lines):
                print(f"   {i+1}: {line[:100]}...")
            
            # Reset file pointer and read with CSV reader
            file.seek(0)
            reader = csv.DictReader(file)
            
            print(f"📋 CSV Headers: {reader.fieldnames}")
            
            row_count = 0
            for row in reader:
                row_count += 1
                
                # Debug first few rows
                if row_count <= 3:
                    print(f"🔍 Row {row_count}: {dict(row)}")
                
                # Safe string extraction with None handling
                symbol_raw = row.get('symbol')
                interval_raw = row.get('interval')
                config_id_raw = row.get('config_id')
                
                # Convert None to empty string and strip
                symbol = str(symbol_raw).strip() if symbol_raw is not None else ''
                interval = str(interval_raw).strip() if interval_raw is not None else ''
                config_id = str(config_id_raw).strip() if config_id_raw is not None else 'unknown'
                
                # Debug raw vs processed values
                if row_count <= 3:
                    print(f"   Raw: symbol={symbol_raw}, interval={interval_raw}")
                    print(f"   Processed: symbol='{symbol}', interval='{interval}'")
                
                if symbol and interval and symbol != 'None' and interval != 'None':
                    pair = (symbol, interval)
                    if pair not in unique_pairs:
                        unique_pairs.add(pair)
                        configs.append({
                            'symbol': symbol,
                            'interval': interval,
                            'config_id': config_id
                        })
                        print(f"   ✅ Config eklendi: {symbol} - {interval}")
                else:
                    print(f"   ⚠️  Row {row_count}: Eksik/geçersiz veri - symbol='{symbol}', interval='{interval}'")
            
            print(f"📊 Toplam satır sayısı: {row_count}")
            print(f"📊 Unique config sayısı: {len(configs)}")
    
    except FileNotFoundError:
        print(f"❌ Config dosyası bulunamadı: {csv_file_path}")
        return []
    except Exception as e:
        print(f"❌ Config dosyası okunurken hata: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return configs

def fetch_and_store_for_config(symbol, interval, config_id=None):
    """
    Fetch and store all data types for a specific symbol/interval configuration
    """
    print(f"\n📥 {symbol} - {interval} için {DAYS} günlük veriler çekiliyor...")
    if config_id:
        print(f"   Config ID: {config_id}")
    
    success_count = 0
    errors = []
    
    # Fetch Kline data
    try:
        df_kline = fetch_kline(symbol, interval, days=DAYS)
        if len(df_kline) > 0:
            insert_kline(df_kline)
            print(f"   ✅ Kline verisi: {len(df_kline)} satır")
            success_count += 1
        else:
            print(f"   ⚠️  Kline verisi boş")
    except Exception as e:
        error_msg = f"Kline hatası: {e}"
        errors.append(error_msg)
        print(f"   ❌ {error_msg}")
    
    # Small delay to avoid rate limiting
    time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Fetch Funding Rate (only once per symbol since it doesn't depend on interval)
    try:
        df_fr = fetch_funding_rate(symbol, days=DAYS)
        if len(df_fr) > 0:
            insert_funding_rate(df_fr)
            print(f"   ✅ Funding rate: {len(df_fr)} satır")
            success_count += 1
        else:
            print(f"   ⚠️  Funding rate verisi boş")
    except Exception as e:
        error_msg = f"Funding rate hatası: {e}"
        errors.append(error_msg)
        print(f"   ❌ {error_msg}")
    
    time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Fetch Open Interest
    try:
        df_oi = fetch_open_interest(symbol, interval=interval, days=DAYS)
        if len(df_oi) > 0:
            insert_open_interest(df_oi)
            print(f"   ✅ Open Interest: {len(df_oi)} satır")
            success_count += 1
        else:
            print(f"   ⚠️  Open Interest verisi boş")
    except Exception as e:
        error_msg = f"Open Interest hatası: {e}"
        errors.append(error_msg)
        print(f"   ❌ {error_msg}")
    
    time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Fetch Long/Short Ratio
    try:
        df_lsr = fetch_long_short_ratio(symbol, interval=interval, days=DAYS)
        if len(df_lsr) > 0:
            insert_long_short_ratio(df_lsr)
            print(f"   ✅ Long/Short Ratio: {len(df_lsr)} satır")
            success_count += 1
        else:
            print(f"   ⚠️  Long/Short Ratio verisi boş")
    except Exception as e:
        error_msg = f"Long/Short Ratio hatası: {e}"
        errors.append(error_msg)
        print(f"   ❌ {error_msg}")
    
    return success_count, errors

def fetch_all_configs():
    """
    Main function to fetch data for all configurations in the CSV
    """
    print("🚀 Config-based toplu veri çekme başlatılıyor...")
    print(f"📁 Config dosyası: {CONFIG_FILE}")
    print(f"📅 Çekilecek gün sayısı: {DAYS}")
    print("=" * 60)
    
    # Load configurations
    configs = load_trading_configs(CONFIG_FILE)
    
    if not configs:
        print("❌ Hiç config bulunamadı. İşlem sonlandırılıyor.")
        return
    
    print(f"📊 Toplam {len(configs)} unique (symbol, interval) çifti bulundu:")
    for i, config in enumerate(configs, 1):
        print(f"   {i:2d}. {config['symbol']} - {config['interval']} ({config['config_id']})")
    
    print("\n" + "=" * 60)
    
    # Process each configuration
    total_configs = len(configs)
    successful_configs = 0
    failed_configs = 0
    all_errors = []
    
    start_time = datetime.now()
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total_configs}] İşleniyor: {config['symbol']} - {config['interval']}")
        
        try:
            success_count, errors = fetch_and_store_for_config(
                config['symbol'], 
                config['interval'], 
                config['config_id']
            )
            
            if errors:
                all_errors.extend([f"{config['symbol']}-{config['interval']}: {err}" for err in errors])
                failed_configs += 1
                print(f"   ⚠️  Kısmi başarı: {success_count}/4 veri türü başarılı")
            else:
                successful_configs += 1
                print(f"   🎉 Tüm veriler başarıyla kaydedildi ({success_count}/4)")
                
        except Exception as e:
            failed_configs += 1
            error_msg = f"{config['symbol']}-{config['interval']}: Genel hata - {e}"
            all_errors.append(error_msg)
            print(f"   ❌ Kritik hata: {e}")
        
        # Progress indicator
        progress = (i / total_configs) * 100
        print(f"   📈 İlerleme: {progress:.1f}% ({i}/{total_configs})")
        
        # Delay between different symbols to avoid rate limiting
        if i < total_configs:
            time.sleep(DELAY_BETWEEN_REQUESTS * 2)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 TOPLU VERİ ÇEKME RAPORU")
    print("=" * 60)
    print(f"🕐 Başlangıç: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Bitiş: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Toplam süre: {duration}")
    print(f"📈 Başarılı: {successful_configs}/{total_configs} config")
    print(f"❌ Başarısız: {failed_configs}/{total_configs} config")
    
    if all_errors:
        print(f"\n⚠️  HATALAR ({len(all_errors)} adet):")
        for error in all_errors:
            print(f"   • {error}")
    else:
        print("\n🎉 Hiç hata olmadan tamamlandı!")
    
    print("=" * 60)

def get_unique_symbols_intervals():
    """
    Utility function to get unique symbols and intervals from config
    """
    configs = load_trading_configs(CONFIG_FILE)
    
    symbols = set()
    intervals = set()
    
    for config in configs:
        symbols.add(config['symbol'])
        intervals.add(config['interval'])
    
    print("📊 Config'deki unique değerler:")
    print(f"💰 Symbols ({len(symbols)}): {sorted(symbols)}")
    print(f"⏰ Intervals ({len(intervals)}): {sorted(intervals)}")
    
    return sorted(symbols), sorted(intervals)

if __name__ == "__main__":
    # Check if user wants to see unique values only
    if len(sys.argv) > 1 and sys.argv[1] == "--show-unique":
        get_unique_symbols_intervals()
    else:
        fetch_all_configs()