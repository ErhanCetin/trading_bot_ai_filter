import sys
import os
import pandas as pd
import csv
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading

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

# FUTURES TRADING: Ultra-short data periods for current market conditions
# Focus on RECENT patterns only - leverage markets change rapidly
STRATEGY_DURATION = {
    'scalping': 3,        # 3 days - Recent micro-patterns only
    'swing': 5,           # 5 days - Very short-term trends  
    'momentum': 4,        # 4 days - Recent momentum only
    'mean_reversion': 5,  # 5 days - Fresh support/resistance
    'advanced': 7         # 7 days - Maximum for any futures strategy
}

# Binance Rate Limiting Optimization
MAX_CONCURRENT_REQUESTS = 8    # Parallel requests (conservative)
RATE_LIMIT_DELAY = 0.1        # Reduced delay between requests (100ms)
BATCH_SIZE = 4                # Requests per batch
INTER_BATCH_DELAY = 1.0       # Delay between batches (1 second)

def load_trading_configs(csv_file_path):
    """
    Load trading configurations from CSV file
    Returns list of unique (symbol, interval) pairs with debug information
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

def get_optimal_days(config_id, interval):
    """
    FUTURES OPTIMIZED: Ultra-short data periods for futures trading
    Futures markets change too rapidly for historical backtesting
    Focus on RECENT patterns that reflect current leverage environment
    """
    # Determine strategy type from config_id
    if 'scalping' in config_id:
        base_days = STRATEGY_DURATION['scalping']
    elif 'swing' in config_id:
        base_days = STRATEGY_DURATION['swing']
    elif 'momentum' in config_id:
        base_days = STRATEGY_DURATION['momentum']
    elif 'mean_reversion' in config_id:
        base_days = STRATEGY_DURATION['mean_reversion']
    elif 'advanced' in config_id:
        base_days = STRATEGY_DURATION['advanced']
    else:
        base_days = 5  # Conservative default for futures (1 week max)
    
    # For futures, we prioritize RECENCY over sample size
    # Better to have 100 recent relevant data points than 1000 stale ones
    
    # Minimal data point validation (lower threshold for futures)
    interval_points_per_day = {
        '1m': 1440,   # 24*60 = 1440 points per day
        '3m': 480,    # 24*20 = 480 points per day  
        '5m': 288,    # 24*12 = 288 points per day
        '15m': 96,    # 24*4 = 96 points per day
        '30m': 48,    # 24*2 = 48 points per day
        '1h': 24,     # 24 points per day
        '4h': 6       # 6 points per day
    }
    
    points_per_day = interval_points_per_day.get(interval, 24)
    expected_points = base_days * points_per_day
    
    # FUTURES: Lower minimum threshold, prioritize recency
    min_required_points = 50  # Much lower for futures (vs 200 for spot)
    if expected_points < min_required_points:
        # Only slightly increase, keep data recent
        min_days = max(base_days, int(min_required_points / points_per_day) + 1)
        optimal_days = min(min_days, base_days + 2)  # Max +2 days extension
        print(f"   📊 Minimal extension {config_id}: {base_days}→{optimal_days} days (RECENT focus)")
    else:
        optimal_days = base_days
    
    # STRICT caps for futures - never go beyond 10 days
    optimal_days = min(optimal_days, 10)   # Hard cap: 10 days max
    optimal_days = max(optimal_days, 2)    # Min 2 days (weekend protection)
    
    return optimal_days
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

# Rate limiting control
rate_limit_lock = threading.Lock()
last_request_time = 0

def rate_limited_request(func, *args, **kwargs):
    """
    Execute function with rate limiting protection
    """
    global last_request_time
    
    with rate_limit_lock:
        current_time = time.time()
        elapsed = current_time - last_request_time
        
        if elapsed < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - elapsed
            time.sleep(sleep_time)
        
        try:
            result = func(*args, **kwargs)
            last_request_time = time.time()
            return result
        except Exception as e:
            last_request_time = time.time()
            raise e

def fetch_and_store_for_config_parallel(symbol, interval, config_id=None):
    """
    Optimized parallel data fetching for a single config
    """
    optimal_days = get_optimal_days(config_id or '', interval)
    
    print(f"\n📥 {symbol} - {interval} için {optimal_days} günlük veriler çekiliyor (PARALLEL)...")
    if config_id:
        print(f"   Config ID: {config_id}")
        print(f"   📊 Strategy-optimized duration: {optimal_days} days")
    
    # Prepare all data fetching tasks
    fetch_tasks = [
        ('kline', lambda: rate_limited_request(fetch_kline, symbol, interval, days=optimal_days))
        # ('funding_rate', lambda: rate_limited_request(fetch_funding_rate, symbol, days=optimal_days)),
        # ('open_interest', lambda: rate_limited_request(fetch_open_interest, symbol, interval=interval, days=optimal_days)),
        # ('long_short_ratio', lambda: rate_limited_request(fetch_long_short_ratio, symbol, interval=interval, days=optimal_days))
    ]
    
    results = {}
    errors = []
    success_count = 0
    
    # Execute tasks in parallel with rate limiting
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(task_func): task_name 
            for task_name, task_func in fetch_tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            
            try:
                data = future.result()
                results[task_name] = data
                
                if len(data) > 0:
                    print(f"   ✅ {task_name}: {len(data)} satır")
                    success_count += 1
                else:
                    print(f"   ⚠️  {task_name}: Boş veri")
                    
            except Exception as e:
                error_msg = f"{task_name} hatası: {e}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
    
    # Store results in database (sequential for data integrity)
    if 'kline' in results and len(results['kline']) > 0:
        try:
            insert_kline(results['kline'])
        except Exception as e:
            print(f"   ❌ Kline DB insert hatası: {e}")
    
    if 'funding_rate' in results and len(results['funding_rate']) > 0:
        try:
            insert_funding_rate(results['funding_rate'])
        except Exception as e:
            print(f"   ❌ Funding rate DB insert hatası: {e}")
    
    if 'open_interest' in results and len(results['open_interest']) > 0:
        try:
            insert_open_interest(results['open_interest'])
        except Exception as e:
            print(f"   ❌ Open Interest DB insert hatası: {e}")
    
    if 'long_short_ratio' in results and len(results['long_short_ratio']) > 0:
        try:
            insert_long_short_ratio(results['long_short_ratio'])
        except Exception as e:
            print(f"   ❌ Long/Short Ratio DB insert hatası: {e}")
    
    return success_count, errors

def fetch_and_store_for_config(symbol, interval, config_id=None):
    """
    Fetch and store all data types for a specific symbol/interval configuration
    """
    # Calculate optimal days based on strategy and interval
    optimal_days = get_optimal_days(config_id or '', interval)
    
    print(f"\n📥 {symbol} - {interval} için {optimal_days} günlük veriler çekiliyor...")
    if config_id:
        print(f"   Config ID: {config_id}")
        print(f"   📊 Strategy-optimized duration: {optimal_days} days")
    
    success_count = 0
    errors = []
    
    # Fetch Kline data
    try:
        df_kline = fetch_kline(symbol, interval, days=optimal_days)
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
        df_fr = fetch_funding_rate(symbol, days=optimal_days)
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
        df_oi = fetch_open_interest(symbol, interval=interval, days=optimal_days)
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
        df_lsr = fetch_long_short_ratio(symbol, interval=interval, days=optimal_days)
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

def fetch_all_configs_parallel():
    """
    OPTIMIZED: Parallel batch fetching with intelligent rate limiting
    """
    print("🚀 FUTURES-FOCUSED Parallel veri çekme başlatılıyor...")
    print(f"📁 Config dosyası: {CONFIG_FILE}")
    print("📅 FUTURES: Ultra-short data duration (RECENCY over VOLUME):")
    for strategy, days in STRATEGY_DURATION.items():
        print(f"   • {strategy}: {days} days (futures-focused)")
    print(f"💡 Vadeli İşlem Mantığı: Son günlerin pattern'leri > Eski data volume")
    print(f"⚡ Parallel Settings: {MAX_CONCURRENT_REQUESTS} concurrent, {RATE_LIMIT_DELAY}s delay")
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
    
    # Process configurations in optimized batches
    total_configs = len(configs)
    successful_configs = 0
    failed_configs = 0
    all_errors = []
    
    start_time = datetime.now()
    
    # Process in batches to respect rate limits
    for batch_start in range(0, total_configs, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_configs)
        batch_configs = configs[batch_start:batch_end]
        
        print(f"\n📦 Batch {batch_start//BATCH_SIZE + 1}: Processing configs {batch_start+1}-{batch_end}")
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(batch_configs))) as executor:
            future_to_config = {
                executor.submit(
                    fetch_and_store_for_config_parallel,
                    config['symbol'],
                    config['interval'],
                    config['config_id']
                ): config for config in batch_configs
            }
            
            # Collect batch results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                
                try:
                    success_count, errors = future.result()
                    
                    if errors:
                        all_errors.extend([f"{config['symbol']}-{config['interval']}: {err}" for err in errors])
                        failed_configs += 1
                        print(f"   ⚠️  {config['symbol']}-{config['interval']}: Kısmi başarı ({success_count}/4)")
                    else:
                        successful_configs += 1
                        print(f"   🎉 {config['symbol']}-{config['interval']}: Tamamlandı ({success_count}/4)")
                        
                except Exception as e:
                    failed_configs += 1
                    error_msg = f"{config['symbol']}-{config['interval']}: Kritik hata - {e}"
                    all_errors.append(error_msg)
                    print(f"   ❌ {error_msg}")
        
        # Inter-batch delay to prevent rate limiting
        if batch_end < total_configs:
            print(f"   ⏳ Batch tamamlandı. {INTER_BATCH_DELAY}s bekleniyor...")
            time.sleep(INTER_BATCH_DELAY)
        
        # Progress update
        progress = (batch_end / total_configs) * 100
        print(f"   📈 Toplam İlerleme: {progress:.1f}% ({batch_end}/{total_configs})")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 PARALLEL TOPLU VERİ ÇEKME RAPORU")
    print("=" * 60)
    print(f"🕐 Başlangıç: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Bitiş: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Toplam süre: {duration}")
    print(f"⚡ Hızlanma: ~{MAX_CONCURRENT_REQUESTS}x faster than sequential")
    print(f"📈 Başarılı: {successful_configs}/{total_configs} config")
    print(f"❌ Başarısız: {failed_configs}/{total_configs} config")
    
    if all_errors:
        print(f"\n⚠️  HATALAR ({len(all_errors)} adet):")
        for error in all_errors[:10]:  # Show first 10 errors
            print(f"   • {error}")
        if len(all_errors) > 10:
            print(f"   ... ve {len(all_errors)-10} adet daha")
    else:
        print("\n🎉 Hiç hata olmadan tamamlandı!")
    
    print("=" * 60)

def fetch_all_configs():
    """
    Main function to fetch data for all configurations in the CSV
    """
    print("🚀 Futures-optimized toplu veri çekme başlatılıyor...")
    print(f"📁 Config dosyası: {CONFIG_FILE}")
    print("📅 Futures trading optimized data duration:")
    for strategy, days in STRATEGY_DURATION.items():
        print(f"   • {strategy}: {days} days (futures-optimized)")
    print("   💡 Kısa süreli data: Recent market conditions ve volatility patterns")
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
            time.sleep(RATE_LIMIT_DELAY * 2)
    
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
    elif len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        fetch_all_configs_parallel()
    else:
        # Default to parallel for better performance
        fetch_all_configs_parallel()