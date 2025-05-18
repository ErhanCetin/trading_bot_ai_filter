"""
Toplu backtest çalıştırma modülü
"""
import os
import pandas as pd
import json
import traceback
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from ..core.backtest_engine import BacktestEngine
from ..utils.data_loader import (
    load_price_data, 
    load_config_combinations, 
    transform_config_row,
    parse_indicators_config
)
from ..analysis.result_analyzer import analyze_batch_results
from ..analysis.result_plotter import plot_batch_results


def run_single_backtest(
    symbol: str,
    interval: str,
    config: Dict[str, Any],
    config_id: Union[int, str],
    db_url: str,
    backtest_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Tek bir backtest çalıştırır
    
    Args:
        symbol: İşlem sembolü
        interval: Zaman aralığı
        config: Konfigürasyon
        config_id: Konfigürasyon ID'si
        db_url: Veritabanı bağlantı URL'si
        backtest_params: Backtest parametreleri
        
    Returns:
        Backtest sonuçları
    """
    try:
        # Veriyi yükle
        df = load_price_data(symbol, interval, db_url)
        
        # Backtest motorunu oluştur
        engine = BacktestEngine(
            symbol=symbol,
            interval=interval,
            initial_balance=backtest_params.get('initial_balance', 10000.0),
            risk_per_trade=backtest_params.get('risk_per_trade', 0.01),
            sl_multiplier=backtest_params.get('sl_multiplier', 1.5),
            tp_multiplier=backtest_params.get('tp_multiplier', 3.0),
            leverage=backtest_params.get('leverage', 1.0),
            position_direction=backtest_params.get('position_direction', {"Long": True, "Short": True}),
            commission_rate=backtest_params.get('commission_rate', 0.001)
        )
        
        # Signal Engine bileşenlerini yapılandır
        engine.configure_signal_engine(
            indicators_config=config.get('indicators', {}),
            strategies_config=config.get('strategies', {}),
            strength_config=config.get('strength', {}),
            filter_config=config.get('filters', {})
        )
        
        # Backtest çalıştır
        result = engine.run(df, config_id=config_id)
        
        # Sonuçları config_id ile işaretle
        result['config_id'] = config_id
        result['config'] = config
        
        # Özet metrikleri yazdır
        if 'error' not in result:
            win_rate = result['metrics'].get('win_rate', 0) if 'metrics' in result else 0
            profit_loss = result.get('profit_loss', 0)
            roi_pct = result.get('roi_pct', 0)
            
            print(f"✅ Config {config_id} completed: "
                f"Trades {result['total_trades']}, "
                f"Win Rate {win_rate:.2f}%, "
                f"Profit ${profit_loss:.2f}, "
                f"ROI {roi_pct:.2f}%")
        else:
            print(f"❌ Config {config_id} failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    except Exception as e:
        print(f"❌ Error in backtest for config_id {config_id}: {e}")
        traceback.print_exc()
        return {
            'config_id': config_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_batch_backtest(
    symbol: str,
    interval: str,
    config_csv_path: str,
    db_url: str,
    output_dir: str,
    backtest_params: Dict[str, Any],
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Toplu backtest çalıştırır
    
    Args:
        symbol: İşlem sembolü
        interval: Zaman aralığı
        config_csv_path: Konfigürasyon CSV dosya yolu
        db_url: Veritabanı bağlantı URL'si
        output_dir: Çıktı dizini
        backtest_params: Backtest parametreleri
        max_workers: Maksimum process sayısı (None = CPU sayısı)
        
    Returns:
        Toplu backtest sonuçları 
    """
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Batch başlangıç zamanı
    batch_start_time = time.time()
    
    # Konfigurasyon kombinasyonlarını yükle
    config_df = load_config_combinations(config_csv_path)
    
    if config_df.empty:
        print("⚠️ No configurations found in CSV.")
        return {"status": "error", "message": "No configurations found."}
    
    # Konfigurasyon sayısını yazdır
    print(f"📊 Found {len(config_df)} configurations in CSV.")
    
    # Tüm sonuçları toplayacak liste
    all_results = []
    all_trades = []
    
    # Paralel işleme ile backtest çalıştır
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Her konfigürasyon için bir future oluştur
        futures = {}
        
        for i, row in config_df.iterrows():
            config_id = row.get("config_id", i)
            config = transform_config_row(row)
            
            # Future oluştur ve sakla
            future = executor.submit(
                run_single_backtest,
                symbol,
                interval,
                config,
                config_id,
                db_url,
                backtest_params
            )
            
            futures[future] = config_id
        
        # Tamamlanan işleri topla
        for future in as_completed(futures):
            config_id = futures[future]
            
            try:
                result = future.result()
                
                if 'error' in result:
                    print(f"❌ Config {config_id} failed: {result['error']}")
                    continue
                
                # Sonuçları sakla
                all_results.append(result)
                
                # Trade'leri sakla
                if 'trades' in result:
                    for trade in result['trades']:
                        trade['config_id'] = config_id
                    all_trades.extend(result['trades'])
            
            except Exception as e:
                print(f"❌ Error processing results for config_id {config_id}: {e}")
                traceback.print_exc()
    
    # Batch süresini hesapla
    batch_duration = time.time() - batch_start_time
    print(f"⏱️ Batch backtest completed in {batch_duration:.2f} seconds.")
    
    # Sonuçları analiz et
    if all_results:
        # Sonuçları DataFrame'e dönüştür
        valid_results = []
        for r in all_results:
            if 'metrics' in r:
                # Metrikler var, ancak bazı değerler eksik olabilir
                result_dict = {
                    'config_id': r['config_id'],
                    'total_trades': r['total_trades'],
                    'win_rate': r['metrics'].get('win_rate', 0),
                    'profit_loss': r.get('profit_loss', 0),
                    'roi_pct': r.get('roi_pct', 0),
                    'max_drawdown_pct': r['metrics'].get('max_drawdown_pct', 0),
                    'sharpe_ratio': r['metrics'].get('sharpe_ratio', 0),
                    'profit_factor': r['metrics'].get('profit_factor', 0)
                }
                valid_results.append(result_dict)

        results_df = pd.DataFrame(valid_results)
        # Trades DataFrame'ini oluştur
        trades_df = pd.DataFrame(all_trades)
        
        # CSV dosyalarına kaydet
        results_df.to_csv(os.path.join(output_dir, "batch_results.csv"), index=False)
        trades_df.to_csv(os.path.join(output_dir, "batch_trades.csv"), index=False)
        
        # Sonuçları analiz et
        analyze_batch_results(results_df, trades_df, output_dir)
        
        # Grafikleri oluştur
        plot_batch_results(results_df, trades_df, output_dir)
       
        print(f"✅ Batch results saved to {output_dir}")
        

        # En iyi sonuçları göster
        if not results_df.empty:
            best_roi = results_df.loc[results_df['roi_pct'].idxmax()]
            best_winrate = results_df.loc[results_df['win_rate'].idxmax()]
            
            print("\n🏆 Best ROI Configuration:")
            print(f"   Config ID: {best_roi['config_id']}")
            print(f"   ROI: {best_roi['roi_pct']:.2f}%")
            print(f"   Win Rate: {best_roi['win_rate']:.2f}%")
            print(f"   Total Trades: {best_roi['total_trades']}")
            
            print("\n🏆 Best Win Rate Configuration:")
            print(f"   Config ID: {best_winrate['config_id']}")
            print(f"   Win Rate: {best_winrate['win_rate']:.2f}%")
            print(f"   ROI: {best_winrate['roi_pct']:.2f}%")
            print(f"   Total Trades: {best_winrate['total_trades']}")
        else:
            print("\n⚠️ No valid configurations with trades found.")
        
        # Sonuçları döndür
        return {
            "status": "success",
            "total_configs": len(config_df),
            "completed_configs": len(all_results),
            "best_roi_config": best_roi['config_id'],
            "best_roi_pct": best_roi['roi_pct'],
            "best_winrate_config": best_winrate['config_id'],
            "best_winrate_pct": best_winrate['win_rate'],
            "results_path": os.path.join(output_dir, "batch_results.csv"),
            "trades_path": os.path.join(output_dir, "batch_trades.csv")
        }
    else:
        print("⚠️ No successful backtest results found.")
        return {
            "status": "warning",
            "message": "No successful backtest results found.",
            "total_configs": len(config_df),
            "completed_configs": 0
        }