"""
Toplu backtest çalıştırma modülü - single_backtest.py iyileştirmeleriyle güncellenmiş
"""
import os
import pandas as pd
import json
import traceback
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging

from ..core.backtest_engine import BacktestEngine
from ..utils.data_loader import load_price_data
from ..analysis.result_analyzer import analyze_batch_results, analyze_single_result
from ..analysis.result_plotter import plot_batch_results, plot_single_result
from ..utils.config_loader import load_env_config


# Logger ayarla
logger = logging.getLogger(__name__)

try:
    from data.enhanced_fetch_and_store_binance import fetch_all_configs_parallel
    ENHANCED_FETCH_AVAILABLE = True
    logger.info("✅ Enhanced fetch available")
except ImportError as e:
    ENHANCED_FETCH_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced fetch not available: {e}")

def load_config_combinations(csv_path: str) -> pd.DataFrame:
    """
    Enhanced CSV config loader with JSON parsing support
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        Konfigürasyon DataFrame'i
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded {len(df)} configurations from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading config CSV: {e}")
        return pd.DataFrame()


def parse_json_config(config_str: str) -> Dict[str, Any]:
    """
    Parse JSON string from CSV to dictionary
    
    Args:
        config_str: JSON string from CSV
        
    Returns:
        Parsed dictionary
    """
    try:
        if pd.isna(config_str) or config_str == "":
            return {}
        return json.loads(config_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"⚠️ JSON parse error: {e}, returning empty config")
        return {}


def transform_csv_row_to_config(row: pd.Series) -> Dict[str, Any]:
    """
    Transform CSV row to backtest configuration
    """
    print(f"🔍 CSV ROW DEBUG: Row index = {row.name}")
    print(f"🔍 CSV ROW DEBUG: Row keys = {row.keys().tolist()}")
    print(f"🔍 CSV ROW DEBUG: Row values = {row.values}")
    
    try:
        config = {
            "indicators": {
                "long": parse_json_config(row.get("indicators_long", "{}")),
                "short": parse_json_config(row.get("indicators_short", "{}"))
            },
            "strategies": parse_json_config(row.get("strategies", "{}")),
            "filters": parse_json_config(row.get("filters", "{}")),
            "strength": parse_json_config(row.get("strength", "{}"))
        }
        
        print(f"🔍 CSV ROW DEBUG: Parsed config = {config}")
        return config
        
    except Exception as e:
        print(f"❌ CSV ROW ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"indicators": {"long": {}, "short": {}}, "strategies": {}, "filters": {}, "strength": {}}



def run_single_backtest_worker(
    symbol: str,
    interval: str,
    config: Dict[str, Any],
    config_id: Union[int, str],
    db_url: str,
    backtest_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Worker function for single backtest execution
    Enhanced with single_backtest.py improvements
    """
    try:
        print(f"🔍 WORKER DEBUG: Config ID = {config_id}")
        print(f"🔍 WORKER DEBUG: Symbol = {symbol}, Interval = {interval}")
        print(f"🔍 WORKER DEBUG: Config = {config}")

        # ✅ TP/Commission filter status check
        if backtest_params.get('enable_tp_commission_filter', False):
            print(f"🔧 TP/Commission filter ENABLED for config {config_id}")
            print(f"   Min TP/Commission ratio: {backtest_params.get('min_tp_commission_ratio', 3.0)}x")
            print(f"   Min position size: ${backtest_params.get('min_position_size', 800.0)}")
        
        # Veriyi yükle
        print("🔍 WORKER DEBUG: Veri yükleniyor...")
        df = load_price_data(symbol, interval, db_url)
        print(f"🔍 WORKER DEBUG: Veri yüklendi. Shape: {df.shape}")
        print(f"🔍 WORKER DEBUG: Columns: {df.columns.tolist()}")
        
        if df.empty:
            print("❌ WORKER DEBUG: DataFrame boş!")
            return {
                'config_id': str(config_id),
                'error': 'Empty DataFrame',
                'total_trades': 0
            }
        
        if len(df) < 2:
            print(f"❌ WORKER DEBUG: Yetersiz veri: {len(df)} satır")
            return {
                'config_id': str(config_id),
                'error': f'Insufficient data: {len(df)} rows',
                'total_trades': 0
            }
        
        print("🔍 WORKER DEBUG: Backtest engine oluşturuluyor...")
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
            commission_rate=backtest_params.get('commission_rate', 0.001),
            # ✅ YENİ: TP/Commission filter parametreleri
            min_tp_commission_ratio=backtest_params.get('min_tp_commission_ratio', 3.0),
            max_commission_impact_pct=backtest_params.get('max_commission_impact_pct', 15.0),
            min_position_size=backtest_params.get('min_position_size', 800.0),
            min_net_rr_ratio=backtest_params.get('min_net_rr_ratio', 1.5),
            enable_tp_commission_filter=backtest_params.get('enable_tp_commission_filter', False)
        )
        
        print("🔍 WORKER DEBUG: Signal Engine yapılandırılıyor...")
        # Signal Engine bileşenlerini yapılandır
        engine.configure_signal_engine(
            indicators_config=config.get('indicators', {}),
            strategies_config=config.get('strategies', {}),
            strength_config=config.get('strength', {}),
            filter_config=config.get('filters', {})
        )
        
        print("🔍 WORKER DEBUG: Backtest çalıştırılıyor...")
        # Backtest çalıştır
        result = engine.run(df, config_id=str(config_id))
        
        print(f"🔍 WORKER DEBUG: Backtest tamamlandı. Result keys: {result.keys()}")
        
        # Sonuçları config_id ile işaretle
        result['config_id'] = str(config_id)
        result['config'] = config

        # ✅ TP/Commission filter sonuçlarını logla
        if 'filter_statistics' in result:
            stats = result['filter_statistics']
            print(f"🔧 Config {config_id} Filter Results:")
            print(f"   Total signals: {stats.get('total_signals', 0)}")
            print(f"   Filtered signals: {stats.get('filtered_signals', 0)}")
            print(f"   Filter efficiency: {stats.get('filter_efficiency_pct', 0):.1f}%")
        
        # Özet metrikleri yazdır (sadece başarılı sonuçlar için)
        if 'error' not in result and result.get('total_trades', 0) > 0:
            win_rate = result['metrics'].get('win_rate', 0) if 'metrics' in result else 0
            profit_loss = result.get('profit_loss', 0)
            roi_pct = result.get('roi_pct', 0)
            
            print(f"✅ Config {config_id} completed: "
                  f"Trades {result['total_trades']}, "
                  f"Win Rate {win_rate:.2f}%, "
                  f"Profit ${profit_loss:.2f}, "
                  f"ROI {roi_pct:.2f}%")
            # ✅ TP/Commission specific metrics
            if backtest_params.get('enable_tp_commission_filter', False) and 'trades' in result:
                trades = result['trades']
                tp_trades = [t for t in trades if t.get('outcome') == 'TP']
                if tp_trades:
                    avg_commission_impact = sum(t.get('commission_impact_pct', 0) for t in tp_trades) / len(tp_trades)
                    print(f"   Avg Commission Impact: {avg_commission_impact:.1f}%")
        elif result.get('total_trades', 0) == 0:
            print(f"⚠️ Config {config_id} completed with no trades")
            if backtest_params.get('enable_tp_commission_filter', False):
                print("   (Possible reason: All trades filtered by TP/Commission filter)")
        
        return result
    
    except Exception as e:
        print(f"❌ WORKER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_id': str(config_id),
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
    max_workers: Optional[int] = None,
    auto_fetch_data: bool = True  # ✅ YENİ PARAMETRE

) -> Dict[str, Any]:
    """
    Enhanced batch backtest runner with single_backtest.py improvements
    
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

       # ✅ DIREKT FUNCTION CALL - super simple
    if auto_fetch_data and ENHANCED_FETCH_AVAILABLE:
        logger.info("🌐 AUTO DATA FETCHING - Calling fetch_all_configs_parallel()...")
        
        try:
            # ✅ TEK FUNCTION CALL
            fetch_all_configs_parallel()
            logger.info("✅ Enhanced fetch completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Enhanced fetch failed: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("⚠️ Continuing with existing data...")
    
    elif auto_fetch_data and not ENHANCED_FETCH_AVAILABLE:
        logger.warning("⚠️ Auto fetch requested but enhanced_fetch not available")
        


     # ✅ TP/Commission filter status logging
    if backtest_params.get('enable_tp_commission_filter', False):
        logger.info("🔧 BATCH BACKTEST: TP/Commission filter ENABLED")
        logger.info(f"   Min TP/Commission ratio: {backtest_params.get('min_tp_commission_ratio', 3.0)}x")
        logger.info(f"   Max commission impact: {backtest_params.get('max_commission_impact_pct', 15.0)}%")
        logger.info(f"   Min position size: ${backtest_params.get('min_position_size', 800.0)}")
        logger.info("   Only profitable trades will be opened across all configurations")
    else:
        logger.warning("⚠️ BATCH BACKTEST: TP/Commission filter DISABLED")
    
    # Batch başlangıç zamanı
    batch_start_time = time.time()
    
    # Konfigurasyon kombinasyonlarını yükle
    config_df = load_config_combinations(config_csv_path)
    
    if config_df.empty:
        logger.warning("⚠️ No configurations found in CSV.")
        return {"status": "error", "message": "No configurations found."}
    
    # Konfigurasyon sayısını yazdır
    logger.info(f"📊 Found {len(config_df)} configurations in CSV.")
    
    # Tüm sonuçları toplayacak liste
    all_results = []
    all_trades = []
    # ✅ Filter statistics tracking
    total_filter_stats = {
        'total_signals': 0,
        'filtered_signals': 0,
        'configs_with_no_trades': 0,
        'configs_with_trades': 0
    }
    
    # Paralel işleme ile backtest çalıştır
    max_workers = max_workers or max(1, os.cpu_count() - 1)
    logger.info(f"🚀 Starting batch backtest with {max_workers} workers")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Her konfigürasyon için bir future oluştur
        futures = {}
        
        for i, row in config_df.iterrows():
            config_id = row.get("config_id", i)
            config = transform_csv_row_to_config(row)
            
            # CSV'den symbol ve interval bilgisini al (varsa)
            csv_symbol = row.get("symbol", symbol)
            csv_interval = row.get("interval", interval)
            
            # Future oluştur ve sakla
            future = executor.submit(
                run_single_backtest_worker,
                csv_symbol,
                csv_interval,
                config,
                config_id,
                db_url,
                backtest_params
            )
            
            futures[future] = config_id
        
        # Tamamlanan işleri topla
        completed_count = 0
        for future in as_completed(futures):
            config_id = futures[future]
            completed_count += 1
            
            try:
                result = future.result()
                
                if 'error' in result:
                    logger.error(f"❌ Config {config_id} failed: {result['error']}")
                    continue
                # ✅ Filter statistics toplama
                if 'filter_statistics' in result:
                    stats = result['filter_statistics']
                    total_filter_stats['total_signals'] += stats.get('total_signals', 0)
                    total_filter_stats['filtered_signals'] += stats.get('filtered_signals', 0)
                
                # Sonuçları sakla
                all_results.append(result)
                # Trade sayısı istatistikleri
                if result.get('total_trades', 0) > 0:
                    total_filter_stats['configs_with_trades'] += 1
                else:
                    total_filter_stats['configs_with_no_trades'] += 1
                
                # Trade'leri sakla
                if 'trades' in result and result['trades']:
                    for trade in result['trades']:
                        trade['config_id'] = str(config_id)
                    all_trades.extend(result['trades'])
                
                # Progress indicator
                if completed_count % 10 == 0 or completed_count == len(config_df):
                    logger.info(f"📈 Progress: {completed_count}/{len(config_df)} configurations completed")
            
            except Exception as e:
                logger.error(f"❌ Error processing results for config_id {config_id}: {e}")
    
    # Batch süresini hesapla
    batch_duration = time.time() - batch_start_time
    logger.info(f"⏱️ Batch backtest completed in {batch_duration:.2f} seconds.")
    logger.info(f"📊 Successfully completed {len(all_results)} out of {len(config_df)} configurations")
      # ✅ ENHANCED: Filter statistics summary
    if backtest_params.get('enable_tp_commission_filter', False):
        logger.info(f"\n🔧 TP/COMMISSION FILTER SUMMARY:")
        logger.info(f"   Total signals across all configs: {total_filter_stats['total_signals']}")
        logger.info(f"   Filtered signals: {total_filter_stats['filtered_signals']}")
        if total_filter_stats['total_signals'] > 0:
            filter_efficiency = (total_filter_stats['filtered_signals'] / total_filter_stats['total_signals']) * 100
            logger.info(f"   Overall filter efficiency: {filter_efficiency:.1f}%")
        logger.info(f"   Configs with trades: {total_filter_stats['configs_with_trades']}")
        logger.info(f"   Configs with no trades: {total_filter_stats['configs_with_no_trades']}")
    
    # Sonuçları analiz et ve kaydet
    if all_results:
        # Sonuçları DataFrame'e dönüştür
        valid_results = []
        for r in all_results:
            if 'metrics' in r:
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
                 # ✅ TP/Commission specific metrics
                if backtest_params.get('enable_tp_commission_filter', False) and 'filter_statistics' in r:
                    stats = r['filter_statistics']
                    result_dict.update({
                        'filter_efficiency_pct': stats.get('filter_efficiency_pct', 0),
                        'total_signals': stats.get('total_signals', 0),
                        'filtered_signals': stats.get('filtered_signals', 0)
                    })
                valid_results.append(result_dict)

        if valid_results:
            results_df = pd.DataFrame(valid_results)
            trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
            
            # CSV dosyalarına kaydet
            results_path = os.path.join(output_dir, "batch_results.csv")
            trades_path = os.path.join(output_dir, "batch_trades.csv")
            
            results_df.to_csv(results_path, index=False)
            if not trades_df.empty:
                trades_df.to_csv(trades_path, index=False)
            
            logger.info(f"💾 Results saved to {results_path}")
            logger.info(f"💾 Trades saved to {trades_path}")
            
            # Sonuçları analiz et
            try:
                analyze_batch_results(results_df, trades_df, output_dir)
                logger.info("📊 Batch analysis completed")
            except Exception as e:
                logger.error(f"⚠️ Error in batch analysis: {e}")
            
            # Grafikleri oluştur
            try:
                plot_batch_results(results_df, trades_df, output_dir)
                logger.info("📈 Batch plotting completed")
            except Exception as e:
                logger.error(f"⚠️ Error in batch plotting: {e}")
            
            # En iyi sonuçları göster
            if not results_df.empty:
                best_roi = results_df.loc[results_df['roi_pct'].idxmax()]
                best_winrate = results_df.loc[results_df['win_rate'].idxmax()]
                
                logger.info("\n🏆 BEST RESULTS SUMMARY:")
                logger.info("=" * 50)
                logger.info(f"🥇 Best ROI Configuration:")
                logger.info(f"   Config ID: {best_roi['config_id']}")
                logger.info(f"   ROI: {best_roi['roi_pct']:.2f}%")
                logger.info(f"   Win Rate: {best_roi['win_rate']:.2f}%")
                logger.info(f"   Total Trades: {best_roi['total_trades']}")
                logger.info(f"   Profit Factor: {best_roi['profit_factor']:.2f}")
                
                 # ✅ TP/Commission filter effectiveness
                if backtest_params.get('enable_tp_commission_filter', False):
                    profitable_configs = (results_df['roi_pct'] > 0).sum()
                    total_configs = len(results_df)
                    success_rate = (profitable_configs / total_configs) * 100
                    
                    logger.info(f"\n🔧 TP/COMMISSION FILTER EFFECTIVENESS:")
                    logger.info(f"   Profitable configurations: {profitable_configs}/{total_configs} ({success_rate:.1f}%)")
                    logger.info(f"   Average ROI: {results_df['roi_pct'].mean():.2f}%")
                    logger.info(f"   Total trades executed: {results_df['total_trades'].sum()}")


                logger.info(f"\n🎯 Best Win Rate Configuration:")
                logger.info(f"   Config ID: {best_winrate['config_id']}")
                logger.info(f"   Win Rate: {best_winrate['win_rate']:.2f}%")
                logger.info(f"   ROI: {best_winrate['roi_pct']:.2f}%")
                logger.info(f"   Total Trades: {best_winrate['total_trades']}")
                logger.info(f"   Profit Factor: {best_winrate['profit_factor']:.2f}")
                
                # İstatistiksel özet
                logger.info(f"\n📊 STATISTICAL SUMMARY:")
                logger.info(f"   Average ROI: {results_df['roi_pct'].mean():.2f}%")
                logger.info(f"   Median ROI: {results_df['roi_pct'].median():.2f}%")
                logger.info(f"   ROI Std Dev: {results_df['roi_pct'].std():.2f}%")
                logger.info(f"   Profitable Configs: {(results_df['roi_pct'] > 0).sum()}/{len(results_df)} ({(results_df['roi_pct'] > 0).mean()*100:.1f}%)")
                logger.info(f"   Average Win Rate: {results_df['win_rate'].mean():.2f}%")
                logger.info(f"   Total Trades Executed: {results_df['total_trades'].sum()}")
                
                # Sonuçları döndür
                return {
                    "status": "success",
                    "total_configs": len(config_df),
                    "completed_configs": len(all_results),
                    "successful_configs": len(valid_results),
                    "best_roi_config": best_roi['config_id'],
                    "best_roi_pct": best_roi['roi_pct'],
                    "best_winrate_config": best_winrate['config_id'],
                    "best_winrate_pct": best_winrate['win_rate'],
                    "average_roi": results_df['roi_pct'].mean(),
                    "profitable_configs_pct": (results_df['roi_pct'] > 0).mean() * 100,
                    "total_trades": results_df['total_trades'].sum(),
                    "results_path": results_path,
                    "trades_path": trades_path if not trades_df.empty else None,
                    "output_dir": output_dir,
                    # ✅ TP/Commission filter results
                    "filter_statistics": total_filter_stats,
                    "tp_commission_filter_enabled": backtest_params.get('enable_tp_commission_filter', False)
                }
            else:
                logger.warning("⚠️ No valid configurations with metrics found.")
                return {
                    "status": "warning",
                    "message": "No valid configurations with metrics found.",
                    "total_configs": len(config_df),
                    "completed_configs": len(all_results),
                    "successful_configs": 0
                }
        else:
            logger.warning("⚠️ No valid results with metrics found.")
            return {
                "status": "warning", 
                "message": "No valid results with metrics found.",
                "total_configs": len(config_df),
                "completed_configs": len(all_results),
                "successful_configs": 0
            }
    else:
        logger.warning("⚠️ No successful backtest results found.")
        return {
            "status": "warning",
            "message": "No successful backtest results found.",
            "total_configs": len(config_df),
            "completed_configs": 0,
            "successful_configs": 0
        }


def analyze_individual_configs(
    results: List[Dict[str, Any]], 
    output_dir: str,
    top_n: int = 5
) -> None:
    """
    Analyze individual configurations in detail
    
    Args:
        results: List of backtest results
        output_dir: Output directory
        top_n: Number of top configurations to analyze in detail
    """
    if not results:
        return
    
    logger.info(f"🔍 Analyzing top {top_n} configurations in detail...")
    
    # Sort by ROI and get top N
    sorted_results = sorted(results, key=lambda x: x.get('roi_pct', 0), reverse=True)
    top_results = sorted_results[:top_n]
    
    # Individual analysis output directory
    individual_dir = os.path.join(output_dir, "individual_analysis")
    os.makedirs(individual_dir, exist_ok=True)
    
    for i, result in enumerate(top_results):
        config_id = result['config_id']
        config_output_dir = os.path.join(individual_dir, f"config_{config_id}")
        os.makedirs(config_output_dir, exist_ok=True)
        
        try:
            # Detailed analysis
            analysis = analyze_single_result(result, config_output_dir, config_id)
            
            # Detailed plots
            plots = plot_single_result(result, config_output_dir, config_id)
            
            logger.info(f"✅ Detailed analysis completed for Config {config_id}")
            
        except Exception as e:
            logger.error(f"❌ Error in detailed analysis for Config {config_id}: {e}")


def save_configuration_summary(
    config_df: pd.DataFrame,
    results: List[Dict[str, Any]], 
    output_dir: str
) -> None:
    """
    Save configuration summary with performance metrics
    
    Args:
        config_df: Configuration DataFrame
        results: List of backtest results
        output_dir: Output directory
    """
    try:
        # Create summary with configs and their performance
        summary_data = []
        
        for result in results:
            config_id = result['config_id']
            
            # Find corresponding config row
            config_row = config_df[config_df['config_id'] == config_id].iloc[0] if len(config_df[config_df['config_id'] == config_id]) > 0 else None
            
            if config_row is not None:
                summary_item = {
                    'config_id': config_id,
                    'roi_pct': result.get('roi_pct', 0),
                    'win_rate': result['metrics'].get('win_rate', 0) if 'metrics' in result else 0,
                    'total_trades': result.get('total_trades', 0),
                    'profit_factor': result['metrics'].get('profit_factor', 0) if 'metrics' in result else 0,
                    'max_drawdown_pct': result['metrics'].get('max_drawdown_pct', 0) if 'metrics' in result else 0,
                    'sharpe_ratio': result['metrics'].get('sharpe_ratio', 0) if 'metrics' in result else 0,
                    'indicators_count': len(parse_json_config(config_row.get('indicators_long', '{}'))) + len(parse_json_config(config_row.get('indicators_short', '{}'))),
                    'strategies_count': len(parse_json_config(config_row.get('strategies', '{}'))),
                    'filters_count': len(parse_json_config(config_row.get('filters', '{}'))) - 2,  # Exclude min_checks and min_strength
                    'strength_count': len(parse_json_config(config_row.get('strength', '{}')))
                }
                summary_data.append(summary_item)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, "configuration_performance_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"📋 Configuration summary saved to {summary_path}")
    
    except Exception as e:
        logger.error(f"❌ Error saving configuration summary: {e}")


# Enhanced main function for testing
if __name__ == "__main__":
    """
    Test runner for batch backtest
    """
    import sys
    import os
    
    # Ana dizini Python path'ine ekle
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from backtest.utils.config_loader import load_env_config
    
    # Loglama ayarla
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Çevre değişkenlerinden konfigürasyonu yükle
    env_config = load_env_config()
    
    # Test parametreleri
    test_params = {
        "symbol": env_config.get("symbol", "ETHFIUSDT"),
        "interval": env_config.get("interval", "5m"),
        "config_csv_path": "backtest/config/config_combinations.csv",
        "db_url": env_config.get("db_url"),
        "output_dir": "backtest/results/batch_test",
        "backtest_params": {
            "initial_balance": float(env_config.get("initial_balance", 10000.0)),
            "risk_per_trade": float(env_config.get("risk_per_trade", 0.015)),
            "sl_multiplier": float(env_config.get("sl_multiplier", 1.2)),
            "tp_multiplier": float(env_config.get("tp_multiplier", 2.8)),
            "leverage": float(env_config.get("leverage", 5.0)),
            "position_direction": env_config.get("position_direction", {"Long": False, "Short": True}),
            "commission_rate": float(env_config.get("commission_rate", 0.001))
        },
        "max_workers": 4  # Test için daha düşük worker sayısı
    }
    
    logger.info("🚀 Starting batch backtest test...")
    logger.info(f"📊 Configuration: {test_params['symbol']} {test_params['interval']}")
    
    # Batch backtest çalıştır
    result = run_batch_backtest(**test_params)
    
    logger.info(f"🏁 Test completed with status: {result['status']}")
    
    if result['status'] == 'success':
        logger.info("✅ Batch backtest test successful!")
        logger.info(f"📈 Best ROI: {result['best_roi_pct']:.2f}% (Config: {result['best_roi_config']})")
    else:
        logger.error(f"❌ Batch backtest test failed: {result.get('message', 'Unknown error')}")