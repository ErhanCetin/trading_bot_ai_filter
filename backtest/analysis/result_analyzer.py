"""
Backtest sonuçlarını analiz etme modülü
ENHANCED: Fixed win rate calculation inconsistencies and added comprehensive metrics
"""
import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, List, Any, Optional, Union
import json
import logging

# Logger ayarla
logger = logging.getLogger(__name__)


def analyze_single_result(
    result: Dict[str, Any], 
    output_dir: str,
    config_id: str = "default"
) -> Dict[str, Any]:
    """
    ENHANCED: Tek bir backtest sonucunu analiz eder with consistent metrics calculation
    
    Args:
        result: Backtest sonucu
        output_dir: Çıktı dizini
        config_id: Konfigürasyon ID'si
        
    Returns:
        Analiz sonuçları
    """
    if 'trades' not in result or not result['trades']:
        return {"status": "warning", "message": "No trades to analyze"}
    
    # Trade'leri DataFrame'e dönüştür
    trades_df = pd.DataFrame(result['trades'])
    
    # NumPy int64 ve float64 tiplerini standart Python int ve float'a çeviren yardımcı fonksiyon
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # ENHANCED: Consistent win rate calculation using net_pnl
    profitable_trades = trades_df[trades_df["net_pnl"] > 0] if "net_pnl" in trades_df.columns else trades_df[trades_df["gain_usd"] > 0]
    total_trades = len(trades_df)
    win_rate = (len(profitable_trades) / total_trades * 100) if total_trades > 0 else 0
    
    # ENHANCED: Comprehensive metrics calculation
    analysis = {
        # Core metrics using consistent calculation
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_loss": result['profit_loss'],
        "roi_pct": result['roi_pct'],
        
        # Enhanced financial metrics
        "gross_profit": profitable_trades["net_pnl"].sum() if "net_pnl" in trades_df.columns and len(profitable_trades) > 0 else profitable_trades["gain_usd"].sum() if len(profitable_trades) > 0 else 0,
        "gross_loss": abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum()) if "net_pnl" in trades_df.columns else abs(trades_df[trades_df["gain_usd"] < 0]["gain_usd"].sum()),
        
        # Risk metrics from result if available, otherwise calculate
        "max_drawdown_pct": result['metrics'].get('max_drawdown_pct', 0) if 'metrics' in result else 0,
        "sharpe_ratio": result['metrics'].get('sharpe_ratio', 0) if 'metrics' in result else 0,
        "profit_factor": result['metrics'].get('profit_factor', 0) if 'metrics' in result else 0,
        
        # Trade performance
        "avg_gain_pct": float(trades_df['gain_pct'].mean()) if 'gain_pct' in trades_df.columns else 0,
        "avg_gain_usd": float(trades_df['gain_usd'].mean()) if 'gain_usd' in trades_df.columns else float(trades_df['net_pnl'].mean()) if 'net_pnl' in trades_df.columns else 0,
        
        # Enhanced metrics
        "largest_win": profitable_trades["net_pnl"].max() if "net_pnl" in trades_df.columns and len(profitable_trades) > 0 else profitable_trades["gain_usd"].max() if len(profitable_trades) > 0 else 0,
        "largest_loss": trades_df[trades_df["net_pnl"] < 0]["net_pnl"].min() if "net_pnl" in trades_df.columns else trades_df[trades_df["gain_usd"] < 0]["gain_usd"].min(),
        "average_win": profitable_trades["net_pnl"].mean() if "net_pnl" in trades_df.columns and len(profitable_trades) > 0 else profitable_trades["gain_usd"].mean() if len(profitable_trades) > 0 else 0,
        "average_loss": trades_df[trades_df["net_pnl"] < 0]["net_pnl"].mean() if "net_pnl" in trades_df.columns else trades_df[trades_df["gain_usd"] < 0]["gain_usd"].mean(),
        "winning_trades": len(profitable_trades),
        "losing_trades": len(trades_df[trades_df["net_pnl"] < 0]) if "net_pnl" in trades_df.columns else len(trades_df[trades_df["gain_usd"] < 0])
    }
    
    # ENHANCED: Direction performance analysis
    if 'direction' in trades_df.columns:
        direction_breakdown = {}
        for direction in trades_df['direction'].unique():
            dir_df = trades_df[trades_df['direction'] == direction]
            
            # Use consistent win calculation
            dir_profitable = dir_df[dir_df["net_pnl"] > 0] if "net_pnl" in dir_df.columns else dir_df[dir_df["gain_usd"] > 0]
            dir_total = len(dir_df)
            dir_win_rate = (len(dir_profitable) / dir_total * 100) if dir_total > 0 else 0
            
            direction_breakdown[direction] = {
                "count": int(dir_total),
                "win_count": int(len(dir_profitable)),
                "win_rate": float(dir_win_rate),
                "avg_gain_pct": float(dir_df['gain_pct'].mean()) if 'gain_pct' in dir_df.columns else 0,
                "total_gain_usd": float(dir_df['gain_usd'].sum()) if 'gain_usd' in dir_df.columns else float(dir_df['net_pnl'].sum()) if 'net_pnl' in dir_df.columns else 0,
                "gross_profit": float(dir_profitable["net_pnl"].sum()) if "net_pnl" in dir_df.columns and len(dir_profitable) > 0 else float(dir_profitable["gain_usd"].sum()) if len(dir_profitable) > 0 else 0,
                "gross_loss": float(abs(dir_df[dir_df["net_pnl"] < 0]["net_pnl"].sum())) if "net_pnl" in dir_df.columns else float(abs(dir_df[dir_df["gain_usd"] < 0]["gain_usd"].sum()))
            }
            
            # Calculate profit factor for direction
            if direction_breakdown[direction]["gross_loss"] > 0:
                direction_breakdown[direction]["profit_factor"] = direction_breakdown[direction]["gross_profit"] / direction_breakdown[direction]["gross_loss"]
            else:
                direction_breakdown[direction]["profit_factor"] = float('inf') if direction_breakdown[direction]["gross_profit"] > 0 else 0
        
        analysis['direction_breakdown'] = direction_breakdown
    
    # ENHANCED: Signal strength analysis
    if 'signal_strength' in trades_df.columns:
        strength_breakdown = {}
        for strength in trades_df['signal_strength'].unique():
            str_df = trades_df[trades_df['signal_strength'] == strength]
            str_profitable = str_df[str_df["net_pnl"] > 0] if "net_pnl" in str_df.columns else str_df[str_df["gain_usd"] > 0]
            str_total = len(str_df)
            str_win_rate = (len(str_profitable) / str_total * 100) if str_total > 0 else 0
            
            # strength değeri NumPy int64 olabilir, string'e çevirirken int() ile dönüştür
            strength_key = str(int(strength)) if not pd.isna(strength) else "unknown"
            
            strength_breakdown[strength_key] = {
                "count": int(str_total),
                "win_count": int(len(str_profitable)),
                "win_rate": float(str_win_rate),
                "avg_gain_pct": float(str_df['gain_pct'].mean()) if 'gain_pct' in str_df.columns else 0,
                "total_gain_usd": float(str_df['gain_usd'].sum()) if 'gain_usd' in str_df.columns else float(str_df['net_pnl'].sum()) if 'net_pnl' in str_df.columns else 0
            }
        
        analysis['strength_breakdown'] = strength_breakdown
    
    # ENHANCED: Outcome distribution analysis
    if 'outcome' in trades_df.columns:
        outcome_stats = trades_df['outcome'].value_counts().to_dict()
        # Convert numpy types
        outcome_stats = {k: int(v) for k, v in outcome_stats.items()}
        analysis['outcome_distribution'] = outcome_stats
        
        # TP/SL ratio
        tp_count = outcome_stats.get('TP', 0)
        sl_count = outcome_stats.get('SL', 0)
        analysis['tp_sl_ratio'] = tp_count / sl_count if sl_count > 0 else float('inf')
    
    # ENHANCED: Risk metrics
    if 'rr_ratio' in trades_df.columns:
        analysis['avg_rr_ratio'] = float(trades_df['rr_ratio'].mean())
        analysis['median_rr_ratio'] = float(trades_df['rr_ratio'].median())
    
    # ENHANCED: Time-based analysis
    if 'time' in trades_df.columns:
        try:
            trades_df['time_dt'] = pd.to_datetime(trades_df['time'], unit='ms')
            analysis['trading_period'] = {
                "start": trades_df['time_dt'].min().strftime('%Y-%m-%d %H:%M'),
                "end": trades_df['time_dt'].max().strftime('%Y-%m-%d %H:%M'),
                "duration_days": (trades_df['time_dt'].max() - trades_df['time_dt'].min()).days
            }
        except:
            pass
    
    # ENHANCED: Consecutive wins/losses analysis
    if 'net_pnl' in trades_df.columns:
        analysis['max_consecutive_wins'] = _calculate_max_consecutive(trades_df['net_pnl'] > 0)
        analysis['max_consecutive_losses'] = _calculate_max_consecutive(trades_df['net_pnl'] < 0)
    elif 'gain_usd' in trades_df.columns:
        analysis['max_consecutive_wins'] = _calculate_max_consecutive(trades_df['gain_usd'] > 0)
        analysis['max_consecutive_losses'] = _calculate_max_consecutive(trades_df['gain_usd'] < 0)
    
    # İndikatör değerlerine göre analiz (mevcut kod korundu)
    indicator_cols = [col for col in trades_df.columns if col.startswith(('rsi', 'macd', 'adx', 'cci', 'supertrend', 'indicator_'))]
    
    if indicator_cols:
        indicator_analysis = {}
        
        for col in indicator_cols:
            # NaN değerleri filtrele
            valid_df = trades_df.dropna(subset=[col])
            
            if len(valid_df) < 5:  # Yeterli veri yok
                continue
                
            # Korelasyon hesapla - NaN veya sonsuz değerler için güvenlik önlemi
            try:
                correlation = valid_df[col].corr(valid_df['gain_pct'] if 'gain_pct' in valid_df.columns else valid_df['net_pnl'])
                if pd.isna(correlation) or np.isinf(correlation):
                    correlation = 0.0  # Geçersiz değerler için varsayılan
            except:
                correlation = 0.0  # Hata durumunda varsayılan
            
            # Değer aralıklarına göre performans analizi
            try:
                # Yeterli benzersiz değer var mı kontrol et
                if len(valid_df[col].unique()) >= 4:
                    # Değer aralıklarını hesapla
                    bins = pd.qcut(valid_df[col], 4, duplicates='drop')
                    pnl_col = 'gain_pct' if 'gain_pct' in valid_df.columns else 'net_pnl'
                    outcome_col = 'outcome' if 'outcome' in valid_df.columns else None
                    
                    if outcome_col:
                        bin_stats = valid_df.groupby(bins, observed=True).agg({
                            pnl_col: 'mean',
                            outcome_col: lambda x: (x == 'TP').mean() * 100,
                            'config_id': 'count'
                        }).rename(columns={outcome_col: 'win_rate', 'config_id': 'count'})
                    else:
                        # Use PnL-based win rate if outcome not available
                        bin_stats = valid_df.groupby(bins, observed=True).agg({
                            pnl_col: 'mean',
                            'net_pnl': lambda x: (x > 0).mean() * 100 if 'net_pnl' in valid_df.columns else (valid_df['gain_usd'] > 0).mean() * 100,
                            'config_id': 'count'
                        }).rename(columns={'net_pnl': 'win_rate', 'config_id': 'count'})
                    
                    # Aralık değerlerini string formatına dönüştür
                    bin_ranges = {}
                    for i, (b, row) in enumerate(bin_stats.iterrows()):
                        # NaN ve sonsuz değerleri kontrol et
                        win_rate = row['win_rate']
                        avg_gain = row[pnl_col]
                        count = row['count']
                        
                        if pd.isna(win_rate) or np.isinf(win_rate):
                            win_rate = 0.0
                        if pd.isna(avg_gain) or np.isinf(avg_gain):
                            avg_gain = 0.0
                        
                        bin_ranges[str(i)] = {
                            'range': f"{b.left:.2f} - {b.right:.2f}",
                            'win_rate': float(win_rate),
                            'avg_gain_pct': float(avg_gain),
                            'count': int(count)
                        }
                    
                    indicator_analysis[col] = {
                        'correlation': float(correlation),
                        'bin_performance': bin_ranges
                    }
                else:
                    # Yetersiz benzersiz değer, sadece korelasyon göster
                    indicator_analysis[col] = {
                        'correlation': float(correlation)
                    }
            except Exception as e:
                # Bin oluşturma hatası olursa basit korelasyon kullan
                indicator_analysis[col] = {
                    'correlation': float(correlation),
                    'error': str(e)
                }
        
        analysis['indicator_analysis'] = indicator_analysis
    
    # NumPy verilerini standart Python türlerine dönüştür
    analysis = convert_numpy_types(analysis)
    
    # Analiz sonuçlarını JSON olarak kaydet
    try:
        analysis_path = os.path.join(output_dir, f"analysis_{config_id}.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=4)
            
        logger.info(f"📊 Analysis saved to {analysis_path}")
        
    except TypeError as e:
        # Serileştirme hatası olursa, hata mesajını yazdır ve
        # problemi debug etmek için detayları logla
        logger.error(f"JSON serialization error: {e}")
        # Problemli değerleri bulmaya çalış
        for key, value in analysis.items():
            try:
                json.dumps({key: value})
            except TypeError:
                logger.error(f"Problem with key: {key}, value type: {type(value)}")
                # Karmaşık veri yapılarını derinlemesine incele
                if isinstance(value, dict):
                    for k, v in value.items():
                        try:
                            json.dumps({k: v})
                        except TypeError:
                            logger.error(f"  Nested problem with key: {k}, value type: {type(v)}")
        
        # Basit bir analiz sonucu döndür, en azından hata olmaz
        simplified_analysis = {
            "status": "error",
            "message": "JSON serialization error",
            "total_trades": len(trades_df),
            "win_rate": win_rate,
            "profit_loss": result.get('profit_loss', 0),
            "roi_pct": result.get('roi_pct', 0)
        }
        return simplified_analysis
    
    return analysis

def _calculate_max_consecutive(boolean_series: pd.Series) -> int:
    """
    ENHANCED: Calculate maximum consecutive True values in a boolean series
    """
    max_consecutive = 0
    current_consecutive = 0
    
    for value in boolean_series:
        if value:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def analyze_batch_results(
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_dir: str
) -> Dict[str, Any]:
    """
    ENHANCED: Toplu backtest sonuçlarını analiz eder with consistent metrics
    
    Args:
        results_df: Sonuç özeti DataFrame'i
        trades_df: Tüm trade'leri içeren DataFrame
        output_dir: Çıktı dizini
        
    Returns:
        Analiz sonuçları
    """
    # NumPy int64 ve float64 tiplerini standart Python int ve float'a çeviren yardımcı fonksiyon
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Klasörü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # ENHANCED: Comprehensive batch statistics
    stats = {
        "total_configs": int(len(results_df)),
        "configs_with_trades": int(len(results_df[results_df['total_trades'] > 0])),
        "total_trades": int(trades_df.shape[0]) if not trades_df.empty else 0,
        "avg_trades_per_config": float(results_df['total_trades'].mean()),
        "median_trades_per_config": float(results_df['total_trades'].median()),
        
        # Performance metrics
        "avg_win_rate": float(results_df['win_rate'].mean()),
        "median_win_rate": float(results_df['win_rate'].median()),
        "win_rate_std": float(results_df['win_rate'].std()),
        
        # ROI statistics
        "avg_roi": float(results_df['roi_pct'].mean()),
        "median_roi": float(results_df['roi_pct'].median()),
        "roi_std": float(results_df['roi_pct'].std()),
        "best_roi": float(results_df['roi_pct'].max()),
        "worst_roi": float(results_df['roi_pct'].min()),
        
        # Profitability analysis
        "profitable_configs": int((results_df['roi_pct'] > 0).sum()),
        "profitable_configs_pct": float((results_df['roi_pct'] > 0).mean() * 100),
        "breakeven_configs": int((results_df['roi_pct'] == 0).sum()),
        "losing_configs": int((results_df['roi_pct'] < 0).sum()),
        
        # Risk metrics
        "avg_max_drawdown": float(results_df['max_drawdown_pct'].mean()) if 'max_drawdown_pct' in results_df.columns else 0,
        "worst_max_drawdown": float(results_df['max_drawdown_pct'].max()) if 'max_drawdown_pct' in results_df.columns else 0,
        "avg_sharpe_ratio": float(results_df['sharpe_ratio'].mean()) if 'sharpe_ratio' in results_df.columns else 0,
        "avg_profit_factor": float(results_df['profit_factor'].mean()) if 'profit_factor' in results_df.columns else 0,
        
        # Performance distribution
        "roi_quartiles": {
            "q25": float(results_df['roi_pct'].quantile(0.25)),
            "q50": float(results_df['roi_pct'].quantile(0.50)),
            "q75": float(results_df['roi_pct'].quantile(0.75)),
            "q90": float(results_df['roi_pct'].quantile(0.90)),
            "q95": float(results_df['roi_pct'].quantile(0.95))
        }
    }
    
    # ENHANCED: Top performing configurations analysis
    if not results_df.empty:
        # En iyi 5 konfigürasyonu bul (ROI'ye göre)
        best_configs = results_df.sort_values('roi_pct', ascending=False).head(5)[
            ['config_id', 'total_trades', 'win_rate', 'roi_pct', 'max_drawdown_pct', 'sharpe_ratio', 'profit_factor']
        ].to_dict('records')
        
        # NumPy tiplerini standart Python tiplerine çevir
        best_configs = convert_numpy_types(best_configs)
        stats['best_configs'] = best_configs
        
        # En kötü 5 konfigürasyonu bul (ROI'ye göre)
        worst_configs = results_df.sort_values('roi_pct').head(5)[
            ['config_id', 'total_trades', 'win_rate', 'roi_pct', 'max_drawdown_pct', 'sharpe_ratio', 'profit_factor']
        ].to_dict('records')
        
        # NumPy tiplerini standart Python tiplerine çevir
        worst_configs = convert_numpy_types(worst_configs)
        stats['worst_configs'] = worst_configs
        
        # ENHANCED: Best by different metrics
        if 'win_rate' in results_df.columns:
            best_winrate_config = results_df.loc[results_df['win_rate'].idxmax()]
            stats['best_winrate_config'] = {
                'config_id': best_winrate_config['config_id'],
                'win_rate': float(best_winrate_config['win_rate']),
                'roi_pct': float(best_winrate_config['roi_pct']),
                'total_trades': int(best_winrate_config['total_trades'])
            }
        
        if 'sharpe_ratio' in results_df.columns and results_df['sharpe_ratio'].max() > 0:
            best_sharpe_config = results_df.loc[results_df['sharpe_ratio'].idxmax()]
            stats['best_sharpe_config'] = {
                'config_id': best_sharpe_config['config_id'],
                'sharpe_ratio': float(best_sharpe_config['sharpe_ratio']),
                'roi_pct': float(best_sharpe_config['roi_pct']),
                'win_rate': float(best_sharpe_config['win_rate'])
            }
        
        if 'profit_factor' in results_df.columns and results_df['profit_factor'].max() > 0:
            best_pf_config = results_df.loc[results_df['profit_factor'].idxmax()]
            stats['best_profit_factor_config'] = {
                'config_id': best_pf_config['config_id'],
                'profit_factor': float(best_pf_config['profit_factor']),
                'roi_pct': float(best_pf_config['roi_pct']),
                'win_rate': float(best_pf_config['win_rate'])
            }
    
    # ENHANCED: Trade-level analysis if trades available
    if not trades_df.empty:
        trade_analysis = _analyze_batch_trades(trades_df)
        stats['trade_analysis'] = trade_analysis
    
    # ENHANCED: Configuration performance correlation analysis
    if len(results_df) > 10:  # Enough data for correlation analysis
        correlation_analysis = _analyze_configuration_correlations(results_df)
        stats['correlation_analysis'] = correlation_analysis
    
    # ENHANCED: Risk-return profile analysis
    risk_return_analysis = _analyze_risk_return_profile(results_df)
    stats['risk_return_analysis'] = risk_return_analysis
    
    # Sonuçları JSON olarak kaydet
    try:
        stats_path = os.path.join(output_dir, "batch_analysis.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"📊 Batch analysis saved to {stats_path}")
    except TypeError as e:
        logger.error(f"JSON serialization error in batch analysis: {e}")
        # Basitleştirilmiş stats nesnesini kullan
        stats = {
            "status": "error",
            "message": "JSON serialization error",
            "total_configs": int(len(results_df)),
            "total_trades": int(trades_df.shape[0]) if not trades_df.empty else 0,
            "profitable_configs_pct": float((results_df['roi_pct'] > 0).mean() * 100),
            "avg_roi": float(results_df['roi_pct'].mean())
        }
        stats_path = os.path.join(output_dir, "batch_analysis_simplified.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
    
    # ENHANCED: Direction and strength breakdown analysis
    direction_path = None
    strength_path = None
    
    if not trades_df.empty:
        # Yöne göre kırılım analizi
        if 'direction' in trades_df.columns:
            direction_df = trades_df.groupby(['config_id', 'direction']).agg({
                'gain_usd': 'sum',
                'outcome': lambda x: (x == 'TP').mean() * 100 if 'outcome' in trades_df.columns else None,
                'net_pnl': lambda x: (x > 0).mean() * 100 if 'net_pnl' in trades_df.columns else None,
                'time': 'count'
            }).reset_index()
            
            # Rename columns appropriately
            if 'outcome' in trades_df.columns:
                direction_df = direction_df.rename(columns={'outcome': 'outcome_win_rate', 'time': 'count'})
            if 'net_pnl' in trades_df.columns:
                direction_df = direction_df.rename(columns={'net_pnl': 'pnl_win_rate', 'time': 'count'})
            else:
                direction_df = direction_df.rename(columns={'time': 'count'})
            
            direction_path = os.path.join(output_dir, "direction_breakdown.csv")
            direction_df.to_csv(direction_path, index=False)
            logger.info(f"📊 Direction breakdown saved to {direction_path}")
        
        # Sinyal gücüne göre kırılım analizi
        if 'signal_strength' in trades_df.columns:
            strength_df = trades_df.groupby(['config_id', 'signal_strength']).agg({
                'gain_usd': 'sum',
                'outcome': lambda x: (x == 'TP').mean() * 100 if 'outcome' in trades_df.columns else None,
                'net_pnl': lambda x: (x > 0).mean() * 100 if 'net_pnl' in trades_df.columns else None,
                'time': 'count'
            }).reset_index()
            
            # Rename columns appropriately
            if 'outcome' in trades_df.columns:
                strength_df = strength_df.rename(columns={'outcome': 'outcome_win_rate', 'time': 'count'})
            if 'net_pnl' in trades_df.columns:
                strength_df = strength_df.rename(columns={'net_pnl': 'pnl_win_rate', 'time': 'count'})
            else:
                strength_df = strength_df.rename(columns={'time': 'count'})
            
            strength_path = os.path.join(output_dir, "strength_breakdown.csv")
            strength_df.to_csv(strength_path, index=False)
            logger.info(f"📊 Strength breakdown saved to {strength_path}")
    
    # Detaylı analiz sonuçlarını döndür
    return {
        "status": "success",
        "stats": stats,
        "stats_path": stats_path,
        "direction_breakdown_path": direction_path,
        "strength_breakdown_path": strength_path
    }

def _analyze_batch_trades(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    ENHANCED: Analyze all trades across batch configurations
    """
    analysis = {}
    
    # Use consistent PnL column
    pnl_col = 'net_pnl' if 'net_pnl' in trades_df.columns else 'gain_usd'
    
    # Overall trade statistics
    analysis['total_trades'] = len(trades_df)
    analysis['profitable_trades'] = int((trades_df[pnl_col] > 0).sum())
    analysis['losing_trades'] = int((trades_df[pnl_col] < 0).sum())
    analysis['overall_win_rate'] = float((trades_df[pnl_col] > 0).mean() * 100)
    
    # PnL distribution
    analysis['total_pnl'] = float(trades_df[pnl_col].sum())
    analysis['avg_trade_pnl'] = float(trades_df[pnl_col].mean())
    analysis['median_trade_pnl'] = float(trades_df[pnl_col].median())
    analysis['pnl_std'] = float(trades_df[pnl_col].std())
    
    # Extreme values
    analysis['largest_win'] = float(trades_df[pnl_col].max())
    analysis['largest_loss'] = float(trades_df[pnl_col].min())
    
    # Direction analysis
    if 'direction' in trades_df.columns:
        direction_stats = {}
        for direction in trades_df['direction'].unique():
            dir_trades = trades_df[trades_df['direction'] == direction]
            direction_stats[direction] = {
                'count': len(dir_trades),
                'win_rate': float((dir_trades[pnl_col] > 0).mean() * 100),
                'avg_pnl': float(dir_trades[pnl_col].mean()),
                'total_pnl': float(dir_trades[pnl_col].sum())
            }
        analysis['direction_performance'] = direction_stats
    
    # Outcome analysis
    if 'outcome' in trades_df.columns:
        outcome_stats = trades_df['outcome'].value_counts().to_dict()
        analysis['outcome_distribution'] = {k: int(v) for k, v in outcome_stats.items()}
    
    return analysis

def _analyze_configuration_correlations(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    ENHANCED: Analyze correlations between configuration parameters and performance
    """
    correlation_analysis = {}
    
    # Select numeric columns for correlation analysis
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    performance_cols = ['roi_pct', 'win_rate', 'total_trades', 'max_drawdown_pct', 'sharpe_ratio', 'profit_factor']
    
    # Find configuration parameters (non-performance columns)
    config_cols = [col for col in numeric_cols if col not in performance_cols and col != 'config_id']
    
    if config_cols and len(results_df) > 10:
        # Calculate correlations with ROI
        roi_correlations = {}
        for col in config_cols:
            try:
                corr = results_df[col].corr(results_df['roi_pct'])
                if not pd.isna(corr):
                    roi_correlations[col] = float(corr)
            except:
                continue
        
        correlation_analysis['roi_correlations'] = roi_correlations
        
        # Calculate correlations with win rate
        if 'win_rate' in results_df.columns:
            winrate_correlations = {}
            for col in config_cols:
                try:
                    corr = results_df[col].corr(results_df['win_rate'])
                    if not pd.isna(corr):
                        winrate_correlations[col] = float(corr)
                except:
                    continue
            
            correlation_analysis['winrate_correlations'] = winrate_correlations
    
    return correlation_analysis

def _analyze_risk_return_profile(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    ENHANCED: Analyze risk-return characteristics of configurations
    """
    risk_return = {}
    
    # Risk-return scatter analysis
    if 'max_drawdown_pct' in results_df.columns and len(results_df) > 5:
        # Categorize configurations by risk-return profile
        median_roi = results_df['roi_pct'].median()
        median_dd = results_df['max_drawdown_pct'].median()
        
        # Quadrant analysis
        high_return_low_risk = results_df[
            (results_df['roi_pct'] > median_roi) & 
            (results_df['max_drawdown_pct'] < median_dd)
        ]
        high_return_high_risk = results_df[
            (results_df['roi_pct'] > median_roi) & 
            (results_df['max_drawdown_pct'] >= median_dd)
        ]
        low_return_low_risk = results_df[
            (results_df['roi_pct'] <= median_roi) & 
            (results_df['max_drawdown_pct'] < median_dd)
        ]
        low_return_high_risk = results_df[
            (results_df['roi_pct'] <= median_roi) & 
            (results_df['max_drawdown_pct'] >= median_dd)
        ]
        
        risk_return['quadrant_analysis'] = {
            'high_return_low_risk': {
                'count': len(high_return_low_risk),
                'avg_roi': float(high_return_low_risk['roi_pct'].mean()) if len(high_return_low_risk) > 0 else 0,
                'avg_drawdown': float(high_return_low_risk['max_drawdown_pct'].mean()) if len(high_return_low_risk) > 0 else 0
            },
            'high_return_high_risk': {
                'count': len(high_return_high_risk),
                'avg_roi': float(high_return_high_risk['roi_pct'].mean()) if len(high_return_high_risk) > 0 else 0,
                'avg_drawdown': float(high_return_high_risk['max_drawdown_pct'].mean()) if len(high_return_high_risk) > 0 else 0
            },
            'low_return_low_risk': {
                'count': len(low_return_low_risk),
                'avg_roi': float(low_return_low_risk['roi_pct'].mean()) if len(low_return_low_risk) > 0 else 0,
                'avg_drawdown': float(low_return_low_risk['max_drawdown_pct'].mean()) if len(low_return_low_risk) > 0 else 0
            },
            'low_return_high_risk': {
                'count': len(low_return_high_risk),
                'avg_roi': float(low_return_high_risk['roi_pct'].mean()) if len(low_return_high_risk) > 0 else 0,
                'avg_drawdown': float(low_return_high_risk['max_drawdown_pct'].mean()) if len(low_return_high_risk) > 0 else 0
            }
        }
    
    # Sharpe ratio analysis
    if 'sharpe_ratio' in results_df.columns:
        positive_sharpe = results_df[results_df['sharpe_ratio'] > 0]
        risk_return['sharpe_analysis'] = {
            'positive_sharpe_count': len(positive_sharpe),
            'positive_sharpe_pct': float(len(positive_sharpe) / len(results_df) * 100),
            'avg_positive_sharpe': float(positive_sharpe['sharpe_ratio'].mean()) if len(positive_sharpe) > 0 else 0,
            'max_sharpe': float(results_df['sharpe_ratio'].max()),
            'min_sharpe': float(results_df['sharpe_ratio'].min())
        }
    
    return risk_return