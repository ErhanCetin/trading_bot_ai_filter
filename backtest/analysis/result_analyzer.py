"""
Backtest sonuçlarını analiz etme modülü
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
    Tek bir backtest sonucunu analiz eder
    
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
    
    # Temel metrikleri hesapla
    analysis = {
        "total_trades": len(trades_df),
        "win_rate": result['metrics']['win_rate'],
        "profit_loss": result['profit_loss'],
        "roi_pct": result['roi_pct'],
        "max_drawdown_pct": result['metrics']['max_drawdown_pct'],
        "sharpe_ratio": result['metrics']['sharpe_ratio'],
        "profit_factor": result['metrics']['profit_factor'],
        "avg_gain_pct": float(trades_df['gain_pct'].mean()),  # NumPy float64'ü Python float'a çevir
        "avg_gain_usd": float(trades_df['gain_usd'].mean())   # NumPy float64'ü Python float'a çevir
    }
    
    # Yöne göre kırılım
    if 'direction' in trades_df.columns:
        direction_breakdown = {}
        for direction in trades_df['direction'].unique():
            dir_df = trades_df[trades_df['direction'] == direction]
            win_count = (dir_df['outcome'] == 'TP').sum()
            total = len(dir_df)
            win_rate = (win_count / total * 100) if total > 0 else 0
            
            direction_breakdown[direction] = {
                "count": int(total),  # NumPy int64'ü Python int'e çevir
                "win_count": int(win_count),  # NumPy int64'ü Python int'e çevir
                "win_rate": float(win_rate),  # NumPy float64'ü Python float'a çevir
                "avg_gain_pct": float(dir_df['gain_pct'].mean()),  # NumPy float64'ü Python float'a çevir
                "total_gain_usd": float(dir_df['gain_usd'].sum())  # NumPy float64'ü Python float'a çevir
            }
        
        analysis['direction_breakdown'] = direction_breakdown
    
    # Sinyal gücüne göre kırılım
    if 'signal_strength' in trades_df.columns:
        strength_breakdown = {}
        for strength in trades_df['signal_strength'].unique():
            str_df = trades_df[trades_df['signal_strength'] == strength]
            win_count = (str_df['outcome'] == 'TP').sum()
            total = len(str_df)
            win_rate = (win_count / total * 100) if total > 0 else 0
            
            # strength değeri NumPy int64 olabilir, string'e çevirirken int() ile dönüştür
            strength_key = str(int(strength))
            
            strength_breakdown[strength_key] = {
                "count": int(total),  # NumPy int64'ü Python int'e çevir
                "win_count": int(win_count),  # NumPy int64'ü Python int'e çevir
                "win_rate": float(win_rate),  # NumPy float64'ü Python float'a çevir
                "avg_gain_pct": float(str_df['gain_pct'].mean()),  # NumPy float64'ü Python float'a çevir
                "total_gain_usd": float(str_df['gain_usd'].sum())  # NumPy float64'ü Python float'a çevir
            }
        
        analysis['strength_breakdown'] = strength_breakdown
    
    # İndikatör değerlerine göre analiz
    indicator_cols = [col for col in trades_df.columns if col.startswith(('rsi', 'macd', 'adx', 'cci', 'supertrend'))]
    
    if indicator_cols:
        indicator_analysis = {}
        
        for col in indicator_cols:
            # NaN değerleri filtrele
            valid_df = trades_df.dropna(subset=[col])
            
            if len(valid_df) < 5:  # Yeterli veri yok
                continue
                
            # Korelasyon hesapla - NaN veya sonsuz değerler için güvenlik önlemi
            try:
                correlation = valid_df[col].corr(valid_df['gain_pct'])
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
                    bin_stats = valid_df.groupby(bins, observed=True).agg({
                        'gain_pct': 'mean',
                        'outcome': lambda x: (x == 'TP').mean() * 100,
                        'config_id': 'count'
                    }).rename(columns={'outcome': 'win_rate', 'config_id': 'count'})
                    
                    # Aralık değerlerini string formatına dönüştür
                    bin_ranges = {}
                    for i, (b, row) in enumerate(bin_stats.iterrows()):
                        # NaN ve sonsuz değerleri kontrol et
                        win_rate = row['win_rate']
                        avg_gain = row['gain_pct']
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
            "total_trades": len(trades_df)
        }
        return simplified_analysis
    
    return analysis


def analyze_batch_results(
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_dir: str
) -> Dict[str, Any]:
    """
    Toplu backtest sonuçlarını analiz eder
    
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
    
    # Temel istatistikleri hesapla
    stats = {
        "total_configs": int(len(results_df)),
        "configs_with_trades": int(len(results_df[results_df['total_trades'] > 0])),
        "total_trades": int(trades_df.shape[0]),
        "avg_trades_per_config": float(results_df['total_trades'].mean()),
        "avg_win_rate": float(results_df['win_rate'].mean()),
        "avg_roi": float(results_df['roi_pct'].mean()),
        "median_roi": float(results_df['roi_pct'].median()),
        "roi_std": float(results_df['roi_pct'].std()),
        "best_roi": float(results_df['roi_pct'].max()),
        "worst_roi": float(results_df['roi_pct'].min()),
        "profitable_configs": int((results_df['roi_pct'] > 0).sum()),
        "profitable_configs_pct": float((results_df['roi_pct'] > 0).mean() * 100)
    }
    
    # En iyi 5 konfigürasyonu bul (ROI'ye göre)
    best_configs = results_df.sort_values('roi_pct', ascending=False).head(5)[
        ['config_id', 'total_trades', 'win_rate', 'roi_pct', 'max_drawdown_pct', 'sharpe_ratio']
    ].to_dict('records')
    
    # NumPy tiplerini standart Python tiplerine çevir
    best_configs = convert_numpy_types(best_configs)
    
    stats['best_configs'] = best_configs
    
    # En kötü 5 konfigürasyonu bul (ROI'ye göre)
    worst_configs = results_df.sort_values('roi_pct').head(5)[
        ['config_id', 'total_trades', 'win_rate', 'roi_pct', 'max_drawdown_pct', 'sharpe_ratio']
    ].to_dict('records')
    
    # NumPy tiplerini standart Python tiplerine çevir
    worst_configs = convert_numpy_types(worst_configs)
    
    stats['worst_configs'] = worst_configs
    
    # Sonuçları JSON olarak kaydet
    try:
        stats_path = os.path.join(output_dir, "batch_analysis.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
    except TypeError as e:
        logger.error(f"JSON serialization error in batch analysis: {e}")
        # Basitleştirilmiş stats nesnesini kullan
        stats = {
            "status": "error",
            "message": "JSON serialization error",
            "total_configs": int(len(results_df)),
            "total_trades": int(trades_df.shape[0])
        }
        stats_path = os.path.join(output_dir, "batch_analysis_simplified.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
    
    # Yöne göre kırılım analizi
    if 'direction' in trades_df.columns:
        direction_df = trades_df.groupby(['config_id', 'direction']).agg({
            'gain_usd': 'sum',
            'outcome': lambda x: (x == 'TP').mean() * 100,
            'time': 'count'
        }).reset_index().rename(columns={'outcome': 'win_rate', 'time': 'count'})
        
        direction_path = os.path.join(output_dir, "direction_breakdown.csv")
        direction_df.to_csv(direction_path, index=False)
    else:
        direction_path = None
    
    # Sinyal gücüne göre kırılım analizi
    if 'signal_strength' in trades_df.columns:
        strength_df = trades_df.groupby(['config_id', 'signal_strength']).agg({
            'gain_usd': 'sum',
            'outcome': lambda x: (x == 'TP').mean() * 100,
            'time': 'count'
        }).reset_index().rename(columns={'outcome': 'win_rate', 'time': 'count'})
        
        strength_path = os.path.join(output_dir, "strength_breakdown.csv")
        strength_df.to_csv(strength_path, index=False)
    else:
        strength_path = None
    
    # Detaylı analiz sonuçlarını döndür
    return {
        "status": "success",
        "stats": stats,
        "stats_path": stats_path,
        "direction_breakdown_path": direction_path,
        "strength_breakdown_path": strength_path
    }