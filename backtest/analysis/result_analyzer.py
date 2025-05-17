"""
Backtest sonuçlarını analiz etme modülü
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Union
import json


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
    
    # Temel metrikleri hesapla
    analysis = {
        "total_trades": len(trades_df),
        "win_rate": result['metrics']['win_rate'],
        "profit_loss": result['profit_loss'],
        "roi_pct": result['roi_pct'],
        "max_drawdown_pct": result['metrics']['max_drawdown_pct'],
        "sharpe_ratio": result['metrics']['sharpe_ratio'],
        "profit_factor": result['metrics']['profit_factor'],
        "avg_gain_pct": trades_df['gain_pct'].mean(),
        "avg_gain_usd": trades_df['gain_usd'].mean()
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
                "count": total,
                "win_count": win_count,
                "win_rate": win_rate,
                "avg_gain_pct": dir_df['gain_pct'].mean(),
                "total_gain_usd": dir_df['gain_usd'].sum()
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
            
            strength_breakdown[str(int(strength))] = {
                "count": total,
                "win_count": win_count,
                "win_rate": win_rate,
                "avg_gain_pct": str_df['gain_pct'].mean(),
                "total_gain_usd": str_df['gain_usd'].sum()
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
                
            # Korelasyon hesapla
            correlation = valid_df[col].corr(valid_df['gain_pct'])
            
            # Değer aralıklarına göre performans analizi
            try:
                # Değer aralıklarını hesapla
                bins = pd.qcut(valid_df[col], 4, duplicates='drop')
                bin_stats = valid_df.groupby(bins).agg({
                    'gain_pct': 'mean',
                    'outcome': lambda x: (x == 'TP').mean() * 100,
                    'config_id': 'count'
                }).rename(columns={'outcome': 'win_rate', 'config_id': 'count'})
                
                # Aralık değerlerini string formatına dönüştür
                bin_ranges = {str(i): {
                    'range': f"{b.left:.2f} - {b.right:.2f}",
                    'win_rate': row['win_rate'],
                    'avg_gain_pct': row['gain_pct'],
                    'count': row['count']
                } for i, (b, row) in enumerate(bin_stats.iterrows())}
                
                indicator_analysis[col] = {
                    'correlation': correlation,
                    'bin_performance': bin_ranges
                }
            except:
                # Bin oluşturma hatası olursa basit korelasyon kullan
                indicator_analysis[col] = {
                    'correlation': correlation
                }
        
        analysis['indicator_analysis'] = indicator_analysis
    
    # Analiz sonuçlarını JSON olarak kaydet
    analysis_path = os.path.join(output_dir, f"analysis_{config_id}.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=4)
    
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
    # Klasörü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Temel istatistikleri hesapla
    stats = {
        "total_configs": len(results_df),
        "configs_with_trades": len(results_df[results_df['total_trades'] > 0]),
        "total_trades": trades_df.shape[0],
        "avg_trades_per_config": results_df['total_trades'].mean(),
        "avg_win_rate": results_df['win_rate'].mean(),
        "avg_roi": results_df['roi_pct'].mean(),
        "median_roi": results_df['roi_pct'].median(),
        "roi_std": results_df['roi_pct'].std(),
        "best_roi": results_df['roi_pct'].max(),
        "worst_roi": results_df['roi_pct'].min(),
        "profitable_configs": (results_df['roi_pct'] > 0).sum(),
        "profitable_configs_pct": (results_df['roi_pct'] > 0).mean() * 100
    }
    
    # En iyi 5 konfigürasyonu bul (ROI'ye göre)
    best_configs = results_df.sort_values('roi_pct', ascending=False).head(5)[
        ['config_id', 'total_trades', 'win_rate', 'roi_pct', 'max_drawdown_pct', 'sharpe_ratio']
    ].to_dict('records')
    
    stats['best_configs'] = best_configs
    
    # En kötü 5 konfigürasyonu bul (ROI'ye göre)
    worst_configs = results_df.sort_values('roi_pct').head(5)[
        ['config_id', 'total_trades', 'win_rate', 'roi_pct', 'max_drawdown_pct', 'sharpe_ratio']
    ].to_dict('records')
    
    stats['worst_configs'] = worst_configs
    
    # Sonuçları JSON olarak kaydet
    stats_path = os.path.join(output_dir, "batch_analysis.json")
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
    
    # Sinyal gücüne göre kırılım analizi
    if 'signal_strength' in trades_df.columns:
        strength_df = trades_df.groupby(['config_id', 'signal_strength']).agg({
            'gain_usd': 'sum',
            'outcome': lambda x: (x == 'TP').mean() * 100,
            'time': 'count'
        }).reset_index().rename(columns={'outcome': 'win_rate', 'time': 'count'})
        
        strength_path = os.path.join(output_dir, "strength_breakdown.csv")
        strength_df.to_csv(strength_path, index=False)
    
    # Detaylı analiz sonuçlarını döndür
    return {
        "status": "success",
        "stats": stats,
        "stats_path": stats_path,
        "direction_breakdown_path": direction_path if 'direction' in trades_df.columns else None,
        "strength_breakdown_path": strength_path if 'signal_strength' in trades_df.columns else None
    }