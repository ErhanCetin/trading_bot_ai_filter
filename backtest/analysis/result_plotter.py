"""
Backtest sonuçlarını görselleştirme modülü
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import os
import numpy as np


def plot_single_result(
    result: Dict[str, Any], 
    output_dir: str,
    config_id: str = "default"
) -> Dict[str, str]:
    """
    Tek bir backtest sonucunu görselleştirir
    
    Args:
        result: Backtest sonucu
        output_dir: Çıktı dizini
        config_id: Konfigürasyon ID'si
        
    Returns:
        Oluşturulan grafiklerin dosya yolları
    """
    if 'trades' not in result or not result['trades']:
        return {"status": "warning", "message": "No trades to plot"}
    
    # Klasörü oluştur
    plot_dir = os.path.join(output_dir, f"plots_{config_id}")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Trade'leri DataFrame'e dönüştür
    trades_df = pd.DataFrame(result['trades'])
    
    # Oluşturulan grafiklerin dosya yollarını sakla
    plot_paths = {}
    
    # 1. Equity Curve
    if 'equity_curve' in result and result['equity_curve']:
        equity_df = pd.DataFrame(result['equity_curve'])
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(equity_df['time']), equity_df['balance'])
        plt.title(f"Equity Curve - Config {config_id}")
        plt.xlabel("Time")
        plt.ylabel("Balance (USD)")
        plt.grid(True)
        plt.tight_layout()
        equity_path = os.path.join(plot_dir, "equity_curve.png")
        plt.savefig(equity_path)
        plt.close()
        plot_paths['equity_curve'] = equity_path
    
    # 2. Kazanç Dağılımı
    if 'gain_pct' in trades_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=trades_df, x='gain_pct', hue='outcome', kde=True, bins=30)
        plt.title(f"Gain Distribution - Config {config_id}")
        plt.xlabel("Gain %")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        gain_path = os.path.join(plot_dir, "gain_distribution.png")
        plt.savefig(gain_path)
        plt.close()
        plot_paths['gain_distribution'] = gain_path
    
    # 3. Yöne Göre Performans
    if 'direction' in trades_df.columns:
        direction_groups = trades_df.groupby('direction')
        win_rates = []
        gain_avgs = []
        trade_counts = []
        directions = []
        
        for direction, group in direction_groups:
            directions.append(direction)
            trade_counts.append(len(group))
            win_rates.append((group['outcome'] == 'TP').mean() * 100)
            gain_avgs.append(group['gain_pct'].mean())
        
        # Win rate bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(directions, win_rates, color=['green', 'red'])
        plt.title(f"Win Rate by Direction - Config {config_id}")
        plt.xlabel("Direction")
        plt.ylabel("Win Rate (%)")
        plt.ylim(0, 100)
        
        # Bar üzerine trade sayısı ve win rate yaz
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{win_rates[i]:.1f}%\n({trade_counts[i]} trades)",
                     ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        direction_path = os.path.join(plot_dir, "direction_performance.png")
        plt.savefig(direction_path)
        plt.close()
        plot_paths['direction_performance'] = direction_path
    
    # 4. Sinyal Gücüne Göre Performans
    if 'signal_strength' in trades_df.columns:
        strength_groups = trades_df.groupby('signal_strength')
        win_rates = []
        gain_avgs = []
        trade_counts = []
        strengths = []
        
        for strength, group in strength_groups:
            strengths.append(int(strength))
            trade_counts.append(len(group))
            win_rates.append((group['outcome'] == 'TP').mean() * 100)
            gain_avgs.append(group['gain_pct'].mean())
        
        # Veriyi sırala
        sorted_indices = np.argsort(strengths)
        strengths = [strengths[i] for i in sorted_indices]
        win_rates = [win_rates[i] for i in sorted_indices]
        trade_counts = [trade_counts[i] for i in sorted_indices]
        gain_avgs = [gain_avgs[i] for i in sorted_indices]
        
        # Win rate bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strengths, win_rates, color='skyblue')
        plt.title(f"Win Rate by Signal Strength - Config {config_id}")
        plt.xlabel("Signal Strength")
        plt.ylabel("Win Rate (%)")
        plt.ylim(0, 100)
        
        # Bar üzerine trade sayısı ve win rate yaz
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{win_rates[i]:.1f}%\n({trade_counts[i]} trades)",
                     ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        strength_path = os.path.join(plot_dir, "strength_performance.png")
        plt.savefig(strength_path)
        plt.close()
        plot_paths['strength_performance'] = strength_path
    
    # 5. İndikatör Değerlerine Göre Kazanç İlişkisi
    indicator_cols = [col for col in trades_df.columns if col.startswith(('rsi', 'macd', 'adx', 'cci'))]
    
    for col in indicator_cols:
        # NaN değerleri filtrele
        valid_df = trades_df.dropna(subset=[col])
        
        if len(valid_df) < 10:  # Yeterli veri yok
            continue
        
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(data=valid_df, x=col, y='gain_pct', hue='outcome', alpha=0.7)
        plt.title(f"{col.upper()} vs Gain % - Config {config_id}")
        plt.xlabel(col.upper())
        plt.ylabel("Gain %")
        
        # Trend çizgisi ekle
        sns.regplot(x=col, y='gain_pct', data=valid_df, scatter=False, line_kws={"color": "red"})
        
        # Korelasyon değerini ekle
        corr = valid_df[col].corr(valid_df['gain_pct'])
        plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        indicator_path = os.path.join(plot_dir, f"{col}_vs_gain.png")
        plt.savefig(indicator_path)
        plt.close()
        plot_paths[f'{col}_vs_gain'] = indicator_path
    
    return plot_paths


def plot_batch_results(
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_dir: str
) -> Dict[str, str]:
    """
    Toplu backtest sonuçlarını görselleştirir
    
    Args:
        results_df: Sonuç özeti DataFrame'i
        trades_df: Tüm trade'leri içeren DataFrame
        output_dir: Çıktı dizini
        
    Returns:
        Oluşturulan grafiklerin dosya yolları
    """
    # Klasörü oluştur
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Oluşturulan grafiklerin dosya yollarını sakla
    plot_paths = {}
    
    # 1. ROI Dağılımı
    if 'roi_pct' in results_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['roi_pct'], bins=30, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title("ROI Distribution Across All Configurations")
        plt.xlabel("ROI %")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roi_path = os.path.join(plot_dir, "roi_distribution.png")
        plt.savefig(roi_path)
        plt.close()
        plot_paths['roi_distribution'] = roi_path
    
    # 2. Win Rate ve ROI İlişkisi
    if 'win_rate' in results_df.columns and 'roi_pct' in results_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x='win_rate', y='roi_pct', alpha=0.7, 
                        size='total_trades', sizes=(20, 200), hue='sharpe_ratio')
        plt.title("Win Rate vs ROI")
        plt.xlabel("Win Rate (%)")
        plt.ylabel("ROI (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        winrate_roi_path = os.path.join(plot_dir, "winrate_vs_roi.png")
        plt.savefig(winrate_roi_path)
        plt.close()
        plot_paths['winrate_vs_roi'] = winrate_roi_path
    
    # 3. En İyi 10 Konfigürasyon (ROI'ye göre)
    if 'roi_pct' in results_df.columns:
        top_10 = results_df.sort_values('roi_pct', ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(top_10['config_id'].astype(str), top_10['roi_pct'])
        plt.title("Top 10 Configurations by ROI")
        plt.xlabel("Configuration ID")
        plt.ylabel("ROI (%)")
        plt.xticks(rotation=45)
        
        # Bar üzerine trade sayısı ve win rate yaz
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{top_10['roi_pct'].iloc[i]:.1f}%\n({top_10['total_trades'].iloc[i]} trades, WR: {top_10['win_rate'].iloc[i]:.1f}%)",
                     ha='center', va='bottom', rotation=0, fontsize=8)
        
        plt.tight_layout()
        top_config_path = os.path.join(plot_dir, "top_configurations.png")
        plt.savefig(top_config_path)
        plt.close()
        plot_paths['top_configurations'] = top_config_path
   
   # 4. Drawdown vs ROI İlişkisi
    if 'max_drawdown_pct' in results_df.columns and 'roi_pct' in results_df.columns:
       plt.figure(figsize=(10, 6))
       sns.scatterplot(data=results_df, x='max_drawdown_pct', y='roi_pct', alpha=0.7, 
                       size='total_trades', sizes=(20, 200), hue='sharpe_ratio')
       plt.title("Max Drawdown vs ROI")
       plt.xlabel("Max Drawdown (%)")
       plt.ylabel("ROI (%)")
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       drawdown_roi_path = os.path.join(plot_dir, "drawdown_vs_roi.png")
       plt.savefig(drawdown_roi_path)
       plt.close()
       plot_paths['drawdown_vs_roi'] = drawdown_roi_path
   
   # 5. Trade Sayısı ve Win Rate İlişkisi
    if 'total_trades' in results_df.columns and 'win_rate' in results_df.columns:
       plt.figure(figsize=(10, 6))
       sns.scatterplot(data=results_df, x='total_trades', y='win_rate', alpha=0.7, 
                       size='roi_pct', sizes=(20, 200), hue='roi_pct')
       plt.title("Number of Trades vs Win Rate")
       plt.xlabel("Total Trades")
       plt.ylabel("Win Rate (%)")
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       trades_winrate_path = os.path.join(plot_dir, "trades_vs_winrate.png")
       plt.savefig(trades_winrate_path)
       plt.close()
       plot_paths['trades_vs_winrate'] = trades_winrate_path
   
   # 6. Yöne Göre Performans Analizi (toplu)
    if 'direction' in trades_df.columns:
       direction_df = trades_df.groupby(['config_id', 'direction']).agg({
           'gain_pct': 'mean',
           'outcome': lambda x: (x == 'TP').mean() * 100,
           'gain_usd': 'sum',
           'entry_price': 'count'
       }).reset_index()
       
       direction_df.rename(columns={'outcome': 'win_rate', 'entry_price': 'count'}, inplace=True)
       
       plt.figure(figsize=(10, 6))
       sns.boxplot(data=direction_df, x='direction', y='win_rate')
       plt.title("Win Rate by Direction (Across All Configurations)")
       plt.xlabel("Direction")
       plt.ylabel("Win Rate (%)")
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       direction_box_path = os.path.join(plot_dir, "direction_winrate_boxplot.png")
       plt.savefig(direction_box_path)
       plt.close()
       plot_paths['direction_winrate_boxplot'] = direction_box_path
       
       plt.figure(figsize=(10, 6))
       sns.boxplot(data=direction_df, x='direction', y='gain_pct')
       plt.title("Gain % by Direction (Across All Configurations)")
       plt.xlabel("Direction")
       plt.ylabel("Average Gain %")
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       direction_gain_path = os.path.join(plot_dir, "direction_gain_boxplot.png")
       plt.savefig(direction_gain_path)
       plt.close()
       plot_paths['direction_gain_boxplot'] = direction_gain_path
   
   # 7. Sinyal Gücüne Göre Performans Analizi (toplu)
    if 'signal_strength' in trades_df.columns:
       strength_df = trades_df.groupby(['config_id', 'signal_strength']).agg({
           'gain_pct': 'mean',
           'outcome': lambda x: (x == 'TP').mean() * 100,
           'gain_usd': 'sum',
           'entry_price': 'count'
       }).reset_index()
       
       strength_df.rename(columns={'outcome': 'win_rate', 'entry_price': 'count'}, inplace=True)
       
       plt.figure(figsize=(10, 6))
       sns.boxplot(data=strength_df, x='signal_strength', y='win_rate')
       plt.title("Win Rate by Signal Strength (Across All Configurations)")
       plt.xlabel("Signal Strength")
       plt.ylabel("Win Rate (%)")
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       strength_box_path = os.path.join(plot_dir, "strength_winrate_boxplot.png")
       plt.savefig(strength_box_path)
       plt.close()
       plot_paths['strength_winrate_boxplot'] = strength_box_path
       
       plt.figure(figsize=(10, 6))
       sns.boxplot(data=strength_df, x='signal_strength', y='gain_pct')
       plt.title("Gain % by Signal Strength (Across All Configurations)")
       plt.xlabel("Signal Strength")
       plt.ylabel("Average Gain %")
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       strength_gain_path = os.path.join(plot_dir, "strength_gain_boxplot.png")
       plt.savefig(strength_gain_path)
       plt.close()
       plot_paths['strength_gain_boxplot'] = strength_gain_path
   
   # 8. Heatmap - Total Trades by Configuration
    if 'config_id' in trades_df.columns:
       # En çok trade içeren top 20 konfigürasyon
       config_counts = trades_df['config_id'].value_counts().nlargest(20)
       
       # Yöne göre pivot tablo oluştur
       if 'direction' in trades_df.columns:
           direction_pivot = pd.crosstab(
               trades_df[trades_df['config_id'].isin(config_counts.index)]['config_id'],
               trades_df[trades_df['config_id'].isin(config_counts.index)]['direction']
           )
           
           plt.figure(figsize=(10, 8))
           sns.heatmap(direction_pivot, annot=True, fmt='d', cmap='Blues')
           plt.title("Number of Trades by Configuration and Direction")
           plt.tight_layout()
           direction_heatmap_path = os.path.join(plot_dir, "direction_heatmap.png")
           plt.savefig(direction_heatmap_path)
           plt.close()
           plot_paths['direction_heatmap'] = direction_heatmap_path
       
       # Sonuca göre pivot tablo oluştur
       if 'outcome' in trades_df.columns:
           outcome_pivot = pd.crosstab(
               trades_df[trades_df['config_id'].isin(config_counts.index)]['config_id'],
               trades_df[trades_df['config_id'].isin(config_counts.index)]['outcome']
           )
           
           plt.figure(figsize=(10, 8))
           sns.heatmap(outcome_pivot, annot=True, fmt='d', cmap='Greens')
           plt.title("Number of Trades by Configuration and Outcome")
           plt.tight_layout()
           outcome_heatmap_path = os.path.join(plot_dir, "outcome_heatmap.png")
           plt.savefig(outcome_heatmap_path)
           plt.close()
           plot_paths['outcome_heatmap'] = outcome_heatmap_path
   
    return plot_paths    