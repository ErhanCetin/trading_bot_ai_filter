import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any, Optional

def plot_results(trades_csv: str, output_dir: str):
    """
    Backtest sonuçlarını görselleştirir
    
    Args:
        trades_csv: Trade verilerini içeren CSV dosyası yolu
        output_dir: Grafiklerin kaydedileceği dizin
    """
    df = pd.read_csv(trades_csv, parse_dates=["time"])
    os.makedirs(output_dir, exist_ok=True)

    # 1. Equity Curve
    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["balance"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Balance (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equity_curve.png"))
    plt.close()

    # 2. Gain Distribution by Outcome
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df[df["outcome"].isin(["TP", "SL"])], x="gain_pct", hue="outcome", bins=50, kde=True)
    plt.title("Gain % Distribution (TP vs SL)")
    plt.xlabel("Gain %")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gain_distribution.png"))
    plt.close()

    # 3. ADX vs Gain (optional)
    if "adx" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df[df["outcome"].isin(["TP", "SL"])], x="adx", y="gain_pct", hue="outcome", alpha=0.6)
        plt.title("ADX vs Gain %")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "adx_vs_gain.png"))
        plt.close()

    # 4. RSI vs Gain (optional)
    if "rsi" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df[df["outcome"].isin(["TP", "SL"])], x="rsi", y="gain_pct", hue="outcome", alpha=0.6)
        plt.title("RSI vs Gain %")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rsi_vs_gain.png"))
        plt.close()
    
    # 5. Yeni: Signal Strength vs Win Rate
    if "signal_strength" in df.columns:
        strength_groups = df.groupby("signal_strength")
        win_rates = {}
        counts = {}
        
        for strength, group in strength_groups:
            outcomes = group["outcome"].value_counts()
            tp_count = outcomes.get("TP", 0)
            sl_count = outcomes.get("SL", 0)
            total = tp_count + sl_count
            
            if total > 0:
                win_rates[strength] = (tp_count / total) * 100
                counts[strength] = total
        
        if win_rates:
            plt.figure(figsize=(8, 4))
            strengths = list(win_rates.keys())
            rates = list(win_rates.values())
            
            bars = plt.bar(strengths, rates, color="skyblue")
            
            # Bar üzerine etiketler ekle
            for i, (bar, count) in enumerate(zip(bars, counts.values())):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                         f"{rates[i]:.1f}%\n(n={count})", 
                         ha="center", va="bottom")
            
            plt.title("Win Rate by Signal Strength")
            plt.xlabel("Signal Strength")
            plt.ylabel("Win Rate (%)")
            plt.ylim(0, 100)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "win_rate_by_signal_strength.png"))
            plt.close()
    
    # 6. Yeni: Direction Performance Comparison
    if "direction" in df.columns:
        direction_groups = df.groupby("direction")
        direction_stats = {}
        
        for direction, group in direction_groups:
            outcomes = group["outcome"].value_counts()
            tp_count = outcomes.get("TP", 0)
            sl_count = outcomes.get("SL", 0)
            total = tp_count + sl_count
            
            if total > 0:
                win_rate = (tp_count / total) * 100
                avg_gain = group["gain_pct"].mean()
                
                direction_stats[direction] = {
                    "win_rate": win_rate,
                    "avg_gain": avg_gain,
                    "count": total
                }
        
        if direction_stats:
            plt.figure(figsize=(10, 5))
            
            directions = list(direction_stats.keys())
            win_rates = [stats["win_rate"] for stats in direction_stats.values()]
            counts = [stats["count"] for stats in direction_stats.values()]
            
            x = range(len(directions))
            width = 0.4
            
            plt.bar(x, win_rates, width=width, label="Win Rate (%)", color="green")
            
            # Etiketler ekle
            for i, (rate, count) in enumerate(zip(win_rates, counts)):
                plt.text(i, rate + 2, f"{rate:.1f}%\n(n={count})", ha="center")
            
            plt.title("Performance by Direction")
            plt.xticks(x, directions)
            plt.ylabel("Win Rate (%)")
            plt.ylim(0, max(win_rates) * 1.2)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "performance_by_direction.png"))
            plt.close()

    print("✅ All plots saved to:", output_dir)