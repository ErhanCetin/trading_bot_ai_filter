import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

RESULTS_DIR = "backtest/results"
TRADES_CSV = os.path.join(RESULTS_DIR, "trades.csv")
BREAKDOWN_CSV = os.path.join(RESULTS_DIR, "signal_breakdown.csv")
HEATMAP_PATH = os.path.join(RESULTS_DIR, "signal_heatmap.png")


def run_backtest_diagnostics():
    """
    Backtest sonuçlarını analiz eder ve görselleştirir.
    - Sinyal gücü ve yöne göre kırılım oluşturur
    - Heatmap grafiği oluşturur
    """
    df = pd.read_csv(TRADES_CSV)
    print(f"📊 Trade verileri yüklendi. Sütunlar: {df.columns.tolist()}")

    # Grupla ve metrikleri hesapla
    grouped = df.groupby(["signal_strength", "direction", "outcome"], dropna=False).agg(
        total_trades=("entry_price", "count"),
        avg_gain_pct=("gain_pct", "mean"),
        avg_gain_usd=("gain_usd", "mean"),
        avg_rr_ratio=("rr_ratio", "mean"),
        avg_atr=("atr", "mean")
    ).reset_index()

    # Eksik sütunlar varsa doldur
    for col in ["signal_strength", "direction", "outcome"]:
        if col not in grouped.columns:
            grouped[col] = "Unknown"

    grouped.to_csv(BREAKDOWN_CSV, index=False)
    print(f"✅ Breakdown saved to: {BREAKDOWN_CSV}")

    # Heatmap
    try:
        pivot = df.groupby(["signal_strength", "direction"]).size().unstack(fill_value=0)
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
        plt.title("Signal Count by Strength and Direction")
        plt.tight_layout()
        plt.savefig(HEATMAP_PATH)
        print(f"✅ Heatmap saved to: {HEATMAP_PATH}")
    except Exception as e:
        print(f"❌ Failed to generate heatmap: {e}")

    # Ek olarak detaylı analiz sonuçları
    generate_additional_analysis(df)


def generate_additional_analysis(df: pd.DataFrame) -> None:
    """
    Backtest verilerine ek analizler uygular
    
    Args:
        df: Trade verilerini içeren DataFrame
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        # 1. Sinyal gücüne göre kazanç dağılımı
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="signal_strength", y="gain_pct", hue="direction", data=df)
        plt.title("Gain % Distribution by Signal Strength and Direction")
        plt.savefig(os.path.join(RESULTS_DIR, "gain_by_strength.png"))
        
        # 2. Strateji başarı oranı analizi (sinyal gücüne göre)
        if "signal_strength" in df.columns:
            win_rates = df.groupby(["signal_strength", "direction"]).apply(
                lambda x: pd.Series({
                    'win_rate': 100 * (x['outcome'] == 'TP').sum() / len(x) if len(x) > 0 else 0,
                    'count': len(x)
                })
            ).reset_index()
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x="signal_strength", y="win_rate", hue="direction", data=win_rates)
            plt.title("Win Rate by Signal Strength and Direction")
            plt.savefig(os.path.join(RESULTS_DIR, "winrate_by_strength.png"))
        
        print(f"✅ Additional analysis saved to {RESULTS_DIR}")
    except Exception as e:
        print(f"❌ Failed to generate additional analysis: {e}")


if __name__ == "__main__":
    run_backtest_diagnostics()
    print("📊 Backtest diagnostics completed.")