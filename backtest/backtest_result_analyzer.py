import pandas as pd
from typing import Dict, Any, Optional, List

def analyze_results(trades_csv_path: str, summary_csv_path: str):
    """
    Backtest sonuçlarını analiz eder ve özet istatistikler oluşturur
    
    Args:
        trades_csv_path: Trade verilerini içeren CSV dosyası yolu
        summary_csv_path: Özet sonuçların kaydedileceği CSV dosyası yolu
    """
    df = pd.read_csv(trades_csv_path)

    total = len(df)
    tp_count = (df["outcome"] == "TP").sum()
    sl_count = (df["outcome"] == "SL").sum()
    open_count = (df["outcome"] == "OPEN").sum()

    win_rate = (tp_count / (tp_count + sl_count)) * 100 if (tp_count + sl_count) > 0 else 0
    rr_ratio = df["rr_ratio"].mean() if "rr_ratio" in df.columns else 0

    summary = {
        "total_trades": total,
        "take_profits": tp_count,
        "stop_losses": sl_count,
        "open_trades": open_count,
        "win_rate_pct": win_rate,
        "avg_rr_ratio": rr_ratio,
        "avg_gain_pct": df["gain_pct"].mean() if "gain_pct" in df.columns else None,
        "avg_gain_usd": df["gain_usd"].mean() if "gain_usd" in df.columns else None,
        "final_balance": df["balance"].iloc[-1] if "balance" in df.columns and not df.empty else None
    }

    # Eğer sinyal gücüne göre kırılım verisi varsa, ekle
    if "signal_strength" in df.columns:
        for strength in df["signal_strength"].unique():
            strength_df = df[df["signal_strength"] == strength]
            if not strength_df.empty:
                strength_win_rate = (strength_df["outcome"] == "TP").sum() / len(strength_df) * 100
                summary[f"strength_{strength}_win_rate"] = strength_win_rate
                summary[f"strength_{strength}_count"] = len(strength_df)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Summary saved to: {summary_csv_path}")

    # Gruplandırılmış kırılım
    group_keys = ["signal_strength", "direction", "outcome"]
    agg_metrics = {
        "gain_pct": "mean",
        "gain_usd": "mean",
        "rr_ratio": "mean",
        "atr": "mean",
        "time": "count"
    }

    valid_metrics = {k: v for k, v in agg_metrics.items() if k in df.columns}

    if all(key in df.columns for key in group_keys) and valid_metrics:
        breakdown = df.groupby(group_keys).agg(valid_metrics).rename(columns={"time": "total_trades"}).reset_index()
        breakdown.to_csv("backtest/results/signal_breakdown.csv", index=False)
        print("✅ Breakdown saved to: backtest/results/signal_breakdown.csv")
    else:
        print("⚠️ Group keys or metrics missing for breakdown.")
        
    # İndikatör değerlerine göre kazanç analizi
    generate_indicator_analysis(df)


def generate_indicator_analysis(df: pd.DataFrame) -> None:
    """
    İndikatör değerlerine göre kazanç analizi
    
    Args:
        df: Trade verilerini içeren DataFrame
    """
    # İndikatör sütunları
    indicator_cols = ["rsi", "macd", "adx", "cci"]
    indicator_cols = [col for col in indicator_cols if col in df.columns]
    
    if not indicator_cols:
        return
        
    # Her indikatör için ortalama kazanç korelasyonu
    correlations = {}
    for col in indicator_cols:
        if col in df.columns and "gain_pct" in df.columns:
            # NaN değerlerini filtrele
            valid_data = df[[col, "gain_pct"]].dropna()
            if not valid_data.empty:
                corr = valid_data[col].corr(valid_data["gain_pct"])
                correlations[col] = corr
    
    if correlations:
        print("\n📊 İndikatör-Kazanç Korelasyonları:")
        for indicator, corr in correlations.items():
            print(f"  {indicator}: {corr:.4f}")
        
        # En yüksek pozitif ve negatif korelasyonu kaydet
        best_indicator = max(correlations.items(), key=lambda x: x[1])
        worst_indicator = min(correlations.items(), key=lambda x: x[1])
        
        print(f"\n✅ En yüksek pozitif korelasyon: {best_indicator[0]} ({best_indicator[1]:.4f})")
        print(f"❌ En yüksek negatif korelasyon: {worst_indicator[0]} ({worst_indicator[1]:.4f})")