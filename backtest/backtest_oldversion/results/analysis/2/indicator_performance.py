#📌 5. Indikatör Kullanımına Göre Performans Kırılımı
#Amaç: Hangi indikatörü kullanan config'ler daha başarılı?
import pandas as pd

config_df = pd.read_csv("config_combinations.csv")
results_df = pd.read_csv("batch_results.csv")
merged = config_df.merge(results_df, on="config_id")

print("📊 Indicator-Based Performance Breakdown\n")

# Kullandığımız tüm indikatörler
indicators = [
    "EMA_FAST", "EMA_SLOW", "RSI", "MACD", "ATR", "OBV", "CCI", "ADX",
    "SUPER_TREND_period", "BOLLINGER_length", "DONCHIAN_period", "Z_SCORE_length"
]

for ind in indicators:
    if ind in ["MACD", "OBV"]:
        filtered = merged[merged[ind] == True]
    else:
        filtered = merged[merged[ind].notna()]
    avg_gain = filtered["total_gain_usd"].mean()
    print(f"🔹 {ind} kullananların ortalama kazancı: {avg_gain:.2f} USDT")
