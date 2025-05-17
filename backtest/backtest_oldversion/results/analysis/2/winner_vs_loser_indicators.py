#3️⃣ Kazanan Config’lerin Ortak Özellikleri
#Sinyal üreten ve kazandıran config’lerde benzer indikatör desenleri var mı?


#Bu script ile kazanan konfigürasyonların ortak noktalarını veri destekli olarak görebileceksin.


import pandas as pd

# Load data
config_df = pd.read_csv("config_combinations.csv")
results_df = pd.read_csv("batch_results.csv")

# Merge config + results
merged_df = config_df.merge(results_df, on="config_id")
winners = merged_df[merged_df["total_gain_usd"] > 0]
losers = merged_df[merged_df["total_gain_usd"] <= 0]

# List of all relevant indicators
indicators = [
    "EMA_FAST", "EMA_SLOW", "RSI", "MACD", "ATR", "OBV", "CCI", "ADX",
    "SUPER_TREND_period", "SUPER_TREND_multiplier",
    "BOLLINGER_length", "BOLLINGER_stddev",
    "DONCHIAN_period", "Z_SCORE_length"
]

print("🏆 Average values and usage in WINNING configs:\n")
for indicator in indicators:
    if indicator in ["MACD", "OBV"]:
        used_count = winners[indicator].sum()
        print(f"🔹 {indicator}: used in {used_count} winning configs")
    else:
        mean_value = winners[indicator].dropna().mean()
        print(f"🔹 {indicator}: average = {mean_value:.2f}")

print("\n❌ Average values and usage in LOSING configs:\n")
for indicator in indicators:
    if indicator in ["MACD", "OBV"]:
        used_count = losers[indicator].sum()
        print(f"🔻 {indicator}: used in {used_count} losing configs")
    else:
        mean_value = losers[indicator].dropna().mean()
        print(f"🔻 {indicator}: average = {mean_value:.2f}")
