import pandas as pd

df = pd.read_csv("config_combinations.csv")

print("📊 Toplam konfigürasyon sayısı:", len(df))
print("📌 Kullanılan indikatörler ve dağılımları:")

columns_to_check = ["EMA_FAST", "EMA_SLOW", "RSI", "MACD", "ATR", "OBV", "CCI", "ADX",
                    "SUPER_TREND_period", "SUPER_TREND_multiplier",
                    "BOLLINGER_length", "BOLLINGER_stddev",
                    "DONCHIAN_period", "Z_SCORE_length"]

for col in columns_to_check:
    used = df[col].notna().sum()
    print(f" - {col}: {used} config içinde kullanılmış")
