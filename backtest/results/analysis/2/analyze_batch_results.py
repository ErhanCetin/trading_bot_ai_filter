import pandas as pd

df = pd.read_csv("batch_results.csv")

print("🏆 Genel Başarı Analizi")
print("Toplam konfigürasyon:", len(df))
print("Kazanan (gain > 0):", (df["total_gain_usd"] > 0).sum())
print("Kaybeden (gain < 0):", (df["total_gain_usd"] < 0).sum())

best = df.sort_values(by="total_gain_usd", ascending=False).head(5)
print("\n🚀 En kârlı 5 konfigürasyon:")
print(best[["config_id", "total_gain_usd", "total_trades", "win_rate"]])
