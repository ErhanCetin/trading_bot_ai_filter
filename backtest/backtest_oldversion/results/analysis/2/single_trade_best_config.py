#📌 3. En İyi Tek Bir İşlem Yapan Config’ler
#Amaç: Az işlem ama yüksek kar sağlayan config’leri bul.
import pandas as pd
df = pd.read_csv("batch_results.csv")
winners = df[df["total_gain_usd"] > 0]
low_trade_winners = winners[winners["total_trades"] < 5]
print("🎯 Az sayıda işlemle kazandıran config_id'ler:")
print(low_trade_winners[["config_id", "total_gain_usd", "total_trades"]])
