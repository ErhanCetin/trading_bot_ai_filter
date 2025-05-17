#2️⃣ Konfigürasyonlara Göre Trade Sıklığı (Yoğunluk)
#Hangi config daha fazla sinyal üretmiş?
import pandas as pd
df = pd.read_csv("batch_trades_all.csv")
trade_count_by_config = df["config_id"].value_counts().sort_values(ascending=False).head(10)
print("🔢 En çok trade üreten 10 config:")
print(trade_count_by_config)
