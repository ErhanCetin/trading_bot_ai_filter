#📌 2. En Çok TP Getiren Config’ler
#Amaç: Kazandıran işlemlerin (TP) hacmini analiz et.
import pandas as pd
df = pd.read_csv("batch_trades_all.csv")
tp_counts = df[df["outcome"] == "TP"]["config_id"].value_counts().head(10)
print("🚀 En çok TP üreten config_id'ler:\n", tp_counts)
