import pandas as pd

df = pd.read_csv("batch_trades_all.csv")

# Sadece TP/SL işlemleri
df = df[df["outcome"].isin(["TP", "SL"])]

print("🧾 Trade Detayları")
print("Toplam işlem sayısı:", len(df))

# TP/SL oranı
tp = (df["outcome"] == "TP").sum()
sl = (df["outcome"] == "SL").sum()
print("TP:", tp, "({:.2f}%)".format(tp / len(df) * 100))
print("SL:", sl, "({:.2f}%)".format(sl / len(df) * 100))

# Ortalama kazanç/zarar
print("Ortalama kazanç (gain_usd):", df["gain_usd"].mean())
