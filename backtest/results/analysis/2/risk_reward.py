#1️⃣ İşlem Başına Ortalama Risk & Getiri (RR) Dağılımı
#Risk/Reward oranlarını çıkar, yüksek RR kazandırıyor mu analiz et
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("batch_trades_all.csv")
df = df[df["outcome"].isin(["TP", "SL"])]
df["rr_ratio"].plot.hist(bins=20, edgecolor="black")
plt.title("Distribution of RR Ratio per Trade")
plt.xlabel("RR Ratio (TP_MULTIPLIER / SL_MULTIPLIER)")
plt.grid(True)
plt.show()
