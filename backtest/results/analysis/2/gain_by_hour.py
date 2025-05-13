#📌 4. İşlem Süresi Dağılımı (Zamana Göre Kar/Zarar)
#Amaç: Günün saatine göre işlem kazancı nasıl değişiyor?
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("batch_trades_all.csv")
df["hour"] = pd.to_datetime(df["time"]).dt.hour
gain_by_hour = df.groupby("hour")["gain_usd"].mean()

print("🕒 Saatlik ortalama kazançlar:")
print(gain_by_hour)

gain_by_hour.plot(kind="bar", title="Hour vs Avg Gain USD")
plt.ylabel("Average Gain USD")
plt.xlabel("Hour of Day")
plt.grid(True)
plt.show()
