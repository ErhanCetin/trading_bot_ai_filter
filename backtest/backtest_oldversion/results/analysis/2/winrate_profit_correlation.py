#📌 1. Win Rate ile Kazanç Arasındaki İlişki (Correlation)
#Amaç: Gerçekten yüksek win rate = yüksek kâr mı?

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("batch_results.csv")

correlation = df["win_rate"].corr(df["total_gain_usd"])
print(f"📈 Correlation between win_rate and total_gain_usd: {correlation:.2f}")

plt.scatter(df["win_rate"], df["total_gain_usd"], alpha=0.5)
plt.xlabel("Win Rate (%)")
plt.ylabel("Total Gain USD")
plt.title("Win Rate vs Gain Correlation")
plt.grid(True)
plt.show()
