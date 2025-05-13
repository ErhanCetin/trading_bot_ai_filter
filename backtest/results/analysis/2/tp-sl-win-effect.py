#4️⃣ TP/SL Oranı Farklılığının Kazanca Etkisi
#RR değiştikçe performans değişiyor mu?
import pandas as pd
results = pd.read_csv("batch_results.csv")
results["rr_ratio"] = results["win_rate"] / 100  # yaklaşık olarak

import matplotlib.pyplot as plt
plt.scatter(results["rr_ratio"], results["total_gain_usd"])
plt.xlabel("RR Ratio Approx (from Win Rate)")
plt.ylabel("Total Gain USD")
plt.title("Win Rate vs Gain USD")
plt.grid(True)
plt.show()
