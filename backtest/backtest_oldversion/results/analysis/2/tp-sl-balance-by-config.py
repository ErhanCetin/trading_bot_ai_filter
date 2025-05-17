#5️⃣ Config Kararlılığı: TP ve SL’ler Arasındaki Denge
#Kârlı config’lerde TP ve SL sayıları ne kadar dengeli?
import pandas as pd
trades = pd.read_csv("batch_trades_all.csv")
tp_sl_counts = trades[trades["outcome"].isin(["TP", "SL"])] \
    .groupby(["config_id", "outcome"])["gain_usd"] \
    .count().unstack().fillna(0)

tp_sl_counts["tp_sl_ratio"] = tp_sl_counts["TP"] / (tp_sl_counts["SL"] + 1)
top = tp_sl_counts.sort_values("tp_sl_ratio", ascending=False).head(10)
print("📈 En dengeli (TP yüksek) config’ler:\n", top[["TP", "SL", "tp_sl_ratio"]])
