import pandas as pd

# Load result and trade data

results_df = pd.read_csv("backtest/results/analysis/2/batch_results.csv",  encoding="utf-8",on_bad_lines='skip')
trades_df = pd.read_csv("backtest/results/analysis/2/batch_trades_all.csv", encoding="utf-8",on_bad_lines='skip')

# Filter only trades with outcome TP/SL
trades_df = trades_df[trades_df["outcome"].isin(["TP", "SL"])]

# --- 1. Top 20 config by gain ---
top20 = results_df.sort_values(by="total_gain_usd", ascending=False).head(20)
print("🏆 Top 20 configurations by total_gain_usd:")
print(top20[["config_id", "total_gain_usd", "total_trades", "win_rate"]])

# --- 2. Worst 10 configs ---
worst10 = results_df.sort_values(by="total_gain_usd", ascending=True).head(10)
print("\n❌ Worst 10 configurations:")
print(worst10[["config_id", "total_gain_usd", "total_trades", "win_rate"]])

# --- 3. Global stats ---
print("\n📊 Global Statistics:")
print("Total trades:", len(trades_df))
print("Total configs tested:", len(results_df))
print("Average gain per trade:", trades_df["gain_usd"].mean())
print("TP count:", (trades_df["outcome"] == "TP").sum())
print("SL count:", (trades_df["outcome"] == "SL").sum())
print("TP ratio: {:.2f}%".format((trades_df["outcome"] == "TP").mean() * 100))

# --- 4. Save filtered data to send me back ---
top20_ids = top20["config_id"].tolist()
filtered_trades = trades_df[trades_df["config_id"].isin(top20_ids)]
filtered_trades.to_csv("top20_trades_only.csv", index=False)
top20.to_csv("top20_results_only.csv", index=False)
