# strategy_graph_report.py

import pandas as pd
import matplotlib.pyplot as plt

# Load previously generated strategy test results
results_df = pd.read_csv("strategy_test_results.csv")

# Plot 1: Win Rate per Strategy
plt.figure(figsize=(10, 6))
plt.barh(results_df["Strategy"], results_df["Win Rate %"], edgecolor='black')
plt.xlabel("Win Rate (%)")
plt.title("Strategy Win Rate Comparison")
plt.grid(axis='x')
plt.tight_layout()
plt.savefig("win_rate_comparison.png")
plt.close()

# Plot 2: Average Gain per Strategy
plt.figure(figsize=(10, 6))
plt.barh(results_df["Strategy"], results_df["Avg Gain %"], color='orange', edgecolor='black')
plt.xlabel("Average Gain per Trade (%)")
plt.title("Strategy Avg Gain Comparison")
plt.grid(axis='x')
plt.tight_layout()
plt.savefig("avg_gain_comparison.png")
plt.close()

print("✅ Graphs saved as 'win_rate_comparison.png' and 'avg_gain_comparison.png'")
