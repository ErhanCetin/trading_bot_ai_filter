import pandas as pd

df = pd.read_csv("backtest/results/analysis/2/config_combinations.csv")
top_ids = [2096, 2084, 2108, 2106, 2104, 2102, 2100, 2098, 2094, 2080]
print(df[df["config_id"].isin(top_ids)])
