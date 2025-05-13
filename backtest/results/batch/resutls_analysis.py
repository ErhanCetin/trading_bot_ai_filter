import pandas as pd

# Gerekli verileri yeniden yükleyelim
config_path = "/mnt/data/config_combinations.csv"
results_path = "/mnt/data/batch_results.csv"

# CSV dosyalarını oku
config_df = pd.read_csv(config_path)
results_df = pd.read_csv(results_path)

# En yüksek kazanca sahip ilk 10 config_id
top_10_ids = results_df.sort_values(by="total_gain_usd", ascending=False).head(10)["config_id"].tolist()

# Bu config_id'lere karşılık gelen satırları al
top_configs_df = config_df[config_df["config_id"].isin(top_10_ids)]

# Sıralamayı kazanca göre yap
top_configs_df = top_configs_df.set_index("config_id").loc[top_10_ids].reset_index()

# Dosya olarak kaydet
export_path = "/mnt/data/top_10_config_combinations.csv"
top_configs_df.to_csv(export_path, index=False)

export_path
