import pandas as pd
from ai_filter.signal_ranker import SignalFilter

# Örnek veri: geçmiş sinyallerin sonucu ile etiketlenmiş olmalı (0 başarısız, 1 başarılı)
df = pd.read_csv("signal_history.csv")

filter_model = SignalFilter()
filter_model.train(df)

# Canlı sinyal örneği
latest = df.iloc[[-1]]
confidence = filter_model.predict(latest)
print(f"Güven skoru: {confidence[0]:.2f}")
