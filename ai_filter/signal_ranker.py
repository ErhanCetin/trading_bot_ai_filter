import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SignalFilter:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, df: pd.DataFrame):
        features = ["rsi", "macd", "atr", "obv", "ema_fast", "ema_slow"]
        target = "signal_success"  # geçmiş sinyalin başarılı olup olmadığı (0/1)

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def predict(self, latest_df: pd.DataFrame):
        features = ["rsi", "macd", "atr", "obv", "ema_fast", "ema_slow"]
        preds = self.model.predict_proba(latest_df[features])[:, 1]  # başarı olasılığı
        return preds
