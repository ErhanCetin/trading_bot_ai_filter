from runner_data_sync import fetch_and_store
import os

class NoOpFetcher:
    def fetch(self, symbol, interval):
        print("🔁 Docker ortamı: Veri çekme pas geçildi.")

class LocalFetcher:
    def fetch(self, symbol, interval):
        fetch_and_store(symbol, interval)

def get_fetcher():
    env = os.getenv("ENV", "local").lower()
    return LocalFetcher() if env == "local" else NoOpFetcher()
