from prometheus_client import start_http_server, Gauge
import random
import time

# Metrikler
signal_count = Gauge("signal_count", "Number of total signals")
strong_signals = Gauge("strong_signal_count", "Number of strong signals")
api_errors = Gauge("binance_api_errors", "Number of Binance API errors")

def simulate_metrics():
    while True:
        signal_count.set(random.randint(50, 100))
        strong_signals.set(random.randint(10, 50))
        api_errors.set(random.randint(0, 5))
        time.sleep(10)

if __name__ == "__main__":
    start_http_server(8000)  # Prometheus bu porttan veri çeker
    simulate_metrics()
