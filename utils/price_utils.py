import requests
import os
import sys  
from tenacity import retry, stop_after_attempt, wait_fixed


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_environment, get_config
load_environment()
config = get_config()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_current_market_price(symbol: str) -> float:
    """
    Binance Futures API'den anlık fiyat bilgisini çeker.
    Hata durumunda 3 kez, her seferinde 2 saniye arayla yeniden dener.
    """
    BINANCE_FUTURES_BASE_URL= config["BINANCE_FUTURES_BASE_URL"]
    url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/ticker/price?symbol={symbol}"
    print(f"Fetching current market price for {symbol} from {url}")
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    return float(data["price"])

def is_entry_price_valid(current_price: float, expected_price: float) -> bool:
    """
    Sinyal üretim fiyatı ile canlı fiyat arasındaki fark toleransı içinde mi?
    """
    try:
        max_deviation_pct = float(config["MAX_DEVIATION_PERCENT"])
        deviation = abs(current_price - expected_price) / expected_price * 100
        return deviation <= max_deviation_pct
    except Exception as e:
        print(f"❌ Fiyat sapma kontrolü hatası: {e}")
        return False
