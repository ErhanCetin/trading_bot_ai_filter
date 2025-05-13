import os
import json
from dotenv import load_dotenv


_config = {}
_loaded = False  # iç kontrol değişkeni

def load_environment():
    load_dotenv()  # varsayılan .env dosyasını yükle
    global _loaded
    if _loaded:
        return  # ✅ Zaten yüklendi, tekrar yapma
    env_type = PROJECT_ENV=os.getenv("PROJECT_GOLDEN_KICK_ENV", "local").lower()

    print(f"✅ ✅ Ortam türü: {env_type}")
    if env_type == "docker":
        env_file = ".env.docker"
    else:
        env_file = ".env.local"

    if os.path.exists(env_file):
        load_dotenv(env_file,override=True)
        print(f"✅ Ortam değişkenleri yüklendi: {env_file}")
        _loaded = True  # artık tekrar yükleme

        _config["SYMBOL"] = os.getenv("SYMBOL")
        _config["INTERVAL"] = os.getenv("INTERVAL")
        _config["ACCOUNT_BALANCE"] = float(os.getenv("ACCOUNT_BALANCE", 1000))
        _config["RISK_PER_TRADE"] = float(os.getenv("RISK_PER_TRADE", 0.01))
        _config["SL_MULTIPLIER"] = float(os.getenv("SL_MULTIPLIER", 1.0))
        _config["TP_MULTIPLIER"] = float(os.getenv("TP_MULTIPLIER", 1.0))
        _config["LIMIT"] = float(os.getenv("LIMIT", 1000))
        _config["LEVERAGE"] = float(os.getenv("LEVERAGE", 10))
        _config["BINANCE_FUTURES_BASE_URL"] = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")
        _config["MAX_DEVIATION_PERCENT"] = os.getenv("MAX_DEVIATION_PERCENT", 1.0),
        _config["ATR_VOLATILITY_THRESHOLD"] = float(os.getenv("ATR_VOLATILITY_THRESHOLD", 0.0015)), 
        
        
        # "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
        # "BINANCE_API_SECRET": os.getenv("BINANCE_API_SECRET"),
        # "TELEGRAM_API_KEY": os.getenv("TELEGRAM_API_KEY"),
        # "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
        # "DATABASE_URL": os.getenv("DATABASE_URL"),
        # "DATABASE_USER": os.getenv("DATABASE_USER"),
        # "DATABASE_PASSWORD": os.getenv("DATABASE_PASSWORD"),
        # "DATABASE_HOST": os.getenv("DATABASE_HOST"),
        # "DATABASE_PORT": os.getenv("DATABASE_PORT"),
        # "DATABASE_NAME": os.getenv("DATABASE_NAME"),

        # Long & Short Indicator configs
        try:
            _config["INDICATORS_LONG"] = json.loads(os.getenv("INDICATORS_LONG", "{}"))
            print("📊 Parsed INDICATORS_LONG:", _config["INDICATORS_LONG"])
        except Exception as e:
            print("❌ Failed to parse INDICATORS_LONG:", e)

        try:
            _config["INDICATORS_SHORT"] = json.loads(os.getenv("INDICATORS_SHORT", "{}"))
            print("📊 Parsed INDICATORS_SHORT:", _config["INDICATORS_SHORT"])
        except Exception as e:
            print("❌ Failed to parse INDICATORS_SHORT:", e)

        try:
            _config["POSITION_DIRECTION"] = json.loads(os.getenv("POSITION_DIRECTION", "{}"))
            print("📊 Parsed POSITION_DIRECTION:", _config["POSITION_DIRECTION"])
        except Exception as e:
            print("❌ Failed to parse POSITION_DIRECTION:", e)
    else:
        raise FileNotFoundError(f"❌ {env_file} bulunamadı. Lütfen oluşturun.")
    
    
def get_config():
  """Return loaded config dictionary."""
  if not _config:
        raise RuntimeError("❌ Config not loaded. Call load_environment() first.")
  return _config


def get_indicator_config(direction: str = "Long") -> dict:
    """Get the indicator configuration for the specified direction (Long or Short)."""
    if not direction:
        raise ValueError("❌ direction parametresi boş olamaz")
    if not isinstance(direction, str):  
        raise ValueError("❌ direction parametresi string olmalı") 

    direction = direction.capitalize()
    if direction not in ["Long", "Short"]:
        raise ValueError(f"❌ Unknown direction: {direction}")
    key = f"INDICATORS_{direction.upper()}"
    if key not in _config:
        raise ValueError(f"❌ {key} parametresi yüklenmemiş. load_environment() çağrıldı mı?")

    return _config[key]

def get_position_direction() -> dict:
    """Returns POSITION_DIRECTION as dict, example: {'Long': true, 'Short': false}"""
    if "POSITION_DIRECTION" not in _config:
        raise ValueError("❌ POSITION_DIRECTION tanımlı değil. load_environment() çağrıldı mı?")
    return _config["POSITION_DIRECTION"]