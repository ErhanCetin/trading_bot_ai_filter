import os
import json
from dotenv import load_dotenv


_config = {}
_loaded = False  # iç kontrol değişkeni

def load_environment():
    load_dotenv()  # varsayılan .env dosyasını yükle
    global _loaded, _config
    if _loaded:
        return  # ✅ Zaten yüklendi, tekrar yapma
    env_type = os.getenv("PROJECT_GOLDEN_KICK_ENV", "local").lower()

    print(f"✅ ✅ Ortam türü: {env_type}")
    if env_type == "docker":
        env_file = ".env.docker"
    else:
        env_file = ".env.local"

    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)
        print(f"✅ Ortam değişkenleri yüklendi: {env_file}")
        _loaded = True  # artık tekrar yükleme

        _config["SYMBOL"] = os.getenv("SYMBOL")
        _config["INTERVAL"] = os.getenv("INTERVAL")
        _config["ACCOUNT_BALANCE"] = float(os.getenv("ACCOUNT_BALANCE", 22222.0))
        _config["RISK_PER_TRADE"] = float(os.getenv("RISK_PER_TRADE", 0.01))
        _config["SL_MULTIPLIER"] = float(os.getenv("SL_MULTIPLIER", 1.0))
        _config["TP_MULTIPLIER"] = float(os.getenv("TP_MULTIPLIER", 1.0))
        _config["LIMIT"] = float(os.getenv("LIMIT", 1000))
        _config["LEVERAGE"] = float(os.getenv("LEVERAGE", 10))
        _config["BINANCE_FUTURES_BASE_URL"] = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")
        _config["MAX_DEVIATION_PERCENT"] = os.getenv("MAX_DEVIATION_PERCENT", 1.0)
        _config["ATR_VOLATILITY_THRESHOLD"] = float(os.getenv("ATR_VOLATILITY_THRESHOLD", 0.0015))
        _config["ATR_PERIOD"] = int(os.getenv("ATR_PERIOD", 14))
        _config["RESULTS_DIR"] = os.getenv("RESULTS_DIR", "backtest/results")
        _config["DB_URL"] = os.getenv("DB_URL", "postgresql://localhost/crypto")
        _config["INITIAL_BALANCE"] = os.getenv("ACCOUNT_BALANCE", 11111.0)
        _config["COMMISSION_RATE"] = os.getenv("COMMISSION_RATE", 0.001)

        # Tüm yapılandırmaları yükle
        try:
            _config["INDICATORS_LONG"] = json.loads(os.getenv("INDICATORS_LONG", "{}"))
        except Exception as e:
            print("❌ Failed to parse INDICATORS_LONG:", e)
            _config["INDICATORS_LONG"] = {}

        try:
            _config["INDICATORS_SHORT"] = json.loads(os.getenv("INDICATORS_SHORT", "{}"))
        except Exception as e:
            print("❌ Failed to parse INDICATORS_SHORT:", e)
            _config["INDICATORS_SHORT"] = {}

        try:
            _config["STRATEGIES_CONFIG"] = json.loads(os.getenv("STRATEGIES_CONFIG", "{}"))
        except Exception as e:
            print("❌ Failed to parse STRATEGIES_CONFIG:", e)
            _config["STRATEGIES_CONFIG"] = {}

        try:
            _config["FILTER_CONFIG"] = json.loads(os.getenv("FILTER_CONFIG", "{}"))
        except Exception as e:
            print("❌ Failed to parse FILTER_CONFIG:", e)
            _config["FILTER_CONFIG"] = {}

        try:
            _config["STRENGTH_CONFIG"] = json.loads(os.getenv("STRENGTH_CONFIG", "{}"))
        except Exception as e:
            print("❌ Failed to parse STRENGTH_CONFIG:", e)
            _config["STRENGTH_CONFIG"] = {}

        try:
            _config["POSITION_DIRECTION"] = json.loads(os.getenv("POSITION_DIRECTION", "{}"))
        except Exception as e:
            print("❌ Failed to parse POSITION_DIRECTION:", e)
            _config["POSITION_DIRECTION"] = {"Long": True, "Short": True}

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


def get_strategies_config() -> dict:
    """Returns the strategies configuration."""
    if "STRATEGIES_CONFIG" not in _config:
        raise ValueError("❌ STRATEGIES_CONFIG tanımlı değil. load_environment() çağrıldı mı?")
    return _config["STRATEGIES_CONFIG"]


def get_filter_config() -> dict:
    """Returns the filter configuration."""
    if "FILTER_CONFIG" not in _config:
        raise ValueError("❌ FILTER_CONFIG tanımlı değil. load_environment() çağrıldı mı?")
    return _config["FILTER_CONFIG"]


def get_strength_config() -> dict:
    """Returns the strength calculator configuration."""
    if "STRENGTH_CONFIG" not in _config:
        raise ValueError("❌ STRENGTH_CONFIG tanımlı değil. load_environment() çağrıldı mı?")
    return _config["STRENGTH_CONFIG"]


def get_position_direction() -> dict:
    """Returns POSITION_DIRECTION as dict, example: {'Long': true, 'Short': false}"""
    if "POSITION_DIRECTION" not in _config:
        raise ValueError("❌ POSITION_DIRECTION tanımlı değil. load_environment() çağrıldı mı?")
    return _config["POSITION_DIRECTION"]


def convert_to_json(python_dict):
    """
    Python dictionary yapısını JSON formatına dönüştürür.
    
    Args:
        python_dict (dict): Python dictionary
        
    Returns:
        str: JSON formatında string
    """
    import json
    # Python dict'i JSON'a dönüştür
    return json.dumps(python_dict)


        # # Tüm yapılandırmaları yükle
        # try:
        #     _config["INDICATORS_LONG"] = json.loads(os.getenv("INDICATORS_LONG", "{}"))
        #     logger.info("✅ INDICATORS_LONG yapılandırması yüklendi.")
        # except json.JSONDecodeError as e:
        #     logger.error(f"❌ INDICATORS_LONG yapılandırması ayrıştırılamadı: {e}")
        #     logger.error("JSON formatına dikkat edin: Tüm özellik adları çift tırnak içinde olmalıdır.")
        #     _config["INDICATORS_LONG"] = {}
        # except Exception as e:
        #     logger.error(f"❌ INDICATORS_LONG yapılandırması yüklenirken hata: {e}")
        #     _config["INDICATORS_LONG"] = {}
