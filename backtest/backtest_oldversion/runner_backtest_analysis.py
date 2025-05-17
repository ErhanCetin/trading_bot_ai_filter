import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union, Dict, List, Any
import pandas as pd
import json
from sqlalchemy import text

from env_loader import load_environment, get_config
from db.postgresql import engine

# Mevcut adapter-based imports (geriye dönük uyumluluk için)
from signal_engine import add_indicators
from signal_engine import generate_signals
from signal_engine import apply_signal_strength
from signal_engine import filter_signals

# Plugin sistemi imports (doğrudan plugin sistemi kullanmak isterseniz)
from signal_engine.indicators import registry as indicator_registry
from signal_engine.signal_indicator_plugin_system import IndicatorManager
from signal_engine.strategies import registry as strategy_registry  
from signal_engine.signal_strategy_system import SignalGeneratorManager
from signal_engine.calculators import registry as strength_registry
from signal_engine.signal_strength_system import SignalStrengthManager
from signal_engine.rules import registry as filter_registry
from signal_engine.signal_filter_system import SignalFilterManager

from utils.price_utils import is_entry_price_valid

from backtest.backtest_result_writer import write_backtest_results
from backtest.backtest_result_analyzer import analyze_results
from backtest.backtest_result_plotter import plot_results
from backtest.backtest_diagnostics import run_backtest_diagnostics

# Load environment and config
load_environment()
config = get_config()

SYMBOL = config["SYMBOL"]
INTERVAL = config["INTERVAL"]
ACCOUNT_BALANCE = config["ACCOUNT_BALANCE"]
RISK_PER_TRADE = config["RISK_PER_TRADE"]
SL_MULTIPLIER = config["SL_MULTIPLIER"]
TP_MULTIPLIER = config["TP_MULTIPLIER"]
LEVERAGE = config["LEVERAGE"]

RESULTS_DIR = "backtest/results"
TRADES_CSV = os.path.join(RESULTS_DIR, "trades.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_stats.csv")
SIGNAL_CSV = os.path.join(RESULTS_DIR, "signal_breakdown.csv")
USE_PLUGIN_SYSTEM = True  # Plugin sistemini doğrudan kullanmak için True yapın


# 📦 Load price data
def load_price_data(symbol: str, interval: str) -> pd.DataFrame:
    """
    Veritabanından fiyat verilerini yükler
    
    Args:
        symbol: İşlem sembolü (örn. "BTCUSDT")
        interval: Zaman aralığı (örn. "1m", "5m", "1h")
        
    Returns:
        Fiyat verilerini içeren DataFrame
    """
    query = f"""
    SELECT * FROM kline_data
    WHERE symbol = '{symbol}' AND interval = '{interval}'
    ORDER BY open_time
    """
    df = pd.read_sql(text(query), engine)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def convert_legacy_to_plugin_config(indicators: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Legacy indicator config formatını plugin sistemi formatına dönüştürür
    
    Args:
        indicators: Eski format indikatör konfigürasyonu
        
    Returns:
        Plugin sistemi formatında konfigürasyon
    """
    plugin_config = {}
    
    # EMA
    if "EMA_FAST" in indicators or "EMA_SLOW" in indicators:
        plugin_config["ema"] = {
            "fast_period": indicators.get("EMA_FAST", 12),
            "slow_period": indicators.get("EMA_SLOW", 26)
        }
    
    # RSI
    if "RSI" in indicators:
        plugin_config["rsi"] = {
            "period": indicators["RSI"]
        }
    
    # MACD
    if "MACD" in indicators and indicators["MACD"]:
        plugin_config["macd"] = {}
    
    # ATR
    if "ATR" in indicators:
        plugin_config["atr"] = {
            "period": indicators["ATR"]
        }
    
    # OBV
    if "OBV" in indicators and indicators["OBV"]:
        plugin_config["obv"] = {}
    
    # ADX
    if "ADX" in indicators:
        plugin_config["adx"] = {
            "period": indicators["ADX"]
        }
    
    # CCI
    if "CCI" in indicators:
        plugin_config["cci"] = {
            "period": indicators["CCI"]
        }
    
    # SUPER TREND
    if "SUPER_TREND" in indicators:
        st_config = indicators["SUPER_TREND"]
        plugin_config["supertrend"] = {
            "period": int(st_config.get("period", 10)),
            "multiplier": float(st_config.get("multiplier", 3))
        }
    
    # BOLLINGER BANDS
    if "BOLLINGER" in indicators:
        bb_config = indicators["BOLLINGER"]
        plugin_config["bollinger"] = {
            "period": bb_config.get("length", 20),
            "std_dev": bb_config.get("stddev", 2)
        }
    
    # DONCHIAN CHANNEL
    if "DONCHIAN" in indicators:
        dc_config = indicators["DONCHIAN"]
        plugin_config["donchian"] = {
            "period": dc_config.get("period", 20)
        }
    
    # Z-SCORE
    if "Z_SCORE" in indicators:
        zs_config = indicators["Z_SCORE"]
        plugin_config["zscore"] = {
            "period": zs_config.get("length", 20)
        }
    
    return plugin_config


def analyze_short_conditions(df: pd.DataFrame) -> None:
    """
    Short sinyal koşullarını analiz eder
    
    Args:
        df: İndikatörleri içeren DataFrame
    """
    print("📊 SHORT sinyali koşulları dağılımı:")
    
    if "ema_fast" in df.columns and "ema_slow" in df.columns:
        print("✅ ema_fast < ema_slow:", (df["ema_fast"] < df["ema_slow"]).sum())
    
    if "rsi" in df.columns:
        print("✅ rsi < 50:", (df["rsi"] < 50).sum())
    
    if "macd" in df.columns:
        print("✅ macd < 0:", (df["macd"] < 0).sum())
    
    if "supertrend" in df.columns:
        print("✅ supertrend == False:", (df["supertrend"] == False).sum())
    
    if "adx" in df.columns:
        print("✅ adx > 20:", (df["adx"] > 20).sum())
    
    if "di_neg" in df.columns and "di_pos" in df.columns:
        print("✅ di_neg > di_pos:", (df["di_neg"] > df["di_pos"]).sum())

    all_passed = (
        (df["ema_fast"] < df["ema_slow"]) &
        (df["rsi"] < 50) &
        (df["macd"] < 0) &
        ((df["supertrend"] == False) if "supertrend" in df.columns else True) &
        (df["adx"] > 20) &
        (df["di_neg"] > df["di_pos"])
    )
    print("🔍 Short koşulunu geçen satır sayısı:", all_passed.sum())
    print("✅ short_signal True sayısı:", df['short_signal'].sum())


def process_indicators_with_plugin_system(df: pd.DataFrame, indicators: Dict[str, Any]) -> pd.DataFrame:
    """
    Plugin sistemini kullanarak indikatörleri hesaplar ve sinyalleri üretir
    
    Args:
        df: İşlenecek fiyat verisi DataFrame
        indicators: İndikatör konfigürasyonu
        
    Returns:
        İndikatörler ve sinyaller eklenmiş DataFrame
    """
    # Plugin sistemi yapısını kullan
    indicator_manager = IndicatorManager(indicator_registry)
    signal_manager = SignalGeneratorManager(strategy_registry)
    strength_manager = SignalStrengthManager(strength_registry)
    filter_manager = SignalFilterManager(filter_registry)
    
    # Legacy konfigürasyonu plugin formatına dönüştür
    plugin_config = convert_legacy_to_plugin_config(indicators)
    
    # İndikatörleri ekle
    for indicator_name, params in plugin_config.items():
        indicator_manager.add_indicator(indicator_name, params)
    
    # İndikatörleri hesapla
    df = indicator_manager.calculate_indicators(df)
    
    # Stratejileri ekle
    signal_manager.add_strategy("trend_following")
    signal_manager.add_strategy("oscillator_signals")
    signal_manager.add_strategy("volatility_breakout")
    
    # Sinyalleri üret
    df = signal_manager.generate_signals(df)
    
    # Sinyal gücünü hesapla
    strength_manager.add_calculator("trend_indicators")
    strength_manager.add_calculator("oscillator_levels")
    strength_manager.add_calculator("volatility_measures")
    df = strength_manager.apply_signal_strength(df)
    
    # Sinyalleri filtrele
    filter_manager.add_rule("rsi_threshold")
    filter_manager.add_rule("macd_confirmation")
    filter_manager.add_rule("atr_volatility")
    filter_manager.set_min_checks_required(2)
    filter_manager.set_min_strength_required(3)
    df = filter_manager.filter_signals(df)
    
    return df


def run_backtest_with_config(
    symbol: str, 
    interval: str, 
    indicators: Dict[str, Any] = None, 
    return_trades_only: bool = False, 
    config_id: Union[int, str] = None
) -> List[Dict[str, Any]]:
    """
    Belirli bir konfigürasyonla backtest çalıştırır
    
    Args:
        symbol: İşlem sembolü
        interval: Zaman aralığı 
        indicators: İndikatör konfigürasyonu
        return_trades_only: Sadece trade'leri dönmek için True
        config_id: Konfigürasyon ID'si
        
    Returns:
        Trade'ler listesi veya None
    """
    df = load_price_data(symbol, interval)
    
    if USE_PLUGIN_SYSTEM:
        # Plugin sistemini doğrudan kullan
        df = process_indicators_with_plugin_system(df, indicators or {})
    else:
        # Mevcut adapter'ları kullan (geriye dönük uyumluluk)
        df = add_indicators(df, custom_config=indicators)
        df = generate_signals(df)
        df = apply_signal_strength(df)
        df = filter_signals(df)  

    df["open_time_fmt"] = pd.to_datetime(df["open_time"], unit="ms")

    trades = []
    balance = ACCOUNT_BALANCE

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if row["signal_strength"] < 3:
            continue

        direction = "LONG" if row["long_signal"] else "SHORT" if row["short_signal"] else None
        if direction is None:
            continue

        entry_price = row["close"]
        atr = row["atr"]
        sl = entry_price - atr * SL_MULTIPLIER if direction == "LONG" else entry_price + atr * SL_MULTIPLIER
        tp = entry_price + atr * TP_MULTIPLIER if direction == "LONG" else entry_price - atr * TP_MULTIPLIER
        high = next_row["high"]
        low = next_row["low"]

        if direction == "LONG":
            outcome = "TP" if high >= tp else "SL" if low <= sl else "OPEN"
        else:
            outcome = "TP" if low <= tp else "SL" if high >= sl else "OPEN"

        rr_ratio = TP_MULTIPLIER / SL_MULTIPLIER
        gain_pct = rr_ratio if outcome == "TP" else -1 if outcome == "SL" else 0
        position_size = balance * LEVERAGE
        gain_usd = (gain_pct / 100) * position_size
        balance += gain_usd

        trades.append({
            "config_id": config_id,
            "time": row["open_time_fmt"],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": next_row["close"],
            "atr": atr,
            "rr_ratio": rr_ratio,
            "outcome": outcome,
            "gain_pct": gain_pct,
            "gain_usd": gain_usd,
            "balance": balance,
            "rsi": row.get("rsi"),
            "macd": row.get("macd"),
            "obv": row.get("obv"),
            "supertrend": row.get("supertrend"),
            "cci": row.get("cci"),
            "adx": row.get("adx"),
            "di_pos": row.get("di_pos"),
            "di_neg": row.get("di_neg"),
            "signal_strength": row.get("signal_strength"),
            "signal_passed_filter": row.get("signal_passed_filter"),
            "indicator_config": json.dumps(indicators)
        })      

    if return_trades_only:
        return trades

    # Standalone kullanım için sonuçları yaz
    analyze_others(trades)
    print("✅ Full backtest completed.")
    return trades


def analyze_others(trades):
    """
    Tüm analiz ve çıktı işlemlerini gerçekleştirir.
    trades: trade verileri (Liste veya DataFrame olabilir)
    """
    # DataFrame kontrolü ekleyelim
    if isinstance(trades, pd.DataFrame):
        if trades.empty:
            print("❌ 'trades' DataFrame'i boş, analiz yapılmadı.")
            return
        trades_df = trades
    else:
        # Liste olduğunu varsayalım
        if not trades:
            print("❌ 'trades' listesi boş, analiz yapılmadı.")
            return
        # Liste ise DataFrame'e çevirelim
        trades_df = pd.DataFrame(trades)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Eğer pandas DataFrame ise, önce liste formatına çevirelim
    if isinstance(trades, pd.DataFrame):
        trades_list = trades.to_dict('records')
        write_backtest_results(trades_list, TRADES_CSV)
    else:
        write_backtest_results(trades, TRADES_CSV)
    
    analyze_results(TRADES_CSV, SUMMARY_CSV)
    plot_results(TRADES_CSV, RESULTS_DIR)
    run_backtest_diagnostics()
    print("✅ Full backtest completed.")


# 🚀 Run with env config
if __name__ == "__main__":
    run_backtest_with_config(SYMBOL, INTERVAL, config_id="env")