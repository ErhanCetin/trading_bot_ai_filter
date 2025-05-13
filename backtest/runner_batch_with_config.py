import os
import traceback
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

from backtest.runner_backtest_analysis import (
    run_backtest_with_config, 
    analyze_others, 
    convert_legacy_to_plugin_config
)
from env_loader import load_environment, get_config, get_position_direction

load_environment()
config = get_config()

CONFIG_FILE = "backtest/config/config_combinations.csv"
RESULTS_DIR = "backtest/results/batch"
RESULTS_CSV = os.path.join(RESULTS_DIR, "batch_results.csv")
TRADES_CSV = os.path.join(RESULTS_DIR, "batch_trades_all.csv")


def transform_row_to_indicator_config(row: pd.Series) -> Dict[str, Any]:
    """
    CSV konfigürasyon satırını legacy indikatör konfigürasyonuna dönüştürür.
    
    Args:
        row: CSV'den okunan konfigürasyon satırı
        
    Returns:
        Legacy format indikatör konfigürasyonu 
    """
    config = {}
    if not pd.isna(row.get("EMA_FAST")):
        config["EMA_FAST"] = int(row["EMA_FAST"])
    if not pd.isna(row.get("EMA_SLOW")):
        config["EMA_SLOW"] = int(row["EMA_SLOW"])
    if not pd.isna(row.get("RSI")):
        config["RSI"] = int(row["RSI"])
    if not pd.isna(row.get("MACD")):
        config["MACD"] = bool(row["MACD"])
    if not pd.isna(row.get("ATR")):
        config["ATR"] = int(row["ATR"])
    if not pd.isna(row.get("OBV")):
        config["OBV"] = bool(row["OBV"])
    if not pd.isna(row.get("CCI")):
        config["CCI"] = int(row["CCI"])
    if not pd.isna(row.get("ADX")):
        config["ADX"] = int(row["ADX"])
    if not pd.isna(row.get("SUPER_TREND_period")) and not pd.isna(row.get("SUPER_TREND_multiplier")):
        config["SUPER_TREND"] = {
            "period": int(row["SUPER_TREND_period"]),
            "multiplier": float(row["SUPER_TREND_multiplier"])
        }
    if not pd.isna(row.get("BOLLINGER_length")) and not pd.isna(row.get("BOLLINGER_stddev")):
        config["BOLLINGER"] = {
            "length": int(row["BOLLINGER_length"]),
            "stddev": float(row["BOLLINGER_stddev"])
        }
    if not pd.isna(row.get("DONCHIAN_period")):
        config["DONCHIAN"] = {
            "period": int(row["DONCHIAN_period"])
        }
    if not pd.isna(row.get("Z_SCORE_length")):
        config["Z_SCORE"] = {
            "length": int(row["Z_SCORE_length"])
        }
    return config


def run_batch():
    """
    CSV'den okunan konfigürasyonlarla batch backtest çalıştırır
    """
    df = pd.read_csv(CONFIG_FILE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    symbol = os.getenv("SYMBOL", "BTCUSDT")
    interval = os.getenv("INTERVAL", "5m")

    # Konfigürasyonları hazırla 
    indicator_config_map = {
        row["config_id"]: transform_row_to_indicator_config(row) for _, row in df.iterrows()
    }
    
    # Optionally convert to plugin system format
    # plugin_config_map = {
    #     config_id: convert_legacy_to_plugin_config(indicators) 
    #     for config_id, indicators in indicator_config_map.items()
    # }

    all_trades = []
    batch_summary = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_config = {
            executor.submit(run_backtest_with_config, symbol, interval, indicators, True, config_id): config_id
            for config_id, indicators in indicator_config_map.items()
        }

        for future in as_completed(future_to_config):
            config_id = future_to_config[future]
            try:
                trades = future.result()
                if trades:
                    df_trades = pd.DataFrame(trades)
                    all_trades.append(df_trades)
                    gain_sum = df_trades["gain_usd"].sum()
                    trade_count = len(df_trades)
                    win_rate = round((df_trades["outcome"] == "TP").sum() / trade_count * 100, 2) if trade_count > 0 else 0.0
                    batch_summary.append({
                        "config_id": config_id,
                        "total_gain_usd": gain_sum,
                        "total_trades": trade_count,
                        "win_rate": win_rate
                    })
                    print(f"✅ Config {config_id} completed: Gain ${gain_sum:.2f}, Trades {trade_count}, Win Rate {win_rate}%")
                else:
                    print(f"⚠️ Config {config_id} produced no trades.")
            except Exception as e:
                print(f"❌ Config {config_id} failed: {e}")
                traceback.print_exc()

    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        all_trades_df.to_csv(TRADES_CSV, index=False)
        print(f"📁 Saved trades to {TRADES_CSV}")
        
        if batch_summary:
            pd.DataFrame(batch_summary).to_csv(RESULTS_CSV, index=False)
            print(f"📁 Saved summary to {RESULTS_CSV}")
            # DataFrame'i doğrudan gönderiyoruz
            analyze_others(all_trades_df)


if __name__ == "__main__":
    run_batch()