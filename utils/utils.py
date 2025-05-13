
import pandas as pd

def transform_row_to_indicator_config(row):
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
