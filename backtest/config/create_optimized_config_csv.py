# config_loader.py - YENİ FONKSİYON EKLEYİN

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
import sys

# Root dizinini Python yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def create_optimized_config_csv():
    """Yüksek ROI için optimize edilmiş config dosyası oluştur"""
    
    configs = []
    config_id = 1
    
    # ✅ YÜKSEK PERFORMANSLI İNDİKATÖR KOMBİNASYONLARI
    indicator_combinations = [
        # Trend + Momentum kombinasyonu
        {
            "long": {
                "ema": {"periods": [8, 21]},
                "rsi": {"periods": [14]}, 
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "supertrend": {"atr_period": 10, "atr_multiplier": 2.5}
            },
            "short": {
                "ema": {"periods": [8, 21]},
                "rsi": {"periods": [14]},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9}, 
                "supertrend": {"atr_period": 10, "atr_multiplier": 2.5}
            }
        },
        # Volatility breakout kombinasyonu
        {
            "long": {
                "bollinger": {"window": 20, "window_dev": 2.0},
                "atr": {"window": 14},
                "rsi": {"periods": [21]},
                "stochastic": {"window": 14, "smooth_window": 3, "d_window": 3}
            },
            "short": {
                "bollinger": {"window": 20, "window_dev": 2.0},
                "atr": {"window": 14}, 
                "rsi": {"periods": [21]},
                "stochastic": {"window": 14, "smooth_window": 3, "d_window": 3}
            }
        }
    ]
    
    # ✅ YÜKSEK PERFORMANSLI STRATEJİ KOMBİNASYONLARI
    strategy_combinations = [
        {
            "trend_following": {"weight": 2.0, "adx_threshold": 20},
            "momentum_breakout": {"weight": 1.5},
            "volatility_expansion": {"weight": 1.0}
        },
        {
            "adaptive_trend": {"weight": 2.5, "adx_min_threshold": 15, "adx_max_threshold": 35},
            "momentum_reversal": {"weight": 1.0},
            "volatility_contraction": {"weight": 0.8}
        }
    ]
    
    # ✅ OPTİMİZE FİLTRE AYARLARI
    filter_combinations = [
        {
            "market_regime": {},
            "dynamic_threshold_filter": {"base_threshold": 0.4, "volatility_impact": 0.3},
            "min_checks": 2,
            "min_strength": 40  # Daha düşük eşik
        },
        {
            "volatility_regime": {},
            "trend_strength_filter": {"min_trend_strength": 0.3},
            "min_checks": 1,  # Daha esnek
            "min_strength": 35
        }
    ]
    
    for indicators in indicator_combinations:
        for strategies in strategy_combinations:
            for filters in filter_combinations:
                configs.append({
                    "config_id": config_id,
                    "indicators_long": json.dumps(indicators["long"]),
                    "indicators_short": json.dumps(indicators["short"]),
                    "strategies": json.dumps(strategies),
                    "filters": json.dumps(filters),
                    "strength": json.dumps({
                        "market_context_strength": {"volatility_adjustment": True},
                        "momentum_strength": {"rsi_weight": 1.2, "macd_weight": 1.0}
                    }),
                    "symbol": "ETHFIUSDT",
                    "interval": "5m"
                })
                config_id += 1
    
    df = pd.DataFrame(configs)
    df.to_csv("backtest/config/high_performance_configs.csv", index=False)
    return df

create_optimized_config_csv()