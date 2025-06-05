# main.py CUSTOM_CONFIG options - Choose one for each CSV config

# =============================================================================
# BEAR MARKET STRATEGIES (for current ETHUSDT bear market)
# =============================================================================

# 1. BEAR_SHORT_BALANCED (Best from diagnostic) - 257 trades, +37% ROI
BEAR_SHORT_BALANCED = {
    "risk_per_trade": 0.02,           # 2% risk
    "leverage": 3.0,                  # 3x leverage
    "sl_multiplier": 1.3,             # 1.3x ATR stop loss
    "tp_multiplier": 2.5,             # 2.5x ATR take profit (2:1 R:R)
    "position_direction": {"Long": False, "Short": True},
    "enable_tp_commission_filter": True,
    "min_tp_commission_ratio": 12.0,   # Balanced filtering
    "max_commission_impact_pct": 10.0,
    "min_position_size": 1000.0,
    "min_net_rr_ratio": 1.5,
    "max_holding_bars": 30,
}

# 2. BEAR_SHORT_STRICT (Fewer trades, higher quality)
BEAR_SHORT_STRICT = {
    "risk_per_trade": 0.025,          # Slightly higher risk
    "leverage": 3.0,
    "sl_multiplier": 1.2,             # Tighter stops
    "tp_multiplier": 3.0,             # Higher targets
    "position_direction": {"Long": False, "Short": True},
    "enable_tp_commission_filter": True,
    "min_tp_commission_ratio": 18.0,   # Stricter filtering
    "max_commission_impact_pct": 8.0,
    "min_position_size": 1500.0,       # Higher minimum
    "min_net_rr_ratio": 2.0,           # Better R:R
    "max_holding_bars": 25,
}

# 3. BEAR_SHORT_RELAXED (More trades, lower quality)
BEAR_SHORT_RELAXED = {
    "risk_per_trade": 0.015,          # Lower risk
    "leverage": 2.5,                  # Lower leverage
    "sl_multiplier": 1.5,             # Wider stops
    "tp_multiplier": 2.2,             # Lower targets
    "position_direction": {"Long": False, "Short": True},
    "enable_tp_commission_filter": True,
    "min_tp_commission_ratio": 8.0,    # Relaxed filtering
    "max_commission_impact_pct": 12.0,
    "min_position_size": 800.0,        # Lower minimum
    "min_net_rr_ratio": 1.3,
    "max_holding_bars": 35,
}

# =============================================================================
# BULL MARKET STRATEGIES (for testing on BTCUSDT or bull periods)
# =============================================================================

# 4. BULL_LONG_AGGRESSIVE (High risk, high reward)
BULL_LONG_AGGRESSIVE = {
    "risk_per_trade": 0.035,          # 3.5% risk
    "leverage": 5.0,                  # Higher leverage in bull
    "sl_multiplier": 1.1,             # Tight stops
    "tp_multiplier": 4.0,             # High targets
    "position_direction": {"Long": True, "Short": False},
    "enable_tp_commission_filter": True,
    "min_tp_commission_ratio": 15.0,
    "max_commission_impact_pct": 8.0,
    "min_position_size": 1200.0,
    "min_net_rr_ratio": 2.5,
    "max_holding_bars": 45,
}

# 5. BULL_LONG_CONSERVATIVE (Lower risk bull strategy)
BULL_LONG_CONSERVATIVE = {
    "risk_per_trade": 0.02,
    "leverage": 3.0,
    "sl_multiplier": 1.5,             # Wider stops for safety
    "tp_multiplier": 3.0,
    "position_direction": {"Long": True, "Short": False},
    "enable_tp_commission_filter": True,
    "min_tp_commission_ratio": 20.0,   # Stricter quality
    "max_commission_impact_pct": 7.0,
    "min_position_size": 1500.0,
    "min_net_rr_ratio": 1.8,
    "max_holding_bars": 50,
}

# =============================================================================
# SIDEWAYS/BREAKOUT STRATEGIES
# =============================================================================

# 6. SIDEWAYS_BOTH_BREAKOUT (Both directions, quick trades)
SIDEWAYS_BOTH_BREAKOUT = {
    "risk_per_trade": 0.018,
    "leverage": 4.0,
    "sl_multiplier": 0.8,             # Very tight stops
    "tp_multiplier": 2.0,             # Quick profits
    "position_direction": {"Long": True, "Short": True},
    "enable_tp_commission_filter": True,
    "min_tp_commission_ratio": 25.0,   # Very strict for breakouts
    "max_commission_impact_pct": 6.0,
    "min_position_size": 2000.0,       # Higher minimum for quality
    "min_net_rr_ratio": 2.2,
    "max_holding_bars": 15,            # Quick exits
}

# =============================================================================
# SCALPING STRATEGIES (Higher frequency)
# =============================================================================

# 7. SCALP_SHORT_FAST (5m timeframe, fast trades)
SCALP_SHORT_FAST = {
    "risk_per_trade": 0.01,           # Lower risk for scalping
    "leverage": 6.0,                  # Higher leverage for small moves
    "sl_multiplier": 0.6,             # Very tight stops
    "tp_multiplier": 1.5,             # Quick targets
    "position_direction": {"Long": False, "Short": True},
    "enable_tp_commission_filter": True,
    "min_tp_commission_ratio": 30.0,   # Very strict for scalping
    "max_commission_impact_pct": 4.0,
    "min_position_size": 3000.0,       # Higher minimum
    "min_net_rr_ratio": 2.0,
    "max_holding_bars": 8,             # Very quick exits
}

# =============================================================================
# CONSERVATIVE STRATEGIES (Lower frequency, higher quality)
# =============================================================================

# 8. CONSERVATIVE_LONG_SLOW (1h timeframe, patient trades)
CONSERVATIVE_LONG_SLOW = {
    "risk_per_trade": 0.025,
    "leverage": 2.0,                  # Lower leverage for safety
    "sl_multiplier": 2.0,             # Wide stops
    "tp_multiplier": 5.0,             # High targets
    "position_direction": {"Long": True, "Short": False},
    "enable_tp_commission_filter": True,
    "min_tp_commission_ratio": 40.0,   # Very strict filtering
    "max_commission_impact_pct": 3.0,
    "min_position_size": 5000.0,       # High minimum for quality
    "min_net_rr_ratio": 3.0,           # High R:R requirement
    "max_holding_bars": 100,           # Long holding allowed
}

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
HOW TO USE:

1. Save the CSV as: backtest/config/config_combinations.csv

2. In main.py, choose one config based on market conditions:

For current bear market (ETHUSDT):
CUSTOM_CONFIG = BEAR_SHORT_BALANCED

For bull market testing:
CUSTOM_CONFIG = BULL_LONG_AGGRESSIVE

For sideways market:
CUSTOM_CONFIG = SIDEWAYS_BOTH_BREAKOUT

3. Run single backtest:
RUN_MODE = "single"

4. Or run batch with all configs:
RUN_MODE = "batch"
"""

# =============================================================================
# CURRENT RECOMMENDATION FOR YOUR BEAR MARKET
# =============================================================================

# Use this for current ETHUSDT bear market:
CUSTOM_CONFIG = BEAR_SHORT_BALANCED

# Alternative if you want fewer, higher quality trades:
# CUSTOM_CONFIG = BEAR_SHORT_STRICT

# Alternative if you want more frequent trading:
# CUSTOM_CONFIG = BEAR_SHORT_RELAXED

Strategy Configs:
8 farklı strateji config'i hazır:
Bear Market (Current):

BEAR_SHORT_BALANCED ← Recommended (257 trades, +37% ROI)
BEAR_SHORT_STRICT ← Fewer trades, higher quality
BEAR_SHORT_RELAXED ← More trades, lower quality

Bull Market:

BULL_LONG_AGGRESSIVE ← High risk/reward
BULL_LONG_CONSERVATIVE ← Safe bull strategy

Sideways Market:

SIDEWAYS_BOTH_BREAKOUT ← Quick breakout trades

Scalping:

SCALP_SHORT_FAST ← 5m fast trades

Conservative:

CONSERVATIVE_LONG_SLOW ← 1h patient trades

3. Test Options:


Clean Configuration Structure: Fixed version with proper CSV formatting and consistent JSON structure across all 50 configurations.
Strategy Distribution:

Scalping Pro (1-10): Ultra-fast strategies with progressive ATR multipliers (1.2-2.1) for 1m/3m/5m timeframes
Swing Trader (11-25): Medium-term strategies with comprehensive indicator sets and progressive ADX thresholds (20-34)
Momentum Day (26-35): Intraday momentum with stochastic analysis and volume confirmation
Mean Reversion (36-45): Counter-trend strategies with Bollinger band variations (2.0-2.9 std dev)
Advanced MTF (46-50): Sophisticated multi-timeframe confluence strategies with SuperTrend integration

Key Improvements in V7:

✅ Clean JSON formatting - No broken strings or malformed data
✅ Proper CSV escaping - All quotes and commas handled correctly
✅ Consistent parameter scaling - Logical progression in all thresholds
✅ Complete data integrity - All 50 rows with 8 columns each
✅ Professional parameter ranges - Industry-standard values throughout

Ready for Batch Testing: Each configuration now has clean, parseable JSON structures that can be directly loaded into your trading system for systematic backtesting.
💡 Proaktif Öneriler:

Performance Testing: Start with scalping configs (1-10) for high-frequency testing, then move to swing configs (11-25) for longer timeframes
Risk Management: Mean reversion configs (36-45) require careful position sizing due to higher drawdown potential
Market Regime Filtering: Advanced MTF configs (46-50) work best in trending markets - consider adding regime detection

Bu temiz versiyon artık batch testlerde sorunsuz çalışacak. Her config unique ve mantıklı parameter kombinasyonları içeriyor.RetryClaude can make mistakes. Please double-check responses.