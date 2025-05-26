"""
Optimized Config Generator for 70%+ ROI Trading System
Generates coin-specific configurations with risk management focus
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class CoinProfile:
    """Coin characteristics for optimization"""
    symbol: str
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    liquidity_score: float  # 0-1
    trend_persistence: float  # 0-1  
    noise_ratio: float  # 0-1
    typical_move_pct: float  # Average significant move %
    risk_grade: str  # 'A', 'B', 'C', 'D'


class OptimizedConfigGenerator:
    """
    Generates optimized configurations for 70%+ ROI with controlled risk
    """
    
    def __init__(self):
        # Define coin profiles based on historical analysis
        self.coin_profiles = {
            'PEPEUSDT': CoinProfile(
                symbol='PEPEUSDT',
                volatility_level='extreme',
                liquidity_score=0.7,
                trend_persistence=0.3,
                noise_ratio=0.8,
                typical_move_pct=25.0,
                risk_grade='D'
            ),
            'DOGEUSDT': CoinProfile(
                symbol='DOGEUSDT', 
                volatility_level='high',
                liquidity_score=0.9,
                trend_persistence=0.6,
                noise_ratio=0.6,
                typical_move_pct=12.0,
                risk_grade='C'
            ),
            'SOLUSDT': CoinProfile(
                symbol='SOLUSDT',
                volatility_level='medium',
                liquidity_score=0.95,
                trend_persistence=0.8,
                noise_ratio=0.4,
                typical_move_pct=8.0,
                risk_grade='B'
            ),
            # Add more stable coins for better risk/reward
            'BTCUSDT': CoinProfile(
                symbol='BTCUSDT',
                volatility_level='low',
                liquidity_score=1.0,
                trend_persistence=0.9,
                noise_ratio=0.2,
                typical_move_pct=4.0,
                risk_grade='A'
            ),
            'ETHUSDT': CoinProfile(
                symbol='ETHUSDT',
                volatility_level='medium',
                liquidity_score=1.0, 
                trend_persistence=0.85,
                noise_ratio=0.3,
                typical_move_pct=6.0,
                risk_grade='A'
            )
        }
        
        # Strategy effectiveness by volatility and risk grade
        self.strategy_effectiveness = {
            'trend_following': {
                'high_vol': 0.6, 'medium_vol': 0.8, 'low_vol': 0.9,
                'grade_A': 0.9, 'grade_B': 0.8, 'grade_C': 0.7, 'grade_D': 0.5
            },
            'overextended_reversal': {
                'high_vol': 0.8, 'medium_vol': 0.7, 'low_vol': 0.6,
                'grade_A': 0.7, 'grade_B': 0.8, 'grade_C': 0.7, 'grade_D': 0.9
            },
            'volatility_breakout': {
                'high_vol': 0.9, 'medium_vol': 0.7, 'low_vol': 0.5,
                'grade_A': 0.6, 'grade_B': 0.8, 'grade_C': 0.9, 'grade_D': 0.8
            },
            'range_breakout': {
                'high_vol': 0.5, 'medium_vol': 0.8, 'low_vol': 0.9,
                'grade_A': 0.8, 'grade_B': 0.9, 'grade_C': 0.7, 'grade_D': 0.5
            }
        }

    def calculate_roi_potential(self, profile: CoinProfile, config: Dict) -> float:
        """
        Calculate expected ROI based on coin profile and config
        Returns expected ROI percentage
        """
        base_roi = 45.0  # Base ROI expectation
        
        # Volatility impact on ROI potential
        vol_multipliers = {
            'low': 1.2,      # Lower vol = more consistent gains
            'medium': 1.4,   # Medium vol = good balance
            'high': 1.6,     # High vol = high potential but risky
            'extreme': 1.8   # Extreme vol = very high potential but very risky
        }
        
        roi = base_roi * vol_multipliers.get(profile.volatility_level, 1.0)
        
        # Risk grade adjustment (higher grade = more reliable ROI)
        risk_adjustments = {'A': 1.3, 'B': 1.2, 'C': 1.0, 'D': 0.8}
        roi *= risk_adjustments.get(profile.risk_grade, 1.0)
        
        # Strategy selection impact
        strategies = config.get('strategies', {})
        strategy_boost = 1.0
        
        for strategy_name in strategies.keys():
            effectiveness = self.strategy_effectiveness.get(strategy_name, {})
            vol_effect = effectiveness.get(f"{profile.volatility_level}_vol", 0.7)
            grade_effect = effectiveness.get(f"grade_{profile.risk_grade}", 0.7)
            strategy_boost *= (1 + (vol_effect * grade_effect - 0.5) * 0.3)
        
        roi *= strategy_boost
        
        # Filter quality impact
        filters = config.get('filters', {})
        filter_quality = len(filters) * 0.05  # Each filter adds 5% potential
        roi *= (1 + filter_quality)
        
        return min(roi, 150.0)  # Cap at 150% to be realistic

    def calculate_risk_score(self, profile: CoinProfile, config: Dict) -> float:
        """
        Calculate risk score (0-100, lower is better)
        """
        base_risk = 30.0
        
        # Volatility impact on risk
        vol_risks = {
            'low': 1.0,
            'medium': 1.3, 
            'high': 1.8,
            'extreme': 2.5
        }
        
        risk = base_risk * vol_risks.get(profile.volatility_level, 1.5)
        
        # Risk grade impact (better grade = lower risk)
        grade_adjustments = {'A': 0.7, 'B': 0.8, 'C': 1.0, 'D': 1.4}
        risk *= grade_adjustments.get(profile.risk_grade, 1.2)
        
        # Strategy aggressiveness
        strategies = config.get('strategies', {})
        
        # Check for aggressive parameters
        if 'volatility_breakout' in strategies:
            atr_mult = strategies['volatility_breakout'].get('atr_multiplier', 2.0)
            if atr_mult < 2.5:  # Too aggressive
                risk *= 1.3
        
        if 'trend_following' in strategies:
            adx_threshold = strategies['trend_following'].get('adx_threshold', 25)
            if adx_threshold < 25:  # Too loose
                risk *= 1.2
        
        # Filter protection (more filters = lower risk)
        filters = config.get('filters', {})
        filter_protection = len(filters) * 0.1
        risk *= (1 - min(filter_protection, 0.4))  # Max 40% risk reduction
        
        return min(risk, 95.0)  # Cap at 95%

    def generate_conservative_config(self, profile: CoinProfile) -> Dict[str, Any]:
        """Generate conservative config optimized for consistent ROI"""
        
        # Base indicators with conservative settings
        indicators = {
            "ema": {"periods": [21, 50, 200]},  # Longer periods for stability
            "sma": {"periods": [50, 200]},
            "rsi": {"periods": [21]},  # Longer RSI for less noise
            "atr": {"window": 21}
        }
        
        # Add coin-specific indicators
        if profile.volatility_level in ['high', 'extreme']:
            indicators["bollinger_bands"] = {"period": 20, "std": 2.5}  # Wider bands
        
        # Conservative strategies based on coin characteristics
        strategies = {}
        
        # Always include trend following with conservative settings
        if profile.risk_grade in ['A', 'B']:
            strategies["trend_following"] = {
                "adx_threshold": 30,  # Higher threshold for quality trends
                "rsi_threshold": 55,  # Slight bias but not extreme
            }
        
        # Add reversal for high-volatility coins
        if profile.volatility_level in ['high', 'extreme']:
            strategies["overextended_reversal"] = {
                "rsi_overbought": 78,  # More extreme levels
                "rsi_oversold": 22,
                "consecutive_candles": 4  # More confirmation
            }
        
        # Breakout for medium volatility coins
        if profile.volatility_level in ['medium', 'high'] and profile.risk_grade in ['A', 'B', 'C']:
            strategies["range_breakout"] = {
                "range_threshold": min(0.04, profile.typical_move_pct * 0.002),
                "breakout_factor": 1.015  # Conservative breakout confirmation
            }
        
        # Conservative filters with multiple layers
        filters = {
            "market_regime": {},
            "volatility_regime": {
                "atr_threshold": 2.0 if profile.volatility_level == 'extreme' else 1.5
            },
            "pattern_recognition_filter": {
                "confidence_threshold": 0.75 if profile.risk_grade in ['A', 'B'] else 0.80
            },
            "min_checks": 3,  # Require multiple confirmations
            "min_strength": 65  # Higher strength requirement
        }
        
        # Conservative strength calculation
        strength = {
            "risk_reward_strength": {
                "risk_factor": 1.5,  # Conservative risk management
                "reward_factor": 2.5  # Target higher rewards
            },
            "market_context_strength": {},
            "indicator_confirmation_strength": {
                "base_strength": 55  # Higher base requirement
            }
        }
        
        return {
            "indicators_long": indicators,
            "indicators_short": indicators,
            "strategies": strategies,
            "filters": filters,
            "strength": strength
        }

    def generate_balanced_config(self, profile: CoinProfile) -> Dict[str, Any]:
        """Generate balanced config for optimal risk/reward"""
        
        # Adaptive indicators
        indicators = {
            "ema": {"periods": [12, 26, 50]},
            "sma": {"periods": [50, 200]},
            "rsi": {"periods": [14]},
            "atr": {"window": 14}
        }
        
        # Add volatility-specific indicators
        if profile.noise_ratio > 0.6:
            indicators["bollinger_bands"] = {"period": 20, "std": 2.0}
            indicators["keltner"] = {"ema_window": 20, "atr_multiplier": 2.0}
        
        # Balanced strategies
        strategies = {}
        
        # Trend following with adaptive thresholds
        strategies["trend_following"] = {
            "adx_threshold": 25 + (profile.noise_ratio * 10),  # Adapt to noise
            "rsi_threshold": 50
        }
        
        # Volatility breakout with profile-based settings
        strategies["volatility_breakout"] = {
            "atr_multiplier": 2.0 + (profile.volatility_level == 'extreme') * 0.5,
            "volume_surge_factor": 1.5 + (profile.liquidity_score * 0.5)
        }
        
        # Range breakout for suitable profiles
        if profile.trend_persistence < 0.7:
            strategies["range_breakout"] = {
                "range_threshold": profile.typical_move_pct * 0.003,
                "breakout_factor": 1.012
            }
        
        # Reversal for high-volatility coins
        if profile.volatility_level in ['high', 'extreme']:
            strategies["overextended_reversal"] = {
                "rsi_overbought": 75,
                "rsi_oversold": 25
            }
        
        # Balanced filters
        filters = {
            "market_regime": {},
            "volatility_regime": {"atr_threshold": 1.5},
            "pattern_recognition_filter": {"confidence_threshold": 0.70},
            "min_checks": 2,
            "min_strength": 55
        }
        
        # Balanced strength calculation
        strength = {
            "risk_reward_strength": {"risk_factor": 1.2},
            "market_context_strength": {},
            "indicator_confirmation_strength": {"base_strength": 50}
        }
        
        return {
            "indicators_long": indicators,
            "indicators_short": indicators,
            "strategies": strategies,
            "filters": filters,
            "strength": strength
        }

    def generate_aggressive_config(self, profile: CoinProfile) -> Dict[str, Any]:
        """Generate aggressive config for high ROI (only for suitable coins)"""
        
        # Only for high-quality, high-volatility coins
        if profile.risk_grade not in ['A', 'B'] or profile.volatility_level not in ['medium', 'high']:
            return self.generate_balanced_config(profile)
        
        # Fast-response indicators
        indicators = {
            "ema": {"periods": [9, 21, 50]},
            "sma": {"periods": [20, 50]},
            "rsi": {"periods": [14]},
            "atr": {"window": 14},
            "bollinger_bands": {"period": 20, "std": 2.0}
        }
        
        # Aggressive strategies
        strategies = {
            "trend_following": {
                "adx_threshold": 22,
                "rsi_threshold": 50
            },
            "volatility_breakout": {
                "atr_multiplier": 1.8,
                "volume_surge_factor": 1.3
            },
            "overextended_reversal": {
                "rsi_overbought": 72,
                "rsi_oversold": 28
            }
        }
        
        # More permissive filters but still with safeguards
        filters = {
            "market_regime": {},
            "volatility_regime": {"atr_threshold": 1.2},
            "pattern_recognition_filter": {"confidence_threshold": 0.65},
            "min_checks": 2,
            "min_strength": 50
        }
        
        # Aggressive strength calculation
        strength = {
            "risk_reward_strength": {"risk_factor": 1.0},
            "market_context_strength": {},
            "indicator_confirmation_strength": {"base_strength": 45}
        }
        
        return {
            "indicators_long": indicators,
            "indicators_short": indicators,
            "strategies": strategies,
            "filters": filters,
            "strength": strength
        }

    def generate_optimized_configs(self) -> List[Dict[str, Any]]:
        """Generate all optimized configurations targeting 70%+ ROI"""
        
        configs = []
        
        for symbol, profile in self.coin_profiles.items():
            # Generate multiple config variants per coin
            config_variants = []
            
            # Conservative config (lower risk, steady ROI)
            conservative = self.generate_conservative_config(profile)
            config_variants.append(('conservative', conservative))
            
            # Balanced config (balanced risk/reward)
            balanced = self.generate_balanced_config(profile)
            config_variants.append(('balanced', balanced))
            
            # Aggressive config (higher risk, higher ROI potential)
            if profile.risk_grade in ['A', 'B']:
                aggressive = self.generate_aggressive_config(profile)
                config_variants.append(('aggressive', aggressive))
            
            # Evaluate each variant
            for variant_name, config in config_variants:
                roi_potential = self.calculate_roi_potential(profile, config)
                risk_score = self.calculate_risk_score(profile, config)
                
                # Only include configs with 70%+ ROI potential and acceptable risk
                if roi_potential >= 70.0 and risk_score <= 60.0:
                    config_entry = {
                        'config_id': f"{symbol.lower().replace('usdt', '')}_{variant_name}",
                        'symbol': symbol,
                        'interval': '5m',
                        'variant': variant_name,
                        'expected_roi': round(roi_potential, 1),
                        'risk_score': round(risk_score, 1),
                        'indicators_long': json.dumps(config['indicators_long']),
                        'indicators_short': json.dumps(config['indicators_short']),
                        'strategies': json.dumps(config['strategies']),
                        'filters': json.dumps(config['filters']),
                        'strength': json.dumps(config['strength'])
                    }
                    configs.append(config_entry)
        
        # Sort by ROI/Risk ratio (higher is better)
        configs.sort(key=lambda x: x['expected_roi'] / x['risk_score'], reverse=True)
        
        return configs

    def export_to_csv(self, filename: str = "optimized_config_combinations.csv") -> pd.DataFrame:
        """Export optimized configurations to CSV"""
        
        configs = self.generate_optimized_configs()
        
        # Convert to DataFrame with required columns
        df_data = []
        for config in configs:
            row = {
                'config_id': config['config_id'],
                'indicators_long': config['indicators_long'],
                'indicators_short': config['indicators_short'],
                'strategies': config['strategies'],
                'filters': config['filters'],
                'strength': config['strength'],
                'symbol': config['symbol'],
                'interval': config['interval']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False)
        
        # Print summary
        print("🎯 OPTIMIZED CONFIGURATIONS GENERATED")
        print(f"📊 Total configs: {len(df)}")
        print(f"💰 ROI range: {min(c['expected_roi'] for c in configs):.1f}% - {max(c['expected_roi'] for c in configs):.1f}%")
        print(f"⚠️  Risk range: {min(c['risk_score'] for c in configs):.1f}% - {max(c['risk_score'] for c in configs):.1f}%")
        print(f"\n🏆 TOP 5 CONFIGS (by ROI/Risk ratio):")
        
        for i, config in enumerate(configs[:5], 1):
            ratio = config['expected_roi'] / config['risk_score']
            print(f"  {i}. {config['config_id']} | ROI: {config['expected_roi']}% | Risk: {config['risk_score']}% | Ratio: {ratio:.2f}")
        
        return df

    def print_detailed_analysis(self):
        """Print detailed analysis of generated configurations"""
        
        configs = self.generate_optimized_configs()
        
        print("\n" + "="*80)
        print("🔬 DETAILED CONFIGURATION ANALYSIS")
        print("="*80)
        
        # Group by coin
        by_coin = {}
        for config in configs:
            symbol = config['symbol']
            if symbol not in by_coin:
                by_coin[symbol] = []
            by_coin[symbol].append(config)
        
        for symbol, coin_configs in by_coin.items():
            profile = self.coin_profiles[symbol]
            print(f"\n📊 {symbol} (Risk Grade: {profile.risk_grade}, Volatility: {profile.volatility_level})")
            print("-" * 50)
            
            for config in coin_configs:
                print(f"  🎯 {config['config_id']}")
                print(f"     💰 Expected ROI: {config['expected_roi']}%")
                print(f"     ⚠️  Risk Score: {config['risk_score']}%")
                print(f"     📈 ROI/Risk Ratio: {config['expected_roi']/config['risk_score']:.2f}")
                
                # Parse strategies
                strategies = json.loads(config['strategies'])
                print(f"     🚀 Strategies: {', '.join(strategies.keys())}")
                print()


# Usage Example
if __name__ == "__main__":
    generator = OptimizedConfigGenerator()
    
    # Generate and export optimized configurations
    df = generator.export_to_csv("optimized_high_roi_configs.csv")
    
    # Print detailed analysis
    generator.print_detailed_analysis()
    
    print(f"\n✅ Configurations exported to optimized_high_roi_configs.csv")
    print(f"📁 Total configurations: {len(df)}")