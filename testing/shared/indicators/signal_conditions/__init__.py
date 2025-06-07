"""
Signal Conditions Package - Main Manager
Centralized management of all signal condition checkers across all phases
"""
import logging
from typing import Dict, Any, List

# Import base conditions (Phase 1)
from .base_conditions import (
    BaseConditionChecker,
    BasicConditionChecker,
    VolumeConditionChecker, 
    MACDConditionChecker,
    BollingerConditionChecker,
    ATRConditionChecker,
    TrendConditionChecker
)

# Import advanced conditions (Phase 2)
from .advanced_conditions import (
    SupertrendConditionChecker,
    HeikinAshiConditionChecker,
    IchimokuConditionChecker,
    AdaptiveRSIConditionChecker,
    MTFEMAConditionChecker
)

# Import feature conditions (Phase 3)
from .feature_conditions import (
    PriceActionConditionChecker,
    VolumePriceConditionChecker,
    MomentumConditionChecker,
    SupportResistanceConditionChecker
)

# Import regime conditions (Phase 4)
from .regime_conditions import (
    MarketRegimeConditionChecker,
    VolatilityRegimeConditionChecker,
    TrendStrengthConditionChecker,
    StatisticalConditionChecker
)

logger = logging.getLogger(__name__)


class SignalConditionsManager:
    """
    Main manager for all signal condition checking across all phases.
    Handles automatic loading of appropriate checkers based on available phases.
    """
    
    def __init__(self, enable_phases: List[int] = None):
        """
        Initialize with configurable phase support.
        
        Args:
            enable_phases: List of phases to enable (1-4). If None, enables all available.
        """
        if enable_phases is None:
            enable_phases = [1, 2, 3, 4]  # Enable all phases by default
        
        self.enabled_phases = enable_phases
        self.checkers = []
        
        # Phase 1 - Base Indicators (Always enabled if requested)
        if 1 in enable_phases:
            self.checkers.extend([
                BasicConditionChecker(),
                VolumeConditionChecker(),
                MACDConditionChecker(),
                BollingerConditionChecker(),
                ATRConditionChecker(),
                TrendConditionChecker()
            ])
            logger.info("✅ Phase 1 (Base) conditions loaded")
        
        # Phase 2 - Advanced Indicators
        if 2 in enable_phases:
            self.checkers.extend([
                SupertrendConditionChecker(),
                HeikinAshiConditionChecker(),
                IchimokuConditionChecker(),
                AdaptiveRSIConditionChecker(),
                MTFEMAConditionChecker()
            ])
            logger.info("✅ Phase 2 (Advanced) conditions loaded")
        
        # Phase 3 - Feature Indicators
        if 3 in enable_phases:
            self.checkers.extend([
                PriceActionConditionChecker(),
                VolumePriceConditionChecker(),
                MomentumConditionChecker(),
                SupportResistanceConditionChecker()
            ])
            logger.info("✅ Phase 3 (Feature) conditions loaded")
        
        # Phase 4 - Regime Indicators
        if 4 in enable_phases:
            self.checkers.extend([
                MarketRegimeConditionChecker(),
                VolatilityRegimeConditionChecker(),
                TrendStrengthConditionChecker(),
                StatisticalConditionChecker()
            ])
            logger.info("✅ Phase 4 (Regime) conditions loaded")
        
        # Build type to checker mapping
        self.type_checker_map = {}
        for checker in self.checkers:
            for condition_type in checker.get_supported_types():
                if condition_type in self.type_checker_map:
                    logger.warning(f"⚠️ Condition type '{condition_type}' already registered, overriding")
                self.type_checker_map[condition_type] = checker
        
        total_types = len(self.type_checker_map)
        total_checkers = len(self.checkers)
        
        logger.info(f"🔧 SignalConditionsManager initialized:")
        logger.info(f"   📊 Phases enabled: {enable_phases}")
        logger.info(f"   🔧 Checkers loaded: {total_checkers}")
        logger.info(f"   📋 Condition types: {total_types}")
    
    def check_conditions(self, row, conditions: List[Dict[str, Any]]) -> bool:
        """
        Check if all conditions are met for a given row.
        
        Args:
            row: Data row to check (pandas Series)
            conditions: List of conditions to check
            
        Returns:
            True if all conditions are met
        """
        if not conditions:
            return False
        
        for condition in conditions:
            condition_type = condition.get('type', '')
            
            if condition_type not in self.type_checker_map:
                logger.warning(f"⚠️ Unknown condition type: {condition_type}")
                logger.debug(f"Available types: {list(self.type_checker_map.keys())}")
                continue
            
            checker = self.type_checker_map[condition_type]
            
            try:
                if not checker.check(row, condition):
                    return False
            except Exception as e:
                logger.warning(f"⚠️ Error checking condition {condition}: {e}")
                return False
        
        return True
    
    def calculate_signal_strength(self, row, conditions: List[Dict[str, Any]]) -> float:
        """
        Calculate signal strength based on conditions.
        
        Args:
            row: Data row (pandas Series)
            conditions: List of conditions
            
        Returns:
            Signal strength (0-100)
        """
        if not conditions:
            return 0.0
        
        total_strength = 0.0
        condition_count = len(conditions)
        
        for condition in conditions:
            strength = 50.0  # Base strength
            
            condition_type = condition.get('type', '')
            
            # Enhanced strength calculation based on condition type
            if condition_type == 'threshold':
                column = condition.get('column', '')
                if column in row.index and not row[column] is None:
                    value = row[column]
                    threshold = condition.get('value', 0)
                    if threshold != 0:
                        difference = abs(value - threshold)
                        strength += min(50.0, difference * 2)
            
            elif condition_type in ['price_cross_above', 'price_cross_below']:
                # Crossover strength based on distance
                price_column = condition.get('price_column', 'close')
                indicator_column = condition.get('indicator_column', '')
                
                if price_column in row.index and indicator_column in row.index:
                    price = row[price_column]
                    indicator = row[indicator_column]
                    if indicator != 0:
                        distance_pct = abs(price - indicator) / indicator * 100
                        strength += min(30.0, distance_pct * 10)
            
            elif condition_type == 'volume_confirmation':
                # Volume strength
                if 'volume' in row.index and 'volume_ma' in row.index:
                    volume = row['volume']
                    volume_ma = row.get('volume_ma', volume)
                    if volume_ma > 0:
                        volume_ratio = volume / volume_ma
                        strength += min(40.0, (volume_ratio - 1) * 50)
            
            total_strength += strength
        
        # Average and cap at 100
        average_strength = total_strength / condition_count if condition_count > 0 else 0
        return min(100.0, average_strength)
    
    def get_supported_condition_types(self) -> List[str]:
        """Get all supported condition types."""
        return list(self.type_checker_map.keys())
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get information about loaded phases."""
        phase_info = {
            'enabled_phases': self.enabled_phases,
            'total_checkers': len(self.checkers),
            'total_condition_types': len(self.type_checker_map),
            'condition_types_by_phase': {}
        }
        
        # Map condition types to phases
        phase_checkers = {
            1: [BasicConditionChecker, VolumeConditionChecker, MACDConditionChecker, 
                BollingerConditionChecker, ATRConditionChecker, TrendConditionChecker],
            2: [SupertrendConditionChecker, HeikinAshiConditionChecker, IchimokuConditionChecker,
                AdaptiveRSIConditionChecker, MTFEMAConditionChecker],
            3: [PriceActionConditionChecker, VolumePriceConditionChecker, MomentumConditionChecker,
                SupportResistanceConditionChecker],
            4: [MarketRegimeConditionChecker, VolatilityRegimeConditionChecker, 
                TrendStrengthConditionChecker, StatisticalConditionChecker]
        }
        
        for phase, checker_classes in phase_checkers.items():
            if phase in self.enabled_phases:
                phase_types = []
                for checker_class in checker_classes:
                    # Create instance to get supported types
                    temp_checker = checker_class()
                    phase_types.extend(temp_checker.get_supported_types())
                phase_info['condition_types_by_phase'][f'phase_{phase}'] = phase_types
        
        return phase_info
    
    def validate_conditions(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a list of conditions.
        
        Args:
            conditions: List of conditions to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'total_conditions': len(conditions),
            'valid_conditions': 0,
            'invalid_conditions': 0,
            'unknown_types': [],
            'missing_parameters': [],
            'warnings': []
        }
        
        for i, condition in enumerate(conditions):
            condition_type = condition.get('type', '')
            
            if not condition_type:
                validation_results['invalid_conditions'] += 1
                validation_results['missing_parameters'].append(f"Condition {i}: missing 'type'")
                continue
            
            if condition_type not in self.type_checker_map:
                validation_results['invalid_conditions'] += 1
                validation_results['unknown_types'].append(condition_type)
                continue
            
            # Basic parameter validation
            required_params = self._get_required_params(condition_type)
            missing_params = [param for param in required_params if param not in condition]
            
            if missing_params:
                validation_results['warnings'].append(
                    f"Condition {i} ({condition_type}): missing parameters {missing_params}"
                )
            
            validation_results['valid_conditions'] += 1
        
        validation_results['is_valid'] = validation_results['invalid_conditions'] == 0
        
        return validation_results
    
    def _get_required_params(self, condition_type: str) -> List[str]:
        """Get required parameters for a condition type."""
        # Define required parameters for different condition types
        param_map = {
            'threshold': ['column', 'operator', 'value'],
            'price_cross_above': ['price_column', 'indicator_column'],
            'price_cross_below': ['price_column', 'indicator_column'],
            'line_cross_above': ['line1', 'line2'],
            'line_cross_below': ['line1', 'line2'],
            'volume_confirmation': ['min_volume_ratio'],
            'zero_line_filter': ['above_zero'],
            'histogram_positive': ['column'],
            'histogram_negative': ['column'],
            # Add more as needed
        }
        
        return param_map.get(condition_type, [])


# Export main classes
__all__ = [
    'SignalConditionsManager',
    'BaseConditionChecker',
    # Phase 1
    'BasicConditionChecker',
    'VolumeConditionChecker', 
    'MACDConditionChecker',
    'BollingerConditionChecker',
    'ATRConditionChecker',
    'TrendConditionChecker',
    # Phase 2
    'SupertrendConditionChecker',
    'HeikinAshiConditionChecker',
    'IchimokuConditionChecker',
    'AdaptiveRSIConditionChecker',
    'MTFEMAConditionChecker',
    # Phase 3
    'PriceActionConditionChecker',
    'VolumePriceConditionChecker',
    'MomentumConditionChecker',
    'SupportResistanceConditionChecker',
    # Phase 4
    'MarketRegimeConditionChecker',
    'VolatilityRegimeConditionChecker',
    'TrendStrengthConditionChecker',
    'StatisticalConditionChecker'
]