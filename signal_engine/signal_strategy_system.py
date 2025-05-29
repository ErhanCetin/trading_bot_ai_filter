"""
Base strategy system for the trading system.
Defines the strategy registry and base strategy class.
"""
import pandas as pd
import logging  
logging.basicConfig
from typing import Dict, Any, List, Optional, Tuple, Union, Type
from abc import ABC, abstractmethod


# Logger ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    name = "base_strategy"
    display_name = "Base Strategy"
    description = "Base class for all strategies"
    category = "base"
    
    default_params = {}
    required_indicators = []
    optional_indicators = []
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with parameters.
        
        Args:
            params: Optional parameters to override defaults
        """
        self.params = self.default_params.copy() if hasattr(self, 'default_params') else {}
        if params:
            self.params.update(params)
    
    @abstractmethod
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate signal conditions.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        pass
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on flexible confirmation logic
        
        The key insight: Not ALL conditions need to be true,
        but ENOUGH conditions should confirm the signal
        """
        signals = pd.Series(0, index=df.index)
        
        if not self.validate_dataframe(df):
            return signals
        
        # Get strategy-specific confirmation requirements
        min_confirmations = self.params.get("confirmation_count", 
                                          len(self.required_indicators) // 2)
        confidence_threshold = self.params.get("confidence_threshold", 0.6)
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            try:
                conditions = self.generate_conditions(df, row, i)
                signal = self._evaluate_signal_strength(conditions, 
                                                      min_confirmations, 
                                                      confidence_threshold)
                signals.iloc[i] = signal
                
            except Exception as e:
                logger.debug(f"Error generating signal at index {i}: {e}")
                continue
        
        return signals
    
    def _evaluate_signal_strength(self, conditions: Dict[str, List[bool]], 
                                min_confirmations: int, 
                                confidence_threshold: float) -> int:
        """
        Evaluate signal strength based on condition confirmations
        
        Args:
            conditions: Dict with 'long' and 'short' condition lists
            min_confirmations: Minimum number of confirming conditions
            confidence_threshold: Percentage of conditions that must be true
            
        Returns:
            1 for long signal, -1 for short signal, 0 for no signal
        """
        long_conditions = [c for c in conditions.get('long', []) if c is not None]
        short_conditions = [c for c in conditions.get('short', []) if c is not None]
        
        # Calculate confirmation scores
        long_score = self._calculate_confirmation_score(long_conditions, 
                                                       min_confirmations, 
                                                       confidence_threshold)
        short_score = self._calculate_confirmation_score(short_conditions, 
                                                        min_confirmations, 
                                                        confidence_threshold)
        
        # Generate signal based on stronger score
        if long_score > short_score and long_score > 0:
            return 1
        elif short_score > long_score and short_score > 0:
            return -1
        else:
            return 0
    
    def _calculate_confirmation_score(self, conditions: List[bool], 
                                    min_confirmations: int, 
                                    confidence_threshold: float) -> float:
        """
        Calculate confirmation score for a set of conditions
        
        Logic:
        1. Count true conditions
        2. Check if minimum confirmations met
        3. Check if confidence threshold met
        4. Return weighted score
        """
        if not conditions:
            return 0.0
        
        true_count = sum(1 for c in conditions if c)
        total_count = len(conditions)
        confidence = true_count / total_count
        
        # Must meet both minimum confirmations AND confidence threshold
        if true_count >= min_confirmations and confidence >= confidence_threshold:
            # Return weighted score (higher is stronger)
            return confidence * (true_count / max(total_count, 1))
        
        return 0.0
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        return all(col in df.columns for col in self.required_indicators)


class StrategyRegistry:
    """Registry for strategy classes."""
    
    def __init__(self):
        """Initialize the registry."""
        self._strategies = {}
    
    def register(self, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class.
        
        Args:
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"Class {strategy_class.__name__} is not a subclass of BaseStrategy")
        
        self._strategies[strategy_class.name] = strategy_class
    
    def get_strategy(self, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get a strategy class by name.
        
        Args:
            name: Name of the strategy class
            
        Returns:
            Strategy class or None if not found
        """
        return self._strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """
        Get all registered strategy classes.
        
        Returns:
            Dictionary of strategy names to strategy classes
        """
        return self._strategies.copy()
    
    def get_strategies_by_category(self, category: str) -> Dict[str, Type[BaseStrategy]]:
        """
        Get all strategy classes in a category.
        
        Args:
            category: Category to filter by
            
        Returns:
            Dictionary of strategy names to strategy classes
        """
        return {name: cls for name, cls in self._strategies.items() if cls.category == category}
    
    def create_strategy(self, name: str, params: Optional[Dict[str, Any]] = None) -> Optional[BaseStrategy]:
        """
        Create a strategy instance by name.
        
        Args:
            name: Name of the strategy
            params: Optional parameters to pass to the strategy
            
        Returns:
            Strategy instance or None if not found
        """
        strategy_class = self.get_strategy(name)
        if strategy_class:
            return strategy_class(params)
        return None
    
    
class StrategyManager:
    """Stratejilerin yönetimini ve sinyal üretimini koordine eden sınıf."""
    
    def __init__(self, registry: StrategyRegistry = None):
        """
        Initialize the strategy manager.
        
        Args:
            registry: Optional strategy registry to use
        """
        self.registry = registry or StrategyRegistry()
    
    def add_strategy(self, strategy_name: str, params: Optional[Dict[str, Any]] = None, weight: float = 1.0) -> None:
        """
        Strateji ekler (isim, parametreler ve ağırlık ile)
        
        Args:
            strategy_name: Strateji adı
            params: Strateji parametreleri
            weight: Strateji ağırlığı (ensemble için)
        """
        self._strategies_to_use = getattr(self, '_strategies_to_use', [])
        self._strategy_params = getattr(self, '_strategy_params', {})
        self._strategy_weights = getattr(self, '_strategy_weights', {})
        
        self._strategies_to_use.append(strategy_name)
        if params:
            self._strategy_params[strategy_name] = params
        
        # Weight desteği eklendi
        self._strategy_weights[strategy_name] = weight

    def generate_signals(self, df: pd.DataFrame, strategy_names: List[str] = None, 
                    params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Generate signals using multiple strategies with weight support.
        
        Args:
            df: DataFrame with indicator data
            strategy_names: List of strategy names to use (None for previously added)
            params: Optional parameters for each strategy as {strategy_name: params_dict}
            
        Returns:
            DataFrame with signals columns
        """
        result_df = df.copy()
        
        # Daha önce add_strategy ile eklenen stratejileri kullan
        if strategy_names is None:
            strategy_names = getattr(self, '_strategies_to_use', [])
            params = getattr(self, '_strategy_params', {})
        else:
            params = params or {}
        
        # Strategy weights'leri al
        strategy_weights = getattr(self, '_strategy_weights', {})
        
        # Initialize signal columns with weighted voting logic
        result_df["long_signal"] = False
        result_df["short_signal"] = False
        result_df["strategy_name"] = None
        result_df["signal_strength"] = 0.0
        
        # For ensemble logic - collect weighted votes
        long_votes = 0.0
        short_votes = 0.0
        total_weight = 0.0
        contributing_strategies = []
        
        # Apply each strategy
        for name in strategy_names:
            # Get strategy params and weight
            strategy_params = params.get(name, {})
            strategy_weight = strategy_weights.get(name, 1.0)
            
            # Create strategy instance
            strategy = self.registry.create_strategy(name, strategy_params)
            
            if strategy:
                try:
                    # Generate signals for this strategy
                    signals = strategy.generate_signals(result_df)
                    
                    # Count positive signals and apply weight
                    long_signal_count = (signals > 0).sum()
                    short_signal_count = (signals < 0).sum()
                    
                    if long_signal_count > 0 or short_signal_count > 0:
                        # Calculate strategy contribution
                        long_contribution = (long_signal_count / len(signals)) * strategy_weight
                        short_contribution = (short_signal_count / len(signals)) * strategy_weight
                        
                        long_votes += long_contribution
                        short_votes += short_contribution
                        total_weight += strategy_weight
                        contributing_strategies.append(name)
                        
                        # Apply individual strategy signals (for non-ensemble mode)
                        for i in range(len(signals)):
                            current_signal_strength = result_df["signal_strength"].iloc[i]
                            
                            if signals.iloc[i] > 0:
                                # Long signal - update if this has higher weighted strength
                                weighted_strength = strategy_weight * abs(signals.iloc[i])
                                if weighted_strength > current_signal_strength:
                                    result_df.loc[result_df.index[i], "long_signal"] = True
                                    result_df.loc[result_df.index[i], "short_signal"] = False
                                    result_df.loc[result_df.index[i], "strategy_name"] = name
                                    result_df.loc[result_df.index[i], "signal_strength"] = weighted_strength
                                    
                            elif signals.iloc[i] < 0:
                                # Short signal - update if this has higher weighted strength
                                weighted_strength = strategy_weight * abs(signals.iloc[i])
                                if weighted_strength > current_signal_strength:
                                    result_df.loc[result_df.index[i], "long_signal"] = False
                                    result_df.loc[result_df.index[i], "short_signal"] = True
                                    result_df.loc[result_df.index[i], "strategy_name"] = name
                                    result_df.loc[result_df.index[i], "signal_strength"] = weighted_strength
                            
                except Exception as e:
                    logger.error(f"Error generating signals with strategy {name}: {e}")
            else:
                logger.warning(f"Strategy {name} not found in registry")
        
        # Apply ensemble voting logic if multiple strategies contributed
        if len(contributing_strategies) > 1 and total_weight > 0:
            vote_threshold = 0.5  # Configurable threshold for ensemble decisions
            
            # Normalize votes
            long_vote_strength = long_votes / total_weight
            short_vote_strength = short_votes / total_weight
            
            # Apply ensemble decisions where individual strategy signals are weak
            weak_signal_threshold = 0.5
            
            for i in range(len(result_df)):
                current_strength = result_df["signal_strength"].iloc[i]
                
                # If current signal is weak, consider ensemble vote
                if current_strength < weak_signal_threshold:
                    if long_vote_strength > vote_threshold and long_vote_strength > short_vote_strength:
                        result_df.loc[result_df.index[i], "long_signal"] = True
                        result_df.loc[result_df.index[i], "short_signal"] = False
                        result_df.loc[result_df.index[i], "strategy_name"] = "ensemble"
                        result_df.loc[result_df.index[i], "signal_strength"] = long_vote_strength
                        
                    elif short_vote_strength > vote_threshold and short_vote_strength > long_vote_strength:
                        result_df.loc[result_df.index[i], "long_signal"] = False
                        result_df.loc[result_df.index[i], "short_signal"] = True
                        result_df.loc[result_df.index[i], "strategy_name"] = "ensemble"
                        result_df.loc[result_df.index[i], "signal_strength"] = short_vote_strength
        
        return result_df
    
    def list_available_strategies(self) -> Dict[str, List[str]]:
        """
        Get a list of available strategies by category.
        
        Returns:
            Dictionary of categories to list of strategy names
        """
        result = {}
        
        # Get all strategies
        all_strategies = self.registry.get_all_strategies()
        
        # Group by category
        for name, strategy_class in all_strategies.items():
            category = strategy_class.category
            
            if category not in result:
                result[category] = []
                
            result[category].append(name)
        
        return result
    
    def get_strategy_details(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy details or None if not found
        """
        strategy_class = self.registry.get_strategy(strategy_name)
        
        if not strategy_class:
            return None
            
        return {
            "name": strategy_class.name,
            "display_name": strategy_class.display_name,
            "description": strategy_class.description,
            "category": strategy_class.category,
            "default_params": strategy_class.default_params,
            "required_indicators": strategy_class.required_indicators,
            "optional_indicators": getattr(strategy_class, "optional_indicators", [])
        }    