"""
Base strategy system for the trading system.
Defines the strategy registry and base strategy class.
"""
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Type
from abc import ABC, abstractmethod


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
        Generate signals for each row in the dataframe.
        
        Args:
            df: DataFrame with indicator data
            
        Returns:
            Series with signal values (1 for long, -1 for short, 0 for no signal)
        """
        # Initialize signals series with zeros
        signals = pd.Series(0, index=df.index)
        
        # Validate dataframe
        if not self.validate_dataframe(df):
            return signals
        
        # Process each row
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Generate conditions
            conditions = self.generate_conditions(df, row, i)
            
            # Get long and short conditions
            long_conditions = conditions.get('long', [])
            short_conditions = conditions.get('short', [])
            
            # Check if any conditions exist
            if not long_conditions and not short_conditions:
                continue
                
            # Calculate signal based on conditions
            if long_conditions and all(long_conditions):
                signals.iloc[i] = 1  # Long signal
            elif short_conditions and all(short_conditions):
                signals.iloc[i] = -1  # Short signal
        
        return signals
    
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
    
    def generate_signals(self, df: pd.DataFrame, strategy_names: List[str], 
                       params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Generate signals using multiple strategies.
        
        Args:
            df: DataFrame with indicator data
            strategy_names: List of strategy names to use
            params: Optional parameters for each strategy as {strategy_name: params_dict}
            
        Returns:
            DataFrame with signals columns
        """
        result_df = df.copy()
        params = params or {}
        
        # Initialize signal columns
        result_df["long_signal"] = False
        result_df["short_signal"] = False
        result_df["strategy_name"] = None
        
        # Apply each strategy
        for name in strategy_names:
            # Get strategy params if provided
            strategy_params = params.get(name, {})
            
            # Create strategy instance
            strategy = self.registry.create_strategy(name, strategy_params)
            
            if strategy:
                try:
                    # Generate signals
                    signals = strategy.generate_signals(result_df)
                    
                    # Convert signals to long/short format
                    for i in range(len(signals)):
                        if signals.iloc[i] > 0 and not result_df["long_signal"].iloc[i]:
                            result_df.loc[result_df.index[i], "long_signal"] = True
                            result_df.loc[result_df.index[i], "strategy_name"] = name
                        elif signals.iloc[i] < 0 and not result_df["short_signal"].iloc[i]:
                            result_df.loc[result_df.index[i], "short_signal"] = True
                            result_df.loc[result_df.index[i], "strategy_name"] = name
                            
                except Exception as e:
                    logger.error(f"Error generating signals with strategy {name}: {e}")
            else:
                logger.warning(f"Strategy {name} not found in registry")
        
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