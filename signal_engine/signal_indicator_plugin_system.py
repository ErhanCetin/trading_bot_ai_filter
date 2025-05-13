"""
Base indicator system for the trading system.
Defines the indicator registry and base indicator class.
"""
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Type
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseIndicator(ABC):
    """Base class for all indicators."""
    
    name = "base_indicator"
    display_name = "Base Indicator"
    description = "Base class for all indicators"
    category = "base"
    
    default_params = {}
    requires_columns = []
    output_columns = []
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the indicator with parameters.
        
        Args:
            params: Optional parameters to override defaults
        """
        self.params = self.default_params.copy() if hasattr(self, 'default_params') else {}
        if params:
            self.params.update(params)
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with indicator columns added
        """
        pass
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        return all(col in df.columns for col in self.requires_columns)


class IndicatorRegistry:
    """Registry for indicator classes."""
    
    def __init__(self):
        """Initialize the registry."""
        self._indicators = {}
    
    def register(self, indicator_class: Type[BaseIndicator]) -> None:
        """
        Register an indicator class.
        
        Args:
            indicator_class: Indicator class to register
        """
        if not issubclass(indicator_class, BaseIndicator):
            raise TypeError(f"Class {indicator_class.__name__} is not a subclass of BaseIndicator")
        
        self._indicators[indicator_class.name] = indicator_class
    
    def get_indicator(self, name: str) -> Optional[Type[BaseIndicator]]:
        """
        Get an indicator class by name.
        
        Args:
            name: Name of the indicator class
            
        Returns:
            Indicator class or None if not found
        """
        return self._indicators.get(name)
    
    def get_all_indicators(self) -> Dict[str, Type[BaseIndicator]]:
        """
        Get all registered indicator classes.
        
        Returns:
            Dictionary of indicator names to indicator classes
        """
        return self._indicators.copy()
    
    def get_indicators_by_category(self, category: str) -> Dict[str, Type[BaseIndicator]]:
        """
        Get all indicator classes in a category.
        
        Args:
            category: Category to filter by
            
        Returns:
            Dictionary of indicator names to indicator classes
        """
        return {name: cls for name, cls in self._indicators.items() if cls.category == category}
    
    def create_indicator(self, name: str, params: Optional[Dict[str, Any]] = None) -> Optional[BaseIndicator]:
        """
        Create an indicator instance by name.
        
        Args:
            name: Name of the indicator
            params: Optional parameters to pass to the indicator
            
        Returns:
            Indicator instance or None if not found
        """
        indicator_class = self.get_indicator(name)
        if indicator_class:
            return indicator_class(params)
        return None
    

# Dosyanın sonuna ekleyin

class IndicatorManager:
    """İndikatörlerin yönetimini ve hesaplanmasını koordine eden sınıf."""
    
    def __init__(self, registry: IndicatorRegistry = None):
        """
        Initialize the indicator manager.
        
        Args:
            registry: Optional indicator registry to use
        """
        self.registry = registry or IndicatorRegistry()
        
    def calculate_indicators(self, df: pd.DataFrame, indicator_names: List[str], 
                          params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Calculate multiple indicators for the dataframe.
        
        Args:
            df: DataFrame with price data
            indicator_names: List of indicator names to calculate
            params: Optional parameters for each indicator as {indicator_name: params_dict}
            
        Returns:
            DataFrame with indicators calculated
        """
        result_df = df.copy()
        params = params or {}
        
        for name in indicator_names:
            # Get indicator params if provided
            indicator_params = params.get(name, {})
            
            # Create and calculate indicator
            indicator = self.registry.create_indicator(name, indicator_params)
            
            if indicator:
                try:
                    result_df = indicator.calculate(result_df)
                except Exception as e:
                    logger.error(f"Error calculating indicator {name}: {e}")
            else:
                logger.warning(f"Indicator {name} not found in registry")
        
        return result_df
    
    def list_available_indicators(self) -> Dict[str, List[str]]:
        """
        Get a list of available indicators by category.
        
        Returns:
            Dictionary of categories to list of indicator names
        """
        result = {}
        
        # Get all indicators
        all_indicators = self.registry.get_all_indicators()
        
        # Group by category
        for name, indicator_class in all_indicators.items():
            category = indicator_class.category
            
            if category not in result:
                result[category] = []
                
            result[category].append(name)
        
        return result    