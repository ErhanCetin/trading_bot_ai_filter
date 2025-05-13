"""
Base filter system for the trading system.
Defines the filter registry and base filter class.
"""
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Type
from abc import ABC, abstractmethod


class BaseFilter(ABC):
    """Base class for all signal filters."""
    
    name = "base_filter"
    display_name = "Base Filter"
    description = "Base class for all filters"
    category = "base"
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the filter with parameters.
        
        Args:
            params: Optional parameters to override defaults
        """
        self.params = self.default_params.copy() if hasattr(self, 'default_params') else {}
        if params:
            self.params.update(params)
    
    @abstractmethod
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply the filter to the signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values (typically 1 for long, -1 for short, 0 for no signal)
            
        Returns:
            Filtered signals series
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
        required_columns = getattr(self, 'required_indicators', [])
        return all(col in df.columns for col in required_columns)


class FilterRuleRegistry:
    """Registry for filter classes."""
    
    def __init__(self):
        """Initialize the registry."""
        self._filters = {}
    
    def register(self, filter_class: Type[BaseFilter]) -> None:
        """
        Register a filter class.
        
        Args:
            filter_class: Filter class to register
        """
        if not issubclass(filter_class, BaseFilter):
            raise TypeError(f"Class {filter_class.__name__} is not a subclass of BaseFilter")
        
        self._filters[filter_class.name] = filter_class
    
    def get_filter(self, name: str) -> Optional[Type[BaseFilter]]:
        """
        Get a filter class by name.
        
        Args:
            name: Name of the filter class
            
        Returns:
            Filter class or None if not found
        """
        return self._filters.get(name)
    
    def get_all_filters(self) -> Dict[str, Type[BaseFilter]]:
        """
        Get all registered filter classes.
        
        Returns:
            Dictionary of filter names to filter classes
        """
        return self._filters.copy()
    
    def get_filters_by_category(self, category: str) -> Dict[str, Type[BaseFilter]]:
        """
        Get all filter classes in a category.
        
        Args:
            category: Category to filter by
            
        Returns:
            Dictionary of filter names to filter classes
        """
        return {name: cls for name, cls in self._filters.items() if cls.category == category}
    
    def create_filter(self, name: str, params: Optional[Dict[str, Any]] = None) -> Optional[BaseFilter]:
        """
        Create a filter instance by name.
        
        Args:
            name: Name of the filter
            params: Optional parameters to pass to the filter
            
        Returns:
            Filter instance or None if not found
        """
        filter_class = self.get_filter(name)
        if filter_class:
            return filter_class(params)
        return None
    def register_filter_rule(self, name: str, rule_class: Type) -> None:
        """
        Register a BaseFilterRule class with the given name.
        This allows compatibility with both BaseFilter and BaseFilterRule.
        
        Args:
            name: Name of the filter rule
            rule_class: BaseFilterRule class to register
        """
        # BaseFilterRule sınıflarını da kabul edelim
        self._filters[name] = rule_class
        

class FilterManager:
    """Filtrelerin yönetimini ve uygulanmasını koordine eden sınıf."""
    
    def __init__(self, registry: FilterRuleRegistry = None):
        """
        Initialize the filter manager.
        
        Args:
            registry: Optional filter registry to use
        """
        self.registry = registry or FilterRuleRegistry()
    
    def apply_filters(self, df: pd.DataFrame, signals_df: pd.DataFrame, 
                    filter_names: List[str], params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Apply multiple filters to signals.
        
        Args:
            df: DataFrame with indicator data
            signals_df: DataFrame with signal columns (long_signal, short_signal)
            filter_names: List of filter names to apply
            params: Optional parameters for each filter as {filter_name: params_dict}
            
        Returns:
            DataFrame with filtered signals
        """
        # Combine indicator data and signals into one dataframe
        result_df = df.copy()
        
        # Ensure signal columns exist in the result dataframe
        if "long_signal" not in result_df.columns:
            result_df["long_signal"] = signals_df.get("long_signal", False)
        if "short_signal" not in result_df.columns:
            result_df["short_signal"] = signals_df.get("short_signal", False)
        
        params = params or {}
        
        # Apply each filter
        for name in filter_names:
            # Get filter params if provided
            filter_params = params.get(name, {})
            
            # Create filter instance
            filter_instance = self.registry.create_filter(name, filter_params)
            
            if filter_instance:
                try:
                    # Apply filter
                    result_df = filter_instance.apply_to_dataframe(result_df)
                except Exception as e:
                    logger.error(f"Error applying filter {name}: {e}")
            else:
                logger.warning(f"Filter {name} not found in registry")
        
        return result_df
    
    def list_available_filters(self) -> Dict[str, List[str]]:
        """
        Get a list of available filters by category.
        
        Returns:
            Dictionary of categories to list of filter names
        """
        result = {}
        
        # Get all filters
        all_filters = self.registry.get_all_filters()
        
        # Group by category
        for name, filter_class in all_filters.items():
            category = filter_class.category
            
            if category not in result:
                result[category] = []
                
            result[category].append(name)
        
        return result
    
    def get_filter_details(self, filter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific filter.
        
        Args:
            filter_name: Name of the filter
            
        Returns:
            Dictionary with filter details or None if not found
        """
        filter_class = self.registry.get_filter(filter_name)
        
        if not filter_class:
            return None
            
        return {
            "name": filter_class.name,
            "display_name": filter_class.display_name,
            "description": filter_class.description,
            "category": filter_class.category,
            "default_params": getattr(filter_class, "default_params", {}),
            "required_indicators": getattr(filter_class, "required_indicators", [])
        }    
    

class BaseFilterRule:
    """Base class for all filter rules in the system."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter to the data and return filtered data."""
        raise NotImplementedError("Subclasses must implement apply method")
    
    def validate_params(self) -> bool:
        """Validate the parameters for this filter."""
        return True    