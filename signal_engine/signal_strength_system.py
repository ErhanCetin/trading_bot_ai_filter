"""
Base strength calculator system for the trading system.
Defines the strength calculator registry and base strength calculator class.
"""
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Type
from abc import ABC, abstractmethod


class BaseStrengthCalculator(ABC):
    """Base class for all signal strength calculators."""
    
    name = "base_strength"
    display_name = "Base Strength Calculator"
    description = "Base class for all strength calculators"
    category = "base"
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strength calculator with parameters.
        
        Args:
            params: Optional parameters to override defaults
        """
        self.params = self.default_params.copy() if hasattr(self, 'default_params') else {}
        if params:
            self.params.update(params)
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate strength values for signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values (typically 1 for long, -1 for short, 0 for no signal)
            
        Returns:
            Series with signal strength values (0-100 scale, 0 = weakest, 100 = strongest)
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


class StrengthCalculatorRegistry:
    """Registry for strength calculator classes."""
    
    def __init__(self):
        """Initialize the registry."""
        self._calculators = {}
    
    def register(self, calculator_class: Type[BaseStrengthCalculator]) -> None:
        """
        Register a strength calculator class.
        
        Args:
            calculator_class: Strength calculator class to register
        """
        if not issubclass(calculator_class, BaseStrengthCalculator):
            raise TypeError(f"Class {calculator_class.__name__} is not a subclass of BaseStrengthCalculator")
        
        self._calculators[calculator_class.name] = calculator_class
    
    def get_calculator_class(self, name: str) -> Optional[Type[BaseStrengthCalculator]]:
        """
        Get a strength calculator class by name.
        
        Args:
            name: Name of the strength calculator class
            
        Returns:
            Strength calculator class or None if not found
        """
        return self._calculators.get(name)

    def create_calculator(self, name: str, params: Optional[Dict[str, Any]] = None) -> Optional[BaseStrengthCalculator]:
        """
        Create a calculator instance by name.
        
        Args:
            name: Name of the calculator
            params: Optional parameters for the calculator
            
        Returns:
            Calculator instance or None if not found
        """
        calculator_class = self.get_calculator_class(name)
        if calculator_class:
            return calculator_class(params)
        return None
    
    def get_all_calculators(self) -> Dict[str, Type[BaseStrengthCalculator]]:
        """
        Get all registered strength calculator classes.
        
        Returns:
            Dictionary of calculator names to calculator classes
        """
        return self._calculators.copy()
    
    def get_calculators_by_category(self, category: str) -> Dict[str, Type[BaseStrengthCalculator]]:
        """
        Get all strength calculator classes in a category.
        
        Args:
            category: Category to filter by
            
        Returns:
            Dictionary of calculator names to calculator classes
        """
        return {name: cls for name, cls in self._calculators.items() if cls.category == category}

class StrengthManager:
    """Sinyal gücü hesaplayıcılarının yönetimini koordine eden sınıf."""
    
    def __init__(self, registry: StrengthCalculatorRegistry = None):
        """
        Initialize the strength manager.
        
        Args:
            registry: Optional strength calculator registry to use
        """
        self.registry = registry or StrengthCalculatorRegistry()
    
    def calculate_strength(self, df: pd.DataFrame, signals_df: pd.DataFrame, 
                         calculator_names: List[str], 
                         params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.Series:
        """
        Calculate signal strength using multiple calculators.
        
        Args:
            df: DataFrame with indicator data
            signals_df: DataFrame with signal columns (long_signal, short_signal)
            calculator_names: List of calculator names to use
            params: Optional parameters for each calculator as {calculator_name: params_dict}
            
        Returns:
            Series with signal strength values (0-100)
        """
        params = params or {}
        
        # Convert signals to a series (-1, 0, 1)
        signals = pd.Series(0, index=df.index)
        
        # Combine long and short signals
        if "long_signal" in signals_df.columns:
            signals[signals_df["long_signal"]] = 1
            
        if "short_signal" in signals_df.columns:
            signals[signals_df["short_signal"]] = -1
        
        # Initialize strength values
        strength_values = pd.Series(0, index=df.index)
        calculator_weights = {}
        
        # Calculate strength with each calculator
        for name in calculator_names:
            # Get calculator params if provided
            calculator_params = params.get(name, {}) if params else {}
            
            # Get weight for this calculator (default to 1.0)
            weight = calculator_params.pop("weight", 1.0)
            calculator_weights[name] = weight
            
            # Create calculator instance - DÜZELTME
            calculator = self.registry.create_calculator(name, calculator_params)
            
            if calculator:
                try:
                    # Calculate strength
                    calculator_strength = calculator.calculate(df, signals)
                    
                    # Add weighted strength
                    strength_values += calculator_strength * weight
                except Exception as e:
                    logger.error(f"Error calculating strength with calculator {name}: {e}")
            else:
                logger.warning(f"Strength calculator {name} not found in registry")
        
        # Normalize strength values based on weights
        total_weight = sum(calculator_weights.values())
        if total_weight > 0:
            strength_values = strength_values / total_weight
        
        # Ensure values are within 0-100 range
        strength_values = strength_values.clip(0, 100).round().astype(int)
        
        # Ensure zero strength for zero signals
        strength_values[signals == 0] = 0
        
        return strength_values
    
    def list_available_calculators(self) -> Dict[str, List[str]]:
        """
        Get a list of available strength calculators by category.
        
        Returns:
            Dictionary of categories to list of calculator names
        """
        result = {}
        
        # Get all calculators
        all_calculators = self.registry.get_all_calculators()
        
        # Group by category
        for name, calculator_class in all_calculators.items():
            category = calculator_class.category
            
            if category not in result:
                result[category] = []
                
            result[category].append(name)
        
        return result
    
    def get_calculator_details(self, calculator_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific strength calculator.
        
        Args:
            calculator_name: Name of the calculator
            
        Returns:
            Dictionary with calculator details or None if not found
        """
        # DÜZELTME
        calculator_class = self.registry.get_calculator_class(calculator_name)
        
        if not calculator_class:
            return None
            
        return {
            "name": calculator_class.name,
            "display_name": calculator_class.display_name,
            "description": calculator_class.description,
            "category": calculator_class.category,
            "default_params": getattr(calculator_class, "default_params", {}),
            "required_indicators": getattr(calculator_class, "required_indicators", [])
        }