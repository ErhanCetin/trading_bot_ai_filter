"""
Base strength calculator for the trading system.
All strength calculators must inherit from this base class.
"""
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
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