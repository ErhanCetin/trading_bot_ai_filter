"""
Base filter class definition for the trading system.
"""
import pandas as pd
from abc import abstractmethod
from typing import Dict, Any, Optional, List

from signal_engine.signal_filter_system import BaseFilterRule


class AdvancedFilterRule(BaseFilterRule):
    """Enhanced base class for advanced filter rules."""
    
    required_indicators = []  # Alt sınıflar tarafından doldurulacak
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with parameters.
        
        Args:
            params: Optional parameters to override defaults
        """
        super().__init__(params)
        
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Temel signal sütunlarını kontrol et
        base_columns = ["long_signal", "short_signal"]
        if not all(col in df.columns for col in base_columns):
            return False
            
        # Gerekli indikatör sütunlarını kontrol et
        required_columns = getattr(self, 'required_indicators', [])
        return all(col in df.columns for col in required_columns)
    
    def prepare_filter(self, df: pd.DataFrame) -> None:
        """
        Prepare the filter before applying to all rows.
        This is called once before checking all rows, useful for pre-calculations.
        
        Args:
            df: DataFrame with indicator and signal data
        """
        pass
    
    def post_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-processing after all rows have been checked.
        This is called once after all rows are processed.
        
        Args:
            df: DataFrame with filtered signals
            
        Returns:
            DataFrame with post-processed signals
        """
        return df
    
    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter to the entire dataframe.
        
        Args:
            df: DataFrame with indicator and signal data
            
        Returns:
            DataFrame with filtered signals
        """
        # Önce DataFrame'i doğrula
        if not self.validate_dataframe(df):
            # Gerekli sütunlar yoksa, orijinal DataFrame'i değiştirmeden döndür
            return df
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Prepare filter
        self.prepare_filter(result_df)
        
        # Apply filter row by row
        for i in range(len(result_df)):
            row = result_df.iloc[i]
            
            # Only process rows with signals
            if row["long_signal"] or row["short_signal"]:
                # Check long signal
                if row["long_signal"]:
                    result_df.loc[result_df.index[i], "long_signal"] = self.check_rule(result_df, row, i, "long")
                
                # Check short signal
                if row["short_signal"]:
                    result_df.loc[result_df.index[i], "short_signal"] = self.check_rule(result_df, row, i, "short")
        
        # Apply post-processing
        result_df = self.post_filter(result_df)
        
        return result_df