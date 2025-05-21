"""
1. Önce yeni bir dosya oluşturun: common_calculations.py
"""

import pandas as pd
import ta
import numpy as np
from typing import Dict, Any, List, Optional

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class ADXCalculator(BaseIndicator):
    """ADX ve ilgili değerleri hesaplayan ortak sınıf."""
    
    name = "adx_calculator"
    display_name = "ADX Calculator"
    description = "Common ADX calculation used by multiple indicators"
    category = "utility"
    
    default_params = {
        "window": 14
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = ["adx", "di_pos", "di_neg"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX and related values.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with ADX columns added
        """
        result_df = df.copy()
        
        # Get parameters
        window = self.params.get("window", self.default_params["window"])
        
        # Kontrol et: Tüm gerekli ADX sütunları var mı ve geçerli mi?
        needs_calculation = (
            "adx" not in result_df.columns or 
            "di_pos" not in result_df.columns or 
            "di_neg" not in result_df.columns or
            result_df["adx"].isna().any() or
            result_df["di_pos"].isna().any() or
            result_df["di_neg"].isna().any()
        )
        
        if needs_calculation:
            # Hesapla: Eksik veya geçersiz değerler var
            try:
                adx_indicator = ta.trend.ADXIndicator(
                    high=result_df["high"],
                    low=result_df["low"],
                    close=result_df["close"],
                    window=window
                )
                
                result_df["adx"] = adx_indicator.adx()
                result_df["di_pos"] = adx_indicator.adx_pos()
                result_df["di_neg"] = adx_indicator.adx_neg()
                
                # NaN değerlerin kontrol edilmesi
                nan_count = result_df[["adx", "di_pos", "di_neg"]].isna().sum().sum()
                if nan_count > 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"ADX calculation produced {nan_count} NaN values. This could affect downstream indicators.")
                    
                    # NaN değerleri işleme - başlangıçtaki NaN'ları ileriye doldur
                    result_df[["adx", "di_pos", "di_neg"]] = result_df[["adx", "di_pos", "di_neg"]].fillna(method='ffill')
                    
                    # Hala NaN kaldıysa (başlangıçta) onları da doldur
                    result_df[["adx", "di_pos", "di_neg"]] = result_df[["adx", "di_pos", "di_neg"]].fillna(0)
            
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error calculating ADX: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Hata durumunda varsayılan değerler ata
                if "adx" not in result_df.columns:
                    result_df["adx"] = 0
                if "di_pos" not in result_df.columns:
                    result_df["di_pos"] = 0
                if "di_neg" not in result_df.columns:
                    result_df["di_neg"] = 0
        
        return result_df