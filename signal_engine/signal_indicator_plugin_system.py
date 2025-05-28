"""
Enhanced Base indicator system with dependency management.
"""
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Type, Set
from abc import ABC, abstractmethod
import logging

# Logger setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseIndicator(ABC):
    """Base class for all indicators with dependency management."""
    
    name = "base_indicator"
    display_name = "Base Indicator"
    description = "Base class for all indicators"
    category = "base"
    
    default_params = {}
    requires_columns = []
    output_columns = []
    dependencies = []  # NEW: List of indicator names this indicator depends on
    
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
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of indicator names this indicator depends on.
        
        Returns:
            List of dependency indicator names
        """
        return self.dependencies.copy()
    
    def has_dependencies_calculated(self, df: pd.DataFrame) -> bool:
        """
        Check if all dependencies are calculated in the dataframe.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if all dependencies are present, False otherwise
        """
        for dep_name in self.dependencies:
            if dep_name not in df.columns:
                return False
        return True
    
    def validate_dependencies(self, df: pd.DataFrame) -> None:
        """
        Validate that all dependencies are present in the dataframe.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If dependencies are missing
        """
        missing_deps = [dep for dep in self.dependencies if dep not in df.columns]
        if missing_deps:
            raise ValueError(
                f"Indicator '{self.name}' is missing required dependencies: {missing_deps}. "
                f"Please ensure these indicators are calculated first."
            )
    
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
    """Registry for indicator classes with dependency resolution."""
    
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
        logger.debug(f"Registered indicator: {indicator_class.name}")
    
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
        logger.warning(f"Indicator '{name}' not found in registry")
        return None


class IndicatorManager:
    """Manager for calculating indicators with smart dependency resolution."""
    
    def __init__(self, registry: IndicatorRegistry = None):
        """
        Initialize the indicator manager.
        
        Args:
            registry: Optional indicator registry to use
        """
        self.registry = registry or IndicatorRegistry()
        self._indicators_to_calculate = []
        self._indicator_params = {}
        
        # Create column-to-indicator mapping for smart resolution
        self._column_to_indicator_map = self._build_column_mapping()
    
    def add_indicator(self, indicator_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        İndikatör ekler (isim ve parametrelerle)
        
        Args:
            indicator_name: İndikatör adı
            params: İndikatör parametreleri
        """
        self._indicators_to_calculate = getattr(self, '_indicators_to_calculate', [])
        self._indicator_params = getattr(self, '_indicator_params', {})
        
        self._indicators_to_calculate.append(indicator_name)
        if params:
            self._indicator_params[indicator_name] = params    
        
    def _build_column_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from output columns to indicator names.
        Handles both static and dynamic column patterns.
        
        Returns:
            Dictionary mapping column names to indicator names
        """
        column_map = {}
        
        for indicator_name, indicator_class in self.registry.get_all_indicators().items():
            # Handle static output columns
            if hasattr(indicator_class, 'output_columns') and indicator_class.output_columns:
                for column in indicator_class.output_columns:
                    if column:  # Skip empty strings
                        column_map[column] = indicator_name
            
            # Handle dynamic patterns for common indicators
            if indicator_name == "rsi":
                # RSI creates rsi_{period} columns
                for period in [7, 14, 21, 28]:  # Common RSI periods
                    column_map[f"rsi_{period}"] = indicator_name
            
            elif indicator_name == "ema":
                # EMA creates ema_{period} columns  
                for period in [9, 12, 20, 21, 26, 50, 100, 200]:  # Common EMA periods
                    column_map[f"ema_{period}"] = indicator_name
            
            elif indicator_name == "sma":
                # SMA creates sma_{period} columns
                for period in [10, 20, 50, 100, 200]:  # Common SMA periods
                    column_map[f"sma_{period}"] = indicator_name
            
            elif indicator_name == "atr":
                # ATR creates atr_{period} columns
                for period in [10, 14, 20, 50]:  # Common ATR periods
                    column_map[f"atr_{period}"] = indicator_name
                    column_map[f"atr_{period}_percent"] = indicator_name
            
            elif indicator_name == "macd":
                # MACD creates specific columns
                column_map["macd_line"] = indicator_name
                column_map["macd_signal"] = indicator_name
                column_map["macd_histogram"] = indicator_name
                column_map["macd_crossover"] = indicator_name
            
            elif indicator_name == "adx_calculator":
                # ADX Calculator creates these specific columns
                column_map["adx"] = indicator_name
                column_map["di_pos"] = indicator_name
                column_map["di_neg"] = indicator_name
            
            elif indicator_name == "bollinger":
                # Bollinger Bands creates these columns
                column_map["bollinger_upper"] = indicator_name
                column_map["bollinger_middle"] = indicator_name
                column_map["bollinger_lower"] = indicator_name
                column_map["bollinger_width"] = indicator_name
                column_map["bollinger_pct_b"] = indicator_name
        
        logger.debug(f"Built column mapping with {len(column_map)} entries")
        return column_map
    
    def _resolve_column_dependencies(self, dependencies: List[str]) -> List[str]:
        """
        Resolve column names to indicator names.
        
        Args:
            dependencies: List of column names or indicator names
            
        Returns:
            List of indicator names needed to satisfy dependencies
        """
        indicator_names = set()
        unresolved = []
        
        for dep in dependencies:
            if dep in self.registry.get_all_indicators():
                # It's already an indicator name
                indicator_names.add(dep)
            elif dep in self._column_to_indicator_map:
                # It's a column name, map to indicator
                indicator_names.add(self._column_to_indicator_map[dep])
            else:
                # Try to infer from patterns
                resolved = self._infer_indicator_from_column(dep)
                if resolved:
                    indicator_names.add(resolved)
                else:
                    unresolved.append(dep)
        
        if unresolved:
            logger.warning(f"Could not resolve dependencies: {unresolved}")
        
        return list(indicator_names)
    
    def _infer_indicator_from_column(self, column_name: str) -> Optional[str]:
        """
        Try to infer indicator name from column name patterns.
        
        Args:
            column_name: Column name to analyze
            
        Returns:
            Inferred indicator name or None
        """
        # Pattern matching for dynamic columns
        if column_name.startswith("rsi_"):
            return "rsi"
        elif column_name.startswith("ema_"):
            return "ema"
        elif column_name.startswith("sma_"):
            return "sma"
        elif column_name.startswith("atr_"):
            return "atr"
        elif column_name.startswith("macd_"):
            return "macd"
        elif column_name.startswith("bollinger_"):
            return "bollinger"
        elif column_name.startswith("keltner_"):
            return "keltner"
        elif column_name.startswith("supertrend"):
            return "supertrend"
        elif column_name.startswith("ha_"):
            return "heikin_ashi"
        elif column_name.startswith(("tenkan_", "kijun_", "senkou_", "chikou_", "cloud_")):
            return "ichimoku"
        elif column_name == "adaptive_rsi" or column_name == "adaptive_rsi_period":
            return "adaptive_rsi"
        elif column_name.endswith("_alignment") and "ema" in column_name:
            return "mtf_ema"
        elif column_name.endswith("_zscore") or column_name.endswith("_percentile"):
            return "zscore"
        elif column_name.startswith("std_") or column_name.startswith("volatility_"):
            return "std_deviation"
        elif column_name.startswith("reg_") or column_name.startswith("slope_"):
            return "linear_regression"
        elif column_name in ["adx", "di_pos", "di_neg"]:
            return "adx_calculator"
        
        return None
    def _infer_indicator_from_column(self, column_name: str) -> Optional[str]:
        """
        Try to infer indicator name from column name patterns.
        
        Args:
            column_name: Column name to analyze
            
        Returns:
            Inferred indicator name or None
        """
        # Pattern matching for dynamic columns
        if column_name.startswith("rsi_"):
            return "rsi"
        elif column_name.startswith("ema_"):
            return "ema"
        elif column_name.startswith("sma_"):
            return "sma"
        elif column_name.startswith("atr_"):
            return "atr"
        elif column_name.startswith("macd_"):
            return "macd"
        elif column_name.startswith("bollinger_"):
            return "bollinger"
        elif column_name.startswith("keltner_"):
            return "keltner"
        elif column_name.startswith("supertrend"):
            return "supertrend"
        elif column_name.startswith("ha_"):
            return "heikin_ashi"
        elif column_name.startswith(("tenkan_", "kijun_", "senkou_", "chikou_", "cloud_")):
            return "ichimoku"
        elif column_name == "adaptive_rsi" or column_name == "adaptive_rsi_period":
            return "adaptive_rsi"
        elif column_name.endswith("_alignment") and "ema" in column_name:
            return "mtf_ema"
        elif column_name.endswith("_zscore") or column_name.endswith("_percentile"):
            return "zscore"
        elif column_name.startswith("std_") or column_name.startswith("volatility_"):
            return "std_deviation"
        elif column_name.startswith("reg_") or column_name.startswith("slope_"):
            return "linear_regression"
        elif column_name in ["adx", "di_pos", "di_neg"]:
            return "adx_calculator"
        
        return None
        """
        Add an indicator to be calculated.
        
        Args:
            indicator_name: Name of the indicator
            params: Optional parameters for the indicator
        """
        if indicator_name not in self._indicators_to_calculate:
            self._indicators_to_calculate.append(indicator_name)
        
        if params:
            self._indicator_params[indicator_name] = params
    
    def _resolve_dependencies(self, indicator_names: List[str]) -> List[str]:
        """
        Enhanced dependency resolution that handles both indicator and column names.
        
        Args:
            indicator_names: List of indicator names to resolve
            
        Returns:
            List of indicator names in dependency order
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        resolved = []
        visited = set()
        visiting = set()  # For circular dependency detection
        
        def resolve_single(name: str):
            if name in visiting:
                raise ValueError(f"Circular dependency detected involving indicator: {name}")
            
            if name in visited:
                return
            
            visiting.add(name)
            indicator_class = self.registry.get_indicator(name)
            
            if indicator_class and hasattr(indicator_class, 'dependencies'):
                # Smart resolution: convert column dependencies to indicator names
                indicator_deps = self._resolve_column_dependencies(indicator_class.dependencies)
                
                # Recursively resolve each dependency
                for dep in indicator_deps:
                    resolve_single(dep)
                
                # Add self if not already added
                if name not in resolved:
                    resolved.append(name)
            elif indicator_class:
                # No dependencies, just add
                if name not in resolved:
                    resolved.append(name)
            else:
                logger.warning(f"Indicator '{name}' not found in registry")
            
            visiting.remove(name)
            visited.add(name)
        
        # Resolve all indicators
        for name in indicator_names:
            resolve_single(name)
        
        logger.debug(f"Smart dependency resolution order: {resolved}")
        return resolved
    
    def calculate_indicators(self, df: pd.DataFrame, 
                           indicator_names: List[str] = None,
                           params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Calculate indicators with smart dependency resolution.
        
        Args:
            df: DataFrame with price data
            indicator_names: List of indicator names to calculate (optional)
            params: Optional parameters for each indicator
            
        Returns:
            DataFrame with calculated indicators
        """
        result_df = df.copy()
        
        # Use previously added indicators if none specified
        if indicator_names is None:
            indicator_names = self._indicators_to_calculate.copy()
            params = self._indicator_params.copy()
        else:
            params = params or {}
        
        if not indicator_names:
            logger.warning("No indicators specified for calculation")
            return result_df
        
        try:
            # Smart dependency resolution (handles both indicator and column names)
            resolved_names = self._resolve_dependencies(indicator_names)
            
            logger.info(f"Smart calculation order: {resolved_names}")
            
            # Calculate indicators in dependency order
            for name in resolved_names:
                indicator_params = params.get(name, {})
                indicator = self.registry.create_indicator(name, indicator_params)
                
                if indicator:
                    try:
                        # Smart optimization: Skip if output columns already exist
                        if self._should_skip_calculation(indicator, result_df):
                            logger.debug(f"Skipping {name} - outputs already exist")
                            continue
                        
                        # Calculate indicator
                        result_df = indicator.calculate(result_df)
                        logger.debug(f"Successfully calculated indicator: {name}")
                        
                    except Exception as e:
                        logger.error(f"Error calculating indicator '{name}': {e}")
                        # Continue with other indicators rather than failing completely
                        
                else:
                    logger.warning(f"Could not create indicator: {name}")
            
        except Exception as e:
            logger.error(f"Error in smart dependency resolution: {e}")
            raise
        
        return result_df
    
    def _should_skip_calculation(self, indicator: BaseIndicator, df: pd.DataFrame) -> bool:
        """
        Check if indicator calculation should be skipped (outputs already exist).
        
        Args:
            indicator: Indicator instance
            df: DataFrame to check
            
        Returns:
            True if calculation should be skipped
        """
        if not hasattr(indicator, 'output_columns') or not indicator.output_columns:
            return False
        
        # Check if all output columns exist and have non-null values
        for col in indicator.output_columns:
            if col not in df.columns or df[col].isnull().all():
                return False
        
        return True
    
    def list_available_indicators(self) -> Dict[str, List[str]]:
        """
        Get a list of available indicators by category.
        
        Returns:
            Dictionary of categories to list of indicator names
        """
        result = {}
        all_indicators = self.registry.get_all_indicators()
        
        for name, indicator_class in all_indicators.items():
            category = indicator_class.category
            if category not in result:
                result[category] = []
            result[category].append(name)
        
        return result
    
    def get_dependency_graph(self, indicator_names: List[str] = None) -> Dict[str, List[str]]:
        """
        Get the dependency graph for specified indicators.
        
        Args:
            indicator_names: List of indicator names (optional)
            
        Returns:
            Dictionary mapping indicator names to their dependencies
        """
        if indicator_names is None:
            indicator_names = self._indicators_to_calculate.copy()
        
        dependency_graph = {}
        
        for name in indicator_names:
            indicator_class = self.registry.get_indicator(name)
            if indicator_class:
                dependency_graph[name] = indicator_class.dependencies.copy()
        
        return dependency_graph