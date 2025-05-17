"""
Base filter system for the trading system.
Defines the filter registry and base filter class.
"""
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Type
from abc import ABC, abstractmethod

# Logger ayarla
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



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
    def add_rule(self, rule_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Filtre kuralı ekler (isim ve parametrelerle)
        
        Args:
            rule_name: Kural adı
            params: Kural parametreleri
        """
        self._rules_to_apply = getattr(self, '_rules_to_apply', [])
        self._rule_params = getattr(self, '_rule_params', {})
        
        self._rules_to_apply.append(rule_name)
        if params:
            self._rule_params[rule_name] = params   
   
    def set_min_checks_required(self, min_checks: int) -> None:
        """
        Minimum geçmesi gereken kontrol sayısını ayarlar.
        
        Args:
            min_checks: Minimum kontrol sayısı
        """
        self._min_checks_required = max(1, min_checks)  # En az 1 kontrol gerekli

    def set_min_strength_required(self, min_strength: int) -> None:
        """
        Minimum sinyal gücünü ayarlar.
        
        Args:
            min_strength: Minimum sinyal gücü (0-10)
        """
        self._min_strength_required = max(0, min(10, min_strength))  # 0-10 arası         
    
    def filter_signals(self, df: pd.DataFrame, rule_names: List[str] = None,
                 params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Sinyalleri filtreler.
        
        Args:
            df: Sinyal sütunları içeren DataFrame
            rule_names: Kullanılacak kural adları listesi (None ise önceden eklenenler)
            params: Her kural için isteğe bağlı parametreler {rule_name: params_dict}
            
        Returns:
            Filtrelenmiş sinyallerle DataFrame
        """
        result_df = df.copy()
        
        # Daha önce add_rule ile eklenen kuralları kullan
        if rule_names is None:
            rule_names = getattr(self, '_rules_to_apply', [])
            params = getattr(self, '_rule_params', {})
        else:
            params = params or {}
        
        # Min kontrol ve güç değerlerini al
        min_checks = getattr(self, '_min_checks_required', 1)
        min_strength = getattr(self, '_min_strength_required', 0)
        
        # Filtre sonuçlarını sakla
        signal_filter_results = pd.DataFrame(index=result_df.index)
        
        # Her kural için sonuçları hesapla
        for name in rule_names:
            # Kural parametrelerini al
            rule_params = params.get(name, {})
            
            # Kural örneği oluştur
            rule = self.registry.create_filter(name, rule_params)
            
            if rule:
                try:
                    # Convert signals to series
                    signals = pd.Series(0, index=result_df.index)
                    if "long_signal" in result_df.columns:
                        signals[result_df["long_signal"]] = 1
                    if "short_signal" in result_df.columns:
                        signals[result_df["short_signal"]] = -1
                    
                    # Kuralı uygula
                    rule_result = rule.apply(result_df, signals)
                    signal_filter_results[name] = rule_result
                except Exception as e:
                    logger.error(f"Error applying filter rule {name}: {e}")
                    # Hata durumunda varsayılan olarak tüm satırlar için True kullan
                    signal_filter_results[name] = True
            else:
                logger.warning(f"Rule {name} not found in registry")
                # Bulunamayan kural için varsayılan olarak tüm satırlar için True kullan
                signal_filter_results[name] = True
        
        # Geçerli kontrol sayısını hesapla
        if not signal_filter_results.empty:
            valid_checks = signal_filter_results.sum(axis=1)
            
            # Sinyalleri filtrele
            result_df["signal_passed_filter"] = False
            
            # Sinyal gücü kontrolü
            has_strength = True
            if "signal_strength" in result_df.columns:
                has_strength = result_df["signal_strength"] >= min_strength
            
            # Kontrol sayısı ve güç kontrolü
            passes_filter = (valid_checks >= min_checks) & has_strength
            
            # Sinyalleri güncelle
            result_df.loc[passes_filter, "signal_passed_filter"] = True
            
            # Failed sinyalleri kaldır
            if "long_signal" in result_df.columns:
                result_df.loc[~passes_filter & result_df["long_signal"], "long_signal"] = False
            
            if "short_signal" in result_df.columns:
                result_df.loc[~passes_filter & result_df["short_signal"], "short_signal"] = False
        
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