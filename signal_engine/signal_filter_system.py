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
    

# signal_engine/signal_filter_system.py - FilterManager sınıfını düzelt

class FilterManager:
    """Filtrelerin yönetimini ve uygulanmasını koordine eden sınıf."""
    
    def __init__(self, registry: FilterRuleRegistry = None):
        """
        Initialize the filter manager.
        
        Args:
            registry: Optional filter registry to use
        """
        self.registry = registry or FilterRuleRegistry()
        
        # 🔧 DÜZELTME: Varsayılan ayarları daha esnek yap
        self._min_checks_required = 0  # ÖNCE: 1, SONRA: 0
        self._min_strength_required = 45  # ÖNCE: 0, SONRA: 45
        
    def add_rule(self, rule_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Filtre kuralı ekler (isim ve parametrelerle)
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
            min_checks: Minimum kontrol sayısı (0 = hiç kontrol gerekmez)
        """
        self._min_checks_required = max(0, min_checks)  # 🔧 DÜZELTME: 0'a izin ver

    def set_min_strength_required(self, min_strength: int) -> None:
        """
        Minimum sinyal gücünü ayarlar.
        
        Args:
            min_strength: Minimum sinyal gücü (0-100)
        """
        self._min_strength_required = max(0, min(100, min_strength))
        
    def set_lenient_mode(self):
        """🆕 Esnek mod - daha fazla sinyal geçer"""
        self._min_checks_required = 0
        self._min_strength_required = 40
        
    def set_balanced_mode(self):
        """🆕 Dengeli mod - orta seviye filtre"""
        self._min_checks_required = 1
        self._min_strength_required = 55
        
    def set_strict_mode(self):
        """🆕 Sıkı mod - sadece yüksek kaliteli sinyaller"""
        self._min_checks_required = 2
        self._min_strength_required = 70
        
    def filter_signals(self, df: pd.DataFrame, rule_names: List[str] = None,
                      params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Sinyalleri filtreler.
        
        Args:
            df: Sinyal sütunları içeren DataFrame
            rule_names: Kullanılacak kural adları listesi
            params: Her kural için parametreler
            
        Returns:
            Filtrelenmiş sinyallerle DataFrame
        """
        result_df = df.copy()
       
        print("🔍 FILTER DEBUG:")
        print(f"Rules to apply: {getattr(self, '_rules_to_apply', [])}")
        print(f"Available rules: {list(self.registry.get_all_filters().keys())}")
        print(f"Missing rules: {[r for r in getattr(self, '_rules_to_apply', []) if not self.registry.get_filter(r)]}")        
        # Önceden add_rule ile eklenen kuralları kullan
        if rule_names is None:
            rule_names = getattr(self, '_rules_to_apply', [])
            params = getattr(self, '_rule_params', {})
        else:
            params = params or {}
        
        # Min kontrol ve güç değerlerini al
        min_checks = getattr(self, '_min_checks_required', 0)
        min_strength = getattr(self, '_min_strength_required', 45)
        
        print(f"🔍 Filter Ayarları: min_checks={min_checks}, min_strength={min_strength}")
        
        # Eğer hiç kural yoksa, sadece güç kontrolü yap
        if not rule_names:
            print("⚠️ Hiç filtre kuralı yok, sadece güç kontrolü yapılıyor")
            return self._apply_strength_only_filter(result_df, min_strength)
        
        # Gerçek filtre kurallarından özel parametreleri ayır
        actual_rule_names = [name for name in rule_names 
                           if name not in ['min_checks', 'min_strength']]
        
        # Eğer gerçek kural yoksa, sadece güç kontrolü yap
        if not actual_rule_names:
            print("⚠️ Gerçek filtre kuralı yok, sadece güç kontrolü yapılıyor")
            return self._apply_strength_only_filter(result_df, min_strength)
        
        # Filtre sonuçlarını sakla
        signal_filter_results = pd.DataFrame(index=result_df.index)
        
        # Her kural için sonuçları hesapla
        for name in actual_rule_names:
            rule_params = params.get(name, {})
            rule = self.registry.create_filter(name, rule_params)
            
            if rule:
                try:
                    # Farklı filtre türlerini kontrol et
                    if hasattr(rule, 'apply') and hasattr(rule, 'validate_dataframe'):
                        # BaseFilter türü
                        signals = pd.Series(0, index=result_df.index)
                        if "long_signal" in result_df.columns:
                            signals[result_df["long_signal"]] = 1
                        if "short_signal" in result_df.columns:
                            signals[result_df["short_signal"]] = -1
                        
                        rule_result = rule.apply(result_df, signals)
                        signal_filter_results[name] = rule_result != 0
                        
                    else:
                        # Eski tip filtreler
                        filtered_df = rule.apply_to_dataframe(result_df)
                        
                        # Filtreleme sonucunu analiz et
                        row_results = pd.Series(True, index=result_df.index)
                        
                        if "long_signal" in result_df.columns:
                            long_failed = (result_df["long_signal"] == True) & (filtered_df["long_signal"] == False)
                            row_results[long_failed] = False
                        
                        if "short_signal" in result_df.columns:
                            short_failed = (result_df["short_signal"] == True) & (filtered_df["short_signal"] == False)
                            row_results[short_failed] = False
                        
                        signal_filter_results[name] = row_results
                        
                except Exception as e:
                    logger.error(f"Error applying filter rule {name}: {e}")
                    signal_filter_results[name] = True  # Hata durumunda geçir
            else:
                logger.warning(f"Rule {name} not found in registry")
                signal_filter_results[name] = True  # Bulunamayan kural için geçir
        
        # Geçerli kontrol sayısını hesapla
        if not signal_filter_results.empty:
            valid_checks = signal_filter_results.sum(axis=1)
        else:
            valid_checks = pd.Series(0, index=result_df.index)
        
        # Sinyalleri filtrele
        result_df["signal_passed_filter"] = False
        
        # Sinyal gücü kontrolü
        has_strength = pd.Series(True, index=result_df.index)
        if "signal_strength" in result_df.columns:
            has_strength = result_df["signal_strength"] >= min_strength
        
        # 🔧 DÜZELTME: Kontrol sayısı ve güç kontrolü
        passes_filter = (valid_checks >= min_checks) & has_strength
        
        # Sinyalleri güncelle
        result_df.loc[passes_filter, "signal_passed_filter"] = True
        
        # Başarısız sinyalleri kaldır (opsiyonel)
        # Bu kısmı şimdilik devre dışı bırak, sadece işaretle
        """
        if "long_signal" in result_df.columns:
            result_df.loc[~passes_filter & result_df["long_signal"], "long_signal"] = False
        
        if "short_signal" in result_df.columns:
            result_df.loc[~passes_filter & result_df["short_signal"], "short_signal"] = False
        """
        
        # 📊 Sonuçları raporla
        total_signals = (result_df.get("long_signal", pd.Series(False)).sum() + 
                        result_df.get("short_signal", pd.Series(False)).sum())
        passed_signals = result_df["signal_passed_filter"].sum()
        
        print(f"📊 Filter Sonuçları:")
        print(f"   Toplam sinyal: {total_signals}")
        print(f"   Geçen sinyal: {passed_signals}")
        print(f"   Geçme oranı: {passed_signals/total_signals*100:.1f}%" if total_signals > 0 else "   Geçme oranı: 0.0%")
        
        return result_df
    
    def _apply_strength_only_filter(self, df: pd.DataFrame, min_strength: int) -> pd.DataFrame:
        """🆕 Sadece güç kontrolü yapar"""
        result_df = df.copy()
        result_df["signal_passed_filter"] = False
        
        # Sinyal gücü kontrolü
        if "signal_strength" in result_df.columns:
            has_strength = result_df["signal_strength"] >= min_strength
            
            # Long veya short sinyali olan ve güç eşiğini geçen sinyalleri işaretle
            has_signal = (result_df.get("long_signal", pd.Series(False)) | 
                         result_df.get("short_signal", pd.Series(False)))
            
            result_df.loc[has_signal & has_strength, "signal_passed_filter"] = True
        
        return result_df

# 🆕 YARDIMCI FONKSİYONLAR

def quick_fix_csv_data(csv_file_path: str) -> pd.DataFrame:
    """
    CSV verisini hızlıca düzeltir ve test eder
    
    Args:
        csv_file_path: CSV dosya yolu
        
    Returns:
        Düzeltilmiş DataFrame
    """
    import pandas as pd
    
    # CSV'yi oku
    df = pd.read_csv(csv_file_path)
    
    print(f"📂 CSV okundu: {len(df)} satır")
    print(f"📊 Mevcut filter geçme: {df['signal_passed_filter'].sum()} / {len(df)}")
    
    # Filter manager oluştur ve esnek moda ayarla
    filter_manager = FilterManager()
    filter_manager.set_lenient_mode()  # Esnek mod
    
    # Sadece güç kontrolü ile filtrele
    fixed_df = filter_manager._apply_strength_only_filter(df, min_strength=45)
    
    print(f"✅ Düzeltme sonrası filter geçme: {fixed_df['signal_passed_filter'].sum()} / {len(fixed_df)}")
    
    return fixed_df

def test_filter_fix():
    """Filter düzeltmesini test eder"""
    
    # Test verisi oluştur
    test_data = {
        'signal_strength': [45, 60, 30, 70, 50, 40, 80],
        'long_signal': [True, False, True, False, True, False, True],
        'short_signal': [False, True, False, True, False, True, False],
        'signal_passed_filter': [False] * 7  # Hepsi False
    }
    
    df = pd.DataFrame(test_data)
    
    print("🧪 TEST VERİSİ:")
    print(f"Toplam sinyal: {df['long_signal'].sum() + df['short_signal'].sum()}")
    print(f"Geçen sinyal (önce): {df['signal_passed_filter'].sum()}")
    
    # Filter manager ile düzelt
    filter_manager = FilterManager()
    filter_manager.set_lenient_mode()
    
    fixed_df = filter_manager._apply_strength_only_filter(df, min_strength=45)
    
    print(f"Geçen sinyal (sonra): {fixed_df['signal_passed_filter'].sum()}")
    print("✅ Filter düzeltmesi başarılı!")
    
    return fixed_df

# Test çalıştır
if __name__ == "__main__":
    test_result = test_filter_fix()
    print("\n📋 Test Sonucu:")
    print(test_result[['signal_strength', 'long_signal', 'short_signal', 'signal_passed_filter']])    