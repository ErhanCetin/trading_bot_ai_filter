# signal_engine/signal_manager.py

class SignalManager:
    """
    Tüm sinyal sistemi bileşenlerini yöneten ana sınıf.
    """
    
    def __init__(self):
        """Ana yönetici sınıfı başlat."""
        from signal_engine.signal_indicator_plugin_system import IndicatorManager
        from signal_engine.signal_strategy_system import StrategyManager
        from signal_engine.signal_filter_system import FilterManager
        from signal_engine.signal_strength_system import StrengthManager
        
        # Yönetici sınıfları oluştur
        self.indicator_manager = IndicatorManager()
        self.strategy_manager = StrategyManager()
        self.filter_manager = FilterManager()
        self.strength_manager = StrengthManager()
        
        # ML modülünü opsiyonel olarak ekle
        try:
            from signal_engine.signal_ml_system import MLManager
            self.ml_manager = MLManager()
        except ImportError:
            self.ml_manager = None
    
    def process_data(self, price_data, config):
        """
        Fiyat verilerini işleme akışı.
        
        Args:
            price_data: Fiyat verilerini içeren DataFrame
            config: İşlem yapılandırması
            
        Returns:
            İşlenmiş veri ve sinyalleri içeren sonuç
        """
        # 1. İndikatörleri hesapla
        indicator_data = self.indicator_manager.calculate_indicators(
            price_data, 
            config.get('indicators', [])
        )
        
        # 2. Sinyalleri üret
        signals = self.strategy_manager.generate_signals(
            indicator_data, 
            config.get('strategies', [])
        )
        
        # 3. Sinyalleri filtrele
        filtered_signals = self.filter_manager.apply_filters(
            indicator_data, 
            signals, 
            config.get('filters', [])
        )
        
        # 4. Sinyal gücünü hesapla
        signal_strength = self.strength_manager.calculate_strength(
            indicator_data, 
            filtered_signals, 
            config.get('strength_calculators', [])
        )
        
        # 5. ML tahminleri ekle (opsiyonel)
        if self.ml_manager and config.get('use_ml', False):
            ml_predictions = self.ml_manager.get_predictions(
                indicator_data, 
                config.get('ml_config', {})
            )
        else:
            ml_predictions = None
        
        # Sonuçları döndür
        return {
            'indicator_data': indicator_data,
            'signals': signals,
            'filtered_signals': filtered_signals,
            'signal_strength': signal_strength,
            'ml_predictions': ml_predictions
        }