# signal_engine/integrated_system.py - YENİ DOSYA OLUŞTUR

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Import our new systems
try:
    from signal_engine.signal_filter_system import FilterManager
    from signal_engine.signal_result_test_advance.risk_management import RiskRewardOptimizer, PositionSizer
    from signal_engine.signal_result_test_advance.position_management import PositionManager
except ImportError as e:
    print(f"⚠️ Import hatası: {e}")
    print("Lütfen yeni dosyaları oluşturun ve modül yollarını kontrol edin")

logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """Tüm iyileştirmeleri birleştiren ana trading sistemi"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize integrated trading system
        
        Args:
            config: Sistem konfigürasyonu
        """
        self.config = config or {}
        
        # Alt sistemleri başlat
        self._initialize_subsystems()
        
        # Performance tracking
        self.performance_metrics = {
            "total_signals": 0,
            "filtered_signals": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_rr_ratio": 0.0
        }
        
        logger.info("🚀 Integrated Trading System initialized")
    
    def _initialize_subsystems(self):
        """Alt sistemleri başlatır"""
        
        # 1. Filter System
        self.filter_manager = FilterManager()
        filter_mode = self.config.get("filter_mode", "lenient")
        
        if filter_mode == "strict":
            self.filter_manager.set_strict_mode()
        elif filter_mode == "balanced":
            self.filter_manager.set_balanced_mode()
        else:
            self.filter_manager.set_lenient_mode()
        
        logger.info(f"🔍 Filter system: {filter_mode} mode")
        
        # 2. Risk/Reward Optimizer
        rr_config = self.config.get("risk_reward", {})
        self.rr_optimizer = RiskRewardOptimizer(rr_config)
        
        logger.info(f"🎯 RR Optimizer: target={self.rr_optimizer.target_rr_ratio}")
        
        # 3. Position Manager
        position_config = self.config.get("position_management", {})
        self.position_manager = PositionManager(position_config)
        
        logger.info(f"📊 Position Manager: max_pos={self.position_manager.max_positions}")
        
        # 4. Position Sizer
        account_balance = self.config.get("account_balance", 10000)
        risk_per_trade = self.config.get("risk_per_trade", 2.0)
        self.position_sizer = PositionSizer(account_balance, risk_per_trade)
        
        logger.info(f"💰 Position Sizer: balance=${account_balance}, risk={risk_per_trade}%")
    
    def process_dataframe(self, df: pd.DataFrame, apply_all: bool = True) -> pd.DataFrame:
        """
        DataFrame'i tüm sistemlerden geçirir
        
        Args:
            df: Ham trading verisi
            apply_all: Tüm sistemleri uygula
            
        Returns:
            İşlenmiş DataFrame
        """
        logger.info(f"🔄 Processing DataFrame: {len(df)} rows")
        
        result_df = df.copy()
        
        # 1. FILTER SYSTEM
        logger.info("🔍 Step 1: Applying filter system...")
        original_passed = result_df.get("signal_passed_filter", pd.Series(False)).sum()
        
        result_df = self.filter_manager._apply_strength_only_filter(
            result_df, 
            min_strength=self.filter_manager._min_strength_required
        )
        
        new_passed = result_df["signal_passed_filter"].sum()
        logger.info(f"   Filter results: {original_passed} → {new_passed} signals passed")
        
        # 2. RISK/REWARD OPTIMIZATION
        if apply_all:
            logger.info("🎯 Step 2: Applying risk/reward optimization...")
            result_df = self.rr_optimizer.optimize_dataframe(result_df)
            
            avg_rr = result_df["optimized_rr"].dropna().mean()
            logger.info(f"   RR optimization: avg ratio = {avg_rr:.2f}")
        
        # 3. POSITION MANAGEMENT
        if apply_all:
            logger.info("📊 Step 3: Applying position management...")
            result_df = self.position_manager.apply_to_dataframe(result_df)
            
            max_open = result_df["open_positions_count"].max()
            logger.info(f"   Position management: max concurrent = {max_open}")
        
        # 4. PERFORMANCE METRICS
        self._update_performance_metrics(result_df)
        
        logger.info("✅ DataFrame processing completed")
        return result_df
    
    def _update_performance_metrics(self, df: pd.DataFrame):
        """Performans metriklerini günceller"""
        
        # Sinyal sayıları
        total_long = df.get("long_signal", pd.Series(False)).sum()
        total_short = df.get("short_signal", pd.Series(False)).sum()
        self.performance_metrics["total_signals"] = total_long + total_short
        
        self.performance_metrics["filtered_signals"] = df.get("signal_passed_filter", pd.Series(False)).sum()
        
        # RR ratio
        if "optimized_rr" in df.columns:
            avg_rr = df["optimized_rr"].dropna().mean()
            self.performance_metrics["avg_rr_ratio"] = avg_rr if not pd.isna(avg_rr) else 0.0
        
        # Position sayıları
        if "position_action" in df.columns:
            opened = df["position_action"].str.contains("OPEN_", na=False).sum()
            closed = df["position_action"].str.contains("CLOSE_", na=False).sum()
            self.performance_metrics["positions_opened"] = opened
            self.performance_metrics["positions_closed"] = closed
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Detaylı performans raporu"""
        metrics = self.performance_metrics.copy()
        
        # Ek hesaplamalar
        if metrics["total_signals"] > 0:
            filter_rate = (metrics["filtered_signals"] / metrics["total_signals"]) * 100
            metrics["filter_pass_rate"] = filter_rate
        
        if metrics["positions_closed"] > 0:
            metrics["position_closure_rate"] = (metrics["positions_closed"] / metrics["positions_opened"]) * 100
        
        return metrics
    
    def quick_fix_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        CSV dosyasını hızlıca düzeltir
        
        Args:
            csv_path: Giriş CSV dosya yolu
            output_path: Çıkış CSV dosya yolu
            
        Returns:
            Düzeltilmiş DataFrame
        """
        logger.info(f"🔧 Quick fixing CSV: {csv_path}")
        
        # CSV'yi oku
        df = pd.read_csv(csv_path)
        original_len = len(df)
        
        print(f"📂 CSV loaded: {original_len} rows")
        print(f"📊 Original stats:")
        print(f"   - Filter passed: {df.get('signal_passed_filter', pd.Series(False)).sum()}")
        print(f"   - Avg RR ratio: {df.get('rr_ratio', pd.Series([1.0])).mean():.2f}")
        print(f"   - Long signals: {df.get('long_signal', pd.Series(False)).sum()}")
        print(f"   - Short signals: {df.get('short_signal', pd.Series(False)).sum()}")
        
        # Sistemden geçir
        fixed_df = self.process_dataframe(df, apply_all=True)
        
        print(f"\n✅ After processing:")
        print(f"   - Filter passed: {fixed_df.get('signal_passed_filter', pd.Series(False)).sum()}")
        print(f"   - Avg RR ratio: {fixed_df.get('optimized_rr', pd.Series([1.0])).dropna().mean():.2f}")
        print(f"   - Max open positions: {fixed_df.get('open_positions_count', pd.Series([0])).max()}")
        
        # Kaydet
        if output_path is None:
            output_path = csv_path.replace('.csv', '_fixed.csv')
        
        fixed_df.to_csv(output_path, index=False)
        logger.info(f"💾 Fixed CSV saved: {output_path}")
        
        return fixed_df


# 🧪 TEST FONKSİYONLARI VE ÖRNEKLER

def create_sample_config() -> Dict[str, Any]:
    """Örnek sistem konfigürasyonu"""
    return {
        "filter_mode": "lenient",  # lenient, balanced, strict
        "account_balance": 10000,
        "risk_per_trade": 2.0,
        
        "risk_reward": {
            "target_rr_ratio": 2.5,
            "min_rr_ratio": 1.8,
            "stop_atr": 1.0,
            "tp_atr": 2.5,
            "update_original": True
        },
        
        "position_management": {
            "max_positions": 8,
            "max_per_symbol": 2,
            "max_risk_pct": 2.0,
            "max_total_risk": 10.0,
            "default_timeout": 100,
            "trailing_stops": True,
            "trailing_atr": 1.5
        }
    }

def test_integrated_system():
    """Entegre sistemi test eder"""
    print("🧪 INTEGRATED SYSTEM TEST")
    print("=" * 60)
    
    # Test verisi oluştur
    np.random.seed(42)
    n_rows = 50
    
    test_data = {
        'close': np.random.normal(100, 10, n_rows),
        'atr': np.random.normal(2.0, 0.5, n_rows),
        'signal_strength': np.random.randint(30, 90, n_rows),
        'long_signal': np.random.choice([True, False], n_rows, p=[0.1, 0.9]),
        'short_signal': np.random.choice([True, False], n_rows, p=[0.1, 0.9]),
        'signal_passed_filter': [False] * n_rows,  # Hepsi False başlangıçta
        'rr_ratio': [1.0] * n_rows,  # Düşük RR
        'sl': [0.0] * n_rows,
        'tp': [0.0] * n_rows
    }
    
    df = pd.DataFrame(test_data)
    
    print("📊 ORIGINAL DATA:")
    print(f"Total signals: {df['long_signal'].sum() + df['short_signal'].sum()}")
    print(f"Passed filter: {df['signal_passed_filter'].sum()}")
    print(f"Avg RR ratio: {df['rr_ratio'].mean():.2f}")
    
    # Sistem oluştur ve test et
    config = create_sample_config()
    system = IntegratedTradingSystem(config)
    
    # Veriyi işle
    result_df = system.process_dataframe(df)
    
    print(f"\n📈 PROCESSED DATA:")
    print(f"Passed filter: {result_df['signal_passed_filter'].sum()}")
    print(f"Avg RR ratio: {result_df.get('optimized_rr', pd.Series([1.0])).dropna().mean():.2f}")
    print(f"Max open positions: {result_df.get('open_positions_count', pd.Series([0])).max()}")
    
    # Performans raporu
    print(f"\n📋 PERFORMANCE REPORT:")
    report = system.get_performance_report()
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    return result_df, system

def compare_before_after_csv(original_csv: str):
    """Orijinal ve düzeltilmiş CSV'yi karşılaştırır"""
    print("📊 BEFORE/AFTER COMPARISON")
    print("=" * 50)
    
    # Orijinal veriyi oku
    original_df = pd.read_csv(original_csv)
    
    # Sistem oluştur
    system = IntegratedTradingSystem(create_sample_config())
    
    # Düzelt
    fixed_df = system.quick_fix_csv(original_csv)
    
    print("\n📈 IMPROVEMENT SUMMARY:")
    print("=" * 30)
    
    # Filter geçme oranı
    orig_passed = original_df.get('signal_passed_filter', pd.Series(False)).sum()
    new_passed = fixed_df.get('signal_passed_filter', pd.Series(False)).sum()
    total_signals = (original_df.get('long_signal', pd.Series(False)).sum() + 
                    original_df.get('short_signal', pd.Series(False)).sum())
    
    print(f"🔍 Filter Pass Rate:")
    print(f"   Before: {orig_passed}/{total_signals} ({orig_passed/total_signals*100:.1f}%)")
    print(f"   After:  {new_passed}/{total_signals} ({new_passed/total_signals*100:.1f}%)")
    print(f"   Improvement: +{new_passed-orig_passed} signals")
    
    # RR Ratio
    orig_rr = original_df.get('rr_ratio', pd.Series([1.0])).mean()
    new_rr = fixed_df.get('optimized_rr', pd.Series([1.0])).dropna().mean()
    
    print(f"\n🎯 Risk/Reward Ratio:")
    print(f"   Before: {orig_rr:.2f}")
    print(f"   After:  {new_rr:.2f}")
    print(f"   Improvement: +{new_rr-orig_rr:.2f}")
    
    # Position Management
    max_open = fixed_df.get('open_positions_count', pd.Series([0])).max()
    print(f"\n📊 Position Management:")
    print(f"   Max concurrent positions: {max_open}")
    print(f"   Position control: ACTIVE")
    
    return fixed_df

# 🚀 HIZLI BAŞLATMA FONKSİYONU

def quick_start_fix(csv_file_path: str):
    """
    CSV dosyası için tek komutla hızlı düzeltme
    
    Args:
        csv_file_path: CSV dosya yolu
        
    Returns:
        Düzeltilmiş DataFrame
    """
    print("🚀 QUICK START - TRADING SYSTEM FIX")
    print("=" * 50)
    
    # Otomatik konfigürasyon
    config = {
        "filter_mode": "lenient",      # Esnek filter
        "account_balance": 10000,
        "risk_per_trade": 2.0,
        
        "risk_reward": {
            "target_rr_ratio": 2.5,   # Hedef RR 2.5
            "min_rr_ratio": 1.8,      # Min RR 1.8
            "stop_atr": 1.0,          # Stop = ATR * 1.0
            "tp_atr": 2.5,            # TP = ATR * 2.5
        },
        
        "position_management": {
            "max_positions": 10,       # Max 10 açık pozisyon
            "default_timeout": 100,    # 100 bar timeout
            "trailing_stops": True,    # Trailing stops aktif
        }
    }
    
    # Sistem oluştur ve çalıştır
    system = IntegratedTradingSystem(config)
    fixed_df = system.quick_fix_csv(csv_file_path)
    
    # Özet rapor
    report = system.get_performance_report()
    
    print(f"\n🎉 MISSION ACCOMPLISHED!")
    print(f"✅ Filter pass rate: {report.get('filter_pass_rate', 0):.1f}%")
    print(f"✅ Avg RR ratio: {report.get('avg_rr_ratio', 0):.2f}")
    print(f"✅ Positions managed: {report.get('positions_opened', 0)}")
    
    return fixed_df

# Ana test
if __name__ == "__main__":
    # Test entegre sistemi
    test_result, system = test_integrated_system()
    
    print(f"\n{'='*60}")
    print("🎯 QUICK START READY!")
    print("Use: quick_start_fix('your_file.csv')")
    print("='*60")