"""
Ana çalıştırıcı modül - VSCode'dan doğrudan çalıştırılabilir
ENHANCED: Added comprehensive validation, analysis and error handling
"""
import os
import sys
import json
from typing import Dict, List, Any, Optional
import logging

# Modül yolunu ayarla - mevcut dizinin bir üst dizinini Python yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Loglama yapılandırması
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backtest modüllerini içe aktar
from backtest.utils.config_loader import load_env_config
from backtest.runners.single_backtest import run_single_backtest
from backtest.runners.batch_backtest import run_batch_backtest
from backtest.utils.config_viewer import print_config_details, print_enhanced_config_summary
from backtest.utils.signal_engine_components import check_signal_engine_components

def print_registered_indicators():
    """
    Kayıtlı indikatörleri gösterir
    """
    # Bu kodu bir yerde çalıştırın (örneğin main.py dosyasına ekleyin)

    from backtest.utils.indicator_helper import check_available_indicators, get_recommended_config

    # Kullanılabilir indikatörleri göster
    available = check_available_indicators()
    print(f"Found {len(available)} indicators in registry.\n")

    # Tavsiye edilen yapılandırmayı al
    recommended = get_recommended_config()

    print("\nRECOMMENDED CONFIGURATION:")
    print("=" * 80)
    print(recommended["recommended_env"])
    print("=" * 80)
    print("\nThis configuration includes only the indicators available in the Signal Engine registry.")

def validate_backtest_config(env_config: Dict[str, Any]) -> List[str]:
    """
    ENHANCED: Validate backtest configuration and return warnings
    
    Args:
        env_config: Environment configuration
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Risk validation
    risk_per_trade = float(env_config.get("risk_per_trade", 0.01))
    if risk_per_trade > 0.05:
        warnings.append(f"⚠️ High risk per trade: {risk_per_trade*100:.1f}% (recommended: <5%)")
    elif risk_per_trade < 0.005:
        warnings.append(f"⚠️ Very low risk per trade: {risk_per_trade*100:.1f}% (may limit profits)")
    
    # Leverage validation
    leverage = float(env_config.get("leverage", 1.0))
    if leverage > 10:
        warnings.append(f"⚠️ Very high leverage: {leverage}x (increases risk significantly)")
    elif leverage > 5:
        warnings.append(f"⚠️ High leverage: {leverage}x (use with caution)")
    
    # SL/TP ratio validation
    sl_mult = float(env_config.get("sl_multiplier", 1.5))
    tp_mult = float(env_config.get("tp_multiplier", 3.0))
    rr_ratio = tp_mult / sl_mult
    
    if rr_ratio < 1.5:
        warnings.append(f"⚠️ Low Risk/Reward ratio: {rr_ratio:.2f} (recommended: >1.5)")
    elif rr_ratio > 5:
        warnings.append(f"⚠️ Very high Risk/Reward ratio: {rr_ratio:.2f} (may reduce win rate)")
    
    # Position direction validation
    pos_dir = env_config.get("position_direction", {})
    if not pos_dir.get("Long", True) and not pos_dir.get("Short", True):
        warnings.append("❌ Both Long and Short disabled - no trades will be executed")
    
    # Commission validation
    commission = float(env_config.get("commission_rate", 0.001))
    if commission > 0.005:
        warnings.append(f"⚠️ High commission rate: {commission*100:.3f}% (may impact profitability)")
    
    # Symbol and interval validation
    symbol = env_config.get("symbol")
    interval = env_config.get("interval")
    if not symbol:
        warnings.append("❌ Symbol not configured")
    if not interval:
        warnings.append("❌ Interval not configured")
    
    # Database URL validation
    db_url = env_config.get("db_url")
    if not db_url:
        warnings.append("❌ Database URL not configured")
    
    # Initial balance validation
    initial_balance = float(env_config.get("initial_balance", 10000))
    if initial_balance < 1000:
        warnings.append(f"⚠️ Low initial balance: ${initial_balance:,.2f} (may affect position sizing)")
    
    return warnings

def calculate_risk_score(metrics: Dict[str, Any]) -> float:
    """
    ENHANCED: Calculate risk score (0-10, lower is better)
    """
    score = 5.0  # Base score
    
    # Drawdown penalty
    max_dd = metrics.get('max_drawdown_pct', 0)
    if max_dd > 20:
        score += 3
    elif max_dd > 10:
        score += 2
    elif max_dd > 5:
        score += 1
    
    # Sharpe ratio bonus
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 2:
        score -= 2
    elif sharpe > 1:
        score -= 1
    elif sharpe < 0:
        score += 2
    
    # Profit factor consideration
    pf = metrics.get('profit_factor', 0)
    if pf < 1:
        score += 2
    elif pf > 2:
        score -= 1
    
    # Consecutive losses penalty
    max_consec_loss = metrics.get('max_consecutive_losses', 0)
    if max_consec_loss > 10:
        score += 2
    elif max_consec_loss > 5:
        score += 1
    
    return max(0, min(10, score))

def calculate_performance_rating(result: Dict[str, Any], metrics: Dict[str, Any]) -> float:
    """
    ENHANCED: Calculate performance rating (0-10, higher is better)
    """
    score = 5.0  # Base score
    
    # ROI bonus/penalty
    roi = result.get('roi_pct', 0)
    if roi > 50:
        score += 3
    elif roi > 20:
        score += 2
    elif roi > 10:
        score += 1
    elif roi < 0:
        score -= 3
    elif roi < 5:
        score -= 1
    
    # Win rate bonus
    win_rate = metrics.get('win_rate', 0)
    if win_rate > 70:
        score += 2
    elif win_rate > 60:
        score += 1
    elif win_rate < 40:
        score -= 1
    elif win_rate < 30:
        score -= 2
    
    # Trade count consideration
    total_trades = result.get('total_trades', 0)
    if total_trades < 10:
        score -= 2  # Not enough trades for reliable statistics
    elif total_trades > 100:
        score += 1  # Good sample size
    
    # Profit factor bonus
    pf = metrics.get('profit_factor', 0)
    if pf > 2:
        score += 1
    elif pf > 1.5:
        score += 0.5
    
    # Risk-adjusted return (Sharpe ratio)
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 2:
        score += 1
    elif sharpe > 1:
        score += 0.5
    
    return max(0, min(10, score))

def get_risk_level(score: float) -> str:
    """Get risk level description"""
    if score <= 3:
        return "LOW RISK"
    elif score <= 6:
        return "MODERATE RISK"
    elif score <= 8:
        return "HIGH RISK"
    else:
        return "VERY HIGH RISK"

def get_performance_level(score: float) -> str:
    """Get performance level description"""
    if score >= 8:
        return "EXCELLENT"
    elif score >= 6:
        return "GOOD"
    elif score >= 4:
        return "AVERAGE"
    elif score >= 2:
        return "POOR"
    else:
        return "VERY POOR"

def analyze_backtest_results(result: Dict[str, Any]) -> None:
    """
    ENHANCED: result analysis with detailed breakdown
    
    Args:
        result: Backtest result dictionary
    """
    if result.get("status") != "success":
        print(f"❌ Backtest failed: {result.get('message', 'Unknown error')}")
        return
    
    backtest_result = result.get("result", {})
    metrics = backtest_result.get("metrics", {})
    
    print("\n" + "="*80)
    print("📊 ENHANCED BACKTEST RESULTS ANALYSIS")
    print("="*80)
    
    # Core Performance Metrics
    print(f"\n🎯 CORE PERFORMANCE:")
    print(f"   Total Trades: {backtest_result.get('total_trades', 0)}")
    print(f"   Win Rate: {metrics.get('win_rate', 0):.2f}%")
    print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"   ROI: {backtest_result.get('roi_pct', 0):.2f}%")
    print(f"   Net Profit: ${backtest_result.get('profit_loss', 0):.2f}")
    
    # Financial Details
    print(f"\n💰 FINANCIAL BREAKDOWN:")
    print(f"   Gross Profit: ${metrics.get('gross_profit', 0):.2f}")
    print(f"   Gross Loss: ${metrics.get('gross_loss', 0):.2f}")
    print(f"   Winning Trades: {metrics.get('winning_trades', 0)}")
    print(f"   Losing Trades: {metrics.get('losing_trades', 0)}")
    if metrics.get('winning_trades', 0) > 0:
        print(f"   Average Win: ${metrics.get('average_win', 0):.2f}")
    if metrics.get('losing_trades', 0) > 0:
        print(f"   Average Loss: ${metrics.get('average_loss', 0):.2f}")
    
    # Risk Metrics
    print(f"\n⚠️ RISK METRICS:")
    print(f"   Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    print(f"   Recovery Factor: {metrics.get('recovery_factor', 0):.2f}")
    print(f"   VaR (95%): {metrics.get('var_95', 0):.2f}%")
    print(f"   Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}")
    print(f"   Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}")
    
    # Direction Performance
    if 'direction_performance' in metrics:
        print(f"\n📈 DIRECTION PERFORMANCE:")
        for direction, stats in metrics['direction_performance'].items():
            print(f"   {direction}:")
            print(f"     Trades: {stats.get('count', 0)}")
            print(f"     Win Rate: {stats.get('win_rate', 0):.2f}%")
            print(f"     Total PnL: ${stats.get('total_pnl', 0):.2f}")
            print(f"     Avg Trade: ${stats.get('avg_pnl', 0):.2f}")
            print(f"     Profit Factor: {stats.get('profit_factor', 0):.2f}")
    
    # Trade Quality Analysis
    if backtest_result.get('total_trades', 0) > 0:
        print(f"\n🔍 TRADE QUALITY:")
        avg_win = metrics.get('average_win', 0)
        avg_loss = metrics.get('average_loss', 0)
        print(f"   Average Win: ${avg_win:.2f}")
        print(f"   Average Loss: ${avg_loss:.2f}")
        print(f"   Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   Win/Loss Ratio: ∞")
        print(f"   Largest Win: ${metrics.get('largest_win', 0):.2f}")
        print(f"   Largest Loss: ${metrics.get('largest_loss', 0):.2f}")
        print(f"   Average R:R Ratio: {metrics.get('avg_rr_ratio', 0):.2f}")
        
        # Outcome distribution
        if 'outcome_distribution' in metrics:
            print(f"\n📊 OUTCOME DISTRIBUTION:")
            total_outcomes = sum(metrics['outcome_distribution'].values())
            for outcome, count in metrics['outcome_distribution'].items():
                pct = (count / total_outcomes * 100) if total_outcomes > 0 else 0
                print(f"   {outcome}: {count} ({pct:.1f}%)")
    
    # Risk Assessment
    print(f"\n🚨 RISK ASSESSMENT:")
    risk_score = calculate_risk_score(metrics)
    print(f"   Risk Score: {risk_score:.1f}/10 ({get_risk_level(risk_score)})")
    
    # Performance Rating
    performance_rating = calculate_performance_rating(backtest_result, metrics)
    print(f"   Performance Rating: {performance_rating:.1f}/10 ({get_performance_level(performance_rating)})")
    
    # Trading Period
    if 'trading_period' in metrics:
        period = metrics['trading_period']
        print(f"\n📅 TRADING PERIOD:")
        print(f"   Start: {period.get('start', 'Unknown')}")
        print(f"   End: {period.get('end', 'Unknown')}")
        print(f"   Duration: {period.get('duration_days', 0)} days")
    
    # Configuration Summary
    config = backtest_result.get('config', {})
    if config:
        print(f"\n⚙️ CONFIGURATION:")
        print(f"   Risk per Trade: {config.get('risk_per_trade', 0)*100:.2f}%")
        pos_dir = config.get('position_direction', {})
        directions = []
        if pos_dir.get('Long', False):
            directions.append('Long')
        if pos_dir.get('Short', False):
            directions.append('Short')
        print(f"   Directions: {', '.join(directions) if directions else 'None'}")


def run_backtest(mode: str = "single", config_id: str = "default", custom_config: Dict[str, Any] = None):
    """
    ENHANCED: Backtest çalıştırır - VSCode'dan direkt çağrılabilir with validation and analysis
    
    Args:
        mode: Çalıştırma modu ("single" veya "batch")
        config_id: Konfigürasyon ID'si (single mod için)
        custom_config: Özel konfigürasyon sözlüğü (varsayılanları ezmek için)
    """
    # Çevre değişkenlerinden konfigürasyon yükle
    env_config = load_env_config()
    #print_config_details(env_config, "BACKTEST CONFIGURATION")

    #print_registered_indicators()
    #check_signal_engine_components()

    # Özel konfigürasyonu entegre et (varsa)
    if custom_config:
        for key, value in custom_config.items():
            env_config[key] = value
    
    # ENHANCED: Configuration validation
    config_warnings = validate_backtest_config(env_config)
    if config_warnings:
        print("\n🚨 CONFIGURATION WARNINGS:")
        for warning in config_warnings:
            print(f"   {warning}")
        print("")
    
    # ENHANCED: Print configuration summary
    print_enhanced_config_summary(env_config)
    
    # Çıktı dizini oluştur
    output_dir = env_config.get("results_dir", "backtest/results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Gerekli parametreleri kontrol et
    symbol = env_config.get("symbol")
    interval = env_config.get("interval")
    
    if not symbol or not interval:
        logger.error("Symbol ve interval parametreleri gerekli. Lütfen ENV dosyasını kontrol edin.")
        return
    backtest_params = {
        "initial_balance": env_config.get("initial_balance"),
        "risk_per_trade": env_config.get("risk_per_trade"),
        "sl_multiplier": env_config.get("sl_multiplier"),
        "tp_multiplier": env_config.get("tp_multiplier"),
        "leverage": env_config.get("leverage"),
        "position_direction": env_config.get("position_direction"),
        "commission_rate": env_config.get("commission_rate"),
        "max_holding_bars": env_config.get("max_holding_bars", 500),
        
        # ✅ TP/Commission filter parameters
        "min_tp_commission_ratio": env_config.get("min_tp_commission_ratio", 3.0),
        "max_commission_impact_pct": env_config.get("max_commission_impact_pct", 15.0),
        "min_position_size": env_config.get("min_position_size", 800.0),
        "min_net_rr_ratio": env_config.get("min_net_rr_ratio", 1.5),
        "enable_tp_commission_filter": env_config.get("enable_tp_commission_filter", False)
    }

    # ✅ VALIDATION: Check if filter parameters are being passed correctly
    logger.info("🔧 Backtest Parameters Validation:")
    logger.info(f"   enable_tp_commission_filter: {backtest_params.get('enable_tp_commission_filter')}")
    logger.info(f"   commission_rate: {backtest_params.get('commission_rate')}")
    logger.info(f"   min_tp_commission_ratio: {backtest_params.get('min_tp_commission_ratio')}")
    logger.info(f"   max_commission_impact_pct: {backtest_params.get('max_commission_impact_pct')}")
    logger.info(f"   min_position_size: {backtest_params.get('min_position_size')}")
    logger.info(f"   min_net_rr_ratio: {backtest_params.get('min_net_rr_ratio')}")
    
    if mode == "single":
        logger.info(f"🚀 {symbol} {interval} için tek backtest çalıştırılıyor (Config ID: {config_id})")
         # ✅ TP/Commission filter status
        if backtest_params.get("enable_tp_commission_filter", False):
            logger.info("🔧 TP/Commission filter ENABLED - Only profitable trades will be opened")
        else:
            logger.info("⚠️ TP/Commission filter DISABLED - All valid signals will be processed")
        # Tek backtest çalıştır
        result = run_single_backtest(
            symbol=symbol,
            interval=interval,
            db_url=env_config.get("db_url"),
            output_dir=os.path.join(output_dir, "single"),
            backtest_params=backtest_params,  # ✅ TP/Commission parameters dahil
            indicators_config=env_config.get("indicators"),
            config_id=config_id
        )
        
        if result.get("status") == "success":
            logger.info(f"✅ Backtest başarıyla tamamlandı. Sonuçlar: {output_dir}/single klasörüne kaydedildi.")
            
            # ENHANCED: Comprehensive result analysis
            analyze_backtest_results(result)
            
            # Kısa özet yazdır (mevcut kod korundu ama enhanced analysis'den sonra)
            logger.info(f"📄 Summary saved to output directory for detailed review.")
            return result  
        else:
            logger.error(f"❌ Backtest sırasında hata oluştu: {result.get('message')}")
            return None 
    
    elif mode == "batch":
        # Config CSV yolunu belirle
        config_csv = os.path.join("backtest", "config", "config_combinations.csv")
        
        if not os.path.exists(config_csv):
            logger.error(f"❌ Konfigürasyon CSV dosyası bulunamadı: {config_csv}")
            return
        
        logger.info(f"🚀 CSV konfigürasyonları ile toplu backtest çalıştırılıyor: {config_csv}")

        # ✅ TP/Commission filter status for batch
        if backtest_params.get("enable_tp_commission_filter", False):
            logger.info("🔧 BATCH MODE: TP/Commission filter ENABLED for all configurations")
            logger.info(f"   Min TP/Commission ratio: {backtest_params['min_tp_commission_ratio']}x")
            logger.info(f"   Min position size: ${backtest_params['min_position_size']}")
        else:
            logger.info("⚠️ BATCH MODE: TP/Commission filter DISABLED")
        
        # Maksimum işlemci sayısını belirle
        max_workers = os.cpu_count() - 1  # Bir CPU boşta bırak
        
        auto_fetch = custom_config.get("auto_fetch_data", True) if custom_config else True

        # Toplu backtest çalıştır
        result = run_batch_backtest(
            symbol=symbol,
            interval=interval,
            config_csv_path=config_csv,
            db_url=env_config.get("db_url"),
            output_dir=os.path.join(output_dir, "batch"),
            backtest_params=backtest_params,  # ✅ TP/Commission parameters dahil
            max_workers=max_workers,
            auto_fetch_data=auto_fetch
        )
        
        if result.get("status") == "success":
            logger.info(f"✅ Toplu backtest başarıyla tamamlandı. Sonuçlar: {output_dir}/batch klasörüne kaydedildi.")
            logger.info(f"   - Toplam Konfigürasyon: {result.get('total_configs')}")
            logger.info(f"   - Tamamlanan: {result.get('completed_configs')}")
            logger.info(f"   - En İyi ROI: {result.get('best_roi_pct'):.2f}% (Config: {result.get('best_roi_config')})")
            logger.info(f"   - En İyi Kazanç Oranı: {result.get('best_winrate_pct'):.2f}% (Config: {result.get('best_winrate_config')})")
            # ✅ ENHANCED: TP/Commission filter effectiveness
            if result.get('tp_commission_filter_enabled', False):
                filter_stats = result.get('filter_statistics', {})
                logger.info(f"\n🔧 TP/COMMISSION FILTER EFFECTIVENESS:")
                logger.info(f"   Total signals: {filter_stats.get('total_signals', 0)}")
                logger.info(f"   Filtered signals: {filter_stats.get('filtered_signals', 0)}")
                logger.info(f"   Configs with trades: {filter_stats.get('configs_with_trades', 0)}")
                logger.info(f"   Configs with no trades: {filter_stats.get('configs_with_no_trades', 0)}")
                
            # ENHANCED: Additional batch statistics
            if result.get('profitable_configs_pct'):
                logger.info(f"   - Karlı Konfigürasyonlar: {result.get('profitable_configs_pct'):.1f}%")
            if result.get('average_roi'):
                logger.info(f"   - Ortalama ROI: {result.get('average_roi'):.2f}%")
            
            return result  # ✅ BUNU EKLE
        else:
            logger.error(f"❌ Toplu backtest sırasında hata oluştu: {result.get('message')}")
            return None 
    
    else:
        logger.error(f"❌ Bilinmeyen çalıştırma modu: {mode}. 'single' veya 'batch' kullanın.")
        return None 

def enhanced_run_backtest(mode: str = "single", config_id: str = "default", 
                         custom_config: Dict[str, Any] = None, validate: bool = True):
    """
    ENHANCED: Enhanced backtest runner with validation and detailed analysis
    
    Args:
        mode: Run mode ("single" or "batch")
        config_id: Configuration ID
        custom_config: Custom configuration overrides
        validate: Whether to validate configuration before running
    """
    # Load environment configuration
    env_config = load_env_config()
    
    # Apply custom config
    if custom_config:
        for key, value in custom_config.items():
            env_config[key] = value
    
    # Validate configuration
    if validate:
        warnings = validate_backtest_config(env_config)
        if warnings:
            print("\n🚨 CONFIGURATION WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")
            
            # Check for critical errors
            critical_errors = [w for w in warnings if w.startswith("❌")]
            if critical_errors:
                print("\n❌ Critical configuration errors found. Please fix before proceeding:")
                for error in critical_errors:
                    print(f"   {error}")
                return None
            
            response = input("\nDo you want to continue? (y/N): ")
            if response.lower() != 'y':
                print("Backtest cancelled.")
                return None
    
    # Run original backtest logic
    result = run_backtest(mode, config_id, custom_config)
    
    return result

if __name__ == "__main__":
    # ENHANCED: VSCode'dan doğrudan çalıştırmak için yapılandırma with validation
    # Burada mod, config_id ve özel parametreler ayarlanabilir
    
    # Çalıştırma modu: "single" veya "batch"
    RUN_MODE = "batch"  # Tek backtest için "single", toplu backtest için "batch"
    
    # Tek backtest için konfigürasyon ID'si
    CONFIG_ID = "default"
    
    # Özel konfigürasyon (çevre değişkenlerini ezmek için)
    CONFIG_ID = "tp_commission_optimized"
    
    # ✅ TP/COMMISSION OPTIMIZE EDİLMİŞ KONFIGÜRASYON
    CUSTOM_CONFIG = {
        # 🔧 CORRECT BINANCE MAKER RATE
           # ✅ REALISTIC COMMISSION
        "commission_rate": 0.0002,  # Binance maker 0.01%
        
        # ✅ AGGRESSIVE RISK MANAGEMENT  
        "risk_per_trade": 0.04,     # %4 risk per trade
        "leverage": 5.0,            # Full 5x leverage
        "initial_balance": 10000.0,
        
        # ✅ TIGHT SL, HIGH TP
        "sl_multiplier": 0.8,       # Very tight SL (0.8x ATR)
        "tp_multiplier": 5.0,       # High TP (6.25:1 R:R ratio)
        
        # ✅ REALISTIC TP/COMMISSION FILTER
        "enable_tp_commission_filter": True,
        "min_tp_commission_ratio": 2.0,      # 2x is enough
        "max_commission_impact_pct": 50.0,   # Up to 50% OK for small trades
        "min_position_size": 50.0,           # $50 minimum (very low)
        "min_net_rr_ratio": 0.8,             # 0.8:1 net (after commission)
        
        # ✅ BOTH DIRECTIONS
        "position_direction": {"Long": True, "Short": True},
        
        # ✅ QUICK SCALPING
        "max_holding_bars": 20,  # 100 minutes max (5m x 20)
        "auto_fetch_data": True,  # Enhanced fetch'i çalıştır

        
        # ✅ RELAXED FILTERS FOR MORE TRADES
        "filters": {
            "market_regime": {},
            "min_checks": 1,        # Only 1 filter check required
            "min_strength": 15      # Very low strength requirement
        }
    }

    # 🎯 EXTREME %70+ ROI TARGETING CONFIG
    CUSTOM_CONFIG_EXTREME = {
        # 🔧 OPTIMIZED COMMISSION
        "commission_rate": 0.0001,  # 0.01% maker (daha düşük)
        
        # 🔧 EXTREME RISK MANAGEMENT
        "risk_per_trade": 0.08,     # %8 risk per trade (was %4)
        "leverage": 15.0,           # 15x leverage (maximum)
        "initial_balance": 10000.0,
        
        # 🔧 ULTRA TIGHT SL, EXTREME TP
        "sl_multiplier": 0.5,       # Çok sıkı SL (0.5x ATR)
        "tp_multiplier": 12.0,      # Çok yüksek TP (24:1 R:R)
        
        # 🔧 MINIMAL TP/COMMISSION FILTER
        "enable_tp_commission_filter": True,
        "min_tp_commission_ratio": 1.2,      # 1.2x (çok gevşek)
        "max_commission_impact_pct": 90.0,   # %90'a kadar
        "min_position_size": 20.0,           # $20 minimum
        "min_net_rr_ratio": 0.3,             # 0.3:1 net (çok gevşek)
        
        # ✅ BOTH DIRECTIONS
        "position_direction": {"Long": True, "Short": True},
        
        # 🔧 ULTRA FAST SCALPING
        "max_holding_bars": 10,  # 50 dakika max (5m x 10)
        "auto_fetch_data": True,
        
        # 🔧 MINIMAL FILTERS
        "filters": {
            "min_checks": 1,
            "min_strength": 1  # Minimum possible
        }
    }
    
    CUSTOM_CONFIG = CUSTOM_CONFIG_EXTREME  # Use extreme config for testing

    # ENHANCED: Backtest çalıştır with validation
    print("🚀 ENHANCED BACKTEST ENGINE STARTING...")
    print("="*50)
    
    result = enhanced_run_backtest(
        mode=RUN_MODE, 
        config_id=CONFIG_ID, 
        custom_config=CUSTOM_CONFIG,
        validate=True  # Enable validation
    )
    
    if result:
        print("\n✅ Enhanced backtest completed successfully!")
    else:
        print("\n❌ Enhanced backtest failed or was cancelled.")