"""
Tek backtest çalıştırma modülü
"""
import os
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union
import logging

from ..core.backtest_engine import BacktestEngine
from ..utils.data_loader import load_price_data, parse_indicators_config
from ..analysis.result_analyzer import analyze_single_result
from ..analysis.result_plotter import plot_single_result
from ..utils.config_loader import load_env_config

# Logger ayarla
logger = logging.getLogger(__name__)


def run_single_backtest(
    symbol: str,
    interval: str,
    db_url: str,
    output_dir: str,
    backtest_params: Dict[str, Any],
    indicators_config: Dict[str, Any] = None,
    config_id: str = "default"
) -> Dict[str, Any]:
    """
    Tek bir backtest çalıştırır ve sonuçları analiz eder
    
    Args:
        symbol: İşlem sembolü
        interval: Zaman aralığı
        db_url: Veritabanı bağlantı URL'si
        output_dir: Çıktı dizini
        backtest_params: Backtest parametreleri
        indicators_config: İndikatör konfigürasyonu (None ise çevre değişkenlerinden alınır)
        config_id: Konfigürasyon ID'si
        
    Returns:
        Backtest sonuçları
    """
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Çevre değişkenlerinden konfigürasyonu yükle
        env_config = load_env_config()
        
        # Veriyi yükle
        df = load_price_data(symbol, interval, db_url)
        logger.info(f"📈 Loaded {len(df)} data points for {symbol} {interval}")
        
        # İndikatör konfigürasyonunu kontrol et
        if indicators_config is None:
            # Çevre değişkenlerinden indikatör yapılandırmasını al
            try:
                indicators_config = {
                    "long": env_config["indicators"].get("long", {}),
                    "short": env_config["indicators"].get("short", {})
                }
                logger.info("✅ Loaded indicator configuration from environment.")
            except (KeyError, TypeError):
                indicators_config = {}
                logger.warning("⚠️ No indicators configuration found in environment. Using empty configuration.")
            
            # İndikatör konfigürasyonu boşsa kullanıcıya bildir
            if not indicators_config.get("long") and not indicators_config.get("short"):
                logger.warning("⚠️ Empty indicator configuration. Signals may not be generated.")
        
        # Backtest motorunu oluştur
        # ✅ ENHANCED: BacktestEngine'i TP/Commission parametreleri ile oluştur
        engine = BacktestEngine(
            symbol=symbol,
            interval=interval,
            # Temel parametreler
            initial_balance=backtest_params.get('initial_balance', 10000.0),
            risk_per_trade=backtest_params.get('risk_per_trade', 0.01),
            sl_multiplier=backtest_params.get('sl_multiplier', 1.5),
            tp_multiplier=backtest_params.get('tp_multiplier', 3.0),
            leverage=backtest_params.get('leverage', 1.0),
            position_direction=backtest_params.get('position_direction', {"Long": True, "Short": True}),
            commission_rate=backtest_params.get('commission_rate', 0.001),
            max_holding_bars=backtest_params.get('max_holding_bars', 500),
            
            # ✅ YENİ: TP/Commission filter parametreleri
            min_tp_commission_ratio=backtest_params.get('min_tp_commission_ratio', 3.0),
            max_commission_impact_pct=backtest_params.get('max_commission_impact_pct', 15.0),
            min_position_size=backtest_params.get('min_position_size', 800.0),
            min_net_rr_ratio=backtest_params.get('min_net_rr_ratio', 1.5),
            enable_tp_commission_filter=backtest_params.get('enable_tp_commission_filter', False)
        )
        
        # Strateji, filtre ve güç hesaplayıcı yapılandırmalarını çevre değişkenlerinden al
        strategies_config = env_config.get("strategies", {})
        filters_config = env_config.get("filters", {})
        strength_config = env_config.get("strength", {})
        
        # Kapsamlı konfigürasyon oluştur
        config = {
            "indicators": indicators_config,
            "strategies": strategies_config,
            "filters": filters_config,
            "strength": strength_config
        }
        
        # Signal Engine bileşenlerini yapılandır
        engine.configure_signal_engine(
            indicators_config=config.get('indicators', {}),
            strategies_config=config.get('strategies', {}),
            strength_config=config.get('strength', {}),
            filter_config=config.get('filters', {})
        )
        
        # ✅ TP/Commission filter status logging
        if backtest_params.get('enable_tp_commission_filter', False):
            logger.info("🔧 Signal Engine configured WITH TP/Commission filtering:")
            logger.info(f"   - Min TP/Commission ratio: {backtest_params.get('min_tp_commission_ratio', 3.0)}x")
            logger.info(f"   - Max commission impact: {backtest_params.get('max_commission_impact_pct', 15.0)}%")
            logger.info(f"   - Min position size: ${backtest_params.get('min_position_size', 800.0)}")
        else:
            logger.info("🔧 Signal Engine configured WITHOUT TP/Commission filtering")
        logger.info("🔧 Signal Engine configured with the following components:")
        logger.info(f"   - Indicators: {len(indicators_config.get('long', {})) + len(indicators_config.get('short', {}))} registered")
        logger.info(f"   - Strategies: {len(strategies_config)} registered")
        logger.info(f"   - Filters: {len(filters_config) - 2 if 'min_checks' in filters_config and 'min_strength' in filters_config else len(filters_config)} registered")
        logger.info(f"   - Strength Calculators: {len(strength_config)} registered")
        
        # Backtest çalıştır
        logger.info(f"🚀 Starting backtest for {symbol} {interval} with config_id: {config_id}")
        result = engine.run(df, config_id=config_id)
        
        # Sonuçları config_id ile işaretle
        result['config_id'] = config_id
        result['config'] = config

         # ✅ ENHANCED: TP/Commission filter results logging
        if 'filter_statistics' in result:
            stats = result['filter_statistics']
            logger.info(f"🔧 TP/Commission Filter Results:")
            logger.info(f"   - Total signals: {stats.get('total_signals', 0)}")
            logger.info(f"   - Filtered signals: {stats.get('filtered_signals', 0)}")
            logger.info(f"   - Filter efficiency: {stats.get('filter_efficiency_pct', 0):.1f}%")
            logger.info(f"   - Trades opened: {stats.get('trades_opened', 0)}")
        
        # Trade'leri kaydet
        if 'trades' in result and result['trades']:
            trades_df = pd.DataFrame(result['trades'])
            trades_path = os.path.join(output_dir, f"trades_{config_id}.csv")
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"📋 Saved {len(trades_df)} trades to {trades_path}")
        else:
            trades_path = None
            logger.warning("⚠️ No trades were executed during backtest.")
        
        # Equity curve'ü kaydet
        if 'equity_curve' in result and result['equity_curve']:
            equity_df = pd.DataFrame(result['equity_curve'])
            equity_path = os.path.join(output_dir, f"equity_{config_id}.csv")
            equity_df.to_csv(equity_path, index=False)
            logger.info(f"📈 Saved equity curve to {equity_path}")
        else:
            equity_path = None
            logger.warning("⚠️ No equity curve data available.")
        
        # Sonuçları analiz et
        analysis = analyze_single_result(result, output_dir, config_id)
        
        # Grafikleri oluştur
        plots = plot_single_result(result, output_dir, config_id)
        
        # Özet raporu yazdır
        print(f"\n📊 Backtest Results for {symbol} {interval} (Config: {config_id}):")
        print(f"   Total Trades: {result['total_trades']}")
        
        if result['total_trades'] > 0 and 'metrics' in result and 'win_rate' in result['metrics']:
            print(f"   Win Rate: {result['metrics']['win_rate']:.2f}%")
            print(f"   Profit/Loss: ${result['profit_loss']:.2f}")
            print(f"   ROI: {result['roi_pct']:.2f}%")
            print(f"   Max Drawdown: {result['metrics']['max_drawdown_pct']:.2f}%")
            
            # ✅ TP/Commission specific metrics
            if backtest_params.get('enable_tp_commission_filter', False):
                filter_stats = result.get('filter_statistics', {})
                print(f"   Filter Efficiency: {filter_stats.get('filter_efficiency_pct', 0):.1f}%")
                
                if 'trades' in result and result['trades']:
                    trades = result['trades']
                    tp_trades = [t for t in trades if t.get('outcome') == 'TP']
                    if tp_trades:
                        avg_commission_impact = sum(t.get('commission_impact_pct', 0) for t in tp_trades) / len(tp_trades)
                        print(f"   Avg Commission Impact: {avg_commission_impact:.1f}%")
           
            if 'sharpe_ratio' in result['metrics']:
                print(f"   Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
            
            if 'profit_factor' in result['metrics']:
                print(f"   Profit Factor: {result['metrics']['profit_factor']:.2f}")
            
            if 'direction_performance' in result['metrics']:
                print("\n   Direction Performance:")
                for direction, stats in result['metrics']['direction_performance'].items():
                    if 'count' in stats and 'win_rate' in stats and 'avg_gain' in stats:
                        print(f"     {direction}: {stats['count']} trades, Win Rate: {stats['win_rate']:.2f}%, Avg Gain: {stats['avg_gain']:.2f}%")
        else:
            print("   No trades were executed or not enough data to calculate metrics.")
        
        print(f"\n✅ Results saved to {output_dir}")
        
        return {
            "status": "success",
            "result": result,
            "trades_path": trades_path,
            "equity_path": equity_path,
            "analysis": analysis,
            "plots": plots
        }
    
    except Exception as e:
        import traceback
        logger.error(f"❌ Error in backtest: {e}")
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    # Test çalıştırması için
    import sys
    import os
    
    # Ana dizini Python path'ine ekle
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from backtest.utils.config_loader import load_env_config
    
    # Çevre değişkenlerinden konfigürasyonu yükle
    env_config = load_env_config()
    
    # Test çalıştırması
    result = run_single_backtest(
        symbol=env_config.get("symbol", "BTCUSDT"),
        interval=env_config.get("interval", "1h"),
        db_url=env_config.get("db_url", "postgresql://localhost/crypto"),
        output_dir="backtest/results/single",
        backtest_params={
            "initial_balance": float(env_config.get("initial_balance", 10000.0)),
            "risk_per_trade": float(env_config.get("risk_per_trade", 0.01)),
            "sl_multiplier": float(env_config.get("sl_multiplier", 1.5)),
            "tp_multiplier": float(env_config.get("tp_multiplier", 3.0)),
            "leverage": float(env_config.get("leverage", 1.0)),
            "position_direction": env_config.get("position_direction", {"Long": True, "Short": True}),
            "commission_rate": float(env_config.get("commission_rate", 0.001))
        },
        config_id="test"
    )
    
    print(f"Test result status: {result['status']}")