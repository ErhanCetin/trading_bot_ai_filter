"""
Tek backtest çalıştırma modülü
"""
import os
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union

from ..core.backtest_engine import BacktestEngine
from ..utils.data_loader import load_price_data, parse_indicators_config
from ..analysis.result_analyzer import analyze_single_result
from ..analysis.result_plotter import plot_single_result


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
        indicators_config: İndikatör konfigürasyonu
        config_id: Konfigürasyon ID'si
        
    Returns:
        Backtest sonuçları
    """
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Veriyi yükle
        df = load_price_data(symbol, interval, db_url)
        print(f"📈 Loaded {len(df)} data points for {symbol} {interval}")
        
        # İndikatör konfigürasyonunu kontrol et
        if indicators_config is None:
            indicators_config = {}
            print("⚠️ No indicators configuration provided. Using defaults.")
        
        # print("⚠️ Indicators configuration:")
        # print(f"🔧 Indicators configuration: {json.dumps(indicators_config, indent=2)}")

        # Backtest motorunu oluştur
        engine = BacktestEngine(
            symbol=symbol,
            interval=interval,
            initial_balance=backtest_params.get('initial_balance', 10000.0),
            risk_per_trade=backtest_params.get('risk_per_trade', 0.01),
            sl_multiplier=backtest_params.get('sl_multiplier', 1.5),
            tp_multiplier=backtest_params.get('tp_multiplier', 3.0),
            leverage=backtest_params.get('leverage', 1.0),
            position_direction=backtest_params.get('position_direction', {"Long": True, "Short": True}),
            commission_rate=backtest_params.get('commission_rate', 0.001)
        )
        
        # Standart konfigürasyon oluştur
        config = {
            "indicators": indicators_config,
            "strategies": {
                "trend_following": {},
                "oscillator_signals": {},
                "volatility_breakout": {}
            },
            "strength": {
                "trend_indicators": {},
                "oscillator_levels": {},
                "volatility_measures": {}
            },
            "filters": {
                "rsi_threshold": {},
                "macd_confirmation": {},
                "atr_volatility": {},
                "min_checks": 2,
                "min_strength": 3
            }
        }
        
        # Signal Engine bileşenlerini yapılandır
        engine.configure_signal_engine(
            indicators_config=config.get('indicators', {}),
            strategies_config=config.get('strategies', {}),
            strength_config=config.get('strength', {}),
            filter_config=config.get('filters', {})
        )
        
        # Backtest çalıştır
        result = engine.run(df, config_id=config_id)
        
        # Sonuçları config_id ile işaretle
        result['config_id'] = config_id
        result['config'] = config
        
        # Trade'leri kaydet
        trades_df = pd.DataFrame(result['trades'])
        trades_path = os.path.join(output_dir, f"trades_{config_id}.csv")
        trades_df.to_csv(trades_path, index=False)
        
        # Equity curve'ü kaydet
        equity_df = pd.DataFrame(result['equity_curve'])
        equity_path = os.path.join(output_dir, f"equity_{config_id}.csv")
        equity_df.to_csv(equity_path, index=False)
        
        # Sonuçları analiz et
        analysis = analyze_single_result(result, output_dir, config_id)
        
        # Grafikleri oluştur
        plots = plot_single_result(result, output_dir, config_id)
        
        # Özet raporu yazdır
        print(f"\n📊 Backtest Results for {symbol} {interval} (Config: {config_id}):")
        print(f"   Total Trades: {result['total_trades']}")
        if result['total_trades'] > 0 and 'win_rate' in result['metrics']:
            print(f"   Win Rate: {result['metrics']['win_rate']:.2f}%")
            print(f"   Profit/Loss: ${result['profit_loss']:.2f}")
            print(f"   ROI: {result['roi_pct']:.2f}%")
            print(f"   Max Drawdown: {result['metrics']['max_drawdown_pct']:.2f}%")
            print(f"   Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
            print(f"   Profit Factor: {result['metrics']['profit_factor']:.2f}")
            
            if result['metrics'].get('direction_performance'):
                print("\n   Direction Performance:")
                for direction, stats in result['metrics']['direction_performance'].items():
                    print(f"     {direction}: {stats['count']} trades, Win Rate: {stats['win_rate']:.2f}%, Avg Gain: {stats['avg_gain']:.2f}%")
        else:
            print("   No trades were executed or not enough data to calculate metrics."  )      
        
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
        print(f"❌ Error in backtest: {e}")
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }