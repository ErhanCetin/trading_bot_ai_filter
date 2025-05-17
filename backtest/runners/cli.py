"""
Komut satırı arayüzü ile backtest çalıştırma modülü
"""
import argparse
import os
import sys
import json
from typing import Dict, List, Any, Optional

# Modül yolunu ayarla
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backtest.utils.config_loader import load_env_config
from backtest.runners.single_backtest import run_single_backtest
from backtest.runners.batch_backtest import run_batch_backtest


def main():
    """Komut satırı arayüzü ile backtest çalıştırma"""
    parser = argparse.ArgumentParser(description="Backtest Runner")
    
    # Ana parametreler
    parser.add_argument("--mode", required=True, choices=["single", "batch"],
                       help="Backtest modu: single (tek) veya batch (toplu)")
    
    # Tek backtest parametreleri
    parser.add_argument("--symbol", help="İşlem sembolü (env: SYMBOL)")
    parser.add_argument("--interval", help="Zaman aralığı (env: INTERVAL)")
    parser.add_argument("--config_id", default="default", help="Konfigürasyon ID'si")
    
    # Batch backtest parametreleri
    parser.add_argument("--config_csv", help="Konfigürasyon CSV dosya yolu")
    parser.add_argument("--max_workers", type=int, help="Maksimum process sayısı")
    
    # Ortak parametreler
    parser.add_argument("--db_url", help="Veritabanı bağlantı URL'si (env: DB_URL)")
    parser.add_argument("--output_dir", help="Çıktı dizini (env: RESULTS_DIR)")
    parser.add_argument("--balance", type=float, help="Başlangıç bakiyesi (env: ACCOUNT_BALANCE)")
    parser.add_argument("--risk", type=float, help="İşlem başına risk oranı (env: RISK_PER_TRADE)")
    parser.add_argument("--sl_multiplier", type=float, help="Stop-loss çarpanı (env: SL_MULTIPLIER)")
    parser.add_argument("--tp_multiplier", type=float, help="Take-profit çarpanı (env: TP_MULTIPLIER)")
    parser.add_argument("--leverage", type=float, help="Kaldıraç oranı (env: LEVERAGE)")
    parser.add_argument("--commission", type=float, help="Komisyon oranı (env: COMMISSION_RATE)")
    parser.add_argument("--indicators", help="İndikatör konfigürasyonu JSON string")
    parser.add_argument("--direction", help="Pozisyon yönü JSON string: {\"Long\": bool, \"Short\": bool}")
    
    args = parser.parse_args()
    
    # Çevre değişkenlerinden konfigürasyon yükle
    env_config = load_env_config()
    
    # Komut satırı argümanlarını işle
    backtest_params = {
        "symbol": args.symbol or env_config.get("symbol"),
        "interval": args.interval or env_config.get("interval"),
        "db_url": args.db_url or env_config.get("db_url"),
        "initial_balance": args.balance or env_config.get("initial_balance"),
        "risk_per_trade": args.risk or env_config.get("risk_per_trade"),
        "sl_multiplier": args.sl_multiplier or env_config.get("sl_multiplier"),
        "tp_multiplier": args.tp_multiplier or env_config.get("tp_multiplier"),
        "leverage": args.leverage or env_config.get("leverage"),
        "commission_rate": args.commission or env_config.get("commission_rate"),
        "position_direction": env_config.get("position_direction"),
    }
    
    # Direction argümanını işle
    if args.direction:
        try:
            backtest_params["position_direction"] = json.loads(args.direction)
        except json.JSONDecodeError:
            print(f"⚠️ Invalid direction format: {args.direction}. Using default.")
    
    # İndikatör argümanını işle
    indicators_config = env_config.get("indicators", {})
    if args.indicators:
        try:
            indicators_config = json.loads(args.indicators)
        except json.JSONDecodeError:
            print(f"⚠️ Invalid indicators format: {args.indicators}. Using default.")
    
    # Çıktı dizinini ayarla
    output_dir = args.output_dir or env_config.get("results_dir", "backtest/results")
    
    # Tek veya toplu backtest çalıştır
    if args.mode == "single":
        # Tek backtest için parametreleri doğrula
        if not backtest_params["symbol"] or not backtest_params["interval"]:
            print("❌ Symbol and interval are required for single backtest.")
            sys.exit(1)
        
        print(f"🚀 Running single backtest for {backtest_params['symbol']} {backtest_params['interval']} (Config ID: {args.config_id})")
        
        # Tek backtest çalıştır
        result = run_single_backtest(
            symbol=backtest_params["symbol"],
            interval=backtest_params["interval"],
            db_url=backtest_params["db_url"],
            output_dir=os.path.join(output_dir, "single"),
            backtest_params=backtest_params,
            indicators_config=indicators_config,
            config_id=args.config_id
        )
        
        print(f"✅ Single backtest completed. Status: {result.get('status')}")
        
        # Özet sonuçları yazdır
        if result.get("status") == "success" and "result" in result:
            metrics = result["result"].get("metrics", {})
            print("\n📊 Summary Results:")
            print(f"   - Total Trades: {result['result']['total_trades']}")
            print(f"   - Win Rate: {metrics.get('win_rate', 0):.2f}%")
            print(f"   - ROI: {result['result']['roi_pct']:.2f}%")
            print(f"   - Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   - Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"\n✅ Results saved to: {os.path.join(output_dir, 'single')}")
        
    elif args.mode == "batch":
        # Toplu backtest için parametreleri doğrula
        config_csv = args.config_csv or os.path.join("backtest", "config", "config_combinations.csv")
        
        if not os.path.exists(config_csv):
            print(f"❌ Config CSV file not found: {config_csv}")
            sys.exit(1)
        
        print(f"🚀 Running batch backtest using config CSV: {config_csv}")
        
        # Toplu backtest çalıştır
        result = run_batch_backtest(
            symbol=backtest_params["symbol"],
            interval=backtest_params["interval"],
            config_csv_path=config_csv,
            db_url=backtest_params["db_url"],
            output_dir=os.path.join(output_dir, "batch"),
            backtest_params=backtest_params,
            max_workers=args.max_workers
        )
        
        print(f"✅ Batch backtest completed. Status: {result.get('status')}")
        print(f"   Results saved to: {result.get('results_path')}")
        
        if result.get("status") == "success":
            print(f"\n🏆 Best ROI: {result.get('best_roi_pct'):.2f}% (Config: {result.get('best_roi_config')})")
            print(f"🏆 Best Win Rate: {result.get('best_winrate_pct'):.2f}% (Config: {result.get('best_winrate_config')})")


if __name__ == "__main__":
    main()

 # # Backtest Runner , how to run.
 # >  python -m backtest.runners.cli --mode single --symbol ETHUSDT --interval 1h --balance 5000 --risk 0.02 --sl_multiplier 2.0 --tp_multiplier 4.0 --leverage 2.0 --direction '{"Long": true, "Short": false}'   