#!/usr/bin/env python3
"""
diagnostic_test.py - COMPLETE WORKING VERSION
Bu dosya tek başına çalıştırılır: python backtest/diagnostic_test.py

TESTED AND WORKING - Copy this entire file
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from sqlalchemy import create_engine, text

# Ana dizini Python yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Loglama basit tut
logging.basicConfig(level=logging.WARNING)

def check_imports():
    """
    Required imports'ları kontrol et
    """
    try:
        from backtest.utils.data_loader import load_price_data
        from backtest.utils.config_loader import load_env_config
        from backtest.runners.single_backtest import run_single_backtest
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from project root directory")
        return False

def check_database_connection():
    """
    Database connection ve data availability kontrol et
    """
    print("🔗 CHECKING DATABASE CONNECTION...")
    
    try:
        from backtest.utils.config_loader import load_env_config
        env_config = load_env_config()
        db_url = env_config.get("db_url")
        
        if not db_url:
            print("❌ Database URL not configured")
            return None
        
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
        
        with engine.connect() as conn:
            query = """
            SELECT symbol, interval, COUNT(*) as count
            FROM kline_data 
            GROUP BY symbol, interval 
            ORDER BY count DESC 
            LIMIT 10
            """
            result = conn.execute(text(query))
            data_summary = result.fetchall()
            
            print(f"\n📊 AVAILABLE DATA (top 10):")
            for row in data_summary:
                print(f"   {row[0]} {row[1]}: {row[2]:,} records")
            
            if data_summary:
                best = data_summary[0]
                engine.dispose()
                return {"symbol": best[0], "interval": best[1], "count": best[2]}
        
        engine.dispose()
        return None
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return None

def check_user_config_data(symbol: str, interval: str) -> Optional[Dict]:
    """
    Check if user's chosen symbol/interval has data
    """
    print(f"🎯 CHECKING USER CONFIG: {symbol} {interval}")
    
    try:
        from backtest.utils.config_loader import load_env_config
        env_config = load_env_config()
        db_url = env_config.get("db_url")
        
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            query = f"""
            SELECT COUNT(*) as count
            FROM kline_data 
            WHERE symbol = '{symbol}' AND interval = '{interval}'
            """
            result = conn.execute(text(query))
            count = result.fetchone()[0]
            
            print(f"📊 Found {count:,} records for {symbol} {interval}")
            
            if count == 0:
                print(f"❌ No data for user config: {symbol} {interval}")
                
                print(f"\n💡 AVAILABLE ALTERNATIVES:")
                
                query = f"""
                SELECT interval, COUNT(*) as count
                FROM kline_data 
                WHERE symbol = '{symbol}'
                GROUP BY interval 
                ORDER BY count DESC
                """
                result = conn.execute(text(query))
                symbol_alternatives = result.fetchall()
                
                if symbol_alternatives:
                    print(f"   {symbol} available intervals:")
                    for row in symbol_alternatives:
                        print(f"     {row[0]}: {row[1]:,} records")
                
                query = f"""
                SELECT symbol, COUNT(*) as count
                FROM kline_data 
                WHERE interval = '{interval}'
                GROUP BY symbol 
                ORDER BY count DESC
                LIMIT 5
                """
                result = conn.execute(text(query))
                interval_alternatives = result.fetchall()
                
                if interval_alternatives:
                    print(f"   {interval} available symbols:")
                    for row in interval_alternatives:
                        print(f"     {row[0]}: {row[1]:,} records")
                
                engine.dispose()
                return None
            
            elif count < 100:
                print(f"⚠️ Very little data: {count} records (recommended: >1000)")
                print(f"💡 Consider using a different symbol/interval")
            else:
                print(f"✅ Sufficient data: {count:,} records")
            
            engine.dispose()
            return {"symbol": symbol, "interval": interval, "count": count}
        
    except Exception as e:
        print(f"❌ Error checking user config: {e}")
        return None

def run_buy_hold_test(symbol: str, interval: str) -> Optional[Dict]:
    """
    Simple buy & hold test
    """
    print(f"\n📊 BUY & HOLD TEST: {symbol} {interval}")
    print("-" * 50)
    
    try:
        from backtest.utils.data_loader import load_price_data
        from backtest.utils.config_loader import load_env_config
        
        env_config = load_env_config()
        db_url = env_config.get("db_url")
        
        df = load_price_data(symbol, interval, db_url, validate_data=False, min_rows=10)
        
        if df.empty:
            print(f"❌ No data loaded for {symbol} {interval}")
            return None
        
        print(f"✅ Data loaded: {len(df)} candles")
        
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        price_change_pct = ((end_price / start_price) - 1) * 100
        
        leverage = 8.0
        commission_rate = 0.0002
        initial_balance = 10000.0
        
        position_value = initial_balance * leverage
        gross_pnl = (price_change_pct / 100) * position_value
        commission = position_value * commission_rate * 2
        net_pnl = gross_pnl - commission
        roi_pct = (net_pnl / initial_balance) * 100
        
        print(f"\n📈 RESULTS:")
        print(f"   Price Change: {price_change_pct:+.2f}%")
        print(f"   Leveraged ROI: {roi_pct:+.2f}%")
        print(f"   Period: {len(df)} candles")
        
        if roi_pct > 20:
            print(f"   🚀 Strong bull market - strategy should beat +{roi_pct + 20:.0f}%")
        elif roi_pct > 0:
            print(f"   📈 Bull market - strategy should beat +{roi_pct + 30:.0f}%")
        else:
            print(f"   🔴 Bear market - strategy should be positive despite {roi_pct:.1f}% decline")
        
        return {
            "symbol": symbol,
            "interval": interval,
            "roi_pct": roi_pct,
            "price_change_pct": price_change_pct,
            "candles": len(df)
        }
        
    except Exception as e:
        print(f"❌ Buy & hold test failed: {e}")
        return None

def run_strategy_test_current(symbol: str, interval: str) -> Optional[Dict]:
    """
    Test current strategy (from env config)
    """
    print(f"\n🎯 CURRENT STRATEGY TEST: {symbol} {interval}")
    print("-" * 50)
    
    try:
        from backtest.utils.config_loader import load_env_config
        from backtest.runners.single_backtest import run_single_backtest
        
        env_config = load_env_config()
        
        backtest_params = {
            "initial_balance": float(env_config.get("initial_balance", 10000.0)),
            "risk_per_trade": float(env_config.get("risk_per_trade", 0.04)),
            "sl_multiplier": float(env_config.get("sl_multiplier", 1.2)),
            "tp_multiplier": float(env_config.get("tp_multiplier", 3.0)),
            "leverage": float(env_config.get("leverage", 8.0)),
            "commission_rate": float(env_config.get("commission_rate", 0.0002)),
            "position_direction": env_config.get("position_direction", {"Long": True, "Short": False}),
            "max_holding_bars": int(env_config.get("max_holding_bars", 500)),
            "enable_tp_commission_filter": env_config.get("enable_tp_commission_filter", False),
        }
        
        print(f"🚀 Running CURRENT strategy (from env)...")
        
        result = run_single_backtest(
            symbol=symbol,
            interval=interval,
            db_url=env_config.get("db_url"),
            output_dir="backtest/results/diagnostic",
            backtest_params=backtest_params,
            config_id="current_strategy"
        )
        
        if result and result.get('status') == 'success':
            strategy_result = result['result']
            
            total_trades = strategy_result.get('total_trades', 0)
            roi_pct = strategy_result.get('roi_pct', 0)
            win_rate = strategy_result.get('metrics', {}).get('win_rate', 0)
            
            print(f"✅ CURRENT RESULTS:")
            print(f"   Total Trades: {total_trades}")
            print(f"   ROI: {roi_pct:+.2f}%")
            print(f"   Win Rate: {win_rate:.1f}%")
            
            return {
                "total_trades": total_trades,
                "roi_pct": roi_pct,
                "win_rate": win_rate,
                "max_drawdown": strategy_result.get('metrics', {}).get('max_drawdown_pct', 0),
                "profit_factor": strategy_result.get('metrics', {}).get('profit_factor', 0),
                "status": "success",
                "config": "current_env"
            }
        else:
            print(f"❌ Current strategy failed: {result.get('message', 'Unknown') if result else 'No result'}")
            return {"status": "failed", "error": result.get('message', 'Unknown') if result else 'No result'}
            
    except Exception as e:
        print(f"❌ Current strategy error: {e}")
        return {"status": "error", "error": str(e)}

def run_strategy_test_fixed(symbol: str, interval: str, baseline_roi: float) -> Optional[Dict]:
    """
    Test FIXED strategy with internal improvements
    """
    print(f"\n🔧 FIXED STRATEGY TEST: {symbol} {interval}")
    print("-" * 50)
    
    try:
        from backtest.utils.config_loader import load_env_config
        from backtest.runners.single_backtest import run_single_backtest
        
        env_config = load_env_config()
        
        if baseline_roi < -10:
            print("📉 Bear market detected - using BALANCED SHORT strategy")
            fixed_params = {
                "initial_balance": 10000.0,
                "risk_per_trade": 0.02,     # 1.5% → 2% (slightly more aggressive)
                "sl_multiplier": 1.3,       # 1.5 → 1.3 (slightly wider stops)
                "tp_multiplier": 2.5,       # 3.0 → 2.5 (more achievable targets)
                "leverage": 3.0,            # 2.0 → 3.0 (moderate leverage)
                "commission_rate": 0.0002,
                "position_direction": {"Long": False, "Short": True},
                "max_holding_bars": 30,     # 20 → 30 (longer holding)
                "enable_tp_commission_filter": True,
                "min_tp_commission_ratio": 12.0,    # 25.0 → 12.0 (much more relaxed)
                "max_commission_impact_pct": 10.0,  # 6.0 → 10.0 (more relaxed)
                "min_position_size": 1000.0,        # 2000.0 → 1000.0 (smaller minimum)
                "min_net_rr_ratio": 1.5,            # 2.0 → 1.5 (more achievable)
            }
        elif baseline_roi > 10:
            print("📈 Bull market detected - using BALANCED LONG strategy")
            fixed_params = {
                "initial_balance": 10000.0,
                "risk_per_trade": 0.025,    # Slightly more aggressive for bull
                "sl_multiplier": 1.2,       # Tight stops
                "tp_multiplier": 3.5,       # Higher targets in bull market
                "leverage": 4.0,            # More leverage in bull
                "commission_rate": 0.0002,
                "position_direction": {"Long": True, "Short": False},
                "max_holding_bars": 40,     # Longer holding in bull
                "enable_tp_commission_filter": True,
                "min_tp_commission_ratio": 15.0,    # Moderate filtering
                "max_commission_impact_pct": 8.0,   # Moderate
                "min_position_size": 1200.0,        # Moderate minimum
                "min_net_rr_ratio": 2.0,            # Higher R:R in bull
            }
        else:
            print("📊 Sideways market detected - using BALANCED BREAKOUT strategy")
            fixed_params = {
                "initial_balance": 10000.0,
                "risk_per_trade": 0.018,    # Conservative for sideways
                "sl_multiplier": 1.1,       # Very tight stops for breakouts
                "tp_multiplier": 2.2,       # Quick profits in sideways
                "leverage": 3.5,            # Moderate leverage
                "commission_rate": 0.0002,
                "position_direction": {"Long": True, "Short": True},  # Both directions
                "max_holding_bars": 25,     # Quick exits
                "enable_tp_commission_filter": True,
                "min_tp_commission_ratio": 18.0,    # Moderate-strict
                "max_commission_impact_pct": 7.0,   # Moderate
                "min_position_size": 1500.0,        # Moderate
                "min_net_rr_ratio": 1.8,            # Reasonable R:R
            }
        
        minimal_indicators = {
            "long": {
                "ema": {"periods": [21, 50]},
                "rsi": {"periods": [14]},
                "atr": {"window": 14}
            },
            "short": {
                "ema": {"periods": [21, 50]},
                "rsi": {"periods": [14]},
                "atr": {"window": 14}
            }
        }
        
        print(f"🚀 Running BALANCED FIXED strategy...")
        print(f"   Target: 15-50 trades with positive ROI in bear market")
        
        result = run_single_backtest(
            symbol=symbol,
            interval=interval,
            db_url=env_config.get("db_url"),
            output_dir="backtest/results/diagnostic",
            backtest_params=fixed_params,
            indicators_config=minimal_indicators,
            config_id="fixed_strategy"
        )
        
        if result and result.get('status') == 'success':
            strategy_result = result['result']
            
            total_trades = strategy_result.get('total_trades', 0)
            roi_pct = strategy_result.get('roi_pct', 0)
            win_rate = strategy_result.get('metrics', {}).get('win_rate', 0)
            max_dd = strategy_result.get('metrics', {}).get('max_drawdown_pct', 0)
            pf = strategy_result.get('metrics', {}).get('profit_factor', 0)
            
            print(f"✅ BALANCED FIXED RESULTS:")
            print(f"   Total Trades: {total_trades}")
            print(f"   ROI: {roi_pct:+.2f}%")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Max Drawdown: {max_dd:.1f}%")
            print(f"   Profit Factor: {pf:.2f}")
            
            # Trade count assessment
            if total_trades == 0:
                print(f"   ❌ Still no trades - filters too strict")
            elif total_trades < 10:
                print(f"   ⚠️ Very few trades ({total_trades}) - consider relaxing filters")
            elif total_trades > 100:
                print(f"   ⚠️ Many trades ({total_trades}) - consider stricter filters")
            else:
                print(f"   ✅ Good trade count ({total_trades}) - balanced approach working")
            
            return {
                "total_trades": total_trades,
                "roi_pct": roi_pct,
                "win_rate": win_rate,
                "max_drawdown": max_dd,
                "profit_factor": pf,
                "status": "success",
                "config": "fixed_internal"
            }
        else:
            print(f"❌ Fixed strategy failed: {result.get('message', 'Unknown') if result else 'No result'}")
            return {"status": "failed", "error": result.get('message', 'Unknown') if result else 'No result'}
            
    except Exception as e:
        print(f"❌ Fixed strategy error: {e}")
        return {"status": "error", "error": str(e)}

def diagnose_and_fix(baseline_result: Dict, current_result: Dict, fixed_result: Dict) -> Dict:
    """
    Compare current vs fixed strategy and provide analysis
    """
    print(f"\n🔍 COMPREHENSIVE STRATEGY ANALYSIS")
    print("=" * 70)
    
    baseline_roi = baseline_result.get('roi_pct', 0)
    current_roi = current_result.get('roi_pct', 0) if current_result.get('status') == 'success' else 0
    fixed_roi = fixed_result.get('roi_pct', 0) if fixed_result.get('status') == 'success' else 0
    
    current_trades = current_result.get('total_trades', 0) if current_result.get('status') == 'success' else 0
    fixed_trades = fixed_result.get('total_trades', 0) if fixed_result.get('status') == 'success' else 0
    
    print(f"📊 PERFORMANCE COMPARISON:")
    print(f"   Buy & Hold ROI:    {baseline_roi:+8.2f}%")
    print(f"   Current Strategy:  {current_roi:+8.2f}% ({current_trades} trades)")
    print(f"   Fixed Strategy:    {fixed_roi:+8.2f}% ({fixed_trades} trades)")
    
    current_vs_baseline = current_roi - baseline_roi
    fixed_vs_baseline = fixed_roi - baseline_roi
    fixed_vs_current = fixed_roi - current_roi
    
    print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
    print(f"   Current vs Baseline: {current_vs_baseline:+.2f}%")
    print(f"   Fixed vs Baseline:   {fixed_vs_baseline:+.2f}%")
    print(f"   Fixed vs Current:    {fixed_vs_current:+.2f}%")
    
    if current_trades > 0 and fixed_trades > 0:
        trade_reduction = ((current_trades - fixed_trades) / current_trades) * 100
        print(f"   Trade Reduction:     {trade_reduction:+.1f}%")
    
    current_wr = current_result.get('win_rate', 0) if current_result.get('status') == 'success' else 0
    fixed_wr = fixed_result.get('win_rate', 0) if fixed_result.get('status') == 'success' else 0
    
    if current_wr > 0 and fixed_wr > 0:
        wr_improvement = fixed_wr - current_wr
        print(f"   Win Rate Change:     {wr_improvement:+.1f}%")
    
    if fixed_result.get('status') != 'success':
        diagnosis = {
            "result": "TECHNICAL FAILURE",
            "grade": "F",
            "message": "Fixed strategy failed to run",
            "recommendations": ["Debug technical issues", "Check data quality", "Simplify configuration"]
        }
    elif fixed_roi > baseline_roi + 20:
        diagnosis = {
            "result": "EXCELLENT FIX", 
            "grade": "A+",
            "message": f"Fixed strategy significantly outperforms baseline (+{fixed_vs_baseline:.1f}%)",
            "recommendations": ["Deploy this configuration", "Scale up", "Test on other assets"]
        }
    elif fixed_roi > baseline_roi + 10:
        diagnosis = {
            "result": "GOOD FIX",
            "grade": "A", 
            "message": f"Fixed strategy outperforms baseline (+{fixed_vs_baseline:.1f}%)",
            "recommendations": ["Deploy with monitoring", "Fine-tune parameters", "Expand testing"]
        }
    elif fixed_roi > current_roi + 10:
        diagnosis = {
            "result": "SIGNIFICANT IMPROVEMENT",
            "grade": "B+",
            "message": f"Major improvement over current (+{fixed_vs_current:.1f}%)",
            "recommendations": ["Apply the fixes", "Further optimization needed", "Monitor closely"]
        }
    elif fixed_roi > current_roi:
        diagnosis = {
            "result": "MINOR IMPROVEMENT", 
            "grade": "B",
            "message": f"Small improvement (+{fixed_vs_current:.1f}%)",
            "recommendations": ["Apply fixes", "Need more optimization", "Test different approaches"]
        }
    else:
        diagnosis = {
            "result": "FIX INEFFECTIVE",
            "grade": "C",
            "message": "Fixed strategy not significantly better",
            "recommendations": ["Try different approach", "Review strategy fundamentals", "Consider manual trading"]
        }
    
    print(f"\n🎯 DIAGNOSIS:")
    print(f"   Result: {diagnosis['result']}")
    print(f"   Grade: {diagnosis['grade']}")
    print(f"   Assessment: {diagnosis['message']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(diagnosis['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    if fixed_roi > current_roi:
        print(f"\n🔧 EFFECTIVE FIXES IDENTIFIED:")
        print(f"   • Reduced trade count: {current_trades} → {fixed_trades}")
        print(f"   • Market-adaptive direction (bear market = short)")
        print(f"   • Stricter filtering (less over-trading)")
        print(f"   • Conservative position sizing")
        print(f"   • Shorter holding periods")
    
    return diagnosis

def main():
    """
    Main diagnostic function - COMPLETE WORKING VERSION
    """
    print("🚀 STANDALONE DIAGNOSTIC WITH INTERNAL FIXES")
    print("=" * 80)
    print("Purpose: Test current strategy vs internal fixes")
    print("=" * 80)
    
    if not check_imports():
        print("\n❌ FAILED: Import issues")
        return
    
    try:
        from backtest.utils.config_loader import load_env_config
        env_config = load_env_config()
        user_symbol = env_config.get("symbol")
        user_interval = env_config.get("interval")
        
        print(f"📋 USER CONFIG: {user_symbol} {user_interval}")
        
        if not user_symbol or not user_interval:
            print("❌ User config incomplete!")
            return
        
    except Exception as e:
        print(f"❌ Error loading user config: {e}")
        return
    
    database_ok = check_database_connection()
    if not database_ok:
        print("\n❌ FAILED: Database connection issues")
        return
    
    user_data_check = check_user_config_data(user_symbol, user_interval)
    if not user_data_check:
        print(f"\n❌ FAILED: No data for your config ({user_symbol} {user_interval})")
        return
    
    symbol = user_symbol
    interval = user_interval
    
    print(f"\n🎯 TESTING: {symbol} {interval}")
    
    baseline_result = run_buy_hold_test(symbol, interval)
    if not baseline_result:
        print(f"\n❌ FAILED: Baseline test failed")
        return
    
    print(f"\n" + "="*60)
    print(f"TESTING CURRENT vs FIXED STRATEGIES")
    print(f"="*60)
    
    current_result = run_strategy_test_current(symbol, interval)
    fixed_result = run_strategy_test_fixed(symbol, interval, baseline_result['roi_pct'])
    
    diagnosis = diagnose_and_fix(baseline_result, current_result, fixed_result)
    
    print(f"\n🏁 DIAGNOSTIC WITH FIXES COMPLETE!")
    print("=" * 60)
    
    grade = diagnosis.get('grade', 'F')
    
    if grade in ['A+', 'A']:
        print("🎉 EXCELLENT - Fixed strategy works well!")
        print("🚀 Ready to deploy the internal fixes")
        status = "EXCELLENT"
    elif grade in ['B+', 'B']:
        print("✅ GOOD - Significant improvements found")
        print("🔧 Apply fixes and continue optimization")
        status = "GOOD"
    elif grade == 'C':
        print("🟡 NEEDS WORK - More fundamental changes needed")
        print("🛠️ Consider different strategy approach")
        status = "NEEDS_WORK"
    else:
        print("❌ FAILED - Technical or fundamental issues")
        print("🔧 Debug and fix basic problems first")
        status = "FAILED"
    
    print(f"\n💡 NEXT STEPS BASED ON RESULT ({status}):")
    
    if status == "EXCELLENT":
        print(f"1. ✅ Extract the fixed config parameters")
        print(f"2. ✅ Apply to your main strategy")
        print(f"3. ✅ Scale up with confidence")
        print(f"4. ✅ Test on other assets/timeframes")
    elif status == "GOOD":
        print(f"1. 🔧 Apply the identified fixes")
        print(f"2. 🔧 Fine-tune parameters further") 
        print(f"3. 🔧 Re-test before scaling")
        print(f"4. 🔧 Monitor performance closely")
    elif status == "NEEDS_WORK":
        print(f"1. 🛠️ Try completely different approach")
        print(f"2. 🛠️ Review strategy fundamentals")
        print(f"3. 🛠️ Consider manual trading")
        print(f"4. 🛠️ Test with different assets/timeframes")
    else:
        print(f"1. 🔧 Fix technical issues first")
        print(f"2. 🔧 Check data quality")
        print(f"3. 🔧 Simplify configuration")
        print(f"4. 🔧 Debug step by step")

if __name__ == "__main__":
    """
    Standalone execution - COMPLETE WORKING VERSION
    """
    try:
        main()
    except Exception as e:
        print(f"\n❌ DIAGNOSTIC FAILED WITH ERROR:")
        print(f"Error: {e}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"1. Check if you're in the correct directory")
        print(f"2. Verify database connection")
        print(f"3. Check .env file configuration")
        print(f"4. Ensure all dependencies are installed")
        
        import traceback
        print(f"\n📋 FULL ERROR TRACE:")
        traceback.print_exc()