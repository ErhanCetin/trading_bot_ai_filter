"""
Temel backtest motoru.
Signal Engine ile entegre çalışır ve ticaret stratejilerini değerlendirir.

Fixed version with proper parametric strategy support and ensemble logic.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os
import logging

# Signal Engine importları
from signal_engine.indicators import registry as indicator_registry
from signal_engine.signal_indicator_plugin_system import IndicatorManager
from signal_engine.strategies import registry as strategy_registry
from signal_engine.signal_strategy_system import StrategyManager  # Updated import
from signal_engine.strength import registry as strength_registry
from signal_engine.signal_strength_system import StrengthManager
from signal_engine.filters import registry as filter_registry
from signal_engine.signal_filter_system import FilterManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Signal Engine ile entegre çalışan backtest motoru with enhanced strategy support."""
    
    def __init__(self, 
                 symbol: str, 
                 interval: str, 
                 initial_balance: float = 10000.0,
                 risk_per_trade: float = 0.01,
                 sl_multiplier: float = 1.5,
                 tp_multiplier: float = 3.0,
                 leverage: float = 1.0,
                 position_direction: Dict[str, bool] = None,
                 commission_rate: float = 0.001):
        """
        Backtest motorunu başlatır
        
        Args:
            symbol: İşlem sembolü
            interval: Zaman aralığı
            initial_balance: Başlangıç bakiyesi
            risk_per_trade: İşlem başına risk oranı (bakiyenin yüzdesi)
            sl_multiplier: Stop-loss ATR çarpanı
            tp_multiplier: Take-profit ATR çarpanı
            leverage: Kaldıraç oranı
            position_direction: İşlem yönü ayarları {"Long": bool, "Short": bool}
            commission_rate: Komisyon oranı (işlem başına)
        """
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.leverage = leverage
        self.position_direction = position_direction or {"Long": True, "Short": True}
        self.commission_rate = commission_rate
        
        # Signal Engine bileşenleri - UPDATED
        self.indicator_manager = IndicatorManager(indicator_registry)
        self.strategy_manager = StrategyManager(strategy_registry)  # Enhanced version
        self.strength_manager = StrengthManager(strength_registry)
        self.filter_manager = FilterManager(filter_registry)
        
        # Strategy configuration storage
        self.strategy_config = {}
        
        # Sonuçları saklayacak konteynerler
        self.trades = []
        self.equity_curve = []
        self.metrics = {}
    
    def configure_signal_engine(self, 
                               indicators_config: Dict[str, Any] = None,
                               strategies_config: Dict[str, Any] = None,
                               strength_config: Dict[str, Any] = None,
                               filter_config: Dict[str, Any] = None) -> None:
        """
        Signal Engine bileşenlerini yapılandırır - ENHANCED VERSION
        
        Args:
            indicators_config: İndikatör yapılandırması
            strategies_config: Strateji yapılandırması (enhanced format)
            strength_config: Sinyal gücü yapılandırması
            filter_config: Filtre yapılandırması
        """
        # İndikatörleri yapılandır
        if indicators_config:
            # İndikatörlerin yöne göre konfigürasyonu
            long_indicators = indicators_config.get('long', {})
            short_indicators = indicators_config.get('short', {})
            
            # Tüm indikatörleri ekle
            all_indicators = {**long_indicators, **short_indicators}
            for indicator_name, params in all_indicators.items():
                self.indicator_manager.add_indicator(indicator_name, params)
        
        # Stratejileri yapılandır - ENHANCED LOGIC
        if strategies_config:
            self.strategy_config = strategies_config
            self._configure_strategies(strategies_config)
        
        # Sinyal gücü hesaplayıcılarını yapılandır
        if strength_config:
            for calculator_name, params in strength_config.items():
                self.strength_manager.add_calculator(calculator_name, params)
        
        # Filtreleri yapılandır
         # 🔧 FIXED: Filtreleri yapılandır - MATCHES YOUR CONFIG
        if filter_config:
            logger.info(f"🔧 Configuring filters from config with {len(filter_config)} items...")
            
            # Configuration parameters to skip
            config_params = {'min_checks', 'min_strength'}
            
            # 1. Add all filter rules (except config params)
            rules_added = 0
            for rule_name, params in filter_config.items():
                if rule_name in config_params:
                    continue  # Skip config parameters
                    
                try:
                    # Add the filter rule with its parameters
                    self.filter_manager.add_rule(rule_name, params)
                    logger.info(f"✅ Added filter rule: {rule_name}")
                    rules_added += 1
                except Exception as e:
                    logger.error(f"❌ Failed to add filter rule {rule_name}: {e}")
            
            logger.info(f"✅ Successfully added {rules_added} filter rules")
            
            # 2. Set configuration parameters
            if "min_checks" in filter_config:
                min_checks = filter_config["min_checks"]
                self.filter_manager.set_min_checks_required(min_checks)
                logger.info(f"🎯 Set min_checks_required: {min_checks}")
                
            if "min_strength" in filter_config:
                min_strength = filter_config["min_strength"]
                self.filter_manager.set_min_strength_required(min_strength)
                logger.info(f"💪 Set min_strength_required: {min_strength}")
                
            # 3. Log final configuration
            total_rules = getattr(self.filter_manager, '_rules_to_apply', [])
            logger.info(f"📋 Final filter rules: {total_rules}")
            logger.info(f"⚙️ Min checks: {getattr(self.filter_manager, '_min_checks_required', 0)}")
            logger.info(f"⚙️ Min strength: {getattr(self.filter_manager, '_min_strength_required', 0)}")
            
        else:
            # Config yoksa varsayılan filtreleri ekle
            logger.info("⚠️ No filter config provided, adding default filters...")
            self._add_default_filters()

    def _add_default_filters(self) -> None:
        """
        🆕 YENİ: Varsayılan filtreleri ekler - eğer config boşsa
        """
        # Temel ve güvenli filtreler
        default_filters = [
            'market_regime',        # Piyasa rejimi kontrolü
            'trend_strength',       # Trend gücü kontrolü  
            'volatility_regime'     # Volatilite kontrolü
        ]
        
        for filter_name in default_filters:
            try:
                self.filter_manager.add_rule(filter_name)
                logger.info(f"✅ Added default filter: {filter_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not add default filter {filter_name}: {e}")
        
        # Varsayılan eşikler
        self.filter_manager.set_min_checks_required(2)    # 3'ten 2'si geçsin
        self.filter_manager.set_min_strength_required(45) # Orta seviye güç
        logger.info("✅ Default filters added with min_checks=2 and min_strength=45")


    def _configure_strategies(self, strategies_config: Dict[str, Any]) -> None:
        """
        Strategy configuration - simplified without complex ensemble logic
        
        Supports multiple configuration formats:
        1. Simple format: {"strategy_name": {"param1": value1}}
        2. Enhanced format: {"strategy_name": {"params": {...}, "weight": 1.0}}
        """
        logger.info("🔧 Configuring strategies...")
        
        # Clear existing strategies
        #self.strategy_manager.clear_strategies()
        
        # Configure individual strategies (including ensemble strategies from registry)
        configured_count = 0
        for strategy_name, strategy_config in strategies_config.items():
            try:
                # Parse strategy configuration
                params, weight = self._parse_strategy_config(strategy_config)
                
                # Add strategy to manager
                self.strategy_manager.add_strategy(
                    strategy_name=strategy_name,
                    params=params,
                    weight=weight
                )
                
                configured_count += 1
                logger.debug(f"✅ Added strategy: {strategy_name} (weight: {weight})")
                
            except Exception as e:
                logger.error(f"❌ Failed to configure strategy {strategy_name}: {e}")
                continue
        
        logger.info(f"✅ Successfully configured {configured_count} strategies")
        
        # Log strategy details
        available_strategies = self.strategy_manager.list_available_strategies()
        logger.info(f"📋 Available strategy categories: {list(available_strategies.keys())}")
    
    def _parse_strategy_config(self, strategy_config: Any) -> Tuple[Dict[str, Any], float]:
        """
        Parse strategy configuration from various formats
        
        Args:
            strategy_config: Strategy configuration (various formats)
            
        Returns:
            Tuple of (params_dict, weight)
        """
        # Handle different configuration formats
        if isinstance(strategy_config, dict):
            if "params" in strategy_config or "weight" in strategy_config:
                # Enhanced format: {"params": {...}, "weight": 1.0}
                params = strategy_config.get("params", {})
                weight = strategy_config.get("weight", 1.0)
            else:
                # Simple format: {"param1": value1, "param2": value2}
                params = strategy_config
                weight = 1.0
        elif isinstance(strategy_config, (int, float)):
            # Weight only: strategy_name: 1.5
            params = {}
            weight = float(strategy_config)
        else:
            # Default case
            params = {}
            weight = 1.0
        
        return params, weight
    
    def run(self, df: pd.DataFrame, config_id: str = None) -> Dict[str, Any]:
        """
        Backtest çalıştırır - ENHANCED VERSION
        
        Args:
            df: Fiyat verisi DataFrame
            config_id: Konfigürasyon ID'si
            
        Returns:
            Backtest sonuç özeti
        """
        logger.info(f"🚀 Starting backtest for {self.symbol} {self.interval}")
        
        # Başlangıç bakiyesi ile backtest için hazırla
        balance = self.initial_balance
        trades = []
        equity_curve = [{"time": df.iloc[0]["open_time"], "balance": balance}]
        
        # Signal Engine sürecini çalıştır
        try:
            # 1. İndikatörleri hesapla
            logger.info("📊 Calculating indicators...")
            df = self.indicator_manager.calculate_indicators(df)
            print(f"🚀 🚀 🚀 İndikatörler hesaplandı: {df.columns.tolist()}")
           
            logger.info(f"✅ Indicators calculated: {len([col for col in df.columns if not col in ['open_time', 'open', 'high', 'low', 'close', 'volume']])} indicators")


            # Debug indicators (commented out by default, uncomment when needed)
            #from backtest.utils.print_calculated_indicator_data.print_calculated_indicator_list import debug_indicators
            #debug_indicators(df, output_type="csv", output_file="calculated_indicators.csv")
            
            # 2. Sinyalleri oluştur - SIMPLIFIED CALL
            logger.info("🎯 Generating strategy signals...")
            df = self.strategy_manager.generate_signals(df)
            print(f"🚀 🚀 🚀 Sinyaller oluşturuldu: {df.columns.tolist()}")
            logger.info("✅ Strategy signals generated")
            
            # Debug signals (commented out by default, uncomment when needed)
            # # Sinyaller için debug modülünü içe aktar
            # from backtest.utils.print_calculated_strategies_data.print_calculated_strategies_list import debug_signals
            # # Sinyalleri debug et
            # debug_signals(df, output_type="csv", output_file="calculated_signals.csv")
            
            # Debug: Signal generation statistics
            long_signals = df["long_signal"].sum() if "long_signal" in df.columns else 0
            short_signals = df["short_signal"].sum() if "short_signal" in df.columns else 0
            logger.info(f"📈 Signal statistics: {long_signals} long, {short_signals} short")
            
            # Yön filtrelemesi uygula
            if not self.position_direction.get("Long", True):
                df["long_signal"] = False
                logger.info("🚫 Long signals disabled by position direction filter")
            if not self.position_direction.get("Short", True):
                df["short_signal"] = False
                logger.info("🚫 Short signals disabled by position direction filter")
            
            # 3. Sinyal gücünü hesapla
            logger.info("💪 Calculating signal strength...")
            # Hesaplayıcı adlarını yöneticiden al
            calculator_names = getattr(self.strength_manager, '_calculators_to_use', [])
            print(f"🚀 🚀 🚀 Sinyal gücü hesaplayıcıları- calculator name: {calculator_names}")
            # Hesaplayıcı parametrelerini al
            calculator_params = getattr(self.strength_manager, '_calculator_params', {})
            print(f"🚀 🚀 🚀 Sinyal gücü hesaplayıcıları- calculator params: {calculator_params}")
            
            if calculator_names:
                # Sinyal gücünü hesapla
                strength_series = self.strength_manager.calculate_strength(
                    df, 
                    df,  # signals_df olarak aynı df'i kullanıyoruz, çünkü long_signal ve short_signal sütunları burada 
                    calculator_names,
                    calculator_params
                )
                # Sonuçları DataFrame'e ekle
                df['signal_strength'] = strength_series
                logger.info(f"✅ Signal strength calculated, avg strength: {strength_series.mean():.2f}")
                
                # Debug strength values (commented out by default, uncomment when needed)
                # from backtest.utils.print_calculated_strength_data.print_calculated_strength_list import debug_strength_values
                # # Sinyalleri debug et
                # debug_strength_values(df, output_type="csv", output_file="calculated_strengths.csv")
            else:
                df['signal_strength'] = 1.0  # Default strength
                logger.info("⚠️ No strength calculators configured, using default strength")
                    
            # 4. Sinyalleri filtrele
            # 4. 🔧 FIXED: Sinyalleri filtrele - YOUR CONFIG READY
            logger.info("🔍 Applying filters...")
            
            # Filter configuration debug
            rules_to_apply = getattr(self.filter_manager, '_rules_to_apply', [])
            min_checks = getattr(self.filter_manager, '_min_checks_required', 0)
            min_strength = getattr(self.filter_manager, '_min_strength_required', 0)
            
            logger.info(f"📋 Filter rules configured: {len(rules_to_apply)} rules")
            logger.info(f"🎯 Rules: {rules_to_apply}")
            logger.info(f"⚙️ Min checks required: {min_checks}")
            logger.info(f"💪 Min strength required: {min_strength}")
            
            if not rules_to_apply:
                logger.error("❌ NO FILTER RULES CONFIGURED! This should not happen with your config.")
                logger.info("🔧 Adding emergency default filters...")
                self._add_default_filters()
            
            # Pre-filter signal count
            pre_long = df.get("long_signal", pd.Series(False)).sum()
            pre_short = df.get("short_signal", pd.Series(False)).sum()
            logger.info(f"📊 Pre-filter signals: {pre_long} long, {pre_short} short")
            
            # Apply filters - PROPER CALL
            try:
                df = self.filter_manager.filter_signals(df)
                logger.info("✅ Filters applied successfully")
            except Exception as e:
                logger.error(f"❌ Error applying filters: {e}")
                import traceback
                traceback.print_exc()
                # Continue without filtering
            
            # Post-filter analysis
            if "signal_passed_filter" in df.columns:
                passed_signals = df["signal_passed_filter"].sum()
                total_signals = pre_long + pre_short
                pass_rate = (passed_signals / total_signals * 100) if total_signals > 0 else 0
                logger.info(f"🎯 Filter results: {passed_signals}/{total_signals} signals passed ({pass_rate:.1f}%)")
            else:
                logger.warning("⚠️ No signal_passed_filter column found after filtering")
            
            # Final signal statistics
            final_long = df["long_signal"].sum() if "long_signal" in df.columns else 0
            final_short = df["short_signal"].sum() if "short_signal" in df.columns else 0
            logger.info(f"🎯 Final signals after filtering: {final_long} long, {final_short} short")
            
        except Exception as e:
            logger.error(f"❌ Error in signal generation: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(str(e))
        
        # İşlemleri simüle et
        logger.info("💼 Simulating trades...")
        trades_executed = 0
        
        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # Sinyal gücü kontrolü - ORJINAL YORUMLA
            if row.get("signal_strength", 0) < 1:  # 3 → 1 (daha çok trade için)
                continue
            
            # Yön belirleme
            direction = None
            if row.get("long_signal", False):
                direction = "LONG"
            elif row.get("short_signal", False):
                direction = "SHORT"
            
            if direction is None:
                continue
            
            # İşlem detaylarını hesapla
            trade_result = self._simulate_trade(row, next_row, direction, balance, df, config_id)
            
            if trade_result:
                balance = trade_result["new_balance"]
                trades.append(trade_result["trade"])
                equity_curve.append({"time": next_row["open_time"], "balance": balance})
                trades_executed += 1
        
        logger.info(f"✅ Trade simulation completed: {trades_executed} trades executed")
        
        # Sonuçları sakla
        self.trades = trades
        self.equity_curve = equity_curve
        
        # Performans metriklerini hesapla
        self._calculate_performance_metrics()
        
        # Sonuç özeti
        result = {
            "total_trades": len(trades),
            "final_balance": balance,
            "profit_loss": balance - self.initial_balance,
            "roi_pct": (balance / self.initial_balance - 1) * 100,
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": self.metrics,
            "config": {
                "strategies": self.strategy_config,
                "position_direction": self.position_direction,
                "risk_per_trade": self.risk_per_trade
            }
        }
        
        logger.info(f"🏁 Backtest completed: {len(trades)} trades, ROI: {result['roi_pct']:.2f}%")
        return result

    def _simulate_trade(self, row: pd.Series, next_row: pd.Series, direction: str, 
                       current_balance: float, df: pd.DataFrame, config_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Simulate a single trade execution
        
        Args:
            row: Current data row
            next_row: Next data row
            direction: Trade direction ("LONG" or "SHORT")
            current_balance: Current account balance
            df: Complete DataFrame (for ATR fallback calculation)
            config_id: Configuration ID
            
        Returns:
            Dictionary with trade result and new balance, or None if trade fails
        """
        try:
            # İşlem detaylarını hesapla
            entry_price = row["close"]
            
            # ATR değerini önce row'dan almaya çalış, yoksa hesapla
            if "atr" in row and not pd.isna(row["atr"]):
                atr = row["atr"]
            elif "atr_14" in row and not pd.isna(row["atr_14"]):
                atr = row["atr_14"]
            else:
                # Fallback: Simple volatility calculation
                atr = df["close"].pct_change().abs().mean() * entry_price
            
            # Stop-loss ve take-profit seviyeleri
            # 🔧 DÜZELTİLMİŞ TP/SL HESAPLAMA
            if direction == "LONG":
                sl = entry_price - (atr * self.sl_multiplier)
                tp = entry_price + (atr * self.tp_multiplier)
            else:  # SHORT
                sl = entry_price + (atr * self.sl_multiplier)  
                tp = entry_price - (atr * self.tp_multiplier)

            # 🚨 CRİTİCAL: ATR ve multiplier kontrolü
            if atr <= 0:
                logger.error(f"ATR is zero or negative: {atr}, using fallback")
                # Fallback: Entry price'ın %0.5'i kadar distance
                fallback_distance = entry_price * 0.005
                if direction == "LONG":
                    sl = entry_price - fallback_distance
                    tp = entry_price + (fallback_distance * 2)
                else:  # SHORT
                    sl = entry_price + fallback_distance
                    tp = entry_price - (fallback_distance * 2)

            # 🚨 CRİTİCAL: TP=SL=Entry kontrolü
            if abs(tp - entry_price) < 0.0001 or abs(sl - entry_price) < 0.0001:
                logger.error(f"TP/SL too close to entry: Entry={entry_price}, TP={tp}, SL={sl}, ATR={atr}")
                return None  # Bu trade'i skip et

            # Debug log
            logger.debug(f"{direction} Trade: Entry={entry_price:.4f}, TP={tp:.4f}, SL={sl:.4f}, ATR={atr:.4f}")
            
            # Next bar değerlerini kontrol et
            high = next_row["high"]
            low = next_row["low"]
            
            # İşlem sonucunu belirle
            # 🔧 DÜZELTİLMİŞ İŞLEM SONUCU BELİRLEME
            if direction == "LONG":
                # LONG: TP yukarıda, SL aşağıda
                if high >= tp:
                    outcome = "TP"
                    exit_price = tp
                elif low <= sl:
                    outcome = "SL"  
                    exit_price = sl
                else:
                    outcome = "OPEN"
                    exit_price = next_row["close"]
            else:  # SHORT
                # SHORT: TP aşağıda, SL yukarıda
                if low <= tp:
                    outcome = "TP"
                    exit_price = tp
                elif high >= sl:
                    outcome = "SL"
                    exit_price = sl  
                else:
                    outcome = "OPEN"
                    exit_price = next_row["close"]



            
            # Risk-reward oranı
            rr_ratio = self.tp_multiplier / self.sl_multiplier
            
            # Kazanç/kayıp hesaplama
            if direction == "LONG":
                gain_pct = ((exit_price / entry_price) - 1) * 100
            else:  # SHORT
                gain_pct = ((entry_price / exit_price) - 1) * 100
            
            # Position size ve commission
            position_size = current_balance * self.risk_per_trade * self.leverage
            commission = position_size * self.commission_rate
            
            # Net kazanç
            gain_usd = (gain_pct / 100) * position_size - commission
            new_balance = current_balance + gain_usd
            
            # İşlem kaydı
            trade = {
                "config_id": config_id,
                "time": row["open_time"],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "atr": atr,
                "sl": sl,
                "tp": tp,
                "rr_ratio": rr_ratio,
                "outcome": outcome,
                "gain_pct": gain_pct,
                "gain_usd": gain_usd,
                "commission": commission,
                "balance": new_balance,
                "position_size": position_size,
                "signal_strength": row.get("signal_strength", 0),
                "signal_passed_filter": row.get("signal_passed_filter", True)
            }
            
            # İndikatör değerlerini ekle
            for col in df.columns:
                if col.startswith(('rsi', 'macd', 'obv', 'adx', 'cci', 'supertrend', 'ema', 'sma')):
                    trade[col] = row.get(col)
            
            return {
                "trade": trade,
                "new_balance": new_balance
            }
            
        except Exception as e:
            logger.error(f"❌ Error simulating trade: {e}")
            return None
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            "total_trades": 0,
            "final_balance": self.initial_balance,
            "profit_loss": 0,
            "roi_pct": 0,
            "trades": [],
            "equity_curve": [],
            "metrics": {},
            "error": error_message
        }
    
    def _calculate_performance_metrics(self) -> None:
        """Performans metriklerini hesaplar ve saklar - ENHANCED VERSION"""
        if not self.trades:
            self.metrics = {
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "direction_performance": {},
                "avg_gain_per_trade": 0,
                "avg_rr_ratio": 0,
                "outcome_distribution": {},
                "strategy_performance": {}
            }
            return
            
        # Trade sonuçlarını pandas DataFrame'e çevir
        trades_df = pd.DataFrame(self.trades)
        
        # Win rate
        # 🔧 DÜZELTİLMİŞ WIN RATE HESAPLAMA
        # Win rate'i gain'e göre hesapla, outcome'a göre değil
        profitable_trades = trades_df[trades_df["gain_pct"] > 0]
        total_trades = len(trades_df)
        win_rate = (len(profitable_trades) / total_trades * 100) if total_trades > 0 else 0

        # Debug log
        logger.info(f"Win rate calculation: {len(profitable_trades)} profitable / {total_trades} total = {win_rate:.2f}%")
        
        # Profit factor
        winning_trades = trades_df[trades_df["gain_usd"] > 0]
        losing_trades = trades_df[trades_df["gain_usd"] < 0]
        
        gross_profit = winning_trades["gain_usd"].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades["gain_usd"].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown hesaplama
        if self.equity_curve:
            equity = pd.DataFrame(self.equity_curve)
            equity["drawdown"] = equity["balance"].cummax() - equity["balance"]
            equity["drawdown_pct"] = equity["drawdown"] / equity["balance"].cummax() * 100
            max_drawdown = equity["drawdown"].max()
            max_drawdown_pct = equity["drawdown_pct"].max()
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Sharpe oranı
        if len(trades_df) > 1:
            returns = trades_df["gain_pct"].values
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        else:
            sharpe = 0
        
        # Yöne göre performans
        direction_stats = {}
        for direction in trades_df["direction"].unique():
            direction_df = trades_df[trades_df["direction"] == direction]
            direction_closed = direction_df[direction_df["outcome"].isin(["TP", "SL"])]
            direction_win_count = (direction_closed["outcome"] == "TP").sum()
            direction_win_rate = (direction_win_count / len(direction_closed) * 100) if len(direction_closed) > 0 else 0
            
            direction_stats[direction] = {
                "count": len(direction_df),
                "win_rate": direction_win_rate,
                "avg_gain": direction_df["gain_pct"].mean(),
                "total_profit": direction_df["gain_usd"].sum(),
                "winning_trades": direction_win_count,
                "losing_trades": len(direction_closed) - direction_win_count
            }
        
        # Outcome distribution
        outcome_stats = trades_df["outcome"].value_counts().to_dict()
        
        # Strategy performance (if multiple strategies used)
        strategy_stats = {}
        if "strategy_name" in trades_df.columns:
            for strategy in trades_df["strategy_name"].unique():
                strategy_df = trades_df[trades_df["strategy_name"] == strategy]
                strategy_stats[strategy] = {
                    "count": len(strategy_df),
                    "win_rate": (strategy_df["outcome"] == "TP").sum() / len(strategy_df) * 100,
                    "avg_gain": strategy_df["gain_pct"].mean(),
                    "total_profit": strategy_df["gain_usd"].sum()
                }
        
        # Tüm metrikleri sakla
        self.metrics = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe,
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "direction_performance": direction_stats,
            "avg_gain_per_trade": trades_df["gain_pct"].mean(),
            "avg_rr_ratio": trades_df["rr_ratio"].mean(),
            "outcome_distribution": outcome_stats,
            "strategy_performance": strategy_stats,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_signal_strength": trades_df["signal_strength"].mean() if "signal_strength" in trades_df.columns else 0
        }
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of configured strategies"""
        return {
            "total_strategies": len(getattr(self.strategy_manager, '_strategies_to_use', [])),
            "strategy_weights": getattr(self.strategy_manager, '_strategy_weights', {}),
            "available_categories": self.strategy_manager.list_available_strategies()
        }