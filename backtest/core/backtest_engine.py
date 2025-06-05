"""
Temel backtest motoru - ENHANCED VERSION with Multi-Bar Trade Tracking
Signal Engine ile entegre çalışır ve ticaret stratejilerini değerlendirir.

MAJOR ENHANCEMENT: Fixed trade outcome simulation with realistic multi-bar tracking
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import os
import logging

# Signal Engine importları
from signal_engine.indicators import registry as indicator_registry
from signal_engine.signal_indicator_plugin_system import IndicatorManager
from signal_engine.strategies import registry as strategy_registry
from signal_engine.signal_strategy_system import StrategyManager
from signal_engine.strength import registry as strength_registry
from signal_engine.signal_strength_system import StrengthManager
from signal_engine.filters import registry as filter_registry
from signal_engine.signal_filter_system import FilterManager

logger = logging.getLogger(__name__)


@dataclass
class TradeState:
    """Enhanced trade state for multi-bar tracking"""
    entry_bar: int
    entry_price: float
    sl: float
    tp: float
    direction: str
    position_size: float
    max_holding_bars: int
    atr: float
    signal_strength: float
    position_details: Dict[str, Any]
    
    # Additional tracking for advanced features
    entry_time: Optional[int] = None
    highest_price: Optional[float] = None  # For trailing stops (future enhancement)
    lowest_price: Optional[float] = None   # For trailing stops (future enhancement)
    
    def __post_init__(self):
        """Initialize tracking prices"""
        if self.direction == "LONG":
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price
        else:
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price


class BacktestEngine:
    """ENHANCED: Signal Engine ile entegre çalışan backtest motoru with realistic trade tracking."""
    
    def __init__(self, 
                 symbol: str, 
                 interval: str, 
                 initial_balance: float = 10000.0,
                 risk_per_trade: float = 0.01,
                 sl_multiplier: float = 1.5,
                 tp_multiplier: float = 3.0,
                 leverage: float = 1.0,
                 position_direction: Dict[str, bool] = None,
                 commission_rate: float = 0.001,
                 max_holding_bars: int = 500,
                # ✅ YENİ: TP/Commission filter parametreleri
                 min_tp_commission_ratio: float = 3.0,
                 max_commission_impact_pct: float = 15.0,
                 min_position_size: float = 800.0,
                 min_net_rr_ratio: float = 1.5,
                 enable_tp_commission_filter: bool = False):
        """
        ENHANCED: Backtest motorunu başlatır with multi-bar trade tracking
        
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
            max_holding_bars: Maximum bars to hold a position
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
        self.max_holding_bars = max_holding_bars
        # ✅ YENİ: TP/Commission filter parametreleri
        self.min_tp_commission_ratio = min_tp_commission_ratio
        self.max_commission_impact_pct = max_commission_impact_pct
        self.min_position_size = min_position_size
        self.min_net_rr_ratio = min_net_rr_ratio
        self.enable_tp_commission_filter = enable_tp_commission_filter
        # ✅ VALIDATION: Log filter status
        if self.enable_tp_commission_filter:
            logger.info("🔧 TP/Commission filter ENABLED with parameters:")
            logger.info(f"   min_tp_commission_ratio: {self.min_tp_commission_ratio}")
            logger.info(f"   max_commission_impact_pct: {self.max_commission_impact_pct}%")
            logger.info(f"   min_position_size: ${self.min_position_size}")
            logger.info(f"   min_net_rr_ratio: {self.min_net_rr_ratio}")
        else:
            logger.warning("⚠️ TP/Commission filter DISABLED")
        
        # Signal Engine bileşenleri
        self.indicator_manager = IndicatorManager(indicator_registry)
        self.strategy_manager = StrategyManager(strategy_registry)
        self.strength_manager = StrengthManager(strength_registry)
        self.filter_manager = FilterManager(filter_registry)
        
        # Strategy configuration storage
        self.strategy_config = {}
        
        # Sonuçları saklayacak konteynerler
        self.trades = []
        self.equity_curve = []
        self.metrics = {}
        # ✅ TP/Commission filter logging
        if self.enable_tp_commission_filter:
            logger.info("🔧 TP/Commission filter ENABLED:")
            logger.info(f"   Min TP/Commission ratio: {self.min_tp_commission_ratio}x")
            logger.info(f"   Max commission impact: {self.max_commission_impact_pct}%")
            logger.info(f"   Min position size: ${self.min_position_size}")
            logger.info(f"   Min net R:R ratio: {self.min_net_rr_ratio}:1")
    
    def configure_signal_engine(self, 
                               indicators_config: Dict[str, Any] = None,
                               strategies_config: Dict[str, Any] = None,
                               strength_config: Dict[str, Any] = None,
                               filter_config: Dict[str, Any] = None) -> None:
        """
        Signal Engine bileşenlerini yapılandırır
        """
        # İndikatörleri yapılandır
        if indicators_config:
            long_indicators = indicators_config.get('long', {})
            short_indicators = indicators_config.get('short', {})
            
            all_indicators = {**long_indicators, **short_indicators}
            for indicator_name, params in all_indicators.items():
                self.indicator_manager.add_indicator(indicator_name, params)
        
        # Stratejileri yapılandır
        if strategies_config:
            self.strategy_config = strategies_config
            self._configure_strategies(strategies_config)
        
        # Sinyal gücü hesaplayıcılarını yapılandır
        if strength_config:
            for calculator_name, params in strength_config.items():
                self.strength_manager.add_calculator(calculator_name, params)
        
        # Filtreleri yapılandır
        if filter_config:
            logger.info(f"🔧 Configuring filters from config with {len(filter_config)} items...")
            
            config_params = {'min_checks', 'min_strength'}
            
            rules_added = 0
            for rule_name, params in filter_config.items():
                if rule_name in config_params:
                    continue
                    
                try:
                    self.filter_manager.add_rule(rule_name, params)
                    logger.info(f"✅ Added filter rule: {rule_name}")
                    rules_added += 1
                except Exception as e:
                    logger.error(f"❌ Failed to add filter rule {rule_name}: {e}")
            
            logger.info(f"✅ Successfully added {rules_added} filter rules")
            
            if "min_checks" in filter_config:
                self.filter_manager.set_min_checks_required(filter_config["min_checks"])
                
            if "min_strength" in filter_config:
                self.filter_manager.set_min_strength_required(filter_config["min_strength"])
                
        else:
            logger.info("⚠️ No filter config provided, adding default filters...")
            self._add_default_filters()

    def _add_default_filters(self) -> None:
        """Varsayılan filtreleri ekler"""
        default_filters = ['market_regime', 'trend_strength', 'volatility_regime']
        
        for filter_name in default_filters:
            try:
                self.filter_manager.add_rule(filter_name)
                logger.info(f"✅ Added default filter: {filter_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not add default filter {filter_name}: {e}")
        
        self.filter_manager.set_min_checks_required(2)
        self.filter_manager.set_min_strength_required(45)
        logger.info("✅ Default filters added with min_checks=2 and min_strength=45")

    def _configure_strategies(self, strategies_config: Dict[str, Any]) -> None:
        """Strategy configuration"""
        logger.info("🔧 Configuring strategies...")
        
        configured_count = 0
        for strategy_name, strategy_config in strategies_config.items():
            try:
                params, weight = self._parse_strategy_config(strategy_config)
                
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
    
    def _parse_strategy_config(self, strategy_config: Any) -> Tuple[Dict[str, Any], float]:
        """Parse strategy configuration from various formats"""
        if isinstance(strategy_config, dict):
            if "params" in strategy_config or "weight" in strategy_config:
                params = strategy_config.get("params", {})
                weight = strategy_config.get("weight", 1.0)
            else:
                params = strategy_config
                weight = 1.0
        elif isinstance(strategy_config, (int, float)):
            params = {}
            weight = float(strategy_config)
        else:
            params = {}
            weight = 1.0
        
        return params, weight
    
    def run(self, df: pd.DataFrame, config_id: str = None) -> Dict[str, Any]:
        """
        ENHANCED: Multi-bar trade tracking ile backtest çalıştırır
        
        Args:
            df: Fiyat verisi DataFrame
            config_id: Konfigürasyon ID'si
            
        Returns:
            Backtest sonuç özeti
        """
        logger.info(f"🚀 Starting ENHANCED backtest for {self.symbol} {self.interval}")

         # ✅ TP/Commission filter status log
        if self.enable_tp_commission_filter:
            logger.info("🔧 TP/Commission filter ACTIVE - Only profitable trades will be opened")
        else:
            logger.warning("⚠️ TP/Commission filter DISABLED - All signals will be processed")
        
        # Başlangıç
        balance = self.initial_balance
        trades = []
        equity_curve = [{"time": df.iloc[0]["open_time"], "balance": balance}]
        
        # ENHANCED: Active trades tracking
        active_trades = []  # List of open TradeState objects
        trade_id_counter = 0

         # ✅ TP/Commission filter statistics
        total_signals = 0
        filtered_signals = 0
        tp_commission_rejections = 0
        position_size_rejections = 0
            
        # Signal Engine sürecini çalıştır
        try:
            # 1. İndikatörleri hesapla
            logger.info("📊 Calculating indicators...")
            df = self.indicator_manager.calculate_indicators(df)
            logger.info(f"✅ Indicators calculated")

            # 2. Sinyalleri oluştur
            logger.info("🎯 Generating strategy signals...")
            df = self.strategy_manager.generate_signals(df)
            logger.info("✅ Strategy signals generated")
            
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
            calculator_names = getattr(self.strength_manager, '_calculators_to_use', [])
            
            if calculator_names:
                calculator_params = getattr(self.strength_manager, '_calculator_params', {})
                strength_series = self.strength_manager.calculate_strength(
                    df, df, calculator_names, calculator_params
                )
                df['signal_strength'] = strength_series
                logger.info(f"✅ Signal strength calculated, avg strength: {strength_series.mean():.2f}")
            else:
                df['signal_strength'] = 1.0
                logger.info("⚠️ No strength calculators configured, using default strength")
                    
            # 4. Sinyalleri filtrele
            logger.info("🔍 Applying filters...")
            
            rules_to_apply = getattr(self.filter_manager, '_rules_to_apply', [])
            min_checks = getattr(self.filter_manager, '_min_checks_required', 0)
            min_strength = getattr(self.filter_manager, '_min_strength_required', 0)
            
            logger.info(f"📋 Filter rules: {len(rules_to_apply)}, min_checks: {min_checks}, min_strength: {min_strength}")
            
            if not rules_to_apply:
                logger.error("❌ NO FILTER RULES CONFIGURED!")
                self._add_default_filters()
            
            # Pre-filter signal count
            pre_long = df.get("long_signal", pd.Series(False)).sum()
            pre_short = df.get("short_signal", pd.Series(False)).sum()
            logger.info(f"📊 Pre-filter signals: {pre_long} long, {pre_short} short")
            
            try:
                df = self.filter_manager.filter_signals(df)
                logger.info("✅ Filters applied successfully")
            except Exception as e:
                logger.error(f"❌ Error applying filters: {e}")
            
            # Final signal statistics
            final_long = df["long_signal"].sum() if "long_signal" in df.columns else 0
            final_short = df["short_signal"].sum() if "short_signal" in df.columns else 0
            logger.info(f"🎯 Final signals after filtering: {final_long} long, {final_short} short")
            
        except Exception as e:
            logger.error(f"❌ Error in signal generation: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(str(e))
        
        # ENHANCED: Multi-bar trade simulation
        logger.info("💼 Starting ENHANCED trade simulation with multi-bar tracking...")
        trades_opened = 0
        trades_closed = 0
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            
            # 1. CHECK EXISTING TRADES FIRST - Update tracking prices
            for trade_state in active_trades:
                self._update_trade_tracking(trade_state, current_row)
            
            # 2. CHECK FOR TRADE EXITS
            trades_to_close = []
            for trade_idx, trade_state in enumerate(active_trades):
                exit_result = self._check_trade_exit(current_row, trade_state, i)
                
                if exit_result:
                    # Close the trade
                    completed_trade = self._close_trade(trade_state, exit_result, current_row, config_id)
                    trades.append(completed_trade)
                    balance += completed_trade["net_pnl"]
                    trades_to_close.append(trade_idx)
                    trades_closed += 1
                    
                    # Update equity curve
                    equity_curve.append({
                        "time": current_row["open_time"], 
                        "balance": balance
                    })
                    
                    logger.debug(f"📉 Closed {trade_state.direction} trade: {exit_result['outcome']} at bar {i}")
            
            # Remove closed trades (reverse order to maintain indices)
            for idx in reversed(trades_to_close):
                active_trades.pop(idx)
            
            # 3. CHECK FOR NEW TRADE SIGNALS (don't open on last bar)
            if i < len(df) - 1:
                new_trade = self._check_new_trade_signal(current_row, balance, i, df)
                # ✅ Filter statistics tracking
                has_signal = (current_row.get("long_signal", False) or 
                            current_row.get("short_signal", False))
            
                if has_signal:
                    if new_trade is None and self.enable_tp_commission_filter:
                        filtered_signals += 1
                        # Check rejection reason (bu detay _check_new_trade_signal'de loglanıyor)
                if new_trade:
                    active_trades.append(new_trade)
                    trades_opened += 1
                    logger.debug(f"📈 Opened {new_trade.direction} trade at bar {i}")
        
        # # 4. CLOSE ANY REMAINING OPEN TRADES AT END
        # final_row = df.iloc[-1]
        # for trade_state in active_trades:
        #     exit_result = {
        #         "outcome": "MARKET_CLOSE",
        #         "exit_price": final_row["close"],
        #         "exit_bar": len(df) - 1
        #     }
        #     completed_trade = self._close_trade(trade_state, exit_result, final_row, config_id)
        #     trades.append(completed_trade)
        #     balance += completed_trade["net_pnl"]
        #     trades_closed += 1
         # 4. HANDLE REMAINING OPEN TRADES AT END
        # CRITICAL: We have two options here:
        # Option A: Close them at market close (add to analysis)
        # Option B: Ignore them completely (exclude from analysis)
        
        # OPTION B: IGNORE OPEN TRADES COMPLETELY
        trades_ignored = len(active_trades)
        logger.info(f"⚠️ Ignoring {trades_ignored} OPEN trades at end of backtest")
        
        # Clear active trades without processing them
        active_trades.clear()
        

        logger.info(f"✅ ENHANCED trade simulation completed:")
        logger.info(f"   📈 Trades opened: {trades_opened}")
        logger.info(f"   📉 Trades closed: {trades_closed}")
        logger.info(f"   🎯 Total signals: {total_signals}")
        logger.info(f"   🔧 Filtered signals: {filtered_signals}")
        logger.info(f"   📊 Filter efficiency: {(filtered_signals/max(total_signals,1)*100):.1f}%")
        logger.info(f"   💰 Final balance: ${balance:.2f}")
        
        # Sonuçları sakla
        self.trades = trades
        self.equity_curve = equity_curve
        
        # Performans metriklerini hesapla
        self._calculate_performance_metrics_enhanced()
        
        # ✅ ENHANCED result with filter statistics
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
                "risk_per_trade": self.risk_per_trade,
                "max_holding_bars": self.max_holding_bars,
                # ✅ TP/Commission filter config
                "enable_tp_commission_filter": self.enable_tp_commission_filter,
                "min_tp_commission_ratio": self.min_tp_commission_ratio,
                "max_commission_impact_pct": self.max_commission_impact_pct,
                "min_position_size": self.min_position_size
            },
            "open_trades_ignored": trades_ignored,
            # ✅ Filter statistics
            "filter_statistics": {
                "total_signals": total_signals,
                "filtered_signals": filtered_signals,
                "filter_efficiency_pct": (filtered_signals/max(total_signals,1)*100),
                "trades_opened": trades_opened
            }
        }
        
        logger.info(f"🏁 ENHANCED backtest completed: {len(trades)} trades, ROI: {result['roi_pct']:.2f}%")
        return result

    def _update_trade_tracking(self, trade_state: TradeState, current_row: pd.Series) -> None:
        """Update trade tracking prices for future enhancements like trailing stops"""
        current_high = current_row["high"]
        current_low = current_row["low"]
        
        if trade_state.direction == "LONG":
            trade_state.highest_price = max(trade_state.highest_price, current_high)
            trade_state.lowest_price = min(trade_state.lowest_price, current_low)
        else:  # SHORT
            trade_state.highest_price = max(trade_state.highest_price, current_high)
            trade_state.lowest_price = min(trade_state.lowest_price, current_low)

    def _check_trade_exit(self, current_row: pd.Series, trade_state: TradeState, current_bar: int) -> Optional[Dict]:
        """
        ENHANCED: Check if an active trade should be closed with realistic logic
        
        Returns:
            Dict with exit details or None if trade should remain open
        """
        high = current_row["high"]
        low = current_row["low"]
        close = current_row["close"]
        
        # Check max holding period
        bars_held = current_bar - trade_state.entry_bar
        if bars_held >= trade_state.max_holding_bars:
            return {
                "outcome": "MAX_HOLDING",
                "exit_price": close,
                "exit_bar": current_bar,
                "bars_held": bars_held
            }
        
        if trade_state.direction == "LONG":
            # Check TP first (favorable scenario)
            if high >= trade_state.tp:
                return {
                    "outcome": "TP",
                    "exit_price": trade_state.tp,
                    "exit_bar": current_bar,
                    "bars_held": bars_held
                }
            # Check SL
            elif low <= trade_state.sl:
                return {
                    "outcome": "SL", 
                    "exit_price": trade_state.sl,
                    "exit_bar": current_bar,
                    "bars_held": bars_held
                }
        
        else:  # SHORT
            # Check TP first (favorable scenario)
            if low <= trade_state.tp:
                return {
                    "outcome": "TP",
                    "exit_price": trade_state.tp,
                    "exit_bar": current_bar,
                    "bars_held": bars_held
                }
            # Check SL
            elif high >= trade_state.sl:
                return {
                    "outcome": "SL",
                    "exit_price": trade_state.sl,
                    "exit_bar": current_bar,
                    "bars_held": bars_held
                }
        
        return None  # Trade remains open

    def _check_new_trade_signal(self, current_row: pd.Series, balance: float, 
                               current_bar: int, df: pd.DataFrame) -> Optional[TradeState]:
        """
        ENHANCED: Check for new trade signals and create TradeState if valid
        """
        # Signal strength check
        if current_row.get("signal_strength", 0) < 1:
            return None
        
        # Direction determination
        direction = None
        if current_row.get("long_signal", False):
            direction = "LONG"
        elif current_row.get("short_signal", False):
            direction = "SHORT"
        
        if direction is None:
            return None
        
        try:
            # Calculate trade parameters
            entry_price = current_row["close"]
            atr = self._get_robust_atr(current_row, df, entry_price)
            sl, tp = self._calculate_sl_tp_levels(entry_price, atr, direction)
            position_details = self._calculate_position_size(balance, entry_price, sl, direction)

            # ✅ CRITICAL: Calculate TP/Commission analysis BEFORE filtering
            tp_analysis = self._analyze_tp_commission_profitability(
                entry_price, sl, tp, position_details["position_value"]
            )

            # 🔧 FIXED: Ensure filter is actually applied when enabled
            if self.enable_tp_commission_filter:
                
                # Debug logging to see what's happening
                logger.debug(f"🔍 FILTERING Trade at bar {current_bar}:")
                logger.debug(f"   Position: ${position_details['position_value']:.2f}")
                logger.debug(f"   Commission impact: {tp_analysis['commission_impact_pct']:.1f}%")
                logger.debug(f"   TP/Commission ratio: {tp_analysis['tp_commission_ratio']:.1f}x")
                
                # Filter 1: Minimum position size
                if position_details["position_value"] < self.min_position_size:
                    logger.debug(f"❌ REJECTED: Position ${position_details['position_value']:.0f} < ${self.min_position_size}")
                    return None
                
                # Filter 2: TP/Commission ratio
                if tp_analysis["tp_commission_ratio"] < self.min_tp_commission_ratio:
                    logger.debug(f"❌ REJECTED: TP/Comm {tp_analysis['tp_commission_ratio']:.1f}x < {self.min_tp_commission_ratio}x")
                    return None
                
                # Filter 3: Commission impact percentage  
                if tp_analysis["commission_impact_pct"] > self.max_commission_impact_pct:
                    logger.debug(f"❌ REJECTED: Comm impact {tp_analysis['commission_impact_pct']:.1f}% > {self.max_commission_impact_pct}%")
                    return None
                
                # Filter 4: Net R:R ratio
                if tp_analysis["net_rr_ratio"] < self.min_net_rr_ratio:
                    logger.debug(f"❌ REJECTED: Net R:R {tp_analysis['net_rr_ratio']:.2f} < {self.min_net_rr_ratio}")
                    return None
                
                # ✅ PASSED ALL FILTERS
                logger.debug(f"✅ APPROVED: All filters passed")

            # Create trade state
            trade_state = TradeState(
                entry_bar=current_bar,
                entry_price=entry_price,
                sl=sl,
                tp=tp,
                direction=direction,
                position_size=position_details["position_size"],
                max_holding_bars=self.max_holding_bars,
                atr=atr,
                signal_strength=current_row.get("signal_strength", 0),
                position_details={**position_details, **tp_analysis},
                entry_time=current_row.get("open_time")
            )
            
            return trade_state
            
        except Exception as e:
            logger.error(f"❌ Error creating new trade: {e}")
            return None
    
    def _analyze_tp_commission_profitability(self, entry_price: float, sl: float, tp: float, 
                                       position_value: float) -> Dict[str, Any]:
        """
        EKSİK FONKSIYON - TP komisyonu karşılama analizini yapar
        """
        # TP'de beklenen brüt kazanç
        tp_distance_pct = abs(tp - entry_price) / entry_price
        expected_tp_gross = position_value * tp_distance_pct
        
        # Toplam komisyon (açılış + kapanış)
        total_commission = position_value * self.commission_rate * 2
        
        # SL'de beklenen brüt zarar
        sl_distance_pct = abs(entry_price - sl) / entry_price  
        expected_sl_gross_loss = position_value * sl_distance_pct

        # ✅ FIXED: Use actual commission rate from self
        total_commission = position_value * self.commission_rate * 2  # Round trip
        
        # Net karlılık hesaplamaları
        expected_tp_net = expected_tp_gross - total_commission
        expected_sl_net_loss = expected_sl_gross_loss + total_commission
        
        # Kritik oranlar
        tp_commission_ratio = expected_tp_gross / total_commission if total_commission > 0 else 0
        commission_impact_pct = (total_commission / expected_tp_gross * 100) if expected_tp_gross > 0 else 100
        net_rr_ratio = expected_tp_net / expected_sl_net_loss if expected_sl_net_loss > 0 else 0

          # Debug validation
        logger.debug(f"📊 Commission Analysis:")
        logger.debug(f"   Expected TP gross: ${expected_tp_gross:.2f}")
        logger.debug(f"   Total commission: ${total_commission:.2f}")
        logger.debug(f"   Commission impact: {commission_impact_pct:.1f}%")
        logger.debug(f"   TP/Commission ratio: {tp_commission_ratio:.1f}x")
        
        return {
            "expected_tp_gross": expected_tp_gross,
            "expected_tp_net": expected_tp_net,
            "total_commission": total_commission,
            "tp_commission_ratio": tp_commission_ratio,
            "commission_impact_pct": commission_impact_pct,
            "net_rr_ratio": net_rr_ratio,
            "expected_sl_net_loss": expected_sl_net_loss,
            "position_value": position_value
        }

    def _close_trade(self, trade_state: TradeState, exit_result: Dict, 
                    current_row: pd.Series, config_id: str) -> Dict:
        """
        ENHANCED: Close a trade and calculate final PnL with comprehensive metrics
        """
        exit_price = exit_result["exit_price"]
        outcome = exit_result["outcome"]
        bars_held = exit_result.get("bars_held", 0)
        
        # Calculate PnL
        pnl_details = self._calculate_pnl(
            trade_state.entry_price, 
            exit_price, 
            trade_state.direction, 
            trade_state.position_details, 
            outcome
        )
        
        # Build comprehensive trade record
        trade = self._build_trade_record_enhanced(
            trade_state, exit_result, current_row, pnl_details, config_id
        )
        
        return trade

        # backtest_engine.py dosyasında _calculate_position_size metodunu değiştir

    def _calculate_position_size(self, balance: float, entry_price: float, 
                           sl: float, direction: str) -> dict:
        """
        ENHANCED: FIXED position sizing with proper leverage-aware logic
        
        Args:
            balance: Account balance
            entry_price: Trade entry price
            sl: Stop loss price
            direction: Trade direction
            
        Returns:
            Position sizing details with leverage consideration
        """
        # Calculate risk amount in USD
        risk_amount = balance * self.risk_per_trade
        
        # Calculate distance to SL in percentage
        sl_distance_pct = abs(entry_price - sl) / entry_price
        
        # Prevent division by zero
        if sl_distance_pct <= 0:
            sl_distance_pct = 0.005  # 0.5% fallback
        
        # FIXED: Calculate position value based on risk
        # Risk Amount = Position Value * SL Distance %
        position_value = risk_amount / sl_distance_pct
        
        # FIXED: Leverage-aware position limits
        max_margin_required = balance * 0.8  # Use max 80% of balance as margin
        max_position_value = max_margin_required * self.leverage  # Leverage multiplier
        
        # Apply position limit
        if position_value > max_position_value:
            logger.warning(f"Position limited: ${position_value:.2f} → ${max_position_value:.2f}")
            position_value = max_position_value
        
        # Calculate required margin
        required_margin = position_value / self.leverage
        
        # Validate margin requirement
        if required_margin > balance * 0.9:  # Don't use more than 90% as margin
            logger.warning(f"Insufficient margin: required ${required_margin:.2f}, available ${balance * 0.9:.2f}")
            required_margin = balance * 0.9
            position_value = required_margin * self.leverage
        
        # FIXED: Position size is same as position value (for crypto)
        position_size = position_value
        
        logger.debug(f"Position Sizing: Balance=${balance}, Risk=${risk_amount}, "
                    f"SL_Distance={sl_distance_pct*100:.2f}%, Position=${position_value:.2f}, "
                    f"Margin=${required_margin:.2f}")
        
        return {
            "position_size": position_size,
            "position_value": position_value,  # Actual trade size
            "required_margin": required_margin,  # Margin needed
            "risk_amount": risk_amount,  # Actual risk
            "sl_distance_pct": sl_distance_pct,
            "leverage_used": self.leverage,
            "margin_usage_pct": (required_margin / balance) * 100
        }


    def _calculate_pnl(self, entry_price: float, exit_price: float, direction: str, 
                    position_details: dict, outcome: str) -> dict:
        """
        ENHANCED: FIXED PnL calculation with accurate commission and margin
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: Trade direction
            position_details: Position details from _calculate_position_size
            outcome: Trade outcome
            
        Returns:
            Comprehensive PnL breakdown
        """
        position_value = position_details["position_value"]
        
        # Calculate price change percentage
        if direction == "LONG":
            price_change_pct = ((exit_price / entry_price) - 1) * 100
        else:  # SHORT
            price_change_pct = ((entry_price / exit_price) - 1) * 100
        
        # Calculate gross PnL based on position value
        gross_pnl = (price_change_pct / 100) * position_value
        
        # FIXED: Calculate commission on position value (trading volume)
        entry_commission = position_value * self.commission_rate
        
        # Exit commission on exit value
        exit_value = position_value  # For simplicity, use same value
        exit_commission = exit_value * self.commission_rate
        total_commission = entry_commission + exit_commission
        
        # Calculate slippage (market impact)
        slippage_rate = 0.0005 if outcome in ["TP", "SL"] else 0.0002
        slippage_cost = position_value * slippage_rate
        
        # Net PnL
        net_pnl = gross_pnl - total_commission - slippage_cost
        
        # Calculate commission as percentage of gross PnL for monitoring
        commission_impact_pct = 0
        if abs(gross_pnl) > 0:
            commission_impact_pct = (total_commission / abs(gross_pnl)) * 100
        
        logger.debug(f"PnL Calculation: Gross=${gross_pnl:.2f}, Commission=${total_commission:.2f} "
                    f"({commission_impact_pct:.1f}%), Net=${net_pnl:.2f}")
        
        return {
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "price_change_pct": price_change_pct,
            "total_commission": total_commission,
            "slippage_cost": slippage_cost,
            "entry_commission": entry_commission,
            "exit_commission": exit_commission,
            "commission_impact_pct": commission_impact_pct
        }


    def _build_trade_record_enhanced(self, trade_state: TradeState, exit_result: Dict,
                               current_row: pd.Series, pnl_details: Dict, config_id: str) -> Dict:
        """
        ENHANCED: Build comprehensive trade record with leverage-aware metrics
        """
        # Calculate risk-reward ratio
        if trade_state.direction == "LONG":
            risk_distance = trade_state.entry_price - trade_state.sl
            reward_distance = trade_state.tp - trade_state.entry_price
        else:
            risk_distance = trade_state.sl - trade_state.entry_price
            reward_distance = trade_state.entry_price - trade_state.tp
        
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        trade = {
            # Basic trade info
            "config_id": config_id,
            "time": trade_state.entry_time or current_row.get("open_time"),
            "exit_time": current_row.get("open_time"),
            "direction": trade_state.direction,
            "entry_price": trade_state.entry_price,
            "exit_price": exit_result["exit_price"],
            "outcome": exit_result["outcome"],
            
            # Multi-bar tracking metrics
            "entry_bar": trade_state.entry_bar,
            "exit_bar": exit_result["exit_bar"],
            "bars_held": exit_result.get("bars_held", 0),
            "highest_price_reached": trade_state.highest_price,
            "lowest_price_reached": trade_state.lowest_price,
            
            # Risk management
            "sl": trade_state.sl,
            "tp": trade_state.tp,
            "rr_ratio": rr_ratio,
            "risk_distance_pct": abs(trade_state.sl - trade_state.entry_price) / trade_state.entry_price * 100,
            "reward_distance_pct": abs(trade_state.tp - trade_state.entry_price) / trade_state.entry_price * 100,
            
            # ENHANCED: Position details with leverage
            "position_size": trade_state.position_size,
            "position_value": trade_state.position_details["position_value"],
            "required_margin": trade_state.position_details.get("required_margin", 0),
            "margin_usage_pct": trade_state.position_details.get("margin_usage_pct", 0),
            "leverage_used": trade_state.position_details["leverage_used"],
            "risk_amount": trade_state.position_details["risk_amount"],
            
            # PnL breakdown
            "gross_pnl": pnl_details["gross_pnl"],
            "net_pnl": pnl_details["net_pnl"],
            "price_change_pct": pnl_details["price_change_pct"],
            "total_commission": pnl_details["total_commission"],
            "commission_impact_pct": pnl_details.get("commission_impact_pct", 0),
            "slippage_cost": pnl_details["slippage_cost"],
            
            # Legacy compatibility
            "gain_pct": pnl_details["price_change_pct"],
            "gain_usd": pnl_details["net_pnl"],
            "commission": pnl_details["total_commission"],
            
            # Signal context
            "signal_strength": trade_state.signal_strength,
            "signal_passed_filter": True,
            
            # Market context
            "atr": trade_state.atr,
            "volatility_regime": current_row.get("volatility_regime", "unknown"),
            
            # ENHANCED: New tracking fields
            "leverage_effective": (trade_state.position_details["position_value"] / 
                                trade_state.position_details.get("required_margin", 1)),
            "position_size_category": "large" if trade_state.position_details["position_value"] > 5000 else 
                                    "medium" if trade_state.position_details["position_value"] > 1000 else "small"
        }
        
        return trade

    def _get_robust_atr(self, row: pd.Series, df: pd.DataFrame, entry_price: float) -> float:
        """ENHANCED: Robust ATR calculation with multiple fallbacks"""
        # Priority 1: Standard ATR column
        if "atr" in row and not pd.isna(row["atr"]) and row["atr"] > 0:
            return row["atr"]
        
        # Priority 2: ATR with period suffix
        for period in [14, 20, 10]:
            col_name = f"atr_{period}"
            if col_name in row and not pd.isna(row[col_name]) and row[col_name] > 0:
                return row[col_name]
        
        # Priority 3: Calculate from recent data
        if len(df) >= 14:
            recent_data = df.tail(14)
            if all(col in recent_data.columns for col in ['high', 'low', 'close']):
                true_ranges = []
                for i in range(1, len(recent_data)):
                    current = recent_data.iloc[i]
                    previous = recent_data.iloc[i-1]
                    
                    tr1 = current['high'] - current['low']
                    tr2 = abs(current['high'] - previous['close'])
                    tr3 = abs(current['low'] - previous['close'])
                    
                    true_ranges.append(max(tr1, tr2, tr3))
                
                if true_ranges:
                    return np.mean(true_ranges)
        
        # Priority 4: Simple volatility fallback
        if 'close' in df.columns and len(df) >= 5:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std()
                if volatility > 0:  # FIXED: Avoid division by zero
                    return volatility * entry_price * np.sqrt(20)
        
        # Priority 5: FIXED fallback - more conservative
        return entry_price * 0.005  # 0.5% instead of 1%

    def _calculate_sl_tp_levels(self, entry_price: float, atr: float, direction: str) -> tuple:
        """ENHANCED: SL/TP calculation with validation"""
        atr_distance = atr * max(self.sl_multiplier, 0.5)
        pct_distance = entry_price * 0.005
        
        sl_distance = max(atr_distance, pct_distance)
        tp_distance = sl_distance * (self.tp_multiplier / self.sl_multiplier)
        
        if direction == "LONG":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        # Validation
        min_distance = entry_price * 0.001
        max_distance = entry_price * 0.05
        
        if abs(sl - entry_price) < min_distance or abs(sl - entry_price) > max_distance:
            logger.warning(f"SL distance out of range, using fallback")
            if direction == "LONG":
                sl = entry_price * (1 - 0.005)
                tp = entry_price * (1 + 0.015)
            else:
                sl = entry_price * (1 + 0.005)
                tp = entry_price * (1 - 0.015)
        
        return sl, tp

  
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
    
    def _calculate_performance_metrics_enhanced(self) -> None:
        """ENHANCED: Performans metriklerini hesaplar - COMPREHENSIVE VERSION"""
        if not self.trades:
            self.metrics = self._get_empty_metrics()
            return
            
        # Trade sonuçlarını pandas DataFrame'e çevir
        trades_df = pd.DataFrame(self.trades)
        
        # ENHANCED WIN RATE CALCULATION - Based on net_pnl (most accurate)
        profitable_trades = trades_df[trades_df["net_pnl"] > 0]
        break_even_trades = trades_df[trades_df["net_pnl"] == 0]
        losing_trades = trades_df[trades_df["net_pnl"] < 0]
        
        total_trades = len(trades_df)
        win_count = len(profitable_trades)
        loss_count = len(losing_trades)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        # ENHANCED PROFIT FACTOR
        gross_profit = profitable_trades["net_pnl"].sum() if len(profitable_trades) > 0 else 0
        gross_loss = abs(losing_trades["net_pnl"].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # ENHANCED DRAWDOWN CALCULATION
        drawdown_metrics = self._calculate_drawdown_metrics()
        
        # ENHANCED SHARPE RATIO
        sharpe_metrics = self._calculate_sharpe_metrics(trades_df)
        
        # DIRECTION PERFORMANCE
        direction_performance = self._calculate_direction_performance(trades_df)
        
        # ENHANCED: Multi-bar tracking metrics
        holding_metrics = self._calculate_holding_metrics(trades_df)
        
        # COMPILE ALL METRICS
        self.metrics = {
            # Core Performance
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "break_even_trades": len(break_even_trades),
            
            # Financial Metrics
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": gross_profit + gross_loss,
            "average_win": profitable_trades["net_pnl"].mean() if len(profitable_trades) > 0 else 0,
            "average_loss": losing_trades["net_pnl"].mean() if len(losing_trades) > 0 else 0,
            "largest_win": profitable_trades["net_pnl"].max() if len(profitable_trades) > 0 else 0,
            "largest_loss": losing_trades["net_pnl"].min() if len(losing_trades) > 0 else 0,
            
            # Risk Metrics
            **drawdown_metrics,
            **sharpe_metrics,
            
            # Performance by Direction
            "direction_performance": direction_performance,
            
            # ENHANCED: Multi-bar tracking metrics
            **holding_metrics,
            
            # Legacy Compatibility
            "max_drawdown": drawdown_metrics.get("max_drawdown", 0),
            "max_drawdown_pct": drawdown_metrics.get("max_drawdown_pct", 0),
            "sharpe_ratio": sharpe_metrics.get("sharpe_ratio", 0),
            "avg_gain_per_trade": trades_df["net_pnl"].mean(),
            "avg_rr_ratio": trades_df.get("rr_ratio", pd.Series([0])).mean()
        }

    def _calculate_holding_metrics(self, trades_df: pd.DataFrame) -> dict:
        """ENHANCED: Calculate metrics related to holding periods"""
        if "bars_held" not in trades_df.columns:
            return {}
        
        return {
            "avg_holding_bars": trades_df["bars_held"].mean(),
            "median_holding_bars": trades_df["bars_held"].median(),
            "max_holding_bars": trades_df["bars_held"].max(),
            "min_holding_bars": trades_df["bars_held"].min(),
            "max_holding_exits": (trades_df["outcome"] == "MAX_HOLDING").sum(),
            "max_holding_exit_rate": (trades_df["outcome"] == "MAX_HOLDING").mean() * 100,
            "timeout_exits": (trades_df["outcome"] == "TIMEOUT").sum(),
            "timeout_exit_rate": (trades_df["outcome"] == "TIMEOUT").mean() * 100
        }

    def _calculate_drawdown_metrics(self) -> dict:
        """ENHANCED: Calculate comprehensive drawdown metrics"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return {
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "drawdown_duration_max": 0,
                "recovery_factor": 0,
                "calmar_ratio": 0
            }
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df["running_max"] = equity_df["balance"].cummax()
        equity_df["drawdown"] = equity_df["running_max"] - equity_df["balance"]
        equity_df["drawdown_pct"] = equity_df["drawdown"] / equity_df["running_max"] * 100
        
        max_drawdown = equity_df["drawdown"].max()
        max_drawdown_pct = equity_df["drawdown_pct"].max()
        
        # Recovery factor
        total_return = equity_df["balance"].iloc[-1] - equity_df["balance"].iloc[0]
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Calmar ratio (simplified)
        annual_return = (equity_df["balance"].iloc[-1] / equity_df["balance"].iloc[0] - 1) * 100
        calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "recovery_factor": recovery_factor,
            "calmar_ratio": calmar_ratio,
            "avg_drawdown": equity_df["drawdown"].mean()
        }

    def _calculate_sharpe_metrics(self, trades_df: pd.DataFrame) -> dict:
        """ENHANCED: Calculate Sharpe and related risk-adjusted metrics"""
        if len(trades_df) < 2:
            return {"sharpe_ratio": 0, "sortino_ratio": 0}
        
        returns = trades_df["net_pnl"] / self.initial_balance
        
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Sortino Ratio (using downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else std_return
        sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "volatility": std_return,
            "downside_deviation": downside_std
        }

    def _calculate_direction_performance(self, trades_df: pd.DataFrame) -> dict:
        """ENHANCED: Calculate performance metrics by direction"""
        direction_stats = {}
        
        if "direction" not in trades_df.columns:
            return direction_stats
        
        for direction in trades_df["direction"].unique():
            direction_trades = trades_df[trades_df["direction"] == direction]
            
            profitable = direction_trades[direction_trades["net_pnl"] > 0]
            losing = direction_trades[direction_trades["net_pnl"] < 0]
            
            stats = {
                "count": len(direction_trades),
                "win_count": len(profitable),
                "loss_count": len(losing),
                "win_rate": (len(profitable) / len(direction_trades) * 100) if len(direction_trades) > 0 else 0,
                "total_pnl": direction_trades["net_pnl"].sum(),
                "avg_pnl": direction_trades["net_pnl"].mean(),
                "gross_profit": profitable["net_pnl"].sum() if len(profitable) > 0 else 0,
                "gross_loss": abs(losing["net_pnl"].sum()) if len(losing) > 0 else 0,
                "avg_holding_bars": direction_trades["bars_held"].mean() if "bars_held" in direction_trades.columns else 0
            }
            
            stats["profit_factor"] = stats["gross_profit"] / stats["gross_loss"] if stats["gross_loss"] > 0 else float('inf')
            direction_stats[direction] = stats
        
        return direction_stats

    def _get_empty_metrics(self) -> dict:
        """ENHANCED: Return empty metrics structure"""
        return {
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
            "avg_holding_bars": 0,
            "max_holding_exits": 0
        }
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of configured strategies"""
        return {
            "total_strategies": len(getattr(self.strategy_manager, '_strategies_to_use', [])),
            "strategy_weights": getattr(self.strategy_manager, '_strategy_weights', {}),
            "available_categories": self.strategy_manager.list_available_strategies(),
            "max_holding_bars": self.max_holding_bars
        }