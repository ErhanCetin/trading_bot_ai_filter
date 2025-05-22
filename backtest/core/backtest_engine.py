"""
Temel backtest motoru.
Signal Engine ile entegre çalışır ve ticaret stratejilerini değerlendirir.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os

# Signal Engine importları
from signal_engine.indicators import registry as indicator_registry
from signal_engine.signal_indicator_plugin_system import IndicatorManager
from signal_engine.strategies import registry as strategy_registry
from signal_engine.signal_strategy_system import StrategyManager
from signal_engine.strength import registry as strength_registry
from signal_engine.signal_strength_system import StrengthManager
from signal_engine.filters import registry as filter_registry
from signal_engine.signal_filter_system import FilterManager


class BacktestEngine:
    """Signal Engine ile entegre çalışan backtest motoru."""
    
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
        
        # Signal Engine bileşenleri
        self.indicator_manager = IndicatorManager(indicator_registry)
        self.strategy_manager = StrategyManager(strategy_registry)
        self.strength_manager = StrengthManager(strength_registry)
        self.filter_manager = FilterManager(filter_registry)
        
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
        Signal Engine bileşenlerini yapılandırır
        
        Args:
            indicators_config: İndikatör yapılandırması
            strategies_config: Strateji yapılandırması
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
        
        # Stratejileri yapılandır
        if strategies_config:
            for strategy_name, params in strategies_config.items():
                self.strategy_manager.add_strategy(strategy_name, params)
        
        # Sinyal gücü hesaplayıcılarını yapılandır
        if strength_config:
            for calculator_name, params in strength_config.items():
                self.strength_manager.add_calculator(calculator_name, params)
        
        # Filtreleri yapılandır
        if filter_config:
            for rule_name, params in filter_config.items():
                self.filter_manager.add_rule(rule_name, params)
            
            # Minimum kontrol ve güç gereksinimleri
            if "min_checks" in filter_config:
                self.filter_manager.set_min_checks_required(filter_config["min_checks"])
            if "min_strength" in filter_config:
                self.filter_manager.set_min_strength_required(filter_config["min_strength"])
    
    def run(self, df: pd.DataFrame, config_id: str = None) -> Dict[str, Any]:
        """
        Backtest çalıştırır
        
        Args:
            df: Fiyat verisi DataFrame
            config_id: Konfigürasyon ID'si
            
        Returns:
            Backtest sonuç özeti
        """
        # Başlangıç bakiyesi ile backtest için hazırla
        balance = self.initial_balance
        trades = []
        equity_curve = [{"time": df.iloc[0]["open_time"], "balance": balance}]
        
        # Signal Engine sürecini çalıştır
        # 1. İndikatörleri hesapla
        df = self.indicator_manager.calculate_indicators(df)
        print(f"🚀 🚀 🚀 İndikatörler hesaplandı: {df.columns.tolist()}")
       
       # from backtest.utils.print_calculated_indicator_data.print_calculated_indicator_list import debug_indicators
       # debug_indicators(df, output_type="csv", output_file="calculated_indicators.csv")
        
        # 2. Sinyalleri oluştur
        df = self.strategy_manager.generate_signals(df)
        print(f"🚀 🚀 🚀 Sinyaller oluşturuldu: {df.columns.tolist()}")

        # # Sinyaller için debug modülünü içe aktar
        # from backtest.utils.print_calculated_strategies_data.print_calculated_strategies_list import debug_signals
        # # Sinyalleri debug et
        # debug_signals(df, output_type="csv", output_file="calculated_signals.csv")
        
        # Yön filtrelemesi uygula
        if not self.position_direction.get("Long", True):
            df["long_signal"] = False
        if not self.position_direction.get("Short", True):
            df["short_signal"] = False
        
        # 3. Sinyal gücünü hesapla
        # Hesaplayıcı adlarını yöneticiden al
        calculator_names = getattr(self.strength_manager, '_calculators_to_use', [])
        print(f"🚀 🚀 🚀 Sinyal gücü hesaplayıcıları- calculator name: {calculator_names}")
        # Hesaplayıcı parametrelerini al
        calculator_params = getattr(self.strength_manager, '_calculator_params', {})
        print(f"🚀 🚀 🚀 Sinyal gücü hesaplayıcıları- calculator params: {calculator_params}")
        # Sinyal gücünü hesapla
        strength_series = self.strength_manager.calculate_strength(
            df, 
            df,  # signals_df olarak aynı df'i kullanıyoruz, çünkü long_signal ve short_signal sütunları burada 
            calculator_names,
            calculator_params
        )
        # Sonuçları DataFrame'e ekle
        df['signal_strength'] = strength_series
        #from backtest.utils.print_calculated_strength_data.print_calculated_strength_list import debug_strength_values
        #  # Sinyalleri debug et
        #debug_strength_values(df, output_type="csv", output_file="calculated_strengths.csv")
                
        # 4. Sinyalleri filtrele
        df = self.filter_manager.filter_signals(df)
        
        # İşlemleri simüle et
        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # Sinyal gücü kontrolü
            if row.get("signal_strength", 0) < 3:
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
            entry_price = row["close"]
            atr = row.get("atr", df["close"].pct_change().abs().mean() * entry_price)
            
            # Stop-loss ve take-profit seviyeleri
            sl = entry_price - atr * self.sl_multiplier if direction == "LONG" else entry_price + atr * self.sl_multiplier
            tp = entry_price + atr * self.tp_multiplier if direction == "LONG" else entry_price - atr * self.tp_multiplier
            
            # Next bar değerlerini kontrol et
            high = next_row["high"]
            low = next_row["low"]
            
            # İşlem sonucunu belirle
            if direction == "LONG":
                outcome = "TP" if high >= tp else "SL" if low <= sl else "OPEN"
            else:
                outcome = "TP" if low <= tp else "SL" if high >= sl else "OPEN"
            
            # Risk-reward oranı
            rr_ratio = self.tp_multiplier / self.sl_multiplier
            
            # Kazanç/kayıp hesaplama
            gain_pct = rr_ratio if outcome == "TP" else -1 if outcome == "SL" else 0
            position_size = balance * self.risk_per_trade * self.leverage
            
            # Komisyon hesapla
            commission = position_size * self.commission_rate
            
            # Net kazanç
            gain_usd = (gain_pct / 100) * position_size - commission
            balance += gain_usd
            
            # İşlemi kaydet
            trade = {
                "config_id": config_id,
                "time": row["open_time"],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": next_row["close"],
                "atr": atr,
                "sl": sl,
                "tp": tp,
                "rr_ratio": rr_ratio,
                "outcome": outcome,
                "gain_pct": gain_pct,
                "gain_usd": gain_usd,
                "commission": commission,
                "balance": balance,
                "position_size": position_size
            }
            
            # İndikatör değerlerini ekle
            for col in df.columns:
                if col.startswith(('rsi', 'macd', 'obv', 'adx', 'cci', 'supertrend')):
                    trade[col] = row.get(col)
            
            # Sinyal ve filtre metriklerini ekle
            trade["signal_strength"] = row.get("signal_strength", 0)
            trade["signal_passed_filter"] = row.get("signal_passed_filter", False)
            
            # İşlemi listeye ekle
            trades.append(trade)
            
            # Equity curve'e ekle
            equity_curve.append({"time": next_row["open_time"], "balance": balance})
        
        # Sonuçları sakla
        self.trades = trades
        self.equity_curve = equity_curve
        
        # Performans metriklerini hesapla
        self._calculate_performance_metrics()
        
        # Sonuç özeti
        return {
            "total_trades": len(trades),
            "final_balance": balance,
            "profit_loss": balance - self.initial_balance,
            "roi_pct": (balance / self.initial_balance - 1) * 100,
            "win_rate": self.metrics.get("win_rate", 0),
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": self.metrics
        }
    
    def _calculate_performance_metrics(self) -> None:
        """Performans metriklerini hesaplar ve saklar"""
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
                "avg_rr_ratio": 0
            }
            return
        if not self.trades:
            self.metrics = {}
            return
            
        # Trade sonuçlarını pandas DataFrame'e çevir
        trades_df = pd.DataFrame(self.trades)
        
        # Win rate
        closed_trades = trades_df[trades_df["outcome"].isin(["TP", "SL"])]
        win_count = (closed_trades["outcome"] == "TP").sum()
        total_closed = len(closed_trades)
        win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df["gain_usd"] > 0]["gain_usd"].sum()
        gross_loss = abs(trades_df[trades_df["gain_usd"] < 0]["gain_usd"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Drawdown hesaplama
        equity = pd.DataFrame(self.equity_curve)
        equity["drawdown"] = equity["balance"].cummax() - equity["balance"]
        equity["drawdown_pct"] = equity["drawdown"] / equity["balance"].cummax() * 100
        max_drawdown = equity["drawdown"].max()
        max_drawdown_pct = equity["drawdown_pct"].max()
        
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
            direction_win_rate = (direction_df["outcome"] == "TP").sum() / len(direction_df) * 100 if len(direction_df) > 0 else 0
            direction_stats[direction] = {
                "count": len(direction_df),
                "win_rate": direction_win_rate,
                "avg_gain": direction_df["gain_pct"].mean()
            }
        
        # Tüm metrikleri sakla
        self.metrics = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe,
            "total_trades": len(trades_df),
            "winning_trades": win_count,
            "losing_trades": total_closed - win_count,
            "direction_performance": direction_stats,
            "avg_gain_per_trade": trades_df["gain_pct"].mean(),
            "avg_rr_ratio": trades_df["rr_ratio"].mean()
        }