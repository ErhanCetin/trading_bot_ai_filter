"""
Performance Analyzer - Advanced Metrics Calculation
Comprehensive performance analysis for trading signals and indicators
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Advanced performance analysis for trading signals.
    Calculates comprehensive metrics for backtesting and signal quality assessment.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        logger.debug("🔧 PerformanceAnalyzer initialized")
    
    def analyze_signals_performance(self, signals_df: pd.DataFrame, price_data: pd.DataFrame, 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive performance analysis of trading signals.
        
        Args:
            signals_df: DataFrame with signals
            price_data: DataFrame with OHLCV data
            config: Test configuration
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        logger.info("📊 Starting comprehensive signal performance analysis...")
        
        if signals_df.empty:
            return self._get_empty_performance_metrics()
        
        try:
            # Execute signal-based trades
            trades = self._execute_signal_trades(signals_df, price_data, config)
            
            if not trades:
                return self._get_empty_performance_metrics()
            
            # Calculate all performance metrics
            performance = {
                'basic_metrics': self._calculate_basic_metrics(trades),
                'risk_metrics': self._calculate_risk_metrics(trades),
                'advanced_metrics': self._calculate_advanced_metrics(trades, price_data),
                'signal_quality': self._calculate_signal_quality_metrics(signals_df, trades),
                'trade_analysis': self._analyze_trade_patterns(trades),
                'drawdown_analysis': self._analyze_drawdowns(trades),
                'time_analysis': self._analyze_time_patterns(trades),
                'meta_information': self._get_meta_information(signals_df, trades, config)
            }
            
            # Calculate summary scores
            performance['summary_scores'] = self._calculate_summary_scores(performance)
            
            logger.info("✅ Performance analysis completed successfully")
            return performance
            
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
            return self._get_error_performance_metrics(str(e))
    
    def _execute_signal_trades(self, signals_df: pd.DataFrame, price_data: pd.DataFrame, 
                             config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute trades based on signals and return trade results."""
        trades = []
        current_position = None
        initial_capital = 10000  # $10,000 starting capital
        current_capital = initial_capital
        
        # Get exit conditions from config
        exit_conditions = config.get('signal_generation', {}).get('exit_conditions', {})
        stop_loss_pct = exit_conditions.get('stop_loss_pct', 3.0)
        take_profit_pct = exit_conditions.get('take_profit_pct', 6.0)
        max_hold_periods = exit_conditions.get('max_hold_periods', 100)
        
        # Sort signals by timestamp/index
        signals_sorted = signals_df.sort_values('index' if 'index' in signals_df.columns else signals_df.index)
        
        for _, signal in signals_sorted.iterrows():
            signal_time = signal.get('index', signal.name)
            signal_type = signal.get('signal_type', 'none')
            signal_price = signal.get('price', 0)
            
            if signal_type in ['buy', 'sell'] and signal_price > 0:
                # Close existing position if signal is opposite
                if current_position and current_position['type'] != signal_type:
                    trade = self._close_position(current_position, signal_price, signal_time, 'signal_change')
                    if trade:
                        trades.append(trade)
                        current_capital = trade['exit_capital']
                    current_position = None
                
                # Open new position
                if not current_position:
                    current_position = {
                        'type': signal_type,
                        'entry_time': signal_time,
                        'entry_price': signal_price,
                        'entry_capital': current_capital,
                        'quantity': current_capital / signal_price,
                        'stop_loss': signal_price * (1 - stop_loss_pct/100) if signal_type == 'buy' else signal_price * (1 + stop_loss_pct/100),
                        'take_profit': signal_price * (1 + take_profit_pct/100) if signal_type == 'buy' else signal_price * (1 - take_profit_pct/100),
                        'max_hold_time': signal_time + max_hold_periods
                    }
            
            # Check for exit conditions on existing position
            if current_position and signal_time > current_position['entry_time']:
                current_price = self._get_price_at_time(price_data, signal_time)
                
                if current_price > 0:
                    exit_reason = None
                    
                    # Check stop loss
                    if current_position['type'] == 'buy' and current_price <= current_position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif current_position['type'] == 'sell' and current_price >= current_position['stop_loss']:
                        exit_reason = 'stop_loss'
                    
                    # Check take profit
                    elif current_position['type'] == 'buy' and current_price >= current_position['take_profit']:
                        exit_reason = 'take_profit'
                    elif current_position['type'] == 'sell' and current_price <= current_position['take_profit']:
                        exit_reason = 'take_profit'
                    
                    # Check max hold time
                    elif signal_time >= current_position['max_hold_time']:
                        exit_reason = 'max_hold_time'
                    
                    # Close position if exit condition met
                    if exit_reason:
                        trade = self._close_position(current_position, current_price, signal_time, exit_reason)
                        if trade:
                            trades.append(trade)
                            current_capital = trade['exit_capital']
                        current_position = None
        
        # Close any remaining position at the end
        if current_position:
            final_price = self._get_price_at_time(price_data, len(price_data) - 1)
            if final_price > 0:
                trade = self._close_position(current_position, final_price, len(price_data) - 1, 'end_of_data')
                if trade:
                    trades.append(trade)
        
        logger.info(f"✅ Executed {len(trades)} trades from {len(signals_df)} signals")
        return trades
    
    def _close_position(self, position: Dict[str, Any], exit_price: float, 
                       exit_time: int, exit_reason: str) -> Optional[Dict[str, Any]]:
        """Close a position and calculate trade metrics."""
        try:
            if position['type'] == 'buy':
                exit_value = position['quantity'] * exit_price
                pnl = exit_value - position['entry_capital']
                pnl_pct = (exit_price / position['entry_price'] - 1) * 100
            else:  # sell/short
                # For short positions, profit when price goes down
                exit_value = position['entry_capital'] - (position['quantity'] * (exit_price - position['entry_price']))
                pnl = exit_value - position['entry_capital']
                pnl_pct = (position['entry_price'] / exit_price - 1) * 100
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': exit_time,
                'duration': exit_time - position['entry_time'],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'entry_capital': position['entry_capital'],
                'exit_capital': position['entry_capital'] + pnl,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'is_winner': pnl > 0
            }
            
            return trade
            
        except Exception as e:
            logger.warning(f"⚠️ Error closing position: {e}")
            return None
    
    def _get_price_at_time(self, price_data: pd.DataFrame, time_index: int) -> float:
        """Get price at specific time index."""
        try:
            if time_index < len(price_data):
                return float(price_data.iloc[time_index]['close'])
            else:
                return float(price_data.iloc[-1]['close'])  # Use last available price
        except:
            return 0.0
    
    def _calculate_basic_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        if not trades:
            return {'total_trades': 0}
        
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['is_winner'])
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL calculations
        total_pnl = sum(trade['pnl'] for trade in trades)
        total_pnl_pct = sum(trade['pnl_pct'] for trade in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_pnl_pct = total_pnl_pct / total_trades if total_trades > 0 else 0
        
        # Winner/loser analysis
        winning_pnl = sum(trade['pnl'] for trade in trades if trade['is_winner'])
        losing_pnl = sum(trade['pnl'] for trade in trades if not trade['is_winner'])
        avg_winner = winning_pnl / winning_trades if winning_trades > 0 else 0
        avg_loser = losing_pnl / losing_trades if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = abs(winning_pnl)
        gross_loss = abs(losing_pnl)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_pnl_pct, 2),
            'avg_trade_pnl': round(avg_pnl, 2),
            'avg_trade_return_pct': round(avg_pnl_pct, 2),
            'avg_winner': round(avg_winner, 2),
            'avg_loser': round(avg_loser, 2),
            'profit_factor': round(profit_factor, 3),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2)
        }
    
    def _calculate_risk_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics."""
        if not trades:
            return {}
        
        returns = [trade['pnl_pct'] for trade in trades]
        
        if not returns:
            return {}
        
        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe ratio (annualized)
        trading_periods_per_year = 252 * 24 * 12  # Assuming 5-minute intervals
        excess_return = mean_return - (self.risk_free_rate / trading_periods_per_year * 100)
        sharpe_ratio = excess_return / std_return if std_return > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns else 0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Other risk metrics
        max_loss = min(returns) if returns else 0
        max_gain = max(returns) if returns else 0
        
        return {
            'sharpe_ratio': round(sharpe_ratio, 3),
            'sortino_ratio': round(sortino_ratio, 3),
            'volatility': round(std_return, 3),
            'max_single_loss_pct': round(max_loss, 2),
            'max_single_gain_pct': round(max_gain, 2),
            'return_to_risk_ratio': round(mean_return / abs(max_loss), 3) if max_loss != 0 else 0
        }
    
    def _calculate_advanced_metrics(self, trades: List[Dict[str, Any]], 
                                  price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced performance metrics."""
        if not trades:
            return {}
        
        # Calculate equity curve
        equity_curve = [10000]  # Starting capital
        for trade in trades:
            equity_curve.append(trade['exit_capital'])
        
        # Maximum drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Recovery metrics
        recovery_factor = abs(equity_curve[-1] - equity_curve[0]) / max_drawdown if max_drawdown > 0 else 0
        
        # Calmar ratio
        total_return_pct = ((equity_curve[-1] / equity_curve[0]) - 1) * 100
        calmar_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        return {
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'recovery_factor': round(recovery_factor, 3),
            'calmar_ratio': round(calmar_ratio, 3),
            'final_capital': round(equity_curve[-1], 2),
            'total_return_pct': round(total_return_pct, 2)
        }
    
    def _calculate_signal_quality_metrics(self, signals_df: pd.DataFrame, 
                                        trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate signal quality and accuracy metrics."""
        total_signals = len(signals_df)
        total_trades = len(trades)
        
        # Signal conversion rate
        signal_to_trade_ratio = (total_trades / total_signals) * 100 if total_signals > 0 else 0
        
        # Signal accuracy (based on trades)
        successful_trades = sum(1 for trade in trades if trade['is_winner'])
        signal_accuracy = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Signal strength analysis
        if 'signal_strength' in signals_df.columns:
            avg_signal_strength = signals_df['signal_strength'].mean()
            signal_strength_std = signals_df['signal_strength'].std()
        else:
            avg_signal_strength = 0
            signal_strength_std = 0
        
        return {
            'total_signals_generated': total_signals,
            'signal_to_trade_conversion_rate': round(signal_to_trade_ratio, 2),
            'signal_accuracy_pct': round(signal_accuracy, 2),
            'avg_signal_strength': round(avg_signal_strength, 2),
            'signal_strength_consistency': round(signal_strength_std, 2)
        }
    
    def _analyze_trade_patterns(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trading patterns and behaviors."""
        if not trades:
            return {}
        
        # Duration analysis
        durations = [trade['duration'] for trade in trades]
        avg_duration = np.mean(durations)
        median_duration = np.median(durations)
        
        # Exit reason analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Trade type analysis
        buy_trades = [trade for trade in trades if trade['type'] == 'buy']
        sell_trades = [trade for trade in trades if trade['type'] == 'sell']
        
        buy_win_rate = (sum(1 for t in buy_trades if t['is_winner']) / len(buy_trades) * 100) if buy_trades else 0
        sell_win_rate = (sum(1 for t in sell_trades if t['is_winner']) / len(sell_trades) * 100) if sell_trades else 0
        
        # Consecutive analysis
        consecutive_wins = self._calculate_consecutive_streaks(trades, True)
        consecutive_losses = self._calculate_consecutive_streaks(trades, False)
        
        return {
            'avg_trade_duration': round(avg_duration, 1),
            'median_trade_duration': round(median_duration, 1),
            'exit_reason_breakdown': exit_reasons,
            'buy_trades_count': len(buy_trades),
            'sell_trades_count': len(sell_trades),
            'buy_win_rate_pct': round(buy_win_rate, 2),
            'sell_win_rate_pct': round(sell_win_rate, 2),
            'max_consecutive_wins': consecutive_wins['max_streak'],
            'max_consecutive_losses': consecutive_losses['max_streak'],
            'avg_consecutive_wins': round(consecutive_wins['avg_streak'], 1),
            'avg_consecutive_losses': round(consecutive_losses['avg_streak'], 1)
        }
    
    def _calculate_consecutive_streaks(self, trades: List[Dict[str, Any]], 
                                     is_winner: bool) -> Dict[str, Any]:
        """Calculate consecutive winning or losing streaks."""
        if not trades:
            return {'max_streak': 0, 'avg_streak': 0, 'streaks': []}
        
        streaks = []
        current_streak = 0
        
        for trade in trades:
            if trade['is_winner'] == is_winner:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        # Add final streak if it exists
        if current_streak > 0:
            streaks.append(current_streak)
        
        max_streak = max(streaks) if streaks else 0
        avg_streak = np.mean(streaks) if streaks else 0
        
        return {
            'max_streak': max_streak,
            'avg_streak': avg_streak,
            'streaks': streaks
        }
    
    def _analyze_drawdowns(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze drawdown patterns and recovery."""
        if not trades:
            return {}
        
        # Build equity curve
        equity_curve = [10000]
        for trade in trades:
            equity_curve.append(trade['exit_capital'])
        
        # Find all drawdown periods
        drawdowns = []
        peak = equity_curve[0]
        drawdown_start = 0
        in_drawdown = False
        
        for i, equity in enumerate(equity_curve):
            if equity > peak:
                # New peak reached
                if in_drawdown:
                    # End of drawdown period
                    drawdown_depth = ((peak - min(equity_curve[drawdown_start:i])) / peak) * 100
                    recovery_periods = i - drawdown_start
                    drawdowns.append({
                        'depth_pct': drawdown_depth,
                        'recovery_periods': recovery_periods,
                        'start_index': drawdown_start,
                        'end_index': i
                    })
                    in_drawdown = False
                
                peak = equity
            elif equity < peak and not in_drawdown:
                # Start of new drawdown
                drawdown_start = i
                in_drawdown = True
        
        # Handle ongoing drawdown at end
        if in_drawdown:
            drawdown_depth = ((peak - min(equity_curve[drawdown_start:])) / peak) * 100
            recovery_periods = len(equity_curve) - drawdown_start
            drawdowns.append({
                'depth_pct': drawdown_depth,
                'recovery_periods': recovery_periods,
                'start_index': drawdown_start,
                'end_index': len(equity_curve) - 1,
                'ongoing': True
            })
        
        if not drawdowns:
            return {'drawdown_periods': 0}
        
        # Analyze drawdown statistics
        drawdown_depths = [dd['depth_pct'] for dd in drawdowns]
        recovery_times = [dd['recovery_periods'] for dd in drawdowns if not dd.get('ongoing', False)]
        
        return {
            'drawdown_periods': len(drawdowns),
            'avg_drawdown_depth_pct': round(np.mean(drawdown_depths), 2),
            'max_drawdown_depth_pct': round(max(drawdown_depths), 2),
            'avg_recovery_periods': round(np.mean(recovery_times), 1) if recovery_times else 0,
            'max_recovery_periods': max(recovery_times) if recovery_times else 0,
            'ongoing_drawdown': any(dd.get('ongoing', False) for dd in drawdowns)
        }
    
    def _analyze_time_patterns(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time-based trading patterns."""
        if not trades:
            return {}
        
        # Monthly performance (if we have enough data)
        monthly_pnl = {}
        for trade in trades:
            # Simplified monthly grouping (using entry time as proxy)
            month_key = f"month_{(trade['entry_time'] // 43200) % 12 + 1}"  # Rough approximation
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = []
            monthly_pnl[month_key].append(trade['pnl_pct'])
        
        # Calculate monthly stats
        monthly_stats = {}
        for month, pnls in monthly_pnl.items():
            monthly_stats[month] = {
                'trades': len(pnls),
                'avg_return': round(np.mean(pnls), 2),
                'win_rate': round((sum(1 for p in pnls if p > 0) / len(pnls) * 100), 2)
            }
        
        return {
            'time_period_analysis': monthly_stats,
            'total_trading_periods': len(set(trade['entry_time'] for trade in trades))
        }
    
    def _get_meta_information(self, signals_df: pd.DataFrame, trades: List[Dict[str, Any]], 
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Get meta information about the analysis."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'config_name': config.get('test_name', 'unknown'),
            'symbol': config.get('symbol', 'unknown'),
            'interval': config.get('interval', 'unknown'),
            'indicator_name': config.get('indicator_name', 'unknown'),
            'test_duration_days': config.get('test_settings', {}).get('test_duration_days', 7),
            'total_data_points': len(signals_df) if not signals_df.empty else 0,
            'analysis_version': '1.0.0'
        }
    
    def _calculate_summary_scores(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary scores for the strategy."""
        basic = performance.get('basic_metrics', {})
        risk = performance.get('risk_metrics', {})
        advanced = performance.get('advanced_metrics', {})
        signal_quality = performance.get('signal_quality', {})
        
        # Accuracy score (0-100)
        win_rate = basic.get('win_rate', 0)
        signal_accuracy = signal_quality.get('signal_accuracy_pct', 0)
        accuracy_score = (win_rate + signal_accuracy) / 2
        
        # Risk-adjusted score (0-100)
        sharpe = risk.get('sharpe_ratio', 0)
        max_dd = advanced.get('max_drawdown_pct', 100)
        risk_score = max(0, min(100, (sharpe * 20) - (max_dd / 2)))
        
        # Profitability score (0-100)
        profit_factor = basic.get('profit_factor', 0)
        total_return = advanced.get('total_return_pct', 0)
        profit_score = min(100, (profit_factor * 20) + max(0, total_return))
        
        # Overall score (weighted average)
        overall_score = (accuracy_score * 0.3 + risk_score * 0.4 + profit_score * 0.3)
        
        return {
            'accuracy_score': round(accuracy_score, 1),
            'risk_adjusted_score': round(risk_score, 1),
            'profitability_score': round(profit_score, 1),
            'overall_score': round(overall_score, 1),
            'grade': self._get_performance_grade(overall_score)
        }
    
    def _get_performance_grade(self, score: float) -> str:
        """Get letter grade based on overall score."""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def _get_empty_performance_metrics(self) -> Dict[str, Any]:
        """Get default performance metrics for empty results."""
        return {
            'basic_metrics': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'profit_factor': 0.0
            },
            'risk_metrics': {
                'sharpe_ratio': 0.0,
                'volatility': 0.0
            },
            'advanced_metrics': {
                'max_drawdown_pct': 0.0,
                'total_return_pct': 0.0
            },
            'signal_quality': {
                'total_signals_generated': 0,
                'signal_accuracy_pct': 0.0
            },
            'summary_scores': {
                'overall_score': 0.0,
                'grade': 'F'
            },
            'meta_information': {
                'analysis_timestamp': datetime.now().isoformat(),
                'status': 'no_signals'
            }
        }
    
    def _get_error_performance_metrics(self, error_message: str) -> Dict[str, Any]:
        """Get error performance metrics."""
        empty_metrics = self._get_empty_performance_metrics()
        empty_metrics['meta_information']['status'] = 'error'
        empty_metrics['meta_information']['error_message'] = error_message
        return empty_metrics
    
    def create_performance_summary(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concise performance summary for reporting."""
        basic = performance.get('basic_metrics', {})
        risk = performance.get('risk_metrics', {})
        advanced = performance.get('advanced_metrics', {})
        scores = performance.get('summary_scores', {})
        meta = performance.get('meta_information', {})
        
        return {
            'test_info': {
                'config_name': meta.get('config_name', 'unknown'),
                'symbol': meta.get('symbol', 'unknown'),
                'interval': meta.get('interval', 'unknown'),
                'indicator_name': meta.get('indicator_name', 'unknown')
            },
            'key_metrics': {
                'total_trades': basic.get('total_trades', 0),
                'win_rate_pct': basic.get('win_rate', 0),
                'profit_factor': basic.get('profit_factor', 0),
                'sharpe_ratio': risk.get('sharpe_ratio', 0),
                'max_drawdown_pct': advanced.get('max_drawdown_pct', 0),
                'total_return_pct': advanced.get('total_return_pct', 0)
            },
            'scores': {
                'overall_score': scores.get('overall_score', 0),
                'grade': scores.get('grade', 'F'),
                'accuracy_score': scores.get('accuracy_score', 0),
                'risk_score': scores.get('risk_adjusted_score', 0),
                'profit_score': scores.get('profitability_score', 0)
            },
            'analysis_date': meta.get('analysis_timestamp', datetime.now().isoformat())
        }


# Convenience function for easy use
def analyze_performance(signals_df: pd.DataFrame, price_data: pd.DataFrame, 
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for performance analysis.
    
    Args:
        signals_df: DataFrame with signals
        price_data: DataFrame with OHLCV data  
        config: Test configuration
        
    Returns:
        Comprehensive performance analysis
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze_signals_performance(signals_df, price_data, config)


if __name__ == "__main__":
    """Test the performance analyzer."""
    
    print("🧪 Testing PerformanceAnalyzer...")
    
    # Create sample data
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Sample price data
    start_time = datetime.now() - timedelta(days=7)
    time_range = pd.date_range(start=start_time, periods=1000, freq='5min')
    
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0, 0.01, len(time_range))
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    price_data = pd.DataFrame({
        'open_time': time_range,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(prices))
    })
    
    # Sample signals
    signal_indices = np.random.choice(len(price_data), size=20, replace=False)
    signals_data = []
    
    for i, idx in enumerate(signal_indices):
        signals_data.append({
            'index': idx,
            'signal_type': 'buy' if i % 2 == 0 else 'sell',
            'price': float(price_data.iloc[idx]['close']),
            'signal_strength': np.random.uniform(50, 100)
        })
    
    signals_df = pd.DataFrame(signals_data)
    
    # Sample config
    config = {
        'test_name': 'Test RSI Strategy',
        'symbol': 'ETHUSDT',
        'interval': '5m',
        'indicator_name': 'rsi',
        'signal_generation': {
            'exit_conditions': {
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'max_hold_periods': 50
            }
        },
        'test_settings': {
            'test_duration_days': 7
        }
    }
    
    # Test performance analysis
    try:
        print("📊 Running performance analysis...")
        analyzer = PerformanceAnalyzer()
        
        performance = analyzer.analyze_signals_performance(signals_df, price_data, config)
        
        print("✅ Performance analysis completed!")
        print(f"📈 Results summary:")
        print(f"   Total trades: {performance['basic_metrics']['total_trades']}")
        print(f"   Win rate: {performance['basic_metrics']['win_rate']}%")
        print(f"   Profit factor: {performance['basic_metrics']['profit_factor']}")
        print(f"   Sharpe ratio: {performance['risk_metrics']['sharpe_ratio']}")
        print(f"   Max drawdown: {performance['advanced_metrics']['max_drawdown_pct']}%")
        print(f"   Overall score: {performance['summary_scores']['overall_score']} ({performance['summary_scores']['grade']})")
        
        # Test summary creation
        print("\n📋 Testing performance summary...")
        summary = analyzer.create_performance_summary(performance)
        print(f"✅ Summary created for {summary['test_info']['config_name']}")
        
        # Test convenience function
        print("\n🔧 Testing convenience function...")
        performance2 = analyze_performance(signals_df, price_data, config)
        print(f"✅ Convenience function works: {performance2['summary_scores']['grade']} grade")
        
    except Exception as e:
        print(f"❌ Performance analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 PerformanceAnalyzer test completed!")