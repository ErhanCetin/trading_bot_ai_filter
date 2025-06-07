"""
Data Processing Module for Testing Framework
Handles data enhancement, metadata addition, and summary statistics
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing and enhancement for testing framework.
    Adds metadata, calculates summary statistics, and prepares data for testing.
    """
    
    def __init__(self):
        """Initialize data processor."""
        self.processing_timestamp = None
        logger.debug("🔧 DataProcessor initialized")
    
    def process_data(self, df: pd.DataFrame, symbol: str, interval: str, 
                    quality_score: float) -> pd.DataFrame:
        """
        Process and enhance data with metadata and derived features.
        
        Args:
            df: Raw DataFrame to process
            symbol: Trading symbol
            interval: Time interval
            quality_score: Data quality score from validation
            
        Returns:
            Enhanced DataFrame with metadata and features
        """
        logger.info(f"🔄 Processing data for {symbol} {interval}")
        
        self.processing_timestamp = datetime.now()
        result_df = df.copy()
        
        try:
            # Add basic metadata
            result_df = self._add_basic_metadata(result_df, symbol, interval, quality_score)
            
            # Add market characteristics
            result_df = self._add_market_characteristics(result_df)
            
            # Add time-based features
            result_df = self._add_time_features(result_df)
            
            # Add technical features
            result_df = self._add_technical_features(result_df)
            
            # Clean and optimize data types
            result_df = self._optimize_data_types(result_df)
            
            logger.info(f"✅ Data processing completed: {len(result_df)} rows, {len(result_df.columns)} columns")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            raise
    
    def _add_basic_metadata(self, df: pd.DataFrame, symbol: str, interval: str, 
                          quality_score: float) -> pd.DataFrame:
        """Add basic metadata columns."""
        
        # Core metadata
        df['symbol'] = symbol
        df['interval'] = interval
        df['data_quality_score'] = quality_score
        df['processing_timestamp'] = self.processing_timestamp
        
        # Data source information
        df['data_source'] = 'binance'
        df['fetch_method'] = 'postgresql'
        
        logger.debug("📋 Added basic metadata")
        return df
    
    def _add_market_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market characteristic indicators."""
        
        # Price volatility (rolling standard deviation)
        df['price_volatility_50'] = df['close'].rolling(window=50, min_periods=10).std() / df['close'].rolling(window=50, min_periods=10).mean()
        df['price_volatility_20'] = df['close'].rolling(window=20, min_periods=5).std() / df['close'].rolling(window=20, min_periods=5).mean()
        
        # Volume characteristics
        df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=5).mean()
        df['volume_volatility'] = df['volume'].rolling(window=50, min_periods=10).std() / df['volume'].rolling(window=50, min_periods=10).mean()
        
        # Price range characteristics
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        df['daily_range_ma'] = df['daily_range'].rolling(window=20, min_periods=5).mean()
        
        # True Range (for ATR calculation later)
        df['prev_close'] = df['close'].shift(1)
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['prev_close']),
                np.abs(df['low'] - df['prev_close'])
            )
        )
        
        # Basic trend indicators
        df['price_change'] = df['close'].pct_change()
        df['price_trend_5'] = np.where(
            df['close'] > df['close'].shift(5), 1,
            np.where(df['close'] < df['close'].shift(5), -1, 0)
        )
        
        logger.debug("📈 Added market characteristics")
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for analysis."""
        
        if 'open_time' not in df.columns:
            logger.warning("⚠️ No open_time column found, skipping time features")
            return df
        
        # Ensure open_time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
            df['open_time'] = pd.to_datetime(df['open_time'])
        
        # Basic time features
        df['hour'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['open_time'].dt.day
        df['month'] = df['open_time'].dt.month
        
        # Trading session indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
        df['is_asian_session'] = df['hour'].between(0, 8)  # UTC 0-8 roughly Asian
        df['is_european_session'] = df['hour'].between(8, 16)  # UTC 8-16 roughly European
        df['is_american_session'] = df['hour'].between(16, 24)  # UTC 16-24 roughly American
        
        # Market opening/closing times (approximate)
        df['is_market_open_hours'] = df['hour'].between(8, 22)  # General trading hours
        df['is_low_liquidity_hours'] = df['hour'].between(22, 2)  # Low liquidity period
        
        # Time-based volatility patterns
        df['hour_volatility_rank'] = df.groupby('hour')['price_volatility_20'].rank(pct=True)
        
        logger.debug("🕐 Added time-based features")
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical analysis features."""
        
        # Simple moving averages for reference
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        
        # Price position relative to moving averages
        df['close_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        df['close_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50'] * 100
        
        # Basic momentum indicators
        df['momentum_5'] = df['close'].pct_change(periods=5) * 100
        df['momentum_20'] = df['close'].pct_change(periods=20) * 100
        
        # Volume-price relationship
        df['volume_price_trend'] = np.where(
            (df['price_change'] > 0) & (df['volume'] > df['volume_ma_20']), 1,  # Price up, volume up
            np.where(
                (df['price_change'] < 0) & (df['volume'] > df['volume_ma_20']), -1,  # Price down, volume up
                0  # Other cases
            )
        )
        
        # Basic support/resistance levels (simplified)
        df['resistance_20'] = df['high'].rolling(window=20, min_periods=1).max()
        df['support_20'] = df['low'].rolling(window=20, min_periods=1).min()
        df['resistance_distance'] = (df['resistance_20'] - df['close']) / df['close'] * 100
        df['support_distance'] = (df['close'] - df['support_20']) / df['close'] * 100
        
        logger.debug("📊 Added technical features")
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        
        # Convert boolean columns
        boolean_columns = [col for col in df.columns if col.startswith('is_')]
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype('bool')
        
        # Convert integer columns
        integer_columns = ['hour', 'day_of_week', 'day_of_month', 'month', 'price_trend_5', 'volume_price_trend']
        for col in integer_columns:
            if col in df.columns:
                df[col] = df[col].astype('int8')
        
        # Convert float32 for memory efficiency (most price/volume data doesn't need float64 precision)
        float_columns = [col for col in df.columns if df[col].dtype == 'float64']
        exclude_from_float32 = ['open_time', 'close_time', 'processing_timestamp']  # Keep datetime precision
        
        for col in float_columns:
            if col not in exclude_from_float32:
                # Check if values are within float32 range
                if df[col].notna().any():
                    max_val = df[col].max()
                    min_val = df[col].min()
                    if pd.isna(max_val) or pd.isna(min_val):
                        continue
                    
                    # Float32 range check
                    if abs(max_val) < 3.4e38 and abs(min_val) < 3.4e38:
                        df[col] = df[col].astype('float32')
        
        logger.debug("🗜️ Optimized data types")
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for the processed data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            summary = {
                'data_info': {
                    'symbol': df['symbol'].iloc[0] if 'symbol' in df.columns else 'Unknown',
                    'interval': df['interval'].iloc[0] if 'interval' in df.columns else 'Unknown',
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'date_range': {
                        'start': df['open_time'].min().isoformat() if 'open_time' in df.columns else None,
                        'end': df['open_time'].max().isoformat() if 'open_time' in df.columns else None,
                        'duration_days': (df['open_time'].max() - df['open_time'].min()).days if 'open_time' in df.columns else 0
                    },
                    'data_quality_score': df['data_quality_score'].iloc[0] if 'data_quality_score' in df.columns else 0,
                    'processing_timestamp': self.processing_timestamp.isoformat() if self.processing_timestamp else None
                },
                'price_statistics': self._get_price_statistics(df),
                'volume_statistics': self._get_volume_statistics(df),
                'volatility_statistics': self._get_volatility_statistics(df),
                'time_statistics': self._get_time_statistics(df),
                'data_completeness': self._get_completeness_statistics(df)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating summary statistics: {e}")
            return {'error': str(e)}
    
    def _get_price_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get price-related statistics."""
        stats = {}
        
        if 'close' in df.columns:
            stats['price_range'] = {
                'min': float(df['low'].min()) if 'low' in df.columns else float(df['close'].min()),
                'max': float(df['high'].max()) if 'high' in df.columns else float(df['close'].max()),
                'current': float(df['close'].iloc[-1]),
                'avg': float(df['close'].mean()),
                'median': float(df['close'].median())
            }
            
            if 'price_change' in df.columns:
                price_changes = df['price_change'].dropna()
                stats['volatility'] = {
                    'daily_volatility_pct': float(price_changes.std() * np.sqrt(288) * 100),  # 5min -> daily
                    'max_single_move_pct': float(price_changes.abs().max() * 100),
                    'avg_absolute_move_pct': float(price_changes.abs().mean() * 100)
                }
        
        return stats
    
    def _get_volume_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get volume-related statistics."""
        stats = {}
        
        if 'volume' in df.columns:
            volume_data = df['volume'].dropna()
            stats = {
                'avg_volume': float(volume_data.mean()),
                'median_volume': float(volume_data.median()),
                'max_volume': float(volume_data.max()),
                'min_volume': float(volume_data.min()),
                'volume_volatility': float(volume_data.std() / volume_data.mean()) if volume_data.mean() > 0 else 0,
                'zero_volume_count': int((volume_data == 0).sum()),
                'zero_volume_pct': float((volume_data == 0).mean() * 100)
            }
        
        return stats
    
    def _get_volatility_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get volatility-related statistics."""
        stats = {}
        
        if 'price_volatility_20' in df.columns:
            vol_data = df['price_volatility_20'].dropna()
            if len(vol_data) > 0:
                stats['price_volatility'] = {
                    'avg_volatility': float(vol_data.mean()),
                    'max_volatility': float(vol_data.max()),
                    'min_volatility': float(vol_data.min()),
                    'volatility_trend': float(vol_data.iloc[-20:].mean() - vol_data.iloc[:20].mean()) if len(vol_data) >= 40 else 0
                }
        
        if 'daily_range' in df.columns:
            range_data = df['daily_range'].dropna()
            if len(range_data) > 0:
                stats['daily_range'] = {
                    'avg_range_pct': float(range_data.mean() * 100),
                    'max_range_pct': float(range_data.max() * 100),
                    'min_range_pct': float(range_data.min() * 100)
                }
        
        return stats
    
    def _get_time_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get time-based statistics."""
        stats = {}
        
        if 'hour' in df.columns:
            stats['time_distribution'] = {
                'most_active_hour': int(df['hour'].mode().iloc[0]) if len(df['hour'].mode()) > 0 else 0,
                'weekend_data_pct': float(df['is_weekend'].mean() * 100) if 'is_weekend' in df.columns else 0,
                'low_liquidity_hours_pct': float(df['is_low_liquidity_hours'].mean() * 100) if 'is_low_liquidity_hours' in df.columns else 0
            }
        
        return stats
    
    def _get_completeness_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get data completeness statistics."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        
        stats = {
            'missing_data_pct': float((missing_cells / total_cells) * 100) if total_cells > 0 else 0,
            'complete_rows': int((~df.isnull().any(axis=1)).sum()),
            'total_rows': len(df),
            'complete_rows_pct': float((~df.isnull().any(axis=1)).mean() * 100),
            'columns_with_missing_data': int(df.isnull().any().sum()),
            'total_columns': len(df.columns)
        }
        
        return stats


if __name__ == "__main__":
    """Test the data processor module."""
    
    print("🧪 Testing DataProcessor...")
    
    # Create sample data
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate test data
    start_time = datetime.now() - timedelta(days=7)
    time_range = pd.date_range(start=start_time, periods=1000, freq='5min')
    
    # Create realistic OHLCV data
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0, 0.01, len(time_range))
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    test_df = pd.DataFrame({
        'open_time': time_range,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, len(prices))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(prices))),
        'close': prices * (1 + np.random.normal(0, 0.005, len(prices))),
        'volume': np.random.uniform(1000, 10000, len(prices))
    })
    
    # Ensure OHLC consistency
    test_df['high'] = test_df[['open', 'high', 'close']].max(axis=1)
    test_df['low'] = test_df[['open', 'low', 'close']].min(axis=1)
    
    # Test processor
    try:
        processor = DataProcessor()
        
        # Process data
        print("🔄 Processing test data...")
        processed_df = processor.process_data(test_df, "ETHUSDT", "5m", 85.5)
        
        print(f"✅ Data processing completed:")
        print(f"   📊 Rows: {len(processed_df)}")
        print(f"   📋 Columns: {len(processed_df.columns)}")
        print(f"   🏷️ Added metadata columns: {len(processed_df.columns) - len(test_df.columns)}")
        
        # Generate summary
        print("📈 Generating summary statistics...")
        summary = processor.generate_summary_statistics(processed_df)
        
        print(f"✅ Summary generated:")
        print(f"   🏆 Data quality score: {summary['data_info']['data_quality_score']}")
        print(f"   💰 Price range: ${summary['price_statistics']['price_range']['min']:.2f} - ${summary['price_statistics']['price_range']['max']:.2f}")
        print(f"   📈 Daily volatility: {summary['price_statistics']['volatility']['daily_volatility_pct']:.2f}%")
        print(f"   📊 Complete rows: {summary['data_completeness']['complete_rows_pct']:.1f}%")
        
        # Test specific features
        print("🔍 Testing specific features...")
        
        # Check metadata
        metadata_cols = ['symbol', 'interval', 'data_quality_score', 'processing_timestamp']
        for col in metadata_cols:
            if col in processed_df.columns:
                print(f"   ✅ {col}: Present")
            else:
                print(f"   ❌ {col}: Missing")
        
        # Check time features
        time_features = ['hour', 'day_of_week', 'is_weekend', 'is_market_open_hours']
        for col in time_features:
            if col in processed_df.columns:
                print(f"   ✅ {col}: Present")
            else:
                print(f"   ❌ {col}: Missing")
        
        # Check technical features
        tech_features = ['sma_20', 'price_volatility_20', 'momentum_5', 'volume_price_trend']
        for col in tech_features:
            if col in processed_df.columns:
                print(f"   ✅ {col}: Present")
            else:
                print(f"   ❌ {col}: Missing")
        
    except Exception as e:
        print(f"❌ DataProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎉 DataProcessor test completed!")