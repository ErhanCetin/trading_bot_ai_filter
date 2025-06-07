"""
Data Fetching Module for Testing Framework
Handles Binance data fetching and database loading operations
"""
import sys
import os
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy import create_engine, text

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import existing functionality
from data.binance_fetch_and_store_historical import fetch_and_store_for_config
from db.postgresql import engine  # Use existing engine
from env_loader import load_environment

# Configure logging
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Handles data fetching from Binance and loading from PostgreSQL database.
    Integrates with existing data infrastructure.
    """
    
    def __init__(self):
        """Initialize data fetcher with database connection."""
        # Load environment
        load_environment()
        
        # Use existing database engine
        self.engine = engine
        
        # Connection settings
        self.connection_timeout = 30
        self.query_timeout = 60
        self.retry_attempts = 3
        
        logger.debug("🔧 DataFetcher initialized")
    
    def fetch_fresh_data(self, symbol: str, interval: str) -> Tuple[int, list]:
        """
        Fetch fresh data from Binance and store in database.
        Uses existing fetch_and_store_for_config method.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            interval: Time interval (e.g., '5m', '1h')
            
        Returns:
            Tuple of (success_count, errors_list)
        """
        logger.info(f"🔄 Fetching fresh data from Binance: {symbol} {interval}")
        
        try:
            # Use existing method (fetches 7 days of data hardcoded)
            success_count, errors = fetch_and_store_for_config(symbol, interval)
            
            if errors:
                logger.warning(f"⚠️ Data fetch completed with {len(errors)} errors")
                for error in errors:
                    logger.warning(f"   • {error}")
            else:
                logger.info(f"✅ Data fetch successful ({success_count}/4 data types)")
            
            # Wait for database commit
            time.sleep(2)
            
            return success_count, errors
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch data from Binance: {e}")
            return 0, [str(e)]
    
    def load_data_from_database(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Load OHLCV data from PostgreSQL database.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If no data found or insufficient data
            Exception: If database query fails
        """
        logger.debug(f"📥 Loading data from database: {symbol} {interval}")
        
        try:
            # SQL query to get OHLCV data for last 8 days (to ensure 7+ days)
            # Fixed: Use text() wrapper for SQLAlchemy and proper timestamp handling
            query = text("""
            SELECT 
                open_time,
                open,
                high,
                low,
                close,
                volume,
                close_time,
                quote_asset_volume,
                number_of_trades,
                taker_buy_base_volume,
                taker_buy_quote_volume
            FROM kline_data 
            WHERE symbol = :symbol AND interval = :interval
            AND open_time >= EXTRACT(EPOCH FROM (NOW() - INTERVAL '8 days')) * 1000
            ORDER BY open_time ASC
            """)
            
            # Execute query with proper parameter format
            df = self._execute_query_with_retry(query, {"symbol": symbol, "interval": interval})
            
            if df.empty:
                raise ValueError(f"No data found for {symbol} {interval} in database")
            
            # Process and clean the data
            df = self._process_raw_data(df)
            
            logger.debug(f"✅ Loaded {len(df)} rows from database")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from database: {e}")
            raise
    
    def _execute_query_with_retry(self, query, params: dict) -> pd.DataFrame:
        """
        Execute database query with retry logic.
        
        Args:
            query: SQL query (text() wrapped or string)
            params: Query parameters as dictionary
            
        Returns:
            DataFrame with query results
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                # Load data using pandas with timeout - use dictionary params
                df = pd.read_sql_query(
                    query, 
                    self.engine, 
                    params=params
                )
                return df
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Database query attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"🔄 Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ All {self.retry_attempts} database query attempts failed")
        
        raise Exception(f"Database query failed after {self.retry_attempts} attempts: {last_error}")
    
    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean raw data from database.
        
        Args:
            df: Raw DataFrame from database
            
        Returns:
            Processed DataFrame
        """
        # Convert timestamp columns to datetime (assuming they are in milliseconds)
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Ensure numeric columns are proper types
        numeric_columns = [
            'open', 'high', 'low', 'close', 'volume', 
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by time to ensure chronological order
        if 'open_time' in df.columns:
            df = df.sort_values('open_time').reset_index(drop=True)
        
        return df
    
    def check_data_freshness(self, symbol: str, interval: str, 
                           min_data_points: int) -> bool:
        """
        Check if we have sufficient recent data in the database.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            min_data_points: Minimum required data points
            
        Returns:
            True if sufficient recent data exists, False otherwise
        """
        try:
            # Query to check data availability and freshness
            # Fixed: Use text() wrapper for SQLAlchemy
            query = text("""
            SELECT 
                COUNT(*) as row_count, 
                MAX(open_time) as latest_time,
                MIN(open_time) as earliest_time
            FROM kline_data 
            WHERE symbol = :symbol AND interval = :interval
            AND open_time >= EXTRACT(EPOCH FROM (NOW() - INTERVAL '8 days')) * 1000
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol, "interval": interval}).fetchone()
                
                if result and result[0] >= min_data_points:
                    latest_time_timestamp = result[1]
                    
                    if latest_time_timestamp:
                        # Convert timestamp to datetime (assuming milliseconds)
                        from datetime import datetime
                        latest_time = datetime.fromtimestamp(latest_time_timestamp / 1000)
                        time_diff = datetime.now() - latest_time
                        hours_old = time_diff.total_seconds() / 3600
                        
                        if hours_old <= 2:  # Data is recent enough
                            logger.debug(f"✅ Fresh data available: {result[0]} rows, {hours_old:.1f}h old")
                            return True
                        else:
                            logger.debug(f"📅 Data exists but stale: {hours_old:.1f}h old")
                            return False
                
                logger.debug(f"📊 Insufficient data: {result[0] if result else 0} < {min_data_points} required")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking data freshness: {e}")
            return False
    
    def get_data_summary(self, symbol: str, interval: str) -> dict:
        """
        Get summary information about available data.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            Dictionary with data summary
        """
        try:
            # Fixed: Use text() wrapper for SQLAlchemy
            query = text("""
            SELECT 
                COUNT(*) as total_rows,
                MIN(open_time) as earliest_time,
                MAX(open_time) as latest_time,
                AVG(volume) as avg_volume,
                MIN(low) as min_price,
                MAX(high) as max_price
            FROM kline_data 
            WHERE symbol = :symbol AND interval = :interval
            AND open_time >= EXTRACT(EPOCH FROM (NOW() - INTERVAL '8 days')) * 1000
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol, "interval": interval}).fetchone()
                
                if result:
                    # Convert timestamps to datetime if they exist
                    earliest_time = None
                    latest_time = None
                    
                    if result[1]:  # earliest_time
                        from datetime import datetime
                        earliest_time = datetime.fromtimestamp(result[1] / 1000)
                    
                    if result[2]:  # latest_time  
                        from datetime import datetime
                        latest_time = datetime.fromtimestamp(result[2] / 1000)
                    
                    return {
                        'total_rows': result[0],
                        'earliest_time': earliest_time,
                        'latest_time': latest_time,
                        'avg_volume': float(result[3]) if result[3] else 0,
                        'price_range': {
                            'min': float(result[4]) if result[4] else 0,
                            'max': float(result[5]) if result[5] else 0
                        }
                    }
                
                return {'total_rows': 0}
                
        except Exception as e:
            logger.error(f"❌ Error getting data summary: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    """Test the data fetcher module."""
    
    print("🧪 Testing DataFetcher...")
    
    fetcher = DataFetcher()
    
    # Test data fetching
    test_symbol = "ETHUSDT"
    test_interval = "5m"
    
    try:
        # Check data freshness
        print(f"🔍 Checking data freshness for {test_symbol} {test_interval}...")
        is_fresh = fetcher.check_data_freshness(test_symbol, test_interval, 100)
        print(f"📅 Data freshness: {'✅ Fresh' if is_fresh else '❌ Stale or missing'}")
        
        # Get data summary
        print(f"📊 Getting data summary...")
        summary = fetcher.get_data_summary(test_symbol, test_interval)
        print(f"📈 Data summary: {summary}")
        
        # Load data from database
        print(f"📥 Loading data from database...")
        df = fetcher.load_data_from_database(test_symbol, test_interval)
        print(f"✅ Loaded {len(df)} rows")
        print(f"📅 Date range: {df['open_time'].min()} to {df['open_time'].max()}")
        
    except Exception as e:
        print(f"❌ DataFetcher test failed: {e}")
    
    print("🎉 DataFetcher test completed!")