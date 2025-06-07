"""
Main Data Pipeline Interface for Testing Framework
Provides clean, modular interface for all data operations
"""
import sys
import os
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import with absolute imports
from config_manager import ConfigManager, get_config_manager
from data_fetcher import DataFetcher  
from data_validator import DataValidator
from data_processor import DataProcessor

# Configure logging
logger = logging.getLogger(__name__)


class TestDataPipeline:
    """
    Main data pipeline interface for indicator testing framework.
    Orchestrates data fetching, validation, and processing operations.
    """
    
    def __init__(self):
        """Initialize the complete data pipeline."""
        # Initialize all components
        self.config_manager = get_config_manager()
        self.data_fetcher = DataFetcher()
        self.data_validator = DataValidator()
        self.data_processor = DataProcessor()
        
        # Load global settings
        global_settings = self.config_manager.get_global_settings()
        data_quality_thresholds = global_settings.get('data_quality_thresholds', {})
        
        # Update validator with global thresholds
        self.data_validator = DataValidator(data_quality_thresholds)
        
        logger.info("🔧 TestDataPipeline initialized with all components")
    
    def fetch_test_data(self, symbol: str, interval: str, 
                       min_data_points: Optional[int] = None,
                       force_refresh: bool = False) -> pd.DataFrame:
        """
        Main method to fetch and prepare test data for indicator testing.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            interval: Time interval (e.g., '5m', '1h')
            min_data_points: Minimum required data points (optional)
            force_refresh: If True, fetches fresh data from Binance
            
        Returns:
            DataFrame with validated and processed test data
            
        Raises:
            ValueError: If data quality is insufficient
            Exception: If any pipeline step fails
        """
        logger.info(f"🚀 Starting data pipeline for {symbol} {interval}")
        
        try:
            # Step 1: Determine minimum data points
            if min_data_points is None:
                min_data_points = self.config_manager.get_min_data_points_for_interval(interval)
            
            # Step 2: Check if fresh data is needed
            if force_refresh or not self._has_sufficient_data(symbol, interval, min_data_points):
                logger.info(f"🔄 Fetching fresh data from Binance...")
                success_count, errors = self.data_fetcher.fetch_fresh_data(symbol, interval)
                
                if errors:
                    logger.warning(f"⚠️ Data fetch completed with errors: {errors}")
                else:
                    logger.info(f"✅ Fresh data fetched successfully")
            
            # Step 3: Load data from database  
            logger.info(f"📥 Loading data from database...")
            raw_df = self.data_fetcher.load_data_from_database(symbol, interval)
            
            # Step 4: Validate data quality
            logger.info(f"🔍 Validating data quality...")
            validation_results = self.data_validator.validate_data_quality(
                raw_df, symbol, interval, min_data_points
            )
            
            # Step 5: Process and enhance data
            logger.info(f"🔄 Processing and enhancing data...")
            processed_df = self.data_processor.process_data(
                raw_df, symbol, interval, validation_results['quality_score']
            )
            
            logger.info(f"✅ Data pipeline completed successfully: {len(processed_df)} rows ready for testing")
            return processed_df
            
        except Exception as e:
            logger.error(f"❌ Data pipeline failed for {symbol} {interval}: {e}")
            raise
    
    def _has_sufficient_data(self, symbol: str, interval: str, min_data_points: int) -> bool:
        """Check if sufficient data is available without fetching."""
        return self.data_fetcher.check_data_freshness(symbol, interval, min_data_points)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for the data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        return self.data_processor.generate_summary_statistics(df)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a test configuration for completeness.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
        """
        return self.config_manager.validate_config_completeness(config)
    
    def get_available_configs(self) -> list:
        """
        Get list of available indicator configurations.
        
        Returns:
            List of configuration names
        """
        return self.config_manager.list_available_indicator_configs()
    
    def load_indicator_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load specific indicator configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary
        """
        return self.config_manager.load_indicator_config(config_name)
    
    def get_data_status(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Get current data status for a symbol/interval pair.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            Dictionary with data status information
        """
        try:
            min_data_points = self.config_manager.get_min_data_points_for_interval(interval)
            
            status = {
                'symbol': symbol,
                'interval': interval,
                'min_required_points': min_data_points,
                'has_sufficient_data': self._has_sufficient_data(symbol, interval, min_data_points),
                'data_summary': self.data_fetcher.get_data_summary(symbol, interval)
            }
            
            return status
            
        except Exception as e:
            return {
                'symbol': symbol,
                'interval': interval,
                'error': str(e)
            }


# Convenience functions for easy importing
def create_pipeline() -> TestDataPipeline:
    """
    Create a new TestDataPipeline instance.
    
    Returns:
        TestDataPipeline instance
    """
    return TestDataPipeline()


def fetch_data(symbol: str, interval: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Convenience function to fetch test data.
    
    Args:
        symbol: Trading symbol
        interval: Time interval  
        force_refresh: Whether to force fresh data fetch
        
    Returns:
        Processed DataFrame ready for testing
    """
    pipeline = create_pipeline()
    return pipeline.fetch_test_data(symbol, interval, force_refresh=force_refresh)


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Convenience function to load indicator configuration.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Configuration dictionary
    """
    pipeline = create_pipeline()
    return pipeline.load_indicator_config(config_name)


# Export main classes and functions
__all__ = [
    'TestDataPipeline',
    'ConfigManager', 
    'DataFetcher',
    'DataValidator',
    'DataProcessor',
    'create_pipeline',
    'fetch_data',
    'load_config'
]


if __name__ == "__main__":
    """Test the complete data pipeline."""
    
    print("🧪 Testing Complete Data Pipeline...")
    
    try:
        # Test pipeline creation
        print("🔧 Creating pipeline...")
        pipeline = TestDataPipeline()
        
        # Test configuration loading
        print("📋 Testing configuration loading...")
        available_configs = pipeline.get_available_configs()
        print(f"✅ Available configs: {available_configs}")
        
        if available_configs:
            # Load first available config
            config_name = available_configs[0]
            config = pipeline.load_indicator_config(config_name)
            print(f"✅ Config '{config_name}' loaded successfully")
            print(f"📊 Config details: {config.get('test_name', 'Unknown')}")
            
            # Test data status
            print(f"📊 Checking data status...")
            status = pipeline.get_data_status(config['symbol'], config['interval'])
            print(f"✅ Data status: {status['has_sufficient_data']}")
            
            # Test data fetching (without forcing refresh for test)
            print(f"📥 Testing data fetch...")
            test_data = pipeline.fetch_test_data(
                config['symbol'], 
                config['interval'],
                force_refresh=False
            )
            
            print(f"✅ Data fetch successful:")
            print(f"   📊 Rows: {len(test_data)}")
            print(f"   📋 Columns: {len(test_data.columns)}")
            
            # Test summary generation
            print(f"📈 Generating summary...")
            summary = pipeline.get_data_summary(test_data)
            
            print(f"✅ Summary generated:")
            print(f"   🏆 Quality score: {summary['data_info']['data_quality_score']}")
            print(f"   📅 Date range: {summary['data_info']['date_range']['duration_days']} days")
            print(f"   💰 Price range: ${summary['price_statistics']['price_range']['min']:.2f} - ${summary['price_statistics']['price_range']['max']:.2f}")
        
        print("🎉 Complete pipeline test successful!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test convenience functions
    print("\n🧪 Testing convenience functions...")
    
    try:
        # Test convenience pipeline creation
        pipeline2 = create_pipeline()
        print("✅ Convenience pipeline creation works")
        
        # Test convenience config loading
        if available_configs:
            config2 = load_config(available_configs[0])
            print("✅ Convenience config loading works")
        
    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
    
    print("🎉 All tests completed!")