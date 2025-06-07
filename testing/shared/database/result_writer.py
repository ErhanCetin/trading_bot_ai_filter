"""
Database Result Writer
Handles writing test results to database with proper error handling
"""
import logging
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import threading

logger = logging.getLogger(__name__)
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class ResultWriter:
    """
    Handles writing test results to database.
    Provides batch operations and proper error handling.
    """
    
    def __init__(self, db_connection):
        """
        Initialize result writer.
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db_connection = db_connection
        self.write_lock = threading.Lock()
        self.batch_size = 100
        
        logger.debug("🔧 ResultWriter initialized")
    
    def write_test_execution(self, execution_data: Dict[str, Any]) -> bool:
        """
        Write test execution metadata to database.
        
        Args:
            execution_data: Test execution metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db_connection.is_available:
            logger.error("❌ Database not available for execution write")
            return False
        
        with self.write_lock:
            try:
                insert_sql = """
                INSERT INTO test_executions (
                    execution_id, execution_type, start_time, end_time,
                    duration_seconds, total_tests, successful_tests, failed_tests,
                    phases_tested, overall_success_rate, best_performer_config,
                    best_performer_accuracy, average_accuracy, average_sharpe,
                    status, error_message, metadata
                ) VALUES (
                    :execution_id, :execution_type, :start_time, :end_time,
                    :duration_seconds, :total_tests, :successful_tests, :failed_tests,
                    :phases_tested, :overall_success_rate, :best_performer_config,
                    :best_performer_accuracy, :average_accuracy, :average_sharpe,
                    :status, :error_message, :metadata
                ) ON CONFLICT (execution_id) DO UPDATE SET
                    end_time = EXCLUDED.end_time,
                    duration_seconds = EXCLUDED.duration_seconds,
                    total_tests = EXCLUDED.total_tests,
                    successful_tests = EXCLUDED.successful_tests,
                    failed_tests = EXCLUDED.failed_tests,
                    overall_success_rate = EXCLUDED.overall_success_rate,
                    best_performer_config = EXCLUDED.best_performer_config,
                    best_performer_accuracy = EXCLUDED.best_performer_accuracy,
                    average_accuracy = EXCLUDED.average_accuracy,
                    average_sharpe = EXCLUDED.average_sharpe,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    metadata = EXCLUDED.metadata;
                """
                
                # Prepare execution data
                params = {
                    'execution_id': execution_data['execution_id'],
                    'execution_type': execution_data.get('execution_type', 'indicators_test'),
                    'start_time': execution_data['start_time'],
                    'end_time': execution_data.get('end_time'),
                    'duration_seconds': execution_data.get('duration_seconds'),
                    'total_tests': execution_data.get('total_tests', 0),
                    'successful_tests': execution_data.get('successful_tests', 0),
                    'failed_tests': execution_data.get('failed_tests', 0),
                    'phases_tested': execution_data.get('phases_tested', []),
                    'overall_success_rate': execution_data.get('overall_success_rate', 0.0),
                    'best_performer_config': execution_data.get('best_performer_config'),
                    'best_performer_accuracy': execution_data.get('best_performer_accuracy', 0.0),
                    'average_accuracy': execution_data.get('average_accuracy', 0.0),
                    'average_sharpe': execution_data.get('average_sharpe', 0.0),
                    'status': execution_data.get('status', 'running'),
                    'error_message': execution_data.get('error_message'),
                    'metadata': json.dumps(execution_data.get('metadata', {}))
                }
                
                self.db_connection.execute_query(insert_sql, params)
                
                logger.debug(f"✅ Test execution {execution_data['execution_id']} written to database")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to write test execution: {e}")
                return False
    
    def write_indicator_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Write indicator test results to database.
        
        Args:
            results: List of test result dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not results:
            logger.warning("⚠️ No results to write")
            return True
        
        if not self.db_connection.is_available:
            logger.error("❌ Database not available for results write")
            return False
        
        with self.write_lock:
            try:
                # Process results in batches for better performance
                for i in range(0, len(results), self.batch_size):
                    batch = results[i:i + self.batch_size]
                    success = self._write_results_batch(batch)
                    
                    if not success:
                        logger.error(f"❌ Failed to write batch {i//self.batch_size + 1}")
                        return False
                
                logger.info(f"💾 Successfully wrote {len(results)} indicator results to database")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to write indicator results: {e}")
                return False
    
    def _write_results_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Write a batch of results to database."""
        try:
            # Convert to DataFrame for efficient batch insert
            df = pd.DataFrame(batch)
            
            # Ensure datetime columns are properly formatted
            if 'test_date' in df.columns:
                df['test_date'] = pd.to_datetime(df['test_date'])
            
            # Write to database using pandas to_sql
            df.to_sql(
                'indicator_test_results',
                self.db_connection.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch write failed: {e}")
            return False
    
    def write_phase_summaries(self, execution_id: str, phase_summaries: Dict[str, Dict[str, Any]]) -> bool:
        """
        Write phase summaries to database.
        
        Args:
            execution_id: Test execution ID
            phase_summaries: Dictionary of phase summaries
            
        Returns:
            True if successful, False otherwise
        """
        if not phase_summaries:
            return True
        
        if not self.db_connection.is_available:
            logger.error("❌ Database not available for phase summaries write")
            return False
        
        with self.write_lock:
            try:
                insert_sql = """
                INSERT INTO phase_summaries (
                    execution_id, phase_name, total_tests, successful_tests,
                    failed_tests, success_rate, average_accuracy, best_accuracy,
                    worst_accuracy, average_sharpe, duration_seconds,
                    best_performer_config, test_date
                ) VALUES (
                    :execution_id, :phase_name, :total_tests, :successful_tests,
                    :failed_tests, :success_rate, :average_accuracy, :best_accuracy,
                    :worst_accuracy, :average_sharpe, :duration_seconds,
                    :best_performer_config, :test_date
                ) ON CONFLICT (execution_id, phase_name) DO UPDATE SET
                    total_tests = EXCLUDED.total_tests,
                    successful_tests = EXCLUDED.successful_tests,
                    failed_tests = EXCLUDED.failed_tests,
                    success_rate = EXCLUDED.success_rate,
                    average_accuracy = EXCLUDED.average_accuracy,
                    best_accuracy = EXCLUDED.best_accuracy,
                    worst_accuracy = EXCLUDED.worst_accuracy,
                    average_sharpe = EXCLUDED.average_sharpe,
                    duration_seconds = EXCLUDED.duration_seconds,
                    best_performer_config = EXCLUDED.best_performer_config,
                    test_date = EXCLUDED.test_date;
                """
                
                for phase_name, phase_data in phase_summaries.items():
                    params = {
                        'execution_id': execution_id,
                        'phase_name': phase_name,
                        'total_tests': phase_data.get('total_tests', 0),
                        'successful_tests': phase_data.get('successful_tests', 0),
                        'failed_tests': phase_data.get('failed_tests', 0),
                        'success_rate': phase_data.get('success_rate', 0.0),
                        'average_accuracy': phase_data.get('average_accuracy', 0.0),
                        'best_accuracy': phase_data.get('best_accuracy', 0.0),
                        'worst_accuracy': phase_data.get('worst_accuracy', 0.0),
                        'average_sharpe': phase_data.get('average_sharpe', 0.0),
                        'duration_seconds': phase_data.get('duration', 0.0),
                        'best_performer_config': phase_data.get('best_performer'),
                        'test_date': datetime.now()
                    }
                    
                    self.db_connection.execute_query(insert_sql, params)
                
                logger.info(f"💾 Phase summaries written for execution {execution_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to write phase summaries: {e}")
                return False
    
    def get_latest_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get latest test results from database.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of test results
        """
        if not self.db_connection.is_available:
            return []
        
        try:
            query = """
            SELECT 
                test_id, test_execution_id, test_date, phase_name,
                config_name, indicator_name, symbol, interval,
                status, accuracy_pct, sharpe_ratio, max_drawdown,
                win_rate, profit_factor, total_trades, signal_count
            FROM indicator_test_results 
            ORDER BY test_date DESC 
            LIMIT :limit;
            """
            
            result = self.db_connection.execute_query(query, {'limit': limit})
            
            # Convert to list of dictionaries
            results = []
            for row in result:
                results.append({
                    'test_id': row[0],
                    'test_execution_id': row[1],
                    'test_date': row[2],
                    'phase_name': row[3],
                    'config_name': row[4],
                    'indicator_name': row[5],
                    'symbol': row[6],
                    'interval': row[7],
                    'status': row[8],
                    'accuracy_pct': float(row[9]) if row[9] else 0.0,
                    'sharpe_ratio': float(row[10]) if row[10] else 0.0,
                    'max_drawdown': float(row[11]) if row[11] else 0.0,
                    'win_rate': float(row[12]) if row[12] else 0.0,
                    'profit_factor': float(row[13]) if row[13] else 0.0,
                    'total_trades': row[14] if row[14] else 0,
                    'signal_count': row[15] if row[15] else 0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest results: {e}")
            return []
    
    def get_execution_summary(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary for a specific test execution.
        
        Args:
            execution_id: Test execution ID
            
        Returns:
            Execution summary dictionary or None
        """
        if not self.db_connection.is_available:
            return None
        
        try:
            query = """
            SELECT 
                execution_id, execution_type, start_time, end_time,
                duration_seconds, total_tests, successful_tests, failed_tests,
                phases_tested, overall_success_rate, best_performer_config,
                best_performer_accuracy, average_accuracy, average_sharpe,
                status, error_message
            FROM test_executions 
            WHERE execution_id = :execution_id;
            """
            
            result = self.db_connection.execute_query(query, {'execution_id': execution_id})
            
            if not result:
                return None
            
            row = result[0]
            return {
                'execution_id': row[0],
                'execution_type': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'duration_seconds': row[4],
                'total_tests': row[5],
                'successful_tests': row[6],
                'failed_tests': row[7],
                'phases_tested': row[8],
                'overall_success_rate': float(row[9]) if row[9] else 0.0,
                'best_performer_config': row[10],
                'best_performer_accuracy': float(row[11]) if row[11] else 0.0,
                'average_accuracy': float(row[12]) if row[12] else 0.0,
                'average_sharpe': float(row[13]) if row[13] else 0.0,
                'status': row[14],
                'error_message': row[15]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get execution summary: {e}")
            return None
    
    def get_phase_results(self, execution_id: str, phase_name: str) -> List[Dict[str, Any]]:
        """
        Get all results for a specific phase in an execution.
        
        Args:
            execution_id: Test execution ID
            phase_name: Phase name
            
        Returns:
            List of phase results
        """
        if not self.db_connection.is_available:
            return []
        
        try:
            query = """
            SELECT 
                test_id, config_name, indicator_name, symbol, interval,
                status, accuracy_pct, sharpe_ratio, max_drawdown,
                win_rate, profit_factor, total_trades, signal_count,
                execution_time_ms, error_message
            FROM indicator_test_results 
            WHERE test_execution_id = :execution_id AND phase_name = :phase_name
            ORDER BY accuracy_pct DESC;
            """
            
            result = self.db_connection.execute_query(query, {
                'execution_id': execution_id,
                'phase_name': phase_name
            })
            
            results = []
            for row in result:
                results.append({
                    'test_id': row[0],
                    'config_name': row[1],
                    'indicator_name': row[2],
                    'symbol': row[3],
                    'interval': row[4],
                    'status': row[5],
                    'accuracy_pct': float(row[6]) if row[6] else 0.0,
                    'sharpe_ratio': float(row[7]) if row[7] else 0.0,
                    'max_drawdown': float(row[8]) if row[8] else 0.0,
                    'win_rate': float(row[9]) if row[9] else 0.0,
                    'profit_factor': float(row[10]) if row[10] else 0.0,
                    'total_trades': row[11] if row[11] else 0,
                    'signal_count': row[12] if row[12] else 0,
                    'execution_time_ms': row[13] if row[13] else 0,
                    'error_message': row[14]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get phase results: {e}")
            return []
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from database.
        
        Returns:
            Statistics summary dictionary
        """
        if not self.db_connection.is_available:
            return {'error': 'Database not available'}
        
        try:
            stats = {
                'total_executions': 0,
                'total_tests': 0,
                'successful_tests': 0,
                'average_accuracy': 0.0,
                'best_accuracy': 0.0,
                'phase_statistics': {},
                'recent_activity': {}
            }
            
            # Get execution statistics
            exec_query = """
            SELECT 
                COUNT(*) as total_executions,
                SUM(total_tests) as total_tests,
                SUM(successful_tests) as successful_tests,
                AVG(average_accuracy) as avg_accuracy,
                MAX(best_performer_accuracy) as best_accuracy
            FROM test_executions;
            """
            
            exec_result = self.db_connection.execute_query(exec_query)
            if exec_result:
                row = exec_result[0]
                stats.update({
                    'total_executions': row[0] or 0,
                    'total_tests': row[1] or 0,
                    'successful_tests': row[2] or 0,
                    'average_accuracy': float(row[3]) if row[3] else 0.0,
                    'best_accuracy': float(row[4]) if row[4] else 0.0
                })
            
            # Get phase statistics
            phase_query = """
            SELECT 
                phase_name,
                COUNT(*) as test_count,
                AVG(accuracy_pct) as avg_accuracy,
                MAX(accuracy_pct) as best_accuracy
            FROM indicator_test_results 
            WHERE status = 'success'
            GROUP BY phase_name;
            """
            
            phase_result = self.db_connection.execute_query(phase_query)
            for row in phase_result:
                stats['phase_statistics'][row[0]] = {
                    'test_count': row[1],
                    'average_accuracy': float(row[2]) if row[2] else 0.0,
                    'best_accuracy': float(row[3]) if row[3] else 0.0
                }
            
            # Get recent activity (last 7 days)
            recent_query = """
            SELECT 
                DATE(test_date) as test_day,
                COUNT(*) as daily_tests,
                AVG(accuracy_pct) as daily_avg_accuracy
            FROM indicator_test_results 
            WHERE test_date >= CURRENT_DATE - INTERVAL '7 days'
            AND status = 'success'
            GROUP BY DATE(test_date)
            ORDER BY test_day DESC;
            """
            
            recent_result = self.db_connection.execute_query(recent_query)
            for row in recent_result:
                stats['recent_activity'][str(row[0])] = {
                    'daily_tests': row[1],
                    'daily_avg_accuracy': float(row[2]) if row[2] else 0.0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics summary: {e}")
            return {'error': str(e)}
    
    def cleanup_old_results(self, days_to_keep: int = 30) -> bool:
        """
        Clean up old test results to manage database size.
        
        Args:
            days_to_keep: Number of days to keep in database
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db_connection.is_available:
            return False
        
        try:
            # Delete old indicator results
            cleanup_query = """
            DELETE FROM indicator_test_results 
            WHERE test_date < CURRENT_DATE - INTERVAL '%s days';
            """ % days_to_keep
            
            result = self.db_connection.execute_query(cleanup_query)
            
            # Delete old executions
            exec_cleanup_query = """
            DELETE FROM test_executions 
            WHERE start_time < CURRENT_DATE - INTERVAL '%s days';
            """ % days_to_keep
            
            self.db_connection.execute_query(exec_cleanup_query)
            
            # Delete old phase summaries
            phase_cleanup_query = """
            DELETE FROM phase_summaries 
            WHERE test_date < CURRENT_DATE - INTERVAL '%s days';
            """ % days_to_keep
            
            self.db_connection.execute_query(phase_cleanup_query)
            
            logger.info(f"🧹 Database cleanup completed: kept last {days_to_keep} days")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database cleanup failed: {e}")
            return False


if __name__ == "__main__":
    """Test the database module."""
    
    print("🧪 Testing Database Module...")
    
    # Test database connection
    print("🔧 Testing database connection...")
    from testing.shared.database import db_connection, schema_manager, result_writer
    
    # Test connection
    health = db_connection.get_health_status()
    print(f"📊 Database health: {health}")
    
    if health['available']:
        print("✅ Database connection successful")
        
        # Test schema creation
        print("🏗️ Testing schema creation...")
        schema_success = schema_manager.create_all_tables()
        print(f"📋 Schema creation: {'✅ Success' if schema_success else '❌ Failed'}")
        
        # Test schema validation
        print("🔍 Testing schema validation...")
        validation = schema_manager.validate_schema()
        print(f"✅ Schema validation: {validation['valid']}")
        
        # Test result writing
        print("💾 Testing result writing...")
        test_execution = {
            'execution_id': 'TEST_20250606_120000',
            'execution_type': 'test',
            'start_time': datetime.now(),
            'total_tests': 1,
            'status': 'completed'
        }
        
        write_success = result_writer.write_test_execution(test_execution)
        print(f"📝 Execution write: {'✅ Success' if write_success else '❌ Failed'}")
        
        # Test statistics
        print("📊 Testing statistics...")
        stats = result_writer.get_statistics_summary()
        print(f"📈 Statistics: {stats.get('total_executions', 0)} executions")
        
    else:
        print("❌ Database connection failed")
    
    print("🎉 Database module test completed!")