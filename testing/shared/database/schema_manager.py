"""
Database Schema Manager
Handles table creation, migrations, and schema validation
"""
import logging
from pathlib import Path
import sys
from typing import Dict, Any, List
from sqlalchemy import text

logger = logging.getLogger(__name__)
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class SchemaManager:
    """
    Manages database schema for testing framework.
    Handles table creation, updates, and validation.
    """
    
    def __init__(self, db_connection):
        """
        Initialize schema manager.
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db_connection = db_connection
        self.current_version = "1.0.0"
        
        logger.debug("🔧 SchemaManager initialized")
    
    def create_all_tables(self) -> bool:
        """
        Create all required tables for testing framework.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.db_connection.is_available:
            logger.error("❌ Database not available for schema creation")
            return False
        
        try:
            # Create indicator test results table
            success = self.create_indicator_test_results_table()
            if not success:
                return False
            
            # Create test execution metadata table
            success = self.create_test_execution_table()
            if not success:
                return False
            
            # Create phase summary table
            success = self.create_phase_summary_table()
            if not success:
                return False
            
            logger.info("✅ All database tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema creation failed: {e}")
            return False
    
    def create_indicator_test_results_table(self) -> bool:
        """Create indicator test results table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS indicator_test_results (
            id SERIAL PRIMARY KEY,
            test_execution_id VARCHAR(50) NOT NULL,
            test_id VARCHAR(50) NOT NULL UNIQUE,
            test_date TIMESTAMP NOT NULL,
            phase_name VARCHAR(50) NOT NULL,
            config_name VARCHAR(100) NOT NULL,
            indicator_name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            status VARCHAR(20) NOT NULL,
            signal_count INTEGER DEFAULT 0,
            accuracy_pct DECIMAL(5,2) DEFAULT 0.0,
            sharpe_ratio DECIMAL(8,3) DEFAULT 0.0,
            max_drawdown DECIMAL(5,2) DEFAULT 0.0,
            win_rate DECIMAL(5,2) DEFAULT 0.0,
            profit_factor DECIMAL(8,3) DEFAULT 0.0,
            total_trades INTEGER DEFAULT 0,
            buy_signals INTEGER DEFAULT 0,
            sell_signals INTEGER DEFAULT 0,
            total_periods INTEGER DEFAULT 0,
            data_quality_score DECIMAL(5,1) DEFAULT 0.0,
            execution_time_ms INTEGER DEFAULT 0,
            test_duration_days INTEGER DEFAULT 7,
            parameters TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Create indexes for performance
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_indicator_test_results_execution_id ON indicator_test_results(test_execution_id);",
            "CREATE INDEX IF NOT EXISTS idx_indicator_test_results_phase ON indicator_test_results(phase_name);",
            "CREATE INDEX IF NOT EXISTS idx_indicator_test_results_indicator ON indicator_test_results(indicator_name);",
            "CREATE INDEX IF NOT EXISTS idx_indicator_test_results_symbol ON indicator_test_results(symbol, interval);",
            "CREATE INDEX IF NOT EXISTS idx_indicator_test_results_date ON indicator_test_results(test_date);",
            "CREATE INDEX IF NOT EXISTS idx_indicator_test_results_status ON indicator_test_results(status);",
            "CREATE INDEX IF NOT EXISTS idx_indicator_test_results_accuracy ON indicator_test_results(accuracy_pct DESC);"
        ]
        
        try:
            # Create table
            self.db_connection.execute_query(create_table_sql)
            
            # Create indexes
            for index_sql in indexes_sql:
                self.db_connection.execute_query(index_sql)
            
            logger.info("✅ indicator_test_results table created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create indicator_test_results table: {e}")
            return False
    
    def create_test_execution_table(self) -> bool:
        """Create test execution metadata table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS test_executions (
            id SERIAL PRIMARY KEY,
            execution_id VARCHAR(50) NOT NULL UNIQUE,
            execution_type VARCHAR(20) NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_seconds INTEGER,
            total_tests INTEGER DEFAULT 0,
            successful_tests INTEGER DEFAULT 0,
            failed_tests INTEGER DEFAULT 0,
            phases_tested TEXT[],
            overall_success_rate DECIMAL(5,2) DEFAULT 0.0,
            best_performer_config VARCHAR(100),
            best_performer_accuracy DECIMAL(5,2) DEFAULT 0.0,
            average_accuracy DECIMAL(5,2) DEFAULT 0.0,
            average_sharpe DECIMAL(8,3) DEFAULT 0.0,
            status VARCHAR(20) NOT NULL,
            error_message TEXT,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_test_executions_id ON test_executions(execution_id);",
            "CREATE INDEX IF NOT EXISTS idx_test_executions_type ON test_executions(execution_type);",
            "CREATE INDEX IF NOT EXISTS idx_test_executions_start_time ON test_executions(start_time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_test_executions_status ON test_executions(status);"
        ]
        
        try:
            self.db_connection.execute_query(create_table_sql)
            
            for index_sql in indexes_sql:
                self.db_connection.execute_query(index_sql)
            
            logger.info("✅ test_executions table created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create test_executions table: {e}")
            return False
    
    def create_phase_summary_table(self) -> bool:
        """Create phase summary table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS phase_summaries (
            id SERIAL PRIMARY KEY,
            execution_id VARCHAR(50) NOT NULL,
            phase_name VARCHAR(50) NOT NULL,
            total_tests INTEGER DEFAULT 0,
            successful_tests INTEGER DEFAULT 0,
            failed_tests INTEGER DEFAULT 0,
            success_rate DECIMAL(5,2) DEFAULT 0.0,
            average_accuracy DECIMAL(5,2) DEFAULT 0.0,
            best_accuracy DECIMAL(5,2) DEFAULT 0.0,
            worst_accuracy DECIMAL(5,2) DEFAULT 0.0,
            average_sharpe DECIMAL(8,3) DEFAULT 0.0,
            duration_seconds DECIMAL(8,2) DEFAULT 0.0,
            best_performer_config VARCHAR(100),
            test_date TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(execution_id, phase_name)
        );
        """
        
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_phase_summaries_execution ON phase_summaries(execution_id);",
            "CREATE INDEX IF NOT EXISTS idx_phase_summaries_phase ON phase_summaries(phase_name);",
            "CREATE INDEX IF NOT EXISTS idx_phase_summaries_date ON phase_summaries(test_date DESC);"
        ]
        
        try:
            self.db_connection.execute_query(create_table_sql)
            
            for index_sql in indexes_sql:
                self.db_connection.execute_query(index_sql)
            
            logger.info("✅ phase_summaries table created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create phase_summaries table: {e}")
            return False
    
    def validate_schema(self) -> Dict[str, Any]:
        """
        Validate database schema integrity.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'tables_exist': {},
            'missing_tables': [],
            'errors': []
        }
        
        required_tables = [
            'indicator_test_results',
            'test_executions', 
            'phase_summaries'
        ]
        
        if not self.db_connection.is_available:
            validation_results['valid'] = False
            validation_results['errors'].append('Database not available')
            return validation_results
        
        try:
            # Check if tables exist
            for table_name in required_tables:
                exists = self._table_exists(table_name)
                validation_results['tables_exist'][table_name] = exists
                
                if not exists:
                    validation_results['missing_tables'].append(table_name)
                    validation_results['valid'] = False
            
            if validation_results['missing_tables']:
                validation_results['errors'].append(f"Missing tables: {validation_results['missing_tables']}")
            
            logger.info(f"📋 Schema validation: {'✅ Valid' if validation_results['valid'] else '❌ Invalid'}")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(str(e))
            logger.error(f"❌ Schema validation failed: {e}")
        
        return validation_results
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :table_name
            );
            """
            
            result = self.db_connection.execute_query(query, {'table_name': table_name})
            return result[0][0] if result else False
            
        except Exception as e:
            logger.error(f"❌ Error checking table existence for {table_name}: {e}")
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information."""
        schema_info = {
            'version': self.current_version,
            'tables': {},
            'total_records': {},
            'last_updated': {},
            'database_size': None
        }
        
        if not self.db_connection.is_available:
            schema_info['error'] = 'Database not available'
            return schema_info
        
        tables = ['indicator_test_results', 'test_executions', 'phase_summaries']
        
        try:
            for table_name in tables:
                # Get table info
                table_info = self._get_table_info(table_name)
                schema_info['tables'][table_name] = table_info
                
                # Get record count
                count = self._get_table_record_count(table_name)
                schema_info['total_records'][table_name] = count
                
                # Get last updated
                last_updated = self._get_table_last_updated(table_name)
                schema_info['last_updated'][table_name] = last_updated
            
        except Exception as e:
            schema_info['error'] = str(e)
            logger.error(f"❌ Error getting schema info: {e}")
        
        return schema_info
    
    def _get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a table."""
        try:
            query = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_name = :table_name
            ORDER BY ordinal_position;
            """
            
            result = self.db_connection.execute_query(query, {'table_name': table_name})
            
            columns = []
            for row in result:
                columns.append({
                    'name': row[0],
                    'type': row[1],
                    'nullable': row[2] == 'YES',
                    'default': row[3]
                })
            
            return {
                'exists': len(columns) > 0,
                'column_count': len(columns),
                'columns': columns
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting table info for {table_name}: {e}")
            return {'exists': False, 'error': str(e)}
    
    def _get_table_record_count(self, table_name: str) -> int:
        """Get record count for a table."""
        try:
            query = f"SELECT COUNT(*) FROM {table_name};"
            result = self.db_connection.execute_query(query)
            return result[0][0] if result else 0
        except Exception:
            return 0
    
    def _get_table_last_updated(self, table_name: str) -> str:
        """Get last updated timestamp for a table."""
        try:
            # Try to get max created_at or updated_at
            for time_col in ['updated_at', 'created_at', 'test_date']:
                try:
                    query = f"SELECT MAX({time_col}) FROM {table_name};"
                    result = self.db_connection.execute_query(query)
                    if result and result[0][0]:
                        return str(result[0][0])
                except:
                    continue
            return 'unknown'
        except Exception:
            return 'unknown'