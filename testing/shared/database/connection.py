"""
Database Connection Manager
Handles PostgreSQL connections with retry logic and health checks
"""
import logging
import time
from typing import Optional, Any, Dict
from contextlib import contextmanager
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.exc import OperationalError
import threading

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Centralized database connection manager for testing framework.
    Handles connection pooling, retry logic, and health monitoring.
    """
    
    def __init__(self):
        """Initialize database connection manager."""
        self._engine: Optional[Engine] = None
        self._connection_lock = threading.Lock()
        self._connection_attempts = 0
        self._max_retries = 3
        self._retry_delay = 2
        
        logger.debug("🔧 DatabaseConnection manager initialized")
    
    @property
    def engine(self) -> Optional[Engine]:
        """Get database engine, creating it if necessary."""
        if self._engine is None:
            with self._connection_lock:
                if self._engine is None:  # Double-check locking
                    self._engine = self._create_engine()
        return self._engine
    
    @property
    def is_available(self) -> bool:
        """Check if database is available."""
        return self.engine is not None and self._test_connection()
    
    def _create_engine(self) -> Optional[Engine]:
        """Create database engine with retry logic."""
        try:
            # Import existing engine from project
            from db.postgresql import engine as project_engine
            
            # Test the connection
            if self._test_project_engine(project_engine):
                logger.info("✅ Using existing project database engine")
                return project_engine
            else:
                logger.warning("⚠️ Project engine connection failed")
                return None
                
        except ImportError:
            logger.warning("⚠️ Project database engine not available")
            return None
        except Exception as e:
            logger.error(f"❌ Database engine creation failed: {e}")
            return None
    
    def _test_project_engine(self, engine: Engine) -> bool:
        """Test if the project engine is working."""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).fetchone()
                return result is not None
        except Exception as e:
            logger.warning(f"⚠️ Project engine test failed: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """Test database connection health."""
        if not self._engine:
            return False
        
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    @contextmanager
    def get_connection(self):
        """
        Get database connection with automatic cleanup.
        
        Usage:
            with db_connection.get_connection() as conn:
                result = conn.execute("SELECT * FROM table")
        """
        if not self.engine:
            raise ConnectionError("Database engine not available")
        
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except Exception as e:
            logger.error(f"❌ Database operation failed: {e}")
            if connection:
                try:
                    connection.rollback()
                except:
                    pass
            raise
        finally:
            if connection:
                try:
                    connection.close()
                except:
                    pass
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """
        Execute a query with retry logic.
        
        Args:
            query: SQL query string or text() object
            params: Query parameters
            
        Returns:
            Query result
        """
        if not self.is_available:
            raise ConnectionError("Database not available")
        
        for attempt in range(self._max_retries):
            try:
                with self.get_connection() as conn:
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    
                    # If it's a SELECT query, fetch results
                    if query.strip().upper().startswith('SELECT'):
                        return result.fetchall()
                    else:
                        conn.commit()
                        return result
                        
            except OperationalError as e:
                self._connection_attempts += 1
                logger.warning(f"⚠️ Database query attempt {attempt + 1} failed: {e}")
                
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"❌ All {self._max_retries} database attempts failed")
                    raise
            except Exception as e:
                logger.error(f"❌ Database query failed: {e}")
                raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get database health status information."""
        health_status = {
            'available': self.is_available,
            'engine_created': self._engine is not None,
            'connection_attempts': self._connection_attempts,
            'last_test': None,
            'error': None
        }
        
        if self.is_available:
            try:
                start_time = time.time()
                with self.get_connection() as conn:
                    conn.execute(text("SELECT 1"))
                response_time = (time.time() - start_time) * 1000
                
                health_status.update({
                    'last_test': 'success',
                    'response_time_ms': round(response_time, 2)
                })
            except Exception as e:
                health_status.update({
                    'last_test': 'failed',
                    'error': str(e)
                })
        
        return health_status

