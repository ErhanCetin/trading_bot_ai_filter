"""
Test ID Manager - Global Unique Test ID Generation
Handles generation of unique test IDs with proper formatting and collision prevention
"""
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TestIDManager:
    """
    Centralized test ID generation and management.
    Ensures globally unique test IDs across all test executions.
    """
    
    def __init__(self):
        """Initialize test ID manager with thread safety."""
        self._lock = threading.Lock()
        self._execution_counter = 0
        self._test_counter = 0
        self._session_id = self._generate_session_id()
        
        # Cache for ID validation
        self._generated_ids = set()
        self._current_execution_id = None
        
        logger.debug("🔧 TestIDManager initialized")
    
    def generate_execution_id(self, test_type: str = "indicators") -> str:
        """
        Generate unique execution ID for test run.
        
        Args:
            test_type: Type of test execution (indicators, strategies, etc.)
            
        Returns:
            Unique execution ID string
        """
        with self._lock:
            self._execution_counter += 1
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            execution_id = f"{test_type.upper()}_EXEC_{timestamp}_{self._session_id}_{self._execution_counter:04d}"
            
            # Ensure uniqueness
            if execution_id in self._generated_ids:
                # Add microseconds for collision resolution
                microseconds = datetime.now().microsecond
                execution_id = f"{execution_id}_{microseconds:06d}"
            
            self._generated_ids.add(execution_id)
            self._current_execution_id = execution_id
            
            logger.info(f"🆔 Generated execution ID: {execution_id}")
            return execution_id
    
    def generate_test_id(self, config_name: str, phase_name: Optional[str] = None) -> str:
        """
        Generate unique test ID for individual test.
        
        Args:
            config_name: Name of the test configuration
            phase_name: Optional phase name for grouping
            
        Returns:
            Unique test ID string
        """
        with self._lock:
            self._test_counter += 1
            
            # Build test ID components
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Clean config name for ID
            clean_config = self._clean_name_for_id(config_name)
            
            if phase_name:
                clean_phase = self._clean_name_for_id(phase_name)
                test_id = f"TEST_{clean_phase}_{clean_config}_{timestamp}_{self._test_counter:06d}"
            else:
                test_id = f"TEST_{clean_config}_{timestamp}_{self._test_counter:06d}"
            
            # Ensure uniqueness
            if test_id in self._generated_ids:
                # Add random suffix for collision resolution
                random_suffix = str(uuid.uuid4())[:8]
                test_id = f"{test_id}_{random_suffix}"
            
            self._generated_ids.add(test_id)
            
            logger.debug(f"🏷️ Generated test ID: {test_id}")
            return test_id
    
    def generate_batch_test_ids(self, config_names: list, phase_name: Optional[str] = None) -> Dict[str, str]:
        """
        Generate test IDs for a batch of configurations.
        
        Args:
            config_names: List of configuration names
            phase_name: Optional phase name
            
        Returns:
            Dictionary mapping config names to test IDs
        """
        # DON'T use lock here - let individual generate_test_id calls handle locking
        batch_ids = {}
        
        for config_name in config_names:
            test_id = self.generate_test_id(config_name, phase_name)
            batch_ids[config_name] = test_id
        
        logger.info(f"🏷️ Generated {len(batch_ids)} test IDs for batch")
        return batch_ids
    
    def _clean_name_for_id(self, name: str) -> str:
        """Clean name to be safe for ID generation."""
        # Remove special characters and limit length
        clean = ''.join(c for c in name if c.isalnum() or c in '_-')
        return clean[:30]  # Limit length to prevent overly long IDs
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        return str(uuid.uuid4())[:8].upper()
    
    def get_current_execution_id(self) -> Optional[str]:
        """Get the current execution ID."""
        return self._current_execution_id
    
    def validate_test_id(self, test_id: str) -> bool:
        """
        Validate if a test ID was generated by this manager.
        
        Args:
            test_id: Test ID to validate
            
        Returns:
            True if ID is valid and known
        """
        return test_id in self._generated_ids
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated IDs.
        
        Returns:
            Dictionary with ID generation statistics
        """
        with self._lock:
            return {
                'session_id': self._session_id,
                'total_executions': self._execution_counter,
                'total_tests': self._test_counter,
                'unique_ids_generated': len(self._generated_ids),
                'current_execution_id': self._current_execution_id
            }
    
    def clear_cache(self, keep_current_execution: bool = True):
        """
        Clear the ID cache to free memory.
        
        Args:
            keep_current_execution: Whether to keep current execution ID
        """
        with self._lock:
            if keep_current_execution and self._current_execution_id:
                # Keep only current execution ID
                self._generated_ids = {self._current_execution_id}
            else:
                self._generated_ids.clear()
                self._current_execution_id = None
            
            logger.info("🗑️ Test ID cache cleared")
    
    def reset_counters(self):
        """Reset all counters (use carefully)."""
        with self._lock:
            self._execution_counter = 0
            self._test_counter = 0
            self._generated_ids.clear()
            self._current_execution_id = None
            self._session_id = self._generate_session_id()
            
            logger.warning("🔄 Test ID manager reset - all counters cleared")


class TestMetadata:
    """
    Helper class for managing test metadata associated with IDs.
    """
    
    def __init__(self):
        """Initialize metadata manager."""
        self._metadata = {}
        self._lock = threading.Lock()
    
    def add_test_metadata(self, test_id: str, metadata: Dict[str, Any]):
        """
        Add metadata for a test ID.
        
        Args:
            test_id: Test ID
            metadata: Metadata dictionary
        """
        with self._lock:
            self._metadata[test_id] = {
                'created_at': datetime.now(),
                **metadata
            }
    
    def get_test_metadata(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a test ID.
        
        Args:
            test_id: Test ID
            
        Returns:
            Metadata dictionary or None
        """
        return self._metadata.get(test_id)
    
    def get_execution_tests(self, execution_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all tests for an execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Dictionary of test metadata
        """
        execution_tests = {}
        
        for test_id, metadata in self._metadata.items():
            if metadata.get('execution_id') == execution_id:
                execution_tests[test_id] = metadata
        
        return execution_tests
    
    def cleanup_old_metadata(self, hours_to_keep: int = 24):
        """
        Clean up old metadata to free memory.
        
        Args:
            hours_to_keep: Hours of metadata to keep
        """
        with self._lock:
            from datetime import timedelta  # Import here for safety
            cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
            
            old_ids = []
            for test_id, metadata in self._metadata.items():
                if metadata.get('created_at', cutoff_time) < cutoff_time:
                    old_ids.append(test_id)
            
            for test_id in old_ids:
                del self._metadata[test_id]
            
            logger.info(f"🗑️ Cleaned up {len(old_ids)} old metadata entries")


# Singleton instances for global use
test_id_manager = TestIDManager()
test_metadata = TestMetadata()


def generate_execution_id(test_type: str = "indicators") -> str:
    """
    Convenience function to generate execution ID.
    
    Args:
        test_type: Type of test execution
        
    Returns:
        Unique execution ID
    """
    return test_id_manager.generate_execution_id(test_type)


def generate_test_id(config_name: str, phase_name: Optional[str] = None) -> str:
    """
    Convenience function to generate test ID.
    
    Args:
        config_name: Configuration name
        phase_name: Optional phase name
        
    Returns:
        Unique test ID
    """
    return test_id_manager.generate_test_id(config_name, phase_name)


def get_current_execution_id() -> Optional[str]:
    """
    Convenience function to get current execution ID.
    
    Returns:
        Current execution ID or None
    """
    return test_id_manager.get_current_execution_id()


def add_test_metadata(test_id: str, metadata: Dict[str, Any]):
    """
    Convenience function to add test metadata.
    
    Args:
        test_id: Test ID
        metadata: Metadata dictionary
    """
    test_metadata.add_test_metadata(test_id, metadata)


if __name__ == "__main__":
    """Test the test ID manager."""
    
    print("🧪 Testing TestIDManager...")
    
    # Test execution ID generation
    print("🆔 Testing execution ID generation...")
    exec_id1 = test_id_manager.generate_execution_id("indicators")
    exec_id2 = test_id_manager.generate_execution_id("strategies")
    
    print(f"✅ Execution ID 1: {exec_id1}")
    print(f"✅ Execution ID 2: {exec_id2}")
    
    # Test test ID generation
    print("\n🏷️ Testing test ID generation...")
    test_configs = ["rsi_14_oversold", "macd_crossover", "bollinger_squeeze"]
    
    for config in test_configs:
        test_id = test_id_manager.generate_test_id(config, "base_indicators")
        print(f"✅ {config} -> {test_id}")
    
    # Test batch generation
    print("\n📦 Testing batch ID generation...")
    batch_ids = test_id_manager.generate_batch_test_ids(test_configs, "advanced_indicators")
    
    for config, test_id in batch_ids.items():
        print(f"✅ Batch: {config} -> {test_id}")
    
    # Test statistics
    print("\n📊 Testing statistics...")
    stats = test_id_manager.get_statistics()
    print(f"✅ Statistics: {stats}")
    
    # Test metadata
    print("\n📋 Testing metadata...")
    sample_metadata = {
        'execution_id': exec_id1,
        'symbol': 'ETHUSDT',
        'interval': '5m',
        'indicator': 'rsi'
    }
    
    first_test_id = list(batch_ids.values())[0]
    test_metadata.add_test_metadata(first_test_id, sample_metadata)
    
    retrieved_metadata = test_metadata.get_test_metadata(first_test_id)
    print(f"✅ Metadata stored and retrieved: {retrieved_metadata}")
    
    # Test convenience functions
    print("\n🔧 Testing convenience functions...")
    conv_exec_id = generate_execution_id("test")
    conv_test_id = generate_test_id("test_config", "test_phase")
    
    print(f"✅ Convenience execution ID: {conv_exec_id}")
    print(f"✅ Convenience test ID: {conv_test_id}")
    
    print("\n🎉 TestIDManager test completed!")