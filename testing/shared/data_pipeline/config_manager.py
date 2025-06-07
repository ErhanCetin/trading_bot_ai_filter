"""
Configuration Management for Testing Framework - UPDATED VERSION
Handles loading of global settings and phase-based indicator configs
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management for testing framework.
    Now supports both single configs and phase-based configurations.
    """
    
    def __init__(self):
        """Initialize configuration manager with phase support."""
        self.global_settings = None
        self._config_cache = {}  # Cache loaded configs for performance
        
        # Define config file paths
        self.shared_dir = os.path.dirname(__file__)
        self.global_settings_path = os.path.join(self.shared_dir, 'global_settings.json')
        
        # Phase-based config file mapping
        self.phase_config_files = {
            "base_conditions": "base_configs.json",
            "advanced_conditions": "advanced_configs.json", 
            "feature_conditions": "feature_configs.json",
            "regime_conditions": "regime_configs.json"
        }
        
        # Legacy single config support
        self.indicators_config_path = os.path.join(
            os.path.dirname(self.shared_dir), 'indicators', 'indicator_test_configs.json'
        )
        
        logger.debug("🔧 ConfigManager initialized with multi-phase support")
    
    def get_global_settings(self) -> Dict[str, Any]:
        """
        Get global settings, loading from file if not already cached.
        
        Returns:
            Dictionary with global settings
        """
        if self.global_settings is None:
            self.global_settings = self._load_global_settings()
        
        return self.global_settings
    
    def get_available_phases(self) -> List[str]:
        """
        Get list of available phase names.
        
        Returns:
            List of phase names
        """
        available_phases = []
        indicators_dir = os.path.join(os.path.dirname(self.shared_dir), 'indicators')
        
        for phase_name, filename in self.phase_config_files.items():
            file_path = os.path.join(indicators_dir, filename)
            if os.path.exists(file_path):
                available_phases.append(phase_name)
        
        logger.debug(f"📂 Available phases: {available_phases}")
        return available_phases
    
    def load_config_by_phase(self, phase_name: str) -> Dict[str, Any]:
        """
        Load all configurations for a specific phase.
        
        Args:
            phase_name: Name of the phase (e.g., 'base_conditions', 'advanced_conditions')
            
        Returns:
            Dictionary containing all configurations for the phase
            
        Raises:
            ValueError: If phase not found
            FileNotFoundError: If config file not found
        """
        if phase_name not in self.phase_config_files:
            available_phases = list(self.phase_config_files.keys())
            raise ValueError(
                f"Phase '{phase_name}' not found. "
                f"Available phases: {available_phases}"
            )
        
        # Check cache first
        cache_key = f"phase_{phase_name}"
        if cache_key in self._config_cache:
            logger.debug(f"📋 Using cached config for phase {phase_name}")
            return self._config_cache[cache_key]
        
        # Load phase config file
        filename = self.phase_config_files[phase_name]
        indicators_dir = os.path.join(os.path.dirname(self.shared_dir), 'indicators')
        file_path = os.path.join(indicators_dir, filename)
        
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            # Cache the loaded config
            self._config_cache[cache_key] = config_data
            
            logger.info(f"✅ Phase config '{phase_name}' loaded successfully from {filename}")
            logger.debug(f"📊 Loaded {len(config_data.get('indicator_test_configurations', {}))} configurations")
            return config_data
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Phase configuration file not found: {file_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in phase config file {filename}: {e}")
        except Exception as e:
            raise Exception(f"Error loading phase configuration '{phase_name}': {e}")
    
    def load_all_phases(self) -> Dict[str, Dict[str, Any]]:
        """
        Load configurations for ALL available phases.
        
        Returns:
            Dictionary mapping phase names to their configurations
        """
        logger.info("🚀 Loading ALL phases...")
        all_phases_config = {}
        available_phases = self.get_available_phases()
        
        for phase_name in available_phases:
            try:
                phase_config = self.load_config_by_phase(phase_name)
                all_phases_config[phase_name] = phase_config
                logger.info(f"✅ Phase '{phase_name}' loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load phase '{phase_name}': {e}")
                all_phases_config[phase_name] = {"error": str(e)}
        
        total_configs = sum(
            len(phase_data.get('indicator_test_configurations', {})) 
            for phase_data in all_phases_config.values() 
            if 'error' not in phase_data
        )
        
        logger.info(f"🎯 All phases loaded: {len(all_phases_config)} phases, {total_configs} total configurations")
        return all_phases_config
    
    def get_phase_configs_list(self, phase_name: str) -> List[str]:
        """
        Get list of configuration names for a specific phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            List of configuration names in the phase
        """
        try:
            phase_data = self.load_config_by_phase(phase_name)
            configs = phase_data.get('indicator_test_configurations', {})
            return list(configs.keys())
        except Exception as e:
            logger.error(f"❌ Error getting config list for phase '{phase_name}': {e}")
            return []
    
    def get_all_configs_list(self) -> Dict[str, List[str]]:
        """
        Get configuration names for ALL phases.
        
        Returns:
            Dictionary mapping phase names to lists of configuration names
        """
        all_configs = {}
        available_phases = self.get_available_phases()
        
        for phase_name in available_phases:
            all_configs[phase_name] = self.get_phase_configs_list(phase_name)
        
        return all_configs
    
    def load_specific_config(self, config_name: str, phase_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a specific configuration by name, optionally from a specific phase.
        
        Args:
            config_name: Name of the configuration (e.g., 'rsi_14_oversold')
            phase_name: Optional phase name. If None, searches all phases.
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If configuration not found
        """
        if phase_name:
            # Search in specific phase
            try:
                phase_data = self.load_config_by_phase(phase_name)
                configs = phase_data.get('indicator_test_configurations', {})
                
                if config_name in configs:
                    logger.debug(f"✅ Config '{config_name}' found in phase '{phase_name}'")
                    return configs[config_name]
                else:
                    raise ValueError(f"Configuration '{config_name}' not found in phase '{phase_name}'")
            except Exception as e:
                raise ValueError(f"Error loading config '{config_name}' from phase '{phase_name}': {e}")
        else:
            # Search in all phases
            available_phases = self.get_available_phases()
            
            for search_phase in available_phases:
                try:
                    phase_data = self.load_config_by_phase(search_phase)
                    configs = phase_data.get('indicator_test_configurations', {})
                    
                    if config_name in configs:
                        logger.debug(f"✅ Config '{config_name}' found in phase '{search_phase}'")
                        return configs[config_name]
                except Exception as e:
                    logger.warning(f"⚠️ Error searching phase '{search_phase}': {e}")
                    continue
            
            # Not found in any phase
            all_configs = self.get_all_configs_list()
            available_configs = []
            for phase, configs in all_configs.items():
                available_configs.extend([f"{phase}:{config}" for config in configs])
            
            raise ValueError(
                f"Configuration '{config_name}' not found in any phase. "
                f"Available configurations: {available_configs}"
            )
    
    def list_available_indicator_configs(self) -> List[str]:
        """
        Get list of ALL available indicator configurations across all phases.
        
        Returns:
            List of configuration names with phase prefixes
        """
        all_configs = []
        all_phase_configs = self.get_all_configs_list()
        
        for phase_name, config_names in all_phase_configs.items():
            for config_name in config_names:
                all_configs.append(f"{phase_name}:{config_name}")
        
        logger.debug(f"📋 Total available configs: {len(all_configs)}")
        return all_configs
    
    def get_phase_metadata(self, phase_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Phase metadata dictionary
        """
        try:
            phase_data = self.load_config_by_phase(phase_name)
            return phase_data.get('test_metadata', {})
        except Exception as e:
            logger.error(f"❌ Error getting metadata for phase '{phase_name}': {e}")
            return {}
    
    def get_all_phases_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for ALL phases.
        
        Returns:
            Dictionary mapping phase names to their metadata
        """
        all_metadata = {}
        available_phases = self.get_available_phases()
        
        for phase_name in available_phases:
            all_metadata[phase_name] = self.get_phase_metadata(phase_name)
        
        return all_metadata
    
    # Legacy support methods (backward compatibility)
    def load_indicator_config(self, config_name: str) -> Dict[str, Any]:
        """
        Legacy method: Load specific indicator configuration by name.
        Now searches across all phases.
        
        Args:
            config_name: Name of the indicator configuration
            
        Returns:
            Configuration dictionary
        """
        logger.warning("⚠️ Using legacy load_indicator_config method. Consider using load_specific_config instead.")
        return self.load_specific_config(config_name)
    
    def _load_global_settings(self) -> Dict[str, Any]:
        """
        Load global settings from global_settings.json file.
        
        Returns:
            Dictionary with global settings
        """
        try:
            with open(self.global_settings_path, 'r') as f:
                config_data = json.load(f)
            
            settings = config_data.get('global_settings', {})
            logger.debug("✅ Global settings loaded successfully")
            return settings
            
        except FileNotFoundError:
            logger.warning(f"⚠️ Global settings file not found: {self.global_settings_path}")
            return self._get_default_global_settings()
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in global settings: {e}")
            return self._get_default_global_settings()
        except Exception as e:
            logger.error(f"❌ Error loading global settings: {e}")
            return self._get_default_global_settings()
    
    def _get_default_global_settings(self) -> Dict[str, Any]:
        """
        Get default global settings as fallback.
        
        Returns:
            Default settings dictionary
        """
        return {
            'data_quality_thresholds': {
                'min_data_points_5m': 2000,
                'min_data_points_15m': 672,
                'min_data_points_1h': 168,
                'max_missing_data_pct': 5.0,
                'min_volume_threshold': 1000,
                'min_data_quality_score': 75.0
            },
            'database_settings': {
                'connection_timeout': 30,
                'query_timeout': 60,
                'retry_attempts': 3
            },
            'test_output_settings': {
                'save_to_db': True,
                'save_to_csv': True,
                'generate_charts': True
            }
        }
    
    def get_data_quality_thresholds(self) -> Dict[str, Any]:
        """
        Get data quality thresholds from global settings.
        
        Returns:
            Data quality thresholds dictionary
        """
        global_settings = self.get_global_settings()
        return global_settings.get('data_quality_thresholds', {})
    
    def get_min_data_points_for_interval(self, interval: str) -> int:
        """
        Get minimum required data points for a specific interval.
        
        Args:
            interval: Time interval (e.g., '5m', '1h')
            
        Returns:
            Minimum data points required
        """
        thresholds = self.get_data_quality_thresholds()
        
        interval_map = {
            '5m': thresholds.get('min_data_points_5m', 2000),
            '15m': thresholds.get('min_data_points_15m', 672),
            '1h': thresholds.get('min_data_points_1h', 168),
            '4h': thresholds.get('min_data_points_4h', 42),
            '1d': thresholds.get('min_data_points_1d', 7),
        }
        
        return interval_map.get(interval, 1000)  # Default fallback
    
    def get_database_settings(self) -> Dict[str, Any]:
        """
        Get database configuration settings.
        
        Returns:
            Database settings dictionary
        """
        global_settings = self.get_global_settings()
        return global_settings.get('database_settings', {})
    
    def get_output_settings(self) -> Dict[str, Any]:
        """
        Get test output configuration settings.
        
        Returns:
            Output settings dictionary
        """
        global_settings = self.get_global_settings()
        return global_settings.get('test_output_settings', {})
    
    def validate_config_completeness(self, config: Dict[str, Any]) -> bool:
        """
        Validate that a configuration has all required fields.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is complete, False otherwise
        """
        required_fields = [
            'symbol', 'interval', 'indicator_name', 'parameters',
            'test_settings', 'benchmark_targets'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(f"⚠️ Configuration missing required fields: {missing_fields}")
            return False
        
        return True
    
    def clear_cache(self):
        """Clear the configuration cache."""
        self._config_cache.clear()
        self.global_settings = None
        logger.debug("🗑️ Configuration cache cleared")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all phases and configurations.
        
        Returns:
            Summary report dictionary
        """
        logger.info("📊 Generating comprehensive configuration summary...")
        
        summary = {
            'total_phases': 0,
            'total_configurations': 0,
            'phases': {},
            'supported_indicators': set(),
            'condition_checkers': set()
        }
        
        try:
            available_phases = self.get_available_phases()
            summary['total_phases'] = len(available_phases)
            
            for phase_name in available_phases:
                try:
                    phase_data = self.load_config_by_phase(phase_name)
                    configs = phase_data.get('indicator_test_configurations', {})
                    metadata = phase_data.get('test_metadata', {})
                    
                    phase_summary = {
                        'config_count': len(configs),
                        'supported_indicators': metadata.get('supported_indicators', []),
                        'condition_checkers': metadata.get('condition_checkers_used', []),
                        'description': metadata.get('description', ''),
                        'version': metadata.get('version', 'unknown')
                    }
                    
                    summary['phases'][phase_name] = phase_summary
                    summary['total_configurations'] += len(configs)
                    
                    # Aggregate indicators and checkers
                    summary['supported_indicators'].update(metadata.get('supported_indicators', []))
                    summary['condition_checkers'].update(metadata.get('condition_checkers_used', []))
                    
                except Exception as e:
                    logger.error(f"❌ Error processing phase '{phase_name}': {e}")
                    summary['phases'][phase_name] = {'error': str(e)}
            
            # Convert sets to lists for JSON serialization
            summary['supported_indicators'] = list(summary['supported_indicators'])
            summary['condition_checkers'] = list(summary['condition_checkers'])
            
            logger.info(f"✅ Summary generated: {summary['total_phases']} phases, {summary['total_configurations']} configurations")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating summary report: {e}")
            return {'error': str(e)}


# Singleton instance for global use
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager instance
    """
    return config_manager


if __name__ == "__main__":
    """Test the updated configuration manager with multi-phase support."""
    
    print("🧪 Testing Updated ConfigManager with Multi-Phase Support...")
    
    cm = ConfigManager()
    
    try:
        # Test 1: Get available phases
        print("\n📂 Testing available phases...")
        phases = cm.get_available_phases()
        print(f"✅ Available phases: {phases}")
        
        # Test 2: Load single phase
        if phases:
            print(f"\n📋 Testing single phase loading...")
            phase_name = phases[0]
            phase_config = cm.load_config_by_phase(phase_name)
            configs = phase_config.get('indicator_test_configurations', {})
            print(f"✅ Phase '{phase_name}' loaded: {len(configs)} configurations")
            
            # Test 3: Get phase config list
            config_list = cm.get_phase_configs_list(phase_name)
            print(f"✅ Config list for '{phase_name}': {config_list}")
        
        # Test 4: Load all phases
        print(f"\n🚀 Testing all phases loading...")
        all_phases = cm.load_all_phases()
        total_configs = sum(
            len(phase_data.get('indicator_test_configurations', {})) 
            for phase_data in all_phases.values() 
            if 'error' not in phase_data
        )
        print(f"✅ All phases loaded: {len(all_phases)} phases, {total_configs} total configurations")
        
        # Test 5: Get summary report
        print(f"\n📊 Testing summary report...")
        summary = cm.get_summary_report()
        print(f"✅ Summary: {summary['total_phases']} phases, {summary['total_configurations']} configs")
        print(f"   Supported indicators: {len(summary['supported_indicators'])}")
        print(f"   Condition checkers: {len(summary['condition_checkers'])}")
        
        # Test 6: Search specific config
        if phases and all_phases:
            print(f"\n🔍 Testing specific config search...")
            all_configs = cm.get_all_configs_list()
            if all_configs:
                # Find first available config
                for phase, configs in all_configs.items():
                    if configs:
                        config_name = configs[0]
                        try:
                            config = cm.load_specific_config(config_name)
                            print(f"✅ Config '{config_name}' found and loaded")
                            break
                        except Exception as e:
                            print(f"❌ Error loading config '{config_name}': {e}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ ConfigManager test failed: {e}")
        import traceback
        traceback.print_exc()