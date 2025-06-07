"""
Debug script to check config manager and fix loading issues
"""
import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def debug_config_system():
    """Debug the configuration system to identify issues."""
    
    print("🔍 Debugging Configuration System...")
    
    try:
        # Check if config_manager.py exists and what's in it
        config_manager_path = Path(__file__).parent / "config_manager.py"
        print(f"📁 Config manager path: {config_manager_path}")
        print(f"📁 File exists: {config_manager_path.exists()}")
        
        if config_manager_path.exists():
            with open(config_manager_path, 'r') as f:
                content = f.read()
                print(f"📄 File size: {len(content)} characters")
                print(f"🔍 Contains 'class ConfigManager': {'class ConfigManager' in content}")
                
                # Show first few lines
                lines = content.split('\n')[:10]
                print(f"📋 First 10 lines:")
                for i, line in enumerate(lines, 1):
                    print(f"   {i}: {line}")
        
        # Try importing directly
        print(f"\n🔄 Attempting direct import...")
        sys.path.insert(0, str(Path(__file__).parent))
        
        try:
            import config_manager
            print(f"✅ config_manager module imported successfully")
            print(f"📋 Available attributes: {dir(config_manager)}")
            
            if hasattr(config_manager, 'ConfigManager'):
                print(f"✅ ConfigManager class found")
                cm = config_manager.ConfigManager()
                print(f"✅ ConfigManager instance created")
            else:
                print(f"❌ ConfigManager class not found")
                
        except Exception as e:
            print(f"❌ Import failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_config_system()