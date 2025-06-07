#!/usr/bin/env python3
"""
Testing Framework Web UI Startup Script
Starts dashboard with correct JSON paths
"""
import sys
import os
import time
import webbrowser
import threading
from pathlib import Path

def check_data_availability():
    """Check if dashboard data is available."""
    json_results_dir = Path("testing/results/json/")
    
    if not json_results_dir.exists():
        print("📁 Creating JSON results directory...")
        json_results_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(json_results_dir.glob("dashboard_data_*.json"))
    
    print(f"📂 JSON Results Directory: {json_results_dir.absolute()}")
    print(f"📄 Available JSON files: {len(json_files)}")
    
    if json_files:
        latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        size_mb = latest_file.stat().st_size / 1024 / 1024
        print(f"📊 Latest file: {latest_file.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("⚠️  No dashboard JSON files found!")
        print("   Run this to generate data:")
        print("   cd testing/indicators && python isolated_indicators_tester.py")
        return False

def open_browser():
    """Open browser after delay."""
    time.sleep(3)
    webbrowser.open("http://localhost:8000/dashboard")

def main():
    print("🚀 Starting Signal Bot Testing Analytics Dashboard...")
    print("=" * 60)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if we're in the right place
    if not Path("testing").exists():
        print("❌ Error: 'testing' directory not found!")
        print("   Please run from the signal_engine root directory")
        print("   Example: cd signal_engine && python testing/web_ui/start.py")
        return
    
    # Check data availability
    has_data = check_data_availability()
    
    if not has_data:
        print("\n❓ Do you want to continue anyway? (Dashboard will show 'No data' message)")
        response = input("Continue? (y/N): ").lower().strip()
        if response != 'y':
            print("Exiting. Generate data first with isolated testing.")
            return
    
    print("\n🌐 Starting web server...")
    print("📊 Dashboard URL: http://localhost:8000/dashboard")
    print("📖 API Documentation: http://localhost:8000/api/docs")
    print("🔄 Auto-refresh enabled (30s intervals)")
    print("\n⚠️  Press Ctrl+C to stop the server")
    
    # Open browser in background
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, str(current_dir))
        
        # Import and run uvicorn
        import uvicorn
        
        # Start the server
        uvicorn.run(
            "testing.web_ui.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        print("✅ Dashboard stopped successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install dependencies: pip install fastapi uvicorn pandas")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()