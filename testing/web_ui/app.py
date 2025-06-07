"""
Testing Web UI - App.py with Correct JSON Paths
FastAPI backend that reads from testing/results/json/
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import uvicorn

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import testing modules
from testing.shared.database import db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Signal Bot Testing Analytics Dashboard",
    description="Real-time analytics dashboard for signal bot testing framework",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DashboardService:
    """Service for dashboard data operations."""
    
    def __init__(self):
        # CORRECT PATHS
        self.results_dir = Path("testing/results")
        self.json_results_dir = self.results_dir / "json"
        self.csv_results_dir = self.results_dir / "csv"
        
        # Ensure directories exist
        self.json_results_dir.mkdir(parents=True, exist_ok=True)
        self.csv_results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📂 JSON results path: {self.json_results_dir}")
        logger.info(f"📂 CSV results path: {self.csv_results_dir}")
    
    def get_latest_dashboard_file(self) -> Optional[Path]:
        """Get the latest dashboard JSON file."""
        try:
            # Look for dashboard_data_*.json files in json subfolder
            json_files = list(self.json_results_dir.glob("dashboard_data_*.json"))
            
            if not json_files:
                logger.warning(f"⚠️ No dashboard JSON files found in {self.json_results_dir}")
                return None
            
            # Sort by modification time, get latest
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            
            logger.info(f"📄 Latest dashboard file: {latest_file.name}")
            return latest_file
            
        except Exception as e:
            logger.error(f"❌ Error finding latest dashboard file: {e}")
            return None
    
    def load_dashboard_data(self, file_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load dashboard data from JSON file."""
        try:
            if file_path is None:
                file_path = self.get_latest_dashboard_file()
            
            if file_path is None or not file_path.exists():
                logger.error(f"❌ Dashboard file not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ Loaded dashboard data: {len(data.get('test_results', []))} results")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading dashboard data: {e}")
            return None
    
    def get_available_files(self) -> List[Dict[str, Any]]:
        """Get list of available dashboard files."""
        try:
            json_files = list(self.json_results_dir.glob("dashboard_data_*.json"))
            
            files_info = []
            for file_path in json_files:
                stat = file_path.stat()
                files_info.append({
                    "filename": file_path.name,
                    "full_path": str(file_path),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "execution_id": self._extract_execution_id(file_path.name)
                })
            
            # Sort by modification time, newest first
            files_info.sort(key=lambda x: x['modified'], reverse=True)
            
            return files_info
            
        except Exception as e:
            logger.error(f"❌ Error getting available files: {e}")
            return []
    
    def _extract_execution_id(self, filename: str) -> str:
        """Extract execution ID from filename."""
        try:
            # filename format: dashboard_data_EXECUTION_ID_TIMESTAMP.json
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 4:
                # Find the part that looks like an execution ID
                for i, part in enumerate(parts):
                    if 'EXEC' in part:
                        # Combine execution ID parts
                        return '_'.join(parts[2:-1])  # Skip 'dashboard_data' and timestamp
            return filename.replace('dashboard_data_', '').replace('.json', '')
        except:
            return "unknown"

# Initialize service
dashboard_service = DashboardService()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with status info."""
    files_info = dashboard_service.get_available_files()
    latest_file = dashboard_service.get_latest_dashboard_file()
    
    return {
        "message": "Signal Bot Testing Analytics Dashboard API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "data_sources": {
            "json_results_path": str(dashboard_service.json_results_dir),
            "csv_results_path": str(dashboard_service.csv_results_dir),
            "available_files": len(files_info),
            "latest_file": latest_file.name if latest_file else None
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    latest_file = dashboard_service.get_latest_dashboard_file()
    files_count = len(dashboard_service.get_available_files())
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_available": latest_file is not None,
        "files_count": files_count,
        "latest_file": latest_file.name if latest_file else None
    }

@app.get("/api/dashboard/latest")
async def get_latest_dashboard():
    """Get latest dashboard data."""
    try:
        dashboard_data = dashboard_service.load_dashboard_data()
        
        if dashboard_data is None:
            raise HTTPException(
                status_code=404, 
                detail="No dashboard data available. Run isolated testing to generate data."
            )
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"❌ Error in get_latest_dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/files")
async def get_available_files():
    """Get list of available dashboard files."""
    try:
        files_info = dashboard_service.get_available_files()
        
        return {
            "files": files_info,
            "total_count": len(files_info),
            "directory": str(dashboard_service.json_results_dir)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting available files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/file/{filename}")
async def get_dashboard_by_filename(filename: str):
    """Get dashboard data by specific filename."""
    try:
        file_path = dashboard_service.json_results_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        dashboard_data = dashboard_service.load_dashboard_data(file_path)
        
        if dashboard_data is None:
            raise HTTPException(status_code=500, detail=f"Failed to load data from: {filename}")
        
        return JSONResponse(content=dashboard_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error loading file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-results")
async def get_test_results(
    execution_id: Optional[str] = Query(None),
    indicator: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get filtered test results."""
    try:
        dashboard_data = dashboard_service.load_dashboard_data()
        
        if dashboard_data is None:
            return {"results": [], "count": 0}
        
        results = dashboard_data.get("test_results", [])
        
        # Apply filters
        if execution_id:
            target_execution = execution_id
            # Filter by execution_id if it matches metadata
            metadata = dashboard_data.get("metadata", {})
            if metadata.get("execution_id") != target_execution:
                results = []
        
        if indicator:
            results = [r for r in results if r.get("indicator") == indicator]
        
        if symbol:
            results = [r for r in results if r.get("symbol") == symbol]
        
        if status:
            results = [r for r in results if r.get("status") == status]
        
        # Apply limit
        results = results[:limit]
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"❌ Error getting test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary."""
    try:
        dashboard_data = dashboard_service.load_dashboard_data()
        
        if dashboard_data is None:
            return {"error": "No data available"}
        
        summary = dashboard_data.get("summary", {})
        test_results = dashboard_data.get("test_results", [])
        
        # Calculate additional analytics
        successful_results = [r for r in test_results if r.get("status") == "success"]
        
        if successful_results:
            accuracies = [r.get("metrics", {}).get("accuracy", 0) for r in successful_results]
            sharpe_ratios = [r.get("metrics", {}).get("sharpe_ratio", 0) for r in successful_results]
            
            analytics = {
                **summary,
                "detailed_analytics": {
                    "accuracy_stats": {
                        "min": min(accuracies) if accuracies else 0,
                        "max": max(accuracies) if accuracies else 0,
                        "avg": sum(accuracies) / len(accuracies) if accuracies else 0,
                        "median": sorted(accuracies)[len(accuracies)//2] if accuracies else 0
                    },
                    "sharpe_stats": {
                        "min": min(sharpe_ratios) if sharpe_ratios else 0,
                        "max": max(sharpe_ratios) if sharpe_ratios else 0,
                        "avg": sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
                    },
                    "performance_distribution": {
                        "excellent": len([a for a in accuracies if a >= 80]),
                        "good": len([a for a in accuracies if 70 <= a < 80]),
                        "average": len([a for a in accuracies if 60 <= a < 70]),
                        "poor": len([a for a in accuracies if a < 60])
                    }
                }
            }
        else:
            analytics = summary
        
        return analytics
        
    except Exception as e:
        logger.error(f"❌ Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/csv")
async def export_to_csv():
    """Export latest results to CSV."""
    try:
        dashboard_data = dashboard_service.load_dashboard_data()
        
        if dashboard_data is None:
            raise HTTPException(status_code=404, detail="No data available for export")
        
        test_results = dashboard_data.get("test_results", [])
        
        if not test_results:
            raise HTTPException(status_code=404, detail="No test results to export")
        
        # Create CSV content
        import pandas as pd
        
        # Flatten results for CSV
        flattened_results = []
        for result in test_results:
            flat_result = {
                "config_name": result.get("config_name", ""),
                "indicator": result.get("indicator", ""),
                "symbol": result.get("symbol", ""),
                "interval": result.get("interval", ""),
                "status": result.get("status", ""),
                "test_date": result.get("test_date", ""),
                **result.get("metrics", {})
            }
            flattened_results.append(flat_result)
        
        df = pd.DataFrame(flattened_results)
        
        # Save to temporary CSV
        execution_id = dashboard_data.get("metadata", {}).get("execution_id", "unknown")
        csv_filename = f"export_{execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = dashboard_service.csv_results_dir / csv_filename
        
        df.to_csv(csv_path, index=False)
        
        return FileResponse(
            path=csv_path,
            filename=csv_filename,
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"❌ Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static dashboard HTML
@app.get("/dashboard")
async def serve_dashboard():
    """Serve the dashboard HTML file."""
    dashboard_html = Path(__file__).parent / "dashboard.html"
    
    if dashboard_html.exists():
        return FileResponse(dashboard_html)
    else:
        return {
            "message": "Dashboard HTML not found",
            "path": str(dashboard_html),
            "api_available": True,
            "api_docs": "/api/docs"
        }

# Main route redirects to dashboard
@app.get("/dashboard/")
async def dashboard_redirect():
    """Redirect to dashboard."""
    return FileResponse(Path(__file__).parent / "dashboard.html")

if __name__ == "__main__":
    print("🚀 Starting Signal Bot Testing Analytics Dashboard...")
    print(f"📂 JSON Path: {dashboard_service.json_results_dir}")
    print(f"📂 CSV Path: {dashboard_service.csv_results_dir}")
    
    # Check for existing data
    latest_file = dashboard_service.get_latest_dashboard_file()
    if latest_file:
        print(f"📄 Latest data file: {latest_file.name}")
    else:
        print("⚠️ No dashboard data found - run isolated testing first!")
    
    print("🌐 Dashboard URL: http://localhost:8000/dashboard")
    print("📖 API docs: http://localhost:8000/api/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )