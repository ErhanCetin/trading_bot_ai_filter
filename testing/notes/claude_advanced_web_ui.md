#!/bin/bash
# Signal Bot Web UI - Complete File Structure

echo "📁 Signal Bot Web UI Complete File Structure"
echo "=============================================="

# Main project structure
cat << 'EOF'
signal_engine/
├── testing/                          # ← Existing testing framework
│   ├── shared/                       # ← Your current shared modules
│   ├── indicators/                   # ← Your current indicators folder
│   └── results/                      # ← Test results (CSV, JSON exports)
│
├── web_ui/                          # ← NEW: Web UI Dashboard
│   ├── __init__.py
│   ├── README.md                    # Setup & usage guide
│   ├── requirements.txt             # Python dependencies
│   ├── config.json                  # Dashboard configuration
│   ├── start_dashboard.py           # Main startup script
│   │
│   ├── backend/                     # FastAPI Backend
│   │   ├── __init__.py
│   │   ├── app.py                   # Main FastAPI application
│   │   ├── requirements.txt         # Backend-specific deps
│   │   │
│   │   ├── api/                     # API Routes
│   │   │   ├── __init__.py
│   │   │   ├── dashboard.py         # Dashboard endpoints
│   │   │   ├── test_results.py      # Test results API
│   │   │   ├── analytics.py         # Analytics endpoints
│   │   │   ├── export.py            # Export functionality
│   │   │   └── health.py            # Health check endpoints
│   │   │
│   │   ├── services/                # Business Logic
│   │   │   ├── __init__.py
│   │   │   ├── database_service.py  # Database operations
│   │   │   ├── file_service.py      # File-based fallback
│   │   │   ├── analytics_service.py # Analytics calculations
│   │   │   ├── chart_service.py     # Chart data generation
│   │   │   └── export_service.py    # Export operations
│   │   │
│   │   ├── models/                  # Pydantic Models
│   │   │   ├── __init__.py
│   │   │   ├── dashboard.py         # Dashboard response models
│   │   │   ├── test_result.py       # Test result models
│   │   │   ├── analytics.py         # Analytics models
│   │   │   └── common.py            # Common/shared models
│   │   │
│   │   ├── database/                # Database Integration
│   │   │   ├── __init__.py
│   │   │   ├── connection.py        # DB connection management
│   │   │   ├── queries.py           # SQL queries
│   │   │   └── models.py            # SQLAlchemy models
│   │   │
│   │   └── utils/                   # Backend Utilities
│   │       ├── __init__.py
│   │       ├── response_utils.py    # API response helpers
│   │       ├── validation.py        # Data validation
│   │       └── logging_config.py    # Logging configuration
│   │
│   ├── frontend/                    # React Frontend
│   │   ├── public/                  # Static assets
│   │   │   ├── index.html           # Main HTML template
│   │   │   ├── favicon.ico
│   │   │   └── manifest.json
│   │   │
│   │   ├── src/                     # React source code
│   │   │   ├── components/          # React Components
│   │   │   │   ├── common/          # Shared components
│   │   │   │   │   ├── Header.jsx
│   │   │   │   │   ├── LoadingSpinner.jsx
│   │   │   │   │   ├── ErrorBoundary.jsx
│   │   │   │   │   └── ExportButton.jsx
│   │   │   │   │
│   │   │   │   ├── dashboard/       # Dashboard-specific
│   │   │   │   │   ├── SummaryCards.jsx
│   │   │   │   │   ├── TabNavigation.jsx
│   │   │   │   │   └── FilterBar.jsx
│   │   │   │   │
│   │   │   │   ├── charts/          # Chart components
│   │   │   │   │   ├── AccuracyChart.jsx
│   │   │   │   │   ├── RiskReturnChart.jsx
│   │   │   │   │   ├── PerformanceChart.jsx
│   │   │   │   │   └── ComparisonChart.jsx
│   │   │   │   │
│   │   │   │   ├── tables/          # Table components
│   │   │   │   │   ├── ResultsTable.jsx
│   │   │   │   │   ├── SortableHeader.jsx
│   │   │   │   │   └── Pagination.jsx
│   │   │   │   │
│   │   │   │   └── analytics/       # Analytics components
│   │   │   │       ├── PerformanceMetrics.jsx
│   │   │   │       ├── StrategyAnalysis.jsx
│   │   │   │       └── ComparisonView.jsx
│   │   │   │
│   │   │   ├── pages/               # Page components
│   │   │   │   ├── Dashboard.jsx    # Main dashboard page
│   │   │   │   ├── Overview.jsx     # Overview tab
│   │   │   │   ├── Results.jsx      # Results tab
│   │   │   │   ├── Analytics.jsx    # Analytics tab
│   │   │   │   └── Comparison.jsx   # Comparison tab
│   │   │   │
│   │   │   ├── hooks/               # Custom React hooks
│   │   │   │   ├── useDashboardData.js
│   │   │   │   ├── useRealTimeUpdates.js
│   │   │   │   ├── useFilters.js
│   │   │   │   └── useExport.js
│   │   │   │
│   │   │   ├── services/            # Frontend services
│   │   │   │   ├── api.js           # API client
│   │   │   │   ├── chartService.js  # Chart utilities
│   │   │   │   └── exportService.js # Export utilities
│   │   │   │
│   │   │   ├── utils/               # Frontend utilities
│   │   │   │   ├── formatters.js    # Data formatters
│   │   │   │   ├── constants.js     # App constants
│   │   │   │   └── helpers.js       # Helper functions
│   │   │   │
│   │   │   ├── styles/              # CSS/Styling
│   │   │   │   ├── globals.css      # Global styles
│   │   │   │   ├── components.css   # Component styles
│   │   │   │   └── charts.css       # Chart-specific styles
│   │   │   │
│   │   │   ├── App.jsx              # Main App component
│   │   │   └── index.js             # React entry point
│   │   │
│   │   ├── package.json             # NPM dependencies
│   │   ├── vite.config.js           # Vite configuration
│   │   ├── tailwind.config.js       # Tailwind CSS config
│   │   └── .gitignore               # Git ignore rules
│   │
│   ├── static/                      # Static files served by backend
│   │   ├── images/                  # Static images
│   │   ├── icons/                   # Icon files
│   │   └── exports/                 # Generated export files
│   │
│   ├── templates/                   # Jinja2 templates (if needed)
│   │   ├── base.html
│   │   └── dashboard.html
│   │
│   ├── tests/                       # Web UI Tests
│   │   ├── __init__.py
│   │   ├── test_api.py              # API endpoint tests
│   │   ├── test_services.py         # Service layer tests
│   │   ├── test_database.py         # Database tests
│   │   └── frontend/                # Frontend tests
│   │       ├── Dashboard.test.jsx
│   │       └── components/
│   │
│   ├── docs/                        # Documentation
│   │   ├── API.md                   # API documentation
│   │   ├── DEPLOYMENT.md            # Deployment guide
│   │   ├── DEVELOPMENT.md           # Development setup
│   │   └── screenshots/             # Dashboard screenshots
│   │
│   ├── docker/                      # Docker configuration
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── docker-compose.prod.yml
│   │   └── nginx.conf               # Nginx config for production
│   │
│   └── scripts/                     # Utility scripts
│       ├── setup.sh                 # Initial setup script
│       ├── deploy.sh                # Deployment script
│       ├── backup.sh                # Backup script
│       └── migrate.py               # Database migration
│
└── backtest_engine/                 # ← Existing backtest engine
    └── ...                          # Your existing backtest code

EOF

echo ""
echo "📋 Key Files Details:"
echo "===================="

# Create file details
cat << 'EOF'

🔧 BACKEND FILES:
├── app.py                    # Main FastAPI app (150 lines)
├── api/dashboard.py          # Dashboard endpoints (100 lines)
├── api/test_results.py       # Results API (120 lines)
├── api/analytics.py          # Analytics API (80 lines)
├── services/database_service.py  # DB operations (200 lines)
├── services/analytics_service.py # Analytics logic (150 lines)
└── models/dashboard.py       # Pydantic models (80 lines)

🎨 FRONTEND FILES:
├── public/index.html         # Main HTML (single-file React setup)
├── src/App.jsx              # Main React app (100 lines)
├── src/components/dashboard/SummaryCards.jsx  (80 lines)
├── src/components/charts/AccuracyChart.jsx    (60 lines)
├── src/components/tables/ResultsTable.jsx     (120 lines)
├── src/hooks/useDashboardData.js              (50 lines)
└── src/services/api.js       # API client (40 lines)

⚙️ CONFIGURATION FILES:
├── config.json               # Dashboard config (30 lines)
├── requirements.txt          # Python deps (10 lines)
├── package.json              # NPM deps (25 lines)
├── docker-compose.yml        # Docker setup (40 lines)
└── start_dashboard.py        # Startup script (50 lines)

📚 DOCUMENTATION:
├── README.md                 # Main setup guide (100 lines)
├── docs/API.md              # API documentation (200 lines)
├── docs/DEPLOYMENT.md        # Deployment guide (150 lines)
└── docs/DEVELOPMENT.md       # Dev setup (100 lines)

EOF

echo ""
echo "🚀 Implementation Approach:"
echo "=========================="

cat << 'EOF'

PHASE 1: Core Backend (Priority 1)
✅ FastAPI app with basic endpoints
✅ Database integration
✅ File-based fallback
✅ Health checks

PHASE 2: Basic Frontend (Priority 1)
✅ Single-file React dashboard
✅ Summary cards
✅ Results table
✅ Basic charts

PHASE 3: Advanced Features (Priority 2)
□ Real-time updates
□ Advanced analytics
□ Strategy comparison
□ Export functionality

PHASE 4: Production Ready (Priority 3)
□ Docker setup
□ Nginx configuration
□ Performance optimization
□ Comprehensive testing

EOF

echo ""
echo "📁 Directory Creation Commands:"
echo "=============================="

# Directory creation script
cat << 'EOF'

# Create main structure
mkdir -p signal_engine/web_ui/{backend,frontend,static,templates,tests,docs,docker,scripts}

# Backend structure
mkdir -p signal_engine/web_ui/backend/{api,services,models,database,utils}

# Frontend structure (if using separate React build)
mkdir -p signal_engine/web_ui/frontend/{public,src}
mkdir -p signal_engine/web_ui/frontend/src/{components,pages,hooks,services,utils,styles}
mkdir -p signal_engine/web_ui/frontend/src/components/{common,dashboard,charts,tables,analytics}

# Static and template directories
mkdir -p signal_engine/web_ui/static/{images,icons,exports}
mkdir -p signal_engine/web_ui/tests/frontend

# Documentation
mkdir -p signal_engine/web_ui/docs/screenshots

EOF

echo ""
echo "💡 RECOMMENDED START:"
echo "===================="
echo "1. Create basic structure"
echo "2. Implement single-file React dashboard (simpler to start)"
echo "3. Add FastAPI backend with database integration"
echo "4. Test with existing test results"
echo "5. Gradually add advanced features"
echo ""
echo "🎯 This structure supports both rapid prototyping and production scaling!"