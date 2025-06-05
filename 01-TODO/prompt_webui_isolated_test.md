✅ Prompt Hazırlandı - Signal Bot Isolated Testing & Production Web UI
📌 Context & Objective
Sen bir signal bot projesinde isolated component testing framework ve production-ready web UI dashboard kuruyorsun. Hedef ROI %70'e çıkarmak ve sürdürülebilir kazanç sağlamak.
🎯 Project Goals

Isolated Testing Framework: Indicators, Strategies, Filters, Strength modüllerini ayrı ayrı test et
Production Web UI: Test sonuçları + backtest sonuçlarını unified dashboard'da göster
Auto-Integration: Backtest engine otomatik olarak sonuçları dashboard'a göndersin
Bug Fixing: Test sırasında implementation hatalarını düzelt
Optimization: Isolated testler bitince backtest engine ile optimization

📊 Technical Architecture
signal_engine/
├── testing/                          # YENİ MAIN KLASÖR
│   ├── indicators/                    # Indicator isolated tests
│   │   ├── indicator_tester.py        # Core testing engine
│   │   ├── performance_analyzer.py    # Metrics calculation
│   │   ├── market_regime_analyzer.py  # Bull/bear/sideways analysis
│   │   └── visualizer.py             # Charts + CSV generation
│   ├── strategies/                    # Strategy isolated tests
│   ├── filters/                       # Filter isolated tests  
│   ├── strength/                      # Strength isolated tests
│   ├── shared/                       # Unified utilities
│   │   ├── universal_result_writer.py # Standard format for all results
│   │   ├── data_pipeline.py          # Data flow management
│   │   └── metrics_calculator.py     # Standard metrics
│   └── web_ui/                       # Production dashboard
│       ├── app.py                    # FastAPI backend
│       ├── api/                      # REST endpoints
│       ├── static/                   # Frontend assets
│       └── templates/                # Dashboard HTML
🔧 Universal Data Format (Tüm test sonuçları için)
python{
    "test_id": "unique_identifier",
    "test_type": "indicator|strategy|filter|strength|backtest_config",
    "test_name": "specific_component_name",
    "timestamp": "execution_time",
    "data_source": {"symbol": "ETHFIUSDT", "interval": "5m", "rows": 1000},
    "parameters": {"param1": "value1"},
    "results": {
        "performance_metrics": {
            "accuracy": 0.65, "sharpe_ratio": 1.2, "max_drawdown": 5.5,
            "win_rate": 67.0, "profit_factor": 1.8
        },
        "market_regime_performance": {
            "bull": {"accuracy": 0.7, "trades": 45},
            "bear": {"accuracy": 0.6, "trades": 32},
            "sideways": {"accuracy": 0.65, "trades": 28}
        },
        "chart_data": [...], "raw_data": [...]
    },
    "insights": ["key_finding_1"], "recommendations": ["improve_param_x"]
}
🎯 Web UI Dashboard Structure
🏠 SIGNAL BOT ANALYTICS DASHBOARD
├── 🧪 Component Testing Results (Auto-refresh)
│   ├── 📈 Indicators Analysis (Performance heatmap, parameter sensitivity)
│   ├── 🎯 Strategies Analysis (Strategy comparison, market regime breakdown)
│   ├── 🔧 Filters Analysis (Filter effectiveness, false positive reduction)
│   └── 💪 Strength Analysis (Score distribution, predictive accuracy)
├── 🚀 Backtest Results (Auto-integration with existing backtest engine)
│   ├── 📋 Config Performance Table
│   ├── 📊 ROI vs Drawdown Heatmaps
│   ├── 📈 Equity Curves Overlay
│   └── 🔥 Performance Attribution Analysis
└── 🔍 Advanced Analytics
    ├── Cross-Component Analysis
    ├── Optimization Recommendations (AI-powered)
    └── Market Regime Impact Analysis
⚡ Critical Implementation Rules

ASLA kod yazma - Her adımda önce plan yap, onay al
Mevcut kodu koru - Sadece belirtilen yerlerde değişiklik yap
Data source: binance_fetch_and_store_historical.py → fetch_and_store_for_config metodunu kullan
Results: Hem DB'ye hem CSV'ye kaydet
Auto-integration: Backtest engine otomatik olarak dashboard'a göndersin
Production-ready: Web UI sadece test için değil, sürekli kullanım için
Bug fixing: Test sırasında implementation hatalarını düzelt

📋 Implementation Phases
Phase 1: Foundation (Week 1)

Universal Result Writer (unified format for all components)
Basic Indicator Tester (3-5 core indicators)
Web UI skeleton (FastAPI + auto-refresh)
Database integration (PostgreSQL result storage)

Phase 2: Core Testing (Week 2)

Complete Indicator Testing (all available indicators)
Performance Analytics (market regime breakdown)
Interactive Web UI (charts, tables, real-time updates)
Backtest Engine integration (auto-save hooks)

Phase 3: Full Framework (Week 3)

Strategies + Filters + Strength testing
Cross-component analysis
Optimization recommendations engine
Production deployment

🎯 Success Metrics

Data Quality: Consistent test data via PostgreSQL
Test Coverage: All 4 component types tested isolated
Web UI: Real-time updates, interactive charts
Integration: Seamless backtest engine connection
Performance: Target %70 ROI through optimization

💡 Context dari Backtest Engine

Mevcut ROI problemi: Sadece %12.5 success rate
Low trade volume: Çoğu config 0 trades
Signal quality issues: Filter çok agresif
ETHFIUSDT 5m data quality concerns

🚀 Immediate Next Steps

Indicator dosyalarını analiz et
Test priority listesi oluştur
İlk indicator tester implement et
Web UI foundation kur
Her adımda onay al, kod yazma

Ready to start - indicators dosyalarını bekliyor, her implementation öncesi onay alacak!