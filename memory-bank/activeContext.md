# Active Context: NinjaTrader 8 ML Strategy Deployer

## Current Work Focus
**Phase**: Project Initialization
**Sprint**: Memory Bank Setup and Foundation Planning
**Priority**: Establish development framework and begin Week 1 activities

## Immediate Next Steps (Today)
1. ✅ Initialize memory bank structure
2. 🔄 Create Git repository `nt8-ml-mvp`
3. 🔄 Export 2 years of ES 1-minute data from NinjaTrader
4. 🔄 Set up Optuna hyperparameter optimization skeleton
5. 🔄 Schedule daily 30-minute stand-ups

## Week 1 Priorities (Data & Model Design)
**Key Outcomes Needed:**
- Data pipeline finalized
- Baseline feature set defined
- Model architecture decision (TCN vs LightGBM)

**Specific Tasks:**
- Export 2-5 years of 1-minute ES data via NT Market Replay API
- Store data in parquet/feather format in `/data/es_1m/`
- Add derived columns: returns, ATR, EMA9, EMA21, VWAP, time-of-day, session flags
- Design feature engineering pipeline with Z-score normalization
- Define lagged target: `close[t+N] – close[t]` ≥ threshold → Long/Short/Flat
- Compare TCN vs LightGBM/XGBoost performance
- Select model based on F1 score, Sharpe ratio, and latency (<20ms)

## Recent Insights & Decisions
- **Model Selection Criteria**: Prioritizing inference speed (<20ms) alongside performance
- **Data Storage**: Using parquet/feather for efficient time-series data handling  
- **Validation Strategy**: Walk-forward testing to prevent look-ahead bias
- **Risk Management**: Starting with 1 MES contract, micro-live testing approach

## Active Technical Considerations
- **Latency Requirements**: Sub-250ms round-trip for signal generation and execution
- **Data Quality**: Need daily sanity checks and K-nearest fill for gaps
- **Model Artifacts**: Standardizing on .pkl (LightGBM) or .onnx (TCN) formats
- **Integration Pattern**: WebSocket bridge between Python ML service and NinjaScript

## Key Patterns Emerging
- **Development Cadence**: Weekly milestone approach with clear deliverables
- **Risk Mitigation**: Simulation → micro-live → full deployment progression
- **Architecture**: Separation of concerns (data/model/signal bridge/execution)
- **Testing Strategy**: Walk-forward validation + simulation + live testing phases

## Current Blockers/Dependencies
- None identified yet (project just initiated)

## Learning Points
- NinjaTrader 8 API integration will be critical path in Week 3
- Model selection should balance performance with operational constraints
- Walk-forward validation essential for realistic performance estimates
- CI/CD pipeline will enable rapid iteration and deployment

## Environment Status
- **Development Machine**: Windows 10 setup
- **NinjaTrader**: Version 8.1.x (need to pin for stability)
- **Git Repository**: Needs creation today
- **Data Sources**: NinjaTrader Market Replay API access required

## Communication Notes
- Daily 30-minute stand-ups planned for accountability
- Weekly milestone reviews to track progress against plan
- Go/No-Go review scheduled for Week 4 before live deployment 