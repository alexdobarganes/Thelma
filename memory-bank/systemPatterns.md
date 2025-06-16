# System Patterns: NinjaTrader 8 ML Strategy Deployer

## System Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │   ML Model       │    │  Signal Bridge  │
│                 │    │                  │    │                 │
│ • NT8 API       │───▶│ • TCN/LightGBM   │───▶│ • WebSocket     │
│ • Market Replay │    │ • Feature Eng    │    │ • TCP Client    │
│ • Parquet Store │    │ • Inference      │    │ • JSON Protocol │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CI/CD         │    │   Monitoring     │    │  NinjaScript    │
│                 │    │                  │    │                 │
│ • GitHub Actions│    │ • Performance    │    │ • Strategy      │
│ • Model Retrain │    │ • Risk Metrics   │    │ • Order Mgmt    │
│ • Auto Deploy   │    │ • System Health  │    │ • Risk Controls │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Technical Decisions

### Model Architecture Choice
**Decision Point**: TCN vs LightGBM/XGBoost
**Criteria**: 
- F1 score at zero cost
- Sharpe ratio in walk-forward testing
- Inference latency < 20ms

**Rationale**: Both handle time-series effectively while maintaining fast inference suitable for intraday execution.

### Data Pipeline Design
**Storage Format**: Parquet/Feather
- Efficient columnar storage for time-series data
- Fast read/write operations
- Cross-platform compatibility

**Feature Engineering**:
- Windowed OHLCV with technical indicators
- Z-score normalization for stability
- Lagged targets for prediction horizons

### Communication Patterns

#### Signal Bridge Protocol
```python
# WebSocket Message Format
{
    "timestamp": "2025-01-15T14:30:00Z",
    "signal": "LONG|SHORT|FLAT", 
    "confidence": 0.85,
    "features": {...},
    "model_version": "v1.2.3"
}

# Historical Data Configuration (Automatic)
{
    "type": "historical_start",
    "lookback_period": "30 Days",
    "chart_timeframe": "1 Minute", 
    "calculated_bars": 43200,
    "total_available": 5000
}
```

#### NinjaScript Integration
- Local TCP/WebSocket listener in NinjaScript
- Market order execution with configurable size
- Parameter-driven configuration system

## Design Patterns in Use

### Walk-Forward Validation Pattern
```python
# 6-month train / 1-month test rolling windows
for train_start in date_range:
    train_end = train_start + 6_months
    test_start = train_end
    test_end = test_start + 1_month
    
    model = train(data[train_start:train_end])
    metrics = validate(model, data[test_start:test_end])
```

### Circuit Breaker Pattern
- Max daily loss limits
- Cooldown after consecutive losses (3+ in a row)
- Position size limits based on account equity

### Adaptive Configuration Pattern
- **Auto-Detection**: System detects chart timeframe automatically
- **Intelligent Calculation**: Converts business terms (days/weeks) to technical parameters (bar counts)
- **Safety Boundaries**: Enforces min/max limits with fallback for edge cases
- **Transparent Operation**: Logs calculation process for user verification
- **Timeframe Agnostic**: Single configuration works across all chart types

### Producer-Consumer Pattern
- Data pipeline produces features continuously
- ML model consumes features for inference
- Signal bridge produces trading signals
- NinjaScript consumes signals for execution

## Component Relationships

### Data Flow
1. **Historical Data**: NT8 Market Replay → Parquet Files
2. **Live Data**: NT8 Real-time → Feature Engineering → Model Inference
3. **Signals**: Model Output → Signal Bridge → NinjaScript
4. **Execution**: NinjaScript → NT8 Broker API → Market

### Dependency Graph
```
NinjaScript Strategy
    ↓ depends on
Signal Bridge Service  
    ↓ depends on
Trained ML Model
    ↓ depends on  
Feature Engineering Pipeline
    ↓ depends on
Historical Data Pipeline
```

## Critical Implementation Paths

### Latency-Critical Path
**Target**: Sub-250ms round-trip latency
1. Live bar data → Feature vector (< 50ms)
2. Model inference (< 20ms)  
3. Signal transmission (< 30ms)
4. Order placement (< 150ms)

### Data Integrity Path
1. **Validation**: Daily sanity checks on incoming data
2. **Handling**: K-nearest neighbor fill for missing ticks
3. **Quality Control**: Outlier detection and filtering
4. **Backup**: Redundant data sources for critical periods

### Model Update Path
1. **Retraining**: Automated nightly retraining pipeline
2. **Validation**: A/B testing new models vs current production
3. **Deployment**: Hot-swapping models without downtime
4. **Rollback**: Quick revert to previous model version

## Scalability Considerations

### Horizontal Scaling
- Multiple model instances for different instruments
- Load balancing across signal bridges
- Distributed training for larger datasets

### Vertical Scaling  
- GPU acceleration for model training
- Optimized feature computation
- Memory-mapped data access patterns

## Error Handling Patterns

### Graceful Degradation
- Fallback to simpler models on compute failure
- Default to flat position on signal service outage
- Manual override capabilities for emergency situations

### Recovery Mechanisms
- Automatic service restart on crashes
- Data replay for missed signals
- State persistence across restarts

## Security Patterns

### API Access Control
- Encrypted credentials for NT8 API access
- Local-only signal bridge communication
- Audit logging for all trading decisions

### Data Protection
- Local data storage (no cloud transmission of signals)
- Encrypted model artifacts
- Secure communication channels 

### Adaptive Configuration Pattern
```csharp
// Intelligent Historical Data Calculation
private int CalculateHistoricalBarsFromLookback()
{
    // Auto-detect chart timeframe and calculate optimal bars
    var totalMinutes = CalculateMinutesFromLookback();
    var barsNeeded = CalculateBarsFromTimeframe(totalMinutes);
    return ApplySafetyLimits(barsNeeded);
}

// Example Results:
// 30 days + 1-minute chart → 43,200 bars
// 7 days + 5-minute chart → 2,016 bars  
// 3 months + daily chart → ~90 bars
``` 