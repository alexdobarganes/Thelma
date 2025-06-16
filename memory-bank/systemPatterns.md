# System Patterns: NinjaTrader 8 ML Strategy Deployer

## System Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │   ML Model       │    │  Signal Bridge  │
│                 │    │                  │    │                 │
│ • NT8 API       │───▶│ • TCN/LightGBM   │───▶│ • WebSocket     │
│ • Market Replay │    │ • Feature Eng    │    │ • TCP Client    │
│ • CSV Store     │    │ • Inference      │    │ • JSON Protocol │
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
**Storage Format**: CSV/OHLCV Standard
- Industry-standard OHLCV format for ML compatibility
- Fast read/write operations with background streaming
- Cross-platform compatibility with all ML tools
- Timezone-aware timestamp formatting

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
    "total_available": 1051200
}
```

#### NinjaScript Integration
- Local TCP/WebSocket listener in NinjaScript
- Market order execution with configurable size
- Parameter-driven configuration system

## Design Patterns in Use

### Ultra-Performance Architecture Pattern
```python
# High-Performance Mode for Large Dataset Processing
class HighPerformanceMode:
    def __init__(self):
        self.historical_loading = False
        self.background_writer = BackgroundCSVWriter()
        self.priority_queue = PriorityMessageQueue()
    
    async def handle_massive_dataset(self, data_stream):
        # Enable high-performance mode
        self.historical_loading = True
        
        # Priority handling for connection maintenance
        if message_type == "ping":
            await self.priority_response("pong")
        
        # Background processing for non-critical operations
        await self.background_writer.queue_data(data)
        
        # Minimal processing during historical load
        if not self.historical_loading:
            await self.full_statistics_processing()
```

### Background Threading Pattern
```python
# Non-blocking CSV Writing with Queue-Based Architecture
class HighSpeedCSVWriter:
    def __init__(self):
        self.write_queue = queue.Queue(maxsize=20000)
        self.writer_thread = threading.Thread(target=self._background_writer)
        
    def write_record(self, data):
        # Non-blocking queue operation
        self.write_queue.put_nowait(data.to_csv_row())
        
    def _background_writer(self):
        # Background thread processes queue in batches
        while not shutdown:
            batch = self._collect_batch(2000)  # Large batches
            self._write_batch_to_file(batch)
```

### Adaptive Processing Pattern
```python
# System Automatically Adjusts Performance Based on Load
class AdaptiveProcessor:
    def process_message(self, message, context):
        if context.historical_loading and context.high_performance_mode:
            # Minimal processing during historical load
            return self.minimal_processing(message)
        else:
            # Full processing with statistics and UI updates
            return self.full_processing(message)
```

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

### Priority Queue Pattern
```python
# Critical Messages Bypass Normal Processing
class PriorityMessageHandler:
    def handle_message(self, message):
        if message.type == "ping":
            # Immediate priority response
            await self.immediate_pong_response()
        elif message.type == "critical_signal":
            # High priority trading signal
            await self.priority_signal_processing()
        else:
            # Normal queue processing
            await self.normal_queue.put(message)
```

## Component Relationships

### Data Flow
1. **Historical Data**: NT8 Market Replay → CSV Files (Background Streaming)
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

### Ultra-Performance Data Path (NEW)
**Target**: Handle 1M+ historical bars without disconnection
1. **Message Reception** → Priority Queue (< 1ms)
2. **Ping/Pong Handling** → Immediate Response (< 5ms)
3. **Data Processing** → Background Thread (Non-blocking)
4. **CSV Writing** → Queued Batch Operations (< 10ms per batch)
5. **Memory Management** → Constant Footprint (No growth)

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

## Ultra-Performance Patterns (NEW)

### Connection Resilience Pattern
```python
# Enhanced Connection Management for High-Volume Data
class ResilientConnection:
    def __init__(self):
        self.ping_interval = 30  # Increased for high-volume
        self.ping_timeout = 25   # Increased timeout
        self.max_message_size = 2**20  # 1MB for data bursts
        self.connection_queue = 50     # Smaller for faster processing
        
    async def maintain_connection(self):
        # Priority ping/pong handling
        while connected:
            if await self.priority_ping_check():
                await self.immediate_pong_response()
```

### Memory Optimization Pattern
```python
# Constant Memory Usage Regardless of Dataset Size
class MemoryOptimizedProcessor:
    def __init__(self):
        self.display_queue = deque(maxlen=500)  # Fixed size
        self.background_processor = BackgroundThread()
        
    def process_large_dataset(self, data):
        # Only keep recent data for display
        if not self.high_performance_mode or self.record_count % 50 == 0:
            self.display_queue.append(data)
        
        # All data goes to background processing
        self.background_processor.queue(data)
```

### Adaptive UI Pattern
```python
# UI Adapts to Processing Load
class AdaptiveUI:
    def update_display(self):
        if self.historical_loading and self.high_performance_mode:
            # Simplified display during high-volume processing
            await self.simple_progress_display()
            await asyncio.sleep(5.0)  # Reduced frequency
        else:
            # Full rich display during normal operation
            await self.full_statistics_display()
            await asyncio.sleep(0.5)  # Normal frequency
``` 