# Technical Context: NinjaTrader 8 ML Strategy Deployer

## Technology Stack

### Core Technologies
**Platform**: Windows 10 (NinjaTrader 8 requirement)
**Languages**: 
- Python 3.9+ (ML pipeline, signal bridge)
- C# (NinjaScript strategy development)
- SQL (data queries and analysis)

**Machine Learning**:
- **Primary Options**: LightGBM vs Temporal Convolutional Networks (TCN)
- **Optimization**: Optuna for hyperparameter tuning
- **Deployment**: ONNX Runtime or pickle serialization
- **Validation**: scikit-learn metrics, custom walk-forward framework

**Data Pipeline**:
- **Storage**: Parquet (via pandas/pyarrow) or Feather format
- **Processing**: pandas, numpy for feature engineering
- **Technical Indicators**: TA-Lib or custom implementations

### Development Environment

#### Required Software
- **NinjaTrader 8**: Version 8.1.x (pinned for API stability)
- **Python**: 3.9+ with conda/pip environment management
- **Git**: Version control and CI/CD integration
- **IDE**: Visual Studio (NinjaScript) + VS Code/PyCharm (Python)

#### Python Dependencies
```python
# Core ML Stack
lightgbm>=3.3.0
torch>=1.12.0  # For TCN implementation
optuna>=3.0.0
scikit-learn>=1.1.0
onnxruntime>=1.12.0

# Data Processing
pandas>=1.4.0
numpy>=1.21.0
pyarrow>=8.0.0  # Parquet support
ta-lib>=0.4.0   # Technical indicators

# Communication
websockets>=10.0
aiohttp>=3.8.0
asyncio

# Development & Testing
pytest>=7.0.0
jupyter>=1.0.0
matplotlib>=3.5.0
plotly>=5.0.0
```

### NinjaTrader 8 Integration

#### NinjaScript Components
- **Strategy**: Main trading logic and signal consumption
- **AddOn**: Optional advanced UI and monitoring
- **Indicator**: Custom technical analysis if needed

#### API Access Patterns
```csharp
// NinjaScript Strategy Pattern
public class MLSignalStrategy : Strategy
{
    private WebSocketClient signalClient;
    private int contractSize = 1; // MES contracts
    
    protected override void OnMarketData(MarketDataEventArgs e)
    {
        // Forward real-time data to signal bridge
    }
    
    protected override void OnSignalReceived(string signal)
    {
        // Execute trades based on ML signals
    }
}
```

### Communication Architecture

#### Signal Bridge Service
**Protocol**: WebSocket or TCP socket
**Format**: JSON message exchange
**Deployment**: Local Python service

```python
# Signal Bridge Pattern
class SignalBridge:
    async def handle_market_data(self, bar_data):
        features = self.feature_engineer.transform(bar_data)
        signal = self.model.predict(features)
        await self.send_signal(signal)
```

### Data Infrastructure

#### File Structure
```
/data/
  └── es_1m/           # ES futures 1-minute bars
      ├── 2023/
      ├── 2024/
      └── 2025/
/models/
  ├── model.pkl        # LightGBM artifact
  ├── model.onnx       # TCN ONNX model
  └── metadata.json    # Model versioning info
/reports/
  ├── week1/           # Training metrics
  ├── week2/           # Validation results
  └── backtest/        # Performance analysis
```

#### Data Access Patterns
- **Historical**: Bulk parquet file reading
- **Real-time**: NinjaTrader 8 Market Replay API
- **Features**: Sliding window computation with caching

### Development Setup

#### Environment Configuration
```bash
# Create conda environment
conda create -n nt8-ml python=3.9
conda activate nt8-ml

# Install dependencies
pip install -r requirements.txt

# NinjaTrader path setup
export NT8_PATH="/c/Users/{user}/Documents/NinjaTrader 8"
```

#### Project Structure
```
nt8-ml-mvp/
├── src/
│   ├── data/           # Data pipeline modules
│   ├── models/         # ML model implementations  
│   ├── bridge/         # Signal bridge service
│   └── ninjatrader/    # NinjaScript files
├── notebooks/          # Jupyter analysis notebooks
├── tests/              # Unit and integration tests
├── configs/            # Configuration files
└── scripts/            # Utility and deployment scripts
```

### CI/CD Pipeline

#### GitHub Actions Workflow
```yaml
# .github/workflows/build.yml
name: Build and Deploy
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest  # NinjaTrader requirement
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Run tests
        run: pytest tests/
      - name: Train model
        run: python scripts/train_model.py
      - name: Deploy to NinjaTrader
        run: python scripts/deploy.py
```

### Performance Requirements

#### Latency Targets
- **Feature Engineering**: < 50ms per bar
- **Model Inference**: < 20ms per prediction
- **Signal Transmission**: < 30ms bridge to NinjaScript
- **Order Execution**: < 150ms NinjaScript to broker

#### Memory Constraints
- **Training**: Up to 16GB RAM for historical data processing
- **Inference**: < 512MB for live model serving
- **Data Cache**: Configurable sliding window (default 1000 bars)

### Monitoring and Logging

#### Application Monitoring
```python
# Logging Pattern
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)
```

#### Metrics Collection
- **Performance**: Latency histograms, throughput rates
- **Trading**: PnL tracking, position sizes, signal accuracy
- **System**: CPU/memory usage, network connectivity

### Security and Compliance

#### Data Protection
- Local data storage only (no cloud transmission)
- Encrypted model artifacts using platform keystore
- Secure credential management for broker API access

#### Risk Controls
```python
# Risk Management Configuration
RISK_LIMITS = {
    'max_daily_loss': 500,      # USD
    'max_position_size': 5,     # Contracts
    'max_drawdown': 0.05,       # 5%
    'cooldown_after_losses': 3  # Consecutive losing trades
}
```

### Tool Usage Patterns

#### Development Workflow
1. **Data Analysis**: Jupyter notebooks for exploratory analysis
2. **Model Development**: Python scripts with Optuna optimization
3. **Testing**: pytest for unit tests, NinjaTrader Sim for integration
4. **Deployment**: Automated scripts for model and strategy deployment

#### Model Training Pipeline
```bash
# Typical development cycle
python scripts/extract_data.py --start=2020-01-01 --end=2025-01-01
python scripts/feature_engineering.py --data-path=data/es_1m/
python scripts/train_model.py --config=configs/tcn_config.yaml
python scripts/validate_model.py --walk-forward --periods=24
``` 