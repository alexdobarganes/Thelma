# Project Brief: NinjaTrader 8 ML Strategy Deployer

## Project Overview
Build a minimum viable product (MVP) that can train a machine-learning model on historical ES mini futures data from NinjaTrader 8, generate trading signals, and automatically place live/sim trades through NinjaTrader.

## Core Objective
Create an end-to-end automated trading system that:
1. Trains ML models on historical ES futures data
2. Generates real-time trading signals
3. Executes trades automatically through NinjaTrader 8
4. Achieves measurable profitability with controlled risk

## Success Criteria
- **Performance**: Sharpe ratio ≥ 1.2 on 2024-2025 out-of-sample data
- **Speed**: Signal latency ≤ 250ms round-trip
- **Reliability**: End-to-end uptime 95% during test week
- **Risk Management**: Max daily loss controls, position sizing limits

## Timeline Commitment
30-day development cycle with weekly milestones:
- Week 1: Data & Model Design
- Week 2: Training & Validation  
- Week 3: Integration with NinjaTrader
- Week 4: Testing & Launch

## Key Deliverables
1. Data pipeline for ES futures data extraction
2. Trained ML model (TCN or LightGBM)
3. Signal bridge service connecting model to NinjaTrader
4. NinjaScript strategy for trade execution
5. CI/CD pipeline for model deployment
6. Comprehensive documentation and user guide

## Project Scope
**In Scope:**
- ES mini futures (MES/ES) trading only
- 1-minute bar data processing
- Simulation and micro-live testing
- Basic risk controls and position management

**Out of Scope:**
- Multi-asset trading
- Options or complex derivatives
- Advanced portfolio optimization
- High-frequency (sub-minute) strategies

## Definition of Done
MVP is complete when:
- Model trains successfully on historical data
- Signals generate in real-time with acceptable latency
- Trades execute automatically in simulation
- Risk controls prevent excessive losses
- System runs reliably for one full trading week
- Documentation allows independent setup and operation 