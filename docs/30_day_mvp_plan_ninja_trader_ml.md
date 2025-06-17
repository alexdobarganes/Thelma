# 30‑Day MVP Plan – Machine‑Learning Strategy Deployer for NinjaTrader 8

**Objective**: Build, within 30 days, a minimum viable product (MVP) that can train a machine‑learning model on historical ES mini futures data from NinjaTrader 8, generate trading signals, and automatically place live/sim trades through NinjaTrader.

## High‑Level Timeline (4 Weeks)

| Week | Focus | Key Outcomes |
|------|-------|--------------|
| 1 | Data & Model Design | Data pipeline finalized, baseline feature set, model architecture decision |
| 2 | Training & Validation | Clean notebook/script that trains & cross‑validates model, walk‑forward back‑test results |
| 3 | Integration | Signal bridge converts model output → NinjaScript, Auto‑trade AddOn operational in sim |
| 4 | Testing & Launch | End‑to‑end rehearsal on sim & micro‑live, metrics collected, MVP release checklist signed off |

## Detailed Work‑Plan

### Week 1 – Data Engineering & Model Selection
1. **Data pipeline**
   * Export 2‑5 years of 1‑minute ES data via NT Market Replay API.
   * Store parquet/feather in `/data/es_1m/`.
   * Add derived columns: returns, ATR, EMA9, EMA21, VWAP, time‑of‑day, session flags.

2. **Feature engineering**
   * Windowed OHLCV + indicators (Z‑score normalized).
   * Lagged target: `close[t+N] – close[t]` ≥ threshold → Long/Short/Flat.

3. **Model architecture decision**
   * Compare:
     * **Temporal Convolutional Network (TCN)** – fast inference, robust.
     * **LightGBM / XGBoost** – tabular, interpretable, lower latency.
   * Criteria: F1 @ 0 cost, Sharpe in walk‑forward, latency < 20 ms.
   * Choose winner for MVP; keep runner‑up as fallback.

### Week 2 – Training & Validation
1. **Walk‑forward training**
   * Rolling 6‑month train / 1‑month test slices (January 2020 → May 2025).
   * Hyper‑parameter sweep with Optuna (e.g., 100 trials).

2. **Metrics dashboard**
   * Accuracy, precision, recall, PnL, MaxDD, Sharpe, avg trade duration.
   * Save plots (`/reports/week2/`) & JSON metrics.

3. **Model artifact**
   * Save as `model.pkl` (LightGBM) or `model.onnx` (TCN) in `/models/`.

### Week 3 – Platform Integration
1. **Signal bridge**
   * Python WebSocket client subscribes to NT AddOn real‑time feed.
   * Converts latest bar into feature vector, runs model, returns `LONG | SHORT | FLAT`.

2. **NinjaScript Strategy wrapper**
   * Receives signals via local TCP/WS.
   * Executes market order size = 1 MES contract; configurable via parameters.

3. **CI/CD skeleton**
   * Git repo w/ GitHub Actions: unit tests, model retrain job, deploy to `/NinjaTrader 8/bin/Custom/Strategies`.

### Week 4 – Testing, Hardening & Release
1. **Sim tests**
   * 1‑week paper trading on NT Sim101; compare live vs back‑test metrics.

2. **Micro‑live test**
   * 0.1 MES contract for 2 trading days; monitor slippage & latency.

3. **Risk controls**
   * Max daily loss, max position size, cooldown after 3 consecutive losers.

4. **Documentation & Handoff**
   * README (setup, retrain, deploy).
   * MVP demo video.
   * Go/No‑Go review.

## Deliverables Checklist
- [ ] `data_pipeline.ipynb` & ETL script  
- [ ] Feature spec & schema  
- [ ] Trained model artifact & metrics report  
- [ ] Signal bridge service (`bridge.py`)  
- [ ] NinjaScript strategy (`MLSignalStrategy.cs`)  
- [ ] CI/CD pipeline file (`.github/workflows/build.yml`)  
- [ ] User guide & demo

## Success Criteria
* Sharpe ≥ 1.2 on 2024‑2025 OOS.
* Sim latency ≤ 250 ms round‑trip.
* End‑to‑end uptime 95 % during test week.

## Risks & Mitigations
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data gaps / bad ticks | M | H | Daily sanity checks, K‑nearest fill |
| Over‑fit model | M | H | Walk‑forward, early‑stopping |
| WS latency spikes | L | M | Local caching, async queue |
| NT API changes | L | H | Pin NT8 version 8.1.x, smoke tests |

## Next Steps (Today)
1. Create Git repo `nt8‑ml‑mvp`.
2. Export 2 years of ES 1 min data.
3. Spin up Optuna sweep skeleton.
4. Schedule daily 30‑min stand‑ups.

---

## Reasoning
We prioritized a **TCN vs LightGBM** bake‑off because both handle time‑series well yet keep inference fast enough for intraday execution. A 4‑week cadence forces early data + model convergence, leaving the risky platform integration for week 3, but still allowing a full‑week soak test. Walk‑forward prevents look‑ahead bias, and micro‑live reduces capital at risk while exposing real‑time quirks. Deliverables map 1‑to‑1 with success criteria, keeping the scope tight and measurable. 🚀

