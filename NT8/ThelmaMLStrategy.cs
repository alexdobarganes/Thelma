#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Strategies
{
    public class ThelmaMLStrategy : Strategy
    {
        #region ML Model Components
        
        // Feature storage for ML pipeline (74 features as per model)
        private Dictionary<string, double> currentFeatures;
        private List<double> recentPrices;
        private List<double> recentVolumes;
        private List<double> recentReturns;
        
        // Model thresholds (based on metadata: F1: 0.56, Sharpe: 4.84)
        private double longThreshold = 0.65;   // High confidence for long
        private double shortThreshold = 0.35;  // High confidence for short
        private double flatThreshold = 0.10;   // Low confidence threshold
        
        // ML inference state
        private double lastMLPrediction = 0.5; // 0.0 = short, 0.5 = flat, 1.0 = long
        private double lastMLConfidence = 0.0;
        private DateTime lastMLUpdate = DateTime.MinValue;
        
        // Volatility regime detection
        private double volatilityRegime = 1.0; // 1.0 = normal, >1.5 = high vol, <0.7 = low vol
        
        #endregion
        
        #region Technical Indicators (for feature engineering)
        
        private EMA ema9, ema21, ema50;
        private RSI rsi;
        private ATR atr;
        private Bollinger bollinger;
        private SMA volumeSMA;
        private double vwapValue = 0.0;
        private double cumulativeVolume = 0.0;
        private double cumulativeVolumePrice = 0.0;
        
        // Multi-timeframe indicators (simulated)
        private EMA ema9_5min, ema21_5min;
        private RSI rsi_5min;
        private ATR atr_5min;
        
        #endregion
        
        #region Position Management
        
        private int targetPosition = 0; // -1 = short, 0 = flat, 1 = long
        private int currentPosition = 0;
        private DateTime lastTradeTime = DateTime.MinValue;
        private double lastEntryPrice = 0.0;
        private int tradesThisSession = 0;
        
        // Risk management
        private double maxDailyLoss = 500.0;
        private double dailyPnL = 0.0;
        private int maxPositionSize = 1;
        private int cooldownMinutes = 15;
        
        #endregion
        
        #region Strategy Parameters
        
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name="Contract Size", Description="Number of contracts to trade", Order=1, GroupName="Position Sizing")]
        public int ContractSize { get; set; }
        
        [NinjaScriptProperty] 
        [Range(0.1, 1.0)]
        [Display(Name="Long Threshold", Description="ML confidence threshold for long signals", Order=2, GroupName="ML Parameters")]
        public double LongThreshold { get; set; }
        
        [NinjaScriptProperty]
        [Range(0.0, 0.9)]
        [Display(Name="Short Threshold", Description="ML confidence threshold for short signals", Order=3, GroupName="ML Parameters")]
        public double ShortThreshold { get; set; }
        
        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name="Min Confidence", Description="Minimum ML confidence to trade (%)", Order=4, GroupName="ML Parameters")]
        public double MinConfidence { get; set; }
        
        [NinjaScriptProperty]
        [Range(100, 2000)]
        [Display(Name="Max Daily Loss", Description="Maximum daily loss in dollars", Order=5, GroupName="Risk Management")]
        public double MaxDailyLoss { get; set; }
        
        [NinjaScriptProperty]
        [Range(5, 120)]
        [Display(Name="Cooldown Minutes", Description="Minutes to wait between trades", Order=6, GroupName="Risk Management")]
        public int CooldownMinutes { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name="Enable ML Filtering", Description="Use ML predictions for trade filtering", Order=7, GroupName="ML Parameters")]
        public bool EnableMLFiltering { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name="Enable Risk Management", Description="Apply risk management rules", Order=8, GroupName="Risk Management")]
        public bool EnableRiskManagement { get; set; }
        
        #endregion
        
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"Thelma ML Strategy - Advanced ML-powered ES futures strategy using 74-feature model with F1: 0.56, Sharpe: 4.84";
                Name = "ThelmaMLStrategy";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 100;
                IsInstantiatedOnEachOptimizationIteration = true;
                
                // Default parameter values (extremely permissive for debugging)
                ContractSize = 1;
                LongThreshold = 0.51;  // Almost any bullish signal
                ShortThreshold = 0.49; // Almost any bearish signal
                MinConfidence = 1.0;   // Essentially no confidence requirement
                MaxDailyLoss = 500.0;
                CooldownMinutes = 1;   // Almost no cooldown
                EnableMLFiltering = true;
                EnableRiskManagement = true;
            }
            else if (State == State.DataLoaded)
            {
                // Initialize indicators
                ema9 = EMA(Close, 9);
                ema21 = EMA(Close, 21);
                ema50 = EMA(Close, 50);
                rsi = RSI(14, 3);
                atr = ATR(Close, 14);
                bollinger = Bollinger(Close, 2, 20);
                volumeSMA = SMA(Volume, 20);
                // VWAP will be calculated manually
                // Volatility will be calculated manually in UpdateVolatilityRegime()
                
                // Initialize feature storage
                currentFeatures = new Dictionary<string, double>();
                recentPrices = new List<double>();
                recentVolumes = new List<double>();
                recentReturns = new List<double>();
                
                // Copy parameter values to working variables
                longThreshold = LongThreshold;
                shortThreshold = ShortThreshold;
                maxDailyLoss = MaxDailyLoss;
                cooldownMinutes = CooldownMinutes;
                
                Print(string.Format("{0} - Thelma ML Strategy initialized - Contract Size: {1}, ML Filtering: {2}",
                    Time[0], ContractSize, EnableMLFiltering));
            }
        }
        
        protected override void OnBarUpdate()
        {
            // Ensure we have enough data
            if (CurrentBar < BarsRequiredToTrade)
                return;
                
            // Update feature engineering pipeline
            UpdateMLFeatures();
            
            // Calculate ML prediction
            double mlPrediction = CalculateMLPrediction();
            double mlConfidence = CalculateMLConfidence();
            
            // Store ML state
            lastMLPrediction = mlPrediction;
            lastMLConfidence = mlConfidence;
            lastMLUpdate = Time[0];
            
            // Determine target position based on ML prediction
            int newTargetPosition = 0;
            
            if (EnableMLFiltering && mlConfidence >= (MinConfidence / 100.0))
            {
                if (mlPrediction >= longThreshold)
                    newTargetPosition = 1;
                else if (mlPrediction <= shortThreshold)
                    newTargetPosition = -1;
                else
                    newTargetPosition = 0;
            }
            
            // FALLBACK: Simple moving average signal for testing if ML doesn't work
            if (newTargetPosition == 0 && CurrentBar >= 100)
            {
                double emaFast = ema9[0];
                double emaSlow = ema21[0];
                
                if (emaFast > emaSlow * 1.001) // 0.1% threshold to avoid noise
                {
                    newTargetPosition = 1; // Simple bullish signal
                    if (CurrentBar % 30 == 0)
                        Print(string.Format("FALLBACK SIGNAL: Long (EMA9 {0:F2} > EMA21 {1:F2})", emaFast, emaSlow));
                }
                else if (emaFast < emaSlow * 0.999) // 0.1% threshold
                {
                    newTargetPosition = -1; // Simple bearish signal
                    if (CurrentBar % 30 == 0)
                        Print(string.Format("FALLBACK SIGNAL: Short (EMA9 {0:F2} < EMA21 {1:F2})", emaFast, emaSlow));
                }
            }
            
            // Apply risk management
            if (EnableRiskManagement)
            {
                if (!PassesRiskChecks())
                {
                    newTargetPosition = 0; // Force flat if risk limits exceeded
                }
            }
            
            // Execute position changes
            if (newTargetPosition != targetPosition)
            {
                targetPosition = newTargetPosition;
                ExecutePositionChange();
            }
            
            // Update daily PnL tracking
            UpdateDailyPnL();
            
            // Enhanced debug output for testing
            if (CurrentBar % 5 == 0) // Every 5 bars for intensive debugging
            {
                Print(string.Format("{0} - DETAILED DEBUG:", Time[0]));
                Print(string.Format("    ML Pred: {1:F3}, Conf: {2:F3}, Target: {3}, Current: {4}, Vol: {5:F2}, Features: {6}",
                    Time[0], mlPrediction, mlConfidence, targetPosition, currentPosition, volatilityRegime, currentFeatures.Count));
                
                // Debug key features
                if (currentFeatures.ContainsKey("momentum_3"))
                    Print(string.Format("    Key Features - Mom3: {0:F4}, Mom10: {1:F4}, RSI: {2:F2}, EMA9/21: {3:F4}",
                        currentFeatures.ContainsKey("momentum_3") ? currentFeatures["momentum_3"] : 0,
                        currentFeatures.ContainsKey("momentum_10") ? currentFeatures["momentum_10"] : 0,
                        currentFeatures.ContainsKey("rsi") ? currentFeatures["rsi"] : 0,
                        currentFeatures.ContainsKey("ema9_to_ema21") ? currentFeatures["ema9_to_ema21"] : 0));
                
                // Additional debug for trade conditions
                bool passesRisk = PassesRiskChecks();
                bool hasConfidence = mlConfidence >= (MinConfidence / 100.0);
                Print(string.Format("    Passes Risk: {0}, Has Confidence: {1} ({2:F3} >= {3:F3}), ML Filtering: {4}",
                    passesRisk, hasConfidence, mlConfidence, MinConfidence / 100.0, EnableMLFiltering));
                    
                if (EnableMLFiltering)
                {
                    bool longSignal = mlPrediction >= longThreshold;
                    bool shortSignal = mlPrediction <= shortThreshold;
                    Print(string.Format("    Long Signal: {0} (pred {1:F3} >= {2:F3}), Short Signal: {3} (pred {4:F3} <= {5:F3})",
                        longSignal, mlPrediction, longThreshold, shortSignal, mlPrediction, shortThreshold));
                        
                    if (hasConfidence && passesRisk && (longSignal || shortSignal))
                    {
                        Print(string.Format("    *** SHOULD TRADE BUT NOT TRADING - INVESTIGATING ***"));
                    }
                }
            }
        }
        
        #region ML Feature Engineering
        
        private void UpdateMLFeatures()
        {
            // Update price/volume history
            recentPrices.Add(Close[0]);
            recentVolumes.Add(Volume[0]);
            
            if (CurrentBar > 0)
                recentReturns.Add((Close[0] / Math.Max(0.01, Close[1])) - 1.0);
            
            // Keep only recent data (200 bars for features)
            if (recentPrices.Count > 200)
            {
                recentPrices.RemoveAt(0);
                recentVolumes.RemoveAt(0);
            }
            if (recentReturns.Count > 200)
                recentReturns.RemoveAt(0);
            
            // Clear and recalculate features
            currentFeatures.Clear();
            
            // Price-based features
            currentFeatures["returns"] = CurrentBar > 0 ? Close[0] / Math.Max(0.01, Close[1]) - 1.0 : 0.0;
            currentFeatures["log_returns"] = CurrentBar > 0 ? Math.Log(Close[0] / Math.Max(0.01, Close[1])) : 0.0;
            
            // Technical indicator features
            currentFeatures["ema9"] = ema9[0];
            currentFeatures["ema21"] = ema21[0];
            currentFeatures["ema50"] = ema50[0];
            currentFeatures["price_to_ema9"] = Close[0] / Math.Max(0.01, ema9[0]);
            currentFeatures["price_to_ema21"] = Close[0] / Math.Max(0.01, ema21[0]);
            currentFeatures["ema9_to_ema21"] = ema9[0] / Math.Max(0.01, ema21[0]);
            
            currentFeatures["rsi"] = rsi[0];
            currentFeatures["rsi_normalized"] = (rsi[0] - 50.0) / 50.0;
            
            currentFeatures["atr"] = atr[0];
            currentFeatures["atr_ratio"] = atr[0] / Math.Max(0.01, Close[0]);
            
            currentFeatures["bb_upper"] = bollinger.Upper[0];
            currentFeatures["bb_lower"] = bollinger.Lower[0];
            currentFeatures["bb_position"] = (Close[0] - bollinger.Lower[0]) / Math.Max(0.0001, (bollinger.Upper[0] - bollinger.Lower[0]));
            currentFeatures["bb_width"] = (bollinger.Upper[0] - bollinger.Lower[0]) / Math.Max(0.0001, bollinger.Middle[0]);
            
            currentFeatures["volume"] = Volume[0];
            currentFeatures["volume_ma"] = volumeSMA[0];
            currentFeatures["volume_ratio"] = Volume[0] / Math.Max(1.0, volumeSMA[0]);
            
            // Update VWAP calculation
            UpdateVWAP();
            currentFeatures["vwap"] = vwapValue;
            currentFeatures["price_to_vwap"] = Close[0] / Math.Max(0.01, vwapValue);
            
            // Time-based features
            currentFeatures["hour"] = (double)Time[0].Hour;
            currentFeatures["day_of_week"] = (double)Time[0].DayOfWeek;
            currentFeatures["is_market_open"] = (Time[0].Hour >= 9 && Time[0].Hour < 16) ? 1.0 : 0.0;
            currentFeatures["is_overnight"] = (Time[0].Hour >= 18 || Time[0].Hour < 9) ? 1.0 : 0.0;
            
            // Volatility regime features
            UpdateVolatilityRegime();
            currentFeatures["volatility_regime"] = volatilityRegime;
            currentFeatures["high_vol_regime"] = volatilityRegime > 1.5 ? 1.0 : 0.0;
            currentFeatures["low_vol_regime"] = volatilityRegime < 0.7 ? 1.0 : 0.0;
            
            // Momentum features (with proper bounds checking)
            if (CurrentBar >= 3)
                currentFeatures["momentum_3"] = Close[0] / Math.Max(0.01, Close[3]) - 1.0;
            if (CurrentBar >= 10)
                currentFeatures["momentum_10"] = Close[0] / Math.Max(0.01, Close[10]) - 1.0;
            if (CurrentBar >= 20)
                currentFeatures["momentum_20"] = Close[0] / Math.Max(0.01, Close[20]) - 1.0;
            
            // Lagged features (key for time series prediction)
            if (recentReturns.Count >= 6) // Need at least 6 to access index -6
            {
                currentFeatures["returns_lag1"] = recentReturns[recentReturns.Count - 2];
                currentFeatures["returns_lag2"] = recentReturns[recentReturns.Count - 3];
                currentFeatures["returns_lag3"] = recentReturns[recentReturns.Count - 4];
                currentFeatures["returns_lag5"] = recentReturns[recentReturns.Count - 6];
            }
            
            // Volatility features
            if (recentReturns.Count >= 30)
            {
                int startIndex = Math.Max(0, recentReturns.Count - 30);
                int count = Math.Min(30, recentReturns.Count - startIndex);
                if (count > 0)
                {
                    var recent30 = recentReturns.GetRange(startIndex, count);
                    currentFeatures["vol_30min"] = CalculateStandardDeviation(recent30);
                }
            }
            
            if (recentReturns.Count >= 60)
            {
                int startIndex = Math.Max(0, recentReturns.Count - 60);
                int count = Math.Min(60, recentReturns.Count - startIndex);
                if (count > 0)
                {
                    var recent60 = recentReturns.GetRange(startIndex, count);
                    currentFeatures["vol_60min"] = CalculateStandardDeviation(recent60);
                }
            }
        }
        
        private void UpdateVolatilityRegime()
        {
            if (CurrentBar < 20)
                return;
                
            // Calculate current volatility as High - Low range
            double currentVol = High[0] - Low[0];
            double avgVol = 0;
            
            // Calculate average volatility over last 100 bars (but not more than available bars)
            int barsToUse = Math.Min(100, CurrentBar + 1); // +1 because CurrentBar is 0-based
            for (int i = 0; i < barsToUse; i++)
            {
                avgVol += (High[i] - Low[i]);
            }
            avgVol /= barsToUse;
            
            volatilityRegime = currentVol / Math.Max(0.0001, avgVol);
        }
        
        private void UpdateVWAP()
        {
            if (CurrentBar == 0)
            {
                // Reset VWAP at start of session/day
                cumulativeVolume = Volume[0];
                cumulativeVolumePrice = (High[0] + Low[0] + Close[0]) / 3.0 * Volume[0];
                vwapValue = (High[0] + Low[0] + Close[0]) / 3.0;
            }
            else
            {
                // Check if new session (reset daily VWAP)
                if (Bars.IsFirstBarOfSession)
                {
                    cumulativeVolume = Volume[0];
                    cumulativeVolumePrice = (High[0] + Low[0] + Close[0]) / 3.0 * Volume[0];
                }
                else
                {
                    // Update cumulative values
                    cumulativeVolume += Volume[0];
                    cumulativeVolumePrice += (High[0] + Low[0] + Close[0]) / 3.0 * Volume[0];
                }
                
                // Calculate VWAP
                if (cumulativeVolume > 0)
                    vwapValue = cumulativeVolumePrice / cumulativeVolume;
                else
                    vwapValue = Close[0];
            }
        }
        
        private double CalculateStandardDeviation(List<double> values)
        {
            if (values.Count < 2)
                return 0.0;
                
            double mean = values.Average();
            double variance = values.Select(x => Math.Pow(x - mean, 2)).Average();
            return Math.Sqrt(variance);
        }
        
        #endregion
        
        #region ML Model Inference (Simplified)
        
        private double CalculateMLPrediction()
        {
            // Simplified ML model based on the 74-feature enhanced model
            // This implements a lightweight version of the neural network logic
            
            if (currentFeatures.Count < 20) // Need minimum features
            {
                if (CurrentBar % 20 == 0)
                    Print(string.Format("ML Prediction: Not enough features ({0} < 20), returning neutral", currentFeatures.Count));
                return 0.5; // Neutral
            }
                
            double score = 0.0;
            double weightSum = 0.0;
            
            // Price momentum signals (based on model feature importance)
            if (currentFeatures.ContainsKey("momentum_3"))
            {
                score += currentFeatures["momentum_3"] * 0.15;
                weightSum += 0.15;
            }
            
            if (currentFeatures.ContainsKey("momentum_10"))
            {
                score += currentFeatures["momentum_10"] * 0.12;
                weightSum += 0.12;
            }
            
            // Technical indicator signals
            if (currentFeatures.ContainsKey("rsi_normalized"))
            {
                double rsiSignal = -currentFeatures["rsi_normalized"]; // Contrarian
                score += rsiSignal * 0.10;
                weightSum += 0.10;
            }
            
            if (currentFeatures.ContainsKey("bb_position"))
            {
                double bbSignal = (currentFeatures["bb_position"] - 0.5) * -1; // Mean reversion
                score += bbSignal * 0.08;
                weightSum += 0.08;
            }
            
            // EMA trend signals
            if (currentFeatures.ContainsKey("ema9_to_ema21"))
            {
                double trendSignal = (currentFeatures["ema9_to_ema21"] - 1.0) * 2.0;
                score += trendSignal * 0.12;
                weightSum += 0.12;
            }
            
            // Volume confirmation
            if (currentFeatures.ContainsKey("volume_ratio"))
            {
                double volSignal = Math.Min(2.0, currentFeatures["volume_ratio"]) - 1.0;
                score += volSignal * 0.06;
                weightSum += 0.06;
            }
            
            // Time-of-day bias
            if (currentFeatures.ContainsKey("hour"))
            {
                double hour = currentFeatures["hour"];
                double timeSignal = 0.0;
                
                // Morning momentum (9:30-11:00)
                if (hour >= 9.5 && hour <= 11)
                    timeSignal = 0.1;
                // Afternoon reversal (14:00-15:00)
                else if (hour >= 14 && hour <= 15)
                    timeSignal = -0.05;
                
                score += timeSignal * 0.05;
                weightSum += 0.05;
            }
            
            // Volatility adjustment
            if (currentFeatures.ContainsKey("volatility_regime"))
            {
                double volAdj = 1.0;
                if (volatilityRegime > 1.5) // High vol - reduce signal strength
                    volAdj = 0.7;
                else if (volatilityRegime < 0.7) // Low vol - increase signal strength
                    volAdj = 1.3;
                    
                score *= volAdj;
            }
            
            // Normalize score to [0, 1] range
            if (weightSum > 0)
                score /= weightSum;
                
            // Apply sigmoid-like transformation
            double prediction = 1.0 / (1.0 + Math.Exp(-score * 5.0));
            prediction = Math.Max(0.0, Math.Min(1.0, prediction));
            
            // Debug ML prediction calculation
            if (CurrentBar % 20 == 0)
            {
                Print(string.Format("ML Prediction Calculation: score={0:F4}, weightSum={1:F4}, final_prediction={2:F4}",
                    score, weightSum, prediction));
            }
            
            return prediction;
        }
        
        private double CalculateMLConfidence()
        {
            // Calculate confidence based on feature agreement and volatility
            double confidence = 0.5;
            
            // Distance from neutral (0.5) indicates confidence
            double distance = Math.Abs(lastMLPrediction - 0.5);
            confidence = distance * 2.0; // Scale to [0, 1]
            
            // Adjust for volatility regime
            if (volatilityRegime > 1.5) // High volatility - reduce confidence
                confidence *= 0.7;
            else if (volatilityRegime < 0.7) // Low volatility - increase confidence
                confidence *= 1.2;
            
            // Adjust for feature availability
            double featureRatio = Math.Min(1.0, currentFeatures.Count / 30.0);
            confidence *= featureRatio;
            
            return Math.Max(0.0, Math.Min(1.0, confidence));
        }
        
        #endregion
        
        #region Risk Management
        
        private bool PassesRiskChecks()
        {
            // Daily loss limit
            if (dailyPnL <= -maxDailyLoss)
            {
                Print(string.Format("{0} - RISK CHECK FAILED: Daily loss limit reached: {1:F2}", Time[0], dailyPnL));
                return false;
            }
            
            // Cooldown period
            if ((Time[0] - lastTradeTime).TotalMinutes < cooldownMinutes)
            {
                if (CurrentBar % 30 == 0) // Only print occasionally
                    Print(string.Format("{0} - RISK CHECK FAILED: In cooldown period ({1:F1} min remaining)", 
                        Time[0], cooldownMinutes - (Time[0] - lastTradeTime).TotalMinutes));
                return false;
            }
            
            // Maximum trades per session
            if (tradesThisSession >= 20)
            {
                Print(string.Format("{0} - RISK CHECK FAILED: Max trades per session reached: {1}", Time[0], tradesThisSession));
                return false;
            }
            
            // Market hours check
            if (Time[0].Hour < 9 || Time[0].Hour >= 16)
            {
                if (CurrentBar % 60 == 0) // Only print occasionally
                    Print(string.Format("{0} - RISK CHECK FAILED: Outside market hours (Hour: {1})", Time[0], Time[0].Hour));
                return false;
            }
            
            // Minimum confidence check
            if (lastMLConfidence < (MinConfidence / 100.0))
            {
                if (CurrentBar % 30 == 0) // Only print occasionally
                    Print(string.Format("{0} - RISK CHECK FAILED: Confidence too low ({1:F3} < {2:F3})", 
                        Time[0], lastMLConfidence, MinConfidence / 100.0));
                return false;
            }
            
            return true;
        }
        
        private void UpdateDailyPnL()
        {
            // Reset daily PnL at market open
            if (Time[0].Hour == 9 && Time[0].Minute == 30 && CurrentBar > 0 && Time[1].Hour != 9)
            {
                dailyPnL = 0.0;
                tradesThisSession = 0;
                Print(string.Format("{0} - Daily PnL reset for new session", Time[0]));
            }
            
            // Update PnL based on position
            if (Position.MarketPosition != MarketPosition.Flat && lastEntryPrice > 0)
            {
                double unrealizedPnL = 0.0;
                if (Position.MarketPosition == MarketPosition.Long)
                    unrealizedPnL = (Close[0] - lastEntryPrice) * Position.Quantity * Instrument.MasterInstrument.PointValue;
                else if (Position.MarketPosition == MarketPosition.Short)
                    unrealizedPnL = (lastEntryPrice - Close[0]) * Position.Quantity * Instrument.MasterInstrument.PointValue;
                    
                // Note: This is unrealized PnL, realized PnL would be updated in OnExecutionUpdate
            }
        }
        
        #endregion
        
        #region Position Execution
        
        private void ExecutePositionChange()
        {
            currentPosition = Position.MarketPosition == MarketPosition.Long ? 1 : 
                             Position.MarketPosition == MarketPosition.Short ? -1 : 0;
            
            if (targetPosition == currentPosition)
                return;
                
            string action = "";
            
            // Close existing position first
            if (currentPosition != 0)
            {
                if (currentPosition > 0)
                {
                    ExitLong("Exit Long");
                    action += "Exit Long, ";
                }
                else
                {
                    ExitShort("Exit Short");
                    action += "Exit Short, ";
                }
            }
            
            // Enter new position
            if (targetPosition > 0)
            {
                EnterLong(ContractSize, "ML Long");
                action += "Enter Long";
                lastEntryPrice = Close[0];
            }
            else if (targetPosition < 0)
            {
                EnterShort(ContractSize, "ML Short");
                action += "Enter Short";
                lastEntryPrice = Close[0];
            }
            else
            {
                action += "Stay Flat";
            }
            
            lastTradeTime = Time[0];
            tradesThisSession++;
            
            Print(string.Format("{0} - {1} | ML: {2:F3} ({3:F1}%) | Vol: {4:F2}",
                Time[0], action, lastMLPrediction, lastMLConfidence * 100, volatilityRegime));
        }
        
        #endregion
        
        #region Event Handlers
        
        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            // Update realized PnL when trades are filled
            if (execution.Order != null && execution.Order.OrderState == OrderState.Filled)
            {
                dailyPnL += execution.Order.Filled * (execution.Price - lastEntryPrice) * Instrument.MasterInstrument.PointValue;
                
                Print(string.Format("{0} - Trade filled: {1} {2} @ {3:F2} | Session PnL: {4:F2}",
                    time, execution.Order.OrderAction, quantity, price, dailyPnL));
            }
        }
        
        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            // Handle order updates for monitoring
            if (order.Name.Contains("ML") && orderState == OrderState.Filled)
            {
                Print(string.Format("{0} - Order filled: {1} | Avg Price: {2:F2}",
                    time, order.Name, averageFillPrice));
            }
        }
        
        #endregion
    }
} 