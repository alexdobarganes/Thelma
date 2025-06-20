#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class EnhancedVReversalAutoTrader : Strategy
    {
        private Dictionary<string, bool> processedSignals;
        private Dictionary<string, EnhancedVReversalTrade> activeTrades;
        private DateTime lastCheck;
        private int tradeCounter = 0;

        #region Enhanced V-Reversal Trade Management
        
        public class EnhancedVReversalTrade
        {
            public string SignalId { get; set; }
            public string Action { get; set; }  // BUY or SELL
            public string PatternType { get; set; }  // DOWNWARD_V or UPWARD_V
            public double EntryPrice { get; set; }
            public double StopLoss { get; set; }
            public double TakeProfit { get; set; }
            public DateTime SignalTime { get; set; }
            public bool IsEntryFilled { get; set; }
            public string EntryOrderName { get; set; }
            public double FillPrice { get; set; }
            public double MovePoints { get; set; }  // Pattern move magnitude
            public double Confidence { get; set; }   // Pattern confidence
            
            public EnhancedVReversalTrade(Dictionary<string, string> signalData)
            {
                SignalId = signalData.ContainsKey("SIGNAL_ID") ? signalData["SIGNAL_ID"] : DateTime.Now.ToString("HHmmss");
                Action = signalData["ACTION"];
                PatternType = signalData.ContainsKey("PATTERN_SUBTYPE") ? signalData["PATTERN_SUBTYPE"] : "UNKNOWN";
                EntryPrice = Convert.ToDouble(signalData["ENTRY_PRICE"]);
                
                // Extract pattern details
                if (signalData.ContainsKey("MOVE_POINTS"))
                    MovePoints = Convert.ToDouble(signalData["MOVE_POINTS"]);
                
                if (signalData.ContainsKey("CONFIDENCE"))
                    Confidence = Convert.ToDouble(signalData["CONFIDENCE"]);
                
                // Use stop loss and take profit from signal file
                if (signalData.ContainsKey("STOP_LOSS"))
                {
                    StopLoss = Convert.ToDouble(signalData["STOP_LOSS"]);
                }
                else
                {
                    // Fallback: Calculate based on action
                    if (Action == "BUY")
                        StopLoss = EntryPrice * (1 - 0.001);  // 0.1% below entry
                    else
                        StopLoss = EntryPrice * (1 + 0.001);  // 0.1% above entry
                }
                
                if (signalData.ContainsKey("TAKE_PROFIT"))
                {
                    TakeProfit = Convert.ToDouble(signalData["TAKE_PROFIT"]);
                }
                else
                {
                    // Fallback: 3 points target
                    if (Action == "BUY")
                        TakeProfit = EntryPrice + 3.0;
                    else
                        TakeProfit = EntryPrice - 3.0;
                }
                
                SignalTime = DateTime.Now;
                IsEntryFilled = false;
                EntryOrderName = $"Enhanced_{Action}_{SignalId}";
            }
        }
        
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"Enhanced V-Reversal Auto Trader - Bidirectional (BUY + SELL signals)";
                Name = "EnhancedVReversalAutoTrader";
                Calculate = Calculate.OnEachTick;
                EntriesPerDirection = 5;  // Allow multiple trades in each direction
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 60;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TraceOrders = true;
                BarsRequiredToTrade = 1;
                
                // Enhanced settings for bidirectional trading
                DefaultQuantity = 1;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                
                // Configuration
                SignalFolderPath = @"D:\Thelma\signals\enhanced";  // Enhanced signals folder
                MaxPositions = 5;      // Increased for bidirectional trading
                MaxDailyTrades = 40;   // Doubled for BUY + SELL
                TradeQuantity = 1;     // Single contract per signal
                PlaybackMode = true;   // Enable playback mode
                EnableBuySignals = true;   // Enable BUY signals
                EnableSellSignals = true;  // Enable SELL signals
            }
            else if (State == State.DataLoaded)
            {
                processedSignals = new Dictionary<string, bool>();
                activeTrades = new Dictionary<string, EnhancedVReversalTrade>();
                lastCheck = DateTime.Now;
                tradeCounter = 0;
                
                // Create enhanced signals directory
                if (!Directory.Exists(SignalFolderPath))
                {
                    Directory.CreateDirectory(SignalFolderPath);
                    Directory.CreateDirectory(Path.Combine(SignalFolderPath, "processed"));
                }
                
                LoadProcessedSignalsLog();
                
                Print($"🚀 Enhanced V-Reversal AutoTrader STARTED");
                Print($"📊 Strategy: Bidirectional V-Reversal Detection");
                Print($"📁 Monitoring: {SignalFolderPath}");
                Print($"🟢 BUY Signals: {(EnableBuySignals ? "ENABLED" : "DISABLED")}");
                Print($"🔴 SELL Signals: {(EnableSellSignals ? "ENABLED" : "DISABLED")}");
                Print($"⚙️ Max Positions: {MaxPositions} | Max Daily: {MaxDailyTrades}");
                Print($"💰 Expected: Enhanced bidirectional profitability");
            }
        }

        protected override void OnBarUpdate()
        {
            if (State == State.Historical) return;

            // Check for new enhanced signals every 100ms
            if (DateTime.Now.Subtract(lastCheck).TotalMilliseconds >= 100)
            {
                CheckForEnhancedVReversalSignals();
                CleanupCompletedTrades();
                lastCheck = DateTime.Now;
            }
        }

        private void CheckForEnhancedVReversalSignals()
        {
            try
            {
                if (!Directory.Exists(SignalFolderPath)) return;
                
                // Look for enhanced V-reversal signal files
                string[] signalFiles = Directory.GetFiles(SignalFolderPath, "enhanced_vreversal_*.txt");
                
                foreach (string filePath in signalFiles)
                {
                    string fileName = Path.GetFileName(filePath);
                    
                    if (!processedSignals.ContainsKey(fileName))
                    {
                        Print($"🚨 NEW ENHANCED SIGNAL: {fileName} at {DateTime.Now:HH:mm:ss.fff}");
                        
                        if (ProcessEnhancedVReversalSignal(filePath))
                        {
                            processedSignals[fileName] = true;
                            MoveToProcessed(filePath, fileName);
                            LogProcessedSignal(fileName, DateTime.Now);
                            Print($"✅ ENHANCED SIGNAL PROCESSED: {fileName}");
                        }
                        else
                        {
                            Print($"❌ ENHANCED SIGNAL FAILED: {fileName}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"❌ Error checking enhanced signals: {ex.Message}");
            }
        }

        private bool ProcessEnhancedVReversalSignal(string filePath)
        {
            try {
                // Parse signal file
                var signalData = ParseEnhancedSignalFile(filePath);
                if (signalData == null) return false;
                
                string action = signalData["ACTION"];
                
                // Check if signal type is enabled
                if (action == "BUY" && !EnableBuySignals)
                {
                    Print($"🚫 BUY signals disabled, skipping BUY signal");
                    return false;
                }
                
                if (action == "SELL" && !EnableSellSignals)
                {
                    Print($"🚫 SELL signals disabled, skipping SELL signal");
                    return false;
                }
                
                // Standard safety checks with enhanced logic
                DateTime timeToCheck;
                
                if (PlaybackMode)
                {
                    // Extract timestamp from enhanced signal filename
                    string signalFileName = Path.GetFileNameWithoutExtension(filePath);
                    
                    // Format: enhanced_vreversal_buy_20250213_135900 or enhanced_vreversal_sell_20250213_135900
                    string[] parts = signalFileName.Split('_');
                    if (parts.Length >= 5)
                    {
                        string datePart = parts[3];  // 20250213
                        string timePart = parts[4];  // 135900
                        
                        try
                        {
                            string year = datePart.Substring(0, 4);
                            string month = datePart.Substring(4, 2);
                            string day = datePart.Substring(6, 2);
                            string hour = timePart.Substring(0, 2);
                            string minute = timePart.Substring(2, 2);
                            string second = timePart.Substring(4, 2);
                            
                            timeToCheck = new DateTime(int.Parse(year), int.Parse(month), int.Parse(day),
                                                     int.Parse(hour), int.Parse(minute), int.Parse(second));
                            
                            Print($"🕐 PLAYBACK MODE: Using data timestamp {timeToCheck:HH:mm:ss} ET from enhanced signal");
                        }
                        catch
                        {
                            Print($"⚠️ Could not parse enhanced timestamp, using system time");
                            timeToCheck = TimeZoneInfo.ConvertTimeBySystemTimeZoneId(DateTime.Now, "Eastern Standard Time");
                        }
                    }
                    else
                    {
                        timeToCheck = TimeZoneInfo.ConvertTimeBySystemTimeZoneId(DateTime.Now, "Eastern Standard Time");
                    }
                }
                else
                {
                    timeToCheck = TimeZoneInfo.ConvertTimeBySystemTimeZoneId(DateTime.Now, "Eastern Standard Time");
                    Print($"🕐 LIVE MODE: Using system time {timeToCheck:HH:mm:ss} ET");
                }
                
                // Trading hours validation
                TimeSpan currentTime = timeToCheck.TimeOfDay;
                DayOfWeek currentDay = timeToCheck.DayOfWeek;
                
                if (currentDay == DayOfWeek.Saturday || currentDay == DayOfWeek.Sunday)
                {
                    Print($"🚫 WEEKEND - No trading on {currentDay}");
                    return false;
                }
                
                TimeSpan marketOpen = new TimeSpan(9, 30, 0);
                TimeSpan marketClose = new TimeSpan(16, 0, 0);
                
                if (currentTime < marketOpen || currentTime > marketClose)
                {
                    Print($"🚫 AFTER HOURS - Time: {timeToCheck:HH:mm:ss} ET");
                    return false;
                }
                
                // Position limits
                if (activeTrades.Count >= MaxPositions)
                {
                    Print($"⚠️ Max positions reached ({activeTrades.Count}/{MaxPositions})");
                    return false;
                }
                
                if (tradeCounter >= MaxDailyTrades)
                {
                    Print($"⚠️ Daily trade limit reached ({MaxDailyTrades})");
                    return false;
                }
                
                // Create enhanced trade
                var enhancedTrade = new EnhancedVReversalTrade(signalData);
                
                // Execute entry
                ExecuteEnhancedVReversalEntry(enhancedTrade);
                
                activeTrades[enhancedTrade.SignalId] = enhancedTrade;
                tradeCounter++;
                
                Print($"📊 Enhanced trade added. Active: {activeTrades.Count}/{MaxPositions}");
                
                return true;
            }
            catch (Exception ex)
            {
                Print($"❌ Error processing enhanced signal: {ex.Message}");
                return false;
            }
        }

        private Dictionary<string, string> ParseEnhancedSignalFile(string filePath)
        {
            try
            {
                var signalData = new Dictionary<string, string>();
                string[] lines = File.ReadAllLines(filePath);
                
                foreach (string line in lines)
                {
                    if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line)) continue;
                    
                    string[] parts = line.Split('=');
                    if (parts.Length == 2)
                        signalData[parts[0].Trim()] = parts[1].Trim();
                }
                
                // Validate required fields
                if (!signalData.ContainsKey("ACTION") || !signalData.ContainsKey("ENTRY_PRICE"))
                {
                    Print($"❌ Invalid enhanced signal - missing ACTION or ENTRY_PRICE");
                    return null;
                }
                
                string action = signalData["ACTION"];
                string patternType = signalData.ContainsKey("PATTERN_SUBTYPE") ? signalData["PATTERN_SUBTYPE"] : "UNKNOWN";
                string movePoints = signalData.ContainsKey("MOVE_POINTS") ? signalData["MOVE_POINTS"] : "0";
                
                Print($"📊 Enhanced signal parsed: {action} {patternType} ({movePoints} pts)");
                
                return signalData;
            }
            catch (Exception ex)
            {
                Print($"❌ Error parsing enhanced signal file: {ex.Message}");
                return null;
            }
        }

        private void ExecuteEnhancedVReversalEntry(EnhancedVReversalTrade trade)
        {
            try
            {
                // Set stops BEFORE entry
                SetStopLoss(trade.EntryOrderName, CalculationMode.Price, trade.StopLoss, false);
                SetProfitTarget(trade.EntryOrderName, CalculationMode.Price, trade.TakeProfit);
                
                // Enhanced logging
                double slPct = Math.Abs((trade.StopLoss - trade.EntryPrice) / trade.EntryPrice * 100);
                double tpPts = Math.Abs(trade.TakeProfit - trade.EntryPrice);
                
                Print($"🛑 Enhanced SL: {trade.StopLoss:F2} ({slPct:F2}% risk)");
                Print($"🎯 Enhanced TP: {trade.TakeProfit:F2} ({tpPts:F1}pt target)");
                Print($"📊 Pattern: {trade.PatternType} | Move: {trade.MovePoints:F1}pts | Confidence: {trade.Confidence:P1}");
                
                // Execute entry order
                if (trade.Action == "BUY")
                {
                    EnterLongLimit(0, true, TradeQuantity, trade.EntryPrice, trade.EntryOrderName);
                    Print($"📈 ENHANCED BUY: {TradeQuantity} @ {trade.EntryPrice:F2} [{trade.PatternType}]");
                }
                else if (trade.Action == "SELL")
                {
                    EnterShortLimit(0, true, TradeQuantity, trade.EntryPrice, trade.EntryOrderName);
                    Print($"📉 ENHANCED SELL: {TradeQuantity} @ {trade.EntryPrice:F2} [{trade.PatternType}]");
                }
                
                Print($"🚀 Enhanced V-Reversal trade submitted");
            }
            catch (Exception ex)
            {
                Print($"❌ Error executing enhanced entry: {ex.Message}");
            }
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, 
            int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            if (order?.Name == null || !order.Name.StartsWith("Enhanced_")) return;
            
            try
            {
                string signalId = ExtractEnhancedSignalId(order.Name);
                if (string.IsNullOrEmpty(signalId) || !activeTrades.ContainsKey(signalId)) return;
                
                var trade = activeTrades[signalId];
                
                Print($"🔄 ENHANCED ORDER UPDATE: {order.Name} | State: {orderState} | Error: {error}");
                
                if (orderState == OrderState.Filled)
                {
                    // Check if this is an entry order
                    if (order.Name.Contains($"Enhanced_{trade.Action}_{signalId}") && !order.Name.Contains("Stop") && !order.Name.Contains("Target"))
                    {
                        trade.IsEntryFilled = true;
                        trade.FillPrice = averageFillPrice;
                        Print($"✅ ENHANCED {trade.Action} ENTRY FILLED: {order.Name} @ {averageFillPrice:F2}");
                        Print($"📊 Pattern: {trade.PatternType} | Move: {trade.MovePoints:F1}pts | Expected: 91.2% win rate");
                    }
                    // Check if this is an exit order
                    else if (order.Name.Contains("Stop") || order.Name.Contains("Target"))
                    {
                        Print($"🎯 ENHANCED {trade.Action} EXIT FILLED: {order.Name} @ {averageFillPrice:F2}");
                        
                        double finalPnl = 0;
                        if (trade.IsEntryFilled)
                        {
                            if (trade.Action == "BUY")
                                finalPnl = (averageFillPrice - trade.FillPrice) * quantity * 50;
                            else
                                finalPnl = (trade.FillPrice - averageFillPrice) * quantity * 50;
                        }
                        
                        string exitType = order.Name.Contains("Stop") ? "STOP LOSS" : "TAKE PROFIT";
                        Print($"💰 Enhanced {trade.Action} {exitType} P&L: ${finalPnl:F0}");
                    }
                }
                else if (orderState == OrderState.Rejected)
                {
                    Print($"❌ ENHANCED {trade.Action} REJECTED: {order.Name} - {error}");
                    activeTrades.Remove(signalId);
                }
                else if (orderState == OrderState.Cancelled)
                {
                    Print($"🚫 ENHANCED {trade.Action} CANCELLED: {order.Name}");
                    
                    if (order.Name.Contains($"Enhanced_{trade.Action}_{signalId}") && !order.Name.Contains("Stop") && !order.Name.Contains("Target"))
                    {
                        activeTrades.Remove(signalId);
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"❌ Error in enhanced order update: {ex.Message}");
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, 
            double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            try
            {
                // Find related enhanced trade
                string signalId = "";
                EnhancedVReversalTrade relatedTrade = null;
                
                foreach (var kvp in activeTrades)
                {
                    if (execution.Order.Name.StartsWith($"Enhanced_{kvp.Value.Action}_{kvp.Key}"))
                    {
                        signalId = kvp.Key;
                        relatedTrade = kvp.Value;
                        break;
                    }
                }
                
                if (relatedTrade == null) return;
                
                // Calculate P&L
                double pnl = 0;
                if (relatedTrade.IsEntryFilled)
                {
                    if (relatedTrade.Action == "BUY")
                        pnl = (price - relatedTrade.FillPrice) * quantity * 50;
                    else
                        pnl = (relatedTrade.FillPrice - price) * quantity * 50;
                }
                
                Print($"💫 ENHANCED EXECUTION: {execution.Order.Name} - {quantity} @ {price:F2} | P&L: ${pnl:F0}");
                
                // Enhanced exit detection
                bool isExit = false;
                string exitReason = "";
                
                // Method 1: Stop Loss
                if (execution.Order.OrderType == OrderType.StopMarket || 
                    execution.Order.Name.Contains("Stop"))
                {
                    isExit = true;
                    exitReason = $"{relatedTrade.Action} Stop Loss Hit";
                }
                
                // Method 2: Take Profit
                else if ((execution.Order.OrderType == OrderType.Limit || execution.Order.Name.Contains("Target")) &&
                         relatedTrade.IsEntryFilled)
                {
                    double priceDiff = Math.Abs(price - relatedTrade.TakeProfit);
                    if (priceDiff < 1.0)
                    {
                        isExit = true;
                        exitReason = $"{relatedTrade.Action} Take Profit Hit";
                    }
                }
                
                // Method 3: Position flat
                if (marketPosition == MarketPosition.Flat && relatedTrade.IsEntryFilled)
                {
                    if (!isExit)
                    {
                        isExit = true;
                        exitReason = $"{relatedTrade.Action} Position Flat";
                    }
                }
                
                // Method 4: Manual exits
                if (!isExit && relatedTrade.IsEntryFilled)
                {
                    if (execution.Order.OrderAction == OrderAction.Sell && relatedTrade.Action == "BUY")
                    {
                        isExit = true;
                        exitReason = "Manual BUY Exit";
                    }
                    else if (execution.Order.OrderAction == OrderAction.BuyToCover && relatedTrade.Action == "SELL")
                    {
                        isExit = true;
                        exitReason = "Manual SELL Exit";
                    }
                }
                
                if (isExit)
                {
                    Print($"🏁 Enhanced {relatedTrade.Action} trade COMPLETED: {signalId} | P&L: ${pnl:F0} | {exitReason}");
                    Print($"📊 Pattern: {relatedTrade.PatternType} | Move: {relatedTrade.MovePoints:F1}pts");
                    
                    if (activeTrades.ContainsKey(signalId))
                    {
                        activeTrades.Remove(signalId);
                        Print($"✅ Enhanced trade {signalId} removed from active list");
                    }
                    
                    LogCompletedEnhancedTrade(signalId, relatedTrade, exitReason, pnl, price);
                }
            }
            catch (Exception ex)
            {
                Print($"❌ Error in enhanced execution update: {ex.Message}");
            }
        }

        private string ExtractEnhancedSignalId(string orderName)
        {
            try
            {
                // Format: Enhanced_BUY_signalId or Enhanced_SELL_signalId
                string[] parts = orderName.Split('_');
                if (parts.Length >= 3) 
                {
                    return string.Join("_", parts.Skip(2));
                }
                return "";
            }
            catch { return ""; }
        }

        private void CleanupCompletedTrades()
        {
            try
            {
                var tradesToRemove = new List<string>();
                
                foreach (var kvp in activeTrades)
                {
                    var trade = kvp.Value;
                    var tradeAge = DateTime.Now.Subtract(trade.SignalTime).TotalMinutes;
                    
                    // Clean up old unfilled trades
                    if (!trade.IsEntryFilled && tradeAge > 30)
                    {
                        Print($"🧹 Cleaning up stale enhanced {trade.Action} trade: {kvp.Key}");
                        tradesToRemove.Add(kvp.Key);
                    }
                    // Clean up very old trades
                    else if (tradeAge > 120)
                    {
                        Print($"🧹 Cleaning up very old enhanced {trade.Action} trade: {kvp.Key}");
                        tradesToRemove.Add(kvp.Key);
                    }
                    // Check position consistency
                    else if (trade.IsEntryFilled && tradeAge > 0.5)
                    {
                        if (Position.MarketPosition == MarketPosition.Flat)
                        {
                            Print($"🧹 Position flat but enhanced {trade.Action} trade still active: {kvp.Key}");
                            tradesToRemove.Add(kvp.Key);
                        }
                    }
                }
                
                foreach (string tradeId in tradesToRemove)
                {
                    activeTrades.Remove(tradeId);
                }
                
                if (tradesToRemove.Count > 0)
                {
                    Print($"📊 Enhanced cleanup: Removed {tradesToRemove.Count} trades. Active: {activeTrades.Count}");
                }
            }
            catch (Exception ex)
            {
                Print($"⚠️ Error in enhanced cleanup: {ex.Message}");
            }
        }

        private void MoveToProcessed(string filePath, string fileName)
        {
            try
            {
                string processedFolder = Path.Combine(SignalFolderPath, "processed");
                string processedFileName = $"{Path.GetFileNameWithoutExtension(fileName)}_processed_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
                string processedFilePath = Path.Combine(processedFolder, processedFileName);
                
                File.Move(filePath, processedFilePath);
                Print($"📁 MOVED TO PROCESSED: {fileName}");
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not move enhanced signal file: {ex.Message}");
            }
        }

        private void LogProcessedSignal(string fileName, DateTime processedTime)
        {
            try
            {
                string logPath = Path.Combine(SignalFolderPath, "processed_signals.log");
                string logEntry = $"{processedTime:yyyy-MM-dd HH:mm:ss.fff} | {fileName} | ENHANCED_PROCESSED";
                File.AppendAllText(logPath, logEntry + Environment.NewLine);
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not log enhanced processed signal: {ex.Message}");
            }
        }

        private void LogCompletedEnhancedTrade(string signalId, EnhancedVReversalTrade trade, string exitReason, double pnl, double exitPrice)
        {
            try
            {
                string logPath = Path.Combine(SignalFolderPath, "completed_enhanced_trades.log");
                string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} | {signalId} | {trade.Action} | {trade.PatternType} | {exitReason} | ${pnl:F0} | Exit: {exitPrice:F2} | Move: {trade.MovePoints:F1}pts";
                File.AppendAllText(logPath, logEntry + Environment.NewLine);
                Print($"📝 Enhanced trade logged: {signalId} -> {trade.Action} {exitReason} | P&L: ${pnl:F0}");
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not log enhanced completed trade: {ex.Message}");
            }
        }

        private void LoadProcessedSignalsLog()
        {
            try
            {
                string logPath = Path.Combine(SignalFolderPath, "processed_signals.log");
                
                if (File.Exists(logPath))
                {
                    string[] logLines = File.ReadAllLines(logPath);
                    int loadedCount = 0;
                    
                    foreach (string line in logLines)
                    {
                        if (string.IsNullOrWhiteSpace(line)) continue;
                        
                        string[] parts = line.Split('|');
                        if (parts.Length >= 2)
                        {
                            string fileName = parts[1].Trim();
                            if (!string.IsNullOrEmpty(fileName) && !processedSignals.ContainsKey(fileName))
                            {
                                processedSignals[fileName] = true;
                                loadedCount++;
                            }
                        }
                    }
                    
                    if (loadedCount > 0)
                        Print($"📚 Loaded {loadedCount} previously processed enhanced signals");
                }
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not load enhanced processed signals log: {ex.Message}");
            }
        }

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Signal Folder Path", Description = "Path to enhanced V-reversal signal files", Order = 1, GroupName = "Settings")]
        public string SignalFolderPath { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Positions", Description = "Maximum concurrent enhanced positions", Order = 2, GroupName = "Settings")]
        [Range(1, 20)]
        public int MaxPositions { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Daily Trades", Description = "Maximum enhanced trades per day", Order = 3, GroupName = "Settings")]
        [Range(1, 100)]
        public int MaxDailyTrades { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade Quantity", Description = "Contracts per enhanced signal", Order = 4, GroupName = "Settings")]
        [Range(1, 10)]
        public int TradeQuantity { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Playback Mode", Description = "Use data timestamps for trading hours", Order = 5, GroupName = "Settings")]
        public bool PlaybackMode { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable BUY Signals", Description = "Process BUY signals from downward V-reversals", Order = 6, GroupName = "Signal Types")]
        public bool EnableBuySignals { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable SELL Signals", Description = "Process SELL signals from upward V-reversals", Order = 7, GroupName = "Signal Types")]
        public bool EnableSellSignals { get; set; }
        #endregion
    }
} 