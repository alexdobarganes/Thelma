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
    public class VReversalAutoTrader : Strategy
    {
        private Dictionary<string, bool> processedSignals;
        private Dictionary<string, VReversalTrade> activeTrades;
        private DateTime lastCheck;
        private int tradeCounter = 0;

        #region V-Reversal Trade Management
        
        public class VReversalTrade
        {
            public string SignalId { get; set; }
            public string Action { get; set; }  // BUY or SELL
            public double EntryPrice { get; set; }
            public double StopLoss { get; set; }
            public double TakeProfit { get; set; }
            public DateTime SignalTime { get; set; }
            public bool IsEntryFilled { get; set; }
            public string EntryOrderName { get; set; }
            public double FillPrice { get; set; }
            
            public VReversalTrade(Dictionary<string, string> signalData)
            {
                SignalId = signalData.ContainsKey("SIGNAL_ID") ? signalData["SIGNAL_ID"] : DateTime.Now.ToString("HHmmss");
                Action = signalData["ACTION"];
                EntryPrice = Convert.ToDouble(signalData["ENTRY_PRICE"]);
                
                // Use stop loss and take profit from signal file if provided
                if (signalData.ContainsKey("STOP_LOSS"))
                {
                    StopLoss = Convert.ToDouble(signalData["STOP_LOSS"]);
                }
                else
                {
                    // Fallback: Calculate stop loss based on action and current stop loss percentage
                    if (Action == "BUY")
                        StopLoss = EntryPrice * (1 - 0.002);  // 0.2% below entry (fallback)
                    else
                        StopLoss = EntryPrice * (1 + 0.002);  // 0.2% above entry (fallback)
                }
                
                if (signalData.ContainsKey("TAKE_PROFIT"))
                {
                    TakeProfit = Convert.ToDouble(signalData["TAKE_PROFIT"]);
                }
                else
                {
                    // Fallback: Calculate take profit based on action
                    if (Action == "BUY")
                        TakeProfit = EntryPrice + 3.0;  // ~3 points target (fallback)
                    else
                        TakeProfit = EntryPrice - 3.0;  // ~3 points target (fallback)
                }
                
                SignalTime = DateTime.Now;
                IsEntryFilled = false;
                EntryOrderName = $"VRev_{SignalId}";
            }
        }
        
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"V-Reversal Auto Trader - Optimized for validated strategy (Drop_3, Stop_0.002)";
                Name = "VReversalAutoTrader";
                Calculate = Calculate.OnEachTick;
                EntriesPerDirection = 3;  // Allow multiple V-reversal trades
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;  // Close at session end for safety
                ExitOnSessionCloseSeconds = 60;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TraceOrders = true;
                BarsRequiredToTrade = 1;
                
                // V-Reversal specific settings
                DefaultQuantity = 1;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                
                // Configuration
                SignalFolderPath = @"D:\Thelma\signals";  // CHANGED: Use main signals folder
                MaxPositions = 3;  // Limit concurrent V-reversal trades
                MaxDailyTrades = 20; // Reasonable daily limit
                TradeQuantity = 1;   // Single contract per signal
                PlaybackMode = true;  // ADDED: Enable playback mode by default
            }
            else if (State == State.DataLoaded)
            {
                processedSignals = new Dictionary<string, bool>();
                activeTrades = new Dictionary<string, VReversalTrade>();
                lastCheck = DateTime.Now;
                tradeCounter = 0;
                
                // Create signals directory if it doesn't exist
                if (!Directory.Exists(SignalFolderPath))
                {
                    Directory.CreateDirectory(SignalFolderPath);
                    Directory.CreateDirectory(Path.Combine(SignalFolderPath, "processed"));
                }
                
                LoadProcessedSignalsLog();
                
                Print($"🎯 V-Reversal AutoTrader STARTED");
                Print($"📊 Strategy: Drop_3 threshold, 0.2% stop loss");
                Print($"📁 Monitoring: {SignalFolderPath}");
                Print($"⚙️ Max Positions: {MaxPositions} | Max Daily: {MaxDailyTrades}");
                Print($"💰 Expected: ~$25k/month, 91%+ win rate");
            }
        }

        protected override void OnBarUpdate()
        {
            if (State == State.Historical) return;

            // Check for new V-reversal signals every 100ms (fast response)
            if (DateTime.Now.Subtract(lastCheck).TotalMilliseconds >= 100)
            {
                CheckForVReversalSignals();
                
                // Run cleanup more frequently to catch missed exits
                CleanupCompletedTrades();
                
                lastCheck = DateTime.Now;
            }
        }

        private void CheckForVReversalSignals()
        {
            try
            {
                if (!Directory.Exists(SignalFolderPath)) return;
                
                // Look for V-reversal signal files
                string[] signalFiles = Directory.GetFiles(SignalFolderPath, "vreversal_*.txt");
                
                foreach (string filePath in signalFiles)
                {
                    string fileName = Path.GetFileName(filePath);
                    
                    if (!processedSignals.ContainsKey(fileName))
                    {
                        Print($"🚨 NEW V-REVERSAL SIGNAL: {fileName} at {DateTime.Now:HH:mm:ss.fff}");
                        
                        if (ProcessVReversalSignal(filePath))
                        {
                            processedSignals[fileName] = true;
                            MoveToProcessed(filePath, fileName);
                            LogProcessedSignal(fileName, DateTime.Now);
                            Print($"✅ V-REVERSAL SIGNAL PROCESSED: {fileName}");
                        }
                        else
                        {
                            Print($"❌ V-REVERSAL SIGNAL FAILED: {fileName}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"❌ Error checking V-reversal signals: {ex.Message}");
            }
        }

        private bool ProcessVReversalSignal(string filePath)
        {
            try
            {
                // CRITICAL SAFETY: Trading hours check (Eastern Time)
                DateTime timeToCheck;
                
                if (PlaybackMode)
                {
                    // In playback mode, extract timestamp from signal filename (generated from data timestamp)
                    string signalFileName = Path.GetFileNameWithoutExtension(filePath);
                    // Format: vreversal_20250213_135900 -> 2025-02-13 13:59:00
                    if (signalFileName.StartsWith("vreversal_") && signalFileName.Length >= 23)
                    {
                        string datePart = signalFileName.Substring(10, 8);  // 20250213
                        string timePart = signalFileName.Substring(19, 6);  // 135900
                        
                        string year = datePart.Substring(0, 4);
                        string month = datePart.Substring(4, 2);
                        string day = datePart.Substring(6, 2);
                        string hour = timePart.Substring(0, 2);
                        string minute = timePart.Substring(2, 2);
                        string second = timePart.Substring(4, 2);
                        
                        timeToCheck = new DateTime(int.Parse(year), int.Parse(month), int.Parse(day),
                                                 int.Parse(hour), int.Parse(minute), int.Parse(second));
                        
                        Print($"🕐 PLAYBACK MODE: Using data timestamp {timeToCheck:HH:mm:ss} ET from signal");
                    }
                    else
                    {
                        Print($"⚠️ Could not parse timestamp from signal file {signalFileName}, using system time");
                        timeToCheck = TimeZoneInfo.ConvertTimeBySystemTimeZoneId(DateTime.Now, "Eastern Standard Time");
                    }
                }
                else
                {
                    // Live mode: use current system time
                    timeToCheck = TimeZoneInfo.ConvertTimeBySystemTimeZoneId(DateTime.Now, "Eastern Standard Time");
                    Print($"🕐 LIVE MODE: Using system time {timeToCheck:HH:mm:ss} ET");
                }
                
                TimeSpan currentTime = timeToCheck.TimeOfDay;
                DayOfWeek currentDay = timeToCheck.DayOfWeek;
                
                // Only trade Monday-Friday, 9:30 AM - 4:00 PM ET
                if (currentDay == DayOfWeek.Saturday || currentDay == DayOfWeek.Sunday)
                {
                    Print($"🚫 WEEKEND - No trading on {currentDay}");
                    return false;
                }
                
                TimeSpan marketOpen = new TimeSpan(9, 30, 0);   // 9:30 AM ET
                TimeSpan marketClose = new TimeSpan(16, 0, 0);  // 4:00 PM ET
                
                if (currentTime < marketOpen || currentTime > marketClose)
                {
                    Print($"🚫 AFTER HOURS - Time: {timeToCheck:HH:mm:ss} ET (Market: 9:30-16:00)");
                    return false;
                }
                
                // Standard safety checks
                if (activeTrades.Count >= MaxPositions)
                {
                    Print($"⚠️ Max positions reached ({activeTrades.Count}/{MaxPositions}), skipping signal");
                    Print($"📊 Current active trades:");
                    foreach (var kvp in activeTrades)
                    {
                        var trade = kvp.Value;
                        var age = DateTime.Now.Subtract(trade.SignalTime).TotalMinutes;
                        Print($"   - {kvp.Key}: {trade.Action} @ {trade.EntryPrice:F2}, Age: {age:F1}min, Filled: {trade.IsEntryFilled}");
                    }
                    return false;
                }
                
                if (tradeCounter >= MaxDailyTrades)
                {
                    Print($"⚠️ Daily trade limit reached ({MaxDailyTrades}), skipping signal");
                    return false;
                }
                
                // Parse signal file  
                var signalData = ParseSignalFile(filePath);
                if (signalData == null) return false;
                
                // Create V-reversal trade
                var vTrade = new VReversalTrade(signalData);
                
                // Execute entry with pre-set exits (V-reversal style)
                ExecuteVReversalEntry(vTrade);
                
                activeTrades[vTrade.SignalId] = vTrade;
                tradeCounter++;
                
                Print($"📊 Trade added to active list. Count: {activeTrades.Count}/{MaxPositions}");
                
                return true;
            }
            catch (Exception ex)
            {
                Print($"❌ Error processing V-reversal signal: {ex.Message}");
                return false;
            }
        }

        private Dictionary<string, string> ParseSignalFile(string filePath)
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
                
                // Validate required V-reversal fields
                if (!signalData.ContainsKey("ACTION") || !signalData.ContainsKey("ENTRY_PRICE"))
                {
                    Print($"❌ Invalid V-reversal signal - missing ACTION or ENTRY_PRICE");
                    return null;
                }
                
                // Log what values we're using (from signal or calculated)
                string slSource = signalData.ContainsKey("STOP_LOSS") ? "signal" : "calculated";
                string tpSource = signalData.ContainsKey("TAKE_PROFIT") ? "signal" : "calculated";
                Print($"📊 Signal parsed - SL: {slSource}, TP: {tpSource}");
                
                return signalData;
            }
            catch (Exception ex)
            {
                Print($"❌ Error parsing signal file: {ex.Message}");
                return null;
            }
        }

        private void ExecuteVReversalEntry(VReversalTrade vTrade)
        {
            try
            {
                // Set stops BEFORE entry (NT8 requirement for managed approach)
                SetStopLoss(vTrade.EntryOrderName, CalculationMode.Price, vTrade.StopLoss, false);
                SetProfitTarget(vTrade.EntryOrderName, CalculationMode.Price, vTrade.TakeProfit);
                
                // Calculate the actual percentages and points for logging
                double slPct = Math.Abs((vTrade.StopLoss - vTrade.EntryPrice) / vTrade.EntryPrice * 100);
                double tpPts = Math.Abs(vTrade.TakeProfit - vTrade.EntryPrice);
                
                Print($"🛑 V-Rev SL set: {vTrade.StopLoss:F2} ({slPct:F2}% risk)");
                Print($"🎯 V-Rev TP set: {vTrade.TakeProfit:F2} ({tpPts:F1}pt target)");
                
                // Execute entry order
                if (vTrade.Action == "BUY")
                {
                    EnterLongLimit(0, true, TradeQuantity, vTrade.EntryPrice, vTrade.EntryOrderName);
                    Print($"📈 V-REVERSAL BUY: {TradeQuantity} @ {vTrade.EntryPrice:F2} [Signal: {vTrade.SignalId}]");
                }
                else if (vTrade.Action == "SELL")
                {
                    EnterShortLimit(0, true, TradeQuantity, vTrade.EntryPrice, vTrade.EntryOrderName);
                    Print($"📉 V-REVERSAL SELL: {TradeQuantity} @ {vTrade.EntryPrice:F2} [Signal: {vTrade.SignalId}]");
                }
                
                Print($"🚀 V-Reversal trade submitted with validated parameters");
            }
            catch (Exception ex)
            {
                Print($"❌ Error executing V-reversal entry: {ex.Message}");
            }
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, 
            int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            if (order?.Name == null || !order.Name.StartsWith("VRev_")) return;
            
            try
            {
                string signalId = ExtractSignalId(order.Name);
                if (string.IsNullOrEmpty(signalId) || !activeTrades.ContainsKey(signalId)) return;
                
                var vTrade = activeTrades[signalId];
                
                Print($"🔄 ORDER UPDATE: {order.Name} | State: {orderState} | Error: {error}");
                
                if (orderState == OrderState.Filled)
                {
                    // Check if this is an entry order
                    if (order.Name.Contains($"VRev_{signalId}") && !order.Name.Contains("Stop") && !order.Name.Contains("Target"))
                    {
                        vTrade.IsEntryFilled = true;
                        vTrade.FillPrice = averageFillPrice;
                        Print($"✅ V-REVERSAL ENTRY FILLED: {order.Name} @ {averageFillPrice:F2}");
                        Print($"📊 Expected: 91.2% win rate, ~{Math.Abs(vTrade.TakeProfit - vTrade.EntryPrice):F1} pt target");
                    }
                    // Check if this is a stop loss or take profit order
                    else if (order.Name.Contains("Stop") || order.Name.Contains("Target"))
                    {
                        Print($"🎯 V-REVERSAL EXIT FILLED: {order.Name} @ {averageFillPrice:F2}");
                        Print($"📊 Exit type detected in OrderUpdate - should be handled in ExecutionUpdate");
                        
                        // Calculate final P&L
                        double finalPnl = 0;
                        if (vTrade.IsEntryFilled)
                        {
                            if (vTrade.Action == "BUY")
                                finalPnl = (averageFillPrice - vTrade.FillPrice) * quantity * 50;
                            else
                                finalPnl = (vTrade.FillPrice - averageFillPrice) * quantity * 50;
                        }
                        
                        string exitType = order.Name.Contains("Stop") ? "STOP LOSS" : "TAKE PROFIT";
                        Print($"💰 {exitType} P&L: ${finalPnl:F0}");
                    }
                }
                else if (orderState == OrderState.Rejected)
                {
                    Print($"❌ V-REVERSAL REJECTED: {order.Name} - {error}");
                    Print($"📊 Active trades before removal: {activeTrades.Count}");
                    activeTrades.Remove(signalId);
                    Print($"📊 Active trades after removal: {activeTrades.Count}");
                }
                else if (orderState == OrderState.Cancelled)
                {
                    Print($"🚫 V-REVERSAL CANCELLED: {order.Name}");
                    
                    // If it's an entry order, remove the trade
                    if (order.Name.Contains($"VRev_{signalId}") && !order.Name.Contains("Stop") && !order.Name.Contains("Target"))
                    {
                        Print($"📊 Active trades before removal: {activeTrades.Count}");
                        activeTrades.Remove(signalId);
                        Print($"📊 Active trades after removal: {activeTrades.Count}");
                    }
                    // If it's a stop/target order cancellation, the position might still be open
                    else
                    {
                        Print($"⚠️ Stop/Target order cancelled - position may still be open");
                        // Don't remove the trade yet, let the cleanup handle it based on position status
                    }
                }
                else if (orderState == OrderState.CancelPending)
                {
                    Print($"⏳ V-REVERSAL CANCEL PENDING: {order.Name}");
                }
                else if (orderState == OrderState.Working)
                {
                    Print($"⏰ V-REVERSAL WORKING: {order.Name} @ {limitPrice:F2}");
                }
                else if (orderState == OrderState.Accepted)
                {
                    Print($"✅ V-REVERSAL ACCEPTED: {order.Name}");
                }
            }
            catch (Exception ex)
            {
                Print($"❌ Error in V-reversal order update: {ex.Message}");
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, 
            double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            try
            {
                // Check if this is related to any of our V-reversal trades
                string signalId = "";
                VReversalTrade relatedTrade = null;
                
                // Find the trade this execution belongs to
                foreach (var kvp in activeTrades)
                {
                    if (execution.Order.Name.StartsWith($"VRev_{kvp.Key}"))
                    {
                        signalId = kvp.Key;
                        relatedTrade = kvp.Value;
                        break;
                    }
                }
                
                if (relatedTrade == null) return;
                
                // Calculate P&L for logging
                double pnl = 0;
                if (relatedTrade.IsEntryFilled)
                {
                    if (relatedTrade.Action == "BUY")
                        pnl = (price - relatedTrade.FillPrice) * quantity * 50; // ES $50 per point
                    else
                        pnl = (relatedTrade.FillPrice - price) * quantity * 50;
                }
                
                Print($"💫 V-REVERSAL EXECUTION: {execution.Order.Name} - {quantity} @ {price:F2} | P&L: ${pnl:F0}");
                
                // Enhanced exit detection with detailed logging
                Print($"🔍 Order details: Type={execution.Order.OrderType}, Name={execution.Order.Name}");
                Print($"🔍 MarketPosition after execution: {marketPosition}");
                
                bool isExit = false;
                string exitReason = "";
                
                // IMPROVED EXIT DETECTION LOGIC
                
                // Method 1: Check for Stop Loss execution (more precise)
                if (execution.Order.OrderType == OrderType.StopMarket || 
                    execution.Order.Name.Contains("Stop"))
                {
                    isExit = true;
                    exitReason = "Stop Loss Hit";
                }
                
                // Method 2: Check for Take Profit execution (more precise)
                else if ((execution.Order.OrderType == OrderType.Limit || execution.Order.Name.Contains("Target")) &&
                         relatedTrade.IsEntryFilled)
                {
                    // Check if this is a profit target by comparing with expected TP price
                    double priceDiff = Math.Abs(price - relatedTrade.TakeProfit);
                    if (priceDiff < 1.0) // Within 1 point of expected TP
                    {
                        isExit = true;
                        exitReason = "Take Profit Hit";
                    }
                    // Alternative: Check if this is an exit based on direction and price improvement
                    else if (relatedTrade.Action == "BUY" && price > relatedTrade.FillPrice)
                    {
                        isExit = true;
                        exitReason = "Profitable Exit (BUY)";
                    }
                    else if (relatedTrade.Action == "SELL" && price < relatedTrade.FillPrice)
                    {
                        isExit = true;
                        exitReason = "Profitable Exit (SELL)";
                    }
                }
                
                // Method 3: Check for any exit that results in flat position (most reliable)
                if (marketPosition == MarketPosition.Flat && relatedTrade.IsEntryFilled)
                {
                    if (!isExit) // Only set if not already detected by methods above
                    {
                        isExit = true;
                        exitReason = "Position Flat - Exit Detected";
                    }
                }
                
                // Method 4: Manual/Emergency exits (covers all other cases)
                if (!isExit && relatedTrade.IsEntryFilled && 
                    execution.Order.OrderAction == OrderAction.Sell && relatedTrade.Action == "BUY")
                {
                    isExit = true;
                    exitReason = "Manual Sell Exit";
                }
                else if (!isExit && relatedTrade.IsEntryFilled && 
                         execution.Order.OrderAction == OrderAction.BuyToCover && relatedTrade.Action == "SELL")
                {
                    isExit = true;
                    exitReason = "Manual Buy to Cover Exit";
                }
                
                Print($"🔍 Exit detected: {isExit}, Reason: {exitReason}");
                
                if (isExit)
                {
                    Print($"🏁 V-Reversal trade COMPLETED: {signalId} | Final P&L: ${pnl:F0} | Reason: {exitReason}");
                    Print($"📊 Active trades before removal: {activeTrades.Count}");
                    
                    // Remove the completed trade from active trades
                    if (activeTrades.ContainsKey(signalId))
                    {
                        activeTrades.Remove(signalId);
                        Print($"✅ Trade {signalId} successfully removed from active trades");
                    }
                    
                    Print($"📊 Active trades after removal: {activeTrades.Count}");
                    
                    // Log the completed trade
                    LogCompletedTrade(signalId, exitReason, pnl, price);
                }
            }
            catch (Exception ex)
            {
                Print($"❌ Error in V-reversal execution update: {ex.Message}");
            }
        }

        private string ExtractSignalId(string orderName)
        {
            try
            {
                if (orderName.Contains("_"))
                {
                    string[] parts = orderName.Split('_');
                    if (parts.Length > 1) return parts[1];
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
                
                // Log current state for debugging
                Print($"🔍 CLEANUP CHECK: Position={Position.MarketPosition}, Quantity={Position.Quantity}, Active Trades={activeTrades.Count}");
                
                foreach (var kvp in activeTrades)
                {
                    var trade = kvp.Value;
                    var tradeAge = DateTime.Now.Subtract(trade.SignalTime).TotalMinutes;
                    
                    // Remove stale trades that never got filled after 30 minutes
                    if (!trade.IsEntryFilled && tradeAge > 30)
                    {
                        Print($"🧹 Cleaning up stale unfilled trade: {kvp.Key} (age: {tradeAge:F1} min)");
                        tradesToRemove.Add(kvp.Key);
                    }
                    // Remove very old trades regardless of status (safety net)
                    else if (tradeAge > 120) // 2 hours
                    {
                        Print($"🧹 Cleaning up very old trade: {kvp.Key} (age: {tradeAge:F1} min)");
                        tradesToRemove.Add(kvp.Key);
                    }
                    // NEW: Check if filled trades are actually still in position (improved logic)
                    else if (trade.IsEntryFilled && tradeAge > 0.5) // Check filled trades older than 30 seconds
                    {
                        // If no active position and trade is marked as filled, it must have been closed
                        if (Position.MarketPosition == MarketPosition.Flat)
                        {
                            Print($"🧹 CLEANUP: Position is flat but trade still active - removing: {kvp.Key}");
                            Print($"   Trade details: {trade.Action} @ {trade.FillPrice:F2}, Age: {tradeAge:F1} min");
                            tradesToRemove.Add(kvp.Key);
                        }
                        // Additional check: If we have multiple filled trades but only one position, clean extras
                        else if (Position.MarketPosition != MarketPosition.Flat)
                        {
                            // Count how many filled trades we have
                            int filledTradesCount = activeTrades.Values.Count(t => t.IsEntryFilled);
                            
                            // If we have more filled trades than our position quantity allows, clean old ones
                            if (filledTradesCount > 1 && Math.Abs(Position.Quantity) < filledTradesCount)
                            {
                                Print($"🧹 CLEANUP: Mismatch between filled trades ({filledTradesCount}) and position size ({Position.Quantity})");
                                Print($"   Removing trade: {kvp.Key} (age: {tradeAge:F1} min)");
                                tradesToRemove.Add(kvp.Key);
                            }
                        }
                    }
                }
                
                // Additional check: If we have multiple filled trades but only one (or no) position
                if (tradesToRemove.Count == 0) // Only if we haven't found trades to remove yet
                {
                    var filledTrades = activeTrades.Where(kvp => kvp.Value.IsEntryFilled).ToList();
                    
                    if (filledTrades.Count > 1)
                    {
                        Print($"⚠️ ANOMALY DETECTED: {filledTrades.Count} filled trades but position quantity is {Position.Quantity}");
                        
                        // If position is flat, remove all filled trades except the newest
                        if (Position.MarketPosition == MarketPosition.Flat)
                        {
                            var oldestTrades = filledTrades.OrderBy(kvp => kvp.Value.SignalTime).Take(filledTrades.Count - 1);
                            foreach (var trade in oldestTrades)
                            {
                                Print($"🧹 Removing excess filled trade (position flat): {trade.Key}");
                                tradesToRemove.Add(trade.Key);
                            }
                        }
                        // If we have a position but too many filled trades, remove oldest ones
                        else if (filledTrades.Count > Math.Max(1, Position.Quantity))
                        {
                            int excessTrades = filledTrades.Count - Math.Max(1, Position.Quantity);
                            var oldestTrades = filledTrades.OrderBy(kvp => kvp.Value.SignalTime).Take(excessTrades);
                            
                            foreach (var trade in oldestTrades)
                            {
                                Print($"🧹 Removing excess filled trade (too many vs position): {trade.Key}");
                                tradesToRemove.Add(trade.Key);
                            }
                        }
                    }
                }
                
                // Remove identified trades
                foreach (string tradeId in tradesToRemove)
                {
                    activeTrades.Remove(tradeId);
                }
                
                if (tradesToRemove.Count > 0)
                {
                    Print($"📊 Cleanup complete: Removed {tradesToRemove.Count} trades. Active: {activeTrades.Count}");
                }
            }
            catch (Exception ex)
            {
                Print($"⚠️ Error cleaning up trades: {ex.Message}");
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
                Print($"⚠️ Could not move signal file: {ex.Message}");
            }
        }

        private void LogProcessedSignal(string fileName, DateTime processedTime)
        {
            try
            {
                string logPath = Path.Combine(SignalFolderPath, "processed_signals.log");
                string logEntry = $"{processedTime:yyyy-MM-dd HH:mm:ss.fff} | {fileName} | V-REVERSAL_PROCESSED";
                File.AppendAllText(logPath, logEntry + Environment.NewLine);
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not log processed signal: {ex.Message}");
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
                        Print($"📚 Loaded {loadedCount} previously processed V-reversal signals");
                }
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not load processed signals log: {ex.Message}");
            }
        }

        private void LogCompletedTrade(string signalId, string exitReason, double pnl, double exitPrice)
        {
            try
            {
                string logPath = Path.Combine(SignalFolderPath, "completed_trades.log");
                string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} | {signalId} | {exitReason} | ${pnl:F0} | Exit: {exitPrice:F2}";
                File.AppendAllText(logPath, logEntry + Environment.NewLine);
                Print($"📝 Completed trade logged: {signalId} -> {exitReason} | P&L: ${pnl:F0}");
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not log completed trade: {ex.Message}");
            }
        }

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Signal Folder Path", Description = "Path to V-reversal signal files", Order = 1, GroupName = "Settings")]
        public string SignalFolderPath { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Positions", Description = "Maximum concurrent V-reversal positions", Order = 2, GroupName = "Settings")]
        [Range(1, 100)]
        public int MaxPositions { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Daily Trades", Description = "Maximum V-reversal trades per day", Order = 3, GroupName = "Settings")]
        [Range(1, 50)]
        public int MaxDailyTrades { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade Quantity", Description = "Contracts per V-reversal signal", Order = 4, GroupName = "Settings")]
        [Range(1, 10)]
        public int TradeQuantity { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Playback Mode", Description = "Use data timestamps instead of system time for trading hours", Order = 5, GroupName = "Settings")]
        public bool PlaybackMode { get; set; }
        #endregion
    }
}