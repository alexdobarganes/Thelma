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
    public class TBOTAutoTraderUnified : Strategy
    {
        private Dictionary<string, bool> processedOrders;
        private Dictionary<string, ITradeStrategy> activeTrades;
        private DateTime lastCheck;

        #region Trade Strategy Interfaces and Classes

        // Interface for trade strategy polymorphism
        public interface ITradeStrategy
        {
            string OrderId { get; }
            string Action { get; }
            void ExecuteEntry(TBOTAutoTraderUnified strategy);
            void SetupExits(TBOTAutoTraderUnified strategy);
            void HandleOrderUpdate(Order order, OrderState orderState, double averageFillPrice, TBOTAutoTraderUnified strategy);
            void HandleExecution(Execution execution, TBOTAutoTraderUnified strategy);
        }

        // Simple Order Strategy
        public class SimpleTradeStrategy : ITradeStrategy
        {
            public string OrderId { get; private set; }
            public string Action { get; private set; }
            public double EntryPrice { get; private set; }
            public double StopLoss { get; private set; }
            public double TakeProfit { get; private set; }
            public bool HasEntry { get; private set; }
            public bool HasStopLoss { get; private set; }
            public bool HasTakeProfit { get; private set; }
            public bool ExitsSet { get; private set; }

            public SimpleTradeStrategy(Dictionary<string, string> orderData)
            {
                OrderId = orderData.ContainsKey("ORDER_ID") ? orderData["ORDER_ID"] : DateTime.Now.ToString("HHmmss");
                Action = orderData["ACTION"];
                
                HasEntry = orderData.ContainsKey("ENTRY_PRICE") && !string.IsNullOrEmpty(orderData["ENTRY_PRICE"]);
                HasStopLoss = orderData.ContainsKey("STOP_LOSS") && !string.IsNullOrEmpty(orderData["STOP_LOSS"]);
                HasTakeProfit = orderData.ContainsKey("TAKE_PROFIT") && !string.IsNullOrEmpty(orderData["TAKE_PROFIT"]);

                if (HasEntry) EntryPrice = Convert.ToDouble(orderData["ENTRY_PRICE"]);
                if (HasStopLoss) StopLoss = Convert.ToDouble(orderData["STOP_LOSS"]);
                if (HasTakeProfit) TakeProfit = Convert.ToDouble(orderData["TAKE_PROFIT"]);
            }

            public void ExecuteEntry(TBOTAutoTraderUnified strategy)
            {
                string entryName = $"TBOTEntry_{OrderId}";

                // SET EXITS FIRST for Simple orders (required by NT8)
                if (HasStopLoss)
                {
                    strategy.SetStopLoss(entryName, CalculationMode.Price, StopLoss, false);
                    strategy.Print($"🛑 Simple SL set BEFORE entry: {StopLoss:F2}");
                }

                if (HasTakeProfit)
                {
                    strategy.SetProfitTarget(entryName, CalculationMode.Price, TakeProfit);
                    strategy.Print($"🎯 Simple TP set BEFORE entry: {TakeProfit:F2}");
                }

                if (HasEntry)
                {
                    // Limit order with GTC (Good Till Cancelled)
                    if (Action == "BUY")
                    {
                        strategy.EnterLongLimit(0, true, 1, EntryPrice, entryName);
                        strategy.Print($"📈 SIMPLE BUY LIMIT: 1 contract @ {EntryPrice:F2} (GTC)");
                    }
                    else if (Action == "SELL")
                    {
                        strategy.EnterShortLimit(0, true, 1, EntryPrice, entryName);
                        strategy.Print($"📉 SIMPLE SELL LIMIT: 1 contract @ {EntryPrice:F2} (GTC)");
                    }
                }
                else
                {
                    // Market order
                    if (Action == "BUY")
                    {
                        strategy.EnterLong(1, entryName);
                        strategy.Print($"📈 SIMPLE BUY MARKET: 1 contract");
                    }
                    else if (Action == "SELL")
                    {
                        strategy.EnterShort(1, entryName);
                        strategy.Print($"📉 SIMPLE SELL MARKET: 1 contract");
                    }
                }
                
                ExitsSet = true;
                strategy.Print($"✅ Simple order with exits submitted");
            }

            public void SetupExits(TBOTAutoTraderUnified strategy)
            {
                // For Simple orders, exits are already set in ExecuteEntry
                if (ExitsSet) 
                {
                    strategy.Print($"✅ Simple exits already configured in ExecuteEntry for {OrderId}");
                    return;
                }

                // This should not be called for Simple orders anymore
                strategy.Print($"⚠️ SetupExits called but exits already handled in ExecuteEntry for Simple order {OrderId}");
            }

            public void HandleOrderUpdate(Order order, OrderState orderState, double averageFillPrice, TBOTAutoTraderUnified strategy)
            {
                if (order.Name.Contains("Entry") && orderState == OrderState.Filled)
                {
                    // Update entry price if it was a market order
                    if (!HasEntry)
                    {
                        EntryPrice = averageFillPrice;
                        strategy.Print($"📊 Simple entry filled @ {EntryPrice:F2}");
                    }
                    SetupExits(strategy);
                }
            }

            public void HandleExecution(Execution execution, TBOTAutoTraderUnified strategy)
            {
                // Simple logging for simple orders
                strategy.Print($"💫 SIMPLE EXECUTION: {execution.Order.Name} - {execution.Quantity} @ {execution.Price:F2}");
            }
        }

        // ATM Order Strategy using NinjaTrader's native ATM functionality
        public class ATMTradeStrategy : ITradeStrategy
        {
            public string OrderId { get; private set; }
            public string Action { get; private set; }
            public double EntryPrice { get; private set; }
            public int StopTicks { get; private set; }
            public int TP1Ticks { get; private set; }
            public int TP2Ticks { get; private set; }
            public bool HasEntry { get; private set; }
            public string AtmStrategyId { get; private set; }
            public string AtmOrderId { get; private set; }
            public bool IsAtmStrategyCreated { get; private set; }

            public ATMTradeStrategy(Dictionary<string, string> orderData)
            {
                OrderId = orderData.ContainsKey("ORDER_ID") ? orderData["ORDER_ID"] : DateTime.Now.ToString("HHmmss");
                Action = orderData["ACTION"];
                
                HasEntry = orderData.ContainsKey("ENTRY_PRICE") && !string.IsNullOrEmpty(orderData["ENTRY_PRICE"]);
                StopTicks = Convert.ToInt32(orderData["STOP_TICKS"]);
                TP1Ticks = Convert.ToInt32(orderData["TP1_TICKS"]);
                TP2Ticks = Convert.ToInt32(orderData["TP2_TICKS"]);

                if (HasEntry) EntryPrice = Convert.ToDouble(orderData["ENTRY_PRICE"]);
                
                AtmStrategyId = string.Empty;
                AtmOrderId = string.Empty;
                IsAtmStrategyCreated = false;
            }

            public void ExecuteEntry(TBOTAutoTraderUnified strategy)
            {
                try
                {
                    // Generate unique IDs for ATM strategy
                    AtmStrategyId = strategy.GetAtmStrategyUniqueId();
                    AtmOrderId = strategy.GetAtmStrategyUniqueId();
                    IsAtmStrategyCreated = false;

                    strategy.Print($"🚀 Creating ATM Strategy with ID: {AtmStrategyId}");
                    strategy.Print($"📊 Using template: 2C-ATM");

                    if (HasEntry)
                    {
                        // Create ATM strategy with limit order
                        if (Action == "BUY")
                        {
                            strategy.AtmStrategyCreate(OrderAction.Buy, OrderType.Limit, EntryPrice, 0, TimeInForce.Gtc, 
                                AtmOrderId, "2C-ATM", AtmStrategyId, (atmCallbackErrorCode, atmCallBackId) => {
                                    if (atmCallbackErrorCode == ErrorCode.NoError && atmCallBackId == AtmStrategyId)
                                    {
                                        IsAtmStrategyCreated = true;
                                        strategy.Print($"✅ ATM BUY LIMIT created: @ {EntryPrice:F2} (GTC)");
                                    }
                                    else
                                    {
                                        strategy.Print($"❌ ATM Strategy creation failed: {atmCallbackErrorCode}");
                                    }
                                });
                        }
                        else if (Action == "SELL")
                        {
                            strategy.AtmStrategyCreate(OrderAction.Sell, OrderType.Limit, EntryPrice, 0, TimeInForce.Gtc, 
                                AtmOrderId, "2C-ATM", AtmStrategyId, (atmCallbackErrorCode, atmCallBackId) => {
                                    if (atmCallbackErrorCode == ErrorCode.NoError && atmCallBackId == AtmStrategyId)
                                    {
                                        IsAtmStrategyCreated = true;
                                        strategy.Print($"✅ ATM SELL LIMIT created: @ {EntryPrice:F2} (GTC)");
                                    }
                                    else
                                    {
                                        strategy.Print($"❌ ATM Strategy creation failed: {atmCallbackErrorCode}");
                                    }
                                });
                        }
                    }
                    else
                    {
                        // Create ATM strategy with market order
                        if (Action == "BUY")
                        {
                            strategy.AtmStrategyCreate(OrderAction.Buy, OrderType.Market, 0, 0, TimeInForce.Day, 
                                AtmOrderId, "2C-ATM", AtmStrategyId, (atmCallbackErrorCode, atmCallBackId) => {
                                    if (atmCallbackErrorCode == ErrorCode.NoError && atmCallBackId == AtmStrategyId)
                                    {
                                        IsAtmStrategyCreated = true;
                                        strategy.Print($"✅ ATM BUY MARKET created");
                                    }
                                    else
                                    {
                                        strategy.Print($"❌ ATM Strategy creation failed: {atmCallbackErrorCode}");
                                    }
                                });
                        }
                        else if (Action == "SELL")
                        {
                            strategy.AtmStrategyCreate(OrderAction.Sell, OrderType.Market, 0, 0, TimeInForce.Day, 
                                AtmOrderId, "2C-ATM", AtmStrategyId, (atmCallbackErrorCode, atmCallBackId) => {
                                    if (atmCallbackErrorCode == ErrorCode.NoError && atmCallBackId == AtmStrategyId)
                                    {
                                        IsAtmStrategyCreated = true;
                                        strategy.Print($"✅ ATM SELL MARKET created");
                                    }
                                    else
                                    {
                                        strategy.Print($"❌ ATM Strategy creation failed: {atmCallbackErrorCode}");
                                    }
                                });
                        }
                    }

                    strategy.Print($"📝 ATM Strategy submitted - waiting for confirmation...");
                }
                catch (Exception ex)
                {
                    strategy.Print($"❌ Error creating ATM strategy: {ex.Message}");
                }
            }

            public void SetupExits(TBOTAutoTraderUnified strategy)
            {
                // ATM strategies handle their own exits automatically via the template
                strategy.Print($"✅ ATM exits handled automatically by 2C-ATM template for {OrderId}");
            }

            public void HandleOrderUpdate(Order order, OrderState orderState, double averageFillPrice, TBOTAutoTraderUnified strategy)
            {
                // ATM strategies are self-managed, but we can monitor their status
                if (!string.IsNullOrEmpty(AtmOrderId))
                {
                    string[] status = strategy.GetAtmStrategyEntryOrderStatus(AtmOrderId);
                    
                    if (status.GetLength(0) > 0)
                    {
                        strategy.Print($"📊 ATM Entry Status: Fill Price={status[0]}, Fill Amount={status[1]}, State={status[2]}");
                        
                        // Reset order ID if terminal state reached
                        if (status[2] == "Filled" || status[2] == "Cancelled" || status[2] == "Rejected")
                        {
                            if (status[2] == "Filled")
                            {
                                EntryPrice = Convert.ToDouble(status[0]);
                                strategy.Print($"🎯 ATM ENTRY FILLED @ {EntryPrice:F2}");
                            }
                            AtmOrderId = string.Empty;
                        }
                    }
                }

                // Check if ATM strategy should be reset
                if (!string.IsNullOrEmpty(AtmStrategyId) && 
                    strategy.GetAtmStrategyMarketPosition(AtmStrategyId) == MarketPosition.Flat)
                {
                    strategy.Print($"🏁 ATM Strategy {AtmStrategyId} completed - position flat");
                    AtmStrategyId = string.Empty;
                    IsAtmStrategyCreated = false;
                }
            }

            public void HandleExecution(Execution execution, TBOTAutoTraderUnified strategy)
            {
                // Log ATM executions and print strategy status
                strategy.Print($"💫 ATM EXECUTION: {execution.Order.Name} - {execution.Quantity} @ {execution.Price:F2}");
                
                if (!string.IsNullOrEmpty(AtmStrategyId))
                {
                    try
                    {
                        var position = strategy.GetAtmStrategyMarketPosition(AtmStrategyId);
                        var quantity = strategy.GetAtmStrategyPositionQuantity(AtmStrategyId);
                        var avgPrice = strategy.GetAtmStrategyPositionAveragePrice(AtmStrategyId);
                        var unrealizedPnL = strategy.GetAtmStrategyUnrealizedProfitLoss(AtmStrategyId);
                        
                        strategy.Print($"📈 ATM Status: Pos={position}, Qty={quantity}, Avg={avgPrice:F2}, PnL={unrealizedPnL:F2}");
                    }
                    catch (Exception ex)
                    {
                        strategy.Print($"⚠️ Error getting ATM status: {ex.Message}");
                    }
                }
            }
        }

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"TBOT Unified Auto Trader - Handles both simple and ATM orders";
                Name = "TBOTAutoTraderUnified";
                Calculate = Calculate.OnEachTick;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;  // Don't auto-exit on session close
                ExitOnSessionCloseSeconds = 30;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TraceOrders = true;
                BarsRequiredToTrade = 1;
                
                // Order expiration settings
                DefaultQuantity = 1;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                
                // Configuration
                OrderFolderPath = @"D:\TBOT\nt_orders";
            }
            else if (State == State.DataLoaded)
            {
                processedOrders = new Dictionary<string, bool>();
                activeTrades = new Dictionary<string, ITradeStrategy>();
                lastCheck = DateTime.Now;
                
                // Load previously processed orders to maintain idempotency
                LoadProcessedOrdersLog();
                
                Print($"TBOT Unified AutoTrader started");
                Print($"Account: {Account?.DisplayName ?? "Not Set"}");
                Print($"Monitoring: {OrderFolderPath}");
                Print($"Supports: Simple orders & ATM strategies");
                Print($"🔒 Idempotent mode: Orders processed only once");
            }
        }

        protected override void OnBarUpdate()
        {
            // Make sure this strategy does not execute against historical data for ATM strategies
            if (State == State.Historical)
                return;

            // Check for new orders every 0.5 seconds (faster response)
            if (DateTime.Now.Subtract(lastCheck).TotalSeconds >= 0.1)
            {
                CheckForNewOrders();
                MonitorATMStrategies();
                lastCheck = DateTime.Now;
            }
        }

        private void MonitorATMStrategies()
        {
            try
            {
                foreach (var trade in activeTrades.Values.OfType<ATMTradeStrategy>())
                {
                    if (!string.IsNullOrEmpty(trade.AtmStrategyId) && trade.IsAtmStrategyCreated)
                    {
                        // Monitor ATM strategy status
                        var position = GetAtmStrategyMarketPosition(trade.AtmStrategyId);
                        
                        // Clean up completed ATM strategies
                        if (position == MarketPosition.Flat && 
                            !string.IsNullOrEmpty(trade.AtmOrderId) && 
                            GetAtmStrategyEntryOrderStatus(trade.AtmOrderId).Length == 0)
                        {
                            Print($"🏁 Cleaning up completed ATM strategy: {trade.AtmStrategyId}");
                            activeTrades.Remove(trade.OrderId);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"⚠️ Error monitoring ATM strategies: {ex.Message}");
            }
        }
        
        private void CheckForNewOrders()
        {
            try
            {
                if (!Directory.Exists(OrderFolderPath))
                    return;
                
                // Create processed folder if it doesn't exist
                string processedFolder = Path.Combine(OrderFolderPath, "processed");
                if (!Directory.Exists(processedFolder))
                    Directory.CreateDirectory(processedFolder);
                
                string[] orderFiles = Directory.GetFiles(OrderFolderPath, "order_*.txt");
                
                foreach (string filePath in orderFiles)
                {
                    string fileName = Path.GetFileName(filePath);
                    
                    if (!processedOrders.ContainsKey(fileName))
                    {
                        Print($"⚡ NEW ORDER DETECTED: {fileName} at {DateTime.Now:HH:mm:ss.fff}");
                        
                        bool success = ProcessOrderFile(filePath);
                        
                        if (success)
                        {
                            // Mark as processed
                            processedOrders[fileName] = true;
                            
                            // Move file to processed folder with timestamp
                            string processedFileName = $"{Path.GetFileNameWithoutExtension(fileName)}_processed_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
                            string processedFilePath = Path.Combine(processedFolder, processedFileName);
                            
                            try
                            {
                                File.Move(filePath, processedFilePath);
                                Print($"📁 MOVED TO PROCESSED: {fileName} → {processedFileName}");
                            }
                            catch (Exception moveEx)
                            {
                                Print($"⚠️ Could not move file: {moveEx.Message}");
                                // Add timestamp suffix to original file as backup
                                string backupPath = filePath.Replace(".txt", $"_processed_{DateTime.Now:HHmmss}.txt");
                                try
                                {
                                    File.Move(filePath, backupPath);
                                    Print($"📁 RENAMED: {fileName} (backup method)");
                                }
                                catch (Exception renameEx)
                                {
                                    Print($"⚠️ Could not rename file: {renameEx.Message}");
                                }
                            }
                            
                            // Log to persistent file
                            LogProcessedOrder(fileName, DateTime.Now);
                            
                            Print($"✅ ORDER PROCESSED: {fileName} at {DateTime.Now:HH:mm:ss.fff}");
                        }
                        else
                        {
                            Print($"❌ ORDER PROCESSING FAILED: {fileName}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Error checking orders: {ex.Message}");
            }
        }
        
        private bool ProcessOrderFile(string filePath)
        {
            try
            {
                var orderData = new Dictionary<string, string>();
                string[] lines = File.ReadAllLines(filePath);
                
                // Parse key=value pairs
                foreach (string line in lines)
                {
                    if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line))
                        continue;
                        
                    string[] parts = line.Split('=');
                    if (parts.Length == 2)
                        orderData[parts[0].Trim()] = parts[1].Trim();
                }
                
                // Validate required ACTION field
                if (!orderData.ContainsKey("ACTION"))
                {
                    Print($"Invalid order file - missing ACTION: {Path.GetFileName(filePath)}");
                    return false;
                }
                
                // Determine order type and create appropriate strategy
                ITradeStrategy tradeStrategy = CreateTradeStrategy(orderData);
                if (tradeStrategy != null)
                {
                    Print($"🚀 Executing {(tradeStrategy is ATMTradeStrategy ? "ATM" : "SIMPLE")} {tradeStrategy.Action} order {tradeStrategy.OrderId}");
                    
                    tradeStrategy.ExecuteEntry(this);
                    activeTrades[tradeStrategy.OrderId] = tradeStrategy;
                    return true;
                }
                else
                {
                    Print($"Failed to create trade strategy for {Path.GetFileName(filePath)}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                Print($"Error processing {Path.GetFileName(filePath)}: {ex.Message}");
                return false;
            }
        }

        private ITradeStrategy CreateTradeStrategy(Dictionary<string, string> orderData)
        {
            // Determine if this is an ATM order based on the presence of tick-based parameters
            bool isATMOrder = orderData.ContainsKey("STOP_TICKS") && 
                             orderData.ContainsKey("TP1_TICKS") && 
                             orderData.ContainsKey("TP2_TICKS");

            if (isATMOrder)
            {
                // Validate ATM requirements
                if (string.IsNullOrEmpty(orderData["STOP_TICKS"]) || 
                    string.IsNullOrEmpty(orderData["TP1_TICKS"]) || 
                    string.IsNullOrEmpty(orderData["TP2_TICKS"]))
                {
                    Print($"❌ ATM order requires STOP_TICKS, TP1_TICKS, and TP2_TICKS");
                    return null;
                }
                
                return new ATMTradeStrategy(orderData);
            }
            else
            {
                // Simple order
                return new SimpleTradeStrategy(orderData);
            }
        }
        
        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, 
            int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            if (order?.Name == null || !order.Name.StartsWith("TBOT")) return;
            
            try
            {
                // Find associated trade
                string tradeId = ExtractTradeId(order.Name);
                if (string.IsNullOrEmpty(tradeId) || !activeTrades.ContainsKey(tradeId)) return;
                
                var tradeStrategy = activeTrades[tradeId];
                
                if (orderState == OrderState.Filled)
                {
                    Print($"✅ FILLED: {order.Name} - {filled} @ {averageFillPrice:F2}");
                    tradeStrategy.HandleOrderUpdate(order, orderState, averageFillPrice, this);
                }
                else if (orderState == OrderState.Rejected)
                {
                    Print($"❌ REJECTED: {order.Name} - {error}");
                }
                else if (orderState == OrderState.Cancelled)
                {
                    Print($"🚫 CANCELLED: {order.Name}");
                }
            }
            catch (Exception ex)
            {
                Print($"Error in order update: {ex.Message}");
            }
        }
        
        private string ExtractTradeId(string orderName)
        {
            try
            {
                if (orderName.Contains("_"))
                {
                    string[] parts = orderName.Split('_');
                    if (parts.Length > 1)
                        return parts[1];
                }
                return "";
            }
            catch
            {
                return "";
            }
        }
        
        private void LogProcessedOrder(string fileName, DateTime processedTime)
        {
            try
            {
                string logPath = Path.Combine(OrderFolderPath, "processed_orders.log");
                string logEntry = $"{processedTime:yyyy-MM-dd HH:mm:ss.fff} | {fileName} | PROCESSED";
                
                File.AppendAllText(logPath, logEntry + Environment.NewLine);
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not log processed order: {ex.Message}");
            }
        }
        
        private void LoadProcessedOrdersLog()
        {
            try
            {
                string logPath = Path.Combine(OrderFolderPath, "processed_orders.log");
                
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
                            if (!string.IsNullOrEmpty(fileName) && !processedOrders.ContainsKey(fileName))
                            {
                                processedOrders[fileName] = true;
                                loadedCount++;
                            }
                        }
                    }
                    
                    if (loadedCount > 0)
                        Print($"📚 Loaded {loadedCount} previously processed orders from log");
                }
            }
            catch (Exception ex)
            {
                Print($"⚠️ Could not load processed orders log: {ex.Message}");
            }
        }
        
        protected override void OnExecutionUpdate(Execution execution, string executionId, 
            double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (execution?.Order?.Name == null || !execution.Order.Name.StartsWith("TBOT")) return;

            try
            {
                string tradeId = ExtractTradeId(execution.Order.Name);
                if (!string.IsNullOrEmpty(tradeId) && activeTrades.ContainsKey(tradeId))
                {
                    activeTrades[tradeId].HandleExecution(execution, this);
                }
            }
            catch (Exception ex)
            {
                Print($"Error in execution update: {ex.Message}");
            }
        }
        
        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Order Folder Path", Description = "Path to order files", Order = 1, GroupName = "Settings")]
        public string OrderFolderPath { get; set; }
        #endregion
    }
} 