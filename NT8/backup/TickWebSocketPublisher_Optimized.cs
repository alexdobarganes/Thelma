#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Net;
using System.Net.WebSockets;
using System.Linq;
using System.Collections.Concurrent;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class TBOTTickWebSocketPublisherOptimized : Indicator
    {
        private HttpListener httpListener;
        private ConcurrentList<WebSocketClient> connectedClients;
        private bool isRunning;
        private Timer cleanupTimer;
        
        // Performance optimizations
        private readonly SemaphoreSlim broadcastSemaphore = new SemaphoreSlim(1, 1);
        private readonly byte[] pingBytes = Encoding.UTF8.GetBytes("ping");
        private CancellationTokenSource cancellationTokenSource = new CancellationTokenSource();
        
        // Historical data management - optimized
        private ConcurrentQueue<HistoricalBar> historicalData;
        private int maxHistoricalBars = 5000; // Store up to 5000 bars
        private int currentHistoricalCount = 0;

        // Historical bar structure
        private class HistoricalBar
        {
            public DateTime Timestamp { get; set; }
            public double Open { get; set; }
            public double High { get; set; }
            public double Low { get; set; }
            public double Close { get; set; }
            public long Volume { get; set; }
            public string Symbol { get; set; }
        }

        // Thread-safe wrapper class for better client management
        private class WebSocketClient
        {
            public WebSocket WebSocket { get; set; }
            public DateTime LastPong { get; set; }
            public bool IsAlive { get; set; }
            public bool HistoricalDataSent { get; set; }
            private readonly SemaphoreSlim sendSemaphore;
            private readonly ConcurrentQueue<byte[]> messageQueue;
            private readonly Task processingTask;
            private readonly CancellationToken cancellationToken;
            
            public WebSocketClient(WebSocket webSocket, CancellationToken token)
            {
                WebSocket = webSocket;
                LastPong = DateTime.UtcNow;
                IsAlive = true;
                HistoricalDataSent = false;
                cancellationToken = token;
                sendSemaphore = new SemaphoreSlim(1, 1);
                messageQueue = new ConcurrentQueue<byte[]>();
                
                // Start background message processing task
                processingTask = Task.Run(ProcessMessageQueue, token);
            }
            
            public void QueueMessage(byte[] buffer)
            {
                if (IsAlive && WebSocket.State == WebSocketState.Open)
                {
                    messageQueue.Enqueue(buffer);
                }
            }
            
            private async Task ProcessMessageQueue()
            {
                while (!cancellationToken.IsCancellationRequested && IsAlive)
                {
                    try
                    {
                        if (messageQueue.TryDequeue(out byte[] buffer))
                        {
                            if (!IsAlive || WebSocket.State != WebSocketState.Open)
                                break;
                                
                            await sendSemaphore.WaitAsync(cancellationToken);
                            try
                            {
                                if (IsAlive && WebSocket.State == WebSocketState.Open)
                                {
                                    await WebSocket.SendAsync(
                                        new ArraySegment<byte>(buffer), 
                                        WebSocketMessageType.Text, 
                                        true, 
                                        cancellationToken
                                    );
                                }
                            }
                            finally
                            {
                                sendSemaphore.Release();
                            }
                        }
                        else
                        {
                            // No messages, wait a bit
                            await Task.Delay(1, cancellationToken);
                        }
                    }
                    catch (Exception)
                    {
                        IsAlive = false;
                        break;
                    }
                }
            }
            
            public async Task<bool> SendDirectAsync(byte[] buffer, WebSocketMessageType messageType, bool endOfMessage, CancellationToken cancellationToken)
            {
                if (!IsAlive || WebSocket.State != WebSocketState.Open)
                    return false;
                
                try
                {
                    await sendSemaphore.WaitAsync(cancellationToken);
                    try
                    {
                        if (!IsAlive || WebSocket.State != WebSocketState.Open)
                            return false;
                            
                        await WebSocket.SendAsync(new ArraySegment<byte>(buffer), messageType, endOfMessage, cancellationToken);
                        return true;
                    }
                    finally
                    {
                        sendSemaphore.Release();
                    }
                }
                catch
                {
                    IsAlive = false;
                    return false;
                }
            }
            
            public void Dispose()
            {
                IsAlive = false;
                sendSemaphore?.Dispose();
                processingTask?.Wait(1000); // Wait up to 1 second for cleanup
            }
        }

        // Thread-safe list for clients
        private class ConcurrentList<T>
        {
            private readonly List<T> list = new List<T>();
            private readonly ReaderWriterLockSlim rwLock = new ReaderWriterLockSlim();

            public void Add(T item)
            {
                rwLock.EnterWriteLock();
                try
                {
                    list.Add(item);
                }
                finally
                {
                    rwLock.ExitWriteLock();
                }
            }

            public bool Remove(T item)
            {
                rwLock.EnterWriteLock();
                try
                {
                    return list.Remove(item);
                }
                finally
                {
                    rwLock.ExitWriteLock();
                }
            }

            public List<T> ToList()
            {
                rwLock.EnterReadLock();
                try
                {
                    return new List<T>(list);
                }
                finally
                {
                    rwLock.ExitReadLock();
                }
            }

            public int Count
            {
                get
                {
                    rwLock.EnterReadLock();
                    try
                    {
                        return list.Count;
                    }
                    finally
                    {
                        rwLock.ExitReadLock();
                    }
                }
            }
        }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"TBOT Optimized WebSocket Publisher - High Performance Tick Streaming";
                Name = "TBOTTickWebSocketPublisherOptimized";
                Calculate = Calculate.OnEachTick;
                IsOverlay = true;
                IsSuspendedWhileInactive = false;
                
                // Optimized default settings
                WebSocketPort = 6789;
                PublishTicks = true;
                PublishBars = true;
                
                // Historical data settings
                SendHistoricalOnConnect = true;
                HistoricalBarsCount = 2000;
                FastHistoricalDelivery = true;
                
                // Performance settings
                MaxConcurrentConnections = 10;
                MessageQueueSize = 1000;
                TickThrottleMs = 0; // No throttling by default
            }
            else if (State == State.DataLoaded)
            {
                // Initialize optimized data structures
                historicalData = new ConcurrentQueue<HistoricalBar>();
                connectedClients = new ConcurrentList<WebSocketClient>();
                cancellationTokenSource = new CancellationTokenSource();
                
                // Load existing historical data
                LoadExistingHistoricalData();
                
                StartOptimizedWebSocketServer();
            }
            else if (State == State.Terminated)
            {
                StopOptimizedWebSocketServer();
            }
        }

        private void LoadExistingHistoricalData()
        {
            try
            {
                // Load historical bars from the chart data efficiently
                int barsToLoad = Math.Min(HistoricalBarsCount, BarsArray[0].Count);
                int startIndex = Math.Max(0, BarsArray[0].Count - barsToLoad);
                
                for (int i = startIndex; i < BarsArray[0].Count; i++)
                {
                    if (i >= 0 && i < BarsArray[0].Count)
                    {
                        var bar = new HistoricalBar
                        {
                            Timestamp = Times[0][i],
                            Open = Opens[0][i],
                            High = Highs[0][i],
                            Low = Lows[0][i],
                            Close = Closes[0][i],
                            Volume = (long)Volumes[0][i],
                            Symbol = Instrument.FullName
                        };
                        
                        historicalData.Enqueue(bar);
                        currentHistoricalCount++;
                        
                        // Maintain max size
                        if (currentHistoricalCount > maxHistoricalBars)
                        {
                            if (historicalData.TryDequeue(out _))
                                currentHistoricalCount--;
                        }
                    }
                }
                
                Print("Loaded " + currentHistoricalCount + " historical bars for optimized streaming");
            }
            catch (Exception ex)
            {
                Print("Error loading historical data: " + ex.Message);
            }
        }

        private void StartOptimizedWebSocketServer()
        {
            try
            {
                isRunning = true;
                
                httpListener = new HttpListener();
                httpListener.Prefixes.Add("http://localhost:" + WebSocketPort + "/");
                httpListener.Start();
                
                Print("TBOT Optimized WebSocket server started on ws://localhost:" + WebSocketPort);
                Print("Performance: Max " + MaxConcurrentConnections + " connections, Queue size " + MessageQueueSize);
                Print("Historical data: " + (SendHistoricalOnConnect ? "ENABLED" : "DISABLED") + " (" + currentHistoricalCount + " bars available)");
                
                // Start optimized cleanup timer (every 5 seconds)
                cleanupTimer = new Timer(CleanupDisconnectedClients, null, TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(5));
                
                // Handle connections in background with higher priority
                Task.Run(HandleWebSocketRequests, cancellationTokenSource.Token);
            }
            catch (Exception ex)
            {
                Print("Error starting optimized WebSocket server: " + ex.Message);
            }
        }

        private async Task HandleWebSocketRequests()
        {
            while (isRunning && httpListener != null && !cancellationTokenSource.Token.IsCancellationRequested)
            {
                try
                {
                    var listenerContext = await httpListener.GetContextAsync();
                    
                    if (listenerContext.Request.IsWebSocketRequest)
                    {
                                                 // Check connection limit
                         if (connectedClients.Count >= MaxConcurrentConnections)
                         {
                             listenerContext.Response.StatusCode = 503; // Service Unavailable
                             listenerContext.Response.Close();
                             Print("Connection rejected: Max connections (" + MaxConcurrentConnections + ") reached");
                             continue;
                         }
                        
                        var webSocketContext = await listenerContext.AcceptWebSocketAsync(null);
                        var webSocket = webSocketContext.WebSocket;
                        
                        var client = new WebSocketClient(webSocket, cancellationTokenSource.Token);
                        
                                                 connectedClients.Add(client);
                         Print("Client connected. Total: " + connectedClients.Count);

                        // Send historical data if enabled (non-blocking)
                        if (SendHistoricalOnConnect)
                        {
                            Task.Run(() => SendHistoricalDataToClientOptimized(client), cancellationTokenSource.Token);
                        }

                        // Monitor this specific connection (non-blocking)
                        Task.Run(() => MonitorConnectionOptimized(client), cancellationTokenSource.Token);
                    }
                    else
                    {
                        listenerContext.Response.StatusCode = 400;
                        listenerContext.Response.Close();
                    }
                }
                catch (Exception ex)
                {
                    if (isRunning)
                        Print("WebSocket accept error: " + ex.Message);
                }
            }
        }

        private async Task SendHistoricalDataToClientOptimized(WebSocketClient client)
        {
            if (!SendHistoricalOnConnect || client.HistoricalDataSent)
                return;

            try
            {
                var allData = historicalData.ToArray();
                var startIndex = Math.Max(0, allData.Length - HistoricalBarsCount);
                var dataToSend = allData.Skip(startIndex).ToList();

                if (dataToSend.Count == 0)
                {
                    Print("No historical data available to send");
                    return;
                }

                Print("Sending " + dataToSend.Count + " historical bars to new client (optimized)...");

                // Send historical data marker
                string startMessage = "{\"type\":\"historical_start\",\"count\":" + dataToSend.Count + "}";
                byte[] startBuffer = Encoding.UTF8.GetBytes(startMessage);
                await client.SendDirectAsync(startBuffer, WebSocketMessageType.Text, true, cancellationTokenSource.Token);

                // Send historical bars in batches for better performance
                int batchSize = FastHistoricalDelivery ? 50 : 10;
                int sent = 0;
                
                for (int i = 0; i < dataToSend.Count; i += batchSize)
                {
                    if (!client.IsAlive || client.WebSocket.State != WebSocketState.Open)
                        break;

                    int batchEnd = Math.Min(i + batchSize, dataToSend.Count);
                    
                    for (int j = i; j < batchEnd; j++)
                    {
                        var bar = dataToSend[j];
                        string json = "{\"type\":\"historical_bar\",\"symbol\":\"" + bar.Symbol + "\",\"open\":" + bar.Open.ToString("F2") + ",\"high\":" + bar.High.ToString("F2") + ",\"low\":" + bar.Low.ToString("F2") + ",\"close\":" + bar.Close.ToString("F2") + ",\"volume\":" + bar.Volume + ",\"timestamp\":\"" + bar.Timestamp.ToString("yyyy-MM-ddTHH:mm:ss.fffZ") + "\"}";
                        byte[] buffer = Encoding.UTF8.GetBytes(json);
                        
                        bool success = await client.SendDirectAsync(buffer, WebSocketMessageType.Text, true, cancellationTokenSource.Token);
                        if (!success)
                        {
                            Print("Failed to send historical bar " + (sent + 1) + ", stopping transmission");
                            return;
                        }
                        sent++;
                    }
                    
                    // Minimal delay between batches
                    if (FastHistoricalDelivery)
                    {
                        // No delay for fast delivery
                    }
                    else
                    {
                        await Task.Delay(1, cancellationTokenSource.Token); // 1ms delay per batch
                    }
                }

                // Send end marker
                string endMessage = "{\"type\":\"historical_end\",\"sent\":" + sent + "}";
                byte[] endBuffer = Encoding.UTF8.GetBytes(endMessage);
                await client.SendDirectAsync(endBuffer, WebSocketMessageType.Text, true, cancellationTokenSource.Token);

                client.HistoricalDataSent = true;
                Print("✅ Optimized historical data transmission completed: " + sent + "/" + dataToSend.Count + " bars sent");
            }
            catch (Exception ex)
            {
                Print("Error sending historical data (optimized): " + ex.Message);
            }
        }

        private async Task MonitorConnectionOptimized(WebSocketClient client)
        {
            var buffer = new byte[1024];
            
            try
            {
                while (client.IsAlive && client.WebSocket.State == WebSocketState.Open && isRunning && !cancellationTokenSource.Token.IsCancellationRequested)
                {
                    try
                    {
                        // Send ping every 15 seconds using optimized send
                        bool pingSent = await client.SendDirectAsync(pingBytes, WebSocketMessageType.Text, true, cancellationTokenSource.Token);
                        
                        if (!pingSent)
                        {
                            Print("Failed to send ping - client connection lost");
                            break;
                        }
                        
                        // Wait for any incoming messages (including pong) with timeout
                        using (var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationTokenSource.Token))
                        {
                            cts.CancelAfter(TimeSpan.FromSeconds(20));
                            
                            try
                            {
                                var result = await client.WebSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);
                                
                                if (result.MessageType == WebSocketMessageType.Text)
                                {
                                    var message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                                    if (message == "pong")
                                    {
                                        client.LastPong = DateTime.UtcNow;
                                    }
                                }
                                else if (result.MessageType == WebSocketMessageType.Close)
                                {
                                    Print("Client requested close");
                                    break;
                                }
                            }
                            catch (OperationCanceledException)
                            {
                                // No response to ping - check if connection is still alive
                                var timeSinceLastPong = DateTime.UtcNow - client.LastPong;
                                if (timeSinceLastPong > TimeSpan.FromSeconds(60))
                                {
                                    Print("Client not responding to pings - marking as dead");
                                    client.IsAlive = false;
                                    break;
                                }
                            }
                        }
                        
                        await Task.Delay(15000, cancellationTokenSource.Token); // Wait 15 seconds before next ping
                    }
                    catch (OperationCanceledException)
                    {
                        // Server shutting down
                        break;
                    }
                    catch (WebSocketException)
                    {
                        Print("WebSocket error in monitoring - client disconnected");
                        break;
                    }
                    catch (Exception ex)
                    {
                        Print($"Monitoring error: {ex.Message}");
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                Print("Monitor connection error: " + ex.Message);
            }
            finally
            {
                client.IsAlive = false;
                
                connectedClients.Remove(client);
                Print("Client disconnected. Total: " + connectedClients.Count);
                
                // Close the WebSocket if it's still open
                if (client.WebSocket.State == WebSocketState.Open)
                {
                    try
                    {
                        await client.WebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Connection closed", CancellationToken.None);
                    }
                    catch { }
                }
                
                // Dispose client resources
                client.Dispose();
            }
        }

        private void CleanupDisconnectedClients(object state)
        {
            if (!isRunning) return;
            
            var clientsToRemove = new List<WebSocketClient>();
            var allClients = connectedClients.ToList();
            
            foreach (var client in allClients)
            {
                if (!client.IsAlive || client.WebSocket.State != WebSocketState.Open)
                {
                    clientsToRemove.Add(client);
                }
            }

            foreach (var client in clientsToRemove)
            {
                connectedClients.Remove(client);
            }
            
            if (clientsToRemove.Count > 0)
            {
                Print("Cleaned up " + clientsToRemove.Count + " disconnected clients");
                
                // Dispose removed clients
                foreach (var client in clientsToRemove)
                {
                    try
                    {
                        client.Dispose();
                    }
                    catch (Exception ex)
                    {
                        Print("Error disposing client: " + ex.Message);
                    }
                }
            }
        }

        private void BroadcastToClientsOptimized(string message)
        {
            if (string.IsNullOrEmpty(message) || !isRunning)
                return;

            var activeClients = connectedClients.ToList().Where(c => c.IsAlive && c.WebSocket.State == WebSocketState.Open).ToList();

            if (activeClients.Count == 0)
                return;

            // Optimize by pre-converting message to bytes once
            byte[] buffer = Encoding.UTF8.GetBytes(message);
            
            // Use queued sending for better performance
            foreach (var client in activeClients)
            {
                try
                {
                    client.QueueMessage(buffer);
                }
                catch (Exception ex)
                {
                    Print("Error queuing message to client: " + ex.Message);
                    client.IsAlive = false;
                }
            }
        }

        private void StopOptimizedWebSocketServer()
        {
            try
            {
                isRunning = false;
                
                // Cancel all operations
                cancellationTokenSource?.Cancel();
                
                // Stop cleanup timer
                cleanupTimer?.Dispose();
                cleanupTimer = null;
                
                if (connectedClients != null)
                {
                    var clientsToClose = connectedClients.ToList();
                    
                    foreach (var client in clientsToClose)
                    {
                        try
                        {
                            client.IsAlive = false;
                            
                            if (client.WebSocket.State == WebSocketState.Open)
                            {
                                Task.Run(async () =>
                                {
                                    try
                                    {
                                        await client.WebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Server stopping", CancellationToken.None);
                                    }
                                    catch { }
                                    finally
                                    {
                                        client.Dispose();
                                    }
                                });
                            }
                            else
                            {
                                client.Dispose();
                            }
                        }
                        catch (Exception ex)
                        {
                            Print("Error closing client: " + ex.Message);
                            try
                            {
                                client.Dispose();
                            }
                            catch { }
                        }
                    }
                }

                httpListener?.Stop();
                httpListener = null;

                // Dispose resources
                cancellationTokenSource?.Dispose();
                broadcastSemaphore?.Dispose();

                Print("TBOT Optimized WebSocket server stopped");
            }
            catch (Exception ex)
            {
                Print("Error stopping optimized server: " + ex.Message);
            }
        }

        protected override void OnMarketData(MarketDataEventArgs e)
        {
            if (!PublishTicks || !isRunning || connectedClients?.Count == 0)
                return;

            // Throttle ticks if configured
            if (TickThrottleMs > 0)
            {
                var now = DateTime.UtcNow;
                if ((now - lastTickTime).TotalMilliseconds < TickThrottleMs)
                    return;
                lastTickTime = now;
            }

            if (e.MarketDataType == MarketDataType.Last)
            {
                string json = "{\"type\":\"tick\",\"symbol\":\"" + Instrument.FullName + "\",\"price\":" + e.Price.ToString("F2") + ",\"volume\":" + e.Volume + ",\"timestamp\":\"" + e.Time.ToString("yyyy-MM-ddTHH:mm:ss.fffZ") + "\"}";
                BroadcastToClientsOptimized(json);
            }
        }

        private DateTime lastTickTime = DateTime.MinValue;

        protected override void OnBarUpdate()
        {
            // Store new bar in historical data (optimized)
            if (BarsInProgress == 0)
            {
                var newBar = new HistoricalBar
                {
                    Timestamp = Time[0],
                    Open = Open[0],
                    High = High[0],
                    Low = Low[0],
                    Close = Close[0],
                    Volume = (long)Volume[0],
                    Symbol = Instrument.FullName
                };
                
                historicalData.Enqueue(newBar);
                currentHistoricalCount++;
                
                // Maintain max size efficiently
                if (currentHistoricalCount > maxHistoricalBars)
                {
                    if (historicalData.TryDequeue(out _))
                        currentHistoricalCount--;
                }
            }

            // Send real-time bar update
            if (!PublishBars || !isRunning || connectedClients?.Count == 0)
                return;

            string json = "{\"type\":\"bar\",\"symbol\":\"" + Instrument.FullName + "\",\"open\":" + Open[0].ToString("F2") + ",\"high\":" + High[0].ToString("F2") + ",\"low\":" + Low[0].ToString("F2") + ",\"close\":" + Close[0].ToString("F2") + ",\"volume\":" + Volume[0] + ",\"timestamp\":\"" + Time[0].ToString("yyyy-MM-ddTHH:mm:ss.fffZ") + "\"}";
            BroadcastToClientsOptimized(json);
        }

        #region Properties
        [NinjaScriptProperty]
        [Range(1024, 65535)]
        [Display(Name = "WebSocket Port", Description = "Port for WebSocket server", Order = 1, GroupName = "Settings")]
        public int WebSocketPort { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Publish Ticks", Description = "Publish tick data", Order = 2, GroupName = "Settings")]
        public bool PublishTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Publish Bars", Description = "Publish bar data", Order = 3, GroupName = "Settings")]
        public bool PublishBars { get; set; }

        // Historical data properties
        [NinjaScriptProperty]
        [Display(Name = "Send Historical On Connect", Description = "Send historical data when client connects", Order = 4, GroupName = "Historical Data")]
        public bool SendHistoricalOnConnect { get; set; }

        [NinjaScriptProperty]
        [Range(100, 10000)]
        [Display(Name = "Historical Bars Count", Description = "Number of historical bars to send", Order = 5, GroupName = "Historical Data")]
        public int HistoricalBarsCount { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Fast Historical Delivery", Description = "Send historical data quickly (recommended for ML training)", Order = 6, GroupName = "Historical Data")]
        public bool FastHistoricalDelivery { get; set; }

        // Performance properties
        [NinjaScriptProperty]
        [Range(1, 50)]
        [Display(Name = "Max Concurrent Connections", Description = "Maximum number of concurrent WebSocket connections", Order = 7, GroupName = "Performance")]
        public int MaxConcurrentConnections { get; set; }

        [NinjaScriptProperty]
        [Range(100, 10000)]
        [Display(Name = "Message Queue Size", Description = "Size of message queue per client", Order = 8, GroupName = "Performance")]
        public int MessageQueueSize { get; set; }

        [NinjaScriptProperty]
        [Range(0, 1000)]
        [Display(Name = "Tick Throttle Ms", Description = "Minimum milliseconds between tick broadcasts (0 = no throttling)", Order = 9, GroupName = "Performance")]
        public int TickThrottleMs { get; set; }
        #endregion
    }
} 

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private TBOTTickWebSocketPublisherOptimized[] cacheTBOTTickWebSocketPublisherOptimized;
		public TBOTTickWebSocketPublisherOptimized TBOTTickWebSocketPublisherOptimized(int webSocketPort, bool publishTicks, bool publishBars, bool sendHistoricalOnConnect, int historicalBarsCount, bool fastHistoricalDelivery, int maxConcurrentConnections, int messageQueueSize, int tickThrottleMs)
		{
			return TBOTTickWebSocketPublisherOptimized(Input, webSocketPort, publishTicks, publishBars, sendHistoricalOnConnect, historicalBarsCount, fastHistoricalDelivery, maxConcurrentConnections, messageQueueSize, tickThrottleMs);
		}

		public TBOTTickWebSocketPublisherOptimized TBOTTickWebSocketPublisherOptimized(ISeries<double> input, int webSocketPort, bool publishTicks, bool publishBars, bool sendHistoricalOnConnect, int historicalBarsCount, bool fastHistoricalDelivery, int maxConcurrentConnections, int messageQueueSize, int tickThrottleMs)
		{
			if (cacheTBOTTickWebSocketPublisherOptimized != null)
				for (int idx = 0; idx < cacheTBOTTickWebSocketPublisherOptimized.Length; idx++)
					if (cacheTBOTTickWebSocketPublisherOptimized[idx] != null && cacheTBOTTickWebSocketPublisherOptimized[idx].WebSocketPort == webSocketPort && cacheTBOTTickWebSocketPublisherOptimized[idx].PublishTicks == publishTicks && cacheTBOTTickWebSocketPublisherOptimized[idx].PublishBars == publishBars && cacheTBOTTickWebSocketPublisherOptimized[idx].SendHistoricalOnConnect == sendHistoricalOnConnect && cacheTBOTTickWebSocketPublisherOptimized[idx].HistoricalBarsCount == historicalBarsCount && cacheTBOTTickWebSocketPublisherOptimized[idx].FastHistoricalDelivery == fastHistoricalDelivery && cacheTBOTTickWebSocketPublisherOptimized[idx].MaxConcurrentConnections == maxConcurrentConnections && cacheTBOTTickWebSocketPublisherOptimized[idx].MessageQueueSize == messageQueueSize && cacheTBOTTickWebSocketPublisherOptimized[idx].TickThrottleMs == tickThrottleMs && cacheTBOTTickWebSocketPublisherOptimized[idx].EqualsInput(input))
						return cacheTBOTTickWebSocketPublisherOptimized[idx];
			return CacheIndicator<TBOTTickWebSocketPublisherOptimized>(new TBOTTickWebSocketPublisherOptimized(){ WebSocketPort = webSocketPort, PublishTicks = publishTicks, PublishBars = publishBars, SendHistoricalOnConnect = sendHistoricalOnConnect, HistoricalBarsCount = historicalBarsCount, FastHistoricalDelivery = fastHistoricalDelivery, MaxConcurrentConnections = maxConcurrentConnections, MessageQueueSize = messageQueueSize, TickThrottleMs = tickThrottleMs }, input, ref cacheTBOTTickWebSocketPublisherOptimized);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.TBOTTickWebSocketPublisherOptimized TBOTTickWebSocketPublisherOptimized(int webSocketPort, bool publishTicks, bool publishBars, bool sendHistoricalOnConnect, int historicalBarsCount, bool fastHistoricalDelivery, int maxConcurrentConnections, int messageQueueSize, int tickThrottleMs)
		{
			return indicator.TBOTTickWebSocketPublisherOptimized(Input, webSocketPort, publishTicks, publishBars, sendHistoricalOnConnect, historicalBarsCount, fastHistoricalDelivery, maxConcurrentConnections, messageQueueSize, tickThrottleMs);
		}

		public Indicators.TBOTTickWebSocketPublisherOptimized TBOTTickWebSocketPublisherOptimized(ISeries<double> input , int webSocketPort, bool publishTicks, bool publishBars, bool sendHistoricalOnConnect, int historicalBarsCount, bool fastHistoricalDelivery, int maxConcurrentConnections, int messageQueueSize, int tickThrottleMs)
		{
			return indicator.TBOTTickWebSocketPublisherOptimized(input, webSocketPort, publishTicks, publishBars, sendHistoricalOnConnect, historicalBarsCount, fastHistoricalDelivery, maxConcurrentConnections, messageQueueSize, tickThrottleMs);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.TBOTTickWebSocketPublisherOptimized TBOTTickWebSocketPublisherOptimized(int webSocketPort, bool publishTicks, bool publishBars, bool sendHistoricalOnConnect, int historicalBarsCount, bool fastHistoricalDelivery, int maxConcurrentConnections, int messageQueueSize, int tickThrottleMs)
		{
			return indicator.TBOTTickWebSocketPublisherOptimized(Input, webSocketPort, publishTicks, publishBars, sendHistoricalOnConnect, historicalBarsCount, fastHistoricalDelivery, maxConcurrentConnections, messageQueueSize, tickThrottleMs);
		}

		public Indicators.TBOTTickWebSocketPublisherOptimized TBOTTickWebSocketPublisherOptimized(ISeries<double> input , int webSocketPort, bool publishTicks, bool publishBars, bool sendHistoricalOnConnect, int historicalBarsCount, bool fastHistoricalDelivery, int maxConcurrentConnections, int messageQueueSize, int tickThrottleMs)
		{
			return indicator.TBOTTickWebSocketPublisherOptimized(input, webSocketPort, publishTicks, publishBars, sendHistoricalOnConnect, historicalBarsCount, fastHistoricalDelivery, maxConcurrentConnections, messageQueueSize, tickThrottleMs);
		}
	}
}

#endregion
