#!/usr/bin/env python3
"""
NinjaTrader WebSocket Client for ML Strategy Testing
==================================================

Advanced WebSocket client to test the TBOTTickWebSocketPublisherOptimized
NinjaScript indicator. Features real-time data reception, historical data
processing, and comprehensive analytics with streaming CSV persistence.

Author: TBOT ML Strategy Team
Date: 2025-06-16
"""

import asyncio
import json
import logging
import time
import csv
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any, TextIO
from dataclasses import dataclass, field
from pathlib import Path
import signal
import sys
from collections import deque
import queue
from concurrent.futures import ThreadPoolExecutor

import websockets
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/websocket_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ClientStats:
    """Statistics tracking for WebSocket client"""
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    total_messages: int = 0
    historical_bars_received: int = 0
    real_time_ticks: int = 0
    real_time_bars: int = 0
    ping_count: int = 0
    errors: int = 0
    last_message_time: Optional[datetime] = None
    message_types: Dict[str, int] = field(default_factory=dict)
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))  # Fixed size for efficiency
    records_written: int = 0
    buffer_flushes: int = 0
    
    @property
    def connection_duration(self) -> Optional[float]:
        if self.connected_at and self.disconnected_at:
            return (self.disconnected_at - self.connected_at).total_seconds()
        elif self.connected_at:
            return (datetime.now(timezone.utc) - self.connected_at).total_seconds()
        return None
    
    @property
    def messages_per_second(self) -> float:
        duration = self.connection_duration
        if duration and duration > 0:
            return self.total_messages / duration
        return 0.0
    
    @property
    def average_latency(self) -> float:
        if self.latency_samples:
            return np.mean(self.latency_samples)
        return 0.0

@dataclass
class MarketData:
    """Market data structure for bars and ticks"""
    timestamp: datetime
    symbol: str
    data_type: str  # 'tick', 'bar', 'historical_bar'
    price: Optional[float] = None
    volume: Optional[int] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    
    def to_csv_row(self) -> str:
        """Convert to pre-formatted CSV line for maximum speed"""
        # Format timestamp as: 2025-05-06 06:50:00+00:00
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S%z')
        if timestamp_str.endswith('+0000'):
            timestamp_str = timestamp_str[:-5] + '+00:00'
        
        # Use OHLCV format - for ticks, use price as all OHLC values
        if self.data_type == 'tick':
            open_val = high_val = low_val = close_val = self.price or ''
            volume_val = self.volume or ''
        else:  # bar data
            open_val = self.open_price or ''
            high_val = self.high_price or ''
            low_val = self.low_price or ''
            close_val = self.close_price or ''
            volume_val = self.volume or ''
        
        return f"{timestamp_str},{open_val},{high_val},{low_val},{close_val},{volume_val}\n"

class HighSpeedCSVWriter:
    """Ultra-fast CSV writer using background thread and large buffers"""
    
    def __init__(self, filepath: str, buffer_size: int = 1000, max_queue_size: int = 10000):
        """
        Initialize high-speed CSV writer
        
        Args:
            filepath: Path to CSV file
            buffer_size: Records per write batch (much larger)
            max_queue_size: Maximum queue size before blocking
        """
        self.filepath = Path(filepath)
        self.buffer_size = buffer_size
        self.max_queue_size = max_queue_size
        
        # Thread-safe queue for data transfer
        self.write_queue = queue.Queue(maxsize=max_queue_size)
        self.shutdown_event = threading.Event()
        
        # Statistics (thread-safe with locks)
        self._stats_lock = threading.Lock()
        self.records_written = 0
        self.flush_count = 0
        self.queue_size = 0
        
        # Background writer thread
        self.writer_thread = None
        self.is_initialized = False
        
        # CSV header
        self.csv_header = "timestamp,open,high,low,close,volume\n"
    
    def start(self):
        """Start the background writer thread"""
        if self.writer_thread is None:
            self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
            self.writer_thread.start()
            logger.info(f"🚀 High-speed CSV writer started: {self.filepath}")
    
    def write_record(self, market_data: MarketData):
        """Add record to write queue (non-blocking)"""
        try:
            # Pre-format the CSV line for maximum speed
            csv_line = market_data.to_csv_row()
            
            # Put in queue (this is very fast)
            self.write_queue.put_nowait(csv_line)
            
            # Update queue size for monitoring
            self.queue_size = self.write_queue.qsize()
            
        except queue.Full:
            # Queue full - this means we're producing data faster than writing
            logger.warning("⚠️ Write queue full - dropping record to prevent blocking")
            with self._stats_lock:
                # Could increment a "dropped records" counter here
                pass
    
    def _writer_worker(self):
        """Background thread worker for file writing"""
        try:
            # Create directory if needed
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Open file in binary mode for maximum speed
            with open(self.filepath, 'ab') as f:
                # Write header if file is empty
                if self.filepath.stat().st_size == 0:
                    f.write(self.csv_header.encode('utf-8'))
                
                buffer = []
                
                while not self.shutdown_event.is_set() or not self.write_queue.empty():
                    try:
                        # Get record with timeout
                        csv_line = self.write_queue.get(timeout=1.0)
                        buffer.append(csv_line)
                        
                        # Write when buffer is full or periodically
                        if len(buffer) >= self.buffer_size:
                            self._flush_buffer(f, buffer)
                            buffer = []
                        
                    except queue.Empty:
                        # Timeout - flush any pending data
                        if buffer:
                            self._flush_buffer(f, buffer)
                            buffer = []
                        continue
                
                # Final flush
                if buffer:
                    self._flush_buffer(f, buffer)
                
        except Exception as e:
            logger.error(f"❌ CSV writer thread error: {e}")
    
    def _flush_buffer(self, file_handle, buffer: List[str]):
        """Flush buffer to file (called from writer thread)"""
        try:
            # Concatenate all lines and write as one operation
            content = ''.join(buffer).encode('utf-8')
            file_handle.write(content)
            file_handle.flush()
            
            # Update statistics
            with self._stats_lock:
                self.records_written += len(buffer)
                self.flush_count += 1
            
            # Update queue size
            self.queue_size = self.write_queue.qsize()
            
            logger.debug(f"💾 Flushed {len(buffer)} records (Total: {self.records_written})")
            
        except Exception as e:
            logger.error(f"Error flushing CSV buffer: {e}")
    
    def close(self):
        """Shutdown writer and close file"""
        logger.info("🛑 Shutting down CSV writer...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for writer thread to finish
        if self.writer_thread:
            self.writer_thread.join(timeout=10.0)
            if self.writer_thread.is_alive():
                logger.warning("⚠️ CSV writer thread did not shutdown cleanly")
        
        logger.info(f"✅ CSV writer closed. Total records: {self.records_written}, Flushes: {self.flush_count}")

class WebSocketClient:
    """Advanced WebSocket client for NinjaTrader integration"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6789,
                 auto_reconnect: bool = True,
                 max_reconnect_attempts: int = 5,
                 csv_file: str = "ultra_fast_market_data.csv",
                 buffer_size: int = 1000,
                 max_queue_size: int = 10000,
                 high_performance_mode: bool = True):
        """
        Initialize WebSocket client
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
            auto_reconnect: Enable automatic reconnection
            max_reconnect_attempts: Maximum reconnection attempts
            csv_file: Path for high-speed CSV output
            buffer_size: CSV buffer size for batch writes (much larger)
            max_queue_size: Maximum queue size before blocking
            high_performance_mode: Enable ultra-fast mode (minimal processing during historical load)
        """
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}/"
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.high_performance_mode = high_performance_mode
        
        # State management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.is_running = False
        self.reconnect_count = 0
        
        # Performance flags
        self.historical_loading = False
        self.historical_count = 0
        self.last_ping_response = time.time()
        
        # High-speed CSV writer
        self.csv_writer = HighSpeedCSVWriter(
            csv_file, 
            buffer_size=buffer_size,
            max_queue_size=max_queue_size
        )
        
        # Keep limited recent data in memory for display (using deque for efficiency)
        self.recent_data = deque(maxlen=500)  # Reduced for better performance
        self.stats = ClientStats()
        
        # Event callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_market_data: Optional[Callable[[MarketData], None]] = None
        self.on_historical_start: Optional[Callable[[Dict], None]] = None
        self.on_historical_end: Optional[Callable[[Dict], None]] = None
        
        # Display
        self.console = Console()
        self.display_table: Optional[Table] = None
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Start CSV writer
        self.csv_writer.start()
    
    async def connect(self) -> bool:
        """Connect to WebSocket server"""
        try:
            logger.info(f"Connecting to {self.uri}")
            
            # Optimized connection settings for high volume data
            self.websocket = await websockets.connect(
                self.uri,
                ping_interval=30,      # Increased from 20 seconds
                ping_timeout=25,       # Increased from 10 seconds  
                close_timeout=15,      # Increased from 10 seconds
                max_size=2**20,        # 1MB max message size
                max_queue=50           # Reduced queue size for faster processing
            )
            
            self.is_connected = True
            self.stats.connected_at = datetime.now(timezone.utc)
            self.reconnect_count = 0
            self.last_ping_response = time.time()
            
            logger.info("✅ Connected to NinjaTrader WebSocket server (High Performance Mode)")
            
            if self.on_connected:
                await self.on_connected()
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.stats.errors += 1
            return False
    
    async def disconnect(self):
        """Gracefully disconnect from server"""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.close()
                logger.info("✅ Disconnected gracefully")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        self.is_connected = False
        self.stats.disconnected_at = datetime.now(timezone.utc)
        
        # Close CSV writer
        self.csv_writer.close()
        
        if self.on_disconnected:
            await self.on_disconnected()
    
    async def listen(self):
        """Main message listening loop - optimized for high performance"""
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket server")
        
        try:
            async for message in self.websocket:
                # Handle ping/pong messages (sent as plain text) - PRIORITY
                if message == "ping":
                    self.stats.ping_count += 1
                    self.stats.total_messages += 1
                    self.last_ping_response = time.time()
                    
                    try:
                        await self.websocket.send("pong")
                        # Only log in non-performance mode to reduce overhead
                        if not self.high_performance_mode or not self.historical_loading:
                            logger.debug("Sent pong response")
                    except Exception as e:
                        logger.error(f"Error sending pong: {e}")
                    continue
                
                # Handle JSON messages with performance optimization
                await self._handle_message_optimized(message)
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            if self.auto_reconnect and self.reconnect_count < self.max_reconnect_attempts:
                await self._attempt_reconnect()
            else:
                self.is_connected = False
                
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
            self.stats.errors += 1
    
    async def _handle_message_optimized(self, message: str):
        """Handle incoming WebSocket message with performance optimizations"""
        try:
            # Parse JSON message (fastest operation first)
            data = json.loads(message)
            message_type = data.get('type', 'unknown')
            
            # Update basic statistics (minimal overhead)
            self.stats.total_messages += 1
            
            # Handle different message types with performance optimizations
            if message_type == "historical_start":
                await self._handle_historical_start_optimized(data)
                
            elif message_type == "historical_bar":
                await self._handle_historical_bar_optimized(data)
                
            elif message_type == "historical_end":
                await self._handle_historical_end_optimized(data)
                
            elif message_type == "tick":
                await self._handle_real_time_tick_optimized(data)
                
            elif message_type == "bar":
                await self._handle_real_time_bar_optimized(data)
                
            else:
                if not self.high_performance_mode or not self.historical_loading:
                    logger.warning(f"Unknown message type: {message_type}")
            
            # Only update detailed stats when NOT in high-performance historical loading
            if not self.high_performance_mode or not self.historical_loading:
                receive_time = datetime.now(timezone.utc)
                self.stats.last_message_time = receive_time
                self.stats.message_types[message_type] = self.stats.message_types.get(message_type, 0) + 1
                
                # Calculate latency if timestamp available (fixed calculation)
                if 'timestamp' in data:
                    try:
                        msg_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                        latency = (receive_time - msg_time).total_seconds() * 1000
                        
                        # Only add reasonable latency values (0-10 seconds)
                        if 0 <= latency <= 10000:
                            self.stats.latency_samples.append(latency)
                            
                    except Exception:
                        pass  # Ignore timestamp parsing errors
                    
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            self.stats.errors += 1
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.stats.errors += 1
    

    async def _handle_historical_start_optimized(self, data: Dict):
        """Handle historical data start marker - optimized"""
        self.historical_loading = True
        self.historical_count = 0
        expected_count = data.get('count', 0)
        
        logger.info(f"📊 Historical data stream starting: {expected_count} bars (High Performance Mode: {self.high_performance_mode})")
        if self.high_performance_mode:
            logger.info("⚡ Enabling ultra-fast processing - minimal logging/stats during historical load")
        
        self.recent_data.clear()
        
        if self.on_historical_start:
            await self.on_historical_start(data)
    
    async def _handle_historical_bar_optimized(self, data: Dict):
        """Handle historical bar data - ultra-fast processing"""
        try:
            # Create MarketData with minimal processing
            bar = MarketData(
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                symbol=data.get('symbol', ''),
                data_type='historical_bar',
                open_price=float(data.get('open', 0)),
                high_price=float(data.get('high', 0)),
                low_price=float(data.get('low', 0)),
                close_price=float(data.get('close', 0)),
                volume=int(data.get('volume', 0))
            )
            
            # Stream to CSV immediately (non-blocking queue operation)
            self.csv_writer.write_record(bar)
            
            # Increment counters
            self.historical_count += 1
            self.stats.historical_bars_received += 1
            
            # Only add to display queue occasionally to save memory during high-volume load
            if not self.high_performance_mode or self.historical_count % 50 == 0:
                self.recent_data.append(bar)
            
            # Log progress much less frequently during high-performance mode
            if self.high_performance_mode:
                # Log every 10,000 bars instead of 100
                if self.historical_count % 10000 == 0:
                    logger.info(f"📈 Processed {self.historical_count} historical bars (High Speed Mode)")
            else:
                # Normal mode: log every 100 bars
                if self.historical_count % 100 == 0:
                    logger.info(f"📈 Received {self.historical_count} historical bars")
                
        except Exception as e:
            logger.error(f"Error processing historical bar: {e}")
            self.stats.errors += 1
    
    async def _handle_historical_end_optimized(self, data: Dict):
        """Handle historical data end marker - optimized"""
        self.historical_loading = False
        total_sent = data.get('sent', 0)
        
        logger.info(f"✅ Historical data complete: {total_sent} bars sent, {self.historical_count} processed")
        logger.info(f"🚀 Returning to normal performance mode - full stats and logging enabled")
        
        if self.on_historical_end:
            await self.on_historical_end(data)
    
    async def _handle_real_time_tick_optimized(self, data: Dict):
        """Handle real-time tick data - optimized"""
        try:
            tick = MarketData(
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                symbol=data.get('symbol', ''),
                data_type='tick',
                price=float(data.get('price', 0)),
                volume=int(data.get('volume', 0))
            )
            
            # Stream to CSV immediately
            self.csv_writer.write_record(tick)
            
            # Keep recent data for display
            self.recent_data.append(tick)
            self.stats.real_time_ticks += 1
            
            if self.on_market_data:
                await self.on_market_data(tick)
                
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            self.stats.errors += 1
    
    async def _handle_real_time_bar_optimized(self, data: Dict):
        """Handle real-time bar data - optimized"""
        try:
            bar = MarketData(
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                symbol=data.get('symbol', ''),
                data_type='bar',
                open_price=float(data.get('open', 0)),
                high_price=float(data.get('high', 0)),
                low_price=float(data.get('low', 0)),
                close_price=float(data.get('close', 0)),
                volume=int(data.get('volume', 0))
            )
            
            # Stream to CSV immediately
            self.csv_writer.write_record(bar)
            
            # Keep recent data for display
            self.recent_data.append(bar)
            self.stats.real_time_bars += 1
            
            if self.on_market_data:
                await self.on_market_data(bar)
                
        except Exception as e:
            logger.error(f"Error processing bar: {e}")
            self.stats.errors += 1
    
    async def _attempt_reconnect(self):
        """Attempt to reconnect to server"""
        self.reconnect_count += 1
        logger.info(f"🔄 Attempting reconnection {self.reconnect_count}/{self.max_reconnect_attempts}")
        
        await asyncio.sleep(2 ** self.reconnect_count)  # Exponential backoff
        
        if await self.connect():
            logger.info("✅ Reconnection successful")
            await self.listen()
        else:
            logger.error("❌ Reconnection failed")
    
    def get_stats_table(self) -> Table:
        """Generate rich table with current statistics"""
        table = Table(title="📊 WebSocket Client Statistics", box=box.ROUNDED)
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Details", style="yellow")
        
        # Connection info
        status = "🟢 Connected" if self.is_connected else "🔴 Disconnected"
        table.add_row("Status", status, f"Reconnects: {self.reconnect_count}")
        
        duration = self.stats.connection_duration
        if duration:
            table.add_row("Duration", f"{duration:.1f}s", f"Rate: {self.stats.messages_per_second:.1f} msg/s")
        
        # Message counts
        table.add_row("Total Messages", str(self.stats.total_messages), f"Errors: {self.stats.errors}")
        table.add_row("Historical Bars", str(self.stats.historical_bars_received), "Backfill data")
        table.add_row("Real-time Ticks", str(self.stats.real_time_ticks), "Live market data")
        table.add_row("Real-time Bars", str(self.stats.real_time_bars), "Live bar updates")
        table.add_row("Ping Messages", str(self.stats.ping_count), "Keep-alive")
        
        # Performance metrics
        if self.stats.latency_samples:
            avg_latency = self.stats.average_latency
            table.add_row("Avg Latency", f"{avg_latency:.1f}ms", f"Samples: {len(self.stats.latency_samples)}")
        
        # CSV streaming metrics
        csv_records = self.csv_writer.records_written
        csv_flushes = self.csv_writer.flush_count
        table.add_row("CSV Records", str(csv_records), f"Flushes: {csv_flushes}")
        
        # Buffer status
        queue_size = self.csv_writer.queue_size
        queue_pct = (queue_size / self.csv_writer.max_queue_size) * 100
        table.add_row("CSV Queue", f"{queue_size}/{self.csv_writer.max_queue_size}", f"{queue_pct:.1f}% full")
        
        return table
    
    def save_data_to_csv(self, filepath: str = "market_data.csv"):
        """Save received market data to CSV file"""
        try:
            all_data = list(self.recent_data)
            if not all_data:
                logger.warning("No data to save")
                return
            
            # Convert to DataFrame
            df_data = []
            for item in all_data:
                row = item.to_csv_row()
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)
            logger.info(f"💾 Data saved to {filepath} ({len(df)} records)")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    async def run(self):
        """Main run loop with graceful shutdown handling"""
        self.is_running = True
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("🛑 Shutdown signal received")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            if await self.connect():
                await self.listen()
            else:
                logger.error("Failed to establish initial connection")
                
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.disconnect()
            logger.info("🏁 Client shutdown complete")


async def main():
    """Main entry point with live statistics display"""
    console = Console()
    
    # Create client with ultra-high-performance configuration
    client = WebSocketClient(
        host="localhost",
        port=6789,
        auto_reconnect=True,
        max_reconnect_attempts=5,
        csv_file="ultra_fast_market_data.csv",
        buffer_size=2000,           # Larger buffer for high-volume data
        max_queue_size=20000,       # Larger queue to handle 1M+ historical bars
        high_performance_mode=True  # Enable ultra-fast mode for historical data
    )
    
    # Setup event handlers
    async def on_connected():
        console.print("🚀 [green]Connected to NinjaTrader WebSocket server![/green]")
    
    async def on_disconnected():
        console.print("🔌 [red]Disconnected from server[/red]")
    
    async def on_market_data(data: MarketData):
        if data.data_type == 'tick':
            console.print(f"📈 TICK: {data.symbol} @ {data.price} (Vol: {data.volume})")
        elif data.data_type == 'bar':
            console.print(f"📊 BAR: {data.symbol} OHLC: {data.open_price}/{data.high_price}/{data.low_price}/{data.close_price}")
    
    client.on_connected = on_connected
    client.on_disconnected = on_disconnected
    client.on_market_data = on_market_data
    
    # Create live display
    layout = Layout()
    layout.split_row(
        Layout(name="stats", size=80),
        Layout(name="info", size=40)
    )
    
    # Run client with live statistics
    try:
        with Live(layout, refresh_per_second=2, screen=True):
            # Update display task (optimized for high-performance mode)
            async def update_display():
                while client.is_running:
                    try:
                        # Reduce update frequency during historical loading to prevent blocking
                        if client.historical_loading and client.high_performance_mode:
                            # Update much less frequently during historical data loading
                            await asyncio.sleep(5.0)
                            # Simple status update during high-performance loading
                            simple_stats = Table(title="📊 High-Performance Mode", box=box.ROUNDED)
                            simple_stats.add_column("Status", style="cyan")
                            simple_stats.add_column("Value", style="green")
                            simple_stats.add_row("Mode", "🚀 Ultra-Fast Historical Loading")
                            simple_stats.add_row("Processed", f"{client.historical_count:,} bars")
                            simple_stats.add_row("CSV Records", f"{client.csv_writer.records_written:,}")
                            simple_stats.add_row("Queue", f"{client.csv_writer.queue_size}")
                            layout["stats"].update(Panel(simple_stats))
                        else:
                            # Normal detailed stats when not in historical loading
                            layout["stats"].update(Panel(client.get_stats_table()))
                            await asyncio.sleep(0.5)
                        
                        info_text = Text()
                        info_text.append("🔗 Connection: ", style="bold")
                        info_text.append(f"{client.uri}\n", style="cyan")
                        info_text.append("📝 Commands:\n", style="bold")
                        info_text.append("  Ctrl+C: Quit\n", style="white")
                        info_text.append("  Data saved on exit\n", style="white")
                        
                        if client.historical_loading:
                            info_text.append("\n⚡ High-Performance Mode\n", style="yellow bold")
                            info_text.append("  Minimal logging/stats\n", style="yellow")
                            info_text.append("  during historical load\n", style="yellow")
                        
                        layout["info"].update(Panel(info_text, title="Info"))
                        
                    except Exception as e:
                        # Don't let display errors affect the main processing
                        await asyncio.sleep(1.0)
            
            # Start both tasks
            await asyncio.gather(
                client.run(),
                update_display()
            )
            
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Shutdown requested by user[/yellow]")
    finally:
        # CSV data is already streamed during operation
        # Optional: Save recent display data as backup
        if len(client.recent_data) > 0:
            client.save_data_to_csv("recent_data_backup.csv")
            console.print(f"✅ [green]Backup of {len(client.recent_data)} recent records saved[/green]")
        console.print("✅ [green]Streaming CSV data and client shutdown complete[/green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Client stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 