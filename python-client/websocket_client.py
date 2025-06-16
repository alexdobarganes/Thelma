#!/usr/bin/env python3
"""
NinjaTrader WebSocket Client for ML Strategy Testing
==================================================

Advanced WebSocket client to test the TBOTTickWebSocketPublisherOptimized
NinjaScript indicator. Features real-time data reception, historical data
processing, and comprehensive analytics.

Author: TBOT ML Strategy Team
Date: 2025-06-16
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import signal
import sys

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
    latency_samples: List[float] = field(default_factory=list)
    
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

class WebSocketClient:
    """Advanced WebSocket client for NinjaTrader integration"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6789,
                 auto_reconnect: bool = True,
                 max_reconnect_attempts: int = 5):
        """
        Initialize WebSocket client
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
            auto_reconnect: Enable automatic reconnection
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}/"
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # State management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.is_running = False
        self.reconnect_count = 0
        
        # Data storage
        self.market_data: List[MarketData] = []
        self.historical_data: List[MarketData] = []
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
    
    async def connect(self) -> bool:
        """Connect to WebSocket server"""
        try:
            logger.info(f"Connecting to {self.uri}")
            self.websocket = await websockets.connect(
                self.uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.stats.connected_at = datetime.now(timezone.utc)
            self.reconnect_count = 0
            
            logger.info("✅ Connected to NinjaTrader WebSocket server")
            
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
        
        if self.on_disconnected:
            await self.on_disconnected()
    
    async def listen(self):
        """Main message listening loop"""
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket server")
        
        try:
            async for message in self.websocket:
                # Handle ping/pong messages (sent as plain text)
                if message == "ping":
                    self.stats.ping_count += 1
                    self.stats.total_messages += 1
                    self.stats.last_message_time = datetime.now(timezone.utc)
                    
                    try:
                        await self.websocket.send("pong")
                        logger.debug("Sent pong response")
                    except Exception as e:
                        logger.error(f"Error sending pong: {e}")
                    continue
                
                await self._handle_message(message)
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            if self.auto_reconnect and self.reconnect_count < self.max_reconnect_attempts:
                await self._attempt_reconnect()
            else:
                self.is_connected = False
                
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
            self.stats.errors += 1
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            # Track timing for latency calculation
            receive_time = datetime.now(timezone.utc)
            
            # Parse JSON message
            data = json.loads(message)
            message_type = data.get('type', 'unknown')
            
            # Update statistics
            self.stats.total_messages += 1
            self.stats.last_message_time = receive_time
            self.stats.message_types[message_type] = self.stats.message_types.get(message_type, 0) + 1
            
            # Handle different message types
            if message_type == "historical_start":
                await self._handle_historical_start(data)
                
            elif message_type == "historical_bar":
                await self._handle_historical_bar(data)
                
            elif message_type == "historical_end":
                await self._handle_historical_end(data)
                
            elif message_type == "tick":
                await self._handle_real_time_tick(data)
                
            elif message_type == "bar":
                await self._handle_real_time_bar(data)
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
            
            # Calculate latency if timestamp available
            if 'timestamp' in data:
                try:
                    msg_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    latency = (receive_time - msg_time).total_seconds() * 1000
                    self.stats.latency_samples.append(latency)
                    
                    # Keep only last 1000 samples for memory efficiency
                    if len(self.stats.latency_samples) > 1000:
                        self.stats.latency_samples = self.stats.latency_samples[-1000:]
                        
                except Exception:
                    pass  # Ignore timestamp parsing errors
                    
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            self.stats.errors += 1
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.stats.errors += 1
    

    async def _handle_historical_start(self, data: Dict):
        """Handle historical data start marker"""
        logger.info(f"📊 Historical data stream starting: {data.get('count', 'unknown')} bars")
        self.historical_data.clear()
        
        if self.on_historical_start:
            await self.on_historical_start(data)
    
    async def _handle_historical_bar(self, data: Dict):
        """Handle historical bar data"""
        try:
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
            
            self.historical_data.append(bar)
            self.stats.historical_bars_received += 1
            
            # Log progress every 100 bars
            if self.stats.historical_bars_received % 100 == 0:
                logger.info(f"📈 Received {self.stats.historical_bars_received} historical bars")
                
        except Exception as e:
            logger.error(f"Error processing historical bar: {e}")
            self.stats.errors += 1
    
    async def _handle_historical_end(self, data: Dict):
        """Handle historical data end marker"""
        total_sent = data.get('sent', 0)
        logger.info(f"✅ Historical data complete: {total_sent} bars sent, {len(self.historical_data)} received")
        
        if self.on_historical_end:
            await self.on_historical_end(data)
    
    async def _handle_real_time_tick(self, data: Dict):
        """Handle real-time tick data"""
        try:
            tick = MarketData(
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                symbol=data.get('symbol', ''),
                data_type='tick',
                price=float(data.get('price', 0)),
                volume=int(data.get('volume', 0))
            )
            
            self.market_data.append(tick)
            self.stats.real_time_ticks += 1
            
            if self.on_market_data:
                await self.on_market_data(tick)
                
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            self.stats.errors += 1
    
    async def _handle_real_time_bar(self, data: Dict):
        """Handle real-time bar data"""
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
            
            self.market_data.append(bar)
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
        
        return table
    
    def save_data_to_csv(self, filepath: str = "market_data.csv"):
        """Save received market data to CSV file"""
        try:
            all_data = self.historical_data + self.market_data
            if not all_data:
                logger.warning("No data to save")
                return
            
            # Convert to DataFrame
            df_data = []
            for item in all_data:
                row = {
                    'timestamp': item.timestamp,
                    'symbol': item.symbol,
                    'type': item.data_type,
                    'price': item.price,
                    'volume': item.volume,
                    'open': item.open_price,
                    'high': item.high_price,
                    'low': item.low_price,
                    'close': item.close_price
                }
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
    
    # Create client
    client = WebSocketClient(
        host="localhost",
        port=6789,
        auto_reconnect=True,
        max_reconnect_attempts=5
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
            # Update display task
            async def update_display():
                while client.is_running:
                    layout["stats"].update(Panel(client.get_stats_table()))
                    
                    info_text = Text()
                    info_text.append("🔗 Connection: ", style="bold")
                    info_text.append(f"{client.uri}\n", style="cyan")
                    info_text.append("📝 Commands:\n", style="bold")
                    info_text.append("  Ctrl+C: Quit\n", style="white")
                    info_text.append("  Data saved on exit\n", style="white")
                    
                    layout["info"].update(Panel(info_text, title="Info"))
                    await asyncio.sleep(0.5)
            
            # Start both tasks
            await asyncio.gather(
                client.run(),
                update_display()
            )
            
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Shutdown requested by user[/yellow]")
    finally:
        # Save data before exit
        client.save_data_to_csv("received_market_data.csv")
        console.print("✅ [green]Data saved and client shutdown complete[/green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Client stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 