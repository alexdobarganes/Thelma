# Copy of launch_vreversal_system.py with EXTENDED trading windows for more signals
# This version trades during regular market hours instead of just 3 specific windows

import asyncio
import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import subprocess
import threading

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "models"))
sys.path.append(str(project_root / "python-client"))

try:
    from vreversal_realtime_detector import VReversalRealtimeDetector, VReversalConfig
    from websocket_client import WebSocketClient, MarketData
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed and paths are correct")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vreversal_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# EXTENDED TRADING WINDOWS VERSION
class ExtendedVReversalConfig(VReversalConfig):
    """Extended config with broader trading windows"""
    pass

# Override the trading window check
class ExtendedVReversalRealtimeDetector(VReversalRealtimeDetector):
    """Extended detector with broader trading windows"""
    
    def _check_for_patterns(self):
        """Override to use extended trading windows"""
        try:
            # Check daily limit
            today = datetime.now().date()
            if self.last_signal_date != today:
                self.daily_signal_count = 0
                self.last_signal_date = today
            
            if self.daily_signal_count >= self.config.max_daily_signals:
                return
            
            # EXTENDED TRADING WINDOWS - Regular market hours
            from datetime import timezone, timedelta
            
            eastern_tz = timezone(timedelta(hours=-5))  # EST
            current_time_et = datetime.now(eastern_tz).time()
            current_date = datetime.now(eastern_tz).date()
            current_weekday = current_date.weekday()  # 0=Monday, 6=Sunday
            
            # Check if it's a trading day (Monday-Friday)
            if current_weekday >= 5:  # Saturday=5, Sunday=6
                logger.debug("🚫 No trading on weekends")
                return
            
            # EXTENDED HOURS: 9:30 AM - 4:00 PM ET (full regular session)
            session_start = datetime.strptime("09:30", "%H:%M").time()
            session_end = datetime.strptime("16:00", "%H:%M").time()
            
            if not (session_start <= current_time_et <= session_end):
                logger.debug(f"🚫 Outside extended trading hours: {current_time_et.strftime('%H:%M')} ET")
                return
            
            # Convert buffer to DataFrame for pattern detection
            import pandas as pd
            df = pd.DataFrame(list(self.bar_buffer))
            if len(df) < 30:  # Need minimum bars
                return
            
            # Run pattern detection on recent bars
            patterns = self._detect_patterns_in_buffer(df)
            
            # Process any detected patterns
            for pattern in patterns:
                if pattern.signal_id not in self.processed_patterns:
                    self._generate_signal_file(pattern)
                    self.processed_patterns.add(pattern.signal_id)
                    self.daily_signal_count += 1
                    logger.info(f"🎯 V-reversal detected! Signal {pattern.signal_id} generated")
                    
                    if self.daily_signal_count >= self.config.max_daily_signals:
                        logger.info(f"📊 Daily signal limit reached ({self.config.max_daily_signals})")
                        break
                        
        except Exception as e:
            logger.error(f"❌ Error checking patterns: {e}")

def main():
    """Main function with extended trading windows"""
    parser = argparse.ArgumentParser(description="V-Reversal Trading System Launcher - Extended Hours")
    parser.add_argument("--websocket-host", default="192.168.1.65", help="NinjaTrader WebSocket host")
    parser.add_argument("--websocket-port", type=int, default=6789, help="NinjaTrader WebSocket port")
    parser.add_argument("--signal-folder", default="signals", help="Signal output folder")
    parser.add_argument("--max-daily-signals", type=int, default=50, help="Maximum signals per day (increased)")
    
    args = parser.parse_args()
    
    # Create configuration with EXTENDED hours
    config = ExtendedVReversalConfig()
    
    # $2300/DAY MODEL PARAMETERS but with EXTENDED trading windows
    config.drop_threshold = 4.0           # Production validated (4.0 points)
    config.drop_window = 15               # 15 minutes for drop completion
    config.breakout_window = 30           # 30 minutes for breakout
    config.pullback_window = 15           # 15 minutes for pullback
    config.continuation_window = 20       # 20 minutes for continuation
    config.pullback_tolerance = 1.0       # 1 point pullback tolerance
    config.stop_loss_pct = 0.001          # 0.1% - PRODUCTION PROVEN parameter
    config.max_hold_time = 25             # 25 minutes max hold
    config.min_pattern_confidence = 0.75   # 75% minimum confidence
    
    # EXTENDED Trading session configuration
    config.signal_folder = args.signal_folder
    config.max_daily_signals = args.max_daily_signals
    config.session_start = "09:30"        # Market open ET
    config.session_end = "16:00"          # Market close ET (EXTENDED)
    
    # Display startup banner
    console = Console()
    console.print(Panel(
        "[bold green]🎯 $2300/DAY V-REVERSAL SYSTEM - EXTENDED HOURS[/bold green]\n\n"
        f"[cyan]Strategy:[/cyan] Drop_4 + 0.1% Stop Loss (PRODUCTION)\n"
        f"[cyan]WebSocket:[/cyan] {args.websocket_host}:{args.websocket_port}\n"
        f"[cyan]Signals:[/cyan] {config.signal_folder}/\n"
        f"[cyan]Hours:[/cyan] 9:30 AM - 4:00 PM ET (FULL SESSION)\n"
        f"[cyan]Max Daily:[/cyan] {config.max_daily_signals} signals\n\n"
        "[yellow]📊 EXTENDED HOURS TRADING:[/yellow]\n"
        "[yellow]• Based on $2300/day model parameters[/yellow]\n"
        "[yellow]• Expanded from 4.5 to 6.5 hours/day[/yellow]\n"
        "[yellow]• More opportunities for signal generation[/yellow]\n\n"
        "[yellow]🚀 Starting extended hours detection...[/yellow]\n"
        "[yellow]🎮 Press Ctrl+C to stop[/yellow]",
        title="EXTENDED HOURS LAUNCH",
        border_style="green"
    ))
    
    # Use the extended detector
    class VReversalSystemManager:
        def __init__(self, config, websocket_host, websocket_port):
            self.config = config
            self.websocket_host = websocket_host
            self.websocket_port = websocket_port
            self.console = Console()
            
            # Use extended detector
            self.detector = None
            self.websocket_client = None
            
            # System state
            self.system_running = False
            self.startup_time = None
            self.total_signals_generated = 0
            self.last_signal_time = None
            self.connection_status = "Disconnected"
            
            # Statistics
            self.bars_processed = 0
            self.patterns_detected = 0
            self.system_errors = 0
            
            # Create required directories
            self._setup_directories()
        
        def _setup_directories(self):
            """Create required directories"""
            dirs = [
                self.config.signal_folder,
                f"{self.config.signal_folder}/processed",
                "logs",
                "data"
            ]
            
            for dir_path in dirs:
                Path(dir_path).mkdir(exist_ok=True, parents=True)
        
        async def start_system(self):
            """Start the extended V-reversal system"""
            try:
                self.system_running = True
                self.startup_time = datetime.now()
                
                logger.info("🚀 Starting V-Reversal Trading System (Extended Hours)...")
                
                # Initialize EXTENDED detector
                self.detector = ExtendedVReversalRealtimeDetector(self.config)
                
                # Initialize WebSocket client
                self.websocket_client = WebSocketClient(
                    host=self.websocket_host,
                    port=self.websocket_port,
                    csv_file="data/realtime_market_data.csv"
                )
                
                # Set up market data callback
                async def on_market_data(data: MarketData):
                    try:
                        self.connection_status = "Connected"
                        
                        # Only process 1-minute bars from NinjaTrader
                        if data.data_type == 'bar':
                            bar_data = {
                                'timestamp': data.timestamp,
                                'open': data.open_price,
                                'high': data.high_price,
                                'low': data.low_price,
                                'close': data.close_price,
                                'volume': data.volume
                            }
                            
                            # Feed to detector
                            self.detector.add_bar_data(bar_data)
                            self.bars_processed += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing market data: {e}")
                        self.system_errors += 1
                
                # Set up callbacks
                self.websocket_client.on_market_data = on_market_data
                
                # Override detector's signal generation to track statistics
                original_generate_signal = self.detector._generate_signal_file
                
                def tracked_generate_signal(pattern):
                    try:
                        original_generate_signal(pattern)
                        self.total_signals_generated += 1
                        self.last_signal_time = datetime.now()
                        self.patterns_detected += 1
                    except Exception as e:
                        logger.error(f"❌ Error generating signal: {e}")
                        self.system_errors += 1
                
                self.detector._generate_signal_file = tracked_generate_signal
                
                # Start WebSocket client
                logger.info(f"🔗 Connecting to NinjaTrader WebSocket: {self.websocket_host}:{self.websocket_port}")
                
                # Run WebSocket client
                await self.websocket_client.run()
                
            except Exception as e:
                logger.error(f"❌ System startup error: {e}")
                self.system_errors += 1
                raise
        
        def stop_system(self):
            """Stop the system gracefully"""
            self.system_running = False
            logger.info("🛑 Stopping V-Reversal Trading System...")
            
            if self.websocket_client:
                asyncio.create_task(self.websocket_client.disconnect())
    
    # Create system manager
    system_manager = VReversalSystemManager(config, args.websocket_host, args.websocket_port)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received...")
        system_manager.stop_system()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run the system
        asyncio.run(system_manager.start_system())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 System stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ System error: {e}[/red]")
        logger.error(f"System error: {e}")
    finally:
        console.print("[green]✅ V-Reversal Trading System shutdown complete[/green]")

if __name__ == "__main__":
    main() 