# Copy of launch_vreversal_system.py with LOWER threshold for more signals
# This version uses 3.0 points instead of 4.0 for more sensitivity

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

def main():
    """Main function with MORE SENSITIVE parameters"""
    parser = argparse.ArgumentParser(description="V-Reversal Trading System - More Sensitive")
    parser.add_argument("--websocket-host", default="192.168.1.65", help="NinjaTrader WebSocket host")
    parser.add_argument("--websocket-port", type=int, default=6789, help="NinjaTrader WebSocket port")
    parser.add_argument("--signal-folder", default="signals", help="Signal output folder")
    parser.add_argument("--max-daily-signals", type=int, default=30, help="Maximum signals per day (increased)")
    
    args = parser.parse_args()
    
    # Create configuration with MORE SENSITIVE parameters
    config = VReversalConfig()
    
    # MORE SENSITIVE PARAMETERS for more signals
    config.drop_threshold = 3.0           # LOWERED from 4.0 to 3.0 (more sensitive)
    config.drop_window = 15               # 15 minutes for drop completion
    config.breakout_window = 30           # 30 minutes for breakout
    config.pullback_window = 15           # 15 minutes for pullback
    config.continuation_window = 20       # 20 minutes for continuation
    config.pullback_tolerance = 1.5       # INCREASED tolerance (1.5 points)
    config.stop_loss_pct = 0.001          # 0.1% - same as production
    config.max_hold_time = 25             # 25 minutes max hold
    config.min_pattern_confidence = 0.60   # LOWERED from 0.75 to 0.60 (more signals)
    
    # Trading session configuration (EXTENDED hours)
    config.signal_folder = args.signal_folder
    config.max_daily_signals = args.max_daily_signals
    config.session_start = "09:30"        # Market open ET
    config.session_end = "16:00"          # Market close ET
    
    # Display startup banner
    console = Console()
    console.print(Panel(
        "[bold green]🎯 MORE SENSITIVE V-REVERSAL SYSTEM[/bold green]\n\n"
        f"[cyan]Strategy:[/cyan] Drop_3 + 0.1% Stop Loss (MORE SENSITIVE)\n"
        f"[cyan]WebSocket:[/cyan] {args.websocket_host}:{args.websocket_port}\n"
        f"[cyan]Signals:[/cyan] {config.signal_folder}/\n"
        f"[cyan]Hours:[/cyan] 9:30 AM - 4:00 PM ET (FULL SESSION)\n"
        f"[cyan]Max Daily:[/cyan] {config.max_daily_signals} signals\n\n"
        "[yellow]📊 MORE SENSITIVE SETTINGS:[/yellow]\n"
        "[yellow]• Drop Threshold: 3.0 points (vs 4.0 original)[/yellow]\n"
        "[yellow]• Confidence: 60% (vs 75% original)[/yellow]\n"
        "[yellow]• Pullback Tolerance: 1.5 points (vs 1.0)[/yellow]\n"
        "[yellow]• Should generate MORE signals![/yellow]\n\n"
        "[yellow]🚀 Starting sensitive pattern detection...[/yellow]\n"
        "[yellow]🎮 Press Ctrl+C to stop[/yellow]",
        title="MORE SENSITIVE LAUNCH",
        border_style="green"
    ))
    
    # Use standard system manager but with updated config
    from scripts.launch_vreversal_system import VReversalSystemManager
    
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