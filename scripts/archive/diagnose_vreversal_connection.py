#!/usr/bin/env python3
"""
V-Reversal System Diagnostic Tool
================================

Comprehensive diagnostic tool to troubleshoot why the V-reversal system
is not generating signals after 985+ minutes of running.

Tests:
1. WebSocket connection to NinjaTrader
2. Data reception and parsing
3. Pattern detection logic
4. Trading window validation
5. Signal generation process

Author: Thelma ML Strategy Team
Date: 2025-06-19
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
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
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VReversalDiagnostics:
    """Comprehensive diagnostics for V-reversal system"""
    
    def __init__(self, websocket_host="192.168.1.65", websocket_port=6789):
        self.console = Console()
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.config = VReversalConfig()
        
        # Test results
        self.test_results = {
            'websocket_connection': False,
            'data_reception': False,
            'pattern_detection': False,
            'trading_windows': False,
            'signal_generation': False
        }
        
        # Statistics
        self.bars_received = 0
        self.connection_attempts = 0
        self.last_bar_time = None
        
    def display_header(self):
        """Display diagnostic header"""
        self.console.print(Panel(
            "[bold red]🔧 V-REVERSAL SYSTEM DIAGNOSTICS[/bold red]\n\n"
            f"[cyan]Target:[/cyan] Diagnose 985+ minutes without signals\n"
            f"[cyan]WebSocket:[/cyan] {self.websocket_host}:{self.websocket_port}\n"
            f"[cyan]Model:[/cyan] $2300/day production parameters\n\n"
            "[yellow]🔍 Running comprehensive system tests...[/yellow]",
            title="SYSTEM DIAGNOSTIC MODE",
            border_style="red"
        ))
    
    async def test_websocket_connection(self):
        """Test 1: WebSocket connection to NinjaTrader"""
        self.console.print("\n[bold]🔗 TEST 1: WebSocket Connection[/bold]")
        
        try:
            # Create WebSocket client
            client = WebSocketClient(
                host=self.websocket_host,
                port=self.websocket_port,
                csv_file="data/diagnostic_data.csv"
            )
            
            connection_successful = False
            
            async def on_connected():
                nonlocal connection_successful
                connection_successful = True
                self.console.print("✅ WebSocket connection established")
                self.test_results['websocket_connection'] = True
            
            async def on_disconnected():
                self.console.print("⚠️ WebSocket disconnected")
            
            async def on_market_data(data: MarketData):
                self.bars_received += 1
                self.last_bar_time = datetime.now()
                self.test_results['data_reception'] = True
                
                if self.bars_received <= 3:
                    self.console.print(f"📊 Bar #{self.bars_received}: {data.timestamp} - Close: {data.close_price}")
            
            # Set callbacks
            client.on_connected = on_connected
            client.on_disconnected = on_disconnected  
            client.on_market_data = on_market_data
            
            self.console.print(f"🔌 Attempting connection to {self.websocket_host}:{self.websocket_port}...")
            
            # Try connection for 10 seconds
            try:
                await asyncio.wait_for(client.connect(), timeout=10.0)
                await asyncio.sleep(5)  # Wait for data
                await client.disconnect()
            except asyncio.TimeoutError:
                self.console.print("❌ Connection timeout after 10 seconds")
                return False
                
            return connection_successful
            
        except Exception as e:
            self.console.print(f"❌ Connection error: {e}")
            return False
    
    def test_trading_windows(self):
        """Test 2: Trading window validation"""
        self.console.print("\n[bold]🕐 TEST 2: Trading Windows[/bold]")
        
        # Get current Eastern Time
        eastern_tz = timezone(timedelta(hours=-5))  # EST 
        current_et = datetime.now(eastern_tz)
        current_hour = current_et.hour
        current_minute = current_et.minute
        
        self.console.print(f"🕒 Current time ET: {current_et.strftime('%H:%M:%S')}")
        
        # Check production windows
        in_window_1 = (3 <= current_hour < 4)  # 3-4 AM
        in_window_2 = (9 <= current_hour < 11)  # 9-11 AM  
        in_window_3 = (current_hour == 13 and current_minute >= 30) or (14 <= current_hour < 15)  # 1:30-3 PM
        
        windows_table = Table(title="Trading Windows Status", box=box.SIMPLE)
        windows_table.add_column("Window", style="cyan")
        windows_table.add_column("Time Range", style="white")
        windows_table.add_column("Status", style="white")
        
        windows_table.add_row("Window 1", "3:00-4:00 AM ET", "✅ ACTIVE" if in_window_1 else "🔴 CLOSED")
        windows_table.add_row("Window 2", "9:00-11:00 AM ET", "✅ ACTIVE" if in_window_2 else "🔴 CLOSED")
        windows_table.add_row("Window 3", "1:30-3:00 PM ET", "✅ ACTIVE" if in_window_3 else "🔴 CLOSED")
        
        self.console.print(windows_table)
        
        is_trading_time = in_window_1 or in_window_2 or in_window_3
        
        if is_trading_time:
            self.console.print("✅ Currently in active trading window")
            self.test_results['trading_windows'] = True
        else:
            self.console.print("❌ Currently OUTSIDE trading windows")
            self.console.print("💡 This explains why no signals are being generated!")
            
            # Calculate next window
            next_windows = []
            if current_hour < 3:
                next_windows.append("3:00 AM")
            if current_hour < 9:
                next_windows.append("9:00 AM")
            if current_hour < 13 or (current_hour == 13 and current_minute < 30):
                next_windows.append("1:30 PM")
            
            if next_windows:
                self.console.print(f"⏰ Next trading window: {next_windows[0]} ET")
        
        return is_trading_time
    
    def test_pattern_detection_logic(self):
        """Test 3: Pattern detection with sample data"""
        self.console.print("\n[bold]🎯 TEST 3: Pattern Detection Logic[/bold]")
        
        try:
            detector = VReversalRealtimeDetector(self.config)
            
            # Create sample V-reversal pattern data
            base_price = 4950.0
            sample_bars = []
            
            # Create a clear V-reversal pattern
            for i in range(60):
                if i < 15:  # Pre-drop phase
                    price = base_price + (i * 0.1)
                    bar = {
                        'timestamp': datetime.now() - timedelta(minutes=60-i),
                        'open': price,
                        'high': price + 0.5,
                        'low': price - 0.5,
                        'close': price,
                        'volume': 1000
                    }
                elif i < 25:  # Drop phase (drop 5 points)
                    drop_progress = (i - 15) / 10
                    price = base_price + 15 - (drop_progress * 5.0)
                    bar = {
                        'timestamp': datetime.now() - timedelta(minutes=60-i),
                        'open': price + 0.2,
                        'high': price + 0.5,
                        'low': price - 1.0,
                        'close': price,
                        'volume': 1500
                    }
                elif i < 35:  # Recovery/breakout phase
                    recovery_progress = (i - 25) / 10
                    price = (base_price + 10) + (recovery_progress * 6.0)
                    bar = {
                        'timestamp': datetime.now() - timedelta(minutes=60-i),
                        'open': price - 0.2,
                        'high': price + 1.0,
                        'low': price - 0.5,
                        'close': price,
                        'volume': 1200
                    }
                else:  # Pullback and continuation
                    price = base_price + 15.5 + ((i-35) * 0.1)
                    bar = {
                        'timestamp': datetime.now() - timedelta(minutes=60-i),
                        'open': price,
                        'high': price + 0.5,
                        'low': price - 0.3,
                        'close': price,
                        'volume': 1000
                    }
                
                sample_bars.append(bar)
            
            # Feed sample data to detector
            patterns_detected = 0
            for bar in sample_bars:
                detector.add_bar_data(bar)
                # Check if any patterns were detected by looking at signal folder
                
            # Check signal folder for generated files
            signal_files = list(Path(self.config.signal_folder).glob("*.txt"))
            recent_signals = [f for f in signal_files if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).seconds < 300]
            
            if recent_signals:
                self.console.print(f"✅ Pattern detection working - {len(recent_signals)} signals generated")
                self.test_results['pattern_detection'] = True
                self.test_results['signal_generation'] = True
            else:
                self.console.print("⚠️ No patterns detected with sample data")
                self.console.print("💡 Pattern detection may be too strict or sample data insufficient")
                
        except Exception as e:
            self.console.print(f"❌ Pattern detection error: {e}")
            return False
    
    def generate_diagnostic_report(self):
        """Generate final diagnostic report"""
        self.console.print("\n[bold]📋 DIAGNOSTIC REPORT[/bold]")
        
        report_table = Table(title="System Health Check", box=box.ROUNDED)
        report_table.add_column("Test", style="cyan")
        report_table.add_column("Status", style="white")
        report_table.add_column("Details", style="white")
        
        # WebSocket Connection
        status = "✅ PASS" if self.test_results['websocket_connection'] else "❌ FAIL"
        details = f"Received {self.bars_received} bars" if self.bars_received > 0 else "No data received"
        report_table.add_row("WebSocket Connection", status, details)
        
        # Data Reception
        status = "✅ PASS" if self.test_results['data_reception'] else "❌ FAIL"
        details = f"Last bar: {self.last_bar_time.strftime('%H:%M:%S') if self.last_bar_time else 'None'}"
        report_table.add_row("Data Reception", status, details)
        
        # Trading Windows
        status = "✅ PASS" if self.test_results['trading_windows'] else "❌ FAIL"
        eastern_tz = timezone(timedelta(hours=-5))
        current_et = datetime.now(eastern_tz)
        details = f"Current: {current_et.strftime('%H:%M')} ET"
        report_table.add_row("Trading Windows", status, details)
        
        # Pattern Detection
        status = "✅ PASS" if self.test_results['pattern_detection'] else "⚠️ NEEDS CHECK"
        details = "Logic functioning" if self.test_results['pattern_detection'] else "May need adjustment"
        report_table.add_row("Pattern Detection", status, details)
        
        # Signal Generation
        status = "✅ PASS" if self.test_results['signal_generation'] else "❌ FAIL"
        details = "Files generated" if self.test_results['signal_generation'] else "No signals created"
        report_table.add_row("Signal Generation", status, details)
        
        self.console.print(report_table)
        
        # Generate recommendations
        self.console.print("\n[bold yellow]💡 RECOMMENDATIONS:[/bold yellow]")
        
        if not self.test_results['websocket_connection']:
            self.console.print("🔧 Fix WebSocket connection:")
            self.console.print("   • Check NinjaTrader is running")
            self.console.print("   • Verify TickWebSocketPublisher indicator is active")
            self.console.print("   • Check firewall/network settings")
            self.console.print(f"   • Confirm host/port: {self.websocket_host}:{self.websocket_port}")
        
        if not self.test_results['trading_windows']:
            self.console.print("⏰ Trading window issue:")
            self.console.print("   • System only trades 3-4 AM, 9-11 AM, 1:30-3 PM ET")
            self.console.print("   • 985 minutes = 16+ hours outside windows is normal")
            self.console.print("   • Consider expanding windows if needed")
        
        if not self.test_results['pattern_detection']:
            self.console.print("🎯 Pattern detection tuning needed:")
            self.console.print("   • Drop threshold may be too high (currently 4.0 points)")
            self.console.print("   • Consider lowering to 3.0 or 3.5 points")
            self.console.print("   • Market conditions may not be producing patterns")
    
    async def run_full_diagnostics(self):
        """Run complete diagnostic suite"""
        self.display_header()
        
        # Test 1: WebSocket Connection
        await self.test_websocket_connection()
        
        # Test 2: Trading Windows
        self.test_trading_windows()
        
        # Test 3: Pattern Detection
        self.test_pattern_detection_logic()
        
        # Generate report
        self.generate_diagnostic_report()
        
        return self.test_results

async def main():
    """Main diagnostic function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V-Reversal System Diagnostics")
    parser.add_argument("--websocket-host", default="192.168.1.65", help="WebSocket host")
    parser.add_argument("--websocket-port", type=int, default=6789, help="WebSocket port")
    
    args = parser.parse_args()
    
    diagnostics = VReversalDiagnostics(args.websocket_host, args.websocket_port)
    results = await diagnostics.run_full_diagnostics()
    
    # Exit with appropriate code
    if all(results.values()):
        print("\n✅ All diagnostics passed!")
        sys.exit(0)
    else:
        print("\n❌ Some diagnostics failed - see recommendations above")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 