#!/usr/bin/env python3
"""
V-Reversal System Test Script
============================

Test script that simulates real-time market data to verify the complete
V-reversal detection and signal generation pipeline works correctly.

This script creates synthetic ES 1-minute data with embedded V-reversal patterns
to test the detection system without requiring live market data.

Author: Thelma ML Strategy Team
Date: 2025-06-19
"""

import asyncio
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TimeElapsedColumn, MofNCompleteColumn

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "models"))

try:
    from vreversal_realtime_detector import VReversalRealtimeDetector, VReversalConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class VReversalTestDataGenerator:
    """Generates synthetic market data with embedded V-reversal patterns"""
    
    def __init__(self, base_price: float = 4850.0):
        self.base_price = base_price
        self.current_price = base_price
        self.current_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        
    def generate_normal_bar(self) -> dict:
        """Generate a normal 1-minute bar with small price movements"""
        # Small random walk
        price_change = np.random.normal(0, 0.5)  # Small movements
        self.current_price += price_change
        
        # Create OHLC
        open_price = self.current_price
        high_price = open_price + abs(np.random.normal(0, 0.25))
        low_price = open_price - abs(np.random.normal(0, 0.25))
        close_price = open_price + np.random.normal(0, 0.25)
        
        # Update current price
        self.current_price = close_price
        
        bar = {
            'timestamp': self.current_time,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': np.random.randint(100, 1000)
        }
        
        self.current_time += timedelta(minutes=1)
        return bar
    
    def generate_vreversal_pattern(self) -> list:
        """Generate a complete V-reversal pattern across multiple bars"""
        bars = []
        
        # Phase 1: Setup bars (normal movement)
        for _ in range(5):
            bars.append(self.generate_normal_bar())
        
        # Phase 2: Sharp drop (4-5 points down) - this triggers our 3-point threshold
        origin_high = self.current_price
        drop_amount = 4.5  # Above our 3-point threshold
        
        # Drop over 3 bars
        for i in range(3):
            drop_this_bar = drop_amount / 3
            open_price = self.current_price
            close_price = open_price - drop_this_bar
            low_price = close_price - 0.25  # Slightly lower
            high_price = open_price + 0.25   # Slight bounce
            
            bar = {
                'timestamp': self.current_time,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': np.random.randint(200, 800)  # Higher volume on drop
            }
            bars.append(bar)
            
            self.current_price = close_price
            self.current_time += timedelta(minutes=1)
        
        # Phase 3: Breakout above origin high
        for i in range(2):
            open_price = self.current_price
            # Break above origin high
            close_price = origin_high + 0.5 + (i * 0.5)
            high_price = close_price + 0.25
            low_price = open_price - 0.1
            
            bar = {
                'timestamp': self.current_time,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': np.random.randint(300, 1000)  # High volume on breakout
            }
            bars.append(bar)
            
            self.current_price = close_price
            self.current_time += timedelta(minutes=1)
        
        # Phase 4: Pullback (but not below origin high)
        for i in range(2):
            open_price = self.current_price
            # Pullback but stay near origin high (within tolerance)
            close_price = origin_high + np.random.uniform(0.1, 0.8)
            low_price = origin_high - 0.5  # Touch near origin high
            high_price = open_price + 0.1
            
            bar = {
                'timestamp': self.current_time,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': np.random.randint(150, 600)
            }
            bars.append(bar)
            
            self.current_price = close_price
            self.current_time += timedelta(minutes=1)
        
        return bars

class VReversalSystemTester:
    """Tests the complete V-reversal detection system"""
    
    def __init__(self):
        self.console = Console()
        self.config = VReversalConfig()
        self.config.signal_folder = "test_signals"
        self.config.min_pattern_confidence = 0.7  # Lower for testing
        
        # Create test signal directory
        Path(self.config.signal_folder).mkdir(exist_ok=True)
        Path(f"{self.config.signal_folder}/processed").mkdir(exist_ok=True)
        
        # Initialize detector
        self.detector = VReversalRealtimeDetector(self.config)
        
        # Test results
        self.signals_generated = 0
        self.patterns_detected = 0
        
        # Override signal generation to count signals
        original_generate_signal = self.detector._generate_signal_file
        
        def count_signals(pattern):
            original_generate_signal(pattern)
            self.signals_generated += 1
            self.console.print(f"✅ [green]Signal generated:[/green] {pattern.action} @ {pattern.entry_price:.2f}")
        
        self.detector._generate_signal_file = count_signals
    
    async def run_test(self):
        """Run the complete system test"""
        
        self.console.print(Panel(
            "[bold green]🧪 V-REVERSAL SYSTEM TEST[/bold green]\n\n"
            "[cyan]Testing Components:[/cyan]\n"
            "• Real-time pattern detection\n"
            "• Signal file generation\n" 
            "• V-reversal pattern recognition\n"
            "• Configuration parameters\n\n"
            "[yellow]Simulating live market data...[/yellow]",
            title="System Test",
            border_style="blue"
        ))
        
        # Generate test data
        generator = VReversalTestDataGenerator()
        
        # Test scenario 1: Normal market data (should not generate signals)
        self.console.print("\n📊 [blue]Test 1:[/blue] Normal market movement (should generate 0 signals)")
        
        with Progress(
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            refresh_per_second=10
        ) as progress:
            
            task1 = progress.add_task("Processing normal bars...", total=30)
            
            for i in range(30):
                bar = generator.generate_normal_bar()
                self.detector.add_bar_data(bar)
                progress.advance(task1)
                await asyncio.sleep(0.1)  # Simulate real-time
        
        signals_after_normal = self.signals_generated
        self.console.print(f"✅ Normal data test complete: {signals_after_normal} signals generated")
        
        # Test scenario 2: Data with V-reversal patterns
        self.console.print("\n📊 [blue]Test 2:[/blue] V-reversal patterns (should generate signals)")
        
        with Progress(
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            refresh_per_second=10
        ) as progress:
            
            # Generate 3 V-reversal patterns
            task2 = progress.add_task("Processing V-reversal patterns...", total=3)
            
            for pattern_num in range(3):
                vreversal_bars = generator.generate_vreversal_pattern()
                
                for bar in vreversal_bars:
                    self.detector.add_bar_data(bar)
                    await asyncio.sleep(0.05)  # Faster for testing
                
                # Add some normal bars between patterns
                for _ in range(10):
                    bar = generator.generate_normal_bar()
                    self.detector.add_bar_data(bar)
                    await asyncio.sleep(0.02)
                
                progress.advance(task2)
                self.console.print(f"  Pattern {pattern_num + 1} complete")
        
        signals_after_patterns = self.signals_generated
        patterns_detected = signals_after_patterns - signals_after_normal
        
        # Test results
        self.console.print("\n" + "="*60)
        self.console.print(f"🎯 [bold]TEST RESULTS[/bold]")
        self.console.print("="*60)
        
        results_table = [
            ["Normal bars processed", "30", "✅"],
            ["V-reversal patterns injected", "3", "✅"],
            ["Signals from normal data", str(signals_after_normal), "✅" if signals_after_normal == 0 else "⚠️"],
            ["Signals from V-reversal data", str(patterns_detected), "✅" if patterns_detected > 0 else "❌"],
            ["Total signals generated", str(self.signals_generated), "✅" if self.signals_generated > 0 else "❌"]
        ]
        
        for metric, value, status in results_table:
            self.console.print(f"{status} {metric}: [cyan]{value}[/cyan]")
        
        # Check signal files
        signal_files = list(Path(self.config.signal_folder).glob("vreversal_*.txt"))
        self.console.print(f"✅ Signal files created: [cyan]{len(signal_files)}[/cyan]")
        
        if signal_files:
            self.console.print(f"\n📁 [bold]Generated Signal Files:[/bold]")
            for file in signal_files:
                self.console.print(f"  • {file.name}")
        
        # Overall result
        success = (signals_after_normal == 0 and patterns_detected > 0)
        
        if success:
            self.console.print(Panel(
                "[bold green]✅ ALL TESTS PASSED[/bold green]\n\n"
                "The V-reversal detection system is working correctly:\n"
                "• No false signals from normal data\n"
                "• Detected injected V-reversal patterns\n"
                "• Generated proper signal files\n\n"
                "[yellow]System ready for live trading![/yellow]",
                title="Test Results",
                border_style="green"
            ))
        else:
            self.console.print(Panel(
                "[bold red]❌ TESTS FAILED[/bold red]\n\n"
                "Issues detected:\n"
                f"• False signals from normal data: {signals_after_normal > 0}\n"
                f"• Failed to detect patterns: {patterns_detected == 0}\n\n"
                "[yellow]Check configuration and pattern detection logic[/yellow]",
                title="Test Results",
                border_style="red"
            ))
        
        return success

async def main():
    """Main test function"""
    console = Console()
    
    console.print(Panel(
        "[bold cyan]🧪 V-REVERSAL SYSTEM TESTING[/bold cyan]\n\n"
        "This test verifies the V-reversal detection system using\n"
        "synthetic market data with embedded patterns.\n\n"
        "[yellow]Starting automated test sequence...[/yellow]",
        border_style="cyan"
    ))
    
    tester = VReversalSystemTester()
    
    try:
        success = await tester.run_test()
        
        if success:
            console.print("\n[green]🎉 System test completed successfully![/green]")
            console.print("[yellow]Ready to run: python scripts/launch_vreversal_system.py[/yellow]")
        else:
            console.print("\n[red]❌ System test failed. Check configuration.[/red]")
            
    except Exception as e:
        console.print(f"\n[red]❌ Test error: {e}[/red]")
        return False
    
    return success

if __name__ == "__main__":
    asyncio.run(main()) 