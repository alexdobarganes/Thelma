#!/usr/bin/env python3
"""
Quick Demo: Enhanced Bidirectional V-Reversal System

Este script demuestra las capacidades del sistema mejorado que puede generar
señales tanto de BUY como de SELL basadas en patrones V-reversal bidireccionales.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.models.enhanced_vreversal.bidirectional_vreversal_detector import (
    BidirectionalVReversalDetector,
    BidirectionalVReversalConfig
)

def create_sample_data():
    """Create sample market data with both BUY and SELL pattern opportunities"""
    
    eastern_tz = pytz.timezone('US/Eastern')
    start_time = datetime(2025, 6, 20, 10, 0, 0)  # 10:00 AM ET (in trading window)
    
    # Create 100 1-minute bars with patterns
    timestamps = [start_time + timedelta(minutes=i) for i in range(100)]
    
    # Base price around 6100
    base_price = 6100.0
    prices = []
    
    for i, timestamp in enumerate(timestamps):
        # Create price movements with embedded patterns
        
        if 10 <= i <= 25:  # Downward V-reversal pattern (BUY signal)
            if i <= 15:
                # Drop phase: 6100 -> 6095 (5 point drop)
                price = base_price - (i - 10) * 1.0
            elif i <= 20:
                # Recovery phase: 6095 -> 6102 (breakout)
                price = 6095.0 + (i - 15) * 1.4
            else:
                # Pullback phase: 6102 -> 6100 (entry opportunity)
                price = 6102.0 - (i - 20) * 0.4
        
        elif 40 <= i <= 55:  # Upward V-reversal pattern (SELL signal)
            if i <= 45:
                # Rise phase: 6100 -> 6104.5 (4.5 point rise)
                price = base_price + (i - 40) * 0.9
            elif i <= 50:
                # Decline phase: 6104.5 -> 6098 (breakdown)
                price = 6104.5 - (i - 45) * 1.3
            else:
                # Bounce phase: 6098 -> 6100 (entry opportunity)
                price = 6098.0 + (i - 50) * 0.4
        
        else:
            # Normal price movement
            noise = np.random.normal(0, 0.3)
            price = base_price + noise
        
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from close price
        high = close + np.random.uniform(0.1, 0.8)
        low = close - np.random.uniform(0.1, 0.8)
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(100, 1000)
        
        # Ensure OHLC logic is correct
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'timestamp': eastern_tz.localize(timestamp),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return data

def run_enhanced_demo():
    """Run the enhanced bidirectional V-reversal detection demo"""
    
    print("🚀 Enhanced Bidirectional V-Reversal Detection Demo")
    print("="*60)
    
    # Create configuration for both BUY and SELL signals
    config = BidirectionalVReversalConfig(
        drop_threshold=4.0,     # 4 points for BUY signals
        rise_threshold=4.0,     # 4 points for SELL signals
        min_pattern_confidence=0.70,  # Lower for demo
        signal_folder="signals/enhanced",
        enable_buy_signals=True,
        enable_sell_signals=True,
        max_daily_signals=20
    )
    
    print(f"📊 Configuration:")
    print(f"   🟢 BUY Signals: {'ENABLED' if config.enable_buy_signals else 'DISABLED'}")
    print(f"   🔴 SELL Signals: {'ENABLED' if config.enable_sell_signals else 'DISABLED'}")
    print(f"   📈 Drop Threshold: {config.drop_threshold} points")
    print(f"   📉 Rise Threshold: {config.rise_threshold} points")
    print(f"   🎯 Min Confidence: {config.min_pattern_confidence:.1%}")
    print()
    
    # Create detector
    detector = BidirectionalVReversalDetector(config)
    
    # Generate sample data with embedded patterns
    print("📊 Generating sample market data with embedded patterns...")
    sample_data = create_sample_data()
    print(f"   ✅ Created {len(sample_data)} bars of data")
    print(f"   📍 Embedded downward V-reversal at bars 10-25 (for BUY)")
    print(f"   📍 Embedded upward V-reversal at bars 40-55 (for SELL)")
    print()
    
    # Process data bar by bar to simulate real-time
    print("🔍 Processing data for pattern detection...")
    detected_patterns = []
    
    for i, bar_data in enumerate(sample_data):
        detector.add_bar_data(bar_data)
        
        # Check if we have enough data to detect patterns
        if i >= 30:  # Need minimum bars
            print(f"   Bar {i+1:3d}: {bar_data['timestamp'].strftime('%H:%M')} | "
                  f"OHLC: {bar_data['open']:6.2f} {bar_data['high']:6.2f} "
                  f"{bar_data['low']:6.2f} {bar_data['close']:6.2f}")
    
    print()
    
    # Display final statistics
    stats = detector.get_statistics()
    print("📊 DETECTION RESULTS")
    print("="*40)
    print(f"🟢 BUY Signals Generated: {stats['buy_signals']}")
    print(f"🔴 SELL Signals Generated: {stats['sell_signals']}")
    print(f"📊 Total Daily Signals: {stats['daily_signals']}")
    print(f"🔍 Patterns Checked: {stats['total_patterns_checked']}")
    print(f"✅ Patterns Above Threshold: {stats['patterns_above_threshold']}")
    print(f"📋 Unique Patterns Processed: {stats['processed_patterns']}")
    print()
    
    # Check generated signal files
    signal_folder = Path(config.signal_folder)
    if signal_folder.exists():
        signal_files = list(signal_folder.glob("enhanced_vreversal_*.txt"))
        
        if signal_files:
            print(f"📁 GENERATED SIGNAL FILES ({len(signal_files)} files):")
            print("-" * 50)
            
            for signal_file in sorted(signal_files):
                print(f"   📄 {signal_file.name}")
                
                # Read and display signal content
                with open(signal_file, 'r') as f:
                    lines = f.readlines()
                
                # Extract key information
                action = ""
                entry_price = ""
                pattern_type = ""
                confidence = ""
                
                for line in lines:
                    if line.startswith("ACTION="):
                        action = line.split("=")[1].strip()
                    elif line.startswith("ENTRY_PRICE="):
                        entry_price = line.split("=")[1].strip()
                    elif line.startswith("PATTERN_SUBTYPE="):
                        pattern_type = line.split("=")[1].strip()
                    elif line.startswith("CONFIDENCE="):
                        confidence = float(line.split("=")[1].strip())
                
                if action and entry_price:
                    action_emoji = "📈" if action == "BUY" else "📉"
                    print(f"      {action_emoji} {action} @ {entry_price} | {pattern_type} | {confidence:.1%}")
            
            print()
        else:
            print("📁 No signal files generated (patterns may not have met confidence threshold)")
            print()
    
    print("✨ Demo completed successfully!")
    print()
    print("🔧 Next Steps:")
    print("   1. Review generated signals in signals/enhanced/")
    print("   2. Compile EnhancedVReversalAutoTrader.cs in NinjaTrader")
    print("   3. Run: python scripts/launch_enhanced_vreversal_system.py")
    print("   4. Configure NinjaTrader to monitor signals/enhanced/ folder")

def main():
    """Main demo function"""
    try:
        # Ensure directories exist
        Path("signals/enhanced").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Run demo
        run_enhanced_demo()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 