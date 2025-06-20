#!/usr/bin/env python3
"""
Enhanced Bidirectional V-Reversal System Launcher

This script launches the enhanced V-reversal detection system that can generate
both BUY and SELL signals based on bidirectional pattern recognition.

Features:
- Downward V-reversals generate BUY signals (price drops then recovers)
- Upward inverted V-reversals generate SELL signals (price rises then declines)
- Based on proven $2300/day model parameters
- Enhanced signal folder and file naming
"""

import asyncio
import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.models.enhanced_vreversal.bidirectional_vreversal_detector import (
    BidirectionalVReversalDetector,
    BidirectionalVReversalConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_vreversal_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_enhanced_config(
    enable_buy: bool = True,
    enable_sell: bool = True,
    drop_threshold: float = 4.0,
    rise_threshold: float = 4.0,
    max_daily_signals: int = 40,
    signal_folder: str = "signals/enhanced"
) -> BidirectionalVReversalConfig:
    """Create enhanced configuration with specified parameters"""
    
    config = BidirectionalVReversalConfig(
        # Pattern thresholds
        drop_threshold=drop_threshold,
        rise_threshold=rise_threshold,
        
        # Risk management (proven parameters)
        stop_loss_pct=0.001,  # 0.1% stop loss
        take_profit_points=3.0,  # 3 points target
        max_hold_time=25,  # 25 minutes
        
        # Signal controls
        min_pattern_confidence=0.75,
        max_daily_signals=max_daily_signals,
        
        # Signal generation
        signal_folder=signal_folder,
        enable_buy_signals=enable_buy,
        enable_sell_signals=enable_sell,
    )
    
    return config

async def run_enhanced_vreversal_system(config: BidirectionalVReversalConfig, 
                                      websocket_host: str = "192.168.1.65",
                                      websocket_port: int = 6789):
    """Run the enhanced V-reversal detection system"""
    try:
        # Create detector
        detector = BidirectionalVReversalDetector(config)
        
        # Display startup information
        print("\n" + "="*80)
        print("🚀 ENHANCED BIDIRECTIONAL V-REVERSAL SYSTEM")
        print("="*80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 WebSocket: {websocket_host}:{websocket_port}")
        print(f"📁 Signals: {config.signal_folder}")
        print(f"📊 BUY Signals: {'✅ ENABLED' if config.enable_buy_signals else '❌ DISABLED'}")
        print(f"📊 SELL Signals: {'✅ ENABLED' if config.enable_sell_signals else '❌ DISABLED'}")
        print(f"📈 Drop Threshold: {config.drop_threshold} points (for BUY)")
        print(f"📉 Rise Threshold: {config.rise_threshold} points (for SELL)")
        print(f"🎯 Stop Loss: {config.stop_loss_pct*100:.1f}% | Take Profit: {config.take_profit_points} points")
        print(f"📋 Max Daily Signals: {config.max_daily_signals}")
        print(f"⏰ Trading Windows: 3-4 AM, 9-11 AM, 1:30-3 PM ET")
        print("="*80)
        print("🔍 Pattern Detection:")
        print("   📈 DOWNWARD V: Price drops → recovers → BUY signal")
        print("   📉 UPWARD V: Price rises → declines → SELL signal")
        print("="*80)
        print("🟢 System running... Press Ctrl+C to stop")
        print()
        
        # Start real-time detection
        await detector.run_realtime_detection(websocket_host, websocket_port)
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
        
        # Display final statistics
        stats = detector.get_statistics()
        print("\n" + "="*60)
        print("📊 FINAL STATISTICS")
        print("="*60)
        print(f"🟢 BUY Signals Generated: {stats['buy_signals']}")
        print(f"🔴 SELL Signals Generated: {stats['sell_signals']}")
        print(f"📊 Total Daily Signals: {stats['daily_signals']}")
        print(f"🔍 Patterns Checked: {stats['total_patterns_checked']}")
        print(f"✅ Patterns Above Threshold: {stats['patterns_above_threshold']}")
        print(f"📋 Processed Unique Patterns: {stats['processed_patterns']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        raise

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Bidirectional V-Reversal System")
    
    # System configuration
    parser.add_argument("--host", default="192.168.1.65", 
                       help="WebSocket host (default: 192.168.1.65)")
    parser.add_argument("--port", type=int, default=6789,
                       help="WebSocket port (default: 6789)")
    
    # Signal type controls
    parser.add_argument("--buy-only", action="store_true",
                       help="Generate only BUY signals (disable SELL)")
    parser.add_argument("--sell-only", action="store_true", 
                       help="Generate only SELL signals (disable BUY)")
    parser.add_argument("--both", action="store_true", default=True,
                       help="Generate both BUY and SELL signals (default)")
    
    # Pattern thresholds
    parser.add_argument("--drop-threshold", type=float, default=4.0,
                       help="Drop threshold for BUY signals (default: 4.0)")
    parser.add_argument("--rise-threshold", type=float, default=4.0,
                       help="Rise threshold for SELL signals (default: 4.0)")
    
    # Signal limits
    parser.add_argument("--max-daily", type=int, default=40,
                       help="Maximum daily signals (default: 40)")
    
    # Output folder
    parser.add_argument("--signal-folder", default="signals/enhanced",
                       help="Signal output folder (default: signals/enhanced)")
    
    args = parser.parse_args()
    
    # Determine signal types
    if args.buy_only:
        enable_buy, enable_sell = True, False
        print("🟢 Mode: BUY SIGNALS ONLY")
    elif args.sell_only:
        enable_buy, enable_sell = False, True
        print("🔴 Mode: SELL SIGNALS ONLY")
    else:
        enable_buy, enable_sell = True, True
        print("🟢🔴 Mode: BIDIRECTIONAL (BUY + SELL)")
    
    # Create configuration
    config = create_enhanced_config(
        enable_buy=enable_buy,
        enable_sell=enable_sell,
        drop_threshold=args.drop_threshold,
        rise_threshold=args.rise_threshold,
        max_daily_signals=args.max_daily,
        signal_folder=args.signal_folder
    )
    
    # Create logs and signals directories
    Path("logs").mkdir(exist_ok=True)
    Path(args.signal_folder).mkdir(parents=True, exist_ok=True)
    
    # Run system
    asyncio.run(run_enhanced_vreversal_system(config, args.host, args.port))

if __name__ == "__main__":
    main() 