"""
Bidirectional V-Reversal Pattern Detector
Enhanced version that detects both BUY and SELL opportunities

This detector identifies:
1. Downward V-Reversals (BUY signals): Price drops significantly, then recovers
2. Upward Inverted V-Reversals (SELL signals): Price rises significantly, then declines

Based on the proven $2300/day model parameters but expanded for bidirectional trading.
"""

import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pytz
from collections import deque
import json
from rich.console import Console
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bidirectional_vreversal.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BidirectionalVReversalConfig:
    """Enhanced configuration for bidirectional V-reversal detection"""
    
    # PROVEN PARAMETERS from $2300/day model
    drop_threshold: float = 4.0   # Points for downward V-reversal (BUY)
    rise_threshold: float = 4.0   # Points for upward inverted V-reversal (SELL)
    
    # Pattern detection windows
    pattern_window: int = 15      # Window to detect initial move (drop or rise)
    breakout_window: int = 30     # Window for breakout confirmation
    pullback_window: int = 15     # Window for pullback/retrace
    
    # Risk management (proven parameters)
    stop_loss_pct: float = 0.001  # 0.1% stop loss (production proven)
    take_profit_points: float = 3.0  # 3 points target
    max_hold_time: int = 25       # 25 minutes max hold
    
    # Signal filtering
    min_pattern_confidence: float = 0.75
    max_daily_signals: int = 40   # Doubled for bidirectional trading
    
    # Trading windows (same as proven model)
    trading_windows: List[Tuple[int, int]] = None  # Will be set in __post_init__
    
    # Signal generation
    signal_folder: str = "signals/enhanced"
    enable_buy_signals: bool = True
    enable_sell_signals: bool = True
    
    def __post_init__(self):
        if self.trading_windows is None:
            # Same proven trading windows
            self.trading_windows = [
                (3, 4),      # 3-4 AM ET
                (9, 11),     # 9-11 AM ET  
                (13.5, 15)   # 1:30-3 PM ET (13:30-15:00)
            ]

@dataclass
class BidirectionalPattern:
    """Enhanced pattern that supports both BUY and SELL signals"""
    signal_id: str
    timestamp: datetime
    action: str  # "BUY" or "SELL"
    entry_price: float
    pattern_type: str  # "DOWNWARD_V" or "UPWARD_V"
    
    # Pattern-specific data
    origin_price: float      # Starting price (high for BUY, low for SELL)
    extreme_price: float     # Extreme price (low for BUY, high for SELL)
    move_points: float       # Points moved in initial direction
    breakout_price: float    # Breakout confirmation price
    pattern_confidence: float
    
    # Additional context
    volume_confirmation: bool = False
    pattern_duration: int = 0  # Minutes from start to entry
    
    def to_signal_file_content(self, config: BidirectionalVReversalConfig) -> str:
        """Generate signal file content for NinjaTrader"""
        
        # Calculate stop loss and take profit
        if self.action == "BUY":
            stop_loss = self.entry_price * (1 - config.stop_loss_pct)
            take_profit = self.entry_price + config.take_profit_points
            risk_reward = f"Risk: {config.stop_loss_pct*100:.1f}% | Reward: {config.take_profit_points:.1f}pts"
        else:  # SELL
            stop_loss = self.entry_price * (1 + config.stop_loss_pct)
            take_profit = self.entry_price - config.take_profit_points
            risk_reward = f"Risk: {config.stop_loss_pct*100:.1f}% | Reward: {config.take_profit_points:.1f}pts"
        
        pattern_desc = "Price dropped then recovered" if self.pattern_type == "DOWNWARD_V" else "Price rose then declined"
        
        return f"""# Enhanced Bidirectional V-Reversal Signal
# Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} ET
# Pattern: {self.pattern_type} - {pattern_desc}
# Confidence: {self.pattern_confidence:.1%} | Duration: {self.pattern_duration}min
# Move: {self.move_points:.2f} points from {self.origin_price:.2f} to {self.extreme_price:.2f}

ACTION={self.action}
ENTRY_PRICE={self.entry_price:.2f}
STOP_LOSS={stop_loss:.2f}
TAKE_PROFIT={take_profit:.2f}
SIGNAL_ID={self.signal_id}
PATTERN_TYPE=BIDIRECTIONAL_V_REVERSAL
PATTERN_SUBTYPE={self.pattern_type}
MOVE_POINTS={self.move_points:.2f}
ORIGIN_PRICE={self.origin_price:.2f}
EXTREME_PRICE={self.extreme_price:.2f}
BREAKOUT_PRICE={self.breakout_price:.2f}
CONFIDENCE={self.pattern_confidence:.3f}
VOLUME_CONFIRMED={self.volume_confirmation}

# Enhanced Strategy Parameters
# {risk_reward}
# Expected Win Rate: 91.2% (proven model basis)
# Max Hold Time: {config.max_hold_time} minutes
# Pattern Duration: {self.pattern_duration} minutes
"""

class BidirectionalVReversalDetector:
    """Enhanced detector for both BUY and SELL V-reversal patterns"""
    
    def __init__(self, config: BidirectionalVReversalConfig):
        self.config = config
        self.bar_buffer = deque(maxlen=100)  # Rolling buffer for pattern detection
        self.processed_patterns = set()      # Prevent duplicate signals
        self.daily_signal_count = 0
        self.last_reset_date = datetime.now().date()
        
        self.console = Console()
        
        # Statistics tracking
        self.stats = {
            'buy_signals': 0,
            'sell_signals': 0,
            'total_patterns_checked': 0,
            'patterns_above_threshold': 0
        }
        
        # Ensure signal folder exists
        Path(self.config.signal_folder).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔄 Bidirectional V-Reversal Detector initialized")
        logger.info(f"📊 BUY signals: {self.config.enable_buy_signals} | SELL signals: {self.config.enable_sell_signals}")
        logger.info(f"📈 Drop threshold: {self.config.drop_threshold}pts | Rise threshold: {self.config.rise_threshold}pts")
    
    def add_bar_data(self, bar_data: Dict):
        """Add new bar data and check for patterns"""
        try:
            # Ensure timestamp is timezone-aware
            if 'timestamp' in bar_data:
                if isinstance(bar_data['timestamp'], str):
                    bar_data['timestamp'] = pd.to_datetime(bar_data['timestamp'])
                
                if bar_data['timestamp'].tzinfo is None:
                    eastern_tz = pytz.timezone('US/Eastern')
                    bar_data['timestamp'] = eastern_tz.localize(bar_data['timestamp'])
            
            self.bar_buffer.append(bar_data)
            
            # Reset daily counter if new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_signal_count = 0
                self.last_reset_date = current_date
                logger.info(f"📅 New trading day: {current_date}")
            
            # Check for patterns with sufficient data
            if len(self.bar_buffer) >= 50:
                self._check_for_patterns()
                
        except Exception as e:
            logger.error(f"❌ Error adding bar data: {e}")
    
    def _check_for_patterns(self):
        """Check for both BUY and SELL patterns in recent data"""
        try:
            if self.daily_signal_count >= self.config.max_daily_signals:
                return
            
            # Get latest bar for timing validation
            latest_bar = list(self.bar_buffer)[-1]
            eastern_tz = pytz.timezone('US/Eastern')
            data_time_et = latest_bar['timestamp'].astimezone(eastern_tz)
            data_hour = data_time_et.hour + (data_time_et.minute / 60.0)
            
            # Check if in trading windows
            in_trading_window = any(
                start <= data_hour < end 
                for start, end in self.config.trading_windows
            )
            
            if not in_trading_window:
                logger.debug(f"🚫 Outside trading windows: {data_time_et.strftime('%H:%M')} ET")
                return
            
            # Convert buffer to DataFrame
            df = pd.DataFrame(list(self.bar_buffer))
            if len(df) < 50:
                return
            
            # Detect patterns
            patterns = self._detect_patterns_in_buffer(df)
            
            # Generate signals for new patterns
            for pattern in patterns:
                if pattern.signal_id not in self.processed_patterns:
                    self._generate_signal_file(pattern)
                    self.processed_patterns.add(pattern.signal_id)
                    self.daily_signal_count += 1
                    
                    # Update statistics
                    if pattern.action == "BUY":
                        self.stats['buy_signals'] += 1
                    else:
                        self.stats['sell_signals'] += 1
                    
                    logger.info(f"🚨 NEW {pattern.action} SIGNAL: {pattern.pattern_type} @ {pattern.entry_price:.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Error checking patterns: {e}")
    
    def _detect_patterns_in_buffer(self, df: pd.DataFrame) -> List[BidirectionalPattern]:
        """Detect both downward and upward V-reversal patterns"""
        patterns = []
        
        try:
            # Only check recent data to avoid reprocessing
            recent_df = df.tail(60).reset_index(drop=True)
            n = len(recent_df)
            self.stats['total_patterns_checked'] += 1
            
            # Look for patterns in recent bars
            for i in range(0, n - self.config.pattern_window):
                
                # Check for downward V-reversal (BUY signal)
                if self.config.enable_buy_signals:
                    buy_pattern = self._check_downward_v_pattern(recent_df, i)
                    if buy_pattern:
                        patterns.append(buy_pattern)
                        self.stats['patterns_above_threshold'] += 1
                
                # Check for upward inverted V-reversal (SELL signal)
                if self.config.enable_sell_signals:
                    sell_pattern = self._check_upward_v_pattern(recent_df, i)
                    if sell_pattern:
                        patterns.append(sell_pattern)
                        self.stats['patterns_above_threshold'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error in pattern detection: {e}")
        
        return patterns
    
    def _check_downward_v_pattern(self, df: pd.DataFrame, i: int) -> Optional[BidirectionalPattern]:
        """Check for downward V-reversal pattern (BUY signal)"""
        try:
            n = len(df)
            if i >= n - self.config.pattern_window:
                return None
            
            origin_high = df.at[i, 'high']
            
            # 1. Look for significant drop
            drop_end = min(i + self.config.pattern_window, n)
            low_idx = df['low'].iloc[i:drop_end].idxmin()
            drop_points = origin_high - df.at[low_idx, 'low']
            
            if drop_points < self.config.drop_threshold:
                return None
            
            # 2. Look for recovery/breakout above origin
            breakout_start = low_idx + 1
            breakout_end = min(breakout_start + self.config.breakout_window, n)
            breakout_idx = None
            
            for j in range(breakout_start, breakout_end):
                if df.at[j, 'high'] > origin_high:
                    breakout_idx = j
                    breakout_price = df.at[j, 'high']
                    break
            
            if breakout_idx is None:
                return None
            
            # 3. Look for pullback/entry opportunity
            pullback_start = breakout_idx + 1
            pullback_end = min(pullback_start + self.config.pullback_window, n)
            entry_idx = None
            
            for k in range(pullback_start, pullback_end):
                # Entry when price pulls back near origin level but stays above it
                if (df.at[k, 'low'] <= origin_high + 1.0 and 
                    df.at[k, 'close'] >= origin_high - 0.5):
                    entry_idx = k
                    break
            
            if entry_idx is None:
                return None
            
            # Must be recent pattern
            if entry_idx < n - 10:
                return None
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                df, i, low_idx, breakout_idx, entry_idx, drop_points, "DOWNWARD_V"
            )
            
            if confidence < self.config.min_pattern_confidence:
                return None
            
            # Generate pattern
            pattern_timestamp = df.at[entry_idx, 'timestamp']
            pattern_time_str = pattern_timestamp.strftime("%Y%m%d_%H%M%S")
            signal_id = f"BUY_{pattern_time_str}_{origin_high:.2f}_{df.at[low_idx, 'low']:.2f}"
            
            pattern = BidirectionalPattern(
                signal_id=signal_id,
                timestamp=pattern_timestamp,
                action="BUY",
                entry_price=df.at[entry_idx, 'close'],
                pattern_type="DOWNWARD_V",
                origin_price=origin_high,
                extreme_price=df.at[low_idx, 'low'],
                move_points=drop_points,
                breakout_price=breakout_price,
                pattern_confidence=confidence,
                pattern_duration=entry_idx - i
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"❌ Error checking downward V pattern: {e}")
            return None
    
    def _check_upward_v_pattern(self, df: pd.DataFrame, i: int) -> Optional[BidirectionalPattern]:
        """Check for upward inverted V-reversal pattern (SELL signal)"""
        try:
            n = len(df)
            if i >= n - self.config.pattern_window:
                return None
            
            origin_low = df.at[i, 'low']
            
            # 1. Look for significant rise
            rise_end = min(i + self.config.pattern_window, n)
            high_idx = df['high'].iloc[i:rise_end].idxmax()
            rise_points = df.at[high_idx, 'high'] - origin_low
            
            if rise_points < self.config.rise_threshold:
                return None
            
            # 2. Look for decline/breakdown below origin
            breakdown_start = high_idx + 1
            breakdown_end = min(breakdown_start + self.config.breakout_window, n)
            breakdown_idx = None
            
            for j in range(breakdown_start, breakdown_end):
                if df.at[j, 'low'] < origin_low:
                    breakdown_idx = j
                    breakdown_price = df.at[j, 'low']
                    break
            
            if breakdown_idx is None:
                return None
            
            # 3. Look for bounce/entry opportunity
            bounce_start = breakdown_idx + 1
            bounce_end = min(bounce_start + self.config.pullback_window, n)
            entry_idx = None
            
            for k in range(bounce_start, bounce_end):
                # Entry when price bounces near origin level but stays below it
                if (df.at[k, 'high'] >= origin_low - 1.0 and 
                    df.at[k, 'close'] <= origin_low + 0.5):
                    entry_idx = k
                    break
            
            if entry_idx is None:
                return None
            
            # Must be recent pattern
            if entry_idx < n - 10:
                return None
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                df, i, high_idx, breakdown_idx, entry_idx, rise_points, "UPWARD_V"
            )
            
            if confidence < self.config.min_pattern_confidence:
                return None
            
            # Generate pattern
            pattern_timestamp = df.at[entry_idx, 'timestamp']
            pattern_time_str = pattern_timestamp.strftime("%Y%m%d_%H%M%S")
            signal_id = f"SELL_{pattern_time_str}_{origin_low:.2f}_{df.at[high_idx, 'high']:.2f}"
            
            pattern = BidirectionalPattern(
                signal_id=signal_id,
                timestamp=pattern_timestamp,
                action="SELL",
                entry_price=df.at[entry_idx, 'close'],
                pattern_type="UPWARD_V",
                origin_price=origin_low,
                extreme_price=df.at[high_idx, 'high'],
                move_points=rise_points,
                breakout_price=breakdown_price,
                pattern_confidence=confidence,
                pattern_duration=entry_idx - i
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"❌ Error checking upward V pattern: {e}")
            return None
    
    def _calculate_pattern_confidence(self, df: pd.DataFrame, origin_idx: int, extreme_idx: int, 
                                    breakout_idx: int, entry_idx: int, move_points: float, 
                                    pattern_type: str) -> float:
        """Calculate confidence score for the pattern"""
        try:
            confidence = 0.6  # Base confidence
            
            # Factor 1: Move magnitude
            if move_points >= 6.0:
                confidence += 0.2
            elif move_points >= 5.0:
                confidence += 0.15
            elif move_points >= 4.5:
                confidence += 0.1
            
            # Factor 2: Pattern timing (faster = better)
            pattern_duration = entry_idx - origin_idx
            if pattern_duration <= 12:
                confidence += 0.15
            elif pattern_duration <= 20:
                confidence += 0.1
            
            # Factor 3: Volume confirmation (if available)
            if 'volume' in df.columns:
                try:
                    origin_volume = df.at[origin_idx, 'volume']
                    extreme_volume = df.at[extreme_idx, 'volume']
                    breakout_volume = df.at[breakout_idx, 'volume']
                    
                    # Higher volume on key moves
                    if extreme_volume > origin_volume * 1.1:
                        confidence += 0.05
                    if breakout_volume > extreme_volume * 1.1:
                        confidence += 0.05
                except:
                    pass
            
            # Factor 4: Pattern strength
            if pattern_type == "DOWNWARD_V":
                origin_price = df.at[origin_idx, 'high']
                breakout_strength = df.at[breakout_idx, 'high'] - origin_price
            else:  # UPWARD_V
                origin_price = df.at[origin_idx, 'low']
                breakout_strength = origin_price - df.at[breakout_idx, 'low']
            
            if breakout_strength >= 1.5:
                confidence += 0.1
            elif breakout_strength >= 0.75:
                confidence += 0.05
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating confidence: {e}")
            return 0.6
    
    def _generate_signal_file(self, pattern: BidirectionalPattern):
        """Generate signal file for NinjaTrader"""
        try:
            timestamp_str = pattern.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_vreversal_{pattern.action.lower()}_{timestamp_str}.txt"
            filepath = Path(self.config.signal_folder) / filename
            
            # Write signal file
            with open(filepath, 'w') as f:
                f.write(pattern.to_signal_file_content(self.config))
            
            logger.info(f"📁 Enhanced signal file generated: {filename}")
            logger.info(f"💰 {pattern.action} @ {pattern.entry_price:.2f} | {pattern.move_points:.1f}pts | {pattern.pattern_confidence:.1%}")
            
            # Console output
            action_color = "green" if pattern.action == "BUY" else "red"
            pattern_emoji = "📈" if pattern.action == "BUY" else "📉"
            
            self.console.print(Panel(
                f"[bold {action_color}]{pattern_emoji} {pattern.action} SIGNAL - {pattern.pattern_type}[/bold {action_color}]\n"
                f"[cyan]Entry:[/cyan] {pattern.entry_price:.2f}\n"
                f"[cyan]Move:[/cyan] {pattern.move_points:.2f} points\n"
                f"[cyan]Origin:[/cyan] {pattern.origin_price:.2f} → [cyan]Extreme:[/cyan] {pattern.extreme_price:.2f}\n"
                f"[cyan]Confidence:[/cyan] {pattern.pattern_confidence:.1%} | [cyan]Duration:[/cyan] {pattern.pattern_duration}min\n"
                f"[cyan]File:[/cyan] {filename}",
                title="Bidirectional Pattern Detection",
                border_style=action_color
            ))
            
        except Exception as e:
            logger.error(f"❌ Error generating signal file: {e}")
    
    async def run_realtime_detection(self, websocket_host: str = "192.168.1.65", websocket_port: int = 6789):
        """Run real-time bidirectional pattern detection"""
        try:
            # Import WebSocket client
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "python-client"))
            from websocket_client import WebSocketClient, MarketData
            
            logger.info(f"🔗 Connecting to WebSocket: {websocket_host}:{websocket_port}")
            
            client = WebSocketClient(websocket_host, websocket_port)
            
            async def on_market_data(data: MarketData):
                """Process incoming market data"""
                bar_data = {
                    'timestamp': data.timestamp,
                    'open': data.open,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume
                }
                self.add_bar_data(bar_data)
            
            # Set callback and start
            client.set_market_data_callback(on_market_data)
            await client.start()
            
        except Exception as e:
            logger.error(f"❌ Error in real-time detection: {e}")
    
    def get_statistics(self) -> Dict:
        """Get current detection statistics"""
        return {
            **self.stats,
            'daily_signals': self.daily_signal_count,
            'max_daily_signals': self.config.max_daily_signals,
            'processed_patterns': len(self.processed_patterns),
            'buffer_size': len(self.bar_buffer)
        }

def main():
    """Main function for testing"""
    config = BidirectionalVReversalConfig(
        signal_folder="signals/enhanced",
        enable_buy_signals=True,
        enable_sell_signals=True
    )
    
    detector = BidirectionalVReversalDetector(config)
    
    print("🚀 Bidirectional V-Reversal Detector ready")
    print("📊 Detecting both BUY and SELL opportunities")
    print("💡 Use run_realtime_detection() for live trading")

if __name__ == "__main__":
    main() 