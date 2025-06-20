"""
advanced_strategy_optimizer.py
-------------------------------
Advanced trading strategy optimizer for $2,300/day target with intelligent position sizing,
dynamic stops, and trailing stop management.

Key Features:
1. Multi-contract position management (2 contracts)
2. Intelligent stop placement based on volatility
3. Trailing stop algorithms
4. Dynamic position sizing based on confidence
5. Risk-adjusted portfolio optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Advanced V-reversal strategy optimizer")
    p.add_argument("--file", required=True, help="Path to CSV with 1-minute data")
    p.add_argument("--out", default="advanced_strategy.csv", help="Output CSV")
    p.add_argument("--target_daily_pnl", type=float, default=2300, help="Target daily P&L")
    p.add_argument("--max_position_size", type=int, default=2, help="Maximum contracts per trade")
    p.add_argument("--base_drop_threshold", type=float, default=4.0, help="Base drop threshold")
    p.add_argument("--confidence_multiplier", type=float, default=1.5, help="Confidence-based sizing multiplier")
    return p.parse_args()

def calculate_atr(df, period=14):
    """Calculate Average True Range for volatility-based stops"""
    df['h_l'] = df['High'] - df['Low']
    df['h_c'] = abs(df['High'] - df['Close'].shift(1))
    df['l_c'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['h_l', 'h_c', 'l_c']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df

def calculate_confidence_score(drop_speed, breakout_strength, volume_surge=1.0):
    """Calculate pattern confidence score (0-100)"""
    # Base confidence from drop speed (faster = more confident)
    speed_score = min(drop_speed * 10, 40)  # Max 40 points
    
    # Breakout strength score
    strength_score = min(breakout_strength * 8, 35)  # Max 35 points
    
    # Volume surge bonus
    volume_score = min((volume_surge - 1) * 50, 25)  # Max 25 points
    
    total_score = speed_score + strength_score + volume_score
    return min(total_score, 100)

def dynamic_position_sizing(confidence_score, max_contracts=2):
    """Determine position size based on confidence"""
    if confidence_score >= 80:
        return max_contracts  # Full size for high confidence
    elif confidence_score >= 60:
        return max_contracts - 1  # Reduced size for medium confidence
    else:
        return 1  # Minimum size for low confidence

def calculate_intelligent_stops(entry_price, atr, action, confidence_score, base_stop_pct=0.002):
    """Calculate intelligent stop loss and take profit levels"""
    
    # Dynamic stop loss based on ATR and confidence
    if confidence_score >= 80:
        stop_multiplier = 1.5  # Tighter stops for high confidence
    elif confidence_score >= 60:
        stop_multiplier = 2.0  # Medium stops
    else:
        stop_multiplier = 2.5  # Wider stops for low confidence
    
    atr_stop = atr * stop_multiplier
    pct_stop = entry_price * base_stop_pct
    
    # Use the tighter of ATR or percentage stop
    stop_distance = min(atr_stop, pct_stop)
    
    if action == "BUY":
        stop_loss = entry_price - stop_distance
        # Dynamic take profit based on confidence and risk-reward ratio
        if confidence_score >= 80:
            take_profit = entry_price + (stop_distance * 4)  # 4:1 R/R
        elif confidence_score >= 60:
            take_profit = entry_price + (stop_distance * 3)  # 3:1 R/R
        else:
            take_profit = entry_price + (stop_distance * 2.5)  # 2.5:1 R/R
    else:  # SELL
        stop_loss = entry_price + stop_distance
        if confidence_score >= 80:
            take_profit = entry_price - (stop_distance * 4)
        elif confidence_score >= 60:
            take_profit = entry_price - (stop_distance * 3)
        else:
            take_profit = entry_price - (stop_distance * 2.5)
    
    return stop_loss, take_profit

def implement_trailing_stop(df, entry_idx, entry_price, action, initial_stop, trail_distance):
    """Implement trailing stop logic"""
    if action == "BUY":
        highest_price = entry_price
        trailing_stop = initial_stop
        
        for i in range(entry_idx + 1, len(df)):
            current_high = df.at[i, 'High']
            current_low = df.at[i, 'Low']
            
            # Update highest price and trailing stop
            if current_high > highest_price:
                highest_price = current_high
                new_trailing_stop = highest_price - trail_distance
                trailing_stop = max(trailing_stop, new_trailing_stop)
            
            # Check if stopped out
            if current_low <= trailing_stop:
                return i, trailing_stop, "TRAILING_STOP"
        
    else:  # SELL
        lowest_price = entry_price
        trailing_stop = initial_stop
        
        for i in range(entry_idx + 1, len(df)):
            current_high = df.at[i, 'High']
            current_low = df.at[i, 'Low']
            
            # Update lowest price and trailing stop
            if current_low < lowest_price:
                lowest_price = current_low
                new_trailing_stop = lowest_price + trail_distance
                trailing_stop = min(trailing_stop, new_trailing_stop)
            
            # Check if stopped out
            if current_high >= trailing_stop:
                return i, trailing_stop, "TRAILING_STOP"
    
    return None, trailing_stop, "NO_EXIT"

def detect_advanced_patterns(df, **kwargs):
    """Enhanced pattern detection with confidence scoring"""
    
    # Calculate ATR for intelligent stops
    df = calculate_atr(df)
    
    successful_patterns = []
    failed_patterns = []
    n = len(df)
    i = 0
    
    daily_pnl = {}  # Track daily P&L
    
    while i < n - 30:  # Need enough bars for pattern completion
        origin_high = df.at[i, "High"]
        origin_time = df.at[i, "Datetime"]
        current_date = origin_time.date()
        
        # Filter trading windows (3-4 AM, 9-11 AM, 1:30-3 PM)
        hour = origin_time.hour
        if not ((3 <= hour < 4) or (9 <= hour < 11) or (13 <= hour < 15)):
            i += 1
            continue
        
        # 1. Find sharp drop
        drop_window = 15
        drop_end_idx = min(i + drop_window, n-1)
        low_idx = df["Low"].iloc[i:drop_end_idx+1].idxmin()
        drop_points = origin_high - df.at[low_idx, "Low"]
        
        if drop_points < kwargs['base_drop_threshold']:
            i += 1
            continue
        
        # Calculate drop speed for confidence
        drop_minutes = low_idx - i + 1
        drop_speed = drop_points / drop_minutes if drop_minutes > 0 else 0
        
        # 2. Find breakout
        breakout_window = 30
        breakout_end_idx = min(low_idx + breakout_window, n-1)
        breakout_idx = None
        breakout_high = 0
        
        for j in range(low_idx+1, breakout_end_idx+1):
            if df.at[j, "High"] > origin_high:
                breakout_idx = j
                breakout_high = df.at[j, "High"]
                break
        
        if breakout_idx is None:
            i = low_idx + 1
            continue
        
        breakout_strength = breakout_high - origin_high
        
        # 3. Find pullback
        pullback_window = 15
        pullback_end_idx = min(breakout_idx + pullback_window, n-1)
        pullback_idx = None
        
        for k in range(breakout_idx+1, pullback_end_idx+1):
            if (abs(df.at[k, "Low"] - origin_high) <= 1.0 and 
                df.at[k, "Close"] >= origin_high - 1.0):
                pullback_idx = k
                break
        
        if pullback_idx is None:
            i = breakout_idx
            continue
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(drop_speed, breakout_strength)
        
        # Determine position size based on confidence
        position_size = dynamic_position_sizing(confidence_score, kwargs['max_position_size'])
        
        # Entry parameters
        entry_price = df.at[breakout_idx, "Close"]
        atr_value = df.at[breakout_idx, "atr"]
        
        # Calculate intelligent stops
        stop_loss, take_profit = calculate_intelligent_stops(
            entry_price, atr_value, "BUY", confidence_score
        )
        
        # Implement trailing stop
        trail_distance = atr_value * 1.0  # 1 ATR trailing distance
        
        # 4. Look for continuation or exit
        continuation_window = 20
        cont_end_idx = min(pullback_idx + continuation_window, n-1)
        
        # Find exit point
        exit_idx = None
        exit_price = None
        exit_reason = ""
        
        for m in range(pullback_idx+1, cont_end_idx+1):
            current_high = df.at[m, "High"]
            current_low = df.at[m, "Low"]
            
            # Check take profit
            if current_high >= take_profit:
                exit_idx = m
                exit_price = take_profit
                exit_reason = "TAKE_PROFIT"
                break
            
            # Check stop loss
            if current_low <= stop_loss:
                exit_idx = m
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break
        
        # If no exit found, use trailing stop or time exit
        if exit_idx is None:
            trail_exit_idx, trail_price, trail_reason = implement_trailing_stop(
                df, pullback_idx, entry_price, "BUY", stop_loss, trail_distance
            )
            
            if trail_exit_idx is not None:
                exit_idx = trail_exit_idx
                exit_price = trail_price
                exit_reason = trail_reason
            else:
                # Time-based exit
                exit_idx = cont_end_idx
                exit_price = df.at[cont_end_idx, "Close"]
                exit_reason = "TIME_EXIT"
        
        # Calculate P&L
        points_gained = exit_price - entry_price
        pnl_dollars = points_gained * position_size * 50  # ES = $50 per point
        
        # Track daily P&L
        if current_date not in daily_pnl:
            daily_pnl[current_date] = 0
        daily_pnl[current_date] += pnl_dollars
        
        # Create trade record
        trade_record = {
            "origin_idx": i,
            "origin_time": origin_time,
            "origin_high": origin_high,
            "low_idx": low_idx,
            "low_time": df.at[low_idx, "Datetime"],
            "low_price": df.at[low_idx, "Low"],
            "drop_points": drop_points,
            "drop_speed": drop_speed,
            "breakout_idx": breakout_idx,
            "breakout_time": df.at[breakout_idx, "Datetime"],
            "breakout_strength": breakout_strength,
            "confidence_score": confidence_score,
            "position_size": position_size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "exit_idx": exit_idx,
            "exit_time": df.at[exit_idx, "Datetime"],
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "points_gained": points_gained,
            "pnl_dollars": pnl_dollars,
            "daily_pnl_running": daily_pnl[current_date]
        }
        
        if points_gained > 0:
            trade_record["pattern_status"] = "SUCCESS"
            successful_patterns.append(trade_record)
        else:
            trade_record["pattern_status"] = "FAILED"
            failed_patterns.append(trade_record)
        
        i = exit_idx + 1
    
    return successful_patterns, failed_patterns, daily_pnl

def load_data(path):
    """Load and prepare data"""
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c.lower() in ["datetime", "date", "timestamp"]), None)
    if time_col is None:
        raise ValueError("CSV must contain a 'Datetime' column")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: "Datetime"})
    df = df.sort_values("Datetime").reset_index(drop=True)
    
    # Standardize columns
    for col in df.columns:
        if col.lower() == "open":
            df = df.rename(columns={col: "Open"})
        elif col.lower() == "high":
            df = df.rename(columns={col: "High"})
        elif col.lower() == "low":
            df = df.rename(columns={col: "Low"})
        elif col.lower() == "close":
            df = df.rename(columns={col: "Close"})
        elif col.lower() == "volume":
            df = df.rename(columns={col: "Volume"})
    
    if "Volume" not in df.columns:
        df["Volume"] = 1000
    
    return df

def main():
    args = parse_args()
    df = load_data(args.file)
    
    # Filter to trading windows
    time_col = df["Datetime"].dt.time
    window1 = (time_col >= pd.to_datetime("03:00").time()) & (time_col <= pd.to_datetime("04:00").time())
    window2 = (time_col >= pd.to_datetime("09:00").time()) & (time_col <= pd.to_datetime("11:00").time())
    window3 = (time_col >= pd.to_datetime("13:30").time()) & (time_col <= pd.to_datetime("15:00").time())
    
    trading_windows = window1 | window2 | window3
    df = df[trading_windows].reset_index(drop=True)
    
    print(f"🚀 ADVANCED STRATEGY OPTIMIZATION")
    print(f"Target Daily P&L: ${args.target_daily_pnl:,.0f}")
    print(f"Max Position Size: {args.max_position_size} contracts")
    print(f"Data points in trading windows: {len(df):,}")
    
    # Run advanced pattern detection
    params = vars(args)
    successful_patterns, failed_patterns, daily_pnl = detect_advanced_patterns(df, **params)
    
    # Combine results
    all_patterns = successful_patterns + failed_patterns
    
    if all_patterns:
        # Save detailed results
        results_df = pd.DataFrame(all_patterns)
        results_df.to_csv(args.out, index=False)
        
        # Calculate performance metrics
        total_trades = len(all_patterns)
        total_pnl = sum([t['pnl_dollars'] for t in all_patterns])
        win_rate = len(successful_patterns) / total_trades * 100
        avg_pnl_per_trade = total_pnl / total_trades
        
        # Daily analysis
        daily_pnl_list = list(daily_pnl.values())
        avg_daily_pnl = np.mean(daily_pnl_list)
        max_daily_pnl = np.max(daily_pnl_list)
        min_daily_pnl = np.min(daily_pnl_list)
        days_above_target = sum(1 for pnl in daily_pnl_list if pnl >= args.target_daily_pnl)
        total_days = len(daily_pnl_list)
        
        print(f"\n📊 ADVANCED STRATEGY RESULTS:")
        print(f"Total Trades: {total_trades:,}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Average P&L per Trade: ${avg_pnl_per_trade:.2f}")
        
        print(f"\n📅 DAILY PERFORMANCE:")
        print(f"Average Daily P&L: ${avg_daily_pnl:.2f}")
        print(f"Best Day: ${max_daily_pnl:.2f}")
        print(f"Worst Day: ${min_daily_pnl:.2f}")
        print(f"Days Above ${args.target_daily_pnl} Target: {days_above_target}/{total_days} ({days_above_target/total_days*100:.1f}%)")
        
        # Position sizing analysis
        size_1_trades = [t for t in all_patterns if t['position_size'] == 1]
        size_2_trades = [t for t in all_patterns if t['position_size'] == 2]
        
        print(f"\n📈 POSITION SIZE ANALYSIS:")
        if size_1_trades:
            size_1_pnl = np.mean([t['pnl_dollars'] for t in size_1_trades])
            print(f"1 Contract Trades: {len(size_1_trades)}, Avg P&L: ${size_1_pnl:.2f}")
        if size_2_trades:
            size_2_pnl = np.mean([t['pnl_dollars'] for t in size_2_trades])
            print(f"2 Contract Trades: {len(size_2_trades)}, Avg P&L: ${size_2_pnl:.2f}")
        
        # Confidence analysis
        high_conf_trades = [t for t in all_patterns if t['confidence_score'] >= 80]
        med_conf_trades = [t for t in all_patterns if 60 <= t['confidence_score'] < 80]
        low_conf_trades = [t for t in all_patterns if t['confidence_score'] < 60]
        
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        if high_conf_trades:
            high_conf_wr = sum(1 for t in high_conf_trades if t['pnl_dollars'] > 0) / len(high_conf_trades) * 100
            print(f"High Confidence (80+): {len(high_conf_trades)} trades, {high_conf_wr:.1f}% win rate")
        if med_conf_trades:
            med_conf_wr = sum(1 for t in med_conf_trades if t['pnl_dollars'] > 0) / len(med_conf_trades) * 100
            print(f"Medium Confidence (60-79): {len(med_conf_trades)} trades, {med_conf_wr:.1f}% win rate")
        if low_conf_trades:
            low_conf_wr = sum(1 for t in low_conf_trades if t['pnl_dollars'] > 0) / len(low_conf_trades) * 100
            print(f"Low Confidence (<60): {len(low_conf_trades)} trades, {low_conf_wr:.1f}% win rate")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS TO REACH ${args.target_daily_pnl}/DAY:")
        
        if avg_daily_pnl < args.target_daily_pnl:
            multiplier_needed = args.target_daily_pnl / avg_daily_pnl
            print(f"Current avg: ${avg_daily_pnl:.0f}/day, need {multiplier_needed:.1f}x improvement")
            print(f"Strategies to consider:")
            print(f"   1. Increase max position size to {int(args.max_position_size * multiplier_needed)} contracts")
            print(f"   2. Tighten entry criteria (higher confidence threshold)")
            print(f"   3. Add more trading sessions or instruments")
            print(f"   4. Implement portfolio of multiple strategies")
        else:
            print(f"✅ Target achieved! Current average: ${avg_daily_pnl:.0f}/day")
        
        print(f"\nSaved detailed results to {args.out}")
    
    else:
        print("No patterns detected with current parameters.")

if __name__ == "__main__":
    main() 