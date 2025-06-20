
"""
detect_v_reversal.py
--------------------
Detects intraday "V‑reversal with breakout–pullback–continuation" patterns on 1‑minute ES data.

Pattern definition (configurable):
    1. Sharp drop: price falls ≥ drop_threshold points from a local high within drop_window minutes.
    2. Breakout: price then closes or makes a high above the *origin high* within breakout_window minutes.
    3. Pull‑back: price retraces toward the origin high without closing below it by more than pullback_tolerance
       within pullback_window minutes after breakout.
    4. Continuation: price subsequently makes a new high above the breakout high within continuation_window minutes.

Outputs a CSV with timestamps for each of the four key events.

Usage (example):
    python detect_v_reversal.py --file ES_1min.csv --out detections.csv \
        --drop_threshold 5 --drop_window 10 --breakout_window 30 \
        --pullback_window 10 --continuation_window 30 --pullback_tolerance 1.0
"""

import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Detect V‑reversal patterns in ES 1‑min data")
    p.add_argument("--file", required=True, help="Path to CSV with 1‑minute data")
    p.add_argument("--out", default="detections.csv", help="Output CSV path")
    p.add_argument("--drop_threshold", type=float, default=4.0, help="Min drop in points")
    p.add_argument("--drop_window", type=int, default=15, help="Minutes to search for drop")
    p.add_argument("--breakout_window", type=int, default=30, help="Minutes allowed to break above origin high")
    p.add_argument("--pullback_window", type=int, default=15, help="Minutes allowed for pull‑back")
    p.add_argument("--continuation_window", type=int, default=20, help="Minutes allowed for continuation higher high")
    p.add_argument("--pullback_tolerance", type=float, default=1.0, help="Tolerance (points) above/below origin high for pull‑back")
    # Trading windows are now hardcoded: 3-4 AM, 9-11 AM, 1:30-3:00 PM
    p.add_argument("--position_size", type=int, default=1, help="Position size for P&L calculation (number of contracts)")
    p.add_argument("--tick_value", type=float, default=12.50, help="Tick value in dollars (ES = $12.50 per 0.25 point)")
    p.add_argument("--stop_loss_pct", type=float, default=0.001, help="Stop loss as percentage of entry price (default 0.1%)")
    p.add_argument("--max_hold_time", type=int, default=60, help="Maximum minutes to hold position if no continuation")
    p.add_argument("--min_volume", type=int, default=0, help="Minimum volume filter for pattern validation")
    return p.parse_args()

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect a column called "Datetime" or "DateTime" (case‑insensitive)
    time_col = next((c for c in df.columns if c.lower() in ["datetime", "date", "timestamp"]), None)
    if time_col is None:
        raise ValueError("CSV must contain a 'Datetime' column (or 'Date'/'Timestamp').")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: "Datetime"})
    df = df.sort_values("Datetime").reset_index(drop=True)
    
    # Standardize column names to title case
    column_mapping = {}
    for col in df.columns:
        if col.lower() == "open":
            column_mapping[col] = "Open"
        elif col.lower() == "high":
            column_mapping[col] = "High"
        elif col.lower() == "low":
            column_mapping[col] = "Low"
        elif col.lower() == "close":
            column_mapping[col] = "Close"
    
    df = df.rename(columns=column_mapping)
    
    required_cols = {"Open","High","Low","Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df

def detect_patterns(df: pd.DataFrame,
                    drop_threshold: float,
                    drop_window: int,
                    breakout_window: int,
                    pullback_window: int,
                    continuation_window: int,
                    pullback_tolerance: float,
                    position_size: int = 1,
                    tick_value: float = 12.50,
                    stop_loss_pct: float = 0.01,
                    max_hold_time: int = 60,
                    min_volume: int = 0) -> tuple:
    successful_patterns = []
    failed_patterns = []
    n = len(df)
    i = 0
    while i < n - drop_window:
        origin_high = df.at[i, "High"]
        # 1. Look for minimum low within drop_window minutes
        low_idx = df["Low"].iloc[i:i+drop_window].idxmin()
        drop_points = origin_high - df.at[low_idx, "Low"]
        if drop_points < drop_threshold:
            i += 1
            continue
        # 2. Search for breakout above origin_high
        breakout_idx = None
        for j in range(low_idx+1, min(low_idx+1+breakout_window, n)):
            if df.at[j, "High"] > origin_high:
                breakout_idx = j
                breakout_high = df.at[j, "High"]
                break
        if breakout_idx is None:
            i = low_idx + 1
            continue
        # 3. Pull‑back test
        pullback_idx = None
        for k in range(breakout_idx+1, min(breakout_idx+1+pullback_window, n)):
            if abs(df.at[k, "Low"] - origin_high) <= pullback_tolerance and df.at[k, "Close"] >= origin_high - pullback_tolerance:
                pullback_idx = k
                break
        if pullback_idx is None:
            i = breakout_idx   # Skip ahead
            continue
        
        # 4. Continuation higher‑high - CHECK FOR BOTH SUCCESS AND FAILURE
        cont_idx = None
        for m in range(pullback_idx+1, min(pullback_idx+1+continuation_window, n)):
            if df.at[m, "High"] > breakout_high:
                cont_idx = m
                break
        
        # Get day of week information
        origin_datetime = df.at[i, "Datetime"]
        day_of_week = origin_datetime.strftime('%A')
        day_number = origin_datetime.weekday()  # Monday=0, Sunday=6
        
        # Calculate entry price (same for both successful and failed patterns)
        entry_price = df.at[breakout_idx, "Close"]  # Enter on close of breakout candle
        
        if cont_idx is not None:
            # SUCCESSFUL PATTERN - continuation occurred
            exit_price = df.at[cont_idx, "High"]        # Exit at high of continuation candle
            points_gained = exit_price - entry_price
            pnl_dollars = points_gained * position_size * (tick_value / 0.25)
            pnl_percent = (points_gained / entry_price) * 100
            
            successful_patterns.append({
                "origin_idx": i,
                "origin_time": origin_datetime,
                "day_of_week": day_of_week,
                "day_number": day_number,
                "origin_high": origin_high,
                "low_idx": low_idx,
                "low_time": df.at[low_idx, "Datetime"],
                "low_price": df.at[low_idx, "Low"],
                "drop_points": drop_points,
                "breakout_idx": breakout_idx,
                "breakout_time": df.at[breakout_idx, "Datetime"],
                "entry_price": entry_price,
                "pullback_idx": pullback_idx,
                "pullback_time": df.at[pullback_idx, "Datetime"],
                "continuation_idx": cont_idx,
                "continuation_time": df.at[cont_idx, "Datetime"],
                "exit_price": exit_price,
                "points_gained": points_gained,
                "pnl_dollars": pnl_dollars,
                "pnl_percent": pnl_percent,
                "pattern_status": "SUCCESS"
            })
            i = cont_idx + 1
        else:
            # FAILED PATTERN - continuation never occurred within time window
            # Use configurable parameters for exit strategy
            exit_idx = min(pullback_idx + min(continuation_window, max_hold_time), n - 1)
            
            # Find the actual exit point - either time limit or stop loss
            actual_exit_idx = exit_idx
            stop_loss_price = entry_price * (1 - stop_loss_pct)  # Configurable stop loss
            
            exit_reason = "TIME_LIMIT"
            for m in range(pullback_idx+1, exit_idx + 1):
                if df.at[m, "Low"] <= stop_loss_price:
                    actual_exit_idx = m
                    exit_reason = "STOP_LOSS"
                    break
            
            if exit_reason == "STOP_LOSS":
                exit_price = stop_loss_price
            else:
                # Exit at market price if time limit reached
                exit_price = df.at[actual_exit_idx, "Close"]
            points_gained = exit_price - entry_price  # This will be negative
            pnl_dollars = points_gained * position_size * (tick_value / 0.25)
            pnl_percent = (points_gained / entry_price) * 100
            
            failed_patterns.append({
                "origin_idx": i,
                "origin_time": origin_datetime,
                "day_of_week": day_of_week,
                "day_number": day_number,
                "origin_high": origin_high,
                "low_idx": low_idx,
                "low_time": df.at[low_idx, "Datetime"],
                "low_price": df.at[low_idx, "Low"],
                "drop_points": drop_points,
                "breakout_idx": breakout_idx,
                "breakout_time": df.at[breakout_idx, "Datetime"],
                "entry_price": entry_price,
                "pullback_idx": pullback_idx,
                "pullback_time": df.at[pullback_idx, "Datetime"],
                "continuation_idx": None,
                "continuation_time": None,
                "exit_price": exit_price,
                "points_gained": points_gained,
                                 "pnl_dollars": pnl_dollars,
                 "pnl_percent": pnl_percent,
                 "pattern_status": "FAILED",
                 "exit_reason": exit_reason
             })
            i = pullback_idx + 1
            
    return successful_patterns, failed_patterns

def main():
    args = parse_args()
    df = load_data(args.file)
    
    # Filter to specific high-volatility trading windows
    # 3-4 AM: Early European session
    # 9-11 AM: Market open + first hour 
    # 1:30-3:00 PM: Afternoon volatility
    
    time_col = df["Datetime"].dt.time
    
    window1 = (time_col >= pd.to_datetime("03:00").time()) & (time_col <= pd.to_datetime("04:00").time())
    window2 = (time_col >= pd.to_datetime("09:00").time()) & (time_col <= pd.to_datetime("11:00").time()) 
    window3 = (time_col >= pd.to_datetime("13:30").time()) & (time_col <= pd.to_datetime("15:00").time())
    
    # Combine all windows with OR logic
    trading_windows = window1 | window2 | window3
    df = df[trading_windows]
    
    print(f"📊 Filtered to trading windows: 3-4 AM, 9-11 AM, 1:30-3:00 PM")
    print(f"📈 Data points in windows: {len(df):,}")
    
    # Reset index after filtering to ensure continuous indexing
    df = df.reset_index(drop=True)

    successful_patterns, failed_patterns = detect_patterns(df,
                                                           drop_threshold=args.drop_threshold,
                                                           drop_window=args.drop_window,
                                                           breakout_window=args.breakout_window,
                                                           pullback_window=args.pullback_window,
                                                           continuation_window=args.continuation_window,
                                                           pullback_tolerance=args.pullback_tolerance,
                                                           position_size=args.position_size,
                                                           tick_value=args.tick_value,
                                                           stop_loss_pct=args.stop_loss_pct,
                                                           max_hold_time=args.max_hold_time,
                                                           min_volume=args.min_volume)
    
    # Combine successful and failed patterns
    all_patterns = successful_patterns + failed_patterns
    
    if all_patterns:
        out_df = pd.DataFrame(all_patterns)
        out_df.to_csv(args.out, index=False)
        
        # Also save separate files for analysis
        success_df = pd.DataFrame(successful_patterns) if successful_patterns else pd.DataFrame()
        failed_df = pd.DataFrame(failed_patterns) if failed_patterns else pd.DataFrame()
        
        if not success_df.empty:
            success_df.to_csv(args.out.replace('.csv', '_successful.csv'), index=False)
        if not failed_df.empty:
            failed_df.to_csv(args.out.replace('.csv', '_failed.csv'), index=False)
        
        # Print realistic trading statistics
        num_successful = len(successful_patterns)
        num_failed = len(failed_patterns)
        total_patterns = num_successful + num_failed
        
        total_pnl = out_df['pnl_dollars'].sum()
        avg_pnl = out_df['pnl_dollars'].mean()
        win_rate = (out_df['pnl_dollars'] > 0).sum() / len(out_df) * 100
        best_trade = out_df['pnl_dollars'].max()
        worst_trade = out_df['pnl_dollars'].min()
        avg_points = out_df['points_gained'].mean()
        
        print(f"Saved {len(out_df)} total pattern detections to {args.out}")
        print(f"\n🚨 REALISTIC TRADING ANALYSIS (Including Failed Patterns):")
        print(f"✅ Successful Patterns: {num_successful:,} ({num_successful/total_patterns*100:.1f}%)")
        print(f"❌ Failed Patterns: {num_failed:,} ({num_failed/total_patterns*100:.1f}%)")
        print(f"\n💰 P&L Summary (Realistic):")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Average P&L per trade: ${avg_pnl:,.2f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Best Trade: ${best_trade:,.2f}")
        print(f"Worst Trade: ${worst_trade:,.2f}")
        print(f"Average Points per Trade: {avg_points:.2f}")
        
        if num_successful > 0:
            success_pnl = success_df['pnl_dollars'].mean()
            print(f"\n📈 Successful Patterns Only:")
            print(f"Average P&L when successful: ${success_pnl:,.2f}")
            print(f"Average points when successful: {success_df['points_gained'].mean():.2f}")
            
        if num_failed > 0:
            failed_pnl = failed_df['pnl_dollars'].mean()
            print(f"\n📉 Failed Patterns:")
            print(f"Average loss when failed: ${failed_pnl:,.2f}")
            print(f"Average points when failed: {failed_df['points_gained'].mean():.2f}")
        
        # Day of week analysis
        print(f"\nDay of Week Analysis:")
        day_stats = out_df.groupby('day_of_week').agg({
            'pnl_dollars': ['count', 'sum', 'mean'],
            'points_gained': 'mean'
        }).round(2)
        
        # Flatten column names for easier access
        day_stats.columns = ['count', 'total_pnl', 'avg_pnl', 'avg_points']
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in day_order:
            if day in day_stats.index:
                count = int(day_stats.loc[day, 'count'])
                total = day_stats.loc[day, 'total_pnl']
                avg_pnl = day_stats.loc[day, 'avg_pnl']
                avg_pts = day_stats.loc[day, 'avg_points']
                pct_of_total = (count / len(out_df)) * 100
                print(f"  {day:9}: {count:4} patterns ({pct_of_total:4.1f}%) | Avg P&L: ${avg_pnl:7,.2f} | Avg Points: {avg_pts:4.2f}")
    else:
        print("No patterns detected with current parameters.")

if __name__ == "__main__":
    main()
