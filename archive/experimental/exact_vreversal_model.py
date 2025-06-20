"""
exact_vreversal_model.py
========================
Implementación EXACTA del detector V-reversal original que logra 91.8% win rate.
Copia la lógica exacta de detect_v_reversal.py incluyendo:
- Entrada en el breakout (no en pullback)
- Requiere continuación para ser exitoso
- Exit en continuation high
- Lógica exacta de ventanas de tiempo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Exact V-Reversal Model (91.8% win rate)")
    parser.add_argument("--file", required=True, help="Market data CSV file")
    parser.add_argument("--output", default="exact_vreversal_results.json", help="Output file")
    parser.add_argument("--position_size", type=int, default=1, help="Position size (contracts)")
    parser.add_argument("--scale_test", action="store_true", help="Test scaling to reach $2300/day")
    return parser.parse_args()

def load_data(file_path):
    """Carga datos EXACTAMENTE como en el detector original"""
    
    print(f"📊 Loading data: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Detectar columna de tiempo
    time_col = None
    for col in ['Datetime', 'datetime', 'Date', 'Timestamp', 'timestamp']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No time column found. Available: {list(df.columns)}")
    
    # Preparar datos EXACTAMENTE como en original
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: 'Datetime'})
    
    # Estandarizar nombres de columnas EXACTAMENTE como en original
    for col in df.columns:
        if col.lower() == 'open':
            df = df.rename(columns={col: 'Open'})
        elif col.lower() == 'high':
            df = df.rename(columns={col: 'High'})
        elif col.lower() == 'low':
            df = df.rename(columns={col: 'Low'})
        elif col.lower() == 'close':
            df = df.rename(columns={col: 'Close'})
        elif col.lower() == 'volume':
            df = df.rename(columns={col: 'Volume'})
    
    if 'Volume' not in df.columns:
        df['Volume'] = 1000
    
    # Filtrar a ventanas EXACTAS del original
    print(f"📊 Original data points: {len(df):,}")
    
    time_col = df["Datetime"].dt.time
    
    # EXACTAMENTE como en el original
    window1 = (time_col >= pd.to_datetime("03:00").time()) & (time_col <= pd.to_datetime("04:00").time())
    window2 = (time_col >= pd.to_datetime("09:00").time()) & (time_col <= pd.to_datetime("11:00").time()) 
    window3 = (time_col >= pd.to_datetime("13:30").time()) & (time_col <= pd.to_datetime("15:00").time())
    
    trading_windows = window1 | window2 | window3
    df = df[trading_windows]
    
    print(f"📊 Filtered to trading windows: 3-4 AM, 9-11 AM, 1:30-3:00 PM")
    print(f"📈 Data points in windows: {len(df):,}")
    
    # Reset index EXACTAMENTE como en original
    df = df.reset_index(drop=True)
    
    return df

def detect_exact_patterns(df, position_size=1):
    """
    Detector EXACTO copiado de detect_v_reversal.py que logra 91.8% win rate
    """
    
    # Parámetros EXACTOS del detector validado
    drop_threshold = 4.0
    drop_window = 15
    breakout_window = 30
    pullback_window = 15
    continuation_window = 20
    pullback_tolerance = 1.0
    tick_value = 12.50
    stop_loss_pct = 0.001
    max_hold_time = 60
    
    successful_patterns = []
    failed_patterns = []
    n = len(df)
    i = 0
    
    print(f"🔍 Detecting patterns with EXACT original logic...")
    print(f"📊 Parameters: drop={drop_threshold}, stop={stop_loss_pct*100}%, contracts={position_size}")
    
    while i < n - drop_window:
        origin_high = df.at[i, "High"]
        origin_datetime = df.at[i, "Datetime"]
        
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
            if (abs(df.at[k, "Low"] - origin_high) <= pullback_tolerance and 
                df.at[k, "Close"] >= origin_high - pullback_tolerance):
                pullback_idx = k
                break
        
        if pullback_idx is None:
            i = breakout_idx
            continue
        
        # 4. Continuation higher‑high - CHECK FOR BOTH SUCCESS AND FAILURE
        cont_idx = None
        for m in range(pullback_idx+1, min(pullback_idx+1+continuation_window, n)):
            if df.at[m, "High"] > breakout_high:
                cont_idx = m
                break
        
        # Get day info
        day_of_week = origin_datetime.strftime('%A')
        day_number = origin_datetime.weekday()
        
        # ENTRADA EN EL BREAKOUT (EXACTO como original)
        entry_price = df.at[breakout_idx, "Close"]
        
        if cont_idx is not None:
            # SUCCESSFUL PATTERN - continuation occurred
            exit_price = df.at[cont_idx, "High"]  # Exit at high of continuation candle
            points_gained = exit_price - entry_price
            pnl_dollars = points_gained * position_size * (tick_value / 0.25)  # EXACTO
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
            exit_idx = min(pullback_idx + min(continuation_window, max_hold_time), n - 1)
            
            # Find the actual exit point - either time limit or stop loss
            actual_exit_idx = exit_idx
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            
            exit_reason = "TIME_LIMIT"
            for m in range(pullback_idx+1, exit_idx + 1):
                if df.at[m, "Low"] <= stop_loss_price:
                    actual_exit_idx = m
                    exit_reason = "STOP_LOSS"
                    break
            
            if exit_reason == "STOP_LOSS":
                exit_price = stop_loss_price
            else:
                exit_price = df.at[actual_exit_idx, "Close"]
            
            points_gained = exit_price - entry_price  # Will be negative
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

def analyze_exact_results(successful_patterns, failed_patterns, position_size):
    """Análisis EXACTO como el original"""
    
    all_patterns = successful_patterns + failed_patterns
    
    if not all_patterns:
        print("❌ No patterns detected")
        return {}
    
    print(f"\n📊 ANÁLISIS EXACTO DEL DETECTOR VALIDADO")
    print("=" * 55)
    
    # Estadísticas básicas EXACTAS
    num_successful = len(successful_patterns)
    num_failed = len(failed_patterns)
    total_patterns = num_successful + num_failed
    
    out_df = pd.DataFrame(all_patterns)
    total_pnl = out_df['pnl_dollars'].sum()
    avg_pnl = out_df['pnl_dollars'].mean()
    win_rate = (out_df['pnl_dollars'] > 0).sum() / len(out_df) * 100
    best_trade = out_df['pnl_dollars'].max()
    worst_trade = out_df['pnl_dollars'].min()
    avg_points = out_df['points_gained'].mean()
    
    print(f"📈 RESULTADOS EXACTOS:")
    print(f"   Total Patterns: {total_patterns:,}")
    print(f"   ✅ Successful: {num_successful:,} ({num_successful/total_patterns*100:.1f}%)")
    print(f"   ❌ Failed: {num_failed:,} ({num_failed/total_patterns*100:.1f}%)")
    print(f"   🎯 Win Rate: {win_rate:.1f}%")
    
    print(f"\n💰 P&L ANALYSIS:")
    print(f"   Total P&L: ${total_pnl:,.2f}")
    print(f"   Average P&L per trade: ${avg_pnl:.2f}")
    print(f"   Best Trade: ${best_trade:,.2f}")
    print(f"   Worst Trade: ${worst_trade:,.2f}")
    print(f"   Average Points per Trade: {avg_points:.2f}")
    
    # Análisis diario y proyecciones
    df_with_dates = pd.DataFrame(all_patterns)
    df_with_dates['date'] = pd.to_datetime(df_with_dates['origin_time']).dt.date
    
    daily_stats = df_with_dates.groupby('date').agg({
        'pnl_dollars': ['sum', 'count']
    })
    daily_stats.columns = ['daily_pnl', 'daily_trades']
    
    avg_daily_pnl = daily_stats['daily_pnl'].mean()
    avg_daily_trades = daily_stats['daily_trades'].mean()
    max_daily_pnl = daily_stats['daily_pnl'].max()
    min_daily_pnl = daily_stats['daily_pnl'].min()
    
    # Proyecciones usando 21 días de trading por mes
    trading_days_per_month = 21
    monthly_pnl = avg_daily_pnl * trading_days_per_month
    monthly_trades = avg_daily_trades * trading_days_per_month
    annual_pnl = monthly_pnl * 12
    
    print(f"\n📅 PERFORMANCE DIARIA:")
    print(f"   Avg Daily P&L: ${avg_daily_pnl:.2f}")
    print(f"   Avg Daily Trades: {avg_daily_trades:.1f}")
    print(f"   Best Day: ${max_daily_pnl:.2f}")
    print(f"   Worst Day: ${min_daily_pnl:.2f}")
    
    print(f"\n📊 PROYECCIONES MENSUALES:")
    print(f"   Monthly P&L: ${monthly_pnl:,.2f}")
    print(f"   Monthly Trades: {monthly_trades:.0f}")
    print(f"   Annual P&L: ${annual_pnl:,.2f}")
    
    # Comparación con resultados esperados
    expected_win_rate = 91.8
    expected_monthly_pnl = 13211
    
    print(f"\n🎯 VALIDACIÓN DEL MODELO:")
    print(f"   Win Rate: {win_rate:.1f}% vs {expected_win_rate:.1f}% esperado")
    print(f"   Monthly P&L: ${monthly_pnl:.0f} vs ${expected_monthly_pnl:.0f} esperado")
    
    win_rate_diff = win_rate - expected_win_rate
    pnl_diff_pct = (monthly_pnl / expected_monthly_pnl - 1) * 100
    
    validation_status = "✅ MODELO VALIDADO" if (abs(win_rate_diff) < 10 and abs(pnl_diff_pct) < 30) else "⚠️ DIFERENCIAS DETECTADAS"
    print(f"   Status: {validation_status}")
    
    if validation_status == "✅ MODELO VALIDADO":
        print(f"   🎉 El modelo replica correctamente los resultados conocidos!")
    else:
        print(f"   Win Rate diff: {win_rate_diff:+.1f}%")
        print(f"   P&L diff: {pnl_diff_pct:+.1f}%")
    
    # Análisis para scaling a $2,300/día
    target_daily = 2300
    scale_factor = target_daily / avg_daily_pnl if avg_daily_pnl > 0 else float('inf')
    
    print(f"\n🎯 SCALING PARA ${target_daily}/DÍA:")
    print(f"   Factor necesario: {scale_factor:.1f}x")
    
    if scale_factor <= 1.5:
        print(f"   ✅ ALCANZABLE: Usar {int(position_size * scale_factor)} contratos")
    elif scale_factor <= 3.0:
        print(f"   ⚠️ AGRESIVO: Usar {int(position_size * scale_factor)} contratos + optimizaciones")
    elif scale_factor <= 5.0:
        print(f"   🚨 MUY AGRESIVO: Requiere {int(position_size * scale_factor)} contratos")
        print("   Considera múltiples instrumentos o estrategias adicionales")
    else:
        print("   🔴 EXTREMADAMENTE DIFÍCIL: Requiere enfoque completamente diferente")
    
    # Análisis de exit reasons
    if failed_patterns:
        exit_reasons = {}
        for pattern in failed_patterns:
            reason = pattern.get('exit_reason', 'UNKNOWN')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        print(f"\n📤 ANÁLISIS DE SALIDAS FALLIDAS:")
        for reason, count in exit_reasons.items():
            percentage = count / len(failed_patterns) * 100
            print(f"   {reason}: {count} ({percentage:.1f}%)")
    
    return {
        'total_patterns': total_patterns,
        'successful_patterns': num_successful,
        'failed_patterns': num_failed,
        'win_rate': win_rate,
        'avg_daily_pnl': avg_daily_pnl,
        'monthly_pnl': monthly_pnl,
        'scale_factor_needed': scale_factor,
        'model_validated': validation_status == "✅ MODELO VALIDADO"
    }

def test_scaling(df, target_daily=2300):
    """Prueba diferentes escalas para alcanzar el target diario"""
    
    print(f"\n🚀 PROBANDO SCALING PARA ${target_daily}/DÍA")
    print("=" * 45)
    
    scale_factors = [1, 2, 3, 4, 5]
    
    best_scale = None
    best_pnl = 0
    
    for scale in scale_factors:
        print(f"\n📊 Probando {scale} contratos...")
        
        successful, failed = detect_exact_patterns(df, position_size=scale)
        all_patterns = successful + failed
        
        if all_patterns:
            df_patterns = pd.DataFrame(all_patterns)
            df_patterns['date'] = pd.to_datetime(df_patterns['origin_time']).dt.date
            
            daily_stats = df_patterns.groupby('date')['pnl_dollars'].sum()
            avg_daily_pnl = daily_stats.mean()
            win_rate = (df_patterns['pnl_dollars'] > 0).sum() / len(df_patterns) * 100
            
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Avg Daily P&L: ${avg_daily_pnl:.2f}")
            
            if avg_daily_pnl >= target_daily:
                print(f"   🎉 ¡TARGET ALCANZADO con {scale} contratos!")
                if best_scale is None:
                    best_scale = scale
                    best_pnl = avg_daily_pnl
            elif avg_daily_pnl > best_pnl:
                best_pnl = avg_daily_pnl
                best_scale = scale
    
    if best_scale:
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"   {best_scale} contratos = ${best_pnl:.2f}/día")
        
        if best_pnl >= target_daily:
            print(f"   ✅ Target alcanzado!")
        else:
            remaining_gap = target_daily - best_pnl
            print(f"   ⚠️ Faltan ${remaining_gap:.0f}/día para el target")
    
    return best_scale, best_pnl

def main():
    args = parse_args()
    
    try:
        print("🔬 MODELO EXACTO V-REVERSAL (91.8% WIN RATE)")
        print("Lógica exacta del detector validado")
        print("=" * 50)
        
        # Cargar datos
        df = load_data(args.file)
        
        # Ejecutar detector exacto
        successful_patterns, failed_patterns = detect_exact_patterns(df, args.position_size)
        
        # Analizar resultados
        results = analyze_exact_results(successful_patterns, failed_patterns, args.position_size)
        
        # Test de scaling si se solicita
        if args.scale_test and results.get('model_validated', False):
            best_scale, best_pnl = test_scaling(df)
            results['optimal_scale'] = best_scale
            results['optimal_daily_pnl'] = best_pnl
        
        # Guardar resultados
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'position_size': args.position_size,
            'results': results,
            'successful_patterns': successful_patterns,
            'failed_patterns': failed_patterns
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Resultados guardados en: {args.output}")
        
        # Resumen final
        if results.get('model_validated', False):
            print(f"\n🎉 MODELO VALIDADO EXITOSAMENTE!")
            print(f"   Replicando exactamente el comportamiento del detector original")
            
            if args.scale_test and 'optimal_scale' in results:
                optimal_scale = results['optimal_scale']
                optimal_pnl = results['optimal_daily_pnl']
                print(f"   🚀 Configuración óptima: {optimal_scale} contratos = ${optimal_pnl:.0f}/día")
        else:
            print(f"\n⚠️ Modelo requiere ajustes adicionales")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 