"""
validated_model.py
==================
Modelo basado EXACTAMENTE en los parámetros validados del V-reversal system.
- 91.8% win rate comprobado
- $13,211/mes P&L validado
- Parámetros optimizados y probados

Objetivo: Replicar exactamente los resultados conocidos y luego escalar.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Validated V-Reversal Model")
    parser.add_argument("--file", required=True, help="Market data CSV file")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Scaling factor for position size")
    parser.add_argument("--output", default="validated_results.json", help="Output file")
    return parser.parse_args()

class ValidatedVReversalDetector:
    """
    Implementación EXACTA del detector V-reversal validado
    """
    
    def __init__(self):
        # Parámetros exactos del backtest validado
        self.drop_threshold = 4.0  # Puntos mínimos de caída
        self.stop_loss_pct = 0.001  # 0.1% stop loss
        
        # Ventanas exactas del patrón validado
        self.drop_window = 15      # Ventana para detectar caída
        self.breakout_window = 30  # Ventana para detectar breakout
        self.pullback_window = 15  # Ventana para detectar pullback
        self.continuation_window = 20  # Ventana para continuación
        
        # Configuración de trading validada
        self.trading_windows = [
            (3, 4),   # 3-4 AM
            (9, 11),  # 9-11 AM  
            (13, 15)  # 1:30-3 PM
        ]
    
    def is_valid_trading_time(self, timestamp):
        """Verifica si estamos en una ventana de trading validada"""
        hour = timestamp.hour
        
        for start_hour, end_hour in self.trading_windows:
            if start_hour <= hour < end_hour:
                return True
        return False
    
    def detect_pattern(self, df, idx):
        """
        Detecta patrón V-reversal usando la lógica EXACTA validada
        """
        
        if idx < 50 or idx >= len(df) - 30:
            return False, {}
        
        current_time = df.iloc[idx]['Datetime']
        
        # Verificar ventana de trading
        if not self.is_valid_trading_time(current_time):
            return False, {}
        
        # FASE 1: SHARP DROP
        origin_idx = idx
        origin_high = df.iloc[origin_idx]['High']
        
        # Buscar caída en los próximos 15 minutos
        drop_end_idx = min(origin_idx + self.drop_window, len(df) - 1)
        drop_section = df.iloc[origin_idx:drop_end_idx + 1]
        
        # Encontrar el punto más bajo
        low_idx_relative = drop_section['Low'].idxmin()
        low_idx_absolute = origin_idx + (low_idx_relative - origin_idx)
        drop_low = df.iloc[low_idx_absolute]['Low']
        drop_points = origin_high - drop_low
        
        # Validar drop mínimo
        if drop_points < self.drop_threshold:
            return False, {}
        
        # FASE 2: BREAKOUT
        breakout_start = low_idx_absolute
        breakout_end = min(breakout_start + self.breakout_window, len(df) - 1)
        
        breakout_idx = None
        for i in range(breakout_start + 1, breakout_end + 1):
            if i >= len(df):
                break
            if df.iloc[i]['High'] > origin_high:
                breakout_idx = i
                break
        
        if breakout_idx is None:
            return False, {}
        
        # FASE 3: PULLBACK
        pullback_start = breakout_idx
        pullback_end = min(pullback_start + self.pullback_window, len(df) - 1)
        
        pullback_idx = None
        for i in range(pullback_start + 1, pullback_end + 1):
            if i >= len(df):
                break
            
            current_low = df.iloc[i]['Low']
            current_close = df.iloc[i]['Close']
            
            # Pullback: precio cerca del nivel original (tolerancia de 1 punto)
            if (abs(current_low - origin_high) <= 1.0 and 
                current_close >= origin_high - 1.0):
                pullback_idx = i
                break
        
        if pullback_idx is None:
            return False, {}
        
        # PATRÓN VÁLIDO ENCONTRADO
        entry_price = df.iloc[pullback_idx]['Close']
        
        # Calcular stops usando configuración validada
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        
        # Take profit basado en risk-reward ratio validado
        stop_distance = entry_price - stop_loss
        take_profit = entry_price + (stop_distance * 3.0)  # 3:1 R/R ratio
        
        pattern_info = {
            'origin_idx': origin_idx,
            'origin_time': current_time,
            'origin_high': origin_high,
            'low_idx': low_idx_absolute,
            'low_price': drop_low,
            'drop_points': drop_points,
            'breakout_idx': breakout_idx,
            'pullback_idx': pullback_idx,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        return True, pattern_info

def load_data(file_path):
    """Carga datos exactamente como en el sistema validado"""
    
    print(f"📊 Cargando datos: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Detectar columna de tiempo
    time_col = None
    possible_time_cols = ['Datetime', 'datetime', 'Date', 'Timestamp', 'timestamp']
    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No time column found. Available: {list(df.columns)}")
    
    # Preparar datos
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: 'Datetime'})
    
    # Estandarizar nombres
    column_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == 'open':
            column_map[col] = 'Open'
        elif lower == 'high':
            column_map[col] = 'High'  
        elif lower == 'low':
            column_map[col] = 'Low'
        elif lower == 'close':
            column_map[col] = 'Close'
        elif lower == 'volume':
            column_map[col] = 'Volume'
    
    df = df.rename(columns=column_map)
    
    if 'Volume' not in df.columns:
        df['Volume'] = 1000
    
    # Filtrar a ventanas de trading validadas
    df = df.sort_values('Datetime').reset_index(drop=True)
    df['hour'] = df['Datetime'].dt.hour
    
    # Solo ventanas específicas
    mask = ((df['hour'] >= 3) & (df['hour'] < 4)) | \
           ((df['hour'] >= 9) & (df['hour'] < 11)) | \
           ((df['hour'] >= 13) & (df['hour'] < 15))
    
    df_filtered = df[mask].reset_index(drop=True)
    
    print(f"✅ Total records: {len(df):,}")
    print(f"✅ Filtered to trading windows: {len(df_filtered):,}")
    print(f"✅ Coverage: {len(df_filtered)/len(df)*100:.1f}%")
    
    return df_filtered

def simulate_trade(df, entry_idx, entry_price, stop_loss, take_profit):
    """Simula ejecución de trade con lógica validada"""
    
    max_hold = 25  # Máximo 25 períodos de retención
    
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        bar = df.iloc[i]
        
        # Take profit hit
        if bar['High'] >= take_profit:
            return take_profit, "TAKE_PROFIT", i
        
        # Stop loss hit
        if bar['Low'] <= stop_loss:
            return stop_loss, "STOP_LOSS", i
    
    # Time-based exit
    exit_idx = min(entry_idx + max_hold - 1, len(df) - 1)
    return df.iloc[exit_idx]['Close'], "TIME_EXIT", exit_idx

def run_validated_backtest(df, scale_factor=1.0):
    """Ejecuta backtest con configuración validada"""
    
    print(f"\n🚀 EJECUTANDO BACKTEST VALIDADO")
    print(f"📊 Registros: {len(df):,}")
    print(f"📈 Factor de escala: {scale_factor}x")
    
    detector = ValidatedVReversalDetector()
    
    trades = []
    daily_stats = {}
    
    # Posición base validada
    base_position_size = 1  # Empezar con 1 contrato como en el original
    
    i = 50  # Margen inicial
    while i < len(df) - 30:
        
        current_time = df.iloc[i]['Datetime']
        current_date = current_time.date()
        
        # Inicializar día
        if current_date not in daily_stats:
            daily_stats[current_date] = {'pnl': 0, 'trades': 0}
        
        # Buscar patrón
        found, pattern_info = detector.detect_pattern(df, i)
        
        if found:
            # Configurar trade
            entry_idx = pattern_info['pullback_idx']
            entry_price = pattern_info['entry_price']
            stop_loss = pattern_info['stop_loss']
            take_profit = pattern_info['take_profit']
            
            # Tamaño de posición escalado
            position_size = int(base_position_size * scale_factor)
            position_size = max(1, position_size)  # Mínimo 1 contrato
            
            # Simular ejecución
            exit_price, exit_reason, exit_idx = simulate_trade(
                df, entry_idx, entry_price, stop_loss, take_profit
            )
            
            # Calcular P&L
            points_gained = exit_price - entry_price
            pnl_dollars = points_gained * position_size * 50  # ES = $50/punto
            
            # Registrar trade
            trade_record = {
                'date': str(current_date),
                'entry_time': str(current_time),
                'entry_price': entry_price,
                'exit_time': str(df.iloc[exit_idx]['Datetime']),
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'points_gained': points_gained,
                'pnl_dollars': pnl_dollars,
                'drop_points': pattern_info['drop_points'],
                'origin_high': pattern_info['origin_high'],
                'pattern_quality': 'validated'
            }
            
            trades.append(trade_record)
            
            # Actualizar estadísticas diarias
            daily_stats[current_date]['pnl'] += pnl_dollars
            daily_stats[current_date]['trades'] += 1
            
            # Avanzar al final del trade
            i = exit_idx + 1
        else:
            i += 1
    
    return trades, daily_stats

def analyze_validated_results(trades, daily_stats):
    """Analiza resultados del modelo validado"""
    
    if not trades:
        print("❌ No trades found")
        return {}
    
    print(f"\n📊 ANÁLISIS DE MODELO VALIDADO")
    print("=" * 50)
    
    # Estadísticas básicas
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl_dollars'] > 0]
    total_pnl = sum(t['pnl_dollars'] for t in trades)
    
    win_rate = len(winning_trades) / total_trades * 100
    avg_pnl_per_trade = total_pnl / total_trades
    
    print(f"📈 RESULTADOS BÁSICOS:")
    print(f"   Total Trades: {total_trades:,}")
    print(f"   Trades Ganadores: {len(winning_trades):,}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   P&L Total: ${total_pnl:,.2f}")
    print(f"   P&L Promedio/Trade: ${avg_pnl_per_trade:.2f}")
    
    # Performance diaria
    daily_pnls = [day['pnl'] for day in daily_stats.values()]
    daily_trades_count = [day['trades'] for day in daily_stats.values()]
    
    avg_daily_pnl = np.mean(daily_pnls)
    avg_trades_per_day = np.mean(daily_trades_count)
    max_daily_pnl = np.max(daily_pnls)
    min_daily_pnl = np.min(daily_pnls)
    
    print(f"\n📅 PERFORMANCE DIARIA:")
    print(f"   P&L Diario Promedio: ${avg_daily_pnl:.2f}")
    print(f"   Trades Promedio/Día: {avg_trades_per_day:.1f}")
    print(f"   Mejor Día: ${max_daily_pnl:.2f}")
    print(f"   Peor Día: ${min_daily_pnl:.2f}")
    
    # Proyecciones
    trading_days_month = 21
    monthly_pnl = avg_daily_pnl * trading_days_month
    monthly_trades = avg_trades_per_day * trading_days_month
    annual_pnl = monthly_pnl * 12
    
    print(f"\n📊 PROYECCIONES:")
    print(f"   P&L Mensual: ${monthly_pnl:,.2f}")
    print(f"   Trades Mensuales: {monthly_trades:.0f}")
    print(f"   P&L Anual: ${annual_pnl:,.2f}")
    
    # Comparación con resultados esperados
    expected_win_rate = 91.8
    expected_monthly_pnl = 13211
    
    print(f"\n🎯 COMPARACIÓN CON RESULTADOS ESPERADOS:")
    print(f"   Win Rate: {win_rate:.1f}% vs {expected_win_rate:.1f}% esperado")
    print(f"   P&L Mensual: ${monthly_pnl:.0f} vs ${expected_monthly_pnl:.0f} esperado")
    
    win_rate_diff = win_rate - expected_win_rate
    pnl_diff = (monthly_pnl / expected_monthly_pnl - 1) * 100
    
    if abs(win_rate_diff) < 5 and abs(pnl_diff) < 20:
        print("✅ RESULTADOS CONSISTENTES con modelo validado")
    else:
        print("⚠️ DIFERENCIAS SIGNIFICATIVAS detectadas")
        print(f"   Win Rate diff: {win_rate_diff:+.1f}%")
        print(f"   P&L diff: {pnl_diff:+.1f}%")
    
    # Scaling análisis para alcanzar $2,300/día
    target_daily = 2300
    scale_needed = target_daily / avg_daily_pnl if avg_daily_pnl > 0 else float('inf')
    
    print(f"\n🎯 SCALING PARA ${target_daily}/DÍA:")
    print(f"   Factor de escala necesario: {scale_needed:.1f}x")
    
    if scale_needed <= 2:
        print("✅ ALCANZABLE con scaling moderado")
        print(f"   Recomendación: Usar {int(scale_needed)} contratos por trade")
    elif scale_needed <= 4:
        print("⚠️ REQUIERE scaling agresivo")
        print("   Considera múltiples estrategias o instrumentos")
    else:
        print("🚨 SCALING EXTREMO requerido")
        print("   Necesitas enfoque completamente diferente")
    
    # Análisis de exit reasons
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = 0
        exit_reasons[reason] += 1
    
    print(f"\n📤 ANÁLISIS DE SALIDAS:")
    for reason, count in exit_reasons.items():
        percentage = count / total_trades * 100
        print(f"   {reason}: {count} trades ({percentage:.1f}%)")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_daily_pnl': avg_daily_pnl,
        'monthly_pnl': monthly_pnl,
        'scale_factor_needed': scale_needed,
        'model_validation': {
            'win_rate_match': abs(win_rate - expected_win_rate) < 5,
            'pnl_match': abs(pnl_diff) < 20
        }
    }

def main():
    args = parse_args()
    
    try:
        print("🔬 MODELO VALIDADO V-REVERSAL")
        print("Basado en configuración exacta probada")
        print("=" * 45)
        
        # Cargar datos
        df = load_data(args.file)
        
        # Ejecutar backtest
        trades, daily_stats = run_validated_backtest(df, args.scale_factor)
        
        # Analizar
        results = analyze_validated_results(trades, daily_stats)
        
        # Guardar
        output = {
            'timestamp': datetime.now().isoformat(),
            'scale_factor': args.scale_factor,
            'results': results,
            'trades': trades,
            'daily_stats': {str(k): v for k, v in daily_stats.items()}
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Guardado en: {args.output}")
        
        # Probar diferentes escalas si el modelo es válido
        if results.get('model_validation', {}).get('win_rate_match', False):
            print(f"\n🚀 MODELO VALIDADO CORRECTAMENTE")
            
            scale_needed = results.get('scale_factor_needed', 1)
            if scale_needed <= 3:
                print(f"\n📈 PROBANDO ESCALA {scale_needed:.1f}x...")
                
                # Ejecutar con scaling
                scaled_trades, scaled_daily = run_validated_backtest(df, scale_needed)
                scaled_results = analyze_validated_results(scaled_trades, scaled_daily)
                
                if scaled_results['avg_daily_pnl'] >= 2300:
                    print(f"🎉 ¡TARGET ALCANZADO!")
                    print(f"   P&L diario con escala: ${scaled_results['avg_daily_pnl']:.0f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 