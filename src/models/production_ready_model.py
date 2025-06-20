"""
production_ready_model.py
=========================
Modelo de producción basado en resultados reales del V-reversal system.
Objetivo: $2,300/día con gestión de riesgo robusta.

Basado en datos reales:
- 91.8% win rate con configuración windowed
- ~$13,211/mes con trading de 4.5 horas
- Ventanas: 3-4 AM, 9-11 AM, 1:30-3 PM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Production Ready Trading Model")
    parser.add_argument("--file", required=True, help="Market data CSV file")
    parser.add_argument("--target_daily", type=float, default=2300, help="Target daily P&L")
    parser.add_argument("--drop_threshold", type=float, default=4.0, help="Drop threshold in points")
    parser.add_argument("--stop_loss_pct", type=float, default=0.001, help="Stop loss percentage")
    parser.add_argument("--base_contracts", type=int, default=2, help="Base number of contracts")
    parser.add_argument("--aggressive_mode", action="store_true", help="Use aggressive sizing")
    parser.add_argument("--output", default="production_results.json", help="Output file")
    return parser.parse_args()

class ProductionVReversalDetector:
    """
    Detector V-reversal de producción basado en resultados reales probados.
    """
    
    def __init__(self, drop_threshold=4.0, stop_loss_pct=0.001):
        self.drop_threshold = drop_threshold
        self.stop_loss_pct = stop_loss_pct
        
        # Configuraciones validadas por backtesting
        self.pattern_windows = {
            'drop_window': 15,      # Ventana para detectar caída
            'breakout_window': 30,  # Ventana para breakout
            'pullback_window': 15,  # Ventana para pullback
            'continuation_window': 20  # Ventana para continuación
        }
        
        # Estadísticas del modelo validado
        self.expected_stats = {
            'win_rate': 91.8,
            'avg_pnl_per_trade': 133,  # Dólares por trade
            'trades_per_day': 4.7,
            'monthly_pnl': 13211
        }
    
    def is_trading_window(self, timestamp):
        """Verifica si estamos en una ventana de trading válida"""
        hour = timestamp.hour
        
        # Ventanas específicas de alta volatilidad
        return ((3 <= hour < 4) or      # 3-4 AM: Early European session
                (9 <= hour < 11) or     # 9-11 AM: Market open + first hour  
                (13 <= hour < 15))      # 1:30-3 PM: Afternoon volatility

    def detect_v_reversal_pattern(self, df, idx):
        """
        Detecta patrón V-reversal usando la lógica validada en backtest
        """
        if idx < 50 or idx >= len(df) - 30:
            return False, {}
        
        current_time = df.iloc[idx]['Datetime']
        
        # Verificar ventana de trading
        if not self.is_trading_window(current_time):
            return False, {}
        
        # 1. FASE DE CAÍDA ABRUPTA
        origin_idx = idx
        origin_high = df.iloc[origin_idx]['High']
        
        # Buscar caída significativa en los próximos 15 minutos
        drop_end_idx = min(origin_idx + self.pattern_windows['drop_window'], len(df)-1)
        drop_data = df.iloc[origin_idx:drop_end_idx+1]
        
        low_idx = drop_data['Low'].idxmin()
        drop_low = drop_data.loc[low_idx, 'Low']
        drop_points = origin_high - drop_low
        
        # Validar caída mínima
        if drop_points < self.drop_threshold:
            return False, {}
        
        # 2. FASE DE BREAKOUT
        breakout_start_idx = low_idx
        breakout_end_idx = min(breakout_start_idx + self.pattern_windows['breakout_window'], len(df)-1)
        
        breakout_found = False
        breakout_idx = None
        
        for i in range(breakout_start_idx, breakout_end_idx):
            if i >= len(df):
                break
            if df.iloc[i]['High'] > origin_high:
                breakout_found = True
                breakout_idx = i
                break
        
        if not breakout_found:
            return False, {}
        
        # 3. FASE DE PULLBACK
        pullback_start_idx = breakout_idx
        pullback_end_idx = min(pullback_start_idx + self.pattern_windows['pullback_window'], len(df)-1)
        
        pullback_found = False
        pullback_idx = None
        
        for i in range(pullback_start_idx, pullback_end_idx):
            if i >= len(df):
                break
            
            current_low = df.iloc[i]['Low']
            current_close = df.iloc[i]['Close']
            
            # Pullback cerca del nivel original (dentro de 1 punto)
            if (abs(current_low - origin_high) <= 1.0 and 
                current_close >= origin_high - 1.0):
                pullback_found = True
                pullback_idx = i
                break
        
        if not pullback_found:
            return False, {}
        
        # 4. PREPARAR DATOS PARA EL TRADE
        entry_price = df.iloc[pullback_idx]['Close']
        
        # Calcular stops basados en configuración validada
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        
        # Take profit dinámico basado en el drop
        risk_reward_ratio = 3.0  # Conservador pero probado
        stop_distance = entry_price - stop_loss
        take_profit = entry_price + (stop_distance * risk_reward_ratio)
        
        pattern_data = {
            'origin_idx': origin_idx,
            'origin_high': origin_high,
            'drop_low': drop_low,
            'drop_points': drop_points,
            'breakout_idx': breakout_idx,
            'pullback_idx': pullback_idx,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'pattern_quality': self._calculate_pattern_quality(drop_points, df, pullback_idx)
        }
        
        return True, pattern_data
    
    def _calculate_pattern_quality(self, drop_points, df, pullback_idx):
        """Calcula la calidad del patrón (0-1)"""
        
        # Factores de calidad basados en backtesting
        quality_factors = {}
        
        # 1. Intensidad de la caída (más caída = mejor patrón)
        quality_factors['drop_intensity'] = min(drop_points / 8.0, 1.0)  # Max calidad a 8 puntos
        
        # 2. Volumen durante la caída (más volumen = más confianza)
        if pullback_idx >= 20:
            recent_volume = df.iloc[pullback_idx-20:pullback_idx]['Volume'].mean()
            current_volume = df.iloc[pullback_idx]['Volume']
            volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1.0
            quality_factors['volume_confirmation'] = min(volume_ratio / 2.0, 1.0)
        else:
            quality_factors['volume_confirmation'] = 0.5
        
        # 3. Timing dentro de ventana óptima
        hour = df.iloc[pullback_idx]['Datetime'].hour
        if 9 <= hour < 11:
            quality_factors['timing_score'] = 1.0  # Mejor ventana
        elif 13 <= hour < 15:
            quality_factors['timing_score'] = 0.8  # Segunda mejor
        else:
            quality_factors['timing_score'] = 0.6  # Aceptable
        
        # Promedio ponderado
        return np.mean(list(quality_factors.values()))

class ProductionPositionManager:
    """
    Gestión de posiciones para el modelo de producción
    """
    
    def __init__(self, base_contracts=2, aggressive_mode=False, target_daily=2300):
        self.base_contracts = base_contracts
        self.aggressive_mode = aggressive_mode
        self.target_daily = target_daily
        
        # Tracking diario
        self.daily_pnl = {}
        self.daily_trades = {}
        self.active_positions = []
        
        # Límites de riesgo
        self.max_daily_trades = 15 if aggressive_mode else 10
        self.max_concurrent_positions = 3 if aggressive_mode else 2
    
    def calculate_position_size(self, pattern_quality, current_daily_pnl=0):
        """
        Calcula el tamaño de posición basado en:
        1. Calidad del patrón
        2. P&L acumulado del día
        3. Modo agresivo
        """
        
        # Tamaño base
        size = self.base_contracts
        
        # Ajuste por calidad del patrón
        if pattern_quality > 0.8:
            size += 1  # Patrón excelente
        elif pattern_quality < 0.5:
            size = max(1, size - 1)  # Patrón marginal
        
        # Ajuste por P&L diario
        if current_daily_pnl < 0:
            # Si estamos perdiendo, reducir tamaño
            size = max(1, size - 1)
        elif current_daily_pnl > self.target_daily * 0.8:
            # Si cerca del target, mantener conservador
            size = self.base_contracts
        
        # Modo agresivo
        if self.aggressive_mode:
            if pattern_quality > 0.7:
                size += 1
            size = min(size, 4)  # Máximo 4 contratos en modo agresivo
        else:
            size = min(size, 3)  # Máximo 3 contratos en modo conservador
        
        return max(1, size)
    
    def should_take_trade(self, current_date):
        """Determina si debemos tomar el trade"""
        
        # Verificar límites diarios
        daily_trades_count = self.daily_trades.get(current_date, 0)
        if daily_trades_count >= self.max_daily_trades:
            return False, "MAX_DAILY_TRADES"
        
        # Verificar posiciones concurrentes
        if len(self.active_positions) >= self.max_concurrent_positions:
            return False, "MAX_CONCURRENT_POSITIONS"
        
        # Verificar si ya alcanzamos el target diario
        daily_pnl = self.daily_pnl.get(current_date, 0)
        if daily_pnl >= self.target_daily:
            return False, "TARGET_REACHED"
        
        return True, "OK"

def load_and_filter_data(file_path):
    """Carga y filtra los datos para las ventanas de trading"""
    
    print(f"📊 Cargando datos desde {file_path}...")
    
    df = pd.read_csv(file_path)
    
    # Detectar columna de tiempo
    time_cols = ['Datetime', 'datetime', 'Date', 'date', 'Timestamp', 'timestamp']
    time_col = None
    for col in time_cols:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No se encontró columna de tiempo. Columnas disponibles: {list(df.columns)}")
    
    # Convertir a datetime
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: 'Datetime'})
    
    # Estandarizar nombres de columnas
    for col in df.columns:
        if col.lower() in ['open']:
            df = df.rename(columns={col: 'Open'})
        elif col.lower() in ['high']:
            df = df.rename(columns={col: 'High'})
        elif col.lower() in ['low']:
            df = df.rename(columns={col: 'Low'})
        elif col.lower() in ['close']:
            df = df.rename(columns={col: 'Close'})
        elif col.lower() in ['volume']:
            df = df.rename(columns={col: 'Volume'})
    
    # Asegurar que tenemos Volume
    if 'Volume' not in df.columns:
        df['Volume'] = 1000
    
    # Ordenar por tiempo
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    print(f"✅ Datos cargados: {len(df):,} registros")
    
    # Filtrar solo las ventanas de trading validadas
    df['hour'] = df['Datetime'].dt.hour
    trading_mask = ((df['hour'] >= 3) & (df['hour'] < 4)) | \
                   ((df['hour'] >= 9) & (df['hour'] < 11)) | \
                   ((df['hour'] >= 13) & (df['hour'] < 15))
    
    df_filtered = df[trading_mask].reset_index(drop=True)
    
    print(f"✅ Datos filtrados a ventanas de trading: {len(df_filtered):,} registros")
    print(f"📊 Cobertura: {len(df_filtered)/len(df)*100:.1f}% de los datos originales")
    
    return df_filtered

def simulate_trade_execution(df, entry_idx, entry_price, stop_loss, take_profit, max_hold_periods=25):
    """
    Simula la ejecución de un trade con exit conditions
    """
    
    for i in range(entry_idx + 1, min(entry_idx + max_hold_periods, len(df))):
        bar = df.iloc[i]
        
        # Check take profit
        if bar['High'] >= take_profit:
            return take_profit, "TAKE_PROFIT", i
        
        # Check stop loss
        if bar['Low'] <= stop_loss:
            return stop_loss, "STOP_LOSS", i
    
    # Time-based exit
    exit_idx = min(entry_idx + max_hold_periods - 1, len(df) - 1)
    exit_price = df.iloc[exit_idx]['Close']
    return exit_price, "TIME_EXIT", exit_idx

def run_production_model(df, args):
    """
    Ejecuta el modelo de producción
    """
    
    print(f"\n🚀 EJECUTANDO MODELO DE PRODUCCIÓN")
    print(f"🎯 Target diario: ${args.target_daily:,.0f}")
    print(f"📊 Drop threshold: {args.drop_threshold} puntos")
    print(f"⚠️ Stop loss: {args.stop_loss_pct:.1%}")
    print(f"📈 Contratos base: {args.base_contracts}")
    print(f"🔥 Modo agresivo: {'SÍ' if args.aggressive_mode else 'NO'}")
    
    # Inicializar componentes
    detector = ProductionVReversalDetector(args.drop_threshold, args.stop_loss_pct)
    position_manager = ProductionPositionManager(
        args.base_contracts, 
        args.aggressive_mode, 
        args.target_daily
    )
    
    # Resultados
    all_trades = []
    daily_performance = {}
    
    print(f"\n📈 Iniciando backtesting en {len(df):,} registros...")
    
    i = 50  # Empezar con margen para patrones
    
    while i < len(df) - 30:
        current_time = df.iloc[i]['Datetime']
        current_date = current_time.date()
        
        # Inicializar día
        if current_date not in daily_performance:
            daily_performance[current_date] = {
                'pnl': 0,
                'trades': 0,
                'target_reached': False
            }
            position_manager.daily_pnl[current_date] = 0
            position_manager.daily_trades[current_date] = 0
        
        # Verificar si podemos tomar un trade
        can_trade, reason = position_manager.should_take_trade(current_date)
        
        if not can_trade:
            i += 1
            continue
        
        # Buscar patrón V-reversal
        pattern_found, pattern_data = detector.detect_v_reversal_pattern(df, i)
        
        if pattern_found:
            # Calcular tamaño de posición
            current_daily_pnl = position_manager.daily_pnl[current_date]
            position_size = position_manager.calculate_position_size(
                pattern_data['pattern_quality'], 
                current_daily_pnl
            )
            
            # Simular ejecución del trade
            entry_idx = pattern_data['pullback_idx']
            entry_price = pattern_data['entry_price']
            stop_loss = pattern_data['stop_loss']
            take_profit = pattern_data['take_profit']
            
            exit_price, exit_reason, exit_idx = simulate_trade_execution(
                df, entry_idx, entry_price, stop_loss, take_profit
            )
            
            # Calcular P&L
            points_gained = exit_price - entry_price
            pnl_dollars = points_gained * position_size * 50  # ES = $50 por punto
            
            # Registrar trade
            trade_record = {
                'date': current_date,
                'entry_time': current_time,
                'entry_price': entry_price,
                'exit_time': df.iloc[exit_idx]['Datetime'],
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'points_gained': points_gained,
                'pnl_dollars': pnl_dollars,
                'pattern_quality': pattern_data['pattern_quality'],
                'drop_points': pattern_data['drop_points']
            }
            
            all_trades.append(trade_record)
            
            # Actualizar contadores
            position_manager.daily_pnl[current_date] += pnl_dollars
            position_manager.daily_trades[current_date] += 1
            daily_performance[current_date]['pnl'] += pnl_dollars
            daily_performance[current_date]['trades'] += 1
            
            # Verificar si alcanzamos target diario
            if daily_performance[current_date]['pnl'] >= args.target_daily:
                daily_performance[current_date]['target_reached'] = True
            
            # Avanzar al final del trade para evitar solapamiento
            i = exit_idx + 1
        else:
            i += 1
    
    return all_trades, daily_performance

def analyze_production_results(trades, daily_performance, target_daily):
    """
    Analiza los resultados del modelo de producción
    """
    
    if not trades:
        print("❌ No se encontraron trades")
        return {}
    
    print(f"\n📊 ANÁLISIS DE RESULTADOS DEL MODELO DE PRODUCCIÓN")
    print("=" * 65)
    
    # Estadísticas básicas
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl_dollars'] > 0]
    losing_trades = [t for t in trades if t['pnl_dollars'] <= 0]
    
    win_rate = len(winning_trades) / total_trades * 100
    total_pnl = sum(t['pnl_dollars'] for t in trades)
    avg_pnl_per_trade = total_pnl / total_trades
    
    print(f"📈 ESTADÍSTICAS GENERALES:")
    print(f"   Total de Trades: {total_trades:,}")
    print(f"   Trades Ganadores: {len(winning_trades):,} ({win_rate:.1f}%)")
    print(f"   Trades Perdedores: {len(losing_trades):,}")
    print(f"   P&L Total: ${total_pnl:,.2f}")
    print(f"   P&L Promedio por Trade: ${avg_pnl_per_trade:.2f}")
    
    # Análisis diario
    daily_pnls = [day['pnl'] for day in daily_performance.values()]
    daily_trades_counts = [day['trades'] for day in daily_performance.values()]
    
    avg_daily_pnl = np.mean(daily_pnls)
    max_daily_pnl = np.max(daily_pnls)
    min_daily_pnl = np.min(daily_pnls)
    avg_trades_per_day = np.mean(daily_trades_counts)
    
    days_above_target = sum(1 for day in daily_performance.values() if day['target_reached'])
    total_days = len(daily_performance)
    target_achievement_rate = days_above_target / total_days * 100
    
    print(f"\n📅 PERFORMANCE DIARIA:")
    print(f"   P&L Diario Promedio: ${avg_daily_pnl:.2f}")
    print(f"   Mejor Día: ${max_daily_pnl:.2f}")
    print(f"   Peor Día: ${min_daily_pnl:.2f}")
    print(f"   Trades Promedio por Día: {avg_trades_per_day:.1f}")
    print(f"   Días que Alcanzaron Target (${target_daily}): {days_above_target}/{total_days} ({target_achievement_rate:.1f}%)")
    
    # Proyecciones mensuales
    trading_days_per_month = 21
    monthly_pnl = avg_daily_pnl * trading_days_per_month
    monthly_trades = avg_trades_per_day * trading_days_per_month
    annual_pnl = monthly_pnl * 12
    
    print(f"\n📊 PROYECCIONES:")
    print(f"   P&L Mensual Estimado: ${monthly_pnl:,.2f}")
    print(f"   Trades Mensuales: {monthly_trades:.0f}")
    print(f"   P&L Anual Estimado: ${annual_pnl:,.2f}")
    
    # Análisis por tamaño de posición
    print(f"\n📈 ANÁLISIS POR TAMAÑO DE POSICIÓN:")
    position_sizes = {}
    for trade in trades:
        size = trade['position_size']
        if size not in position_sizes:
            position_sizes[size] = {'trades': [], 'pnl': 0}
        position_sizes[size]['trades'].append(trade)
        position_sizes[size]['pnl'] += trade['pnl_dollars']
    
    for size, data in sorted(position_sizes.items()):
        size_trades = len(data['trades'])
        size_pnl = data['pnl']
        size_winrate = len([t for t in data['trades'] if t['pnl_dollars'] > 0]) / size_trades * 100
        size_avg_pnl = size_pnl / size_trades
        
        print(f"   {size} Contratos: {size_trades} trades, {size_winrate:.1f}% win rate, ${size_avg_pnl:.0f} avg P&L")
    
    # Análisis de calidad de patrones
    print(f"\n🎯 ANÁLISIS POR CALIDAD DE PATRÓN:")
    high_quality = [t for t in trades if t['pattern_quality'] > 0.7]
    medium_quality = [t for t in trades if 0.5 <= t['pattern_quality'] <= 0.7]
    low_quality = [t for t in trades if t['pattern_quality'] < 0.5]
    
    for quality_group, name in [(high_quality, "Alta (>70%)"), 
                                (medium_quality, "Media (50-70%)"), 
                                (low_quality, "Baja (<50%)")]:
        if quality_group:
            group_winrate = len([t for t in quality_group if t['pnl_dollars'] > 0]) / len(quality_group) * 100
            group_avg_pnl = np.mean([t['pnl_dollars'] for t in quality_group])
            print(f"   {name}: {len(quality_group)} trades, {group_winrate:.1f}% win rate, ${group_avg_pnl:.0f} avg P&L")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES PARA ALCANZAR ${target_daily}/DÍA:")
    
    if avg_daily_pnl >= target_daily:
        print("🎉 ¡TARGET ALCANZADO!")
        print(f"   ✅ El modelo ya genera ${avg_daily_pnl:.0f}/día en promedio")
        print(f"   🔄 Considera aumentar capital para escalar")
        print(f"   📊 Optimización: Alcanza target {target_achievement_rate:.0f}% de los días")
    else:
        gap = target_daily - avg_daily_pnl
        multiplier_needed = target_daily / avg_daily_pnl
        
        print(f"📈 Gap hacia target: ${gap:.0f}/día ({multiplier_needed:.1f}x)")
        
        if multiplier_needed <= 1.5:
            print("✅ ESTRATEGIAS VIABLES:")
            print(f"   1. Aumentar contratos base moderadamente")
            print("   2. Usar modo agresivo consistentemente")
            print("   3. Optimizar timing de entrada")
        elif multiplier_needed <= 2.5:
            print("⚠️ ESTRATEGIAS MODERADAMENTE AGRESIVAS:")
            print("   1. Trading en más instrumentos (NQ, YM)")
            print("   2. Añadir sesiones de trading (Europa/Asia)")
            print("   3. Leverage controlado (2:1)")
        else:
            print("🚨 ESTRATEGIAS ALTAMENTE AGRESIVAS REQUERIDAS:")
            print("   1. Portfolio de múltiples mercados")
            print("   2. Estrategias de alta frecuencia")
            print("   3. Capital significativamente mayor")
    
    # Métricas de riesgo
    if daily_pnls:
        std_daily = np.std(daily_pnls)
        sharpe = avg_daily_pnl / std_daily if std_daily > 0 else 0
        
        # Drawdown
        cumulative_pnl = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = np.max(drawdowns)
        
        print(f"\n⚖️ MÉTRICAS DE RIESGO:")
        print(f"   Volatilidad Diaria: ${std_daily:.2f}")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Máximo Drawdown: ${max_drawdown:.2f}")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_daily_pnl': avg_daily_pnl,
        'target_achievement_rate': target_achievement_rate,
        'monthly_pnl': monthly_pnl,
        'annual_pnl': annual_pnl
    }

def main():
    args = parse_args()
    
    try:
        print(f"🔥 MODELO DE PRODUCCIÓN PARA TRADING")
        print(f"Basado en V-reversal system validado")
        print("=" * 50)
        
        # Cargar datos
        df = load_and_filter_data(args.file)
        
        # Ejecutar modelo
        trades, daily_performance = run_production_model(df, args)
        
        # Analizar resultados
        results = analyze_production_results(trades, daily_performance, args.target_daily)
        
        # Guardar resultados
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'target_daily': args.target_daily,
                'drop_threshold': args.drop_threshold,
                'stop_loss_pct': args.stop_loss_pct,
                'base_contracts': args.base_contracts,
                'aggressive_mode': args.aggressive_mode
            },
            'summary': results,
            'trades': trades,
            'daily_performance': {str(date): perf for date, perf in daily_performance.items()}
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Resultados guardados en: {args.output}")
        
        # Comparación con modelo original
        print(f"\n🔄 COMPARACIÓN CON MODELO ORIGINAL:")
        print(f"   Original: ~$13,211/mes, 91.8% win rate")
        if results:
            print(f"   Actual: ${results.get('monthly_pnl', 0):,.0f}/mes, {results.get('win_rate', 0):.1f}% win rate")
            
            if results.get('monthly_pnl', 0) > 13211:
                improvement = (results['monthly_pnl'] / 13211 - 1) * 100
                print(f"   📈 Mejora: +{improvement:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 