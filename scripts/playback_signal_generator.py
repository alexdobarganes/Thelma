"""
playback_signal_generator.py
============================
Generador de señales V-Reversal para modo Playback de NinjaTrader.
Pre-genera todas las señales basadas en datos históricos para una fecha específica.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import argparse
from exact_vreversal_model import detect_exact_patterns, load_data

class PlaybackSignalGenerator:
    def __init__(self, output_dir="signals/playback"):
        self.output_dir = output_dir
        
        # Asegurar que existe el directorio
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_signals_for_date(self, df, target_date, position_size=1):
        """Generar todas las señales para una fecha específica"""
        
        print(f"🎮 GENERANDO SEÑALES PARA PLAYBACK")
        print(f"📅 Fecha objetivo: {target_date}")
        print("=" * 50)
        
        # Filtrar datos para la fecha objetivo
        target_date_str = target_date.strftime('%Y-%m-%d')
        day_data = df[df['Datetime'].dt.date == target_date.normalize().date()].copy()
        
        if day_data.empty:
            print(f"❌ No hay datos para {target_date_str}")
            return []
        
        print(f"📊 Datos encontrados: {len(day_data)} barras para {target_date_str}")
        
        # Obtener datos históricos hasta la fecha objetivo (para contexto)
        historical_cutoff = target_date - timedelta(days=30)  # 30 días de historia
        historical_data = df[df['Datetime'] <= target_date].copy()
        historical_data = historical_data[historical_data['Datetime'] >= historical_cutoff]
        
        print(f"📈 Datos históricos: {len(historical_data)} barras (últimos 30 días)")
        
        # Detectar patrones usando función exacta validada
        print("🔍 Detectando patrones V-Reversal...")
        successful_patterns, failed_patterns = detect_exact_patterns(historical_data, position_size=1)
        patterns = successful_patterns + failed_patterns
        
        # Filtrar solo patrones del día objetivo
        day_patterns = []
        for pattern in patterns:
            pattern_date = pd.to_datetime(pattern['origin_time']).normalize().date()
            if pattern_date == target_date.normalize().date():
                day_patterns.append(pattern)
        
        print(f"🎯 Patrones V-Reversal encontrados: {len(day_patterns)}")
        
        # Generar archivos de señales
        signals_generated = []
        
        for i, pattern in enumerate(day_patterns):
            signal_file = self.create_signal_file(pattern, i, position_size)
            if signal_file:
                signals_generated.append(signal_file)
        
        print(f"✅ Señales generadas: {len(signals_generated)}")
        
        # Crear resumen del día
        self.create_day_summary(target_date, day_patterns, signals_generated)
        
        return signals_generated
    
    def create_signal_file(self, pattern, index, position_size):
        """Crear archivo de señal individual"""
        
        try:
            # Calcular timestamp único basado en el patrón
            origin_time = pd.to_datetime(pattern['origin_time'])
            timestamp = int(origin_time.timestamp())
            
            # Generar ID único
            signal_id = f"{origin_time.strftime('%Y%m%d')}_{index:03d}"
                        # Calcular precios usando campos del patrón exacto
            entry_price = pattern['entry_price']
            stop_loss = entry_price * (1 - 0.001)  # 0.1% stop loss
            take_profit = entry_price + (3.0 * (entry_price - stop_loss))  # 3:1 R/R
            
            # Contenido del archivo de señal
            signal_content = f"""# V-Reversal Signal - Playback Mode
ACTION = BUY
ENTRY_PRICE = {entry_price:.2f}
STOP_LOSS = {stop_loss:.2f}
TAKE_PROFIT = {take_profit:.2f}
SIGNAL_ID = {signal_id}
TIMESTAMP = {timestamp}
CONFIDENCE = HIGH
PATTERN_TYPE = V_REVERSAL
ORIGIN_TIME = {pattern['origin_time']}
DROP_POINTS = {pattern['drop_points']:.2f}
BREAKOUT_TIME = {pattern['breakout_time']}
POSITION_SIZE = {position_size}
PLAYBACK_MODE = TRUE
"""
            
            # Nombre del archivo basado en tiempo de origen
            filename = f"vreversal_{signal_id}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            # Escribir archivo
            with open(filepath, 'w') as f:
                f.write(signal_content)
            
            print(f"📝 Señal creada: {filename}")
            print(f"   ⏰ Tiempo: {pattern['origin_time']}")
            print(f"   💰 Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
            print(f"   📉 Drop: {pattern['drop_points']:.2f} puntos")
            
            return {
                'filename': filename,
                'filepath': filepath,
                'signal_id': signal_id,
                'pattern': pattern,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            print(f"❌ Error creando señal {index}: {e}")
            return None
    
    def create_day_summary(self, target_date, patterns, signals):
        """Crear resumen del día para referencia"""
        
        summary = {
            'date': target_date.strftime('%Y-%m-%d'),
            'total_patterns': len(patterns),
            'signals_generated': len(signals),
            'expected_performance': {
                'win_rate': 91.8,
                'avg_points_per_trade': 2.0,
                'expected_daily_pnl_1_contract': len(signals) * 2.0 * 50  # ~$100 per point
            },
            'signals': signals,
            'generated_at': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.output_dir, f"day_summary_{target_date.strftime('%Y%m%d')}.json")
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📋 Resumen guardado: {summary_file}")
        
        # Mostrar resumen en consola
        print(f"\n📊 RESUMEN DEL DÍA {target_date.strftime('%Y-%m-%d')}")
        print("=" * 50)
        print(f"   Patrones detectados: {len(patterns)}")
        print(f"   Señales generadas: {len(signals)}")
        print(f"   P&L esperado (1 contrato): ${summary['expected_performance']['expected_daily_pnl_1_contract']:.0f}")
        print(f"   Win rate esperado: {summary['expected_performance']['win_rate']:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Generar señales V-Reversal para Playback')
    parser.add_argument('--file', required=True, help='Archivo de datos CSV')
    parser.add_argument('--date', required=True, help='Fecha para playback (YYYY-MM-DD)')
    parser.add_argument('--output', default='signals/playback', help='Carpeta de salida')
    parser.add_argument('--contracts', type=int, default=1, help='Número de contratos por señal')
    
    args = parser.parse_args()
    
    try:
        # Cargar datos
        print(f"📊 Cargando datos de: {args.file}")
        df = load_data(args.file)
        
        # Parsear fecha objetivo con timezone
        target_date = pd.to_datetime(args.date).tz_localize('UTC')
        
        # Crear generador
        generator = PlaybackSignalGenerator(args.output)
        
        # Generar señales
        signals = generator.generate_signals_for_date(df, target_date, args.contracts)
        
        if signals:
            print(f"\n🎮 LISTO PARA PLAYBACK")
            print("=" * 50)
            print(f"📁 Señales en: {args.output}")
            print(f"📅 Fecha: {args.date}")
            print(f"🔢 Total señales: {len(signals)}")
            print(f"\n📋 INSTRUCCIONES PARA NT:")
            print(f"1. Configurar Playback para {args.date}")
            print(f"2. Establecer VReversalAutoTrader signal path: {os.path.abspath(args.output)}")
            print(f"3. Iniciar Playback a velocidad deseada")
            print(f"4. Monitorear ejecución de {len(signals)} señales esperadas")
        else:
            print(f"⚠️ No se generaron señales para {args.date}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 