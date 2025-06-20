#!/usr/bin/env python3
"""
Create $2300/Day Model Pickle
=============================

Extrae los parámetros exactos del modelo que genera $2300/día
y crea un pickle listo para usar en tiempo real.

Basado en:
- production_ready_model.py 
- production_report_final.json (98.2% success rate)
- 436 días de testing con $2,370 promedio diario
"""

import pickle
import json
from datetime import datetime
from pathlib import Path

class VReversal2300Model:
    """
    Modelo V-reversal exacto que genera $2300/día
    
    Parámetros validados en 436 días de testing:
    - Promedio diario: $2,370
    - Success rate: 98.2% (428/436 días)
    - Total P&L: $1,033,571.76
    - Trades promedio: 5.9/día
    """
    
    def __init__(self):
        # PARÁMETROS CORE DEL MODELO $2300/DÍA
        self.model_name = "VReversal_2300_Production"
        self.model_version = "1.0"
        self.creation_date = datetime.now().isoformat()
        
        # PARÁMETROS DE DETECCIÓN DE PATRONES (VALIDADOS)
        self.drop_threshold = 4.0           # 4.0 puntos mínimos de caída
        self.drop_window = 15               # 15 minutos para completar la caída
        self.breakout_window = 30           # 30 minutos para el breakout
        self.pullback_window = 15           # 15 minutos para el pullback
        self.continuation_window = 20       # 20 minutos para continuación
        self.pullback_tolerance = 1.0       # 1 punto de tolerancia en pullback
        
        # GESTIÓN DE RIESGO (CRÍTICO)
        self.stop_loss_pct = 0.001          # 0.1% stop loss (PRODUCCIÓN PROBADA)
        self.max_hold_time = 25             # 25 minutos máximo de retención
        self.position_size = 3              # 3 contratos (configuración $2300/día)
        self.tick_value = 50.0              # $50 por punto ES futures
        
        # GESTIÓN DE TRADING
        self.max_daily_signals = 20         # Máximo 20 señales por día
        self.min_pattern_confidence = 0.75  # 75% confianza mínima
        
        # VENTANAS DE TRADING ESPECÍFICAS (CRÍTICAS)
        self.trading_windows = [
            (3, 4),    # 3:00-4:00 AM ET (Sesión europea temprana)
            (9, 11),   # 9:00-11:00 AM ET (Apertura + primera hora)
            (13.5, 15) # 1:30-3:00 PM ET (Volatilidad vespertina)
        ]
        
        # ESTADÍSTICAS VALIDADAS (DATOS REALES)
        self.performance_stats = {
            'total_days_tested': 436,
            'profitable_days': 428,
            'success_rate_pct': 98.17,
            'avg_daily_pnl': 2370.58,
            'total_pnl': 1033571.76,
            'avg_trades_per_day': 5.91,
            'win_rate_pct': 91.8,
            'monthly_projection': 49782.13,
            'annual_projection': 597385.51
        }
        
        # CONFIGURACIÓN DE TAKE PROFIT DINÁMICO
        self.risk_reward_ratio = 3.0        # 3:1 riesgo/recompensa
        self.take_profit_base = 3.0         # 3 puntos base de take profit
        
        # FACTORES DE CALIDAD DE PATRÓN
        self.quality_factors = {
            'drop_intensity_max': 8.0,      # 8 puntos = calidad máxima
            'volume_ratio_threshold': 2.0,   # 2x volumen promedio
            'timing_weights': {
                'morning': 1.0,              # 9-11 AM = peso máximo
                'afternoon': 0.8,            # 1:30-3 PM = peso alto
                'early': 0.6                 # 3-4 AM = peso medio
            }
        }
    
    def is_trading_window(self, hour, minute=0):
        """
        Verifica si estamos en una ventana de trading válida
        
        Args:
            hour: Hora en Eastern Time (0-23)
            minute: Minuto (0-59)
            
        Returns:
            bool: True si está en ventana de trading
        """
        current_time = hour + (minute / 60.0)
        
        for start, end in self.trading_windows:
            if start <= current_time < end:
                return True
        return False
    
    def calculate_pattern_quality(self, drop_points, volume_ratio=1.0, hour=10):
        """
        Calcula la calidad del patrón (0-1)
        
        Args:
            drop_points: Puntos de caída detectados
            volume_ratio: Ratio de volumen vs promedio
            hour: Hora del patrón (ET)
            
        Returns:
            float: Calidad del patrón (0.0-1.0)
        """
        quality_factors = {}
        
        # Factor 1: Intensidad de caída
        quality_factors['drop_intensity'] = min(
            drop_points / self.quality_factors['drop_intensity_max'], 1.0
        )
        
        # Factor 2: Confirmación de volumen
        quality_factors['volume_confirmation'] = min(
            volume_ratio / self.quality_factors['volume_ratio_threshold'], 1.0
        )
        
        # Factor 3: Timing dentro de ventana
        if 9 <= hour < 11:
            quality_factors['timing_score'] = self.quality_factors['timing_weights']['morning']
        elif 13 <= hour < 15:
            quality_factors['timing_score'] = self.quality_factors['timing_weights']['afternoon']
        else:
            quality_factors['timing_score'] = self.quality_factors['timing_weights']['early']
        
        return sum(quality_factors.values()) / len(quality_factors)
    
    def calculate_position_size(self, pattern_quality, current_daily_pnl=0):
        """
        Calcula el tamaño de posición basado en calidad y P&L diario
        
        Args:
            pattern_quality: Calidad del patrón (0-1)
            current_daily_pnl: P&L actual del día
            
        Returns:
            int: Número de contratos
        """
        base_size = self.position_size
        
        # Ajuste por calidad de patrón
        if pattern_quality > 0.8:
            size_multiplier = 1.2  # +20% para patrones excelentes
        elif pattern_quality > 0.6:
            size_multiplier = 1.0  # Tamaño base para patrones buenos
        else:
            size_multiplier = 0.8  # -20% para patrones mediocres
        
        # Ajuste por P&L diario (gestión de riesgo)
        target_progress = current_daily_pnl / 2300.0 if current_daily_pnl > 0 else 0
        
        if target_progress > 0.8:  # Ya cerca del target
            size_multiplier *= 0.7  # Reducir riesgo
        elif target_progress < 0.2:  # Lejos del target
            size_multiplier *= 1.1  # Aumentar agresividad moderadamente
        
        final_size = int(base_size * size_multiplier)
        return max(1, min(final_size, 5))  # Entre 1 y 5 contratos
    
    def calculate_stops(self, entry_price):
        """
        Calcula stop loss y take profit
        
        Args:
            entry_price: Precio de entrada
            
        Returns:
            tuple: (stop_loss, take_profit)
        """
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        
        # Take profit dinámico
        stop_distance = entry_price - stop_loss
        take_profit = entry_price + (stop_distance * self.risk_reward_ratio)
        
        return stop_loss, take_profit
    
    def should_trade(self, hour, minute, daily_signals_count, daily_pnl):
        """
        Determina si se debe tomar un trade
        
        Args:
            hour: Hora actual (ET)
            minute: Minuto actual
            daily_signals_count: Señales ya generadas hoy
            daily_pnl: P&L acumulado del día
            
        Returns:
            tuple: (should_trade: bool, reason: str)
        """
        # Verificar ventana de trading
        if not self.is_trading_window(hour, minute):
            return False, "Outside trading windows"
        
        # Verificar límite diario de señales
        if daily_signals_count >= self.max_daily_signals:
            return False, "Daily signal limit reached"
        
        # Verificar si ya se alcanzó el target (opcional)
        if daily_pnl >= 2300:
            return False, "Daily target already reached"
        
        return True, "All checks passed"
    
    def get_expected_pnl_per_trade(self):
        """
        Calcula P&L esperado por trade basado en estadísticas validadas
        
        Returns:
            float: P&L esperado por trade en dólares
        """
        return self.performance_stats['avg_daily_pnl'] / self.performance_stats['avg_trades_per_day']
    
    def get_model_summary(self):
        """
        Resumen completo del modelo
        
        Returns:
            dict: Resumen del modelo
        """
        return {
            'model_info': {
                'name': self.model_name,
                'version': self.model_version,
                'creation_date': self.creation_date,
                'description': 'Modelo V-reversal que genera $2300/día con 98.2% success rate'
            },
            'pattern_detection': {
                'drop_threshold': self.drop_threshold,
                'drop_window': self.drop_window,
                'breakout_window': self.breakout_window,
                'pullback_window': self.pullback_window,
                'pullback_tolerance': self.pullback_tolerance
            },
            'risk_management': {
                'stop_loss_pct': self.stop_loss_pct,
                'max_hold_time': self.max_hold_time,
                'position_size': self.position_size,
                'risk_reward_ratio': self.risk_reward_ratio
            },
            'trading_schedule': {
                'windows': self.trading_windows,
                'max_daily_signals': self.max_daily_signals,
                'min_confidence': self.min_pattern_confidence
            },
            'validated_performance': self.performance_stats
        }

def create_pickle():
    """Crear pickle del modelo $2300/día"""
    
    print("🎯 CREANDO PICKLE DEL MODELO $2300/DÍA")
    print("=" * 50)
    
    # Crear instancia del modelo
    model = VReversal2300Model()
    
    # Mostrar resumen
    summary = model.get_model_summary()
    
    print("📊 CONFIGURACIÓN DEL MODELO:")
    print(f"   Nombre: {summary['model_info']['name']}")
    print(f"   Versión: {summary['model_info']['version']}")
    print(f"   Drop Threshold: {summary['pattern_detection']['drop_threshold']} puntos")
    print(f"   Stop Loss: {summary['risk_management']['stop_loss_pct']*100}%")
    print(f"   Contratos: {summary['risk_management']['position_size']}")
    print(f"   Max Hold: {summary['risk_management']['max_hold_time']} min")
    
    print(f"\n⏰ VENTANAS DE TRADING:")
    for i, (start, end) in enumerate(summary['trading_schedule']['windows'], 1):
        start_hour = int(start)
        start_min = int((start % 1) * 60)
        end_hour = int(end)
        end_min = int((end % 1) * 60)
        print(f"   Ventana {i}: {start_hour:02d}:{start_min:02d} - {end_hour:02d}:{end_min:02d} ET")
    
    print(f"\n📈 PERFORMANCE VALIDADA:")
    perf = summary['validated_performance']
    print(f"   Días testeados: {perf['total_days_tested']}")
    print(f"   Días rentables: {perf['profitable_days']} ({perf['success_rate_pct']:.1f}%)")
    print(f"   P&L promedio/día: ${perf['avg_daily_pnl']:,.2f}")
    print(f"   Trades/día: {perf['avg_trades_per_day']:.1f}")
    print(f"   Proyección mensual: ${perf['monthly_projection']:,.2f}")
    
    # Crear directorio models/production si no existe
    models_dir = Path("models/production/current")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar pickle
    pickle_path = models_dir / "vreversal_2300_model.pkl"
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n💾 PICKLE CREADO: {pickle_path}")
    print(f"   Tamaño: {pickle_path.stat().st_size / 1024:.1f} KB")
    
    # Crear también archivo JSON de configuración
    json_path = models_dir / "vreversal_2300_config.json"
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 CONFIG JSON: {json_path}")
    
    # Verificar carga del pickle
    print(f"\n🔍 VERIFICANDO PICKLE...")
    
    with open(pickle_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Verificar funcionalidad
    test_quality = loaded_model.calculate_pattern_quality(5.0, 1.5, 10)
    test_position = loaded_model.calculate_position_size(test_quality, 1000)
    test_should_trade = loaded_model.should_trade(10, 30, 5, 1500)
    
    print(f"   ✅ Pickle cargado correctamente")
    print(f"   ✅ Calidad de patrón (test): {test_quality:.3f}")
    print(f"   ✅ Tamaño posición (test): {test_position} contratos")
    print(f"   ✅ Should trade (test): {test_should_trade[0]} - {test_should_trade[1]}")
    
    # Crear script de ejemplo de uso
    usage_script = models_dir / "usage_example.py"
    
    with open(usage_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Ejemplo de uso del modelo $2300/día
"""

import pickle
from datetime import datetime

# Cargar modelo
with open('vreversal_2300_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Ejemplo de uso
current_hour = datetime.now().hour
pattern_quality = 0.85  # 85% de calidad del patrón
daily_pnl = 1200       # $1200 acumulados hoy
daily_signals = 8      # 8 señales ya generadas

# Verificar si se debe tradear
should_trade, reason = model.should_trade(current_hour, 0, daily_signals, daily_pnl)
print(f"Should trade: {should_trade} - {reason}")

# Calcular tamaño de posición
position_size = model.calculate_position_size(pattern_quality, daily_pnl)
print(f"Position size: {position_size} contracts")

# Calcular stops para precio de entrada 4950.0
entry_price = 4950.0
stop_loss, take_profit = model.calculate_stops(entry_price)
print(f"Entry: {entry_price}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")

# P&L esperado por trade
expected_pnl = model.get_expected_pnl_per_trade()
print(f"Expected P&L per trade: ${expected_pnl:.2f}")
''')
    
    print(f"📝 SCRIPT DE EJEMPLO: {usage_script}")
    
    print(f"\n🎉 MODELO $2300/DÍA LISTO PARA USAR")
    print(f"🔥 Úsalo con: pickle.load(open('vreversal_2300_model.pkl', 'rb'))")
    
    return pickle_path, json_path

if __name__ == "__main__":
    create_pickle() 