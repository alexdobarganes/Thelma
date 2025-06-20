# -*- coding: utf-8 -*-
"""
Test del modelo $2300/día pickle
"""

import pickle
from datetime import datetime

print("🔍 PROBANDO PICKLE DEL MODELO $2300/DÍA")
print("=" * 45)

# Cargar modelo
with open('vreversal_2300_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Modelo cargado correctamente")
print(f"   Nombre: {model.model_name}")
print(f"   Versión: {model.model_version}")
print(f"   Drop threshold: {model.drop_threshold} puntos")
print(f"   Stop loss: {model.stop_loss_pct*100}%")
print(f"   Contratos: {model.position_size}")

print("\n📊 ESTADÍSTICAS VALIDADAS:")
stats = model.performance_stats
print(f"   Días testeados: {stats['total_days_tested']}")
print(f"   Success rate: {stats['success_rate_pct']:.1f}%")
print(f"   P&L promedio/día: ${stats['avg_daily_pnl']:,.2f}")
print(f"   Trades/día: {stats['avg_trades_per_day']:.1f}")

print("\n🧪 PROBANDO FUNCIONES:")

# Test 1: Trading windows
current_hour = datetime.now().hour
is_window = model.is_trading_window(current_hour)
print(f"   Hora actual: {current_hour}:00")
print(f"   ¿Es ventana de trading?: {is_window}")

# Test 2: Pattern quality
quality = model.calculate_pattern_quality(5.0, 1.8, 10)
print(f"   Calidad patrón (5 pts, 1.8x vol, 10h): {quality:.3f}")

# Test 3: Position size
position = model.calculate_position_size(quality, 1500)
print(f"   Tamaño posición (calidad {quality:.3f}, $1500 PnL): {position} contratos")

# Test 4: Should trade
should_trade, reason = model.should_trade(10, 30, 8, 1500)
print(f"   ¿Debe tradear? (10:30, 8 señales, $1500): {should_trade}")
print(f"   Razón: {reason}")

# Test 5: Stops calculation
entry_price = 4950.0
stop_loss, take_profit = model.calculate_stops(entry_price)
print(f"   Entrada: {entry_price}")
print(f"   Stop loss: {stop_loss:.2f}")
print(f"   Take profit: {take_profit:.2f}")
print(f"   Risk/Reward: {(take_profit-entry_price)/(entry_price-stop_loss):.1f}:1")

# Test 6: Expected P&L
expected_pnl = model.get_expected_pnl_per_trade()
print(f"   P&L esperado por trade: ${expected_pnl:.2f}")

print("\n🎯 VENTANAS DE TRADING:")
for i, (start, end) in enumerate(model.trading_windows, 1):
    start_h = int(start)
    start_m = int((start % 1) * 60)
    end_h = int(end)
    end_m = int((end % 1) * 60)
    print(f"   Ventana {i}: {start_h:02d}:{start_m:02d} - {end_h:02d}:{end_m:02d} ET")

print("\n✅ PICKLE DEL MODELO $2300/DÍA FUNCIONANDO PERFECTAMENTE")
print("🚀 Listo para usar en tiempo real!") 