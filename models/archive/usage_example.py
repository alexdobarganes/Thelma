#!/usr/bin/env python3
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
