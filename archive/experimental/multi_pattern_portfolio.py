"""
multi_pattern_portfolio.py
==========================
Sistema de portfolio con múltiples patrones de trading para maximizar P&L diario:
1. V-Reversal mejorado
2. Breakout de rango
3. Pullback a media móvil
4. Divergencia RSI
5. Gap fills
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Pattern Portfolio Strategy")
    parser.add_argument("--file", required=True, help="Path to market data CSV")
    parser.add_argument("--target_daily", type=float, default=2300, help="Target daily P&L")
    parser.add_argument("--capital", type=float, default=100000, help="Trading capital")
    parser.add_argument("--max_concurrent", type=int, default=3, help="Max concurrent positions")
    parser.add_argument("--output", default="portfolio_results.json", help="Output file")
    return parser.parse_args()

class PatternPortfolio:
    """Gestiona múltiples estrategias de patrones"""
    
    def __init__(self, capital=100000, max_concurrent=3):
        self.capital = capital
        self.max_concurrent = max_concurrent
        self.active_positions = []
        self.all_trades = []
        self.daily_pnl = {}
        
        # Configuraciones de cada estrategia
        self.strategies = {
            'v_reversal': {
                'allocation': 0.4,  # 40% del capital
                'min_confidence': 0.7,
                'max_positions': 2
            },
            'breakout': {
                'allocation': 0.25,  # 25% del capital
                'min_confidence': 0.6,
                'max_positions': 1
            },
            'pullback': {
                'allocation': 0.2,   # 20% del capital
                'min_confidence': 0.65,
                'max_positions': 1
            },
            'divergence': {
                'allocation': 0.1,   # 10% del capital
                'min_confidence': 0.8,
                'max_positions': 1
            },
            'gap_fill': {
                'allocation': 0.05,  # 5% del capital
                'min_confidence': 0.75,
                'max_positions': 1
            }
        }
    
    def calculate_position_size(self, strategy_name, entry_price, stop_loss):
        """Calcula tamaño de posición por estrategia"""
        allocation = self.strategies[strategy_name]['allocation']
        risk_capital = self.capital * allocation * 0.02  # 2% de riesgo por estrategia
        
        risk_per_contract = abs(entry_price - stop_loss) * 50
        if risk_per_contract > 0:
            contracts = int(risk_capital / risk_per_contract)
            return max(1, min(contracts, 3))  # Entre 1 y 3 contratos
        return 1
    
    def can_enter_position(self, strategy_name):
        """Verifica si podemos entrar en una nueva posición"""
        # Verificar límite total
        if len(self.active_positions) >= self.max_concurrent:
            return False
        
        # Verificar límite por estrategia
        strategy_positions = [p for p in self.active_positions if p['strategy'] == strategy_name]
        max_for_strategy = self.strategies[strategy_name]['max_positions']
        
        return len(strategy_positions) < max_for_strategy

class VReversalImproved:
    """V-Reversal mejorado con filtros adicionales"""
    
    @staticmethod
    def detect_pattern(df, idx, min_confidence=0.7):
        if idx < 50:
            return False, 0.0, {}
        
        data = df.iloc[max(0, idx-50):idx+1]
        current_price = data['Close'].iloc[-1]
        
        # 1. Detectar caída significativa
        lookback = 15
        recent_high = data['High'].rolling(lookback).max().iloc[-1]
        drop_points = recent_high - current_price
        drop_pct = drop_points / recent_high
        
        if drop_points < 4.0 or drop_pct < 0.008:
            return False, 0.0, {}
        
        # 2. Velocidad de caída
        drop_start_idx = data['High'].rolling(lookback).idxmax().iloc[-1] - data.index[0]
        drop_duration = len(data) - 1 - drop_start_idx
        drop_speed = drop_points / max(drop_duration, 1)
        
        # 3. RSI oversold
        rsi = calculate_rsi(data['Close'])
        
        # 4. Volumen
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
        
        # 5. Soporte/Resistencia
        support_nearby = check_support_level(data, current_price)
        
        # Calcular confianza
        confidence_factors = {
            'drop_strength': min(drop_pct / 0.015, 1.0),  # Max at 1.5% drop
            'drop_speed': min(drop_speed / 2.0, 1.0),     # Max at 2 points/min
            'rsi_oversold': max(0, (40 - rsi) / 20) if rsi < 40 else 0,
            'volume_surge': min((volume_surge - 1) / 2, 1.0),
            'support_bounce': 0.3 if support_nearby else 0
        }
        
        confidence = sum(confidence_factors.values()) / len(confidence_factors)
        
        if confidence >= min_confidence:
            # Calcular stops
            atr = calculate_atr(data)
            stop_loss = current_price - (atr * 2.0)
            take_profit = current_price + (atr * 4.0)
            
            return True, confidence, {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern_details': confidence_factors
            }
        
        return False, confidence, {}

class BreakoutStrategy:
    """Estrategia de breakout de rangos"""
    
    @staticmethod
    def detect_pattern(df, idx, min_confidence=0.6):
        if idx < 100:
            return False, 0.0, {}
        
        data = df.iloc[max(0, idx-100):idx+1]
        current_price = data['Close'].iloc[-1]
        
        # Detectar rango consolidación (últimos 30 períodos)
        consolidation_period = 30
        recent_data = data.tail(consolidation_period)
        
        range_high = recent_data['High'].max()
        range_low = recent_data['Low'].min()
        range_size = range_high - range_low
        range_midpoint = (range_high + range_low) / 2
        
        # Verificar que es un rango válido
        min_range_size = current_price * 0.005  # Mínimo 0.5% del precio
        if range_size < min_range_size:
            return False, 0.0, {}
        
        # Verificar breakout
        breakout_threshold = range_high + (range_size * 0.1)  # 10% arriba del rango
        
        if current_price <= breakout_threshold:
            return False, 0.0, {}
        
        # Factores de confianza
        confidence_factors = {
            'range_quality': 1.0 - (range_size / (current_price * 0.02)),  # Rango no muy amplio
            'breakout_strength': min((current_price - range_high) / range_size, 1.0),
            'volume_confirmation': min(data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1] / 1.5, 1.0),
            'consolidation_time': min(consolidation_period / 50, 1.0)
        }
        
        confidence = sum(confidence_factors.values()) / len(confidence_factors)
        
        if confidence >= min_confidence:
            atr = calculate_atr(data)
            stop_loss = range_high - (atr * 0.5)  # Stop justo debajo del rango
            take_profit = current_price + (range_size * 1.5)  # Target 1.5x rango
            
            return True, confidence, {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern_details': confidence_factors
            }
        
        return False, confidence, {}

class PullbackStrategy:
    """Pullback a media móvil en tendencia alcista"""
    
    @staticmethod
    def detect_pattern(df, idx, min_confidence=0.65):
        if idx < 100:
            return False, 0.0, {}
        
        data = df.iloc[max(0, idx-100):idx+1]
        current_price = data['Close'].iloc[-1]
        
        # Calcular EMAs
        ema_20 = data['Close'].ewm(span=20).mean().iloc[-1]
        ema_50 = data['Close'].ewm(span=50).mean().iloc[-1]
        
        # Verificar tendencia alcista
        if ema_20 <= ema_50:
            return False, 0.0, {}
        
        # Verificar pullback hacia EMA20
        distance_to_ema = abs(current_price - ema_20) / current_price
        
        # Debe estar cerca pero no muy lejos de la EMA
        if distance_to_ema > 0.003:  # Más de 0.3%
            return False, 0.0, {}
        
        # Factores de confianza
        trend_strength = (ema_20 - ema_50) / ema_50
        rsi = calculate_rsi(data['Close'])
        
        confidence_factors = {
            'trend_strength': min(trend_strength / 0.02, 1.0),  # Tendencia fuerte
            'pullback_depth': 1.0 - (distance_to_ema / 0.003),  # Cerca de EMA
            'rsi_level': max(0, (45 - rsi) / 15) if rsi < 45 else 0,  # RSI no sobrecomprado
            'ema_slope': 1.0 if ema_20 > data['Close'].ewm(span=20).mean().iloc[-5] else 0.3
        }
        
        confidence = sum(confidence_factors.values()) / len(confidence_factors)
        
        if confidence >= min_confidence:
            atr = calculate_atr(data)
            stop_loss = ema_50 - atr  # Stop debajo de EMA50
            take_profit = current_price + (atr * 3.0)
            
            return True, confidence, {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern_details': confidence_factors
            }
        
        return False, confidence, {}

class RSIDivergenceStrategy:
    """Detecta divergencias bullish en RSI"""
    
    @staticmethod
    def detect_pattern(df, idx, min_confidence=0.8):
        if idx < 100:
            return False, 0.0, {}
        
        data = df.iloc[max(0, idx-100):idx+1]
        
        # Calcular RSI
        rsi_series = data['Close'].rolling(window=1).apply(lambda x: calculate_rsi(data['Close'].iloc[:data['Close'].get_loc(x.index[0])+1]))
        
        if len(rsi_series) < 50:
            return False, 0.0, {}
        
        # Buscar divergencia en los últimos 30 períodos
        recent_prices = data['Low'].tail(30)
        recent_rsi = rsi_series.tail(30)
        
        # Encontrar mínimos locales
        price_lows = []
        rsi_lows = []
        
        for i in range(2, len(recent_prices)-2):
            if (recent_prices.iloc[i] < recent_prices.iloc[i-1] and 
                recent_prices.iloc[i] < recent_prices.iloc[i+1] and
                recent_prices.iloc[i] < recent_prices.iloc[i-2] and 
                recent_prices.iloc[i] < recent_prices.iloc[i+2]):
                price_lows.append((i, recent_prices.iloc[i]))
                rsi_lows.append((i, recent_rsi.iloc[i]))
        
        # Verificar divergencia
        if len(price_lows) < 2:
            return False, 0.0, {}
        
        # Comparar los dos últimos mínimos
        last_price_low = price_lows[-1][1]
        prev_price_low = price_lows[-2][1]
        last_rsi_low = rsi_lows[-1][1]
        prev_rsi_low = rsi_lows[-2][1]
        
        # Divergencia bullish: precio hace mínimo más bajo, RSI hace mínimo más alto
        price_lower = last_price_low < prev_price_low
        rsi_higher = last_rsi_low > prev_rsi_low
        
        if not (price_lower and rsi_higher):
            return False, 0.0, {}
        
        # Factores de confianza
        price_divergence_strength = (prev_price_low - last_price_low) / prev_price_low
        rsi_divergence_strength = (last_rsi_low - prev_rsi_low) / 100
        
        confidence_factors = {
            'price_divergence': min(price_divergence_strength / 0.01, 1.0),
            'rsi_divergence': min(rsi_divergence_strength / 0.1, 1.0),
            'rsi_oversold': max(0, (35 - last_rsi_low) / 15) if last_rsi_low < 35 else 0,
            'timing': 1.0  # Por estar en el punto correcto
        }
        
        confidence = sum(confidence_factors.values()) / len(confidence_factors)
        
        if confidence >= min_confidence:
            current_price = data['Close'].iloc[-1]
            atr = calculate_atr(data)
            
            stop_loss = last_price_low - (atr * 0.5)
            take_profit = current_price + (atr * 4.0)
            
            return True, confidence, {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern_details': confidence_factors
            }
        
        return False, confidence, {}

class GapFillStrategy:
    """Estrategia de llenado de gaps"""
    
    @staticmethod
    def detect_pattern(df, idx, min_confidence=0.75):
        if idx < 10:
            return False, 0.0, {}
        
        current_bar = df.iloc[idx]
        previous_bar = df.iloc[idx-1]
        
        # Detectar gap bajista (gap que puede ser llenado al alza)
        gap_down = previous_bar['Low'] > current_bar['High']
        
        if not gap_down:
            return False, 0.0, {}
        
        gap_size = previous_bar['Low'] - current_bar['High']
        gap_percentage = gap_size / current_bar['Close']
        
        # Gap debe ser significativo pero no excesivo
        if gap_percentage < 0.002 or gap_percentage > 0.01:  # Entre 0.2% y 1%
            return False, 0.0, {}
        
        # Verificar que el precio actual está cerca del gap
        current_price = current_bar['Close']
        distance_to_gap = abs(current_price - previous_bar['Low']) / current_price
        
        if distance_to_gap > 0.005:  # Más de 0.5% del gap
            return False, 0.0, {}
        
        # Factores de confianza
        confidence_factors = {
            'gap_size': 1.0 - abs(gap_percentage - 0.005) / 0.005,  # Óptimo en 0.5%
            'proximity_to_gap': 1.0 - (distance_to_gap / 0.005),
            'volume_context': min(current_bar['Volume'] / df['Volume'].rolling(20).mean().iloc[idx] / 1.2, 1.0),
            'market_context': 0.8  # Gap fills tienen alta probabilidad
        }
        
        confidence = sum(confidence_factors.values()) / len(confidence_factors)
        
        if confidence >= min_confidence:
            # Entry al precio actual, target el gap fill
            entry_price = current_price
            take_profit = previous_bar['Low']  # Llenar el gap
            stop_loss = current_price - (gap_size * 0.5)  # Stop conservador
            
            return True, confidence, {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern_details': confidence_factors
            }
        
        return False, confidence, {}

# Funciones auxiliares
def calculate_rsi(prices, period=14):
    """Calcula RSI"""
    if len(prices) < period + 1:
        return 50
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

def calculate_atr(data, period=14):
    """Calcula Average True Range"""
    if len(data) < period:
        return data['Close'].iloc[-1] * 0.01
    
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean().iloc[-1]
    
    return atr if not pd.isna(atr) else data['Close'].iloc[-1] * 0.01

def check_support_level(data, current_price, tolerance=0.002):
    """Verifica si hay un nivel de soporte cercano"""
    lows = data['Low']
    
    for i in range(len(lows)-5, 0, -1):
        if abs(lows.iloc[i] - current_price) / current_price <= tolerance:
            # Verificar que es un mínimo local
            window = 3
            start_idx = max(0, i - window)
            end_idx = min(len(lows), i + window + 1)
            
            if lows.iloc[i] == lows.iloc[start_idx:end_idx].min():
                return True
    
    return False

def filter_trading_hours(df):
    """Filtra solo horarios de alta volatilidad"""
    time_col = df['Datetime'].dt.time
    
    window1 = (time_col >= pd.to_datetime("03:00").time()) & (time_col < pd.to_datetime("04:00").time())
    window2 = (time_col >= pd.to_datetime("09:00").time()) & (time_col < pd.to_datetime("11:00").time())
    window3 = (time_col >= pd.to_datetime("13:30").time()) & (time_col < pd.to_datetime("15:00").time())
    
    return df[window1 | window2 | window3].reset_index(drop=True)

def load_data(file_path):
    """Carga y prepara los datos"""
    print(f"📊 Cargando datos desde {file_path}...")
    
    df = pd.read_csv(file_path)
    
    # Detectar columna de tiempo
    time_col = None
    for col in df.columns:
        if col.lower() in ['datetime', 'timestamp', 'date']:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError("No se encontró columna de tiempo")
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: 'Datetime'})
    
    # Estandarizar columnas
    column_mapping = {}
    for col in df.columns:
        if col.lower() == 'open':
            column_mapping[col] = 'Open'
        elif col.lower() == 'high':
            column_mapping[col] = 'High'
        elif col.lower() == 'low':
            column_mapping[col] = 'Low'
        elif col.lower() == 'close':
            column_mapping[col] = 'Close'
        elif col.lower() == 'volume':
            column_mapping[col] = 'Volume'
    
    df = df.rename(columns=column_mapping)
    
    if 'Volume' not in df.columns:
        df['Volume'] = 1000
    
    df = filter_trading_hours(df)
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    print(f"✅ Datos cargados: {len(df):,} registros")
    return df

def run_portfolio_strategy(df, args):
    """Ejecuta el portfolio de estrategias"""
    
    print("🚀 Ejecutando Portfolio Multi-Patrón...")
    print(f"🎯 Target diario: ${args.target_daily:,.0f}")
    print(f"💰 Capital: ${args.capital:,.0f}")
    
    portfolio = PatternPortfolio(args.capital, args.max_concurrent)
    
    # Instanciar estrategias
    strategies = {
        'v_reversal': VReversalImproved(),
        'breakout': BreakoutStrategy(),
        'pullback': PullbackStrategy(),
        'divergence': RSIDivergenceStrategy(),
        'gap_fill': GapFillStrategy()
    }
    
    print(f"\n📈 Estrategias activas: {len(strategies)}")
    for name, config in portfolio.strategies.items():
        print(f"  • {name}: {config['allocation']:.0%} asignación, confianza mín: {config['min_confidence']:.0%}")
    
    all_trades = []
    daily_performance = {}
    
    for i in range(100, len(df) - 30):
        current_time = df.iloc[i]['Datetime']
        current_date = current_time.date()
        
        # Inicializar día
        if current_date not in daily_performance:
            daily_performance[current_date] = {'pnl': 0, 'trades': 0}
        
        # Buscar patrones en cada estrategia
        for strategy_name, strategy_obj in strategies.items():
            if not portfolio.can_enter_position(strategy_name):
                continue
            
            config = portfolio.strategies[strategy_name]
            found, confidence, details = strategy_obj.detect_pattern(df, i, config['min_confidence'])
            
            if found and details:
                entry_price = details['entry_price']
                stop_loss = details['stop_loss']
                take_profit = details['take_profit']
                
                # Calcular tamaño de posición
                position_size = portfolio.calculate_position_size(strategy_name, entry_price, stop_loss)
                
                # Simular ejecución del trade
                exit_price, exit_reason, exit_idx = simulate_trade_execution(
                    df, i, entry_price, stop_loss, take_profit
                )
                
                # Calcular P&L
                points_gained = exit_price - entry_price
                pnl_dollars = points_gained * position_size * 50
                
                # Crear registro
                trade = {
                    'strategy': strategy_name,
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'exit_time': df.iloc[exit_idx]['Datetime'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'points_gained': points_gained,
                    'pnl_dollars': pnl_dollars,
                    'date': current_date,
                    'pattern_details': details.get('pattern_details', {})
                }
                
                all_trades.append(trade)
                daily_performance[current_date]['pnl'] += pnl_dollars
                daily_performance[current_date]['trades'] += 1
                
                # Saltar al final del trade
                i = exit_idx
    
    return all_trades, daily_performance

def simulate_trade_execution(df, entry_idx, entry_price, stop_loss, take_profit, max_hold=25):
    """Simula la ejecución de un trade"""
    
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        bar = df.iloc[i]
        
        # Check take profit
        if bar['High'] >= take_profit:
            return take_profit, "TAKE_PROFIT", i
        
        # Check stop loss
        if bar['Low'] <= stop_loss:
            return stop_loss, "STOP_LOSS", i
    
    # Time exit
    final_idx = min(entry_idx + max_hold - 1, len(df) - 1)
    return df.iloc[final_idx]['Close'], "TIME_EXIT", final_idx

def analyze_portfolio_results(trades, daily_performance, target_daily):
    """Analiza resultados del portfolio"""
    
    if not trades:
        print("❌ No se encontraron trades")
        return {}
    
    print(f"\n📊 ANÁLISIS DEL PORTFOLIO")
    print("=" * 60)
    
    # Estadísticas generales
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl_dollars'] > 0]
    total_pnl = sum(t['pnl_dollars'] for t in trades)
    
    print(f"Total de Trades: {total_trades:,}")
    print(f"Trades Ganadores: {len(winning_trades):,} ({len(winning_trades)/total_trades*100:.1f}%)")
    print(f"P&L Total: ${total_pnl:,.2f}")
    print(f"P&L Promedio por Trade: ${total_pnl/total_trades:.2f}")
    
    # Análisis por estrategia
    print(f"\n📈 PERFORMANCE POR ESTRATEGIA:")
    strategies = {}
    for trade in trades:
        strategy = trade['strategy']
        if strategy not in strategies:
            strategies[strategy] = {'trades': [], 'pnl': 0}
        strategies[strategy]['trades'].append(trade)
        strategies[strategy]['pnl'] += trade['pnl_dollars']
    
    for strategy, data in strategies.items():
        strategy_trades = data['trades']
        strategy_pnl = data['pnl']
        strategy_wins = [t for t in strategy_trades if t['pnl_dollars'] > 0]
        win_rate = len(strategy_wins) / len(strategy_trades) * 100
        avg_pnl = strategy_pnl / len(strategy_trades)
        
        print(f"  • {strategy.upper()}:")
        print(f"    Trades: {len(strategy_trades)}, Win Rate: {win_rate:.1f}%, P&L: ${strategy_pnl:.2f}, Avg: ${avg_pnl:.2f}")
    
    # Performance diaria
    daily_pnls = [day['pnl'] for day in daily_performance.values()]
    if daily_pnls:
        avg_daily_pnl = np.mean(daily_pnls)
        max_daily_pnl = np.max(daily_pnls)
        min_daily_pnl = np.min(daily_pnls)
        days_above_target = sum(1 for pnl in daily_pnls if pnl >= target_daily)
        
        print(f"\n📅 PERFORMANCE DIARIA:")
        print(f"P&L Diario Promedio: ${avg_daily_pnl:.2f}")
        print(f"Mejor Día: ${max_daily_pnl:.2f}")
        print(f"Peor Día: ${min_daily_pnl:.2f}")
        print(f"Días > ${target_daily}: {days_above_target}/{len(daily_pnls)} ({days_above_target/len(daily_pnls)*100:.1f}%)")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        if avg_daily_pnl >= target_daily:
            print("🎉 ¡TARGET ALCANZADO! Estrategias de optimización:")
            print("   1. Aumentar capital para escalar proporcionalmente")
            print("   2. Añadir más patrones especializados")
            print("   3. Optimizar horarios por estrategia")
        else:
            improvement_needed = target_daily / avg_daily_pnl
            print(f"Necesitas {improvement_needed:.1f}x más P&L diario")
            
            if improvement_needed <= 2:
                print("✅ Estrategias viables:")
                print("   1. Duplicar tamaño de posiciones")
                print("   2. Optimizar thresholds de confianza")
                print("   3. Añadir sesiones de trading (Europa/Asia)")
            elif improvement_needed <= 4:
                print("⚠️ Estrategias más agresivas:")
                print("   1. Trading de múltiples instrumentos (NQ, YM, RTY)")
                print("   2. Estrategias de scalping de alta frecuencia")
                print("   3. Leverage controlado (2:1)")
            else:
                print("🔄 Necesitas un enfoque completamente diferente:")
                print("   1. Portfolio de múltiples mercados")
                print("   2. Estrategias de options/derivatives")
                print("   3. Algoritmic trading institucional")
        
        return {
            'total_trades': total_trades,
            'win_rate': len(winning_trades)/total_trades*100,
            'total_pnl': total_pnl,
            'avg_daily_pnl': avg_daily_pnl,
            'days_above_target': days_above_target,
            'strategies_performance': {name: {
                'trades': len(data['trades']),
                'pnl': data['pnl'],
                'win_rate': len([t for t in data['trades'] if t['pnl_dollars'] > 0]) / len(data['trades']) * 100
            } for name, data in strategies.items()}
        }
    
    return {}

def main():
    args = parse_args()
    
    try:
        # Cargar datos
        df = load_data(args.file)
        
        # Ejecutar portfolio
        trades, daily_performance = run_portfolio_strategy(df, args)
        
        # Analizar resultados
        results = analyze_portfolio_results(trades, daily_performance, args.target_daily)
        
        # Guardar resultados
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': vars(args),
            'summary': results,
            'trades': trades,
            'daily_performance': {str(date): perf for date, perf in daily_performance.items()}
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Resultados guardados en: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 