"""
paper_trading_simulator.py
===========================
Simulador de paper trading para probar el sistema V-reversal en tiempo real
sin riesgo financiero. Simula condiciones reales de trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import random
from pathlib import Path
from production_system_clean import ProductionVReversalSystem

class PaperTradingSimulator:
    def __init__(self, initial_balance=100000, position_size=3):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position_size = position_size
        self.open_positions = []
        self.closed_trades = []
        self.daily_pnl = []
        self.system = ProductionVReversalSystem()
        
        # Métricas de performance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        
        # Log de actividad
        self.activity_log = []
        
    def log_activity(self, message, level="INFO"):
        """Registrar actividad del simulador"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.activity_log.append(log_entry)
        print(f"[{timestamp}] {level}: {message}")
    
    def simulate_real_time_data(self, df, start_date=None, days_to_simulate=30):
        """Simular datos en tiempo real desde un dataset histórico"""
        
        if start_date:
            df = df[df['Datetime'] >= start_date].reset_index(drop=True)
        
        # Tomar solo los días solicitados
        unique_dates = df['Datetime'].dt.date.unique()
        if len(unique_dates) > days_to_simulate:
            end_date = unique_dates[days_to_simulate - 1]
            df = df[df['Datetime'].dt.date <= end_date].reset_index(drop=True)
        
        self.log_activity(f"Iniciando simulación de {days_to_simulate} días")
        self.log_activity(f"Período: {df['Datetime'].min()} → {df['Datetime'].max()}")
        self.log_activity(f"Bars totales: {len(df):,}")
        
        return df
    
    def add_market_noise(self, price, volatility=0.001):
        """Añadir ruido de mercado para simular condiciones reales"""
        noise = random.gauss(0, volatility * price)
        return price + noise
    
    def simulate_slippage(self, intended_price, market_impact=0.25):
        """Simular slippage en ejecución de órdenes"""
        # Slippage típico de 0.25 puntos en ES
        slippage = random.uniform(0, market_impact)
        return intended_price + slippage  # Siempre desfavorable
    
    def execute_trade(self, signal_data, current_bar):
        """Ejecutar trade basado en señal"""
        
        try:
            # Simular slippage en entrada
            entry_price = self.simulate_slippage(signal_data['entry_price'])
            stop_price = signal_data['stop_price']
            target_price = signal_data['target_price']
            
            # Crear posición
            position = {
                'id': len(self.closed_trades) + len(self.open_positions) + 1,
                'entry_time': current_bar['Datetime'],
                'entry_price': entry_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'position_size': self.position_size,
                'direction': 'LONG',  # V-reversal siempre es long
                'status': 'OPEN'
            }
            
            self.open_positions.append(position)
            self.total_trades += 1
            
            self.log_activity(f"TRADE ABIERTO #{position['id']}: Entry ${entry_price:.2f}, Stop ${stop_price:.2f}, Target ${target_price:.2f}")
            
        except Exception as e:
            self.log_activity(f"Error ejecutando trade: {e}", "ERROR")
    
    def manage_positions(self, current_bar):
        """Gestionar posiciones abiertas"""
        
        positions_to_close = []
        current_price = current_bar['Close']
        
        for position in self.open_positions:
            # Verificar stop loss
            if current_price <= position['stop_price']:
                positions_to_close.append((position, 'STOP_LOSS', position['stop_price']))
                
            # Verificar target
            elif current_price >= position['target_price']:
                positions_to_close.append((position, 'TARGET', position['target_price']))
                
            # Verificar time-based exit (final del día)
            elif current_bar['Datetime'].hour >= 15:  # 3 PM ET
                positions_to_close.append((position, 'TIME_EXIT', current_price))
        
        # Cerrar posiciones
        for position, exit_reason, exit_price in positions_to_close:
            self.close_position(position, exit_price, exit_reason, current_bar['Datetime'])
    
    def close_position(self, position, exit_price, exit_reason, exit_time):
        """Cerrar posición y calcular P&L"""
        
        # Simular slippage en salida
        actual_exit_price = self.simulate_slippage(exit_price)
        
        # Calcular P&L
        pnl_per_contract = (actual_exit_price - position['entry_price']) * 50  # ES multiplier
        total_pnl = pnl_per_contract * position['position_size']
        
        # Actualizar balance
        self.current_balance += total_pnl
        
        # Actualizar métricas
        if total_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Calcular drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Registrar trade cerrado
        closed_trade = {
            'id': position['id'],
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': position['entry_price'],
            'exit_price': actual_exit_price,
            'position_size': position['position_size'],
            'pnl': total_pnl,
            'exit_reason': exit_reason,
            'duration_minutes': (exit_time - position['entry_time']).total_seconds() / 60
        }
        
        self.closed_trades.append(closed_trade)
        self.open_positions.remove(position)
        
        self.log_activity(f"TRADE CERRADO #{position['id']}: {exit_reason} @ ${actual_exit_price:.2f}, P&L: ${total_pnl:.2f}")
    
    def run_simulation(self, df, days_to_simulate=30):
        """Ejecutar simulación completa"""
        
        self.log_activity("Iniciando simulación de paper trading")
        
        # Preparar datos
        simulation_df = self.simulate_real_time_data(df, days_to_simulate=days_to_simulate)
        
        # Simular trading bar por bar
        for i, row in simulation_df.iterrows():
            current_bar = row.to_dict()
            current_bar['Datetime'] = pd.to_datetime(current_bar['Datetime'])
            
            # Gestionar posiciones existentes
            if self.open_positions:
                self.manage_positions(current_bar)
            
            # Buscar nuevas señales (solo durante horas de trading)
            if self.is_trading_hours(current_bar['Datetime']):
                # Obtener datos hasta este punto para análisis
                historical_data = simulation_df.iloc[:i+1].copy()
                
                # Detectar patrones (usando últimas 100 barras para eficiencia)
                if len(historical_data) >= 100:
                    recent_data = historical_data.tail(100).reset_index(drop=True)
                    
                    try:
                        # Usar el sistema de producción para detectar señales
                        patterns = self.system.detect_patterns(recent_data)
                        
                        # Si hay un patrón nuevo en la última barra
                        if patterns and len(patterns) > 0:
                            latest_pattern = patterns[-1]
                            
                            # Verificar que es un patrón nuevo (no ya ejecutado)
                            if not self.is_pattern_already_traded(latest_pattern, current_bar['Datetime']):
                                signal_data = self.system.generate_signal(latest_pattern, current_bar)
                                if signal_data:
                                    self.execute_trade(signal_data, current_bar)
                    
                    except Exception as e:
                        self.log_activity(f"Error detectando patrones: {e}", "ERROR")
            
            # Calcular P&L diario
            if i > 0 and current_bar['Datetime'].date() != simulation_df.iloc[i-1]['Datetime'].date():
                daily_pnl = self.current_balance - (self.daily_pnl[-1] if self.daily_pnl else self.initial_balance)
                self.daily_pnl.append(self.current_balance)
                
                if daily_pnl != 0:
                    self.log_activity(f"P&L Diario: ${daily_pnl:.2f}, Balance: ${self.current_balance:.2f}")
        
        # Cerrar posiciones abiertas al final
        if self.open_positions:
            final_bar = simulation_df.iloc[-1]
            for position in self.open_positions.copy():
                self.close_position(position, final_bar['Close'], 'SIMULATION_END', final_bar['Datetime'])
        
        self.log_activity("Simulación completada")
        return self.generate_performance_report()
    
    def is_trading_hours(self, timestamp):
        """Verificar si estamos en horas de trading"""
        hour = timestamp.hour
        # Horas principales de trading: 9:30 AM - 4:00 PM ET
        return 9 <= hour <= 16
    
    def is_pattern_already_traded(self, pattern, current_time):
        """Verificar si ya hemos ejecutado un trade para este patrón"""
        # Buscar trades recientes (últimas 2 horas)
        recent_threshold = current_time - timedelta(hours=2)
        
        for trade in self.closed_trades:
            if trade['entry_time'] >= recent_threshold:
                # Si hay un trade muy reciente, probablemente es el mismo patrón
                return True
        
        for position in self.open_positions:
            if position['entry_time'] >= recent_threshold:
                return True
        
        return False
    
    def generate_performance_report(self):
        """Generar reporte de performance"""
        
        if not self.closed_trades:
            return {"error": "No trades executed"}
        
        # Calcular métricas
        total_pnl = sum(trade['pnl'] for trade in self.closed_trades)
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        winning_trades_pnl = [trade['pnl'] for trade in self.closed_trades if trade['pnl'] > 0]
        losing_trades_pnl = [trade['pnl'] for trade in self.closed_trades if trade['pnl'] < 0]
        
        avg_winner = np.mean(winning_trades_pnl) if winning_trades_pnl else 0
        avg_loser = np.mean(losing_trades_pnl) if losing_trades_pnl else 0
        
        # Calcular métricas adicionales
        daily_pnls = []
        if self.closed_trades:
            trades_by_date = {}
            for trade in self.closed_trades:
                date = trade['entry_time'].date()
                if date not in trades_by_date:
                    trades_by_date[date] = 0
                trades_by_date[date] += trade['pnl']
            
            daily_pnls = list(trades_by_date.values())
        
        avg_daily_pnl = np.mean(daily_pnls) if daily_pnls else 0
        
        report = {
            'simulation_summary': {
                'initial_balance': self.initial_balance,
                'final_balance': self.current_balance,
                'total_pnl': total_pnl,
                'total_return_pct': (total_pnl / self.initial_balance) * 100,
                'max_drawdown_pct': self.max_drawdown * 100
            },
            'trading_metrics': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate_pct': win_rate,
                'avg_winner': avg_winner,
                'avg_loser': avg_loser,
                'profit_factor': abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf')
            },
            'daily_performance': {
                'avg_daily_pnl': avg_daily_pnl,
                'trading_days': len(daily_pnls),
                'profitable_days': sum(1 for pnl in daily_pnls if pnl > 0),
                'daily_pnls': daily_pnls
            },
            'target_analysis': {
                'target_daily_pnl': 2300,
                'target_achieved': avg_daily_pnl >= 2300,
                'target_gap': 2300 - avg_daily_pnl,
                'success_rate': (avg_daily_pnl / 2300) * 100 if avg_daily_pnl > 0 else 0
            }
        }
        
        return report
    
    def print_performance_report(self, report):
        """Imprimir reporte de performance"""
        
        print("\n" + "="*60)
        print("📊 REPORTE DE PAPER TRADING")
        print("="*60)
        
        # Resumen de simulación
        print(f"\n💰 RESUMEN FINANCIERO:")
        print(f"   Balance Inicial: ${report['simulation_summary']['initial_balance']:,.2f}")
        print(f"   Balance Final: ${report['simulation_summary']['final_balance']:,.2f}")
        print(f"   P&L Total: ${report['simulation_summary']['total_pnl']:,.2f}")
        print(f"   Retorno: {report['simulation_summary']['total_return_pct']:.2f}%")
        print(f"   Max Drawdown: {report['simulation_summary']['max_drawdown_pct']:.2f}%")
        
        # Métricas de trading
        print(f"\n📈 MÉTRICAS DE TRADING:")
        print(f"   Total Trades: {report['trading_metrics']['total_trades']}")
        print(f"   Win Rate: {report['trading_metrics']['win_rate_pct']:.1f}%")
        print(f"   Trades Ganadores: {report['trading_metrics']['winning_trades']}")
        print(f"   Trades Perdedores: {report['trading_metrics']['losing_trades']}")
        print(f"   Ganancia Promedio: ${report['trading_metrics']['avg_winner']:.2f}")
        print(f"   Pérdida Promedio: ${report['trading_metrics']['avg_loser']:.2f}")
        print(f"   Profit Factor: {report['trading_metrics']['profit_factor']:.2f}")
        
        # Performance diaria
        print(f"\n📅 PERFORMANCE DIARIA:")
        print(f"   P&L Diario Promedio: ${report['daily_performance']['avg_daily_pnl']:.2f}")
        print(f"   Días Operados: {report['daily_performance']['trading_days']}")
        print(f"   Días Rentables: {report['daily_performance']['profitable_days']}")
        if report['daily_performance']['trading_days'] > 0:
            daily_success_rate = (report['daily_performance']['profitable_days'] / report['daily_performance']['trading_days']) * 100
            print(f"   Días Rentables %: {daily_success_rate:.1f}%")
        
        # Análisis de target
        print(f"\n🎯 ANÁLISIS DE TARGET ($2,300/día):")
        if report['target_analysis']['target_achieved']:
            print(f"   ✅ TARGET ALCANZADO!")
            print(f"   Exceso: ${report['target_analysis']['target_gap']*-1:.2f}/día")
        else:
            print(f"   ⚠️ Target no alcanzado")
            print(f"   Gap: ${report['target_analysis']['target_gap']:.2f}/día")
        
        print(f"   Porcentaje del Target: {report['target_analysis']['success_rate']:.1f}%")
        
        # Veredicto final
        print(f"\n🏆 VEREDICTO:")
        if report['target_analysis']['target_achieved'] and report['trading_metrics']['win_rate_pct'] >= 80:
            print("   🎉 EXCELENTE - Listo para trading en vivo")
        elif report['target_analysis']['success_rate'] >= 80:
            print("   ✅ BUENO - Considerar ajustes menores")
        elif report['target_analysis']['success_rate'] >= 60:
            print("   ⚠️ REGULAR - Requiere optimización")
        else:
            print("   ❌ POBRE - Necesita revisión completa")

def main():
    try:
        print("🧪 SIMULADOR DE PAPER TRADING")
        print("Probando modelo V-reversal en condiciones reales")
        print("="*50)
        
        # Cargar datos
        df = pd.read_csv("data/raw/es_1m/market_data.csv")
        df['Datetime'] = pd.to_datetime(df['timestamp'])
        df['Open'] = df['open']
        df['High'] = df['high'] 
        df['Low'] = df['low']
        df['Close'] = df['close']
        df['Volume'] = df['volume']
        
        # Crear simulador
        simulator = PaperTradingSimulator(
            initial_balance=100000,
            position_size=3
        )
        
        # Ejecutar simulación
        report = simulator.run_simulation(df, days_to_simulate=30)
        
        # Mostrar resultados
        if 'error' not in report:
            simulator.print_performance_report(report)
            
            # Guardar resultados
            with open('paper_trading_results.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\n💾 Resultados guardados en: paper_trading_results.json")
        else:
            print(f"❌ Error en simulación: {report['error']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 