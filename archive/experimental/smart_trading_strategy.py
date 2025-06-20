"""
smart_trading_strategy.py
=========================
Estrategia de trading inteligente para alcanzar $2,300/día usando:
- Machine Learning para detección de patrones
- Gestión de riesgo dinámica
- Portfolio optimization
- Stop loss inteligentes y trailing stops
- Position sizing basado en volatilidad
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Intentar importar librerías ML
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: Scikit-learn not available. Using simple rules.")

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Trading Strategy Optimizer")
    parser.add_argument("--file", required=True, help="Path to market data CSV")
    parser.add_argument("--target_daily", type=float, default=2300, help="Target daily P&L")
    parser.add_argument("--max_risk_per_trade", type=float, default=0.005, help="Max risk per trade (0.5%)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="ML confidence threshold")
    parser.add_argument("--lookback_days", type=int, default=30, help="Days for volatility calculation")
    parser.add_argument("--output", default="smart_strategy_results.json", help="Output file")
    return parser.parse_args()

class VolatilityManager:
    """Gestiona la volatilidad y el dimensionamiento de posiciones"""
    
    def __init__(self, lookback_period=30):
        self.lookback_period = lookback_period
        self.price_history = []
        self.volatility_cache = {}
    
    def update_price(self, price, timestamp):
        """Actualiza el historial de precios"""
        self.price_history.append((timestamp, price))
        
        # Mantener solo el período de lookback
        cutoff_time = timestamp - timedelta(days=self.lookback_period)
        self.price_history = [(t, p) for t, p in self.price_history if t >= cutoff_time]
    
    def get_current_volatility(self):
        """Calcula la volatilidad actual basada en ATR y desviación estándar"""
        if len(self.price_history) < 14:
            return 0.01  # Default 1%
        
        prices = [p for _, p in self.price_history[-20:]]  # Últimos 20 períodos
        returns = np.diff(np.log(prices))
        
        # Volatilidad anualizada (asumiendo 252 días de trading)
        volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # Minutos en un año
        return max(volatility, 0.005)  # Mínimo 0.5%

class AdvancedPatternDetector:
    """Detector de patrones avanzado con ML"""
    
    def __init__(self, use_ml=True):
        self.use_ml = use_ml and ML_AVAILABLE
        self.model = None
        self.is_trained = False
        self.feature_history = []
        
        if self.use_ml:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
    
    def extract_features(self, df, idx, window=50):
        """Extrae características técnicas para ML"""
        if idx < window:
            return None
        
        data = df.iloc[max(0, idx-window):idx+1]
        
        # Características básicas
        features = {
            'rsi': self.calculate_rsi(data['Close']),
            'macd_signal': self.calculate_macd_signal(data['Close']),
            'bb_position': self.calculate_bb_position(data['Close']),
            'volume_surge': data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1],
            'price_momentum': (data['Close'].iloc[-1] / data['Close'].iloc[-10] - 1) * 100,
            'volatility': data['High'].rolling(20).std().iloc[-1] / data['Close'].iloc[-1],
        }
        
        # Características de velas
        features.update(self.get_candlestick_features(data.tail(5)))
        
        # Características de soporte/resistencia
        features.update(self.get_support_resistance_features(data))
        
        return np.array(list(features.values()))
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def calculate_macd_signal(self, prices):
        """Calcula señal MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return (macd - signal).iloc[-1]
    
    def calculate_bb_position(self, prices, period=20):
        """Posición en Bandas de Bollinger"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        current_price = prices.iloc[-1]
        upper_band = sma.iloc[-1] + (2 * std.iloc[-1])
        lower_band = sma.iloc[-1] - (2 * std.iloc[-1])
        
        if upper_band == lower_band:
            return 0.5
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return np.clip(position, 0, 1)
    
    def get_candlestick_features(self, data):
        """Características de velas japonesas"""
        features = {}
        
        for i, (_, row) in enumerate(data.iterrows()):
            body_size = abs(row['Close'] - row['Open']) / row['Open']
            upper_shadow = (row['High'] - max(row['Open'], row['Close'])) / row['Open']
            lower_shadow = (min(row['Open'], row['Close']) - row['Low']) / row['Open']
            
            features[f'body_size_{i}'] = body_size
            features[f'upper_shadow_{i}'] = upper_shadow
            features[f'lower_shadow_{i}'] = lower_shadow
            features[f'is_bullish_{i}'] = 1 if row['Close'] > row['Open'] else 0
        
        return features
    
    def get_support_resistance_features(self, data):
        """Características de soporte y resistencia"""
        highs = data['High']
        lows = data['Low']
        current_price = data['Close'].iloc[-1]
        
        # Niveles de resistencia (máximos locales)
        resistance_levels = []
        for i in range(2, len(highs)-2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                if highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i+2]:
                    resistance_levels.append(highs.iloc[i])
        
        # Niveles de soporte (mínimos locales)
        support_levels = []
        for i in range(2, len(lows)-2):
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                if lows.iloc[i] < lows.iloc[i-2] and lows.iloc[i] < lows.iloc[i+2]:
                    support_levels.append(lows.iloc[i])
        
        features = {
            'distance_to_resistance': min([abs(current_price - r)/current_price for r in resistance_levels]) if resistance_levels else 0.1,
            'distance_to_support': min([abs(current_price - s)/current_price for s in support_levels]) if support_levels else 0.1,
            'num_resistance_levels': len(resistance_levels),
            'num_support_levels': len(support_levels)
        }
        
        return features
    
    def train_model(self, df):
        """Entrena el modelo ML con datos históricos"""
        if not self.use_ml:
            return
        
        print("🤖 Entrenando modelo ML...")
        
        # Preparar datos de entrenamiento
        X, y = [], []
        
        for i in range(100, len(df) - 20):  # Necesitamos datos futuros para labels
            features = self.extract_features(df, i)
            if features is None:
                continue
            
            # Label: ¿El precio subió significativamente en los próximos 20 períodos?
            current_price = df.iloc[i]['Close']
            future_max = df.iloc[i+1:i+21]['High'].max()
            price_increase = (future_max - current_price) / current_price
            
            # Clasificación: 0=no trade, 1=profitable trade
            label = 1 if price_increase > 0.002 else 0  # >0.2% gain
            
            X.append(features)
            y.append(label)
        
        if len(X) > 100:
            X = np.array(X)
            y = np.array(y)
            
            # Filtrar NaN e infinitos
            valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) > 50:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.model.fit(X_train, y_train)
                
                # Evaluar modelo
                score = self.model.score(X_test, y_test)
                print(f"✅ Modelo entrenado. Precisión: {score:.3f}")
                self.is_trained = True
            else:
                print("❌ Datos insuficientes para entrenamiento")
        else:
            print("❌ Datos insuficientes para entrenamiento")
    
    def predict_pattern(self, df, idx, confidence_threshold=0.7):
        """Predice si hay un patrón rentable"""
        if self.use_ml and self.is_trained:
            features = self.extract_features(df, idx)
            if features is None:
                return False, 0.0
            
            # Verificar características válidas
            if not np.isfinite(features).all():
                return False, 0.0
            
            try:
                # Predicción con probabilidades
                probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
                confidence = probabilities[1]  # Probabilidad de clase positiva
                
                return confidence > confidence_threshold, confidence
            except:
                return False, 0.0
        else:
            # Usar reglas simples como fallback
            return self.simple_pattern_detection(df, idx)
    
    def simple_pattern_detection(self, df, idx):
        """Detección de patrones usando reglas simples"""
        if idx < 50:
            return False, 0.0
        
        data = df.iloc[max(0, idx-50):idx+1]
        current_price = data['Close'].iloc[-1]
        
        # Buscar caída significativa
        recent_high = data['High'].rolling(15).max().iloc[-1]
        drop_percentage = (recent_high - current_price) / recent_high
        
        # RSI oversold
        rsi = self.calculate_rsi(data['Close'])
        
        # Volumen alto
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Condiciones para patrón válido
        conditions = [
            drop_percentage > 0.008,  # Caída > 0.8%
            rsi < 35,  # RSI oversold
            volume_surge > 1.5,  # Volumen alto
        ]
        
        confidence = sum(conditions) / len(conditions)
        return confidence > 0.6, confidence

class SmartPositionManager:
    """Gestión inteligente de posiciones"""
    
    def __init__(self, max_risk_per_trade=0.005, account_size=100000):
        self.max_risk_per_trade = max_risk_per_trade
        self.account_size = account_size
        self.active_positions = []
        self.daily_pnl = 0
        self.daily_trades = 0
        self.max_daily_trades = 10
    
    def calculate_position_size(self, entry_price, stop_loss, confidence, volatility):
        """Calcula tamaño de posición basado en riesgo y confianza"""
        
        # Riesgo por contrato
        risk_per_contract = abs(entry_price - stop_loss) * 50  # ES = $50 por punto
        
        # Riesgo máximo en dólares
        max_risk_dollars = self.account_size * self.max_risk_per_trade
        
        # Ajustar por confianza (más confianza = más tamaño)
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x a 2.0x
        
        # Ajustar por volatilidad (más volatilidad = menos tamaño)
        volatility_multiplier = 1 / (1 + volatility * 10)  # Reduce tamaño con alta volatilidad
        
        # Calcular tamaño base
        base_contracts = max_risk_dollars / risk_per_contract
        
        # Aplicar multiplicadores
        adjusted_contracts = base_contracts * confidence_multiplier * volatility_multiplier
        
        # Limitar entre 1 y 3 contratos
        position_size = max(1, min(3, int(adjusted_contracts)))
        
        return position_size
    
    def calculate_dynamic_stops(self, entry_price, volatility, confidence, action="BUY"):
        """Calcula stops dinámicos basados en volatilidad y confianza"""
        
        # Stop loss base (más volátil = stop más amplio)
        base_stop_pct = 0.002 + (volatility * 2)  # 0.2% a 2.2%
        
        # Ajustar por confianza (más confianza = stop más ajustado)
        confidence_adjustment = 1 - (confidence - 0.5) * 0.3  # 0.85x a 1.15x
        
        stop_distance = entry_price * base_stop_pct * confidence_adjustment
        
        if action == "BUY":
            stop_loss = entry_price - stop_distance
            
            # Take profit dinámico (risk-reward ratio basado en confianza)
            if confidence > 0.8:
                rr_ratio = 4.0  # 4:1 para alta confianza
            elif confidence > 0.7:
                rr_ratio = 3.0  # 3:1 para confianza media
            else:
                rr_ratio = 2.5  # 2.5:1 para baja confianza
                
            take_profit = entry_price + (stop_distance * rr_ratio)
        else:
            stop_loss = entry_price + stop_distance
            
            # Risk-reward similar para shorts
            if confidence > 0.8:
                rr_ratio = 4.0
            elif confidence > 0.7:
                rr_ratio = 3.0
            else:
                rr_ratio = 2.5
                
            take_profit = entry_price - (stop_distance * rr_ratio)
        
        return stop_loss, take_profit
    
    def should_enter_trade(self):
        """Determina si deberíamos entrar en un nuevo trade"""
        return (len(self.active_positions) < 2 and 
                self.daily_trades < self.max_daily_trades)

def load_and_prepare_data(file_path):
    """Carga y prepara los datos de mercado"""
    print(f"📊 Cargando datos desde {file_path}...")
    
    df = pd.read_csv(file_path)
    
    # Detectar columna de tiempo
    time_col = None
    for col in df.columns:
        if col.lower() in ['datetime', 'timestamp', 'date', 'time']:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError("No se encontró columna de tiempo")
    
    # Convertir a datetime
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: 'Datetime'})
    
    # Estandarizar nombres de columnas
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
    
    # Asegurar que tenemos Volume
    if 'Volume' not in df.columns:
        df['Volume'] = 1000  # Default volume
    
    # Filtrar horarios de trading
    df = filter_trading_hours(df)
    
    # Ordenar por fecha
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    print(f"✅ Datos cargados: {len(df):,} registros")
    return df

def filter_trading_hours(df):
    """Filtra solo los horarios de alta volatilidad"""
    time_col = df['Datetime'].dt.time
    
    # Ventanas de alta volatilidad: 3-4 AM, 9-11 AM, 1:30-3 PM
    window1 = (time_col >= pd.to_datetime("03:00").time()) & (time_col < pd.to_datetime("04:00").time())
    window2 = (time_col >= pd.to_datetime("09:00").time()) & (time_col < pd.to_datetime("11:00").time())
    window3 = (time_col >= pd.to_datetime("13:30").time()) & (time_col < pd.to_datetime("15:00").time())
    
    trading_mask = window1 | window2 | window3
    return df[trading_mask].reset_index(drop=True)

def run_smart_strategy(df, args):
    """Ejecuta la estrategia inteligente"""
    
    print("🚀 Iniciando estrategia inteligente de trading...")
    print(f"🎯 Target diario: ${args.target_daily:,.0f}")
    print(f"⚡ Riesgo máximo por trade: {args.max_risk_per_trade:.1%}")
    
    # Inicializar componentes
    volatility_manager = VolatilityManager(args.lookback_days)
    pattern_detector = AdvancedPatternDetector(use_ml=ML_AVAILABLE)
    position_manager = SmartPositionManager(args.max_risk_per_trade)
    
    # Entrenar modelo ML
    pattern_detector.train_model(df)
    
    # Resultados
    all_trades = []
    daily_performance = {}
    
    print("\n📈 Ejecutando backtesting...")
    
    for i in range(100, len(df) - 20):  # Necesitamos margen para patrones y salidas
        current_row = df.iloc[i]
        current_time = current_row['Datetime']
        current_date = current_time.date()
        current_price = current_row['Close']
        
        # Actualizar volatilidad
        volatility_manager.update_price(current_price, current_time)
        current_volatility = volatility_manager.get_current_volatility()
        
        # Reiniciar contadores diarios
        if current_date not in daily_performance:
            daily_performance[current_date] = {'pnl': 0, 'trades': 0}
            position_manager.daily_pnl = 0
            position_manager.daily_trades = 0
        
        # Buscar patrones si podemos entrar en un trade
        if position_manager.should_enter_trade():
            is_pattern, confidence = pattern_detector.predict_pattern(df, i, args.confidence_threshold)
            
            if is_pattern:
                # Calcular entrada y stops
                entry_price = current_price
                volatility = current_volatility
                
                stop_loss, take_profit = position_manager.calculate_dynamic_stops(
                    entry_price, volatility, confidence, "BUY"
                )
                
                position_size = position_manager.calculate_position_size(
                    entry_price, stop_loss, confidence, volatility
                )
                
                # Buscar salida del trade
                exit_found = False
                exit_price = None
                exit_reason = ""
                exit_idx = None
                
                for j in range(i + 1, min(i + 30, len(df))):  # Máximo 30 períodos
                    future_row = df.iloc[j]
                    high_price = future_row['High']
                    low_price = future_row['Low']
                    
                    # Check take profit
                    if high_price >= take_profit:
                        exit_price = take_profit
                        exit_reason = "TAKE_PROFIT"
                        exit_idx = j
                        exit_found = True
                        break
                    
                    # Check stop loss
                    if low_price <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "STOP_LOSS"
                        exit_idx = j
                        exit_found = True
                        break
                
                # Si no hay salida, salir al final del período
                if not exit_found:
                    exit_idx = min(i + 25, len(df) - 1)
                    exit_price = df.iloc[exit_idx]['Close']
                    exit_reason = "TIME_EXIT"
                
                # Calcular P&L
                points_gained = exit_price - entry_price
                pnl_dollars = points_gained * position_size * 50  # ES = $50 por punto
                
                # Crear registro del trade
                trade_record = {
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'exit_time': df.iloc[exit_idx]['Datetime'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'volatility': volatility,
                    'points_gained': points_gained,
                    'pnl_dollars': pnl_dollars,
                    'date': current_date
                }
                
                all_trades.append(trade_record)
                
                # Actualizar contadores
                daily_performance[current_date]['pnl'] += pnl_dollars
                daily_performance[current_date]['trades'] += 1
                position_manager.daily_pnl += pnl_dollars
                position_manager.daily_trades += 1
                
                # Saltar al final del trade para evitar solapamiento
                i = exit_idx
    
    return all_trades, daily_performance

def analyze_results(trades, daily_performance, target_daily):
    """Analiza los resultados de la estrategia"""
    
    if not trades:
        print("❌ No se encontraron trades")
        return {}
    
    print(f"\n📊 ANÁLISIS DE RESULTADOS")
    print("=" * 50)
    
    # Estadísticas de trades
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl_dollars'] > 0]
    losing_trades = [t for t in trades if t['pnl_dollars'] <= 0]
    
    win_rate = len(winning_trades) / total_trades * 100
    total_pnl = sum(t['pnl_dollars'] for t in trades)
    avg_pnl_per_trade = total_pnl / total_trades
    
    print(f"Total Trades: {total_trades:,}")
    print(f"Trades Ganadores: {len(winning_trades):,} ({win_rate:.1f}%)")
    print(f"Trades Perdedores: {len(losing_trades):,}")
    print(f"P&L Total: ${total_pnl:,.2f}")
    print(f"P&L Promedio por Trade: ${avg_pnl_per_trade:.2f}")
    
    # Estadísticas diarias
    daily_pnls = [day['pnl'] for day in daily_performance.values()]
    avg_daily_pnl = np.mean(daily_pnls)
    max_daily_pnl = np.max(daily_pnls)
    min_daily_pnl = np.min(daily_pnls)
    
    days_above_target = sum(1 for pnl in daily_pnls if pnl >= target_daily)
    total_days = len(daily_pnls)
    
    print(f"\n📅 PERFORMANCE DIARIA:")
    print(f"P&L Diario Promedio: ${avg_daily_pnl:.2f}")
    print(f"Mejor Día: ${max_daily_pnl:.2f}")
    print(f"Peor Día: ${min_daily_pnl:.2f}")
    print(f"Días que alcanzaron target (${target_daily}): {days_above_target}/{total_days} ({days_above_target/total_days*100:.1f}%)")
    
    # Análisis por confianza
    high_confidence_trades = [t for t in trades if t['confidence'] > 0.8]
    medium_confidence_trades = [t for t in trades if 0.7 <= t['confidence'] <= 0.8]
    low_confidence_trades = [t for t in trades if t['confidence'] < 0.7]
    
    print(f"\n🎯 ANÁLISIS POR CONFIANZA:")
    if high_confidence_trades:
        hc_winrate = sum(1 for t in high_confidence_trades if t['pnl_dollars'] > 0) / len(high_confidence_trades) * 100
        hc_avg_pnl = np.mean([t['pnl_dollars'] for t in high_confidence_trades])
        print(f"Alta Confianza (>80%): {len(high_confidence_trades)} trades, {hc_winrate:.1f}% win rate, ${hc_avg_pnl:.2f} avg P&L")
    
    if medium_confidence_trades:
        mc_winrate = sum(1 for t in medium_confidence_trades if t['pnl_dollars'] > 0) / len(medium_confidence_trades) * 100
        mc_avg_pnl = np.mean([t['pnl_dollars'] for t in medium_confidence_trades])
        print(f"Confianza Media (70-80%): {len(medium_confidence_trades)} trades, {mc_winrate:.1f}% win rate, ${mc_avg_pnl:.2f} avg P&L")
    
    if low_confidence_trades:
        lc_winrate = sum(1 for t in low_confidence_trades if t['pnl_dollars'] > 0) / len(low_confidence_trades) * 100
        lc_avg_pnl = np.mean([t['pnl_dollars'] for t in low_confidence_trades])
        print(f"Baja Confianza (<70%): {len(low_confidence_trades)} trades, {lc_winrate:.1f}% win rate, ${lc_avg_pnl:.2f} avg P&L")
    
    # Recommendations
    print(f"\n💡 RECOMENDACIONES PARA ALCANZAR ${target_daily}/DÍA:")
    
    if avg_daily_pnl < target_daily:
        multiplier_needed = target_daily / avg_daily_pnl if avg_daily_pnl > 0 else float('inf')
        print(f"Necesitas {multiplier_needed:.1f}x más P&L diario")
        
        if multiplier_needed < 3:
            print("✅ Estrategias viables:")
            print(f"   1. Aumentar tamaño de posición a {int(2 * multiplier_needed)} contratos")
            print("   2. Aumentar threshold de confianza para trades de mayor calidad")
            print("   3. Optimizar horarios de trading")
            if ML_AVAILABLE:
                print("   4. Mejorar modelo ML con más características")
        else:
            print("⚠️ Target muy ambicioso con estos datos. Considera:")
            print("   1. Añadir más instrumentos (NQ, YM, etc.)")
            print("   2. Trading de alta frecuencia")
            print("   3. Estrategias de arbitraje")
    else:
        print("🎉 ¡Target alcanzable! Optimizaciones sugeridas:")
        print("   1. Aumentar capital para escalar")
        print("   2. Automatizar completamente")
        print("   3. Diversificar instrumentos")
    
    # Métricas de riesgo
    if daily_pnls:
        sharpe_ratio = np.mean(daily_pnls) / np.std(daily_pnls) if np.std(daily_pnls) > 0 else 0
        max_drawdown = calculate_max_drawdown(daily_pnls)
        
        print(f"\n⚖️ MÉTRICAS DE RIESGO:")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Máximo Drawdown: ${max_drawdown:.2f}")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_daily_pnl': avg_daily_pnl,
        'days_above_target': days_above_target,
        'target_achievement_rate': days_above_target/total_days*100 if total_days > 0 else 0
    }

def calculate_max_drawdown(daily_pnls):
    """Calcula el máximo drawdown"""
    cumulative = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return np.max(drawdowns) if len(drawdowns) > 0 else 0

def main():
    args = parse_args()
    
    try:
        # Cargar datos
        df = load_and_prepare_data(args.file)
        
        # Ejecutar estrategia
        trades, daily_performance = run_smart_strategy(df, args)
        
        # Analizar resultados
        results = analyze_results(trades, daily_performance, args.target_daily)
        
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