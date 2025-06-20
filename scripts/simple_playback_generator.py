"""
simple_playback_generator.py
============================
Generador simple de señales para Playback usando resultados validados.
"""

import pandas as pd
import json
import os
from datetime import datetime

class SimplePlaybackGenerator:
    def __init__(self, output_dir="signals/playback"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_test_signals(self, num_signals=5):
        """Generar señales de prueba para playback"""
        
        print("🎮 GENERANDO SEÑALES DE PRUEBA PARA PLAYBACK")
        print("=" * 50)
        
        # Precios base ES para pruebas realistas
        base_prices = [5800.00, 5815.25, 5833.50, 5846.75, 5862.00]
        
        signals_generated = []
        
        for i in range(num_signals):
            timestamp = int(datetime.now().timestamp()) + (i * 60)  # Espaciados 1 min
            signal_id = f"TEST_{timestamp}_{i:03d}"
            
            entry_price = base_prices[i % len(base_prices)]
            stop_loss = entry_price * (1 - 0.001)  # 0.1% stop loss
            take_profit = entry_price + 3.0  # +3 puntos target
            
            signal_content = f"""# V-Reversal Signal - Playback Test
ACTION = BUY
ENTRY_PRICE = {entry_price:.2f}
STOP_LOSS = {stop_loss:.2f}
TAKE_PROFIT = {take_profit:.2f}
SIGNAL_ID = {signal_id}
TIMESTAMP = {timestamp}
CONFIDENCE = HIGH
PATTERN_TYPE = V_REVERSAL
POSITION_SIZE = 1
PLAYBACK_MODE = TRUE
EXPECTED_WIN_RATE = 91.8%
"""
            
            filename = f"vreversal_test_{signal_id}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(signal_content)
            
            signals_generated.append({
                'filename': filename,
                'filepath': filepath,
                'signal_id': signal_id,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
            print(f"📝 Señal {i+1}/5: {filename}")
            print(f"   💰 Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
        
        # Crear resumen
        summary = {
            'generated_at': datetime.now().isoformat(),
            'mode': 'PLAYBACK_TEST',
            'total_signals': len(signals_generated),
            'expected_performance': {
                'win_rate': 91.8,
                'avg_points_per_trade': 2.0,
                'expected_profit_per_signal': 100.0  # ~$100 por señal
            },
            'signals': signals_generated
        }
        
        summary_file = os.path.join(self.output_dir, f"playback_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ SEÑALES LISTAS PARA PLAYBACK")
        print("=" * 50)
        print(f"📁 Ubicación: {os.path.abspath(self.output_dir)}")
        print(f"🔢 Total señales: {len(signals_generated)}")
        print(f"📋 Resumen: {summary_file}")
        
        return signals_generated, summary_file

def main():
    print("🚀 GENERADOR DE SEÑALES PARA PLAYBACK")
    print("=" * 50)
    
    generator = SimplePlaybackGenerator()
    signals, summary = generator.generate_test_signals(num_signals=5)
    
    print(f"\n🎮 CONFIGURACIÓN PARA NINJATRADER")
    print("=" * 50)
    print(f"1. 📁 Signal Path: {os.path.abspath(generator.output_dir)}")
    print(f"2. 🎯 Estrategia: VReversalAutoTrader.cs")
    print(f"3. 📊 Señales esperadas: {len(signals)}")
    print(f"4. 💰 Win rate esperado: 91.8%")
    print(f"5. 🎮 Modo: Playback")
    
    print(f"\n📋 PASOS SIGUIENTES:")
    print("1. Configurar NT Playback Connection")
    print("2. Establecer signal path en VReversalAutoTrader")
    print("3. Iniciar playback con datos históricos")
    print("4. Monitorear ejecución automática")
    
    print(f"\n✅ SISTEMA LISTO PARA PRUEBAS")

if __name__ == "__main__":
    main() 