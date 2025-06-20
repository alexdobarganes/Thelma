#!/usr/bin/env python3
"""
Diagnóstico de conexión NinjaTrader
===================================
Script para verificar qué datos está enviando realmente NinjaTrader
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "python-client"))

from websocket_client import WebSocketClient, MarketData

class NTDiagnostic:
    def __init__(self):
        self.total_messages = 0
        self.tick_count = 0
        self.bar_count = 0
        self.historical_count = 0
        self.other_count = 0
        self.last_bar_time = None
        self.first_message_time = None
        self.connection_start = None
        
    async def run_diagnostic(self):
        print("🔍 DIAGNÓSTICO NINJATRADER WEBSOCKET")
        print("=" * 50)
        print("🎯 Verificando qué datos envía NinjaTrader...")
        print("📡 Conectando a 192.168.1.65:6789")
        print("\n⏱️ Presiona Ctrl+C después de 1-2 minutos para ver resultados\n")
        
        client = WebSocketClient(
            host="192.168.1.65",
            port=6789,
            csv_file="data/diagnostic_data.csv"
        )
        
        client.on_market_data = self.analyze_message
        
        self.connection_start = datetime.now()
        
        try:
            await client.run()
        except KeyboardInterrupt:
            await self.show_diagnostic_results()
    
    async def analyze_message(self, data: MarketData):
        """Analizar cada mensaje recibido"""
        self.total_messages += 1
        
        if self.first_message_time is None:
            self.first_message_time = datetime.now()
            print(f"✅ PRIMER MENSAJE RECIBIDO: {data.data_type}")
        
        if data.data_type == 'tick':
            self.tick_count += 1
            if self.tick_count <= 3:
                print(f"🔴 TICK #{self.tick_count}: {data.symbol} @ {data.price} - {data.timestamp.strftime('%H:%M:%S')}")
            elif self.tick_count == 4:
                print("   ... (más ticks - mostrando solo primeros 3)")
                
        elif data.data_type == 'bar':
            self.bar_count += 1
            self.last_bar_time = data.timestamp
            print(f"🟢 BAR #{self.bar_count}: {data.symbol}")
            print(f"    OHLC: {data.open_price}/{data.high_price}/{data.low_price}/{data.close_price}")
            print(f"    Time: {data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Volume: {data.volume}")
            
        elif data.data_type == 'historical_bar':
            self.historical_count += 1
            if self.historical_count <= 3 or self.historical_count % 100 == 0:
                print(f"🟡 HISTORICAL #{self.historical_count}: {data.timestamp.strftime('%H:%M:%S')}")
        else:
            self.other_count += 1
            print(f"⚪ OTHER: {data.data_type}")
        
        # Resumen cada 100 mensajes
        if self.total_messages % 100 == 0:
            print(f"\n📊 RESUMEN PARCIAL ({self.total_messages} mensajes):")
            print(f"   🔴 Ticks: {self.tick_count}")
            print(f"   🟢 Bars: {self.bar_count}")
            print(f"   🟡 Historical: {self.historical_count}")
            print("   " + "-" * 30)
    
    async def show_diagnostic_results(self):
        """Mostrar resultados del diagnóstico"""
        print("\n" + "=" * 60)
        print("📋 RESULTADOS DEL DIAGNÓSTICO")
        print("=" * 60)
        
        connection_time = (datetime.now() - self.connection_start).total_seconds()
        
        print(f"⏰ Tiempo de conexión: {connection_time:.1f} segundos")
        print(f"📨 Total mensajes: {self.total_messages}")
        print(f"🔴 Ticks recibidos: {self.tick_count}")
        print(f"🟢 Barras recibidas: {self.bar_count}")
        print(f"🟡 Barras históricas: {self.historical_count}")
        print(f"⚪ Otros mensajes: {self.other_count}")
        
        if self.last_bar_time:
            print(f"🕐 Última barra: {self.last_bar_time.strftime('%H:%M:%S')}")
        
        print("\n" + "=" * 60)
        print("🎯 DIAGNÓSTICO:")
        
        if self.total_messages == 0:
            print("❌ PROBLEMA CRÍTICO: No se recibieron mensajes")
            print("   ▶️ Verificar:")
            print("     1. NinjaTrader está ejecutándose")
            print("     2. Indicador está aplicado al gráfico")
            print("     3. WebSocket Port = 6789")
            print("     4. Firewall no bloquea puerto 6789")
            
        elif self.bar_count == 0:
            print("❌ PROBLEMA: No se recibieron barras")
            print("   ▶️ En NinjaTrader, verificar:")
            print("     1. PublishBars = TRUE")
            print("     2. Gráfico en timeframe 1 minuto")
            print("     3. Mercado está abierto/hay actividad")
            
            if self.tick_count > 0:
                print(f"   ✅ Se reciben ticks ({self.tick_count}) - conexión OK")
                print("   🔧 Solo necesitas habilitar PublishBars")
                
        elif self.bar_count > 0:
            print("✅ CONFIGURACIÓN CORRECTA")
            print(f"   📊 Se están recibiendo barras ({self.bar_count})")
            
            if self.tick_count > self.bar_count * 10:
                print("   ⚠️ SUGERENCIA: Deshabilitar PublishTicks para mejor rendimiento")
            
            bars_per_minute = self.bar_count / (connection_time / 60) if connection_time > 0 else 0
            print(f"   📈 Velocidad: {bars_per_minute:.1f} barras/minuto")
            
        print("\n" + "=" * 60)

async def main():
    diagnostic = NTDiagnostic()
    await diagnostic.run_diagnostic()

if __name__ == "__main__":
    asyncio.run(main()) 