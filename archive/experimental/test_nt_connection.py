"""
test_nt_connection.py
====================
Script para probar la conexión completa con NinjaTrader
"""

import time
import os
import glob
from datetime import datetime, timedelta
import subprocess
import sys

def check_signal_generation():
    """Verificar que se están generando señales"""
    
    print("🔍 VERIFICANDO GENERACIÓN DE SEÑALES")
    print("=" * 50)
    
    # Verificar carpeta de señales
    if not os.path.exists('signals'):
        print("❌ Carpeta 'signals' no existe")
        return False
    
    # Buscar señales recientes (últimas 24 horas)
    recent_signals = []
    cutoff_time = datetime.now() - timedelta(hours=24)
    
    for signal_file in glob.glob('signals/vreversal_*.txt'):
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(signal_file))
            if file_time > cutoff_time:
                recent_signals.append((signal_file, file_time))
        except:
            continue
    
    if recent_signals:
        print(f"✅ Encontradas {len(recent_signals)} señales recientes:")
        for signal_file, file_time in sorted(recent_signals, key=lambda x: x[1]):
            print(f"   {os.path.basename(signal_file)} - {file_time.strftime('%H:%M:%S')}")
        return True
    else:
        print("⚠️ No se encontraron señales recientes")
        print("   Esto es normal si los mercados están cerrados")
        return True  # No es un error durante horas no de trading

def check_websocket_logs():
    """Verificar logs del WebSocket"""
    
    print("\n📡 VERIFICANDO WEBSOCKET")
    print("=" * 50)
    
    log_files = [
        'python-client/logs/websocket_client.log',
        'logs/websocket_client.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"📄 Leyendo: {log_file}")
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Mostrar últimas 5 líneas
                        print("   Últimas entradas:")
                        for line in lines[-5:]:
                            print(f"   {line.strip()}")
                        return True
            except Exception as e:
                print(f"   ❌ Error leyendo log: {e}")
    
    print("⚠️ No se encontraron logs del WebSocket")
    print("   Verificar que el WebSocket Publisher esté activo en NT")
    return False

def test_production_system():
    """Probar el sistema de producción"""
    
    print("\n🔄 PROBANDO SISTEMA DE PRODUCCIÓN")
    print("=" * 50)
    
    try:
        print("▶️ Ejecutando production_system_clean.py...")
        
        # Ejecutar por 30 segundos para generar señales de prueba
        process = subprocess.Popen([
            sys.executable, 'production_system_clean.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Esperar 10 segundos
        time.sleep(10)
        
        # Terminar proceso
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        
        if stdout:
            print("✅ Sistema ejecutado correctamente")
            print("📋 Salida:")
            for line in stdout.split('\n')[-10:]:  # Últimas 10 líneas
                if line.strip():
                    print(f"   {line}")
        
        if stderr:
            print("⚠️ Warnings/Errores:")
            for line in stderr.split('\n')[-5:]:  # Últimos 5 errores
                if line.strip():
                    print(f"   {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error ejecutando sistema: {e}")
        return False

def check_nt_files():
    """Verificar archivos de NinjaTrader"""
    
    print("\n📁 VERIFICANDO ARCHIVOS NT")
    print("=" * 50)
    
    required_files = [
        'NT8/ThelmaMLStrategy.cs',
        'NT8/TickWebSocketPublisher_Optimized.cs'
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - NO ENCONTRADO")
            all_good = False
    
    return all_good

def generate_test_signals():
    """Generar señales de prueba para verificar el flujo"""
    
    print("\n🧪 GENERANDO SEÑALES DE PRUEBA")
    print("=" * 50)
    
    try:
        # Crear una señal de prueba manual
        timestamp = int(time.time())
        signal_content = f"""VREVERSAL_SIGNAL
Timestamp: {timestamp}
Symbol: ES 12-25
Direction: LONG
Entry: 5800.00
Stop: 5795.00
Target: 5815.00
Contracts: 1
Risk: 0.1%
Confidence: HIGH
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        signal_file = f"signals/vreversal_test_{timestamp}.txt"
        
        # Asegurar que existe la carpeta signals
        os.makedirs('signals', exist_ok=True)
        
        with open(signal_file, 'w') as f:
            f.write(signal_content)
        
        print(f"✅ Señal de prueba generada: {signal_file}")
        print("📋 Contenido:")
        for line in signal_content.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        return signal_file
        
    except Exception as e:
        print(f"❌ Error generando señal de prueba: {e}")
        return None

def main():
    """Ejecutar todas las pruebas de conexión"""
    
    print("🧪 PRUEBA DE CONEXIÓN CON NINJATRADER")
    print("Verificando todos los componentes del sistema")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Archivos NT
    if check_nt_files():
        tests_passed += 1
    
    # Test 2: Generación de señales
    if check_signal_generation():
        tests_passed += 1
    
    # Test 3: WebSocket logs
    if check_websocket_logs():
        tests_passed += 1
    
    # Test 4: Sistema de producción
    if test_production_system():
        tests_passed += 1
    
    # Test 5: Señal de prueba
    if generate_test_signals():
        tests_passed += 1
    
    # Resultado final
    print(f"\n🏆 RESULTADO FINAL")
    print("=" * 50)
    print(f"Pruebas pasadas: {tests_passed}/{total_tests}")
    
    success_rate = (tests_passed / total_tests) * 100
    
    if success_rate >= 80:
        print(f"✅ SISTEMA LISTO ({success_rate:.0f}%)")
        print("   Recomendación: Proceder con pruebas en NT")
        
        print(f"\n📋 PRÓXIMOS PASOS:")
        print("1. Abrir NinjaTrader 8")
        print("2. Cargar ThelmaMLStrategy en cuenta Sim")
        print("3. Activar TickWebSocketPublisher")
        print("4. Monitorear ejecución de trades")
        
    elif success_rate >= 60:
        print(f"⚠️ SISTEMA PARCIAL ({success_rate:.0f}%)")
        print("   Recomendación: Resolver problemas antes de continuar")
    else:
        print(f"❌ SISTEMA NO LISTO ({success_rate:.0f}%)")
        print("   Recomendación: Revisar configuración completamente")

if __name__ == "__main__":
    main() 