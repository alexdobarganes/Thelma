"""
playback_launcher.py
===================
Script para lanzar y gestionar testing en modo Playback.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    print("🎮" * 20)
    print("🎮 THELMA V-REVERSAL PLAYBACK LAUNCHER")
    print("🎮" * 20)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_system_ready():
    """Verificar que todo esté listo para playback"""
    
    print("🔍 VERIFICANDO SISTEMA...")
    print("=" * 40)
    
    checks = []
    
    # 1. Verificar carpeta signals/playback
    playback_dir = "signals/playback"
    if os.path.exists(playback_dir):
        signal_files = [f for f in os.listdir(playback_dir) if f.endswith('.txt')]
        print(f"✅ Directorio playback: {len(signal_files)} señales")
        checks.append(True)
    else:
        print(f"❌ Directorio playback no existe: {playback_dir}")
        checks.append(False)
    
    # 2. Verificar VReversalAutoTrader.cs
    nt_file = "NT8/VReversalAutoTrader.cs"
    if os.path.exists(nt_file):
        print(f"✅ Strategy disponible: {nt_file}")
        checks.append(True)
    else:
        print(f"❌ Strategy no encontrado: {nt_file}")
        checks.append(False)
    
    # 3. Verificar exact model
    model_file = "exact_vreversal_model.py"
    if os.path.exists(model_file):
        print(f"✅ Modelo validado: {model_file}")
        checks.append(True)
    else:
        print(f"❌ Modelo no encontrado: {model_file}")
        checks.append(False)
    
    print()
    if all(checks):
        print("🟢 SISTEMA LISTO PARA PLAYBACK")
        return True
    else:
        print("🔴 SISTEMA NO ESTÁ LISTO")
        return False

def generate_fresh_signals():
    """Generar nuevas señales para testing"""
    
    print("🎯 GENERANDO SEÑALES FRESCAS...")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, "simple_playback_generator.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Señales generadas exitosamente")
            return True
        else:
            print(f"❌ Error generando señales: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando generador: {e}")
        return False

def show_playbook_instructions():
    """Mostrar instrucciones para NT Playback"""
    
    print()
    print("📋 INSTRUCCIONES PARA NINJATRADER")
    print("=" * 50)
    
    instructions = [
        "1. 🚀 Abrir NinjaTrader 8",
        "2. 🔌 Tools → Playback Connection",
        "3. ⚙️ Configurar:",
        "   • Market Data: ES futures",
        "   • Speed: 100x (o menor para observar)",
        "   • Start Date: Cualquier día reciente",
        "4. 📊 New → Chart → ES (1 minute)",
        "5. 🎯 Strategies → VReversalAutoTrader",
        "6. ✅ Verificar signal path está en playback mode",
        "7. ▶️ Connect to Playback",
        "8. 🎮 Start Playback",
        "9. 👀 Monitorear Messages tab para logs"
    ]
    
    for instruction in instructions:
        print(f"  {instruction}")
    
    print()
    print("🎯 RESULTADOS ESPERADOS:")
    print(f"  • Win Rate: 91.8%")
    print(f"  • Señales procesadas: 5/5")
    print(f"  • P&L target: +$500")
    print()

def monitor_playback():
    """Función para monitoreo manual"""
    
    print("📊 MODO MONITOREO ACTIVADO")
    print("=" * 40)
    print("Presiona CTRL+C para salir")
    print()
    
    try:
        while True:
            # Verificar si quedan señales sin procesar
            playback_dir = "signals/playback"
            if os.path.exists(playback_dir):
                txt_files = [f for f in os.listdir(playback_dir) if f.endswith('.txt')]
                processed_dir = os.path.join(playback_dir, "processed")
                processed_files = []
                if os.path.exists(processed_dir):
                    processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.txt')]
                
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Señales pendientes: {len(txt_files)} | Procesadas: {len(processed_files)}")
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoreo terminado")

def main():
    """Función principal del launcher"""
    
    print_banner()
    
    if not check_system_ready():
        print("\n🔧 Ejecuta setup primero:")
        print("   python simple_playback_generator.py")
        return
    
    print("\n🎮 OPCIONES DISPONIBLES:")
    print("=" * 30)
    print("1. 🎯 Generar señales frescas")
    print("2. 📋 Mostrar instrucciones NT")
    print("3. 📊 Monitorear playback")
    print("4. 🔍 Verificar estado")
    print("5. 🚪 Salir")
    
    while True:
        choice = input("\n🎮 Selecciona opción (1-5): ").strip()
        
        if choice == "1":
            generate_fresh_signals()
        elif choice == "2":
            show_playbook_instructions()
        elif choice == "3":
            monitor_playback()
        elif choice == "4":
            check_system_ready()
        elif choice == "5":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida. Usa 1-5.")

if __name__ == "__main__":
    main() 