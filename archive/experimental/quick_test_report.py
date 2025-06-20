"""
quick_test_report.py
====================
Reporte rápido de pruebas para validar el modelo V-reversal
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def quick_validation_test():
    """Prueba rápida de validación del modelo"""
    
    print("🚀 PRUEBA RÁPIDA DE VALIDACIÓN")
    print("=" * 50)
    
    try:
        # Ejecutar el modelo exacto
        from exact_vreversal_model import main as run_exact_model
        
        print("📊 Ejecutando modelo exacto...")
        run_exact_model()
        
        # Leer resultados
        with open('exact_vreversal_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"\n✅ RESULTADOS OBTENIDOS:")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   P&L Diario: ${results['avg_daily_pnl']:.2f}")
        print(f"   P&L Mensual: ${results['monthly_pnl']:.2f}")
        print(f"   Total Trades: {results['total_patterns']}")
        
        # Verificar target
        target_daily = 2300
        target_achieved = results['avg_daily_pnl'] >= target_daily
        
        print(f"\n🎯 ANÁLISIS DE TARGET:")
        print(f"   Target: ${target_daily}/día")
        print(f"   Actual: ${results['avg_daily_pnl']:.2f}/día")
        
        if target_achieved:
            print(f"   ✅ TARGET ALCANZADO!")
            excess = results['avg_daily_pnl'] - target_daily
            print(f"   Exceso: ${excess:.2f}/día")
        else:
            gap = target_daily - results['avg_daily_pnl']
            print(f"   ⚠️ Gap: ${gap:.2f}/día")
            scaling_factor = target_daily / results['avg_daily_pnl']
            print(f"   Factor de escalado necesario: {scaling_factor:.1f}x")
        
        # Verificar consistencia
        print(f"\n📈 VERIFICACIÓN DE CONSISTENCIA:")
        if results['win_rate'] >= 90:
            print(f"   ✅ Win Rate excelente (>90%)")
        elif results['win_rate'] >= 85:
            print(f"   ✅ Win Rate bueno (>85%)")
        else:
            print(f"   ⚠️ Win Rate bajo (<85%)")
        
        # Proyecciones
        monthly_projection = results['avg_daily_pnl'] * 21  # 21 días trading
        annual_projection = monthly_projection * 12
        
        print(f"\n📊 PROYECCIONES:")
        print(f"   Mensual: ${monthly_projection:,.2f}")
        print(f"   Anual: ${annual_projection:,.2f}")
        
        # Veredicto final
        print(f"\n🏆 VEREDICTO FINAL:")
        
        criteria_met = 0
        total_criteria = 3
        
        if results['win_rate'] >= 85:
            criteria_met += 1
            
        if results['avg_daily_pnl'] >= 1500:  # Al menos 65% del target
            criteria_met += 1
            
        if results['total_patterns'] >= 1000:  # Suficientes trades para validación
            criteria_met += 1
        
        success_rate = (criteria_met / total_criteria) * 100
        
        if success_rate >= 100:
            print(f"   🎉 EXCELENTE ({success_rate:.0f}% criterios)")
            print(f"   Recomendación: PROCEDER con trading en vivo")
        elif success_rate >= 67:
            print(f"   ✅ BUENO ({success_rate:.0f}% criterios)")
            print(f"   Recomendación: Considerar paper trading adicional")
        else:
            print(f"   ⚠️ NECESITA MEJORAS ({success_rate:.0f}% criterios)")
            print(f"   Recomendación: Optimización requerida")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return None

def test_scaling_scenarios():
    """Probar diferentes escenarios de escalado"""
    
    print(f"\n🔧 PRUEBA DE ESCALADO")
    print("=" * 50)
    
    try:
        # Leer resultados base
        with open('exact_vreversal_results.json', 'r') as f:
            base_results = json.load(f)
        
        base_daily_pnl = base_results['avg_daily_pnl']
        
        # Escenarios de escalado
        scenarios = [
            {'contracts': 1, 'name': '1 Contrato (Conservador)'},
            {'contracts': 2, 'name': '2 Contratos (Moderado)'},
            {'contracts': 3, 'name': '3 Contratos (Base)'},
            {'contracts': 4, 'name': '4 Contratos (Agresivo)'},
            {'contracts': 5, 'name': '5 Contratos (Muy Agresivo)'}
        ]
        
        target_daily = 2300
        
        print(f"Base P&L con 3 contratos: ${base_daily_pnl:.2f}/día")
        print(f"Target: ${target_daily}/día")
        print()
        
        for scenario in scenarios:
            # Calcular P&L escalado
            scaling_factor = scenario['contracts'] / 3  # Base es 3 contratos
            scaled_pnl = base_daily_pnl * scaling_factor
            
            # Verificar si alcanza target
            target_achieved = scaled_pnl >= target_daily
            status = "✅" if target_achieved else "⚠️"
            
            print(f"{scenario['name']}: ${scaled_pnl:.2f}/día {status}")
            
            if target_achieved:
                excess = scaled_pnl - target_daily
                print(f"   Exceso: ${excess:.2f}/día")
            else:
                gap = target_daily - scaled_pnl
                print(f"   Gap: ${gap:.2f}/día")
            
            # Proyección mensual
            monthly = scaled_pnl * 21
            print(f"   Mensual: ${monthly:,.2f}")
            print()
        
        # Recomendación
        optimal_contracts = target_daily / (base_daily_pnl / 3)
        
        print(f"📋 RECOMENDACIÓN:")
        print(f"   Contratos óptimos para ${target_daily}/día: {optimal_contracts:.1f}")
        
        if optimal_contracts <= 3:
            print(f"   ✅ CONSERVADOR: Usar {int(np.ceil(optimal_contracts))} contratos")
        elif optimal_contracts <= 5:
            print(f"   ⚠️ MODERADO: Usar {int(np.ceil(optimal_contracts))} contratos")
        else:
            print(f"   ❌ RIESGOSO: {optimal_contracts:.1f} contratos es muy agresivo")
            print(f"   Considerar optimización de parámetros primero")
        
    except Exception as e:
        print(f"❌ Error en escalado: {e}")

def main():
    try:
        print("🧪 REPORTE RÁPIDO DE PRUEBAS")
        print("Validación eficiente del modelo V-reversal")
        print("=" * 60)
        
        # Ejecutar validación rápida
        results = quick_validation_test()
        
        if results:
            # Probar escenarios de escalado
            test_scaling_scenarios()
            
            # Guardar reporte
            report = {
                'timestamp': datetime.now().isoformat(),
                'validation_results': results,
                'target_daily': 2300,
                'target_achieved': results['avg_daily_pnl'] >= 2300,
                'scaling_factor_needed': 2300 / results['avg_daily_pnl'],
                'recommended_contracts': 2300 / (results['avg_daily_pnl'] / 3)
            }
            
            with open('quick_test_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\n💾 Reporte guardado en: quick_test_report.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 