"""
simple_validation.py
====================
Validación simple usando resultados existentes
"""

import json
import os
from datetime import datetime

def validate_existing_results():
    """Validar usando resultados existentes"""
    
    print("🧪 VALIDACIÓN SIMPLE DEL MODELO")
    print("=" * 50)
    
    # Buscar archivos de resultados existentes
    result_files = [
        'exact_vreversal_results.json',
        'production_report_final.json',
        'comprehensive_test_results.json'
    ]
    
    results = None
    
    for file in result_files:
        if os.path.exists(file):
            print(f"📊 Leyendo resultados de: {file}")
            with open(file, 'r') as f:
                results = json.load(f)
            break
    
    if not results:
        print("❌ No se encontraron archivos de resultados")
        return None
    
    # Extraer métricas clave
    if 'results' in results:
        # Formato del exact_vreversal_results.json
        data = results['results']
        win_rate = data.get('win_rate', 0)
        avg_daily_pnl = data.get('avg_daily_pnl', 0)
        total_patterns = data.get('total_patterns', 0)
        monthly_pnl = data.get('monthly_pnl', avg_daily_pnl * 21)
        position_size = results.get('position_size', 1)
        
        # Escalar a 3 contratos si es necesario
        if position_size != 3:
            scaling_factor = 3 / position_size
            avg_daily_pnl *= scaling_factor
            monthly_pnl *= scaling_factor
            
    elif 'win_rate' in results:
        # Formato directo
        win_rate = results['win_rate']
        avg_daily_pnl = results.get('avg_daily_pnl', 0)
        total_patterns = results.get('total_patterns', 0)
        monthly_pnl = results.get('monthly_pnl', avg_daily_pnl * 21)
    else:
        print("❌ Formato de resultados no reconocido")
        return None
    
    print(f"\n✅ RESULTADOS VALIDADOS:")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   P&L Diario: ${avg_daily_pnl:.2f}")
    print(f"   P&L Mensual: ${monthly_pnl:.2f}")
    print(f"   Total Trades: {total_patterns}")
    
    # Análisis de target
    target_daily = 2300
    target_achieved = avg_daily_pnl >= target_daily
    
    print(f"\n🎯 ANÁLISIS DE TARGET:")
    print(f"   Target: ${target_daily}/día")
    print(f"   Actual: ${avg_daily_pnl:.2f}/día")
    
    if target_achieved:
        print(f"   ✅ TARGET ALCANZADO!")
        excess = avg_daily_pnl - target_daily
        print(f"   Exceso: ${excess:.2f}/día")
    else:
        gap = target_daily - avg_daily_pnl
        print(f"   ⚠️ Gap: ${gap:.2f}/día")
        scaling_factor = target_daily / avg_daily_pnl if avg_daily_pnl > 0 else 0
        print(f"   Factor de escalado necesario: {scaling_factor:.1f}x")
    
    # Escenarios de escalado
    print(f"\n🔧 ESCENARIOS DE ESCALADO:")
    
    base_pnl_per_contract = avg_daily_pnl / 3  # Asumiendo 3 contratos base
    
    scenarios = [
        (1, "Conservador"),
        (2, "Moderado"), 
        (3, "Base"),
        (4, "Agresivo"),
        (5, "Muy Agresivo")
    ]
    
    for contracts, risk_level in scenarios:
        scaled_pnl = base_pnl_per_contract * contracts
        status = "✅" if scaled_pnl >= target_daily else "⚠️"
        print(f"   {contracts} contratos ({risk_level}): ${scaled_pnl:.2f}/día {status}")
    
    # Recomendación final
    optimal_contracts = target_daily / base_pnl_per_contract if base_pnl_per_contract > 0 else 0
    
    print(f"\n📋 RECOMENDACIÓN:")
    print(f"   Contratos óptimos: {optimal_contracts:.1f}")
    
    if optimal_contracts <= 3:
        print(f"   ✅ BAJO RIESGO - Usar {int(optimal_contracts)} contratos")
    elif optimal_contracts <= 5:
        print(f"   ⚠️ RIESGO MODERADO - Usar {int(optimal_contracts)} contratos")
    else:
        print(f"   ❌ ALTO RIESGO - {optimal_contracts:.1f} contratos")
        print(f"   Considerar optimización antes de escalar")
    
    # Veredicto de calidad
    print(f"\n🏆 CALIDAD DEL MODELO:")
    
    quality_score = 0
    
    if win_rate >= 90:
        print(f"   ✅ Win Rate Excelente (≥90%)")
        quality_score += 3
    elif win_rate >= 85:
        print(f"   ✅ Win Rate Bueno (≥85%)")
        quality_score += 2
    elif win_rate >= 80:
        print(f"   ⚠️ Win Rate Aceptable (≥80%)")
        quality_score += 1
    else:
        print(f"   ❌ Win Rate Bajo (<80%)")
    
    if avg_daily_pnl >= 2000:
        print(f"   ✅ P&L Excelente (≥$2000/día)")
        quality_score += 3
    elif avg_daily_pnl >= 1500:
        print(f"   ✅ P&L Bueno (≥$1500/día)")
        quality_score += 2
    elif avg_daily_pnl >= 1000:
        print(f"   ⚠️ P&L Aceptable (≥$1000/día)")
        quality_score += 1
    else:
        print(f"   ❌ P&L Bajo (<$1000/día)")
    
    if total_patterns >= 2000:
        print(f"   ✅ Volumen Excelente (≥2000 trades)")
        quality_score += 3
    elif total_patterns >= 1000:
        print(f"   ✅ Volumen Bueno (≥1000 trades)")
        quality_score += 2
    elif total_patterns >= 500:
        print(f"   ⚠️ Volumen Aceptable (≥500 trades)")
        quality_score += 1
    else:
        print(f"   ❌ Volumen Bajo (<500 trades)")
    
    # Veredicto final
    max_score = 9
    quality_percentage = (quality_score / max_score) * 100
    
    print(f"\n🎖️ PUNTUACIÓN TOTAL: {quality_score}/{max_score} ({quality_percentage:.0f}%)")
    
    if quality_percentage >= 80:
        print(f"   🎉 MODELO EXCELENTE - Listo para producción")
        recommendation = "PROCEDER con trading en vivo"
    elif quality_percentage >= 60:
        print(f"   ✅ MODELO BUENO - Considerar paper trading")
        recommendation = "Paper trading recomendado antes de ir en vivo"
    elif quality_percentage >= 40:
        print(f"   ⚠️ MODELO REGULAR - Necesita optimización")
        recommendation = "Optimizar parámetros antes de usar"
    else:
        print(f"   ❌ MODELO POBRE - Requiere revisión completa")
        recommendation = "Revisar estrategia completamente"
    
    print(f"   Recomendación: {recommendation}")
    
    return {
        'win_rate': win_rate,
        'avg_daily_pnl': avg_daily_pnl,
        'monthly_pnl': monthly_pnl,
        'total_patterns': total_patterns,
        'target_achieved': target_achieved,
        'optimal_contracts': optimal_contracts,
        'quality_score': quality_score,
        'quality_percentage': quality_percentage,
        'recommendation': recommendation
    }

def main():
    try:
        print("🚀 VALIDACIÓN SIMPLE DEL MODELO V-REVERSAL")
        print("Análisis rápido usando resultados existentes")
        print("=" * 60)
        
        results = validate_existing_results()
        
        if results:
            # Guardar reporte
            report = {
                'timestamp': datetime.now().isoformat(),
                'validation_results': results,
                'status': 'COMPLETED'
            }
            
            with open('simple_validation_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\n💾 Reporte guardado en: simple_validation_report.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 