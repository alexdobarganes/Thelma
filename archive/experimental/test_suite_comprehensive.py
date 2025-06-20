"""
test_suite_comprehensive.py
===========================
Suite de pruebas completa para validar el modelo V-reversal en diferentes condiciones.
Incluye backtesting por períodos, análisis de robustez, y validación de parámetros.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from production_system_clean import ProductionVReversalSystem
from exact_vreversal_model import load_data, detect_exact_patterns, analyze_exact_results

def split_data_by_periods(df, num_periods=6):
    """Dividir datos en períodos para análisis de robustez"""
    
    df_sorted = df.sort_values('Datetime').reset_index(drop=True)
    period_size = len(df_sorted) // num_periods
    
    periods = []
    for i in range(num_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < num_periods - 1 else len(df_sorted)
        
        period_df = df_sorted.iloc[start_idx:end_idx].reset_index(drop=True)
        start_date = period_df['Datetime'].min().strftime('%Y-%m-%d')
        end_date = period_df['Datetime'].max().strftime('%Y-%m-%d')
        
        periods.append({
            'period': i + 1,
            'data': period_df,
            'start_date': start_date,
            'end_date': end_date,
            'bars': len(period_df)
        })
    
    return periods

def test_model_robustness(df, position_size=3):
    """Probar robustez del modelo en diferentes períodos"""
    
    print("🧪 PRUEBA DE ROBUSTEZ DEL MODELO")
    print("=" * 50)
    
    periods = split_data_by_periods(df, num_periods=6)
    results = []
    
    for period in periods:
        print(f"\n📊 Período {period['period']}: {period['start_date']} → {period['end_date']}")
        print(f"   Bars: {period['bars']:,}")
        
        try:
            # Ejecutar detector en este período
            successful, failed = detect_exact_patterns(period['data'], position_size)
            
            if successful or failed:
                period_results = analyze_exact_results(successful, failed, position_size)
                period_results['period'] = period['period']
                period_results['start_date'] = period['start_date']
                period_results['end_date'] = period['end_date']
                period_results['bars'] = period['bars']
                
                results.append(period_results)
                
                print(f"   Win Rate: {period_results['win_rate']:.1f}%")
                print(f"   Daily P&L: ${period_results['avg_daily_pnl']:.2f}")
                print(f"   Total Trades: {period_results['total_patterns']}")
            else:
                print("   ⚠️ No patterns detected in this period")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return results

def test_parameter_sensitivity(df):
    """Probar sensibilidad a diferentes parámetros"""
    
    print("\n🔧 PRUEBA DE SENSIBILIDAD DE PARÁMETROS")
    print("=" * 50)
    
    # Diferentes configuraciones para probar
    configs = [
        {'drop_threshold': 3.5, 'position_size': 3, 'name': 'Drop 3.5pts'},
        {'drop_threshold': 4.0, 'position_size': 3, 'name': 'Drop 4.0pts (Base)'},
        {'drop_threshold': 4.5, 'position_size': 3, 'name': 'Drop 4.5pts'},
        {'drop_threshold': 4.0, 'position_size': 2, 'name': '2 Contratos'},
        {'drop_threshold': 4.0, 'position_size': 4, 'name': '4 Contratos'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Probando: {config['name']}")
        
        try:
            # Crear detector temporal con parámetros modificados
            from exact_vreversal_model import ValidatedVReversalDetector
            
            # Modificar temporalmente los parámetros
            original_threshold = ValidatedVReversalDetector().drop_threshold
            ValidatedVReversalDetector.drop_threshold = config['drop_threshold']
            
            successful, failed = detect_exact_patterns(df, config['position_size'])
            
            # Restaurar parámetro original
            ValidatedVReversalDetector.drop_threshold = original_threshold
            
            if successful or failed:
                config_results = analyze_exact_results(successful, failed, config['position_size'])
                config_results['config_name'] = config['name']
                config_results['drop_threshold'] = config['drop_threshold']
                config_results['position_size'] = config['position_size']
                
                results.append(config_results)
                
                print(f"   Win Rate: {config_results['win_rate']:.1f}%")
                print(f"   Daily P&L: ${config_results['avg_daily_pnl']:.2f}")
                print(f"   Total Trades: {config_results['total_patterns']}")
                
                # Verificar si alcanza target
                if config_results['avg_daily_pnl'] >= 2300:
                    print(f"   ✅ TARGET ALCANZADO!")
                else:
                    gap = 2300 - config_results['avg_daily_pnl']
                    print(f"   ⚠️ Gap: ${gap:.0f}/día")
            else:
                print("   ⚠️ No patterns detected")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return results

def test_market_conditions(df):
    """Probar performance en diferentes condiciones de mercado"""
    
    print("\n📈 PRUEBA POR CONDICIONES DE MERCADO")
    print("=" * 50)
    
    # Calcular volatilidad diaria
    df['date'] = df['Datetime'].dt.date
    daily_stats = df.groupby('date').agg({
        'High': 'max',
        'Low': 'min',
        'Close': ['first', 'last']
    }).reset_index()
    
    daily_stats.columns = ['date', 'daily_high', 'daily_low', 'open', 'close']
    daily_stats['daily_range'] = daily_stats['daily_high'] - daily_stats['daily_low']
    daily_stats['daily_return'] = (daily_stats['close'] - daily_stats['open']) / daily_stats['open']
    
    # Clasificar días por volatilidad
    volatility_threshold = daily_stats['daily_range'].quantile(0.67)
    low_vol_days = daily_stats[daily_stats['daily_range'] <= daily_stats['daily_range'].quantile(0.33)]['date']
    high_vol_days = daily_stats[daily_stats['daily_range'] >= volatility_threshold]['date']
    
    conditions = [
        {'name': 'Baja Volatilidad', 'filter': df['date'].isin(low_vol_days)},
        {'name': 'Alta Volatilidad', 'filter': df['date'].isin(high_vol_days)},
        {'name': 'Días Alcistas', 'filter': df['date'].isin(daily_stats[daily_stats['daily_return'] > 0.005]['date'])},
        {'name': 'Días Bajistas', 'filter': df['date'].isin(daily_stats[daily_stats['daily_return'] < -0.005]['date'])},
    ]
    
    results = []
    
    for condition in conditions:
        filtered_df = df[condition['filter']].reset_index(drop=True)
        
        print(f"\n📊 Condición: {condition['name']}")
        print(f"   Bars: {len(filtered_df):,}")
        
        if len(filtered_df) > 1000:  # Suficientes datos para análisis
            try:
                successful, failed = detect_exact_patterns(filtered_df, position_size=3)
                
                if successful or failed:
                    condition_results = analyze_exact_results(successful, failed, 3)
                    condition_results['condition'] = condition['name']
                    condition_results['bars'] = len(filtered_df)
                    
                    results.append(condition_results)
                    
                    print(f"   Win Rate: {condition_results['win_rate']:.1f}%")
                    print(f"   Daily P&L: ${condition_results['avg_daily_pnl']:.2f}")
                    print(f"   Total Trades: {condition_results['total_patterns']}")
                else:
                    print("   ⚠️ No patterns detected")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print("   ⚠️ Datos insuficientes para análisis")
    
    return results

def test_walk_forward_validation(df, num_folds=5):
    """Validación walk-forward para simular trading en tiempo real"""
    
    print("\n⏭️ VALIDACIÓN WALK-FORWARD")
    print("=" * 50)
    
    df_sorted = df.sort_values('Datetime').reset_index(drop=True)
    fold_size = len(df_sorted) // num_folds
    
    results = []
    
    for i in range(num_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(df_sorted)
        
        fold_df = df_sorted.iloc[start_idx:end_idx].reset_index(drop=True)
        start_date = fold_df['Datetime'].min().strftime('%Y-%m-%d')
        end_date = fold_df['Datetime'].max().strftime('%Y-%m-%d')
        
        print(f"\n📊 Fold {i+1}/{num_folds}: {start_date} → {end_date}")
        
        try:
            successful, failed = detect_exact_patterns(fold_df, position_size=3)
            
            if successful or failed:
                fold_results = analyze_exact_results(successful, failed, 3)
                fold_results['fold'] = i + 1
                fold_results['start_date'] = start_date
                fold_results['end_date'] = end_date
                
                results.append(fold_results)
                
                print(f"   Win Rate: {fold_results['win_rate']:.1f}%")
                print(f"   Daily P&L: ${fold_results['avg_daily_pnl']:.2f}")
                print(f"   Total Trades: {fold_results['total_patterns']}")
                
                # Verificar consistencia
                if fold_results['win_rate'] >= 85:  # Threshold de consistencia
                    print(f"   ✅ CONSISTENTE")
                else:
                    print(f"   ⚠️ INCONSISTENTE")
            else:
                print("   ⚠️ No patterns detected")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return results

def generate_test_report(robustness_results, parameter_results, market_results, walkforward_results):
    """Generar reporte completo de pruebas"""
    
    print("\n📋 REPORTE COMPLETO DE PRUEBAS")
    print("=" * 60)
    
    # Análisis de robustez
    if robustness_results:
        win_rates = [r['win_rate'] for r in robustness_results]
        daily_pnls = [r['avg_daily_pnl'] for r in robustness_results]
        
        print(f"\n🧪 ROBUSTEZ POR PERÍODOS:")
        print(f"   Win Rate - Promedio: {np.mean(win_rates):.1f}% ± {np.std(win_rates):.1f}%")
        print(f"   Win Rate - Rango: {np.min(win_rates):.1f}% - {np.max(win_rates):.1f}%")
        print(f"   Daily P&L - Promedio: ${np.mean(daily_pnls):.2f} ± ${np.std(daily_pnls):.2f}")
        print(f"   Daily P&L - Rango: ${np.min(daily_pnls):.2f} - ${np.max(daily_pnls):.2f}")
        
        consistent_periods = sum(1 for r in robustness_results if r['win_rate'] >= 85)
        print(f"   Períodos Consistentes: {consistent_periods}/{len(robustness_results)} ({consistent_periods/len(robustness_results)*100:.1f}%)")
    
    # Análisis de parámetros
    if parameter_results:
        print(f"\n🔧 SENSIBILIDAD DE PARÁMETROS:")
        for result in parameter_results:
            status = "✅" if result['avg_daily_pnl'] >= 2300 else "⚠️"
            print(f"   {result['config_name']}: {result['win_rate']:.1f}% win, ${result['avg_daily_pnl']:.0f}/día {status}")
    
    # Análisis por condiciones de mercado
    if market_results:
        print(f"\n📈 PERFORMANCE POR CONDICIONES:")
        for result in market_results:
            print(f"   {result['condition']}: {result['win_rate']:.1f}% win, ${result['avg_daily_pnl']:.0f}/día")
    
    # Validación walk-forward
    if walkforward_results:
        wf_win_rates = [r['win_rate'] for r in walkforward_results]
        wf_daily_pnls = [r['avg_daily_pnl'] for r in walkforward_results]
        
        print(f"\n⏭️ VALIDACIÓN WALK-FORWARD:")
        print(f"   Win Rate Consistencia: {np.mean(wf_win_rates):.1f}% ± {np.std(wf_win_rates):.1f}%")
        print(f"   P&L Consistencia: ${np.mean(wf_daily_pnls):.2f} ± ${np.std(wf_daily_pnls):.2f}")
        
        consistent_folds = sum(1 for r in walkforward_results if r['win_rate'] >= 85)
        print(f"   Folds Consistentes: {consistent_folds}/{len(walkforward_results)} ({consistent_folds/len(walkforward_results)*100:.1f}%)")
    
    # Veredicto final
    print(f"\n🏆 VEREDICTO FINAL:")
    
    # Criterios de aprobación
    criteria_passed = 0
    total_criteria = 4
    
    if robustness_results and np.mean([r['win_rate'] for r in robustness_results]) >= 85:
        print("   ✅ Robustez: Win rate consistente >85%")
        criteria_passed += 1
    else:
        print("   ❌ Robustez: Win rate inconsistente")
    
    if parameter_results and any(r['avg_daily_pnl'] >= 2300 for r in parameter_results):
        print("   ✅ Parámetros: Target alcanzable con múltiples configuraciones")
        criteria_passed += 1
    else:
        print("   ❌ Parámetros: Target no robusto")
    
    if market_results and all(r['win_rate'] >= 70 for r in market_results):
        print("   ✅ Condiciones: Performance estable en diferentes mercados")
        criteria_passed += 1
    else:
        print("   ❌ Condiciones: Performance inconsistente")
    
    if walkforward_results and np.std([r['win_rate'] for r in walkforward_results]) <= 10:
        print("   ✅ Consistencia: Baja variabilidad temporal")
        criteria_passed += 1
    else:
        print("   ❌ Consistencia: Alta variabilidad temporal")
    
    success_rate = criteria_passed / total_criteria * 100
    
    if success_rate >= 75:
        print(f"\n🎉 MODELO APROBADO PARA PRODUCCIÓN ({success_rate:.0f}% criterios)")
        print("   Recomendación: PROCEDER con trading en vivo")
    elif success_rate >= 50:
        print(f"\n⚠️ MODELO CONDICIONAL ({success_rate:.0f}% criterios)")
        print("   Recomendación: Paper trading adicional requerido")
    else:
        print(f"\n❌ MODELO NO APROBADO ({success_rate:.0f}% criterios)")
        print("   Recomendación: Optimización adicional requerida")

def main():
    try:
        print("🧪 SUITE DE PRUEBAS COMPLETA - MODELO V-REVERSAL")
        print("Validación exhaustiva para trading en producción")
        print("=" * 60)
        
        # Cargar datos
        df = load_data("data/raw/es_1m/market_data.csv")
        
        # Ejecutar todas las pruebas
        print("\n🚀 Iniciando suite de pruebas...")
        
        # 1. Prueba de robustez por períodos
        robustness_results = test_model_robustness(df)
        
        # 2. Prueba de sensibilidad de parámetros
        parameter_results = test_parameter_sensitivity(df)
        
        # 3. Prueba por condiciones de mercado
        market_results = test_market_conditions(df)
        
        # 4. Validación walk-forward
        walkforward_results = test_walk_forward_validation(df)
        
        # 5. Generar reporte final
        generate_test_report(robustness_results, parameter_results, market_results, walkforward_results)
        
        # Guardar resultados
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'robustness_results': robustness_results,
            'parameter_results': parameter_results,
            'market_results': market_results,
            'walkforward_results': walkforward_results
        }
        
        with open('comprehensive_test_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 Resultados guardados en: comprehensive_test_results.json")
        
    except Exception as e:
        print(f"❌ Error en suite de pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 