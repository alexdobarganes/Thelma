import pandas as pd
from datetime import datetime, timedelta

def analyze_windowed_configuration(file_path, config_name):
    """Analyze windowed configuration and return projections"""
    df = pd.read_csv(file_path)
    df['origin_time'] = pd.to_datetime(df['origin_time'])
    
    # Get data range
    start_date = df['origin_time'].min()
    end_date = df['origin_time'].max()
    
    # Calculate statistics
    total_trades = len(df)
    total_pnl = df['pnl_dollars'].sum()
    avg_pnl_per_trade = df['pnl_dollars'].mean()
    win_rate = (df['pnl_dollars'] > 0).sum() / len(df) * 100
    
    # Calculate trading days
    trading_days = 0
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            trading_days += 1
        current_date += timedelta(days=1)
    
    # Calculate daily averages
    trades_per_day = total_trades / trading_days if trading_days > 0 else 0
    pnl_per_day = total_pnl / trading_days if trading_days > 0 else 0
    
    # Monthly projections (21 trading days per month)
    trading_days_per_month = 21
    trades_per_month = trades_per_day * trading_days_per_month
    pnl_per_month = pnl_per_day * trading_days_per_month
    
    # Risk metrics
    worst_trade = df['pnl_dollars'].min()
    best_trade = df['pnl_dollars'].max()
    
    # Analyze by time windows
    df['hour'] = df['origin_time'].dt.hour
    
    window_stats = []
    # Early morning (3-4 AM)
    early = df[(df['hour'] >= 3) & (df['hour'] < 4)]
    if len(early) > 0:
        window_stats.append({
            'window': '3-4 AM',
            'trades': len(early),
            'avg_pnl': early['pnl_dollars'].mean(),
            'win_rate': (early['pnl_dollars'] > 0).sum() / len(early) * 100
        })
    
    # Morning (9-11 AM)
    morning = df[(df['hour'] >= 9) & (df['hour'] < 11)]
    if len(morning) > 0:
        window_stats.append({
            'window': '9-11 AM',
            'trades': len(morning),
            'avg_pnl': morning['pnl_dollars'].mean(),
            'win_rate': (morning['pnl_dollars'] > 0).sum() / len(morning) * 100
        })
    
    # Afternoon (1:30-3 PM, hour 13-14)
    afternoon = df[(df['hour'] >= 13) & (df['hour'] < 15)]
    if len(afternoon) > 0:
        window_stats.append({
            'window': '1:30-3 PM',
            'trades': len(afternoon),
            'avg_pnl': afternoon['pnl_dollars'].mean(),
            'win_rate': (afternoon['pnl_dollars'] > 0).sum() / len(afternoon) * 100
        })
    
    return {
        'config_name': config_name,
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'win_rate': win_rate,
        'trades_per_day': trades_per_day,
        'pnl_per_day': pnl_per_day,
        'trades_per_month': trades_per_month,
        'pnl_per_month': pnl_per_month,
        'worst_trade': worst_trade,
        'best_trade': best_trade,
        'trading_days': trading_days,
        'window_stats': window_stats
    }

def main():
    print("🕐 ANÁLISIS DE ESTRATEGIAS CON VENTANAS ESPECÍFICAS")
    print("Ventanas: 3-4 AM, 9-11 AM, 1:30-3:00 PM")
    print("=" * 70)
    
    # Analyze configurations
    configs = [
        {
            'file': 'windowed_strategy.csv',
            'name': 'VENTANAS ESTÁNDAR',
            'desc': 'Drop 4.0pts en ventanas específicas'
        },
        {
            'file': 'windowed_conservative.csv', 
            'name': 'VENTANAS CONSERVADORA',
            'desc': 'Drop 6.0pts en ventanas específicas'
        }
    ]
    
    results = []
    for config in configs:
        try:
            result = analyze_windowed_configuration(config['file'], config['name'])
            results.append(result)
        except FileNotFoundError:
            print(f"⚠️ Archivo no encontrado: {config['file']}")
            continue
    
    # Display comparison
    print(f"📊 RESUMEN COMPARATIVO:")
    print("-" * 70)
    print(f"{'Configuración':<20} {'Win Rate':<10} {'Trades/Mes':<12} {'P&L/Mes':<15}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['config_name']:<20} {result['win_rate']:<9.1f}% {result['trades_per_month']:<11.0f} ${result['pnl_per_month']:<14,.0f}")
    
    print("-" * 70)
    
    # Detailed analysis
    for result in results:
        print(f"\n🔍 ANÁLISIS DETALLADO: {result['config_name']}")
        print(f"   📊 Histórico: {result['total_trades']:,} trades en {result['trading_days']} días")
        print(f"   💰 P&L total: ${result['total_pnl']:,.2f}")
        print(f"   📈 Win rate: {result['win_rate']:.1f}%")
        print(f"   📅 Promedios: {result['trades_per_day']:.1f} trades/día, ${result['pnl_per_day']:.2f}/día")
        print(f"   🗓️ PROYECCIÓN MENSUAL:")
        print(f"      💰 P&L estimado: ${result['pnl_per_month']:,.2f}")
        print(f"      📊 Trades estimados: {result['trades_per_month']:.0f}")
        print(f"      📈 P&L anual: ${result['pnl_per_month'] * 12:,.2f}")
        
        # Window breakdown
        print(f"   🕐 Rendimiento por ventana:")
        for window in result['window_stats']:
            print(f"      {window['window']:<10}: {window['trades']:3} trades, {window['win_rate']:5.1f}% win, ${window['avg_pnl']:6.0f} avg")
    
    # Compare with previous full-day configuration
    if results:
        best_windowed = max(results, key=lambda x: x['pnl_per_month'])
        
        print(f"\n🏆 MEJOR CONFIGURACIÓN CON VENTANAS:")
        print(f"   {best_windowed['config_name']}: ${best_windowed['pnl_per_month']:,.0f}/mes")
        
        print(f"\n📈 COMPARACIÓN CON CONFIGURACIÓN ANTERIOR:")
        print(f"   Anterior (todo el día): ~$14,140/mes")
        print(f"   Nueva (ventanas): ${best_windowed['pnl_per_month']:,.0f}/mes")
        
        improvement = ((best_windowed['pnl_per_month'] / 14140) - 1) * 100
        if improvement > 0:
            print(f"   🎯 Mejora: +{improvement:.1f}%")
        else:
            print(f"   ⚠️ Diferencia: {improvement:.1f}%")
        
        print(f"\n💡 VENTAJAS DE LAS VENTANAS ESPECÍFICAS:")
        print(f"   ✅ Mayor precisión en horas de alta volatilidad")
        print(f"   ✅ Evita ruido de horas de baja actividad") 
        print(f"   ✅ Concentra trading en momentos óptimos")
        print(f"   ✅ Reduce tiempo de monitoreo")
        
        # Risk analysis
        print(f"\n⚠️ ANÁLISIS DE RIESGO:")
        for result in results:
            max_drawdown = abs(result['worst_trade']) * result['trades_per_month'] * 0.08
            print(f"   {result['config_name']}: Máximo drawdown estimado ~${max_drawdown:.0f}/mes")

if __name__ == "__main__":
    main() 