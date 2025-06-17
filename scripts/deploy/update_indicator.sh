#!/bin/bash

echo "======================================"
echo "  Actualizando Indicador en NinjaTrader"
echo "======================================"

# Configuración de rutas
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="$SCRIPT_DIR/NT8/TickWebSocketPublisher_Optimized.cs"

# Convertir ruta de Windows para Git Bash
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    TARGET_DIR="$(cygpath -u "$USERPROFILE")/Documents/NinjaTrader 8/bin/Custom/Indicators"
else
    TARGET_DIR="$HOME/Documents/NinjaTrader 8/bin/Custom/Indicators"
fi

TARGET_FILE="$TARGET_DIR/TBOTTickWebSocketPublisherOptimized.cs"

echo ""
echo "🔍 Verificando archivos..."

if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ ERROR: No se encuentra el archivo fuente:"
    echo "   $SOURCE_FILE"
    read -p "Presiona Enter para continuar..."
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ ERROR: No se encuentra el directorio de NinjaTrader:"
    echo "   $TARGET_DIR"
    echo ""
    echo "💡 Verifica que NinjaTrader 8 esté instalado correctamente"
    read -p "Presiona Enter para continuar..."
    exit 1
fi

echo "✅ Archivo fuente encontrado"
echo "✅ Directorio de NinjaTrader encontrado"

echo ""
echo "📋 Copiando archivo actualizado..."

# Copiar archivo
cp "$SOURCE_FILE" "$TARGET_FILE"

if [ $? -eq 0 ]; then
    echo "✅ ¡Archivo copiado exitosamente!"
    echo ""
    echo "📝 PRÓXIMOS PASOS:"
    echo "   1. Abre NinjaTrader 8"
    echo "   2. Ve a Tools > Edit NinjaScript > Indicator"
    echo "   3. Busca: TBOTTickWebSocketPublisherOptimized"
    echo "   4. Presiona F5 para recompilar"
    echo "   5. Configura tu chart con \"Days to load: 730\""
    echo "   6. Añade el indicador al chart"
    echo ""
    echo "🎯 Después ejecuta: cd python-client && python simple_test.py"
    echo ""
    echo "📊 CONFIGURACIÓN RECOMENDADA DEL INDICADOR:"
    echo "   • Historical Lookback: 2"
    echo "   • Historical Lookback Unit: Years"
    echo "   • Historical Bars Count: 1,050,000"
    echo "   • Auto-Configure Chart Data: ✅ ENABLED"
    echo "   • Fast Historical Delivery: ✅ ENABLED"
else
    echo "❌ ERROR: No se pudo copiar el archivo"
    echo "   Verifica que tengas permisos de escritura"
    echo "   o que NinjaTrader esté cerrado"
fi

echo ""
read -p "Presiona Enter para continuar..." 