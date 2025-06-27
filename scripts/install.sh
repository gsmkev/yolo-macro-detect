#!/bin/bash

# Script de instalación para YOLO Macroinvertebrados
# Autor: Kevin Galeano
# Proyecto: PINV01-1159

set -e  # Salir en caso de error

echo "🦐 Instalando YOLO Macroinvertebrados..."
echo "======================================"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar Python
print_status "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 no está instalado. Por favor instala Python 3.8 o superior."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION encontrado"

# Verificar pip
print_status "Verificando pip..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 no está instalado. Por favor instala pip."
    exit 1
fi

print_success "pip3 encontrado"

# Crear entorno virtual
print_status "Creando entorno virtual..."
if [ -d "venv" ]; then
    print_warning "El directorio venv ya existe. Eliminando..."
    rm -rf venv
fi

python3 -m venv venv
print_success "Entorno virtual creado"

# Activar entorno virtual
print_status "Activando entorno virtual..."
source venv/bin/activate
print_success "Entorno virtual activado"

# Actualizar pip
print_status "Actualizando pip..."
pip install --upgrade pip
print_success "pip actualizado"

# Instalar dependencias
print_status "Instalando dependencias..."
pip install -r requirements.txt
print_success "Dependencias instaladas"

# Crear directorios necesarios
print_status "Creando directorios del proyecto..."
mkdir -p logs datasets models results
print_success "Directorios creados"

# Configurar archivo .env
print_status "Configurando variables de entorno..."
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        print_warning "Archivo .env creado desde env.example"
        print_warning "Por favor edita .env con tus credenciales de Roboflow"
    else
        print_warning "Archivo env.example no encontrado"
        print_warning "Por favor crea manualmente el archivo .env"
    fi
else
    print_success "Archivo .env ya existe"
fi

# Verificar CUDA (opcional)
print_status "Verificando CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)
    print_success "CUDA $CUDA_VERSION detectado"
    print_success "GPU disponible para entrenamiento"
else
    print_warning "CUDA no detectado. El entrenamiento será más lento usando CPU"
fi

# Instalación completada
echo ""
echo "🎉 Instalación completada exitosamente!"
echo "======================================"
echo ""
echo "Para comenzar a usar el proyecto:"
echo "1. Activa el entorno virtual: source venv/bin/activate"
echo "2. Configura tu API key de Roboflow en el archivo .env"
echo "3. Ejecuta el pipeline: python main.py --pipeline-complete"
echo ""
echo "Para más información, consulta el README.md"
echo ""
print_success "¡YOLO Macroinvertebrados está listo para usar!" 