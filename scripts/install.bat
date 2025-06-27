@echo off
REM Script de instalación para YOLO Macroinvertebrados (Windows)
REM Autor: Kevin Galeano
REM Proyecto: PINV01-1159

echo 🦐 Instalando YOLO Macroinvertebrados...
echo ======================================

REM Verificar Python
echo [INFO] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no está instalado. Por favor instala Python 3.8 o superior.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% encontrado

REM Verificar pip
echo [INFO] Verificando pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip no está instalado. Por favor instala pip.
    pause
    exit /b 1
)

echo [SUCCESS] pip encontrado

REM Crear entorno virtual
echo [INFO] Creando entorno virtual...
if exist venv (
    echo [WARNING] El directorio venv ya existe. Eliminando...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo [ERROR] Error al crear el entorno virtual.
    pause
    exit /b 1
)
echo [SUCCESS] Entorno virtual creado

REM Activar entorno virtual
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Error al activar el entorno virtual.
    pause
    exit /b 1
)
echo [SUCCESS] Entorno virtual activado

REM Actualizar pip
echo [INFO] Actualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Error al actualizar pip. Continuando...
)

REM Instalar dependencias
echo [INFO] Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Error al instalar dependencias.
    pause
    exit /b 1
)
echo [SUCCESS] Dependencias instaladas

REM Crear directorios necesarios
echo [INFO] Creando directorios del proyecto...
if not exist logs mkdir logs
if not exist datasets mkdir datasets
if not exist models mkdir models
if not exist results mkdir results
echo [SUCCESS] Directorios creados

REM Configurar archivo .env
echo [INFO] Configurando variables de entorno...
if not exist .env (
    if exist env.example (
        copy env.example .env >nul
        echo [WARNING] Archivo .env creado desde env.example
        echo [WARNING] Por favor edita .env con tus credenciales de Roboflow
    ) else (
        echo [WARNING] Archivo env.example no encontrado
        echo [WARNING] Por favor crea manualmente el archivo .env
    )
) else (
    echo [SUCCESS] Archivo .env ya existe
)

REM Verificar CUDA (opcional)
echo [INFO] Verificando CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] CUDA no detectado. El entrenamiento será más lento usando CPU
) else (
    echo [SUCCESS] CUDA detectado
    echo [SUCCESS] GPU disponible para entrenamiento
)

REM Instalación completada
echo.
echo 🎉 Instalación completada exitosamente!
echo ======================================
echo.
echo Para comenzar a usar el proyecto:
echo 1. Activa el entorno virtual: venv\Scripts\activate.bat
echo 2. Configura tu API key de Roboflow en el archivo .env
echo 3. Ejecuta el pipeline: python main.py --pipeline-complete
echo.
echo Para más información, consulta el README.md
echo.
echo [SUCCESS] ¡YOLO Macroinvertebrados está listo para usar!
pause 