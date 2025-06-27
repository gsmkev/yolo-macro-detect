# 🚀 Guía de Despliegue en Producción

Esta guía te ayudará a desplegar el sistema de detección de macroinvertebrados en un entorno de producción.

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB
- **Almacenamiento**: 50 GB SSD
- **GPU**: NVIDIA GTX 1060 o superior (recomendado)

### Hardware Recomendado
- **CPU**: 8+ cores, 3.0 GHz
- **RAM**: 16+ GB
- **Almacenamiento**: 100+ GB NVMe SSD
- **GPU**: NVIDIA RTX 3080 o superior

### Software
- **OS**: Ubuntu 20.04 LTS o superior
- **Python**: 3.8+
- **CUDA**: 11.0+ (si usa GPU)
- **Docker**: 20.10+ (opcional)

## 🐳 Despliegue con Docker

### 1. Crear Dockerfile

```dockerfile
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del proyecto
COPY . .

# Crear directorios necesarios
RUN mkdir -p logs datasets models results

# Exponer puerto (si se usa API)
EXPOSE 8000

# Comando por defecto
CMD ["python", "main.py", "--help"]
```

### 2. Crear docker-compose.yml

```yaml
version: '3.8'

services:
  yolo-macro-detect:
    build: .
    container_name: yolo-macro-detect
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./logs:/app/logs
      - ./.env:/app/.env
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # Servicio de API (opcional)
  api:
    build: .
    container_name: yolo-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: ["python", "api.py"]
    depends_on:
      - yolo-macro-detect
    restart: unless-stopped
```

### 3. Construir y ejecutar

```bash
# Construir imagen
docker-compose build

# Ejecutar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f
```

## 🖥️ Despliegue Manual

### 1. Preparar el servidor

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias
sudo apt install -y python3 python3-pip python3-venv git nvidia-driver-470

# Verificar GPU
nvidia-smi
```

### 2. Clonar y configurar proyecto

```bash
# Clonar repositorio
git clone https://github.com/kevingaleano/yolo-macro-detect.git
cd yolo-macro-detect

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp env.example .env
# Editar .env con credenciales reales
```

### 3. Configurar sistema de archivos

```bash
# Crear directorios
mkdir -p /opt/yolo-macro-detect/{data,models,results,logs}
chown -R $USER:$USER /opt/yolo-macro-detect

# Crear enlaces simbólicos
ln -s /opt/yolo-macro-detect/data ./datasets
ln -s /opt/yolo-macro-detect/models ./models
ln -s /opt/yolo-macro-detect/results ./results
ln -s /opt/yolo-macro-detect/logs ./logs
```

## 🔧 Configuración de Producción

### 1. Variables de Entorno de Producción

```env
# Configuración de Roboflow
ROBOFLOW_API_KEY=tu_api_key_produccion
ROBOFLOW_WORKSPACE=pinv011159
ROBOFLOW_PROJECT=macroinvertebrados-acuaticos

# Configuración del modelo
MODEL_NAME=yolov8x.pt
EXPERIMENT_NAME=macros_prod
TRAINING_EPOCHS=100
IMG_SIZE=640
BATCH_SIZE=16
WORKERS=8

# Configuración de inferencia
CONFIDENCE_THRESHOLD=0.4
IOU_THRESHOLD=0.5

# Configuración de logging
LOG_LEVEL=WARNING
SAVE_RESULTS=True

# Configuración de producción
ENVIRONMENT=production
DEBUG=False
```

### 2. Configurar Logging

```python
# En config.py, agregar configuración de producción
import logging.handlers

def setup_production_logging():
    """Configurar logging para producción."""
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    
    # Handler para archivo con rotación
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Handler para syslog
    syslog_handler = logging.handlers.SysLogHandler()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    syslog_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(syslog_handler)
```

### 3. Configurar Monitoreo

```bash
# Instalar herramientas de monitoreo
sudo apt install -y htop iotop nvtop

# Configurar monitoreo de GPU
nvidia-smi -l 1  # Monitoreo cada segundo
```

## 🔒 Seguridad

### 1. Firewall

```bash
# Configurar UFW
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8000  # Si usa API
sudo ufw status
```

### 2. Usuario dedicado

```bash
# Crear usuario para la aplicación
sudo adduser yolo-app
sudo usermod -aG docker yolo-app

# Cambiar permisos
sudo chown -R yolo-app:yolo-app /opt/yolo-macro-detect
```

### 3. Certificados SSL (si usa API)

```bash
# Instalar Certbot
sudo apt install certbot python3-certbot-nginx

# Obtener certificado
sudo certbot --nginx -d tu-dominio.com
```

## 📊 Monitoreo y Mantenimiento

### 1. Script de monitoreo

```bash
#!/bin/bash
# monitor.sh

# Verificar uso de GPU
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ $GPU_UTIL -gt 90 ]; then
    echo "WARNING: GPU usage is $GPU_UTIL%"
fi

# Verificar espacio en disco
DISK_USAGE=$(df /opt/yolo-macro-detect | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "WARNING: Disk usage is $DISK_USAGE%"
fi

# Verificar logs de errores
ERROR_COUNT=$(grep -c "ERROR" logs/app.log | tail -1)
if [ $ERROR_COUNT -gt 10 ]; then
    echo "WARNING: High error count: $ERROR_COUNT"
fi
```

### 2. Backup automático

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/yolo-macro-detect"

# Crear backup
tar -czf "$BACKUP_DIR/backup_$DATE.tar.gz" \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    .

# Mantener solo los últimos 7 backups
find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +7 -delete
```

### 3. Cron jobs

```bash
# Agregar a crontab
crontab -e

# Monitoreo cada 5 minutos
*/5 * * * * /opt/yolo-macro-detect/scripts/monitor.sh

# Backup diario a las 2 AM
0 2 * * * /opt/yolo-macro-detect/scripts/backup.sh

# Limpieza de logs semanal
0 3 * * 0 find /opt/yolo-macro-detect/logs -name "*.log" -mtime +30 -delete
```

## 🚨 Troubleshooting

### Problemas Comunes

1. **GPU no detectada**
   ```bash
   # Verificar drivers
   nvidia-smi
   
   # Reinstalar drivers si es necesario
   sudo apt install --reinstall nvidia-driver-470
   ```

2. **Memoria insuficiente**
   ```bash
   # Reducir batch size en .env
   BATCH_SIZE=8
   WORKERS=4
   ```

3. **Espacio en disco**
   ```bash
   # Limpiar archivos temporales
   find /tmp -name "*.tmp" -delete
   
   # Limpiar logs antiguos
   find logs -name "*.log" -mtime +7 -delete
   ```

4. **Problemas de red**
   ```bash
   # Verificar conectividad
   ping roboflow.com
   
   # Verificar DNS
   nslookup roboflow.com
   ```

## 📞 Soporte

Para soporte técnico en producción:

- 📧 Email: soporte@ejemplo.com
- 📱 Teléfono: +1-234-567-8900
- 🐛 Issues: [GitHub Issues](https://github.com/kevingaleano/yolo-macro-detect/issues)

---

**Nota**: Esta guía asume un entorno Linux. Para Windows Server, adapta los comandos según sea necesario. 