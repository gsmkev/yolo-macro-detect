# 🦐 YOLO Macroinvertebrados - Detección Automática de Macroinvertebrados Acuáticos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%2Cv11%2Cv12-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project](https://img.shields.io/badge/Project-PINV01--1159-red.svg)]()
[![Research](https://img.shields.io/badge/Research-IEEE%20Conference-blue.svg)]()

Sistema de visión por computadora para la detección automática de macroinvertebrados acuáticos y evaluación de calidad del agua mediante inteligencia artificial. Este proyecto implementa tres modelos de detección de objetos (YOLOv11, YOLOv12 y YOLOv8x) para identificar familias de macroinvertebrados y calcular índices bióticos BMWP para inferir la calidad ecológica del agua.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Resultados Destacados](#-resultados-destacados)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Pipeline](#-pipeline)
- [Modelos Implementados](#-modelos-implementados)
- [Evaluación de Calidad del Agua](#-evaluación-de-calidad-del-agua)
- [Resultados Experimentales](#-resultados-experimentales)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [API Reference](#-api-reference)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## ✨ Características

- 🔍 **Detección Automática**: Identificación precisa de macroinvertebrados usando tres modelos YOLO
- 📊 **9 Familias Detectadas**: Belostomatidae, Chironomidae, Coenagrionidae, Dytiscidae, Hirudinidae, Libellulidae, Noteridae, Physidae, Planorbidae
- 🎯 **Alta Precisión**: Modelos con 99.4-100% de precisión y recall perfecto
- 🌊 **Evaluación de Calidad del Agua**: Cálculo automático del índice BMWP
- 📈 **Métricas Detalladas**: Evaluación completa con mAP@0.5, precisión y recall
- 🖼️ **Anotación Visual**: Generación automática de imágenes anotadas
- 📝 **Logging Completo**: Sistema de logs para seguimiento de entrenamiento e inferencia
- ⚙️ **Configuración Flexible**: Variables de entorno para personalización
- 🚀 **Pipeline Automatizado**: Proceso completo desde dataset hasta inferencia
- 🔬 **Validación en Campo**: Sistema probado en condiciones reales con 94% de coincidencia

## 🏆 Resultados Destacados

### Métricas de Rendimiento

| Modelo | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 |
|--------|-----------|--------|---------|--------------|
| **YOLOv11** | 99.8% | 100% | 99.4% | - |
| **YOLOv12** | 99.8% | 100% | 99.4% | - |
| **YOLOv8x** | 99.9% | 100% | 99.4% | 96.3% |

### Dataset
- **7,492 imágenes** anotadas manualmente por expertos
- **9 familias** de macroinvertebrados acuáticos
- **Aumentación automática** con técnicas avanzadas (rotación, cizallamiento, saturación, ruido)
- **División**: 87% entrenamiento, 8% validación, 4% prueba

### Validación en Campo
- **94% de coincidencia** con identificación humana
- **Validación exitosa** en arroyo Itay y Tres Puentes
- **Inferencia de calidad del agua** mediante BMWP

## 🛠️ Instalación

### Requisitos Previos

- Python 3.8 o superior
- CUDA compatible (recomendado para GPU)
- Git

### Instalación del Proyecto

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/gsmkev/yolo-macro-detect.git
   cd yolo-macro-detect
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o
   venv\Scripts\activate     # Windows
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**
   ```bash
   cp env.example .env
   # Editar .env con tus credenciales
   ```

## ⚙️ Configuración

### Variables de Entorno

Crea un archivo `.env` basado en `env.example`:

```env
# Configuración de Roboflow
ROBOFLOW_API_KEY=tu_api_key_aqui
ROBOFLOW_WORKSPACE=pinv011159
ROBOFLOW_PROJECT=macroinvertebrados-acuaticos

# Configuración del modelo
MODEL_NAME=yolov8x.pt
EXPERIMENT_NAME=macros
TRAINING_EPOCHS=50
IMG_SIZE=640
BATCH_SIZE=16
WORKERS=8

# Configuración de inferencia
CONFIDENCE_THRESHOLD=0.3
IOU_THRESHOLD=0.6

# Configuración de logging
LOG_LEVEL=INFO
SAVE_RESULTS=True

# Configuración BMWP (Evaluación de Calidad del Agua)
ENABLE_BMWP=True
BMWP_CONFIDENCE_WEIGHT=True
```

### Obtener API Key de Roboflow

1. Registrarse en [Roboflow](https://roboflow.com)
2. Ir a Account Settings > API Key
3. Copiar la API key y agregarla al archivo `.env`

## 🚀 Uso

### Pipeline Completo

Ejecutar todo el proceso desde la descarga del dataset hasta el entrenamiento:

```bash
python main.py --pipeline-complete
```

### Solo Configurar Dataset

Descargar y configurar el dataset sin entrenar:

```bash
python main.py --setup-dataset
```

### Solo Entrenamiento

Entrenar modelo con dataset existente:

```bash
python main.py --train --data-yaml datasets/data.yaml
```

### Solo Predicción

Realizar predicción en una imagen:

```bash
python main.py --predict --image test.jpg --model runs/detect/macros/weights/best.pt
```

### Predicción con Evaluación de Calidad del Agua

Calcular índice BMWP basado en las detecciones:

```bash
python main.py --predict --image sample.jpg --model best_model.pt --calculate-bmwp
```

### Opciones Avanzadas

```bash
# Pipeline completo con parámetros personalizados
python main.py --pipeline-complete \
    --dataset-version 5 \
    --epochs 100 \
    --experiment-name "macros_v2"

# Predicción con umbral personalizado y BMWP
python main.py --predict \
    --image sample.jpg \
    --model best_model.pt \
    --confidence 0.5 \
    --calculate-bmwp
```

## 🔄 Pipeline

### 1. Descarga de Dataset
- Conexión automática con Roboflow
- Descarga de la última versión del dataset (7,492 imágenes)
- Validación de estructura y contenido
- Generación de archivo `data.yaml`

### 2. Entrenamiento del Modelo
- Carga del modelo base YOLO (v8x, v11, v12)
- Configuración de hiperparámetros optimizados
- Entrenamiento con early stopping
- Guardado de checkpoints

### 3. Evaluación
- Validación en conjunto de test
- Cálculo de métricas (mAP, precisión, recall)
- Generación de gráficos de rendimiento
- Análisis de matriz de confusión

### 4. Inferencia
- Carga del modelo entrenado
- Predicción en imágenes nuevas
- Anotación automática
- Cálculo de índice BMWP (opcional)
- Exportación de resultados

## 🤖 Modelos Implementados

### YOLOv11 y YOLOv12 (Roboflow)
- **Entrenamiento**: Automático en plataforma Roboflow
- **Configuración**: Object Detection (Accurate) con checkpoint COCOs
- **Ventajas**: Despliegue inmediato, estabilidad, simplicidad
- **Rendimiento**: 99.8% precisión, 100% recall

### YOLOv8x (Entrenamiento Manual)
- **Entrenamiento**: Manual con PyTorch
- **Arquitectura**: 68.1M parámetros, 257.4 GFLOPs
- **Hardware**: NVIDIA RTX 3090 (24 GB)
- **Ventajas**: Control total, métricas ligeramente superiores
- **Rendimiento**: 99.9% precisión, 100% recall, mAP@0.5:0.95 = 96.3%

## 🌊 Evaluación de Calidad del Agua

### Índice BMWP (Biological Monitoring Working Party)

El sistema calcula automáticamente la calidad del agua basándose en las familias detectadas:

| Clase | BMWP | Calidad del Agua |
|-------|------|------------------|
| I | >101 | Muy limpia |
| II | 61–100 | Aceptable |
| III | 36–60 | Dudosa |
| IV | 16–35 | Crítica |
| V | <15 | Muy crítica |

### Puntajes por Familia

| Familia | Puntaje BMWP |
|---------|--------------|
| Belostomatidae | 5 |
| Coenagrionidae | 7 |
| Dytiscidae | 3 |
| Physidae | 3 |
| Planorbidae | 5 |
| Chironomidae | 8 |
| Noteridae | 4 |
| Libellulidae | 8 |
| Hirudinidae | 9 |

### Ejemplo de Cálculo

```json
{
  "detecciones": [
    {"familia": "Physidae", "cantidad": 6, "bmwp": 3},
    {"familia": "Planorbidae", "cantidad": 4, "bmwp": 5},
    {"familia": "Chironomidae", "cantidad": 3, "bmwp": 8},
    {"familia": "Hirudinidae", "cantidad": 2, "bmwp": 9}
  ],
  "bmwp_total": 55,
  "calidad_agua": "Dudosa (Clase III)",
  "confianza": 0.94
}
```

### Uso de la Calculadora BMWP

```python
from utils.bmwp_calculator import bmwp_calculator

# Detecciones de ejemplo
detections = [
    {"familia": "Physidae", "cantidad": 6, "confidence_promedio": 0.95},
    {"familia": "Chironomidae", "cantidad": 3, "confidence_promedio": 0.88}
]

# Calcular BMWP
result = bmwp_calculator.calculate_bmwp(detections)
print(f"BMWP: {result.total_score} - {result.water_quality_description}")
```

## 📊 Resultados Experimentales

### Métricas por Clase (YOLOv8x)

| Clase | mAP@0.5 | Precisión | Recall |
|-------|---------|-----------|--------|
| Belostomatidae | 100% | 100% | 100% |
| Chironomidae | 100% | 100% | 100% |
| Coenagrionidae | 100% | 100% | 100% |
| Dytiscidae | 100% | 100% | 100% |
| Hirudinidae | 100% | 100% | 100% |
| Libellulidae | 100% | 100% | 100% |
| Noteridae | 100% | 100% | 100% |
| Physidae | 99.0% | 99.0% | 99.0% |
| Planorbidae | 100% | 100% | 100% |

### Validación en Campo

- **Sitio de prueba**: Arroyo Itay y Tres Puentes
- **Coincidencia con expertos**: 94%
- **Aplicabilidad**: Confirmada para uso en condiciones reales
- **Tiempo de procesamiento**: Inferencia en tiempo casi real

### Curvas de Rendimiento

- **F1-Score**: Máximo 0.998 en umbral ~0.71
- **Precisión-Recall**: Área bajo curva cercana a 1.0 para todas las clases
- **Estabilidad**: Comportamiento estable en umbrales altos

## 📁 Estructura del Proyecto

```
yolo-macro-detect/
├── main.py                 # Script principal
├── config.py              # Configuración centralizada
├── requirements.txt       # Dependencias
├── env.example           # Ejemplo de variables de entorno
├── README.md             # Documentación
├── LICENSE               # Licencia
│
├── data/                 # Manejo de datasets
│   ├── __init__.py
│   └── dataset_manager.py
│
├── models/               # Modelos y entrenamiento
│   ├── __init__.py
│   ├── trainer.py
│   └── inference.py
│
├── utils/                # Utilidades
│   ├── __init__.py
│   ├── logger.py
│   ├── validators.py
│   └── bmwp_calculator.py
│
├── examples/             # Ejemplos de uso
│   └── example_usage.py
│
├── logs/                 # Logs del sistema
├── datasets/             # Datasets descargados
├── models/               # Modelos guardados
├── results/              # Resultados de inferencia
└── runs/                 # Resultados de entrenamiento
```

## 📚 API Reference

### MacroinvertebratePipeline

Clase principal para manejo del pipeline completo.

```python
from main import MacroinvertebratePipeline

pipeline = MacroinvertebratePipeline()

# Configurar dataset
data_yaml = pipeline.setup_dataset(version=5)

# Entrenar modelo
model_path = pipeline.train_model(data_yaml, epochs=50)

# Realizar predicción con BMWP
results = pipeline.predict_image("test.jpg", model_path, calculate_bmwp=True)
```

### YOLOTrainer

Clase para entrenamiento de modelos YOLO.

```python
from models import YOLOTrainer

trainer = YOLOTrainer("experimento_1")
trainer.load_model("yolov8x.pt")
model_path = trainer.train("data.yaml", epochs=50)
metrics = trainer.evaluate(model_path, "data.yaml")
```

### YOLOInference

Clase para inferencia con modelos entrenados.

```python
from models import YOLOInference

inference = YOLOInference("best_model.pt")
results = inference.predict_image("image.jpg", conf_threshold=0.3, calculate_bmwp=True)
bmwp_score = inference.calculate_bmwp(results['detecciones'])
inference.export_results(results, "output.json")
```

### BMWPCalculator

Clase para cálculo del índice BMWP.

```python
from utils.bmwp_calculator import bmwp_calculator

# Calcular BMWP
result = bmwp_calculator.calculate_bmwp(detections)

# Obtener información
families = bmwp_calculator.get_available_families()
water_quality_info = bmwp_calculator.get_water_quality_info()

# Formatear para JSON
json_result = bmwp_calculator.format_result_for_json(result)
```

### DatasetManager

Clase para manejo de datasets.

```python
from data import DatasetManager

manager = DatasetManager()
manager.setup_roboflow_connection()
dataset_info = manager.download_dataset(version=5)
manager.validate_dataset_structure(dataset_info["location"])
```


### Guías de Contribución

- Seguir las convenciones de código PEP 8
- Agregar docstrings a todas las funciones
- Incluir tests para nuevas funcionalidades
- Actualizar documentación según sea necesario

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Kevin M. Galeano**
- **Proyecto**: PINV01-1159
- **Institución**: Universidad Nacional de Asunción
- **Email**: [gsmkev@gmail.com](mailto:gsmkev@gmail.com)
- **GitHub**: [@gsmkev](https://github.com/gsmkev)


## 🙏 Agradecimientos

- **CONACYT Paraguay** por el financiamiento del proyecto PROCIENCIA
- [Ultralytics](https://github.com/ultralytics/ultralytics) por YOLOv8
- [Roboflow](https://roboflow.com) por la plataforma de datasets y entrenamiento automático
- [Supervision](https://github.com/roboflow/supervision) por las herramientas de anotación
- Equipo del proyecto PINV01-1159 por el soporte y colaboración
- Biólogos especialistas por la validación en campo

---

⭐ Si este proyecto te ha sido útil, ¡considera darle una estrella en GitHub!
 
