"""
Modelos para el proyecto YOLO Macroinvertebrados.

Este paquete contiene las clases y módulos para entrenamiento
e inferencia de modelos YOLO.
"""

from .trainer import YOLOTrainer
from .inference import YOLOInference

__all__ = ['YOLOTrainer', 'YOLOInference'] 