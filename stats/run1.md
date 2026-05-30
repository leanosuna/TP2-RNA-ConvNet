# Run 1 - Entrenamiento ConvNet Billetes Argentinos (Dataset Ampliado)

**Fecha:** 2026-05-21
**Modelo:** ConvNet (arquitectura custom con TensorFlow/Keras)
**Archivo del modelo:** `models/convnet_billetes.keras`
**Gráfico de entrenamiento:** `models/convnet_billetes_training.png`
**Run anterior:** [run0.md](./run0.md)

---

## Dataset

### Origen
- **Directorio:** `dataset/raw/`
- **Imágenes tomadas manualmente** con cámara celular y ejemplos de internet
- **Clases:** 6 denominaciones de billetes argentinos

### Distribución por clase

| Clase   | Imágenes |
|---------|----------|
| 100     | 15       |
| 200     | 22       |
| 500     | 18       |
| 1000    | 25       |
| 2000    | 16       |
| 10000   | 16       |
| **Total** | **112** |

**Comparación con Run 0:** 64 → 112 imágenes (+48 imágenes, +75%)

### Split Train / Validation

- **Estrategia:** `train_test_split` con estratificación (`stratify=True`)
- **Porcentaje:** 80% train / 20% validation
- **Random seed:** 42

#### Train (89 imágenes)

| Clase   | Imágenes |
|---------|----------|
| 100     | 12       |
| 200     | 17       |
| 500     | 14       |
| 1000    | 20       |
| 2000    | 13       |
| 10000   | 13       |

#### Validation (23 imágenes)

| Clase   | Imágenes |
|---------|----------|
| 100     | 3        |
| 200     | 5        |
| 500     | 4        |
| 1000    | 5        |
| 2000    | 3        |
| 10000   | 3        |

---

## Preprocesamiento de Imágenes

- **Tamaño:** 128x128 píxeles, RGB (3 canales)
- **Formato de entrada:** `(batch, 128, 128, 3)`
- **Escalado:** Normalización a `[0, 1]` vía `Rescaling(1./255)`

### Data Augmentation (online, durante entrenamiento)

| Transformación        | Parámetro |
|-----------------------|-----------|
| RandomFlip            | horizontal |
| RandomRotation        | factor 0.2 |
| RandomTranslation     | width 0.2, height 0.2 |
| RandomZoom            | factor 0.2 |
| RandomContrast        | factor 0.2 |
| RandomBrightness      | factor 0.2 |

---

## Arquitectura del Modelo

### Resumen de capas

| Capa | Tipo | Output Shape | Params |
|------|------|-------------|--------|
| input_img | Input | (None, 128, 128, 3) | 0 |
| random_flip | RandomFlip | (None, 128, 128, 3) | 0 |
| random_rotation | RandomRotation | (None, 128, 128, 3) | 0 |
| random_translation | RandomTranslation | (None, 128, 128, 3) | 0 |
| random_zoom | RandomZoom | (None, 128, 128, 3) | 0 |
| random_contrast | RandomContrast | (None, 128, 128, 3) | 0 |
| random_brightness | RandomBrightness | (None, 128, 128, 3) | 0 |
| rescale | Rescaling | (None, 128, 128, 3) | 0 |
| conv_1 | Conv2D (32 filters, 3x3, relu) | (None, 128, 128, 32) | 896 |
| pool_1 | MaxPooling2D (2x2) | (None, 64, 64, 32) | 0 |
| conv_2 | Conv2D (16 filters, 3x3, relu) | (None, 64, 64, 16) | 4,624 |
| pool_2 | MaxPooling2D (2x2) | (None, 32, 32, 16) | 0 |
| conv_3 | Conv2D (8 filters, 3x3, relu) | (None, 32, 32, 8) | 1,160 |
| pool_3 | MaxPooling2D (2x2) | (None, 16, 16, 8) | 0 |
| flatten | Flatten | (None, 2048) | 0 |
| dense_1 | Dense (64, relu) | (None, 64) | 131,136 |
| dense_2 | Dense (32, relu) | (None, 32) | 2,080 |
| output | Dense (6, softmax) | (None, 6) | 198 |

**Total de parámetros:** 140,094 (todos entrenables)

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Optimizador | Adam |
| Learning rate inicial | 0.001 |
| Loss function | categorical_crossentropy |
| Métrica | accuracy |
| Batch size | 8 |
| Épocas máximas | 100 |

---

## Entrenamiento

### Callbacks

| Callback | Configuración |
|----------|--------------|
| EarlyStopping | monitor=val_loss, patience=20, min_delta=0.001, start_from_epoch=15, restore_best_weights=True |
| ReduceLROnPlateau | factor=0.5, patience=10, min_lr=1e-6 |

### Reducciones de Learning Rate

| Época | LR anterior | LR nuevo |
|-------|------------|----------|
| 28 | 0.001 | 0.0005 |
| 45 | 0.0005 | 0.00025 |
| 55 | 0.00025 | 0.000125 |

### Epoch de finalización

- **Época alcanzada:** 55 (detenido por EarlyStopping)
- **Mejor época (restored):** 35 (menor val_loss)

### Métricas finales (época 55, antes de early stopping)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 1.4336 | 1.7459 |
| Accuracy | 41.57% | 26.09% |

### Mejores métricas (época 35, pesos restaurados)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 1.4831 | 1.6533 |
| Accuracy | 34.83% | 26.09% |

---

## Evaluación en Validation Set

### Classification Report

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.00      | 0.00   | 0.00     | 3       |
| 200     | 0.17      | 0.40   | 0.24     | 5       |
| 500     | 0.00      | 0.00   | 0.00     | 4       |
| 1000    | 0.50      | 0.80   | 0.62     | 5       |
| 2000    | 0.00      | 0.00   | 0.00     | 3       |
| 10000   | 0.00      | 0.00   | 0.00     | 3       |
| **Accuracy** | | | **0.26** | **23** |
| **Macro Avg** | 0.11 | 0.20 | 0.14 | 23 |
| **Weighted Avg** | 0.14 | 0.26 | 0.18 | 23 |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 0   | 2   | 0   | 1    | 0    | 0     |
| **200**     | 0   | 2   | 1   | 1    | 1    | 0     |
| **500**     | 0   | 2   | 0   | 2    | 0    | 0     |
| **1000**    | 0   | 1   | 0   | 4    | 0    | 0     |
| **2000**    | 0   | 3   | 0   | 0    | 0    | 0     |
| **10000**   | 1   | 2   | 0   | 0    | 0    | 0     |

### Observaciones

- **1000** es nuevamente la clase mejor reconocida (precision 0.50, recall 0.80)
- **100, 500, 2000, 10000** tienen 0% precision y recall
- El modelo tiene un **sesgo fuerte hacia predecir 200** (12 de 23 predicciones)
- **2000** se confunde completamente con **200** (3 de 3)
- **500** se confunde entre **200** y **1000** (2 y 2 respectivamente)
- **10000** se confunde entre **100** y **200**

---

## Hardware y Software

| Componente | Detalle |
|------------|---------|
| OS | Windows (native) |
| Python | 3.12.5 |
| TensorFlow | 2.21.0 |
| GPU | No disponible |
| Device | CPU |

---

## Comparación Run 0 vs Run 1

| Métrica | Run 0 | Run 1 | Cambio |
|---------|-------|-------|--------|
| Total imágenes | 64 | 112 | +75% |
| Train | 51 | 89 | +75% |
| Validation | 13 | 23 | +77% |
| Épocas completadas | 100 | 55 (early stop) | - |
| Mejor época | 97 | 35 | - |
| Best val_loss | 1.1792 | 1.6533 | +40% (peor) |
| Best val_accuracy | 38.46% | 26.09% | -12.37pp |
| Train accuracy (best epoch) | 54.00% | 34.83% | -19.17pp |
| Clases con recall > 0 | 3 (200, 500, 1000) | 2 (200, 1000) | -1 |
| Mejor clase (F1) | 1000 (0.80) | 1000 (0.62) | -0.18 |

### Análisis

Paradójicamente, **el rendimiento empeoró** a pesar de tener 75% más de datos. Posibles causas:

1. **Las nuevas imágenes pueden tener mayor variabilidad** (diferentes ángulos, iluminación, fondos) que dificultan el aprendizaje con una arquitectura tan simple.
2. **El modelo es demasiado pequeño** (140K params) para capturar la complejidad del dataset ampliado.
3. **Las clases de billetes similares** (200 vs 2000, 500 vs 1000) comparten patrones visuales que la red no logra distinguir sin más capacidad.
4. **El augmentation online puede estar destruyendo señales** importantes en billetes con detalles finos (números de denominación).

---

## Conclusiones y Próximos Pasos

### Problemas persistentes

1. **Arquitectura insuficiente:** 3 capas convolucionales con filtros decrecientes (32→16→8) no capturan suficientes features.
2. **Sin GPU:** Entrenamiento lento, limita experimentación.
3. **Dataset sigue siendo pequeño:** 112 imágenes para 6 clases es insuficiente para una red desde cero.
4. **Bias hacia clase 200:** El modelo colapsa a predecir una sola clase.

### Mejoras recomendadas (priorizadas)

1. **Transfer Learning (máxima prioridad):** Usar MobileNetV2/EfficientNet pre-entrenados con ImageNet. Con ~100 imágenes, fine-tuning es mucho más efectivo que entrenar desde cero.
2. **Reducir augmentation:** Rotación y zoom excesivos pueden distorsionar los números de denominación que son la señal principal.
3. **Aumentar dataset a 200+ imágenes** con enfoque en clases problemáticas (100, 500, 2000, 10000).
4. **Migrar a WSL2** para habilitar GPU acceleration.
5. **Agregar regularización:** BatchNormalization, mayor Dropout, weight decay.
