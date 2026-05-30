# Run 0 - Entrenamiento Inicial ConvNet Billetes Argentinos

**Fecha:** 2026-05-21
**Modelo:** ConvNet (arquitectura custom con TensorFlow/Keras)
**Archivo del modelo:** `models/convnet_billetes.keras`
**Gráfico de entrenamiento:** `models/convnet_billetes_training.png`

---

## Dataset

### Origen
- **Directorio:** `dataset/raw/`
- **Imágenes tomadas manualmente** con cámara celular y ejemplos de internet
- **Clases:** 6 denominaciones de billetes argentinos

### Distribución por clase

| Clase   | Imágenes |
|---------|----------|
| 100     | 7        |
| 200     | 14       |
| 500     | 10       |
| 1000    | 17       |
| 2000    | 8        |
| 10000   | 8        |
| **Total** | **64** |

### Split Train / Validation

- **Estrategia:** `train_test_split` con estratificación (`stratify=True`)
- **Porcentaje:** 80% train / 20% validation
- **Random seed:** 42

#### Train (51 imágenes)

| Clase   | Imágenes |
|---------|----------|
| 100     | 6        |
| 200     | 11       |
| 500     | 8        |
| 1000    | 14       |
| 2000    | 6        |
| 10000   | 6        |

#### Validation (13 imágenes)

| Clase   | Imágenes |
|---------|----------|
| 100     | 1        |
| 200     | 3        |
| 500     | 2        |
| 1000    | 3        |
| 2000    | 2        |
| 10000   | 2        |

---

## Preprocesamiento de Imágenes

- **Tamaño:** 128x128 píxeles, RGB (3 canales)
- **Formato de entrada:** `(batch, 128, 128, 3)`
- **Escalado:** Normalización a `[0, 1]` vía `Rescaling(1./255)`

### Data Augmentation (online, durante entrenamiento)

Transformaciones aplicadas como capas del modelo (solo activas durante `fit()`):

| Transformación        | Parámetro |
|-----------------------|-----------|
| RandomFlip            | horizontal |
| RandomRotation        | factor 0.2 |
| RandomTranslation     | width 0.2, height 0.2 |
| RandomZoom            | factor 0.2 |
| RandomContrast        | factor 0.2 |
| RandomBrightness      | factor 0.2 |

**Decisión:** Augmentation online (no offline). Cada epoch genera variaciones distintas sin ocupar disco, y el modelo nunca ve la misma imagen dos veces.

---

## Arquitectura del Modelo

### Resumen de capas

| Capa | Tipo | Output Shape | Params |
|------|------|-------------|--------|
| input_img | Input | (None, 128, 128, 3) | 0 |
| da_rndFlip | RandomFlip | (None, 128, 128, 3) | 0 |
| da_rndRotation | RandomRotation | (None, 128, 128, 3) | 0 |
| da_rndTranslation | RandomTranslation | (None, 128, 128, 3) | 0 |
| da_rndZoom | RandomZoom | (None, 128, 128, 3) | 0 |
| da_rndContrast | RandomContrast | (None, 128, 128, 3) | 0 |
| da_rndBrightness | RandomBrightness | (None, 128, 128, 3) | 0 |
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
| Dropout rate | 0.3 (configurado, no activo en esta arquitectura) |

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
| 19 | 0.001 | 0.0005 |
| 29 | 0.0005 | 0.00025 |

### Epoch de finalización

- **Época alcanzada:** 100 (completó todas, no se activó EarlyStopping)
- **Mejor época (restored):** 97 (menor val_loss)

### Métricas finales (época 100)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 1.0228 | 1.1976 |
| Accuracy | 58.00% | 38.46% |

### Mejores métricas (época 97, pesos restaurados)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.9427 | 1.1792 |
| Accuracy | 54.00% | 38.46% |

---

## Evaluación en Validation Set

### Classification Report

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.00      | 0.00   | 0.00     | 1       |
| 200     | 0.33      | 0.67   | 0.44     | 3       |
| 500     | 0.50      | 0.50   | 0.50     | 2       |
| 1000    | 1.00      | 0.67   | 0.80     | 3       |
| 2000    | 0.00      | 0.00   | 0.00     | 2       |
| 10000   | 0.00      | 0.00   | 0.00     | 2       |
| **Accuracy** | | | **0.38** | **13** |
| **Macro Avg** | 0.31 | 0.31 | 0.29 | 13 |
| **Weighted Avg** | 0.38 | 0.38 | 0.36 | 13 |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 0   | 0   | 1   | 0    | 0    | 0     |
| **200**     | 0   | 2   | 0   | 0    | 0    | 1     |
| **500**     | 0   | 1   | 1   | 0    | 0    | 0     |
| **1000**    | 0   | 1   | 0   | 2    | 0    | 0     |
| **2000**    | 0   | 0   | 0   | 0    | 0    | 2     |
| **10000**   | 0   | 2   | 0   | 0    | 0    | 0     |

### Observaciones

- **1000** es la clase mejor reconocida (precision 1.00, recall 0.67)
- **2000** y **10000** tienen 0% de acierto (confundidas con otras clases)
- **100** tiene solo 1 imagen en validación, insuficiente para evaluar
- El modelo tiende a predecir **200** como clase por defecto (5 de 13 predicciones)
- **Overfitting notable:** train accuracy 58% vs val accuracy 38%

---

## Hardware y Software

| Componente | Detalle |
|------------|---------|
| OS | Windows (native) |
| Python | 3.12.5 |
| TensorFlow | 2.21.0 |
| GPU | No disponible (TF >= 2.11 no soporta GPU en Windows nativo) |
| Device | CPU |

---

## Conclusiones y Próximos Pasos

### Problemas identificados

1. **Dataset extremadamente pequeño** (64 imágenes totales, 51 train). Se necesitan al menos 50-100 imágenes por clase.
2. **Validation set insuficiente** (13 imágenes). Con 1-3 imágenes por clase las métricas no son confiables.
3. **No se usó GPU** por limitación de TensorFlow en Windows nativo.
4. **Overfitting:** la brecha train/val accuracy indica que el modelo memoriza en lugar de generalizar.

### Mejoras recomendadas

1. **Recolectar más imágenes** (objetivo: 100+ por clase, con variedad de ángulos, iluminación, fondos)
2. **Transfer Learning:** usar MobileNetV2 o EfficientNet pre-entrenados con ImageNet y fine-tuning
3. **Aumentar Dropout** o agregar BatchNormalization para reducir overfitting
4. **Cross-validation** k-fold dado el dataset pequeño
5. **Migrar a WSL2** para habilitar GPU acceleration con TensorFlow
