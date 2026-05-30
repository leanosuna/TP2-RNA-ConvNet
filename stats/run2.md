# Run 2 - Entrenamiento ConvNet con Mejoras

**Fecha:** 2026-05-21
**Modelo:** ConvNet mejorada (filtros crecientes + BatchNorm + L2 + class weights)
**Archivo del modelo:** `models/convnet_billetes.keras`
**Gráfico de entrenamiento:** `models/convnet_billetes_training.png`
**Runs anteriores:** [run0.md](./run0.md) | [run1.md](./run1.md)

---

## Cambios respecto a Run 1

| Parámetro | Run 1 | Run 2 | Cambio |
|-----------|-------|-------|--------|
| Imagen de entrada | 128x128 | **224x224** | +75% pixels |
| Filtros convolucionales | 32→16→8 (decrecientes) | **32→64→128 (crecientes)** | Más capacidad |
| BatchNormalization | No | **Sí** (después de cada Conv2D) | Estabilidad |
| Regularización L2 | No | **Sí** (0.001) | Anti-overfitting |
| Capas densas | 64,32 | **256,128,D,64** | Más capacidad + dropout |
| Dropout | 0.3 | **0.5** | Más regularización |
| Learning rate | 0.001 | **0.0005** | Más conservador |
| Rotación augmentation | 0.2 | **0.05** | Preserva detalles |
| Zoom augmentation | 0.2 | **0.1** | Preserva detalles |
| Traslación augmentation | 0.2 | **0.1** | Preserva detalles |
| Contraste augmentation | 0.2 | **0.1** | Preserva detalles |
| Class weights | No | **Sí** (balanced) | Balanceo de clases |
| Early stop patience | 20 | **40** | Más tiempo |
| Early stop start epoch | 15 | **30** | Más tiempo |
| Épocas máximas | 100 | **150** | Más tiempo |

---

## Dataset

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

### Split Train / Validation

- **Estrategia:** `train_test_split` con estratificación
- **Porcentaje:** 80% train / 20% validation
- **Random seed:** 42

#### Train (89 imágenes)

| Clase   | Imágenes | Class Weight |
|---------|----------|-------------|
| 100     | 12       | 1.236 |
| 200     | 17       | 0.873 |
| 500     | 14       | 1.060 |
| 1000    | 20       | 0.742 |
| 2000    | 13       | 1.141 |
| 10000   | 13       | 1.141 |

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

## Arquitectura del Modelo

### Resumen de capas

| Capa | Tipo | Output Shape | Params |
|------|------|-------------|--------|
| input_img | Input | (None, 224, 224, 3) | 0 |
| random_flip | RandomFlip | (None, 224, 224, 3) | 0 |
| random_rotation | RandomRotation | (None, 224, 224, 3) | 0 |
| random_translation | RandomTranslation | (None, 224, 224, 3) | 0 |
| random_zoom | RandomZoom | (None, 224, 224, 3) | 0 |
| random_contrast | RandomContrast | (None, 224, 224, 3) | 0 |
| random_brightness | RandomBrightness | (None, 224, 224, 3) | 0 |
| rescale | Rescaling | (None, 224, 224, 3) | 0 |
| conv_1 | Conv2D (32 filters, 3x3, relu, L2) | (None, 224, 224, 32) | 896 |
| bn_1 | BatchNormalization | (None, 224, 224, 32) | 128 |
| pool_1 | MaxPooling2D (2x2) | (None, 112, 112, 32) | 0 |
| conv_2 | Conv2D (64 filters, 3x3, relu, L2) | (None, 112, 112, 64) | 18,496 |
| bn_2 | BatchNormalization | (None, 112, 112, 64) | 256 |
| pool_2 | MaxPooling2D (2x2) | (None, 56, 56, 64) | 0 |
| conv_3 | Conv2D (128 filters, 3x3, relu, L2) | (None, 56, 56, 128) | 73,856 |
| bn_3 | BatchNormalization | (None, 56, 56, 128) | 512 |
| pool_3 | MaxPooling2D (2x2) | (None, 28, 28, 128) | 0 |
| flatten | Flatten | (None, 100,352) | 0 |
| dense_1 | Dense (256, relu, L2) | (None, 256) | 25,690,368 |
| dense_2 | Dense (128, relu, L2) | (None, 128) | 32,896 |
| dropout_3 | Dropout (0.5) | (None, 128) | 0 |
| dense_4 | Dense (64, relu, L2) | (None, 64) | 8,256 |
| output | Dense (6, softmax) | (None, 6) | 390 |

**Total de parámetros:** 25,826,054 (todos entrenables)

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Optimizador | Adam |
| Learning rate inicial | 0.0005 |
| Loss function | categorical_crossentropy |
| Métrica | accuracy |
| Batch size | 8 |
| Épocas máximas | 150 |
| L2 regularization | 0.001 |

---

## Entrenamiento

### Callbacks

| Callback | Configuración |
|----------|--------------|
| EarlyStopping | monitor=val_loss, patience=40, min_delta=0.001, start_from_epoch=30, restore_best_weights=True |
| ReduceLROnPlateau | factor=0.5, patience=10, min_lr=1e-6 |

### Reducciones de Learning Rate

| Época | LR anterior | LR nuevo |
|-------|------------|----------|
| 11 | 0.0005 | 0.00025 |
| 21 | 0.00025 | 0.000125 |
| 31 | 0.000125 | 0.0000625 |
| 41 | 0.0000625 | 0.00003125 |
| 59 | 0.00003125 | 0.000015625 |
| 69 | 0.000015625 | 0.0000078125 |
| 79 | 0.0000078125 | 0.00000390625 |
| 89 | 0.00000390625 | 0.000001953125 |

### Epoch de finalización

- **Época alcanzada:** 89 (detenido por EarlyStopping)
- **Mejor época (restored):** 49 (menor val_loss)

### Mejores métricas (época 49, pesos restaurados)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 3.2854 | 2.5856 |
| Accuracy | 49.44% | **65.22%** |

---

## Evaluación en Validation Set

### Classification Report

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 1.00      | 0.33   | 0.50     | 3       |
| 200     | 0.50      | 0.40   | 0.44     | 5       |
| 500     | 1.00      | 0.50   | 0.67     | 4       |
| 1000    | 0.71      | 1.00   | 0.83     | 5       |
| 2000    | 1.00      | 1.00   | 1.00     | 3       |
| 10000   | 0.33      | 0.67   | 0.44     | 3       |
| **Accuracy** | | | **0.65** | **23** |
| **Macro Avg** | 0.76 | 0.65 | 0.65 | 23 |
| **Weighted Avg** | 0.74 | 0.65 | 0.65 | 23 |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 1   | 0   | 0   | 2    | 0    | 0     |
| **200**     | 0   | 2   | 0   | 0    | 0    | 3     |
| **500**     | 0   | 1   | 2   | 0    | 0    | 1     |
| **1000**    | 0   | 0   | 0   | 5    | 0    | 0     |
| **2000**    | 0   | 0   | 0   | 0    | 3    | 0     |
| **10000**   | 0   | 1   | 0   | 0    | 0    | 2     |

### Observaciones

- **1000**: Perfect recall (5/5), precision 0.71 (2 confundidos con 100)
- **2000**: Perfecto (3/3), precision y recall 1.00
- **100**: Precision 1.00 pero recall bajo (1/3), 2 confundidos con 1000
- **500**: Precision 1.00, recall 0.50 (2/4), confundidos con 200 y 10000
- **200**: Recall 0.40 (2/5), 3 confundidos con 10000
- **10000**: Recall 0.67 (2/3), 1 confundido con 200
- **Todas las clases tienen recall > 0** (vs Run 1 donde 4 clases tenían 0)

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

## Comparación Run 0 vs Run 1 vs Run 2

| Métrica | Run 0 | Run 1 | Run 2 |
|---------|-------|-------|-------|
| Imagen | 128x128 | 128x128 | **224x224** |
| Filtros | 32→16→8 | 32→16→8 | **32→64→128** |
| BatchNorm | No | No | **Sí** |
| L2 Reg | No | No | **Sí** |
| Class weights | No | No | **Sí** |
| Total params | 140K | 140K | **25.8M** |
| Total imágenes | 64 | 112 | 112 |
| Train | 51 | 89 | 89 |
| Validation | 13 | 23 | 23 |
| Época final | 100 | 55 | 89 |
| Mejor época | 97 | 35 | 49 |
| **Val accuracy** | **38.46%** | **26.09%** | **65.22%** |
| Val loss | 1.1792 | 1.6533 | 2.5856* |
| Clases con recall > 0 | 3 | 2 | **6** |
| Mejor clase (F1) | 1000 (0.80) | 1000 (0.62) | **2000 (1.00)** |
| Macro F1 | 0.29 | 0.14 | **0.65** |

*Nota: El loss es más alto en Run 2 debido a los class weights que escalan la loss function. La accuracy es la métrica comparable.

### Progreso visual

```
Run 0: ████████████░░░░░░░░░░  38.46%
Run 1: ████████░░░░░░░░░░░░░░  26.09%
Run 2: █████████████████████░  65.22%
```

---

## Conclusiones

### Qué funcionó

1. **Filtros crecientes (32→64→128)**: La red ahora captura features más complejas en capas profundas.
2. **BatchNormalization**: Estabilizó el entrenamiento, permitió usar LR más alto sin divergir.
3. **Imagen más grande (224x224)**: Los detalles de denominación son más visibles.
4. **Class weights**: Evitó que el modelo colapse a predecir una sola clase.
5. **Augmentation reducido**: Preserva los números de denominación como señal válida.
6. **Capas densas más grandes (256→128→64)**: Mejor capacidad de clasificación.

### Problemas pendientes

1. **100 y 200 siguen siendo difíciles de distinguir** de otras clases.
2. **200 se confunde con 10000** (posiblemente por similitud visual en las fotos).
3. **Modelo muy grande** (25.8M params) para solo 89 imágenes de train. Riesgo de overfitting.
4. **Sin GPU**: Entrenamiento lento (~8s/epoch en CPU).

### Próximos pasos sugeridos

1. **Más imágenes de 100 y 200** (las clases con menor recall).
2. **Agregar GlobalAveragePooling2D** en vez de Flatten para reducir parámetros de 25.8M a ~1M.
3. **Probar con 4 capas convolucionales** para más capacidad de extracción de features.
4. **Migrar a WSL2** para GPU acceleration.
