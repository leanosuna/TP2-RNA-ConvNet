# Run 3 - Entrenamiento PyTorch con GPU (Dataset Balanceado)

**Fecha:** 2026-05-21
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti (8.6 GB)
**Modelo:** ConvNet (filtros crecientes + BatchNorm + L2)
**Archivo del modelo:** `models/convnet_billetes.pth`
**Gráfico de entrenamiento:** `models/convnet_billetes_training.png`
**Runs anteriores:** [run0.md](./run0.md) (TF CPU) | [run1.md](./run1.md) (TF CPU) | [run2.md](./run2.md) (TF CPU)

---

## Cambios respecto a Run 2

| Aspecto | Run 2 (TensorFlow) | Run 3 (PyTorch) |
|---------|-------------------|-----------------|
| Framework | TensorFlow 2.21 | **PyTorch 2.5.1** |
| Device | CPU | **GPU RTX 3060 Ti** |
| Tiempo de entrenamiento | ~8 min | **36 segundos** |
| Velocidad | ~8s/epoch | **~0.5s/epoch** |
| Dataset | 112 imágenes | **144 imágenes** (balanceado) |
| Batch size | 8 | **16** |
| Learning rate | 0.0005 | **0.001** |
| Weight decay | 0.001 | **0.0005** |
| Dropout | 0.5 | **0.4** |
| Imágenes pre-cargadas en GPU | No | **Sí** |
| Augmentation en GPU | No | **Sí** |
| cudnn.benchmark | N/A | **True** |

---

## Dataset

### Distribución por clase (balanceado)

| Clase   | Imágenes |
|---------|----------|
| 100     | 23       |
| 200     | 26       |
| 500     | 24       |
| 1000    | 25       |
| 2000    | 23       |
| 10000   | 23       |
| **Total** | **144** |

### Split Train / Validation

- **Estrategia:** `torch.randperm` con seed 42
- **Porcentaje:** 80% train / 20% validation

#### Train (115 imágenes)

| Clase   | Imágenes |
|---------|----------|
| 100     | 19       |
| 200     | 19       |
| 500     | 18       |
| 1000    | 19       |
| 2000    | 20       |
| 10000   | 20       |

#### Validation (29 imágenes)

| Clase   | Imágenes |
|---------|----------|
| 100     | 4        |
| 200     | 7        |
| 500     | 6        |
| 1000    | 6        |
| 2000    | 3        |
| 10000   | 3        |

---

## Arquitectura del Modelo

| Capa | Tipo | Params |
|------|------|--------|
| conv_1 | Conv2D (32, 3x3) + BatchNorm + ReLU + MaxPool | 896 + 128 |
| conv_2 | Conv2D (64, 3x3) + BatchNorm + ReLU + MaxPool | 18,496 + 256 |
| conv_3 | Conv2D (128, 3x3) + BatchNorm + ReLU + MaxPool | 73,856 + 512 |
| flatten | Flatten | - |
| dense_1 | Linear (256) + ReLU | 25,690,368 |
| dense_2 | Linear (128) + ReLU | 32,896 |
| dropout | Dropout (0.4) | - |
| dense_3 | Linear (64) + ReLU | 8,256 |
| output | Linear (6) | 390 |

**Total de parámetros:** 27,693,158

---

## Entrenamiento

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Optimizador | Adam |
| Learning rate inicial | 0.001 |
| Weight decay | 0.0005 |
| Batch size | 16 |
| Épocas máximas | 150 |
| Early stop patience | 40 (start epoch 30) |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 10) |

### Reducciones de Learning Rate

| Época | LR nuevo |
|-------|----------|
| 20 | 5.00e-04 |
| 31 | 2.50e-04 |
| 42 | 1.25e-04 |
| 53 | 6.25e-05 |
| 64 | 3.13e-05 |
| 75 | 1.56e-05 |

### Resultado

- **Época alcanzada:** 76 (early stopping)
- **Mejor época:** 36
- **Tiempo total:** 36.2 segundos (0.6 min)

### Mejores métricas (época 36)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.9296 | 2.1324 |
| Accuracy | 63.48% | **55.17%** |

---

## Evaluación en Validation Set

### Classification Report

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.27      | 0.75   | 0.40     | 4       |
| 200     | 0.67      | 0.29   | 0.40     | 7       |
| 500     | 0.80      | 0.67   | 0.73     | 6       |
| 1000    | 0.80      | 0.67   | 0.73     | 6       |
| 2000    | 0.50      | 0.67   | 0.57     | 3       |
| 10000   | 1.00      | 0.33   | 0.50     | 3       |
| **Accuracy** | | | **0.55** | **29** |
| **Macro Avg** | 0.67 | 0.56 | 0.55 | 29 |
| **Weighted Avg** | 0.68 | 0.55 | 0.56 | 29 |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 3   | 0   | 1   | 0    | 0    | 0     |
| **200**     | 5   | 2   | 0   | 0    | 0    | 0     |
| **500**     | 0   | 0   | 4   | 1    | 1    | 0     |
| **1000**    | 2   | 0   | 0   | 4    | 0    | 0     |
| **2000**    | 0   | 1   | 0   | 0    | 2    | 0     |
| **10000**   | 1   | 0   | 0   | 0    | 1    | 1     |

### Observaciones

- **100**: Recall alto (0.75) pero precision bajo (0.27). El modelo predice 100 demasiado (8 veces de 29).
- **200**: Precision decente (0.67) pero recall bajo (0.29). 5 de 7 confundidos con 100.
- **500 y 1000**: Las mejores clases (F1=0.73 ambas).
- **2000**: Recall 0.67, 1 confundido con 200.
- **10000**: Precision 1.00 pero recall 0.33 (solo 1 de 3 correcto).
- **Todas las clases tienen recall > 0** ✓

---

## Comparación de Todos los Runs

| Métrica | Run 0 | Run 1 | Run 2 | Run 3 |
|---------|-------|-------|-------|-------|
| Framework | TF 2.21 | TF 2.21 | TF 2.21 | **PyTorch 2.5** |
| Device | CPU | CPU | CPU | **GPU RTX 3060 Ti** |
| Tiempo | ~3 min | ~5 min | ~8 min | **36s** |
| Imágenes | 64 | 112 | 112 | **144** |
| Val accuracy | 38.46% | 26.09% | 65.22% | **55.17%** |
| Macro F1 | 0.29 | 0.14 | 0.65 | **0.55** |
| Clases recall > 0 | 3 | 2 | 6 | **6** |
| Mejor clase (F1) | 1000 (0.80) | 1000 (0.62) | 2000 (1.00) | **500/1000 (0.73)** |

### Progreso visual

```
Run 0: ████████████░░░░░░░░░░  38.46%
Run 1: ████████░░░░░░░░░░░░░░  26.09%
Run 2: █████████████████████░  65.22%
Run 3: █████████████████░░░░░  55.17%
```

---

## Análisis de Run 3

### Lo positivo

1. **Velocidad extrema:** 36 segundos vs 8 minutos de TensorFlow CPU (13x más rápido).
2. **GPU utilizada:** RTX 3060 Ti con CUDA 12.1, imágenes pre-cargadas en VRAM.
3. **Dataset balanceado:** 23-26 imágenes por clase, sin sesgo mayor.
4. **Todas las clases reconocidas:** 6/6 con recall > 0.
5. **500 y 1000 sólidas:** F1=0.73 ambas.

### Lo negativo

1. **Overfitting severo:** Train accuracy 63% → Val accuracy 55%, pero train llega a 82% mientras val se estanca en 55%.
2. **100 y 200 se confunden entre sí:** 5 de 7 billetes de 200 clasificados como 100.
3. **Validation loss inestable:** Sube de 2.13 a 4.97 entre epoch 36 y 48, señal clara de overfitting.
4. **Peor que Run 2:** 55% vs 65%. El dataset más grande no compensó el overfitting.

### Causas del overfitting

- **27.7M parámetros para 115 imágenes de train** → ratio de 240K params por imagen.
- **Augmentation en GPU** puede no ser suficiente para regularizar un modelo tan grande.
- **Dropout 0.4** puede ser insuficiente para 27M params.

---

## Recomendaciones para Run 4

1. **Reducir parámetros drásticamente:** Reemplazar `Flatten` por `GlobalAveragePooling2D` → de 27M a ~1M params.
2. **Aumentar Dropout** a 0.6-0.7.
3. **Reducir dense layers:** `128,D,64` en vez de `256,128,D,64`.
4. **Más datos:** Apuntar a 200+ imágenes totales.
5. **Weight decay más alto:** 0.001 o 0.005.
