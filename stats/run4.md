# Run 4 - Entrenamiento PyTorch GPU con AdaptiveAvgPool2d

**Fecha:** 2026-05-21
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti (8.6 GB)
**Modelo:** ConvNet + AdaptiveAvgPool2d(4,4) + BatchNorm
**Archivo del modelo:** `models/convnet_billetes.pth`
**Gráfico de entrenamiento:** `models/convnet_billetes_training.png`
**Runs anteriores:** [run0.md](./run0.md) | [run1.md](./run1.md) | [run2.md](./run2.md) | [run3.md](./run3.md)

---

## Cambios clave respecto a Run 3

| Aspecto | Run 3 | Run 4 |
|---------|-------|-------|
| Spatial pooling | Flatten (100K features) | **AdaptiveAvgPool2d(4,4)** (4K features) |
| Parámetros | 27,693,158 | **1,453,766** (19x menos) |
| Imagen | 160x160 | 160x160 |
| Conv filters | [32, 64, 128] | **[64, 128, 256]** |
| Linear config | 128,D,64 | **256,D,128** |
| Dropout | 0.6 | **0.5** |
| Weight decay | 0.005 | **0.001** |
| Early stop patience | 25 | **30** |

---

## Dataset

144 imágenes balanceadas (23-26 por clase). Train: 115, Validation: 29.

---

## Arquitectura

| Bloque | Detalle |
|--------|---------|
| Conv 1 | Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) |
| Conv 2 | Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPool(2x2) |
| Conv 3 | Conv2D(256, 3x3) + BatchNorm + ReLU + MaxPool(2x2) |
| Pooling | **AdaptiveAvgPool2d(4, 4)** → 256×4×4 = 4,096 features |
| Dense 1 | Linear(4096→256) + ReLU |
| Dropout | Dropout(0.5) |
| Dense 2 | Linear(256→128) + ReLU |
| Output | Linear(128→6) softmax |

**Total parámetros:** 1,453,766 (vs 27.7M en Run 3)

---

## Entrenamiento

- **Época alcanzada:** 71 (early stopping)
- **Mejor época:** 41
- **Tiempo:** 21.3 segundos

### Mejores métricas (época 41)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.8796 | **0.8006** |
| Accuracy | 62.61% | **72.41%** |

---

## Evaluación

### Classification Report

| Clase   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| 100     | 0.60      | 0.75   | 0.67     |
| 200     | 0.78      | 1.00   | **0.88** |
| 500     | 0.83      | 0.83   | **0.83** |
| 1000    | 0.80      | 0.67   | 0.73     |
| 2000    | 1.00      | 0.33   | 0.50     |
| 10000   | 0.33      | 0.33   | 0.33     |
| **Accuracy** | | | **0.72** |
| **Macro F1** | | | **0.66** |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 3   | 0   | 0   | 1    | 0    | 0     |
| **200**     | 0   | 7   | 0   | 0    | 0    | 0     |
| **500**     | 0   | 1   | 5   | 0    | 0    | 0     |
| **1000**    | 2   | 0   | 0   | 4    | 0    | 0     |
| **2000**    | 0   | 0   | 0   | 0    | 1    | 2     |
| **10000**   | 0   | 1   | 1   | 0    | 0    | 1     |

---

## Comparación de Todos los Runs

| Métrica | Run 0 | Run 1 | Run 2 | Run 3 | Run 4 |
|---------|-------|-------|-------|-------|-------|
| Framework | TF CPU | TF CPU | TF CPU | PyT GPU | **PyT GPU** |
| Tiempo | ~3 min | ~5 min | ~8 min | 36s | **21s** |
| Imágenes | 64 | 112 | 112 | 144 | **144** |
| Parámetros | 140K | 140K | 27.7M | 27.7M | **1.45M** |
| **Val accuracy** | 38.46% | 26.09% | 65.22% | 55.17% | **72.41%** |
| **Macro F1** | 0.29 | 0.14 | 0.65 | 0.55 | **0.66** |
| Overfitting | Alto | Alto | Alto | Severo | **Bajo** |

### Progreso visual

```
Run 0: ████████████░░░░░░░░░░  38.46%
Run 1: ████████░░░░░░░░░░░░░░  26.09%
Run 2: █████████████████████░  65.22%
Run 3: █████████████████░░░░░  55.17%
Run 4: ██████████████████████░ 72.41% ← MEJOR
```

---

## Conclusiones

**AdaptiveAvgPool2d(4,4) fue la clave:** redujo parámetros 19x manteniendo información espacial suficiente. El overfitting se redujo drásticamente (train 63% vs val 72%, incluso val > train en la mejor época).

**200 perfecto** (7/7), **500 excelente** (5/6). 2000 y 10000 siguen difíciles por tener solo 3 imágenes en validation.
