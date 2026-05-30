# Run 11 - Arquitectura Chica + Label Smoothing + Weight Decay 0.01

**Fecha:** 2026-05-23
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128 filters, 651K params)
**Archivo del modelo:** `models/convnet_billetes.pth`
**Runs anteriores:** [run0.md](./run0.md) | [run1.md](./run1.md) | [run2.md](./run2.md) | [run3.md](./run3.md) | [run4.md](./run4.md) | [run5.md](./run5.md) | [run6.md](./run6.md) | [run7.md](./run7.md) | [run8.md](./run8.md) | [run9.md](./run9.md) | [run10.md](./run10.md)

---

## Cambios respecto a Run 10

| Aspecto | Run 10 | Run 11 |
|---------|--------|--------|
| Conv filters | [64, 128, 256] | **[32, 64, 128]** |
| Parámetros | 1,453,766 | **651,686** (55% menos) |
| Dropout | 0.5 | **0.3** |
| Weight decay | 0.001 | **0.01** |
| Label smoothing | No | **0.1** |
| MaxPool2d padding | 1 | **0** |

---

## Dataset

186 imágenes balanceadas (31×6). Split estratificado: 144 train / 42 val (7×6).

---

## Arquitectura

| Bloque | Detalle |
|--------|---------|
| Conv 1 | Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2) |
| Conv 2 | Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2) |
| Conv 3 | Conv2D(128, 3×3) + BatchNorm + ReLU + MaxPool(2×2) |
| Pooling | AdaptiveAvgPool2d(4, 4) → 128×4×4 = 2,048 features |
| Dense 1 | Linear(2048→256) + ReLU |
| Dropout | Dropout(**0.3**) |
| Dense 2 | Linear(256→128) + ReLU |
| Output | Linear(128→6) |

**Total parámetros:** 651,686 (vs 1,453,766 en Run 4-10)

---

## Entrenamiento

- **Época alcanzada:** 80 (early stopping)
- **Mejor época:** 30
- **Tiempo:** 19.9 segundos

### Mejores métricas (época 30)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.8408 | **1.8392** |
| Accuracy | 81.25% | **73.81%** |

---

## Evaluación

### Classification Report

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.70      | **1.00** | **0.82** | 7 |
| 200     | 0.67      | 0.29   | 0.40     | 7 |
| 500     | 0.88      | **1.00** | **0.93** | 7 |
| 1000    | 1.00      | 0.86   | **0.92** | 7 |
| 2000    | 0.50      | 0.71   | 0.59     | 7 |
| 10000   | 0.80      | 0.57   | 0.67     | 7 |
| **Accuracy** | | | **0.74** | **42** |
| **Macro Avg** | 0.76 | 0.74 | **0.72** | 42 |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **7** | 0   | 0   | 0    | 0    | 0     |
| **200**     | 3   | 2   | 0   | 0    | 2    | 0     |
| **500**     | 0   | 0   | **7** | 0  | 0    | 0     |
| **1000**    | 0   | 0   | 1   | **6**| 0    | 0     |
| **2000**    | 0   | 1   | 0   | 0    | **5**| 1     |
| **10000**   | 0   | 0   | 0   | 0    | 3    | **4** |

---

## Comparación con Mejores Runs Anteriores

| Métrica | Run 4 | Run 6 | Run 7 | **Run 11** |
|---------|-------|-------|-------|------------|
| Val accuracy | 72.41% | 71.05% | 69.05% | **73.81%** |
| Macro F1 | 0.66 | 0.70 | 0.68 | **0.72** |
| Parámetros | 1.45M | 1.45M | 1.45M | **651K** |
| Tiempo | 21s | 20s | 25s | **20s** |
| Val loss estable | No | No | No | **Sí (~1.8-2.4)** |

### Progreso visual

```
Run 0:  ████████████░░░░░░░░░░  38.46%
Run 1:  ████████░░░░░░░░░░░░░░  26.09%
Run 2:  █████████████████████░  65.22%
Run 3:  █████████████████░░░░░  55.17%
Run 4:  ██████████████████████░ 72.41%
Run 5:  ████████████████░░░░░░  48.28%
Run 6:  ██████████████████████░ 71.05%
Run 7:  █████████████████████░░ 69.05%
Run 8:  ████████████████░░░░░░  52.38%
Run 9:  ████████████████░░░░░░  52.38%
Run 10: ███████████████████░░░  61.90%
Run 11: ███████████████████████ 73.81% ← NUEVO RÉCORD
```

---

## Análisis

**Label smoothing + weight decay 0.01 fueron la clave.** La pérdida de validación se estabilizó completamente (entre 1.8 y 2.4) sin diverger, algo que ninguna corrida anterior había logrado. La arquitectura más chica (651K params) redujo el sobreajuste.

**100 es perfecto** (7/7) y **500 también** (7/7). **1000** casi perfecto (6/7). **10000** mejoró a 4/7 pero sigue confundiéndose con 2000 (3 de 7). **200** es ahora la clase más problemática (solo 2/7, confundido con 100 principalmente).

---

## Próximos Pasos

- Investigar por qué **200** empeoró (3 de 7 clasificados como 100) — posiblemente las imágenes de 200 y 100 se parecen más después del resize a 160×160
- Agregar más imágenes de **200 y 10000** que son las clases con peor recall
- Probar si aumentar las épocas máximas ayuda (el modelo se estabilizó pero cortó temprano por early stopping)
