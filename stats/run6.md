# Run 6 - Dataset Balanceado + Sin Class Weights

**Fecha:** 2026-05-23
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti (8.6 GB)
**Modelo:** ConvNet + AdaptiveAvgPool2d(4,4) + BatchNorm (misma arquitectura Run 4)
**Archivo del modelo:** `models/convnet_billetes.pth`
**Gráfico de entrenamiento:** `models/convnet_billetes_training.png`
**Runs anteriores:** [run0.md](./run0.md) | [run1.md](./run1.md) | [run2.md](./run2.md) | [run3.md](./run3.md) | [run4.md](./run4.md) | [run5.md](./run5.md)

---

## Cambios clave respecto a Run 4

| Aspecto | Run 4 | Run 6 |
|---------|-------|-------|
| Imágenes totales | 144 | **186** |
| Balance por clase | 23–26 | **31 exactas** |
| Class weights | Sí | **No** (ya no necesario) |

---

## Dataset

186 imágenes balanceadas (31 por clase). Train: 148, Validation: 38.

| Clase   | Train | Val |
|---------|-------|-----|
| 100     | 27    | 4   |
| 200     | 26    | 5   |
| 500     | 27    | 4   |
| 1000    | 19    | 12  |
| 2000    | 25    | 6   |
| 10000   | 24    | 7   |
| **Total** | **148** | **38** |

---

## Arquitectura (idéntica a Run 4)

| Bloque | Detalle |
|--------|---------|
| Conv 1 | Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) |
| Conv 2 | Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPool(2x2) |
| Conv 3 | Conv2D(256, 3x3) + BatchNorm + ReLU + MaxPool(2x2) |
| Pooling | AdaptiveAvgPool2d(4, 4) → 256×4×4 = 4,096 features |
| Dense 1 | Linear(4096→256) + ReLU |
| Dropout | Dropout(0.5) |
| Dense 2 | Linear(256→128) + ReLU |
| Output | Linear(128→6) |

**Total parámetros:** 1,453,766

---

## Entrenamiento

- **Época alcanzada:** 55 (early stopping)
- **Mejor época:** 25
- **Tiempo:** 20.0 segundos

### Mejores métricas (época 25)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.8390 | **0.9600** |
| Accuracy | 70.27% | **71.05%** |

---

## Evaluación

### Classification Report

| Clase   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| 100     | 0.33      | 0.50   | 0.40     |
| 200     | 0.80      | 0.80   | **0.80** |
| 500     | 0.80      | 1.00   | **0.89** |
| 1000    | 0.90      | 0.75   | **0.82** |
| 2000    | 0.56      | 0.83   | **0.67** |
| 10000   | 1.00      | 0.43   | 0.60     |
| **Accuracy** | | | **0.71** |
| **Macro F1** | | | **0.70** |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 2   | 1   | 0   | 1    | 0    | 0     |
| **200**     | 1   | 4   | 0   | 0    | 0    | 0     |
| **500**     | 0   | 0   | 4   | 0    | 0    | 0     |
| **1000**    | 2   | 0   | 1   | 9    | 0    | 0     |
| **2000**    | 1   | 0   | 0   | 0    | 5    | 0     |
| **10000**   | 0   | 0   | 0   | 0    | 4    | 3     |

---

## Comparación con Run 4

| Aspecto | Run 4 | Run 6 | Diferencia |
|---------|-------|-------|------------|
| Imágenes | 144 | **186** | +42 |
| Val accuracy | 72.41% | 71.05% | -1.36 pp |
| Macro F1 | 0.66 | **0.70** | +0.04 |
| **2000 recall** | 33% | **83%** | **+50 pp** |
| **10000 recall** | 33% | **43%** | +10 pp |
| Tiempo | 21.3s | 20.0s | similar |

---

## Comparación de Todos los Runs

| Métrica | Run 0 | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |
|---------|-------|-------|-------|-------|-------|-------|-------|
| Framework | TF CPU | TF CPU | TF CPU | PyT GPU | PyT GPU | PyT GPU | **PyT GPU** |
| Tiempo | ~3 min | ~5 min | ~8 min | 36s | 21s | 27s | **20s** |
| Imágenes | 64 | 112 | 112 | 144 | 144 | 144 | **186** |
| Parámetros | 140K | 140K | 27.7M | 27.7M | 1.45M | 2.04M | **1.45M** |
| **Val accuracy** | 38.46% | 26.09% | 65.22% | 55.17% | 72.41% | 48.28% | **71.05%** |
| **Macro F1** | 0.29 | 0.14 | 0.65 | 0.55 | 0.66 | — | **0.70** |
| Overfitting | Alto | Alto | Alto | Severo | Bajo | Alto | **Bajo** |

### Progreso visual

```
Run 0: ████████████░░░░░░░░░░  38.46%
Run 1: ████████░░░░░░░░░░░░░░  26.09%
Run 2: █████████████████████░  65.22%
Run 3: █████████████████░░░░░  55.17%
Run 4: ██████████████████████░ 72.41%
Run 5: ████████████████░░░░░░  48.28%
Run 6: ██████████████████████░ 71.05%
```

---

## Conclusiones

**El balanceo de clases funcionó.** 2000 pasó de 33% → 83% recall, y 10000 mejoró de 33% → 43%. La accuracy se mantuvo similar (~71%) pese a tener más imágenes y sin usar class weights.

**Clases problemáticas restantes:** 100 (solo 2/4 correcta) y 10000 (3/7, confundida principalmente con 2000). Siguen siendo las más difíciles de distinguir visualmente.
