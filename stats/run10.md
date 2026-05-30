# Run 10 - Config Restaurada + Early Stopping por Val Loss

**Fecha:** 2026-05-23
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti

---

## Cambios respecto a Run 9

| Aspecto | Run 9 | Run 10 |
|---------|-------|--------|
| Early stopping target | val_acc | **val_loss** (como Run 7) |
| Best val_acc obtenido | 52.38% | **66.67%** (época 33) |

---

## Dataset

186 imágenes balanceadas (31×6). Split estratificado: 144 train / 42 val.

---

## Entrenamiento

- **Época alcanzada:** 88 (early stopping)
- **Mejor época:** 38 (menor val_loss: 2.0203)
- **Tiempo:** 29.5s

### Mejores métricas (época 38 — menor val_loss)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.5528 | **2.0203** |
| Accuracy | 81.94% | **61.90%** |

### Mejor val_acc observado (época 33)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.7038 | **2.4236** |
| Accuracy | 72.92% | **66.67%** |

---

## Evaluación (modelo de época 38)

### Classification Report

| Clase   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| 100     | 0.50      | 0.43   | 0.46     |
| 200     | 1.00      | 0.14   | 0.25     |
| 500     | 1.00      | 1.00   | **1.00** |
| 1000    | 0.70      | 1.00   | **0.82** |
| 2000    | 0.42      | 0.71   | 0.53     |
| 10000   | 0.50      | 0.43   | 0.46     |
| **Accuracy** | | | **0.62** |
| **Macro F1** | | | **0.59** |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 3   | 0   | 0   | 2    | 2    | 0     |
| **200**     | 2   | 1   | 0   | 0    | 2    | 2     |
| **500**     | 0   | 0   | 7   | 0    | 0    | 0     |
| **1000**    | 0   | 0   | 0   | 7    | 0    | 0     |
| **2000**    | 1   | 0   | 0   | 0    | 5    | 1     |
| **10000**   | 0   | 0   | 0   | 1    | 3    | 3     |

---

## Comparación General

| Run | Val Acc | Macro F1 | Método |
|-----|---------|----------|--------|
| Run 4 | 72.41% | 0.66 | val_loss, 144 img, sin estratificar |
| Run 6 | 71.05% | 0.70 | val_loss, 186 img balanceadas |
| Run 7 | **69.05%** | **0.68** | val_loss + split estratificado |
| Run 8 | 52.38% | 0.52 | 224px + val_acc (regresión) |
| Run 9 | 52.38% | 0.48 | config restaurada + val_acc |
| Run 10 | **61.90%** | 0.59 | config restaurada + val_loss |

---

## Diagnóstico

**El modelo es inherentemente inestable con este dataset.** Entre corridas con la misma configuración, la accuracy varía ~10 pp (52%→62%→69%) por la alta varianza del validation loss. La causa raíz es el dataset pequeño (144 train), no la configuración.

Las mejores corridas (4, 6, 7) usaban **early stopping por val_loss** y las peores (8, 9) usaban **val_acc**. El val_loss es más estable como métrica de early stopping aunque no siempre seleccione el punto de mayor accuracy.
