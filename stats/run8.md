# Run 8 - 224px + Val Acc Priority (regresión)

**Fecha:** 2026-05-23
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet + AdaptiveAvgPool2d(4,4) + BatchNorm

---

## Cambios respecto a Run 7

| Aspecto | Run 7 | Run 8 |
|---------|-------|-------|
| Imagen | 160×160 | **224×224** |
| Weight decay | 0.001 | **1e-4** |
| Horizontal flip | 0.5 | **0** |
| MaxPool2d padding | 1 | **0** |
| Early stopping target | val_loss | **val_acc** |

---

## Dataset

186 imágenes balanceadas (31 por clase). Split estratificado: 144 train / 42 val (7×6).

---

## Entrenamiento

- **Época alcanzada:** 71 (early stopping por paciencia)
- **Mejor época:** 21
- **Tiempo:** 46.4s (el doble que Run 7 por 224×224)

### Mejores métricas (época 21)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.6481 | **4.9106** |
| Accuracy | 75.69% | **52.38%** |

---

## Evaluación

### Classification Report

| Clase   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| 100     | 1.00      | 0.43   | 0.60     |
| 200     | 0.60      | 0.43   | 0.50     |
| 500     | 0.83      | 0.71   | 0.77     |
| 1000    | 0.33      | 1.00   | 0.50     |
| 2000    | 0.33      | 0.14   | 0.20     |
| 10000   | 0.75      | 0.43   | 0.55     |
| **Accuracy** | | | **0.52** |
| **Macro F1** | | | **0.52** |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 3   | 0   | 0   | 4    | 0    | 0     |
| **200**     | 0   | 3   | 1   | 2    | 1    | 0     |
| **500**     | 0   | 0   | 5   | 2    | 0    | 0     |
| **1000**    | 0   | 0   | 0   | 7    | 0    | 0     |
| **2000**    | 0   | 2   | 0   | 3    | 1    | 1     |
| **10000**   | 0   | 0   | 0   | 3    | 1    | 3     |

---

## Comparación con Runs Anteriores

| Métrica | Run 4 | Run 6 | Run 7 | **Run 8** |
|---------|-------|-------|-------|-----------|
| Val accuracy | 72.41% | 71.05% | 69.05% | **52.38%** |
| Macro F1 | 0.66 | 0.70 | 0.68 | **0.52** |
| Tiempo | 21s | 20s | 25s | **46s** |
| Resolución | 160 | 160 | 160 | **224** |

---

## Diagnóstico

**Los cambios empeoraron drásticamente el modelo:**

1. **224×224** sin más datos → más parámetros para sobreajustar con el mismo dataset chico
2. **Weight decay 1e-4** (vs 0.001) → menos regularización, el modelo memoriza más rápido
3. **Sin horizontal flip** → menos variedad en aumentos, peor generalización
4. **Val_loss divergió a 9+** mientras train_loss bajó a 0.3 — sobreajuste extremo
5. **Val_acc nunca superó 52%** pese a priorizarlo en early stopping

El modelo predice **1000 como clase default** (13 de 42 predicciones son 1000), señal de que aprendió un sesgo en lugar de patrones reales.
