# Run 12 - 224×224 con Aspect Ratio Preservation

**Fecha:** 2026-05-23
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, 651K params)

---

## Cambios respecto a Run 11

| Aspecto | Run 11 | Run 12 |
|---------|--------|--------|
| Resolución | 160×160 directo | **224×224 via Resize(256)+CenterCrop(224)** |
| Aspect ratio | Squash | **Preservado** |
| Tiempo | 19.9s | **43.5s** |

---

## Dataset

186 imágenes balanceadas (31×6). Split estratificado: 144 train / 42 val.

---

## Entrenamiento

- **Época alcanzada:** 88 (early stopping)
- **Mejor época:** 38
- **Tiempo:** 43.5s

### Mejores métricas (época 38)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.6617 | **1.5952** (mejor val_loss histórico) |
| Accuracy | 90.97% | **66.67%** |

---

## Evaluación

### Classification Report

| Clase   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| 100     | 0.62      | 0.71   | 0.67     |
| 200     | 0.38      | 0.43   | 0.40     |
| 500     | 0.78      | 1.00   | **0.88** |
| 1000    | 1.00      | 0.71   | **0.83** |
| 2000    | 0.56      | 0.71   | 0.62     |
| 10000   | 1.00      | 0.43   | 0.60     |
| **Accuracy** | | | **0.67** |
| **Macro F1** | | | **0.67** |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 5   | 2   | 0   | 0    | 0    | 0     |
| **200**     | 2   | 3   | 1   | 0    | 1    | 0     |
| **500**     | 0   | 0   | 7   | 0    | 0    | 0     |
| **1000**    | 1   | 0   | 1   | 5    | 0    | 0     |
| **2000**    | 0   | 2   | 0   | 0    | 5    | 0     |
| **10000**   | 0   | 1   | 0   | 0    | 3    | 3     |

---

## Comparación con Run 11

| Métrica | Run 11 (160×160) | Run 12 (224×224) |
|---------|-----------------|-----------------|
| Val accuracy | **73.81%** | 66.67% |
| Macro F1 | **0.72** | 0.67 |
| Val loss | 1.8392 | **1.5952** |
| Tiempo | **19.9s** | 43.5s |

---

## Conclusión

Mayor resolución no ayudó. Pese a preservar aspect ratio, el val_acc bajó de 73.81% → 66.67% y el tiempo se duplicó. El dataset de 144 imágenes es muy chico para beneficiarse de 224×224 — los píxeles extra son principalmente ruido que el modelo sobreajusta. La configuración de **Run 11 (160×160)** sigue siendo la mejor.
