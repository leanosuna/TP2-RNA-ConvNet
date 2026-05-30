# Run 7 - Split Estratificado + Early Stopping Extendido

**Fecha:** 2026-05-23
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti (8.6 GB)
**Modelo:** ConvNet + AdaptiveAvgPool2d(4,4) + BatchNorm (misma arquitectura)
**Archivo del modelo:** `models/convnet_billetes.pth`
**Gráfico de entrenamiento:** `models/convnet_billetes_training.png`
**Runs anteriores:** [run0.md](./run0.md) | [run1.md](./run1.md) | [run2.md](./run2.md) | [run3.md](./run3.md) | [run4.md](./run4.md) | [run5.md](./run5.md) | [run6.md](./run6.md)

---

## Cambios respecto a Run 6

| Aspecto | Run 6 | Run 7 |
|---------|-------|-------|
| Split | Aleatorio simple | **Estratificado (7 imágenes por clase en val)** |
| Early stopping patience | 30 | **50** |
| Class weights | No | No |
| Val set | 38 (4–12 por clase) | **42 (7 exactas por clase)** |

---

## Dataset

| Clase   | Train | Val |
|---------|-------|-----|
| 100     | 24    | 7   |
| 200     | 24    | 7   |
| 500     | 24    | 7   |
| 1000    | 24    | 7   |
| 2000    | 24    | 7   |
| 10000   | 24    | 7   |
| **Total** | **144** | **42** |

---

## Entrenamiento

- **Época alcanzada:** 77 (early stopping, paciencia 50)
- **Mejor época:** 27
- **Tiempo:** 25.1 segundos

### Mejores métricas (época 27)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.5959 | **2.2134** |
| Accuracy | 76.39% | **69.05%** |

---

## Evaluación

### Classification Report

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.71      | 0.71   | **0.71** | 7 |
| 200     | 0.75      | 0.43   | 0.55     | 7 |
| 500     | 0.70      | 1.00   | **0.82** | 7 |
| 1000    | 0.75      | 0.86   | **0.80** | 7 |
| 2000    | 0.56      | 0.71   | 0.62     | 7 |
| 10000   | 0.75      | 0.43   | 0.55     | 7 |
| **Accuracy** | | | **0.69** | **42** |
| **Macro Avg** | 0.70 | 0.69 | **0.68** | 42 |

### Matriz de Confusión

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | 5   | 0   | 0   | 2    | 0    | 0     |
| **200**     | 2   | 3   | 1   | 0    | 1    | 0     |
| **500**     | 0   | 0   | 7   | 0    | 0    | 0     |
| **1000**    | 0   | 0   | 1   | 6    | 0    | 0     |
| **2000**    | 0   | 1   | 0   | 0    | 5    | 1     |
| **10000**   | 0   | 0   | 1   | 0    | 3    | 3     |

---

## Comparación con Runs Anteriores

| Métrica | Run 4 | Run 6 | Run 7 |
|---------|-------|-------|-------|
| Val accuracy | 72.41% | 71.05% | 69.05% |
| Macro F1 | 0.66 | **0.70** | 0.68 |
| Val support | 29 | 38 | **42** |
| **100 F1** | 0.67 | 0.40 | **0.71** |
| **200 F1** | **0.88** | 0.80 | 0.55 |
| **500 F1** | 0.83 | **0.89** | 0.82 |
| **1000 F1** | 0.73 | 0.82 | 0.80 |
| **2000 F1** | 0.50 | 0.67 | 0.62 |
| **10000 F1** | 0.33 | 0.60 | 0.55 |

### Progreso visual

```
Run 0: ████████████░░░░░░░░░░  38.46%
Run 1: ████████░░░░░░░░░░░░░░  26.09%
Run 2: █████████████████████░  65.22%
Run 3: █████████████████░░░░░  55.17%
Run 4: ██████████████████████░ 72.41%
Run 5: ████████████████░░░░░░  48.28%
Run 6: ██████████████████████░ 71.05%
Run 7: █████████████████████░░ 69.05%
```

---

## Problemas Persistentes

1. **Val loss nunca converge** — oscila entre 1.7 y 4.7 durante todo el entrenamiento, sin tendencia a la baja. El early stopping se activa siempre por paciencia, no por mejora real.

2. **Sobreajiste severo** — train_loss baja a 0.31 mientras val_loss se mantiene en ~4. El modelo memoriza en vez de generalizar.

3. **500 perfecto** (7/7), pero **200 y 10000 se caen** — 200 tiene solo 3/7 (confundido con 100 y 500), 10000 tiene 3/7 (confundido con 2000 principalmente).

4. **Dropout 0.5 + weight_decay 0.001 insuficientes** para controlar el sobreajuste con solo 144 imágenes de entrenamiento.

---

## Próximos Pasos Recomendados

- Reducir **Dropout de 0.5 → 0.3** para permitir mejor aprendizaje (el modelo parece undertraineado)
- Aumentar **weight_decay a 0.01** para regularización más fuerte
- Probar **Label Smoothing** en CrossEntropyLoss para reducir confianza excesiva
- Agregar más imágenes o más variedad de data augmentation para que el modelo generalice mejor
