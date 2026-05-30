# Run 19 - Sin Class Weights

**Fecha:** 2026-05-28
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 18

| Aspecto | Run 18 | Run 19 |
|---------|--------|--------|
| Class weights | Sí (dinámicos) | **No** |
| Criterion | CrossEntropyLoss(weight=...) | CrossEntropyLoss() |

---

## Dataset

2210 imágenes (1686 train, 317 val, 207 test). Sin cambios.

---

## Entrenamiento

- **Época alcanzada:** 72 (early stopping — 50 épocas sin mejora desde época 22)
- **Mejor época:** 22
- **Mejor val_loss:** **0.9968** (récord)
- **Mejor val_acc:** **76.97%**
- **Tiempo:** 206.1s (3.4 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.65      | **0.87** | 0.75     | 54      |
| 200     | **0.89**  | 0.61   | 0.72     | 51      |
| 500     | **0.94**  | **0.84** | **0.88** | 55      |
| 1000    | 0.82      | 0.67   | 0.74     | 69      |
| 2000    | 0.67      | **0.98** | **0.79** | 53      |
| 10000   | 0.81      | 0.63   | 0.71     | 35      |
| **Accuracy** | | | **0.77** | **317** |
| **Macro Avg** | 0.80 | 0.77 | **0.77** | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **47** | 3   | 0   | 4    | 0    | 0     |
| **200**     | 3   | **31** | 2   | 3    | 7    | 5     |
| **500**     | 0   | 1   | **46** | 3    | 5    | 0     |
| **1000**    | 22  | 0   | 0   | **46** | 1    | 0     |
| **2000**    | 0   | 0   | 1   | 0    | **52** | 0     |
| **10000**   | 0   | 0   | 0   | 0    | 13   | **22** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.63      | 0.68   | 0.66     | 38      |
| 200     | **0.88**  | 0.64   | 0.74     | 44      |
| 500     | **0.88**  | **0.78** | **0.82** | 27      |
| 1000    | 0.71      | 0.69   | 0.70     | 42      |
| 2000    | 0.63      | **0.97** | **0.76** | 30      |
| 10000   | **0.91**  | 0.81   | **0.86** | 26      |
| **Accuracy** | | | **0.74** | **207** |
| **Macro Avg** | 0.77 | 0.76 | **0.76** | 207 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **26** | 4   | 0   | 4    | 4    | 0     |
| **200**     | 1   | **28** | 2   | 6    | 5    | 2     |
| **500**     | 0   | 0   | **21** | 2    | 4    | 0     |
| **1000**    | 13  | 0   | 0   | **29** | 0    | 0     |
| **2000**    | 1   | 0   | 0   | 0    | **29** | 0     |
| **10000**   | 0   | 0   | 1   | 0    | 4    | **21** |

---

## Comparación con Run 18

| Métrica | Run 18 (con weights) | Run 19 (sin weights) | Diferencia |
|---------|---------------------|---------------------|------------|
| **Val accuracy** | **79.18%** | 76.97% | **-2.2 pp** |
| **Test accuracy** | **76.81%** | 73.91% | -2.9 pp |
| Macro F1 (val) | **0.79** | 0.77 | -0.02 |
| Best val_loss | 1.0747 | **0.9968** | mejor loss |
| Best epoch | 31 | **22** | más rápido |
| Tiempo | 242s | **206s** | más rápido |

### Cambios por clase (F1 val)

| Clase | Run 18 | Run 19 | Cambio |
|-------|--------|--------|--------|
| 100   | **0.78** | 0.75   | -0.03  |
| 200   | 0.66   | **0.72** | **+0.06** |
| 500   | **0.91** | 0.88   | -0.03  |
| 1000  | **0.75** | 0.74   | -0.01  |
| 2000  | **0.78** | **0.79** | +0.01  |
| 10000 | **0.85** | 0.71   | **-0.14** |

---

## Análisis

### Sin class weights: la clase minoritaria pierde

**10000** fue la más perjudicada: F1 bajó de 0.85 → 0.71. Recall cayó de 0.74 → 0.63 (13 de 35 clasificados como 2000). Sin el weight penalty, el modelo ignora a la clase con menos ejemplos (246 train vs 364 de 1000).

### 200 mejoró ligeramente

Recall subió de 0.49 → 0.61. Sin class weights, 200 ya no está penalizada y compite más equitativamente.

### Mejor val_loss histórico (0.9968)

La pérdida sin weights es más baja porque el criterio no pondera las clases minoritarias. Pero accuracy es peor — la loss más baja no siempre significa mejor clasificación.

---

## Conclusión

**Los class weights siguen siendo beneficiosos.** Run 19 (76.97%) rindió ~2 pp peor que Run 18 (79.18%) al eliminar los weights. La clase 10000 fue la más afectada (F1 -0.14), mientras que 200 mejoró marginalmente (+0.06).

El mejor resultado hasta ahora sigue siendo **Run 17 (80.13% val, 85.02% test)** con class weights + WD 0.01 + arquitectura chica.

Vuelvo a dejar `USE_CLASS_WEIGHTS = True` en config.py.
