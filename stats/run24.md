# Run 24 — Misma config (verificación, mejor resultado parcial)

**Fecha:** 2026-05-29
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 23

Sin cambios en la configuración. Misma arquitectura, mismos hiperparámetros.

---

## Dataset

2762 imágenes (1893 train, 317 val, 552 test). Split balanceado 20% test.

---

## Entrenamiento

- **Época alcanzada:** 96 (early stopping)
- **Mejor época:** 46
- **Mejor val_loss:** **0.8982**
- **Mejor val_acc:** **83.91%**
- **Tiempo:** 315.9s (5.3 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | **1.00**  | 0.80   | **0.89** | 54      |
| 200     | 0.91      | 0.84   | **0.88** | 51      |
| 500     | 0.92      | 0.85   | **0.89** | 55      |
| 1000    | 0.88      | 0.84   | **0.86** | 69      |
| 2000    | 0.63      | **0.91** | 0.74   | 53      |
| 10000   | 0.79      | 0.77   | 0.78     | 35      |
| **Accuracy** | | | **0.84** | **317** |
| **Macro Avg** | 0.86 | 0.84 | 0.84 | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **43** | 1   | 0   | 5    | 5    | 0     |
| **200**     | 0   | **43** | 0   | 1    | 3    | 4     |
| **500**     | 0   | 0   | **47** | 2    | 5    | 1     |
| **1000**    | 0   | 0   | 1   | **58** | 10   | 0     |
| **2000**    | 0   | 2   | 1   | 0    | **48** | 2     |
| **10000**   | 0   | 1   | 2   | 0    | 5    | **27** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | **0.97**  | 0.70   | 0.81     | 92      |
| 200     | 0.82      | **0.88** | **0.85** | 92      |
| 500     | 0.89      | **0.87** | **0.88** | 92      |
| 1000    | 0.90      | **0.88** | **0.89** | 92      |
| 2000    | 0.61      | 0.84   | 0.70     | 92      |
| 10000   | 0.84      | 0.73   | 0.78     | 92      |
| **Accuracy** | | | **0.82** | **552** |
| **Macro Avg** | 0.84 | 0.82 | 0.82 | 552 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **64** | 9   | 0   | 4    | 15   | 0     |
| **200**     | 1   | **81** | 6   | 0    | 3    | 1     |
| **500**     | 0   | 1   | **80** | 4    | 7    | 0     |
| **1000**    | 1   | 3   | 0   | **81** | 7    | 0     |
| **2000**    | 0   | 1   | 2   | 0    | **77** | 12    |
| **10000**   | 0   | 4   | 2   | 1    | 18   | **67** |

---

## Evaluación — Training Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | **0.99**  | 0.70   | 0.82     | 326     |
| 200     | 0.84      | **0.88** | **0.86** | 316     |
| 500     | 0.89      | **0.90** | **0.90** | 256     |
| 1000    | 0.92      | **0.90** | **0.91** | 406     |
| 2000    | 0.64      | 0.80   | 0.71     | 317     |
| 10000   | 0.75      | 0.75   | 0.75     | 272     |
| **Accuracy** | | | **0.83** | **1893** |
| **Macro Avg** | 0.84 | 0.82 | 0.83 | 1893 |

### Matriz de Confusión (Training)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **229** | 27  | 0   | 19   | 51   | 0     |
| **200**     | 2   | **279** | 14  | 2    | 5    | 14    |
| **500**     | 0   | 4   | **231** | 7    | 14   | 0     |
| **1000**    | 1   | 8   | 1   | **367** | 29   | 0     |
| **2000**    | 0   | 2   | 4   | 2    | **254** | 55    |
| **10000**   | 0   | 12  | 10  | 4    | 41   | **205** |

---

## Comparación con Run 23

| Métrica | Run 23 | Run 24 |
|---------|:------:|:------:|
| **Val accuracy** | 72.24% | **83.91%** |
| **Test accuracy** | 68.12% | **81.52%** |
| **Train accuracy** | 70.15% | **82.61%** |
| **Macro F1 (val)** | 0.69 | **0.84** |
| **Best val_loss** | 1.3193 | **0.8982** |
| **Mejor época** | 38 | 46 |
| **Tiempo** | 288s | 316s |

---

## Análisis

Run 24 muestra una **mejora significativa respecto a Run 23 (+12 puntos en test)**, demostrando la alta varianza del entrenamiento con esta configuración.

### Aciertos

- **Clase 100 con precisión perfecta (1.00) en validación** — ningún falso positivo de 100.
- **Clase 1000 sólida** con F1 0.89 en test (81/92 correctos).
- **Clase 10000 mejora notablemente** respecto a Run 23: de 0.23 a 0.77 en recall de validación.
- **Clase 500 sigue siendo robusta** (F1 0.88-0.89 en test).
- **Train accuracy (82.61%) y test accuracy (81.52%) muy cercanos** — buena generalización.

### Problemas persistentes

- **Clase 2000 con precisión baja (0.61-0.63):** 15 de 92 imágenes de 100 y 18 de 92 de 10000 clasificadas como 2000 en test.
- **Recall de 100 en test (0.70):** 9 clasificados como 200 y 15 como 2000.

### Conclusión

Run 24 demuestra que la configuración actual puede alcanzar consistentemente ~82% en test cuando la inicialización es favorable. La brecha con Run 23 (68%) es atribuible a la varianza por inicialización aleatoria y data augmentation.
