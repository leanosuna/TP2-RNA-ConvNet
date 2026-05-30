# Run 27 — Misma config (verificación)

**Fecha:** 2026-05-29
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 26

Sin cambios en la configuración. Misma arquitectura, mismos hiperparámetros.

---

## Dataset

2762 imágenes (1893 train, 317 val, 552 test). Split balanceado 20% test.

---

## Entrenamiento

- **Época alcanzada:** 88 (early stopping — 50 épocas sin mejora desde época 38)
- **Mejor época:** 38
- **Mejor val_loss:** 1.1441
- **Mejor val_acc:** **73.82%**
- **Tiempo:** 317.7s (5.3 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.69      | **0.91** | 0.78     | 54      |
| 200     | 0.75      | 0.82   | 0.79     | 51      |
| 500     | **1.00**  | 0.51   | 0.67     | 55      |
| 1000    | **0.94**  | 0.64   | 0.76     | 69      |
| 2000    | 0.62      | 0.79   | 0.69     | 53      |
| 10000   | 0.62      | 0.83   | 0.71     | 35      |
| **Accuracy** | | | **0.74** | **317** |
| **Macro Avg** | 0.77 | 0.75 | 0.73 | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **49** | 1   | 0   | 1    | 3    | 0     |
| **200**     | 6   | **42** | 0   | 1    | 1    | 1     |
| **500**     | 0   | 11  | **28** | 1    | 9    | 6     |
| **1000**    | 12  | 1   | 0   | **44** | 7    | 5     |
| **2000**    | 4   | 1   | 0   | 0    | **42** | 6     |
| **10000**   | 0   | 0   | 0   | 0    | 6    | **29** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.71      | **0.86** | 0.77     | 92      |
| 200     | 0.77      | **0.90** | **0.83** | 92      |
| 500     | **1.00**  | 0.60   | 0.75     | 92      |
| 1000    | **0.95**  | 0.77   | **0.85** | 92      |
| 2000    | 0.65      | 0.72   | 0.68     | 92      |
| 10000   | 0.80      | **0.88** | 0.84     | 92      |
| **Accuracy** | | | **0.79** | **552** |
| **Macro Avg** | 0.81 | 0.79 | 0.79 | 552 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **79** | 3   | 0   | 3    | 6    | 1     |
| **200**     | 6   | **83** | 0   | 0    | 1    | 2     |
| **500**     | 0   | 18  | **55** | 1    | 13   | 5     |
| **1000**    | 15  | 0   | 0   | **71** | 6    | 0     |
| **2000**    | 12  | 2   | 0   | 0    | **66** | 12    |
| **10000**   | 0   | 2   | 0   | 0    | 9    | **81** |

---

## Evaluación — Training Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.70      | 0.84   | 0.76     | 326     |
| 200     | 0.80      | **0.90** | **0.85** | 316     |
| 500     | **1.00**  | 0.63   | 0.78     | 256     |
| 1000    | **0.93**  | 0.76   | 0.84     | 406     |
| 2000    | 0.67      | 0.67   | 0.67     | 317     |
| 10000   | 0.72      | **0.88** | 0.79     | 272     |
| **Accuracy** | | | **0.78** | **1893** |
| **Macro Avg** | 0.80 | 0.78 | 0.78 | 1893 |

### Matriz de Confusión (Training)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **274** | 17  | 0   | 14   | 18   | 3     |
| **200**     | 22  | **284** | 0   | 3    | 1    | 6     |
| **500**     | 1   | 42  | **162** | 3    | 36   | 12    |
| **1000**    | 58  | 6   | 0   | **307** | 27   | 8     |
| **2000**    | 35  | 2   | 0   | 1    | **213** | 66    |
| **10000**   | 3   | 5   | 0   | 1    | 23   | **240** |

---

## Comparación con Run 26

| Métrica | Run 26 (mejor) | Run 27 |
|---------|:--------------:|:------:|
| **Val accuracy** | **88.33%** | 73.82% |
| **Test accuracy** | **92.75%** | 78.62% |
| **Train accuracy** | **91.07%** | 77.81% |
| **Macro F1 (val)** | **0.88** | 0.73 |
| **Best val_loss** | **0.7418** | 1.1441 |
| **Mejor época** | **87** | 38 |
| **Tiempo** | 486s | 318s |

---

## Análisis

Run 27 representa el **extremo inferior de la variabilidad** junto con Run 23. Con 78.62% en test, está ~14 puntos por debajo del mejor resultado (Run 26, 92.75%).

### Problemas principales

- **Clase 500 con recall muy bajo (0.51 val, 0.60 test):** 18 de 92 billetes de 500 clasificados como 200 en test, y 11 de 55 en validación. El billete de 500 (verde) se confunde sistemáticamente con el de 200 (azul claro) en este run.
- **Clase 2000 con precisión baja (0.62-0.65):** 12 de 92 de 100 y 9 de 92 de 10000 clasificados como 2000 en test. 6 de 53 de 10000 en val.
- **Clase 1000 también afectada:** 15 clasificados como 100 y 6 como 2000 en test. 12 como 100 en val.
- **Clase 2000 con recall bajo (0.72 test):** 12 clasificados como 10000.

### Clases que se mantienen

- **10000 con recall decente (0.83 val, 0.88 test):** pese al mal resultado general.
- **200 con recall 0.90 en test** y **100 con recall 0.86** — comportamiento aceptable.
- **500 con precision perfecta (1.00)** — ningún falso positivo.

### Early stopping temprano

La mejor época fue la 38 (la más temprana de los 5 runs), y a partir de ahí el val_loss subió sin recuperarse, indicando que el modelo comenzó a divergir temprano.

### Conclusión

Run 27, junto con Run 23, demuestra que **la misma configuración puede producir resultados muy variables** (68% a 93% en test). Para la documentación del TP, sirve como ejemplo de que el éxito del entrenamiento depende no solo de la arquitectura y los hiperparámetros, sino también de factores estocásticos como la inicialización y el data augmentation. El rango de variabilidad observado (68%-93%) sugiere que ejecuciones múltiples y selección del mejor modelo son prácticas recomendadas.
