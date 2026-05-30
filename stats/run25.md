# Run 25 — Misma config (verificación)

**Fecha:** 2026-05-29
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 24

Sin cambios en la configuración. Misma arquitectura, mismos hiperparámetros.

---

## Dataset

2762 imágenes (1893 train, 317 val, 552 test). Split balanceado 20% test.

---

## Entrenamiento

- **Época alcanzada:** 115 (early stopping)
- **Mejor época:** 65
- **Mejor val_loss:** 0.9278
- **Mejor val_acc:** **82.65%**
- **Tiempo:** 389.2s (6.5 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.75      | **0.96** | **0.85** | 54      |
| 200     | **0.90**  | 0.69   | 0.78     | 51      |
| 500     | **1.00**  | 0.73   | 0.84     | 55      |
| 1000    | **0.98**  | 0.78   | **0.87** | 69      |
| 2000    | 0.68      | **0.98** | 0.80     | 53      |
| 10000   | 0.78      | 0.83   | 0.81     | 35      |
| **Accuracy** | | | **0.83** | **317** |
| **Macro Avg** | 0.85 | 0.83 | 0.82 | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **52** | 0   | 0   | 0    | 2    | 0     |
| **200**     | 2   | **35** | 0   | 0    | 10   | 4     |
| **500**     | 0   | 4   | **40** | 1    | 7    | 3     |
| **1000**    | 15  | 0   | 0   | **54** | 0    | 0     |
| **2000**    | 0   | 0   | 0   | 0    | **52** | 1     |
| **10000**   | 0   | 0   | 0   | 0    | 6    | **29** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.86      | 0.82   | 0.84     | 92      |
| 200     | 0.84      | 0.77   | 0.80     | 92      |
| 500     | **0.98**  | 0.70   | 0.82     | 92      |
| 1000    | **0.95**  | **0.91** | **0.93** | 92      |
| 2000    | 0.60      | **0.90** | 0.72     | 92      |
| 10000   | 0.82      | 0.78   | 0.80     | 92      |
| **Accuracy** | | | **0.81** | **552** |
| **Macro Avg** | 0.84 | 0.81 | 0.82 | 552 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **75** | 4   | 0   | 0    | 13   | 0     |
| **200**     | 2   | **71** | 0   | 0    | 12   | 7     |
| **500**     | 0   | 10  | **64** | 4    | 13   | 1     |
| **1000**    | 8   | 0   | 0   | **84** | 0    | 0     |
| **2000**    | 1   | 0   | 0   | 0    | **83** | 8     |
| **10000**   | 1   | 0   | 1   | 0    | 18   | **72** |

---

## Evaluación — Training Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.82      | 0.83   | 0.82     | 326     |
| 200     | **0.89**  | 0.81   | **0.85** | 316     |
| 500     | **0.98**  | 0.77   | 0.86     | 256     |
| 1000    | **0.97**  | **0.88** | **0.92** | 406     |
| 2000    | 0.65      | **0.89** | 0.75     | 317     |
| 10000   | 0.80      | 0.81   | 0.80     | 272     |
| **Accuracy** | | | **0.84** | **1893** |
| **Macro Avg** | 0.85 | 0.83 | 0.84 | 1893 |

### Matriz de Confusión (Training)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **271** | 12  | 0   | 1    | 42   | 0     |
| **200**     | 10  | **257** | 0   | 3    | 29   | 17    |
| **500**     | 0   | 19  | **196** | 7    | 32   | 2     |
| **1000**    | 46  | 0   | 0   | **357** | 0    | 3     |
| **2000**    | 1   | 0   | 0   | 1    | **283** | 32    |
| **10000**   | 3   | 0   | 3   | 0    | 47   | **219** |

---

## Comparación con Run 24

| Métrica | Run 24 | Run 25 |
|---------|:------:|:------:|
| **Val accuracy** | **83.91%** | 82.65% |
| **Test accuracy** | **81.52%** | 81.34% |
| **Train accuracy** | 82.61% | **83.52%** |
| **Macro F1 (val)** | **0.84** | 0.82 |
| **Best val_loss** | **0.8982** | 0.9278 |
| **Mejor época** | 46 | 65 |
| **Tiempo** | 316s | 389s |

---

## Análisis

Run 25 es **consistente con Run 24**: test accuracy 81.34% vs 81.52%, val accuracy 82.65% vs 83.91%. La diferencia es marginal y atribuible a la varianza normal del entrenamiento.

### Aciertos

- **Clase 1000 con F1=0.93 en test** — la mejor clase del run.
- **Clase 100 con recall 0.96 en validación** — solo 2 errores en 54 imágenes.
- **Clase 2000 con recall 0.98 en validación y 0.90 en test** — muy pocos falsos negativos.
- **Clase 500 con precision 0.98-1.00** — sigue siendo la clase más distintiva.

### Problemas

- **Precisión de 2000 baja (0.60-0.68):** 13 de 92 imágenes de 100, 12 de 92 de 200, 13 de 92 de 500 y 18 de 92 de 10000 clasificados como 2000 en test. Es la principal fuente de error.
- **Recall de 500 en test (0.70):** 10 clasificados como 200, 13 como 2000.
- **Recall de 200 en test (0.77):** 12 clasificados como 2000.

### Conclusión

Run 25 confirma que la configuración actual produce resultados estables alrededor de **81-82% en test**. La variabilidad es acotada (~1 punto entre Run 24 y 25), mucho menor que la brecha observada con Run 23. Esto sugiere que, en promedio, la configuración converge a ~82%.
