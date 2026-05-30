# Run 23 — Misma config que Run 22 (verificación)

**Fecha:** 2026-05-29
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 22

Sin cambios en la configuración. Misma arquitectura, mismos hiperparámetros. Ejecutado para generar datos para la documentación del TP (3+ runs requeridos).

---

## Dataset

2762 imágenes (1893 train, 317 val, 552 test). Split balanceado 20% test.

---

## Entrenamiento

- **Época alcanzada:** 88 (early stopping — 50 épocas sin mejora desde época 38)
- **Mejor época:** 38
- **Mejor val_loss:** 1.3193
- **Mejor val_acc:** **72.24%**
- **Tiempo:** 288.4s (4.8 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.72      | 0.78   | 0.75     | 54      |
| 200     | 0.62      | 0.86   | 0.72     | 51      |
| 500     | **1.00**  | 0.84   | **0.91** | 55      |
| 1000    | 0.94      | 0.64   | 0.76     | 69      |
| 2000    | 0.54      | 0.85   | 0.66     | 53      |
| 10000   | 0.73      | 0.23   | 0.35     | 35      |
| **Accuracy** | | | **0.72** | **317** |
| **Macro Avg** | 0.76 | 0.70 | 0.69 | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **42** | 12  | 0   | 0    | 0    | 0     |
| **200**     | 5   | **44** | 0   | 2    | 0    | 0     |
| **500**     | 0   | 4   | **46** | 1    | 3    | 1     |
| **1000**    | 9   | 4   | 0   | **44** | 11   | 1     |
| **2000**    | 0   | 7   | 0   | 0    | **45** | 1     |
| **10000**   | 2   | 0   | 0   | 0    | 25   | **8**  |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.63      | 0.68   | 0.66     | 92      |
| 200     | 0.56      | 0.87   | 0.68     | 92      |
| 500     | **0.99**  | 0.76   | **0.86** | 92      |
| 1000    | 0.89      | 0.67   | 0.77     | 92      |
| 2000    | 0.52      | 0.68   | 0.59     | 92      |
| 10000   | 0.85      | 0.43   | 0.58     | 92      |
| **Accuracy** | | | **0.68** | **552** |
| **Macro Avg** | 0.74 | 0.68 | 0.69 | 552 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **63** | 27  | 0   | 0    | 2    | 0     |
| **200**     | 8   | **80** | 0   | 2    | 1    | 1     |
| **500**     | 0   | 14  | **70** | 4    | 2    | 2     |
| **1000**    | 20  | 1   | 0   | **62** | 9    | 0     |
| **2000**    | 4   | 20  | 0   | 1    | **63** | 4     |
| **10000**   | 5   | 0   | 1   | 1    | 45   | **40** |

---

## Evaluación — Training Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.63      | 0.71   | 0.67     | 326     |
| 200     | 0.60      | 0.85   | 0.71     | 316     |
| 500     | **0.97**  | 0.84   | **0.90** | 256     |
| 1000    | 0.90      | 0.69   | 0.78     | 406     |
| 2000    | 0.54      | 0.65   | 0.59     | 317     |
| 10000   | 0.78      | 0.48   | 0.59     | 272     |
| **Accuracy** | | | **0.70** | **1893** |
| **Macro Avg** | 0.74 | 0.70 | 0.71 | 1893 |

### Matriz de Confusión (Training)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **232** | 83  | 0   | 3    | 8    | 0     |
| **200**     | 26  | **270** | 0   | 13   | 4    | 3     |
| **500**     | 0   | 25  | **215** | 8    | 5    | 3     |
| **1000**    | 76  | 8   | 1   | **280** | 41   | 0     |
| **2000**    | 18  | 60  | 0   | 3    | **206** | 30    |
| **10000**   | 14  | 1   | 5   | 3    | 119  | **130** |

---

## Comparación con Run 22

| Métrica | Run 22 | Run 23 |
|---------|:------:|:------:|
| **Val accuracy** | **81.70%** | 72.24% |
| **Test accuracy** | **81.70%** | 68.12% |
| **Train accuracy** | — | 70.15% |
| **Macro F1 (val)** | 0.82 | 0.69 |
| **Best val_loss** | **0.9062** | 1.3193 |
| **Mejor época** | 49 | 38 |
| **Tiempo** | 374s | 288s |

---

## Análisis

Run 23 muestra el **extremo inferior de la variabilidad**. Con la misma configuración que Run 22 (81.70%), este run cayó a 68.12% en test. La diferencia se debe enteramente a la inicialización aleatoria y al data augmentation estocástico.

### Problemas principales

- **Clase 10000 con recall muy bajo (0.23 val, 0.43 test):** 25 de 35 billetes de 10000 en validación fueron clasificados como 2000. La clase minoritaria es la más afectada.
- **Clase 2000 con precisión baja (0.52-0.54):** muchas imágenes de otras clases (especialmente 100 y 10000) se clasifican como 2000.
- **Confusión masiva 100↔200:** 27 de 92 billetes de 100 en test clasificados como 200.
- **Clase 1000 también afectada:** 20 clasificados como 100 y 9 como 2000 en test.

### Clases fuertes

- **500 se mantiene robusta** con precision 0.99-1.00 incluso en este run, gracias a su color verde distintivo.

### Conclusión

Este run confirma la alta varianza del entrenamiento. Sirve como contraste para demostrar que, si bien la configuración puede alcanzar ~82% (Run 22), también puede producir ~68% dependiendo de la inicialización. Para la documentación del TP, es un ejemplo válido de un entrenamiento "no exitoso" con la misma arquitectura.
