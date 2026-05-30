# Reporte de Entrenamientos — Runs 23 a 27

## Resumen

Últimos 5 runs con la configuración óptima determinada en los experimentos previos: arquitectura de 651K parámetros (3 bloques Conv → BatchNorm → ReLU → MaxPool, AdaptiveAvgPool2d(4,4), Dropout 0.3, capas FC 2048→256→128→6), weight decay 0.01, label smoothing 0.1, class weights balanceados, data augmentation online. Todos ejecutados con el dataset rebalanceado (1893 train / 552 test / 317 valid).

---

# Run 23

## Información general

| Parámetro | Valor |
|:----------|:------|
| Mejor época | 38 |
| Mejor val_loss | 1.3193 |
| Mejor val_acc | 72.24% |
| Tiempo de entrenamiento | 288.4s (4.8 min) |
| Épocas totales | 88 (early stopping) |
| Total parámetros | 651,686 |

## Métricas por conjunto

### Validation Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.72 | 0.78 | 0.75 | 54 |
| 200 | 0.62 | 0.86 | 0.72 | 51 |
| 500 | 1.00 | 0.84 | 0.91 | 55 |
| 1000 | 0.94 | 0.64 | 0.76 | 69 |
| 2000 | 0.54 | 0.85 | 0.66 | 53 |
| 10000 | 0.73 | 0.23 | 0.35 | 35 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.72 | 0.76 / 0.70 / 0.69 | 0.77 / 0.72 / 0.72 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 42 | 12 | 0 | 0 | 0 | 0 |
| **200** | 5 | 44 | 0 | 2 | 0 | 0 |
| **500** | 0 | 4 | 46 | 1 | 3 | 1 |
| **1000** | 9 | 4 | 0 | 44 | 11 | 1 |
| **2000** | 0 | 7 | 0 | 0 | 45 | 1 |
| **10000** | 2 | 0 | 0 | 0 | 25 | 8 |

### Test Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.63 | 0.68 | 0.66 | 92 |
| 200 | 0.56 | 0.87 | 0.68 | 92 |
| 500 | 0.99 | 0.76 | 0.86 | 92 |
| 1000 | 0.89 | 0.67 | 0.77 | 92 |
| 2000 | 0.52 | 0.68 | 0.59 | 92 |
| 10000 | 0.85 | 0.43 | 0.58 | 92 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.68 | 0.74 / 0.68 / 0.69 | 0.74 / 0.68 / 0.69 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 63 | 27 | 0 | 0 | 2 | 0 |
| **200** | 8 | 80 | 0 | 2 | 1 | 1 |
| **500** | 0 | 14 | 70 | 4 | 2 | 2 |
| **1000** | 20 | 1 | 0 | 62 | 9 | 0 |
| **2000** | 4 | 20 | 0 | 1 | 63 | 4 |
| **10000** | 5 | 0 | 1 | 1 | 45 | 40 |

### Training Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.63 | 0.71 | 0.67 | 326 |
| 200 | 0.60 | 0.85 | 0.71 | 316 |
| 500 | 0.97 | 0.84 | 0.90 | 256 |
| 1000 | 0.90 | 0.69 | 0.78 | 406 |
| 2000 | 0.54 | 0.65 | 0.59 | 317 |
| 10000 | 0.78 | 0.48 | 0.59 | 272 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.70 | 0.74 / 0.70 / 0.71 | 0.74 / 0.70 / 0.71 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 232 | 83 | 0 | 3 | 8 | 0 |
| **200** | 26 | 270 | 0 | 13 | 4 | 3 |
| **500** | 0 | 25 | 215 | 8 | 5 | 3 |
| **1000** | 76 | 8 | 1 | 280 | 41 | 0 |
| **2000** | 18 | 60 | 0 | 3 | 206 | 30 |
| **10000** | 14 | 1 | 5 | 3 | 119 | 130 |

## Análisis

Run 23 representa la línea base inferior de las 5 corridas. Se observa:
- **Clase 10000 con recall muy bajo (0.23 en val, 0.43 en test, 0.48 en train)**: severamente afectada, confundida principalmente con 2000.
- **Clase 2000 con precisión baja (0.54 en val, 0.52 en test)**: muchos falsos positivos, especialmente de 100 y 10000.
- **Clase 500 con excelente precisión (0.99-1.00)**: billete de color verde fácilmente distinguible.
- **Confusión masiva 100↔200 (billetes rojos)** y **2000↔10000 (violetas)**.

---

# Run 24

## Información general

| Parámetro | Valor |
|:----------|:------|
| Mejor época | 46 |
| Mejor val_loss | 0.8982 |
| Mejor val_acc | 83.91% |
| Tiempo de entrenamiento | 315.9s (5.3 min) |
| Épocas totales | 96 (early stopping) |
| Total parámetros | 651,686 |

## Métricas por conjunto

### Validation Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 1.00 | 0.80 | 0.89 | 54 |
| 200 | 0.91 | 0.84 | 0.88 | 51 |
| 500 | 0.92 | 0.85 | 0.89 | 55 |
| 1000 | 0.88 | 0.84 | 0.86 | 69 |
| 2000 | 0.63 | 0.91 | 0.74 | 53 |
| 10000 | 0.79 | 0.77 | 0.78 | 35 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.84 | 0.86 / 0.84 / 0.84 | 0.86 / 0.84 / 0.84 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 43 | 1 | 0 | 5 | 5 | 0 |
| **200** | 0 | 43 | 0 | 1 | 3 | 4 |
| **500** | 0 | 0 | 47 | 2 | 5 | 1 |
| **1000** | 0 | 0 | 1 | 58 | 10 | 0 |
| **2000** | 0 | 2 | 1 | 0 | 48 | 2 |
| **10000** | 0 | 1 | 2 | 0 | 5 | 27 |

### Test Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.97 | 0.70 | 0.81 | 92 |
| 200 | 0.82 | 0.88 | 0.85 | 92 |
| 500 | 0.89 | 0.87 | 0.88 | 92 |
| 1000 | 0.90 | 0.88 | 0.89 | 92 |
| 2000 | 0.61 | 0.84 | 0.70 | 92 |
| 10000 | 0.84 | 0.73 | 0.78 | 92 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.82 | 0.84 / 0.82 / 0.82 | 0.84 / 0.82 / 0.82 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 64 | 9 | 0 | 4 | 15 | 0 |
| **200** | 1 | 81 | 6 | 0 | 3 | 1 |
| **500** | 0 | 1 | 80 | 4 | 7 | 0 |
| **1000** | 1 | 3 | 0 | 81 | 7 | 0 |
| **2000** | 0 | 1 | 2 | 0 | 77 | 12 |
| **10000** | 0 | 4 | 2 | 1 | 18 | 67 |

### Training Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.99 | 0.70 | 0.82 | 326 |
| 200 | 0.84 | 0.88 | 0.86 | 316 |
| 500 | 0.89 | 0.90 | 0.90 | 256 |
| 1000 | 0.92 | 0.90 | 0.91 | 406 |
| 2000 | 0.64 | 0.80 | 0.71 | 317 |
| 10000 | 0.75 | 0.75 | 0.75 | 272 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.83 | 0.84 / 0.82 / 0.83 | 0.84 / 0.83 / 0.83 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 229 | 27 | 0 | 19 | 51 | 0 |
| **200** | 2 | 279 | 14 | 2 | 5 | 14 |
| **500** | 0 | 4 | 231 | 7 | 14 | 0 |
| **1000** | 1 | 8 | 1 | 367 | 29 | 0 |
| **2000** | 0 | 2 | 4 | 2 | 254 | 55 |
| **10000** | 0 | 12 | 10 | 4 | 41 | 205 |

## Análisis

Run 24 muestra una mejora significativa respecto a Run 23:
- **Val accuracy 83.91%** — incremento de ~12 puntos porcentuales sobre Run 23
- **Clase 100 con precisión perfecta (1.00) en val**: el modelo aprendió a identificar billetes de 100 sin falsos positivos
- **Clase 10000 mejora notablemente**: de 0.23 a 0.77 en recall de validación
- **Clase 500 sigue destacando** con alto rendimiento en todos los conjuntos
- **Clase 2000 sigue siendo la más débil** en precisión (0.63 val, 0.61 test), confundida con 10000
- La diferencia entre train (82.61%) y test (81.52%) es mínima, indicando buena generalización
- Se estabiliza respecto a Run 23: la inicialización aleatoria esta vez favoreció una mejor convergencia

---

# Run 25

## Información general

| Parámetro | Valor |
|:----------|:------|
| Mejor época | 65 |
| Mejor val_loss | 0.9278 |
| Mejor val_acc | 82.65% |
| Tiempo de entrenamiento | 389.2s (6.5 min) |
| Épocas totales | 115 (early stopping) |
| Total parámetros | 651,686 |

## Métricas por conjunto

### Validation Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.75 | 0.96 | 0.85 | 54 |
| 200 | 0.90 | 0.69 | 0.78 | 51 |
| 500 | 1.00 | 0.73 | 0.84 | 55 |
| 1000 | 0.98 | 0.78 | 0.87 | 69 |
| 2000 | 0.68 | 0.98 | 0.80 | 53 |
| 10000 | 0.78 | 0.83 | 0.81 | 35 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.83 | 0.85 / 0.83 / 0.82 | 0.86 / 0.83 / 0.83 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 52 | 0 | 0 | 0 | 2 | 0 |
| **200** | 2 | 35 | 0 | 0 | 10 | 4 |
| **500** | 0 | 4 | 40 | 1 | 7 | 3 |
| **1000** | 15 | 0 | 0 | 54 | 0 | 0 |
| **2000** | 0 | 0 | 0 | 0 | 52 | 1 |
| **10000** | 0 | 0 | 0 | 0 | 6 | 29 |

### Test Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.86 | 0.82 | 0.84 | 92 |
| 200 | 0.84 | 0.77 | 0.80 | 92 |
| 500 | 0.98 | 0.70 | 0.82 | 92 |
| 1000 | 0.95 | 0.91 | 0.93 | 92 |
| 2000 | 0.60 | 0.90 | 0.72 | 92 |
| 10000 | 0.82 | 0.78 | 0.80 | 92 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.81 | 0.84 / 0.81 / 0.82 | 0.84 / 0.81 / 0.82 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 75 | 4 | 0 | 0 | 13 | 0 |
| **200** | 2 | 71 | 0 | 0 | 12 | 7 |
| **500** | 0 | 10 | 64 | 4 | 13 | 1 |
| **1000** | 8 | 0 | 0 | 84 | 0 | 0 |
| **2000** | 1 | 0 | 0 | 0 | 83 | 8 |
| **10000** | 1 | 0 | 1 | 0 | 18 | 72 |

### Training Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.82 | 0.83 | 0.82 | 326 |
| 200 | 0.89 | 0.81 | 0.85 | 316 |
| 500 | 0.98 | 0.77 | 0.86 | 256 |
| 1000 | 0.97 | 0.88 | 0.92 | 406 |
| 2000 | 0.65 | 0.89 | 0.75 | 317 |
| 10000 | 0.80 | 0.81 | 0.80 | 272 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.84 | 0.85 / 0.83 / 0.84 | 0.86 / 0.84 / 0.84 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 271 | 12 | 0 | 1 | 42 | 0 |
| **200** | 10 | 257 | 0 | 3 | 29 | 17 |
| **500** | 0 | 19 | 196 | 7 | 32 | 2 |
| **1000** | 46 | 0 | 0 | 357 | 0 | 3 |
| **2000** | 1 | 0 | 0 | 1 | 283 | 32 |
| **10000** | 3 | 0 | 3 | 0 | 47 | 219 |

## Análisis

Run 25 resultados muy similares a Run 24, con accuracy general ligeramente inferior:
- **Val accuracy 82.65%** vs 83.91% de Run 24 — dentro de la variabilidad esperada
- **Test accuracy 81.34%** — consistente con val, buen balance
- **Clase 1000 con F1=0.93 en test** — la clase mejor clasificada en este run
- **Patrón consistente de confusión 2000↔10000**: 18 de 92 billetes de 10000 fueron clasificados como 2000 en test, y viceversa (8 de 92)
- **Clase 100 con recall alto en val (0.96)**: muy pocos falsos negativos
- Entrenó más épocas (115) que Run 24 (96) pero con resultados ligeramente peores
- La precisión 2000 (0.60 test, 0.68 val) sigue siendo el talón de Aquiles

---

# Run 26 — MEJOR RESULTADO

## Información general

| Parámetro | Valor |
|:----------|:------|
| Mejor época | 87 |
| Mejor val_loss | 0.7418 |
| Mejor val_acc | **88.33%** |
| Tiempo de entrenamiento | 485.9s (8.1 min) |
| Épocas totales | 137 (early stopping) |
| Total parámetros | 651,686 |

## Métricas por conjunto

### Validation Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.82 | 0.94 | 0.88 | 54 |
| 200 | 0.91 | 0.80 | 0.85 | 51 |
| 500 | 1.00 | 0.95 | 0.97 | 55 |
| 1000 | 0.89 | 0.90 | 0.89 | 69 |
| 2000 | 0.91 | 0.77 | 0.84 | 53 |
| 10000 | 0.77 | 0.94 | 0.85 | 35 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| **0.88** | 0.88 / 0.88 / 0.88 | 0.89 / 0.88 / 0.88 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 51 | 1 | 0 | 1 | 1 | 0 |
| **200** | 4 | 41 | 0 | 1 | 1 | 4 |
| **500** | 0 | 0 | 52 | 3 | 0 | 0 |
| **1000** | 5 | 0 | 0 | 62 | 0 | 2 |
| **2000** | 2 | 3 | 0 | 3 | 41 | 4 |
| **10000** | 0 | 0 | 0 | 0 | 2 | 33 |

### Test Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.87 | 0.95 | 0.91 | 92 |
| 200 | 0.97 | 0.91 | 0.94 | 92 |
| 500 | 1.00 | 0.92 | 0.96 | 92 |
| 1000 | 0.91 | 0.91 | 0.91 | 92 |
| 2000 | 0.95 | 0.88 | 0.92 | 92 |
| 10000 | 0.88 | 0.99 | 0.93 | 92 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| **0.93** | 0.93 / 0.93 / 0.93 | 0.93 / 0.93 / 0.93 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 87 | 2 | 0 | 0 | 3 | 0 |
| **200** | 4 | 84 | 0 | 2 | 0 | 2 |
| **500** | 0 | 0 | 85 | 4 | 1 | 2 |
| **1000** | 7 | 0 | 0 | 84 | 0 | 1 |
| **2000** | 2 | 1 | 0 | 1 | 81 | 7 |
| **10000** | 0 | 0 | 0 | 1 | 0 | 91 |

### Training Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.88 | 0.92 | 0.90 | 326 |
| 200 | 0.94 | 0.89 | 0.92 | 316 |
| 500 | 1.00 | 0.96 | 0.98 | 256 |
| 1000 | 0.93 | 0.94 | 0.93 | 406 |
| 2000 | 0.95 | 0.79 | 0.86 | 317 |
| 10000 | 0.80 | 0.98 | 0.88 | 272 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.91 | 0.92 / 0.91 / 0.91 | 0.92 / 0.91 / 0.91 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 301 | 13 | 0 | 3 | 9 | 0 |
| **200** | 15 | 282 | 0 | 6 | 1 | 12 |
| **500** | 0 | 1 | 246 | 6 | 1 | 2 |
| **1000** | 19 | 0 | 0 | 380 | 2 | 5 |
| **2000** | 8 | 2 | 0 | 9 | 251 | 47 |
| **10000** | 0 | 1 | 0 | 3 | 1 | 267 |

## Análisis

Run 26 es **el mejor resultado de los 5 runs** y el mejor registro histórico del proyecto:
- **Val accuracy 88.33%** — récord absoluto
- **Test accuracy 92.75%** — el más alto de todos los runs (92 de 552 imágenes mal clasificadas)
- **Train accuracy 91.07%** — consistente con test

**Destacados por clase (test):**
- **Clase 500 con precisión perfecta 1.00** — ningún falso positivo
- **Clase 2000 con precisión 0.95** — mejora drástica respecto a runs anteriores (0.52-0.65)
- **Clase 10000 con recall de 0.99** — 91 de 92 billetes de 10000 correctamente identificados
- **Clase 100 con recall 0.95** — solo 5 errores en 92 imágenes
- **Clase 200 con F1=0.94** — excelente balance precisión-recall

**Matriz de confusión casi diagonal** en test: las confusiones se redujeron al mínimo. Los pares problemáticos históricos (100↔200, 2000↔10000) mejoraron sustancialmente.

Entrenó más épocas (137) que los otros runs, lo que combinado con una inicialización favorable, permitió alcanzar esta performance superior.

---

# Run 27

## Información general

| Parámetro | Valor |
|:----------|:------|
| Mejor época | 38 |
| Mejor val_loss | 1.1441 |
| Mejor val_acc | 73.82% |
| Tiempo de entrenamiento | 317.7s (5.3 min) |
| Épocas totales | 88 (early stopping) |
| Total parámetros | 651,686 |

## Métricas por conjunto

### Validation Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.69 | 0.91 | 0.78 | 54 |
| 200 | 0.75 | 0.82 | 0.79 | 51 |
| 500 | 1.00 | 0.51 | 0.67 | 55 |
| 1000 | 0.94 | 0.64 | 0.76 | 69 |
| 2000 | 0.62 | 0.79 | 0.69 | 53 |
| 10000 | 0.62 | 0.83 | 0.71 | 35 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.74 | 0.77 / 0.75 / 0.73 | 0.79 / 0.74 / 0.74 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 49 | 1 | 0 | 1 | 3 | 0 |
| **200** | 6 | 42 | 0 | 1 | 1 | 1 |
| **500** | 0 | 11 | 28 | 1 | 9 | 6 |
| **1000** | 12 | 1 | 0 | 44 | 7 | 5 |
| **2000** | 4 | 1 | 0 | 0 | 42 | 6 |
| **10000** | 0 | 0 | 0 | 0 | 6 | 29 |

### Test Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.71 | 0.86 | 0.77 | 92 |
| 200 | 0.77 | 0.90 | 0.83 | 92 |
| 500 | 1.00 | 0.60 | 0.75 | 92 |
| 1000 | 0.95 | 0.77 | 0.85 | 92 |
| 2000 | 0.65 | 0.72 | 0.68 | 92 |
| 10000 | 0.80 | 0.88 | 0.84 | 92 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.79 | 0.81 / 0.79 / 0.79 | 0.81 / 0.79 / 0.79 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 79 | 3 | 0 | 3 | 6 | 1 |
| **200** | 6 | 83 | 0 | 0 | 1 | 2 |
| **500** | 0 | 18 | 55 | 1 | 13 | 5 |
| **1000** | 15 | 0 | 0 | 71 | 6 | 0 |
| **2000** | 12 | 2 | 0 | 0 | 66 | 12 |
| **10000** | 0 | 2 | 0 | 0 | 9 | 81 |

### Training Set

| Clase | Precisión | Recuperación | F1-Score | Support |
|:----:|:---------:|:------------:|:--------:|:-------:|
| 100 | 0.70 | 0.84 | 0.76 | 326 |
| 200 | 0.80 | 0.90 | 0.85 | 316 |
| 500 | 1.00 | 0.63 | 0.78 | 256 |
| 1000 | 0.93 | 0.76 | 0.84 | 406 |
| 2000 | 0.67 | 0.67 | 0.67 | 317 |
| 10000 | 0.72 | 0.88 | 0.79 | 272 |

| Accuracy | Macro avg | Weighted avg |
|:-------:|:---------:|:------------:|
| 0.78 | 0.80 / 0.78 / 0.78 | 0.80 / 0.78 / 0.78 |

**Matriz de confusión:**

| | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|:---|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100** | 274 | 17 | 0 | 14 | 18 | 3 |
| **200** | 22 | 284 | 0 | 3 | 1 | 6 |
| **500** | 1 | 42 | 162 | 3 | 36 | 12 |
| **1000** | 58 | 6 | 0 | 307 | 27 | 8 |
| **2000** | 35 | 2 | 0 | 1 | 213 | 66 |
| **10000** | 3 | 5 | 0 | 1 | 23 | 240 |

## Análisis

Run 27 representa el extremo inferior de la variabilidad, junto con Run 23:
- **Val accuracy 73.82%** — uno de los más bajos del grupo
- **Test accuracy 78.62%** — también bajo comparativamente
- **Clase 500 con recall muy bajo (0.51 val, 0.60 test)**: 18 de 92 billetes de 500 clasificados como 200 en test; 11 de 55 confundidos con 200 en val
- **Clase 2000 con precisión baja (0.62 val, 0.65 test)**: 12+6 falsos positivos en test
- **Clase 1000 también afectada**: 15 clasificados como 100 en test
- **Early stopping en época 38** — la mejor época fue temprana, luego el val_loss subió sin recuperarse

Este run confirma la alta varianza del entrenamiento: la misma configuración que produjo 88.33%/92.75% en Run 26 dio 73.82%/78.62% en Run 27, simplemente por la inicialización aleatoria y el data augmentation.

---

## Comparativa final

| Métrica | Run 23 | Run 24 | Run 25 | **Run 26** | Run 27 |
|:--------|:-----:|:-----:|:-----:|:---------:|:-----:|
| **Val accuracy** | 72.24% | 83.91% | 82.65% | **88.33%** | 73.82% |
| **Test accuracy** | 68.12% | 81.52% | 81.34% | **92.75%** | 78.62% |
| **Train accuracy** | 70.15% | 82.61% | 83.52% | **91.07%** | 77.81% |
| **Mejor época** | 38 | 46 | 65 | **87** | 38 |
| **Épocas totales** | 88 | 96 | 115 | **137** | 88 |
| **Tiempo** | 4.8 min | 5.3 min | 6.5 min | **8.1 min** | 5.3 min |
| **Val loss (mejor)** | 1.3193 | 0.8982 | 0.9278 | **0.7418** | 1.1441 |

Run 26 se destaca claramente como el mejor, demostrando que con la configuración actual el modelo puede alcanzar ~93% de accuracy en test cuando la inicialización y el data augmentation favorecen una buena convergencia.
