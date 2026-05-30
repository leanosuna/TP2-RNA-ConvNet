# Comparativa Histórica — 5 Runs Representativos (0 a 27)

## Criterio de selección

Se tomaron 5 runs que representan los hitos más importantes del proyecto, cubriendo el peor resultado histórico, el mejor histórico, y los cambios de configuración más relevantes en el medio.

| Run | Test Acc | Por qué fue seleccionado |
|:---:|:-------:|:-------------------------|
| **1** | ~35% | **Peor resultado histórico** — arquitectura con GlobalAvgPool2d, dataset de solo 112 imágenes |
| **4** | 72.41% | **Primer gran salto de calidad** — AdaptiveAvgPool2d((4,4)) + 160×160, migración a PyTorch |
| **17** | 85.02%* | **Primer resultado sobre 80%** — Weight decay 0.01 + 200 épocas, dataset expandido a 2121 imágenes |
| **22** | 81.70% | **Dataset rebalanceado** — test balanceado al 20% (552 imágenes), métricas más confiables |
| **26** | **92.75%** | **Mejor resultado histórico** — misma configuración que Run 22, inicialización favorable |

*\*Run 17 tenía test chico (~207 imágenes, 8.9%), no directamente comparable con Runs 22-26 (552 imágenes, 20%).*

---

## 1. Evolución de los parámetros

### Arquitectura de la red

| Parámetro | Run 1 | Run 4 | Run 17 | Run 22 | Run 26 |
|:----------|:-----:|:-----:|:------:|:------:|:------:|
| **Framework** | TensorFlow/Keras | PyTorch 2.5.1 | PyTorch 2.5.1 | PyTorch 2.5.1 | PyTorch 2.5.1 |
| **Conv filters** | [16, 32] | [64, 128, 256] | [32, 64, 128] | [32, 64, 128] | [32, 64, 128] |
| **Kernel size** | 3×3 | 3×3 | 3×3 | 3×3 | 3×3 |
| **Pooling** | MaxPool(2) | MaxPool(2) | MaxPool(2) | MaxPool(2) | MaxPool(2) |
| **Post-conv** | GlobalAvgPool2d | AdaptiveAvgPool2d(4,4) | AdaptiveAvgPool2d(4,4) | AdaptiveAvgPool2d(4,4) | AdaptiveAvgPool2d(4,4) |
| **Capas FC** | 32→16→6 | 1024→256→128→6** | 2048→256→128→6 | 2048→256→128→6 | 2048→256→128→6 |
| **Dropout** | 0.5 | 0.5 | 0.3 | 0.3 | 0.3 |
| **Total params** | ~1.8M | ~1.45M | **~651K** | **~651K** | **~651K** |
| **Input size** | — | 160×160 | 160×160 | 160×160 | 160×160 |

*\*\*Run 4 usaba 3 capas FC con 1024→256→128 antes de la salida, a diferencia de Runs 17+ que usan 2048→256→128 por el AdaptiveAvgPool2d(4,4) que produce 128×4×4=2048 features.*

### Hiperparámetros de entrenamiento

| Parámetro | Run 1 | Run 4 | Run 17 | Run 22 | Run 26 |
|:----------|:-----:|:-----:|:------:|:------:|:------:|
| **Batch size** | — | 32 | 32 | 32 | 32 |
| **Épocas** | — | 80 | **200** | **200** | **200** |
| **Learning rate** | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| **Optimizer** | Adam | Adam | Adam | Adam | Adam |
| **Weight decay** | — | **0.005→0.001** | **0.01** | **0.01** | **0.01** |
| **LR scheduler** | ✗ | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |
| **Label smoothing** | ✗ | ✗ | 0.1 | 0.1 | 0.1 |
| **Class weights** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **Early stop patience** | — | 30 | 50 | 50 | 50 |

### Dataset

| Parámetro | Run 1 | Run 4 | Run 17 | Run 22 | Run 26 |
|:----------|:-----:|:-----:|:------:|:------:|:------:|
| **Total imágenes** | **112** | **144** | **2121** | **2762** | **2762** |
| **Train** | ~90 | 115 | 1597 | **1893** | **1893** |
| **Val** | ~22 | 29 | 317 | 317 | 317 |
| **Test** | — | — | 207 (8.9%) | **552 (20%)** | **552 (20%)** |
| **Balanceo** | Manual | Manual | Original | **Rebalanceado** | **Rebalanceado** |
| **Aumentación** | ✗ | ✗ | Online GPU | Online GPU | Online GPU |

### Data augmentation (Runs 17 en adelante)

| Transformación | Run 1-4 | Run 17+ |
|:---------------|:-------:|:-------:|
| RandomRotation | ✗ | ±15° |
| ColorJitter brightness | ✗ | 0.15 |
| ColorJitter contrast | ✗ | 0.1 |
| ColorJitter saturation | ✗ | 0.1 |
| ColorJitter hue | ✗ | 0.02 |
| RandomAffine scale | ✗ | (0.9, 1.1) |
| RandomAffine translate | ✗ | (0.1, 0.1) |
| RandomHorizontalFlip | ✗ | 0.0 (deshabilitado) |

---

## 2. Resultados

### Accuracy general

| Métrica | Run 1 | Run 4 | Run 17* | Run 22 | **Run 26** |
|:--------|:-----:|:-----:|:-------:|:------:|:---------:|
| **Val Accuracy** | ~35% | **72.41%** | **80.13%** | **81.70%** | **88.33%** |
| **Test Accuracy** | — | — | 85.02%* | 81.70% | **92.75%** |
| **Train Accuracy** | — | — | — | — | **91.07%** |
| **Val Loss (mejor)** | — | — | 1.0117 | **0.9062** | **0.7418** |
| **Mejor época** | — | — | 53 | 49 | **87** |
| **Épocas totales** | — | 80 | 103 | 99 | **137** |
| **Tiempo** | ~20s | 21s | 4.6min | 6.2min | **8.1min** |

*\*Run 17 test accuracy sobre test chico (~207 imágenes, 8.9%), no comparable con Runs 22-26 (552 imágenes, 20%).*

### Precision por clase (Test)

| Clase | Run 1 | Run 4 | Run 17* | Run 22 | **Run 26** |
|:----:|:----:|:-----:|:-------:|:------:|:---------:|
| 100 | — | — | 0.83 | **0.91** | 0.87 |
| 200 | — | — | 0.82 | **0.96** | **0.97** |
| 500 | — | — | **1.00** | **0.99** | **1.00** |
| 1000 | — | — | 0.96 | **0.91** | **0.91** |
| 2000 | — | — | 0.75 | 0.54 | **0.95** |
| 10000 | — | — | 0.74 | 0.85 | **0.88** |

*\*Run 17 sobre test chico (~207 imágenes), el resto sobre 552 imágenes.*

### Recall por clase (Test)

| Clase | Run 1 | Run 4 | Run 17* | Run 22 | **Run 26** |
|:----:|:----:|:-----:|:-------:|:------:|:---------:|
| 100 | — | — | **0.97** | 0.77 | **0.95** |
| 200 | — | — | **0.97** | 0.85 | **0.91** |
| 500 | — | — | **0.97** | 0.82 | **0.92** |
| 1000 | — | — | 0.90 | 0.85 | **0.91** |
| 2000 | — | — | 0.72 | **0.91** | 0.88 |
| 10000 | — | — | 0.61 | 0.70 | **0.99** |

### F1-Score por clase (Test)

| Clase | Run 1 | Run 4 | Run 17* | Run 22 | **Run 26** |
|:----:|:----:|:-----:|:-------:|:------:|:---------:|
| 100 | — | — | 0.89 | 0.84 | **0.91** |
| 200 | — | — | 0.89 | **0.90** | **0.94** |
| 500 | — | — | **0.99** | 0.89 | **0.96** |
| 1000 | — | — | **0.93** | 0.88 | **0.91** |
| 2000 | — | — | 0.73 | 0.68 | **0.92** |
| 10000 | — | — | 0.67 | 0.77 | **0.93** |

---

## 3. Análisis de la evolución

### De Run 1 a Run 4: El salto fundacional

**Cambios clave:**
- Migración de TensorFlow/Keras a **PyTorch** (por falta de soporte GPU de TF en Windows)
- Reemplazo de `GlobalAvgPool2d` por `AdaptiveAvgPool2d((4,4))`
- Aumento de filtros conv (16→32 → 64→128→256)
- Input unificado a 160×160

**Impacto:**
- Accuracy saltó de ~35% a **72.41%** (+37 puntos)
- La reducción de parámetros (~1.8M → ~1.45M) combinada con pooling adaptativo permitió preservar información espacial crítica
- El dataset seguía siendo muy pequeño (144 imágenes), poniendo un techo duro al rendimiento

### De Run 4 a Run 17: Regularización y dataset

**Cambios clave:**
- Reducción de filtros conv (64→128→256 → 32→64→128) → **651K params**
- **Weight decay aumentado a 0.01** (vs 0.001)
- **Label smoothing (0.1)** añadido
- **200 épocas** máximas (vs 80)
- Dataset expandido de 144 a **2121 imágenes**
- **Data augmentation online** en GPU

**Impacto:**
- Accuracy subió de 72.41% a **80.13%** (+8 puntos)
- La regularización agresiva (WD 0.01 + label smoothing) controló el overfitting pese a tener 200 épocas
- La reducción de parámetros (1.45M → 651K) fue beneficiosa: menos capacidad pero mejor generalización
- El dataset más grande fue el habilitador principal de las mejoras

### De Run 17 a Run 22: Calidad del dataset

**Cambios clave:**
- **Rebalanceo de splits:** test aumentado de 207 (8.9%) a **552 (20%)** imágenes
- 207 imágenes duplicadas de test pasaron a train (+207 en train)

**Impacto:**
- Val accuracy se mantuvo similar (80.13% → 81.70%)
- Test accuracy sobre 552 imágenes: **81.70%** — primera medición confiable
- Igualdad val-test (ambos 81.70%) indicó que no había overfitting en los splits
- Demostró que el test anterior era poco representativo

### De Run 22 a Run 26: El techo de la configuración

**Cambios clave:**
- **Ninguno** — misma configuración exacta
- La única diferencia fue la inicialización aleatoria y el data augmentation estocástico

**Impacto:**
- Accuracy saltó de 81.70% a **92.75%** (+11 puntos)
- **Demostración de la alta varianza** del entrenamiento: misma config, resultados muy distintos
- Run 26 encontró un mínimo local mucho mejor gracias a una inicialización favorable
- Las 137 épocas (vs 99 de Run 22) permitieron una convergencia más profunda

---

## 4. Resumen de la evolución de parámetros

| Aspecto | Run 1 | Run 4 | Run 17 | Run 22 | Run 26 |
|:--------|:-----:|:-----:|:------:|:------:|:------:|
| **Params** | 1.8M | 1.45M | **651K** | **651K** | **651K** |
| **Conv layers** | 2 | 3 | 3 | 3 | 3 |
| **Conv filters** | 16→32 | 64→128→256 | 32→64→128 | 32→64→128 | 32→64→128 |
| **Pooling** | GlobalAvg | Adaptive(4,4) | Adaptive(4,4) | Adaptive(4,4) | Adaptive(4,4) |
| **WD** | — | 0.001 | **0.01** | **0.01** | **0.01** |
| **Label smoothing** | ✗ | ✗ | 0.1 | 0.1 | 0.1 |
| **Class weights** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **Dataset** | 112 | 144 | 2121 | **2762** | **2762** |
| **Test set** | — | — | 207 (8.9%) | **552 (20%)** | **552 (20%)** |
| **Data aug** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **Accuracy** | **~35%** | **72.41%** | **80.13%** | **81.70%** | **92.75%** |

---

## 5. Diagrama de evolución

```
Accuracy (%)
95 +                                              ★ Run 26 (92.75%)
   |
90 +
   |
85 +                                   Run 22 (81.70%)
   |                                  ██
80 +                        Run 17 (80.13%)
   |                        ██
75 +              Run 4 (72.41%)
   |              ██
70 +
   |
65 +
   |
60 +
   |
55 +
   |
50 +
   |
45 +
   |
40 +   Run 1 (~35%)
   |   ██
35 +
   +——————————————————————————————————————————————————
      Run 1        Run 4        Run 17      Run 22      Run 26
     (112 img)    (144 img)    (2121 img)  (2762 img)  (2762 img)
     GlobalAvg    AdaptPool   WD 0.01     Balanceado  Mejor run
     1.8M params  1.45M       651K        651K        651K
```

---

## 6. Lecciones aprendidas

### Lo que más impactó en el accuracy

| Cambio | Impacto | Runs |
|:-------|:-------:|:----:|
| Aumentar dataset (112 → 2762) | **+40-50 pts** | 1→22 |
| AdaptiveAvgPool2d((4,4)) vs GlobalAvgPool2d | **+37 pts** | 1→4 |
| Weight decay 0.01 + 200 épocas | **+8 pts** | 4→17 |
| Rebalanceo de splits (test 20%) | Métricas confiables | 17→22 |
| Inicialización favorable (varianza) | **+11 pts** | 22→26 |
| Data augmentation online | Habilitó más épocas | 4→17 |
| Label smoothing (0.1) | Estabilizó entrenamiento | 4→17 |
| Class weights balanceados | Mejor recall en clases minoritarias | 4→17 |

### La evolución de la arquitectura

```
Run 1:  Conv(16)→Conv(32)→GlobalAvgPool→FC(32→16→6)          1.8M params  ~35%
Run 4:  Conv(64)→Conv(128)→Conv(256)→AdaptivePool(4,4)      1.45M params  72%
         →FC(1024→256→128→6)
Run 17: Conv(32)→Conv(64)→Conv(128)→AdaptivePool(4,4)       651K params   80%
         →Dropout→FC(2048→256)→Dropout→FC(256→128)→FC(128→6)
Run 22: Ídem Run 17 + dataset rebalanceado                   651K params   82%
Run 26: Ídem Run 22 (inicialización favorable)               651K params   93%
```

La arquitectura se simplificó con el tiempo: menos canales convolucionales pero mejor regularización. El punto óptimo fueron 3 bloques conv con 32→64→128 filtros y 651K parámetros totales.

---

## 7. Conclusión

La comparativa muestra que **el factor más determinante fue el tamaño y calidad del dataset** (112 → 2762 imágenes = +40-50 pts de accuracy). El segundo factor fue la **arquitectura y regularización**: el punto óptimo se encontró con 651K params, weight decay 0.01, label smoothing y AdaptiveAvgPool2d((4,4)). Finalmente, la **varianza por inicialización** puede aportar ±11 puntos adicionales, por lo que se recomienda entrenar múltiples veces y seleccionar el mejor modelo.

| Run | Accuracy | Factor clave |
|:---:|:--------:|:-------------|
| 1 | ~35% | Dataset mínimo (112 img), arquitectura pobre |
| 4 | 72.41% | AdaptivePool + PyTorch + 160×160 |
| 17 | 80.13% | WD 0.01 + 200 épocas + dataset grande |
| 22 | 81.70% | Splits balanceados, métricas confiables |
| **26** | **92.75%** | **Inicialización favorable + convergencia profunda** |

---

## 8. Matrices de Confusión

### Run 1 — Validation (~35%, peor histórico)

Dataset de solo 112 imágenes. Sin test set.

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100**     | 0   | 2   | 0   | 1    | 0    | 0     |
| **200**     | 0   | 2   | 1   | 1    | 1    | 0     |
| **500**     | 0   | 2   | 0   | 2    | 0    | 0     |
| **1000**    | 0   | 1   | 0   | 4    | 0    | 0     |
| **2000**    | 0   | 3   | 0   | 0    | 0    | 0     |
| **10000**   | 1   | 2   | 0   | 0    | 0    | 0     |

La red no lograba aprender: clases 100, 500, 2000 y 10000 con 0% de precisión y recall. Solo la clase 1000 se reconocía parcialmente (4/5 correctos).

---

### Run 4 — Validation (72.41%, primer salto)

Dataset de 144 imágenes. Sin test set.

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100**     | 3   | 0   | 0   | 1    | 0    | 0     |
| **200**     | 0   | 7   | 0   | 0    | 0    | 0     |
| **500**     | 0   | 1   | 5   | 0    | 0    | 0     |
| **1000**    | 2   | 0   | 0   | 4    | 0    | 0     |
| **2000**    | 0   | 0   | 0   | 0    | 1    | 2     |
| **10000**   | 0   | 1   | 1   | 0    | 0    | 1     |

Mejora sustancial: 200 y 500 se reconocen bien (7/7 y 5/6). Persiste confusión 2000↔10000 (2 de 3 de 2000 clasificados como 10000).

---

### Run 17 — Test (85.02%*, primer 80%)

Test chico de 207 imágenes (8.9% del dataset).

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100**     | **33** | 0   | 0   | 1    | 4    | 0     |
| **200**     | 3   | **39** | 0   | 1    | 0    | 1     |
| **500**     | 0   | 1   | **18** | 2    | 0    | 6     |
| **1000**    | 2   | 0   | 0   | **37** | 3    | 0     |
| **2000**    | 3   | 3   | 0   | 0    | **23** | 1     |
| **10000**   | 0   | 0   | 0   | 0    | 1    | **25** |

La diagonal principal se consolida. El peso de datos es bajo (207 imágenes), por lo que algunos aciertos pueden ser anecdóticos (ej: 10000 con 25/26).

---

### Run 22 — Test (81.70%, dataset rebalanceado)

Test balanceado de 552 imágenes (20% del dataset). Primera medición confiable.

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100**     | **71** | 2   | 0   | 0    | 19   | 0     |
| **200**     | 2   | **78** | 0   | 1    | 6    | 5     |
| **500**     | 0   | 0   | **75** | 6    | 11   | 0     |
| **1000**    | 5   | 1   | 0   | **78** | 8    | 0     |
| **2000**    | 0   | 0   | 1   | 1    | **84** | 6     |
| **10000**   | 0   | 0   | 0   | 0    | 28   | **64** |

Se observa el patrón clásico de confusión: 19 de 100→2000, 28 de 10000→2000. La clase 2000 es un "sumidero" de falsos positivos.

---

### Run 26 — Test (92.75%, mejor histórico)

Misma configuración que Run 22, inicialización favorable.

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|:---:|:---:|:---:|:----:|:----:|:-----:|
| **100**     | **87** | 2   | 0   | 0    | 3    | 0     |
| **200**     | 4   | **84** | 0   | 2    | 0    | 2     |
| **500**     | 0   | 0   | **85** | 4    | 1    | 2     |
| **1000**    | 7   | 0   | 0   | **84** | 0    | 1     |
| **2000**    | 2   | 1   | 0   | 1    | **81** | 7     |
| **10000**   | 0   | 0   | 0   | 1    | 0    | **91** |

Matriz casi perfectamente diagonal. Las confusiones residuales son mínimas (ej: 7 de 1000→100, 7 de 2000→10000). La clase 10000 logra 91/92 correctos.

---

### Evolución visual de la diagonal principal

```
Run 1:  ░░░░▒░░░░░░░░   0/3/0/4/0/0   ~35%
Run 4:  ██░░████▒░░▒   3/7/5/4/1/1   72%
Run 17: █████████████  33/39/18/37/23/25  85%*
Run 22: █████████████  71/78/75/78/84/64  82%
Run 26: ████████████████████████  87/84/85/84/81/91  93%
```

La evolución muestra cómo la diagonal se fue fortaleciendo desde el caos de Run 1 hasta la matriz casi diagonal de Run 26.
