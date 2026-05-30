# Run 16 - Arquitectura Chica (32→64→128) + Dataset Completo

**Fecha:** 2026-05-28
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128 filters, **651K params**)

---

## Cambios respecto a Run 15

| Aspecto | Run 15 | Run 16 |
|---------|--------|--------|
| Conv filters | [64, 128, 256, 512] | **[32, 64, 128]** |
| Parámetros | 5.88M | **651K** (9× menos) |
| Dropout | 0.5 | **0.3** |
| Capas lineales | 8192→512→D→256→6 | **2048→256→D→128→6** |
| Flip horizontal | No (p=0) | No (p=0) |
| SAVE_KERAS | N/A | **False** |
| Run dir | — | **models/run16/** |

---

## Dataset

2121 imágenes totales (1597 train, 317 val, 207 test).

| Clase  | Train | Val | Test |
|--------|------:|----:|-----:|
| 100    | 288   | 54  | 38   |
| 200    | 272   | 51  | 44   |
| 500    | 229   | 55  | 27   |
| 1000   | 364   | 69  | 42   |
| 2000   | 287   | 53  | 30   |
| 10000  | 157   | 35  | 26   |
| **Total** | **1597** | **317** | **207** |

---

## Entrenamiento

- **Épocas completadas:** 100 (no hubo early stopping — patience 50 desde época 20, pero siguió mejorando lentamente)
- **Mejor época:** 67
- **Mejor val_loss:** 1.6101
- **Mejor val_acc:** **71.29%**
- **Tiempo:** 289.6s (4.8 min)

### Curva de entrenamiento

El modelo converge rápido: train_acc llega a ~98% en época 26 y se mantiene. El val_acc oscila entre 66-71% desde época 26 hasta el final, sin overfitting severo (val_loss no diverge). LR decay de 1e-3 → 3.9e-6 por ReduceLROnPlateau.

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.67      | **0.96** | **0.79** | 54      |
| 200     | 0.54      | **0.94** | 0.69     | 51      |
| 500     | **1.00**  | 0.42   | 0.59     | 55      |
| 1000    | **0.98**  | 0.58   | 0.73     | 69      |
| 2000    | 0.76      | **0.79** | **0.78** | 53      |
| 10000   | 0.68      | 0.60   | 0.64     | 35      |
| **Accuracy** | | | **0.71** | **317** |
| **Macro Avg** | 0.77 | 0.72 | **0.70** | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **52** | 2   | 0   | 0    | 0    | 0     |
| **200**     | 3   | **48** | 0   | 0    | 0    | 0     |
| **500**     | 0   | 17  | **23** | 1  | 4    | 10    |
| **1000**    | 22  | 4   | 0   | **40** | 3    | 0     |
| **2000**    | 1   | 10  | 0   | 0    | **42** | 0     |
| **10000**   | 0   | 8   | 0   | 0    | 6    | **21** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.65      | 0.84   | 0.74     | 38      |
| 200     | 0.51      | **0.93** | 0.66     | 44      |
| 500     | **1.00**  | 0.41   | 0.58     | 27      |
| 1000    | **0.96**  | 0.55   | 0.70     | 42      |
| 2000    | **0.82**  | 0.60   | 0.69     | 30      |
| 10000   | 0.76      | 0.62   | 0.68     | 26      |
| **Accuracy** | | | **0.68** | **207** |
| **Macro Avg** | 0.78 | 0.66 | **0.67** | 207 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **32** | 6   | 0   | 0    | 0    | 0     |
| **200**     | 3   | **41** | 0   | 0    | 0    | 0     |
| **500**     | 0   | 11  | **11** | 1  | 0    | 4     |
| **1000**    | 13  | 5   | 0   | **23** | 1    | 0     |
| **2000**    | 1   | 10  | 0   | 0    | **18** | 1     |
| **10000**   | 0   | 7   | 0   | 0    | 3    | **16** |

---

## Comparación con Run 15

| Métrica | Run 15 (5.88M, 61.8%) | Run 16 (651K, **71.3%**) |
|---------|----------------------|------------------------|
| **Val accuracy** | 61.83% | **71.29%** (+9.5 pp) |
| **Macro F1** | 0.63 | **0.70** (+0.07) |
| Best val_loss | 1.5997 | 1.6101 |
| Parámetros | 5.88M | **651K** (9× menos) |
| Tiempo | ~7 min | **4.8 min** |
| Sesgo hacia 100 | **Severo** (48/69 1000→100) | **Moderado** (22/69 1000→100) |

**Mejora significativa** — solo cambiando la arquitectura a una más chica se ganaron 9.5 puntos porcentuales.

---

## Comparación histórica

| Run | Val Acc | Params | Dataset | Clave del éxito |
|-----|---------|--------|---------|-----------------|
| Run 11 | **73.81%** | 651K | 144 train | Label smoothing + WD 0.01 |
| Run 4 | **72.41%** | 1.45M | 144 train | AdaptiveAvgPool2d |
| **Run 16** | **71.29%** | **651K** | **1597 train** | **Arquitectura chica + dataset completo** |
| Run 6 | 71.05% | 1.45M | 148 train | Dataset balanceado |
| Run 14 | 61.90% | 651K | 144 train | Misma config que Run 11 (varianza) |
| Run 15 | 61.83% | 5.88M | 1597 train | Arquitectura grande (sesgo) |

Run 16 es el **tercer mejor resultado histórico**, a solo 2.5 pp del récord (Run 11, 73.81%).

---

## Análisis

### El sesgo hacia clase 100 se redujo drásticamente

En Run 15, **48 de 69** imágenes de 1000 se clasificaban como 100. En Run 16, solo **22 de 69**. La arquitectura más chica con más regularización (dropout 0.3) evita que el modelo aprenda el atajo de "decir 100 siempre".

### Clase 500: el nuevo talón de Aquiles

**Solo 23/55 (42%)** en validación. Se confunde masivamente con 200 (17) y 10000 (10). El recall de 500 cayó de 55% (Run 15) a 42% (Run 16). A 160×160, los billetes de 500 probablemente se parecen mucho a 200 y 10000.

### 2000 mejoró notablemente

Pasó de 58% recall (Run 15) a **79%** en validación. La confusión con 200 se mantiene (10/53), pero manejable.

### 10000 se mantiene estable

60% recall en val, 62% en test. Sigue confundiéndose con 200 (8 en val) y 2000 (6 en val). Clase con menos imágenes (157 train), los class weights ayudan pero no alcanzan.

### El modelo no overfitea

Train_acc llega a 99% pero val_loss se mantiene estable (~1.6-2.0) durante 80 épocas. El ReduceLROnPlateau lleva el LR de 1e-3 a 3.9e-6, lo que estabiliza el entrenamiento. No habría early stopping porque la paciencia de 50 desde época 20 significaría esperar 70 épocas sin mejora — eso no ocurrió porque cada ~10 épocas hay un val_acc >70%.

---

## Conclusión

**Run 16 confirma que la arquitectura chica (651K params, 32→64→128) es la mejor opción para este dataset.** Con el dataset completo de 1597 imágenes, alcanza 71.29% val_acc — casi empatando el récord histórico (73.81%) que se logró con solo 144 imágenes.

La diferencia clave respecto a Run 15: **menos parámetros + más regularización = mejor generalización**. El modelo grande de 5.88M aprendía atajos (sesgo hacia clase 100), mientras que el chico se ve forzado a aprender características más generales.

**Problemas restantes:**
1. **500** (42% recall) — confundido con 200 y 10000. Quizás aumentar rotación o agregar variedad de ángulos para 500 ayude.
2. **1000** (58% recall) — confundido con 100. Son visualmente similares (ambos terrosos).
3. **10000** (60% recall) — necesita más imágenes (solo 157 train).

**Próximos pasos sugeridos:**
- Probar weight decay 0.01 (como Run 11) con esta arquitectura y dataset completo
- Aumentar épocas máximas (no llegó a early stopping)
- Data augmentation específica para los pares problemáticos
