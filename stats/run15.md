# Run 15 - Arquitectura Grande (64→128→256→512) con dataset completo

**Fecha:** 2026-05-28
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet 4 bloques (64→128→256→512 filters, **5.88M params**)

---

## Cambios respecto a Run 14

| Aspecto | Run 14 (y Run 11) | Run 15 |
|---------|-------------------|--------|
| Conv filters | [32, 64, 128] | **[64, 128, 256, 512]** |
| Parámetros | 651K | **5.88M** (9× más) |
| Dropout | 0.3 | **0.5** |
| Weight decay | 0.01 | **0.0001** |
| Label smoothing | 0.1 | 0.1 |
| Batch size | 16 | **32** |
| Capas lineales | 2048→256→128→6 | **8192→512→D→256→6** |
| Early stopping | val_loss, patience 10 | val_loss, **patience 50** |
| Dataset | 144 train / 42 val | **1597 train / 317 val** |

---

## Dataset

Dataset expandido con **2121 imágenes totales** (1597 train, 317 val, 207 test).

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

- **Época alcanzada:** 87 (early stopping — sin mejora por 50 épocas)
- **Mejor época:** 87
- **Mejor val_loss:** 1.5997
- **Mejor val_acc:** **61.83%**
- **Tiempo:** ~7 min

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.39      | **1.00** | 0.56     | 54      |
| 200     | 0.60      | **0.80** | 0.69     | 51      |
| 500     | **1.00**  | 0.55   | 0.71     | 55      |
| 1000    | 0.95      | **0.29** | 0.44     | 69      |
| 2000    | 0.84      | 0.58   | 0.69     | 53      |
| 10000   | 0.87      | 0.57   | 0.69     | 35      |
| **Accuracy** | | | **0.62** | **317** |
| **Macro Avg** | 0.78 | 0.63 | **0.63** | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **54** | 0   | 0   | 0    | 0    | 0     |
| **200**     | 9   | **41** | 0   | 0    | 1    | 0     |
| **500**     | 3   | 15  | **30** | 0  | 4    | 3     |
| **1000**    | **48** | 0   | 0   | **20** | 1    | 0     |
| **2000**    | 14  | 7   | 0   | 1    | **31** | 0     |
| **10000**   | 10  | 5   | 0   | 0    | 0    | **20** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.45      | **1.00** | 0.62     | 38      |
| 200     | 0.69      | **0.95** | 0.80     | 44      |
| 500     | **1.00**  | 0.52   | 0.68     | 27      |
| 1000    | **1.00**  | **0.29** | 0.44     | 42      |
| 2000    | 0.81      | 0.57   | 0.67     | 30      |
| 10000   | 0.93      | 0.54   | 0.68     | 26      |
| **Accuracy** | | | **0.66** | **207** |
| **Macro Avg** | 0.81 | 0.64 | **0.65** | 207 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **38** | 0   | 0   | 0    | 0    | 0     |
| **200**     | 2   | **42** | 0   | 0    | 0    | 0     |
| **500**     | 0   | 8   | **14** | 0  | 4    | 1     |
| **1000**    | **29** | 1   | 0   | **12** | 0    | 0     |
| **2000**    | 8   | 5   | 0   | 0    | **17** | 0     |
| **10000**   | 7   | 5   | 0   | 0    | 0    | **14** |

---

## Comparación con Run 14

| Métrica | Run 14 (651K, 144 train) | Run 15 (5.88M, 1597 train) |
|---------|--------------------------|---------------------------|
| **Val accuracy** | **61.90%** | 61.83% |
| Macro F1 | 0.60 | **0.63** |
| Best val_loss | **1.2462** | 1.5997 |
| Parámetros | 651K | **5.88M** |
| Tiempo | **~17s** | ~7 min |

Sorprendentemente, **el resultado es casi idéntico al de Run 14** (61.9% vs 61.8%) pese a tener 9× más parámetros y 11× más datos de entrenamiento.

---

## Comparación histórica

| Run | Val Acc | Params | Dataset (train) | Técnica clave |
|-----|---------|--------|-----------------|---------------|
| Run 11 | **73.81%** | 651K | 144 | Label smoothing + WD 0.01 |
| Run 4 | **72.41%** | 1.45M | 144 | AdaptiveAvgPool2d((4,4)) |
| Run 6 | 71.05% | 1.45M | 148 | Dataset balanceado |
| Run 14 | 61.90% | 651K | 144 | Misma config que Run 11 (varianza) |
| **Run 15** | **61.83%** | **5.88M** | **1597** | **Arquitectura grande + dataset completo** |
| Run 10 | 61.90% | 1.45M | 144 | Early stopping por val_loss |

---

## Análisis

### Problema principal: sesgo masivo hacia clase 100

La clase **100 tiene recall perfecto (100%)** pero a costa de precision bajísima (39% en val, 45% en test). **48 de 69 imágenes de 1000** y **14 de 53 de 2000** fueron clasificadas como 100. Esto indica que el modelo aprendió a "decir 100 por defecto".

Causa probable: **desequilibrio de clases en el dataset**. Aunque se usaron class weights (USE_CLASS_WEIGHTS = True), la clase 100 tiene 288 imágenes vs 157 de 10000. La red de 5.88M params es tan expresiva que encontró el atajo de clasificar todo como 100.

### La clase 1000 es la más perjudicada

- **29% recall** en validation, **29%** en test
- Casi la mitad de las imágenes de 1000 se confunden con 100
- Segunda clase más frecuente (364 train), lo que sugiere que el problema no es cantidad de datos sino **similitud visual entre 100 y 1000**

### 500 y 200 también sufren

- 500 se confunde mucho con 200 (15/55 en val, 8/27 en test)
- Esto sugiere que a 160×160, los billetes de 500 y 200 se ven muy similares

### La arquitectura grande no ayudó

Pese a tener 9× más parámetros que Run 11/14, el resultado es **virtualmente idéntico** (61.8% vs 61.9%). Más capacidad de modelo no compensa el desbalance de clases ni la similitud visual entre denominaciones.

---

## Conclusión

**Run 15 confirma que aumentar la capacidad del modelo no es el camino.** La arquitectura de 5.88M params con dataset de 1597 imágenes sufre de:
1. **Sesgo hacia la clase mayoritaria** (100) pese a los class weights
2. **Alta confusión entre pares visualmente similares**: 100↔1000, 200↔500, 2000↔100
3. **Mismo techo de ~62%** que modelos mucho más chicos con menos datos

La mejor configuración hasta ahora sigue siendo la de **Run 11 (73.81%)**, que con solo 144 imágenes y 651K params logró mejor accuracy usando **label smoothing + weight decay 0.01**. El dataset más grande (1597 train) no ayudó porque introdujo **desequilibrio de clases** que el modelo explota como atajo.

**Próximos pasos sugeridos:**
- Re-balancear el dataset para que todas las clases tengan ~300 imágenes
- Volver a la arquitectura chica (32→64→128) que funcionó mejor
- Probar weight decay más agresivo (0.01 como en Run 11)
- Investigar aumentación específica para los pares problemáticos (100↔1000, 200↔500)
