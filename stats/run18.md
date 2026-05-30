# Run 18 - 10000 aumentado (157 → 246 train)

**Fecha:** 2026-05-28
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 17

| Aspecto | Run 17 | Run 18 |
|---------|--------|--------|
| Imágenes 10000 (train) | 157 | **246** (+89) |
| Total train | 1597 | **1686** |
| Class weight 10000 | 1.585 | **1.118** |

---

## Dataset

2210 imágenes totales (1686 train, 317 val, 207 test).

| Clase  | Train | Val | Test |
|--------|------:|----:|-----:|
| 100    | 288   | 54  | 38   |
| 200    | 272   | 51  | 44   |
| 500    | 229   | 55  | 27   |
| 1000   | 364   | 69  | 42   |
| 2000   | 287   | 53  | 30   |
| 10000  | **246** | 35  | 26   |
| **Total** | **1686** | **317** | **207** |

---

## Entrenamiento

- **Época alcanzada:** 81 (early stopping — 50 épocas sin mejora desde época 31)
- **Mejor época:** 31
- **Mejor val_loss:** 1.0747
- **Mejor val_acc:** **79.18%**
- **Tiempo:** 242.1s (4.0 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.69      | **0.91** | **0.78** | 54      |
| 200     | **1.00**  | 0.49   | 0.66     | 51      |
| 500     | 0.88      | **0.95** | **0.91** | 55      |
| 1000    | 0.87      | 0.67   | 0.75     | 69      |
| 2000    | 0.64      | **1.00** | **0.78** | 53      |
| 10000   | **1.00**  | 0.74   | **0.85** | 35      |
| **Accuracy** | | | **0.79** | **317** |
| **Macro Avg** | **0.85** | 0.79 | **0.79** | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **49** | 0   | 0   | 0    | 5    | 0     |
| **200**     | 4   | **25** | 4   | 5    | 13   | 0     |
| **500**     | 0   | 0   | **52** | 2    | 1    | 0     |
| **1000**    | 18  | 0   | 0   | **46** | 5    | 0     |
| **2000**    | 0   | 0   | 0   | 0    | **53** | 0     |
| **10000**   | 0   | 0   | 3   | 0    | 6    | **26** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.67      | **0.82** | 0.74     | 38      |
| 200     | **1.00**  | 0.55   | 0.71     | 44      |
| 500     | 0.81      | **0.93** | **0.86** | 27      |
| 1000    | 0.81      | 0.69   | 0.74     | 42      |
| 2000    | 0.60      | **0.97** | 0.74     | 30      |
| 10000   | **0.95**  | 0.81   | **0.88** | 26      |
| **Accuracy** | | | **0.77** | **207** |
| **Macro Avg** | 0.81 | 0.79 | **0.78** | 207 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **31** | 0   | 0   | 0    | 7    | 0     |
| **200**     | 2   | **24** | 3   | 5    | 9    | 1     |
| **500**     | 0   | 0   | **25** | 2    | 0    | 0     |
| **1000**    | 12  | 0   | 0   | **29** | 1    | 0     |
| **2000**    | 1   | 0   | 0   | 0    | **29** | 0     |
| **10000**   | 0   | 0   | 3   | 0    | 2    | **21** |

---

## Comparación con Run 17

| Métrica | Run 17 | Run 18 | Diferencia |
|---------|--------|--------|------------|
| **Val accuracy** | **80.13%** | 79.18% | -0.95 pp |
| **Test accuracy** | **85.02%** | 76.81% | -8.21 pp |
| Macro F1 (val) | **0.80** | 0.79 | -0.01 |
| Best val_loss | **1.0117** | 1.0747 | +0.06 |
| Best epoch | 53 | **31** | más rápido |

### Cambios por clase (F1 val)

| Clase | Run 17 | Run 18 | Cambio |
|-------|--------|--------|--------|
| 100   | 0.85   | 0.78   | -0.07  |
| 200   | **0.86** | 0.66   | **-0.20** |
| 500   | 0.82   | **0.91** | **+0.09** |
| 1000  | 0.81   | 0.75   | -0.06  |
| 2000  | 0.69   | **0.78** | **+0.09** |
| 10000 | 0.78   | **0.85** | **+0.07** |

---

## Análisis

### Efecto de las 89 imágenes extra de 10000

**Clase 10000 mejoró:** F1 subió de 0.78 → 0.85. Precision perfecta (1.00) aunque recall bajó de 0.89 → 0.74. El class weight se redujo de 1.585 → 1.118, lo que hace que el modelo sea más conservador con 10000.

### 2000 tuvo recall perfecto (1.00)

Todas las 53 imágenes de 2000 en validation fueron clasificadas correctamente. Sin embargo, precision es baja (0.64) porque hay muchos falsos positivos de 200 (13) y 100 (5). El modelo está "adivinando 2000" con frecuencia.

### 200 colapsó

Recall de 200 cayó de 0.80 → 0.49. Más de la mitad de los 200 se clasificaron como 2000 (13) o 100/500/1000. Posiblemente el cambio en los class weights (al redistribuirse con más 10000) afectó desproporcionadamente a 200.

### Alta varianza entre corridas

Run 17 y Run 18 usan la misma configuración (salvo dataset), y difieren ~1 pp en val pero **~8 pp en test**. Esto sugiere que el split de test tiene alta varianza y/o que el modelo converge a mínimos locales distintos según la inicialización aleatoria.

---

## Conclusión

**Run 18 valida que estamos cerca del techo del modelo actual (~80% val acc).** Las 89 imágenes extra de 10000 mejoraron su F1 (+0.07) pero causaron una redistribución en los class weights que perjudicó a la clase 200.

El resultado es comparable al récord de Run 17 (79.18% vs 80.13%), lo que sugiere que la configuración es estable y el principal factor de variación es la inicialización aleatoria y los class weights dinámicos.

**Próximos pasos sugeridos:**
- Balancear las 89 imágenes de 10000 también en valid y test para que el class weight refleje mejor la distribución real
- Probar sin class weights ahora que el dataset está más balanceado
- Investigar aumentación específica para el par problemático 200↔2000
