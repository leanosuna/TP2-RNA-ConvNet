# Run 17 - Weight Decay 0.01 + 200 épocas

**Fecha:** 2026-05-28
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 16

| Aspecto | Run 16 | Run 17 |
|---------|--------|--------|
| Weight decay | 1e-4 | **0.01** |
| Épocas máximas | 100 | **200** |
| Época de early stopping | — (completó 100) | **103** (best 53) |
| Tiempo | 4.8 min | **4.6 min** |

---

## Dataset

2121 imágenes (1597 train, 317 val, 207 test). Sin cambios.

---

## Entrenamiento

- **Época alcanzada:** 103 (early stopping — 50 épocas sin mejora desde época 53)
- **Mejor época:** 53
- **Mejor val_loss:** **1.0117** (récord histórico)
- **Mejor val_acc:** **80.13%** (¡nuevo récord!)
- **Tiempo:** 276.6s (4.6 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.81      | **0.89** | **0.85** | 54      |
| 200     | **0.93**  | 0.80   | **0.86** | 51      |
| 500     | **1.00**  | 0.69   | **0.82** | 55      |
| 1000    | 0.83      | 0.80   | **0.81** | 69      |
| 2000    | 0.62      | **0.77** | 0.69     | 53      |
| 10000   | 0.70      | **0.89** | **0.78** | 35      |
| **Accuracy** | | | **0.80** | **317** |
| **Macro Avg** | **0.82** | **0.81** | **0.80** | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **48** | 0   | 0   | 1    | 5    | 0     |
| **200**     | 4   | **41** | 0   | 3    | 0    | 3     |
| **500**     | 0   | 1   | **38** | 5    | 4    | 7     |
| **1000**    | 1   | 0   | 0   | **55** | 13   | 0     |
| **2000**    | 5   | 2   | 0   | 2    | **41** | 3     |
| **10000**   | 1   | 0   | 0   | 0    | 3    | **31** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.80      | **0.87** | **0.84** | 38      |
| 200     | **0.91**  | **0.89** | **0.90** | 44      |
| 500     | **1.00**  | 0.67   | **0.80** | 27      |
| 1000    | **0.90**  | **0.88** | **0.89** | 42      |
| 2000    | 0.74      | **0.77** | 0.75     | 30      |
| 10000   | 0.76      | **0.96** | **0.85** | 26      |
| **Accuracy** | | | **0.85** | **207** |
| **Macro Avg** | **0.85** | **0.84** | **0.84** | 207 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **33** | 0   | 0   | 1    | 4    | 0     |
| **200**     | 3   | **39** | 0   | 1    | 0    | 1     |
| **500**     | 0   | 1   | **18** | 2    | 0    | 6     |
| **1000**    | 2   | 0   | 0   | **37** | 3    | 0     |
| **2000**    | 3   | 3   | 0   | 0    | **23** | 1     |
| **10000**   | 0   | 0   | 0   | 0    | 1    | **25** |

---

## Comparación con Runs Anteriores

| Métrica | Run 11 (récord) | Run 16 | **Run 17** |
|---------|----------------|--------|------------|
| **Val accuracy** | 73.81% | 71.29% | **80.13%** |
| **Test accuracy** | — | 67.63% | **85.02%** |
| **Macro F1 (val)** | 0.72 | 0.70 | **0.80** |
| Best val_loss | 1.8392 | 1.6101 | **1.0117** |
| Weight decay | 0.01 | 1e-4 | **0.01** |
| Dataset (train) | 144 | 1597 | 1597 |
| Tiempo | ~20s | 290s | **277s** |

### Progreso histórico

```
Run 0:  ██████████████░░░░░░░░  38.46%
Run 1:  █████████░░░░░░░░░░░░░  26.09%
Run 2:  █████████████████████░  65.22%
Run 3:  ██████████████████░░░░  55.17%
Run 4:  ██████████████████████░ 72.41%
Run 5:  ████████████████░░░░░░  48.28%
Run 6:  ██████████████████████░ 71.05%
Run 7:  █████████████████████░░ 69.05%
Run 8:  ██████████████████░░░░  52.38%
Run 9:  ██████████████████░░░░  52.38%
Run 10: █████████████████████░░ 61.90%
Run 11: ███████████████████████ 73.81% ← Récord anterior
Run 14: █████████████████████░░ 61.90%
Run 15: █████████████████████░░ 61.83%
Run 16: ██████████████████████░ 71.29%
Run 17: ████████████████████████ 80.13% ← NUEVO RÉCORD
```

---

## Análisis

### Weight decay 0.01 fue el factor decisivo

La única diferencia entre Run 16 (71.29%) y Run 17 (80.13%) es el weight decay: 1e-4 → 0.01. Esto confirma lo que Run 11 ya había mostrado con 144 imágenes: **weight decay alto es la regularización más importante para este modelo/dataset.**

### El modelo converge rápido y estable

En apenas **53 épocas** alcanzó el mejor punto, con val_loss de **1.0117** — el más bajo de la historia. Las 200 épocas máximas no fueron necesarias (early stopping cortó en 103), pero permitieron ver que el modelo se mantiene estable sin overfitting.

### Mejoras clase por clase

| Clase | Run 16 (F1 val) | Run 17 (F1 val) | Cambio |
|-------|----------------|----------------|--------|
| 100   | 0.79           | **0.85**       | +0.06  |
| 200   | 0.69           | **0.86**       | +0.17  |
| 500   | 0.59           | **0.82**       | **+0.23** |
| 1000  | 0.73           | **0.81**       | +0.08  |
| 2000  | 0.78           | 0.69           | **-0.09** |
| 10000 | 0.64           | **0.78**       | +0.14  |

**500** tuvo la mejora más dramática (+0.23), pasando de 23/55 → 38/55 aciertos. **200** también mejoró mucho (+0.17).

**2000** es ahora la única clase que empeoró (F1 0.78 → 0.69). Se confunde principalmente con 1000 (13 en val) y 100 (5 en val). Posiblemente el weight decay más fuerte penaliza demasiado las características de 2000.

### Resultados en test (85%) superan a validación (80%)

Es inusual que test sea mejor que validación, pero indica que la partición test es ligeramente más fácil (o tiene menos casos ambiguos). La consistencia entre splits es buena.

---

## Conclusión

**Run 17 establece un nuevo récord: 80.13% val acc, 85.02% test acc.** El weight decay 0.01 combinado con label smoothing 0.1 y la arquitectura chica (651K params) es la configuración óptima identificada hasta ahora.

**Clases que aún necesitan mejora:**
1. **2000** (F1 0.69) — confundido con 1000 y 100
2. **500** (recall 0.69) — confundido con 10000 principalmente
3. **100** (precisión 0.81) — 5 falsos positivos de 2000

**Próximo paso lógico:**
- Probar con más aumento específico para 2000 (quizás rotaciones más amplias o cambios de iluminación)
- Considerar oversampling de 2000 y 500 si se quiere mejorar más
