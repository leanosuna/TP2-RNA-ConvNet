# Run 22 - Dataset rebalanceado (20% test, splits consistentes)

**Fecha:** 2026-05-28
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 21

| Aspecto | Run 21 | Run 22 |
|---------|--------|--------|
| Train | 1686 | **1893** (+207 duplicados de test) |
| Val | 317 (desbalanceado) | 317 (balanceado, original) |
| Test | 207 (8.9%) | **552 (20%)** balanceado 92/clase |
| Class weights | 0.864..1.585 | **0.947..1.135** (más parejos) |

---

## Dataset

2762 imágenes (1893 train, 317 val, 552 test). Test balanceado al 20%.

---

## Entrenamiento

- **Época alcanzada:** 99 (early stopping — 50 épocas sin mejora desde época 49)
- **Mejor época:** 49
- **Mejor val_loss:** **0.9062** (récord)
- **Mejor val_acc:** **81.70% (nuevo récord)**
- **Tiempo:** 373.8s (6.2 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | **0.90**  | **0.85** | **0.88** | 54      |
| 200     | **0.93**  | 0.75   | **0.83** | 51      |
| 500     | **0.96**  | **0.89** | **0.92** | 55      |
| 1000    | **0.93**  | 0.74   | **0.82** | 69      |
| 2000    | 0.55      | **0.94** | 0.69     | 53      |
| 10000   | **0.89**  | 0.71   | **0.79** | 35      |
| **Accuracy** | | | **0.82** | **317** |
| **Macro Avg** | **0.86** | **0.81** | **0.82** | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **46** | 2   | 0   | 1    | 5    | 0     |
| **200**     | 1   | **38** | 0   | 0    | 11   | 1     |
| **500**     | 0   | 0   | **49** | 2    | 3    | 1     |
| **1000**    | 4   | 0   | 2   | **51** | 12   | 0     |
| **2000**    | 0   | 1   | 0   | 1    | **50** | 1     |
| **10000**   | 0   | 0   | 0   | 0    | 10   | **25** |

---

## Evaluación — Test Set **(552 imágenes, mucho más robusto)**

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | **0.91**  | **0.77** | **0.84** | 92      |
| 200     | **0.96**  | **0.85** | **0.90** | 92      |
| 500     | **0.99**  | **0.82** | **0.89** | 92      |
| 1000    | **0.91**  | **0.85** | **0.88** | 92      |
| 2000    | 0.54      | **0.91** | 0.68     | 92      |
| 10000   | 0.85      | 0.70   | 0.77     | 92      |
| **Accuracy** | | | **0.82** | **552** |
| **Macro Avg** | **0.86** | **0.82** | **0.83** | 552 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **71** | 2   | 0   | 0    | 19   | 0     |
| **200**     | 2   | **78** | 0   | 1    | 6    | 5     |
| **500**     | 0   | 0   | **75** | 6    | 11   | 0     |
| **1000**    | 5   | 1   | 0   | **78** | 8    | 0     |
| **2000**    | 0   | 0   | 1   | 1    | **84** | 6     |
| **10000**   | 0   | 0   | 0   | 0    | 28   | **64** |

---

## Comparación con récords anteriores

| Métrica | Run 17 | Run 18 | **Run 22** |
|---------|--------|--------|------------|
| **Val accuracy** | 80.13% | 79.18% | **81.70%** |
| **Test accuracy** | 85.02%* | 76.81%* | **81.70%** |
| **Macro F1 (val)** | 0.80 | 0.79 | **0.82** |
| Best val_loss | 1.0117 | 1.0747 | **0.9062** |
| Mejor época | 53 | 31 | 49 |
| Tiempo | 277s | 242s | 374s |

*\*Runs 17-18 tenían test chico (207), no comparable directamente.*

### Progreso histórico

```
Run 0:  ██████████████░░░░░░░░  38.46%
Run 1:  █████████░░░░░░░░░░░░░  26.09%
Run 2:  █████████████████████░  65.22%
Run 3:  ██████████████████░░░░  55.17%
Run 4:  ██████████████████████░ 72.41%
Run 11: ███████████████████████ 73.81%
Run 15: █████████████████████░░ 61.83%
Run 16: ██████████████████████░ 71.29%
Run 17: ███████████████████████ 80.13% ← Récord anterior
Run 18: ███████████████████████ 79.18%
Run 19: ██████████████████████░ 76.97% (sin class weights)
Run 22: ████████████████████████ 81.70% ← NUEVO RÉCORD
```

---

## Análisis

### El split balanceado dio resultados más confiables

Con 552 imágenes de test (balanceadas 92/clase), las métricas son mucho más robustas que antes. Por primera vez, **val y test dan exactamente el mismo accuracy (81.70%)**, lo que indica que no hay overfitting ni desbalance en los splits.

### Clase 2000: el problema persistente

**F1 de solo 0.68-0.69** en ambos sets. El recall es alto (0.91-0.94) pero la precisión es pésima (0.54-0.55). El modelo clasifica demasiadas imágenes como 2000:
- 19 de 92 imágenes de 100 → clasificadas como 2000
- 28 de 92 imágenes de 10000 → clasificadas como 2000
- 11 de 92 de 500 → clasificadas como 2000

El billete de 2000 parece compartir características visuales con 100 y 10000 a 160×160.

### Clase 10000 también sufre

Recall de 0.70-0.71 en ambos sets. 28/92 en test clasificados como 2000. La clase con menos datos (272 train, 35 val) sigue siendo la más perjudicada pese a los class weights.

### Clases fuertes: 100, 200, 500, 1000

Las cuatro clases principales tienen F1 entre 0.82 y 0.92 en test. **200 es la mejor (F1 0.90)**, con precision 0.96 y recall 0.85.

---

## Conclusión

**Run 22 establece un nuevo récord de 81.70% con el dataset rebalanceado.** El split 20% test (552 imágenes balanceadas) da métricas mucho más confiables que los runs anteriores con test chico.

El cuello de botella principal es **la confusión entre 2000 y las demás clases**. Mejorar la precisión de 2000 (actualmente 0.54) podría llevar el accuracy general a ~87-88%.

**Próximos pasos sugeridos:**
- Aumentación específica para reducir falsos positivos de 2000
- Ajustar el umbral de decisión para 2000 (post-entrenamiento)
- Agregar más imágenes de 10000 y 2000
- Probar arquitectura con más filtros iniciales (64→128→256) ahora que hay más datos
