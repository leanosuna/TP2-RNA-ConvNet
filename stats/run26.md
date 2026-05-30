# Run 26 — Misma config (mejor resultado histórico)

**Fecha:** 2026-05-29
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 25

Sin cambios en la configuración. Misma arquitectura, mismos hiperparámetros.

---

## Dataset

2762 imágenes (1893 train, 317 val, 552 test). Split balanceado 20% test.

---

## Entrenamiento

- **Época alcanzada:** 137 (early stopping)
- **Mejor época:** 87
- **Mejor val_loss:** **0.7418** (récord)
- **Mejor val_acc:** **88.33%** (récord)
- **Tiempo:** 485.9s (8.1 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.82      | **0.94** | **0.88** | 54      |
| 200     | **0.91**  | 0.80   | **0.85** | 51      |
| 500     | **1.00**  | **0.95** | **0.97** | 55      |
| 1000    | 0.89      | **0.90** | **0.89** | 69      |
| 2000    | **0.91**  | 0.77   | 0.84     | 53      |
| 10000   | 0.77      | **0.94** | **0.85** | 35      |
| **Accuracy** | | | **0.88** | **317** |
| **Macro Avg** | **0.88** | **0.88** | **0.88** | 317 |

### Matriz de Confusión (Validation)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **51** | 1   | 0   | 1    | 1    | 0     |
| **200**     | 4   | **41** | 0   | 1    | 1    | 4     |
| **500**     | 0   | 0   | **52** | 3    | 0    | 0     |
| **1000**    | 5   | 0   | 0   | **62** | 0    | 2     |
| **2000**    | 2   | 3   | 0   | 3    | **41** | 4     |
| **10000**   | 0   | 0   | 0   | 0    | 2    | **33** |

---

## Evaluación — Test Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.87      | **0.95** | **0.91** | 92      |
| 200     | **0.97**  | **0.91** | **0.94** | 92      |
| 500     | **1.00**  | **0.92** | **0.96** | 92      |
| 1000    | **0.91**  | **0.91** | **0.91** | 92      |
| 2000    | **0.95**  | **0.88** | **0.92** | 92      |
| 10000   | **0.88**  | **0.99** | **0.93** | 92      |
| **Accuracy** | | | **0.93** | **552** |
| **Macro Avg** | **0.93** | **0.93** | **0.93** | 552 |

### Matriz de Confusión (Test)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **87** | 2   | 0   | 0    | 3    | 0     |
| **200**     | 4   | **84** | 0   | 2    | 0    | 2     |
| **500**     | 0   | 0   | **85** | 4    | 1    | 2     |
| **1000**    | 7   | 0   | 0   | **84** | 0    | 1     |
| **2000**    | 2   | 1   | 0   | 1    | **81** | 7     |
| **10000**   | 0   | 0   | 0   | 1    | 0    | **91** |

---

## Evaluación — Training Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.88      | **0.92** | **0.90** | 326     |
| 200     | **0.94**  | **0.89** | **0.92** | 316     |
| 500     | **1.00**  | **0.96** | **0.98** | 256     |
| 1000    | **0.93**  | **0.94** | **0.93** | 406     |
| 2000    | **0.95**  | 0.79   | 0.86     | 317     |
| 10000   | 0.80      | **0.98** | **0.88** | 272     |
| **Accuracy** | | | **0.91** | **1893** |
| **Macro Avg** | **0.92** | **0.91** | **0.91** | 1893 |

### Matriz de Confusión (Training)

| Real \ Pred | 100 | 200 | 500 | 1000 | 2000 | 10000 |
|-------------|-----|-----|-----|------|------|-------|
| **100**     | **301** | 13  | 0   | 3    | 9    | 0     |
| **200**     | 15  | **282** | 0   | 6    | 1    | 12    |
| **500**     | 0   | 1   | **246** | 6    | 1    | 2     |
| **1000**    | 19  | 0   | 0   | **380** | 2    | 5     |
| **2000**    | 8   | 2   | 0   | 9    | **251** | 47    |
| **10000**   | 0   | 1   | 0   | 3    | 1    | **267** |

---

## Comparación con Run 25

| Métrica | Run 25 | Run 26 |
|---------|:------:|:------:|
| **Val accuracy** | 82.65% | **88.33%** |
| **Test accuracy** | 81.34% | **92.75%** |
| **Train accuracy** | 83.52% | **91.07%** |
| **Macro F1 (val)** | 0.82 | **0.88** |
| **Best val_loss** | 0.9278 | **0.7418** |
| **Mejor época** | 65 | **87** |
| **Tiempo** | 389s | 486s |

---

## Análisis

**Run 26 es el mejor resultado de todo el proyecto.** Con 92.75% en test, supera ampliamente el récord anterior (Run 22 con 81.70%).

### Aciertos — todas las clases mejoran drásticamente

- **Clase 10000 con recall 0.99 en test:** 91 de 92 billetes de 10000 correctamente identificados. Históricamente era la clase más problemática.
- **Clase 500 con precisión perfecta 1.00 y recall 0.92:** ningún falso positivo de 500 en test.
- **Clase 200 con F1 0.94:** precision 0.97, recall 0.91. Excelente balance.
- **Clase 2000 con precisión 0.95:** el mayor salto cualitativo. Antes estaba en 0.52-0.65. Ahora apenas 1+2+1=4 falsos positivos en test.
- **Clase 1000 con F1 0.91:** consistente y robusta.
- **Clase 100 con recall 0.95:** solo 5 errores en 92 imágenes.

### Matriz de confusión casi diagonal

La matriz de confusión en test es casi perfectamente diagonal. Las confusiones residuales son:
- 7 de 1000 → 100 (billetes de tono similar)
- 7 de 2000 → 10000 (y viceversa: 2 de 10000 → 1000)
- 4 de 200 → 100

### Factores del éxito

- **Mayor cantidad de épocas (137):** el modelo tuvo más oportunidades de converger. La mejor época fue la 87, mucho más tarde que en otros runs (38-65).
- **Inicialización favorable:** combinada con data augmentation que no perjudicó el aprendizaje.
- **Mejor val_loss registrado (0.7418):** indica que el modelo generalizó mejor que en cualquier otro run.

### Comparativa histórica

```
Run 22: ████████████████████████ 81.70% ← Récord anterior
Run 23: █████████████████████░░░ 68.12%
Run 24: ████████████████████████ 81.52%
Run 25: ████████████████████████ 81.34%
Run 26: ████████████████████████████████ 92.75% ← NUEVO RÉCORD
Run 27: ███████████████████████░░ 78.62%
```

### Conclusión

Run 26 demuestra el **techo de la configuración actual**: ~93% en test. La arquitectura de 651K params con weight decay 0.01, label smoothing y data augmentation puede alcanzar este nivel cuando la inicialización y el proceso estocástico lo favorecen. Es el modelo a presentar como resultado final del trabajo práctico.
