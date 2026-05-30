# Run 14 - Misma config que Run 11 (verificación de consistencia)

**Fecha:** 2026-05-23
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, 651K params)

---

## Config (idéntica a Run 11)

| Parámetro | Valor |
|-----------|-------|
| Resolución | 160×160 squash |
| Conv filters | [32, 64, 128] |
| Dropout | 0.3 |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| MaxPool2d padding | 0 |
| Early stopping | val_loss |

---

## Resultado

| Métrica | Run 11 | Run 14 |
|---------|--------|--------|
| **Val accuracy** | **73.81%** | **61.90%** |
| Macro F1 | 0.72 | 0.60 |
| Best val_loss | 1.8392 | **1.2462** (récord) |
| Best epoch | 30 | 24 |
| Tiempo | 19.9s | 17.2s |

---

## Evaluación

| Clase   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| 100     | 0.60      | 0.43   | 0.50     |
| 200     | 0.67      | 0.29   | 0.40     |
| 500     | 0.78      | 1.00   | **0.88** |
| 1000    | 0.67      | 0.86   | 0.75     |
| 2000    | 0.50      | 0.71   | 0.59     |
| 10000   | 0.50      | 0.43   | 0.46     |
| **Accuracy** | | | **0.62** |
| **Macro F1** | | | **0.60** |

---

## Conclusión

**La varianza entre corridas es de ~12 pp (62%–74%) con la misma configuración.** Esto confirma que el cuello de botella es el tamaño del dataset (144 train), no la config del modelo. Para obtener resultados consistentes harían falta más imágenes.
