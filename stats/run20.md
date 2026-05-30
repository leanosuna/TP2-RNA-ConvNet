# Run 20 - Misma config que Run 18 (verificación de consistencia)

**Fecha:** 2026-05-28
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti
**Modelo:** ConvNet reducida (32→64→128, **651K params**)

---

## Cambios respecto a Run 18

Ninguno. Misma configuración: class weights, WD 0.01, label smoothing 0.1.

---

## Entrenamiento

- **Época alcanzada:** 74 (early stopping — val_loss divergió después de época 24)
- **Mejor época:** 24
- **Mejor val_loss:** 1.2979
- **Mejor val_acc:** **74.76%**
- **Tiempo:** 223.9s (3.7 min)

---

## Evaluación — Validation Set

| Clase   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 100     | 0.71      | 0.76   | 0.73     | 54      |
| 200     | 0.79      | 0.65   | 0.71     | 51      |
| 500     | **1.00**  | 0.73   | **0.84** | 55      |
| 1000    | 0.83      | 0.75   | **0.79** | 69      |
| 2000    | 0.52      | **0.81** | 0.64     | 53      |
| 10000   | 0.88      | 0.80   | **0.84** | 35      |
| **Accuracy** | | | **0.75** | **317** |
| **Macro Avg** | 0.79 | 0.75 | **0.76** | 317 |

## Comparación con Runs anteriores

| Métrica | Run 17 | Run 18 | Run 20 |
|---------|--------|--------|--------|
| Val accuracy | **80.13%** | **79.18%** | 74.76% |
| Test accuracy | **85.02%** | 76.81% | 76.81% |
| Macro F1 | **0.80** | 0.79 | 0.76 |
| Mejor época | 53 | 31 | **24** |
| Tiempo | 277s | 242s | **224s** |

## Conclusión

**Alta varianza entre corridas idénticas.** Run 20 (74.76%) rindió ~5 pp peor que Run 18 (79.18%) con la misma configuración. El val_loss divergió abruptamente después de época 24 (de 1.3 a 3.8+), lo que indica que el modelo cayó en un mal mínimo local.

Esto confirma que el principal cuello de botella sigue siendo el **tamaño del dataset**. Con ~1700 imágenes de entrenamiento y 6 clases (~280 promedio por clase), el modelo es sensible a la inicialización aleatoria. El techo práctico parece estar en ~78-80% val acc.
