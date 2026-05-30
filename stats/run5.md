# Run 5 - 4 Conv Layers + 200x200

**Fecha:** 2026-05-21
**Framework:** PyTorch 2.5.1 + CUDA 12.1
**GPU:** NVIDIA GeForce RTX 3060 Ti

## Cambios vs Run 4
| Aspecto | Run 4 | Run 5 |
|---------|-------|-------|
| Imagen | 160x160 | 200x200 |
| Conv layers | 3 (64,128,256) | 4 (64,128,256,**256**) |
| Parámetros | 1,453,766 | 2,044,102 |

## Resultado
- **Val accuracy: 48.28%** (vs 72.41% en Run 4)
- Mejor época: 24, Early stopping en 54
- Tiempo: 26.9s

## Conclusión
**Peor que Run 4.** La 4ta capa convolucional y la imagen más grande agregaron complejidad sin beneficio, causando inestabilidad en el validation loss y overfitting más temprano. Run 4 (3 capas + 160x160 + AdaptiveAvgPool2d) sigue siendo la mejor configuración.
