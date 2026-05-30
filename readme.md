# RNA ConvNet — Reconocimiento de Billetes Argentinos (AR$)

Red neuronal convolucional para clasificar billetes argentinos de denominaciones **100, 200, 500, 1000, 2000 y 10000**.

## Requisitos

- Python 3.12
- PyTorch 2.5.1 + CUDA 12.1
- GPU NVIDIA con soporte CUDA (recomendado)

## Entrenamiento

```powershell
py -3.12 src/train.py --run runXX
```

Esto entrena el modelo y guarda en `models/runXX/`:
- `convnet_billetes.pth` — checkpoint del modelo
- `convnet_billetes_training.png` — gráfico de loss y accuracy por época
- `metrics.txt` — classification report y matriz de confusión para train/val/test

