import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset
RAW_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "convnet_billetes.pth")

# Classes (bill denominations)
CLASSES = ["100", "200", "500", "1000", "2000", "10000"]
NUM_CLASSES = len(CLASSES)

# Image parameters
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3

# Class weights (balanced by frequency)
USE_CLASS_WEIGHTS = False

# Training parameters
BATCH_SIZE = 16
EPOCHS = 150
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Data augmentation parameters
AUGMENT_ROTATION = 15
AUGMENT_HFLIP = 0.5
AUGMENT_BRIGHTNESS = 0.15
AUGMENT_CONTRAST = 0.1
AUGMENT_SATURATION = 0.1
AUGMENT_HUE = 0.05
AUGMENT_AFFINE_SCALE = (0.9, 1.1)
AUGMENT_AFFINE_TRANSLATE = (0.1, 0.1)

# ConvNet architecture (Run 4 best - [64, 128, 256])
CONV_FILTERS = [64, 128, 256]
CONV_KERNEL_SIZE = 3
CONV_POOL_SIZE = 2

# Linear layers (AdaptiveAvgPool2d(4,4) -> 256*4*4 = 4096 input)
LINEAR_LAYER_CONFIG = "256,D,128"
DROPOUT_RATE = 0.5

# Early stopping
EARLY_STOP_PATIENCE = 50
EARLY_STOP_MIN_DELTA = 0.001
EARLY_STOP_START_EPOCH = 20
