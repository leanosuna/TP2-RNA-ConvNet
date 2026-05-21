import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset
RAW_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")
AUGMENTED_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "augmented")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "convnet_billetes.keras")

# Classes (bill denominations)
CLASSES = ["100", "200", "500", "1000", "2000", "10000"]
NUM_CLASSES = len(CLASSES)

# Image parameters
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

# Training parameters
BATCH_SIZE = 8
EPOCHS = 100
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Data augmentation parameters
AUGMENT_ROTATION = 0.2
AUGMENT_WIDTH_SHIFT = 0.2
AUGMENT_HEIGHT_SHIFT = 0.2
AUGMENT_ZOOM = 0.2
AUGMENT_BRIGHTNESS = 0.2
AUGMENT_CONTRAST = 0.2

# ConvNet architecture
CONV_LAYERS = 3
CONV_KERNEL_SIZE = 3
CONV_POOL_SIZE = 2
CONV_ACTIVATION = "relu"

# Linear layers
LINEAR_LAYER_CONFIG = "64,32"
LINEAR_ACTIVATION = "relu"
DROPOUT_RATE = 0.3

# Early stopping
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 0.001
EARLY_STOP_START_EPOCH = 15
