import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import (
    RAW_DATASET_DIR, MODEL_SAVE_PATH,
    CLASSES, NUM_CLASSES,
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, IMAGE_SHAPE,
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, LEARNING_RATE,
    AUGMENT_ROTATION, AUGMENT_WIDTH_SHIFT, AUGMENT_HEIGHT_SHIFT,
    AUGMENT_ZOOM, AUGMENT_BRIGHTNESS, AUGMENT_CONTRAST,
    CONV_LAYERS, CONV_KERNEL_SIZE, CONV_POOL_SIZE, CONV_ACTIVATION,
    LINEAR_LAYER_CONFIG, LINEAR_ACTIVATION, DROPOUT_RATE,
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, EARLY_STOP_START_EPOCH,
)


def load_images_from_dir(base_dir, classes_list, img_shape):
    images = []
    labels = []
    img_w, img_h, img_c = img_shape

    for class_idx, class_name in enumerate(classes_list):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  Warning: directory not found for class '{class_name}', skipping")
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(class_dir, filename)
                try:
                    img = Image.open(filepath).convert("RGB")
                    img = img.resize((img_w, img_h), Image.LANCZOS)
                    images.append(np.array(img))
                    labels.append(class_idx)
                except Exception as e:
                    print(f"  Error loading {filepath}: {e}")

    return np.array(images), np.array(labels)


def build_augmentation_layers():
    layers = [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(AUGMENT_ROTATION),
        keras.layers.RandomTranslation(AUGMENT_WIDTH_SHIFT, AUGMENT_HEIGHT_SHIFT, fill_mode="nearest"),
        keras.layers.RandomZoom(AUGMENT_ZOOM, fill_mode="nearest"),
        keras.layers.RandomContrast(AUGMENT_CONTRAST),
        keras.layers.RandomBrightness(AUGMENT_BRIGHTNESS),
    ]
    return layers


def build_convnet(input_shape, num_classes, augment_layers):
    input_layer = keras.layers.Input(shape=input_shape, name="input_img")
    x = input_layer

    for aug_layer in augment_layers:
        x = aug_layer(x)

    x = keras.layers.Rescaling(1.0 / 255, name="rescale")(x)

    for i in range(CONV_LAYERS, 0, -1):
        filters = 2 ** (i + 2)
        x = keras.layers.Conv2D(
            filters, CONV_KERNEL_SIZE, activation=CONV_ACTIVATION,
            padding="same", name=f"conv_{CONV_LAYERS - i + 1}"
        )(x)
        x = keras.layers.MaxPooling2D(
            CONV_POOL_SIZE, padding="same", name=f"pool_{CONV_LAYERS - i + 1}"
        )(x)

    x = keras.layers.Flatten(name="flatten")(x)

    linear_config = [s.strip() for s in LINEAR_LAYER_CONFIG.split(",")]
    for idx, val in enumerate(linear_config):
        if val.upper() == "D":
            x = keras.layers.Dropout(DROPOUT_RATE, name=f"dropout_{idx + 1}")(x)
        elif val.upper() == "BN":
            x = keras.layers.BatchNormalization(name=f"bn_{idx + 1}")(x)
        elif val.isnumeric():
            x = keras.layers.Dense(
                int(val), activation=LINEAR_ACTIVATION, name=f"dense_{idx + 1}"
            )(x)

    output_layer = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(input_layer, output_layer, name="ConvNet")
    return model


def plot_training_history(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    plt.tight_layout()
    plot_path = save_path.replace(".keras", "_training.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to: {plot_path}")


def main():
    print("=" * 60)
    print("ConvNet - Reconocimiento de Billetes Argentinos")
    print("=" * 60)

    print(f"\nTensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU devices found: {gpus}")
    else:
        print("No GPU devices found. Running on CPU.")
        print("Note: TensorFlow >= 2.11 does not support GPU on native Windows.")
        print("Use WSL2 or tensorflow-directml plugin for GPU acceleration.")

    print(f"\nLoading images from: {RAW_DATASET_DIR}")
    images, labels = load_images_from_dir(RAW_DATASET_DIR, CLASSES, IMAGE_SHAPE)
    print(f"Total images loaded: {len(images)}")

    if len(images) == 0:
        print("ERROR: No images loaded. Check dataset path.")
        return

    for i, cls in enumerate(CLASSES):
        count = np.sum(labels == i)
        print(f"  Class '{cls}': {count} images")

    print(f"\nSplitting data: {100 - VALIDATION_SPLIT*100:.0f}% train, {VALIDATION_SPLIT*100:.0f}% validation")
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=VALIDATION_SPLIT, stratify=labels, random_state=42
    )
    print(f"  Train: {len(x_train)} images")
    print(f"  Validation: {len(x_val)} images")

    y_train_enc = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_enc = to_categorical(y_val, num_classes=NUM_CLASSES)

    print("\nBuilding augmentation layers (applied online during training)...")
    augment_layers = build_augmentation_layers()
    for layer in augment_layers:
        print(f"  - {layer.name}")

    print("\nBuilding ConvNet model...")
    model = build_convnet(IMAGE_SHAPE, NUM_CLASSES, augment_layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=EARLY_STOP_MIN_DELTA,
            patience=EARLY_STOP_PATIENCE,
            verbose=1,
            mode="min",
            restore_best_weights=True,
            start_from_epoch=EARLY_STOP_START_EPOCH,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        x_train, y_train_enc,
        validation_data=(x_val, y_val_enc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on validation set...")
    y_pred = model.predict(x_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=CLASSES, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred_classes)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print(cm_df)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    plot_training_history(history, MODEL_SAVE_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()
