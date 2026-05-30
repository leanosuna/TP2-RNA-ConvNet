import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.v2 as transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

torch.backends.cudnn.benchmark = True

from config import (
    MODEL_SAVE_PATH,
    TRAIN_DATASET_DIR, VAL_DATASET_DIR, TEST_DATASET_DIR,
    CLASSES, NUM_CLASSES,
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    AUGMENT_ROTATION, AUGMENT_HFLIP, AUGMENT_BRIGHTNESS, AUGMENT_CONTRAST,
    AUGMENT_SATURATION, AUGMENT_HUE, AUGMENT_AFFINE_SCALE, AUGMENT_AFFINE_TRANSLATE,
    CONV_FILTERS, CONV_KERNEL_SIZE, CONV_POOL_SIZE,
    LINEAR_LAYER_CONFIG, DROPOUT_RATE,
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, EARLY_STOP_START_EPOCH,
    USE_CLASS_WEIGHTS, SAVE_KERAS,
)


class ConvNet(nn.Module):
    def __init__(self, num_classes, conv_filters, linear_config, dropout_rate):
        super().__init__()
        layers = []
        in_channels = IMAGE_CHANNELS

        for out_channels in conv_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=CONV_KERNEL_SIZE, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=CONV_POOL_SIZE))
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        gap_output = conv_filters[-1] * 4 * 4

        linear_parts = [s.strip() for s in linear_config.split(",")]
        linear_layers = []
        prev_size = gap_output

        for val in linear_parts:
            if val.upper() == "D":
                linear_layers.append(nn.Dropout(dropout_rate))
            elif val.isnumeric():
                out_size = int(val)
                linear_layers.append(nn.Linear(prev_size, out_size))
                linear_layers.append(nn.ReLU(inplace=True))
                prev_size = out_size

        self.linear_blocks = nn.Sequential(*linear_layers)
        self.classifier = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear_blocks(x)
        x = self.classifier(x)
        return x


def get_augment_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=AUGMENT_HFLIP),
        transforms.RandomRotation(AUGMENT_ROTATION),
        transforms.ColorJitter(
            brightness=AUGMENT_BRIGHTNESS,
            contrast=AUGMENT_CONTRAST,
            saturation=AUGMENT_SATURATION,
            hue=AUGMENT_HUE,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=AUGMENT_AFFINE_TRANSLATE,
            scale=AUGMENT_AFFINE_SCALE,
        ),
    ])


def load_images(data_dir, classes_list):
    images = []
    labels = []
    base_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_to_idx = {c: i for i, c in enumerate(classes_list)}

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        prefix = fname.split("_")[0]
        if prefix not in class_to_idx:
            continue
        img = Image.open(os.path.join(data_dir, fname)).convert("RGB")
        img = base_transform(img)
        images.append(img)
        labels.append(class_to_idx[prefix])

    print(f"  Loaded {len(images)} images from {data_dir}")
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def plot_training_history(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    plot_path = save_path.replace(".pth", "_training.png")

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Entrenar ConvNet para billetes argentinos")
    parser.add_argument("--run", type=str, default=None, help="Nombre del run (ej: run16). Crea models/runXX/ con checkpoint y gráfico.")
    args = parser.parse_args()

    print("=" * 60)
    print("ConvNet - Reconocimiento de Billetes Argentinos (PyTorch)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading training images from: {TRAIN_DATASET_DIR}")
    train_images, train_labels = load_images(TRAIN_DATASET_DIR, CLASSES)

    print(f"\nLoading validation images from: {VAL_DATASET_DIR}")
    val_images, val_labels = load_images(VAL_DATASET_DIR, CLASSES)

    print(f"\nLoading test images from: {TEST_DATASET_DIR}")
    test_images, test_labels = load_images(TEST_DATASET_DIR, CLASSES)

    if len(train_images) == 0:
        print("ERROR: No training images loaded. Check dataset path.")
        return

    print(f"\nDataset distribution:")
    for i, cls in enumerate(CLASSES):
        t_count = (train_labels == i).sum().item()
        v_count = (val_labels == i).sum().item()
        test_count = (test_labels == i).sum().item()
        print(f"  Class '{cls}': train={t_count}, val={v_count}, test={test_count}")

    val_images = val_images.to(device)
    val_labels = val_labels.to(device)
    val_dataset = TensorDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nBuilding ConvNet model...")
    model = ConvNet(NUM_CLASSES, CONV_FILTERS, LINEAR_LAYER_CONFIG, DROPOUT_RATE).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    if USE_CLASS_WEIGHTS:
        class_counts = torch.bincount(train_labels)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        class_weights = class_weights.to(device)
        print(f"  Class weights: {[f'{w:.3f}' for w in class_weights]}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    augment = get_augment_transform()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0

    print(f"\nTraining for up to {EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        perm = torch.randperm(len(train_images))
        train_images_shuffled = train_images[perm]
        train_labels_shuffled = train_labels[perm]

        for i in range(0, len(train_images), BATCH_SIZE):
            batch_imgs = train_images_shuffled[i:i+BATCH_SIZE]
            batch_labels = train_labels_shuffled[i:i+BATCH_SIZE]

            batch_imgs = augment(batch_imgs).to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_imgs.size(0)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA and epoch >= EARLY_STOP_START_EPOCH:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        elif epoch >= EARLY_STOP_START_EPOCH:
            epochs_no_improve += 1

        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} "
              f"lr: {current_lr:.2e}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE and epoch >= EARLY_STOP_START_EPOCH:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)

    print(f"\nEvaluating on validation set...")
    model.eval()
    all_preds = []
    all_labels_list = []

    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels_np = np.array(all_labels_list)

    print("\nClassification Report:")
    print(classification_report(all_labels_np, all_preds, target_names=CLASSES, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels_np, all_preds)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print(cm_df)

    print(f"\nEvaluating on test set...")
    model.eval()
    all_preds_test = []
    all_labels_test = []

    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds_test.extend(predicted.cpu().numpy())
            all_labels_test.extend(targets.cpu().numpy())

    all_preds_test = np.array(all_preds_test)
    all_labels_test = np.array(all_labels_test)

    print("\nTest Set Classification Report:")
    print(classification_report(all_labels_test, all_preds_test, target_names=CLASSES, zero_division=0))

    print("\nTest Set Confusion Matrix:")
    cm_test = confusion_matrix(all_labels_test, all_preds_test)
    cm_test_df = pd.DataFrame(cm_test, index=CLASSES, columns=CLASSES)
    print(cm_test_df)

    print(f"\nEvaluating on training set...")
    train_images_eval = train_images.to(device)
    train_labels_eval = train_labels.to(device)
    train_dataset_eval = TensorDataset(train_images_eval, train_labels_eval)
    train_loader_eval = DataLoader(train_dataset_eval, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds_train = []
    all_labels_train = []

    with torch.no_grad():
        for images, targets in train_loader_eval:
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds_train.extend(predicted.cpu().numpy())
            all_labels_train.extend(targets.cpu().numpy())

    all_preds_train = np.array(all_preds_train)
    all_labels_train = np.array(all_labels_train)

    print("\nTraining Set Classification Report:")
    print(classification_report(all_labels_train, all_preds_train, target_names=CLASSES, zero_division=0))

    print("\nTraining Set Confusion Matrix:")
    cm_train = confusion_matrix(all_labels_train, all_preds_train)
    cm_train_df = pd.DataFrame(cm_train, index=CLASSES, columns=CLASSES)
    print(cm_train_df)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "image_size": (IMAGE_WIDTH, IMAGE_HEIGHT),
        "conv_filters": CONV_FILTERS,
        "linear_config": LINEAR_LAYER_CONFIG,
        "dropout_rate": DROPOUT_RATE,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": history["val_acc"][best_epoch - 1] if best_epoch > 0 else 0,
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    if args.run:
        run_dir = os.path.join(os.path.dirname(MODEL_SAVE_PATH), args.run)
        os.makedirs(run_dir, exist_ok=True)
        run_model_path = os.path.join(run_dir, "convnet_billetes.pth")
        torch.save(checkpoint, run_model_path)
        print(f"Model also saved to: {run_model_path}")
        plot_training_history(history, os.path.join(run_dir, "convnet_billetes.pth"))

        metrics_path = os.path.join(run_dir, "metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"Run: {args.run}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best val_loss: {best_val_loss:.4f}\n")
            if best_epoch > 0:
                f.write(f"Best val_acc: {history['val_acc'][best_epoch - 1]:.4f}\n")
            f.write(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n\n")

            f.write("=" * 60 + "\n")
            f.write("VALIDATION SET\n")
            f.write("=" * 60 + "\n")
            f.write(classification_report(all_labels_np, all_preds, target_names=CLASSES, zero_division=0))
            f.write("\nConfusion Matrix:\n")
            f.write(cm_df.to_string())
            f.write("\n\n")

            f.write("=" * 60 + "\n")
            f.write("TEST SET\n")
            f.write("=" * 60 + "\n")
            f.write(classification_report(all_labels_test, all_preds_test, target_names=CLASSES, zero_division=0))
            f.write("\nConfusion Matrix:\n")
            f.write(cm_test_df.to_string())
            f.write("\n\n")

            f.write("=" * 60 + "\n")
            f.write("TRAINING SET\n")
            f.write("=" * 60 + "\n")
            f.write(classification_report(all_labels_train, all_preds_train, target_names=CLASSES, zero_division=0))
            f.write("\nConfusion Matrix:\n")
            f.write(cm_train_df.to_string())
            f.write("\n")

        print(f"Metrics saved to: {metrics_path}")
    else:
        plot_training_history(history, MODEL_SAVE_PATH)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    if best_epoch > 0:
        print(f"Best val_acc: {history['val_acc'][best_epoch - 1]:.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
