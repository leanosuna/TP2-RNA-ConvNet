import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image

from config import (
    CLASSES, NUM_CLASSES, MODEL_SAVE_PATH,
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS,
    CONV_FILTERS, LINEAR_LAYER_CONFIG, DROPOUT_RATE,
)


class ConvNet(nn.Module):
    def __init__(self, num_classes, conv_filters, linear_config, dropout_rate, img_w, img_h):
        super().__init__()
        layers = []
        in_channels = IMAGE_CHANNELS

        for out_channels in conv_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, padding=1))
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


def get_inference_transform(img_w, img_h):
    return transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(model, transform, image_path, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        predicted_class = probs.argmax().item()
        confidence = probs[predicted_class].item()

    all_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    return CLASSES[predicted_class], confidence, all_probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [image_path2 ...]")
        sys.exit(1)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}")
        print("Run train.py first to train the model.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from: {MODEL_SAVE_PATH}")

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    img_w, img_h = checkpoint["image_size"]

    model = ConvNet(
        NUM_CLASSES,
        checkpoint["conv_filters"],
        checkpoint["linear_config"],
        checkpoint["dropout_rate"],
        img_w, img_h,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_inference_transform(img_w, img_h)

    for image_path in sys.argv[1:]:
        if not os.path.exists(image_path):
            print(f"\nFile not found: {image_path}")
            continue

        print(f"\nImage: {image_path}")
        predicted_class, confidence, all_probs = predict(model, transform, image_path, device)
        print(f"  Prediction: ${predicted_class}")
        print(f"  Confidence: {confidence:.2%}")
        print("  Probabilities:")
        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(prob * 30)
            print(f"    ${cls:>5}: {prob:.2%} {bar}")


if __name__ == "__main__":
    main()
