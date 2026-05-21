import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

from config import CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT, MODEL_SAVE_PATH


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
    return np.array(img)


def predict(model, image_path):
    img_array = load_and_preprocess_image(image_path)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    all_probs = {cls: float(predictions[0][i]) for i, cls in enumerate(CLASSES)}

    return CLASSES[predicted_class], confidence, all_probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [image_path2 ...]")
        sys.exit(1)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}")
        print("Run train.py first to train the model.")
        sys.exit(1)

    print(f"Loading model from: {MODEL_SAVE_PATH}")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    for image_path in sys.argv[1:]:
        if not os.path.exists(image_path):
            print(f"\nFile not found: {image_path}")
            continue

        print(f"\nImage: {image_path}")
        predicted_class, confidence, all_probs = predict(model, image_path)
        print(f"  Prediction: ${predicted_class}")
        print(f"  Confidence: {confidence:.2%}")
        print("  Probabilities:")
        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(prob * 30)
            print(f"    ${cls:>5}: {prob:.2%} {bar}")


if __name__ == "__main__":
    main()
