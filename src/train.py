import glob
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils import extract_feature, add_noise_to_feature

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

OBSERVED_EMOTIONS = {"calm", "happy", "fearful", "disgust"}
DATA_GLOB = "data/Actors/Actor_*/*.wav"
MODEL_PATH = Path("models/mlp_emotion_model.pkl")


def load_data(test_size: float = 0.15, random_state: int = 42):
    features = []
    labels = []

    files = glob.glob(DATA_GLOB)
    if not files:
        raise FileNotFoundError(
            f"No audio files found. Expected files matching: {DATA_GLOB}"
        )

    for file_path in files:
        file_name = os.path.basename(file_path)
        parts = file_name.split("-")
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        emotion = EMOTIONS.get(emotion_code)
        if emotion not in OBSERVED_EMOTIONS:
            continue

        feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        feature = add_noise_to_feature(feature)

        features.append(feature)
        labels.append(emotion)

    if not features:
        raise ValueError("No valid training samples found after filtering emotions.")

    x = np.array(features, dtype=np.float32)
    y = np.array(labels)

    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="tanh",
        alpha=0.01,
        batch_size=256,
        learning_rate="constant",
        max_iter=500,
        random_state=42,
    )


def main():
    print("Loading data...")
    x_train, x_test, y_train, y_test = load_data()

    print(f"Train shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")
    print(f"Features extracted: {x_train.shape[1]}")

    model = build_model()

    print("Training model...")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
