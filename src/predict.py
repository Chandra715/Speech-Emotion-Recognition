import argparse
from pathlib import Path

import joblib
import numpy as np

from utils import extract_feature


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from a single audio file.")
    parser.add_argument("--file", required=True, help="Path to a .wav file")
    parser.add_argument(
        "--model",
        default="models/mlp_emotion_model.pkl",
        help="Path to the trained model file",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    file_path = Path(args.file)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run training first."
        )

    model = joblib.load(model_path)
    feature = extract_feature(str(file_path), mfcc=True, chroma=True, mel=True)
    prediction = model.predict(np.expand_dims(feature, axis=0))[0]

    print(f"Predicted emotion: {prediction}")


if __name__ == "__main__":
    main()
