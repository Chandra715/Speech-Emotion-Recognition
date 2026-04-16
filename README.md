# Speech Emotion Recognition

A clean, GitHub-ready machine learning project for speech emotion recognition using handcrafted audio features and an MLP classifier.

## Overview
This project classifies speech clips into emotion categories using:
- MFCC features
- Chroma features
- Mel spectrogram features
- noise-based augmentation
- Scikit-learn MLPClassifier

## Supported emotions
- calm
- happy
- fearful
- disgust

## Project structure
```text
speech-emotion-recognition-github/
├── src/
│   ├── train.py
│   ├── utils.py
│   └── predict.py
├── data/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset layout
Place your `.wav` files like this:
```text
data/Actors/Actor_01/*.wav
data/Actors/Actor_02/*.wav
...
```

The code expects filenames in RAVDESS-style format where the third token indicates emotion:
`03-01-06-01-02-01-12.wav`

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python src/train.py
```

## Predict
```bash
python src/predict.py --file "data/Actors/Actor_01/example.wav"
```

## Output
- Trained model: `models/mlp_emotion_model.pkl`

## Notes
- The dataset itself is not included in this repository.
- If you use a different folder layout, update the glob pattern in `src/train.py`.

## Resume-ready description
Built a speech emotion recognition pipeline in Python using Librosa-based audio feature extraction, augmentation, and an MLP classifier to predict emotions from speech recordings.
