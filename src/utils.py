import numpy as np
import librosa
import soundfile


def extract_feature(file_name: str, mfcc: bool = True, chroma: bool = True, mel: bool = True) -> np.ndarray:
    with soundfile.SoundFile(file_name) as sound_file:
        x = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        result = np.array([], dtype=np.float32)

        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T,
                axis=0,
            )
            result = np.hstack((result, mfccs))

        if chroma:
            stft = np.abs(librosa.stft(x))
            chroma_features = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                axis=0,
            )
            result = np.hstack((result, chroma_features))

        if mel:
            mel_features = np.mean(
                librosa.feature.melspectrogram(y=x, sr=sample_rate).T,
                axis=0,
            )
            result = np.hstack((result, mel_features))

    return result.astype(np.float32)


def add_noise_to_feature(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    noise_amp = 0.005 * np.random.uniform() * np.max(np.abs(data))
    noise = noise_amp * np.random.normal(size=data.shape[0]).astype(np.float32)
    return data + noise
