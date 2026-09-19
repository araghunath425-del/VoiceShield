import os
import librosa
import numpy as np
import pandas as pd
import joblib


# ============================================================
# VoiceShield - Acoustic SVM Testing
# ============================================================

MODEL_FILE = "acoustic_svm_model.pkl"


# ------------------------------------------------------------
# Feature extraction
# Must be EXACTLY the same as training
# ------------------------------------------------------------

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    # RMS
    rms = np.mean(
        librosa.feature.rms(y=audio)
    )

    # Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    # Spectral Centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    # Spectral Bandwidth
    spectral_bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )
    )

    # Spectral Rolloff
    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )
    )

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    try:

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        f0_valid = f0[~np.isnan(f0)]

        if len(f0_valid) > 0:
            pitch_mean = np.mean(f0_valid)
            pitch_std = np.std(f0_valid)
        else:
            pitch_mean = 0
            pitch_std = 0

    except Exception:

        pitch_mean = 0
        pitch_std = 0

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    # --------------------------------------------------------
    # Build feature dictionary
    # --------------------------------------------------------

    features = {

        "rms": rms,

        "zcr": zcr,

        "spectral_centroid": spectral_centroid,

        "spectral_bandwidth": spectral_bandwidth,

        "spectral_rolloff": spectral_rolloff,

        "pitch_mean": pitch_mean,

        "pitch_std": pitch_std
    }

    # MFCC means
    for i in range(13):

        features[f"mfcc_mean_{i + 1}"] = mfcc_mean[i]

    # MFCC standard deviations
    for i in range(13):

        features[f"mfcc_std_{i + 1}"] = mfcc_std[i]

    return features


# ------------------------------------------------------------
# Test one audio file
# ------------------------------------------------------------

def test_audio(model, file_path):

    print()
    print("-" * 60)
    print("Testing:", file_path)
    print("-" * 60)

    if not os.path.exists(file_path):

        print("FILE NOT FOUND")

        return

    try:

        features = extract_features(
            file_path
        )

        # Convert dictionary to DataFrame
        X = pd.DataFrame(
            [features]
        )

        # Prediction
        prediction = model.predict(X)[0]

        # Probability
        probabilities = model.predict_proba(X)[0]

        human_probability = probabilities[0] * 100
        synthetic_probability = probabilities[1] * 100

        if prediction == 0:

            result = "HUMAN"

        else:

            result = "SYNTHETIC / AI"

        print()

        print("Prediction:")
        print(result)

        print()

        print(
            f"Human probability:     {human_probability:.2f}%"
        )

        print(
            f"Synthetic probability: {synthetic_probability:.2f}%"
        )

    except Exception as e:

        print()
        print("ERROR:")
        print(e)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("VoiceShield Acoustic SVM Test")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_FILE):

        print(
            "ERROR: acoustic_svm_model.pkl not found."
        )

        return

    print("Loading acoustic SVM model...")

    model = joblib.load(
        MODEL_FILE
    )

    print("Model loaded successfully!")

    # --------------------------------------------------------
    # Test files
    # --------------------------------------------------------

    test_files = [

        "human_recorded.wav",

        "chatgpt_recorded.wav",

        "gemini_recorded.wav",

        "fake_voice.wav"
    ]

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    for file_path in test_files:

        test_audio(
            model,
            file_path
        )

    print()
    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print()


# ------------------------------------------------------------
# Start
# ------------------------------------------------------------

if __name__ == "__main__":

    main()