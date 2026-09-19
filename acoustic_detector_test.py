import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -------------------------------------------------
# Extract acoustic features
# -------------------------------------------------

def extract_features(file):
    audio, sr = librosa.load(file, sr=16000, mono=True)

    # Basic features
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    # Pitch
    pitch = librosa.yin(
        audio,
        fmin=70,
        fmax=400,
        sr=sr
    )

    pitch_mean = np.nanmean(pitch)
    pitch_std = np.nanstd(pitch)

    features = np.concatenate([
        [
            rms,
            zcr,
            centroid,
            bandwidth,
            rolloff,
            pitch_mean,
            pitch_std
        ],
        mfcc_mean
    ])

    return features


# -------------------------------------------------
# Dataset
# -------------------------------------------------

files = [
    "human_recorded.wav",
    "chatgpt_recorded.wav",
    "gemini_recorded.wav",
    "fake_voice.wav"
]

# 0 = Human
# 1 = Synthetic / AI
labels = np.array([0, 1, 1, 1])

print("\n========================================")
print("     VOICESHIELD ACOUSTIC EXPERIMENT")
print("========================================")

X = []

for file in files:
    print("\nExtracting:", file)

    features = extract_features(file)

    X.append(features)

    print("Features extracted:", len(features))

X = np.array(X)

print("\nFeature matrix shape:", X.shape)

# -------------------------------------------------
# Scale features
# -------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# SVM
# -------------------------------------------------

model = SVC(
    kernel="rbf",
    probability=True
)

model.fit(X_scaled, labels)

# -------------------------------------------------
# Predictions
# -------------------------------------------------

print("\n========================================")
print("          PREDICTION RESULTS")
print("========================================")

predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)

for i, file in enumerate(files):

    prediction = predictions[i]

    if prediction == 0:
        result = "HUMAN"
    else:
        result = "SYNTHETIC / AI"

    confidence = probabilities[i][prediction] * 100

    print("\nFile:", file)
    print("Prediction:", result)
    print("Confidence:", round(confidence, 2), "%")