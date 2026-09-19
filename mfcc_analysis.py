import librosa
import numpy as np

files = [
    "human_recorded.wav",
    "chatgpt_recorded.wav",
    "gemini_recorded.wav",
    "fake_voice.wav"
]

print("MFCC + PITCH ANALYSIS")
print("=====================")

for f in files:
    audio, sr = librosa.load(f, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    pitch = librosa.yin(
        audio,
        fmin=70,
        fmax=400,
        sr=sr
    )

    print("\n" + f)
    print("---------------------")

    print(
        "MFCC Mean:",
        np.round(np.mean(mfcc, axis=1), 2)
    )

    print(
        "MFCC Std:",
        np.round(np.std(mfcc, axis=1), 2)
    )

    print(
        "Pitch Mean:",
        round(float(np.nanmean(pitch)), 2),
        "Hz"
    )

    print(
        "Pitch Std:",
        round(float(np.nanstd(pitch)), 2),
        "Hz"
    )