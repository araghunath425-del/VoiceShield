import os
import glob
import librosa
import numpy as np
import pandas as pd


# ============================================================
# VoiceShield - Feature Dataset Builder
# ============================================================

DATASET_PATH = "dataset"
OUTPUT_FILE = "voiceshield_features.csv"


# ------------------------------------------------------------
# Extract features from one audio file
# ------------------------------------------------------------
def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # Avoid problems with empty/silent audio
    if len(audio) == 0:
        return None

    # --------------------------------------------------------
    # Basic audio features
    # --------------------------------------------------------

    # RMS Energy
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
    # Pitch features
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
    # MFCC features
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
    # Store features
    # --------------------------------------------------------

    features = {

        "filename": os.path.basename(file_path),

        "rms": rms,

        "zcr": zcr,

        "spectral_centroid": spectral_centroid,

        "spectral_bandwidth": spectral_bandwidth,

        "spectral_rolloff": spectral_rolloff,

        "pitch_mean": pitch_mean,

        "pitch_std": pitch_std
    }

    # --------------------------------------------------------
    # Add MFCC mean features
    # --------------------------------------------------------

    for i in range(13):

        features[f"mfcc_mean_{i + 1}"] = mfcc_mean[i]

    # --------------------------------------------------------
    # Add MFCC standard deviation features
    # --------------------------------------------------------

    for i in range(13):

        features[f"mfcc_std_{i + 1}"] = mfcc_std[i]

    # --------------------------------------------------------
    # IMPORTANT:
    # Determine label from parent folder
    #
    # dataset/
    #    human/
    #       human_01.wav
    #
    #    synthetic/
    #       synthetic_01.wav
    # --------------------------------------------------------

    parent_folder = os.path.basename(
        os.path.dirname(file_path)
    ).lower()

    if parent_folder == "human":

        features["label"] = 0

    elif parent_folder == "synthetic":

        features["label"] = 1

    else:

        print(
            "WARNING: Unknown folder for:",
            file_path
        )

        return None

    return features


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 50)
    print("VoiceShield Feature Dataset Builder")
    print("=" * 50)
    print()

    # --------------------------------------------------------
    # Find human recordings
    # --------------------------------------------------------

    human_files = glob.glob(
        os.path.join(
            DATASET_PATH,
            "human",
            "*.wav"
        )
    )

    # --------------------------------------------------------
    # Find synthetic recordings
    # --------------------------------------------------------

    synthetic_files = glob.glob(
        os.path.join(
            DATASET_PATH,
            "synthetic",
            "*.wav"
        )
    )

    all_files = human_files + synthetic_files

    print("Human files:", len(human_files))
    print("Synthetic files:", len(synthetic_files))
    print("Total files:", len(all_files))
    print()

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if len(all_files) == 0:

        print("ERROR: No WAV files found!")

        print()
        print("Expected structure:")
        print()
        print("dataset/")
        print("├── human/")
        print("│   ├── human_01.wav")
        print("│   ├── human_02.wav")
        print("│   └── ...")
        print("│")
        print("└── synthetic/")
        print("    ├── synthetic_01.wav")
        print("    ├── synthetic_02.wav")
        print("    └── ...")
        print()

        return

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    dataset = []

    for file_path in all_files:

        print(
            "Processing:",
            file_path
        )

        try:

            features = extract_features(
                file_path
            )

            if features is not None:

                dataset.append(
                    features
                )

        except Exception as e:

            print(
                "ERROR processing",
                file_path
            )

            print(e)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    if len(dataset) == 0:

        print()
        print("ERROR: No features were extracted.")
        return

    df = pd.DataFrame(dataset)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print("FEATURE DATASET CREATED")
    print("=" * 50)

    print(
        "Samples:",
        len(df)
    )

    print(
        "Features:",
        len(df.columns) - 2
    )

    print()

    print("Labels:")

    print(
        "Human:",
        int((df["label"] == 0).sum())
    )

    print(
        "Synthetic:",
        int((df["label"] == 1).sum())
    )

    print()

    print(
        "Saved as:",
        OUTPUT_FILE
    )

    print()

    # --------------------------------------------------------
    # Show first few rows
    # --------------------------------------------------------

    print("Dataset preview:")
    print()

    print(
        df[
            ["filename", "label"]
        ].to_string(index=False)
    )

    print()

    print("=" * 50)
    print("DONE")
    print("=" * 50)


# ============================================================
# Run program
# ============================================================

if __name__ == "__main__":

    main()