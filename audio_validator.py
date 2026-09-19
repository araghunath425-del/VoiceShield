import librosa
import numpy as np


# ============================================================
# VoiceShield - Audio Validation Layer
# ============================================================

TARGET_SR = 16000

# Minimum useful duration for analysis
MIN_DURATION = 1.0

# RMS thresholds
VERY_LOW_RMS = 0.005
LOW_RMS = 0.015


def validate_audio(audio_path):

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    try:

        audio, sr = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True
        )

    except Exception as e:

        return {
            "valid": False,
            "status": "ERROR",
            "message": f"Could not read audio: {e}"
        }

    # --------------------------------------------------------
    # Check empty audio
    # --------------------------------------------------------

    if len(audio) == 0:

        return {
            "valid": False,
            "status": "INVALID",
            "message": "The audio file is empty."
        }

    # --------------------------------------------------------
    # Basic measurements
    # --------------------------------------------------------

    duration = len(audio) / sr

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    # --------------------------------------------------------
    # Check duration
    # --------------------------------------------------------

    if duration < MIN_DURATION:

        return {
            "valid": False,
            "status": "TOO_SHORT",
            "message": (
                f"Audio is only {duration:.2f} seconds long. "
                "Please provide at least 1 second of audio."
            ),
            "duration": duration,
            "sample_rate": sr,
            "rms": rms,
            "peak": peak
        }

    # --------------------------------------------------------
    # Check silence
    # --------------------------------------------------------

    if rms < VERY_LOW_RMS:

        return {
            "valid": False,
            "status": "SILENT",
            "message": (
                "Very little audio energy was detected. "
                "Please check your microphone or recording."
            ),
            "duration": duration,
            "sample_rate": sr,
            "rms": rms,
            "peak": peak
        }

    # --------------------------------------------------------
    # Audio level classification
    # --------------------------------------------------------

    if rms < LOW_RMS:

        level = "LOW"

    elif rms < 0.15:

        level = "NORMAL"

    else:

        level = "HIGH"

    # --------------------------------------------------------
    # Zero Crossing Rate
    # --------------------------------------------------------

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                audio
            )
        )
    )

    # --------------------------------------------------------
    # Spectral features
    # --------------------------------------------------------

    spectral_centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=sr
            )
        )
    )

    spectral_bandwidth = float(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=audio,
                sr=sr
            )
        )
    )

    spectral_rolloff = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=audio,
                sr=sr
            )
        )
    )

    # --------------------------------------------------------
    # Simple speech-likeness heuristic
    #
    # This is NOT an AI detector.
    # It only provides an approximate audio-type indication.
    # --------------------------------------------------------

    if (
        70 <= spectral_centroid <= 3500
        and 0.01 <= zcr <= 0.30
    ):

        audio_type = "SPEECH-LIKE"

    elif spectral_centroid > 3500:

        audio_type = "HIGH-FREQUENCY / POSSIBLE MUSIC"

    else:

        audio_type = "NON-SPEECH / UNCERTAIN"

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    return {
        "valid": True,
        "status": "VALID",
        "message": "Audio is suitable for further analysis.",

        "duration": duration,

        "sample_rate": sr,

        "samples": len(audio),

        "rms": rms,

        "peak": peak,

        "audio_level": level,

        "zcr": zcr,

        "spectral_centroid": spectral_centroid,

        "spectral_bandwidth": spectral_bandwidth,

        "spectral_rolloff": spectral_rolloff,

        "audio_type": audio_type
    }


# ============================================================
# Command-line test
# ============================================================

if __name__ == "__main__":

    import sys

    print()
    print("=" * 60)
    print("VoiceShield Audio Validation Test")
    print("=" * 60)
    print()

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python audio_validator.py <audio_file>"
        )

        print()

        sys.exit()

    audio_file = sys.argv[1]

    print(
        "Testing:",
        audio_file
    )

    print()

    result = validate_audio(
        audio_file
    )

    print("-" * 60)

    for key, value in result.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    print("-" * 60)
    print()