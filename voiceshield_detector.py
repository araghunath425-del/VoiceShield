import torch
import librosa
import numpy as np
import json
import sys
import os
import time
from functools import lru_cache


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "aasist/models/weights/AASIST.pth"
CONFIG_PATH = "aasist/config/AASIST.conf"

TARGET_LENGTH = 64600
TARGET_SR = 16000


# ============================================================
# CPU SAFETY
# ============================================================

# Limit PyTorch CPU usage so the laptop does not become
# unresponsive during AASIST inference.

try:
    torch.set_num_threads(2)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ============================================================
# LOAD AASIST SOURCE
# ============================================================

sys.path.insert(
    0,
    "aasist"
)

from models.AASIST import Model


# ============================================================
# LOAD MODEL ONLY ONCE
# ============================================================

@lru_cache(maxsize=1)
def load_model():

    print()
    print("=" * 60)
    print("Loading VoiceShield AASIST model...")
    print("=" * 60)

    start_time = time.time()


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(CONFIG_PATH):

        raise FileNotFoundError(
            f"AASIST configuration not found: {CONFIG_PATH}"
        )


    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"AASIST model not found: {MODEL_PATH}"
        )


    # --------------------------------------------------------
    # Read configuration
    # --------------------------------------------------------

    with open(
        CONFIG_PATH,
        "r"
    ) as f:

        config = json.load(f)


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = Model(
        config["model_config"]
    )


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )


    model.load_state_dict(
        checkpoint
    )


    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()


    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    model = model.to(
        torch.device("cpu")
    )


    elapsed = time.time() - start_time


    print(
        f"AASIST model loaded in {elapsed:.2f} seconds"
    )

    print(
        "Device: CPU"
    )

    print(
        "PyTorch CPU threads:",
        torch.get_num_threads()
    )

    print(
        "=" * 60
    )

    return model


# ============================================================
# PREPARE AUDIO
# ============================================================

def prepare_audio(audio_path):

    print()
    print("Preparing audio...")

    start_time = time.time()


    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )


    if len(audio) == 0:

        raise ValueError(
            "Audio file is empty."
        )


    print(
        "Sample rate:",
        sample_rate
    )

    print(
        "Original samples:",
        len(audio)
    )


    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    audio = audio.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = np.max(
        np.abs(audio)
    )


    if max_value > 0:

        audio = audio / max_value


    # --------------------------------------------------------
    # Pad short audio
    # --------------------------------------------------------

    if len(audio) < TARGET_LENGTH:

        repeats = int(
            np.ceil(
                TARGET_LENGTH / len(audio)
            )
        )

        audio = np.tile(
            audio,
            repeats
        )


    # --------------------------------------------------------
    # Crop to AASIST input length
    # --------------------------------------------------------

    audio = audio[
        :TARGET_LENGTH
    ]


    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)


    elapsed = time.time() - start_time


    print(
        f"Audio preparation completed in {elapsed:.2f} seconds"
    )

    print(
        "Final samples:",
        audio_tensor.shape[1]
    )


    return audio_tensor


# ============================================================
# AUDIO CHARACTERISTICS
# ============================================================

def get_audio_characteristics(
    audio_path
):

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )


    if len(audio) == 0:

        return {}


    duration = (
        len(audio) / sr
    )


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


    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                audio
            )
        )
    )


    spectral_centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=sr
            )
        )
    )


    return {

        "duration": duration,

        "sample_rate": sr,

        "rms": rms,

        "peak": peak,

        "zcr": zcr,

        "spectral_centroid": spectral_centroid
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_voice(
    audio_path
):

    if not os.path.exists(
        audio_path
    ):

        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )


    print()
    print("=" * 60)
    print("VoiceShield AI Analysis")
    print("=" * 60)


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Step 1/4: Loading AI model...")


    model = load_model()


    # ========================================================
    # PREPARE AUDIO
    # ========================================================

    print()
    print("Step 2/4: Preparing audio...")


    audio_tensor = prepare_audio(
        audio_path
    )


    # ========================================================
    # AASIST INFERENCE
    # ========================================================

    print()
    print("Step 3/4: Running AASIST inference...")
    print(
        "CPU threads:",
        torch.get_num_threads()
    )


    start_time = time.time()


    with torch.inference_mode():

        _, output = model(
            audio_tensor
        )


    inference_time = (
        time.time() - start_time
    )


    print(
        f"AASIST inference completed in "
        f"{inference_time:.2f} seconds"
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = torch.softmax(
        output,
        dim=1
    )


    class_0_probability = float(
        probabilities[0, 0].item()
    )


    class_1_probability = float(
        probabilities[0, 1].item()
    )


    predicted_class = int(
        torch.argmax(
            output,
            dim=1
        ).item()
    )


    # ========================================================
    # AUDIO INFORMATION
    # ========================================================

    audio_info = get_audio_characteristics(
        audio_path
    )


    # ========================================================
    # PROTOTYPE CLASS MAPPING
    # ========================================================

    if predicted_class == 1:

        result = (
            "SYNTHETIC / SPOOF"
        )

        confidence = (
            class_1_probability * 100
        )


        if confidence >= 80:

            risk = "HIGH"

        elif confidence >= 60:

            risk = "MEDIUM"

        else:

            risk = "LOW"


        explanation = (
            "The AASIST model detected acoustic "
            "patterns that are consistent with "
            "synthetic or spoofed speech."
        )


        action = (
            "REQUEST VERIFICATION / BLOCK"
        )


    else:

        result = (
            "REAL / BONAFIDE"
        )

        confidence = (
            class_0_probability * 100
        )


        if confidence >= 80:

            risk = "LOW"

        elif confidence >= 60:

            risk = "MEDIUM"

        else:

            risk = "HIGH"


        explanation = (
            "The AASIST model found the audio "
            "more consistent with bonafide speech. "
            "This result does not guarantee that "
            "the voice is human."
        )


        action = (
            "ALLOW / CONTINUE"
        )


    # ========================================================
    # RELIABILITY
    # ========================================================

    reliability = "STANDARD"


    reliability_message = (
        "The input is within the basic audio "
        "conditions used by VoiceShield."
    )


    # --------------------------------------------------------
    # Short recording
    # --------------------------------------------------------

    if audio_info.get(
        "duration",
        0
    ) < 3:

        reliability = "LIMITED"

        reliability_message = (
            "The recording is short. Detection "
            "reliability may be limited."
        )


    # --------------------------------------------------------
    # Low audio energy
    # --------------------------------------------------------

    elif audio_info.get(
        "rms",
        0
    ) < 0.015:

        reliability = "LIMITED"

        reliability_message = (
            "The recording has low audio energy. "
            "Detection reliability may be limited."
        )


    # --------------------------------------------------------
    # Very strong model score
    # --------------------------------------------------------

    if confidence >= 99:

        reliability = "CAUTION"

        reliability_message = (
            "The model produced a very strong "
            "classification score. This should "
            "still be interpreted as model evidence, "
            "not proof of authenticity."
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    result_data = {

        "result": result,

        "confidence": confidence,

        "risk": risk,

        "explanation": explanation,

        "action": action,

        "class_0_probability":
            class_0_probability,

        "class_1_probability":
            class_1_probability,

        "predicted_class":
            predicted_class,

        "reliability":
            reliability,

        "reliability_message":
            reliability_message,

        "audio_info":
            audio_info,

        "inference_time":
            inference_time
    }


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("DETECTION RESULT")
    print("=" * 60)

    print(
        "Result:",
        result
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    print(
        "Risk:",
        risk
    )

    print(
        "Reliability:",
        reliability
    )

    print(
        "Class 0:",
        f"{class_0_probability * 100:.2f}%"
    )

    print(
        "Class 1:",
        f"{class_1_probability * 100:.2f}%"
    )

    print(
        "Predicted class:",
        predicted_class
    )

    print(
        "Inference time:",
        f"{inference_time:.2f} seconds"
    )

    print(
        "=" * 60
    )


    return result_data


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("VoiceShield AASIST Detection Test")
    print("=" * 60)


    test_file = "fake_voice.wav"


    result = analyze_voice(
        test_file
    )


    print()
    print(
        "Final result:",
        result["result"]
    )