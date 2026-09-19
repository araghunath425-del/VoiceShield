import streamlit as st
import tempfile
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# VOICESHIELD - AUDIO ANALYSIS
# =========================================================

st.set_page_config(
    page_title="Audio Analysis - VoiceShield",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🛡️ VoiceShield")
    st.write("AI Voice Security System")

    st.divider()

    st.subheader("Navigation")

    st.page_link("app.py", label="🏠 Home")

    st.page_link(
        "pages/1_🎙️_Voice_Input.py",
        label="🎙️ Voice Input"
    )

    st.page_link(
        "pages/2_🧠_AI_Detection.py",
        label="🧠 AI Detection"
    )

    st.page_link(
        "pages/3_📊_Audio_Analysis.py",
        label="📊 Audio Analysis"
    )

    st.page_link(
        "pages/4_🔐_Verification.py",
        label="🔐 Verification"
    )

    st.page_link(
        "pages/5_🚦_Security_Decision.py",
        label="🚦 Security Decision"
    )

    st.divider()

    st.subheader("System Status")

    st.success("AI Model: Online")
    st.info("Inference: CPU")


# =========================================================
# HEADER
# =========================================================

st.title("📊 Audio Analysis")

st.write(
    "Visual and acoustic analysis of the selected voice recording."
)

st.divider()


# =========================================================
# CHECK AUDIO
# =========================================================

if st.session_state.get("audio_data") is None:

    st.warning("⚠️ No audio is currently selected.")

    if st.button(
        "🎙️ Go to Voice Input",
        type="primary",
        use_container_width=True
    ):

        st.switch_page(
            "pages/1_🎙️_Voice_Input.py"
        )

    st.stop()


# =========================================================
# AUDIO INFORMATION
# =========================================================

audio_data = st.session_state.audio_data

audio_source = st.session_state.get(
    "audio_source",
    ("unknown", "Audio")
)

source_type, source_name = audio_source


st.header("🎵 Selected Audio")

st.write(
    f"**Source:** {source_type.capitalize()}"
)

st.write(
    f"**File:** {source_name}"
)

st.audio(audio_data)

st.divider()


# =========================================================
# CREATE TEMPORARY FILE
# =========================================================

temp_audio_path = None

try:

    extension = os.path.splitext(
        source_name
    )[1]

    if extension == "":
        extension = ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp_file:

        temp_file.write(audio_data)

        temp_audio_path = temp_file.name


    # =====================================================
    # LOAD AUDIO
    # =====================================================

    audio, sample_rate = librosa.load(
        temp_audio_path,
        sr=16000,
        mono=True
    )


    # =====================================================
    # BASIC AUDIO INFORMATION
    # =====================================================

    duration = len(audio) / sample_rate

    maximum_amplitude = float(
        np.max(np.abs(audio))
    )

    rms_energy = float(
        np.sqrt(np.mean(audio ** 2))
    )


    st.header("📋 Audio Information")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Duration",
            f"{duration:.2f} sec"
        )


    with col2:

        st.metric(
            "Sample Rate",
            f"{sample_rate} Hz"
        )


    with col3:

        st.metric(
            "Samples",
            f"{len(audio):,}"
        )


    with col4:

        st.metric(
            "RMS Energy",
            f"{rms_energy:.4f}"
        )


    st.divider()


    # =====================================================
    # WAVEFORM
    # =====================================================

    st.header("📈 Voice Waveform")

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    librosa.display.waveshow(
        audio,
        sr=sample_rate,
        ax=ax
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_title(
        "Audio Waveform"
    )

    st.pyplot(fig)

    plt.close(fig)


    st.divider()


    # =====================================================
    # MEL-SPECTROGRAM
    # =====================================================

    st.header("🎵 Mel-Spectrogram")

    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128
    )

    spectrogram_db = librosa.power_to_db(
        spectrogram,
        ref=np.max
    )


    fig2, ax2 = plt.subplots(
        figsize=(12, 5)
    )

    img = librosa.display.specshow(
        spectrogram_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        ax=ax2
    )

    ax2.set_title(
        "Mel-Spectrogram"
    )

    fig2.colorbar(
        img,
        ax=ax2,
        format="%+2.0f dB"
    )

    st.pyplot(fig2)

    plt.close(fig2)


    st.divider()


    # =====================================================
    # NEXT STEPS
    # =====================================================

    st.header("➡️ Continue")

    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "🧠 Back to AI Detection",
            use_container_width=True
        ):

            st.switch_page(
                "pages/2_🧠_AI_Detection.py"
            )


    with col2:

        if st.button(
            "🔐 Continue to Verification →",
            type="primary",
            use_container_width=True
        ):

            st.switch_page(
                "pages/4_🔐_Verification.py"
            )


except Exception as e:

    st.error(
        "❌ Unable to analyze the audio."
    )

    st.exception(e)


finally:

    if (
        temp_audio_path is not None
        and os.path.exists(temp_audio_path)
    ):

        os.remove(temp_audio_path)