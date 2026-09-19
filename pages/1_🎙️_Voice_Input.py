import streamlit as st
import os
import tempfile


# =========================================================
# VOICESHIELD - VOICE INPUT
# =========================================================

st.set_page_config(
    page_title="Voice Input - VoiceShield",
    page_icon="🎙️",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

if "audio_source" not in st.session_state:
    st.session_state.audio_source = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🛡️ VoiceShield")

    st.write("AI Voice Security System")

    st.divider()

    st.subheader("Navigation")

    st.page_link(
        "app.py",
        label="🏠 Home"
    )

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

st.title("🎙️ Voice Input")

st.write(
    "Provide a voice recording for VoiceShield to analyze."
)

st.divider()


# =========================================================
# UPLOAD AUDIO
# =========================================================

st.header("📁 Upload Voice")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a"
    ],
    help="Supported formats: WAV, MP3, OGG, FLAC and M4A"
)


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    st.audio(uploaded_file)

    if st.button(
        "✅ Use This Audio",
        type="primary",
        use_container_width=True
    ):

        st.session_state.audio_data = uploaded_file.getvalue()

        st.session_state.audio_source = (
            "upload",
            uploaded_file.name
        )

        st.session_state.analysis_result = None

        st.success(
            "Audio selected successfully!"
        )

        st.switch_page(
            "pages/2_🧠_AI_Detection.py"
        )


st.divider()


# =========================================================
# RECORD AUDIO
# =========================================================

st.header("🎙️ Record Voice")

st.write(
    "Record a voice sample using your microphone."
)


# Create a changing key to allow recording reset
if "recorder_version" not in st.session_state:

    st.session_state.recorder_version = 0


recorded_audio = st.audio_input(
    "🎙️ Record Voice",
    key=f"voice_recorder_{st.session_state.recorder_version}"
)


# =========================================================
# RECORDING RECEIVED
# =========================================================

if recorded_audio is not None:

    st.success(
        "Recording received successfully!"
    )

    st.audio(recorded_audio)


    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # USE RECORDING
    # -----------------------------------------------------

    with col1:

        if st.button(
            "✅ Use This Recording",
            type="primary",
            use_container_width=True
        ):

            st.session_state.audio_data = (
                recorded_audio.getvalue()
            )

            st.session_state.audio_source = (
                "recording",
                "Recorded Voice"
            )

            st.session_state.analysis_result = None

            st.success(
                "Recording selected successfully!"
            )

            st.switch_page(
                "pages/2_🧠_AI_Detection.py"
            )


    # -----------------------------------------------------
    # REMOVE RECORDING
    # -----------------------------------------------------

    with col2:

        if st.button(
            "🗑️ Remove Recording",
            use_container_width=True
        ):

            st.session_state.recorder_version += 1

            st.session_state.audio_data = None

            st.session_state.audio_source = None

            st.session_state.analysis_result = None

            st.rerun()


# =========================================================
# INFORMATION
# =========================================================

st.divider()

st.info(
    "💡 You can either upload an existing audio file "
    "or record a new voice sample. After selecting the "
    "audio, VoiceShield will move to the AI Detection page."
)