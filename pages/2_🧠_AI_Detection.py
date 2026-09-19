import streamlit as st
import tempfile
import os

from voiceshield_detector import analyze_voice


# =========================================================
# VOICESHIELD - AI DETECTION
# =========================================================

st.set_page_config(
    page_title="AI Detection - VoiceShield",
    page_icon="🧠",
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

st.title("🧠 AI Voice Detection")

st.write(
    "VoiceShield uses the AASIST anti-spoofing model "
    "to analyze the selected audio."
)

st.divider()


# =========================================================
# CHECK AUDIO
# =========================================================

if st.session_state.audio_data is None:

    st.warning(
        "⚠️ No audio selected."
    )

    st.info(
        "Go to Voice Input and record or upload an audio file."
    )

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

source_type, source_name = st.session_state.audio_source

st.subheader("🎵 Selected Audio")

st.write(
    f"**Source:** {source_type.capitalize()}"
)

st.write(
    f"**File:** {source_name}"
)

st.audio(
    st.session_state.audio_data
)

st.divider()


# =========================================================
# ANALYZE
# =========================================================

st.subheader("🔍 Run AI Analysis")

if st.button(
    "🧠 ANALYZE WITH AASIST",
    type="primary",
    use_container_width=True
):

    temp_audio_path = None

    try:

        # -------------------------------------------------
        # Create temporary audio file
        # -------------------------------------------------

        extension = os.path.splitext(
            source_name
        )[1]

        if extension == "":
            extension = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(
                st.session_state.audio_data
            )

            temp_audio_path = temp_file.name


        # -------------------------------------------------
        # AI ANALYSIS
        # -------------------------------------------------

        with st.spinner(
            "🧠 VoiceShield AI is analyzing..."
        ):

            result = analyze_voice(
                temp_audio_path
            )


        # Save result
        st.session_state.analysis_result = result


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.header(
            "🛡️ Detection Result"
        )


        result_col, confidence_col, risk_col = st.columns(3)


        with result_col:

            st.metric(
                "Detection",
                result["result"]
            )


        with confidence_col:

            st.metric(
                "Confidence",
                f'{result["confidence"]:.2f}%'
            )


        with risk_col:

            st.metric(
                "Risk Level",
                result["risk"]
            )


        # -------------------------------------------------
        # SECURITY STATUS
        # -------------------------------------------------

        st.subheader(
            "🔐 Security Status"
        )

        if result["result"] == "SYNTHETIC / SPOOF":

            st.error(
                "⚠️ POTENTIAL SYNTHETIC / SPOOFED "
                "VOICE DETECTED"
            )

        else:

            st.success(
                "✅ VOICE CONSISTENT WITH "
                "NATURAL SPEECH"
            )


        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        st.subheader(
            "🧠 AI Explanation"
        )

        st.info(
            result["explanation"]
        )


        # -------------------------------------------------
        # ACTION
        # -------------------------------------------------

        st.subheader(
            "🚦 Recommended Action"
        )

        if "BLOCK" in result["action"]:

            st.error(
                f'🚫 {result["action"]}'
            )

        else:

            st.success(
                f'✅ {result["action"]}'
            )


        # -------------------------------------------------
        # MODEL PROBABILITY
        # -------------------------------------------------

        st.subheader(
            "📊 AASIST Model Probability"
        )

        probability_col1, probability_col2 = st.columns(2)


        with probability_col1:

            st.metric(
                "Class 0",
                f'{result["class_0_probability"] * 100:.2f}%'
            )


        with probability_col2:

            st.metric(
                "Class 1",
                f'{result["class_1_probability"] * 100:.2f}%'
            )


        st.caption(
            "Current validated prototype mapping: "
            "Class 0 → Real/Bonafide | "
            "Class 1 → Synthetic/Spoof."
        )


        # -------------------------------------------------
        # NEXT PAGE
        # -------------------------------------------------

        st.divider()

        if st.button(
            "📊 Continue to Audio Analysis →",
            type="primary",
            use_container_width=True
        ):

            st.switch_page(
                "pages/3_📊_Audio_Analysis.py"
            )


    except Exception as e:

        st.error(
            "❌ VoiceShield encountered an error "
            "while analyzing the audio."
        )

        st.exception(e)


    finally:

        if (
            temp_audio_path is not None
            and os.path.exists(temp_audio_path)
        ):

            os.remove(temp_audio_path)


# =========================================================
# EXISTING RESULT
# =========================================================

elif st.session_state.analysis_result is not None:

    result = st.session_state.analysis_result

    st.divider()

    st.header(
        "🛡️ Previous Analysis"
    )

    result_col, confidence_col, risk_col = st.columns(3)

    with result_col:

        st.metric(
            "Detection",
            result["result"]
        )

    with confidence_col:

        st.metric(
            "Confidence",
            f'{result["confidence"]:.2f}%'
        )

    with risk_col:

        st.metric(
            "Risk Level",
            result["risk"]
        )

    if st.button(
        "📊 Continue to Audio Analysis →",
        type="primary",
        use_container_width=True
    ):

        st.switch_page(
            "pages/3_📊_Audio_Analysis.py"
        )