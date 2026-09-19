import streamlit as st


# =========================================================
# VOICESHIELD - MAIN APPLICATION
# =========================================================

st.set_page_config(
    page_title="VoiceShield",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "audio_source" not in st.session_state:
    st.session_state.audio_source = None

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 21px;
        color: #666666;
        margin-bottom: 25px;
    }

    .feature-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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

    st.divider()

    st.caption(
        "VoiceShield Hackathon Prototype"
    )


# =========================================================
# HOME PAGE
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ VoiceShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Voice Clone & Synthetic Speech Detection'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "VoiceShield is an AI-based voice security system "
    "designed to analyze audio recordings and identify "
    "potentially synthetic or spoofed speech."
)

st.divider()


# =========================================================
# WELCOME
# =========================================================

st.header("🚀 Welcome to VoiceShield")

st.write(
    "Use the navigation menu on the left to move through "
    "the voice security analysis process."
)


# =========================================================
# PIPELINE
# =========================================================

st.subheader("🔍 Detection Pipeline")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        ### 🎙️
        **Voice Input**

        Record or upload audio.
        """
    )

with col2:
    st.markdown(
        """
        ### 🧠
        **AI Detection**

        Analyze using AASIST.
        """
    )

with col3:
    st.markdown(
        """
        ### 📊
        **Audio Analysis**

        Inspect waveform and
        spectrogram.
        """
    )

with col4:
    st.markdown(
        """
        ### 🔐
        **Verification**

        Perform additional
        voice verification.
        """
    )

with col5:
    st.markdown(
        """
        ### 🚦
        **Security Decision**

        Determine the recommended
        security action.
        """
    )


st.divider()


# =========================================================
# START BUTTON
# =========================================================

st.subheader("🎯 Start Analysis")

if st.button(
    "🎙️ Go to Voice Input",
    type="primary",
    use_container_width=True
):

    st.switch_page(
        "pages/1_🎙️_Voice_Input.py"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "VoiceShield — AI-powered synthetic voice detection "
    "hackathon prototype."
)