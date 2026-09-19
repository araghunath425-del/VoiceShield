import streamlit as st


# =========================================================
# VOICESHIELD - SECURITY DECISION
# =========================================================

st.set_page_config(
    page_title="Security Decision - VoiceShield",
    page_icon="🚦",
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

st.title("🚦 Security Decision")

st.write(
    "Final security assessment based on the current "
    "VoiceShield analysis."
)

st.divider()


# =========================================================
# CHECK ANALYSIS
# =========================================================

result = st.session_state.get(
    "analysis_result"
)


if result is None:

    st.warning(
        "⚠️ No voice analysis result is available."
    )

    st.info(
        "Please analyze a voice recording before viewing "
        "the security decision."
    )

    if st.button(
        "🧠 Go to AI Detection",
        type="primary",
        use_container_width=True
    ):

        st.switch_page(
            "pages/2_🧠_AI_Detection.py"
        )

    st.stop()


# =========================================================
# ANALYSIS SUMMARY
# =========================================================

st.header("🧠 Analysis Summary")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Detection Result",
        result["result"]
    )


with col2:

    st.metric(
        "Confidence",
        f'{result["confidence"]:.2f}%'
    )


with col3:

    st.metric(
        "Risk Level",
        result["risk"]
    )


st.divider()


# =========================================================
# SECURITY DECISION
# =========================================================

st.header("🚦 Final Security Decision")


if result["result"] == "SYNTHETIC / SPOOF":

    # -----------------------------------------------------
    # SYNTHETIC / SPOOF
    # -----------------------------------------------------

    st.error(
        "🚨 POTENTIAL SYNTHETIC / SPOOFED VOICE"
    )

    st.subheader(
        "🔴 Recommended Action"
    )

    st.markdown(
        """
        ### REQUEST VERIFICATION / BLOCK

        The current AI analysis indicates that the audio
        contains characteristics associated with
        synthetic or spoofed speech.

        **Recommended security response:**

        1. Request challenge-response verification.
        2. Do not rely on the voice alone for sensitive actions.
        3. Block or hold the request when verification is unsuccessful.
        """
    )


elif result["risk"] == "MEDIUM":

    # -----------------------------------------------------
    # MEDIUM RISK
    # -----------------------------------------------------

    st.warning(
        "⚠️ MEDIUM RISK VOICE"
    )

    st.subheader(
        "🟡 Recommended Action"
    )

    st.markdown(
        """
        ### REQUEST ADDITIONAL VERIFICATION

        The current analysis does not provide a strong
        synthetic-voice indication, but additional
        verification is recommended before sensitive
        actions.
        """
    )


else:

    # -----------------------------------------------------
    # LOW RISK
    # -----------------------------------------------------

    st.success(
        "🟢 VOICE CONSISTENT WITH NATURAL SPEECH"
    )

    st.subheader(
        "🟢 Recommended Action"
    )

    st.markdown(
        """
        ### ALLOW / CONTINUE

        The current analysis did not produce a strong
        synthetic-voice indication.

        The result should still be treated as an
        automated screening result rather than proof
        of speaker identity.
        """
    )


st.divider()


# =========================================================
# MODEL DETAILS
# =========================================================

st.header("📊 Model Evidence")

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Class 0",
        f'{result["class_0_probability"] * 100:.2f}%'
    )


with col2:

    st.metric(
        "Class 1",
        f'{result["class_1_probability"] * 100:.2f}%'
    )


st.caption(
    "Prototype mapping validated using the current test samples: "
    "Class 0 → Real/Bonafide | Class 1 → Synthetic/Spoof."
)


# =========================================================
# EXPLANATION
# =========================================================

st.subheader("🧠 AI Explanation")

st.info(
    result["explanation"]
)


# =========================================================
# SECURITY WORKFLOW
# =========================================================

st.divider()

st.header("🔐 VoiceShield Response Workflow")

flow1, flow2, flow3, flow4 = st.columns(4)


with flow1:

    st.markdown(
        """
        ### 1️⃣
        **Detect**

        Analyze the voice
        with AASIST.
        """
    )


with flow2:

    st.markdown(
        """
        ### 2️⃣
        **Assess**

        Determine the
        risk level.
        """
    )


with flow3:

    st.markdown(
        """
        ### 3️⃣
        **Verify**

        Request a fresh
        challenge response.
        """
    )


with flow4:

    st.markdown(
        """
        ### 4️⃣
        **Decide**

        Allow, verify,
        or hold the request.
        """
    )


# =========================================================
# NAVIGATION
# =========================================================

st.divider()

st.header("➡️ Navigation")

col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "🎙️ New Voice Analysis",
        use_container_width=True
    ):

        # Clear previous analysis
        st.session_state.audio_data = None
        st.session_state.audio_source = None
        st.session_state.analysis_result = None

        st.switch_page(
            "pages/1_🎙️_Voice_Input.py"
        )


with col2:

    if st.button(
        "🔐 Verification",
        use_container_width=True
    ):

        st.switch_page(
            "pages/4_🔐_Verification.py"
        )


with col3:

    if st.button(
        "🏠 Home",
        use_container_width=True
    ):

        st.switch_page(
            "app.py"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "VoiceShield — AI-powered synthetic voice detection "
    "hackathon prototype."
)