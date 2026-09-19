import streamlit as st
import random
import tempfile
import os
import librosa
import numpy as np

from voiceshield_detector import analyze_voice


# =========================================================
# VOICESHIELD - VOICE VERIFICATION
# =========================================================

st.set_page_config(
    page_title="Verification - VoiceShield",
    page_icon="🔐",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "verification_code" not in st.session_state:
    st.session_state.verification_code = None

if "verification_generated" not in st.session_state:
    st.session_state.verification_generated = False

if "verification_audio" not in st.session_state:
    st.session_state.verification_audio = None

if "verification_result" not in st.session_state:
    st.session_state.verification_result = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🛡️ VoiceShield")

    st.write(
        "AI Voice Security System"
    )

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

    st.success(
        "AI Model: Online"
    )

    st.info(
        "Inference: CPU"
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🔐 Voice Verification"
)

st.write(
    "Challenge-response verification provides an additional "
    "security layer for live voice recordings."
)

st.divider()


# =========================================================
# CHECK ANALYSIS RESULT
# =========================================================

result = st.session_state.get(
    "analysis_result"
)


if result is None:

    st.warning(
        "⚠️ No voice analysis result is available."
    )

    st.info(
        "Please record or upload audio and complete "
        "AI Detection first."
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
# CURRENT ANALYSIS
# =========================================================

st.header(
    "🧠 Current Voice Analysis"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Detection",
        result["result"]
    )


with col2:

    st.metric(
        "Confidence",
        f'{result["confidence"]:.2f}%'
    )


with col3:

    st.metric(
        "Risk",
        result["risk"]
    )


st.divider()


# =========================================================
# DETERMINE AUDIO SOURCE
# =========================================================

audio_source = st.session_state.get(
    "audio_source",
    ("unknown", "Audio")
)

source_type = audio_source[0]

source_name = audio_source[1]


# =========================================================
# LIVE RECORDING ONLY
# =========================================================

if source_type == "recording":

    st.header(
        "🎯 Challenge-Response Verification"
    )

    st.write(
        "This audio was recorded live. A fresh challenge "
        "can therefore be given to the speaker."
    )


    # =====================================================
    # GENERATE CHALLENGE
    # =====================================================

    if st.button(
        "🎲 Generate New Challenge",
        type="primary",
        use_container_width=True
    ):

        st.session_state.verification_code = (
            random.randint(1000, 9999)
        )

        st.session_state.verification_generated = True

        st.session_state.verification_audio = None

        st.session_state.verification_result = None


    # =====================================================
    # SHOW CHALLENGE
    # =====================================================

    if st.session_state.verification_generated:

        code = (
            st.session_state.verification_code
        )

        phrase = (
            f"My VoiceShield verification code is {code}"
        )


        st.success(
            "🔐 New verification challenge generated."
        )


        st.subheader(
            "📢 Ask the speaker to say:"
        )


        st.markdown(
            f"### “{phrase}”"
        )


        st.caption(
            "The phrase is intentionally different from "
            "the original recording."
        )


        st.divider()


        # =================================================
        # RECORD CHALLENGE RESPONSE
        # =================================================

        st.header(
            "🎙️ Record Challenge Response"
        )

        st.write(
            "Ask the speaker to read the displayed phrase "
            "clearly into the microphone."
        )


        verification_audio = st.audio_input(
            "🎙️ Record Response",
            key=f"verification_recorder_{code}"
        )


        # =================================================
        # RESPONSE RECEIVED
        # =================================================

        if verification_audio is not None:

            st.success(
                "✅ Verification response recorded."
            )


            st.audio(
                verification_audio
            )


            st.session_state.verification_audio = (
                verification_audio.getvalue()
            )


            st.divider()


            # =================================================
            # ANALYZE VERIFICATION RESPONSE
            # =================================================

            if st.button(
                "🔍 ANALYZE VERIFICATION RESPONSE",
                type="primary",
                use_container_width=True
            ):

                temp_audio_path = None

                try:

                    # -----------------------------------------
                    # CREATE TEMPORARY FILE
                    # -----------------------------------------

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav"
                    ) as temp_file:

                        temp_file.write(
                            verification_audio.getvalue()
                        )

                        temp_audio_path = (
                            temp_file.name
                        )


                    # -----------------------------------------
                    # CHECK MICROPHONE SIGNAL
                    # -----------------------------------------

                    check_audio, check_sr = (
                        librosa.load(
                            temp_audio_path,
                            sr=16000,
                            mono=True
                        )
                    )


                    audio_max = float(
                        np.max(
                            np.abs(check_audio)
                        )
                    )


                    audio_std = float(
                        np.std(check_audio)
                    )


                    if (
                        audio_max == 0
                        or audio_std == 0
                    ):

                        st.error(
                            "❌ No microphone signal detected. "
                            "Please record the challenge again."
                        )

                        st.stop()


                    # -----------------------------------------
                    # RUN AASIST
                    # -----------------------------------------

                    with st.spinner(
                        "🧠 VoiceShield AI is analyzing "
                        "the verification response..."
                    ):

                        verification_result = (
                            analyze_voice(
                                temp_audio_path
                            )
                        )


                    # Save result
                    st.session_state.verification_result = (
                        verification_result
                    )


                    # =================================================
                    # VERIFICATION RESULT
                    # =================================================

                    st.divider()

                    st.header(
                        "🛡️ Verification Analysis"
                    )


                    vcol1, vcol2, vcol3 = (
                        st.columns(3)
                    )


                    with vcol1:

                        st.metric(
                            "Voice Result",
                            verification_result[
                                "result"
                            ]
                        )


                    with vcol2:

                        st.metric(
                            "Confidence",
                            f'{verification_result["confidence"]:.2f}%'
                        )


                    with vcol3:

                        st.metric(
                            "Risk",
                            verification_result[
                                "risk"
                            ]
                        )


                    # =================================================
                    # VERIFICATION STATUS
                    # =================================================

                    if (
                        verification_result[
                            "result"
                        ]
                        == "SYNTHETIC / SPOOF"
                    ):

                        st.error(
                            "🚨 The verification response "
                            "shows potential synthetic or "
                            "spoof characteristics."
                        )

                    else:

                        st.success(
                            "✅ The verification response "
                            "is consistent with natural speech."
                        )


                    st.info(
                        "This checks the new response for "
                        "synthetic/spoof characteristics. "
                        "It does not by itself prove the "
                        "speaker's identity."
                    )


                except Exception as e:

                    st.error(
                        "❌ Error while analyzing "
                        "verification response."
                    )

                    st.exception(
                        e
                    )


                finally:

                    # -----------------------------------------
                    # DELETE TEMPORARY FILE
                    # -----------------------------------------

                    if (
                        temp_audio_path is not None
                        and os.path.exists(
                            temp_audio_path
                        )
                    ):

                        os.remove(
                            temp_audio_path
                        )


# =========================================================
# UPLOADED AUDIO
# =========================================================

else:

    st.header(
        "📁 Uploaded Audio"
    )


    st.info(
        "This audio was uploaded as an existing recording."
    )


    st.write(
        "Challenge-response verification is available "
        "only for live microphone recordings because "
        "a person must respond to a newly generated phrase."
    )


    st.warning(
        "🎙️ To perform challenge-response verification, "
        "go back to Voice Input and make a new live recording."
    )


# =========================================================
# EXISTING VERIFICATION RESULT
# =========================================================

if (
    st.session_state.verification_result
    is not None
):

    verification_result = (
        st.session_state.verification_result
    )


    st.divider()

    st.header(
        "📋 Verification Summary"
    )


    summary_col1, summary_col2, summary_col3 = (
        st.columns(3)
    )


    with summary_col1:

        st.metric(
            "Result",
            verification_result[
                "result"
            ]
        )


    with summary_col2:

        st.metric(
            "Confidence",
            f'{verification_result["confidence"]:.2f}%'
        )


    with summary_col3:

        st.metric(
            "Risk",
            verification_result[
                "risk"
            ]
        )


# =========================================================
# NAVIGATION
# =========================================================

st.divider()

st.header(
    "➡️ Continue"
)


nav_col1, nav_col2 = st.columns(2)


with nav_col1:

    if st.button(
        "📊 Back to Audio Analysis",
        use_container_width=True
    ):

        st.switch_page(
            "pages/3_📊_Audio_Analysis.py"
        )


with nav_col2:

    if st.button(
        "🚦 Security Decision →",
        type="primary",
        use_container_width=True
    ):

        st.switch_page(
            "pages/5_🚦_Security_Decision.py"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "VoiceShield — AI-powered synthetic voice detection "
    "hackathon prototype."
)