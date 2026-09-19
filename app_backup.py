import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from voiceshield_detector import analyze_voice


# =========================================================
# VOICESHIELD - AI VOICE SECURITY SYSTEM
# =========================================================

st.set_page_config(
    page_title="VoiceShield",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #666666;
        margin-top: 0px;
    }

    .security-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 10px;
        border: 1px solid #dddddd;
    }

    .small-text {
        color: #777777;
        font-size: 14px;
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

    st.write(
        "AI Voice Security System"
    )

    st.divider()

    st.subheader("Detection Pipeline")

    st.write("🎙️ 1. Voice Input")
    st.write("↓")
    st.write("🤖 2. AASIST AI")
    st.write("↓")
    st.write("📊 3. Risk Analysis")
    st.write("↓")
    st.write("🔐 4. Verification")
    st.write("↓")
    st.write("🚦 5. Security Decision")

    st.divider()

    st.subheader("System Status")

    st.success("AI Model: Online")
    st.info("Inference: CPU")

    st.divider()

    st.caption(
        "VoiceShield Hackathon Prototype"
    )


# =========================================================
# HEADER
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
    "Detect potentially synthetic or spoofed speech "
    "using an AI-based speech analysis system."
)

st.divider()


# =========================================================
# VOICE INPUT
# =========================================================

st.header("🎙️ Voice Analysis")

st.write(
    "Choose one of the following methods to provide audio."
)


# =========================================================
# UPLOAD AUDIO
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a voice recording",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a"
    ],
    help="Supported audio formats: WAV, MP3, OGG, FLAC and M4A"
)


# =========================================================
# RECORD AUDIO
# =========================================================
st.write("### Or record your voice")

# Create a changing key so the recorder can be reset
if "recorder_version" not in st.session_state:
    st.session_state.recorder_version = 0

recorded_audio = st.audio_input(
    "🎙️ Record Voice",
    key=f"voice_recorder_{st.session_state.recorder_version}"
)

# Reset / remove recording
if recorded_audio is not None:

    reset_col1, reset_col2 = st.columns(2)

    with reset_col1:
        if st.button(
            "🗑️ Remove Recording",
            use_container_width=True
        ):
            st.session_state.recorder_version += 1
            st.rerun()

    with reset_col2:
        st.info("📁 You can also upload another audio file above.")


# =========================================================
# RECORDING DEBUG INFORMATION
# =========================================================

if recorded_audio is not None:

    st.write(
        "Recording received:",
        recorded_audio.size,
        "bytes"
    )

    st.audio(
        recorded_audio
    )


# =========================================================
# SELECT INPUT
# =========================================================

audio_file = None

if recorded_audio is not None:

    audio_file = recorded_audio

elif uploaded_file is not None:

    audio_file = uploaded_file


# =========================================================
# AUDIO SELECTED
# =========================================================

if audio_file is not None:

    st.success(
        f"Audio ready: {audio_file.name}"
    )

    st.audio(
        audio_file
    )

    st.divider()


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    analyze_button = st.button(
        "🔍 ANALYZE VOICE",
        type="primary",
        use_container_width=True
    )


    if analyze_button:

        # =================================================
        # TEMPORARY AUDIO FILE
        # =================================================

        file_extension = os.path.splitext(
            audio_file.name
        )[1]

        if file_extension == "":
            file_extension = ".wav"


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                audio_file.getbuffer()
            )

            temp_audio_path = temp_file.name


        try:

            # =============================================
            # CHECK AUDIO SIGNAL
            # =============================================

            check_audio, check_sr = librosa.load(
                temp_audio_path,
                sr=16000,
                mono=True
            )

            audio_std = float(np.std(check_audio))
            audio_max = float(np.max(np.abs(check_audio)))

            if audio_std == 0 or audio_max == 0:

                st.error(
                    "❌ The recorded audio contains no microphone signal. "
                    "Please check your browser and Windows microphone permissions "
                    "and record again."
                )

                st.stop()


            # =============================================
            # AI ANALYSIS
            # =============================================

            with st.spinner(
                "🧠 VoiceShield AI is analyzing the recording..."
            ):

                result = analyze_voice(
                    temp_audio_path
                )


            # =============================================
            # RESULT HEADER
            # =============================================

            st.divider()

            st.header(
                "🛡️ VoiceShield Analysis"
            )


            # =============================================
            # RESULT CARDS
            # =============================================

            result_col, confidence_col, risk_col = st.columns(3)


            with result_col:

                st.metric(
                    "Detection Result",
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


            # =============================================
            # SECURITY STATUS
            # =============================================

            st.subheader(
                "🔐 Security Status"
            )


            if result["result"] == "SYNTHETIC / SPOOF":

                st.error(
                    "⚠️ POTENTIAL SYNTHETIC / SPOOFED VOICE DETECTED"
                )

            else:

                st.success(
                    "✅ VOICE CONSISTENT WITH NATURAL SPEECH"
                )


            # =============================================
            # AI EXPLANATION
            # =============================================

            st.subheader(
                "🧠 AI Analysis"
            )

            st.info(
                result["explanation"]
            )


            # =============================================
            # RECOMMENDED ACTION
            # =============================================

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


            # =============================================
            # MODEL PROBABILITY
            # =============================================

            st.subheader(
                "📊 Model Probability"
            )

            probability_col1, probability_col2 = st.columns(2)


            with probability_col1:

                st.metric(
                    "Model Class 0",
                    f'{result["class_0_probability"] * 100:.2f}%'
                )


            with probability_col2:

                st.metric(
                    "Model Class 1",
                    f'{result["class_1_probability"] * 100:.2f}%'
                )


            st.caption(
                "Class numbering is shown directly from the current AASIST model output."
            )


            # =============================================
            # PROBABILITY BAR
            # =============================================

            st.write(
                "Model confidence distribution"
            )


            probability_data = {

                "Class 0": result[
                    "class_0_probability"
                ],

                "Class 1": result[
                    "class_1_probability"
                ]

            }


            st.bar_chart(
                probability_data
            )


            # =============================================
            # WAVEFORM
            # =============================================

            st.divider()

            st.subheader(
                "📈 Voice Waveform"
            )


            fig, ax = plt.subplots(
                figsize=(12, 3)
            )


            librosa.display.waveshow(
                check_audio,
                sr=check_sr,
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


            st.pyplot(
                fig
            )

            plt.close(
                fig
            )


            # =============================================
            # MEL SPECTROGRAM
            # =============================================

            st.subheader(
                "🎵 Mel-Spectrogram"
            )


            spectrogram = librosa.feature.melspectrogram(
                y=check_audio,
                sr=check_sr,
                n_mels=128
            )


            spectrogram_db = librosa.power_to_db(
                spectrogram,
                ref=np.max
            )


            fig2, ax2 = plt.subplots(
                figsize=(12, 4)
            )


            img = librosa.display.specshow(
                spectrogram_db,
                sr=check_sr,
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


            st.pyplot(
                fig2
            )

            plt.close(
                fig2
            )


            # =============================================
            # VOICE VERIFICATION
            # =============================================

            st.divider()

            st.header(
                "🔐 Voice Verification"
            )


            if result["result"] == "SYNTHETIC / SPOOF":

                st.warning(
                    "Additional verification is recommended "
                    "before accepting the voice request."
                )


                st.write(
                    "Generate a fresh challenge phrase and "
                    "ask the speaker to respond to it."
                )


                verification_phrase = st.text_input(
                    "Verification phrase",
                    placeholder=(
                        "Example: My VoiceShield verification code is 4827"
                    )
                )


                if st.button(
                    "🎯 Generate Verification Challenge",
                    key="verification_button"
                ):

                    if verification_phrase.strip():

                        st.success(
                            "Verification challenge generated."
                        )

                        st.info(
                            verification_phrase
                        )

                    else:

                        st.warning(
                            "Please enter a verification phrase first."
                        )


            else:

                st.success(
                    "The current analysis did not produce "
                    "a synthetic-voice indication."
                )


            # =============================================
            # FINAL DECISION
            # =============================================

            st.divider()

            st.header(
                "🚦 Security Decision"
            )


            if result["risk"] == "HIGH":

                st.error(
                    "🚫 HIGH RISK — REQUEST VERIFICATION / BLOCK"
                )


            elif result["risk"] == "MEDIUM":

                st.warning(
                    "⚠️ MEDIUM RISK — REQUEST ADDITIONAL VERIFICATION"
                )


            else:

                st.success(
                    "✅ LOW RISK — ALLOW / CONTINUE"
                )


        except Exception as e:

            st.error(
                "VoiceShield encountered an error "
                "while analyzing the audio."
            )

            st.exception(
                e
            )


        finally:

            # =============================================
            # DELETE TEMPORARY FILE
            # =============================================

            if os.path.exists(
                temp_audio_path
            ):

                os.remove(
                    temp_audio_path
                )


# =========================================================
# NO AUDIO
# =========================================================

else:

    st.info(
        "👆 Upload a voice recording or use the microphone "
        "to begin analysis."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "VoiceShield — AI-powered synthetic voice detection "
    "hackathon prototype."
)