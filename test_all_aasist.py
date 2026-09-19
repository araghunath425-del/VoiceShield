from voiceshield_detector import analyze_voice

files = [
    "human_recorded.wav",
    "chatgpt_recorded.wav",
    "gemini_recorded.wav",
    "fake_voice.wav"
]

print("\n========================================")
print("       VOICESHIELD AASIST TEST")
print("========================================")

for file in files:
    print("\n\nFILE:", file)
    print("----------------------------------------")

    result = analyze_voice(file)

    print("Result:", result["result"])
    print("Confidence:", round(result["confidence"], 2), "%")
    print("Risk:", result["risk"])
    print("Class 0:", round(result["class_0_probability"] * 100, 2), "%")
    print("Class 1:", round(result["class_1_probability"] * 100, 2), "%")