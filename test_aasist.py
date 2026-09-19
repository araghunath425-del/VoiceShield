import torch
import soundfile as sf
import numpy as np

from aasist.models.AASIST import Model


# AASIST configuration
config = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}


print("Loading VoiceShield AI...")

model = Model(config)

checkpoint = torch.load(
    "aasist/models/weights/AASIST.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint)
model.eval()

print("AI model loaded successfully!")


# Load real voice recording
audio, sample_rate = sf.read("human_test2.wav")

print("Audio loaded!")
print("Sample rate:", sample_rate)
print("Original samples:", len(audio))


# Convert stereo to mono if necessary
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)


# AASIST expects 64600 samples
max_len = 64600

if len(audio) >= max_len:
    audio = audio[:max_len]
else:
    repeats = int(max_len / len(audio)) + 1
    audio = np.tile(audio, repeats)[:max_len]


audio_tensor = torch.tensor(
    audio,
    dtype=torch.float32
).unsqueeze(0)


print("Analyzing voice...")
 
with torch.no_grad():
    _, output = model(audio_tensor)


# AASIST output
logit_0 = output[0, 0].item()
logit_1 = output[0, 1].item()

# Convert logits to probabilities
probabilities = torch.softmax(output, dim=1)

prob_0 = probabilities[0, 0].item()
prob_1 = probabilities[0, 1].item()

predicted_class = torch.argmax(output, dim=1).item()

print()
print("================================")
print("       VOICESHIELD RESULT")
print("================================")

print("Class 0 logit:", logit_0)
print("Class 1 logit:", logit_1)

print()
print("Class 0 probability:", round(prob_0 * 100, 2), "%")
print("Class 1 probability:", round(prob_1 * 100, 2), "%")

print()
print("Predicted class:", predicted_class)

print("================================")