import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# VoiceShield - Acoustic Feature SVM Trainer
# ============================================================

DATASET_FILE = "voiceshield_features.csv"
MODEL_FILE = "acoustic_svm_model.pkl"


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print()
print("=" * 60)
print("VoiceShield Acoustic SVM Training")
print("=" * 60)
print()

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")
print("Samples:", len(df))
print("Total columns:", len(df.columns))
print()


# ------------------------------------------------------------
# Separate features and labels
# ------------------------------------------------------------

X = df.drop(
    columns=["filename", "label"]
)

y = df["label"]


print("Feature count:", X.shape[1])
print()

print("Labels:")
print("Human:", int((y == 0).sum()))
print("Synthetic:", int((y == 1).sum()))
print()


# ------------------------------------------------------------
# Create SVM pipeline
#
# StandardScaler:
# Makes all features comparable in scale.
#
# SVC:
# RBF kernel learns nonlinear patterns.
#
# probability=True:
# Allows us to obtain probability estimates later.
# ------------------------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# Leave-One-Out validation
#
# With only 10 samples, a normal train/test split would be
# unstable. Leave-One-Out uses 9 samples for training and
# 1 sample for testing, repeated for every sample.
# ------------------------------------------------------------

print("Running Leave-One-Out validation...")
print()

loo = LeaveOneOut()

scores = cross_val_score(
    model,
    X,
    y,
    cv=loo
)

print(
    "LOO validation accuracy:",
    f"{scores.mean() * 100:.2f}%"
)

print(
    "Correct predictions:",
    int(scores.sum()),
    "/",
    len(scores)
)

print()

print(
    "NOTE: This validation result is experimental only because",
    "the current dataset contains just 10 recordings."
)

print()


# ------------------------------------------------------------
# Train final model using ALL available samples
# ------------------------------------------------------------

print("Training final SVM using all 10 samples...")

model.fit(
    X,
    y
)

print("Training completed!")
print()


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "Model saved as:",
    MODEL_FILE
)

print()


# ------------------------------------------------------------
# Training-set prediction
# ------------------------------------------------------------

predictions = model.predict(X)

print("=" * 60)
print("TRAINING DATA CHECK")
print("=" * 60)
print()

print(
    classification_report(
        y,
        predictions,
        target_names=[
            "Human",
            "Synthetic"
        ],
        zero_division=0
    )
)

print("Confusion Matrix:")
print()

print(
    confusion_matrix(
        y,
        predictions
    )
)

print()

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print()