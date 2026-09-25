import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report


# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load training and validation data
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")


# Convert labels into lists
y_train_labels = train_df["labels"].str.split("|")
y_val_labels = val_df["labels"].str.split("|")


# Convert emotion labels into binary columns
mlb = MultiLabelBinarizer()

y_train = mlb.fit_transform(y_train_labels)
y_val = mlb.transform(y_val_labels)


# Build the text classification pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                solver="liblinear"
            )
        )
    )
])


# Train the model
print("Training model...")
model.fit(train_df["text"], y_train)


# Predict emotions for validation data
print("Evaluating model...")
y_pred = model.predict(val_df["text"])


# Evaluate performance
micro_f1 = f1_score(y_val, y_pred, average="micro", zero_division=0)
macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

print("\n--- Validation Results ---")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\n--- Per-Emotion Results ---")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=mlb.classes_,
        zero_division=0
    )
)


# Save model and label encoder
joblib.dump(model, MODEL_DIR / "emotion_model.joblib")
joblib.dump(mlb, MODEL_DIR / "label_binarizer.joblib")

print("\nModel and label binarizer saved in ml/models/")