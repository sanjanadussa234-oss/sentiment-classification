import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score
)

from tensorflow.keras.models import load_model

# -----------------------------------
# CREATE DIRECTORIES
# -----------------------------------

os.makedirs(
    "reports/evaluation",
    exist_ok=True
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = load_model(
    "models/best_sentiment_model.keras"
)

print("Model Loaded Successfully")

# -----------------------------------
# LOAD DATA
# -----------------------------------

X_val = np.load(
    "data/processed/X_val.npy"
)

y_val = np.load(
    "data/processed/y_val.npy"
)

print("Validation Data Loaded")

# -----------------------------------
# LOAD LABEL ENCODER
# -----------------------------------

with open(
    "models/label_encoder.pkl",
    "rb"
) as f:

    label_encoder = pickle.load(f)

# -----------------------------------
# MODEL PREDICTIONS
# -----------------------------------

y_pred_probs = model.predict(X_val)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

# -----------------------------------
# ACCURACY
# -----------------------------------

accuracy = accuracy_score(
    y_val,
    y_pred
)

print(f"\nValidation Accuracy: {accuracy:.4f}")

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------

report = classification_report(
    y_val,
    y_pred,
    target_names=label_encoder.classes_
)

print("\nClassification Report:\n")
print(report)

# Save report
with open(
    "reports/evaluation/classification_report.txt",
    "w"
) as f:

    f.write(report)

print("\nClassification report saved")

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

cm = confusion_matrix(
    y_val,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

plt.figure(figsize=(8, 6))

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    "reports/evaluation/confusion_matrix.png"
)

print("Confusion matrix saved")

# -----------------------------------
# SAVE PREDICTIONS CSV
# -----------------------------------

results_df = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_val),
    "Predicted": label_encoder.inverse_transform(y_pred)
})

results_df.to_csv(
    "data/predictions/predictions.csv",
    index=False
)

print("Predictions CSV saved")

print("\nEVALUATION COMPLETED SUCCESSFULLY")