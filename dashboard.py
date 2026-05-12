import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from PIL import Image

# =========================
# BASE DIR (IMPORTANT FIX)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analysis System",
    layout="wide"
)

st.title("💬 Sentiment Classification Dashboard (LSTM)")

# =========================
# PATHS
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "models/best_sentiment_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models/tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models/label_encoder.pkl")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data/predictions/predictions.csv")
TRAIN_HISTORY_PATH = os.path.join(BASE_DIR, "reports/logs/training_history.csv")

MAX_LEN = 150

# =========================
# SAFE LOAD CSV
# =========================
if os.path.exists(PREDICTIONS_PATH):
    df = pd.read_csv(PREDICTIONS_PATH)
else:
    df = pd.DataFrame(columns=["Actual", "Predicted"])

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"Tokenizer not found: {TOKENIZER_PATH}")
        st.stop()

    if not os.path.exists(LABEL_ENCODER_PATH):
        st.error(f"Label encoder not found: {LABEL_ENCODER_PATH}")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_artifacts()

# =========================
# SIDEBAR NAVIGATION
# =========================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🔮 Predict Sentiment", "📊 Dashboard", "📈 Analytics"]
)

# =====================================================
# 1. PREDICTION PAGE
# =====================================================
if menu == "🔮 Predict Sentiment":

    st.header("🧠 Live Sentiment Prediction")

    text = st.text_area("Enter your text")

    if st.button("Predict Sentiment"):

        if text.strip() == "":
            st.warning("Please enter some text")
        else:
            seq = tokenizer.texts_to_sequences([text])
            padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)

            pred = model.predict(padded)

            label_index = np.argmax(pred)
            sentiment = label_encoder.inverse_transform([label_index])[0]

            confidence = float(np.max(pred))

            st.success(f"🎯 Sentiment: {sentiment}")
            st.info(f"Confidence Score: {confidence:.2f}")

# =====================================================
# 2. DASHBOARD PAGE
# =====================================================
elif menu == "📊 Dashboard":

    st.subheader("📌 Sentiment Distribution")

    st.write("### Actual Sentiment Distribution")
    if "Actual" in df.columns:
        st.bar_chart(df["Actual"].value_counts())

    st.write("### Predicted Sentiment Distribution")
    if "Predicted" in df.columns:
        st.bar_chart(df["Predicted"].value_counts())

    st.subheader("📊 Actual vs Predicted Comparison")
    st.dataframe(df.head(20))

    st.subheader("📉 Confusion Matrix")

    cm_path = os.path.join(BASE_DIR, "reports/evaluation/confusion_matrix.png")

    if os.path.exists(cm_path):
        image = Image.open(cm_path)
        st.image(image, caption="Confusion Matrix", use_container_width=True)
    else:
        st.warning("Confusion matrix not found")

# =====================================================
# 3. ANALYTICS PAGE
# =====================================================
elif menu == "📈 Analytics":

    st.header("📈 Model Performance Analytics")

    if os.path.exists(TRAIN_HISTORY_PATH):

        history = pd.read_csv(TRAIN_HISTORY_PATH)

        st.subheader("Loss Curve")
        st.line_chart(history[["loss", "val_loss"]])

        st.subheader("Accuracy Curve")
        st.line_chart(history[["accuracy", "val_accuracy"]])

    else:
        st.warning("Training history not available")

    st.subheader("Model Info")
    st.write("✔ LSTM Sentiment Classifier")
    st.write("✔ Tokenizer + Label Encoder pipeline")
    st.write("✔ TensorFlow/Keras model")