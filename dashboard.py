import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
import pandas as pd

df = pd.read_csv("data/predictions/predictions.csv")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analysis System",
    layout="wide"
)

st.title("💬 Sentiment Classification Dashboard (LSTM)")

# Force correct working directory (IMPORTANT for deployment)
os.chdir(os.path.dirname(__file__))

# =========================
# PATHS (FIXED STRUCTURE)
# =========================
MODEL_PATH = "models/best_sentiment_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
PREDICTIONS_PATH = "data/predictions/predictions.csv"
TRAIN_HISTORY_PATH = "reports/logs/training_history.csv"

MAX_LEN = 150


# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():

    # Safety checks
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"❌ Tokenizer not found: {TOKENIZER_PATH}")
        st.stop()

    if not os.path.exists(LABEL_ENCODER_PATH):
        st.error(f"❌ Label encoder not found: {LABEL_ENCODER_PATH}")
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
            st.warning("⚠️ Please enter some text")
        else:
            seq = tokenizer.texts_to_sequences([text])
            padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)

            pred = model.predict(padded)

            label_index = np.argmax(pred)
            sentiment = label_encoder.inverse_transform([label_index])[0]

            confidence = float(np.max(pred))

            st.success(f"🎯 Sentiment: **{sentiment}**")
            st.info(f"Confidence Score: {confidence:.2f}")

# =====================================================
# 2. DASHBOARD PAGE
# =====================================================

elif menu == "📊 Dashboard":

    st.subheader("📌 Sentiment Distribution")

    # Actual distribution
    st.write("### Actual Sentiment Distribution")
    st.bar_chart(df["Actual"].value_counts())

    # Predicted distribution
    st.write("### Predicted Sentiment Distribution")
    st.bar_chart(df["Predicted"].value_counts())


    st.subheader("📊 Actual vs Predicted Comparison")

    compare_df = df.copy()
    st.dataframe(compare_df.head(20))

    # optional visualization (clean comparison)
    # st.write("### Comparison View")
    # st.line_chart(compare_df.replace({
    #     "Negative": 0,
    #     "Neutral": 1,
    #     "Positive": 2
    # }))
    import os
    from PIL import Image
    import streamlit as st
    st.subheader("📉 Confusion Matrix")

    cm_path = "reports/evaluation/confusion_matrix.png"

    if os.path.exists(cm_path):
        image = Image.open(cm_path)
        st.image(image, caption="Confusion Matrix - LSTM Model", use_container_width=True)
    else:
        st.warning("Confusion matrix not found. Please check reports/evaluation folder.")
# =====================================================
# 3. ANALYTICS PAGE
# =====================================================
elif menu == "📈 Analytics":

    st.header("📈 Model Performance Analytics")

    # Training history
    if os.path.exists(TRAIN_HISTORY_PATH):

        history = pd.read_csv(TRAIN_HISTORY_PATH)

        st.subheader("Loss Curve")
        st.line_chart(history[["loss", "val_loss"]])

        st.subheader("Accuracy Curve")
        st.line_chart(history[["accuracy", "val_accuracy"]])

    else:
        st.warning("Training history not available.")

    st.subheader("Model Info")
    st.write("✔ LSTM-based Sentiment Classifier")
    st.write("✔ Tokenizer + Label Encoder pipeline")
    st.write("✔ Trained using TensorFlow/Keras")