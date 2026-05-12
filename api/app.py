import os
import re
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# BASE DIRECTORY (IMPORTANT FIX)
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------
# CREATE LOG DIRECTORY
# -----------------------------------

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_sentiment_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

model = load_model(MODEL_PATH)
print("Model Loaded")

# warm-up (important for first request speed)
_ = model.predict(np.zeros((1, 100)))

# -----------------------------------
# LOAD TOKENIZER
# -----------------------------------

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer Loaded")

# -----------------------------------
# LOAD LABEL ENCODER
# -----------------------------------

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("Label Encoder Loaded")

# -----------------------------------
# FASTAPI APP
# -----------------------------------

app = FastAPI(title="Customer Sentiment API")

# -----------------------------------
# INPUT SCHEMA
# -----------------------------------

class TextRequest(BaseModel):
    text: str

# -----------------------------------
# CONFIG
# -----------------------------------

MAX_LEN = 100  # MUST match training

# -----------------------------------
# TEXT CLEANING
# -----------------------------------

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# -----------------------------------
# HOME ROUTE
# -----------------------------------

@app.get("/")
def home():
    return {"message": "Customer Sentiment API Running Successfully"}

# -----------------------------------
# PREDICTION ROUTE
# -----------------------------------

@app.post("/predict")
def predict_sentiment(request: TextRequest):

    # Clean text
    cleaned_text = clean_text(request.text)

    # Tokenize
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    # Predict
    prediction_probs = model.predict(padded_sequence, verbose=0)

    predicted_class = np.argmax(prediction_probs, axis=1)[0]

    sentiment = label_encoder.inverse_transform([predicted_class])[0]

    confidence = float(np.max(prediction_probs))

    # -----------------------------------
    # LOGGING (SAFE FOR HF)
    # -----------------------------------

    log_file = os.path.join(LOG_DIR, "prediction_logs.csv")

    log_data = {
        "timestamp": [str(datetime.now())],
        "input_text": [request.text],
        "cleaned_text": [cleaned_text],
        "predicted_sentiment": [sentiment],
        "confidence": [round(confidence, 4)]
    }

    log_df = pd.DataFrame(log_data)

    try:
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
    except Exception as e:
        print("Logging failed:", e)

    # -----------------------------------
    # RESPONSE
    # -----------------------------------

    return {
        "input_text": request.text,
        "cleaned_text": cleaned_text,
        "predicted_sentiment": sentiment,
        "confidence": round(confidence, 4)
    }