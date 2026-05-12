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
# CREATE LOG DIRECTORY
# -----------------------------------

os.makedirs(
    "logs",
    exist_ok=True
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = load_model(
    "models/best_sentiment_model.keras"
)

print("Model Loaded")

# -----------------------------------
# LOAD TOKENIZER
# -----------------------------------

with open(
    "models/tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(f)

print("Tokenizer Loaded")

# -----------------------------------
# LOAD LABEL ENCODER
# -----------------------------------

with open(
    "models/label_encoder.pkl",
    "rb"
) as f:

    label_encoder = pickle.load(f)

print("Label Encoder Loaded")

# -----------------------------------
# FASTAPI APP
# -----------------------------------

app = FastAPI(
    title="Customer Sentiment API"
)

# -----------------------------------
# INPUT SCHEMA
# -----------------------------------

class TextRequest(BaseModel):
    text: str

# -----------------------------------
# TEXT CLEANING
# -----------------------------------

MAX_LEN = 150

def clean_text(text):

    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove mentions
    text = re.sub(r"@\w+", "", text)

    # remove hashtags symbol only
    text = re.sub(r"#", "", text)

    # keep alphabets only
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# -----------------------------------
# HOME ROUTE
# -----------------------------------

@app.get("/")

def home():

    return {
        "message": "Customer Sentiment API Running Successfully"
    }

# -----------------------------------
# PREDICTION ROUTE
# -----------------------------------

@app.post("/predict")

def predict_sentiment(request: TextRequest):

    # Clean input text
    cleaned_text = clean_text(
        request.text
    )

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    # Predict
    prediction_probs = model.predict(
        padded_sequence
    )

    predicted_class = np.argmax(
        prediction_probs,
        axis=1
    )[0]

    sentiment = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = float(
        np.max(prediction_probs)
    )

    # -----------------------------------
    # CREATE LOG ENTRY
    # -----------------------------------

    log_data = {
        "timestamp": [str(datetime.now())],
        "input_text": [request.text],
        "cleaned_text": [cleaned_text],
        "predicted_sentiment": [sentiment],
        "confidence": [round(confidence, 4)]
    }

    log_df = pd.DataFrame(log_data)

    log_file = "logs/prediction_logs.csv"

    # Append logs
    if os.path.exists(log_file):

        log_df.to_csv(
            log_file,
            mode='a',
            header=False,
            index=False
        )

    else:

        log_df.to_csv(
            log_file,
            index=False
        )

    # -----------------------------------
    # RETURN RESPONSE
    # -----------------------------------

    return {
        "input_text": request.text,
        "cleaned_text": cleaned_text,
        "predicted_sentiment": sentiment,
        "confidence": round(confidence, 4)
    }