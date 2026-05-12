import os
import re
import string
import pickle
import nltk
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# CREATE DIRECTORIES
# -----------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -----------------------------------
# DOWNLOAD NLTK
# -----------------------------------

nltk.download('punkt')

# -----------------------------------
# LOAD DATASETS
# -----------------------------------

columns = ['id', 'entity', 'sentiment', 'review_text']

# Training dataset
train_df = pd.read_csv(
    "data/raw/twitter_training.csv"
)

# Validation dataset
val_df = pd.read_csv(
    "data/raw/twitter_validation.csv",
    header=None,
    names=columns
)

print("\nTrain Shape:", train_df.shape)
print("Validation Shape:", val_df.shape)

# -----------------------------------
# REMOVE NULL VALUES
# -----------------------------------

train_df.dropna(subset=['review_text'], inplace=True)
val_df.dropna(subset=['review_text'], inplace=True)

# -----------------------------------
# REMOVE IRRELEVANT CLASS
# -----------------------------------

train_df = train_df[
    train_df['sentiment'] != 'Irrelevant'
]

val_df = val_df[
    val_df['sentiment'] != 'Irrelevant'
]

# -----------------------------------
# REDUCE DATASET SIZE
# -----------------------------------

SAMPLES_PER_CLASS = 10000

train_df = train_df.groupby('sentiment').apply(
    lambda x: x.sample(
        min(len(x), SAMPLES_PER_CLASS),
        random_state=42
    )
).reset_index(drop=True)

print("\nReduced Training Distribution:")
print(train_df['sentiment'].value_counts())

# -----------------------------------
# TEXT CLEANING
# -----------------------------------

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

# Apply cleaning
train_df['clean_text'] = train_df['review_text'].apply(clean_text)
val_df['clean_text'] = val_df['review_text'].apply(clean_text)

print("\nText Cleaning Completed")

# -----------------------------------
# LABEL ENCODING
# -----------------------------------

label_encoder = LabelEncoder()

label_encoder.fit(train_df['sentiment'])

y_train = label_encoder.transform(
    train_df['sentiment']
)

y_val = label_encoder.transform(
    val_df['sentiment']
)

print("\nLabel Mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")

# Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# -----------------------------------
# TOKENIZATION
# -----------------------------------

VOCAB_SIZE = 20000
MAX_LEN = 150

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

# Fit ONLY on train data
tokenizer.fit_on_texts(
    train_df['clean_text']
)

# Convert to sequences
train_sequences = tokenizer.texts_to_sequences(
    train_df['clean_text']
)

val_sequences = tokenizer.texts_to_sequences(
    val_df['clean_text']
)

# Padding
X_train = pad_sequences(
    train_sequences,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)

X_val = pad_sequences(
    val_sequences,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)

print("\nX_train Shape:", X_train.shape)
print("X_val Shape:", X_val.shape)

# -----------------------------------
# SAVE ARRAYS
# -----------------------------------

np.save(
    "data/processed/X_train.npy",
    X_train
)

np.save(
    "data/processed/X_val.npy",
    X_val
)

np.save(
    "data/processed/y_train.npy",
    y_train
)

np.save(
    "data/processed/y_val.npy",
    y_val
)

print("\nProcessed arrays saved")

# -----------------------------------
# SAVE TOKENIZER
# -----------------------------------

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved")

# -----------------------------------
# SAVE CLEANED DATA
# -----------------------------------

train_df.to_csv(
    "data/processed/processed_train.csv",
    index=False
)

val_df.to_csv(
    "data/processed/processed_validation.csv",
    index=False
)

print("Processed CSVs saved")

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")