import os
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# -----------------------------------
# CREATE DIRECTORIES
# -----------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("reports/logs", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# -----------------------------------
# LOAD DATA
# -----------------------------------

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")

y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")

print("\nX_train:", X_train.shape)
print("X_val:", X_val.shape)

# -----------------------------------
# PARAMETERS
# -----------------------------------

VOCAB_SIZE = 20000
MAX_LEN = 150
NUM_CLASSES = 3

# -----------------------------------
# BUILD MODEL
# -----------------------------------

model = Sequential([

    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
        input_length=MAX_LEN
    ),

    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        )
    ),

    Dropout(0.3),

    Bidirectional(
        LSTM(64)
    ),

    Dropout(0.3),

    Dense(64, activation='relu'),

    Dropout(0.2),

    Dense(NUM_CLASSES, activation='softmax')
])

# -----------------------------------
# COMPILE MODEL
# -----------------------------------

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.build(input_shape=(None, MAX_LEN))

model.summary()

# -----------------------------------
# CALLBACKS
# -----------------------------------

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/best_sentiment_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)

# -----------------------------------
# MLFLOW
# -----------------------------------

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment(
    "Customer_Sentiment_LSTM"
)

with mlflow.start_run():

    # log parameters
    mlflow.log_param("vocab_size", VOCAB_SIZE)
    mlflow.log_param("max_len", MAX_LEN)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 64)

    # -----------------------------------
    # TRAIN
    # -----------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[
            early_stopping,
            reduce_lr,
            checkpoint
        ]
    )

    # -----------------------------------
    # SAVE MODEL
    # -----------------------------------

    model.save(
        "models/sentiment_lstm.keras"
    )

    print("\nModel Saved")

    # -----------------------------------
    # LOG MODEL
    # -----------------------------------

    mlflow.tensorflow.log_model(
        model,
        "sentiment_model"
    )

    # -----------------------------------
    # LOG METRICS
    # -----------------------------------

    best_val_acc = max(
        history.history['val_accuracy']
    )

    best_train_acc = max(
        history.history['accuracy']
    )

    mlflow.log_metric(
        "best_train_accuracy",
        float(best_train_acc)
    )

    mlflow.log_metric(
        "best_val_accuracy",
        float(best_val_acc)
    )

    # -----------------------------------
    # SAVE HISTORY
    # -----------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.to_csv(
        "reports/logs/training_history.csv",
        index=False
    )

    print("Training history saved")

    # -----------------------------------
    # ACCURACY PLOT
    # -----------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history['accuracy'],
        label='Train Accuracy'
    )

    plt.plot(
        history.history['val_accuracy'],
        label='Validation Accuracy'
    )

    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(
        "reports/figures/training_accuracy.png"
    )

    # -----------------------------------
    # LOSS PLOT
    # -----------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history['loss'],
        label='Train Loss'
    )

    plt.plot(
        history.history['val_loss'],
        label='Validation Loss'
    )

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(
        "reports/figures/training_loss.png"
    )

print("\nTRAINING COMPLETED SUCCESSFULLY")