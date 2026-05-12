import pandas as pd

# Load training dataset
train_df = pd.read_csv("data/raw/twitter_training.csv")

# Load validation dataset
val_df = pd.read_csv("data/raw/twitter_validation.csv")

print("\n===== TRAIN DATASET =====")
print(train_df.head())
print("\nColumns:")
print(train_df.columns)
print("\nShape:")
print(train_df.shape)

print("\n============================\n")

print("\n===== VALIDATION DATASET =====")
print(val_df.head())
print("\nColumns:")
print(val_df.columns)
print("\nShape:")
print(val_df.shape)