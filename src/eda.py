import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# -----------------------------------
# LOAD PROCESSED DATA
# -----------------------------------

train_df = pd.read_csv(
    "data/processed/processed_train.csv"
)

val_df = pd.read_csv(
    "data/processed/processed_validation.csv"
)

# Combine for EDA
df = pd.concat([train_df, val_df], ignore_index=True)

print("Dataset Shape:", df.shape)

# -----------------------------------
# CREATE FIGURES DIRECTORY
# -----------------------------------

import os

os.makedirs(
    "reports/figures",
    exist_ok=True
)

# -----------------------------------
# SENTIMENT DISTRIBUTION
# -----------------------------------

plt.figure(figsize=(8, 5))

df['sentiment'].value_counts().plot(
    kind='bar'
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "reports/figures/sentiment_distribution.png"
)

print("Saved: sentiment_distribution.png")

# -----------------------------------
# REVIEW LENGTH DISTRIBUTION
# -----------------------------------

df['review_length'] = df['clean_text'].apply(
    lambda x: len(str(x).split())
)

plt.figure(figsize=(10, 5))

plt.hist(
    df['review_length'],
    bins=50
)

plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "reports/figures/review_length_distribution.png"
)

print("Saved: review_length_distribution.png")

# -----------------------------------
# WORD CLOUDS
# -----------------------------------

sentiments = ['Positive', 'Negative', 'Neutral']

for sentiment in sentiments:

    text = " ".join(
    df[df['sentiment'] == sentiment]['clean_text']
    .dropna()
    .astype(str)
)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(text)

    plt.figure(figsize=(10, 5))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.title(f"{sentiment} Word Cloud")

    filename = f"reports/figures/{sentiment.lower()}_wordcloud.png"

    plt.savefig(filename)

    print(f"Saved: {filename}")

# -----------------------------------
# MOST COMMON WORDS
# -----------------------------------

all_words = " ".join(
    df['clean_text']
    .dropna()
    .astype(str)
).split()

word_counts = Counter(all_words)

common_words = word_counts.most_common(20)

words = [word[0] for word in common_words]
counts = [word[1] for word in common_words]

plt.figure(figsize=(12, 6))

plt.bar(words, counts)

plt.xticks(rotation=45)

plt.title("Top 20 Most Common Words")
plt.xlabel("Words")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "reports/figures/top_common_words.png"
)

print("Saved: top_common_words.png")

print("\nEDA COMPLETED SUCCESSFULLY")