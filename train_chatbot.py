import json
import nltk
import random
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Data preprocessing
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = pattern.split()  # Tokenizes based on spaces
        all_words.extend(w)
        xy.append((w, tag))

# Lemmatization
ignore_words = ['?', '!', '.', ',']
all_words = [lemmatizer.lemmatize(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Bag of words
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = [0] * len(all_words)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_sentence]
    for word in pattern_words:
        if word in all_words:
            idx = all_words.index(word)
            bag[idx] = 1

    X_train.append(bag)
    y_train.append(tags.index(tag))

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and data
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump((model, all_words, tags), f)

print("Model trained and saved.")
