"""
import random
import json
import pickle
import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load model
model, all_words, tags = pickle.load(open("chatbot_model.pkl", "rb"))

with open("intents.json") as f:
    intents = json.load(f)

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(all_words)
    for word in sentence_words:
        for i, w in enumerate(all_words):
            if w == word:
                bag[i] = 1
    return np.array(bag)

def chatbot_response(msg):
    bow = bag_of_words(msg)
    result = model.predict([bow])[0]
    tag = tags[result]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# CLI Chat
print("Chatbot is running! (type 'quit' to stop)")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    response = chatbot_response(message)
    print("Bot:", response)
"""


import random
import json
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model and data
model, all_words, tags = pickle.load(open("chatbot_model.pkl", "rb"))

# Load intents from the intents.json file
with open("intents.json") as f:
    intents = json.load(f)

# Function to create a bag of words from a sentence
def bag_of_words(sentence):
    # Tokenize the sentence and lemmatize each word
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
    # Create a list of 0s and set 1 for the words found in the sentence
    bag = [0] * len(all_words)
    for word in sentence_words:
        if word in all_words:
            bag[all_words.index(word)] = 1
    return np.array(bag)

# Function to get chatbot response based on the message
def chatbot_response(msg):
    # Convert the message to a bag of words
    bow = bag_of_words(msg)
    
    # Predict the tag from the bag of words
    result = model.predict([bow])[0]
    tag = tags[result]

    # Loop through intents and return a random response based on the tag
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Start a simple command-line interface chat
print("Chatbot is running! (type 'quit' to stop)")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    response = chatbot_response(message)
    print("Bot:", response)
