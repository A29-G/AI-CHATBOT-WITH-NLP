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

def chatbot_response(msg):
    # Convert the message to a bag of words
    bow = bag_of_words(msg)

    # Predict probabilities for each tag
    results = model.predict_proba([bow])[0]
    max_index = np.argmax(results)
    max_prob = results[max_index]

    # Lower the threshold to 0.3
    if max_prob > 0.3:  # You can adjust this value to your liking
        tag = tags[max_index]
        
        # Return a random response from the matching intent
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    else:
        # Fallback for low-confidence input
        return "I'm not sure I understand. Can you rephrase that?"

# Start a simple command-line interface chat
print("Chatbot is running! (type 'quit' to stop)")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    response = chatbot_response(message)
    print("Bot:", response)
