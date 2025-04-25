# AI-CHATBOT-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: AISHWARYA VEERANAGOUDA GIRIYAL

INTERN ID: CT04WU10

DOMAIN: PYTHON PROGRAMMING

DURATION: 4 WEEKS

MENTOR: NEELA SANTHOSH KUMAR

DESCRIPTION

###Project Description: AI-CHATBOT-WITH-NLP

This project is a **basic chatbot** made using Python, machine learning, and natural language processing (NLP). The chatbot is designed to understand and reply to simple user inputs like greetings, farewells, and questions about its identity. It is trained using a file that contains a list of possible user questions and the correct responses. These are called **intents**.


#### Main Components:

1. **intents.json**:  
   This file has all the training data for the chatbot. It contains different groups or "tags" such as `greeting`, `goodbye`, `thanks`, and `name`. Each tag includes:
   - **Patterns**: Example inputs the user might type, like “Hi” or “What is your name?”
   - **Responses**: Possible replies the bot can give, like “Hello!” or “I’m a chatbot created using Python.”

   For example:
   - **Tag**: greeting  
   - **Patterns**: "Hi", "Hello", "Hey"  
   - **Responses**: "Hello!", "Hi there!", "Greetings!"

2. **train_chatbot.py**:  
   This Python script trains the chatbot using the intents data. Here’s what it does:

   - Loads the JSON file and reads all patterns and tags.
   - Breaks down each sentence into words.
   - Uses lemmatization (changes words to their base form, like “running” to “run”).
   - Creates a **bag of words**. This means converting sentences into a list of numbers showing which words are present.
   - These numbers (X) and their tags (y) are used to train a machine learning model.
   - The model used is called **Multinomial Naive Bayes**, which is good for text-based classification.
   - The trained model, list of all words, and list of tags are saved in a file named `chatbot_model.pkl`.

3. **chatbot_model.pkl**:  
   This is a saved file that contains:
   - The trained machine learning model.
   - The vocabulary (all words used in training).
   - The list of tags.

   It is used by the chatbot to understand and respond to messages.

4. **chatbot.py**:  
   This is the main script to run the chatbot. It loads the model and data, then starts a simple chat through the command line. Here's how it works:

   - Takes user input.
   - Converts it into a bag of words.
   - Uses the model to predict the tag (meaning) of the message.
   - Based on the tag, it selects a random matching response from `intents.json`.

   For example:
   - If you type “Hello”, the bot may reply “Hi there!”



#### How It Works Simply:
1. You type a message like “Hi”.
2. The bot processes your message and converts it into numbers.
3. The model guesses what you meant (like a greeting).
4. The bot replies with a matching answer.



### Conclusion

This project is a great starting point for learning about chatbots. It uses simple concepts from Python, machine learning, and NLP. You can easily change the chatbot’s behavior by editing the `intents.json` file. It can be improved by adding more data, more advanced models, or even voice features in the future.

*OUTPUT*

https://github.com/A29-G/AI-CHATBOT-WITH-NLP/issues/1#issue-3018956801
