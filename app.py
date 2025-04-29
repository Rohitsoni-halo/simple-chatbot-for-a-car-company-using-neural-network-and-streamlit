import streamlit as st
import numpy as np
import json
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import random
import time


#the recommender system is not added ye

# Download the required NLTK resources
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

# Initialize the stemmer
stemmer = LancasterStemmer()

# Load intents data for the first model
with open("intents.json") as file:
    data = json.load(file)

# Load or preprocess data for the first model
try:
    with open("data.pickle", "rb") as f:
        print("Loading data from pickle")
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    print("Preprocessing data")
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = [1 if w in [stemmer.stem(w.lower()) for w in doc] else 0 for w in words]
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Define and compile the first model
print("Defining the first model")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(training[0]),)),
    tf.keras.layers.Dense(13, activation='relu'),
    tf.keras.layers.Dense(13, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load or train the first model
try:
    print("Loading model weights for the first model")
    model.load_weights("model.weights.h5")
except :
    model.fit(training, output, epochs=100, batch_size=8, verbose=1)
    model.save_weights("model.weights.h5")

# Load intents data for the second model
with open("intents1.json") as file:
    data1 = json.load(file)

# Load or preprocess data for the second model
try:
    with open("data1.pickle", "rb") as f:
        print("Loading data from pickle for the second model")
        words1, labels1, training1, output1 = pickle.load(f)
except FileNotFoundError:
    words1 = []
    labels1 = []
    docs_x = []
    docs_y = []
    print("Preprocessing data for the second model")

    for intent in data1["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words1.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels1:
            labels1.append(intent["tag"])

    words1 = [stemmer.stem(w.lower()) for w in words1 if w != "?"]
    words1 = sorted(list(set(words1)))

    labels1 = sorted(labels1)

    training1 = []
    output1 = []

    out_empty = [0 for _ in range(len(labels1))]

    for x, doc in enumerate(docs_x):
        bag = [1 if w in [stemmer.stem(w.lower()) for w in doc] else 0 for w in words1]
        output_row = out_empty[:]
        output_row[labels1.index(docs_y[x])] = 1
        training1.append(bag)
        output1.append(output_row)

    training1 = np.array(training1)
    output1 = np.array(output1)

    with open("data1.pickle", "wb") as f:
        pickle.dump((words1, labels1, training1, output1), f)

tf.compat.v1.reset_default_graph()
# Define and compile the second model
print("Defining the second model")
model1 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(training1[0]),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(output1[0]), activation='softmax')
])

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("2nd model compiled")
# Load or train the second model

try:
    model1.load_weights("model1.weights.h5")
    print("Loading model weights for the second model")
except :
    model1.fit(training1, output1, epochs=100, batch_size=8, verbose=1)
    model1.save_weights("model1.weights.h5")

def bag_of_words(s, words, stemmer):
    # Initialize the bag of words
    bag = [0 for _ in range(len(words))]

    # Tokenize the input sentence
    s_words = nltk.word_tokenize(s)

    # Stem and lower-case the tokens
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    # Populate the bag of words array
    for s_word in s_words:
        if s_word in words:
            bag[words.index(s_word)] = 1

    return np.array(bag)

def get_response(tag,prompt):
    if tag == "cars":
        # Predict using the second model if the tag is "cars"
        prediction1 = model1.predict(np.array([bag_of_words(prompt, words1, stemmer)]))
        prediction1 = prediction1[0]
        results_index1 = np.argmax(prediction1)
        tag1 = labels1[results_index1]

        # Find the response associated with the tag1
        for tg in data1["intents"]:
            if tg['tag'] == tag1:
                responses = tg['responses']
                return(random.choice(responses))
    else:
        # Find the response associated with the tag
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                return(random.choice(responses))



image="car.jpg"
# Inject custom CSS to set the background image
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("WELCOME TO PININFARINA")

for message in st.session_state.chat_history:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(message['content'])
    else:
        with st.chat_message("assistant"):
            st.write(message['content'])

        
# Accept user input
if prompt := st.chat_input("What is up?"):
    
    # Add user message to chat history
    prediction = model.predict(np.array([bag_of_words(prompt, words, stemmer)]))
    prediction = prediction[0]
    results_index = np.argmax(prediction)
    tag = labels[results_index]

    # Get the response
    response = get_response(tag,prompt)
    
    p=prompt
    r=response

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    placeholder = st.empty()
    message = ""
    
    # Add each word to the message string and update the placeholder
    if(response!=""):
        for word in response.split(" "):
            message += word + " "  # Add word to message
            placeholder.text(message)  # Update the placeholder text
            time.sleep(random.uniform(0.0,0.1))  # Delay between each word

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

