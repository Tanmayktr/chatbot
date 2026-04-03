import streamlit as st
import pickle
import json
import random

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    data = json.load(file)

def get_response(user_input):
    X = vectorizer.transform([user_input])
    tag = model.predict(X)[0]

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# UI
st.title("🤖 Chatbot")

user_input = st.text_input("You:")

if user_input:
    response = get_response(user_input)
    st.write("Bot:", response)