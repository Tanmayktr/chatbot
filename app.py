import json
import random
import pickle
import time
from urllib import response


# Load trained model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    data = json.load(file)

def get_response(user_input):
    X = vectorizer.transform([user_input])

    probs = model.predict_proba(X)
    confidence = max(probs[0])

    tag = model.predict(X)[0]

    # Fallback responses (variation)
    fallback_responses = [
        "Hmm 🤔 I didn't understand that.",
        "Can you rephrase that?",
        "I'm not sure I get it. Try asking differently.",
        "Sorry, I didn’t catch that.",
        "Could you ask in another way?"
    ]

    # If low confidence
    if confidence < 0.5:
        return random.choice(fallback_responses)

    for intent in data['intents']:
        if intent['tag'] == tag:
            base_response = random.choice(intent['responses'])

            # Add variation
            prefixes = ["", "Okay! ", "Sure! ", "Got it! ", ""]
            suffixes = ["", " 😊", " 👍", "", ""]

            return random.choice(prefixes) + base_response + random.choice(suffixes)

    response = get_response(user_input)
    print("Bot is typing...")
    time.sleep(1)
    print("Bot:", response)