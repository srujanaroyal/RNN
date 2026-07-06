# RNN Practical (Many to One)

# SMS Spam detection using simple RNN

# Dataset : spam.csv

# Importing Libraries

import os
import re
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50


# ----------------------------
# Clean Text
# ----------------------------
def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ----------------------------
# Train Model
# ----------------------------
def train_model():

    st.write("Training model...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[['v1', 'v2']]

    df.columns = ['label', 'text']

    df['label'] = df['label'].map({
        'ham': 0,
        'spam': 1
    })

    df['text'] = df['text'].apply(clean_text)

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df['text'])

    sequences = tokenizer.texts_to_sequences(df['text'])

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding='post'
    )

    y = df['label']

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        )
    )

    model.add(SimpleRNN(128))

    model.add(Dense(
        32,
        activation="relu"
    ))

    model.add(Dense(
        1,
        activation="sigmoid"
    ))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    model.save(MODEL)

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("Accuracy:", accuracy)

    predictions = (
        model.predict(
            X_test,
            verbose=0
        ) > 0.5
    ).astype(int)

    print(classification_report(
        y_test,
        predictions
    ))

    print(confusion_matrix(
        y_test,
        predictions
    ))


# ----------------------------
# Prediction Function
# ----------------------------
def predict_sms(message):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    probability = model.predict(
        sequence,
        verbose=0
    )[0][0]

    if probability > 0.5:
        return "Spam", probability
    else:
        return "Ham", probability


# ----------------------------
# Train only once
# ----------------------------
if not os.path.exists(MODEL):
    train_model()


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("SMS Spam Detection using Simple RNN")

st.write("Many-to-One RNN")

message = st.text_area("Enter SMS")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        prediction, probability = predict_sms(message)

        st.success(f"Prediction : {prediction}")

        st.write(
            "Confidence:",
            round(probability * 100, 2),
            "%"
        )