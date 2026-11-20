import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache_resource
def train():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    tfidf = TfidfVectorizer(stop_words="english")
    model = MultinomialNB()
    model.fit(tfidf.fit_transform(df["text"]), df["label"])
    return tfidf, model

tfidf, model = train()

st.title("Spam SMS Detector")
msg = st.text_area("Enter an SMS message")

if st.button("Check"):
    if msg.strip():
        pred = model.predict(tfidf.transform([msg]))[0]
        if pred == "spam":
            st.error("Spam")
        else:
            st.success("Not spam")
    else:
        st.warning("Please enter a message")
