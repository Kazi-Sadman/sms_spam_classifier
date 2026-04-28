import streamlit as st
import pickle
import string
import nltk

# -----------------------------
# Download required NLTK data
# -----------------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# -----------------------------
# Text preprocessing function
# -----------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -----------------------------
# Load model and vectorizer
# -----------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Output
        if result == 1:
            st.error(" Spam Message")
        else:
            st.success(" Not Spam")