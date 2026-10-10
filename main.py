import streamlit as st
import nltk
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="wide")

# Custom CSS for a modern, professional UI
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.2rem;
        text-align: center;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }
    /* Metric cards styling */
    .metric-card {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #333;
    }
    /* Button styling */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BACKEND LOGIC (Preprocessing & Model)
# ==========================================

# Download NLTK data (Cached so it only runs once)
@st.cache_resource
def download_nltk():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')

download_nltk()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text Preprocessing Function (Exactly from your .ipynb)
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
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the trained model and vectorizer (Cached)
@st.cache_resource
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except FileNotFoundError:
        return None, None

tfidf, model = load_models()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dashboard", "🔍 Predict Spam", "📁 Batch Predict"])

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit & Scikit-Learn. Model: Bernoulli Naive Bayes.")

# ==========================================
# 4. PAGE 1: DASHBOARD
# ==========================================
if page == "📊 Dashboard":
    st.markdown('<div class="main-title">📩 SMS Spam Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An AI-powered tool to detect spam messages instantly.</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>98.3%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>99.1%</h3><p>Precision</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>93.4%</h3><p>F1-Score</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>BernoulliNB</h3><p>Model Used</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚡ Quick Demo")
    st.write("Try it out right here!")
    
    quick_input = st.text_area("Enter an SMS message:", height=100, placeholder="e.g., WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!")
    
    if st.button('Predict Now'):
        if not quick_input.strip():
            st.warning("Please enter a message first!")
        elif tfidf is None or model is None:
            st.error("⚠️ Model files not found! Please ensure 'vectorizer.pkl' and 'model.pkl' are in the root directory.")
        else:
            transformed_sms = transform_text(quick_input)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            if result == 1:
                st.error("🚨 **Spam Detected!** This message is classified as Spam.")
            else:
                st.success("✅ **Safe Message.** This message is Not Spam.")

# ==========================================
# 5. PAGE 2: PREDICT SPAM
# ==========================================
elif page == "🔍 Predict Spam":
    st.title("🔍 Predict SMS Spam")
    st.write("Enter a message below to check if it's spam.")

    input_sms = st.text_area("Enter the message:", height=150)

    if st.button('Predict'):
        if input_sms.strip() == "":
            st.warning("Please enter a message!")
        elif tfidf is None or model is None:
            st.error("⚠️ Model files missing. Please add 'vectorizer.pkl' and 'model.pkl'.")
        else:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]
            
            # 4. Get Confidence (BernoulliNB supports predict_proba)
            try:
                proba = model.predict_proba(vector_input)[0]
                confidence = proba[1] if result == 1 else proba[0]
                confidence_pct = f"{confidence * 100:.2f}%"
            except AttributeError:
                confidence_pct = "N/A"

            # 5. Output
            st.markdown("### Results")
            if result == 1:
                st.error(f"🚨 **Spam Message** (Confidence: {confidence_pct})")
            else:
                st.success(f"✅ **Not Spam** (Confidence: {confidence_pct})")

            # 6. Show Preprocessing Details
            with st.expander("🔍 See how the text was preprocessed"):
                st.write("**Original Text:**")
                st.info(input_sms)
                st.write("**Transformed Text (Stemmed & Cleaned):**")
                st.code(transformed_sms if transformed_sms else "[Empty - all words were stopwords or punctuation]")

# ==========================================
# 6. PAGE 3: BATCH PREDICT
# ==========================================
elif page == "📁 Batch Predict":
    st.title("📁 Batch Prediction")
    st.write("Upload a CSV file containing SMS messages to predict them in bulk.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        if tfidf is None or model is None:
            st.error("⚠️ Model files missing. Please add 'vectorizer.pkl' and 'model.pkl'.")
        else:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Allow user to select the text column
            text_column = st.selectbox("Select the column containing SMS text:", df.columns)
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing..."):
                    # Preprocess and predict
                    df['Transformed'] = df[text_column].astype(str).apply(transform_text)
                    df['Prediction'] = df['Transformed'].apply(lambda x: model.predict(tfidf.transform([x]))[0])
                    df['Result'] = df['Prediction'].apply(lambda x: 'Spam' if x == 1 else 'Not Spam')
                    
                    st.success("✅ Prediction Complete!")
                    st.write("### Results")
                    st.dataframe(df[[text_column, 'Result']])
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name='spam_predictions.csv',
                        mime='text/csv',
                    )
