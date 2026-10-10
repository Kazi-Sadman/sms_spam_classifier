import streamlit as st
import nltk
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="wide")

# ==========================================
# 2. CUSTOM CSS (ONLY FOR NAVIGATION TABS)
# ==========================================
st.markdown("""
<style>
    /* Hide the Sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Dark Background */
    [data-testid="stAppViewContainer"] {
        background-color: #0a0a0f !important;
        color: white !important;
    }
    
    /* Center the Header Logo */
    .header-logo {
        text-align: center !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: white !important;
        padding-top: 20px !important;
        padding-bottom: 10px !important;
        display: block !important;
    }
    .header-logo span {
        color: #0055ff !important;
    }

    /* ==========================================
       FORCE CENTER & STYLE ONLY THE NAVIGATION TABS
       ========================================== */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important; /* Centers the tabs */
        width: 100% !important;
        gap: 40px !important; /* Space between tabs */
        background-color: transparent !important;
        border-bottom: 2px solid #2a2a3e !important;
        padding-bottom: 0px !important;
        margin-top: 10px !important;
    }

    /* Individual Tab Styling (Big and Bold) */
    .stTabs [data-baseweb="tab"] {
        height: 60px !important;
        background-color: transparent !important;
        border-radius: 0px !important;
        color: #a0a0b0 !important; /* Default text color */
        font-weight: 700 !important; /* Bold */
        font-size: 1.3rem !important; /* Bigger text */
        padding: 0px 15px !important;
        border-bottom: 3px solid transparent !important;
        transition: all 0.3s ease !important;
    }

    /* Hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        color: white !important;
        border-bottom: 3px solid rgba(0, 85, 255, 0.5) !important;
    }

    /* Active tab styling (Glowing effect) */
    .stTabs [aria-selected="true"] {
        color: #00f2fe !important; /* Bright Cyan */
        font-weight: 900 !important;
        border-bottom: 3px solid #00f2fe !important;
        text-shadow: 0px 0px 15px rgba(0, 242, 254, 0.6) !important; /* Glow */
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. BACKEND LOGIC (Preprocessing & Model)
# ==========================================
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
# 4. TOP NAVIGATION HEADER & TABS
# ==========================================

# Top Logo (Centered)
st.markdown('<div class="header-logo">📩 Spam<span>Shield</span></div>', unsafe_allow_html=True)

# Navigation Tabs (Forced Center in CSS above)
tab_dash, tab_pred, tab_batch, tab_about = st.tabs([
    "📊 Dashboard", 
    "🔍 Predict Spam", 
    "📁 Batch Predict", 
    "ℹ️ About"
])

# ==========================================
# 5. TAB 1: DASHBOARD (Standard Content)
# ==========================================
with tab_dash:
    st.title("📊 Dashboard")
    st.write("Share high-quality messages, filter out spam instantly, and protect your inbox — all in one place.")
    st.markdown("---")
    
    st.subheader("Model Performance Metrics")
    
    # UPDATED METRICS FOR STACKING CLASSIFIER
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Accuracy", value="98.65%")
    with col2:
        st.metric(label="Precision", value="98.43%")
    with col3:
        st.metric(label="F1-Score", value="94.73%")
    with col4:
        st.metric(label="Model Used", value="Stacking Clf")

    st.markdown("---")
    st.subheader("⚡ Quick Demo")
    st.write("Try it out right here! Enter a message to see if it's Spam or Ham.")
    
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
# 6. TAB 2: PREDICT SPAM
# ==========================================
with tab_pred:
    st.title("🔍 Predict SMS Spam")
    st.write("Enter a message below to check if it's spam.")

    input_sms = st.text_area("Enter the message:", height=150)

    if st.button('Predict'):
        if input_sms.strip() == "":
            st.warning("Please enter a message!")
        elif tfidf is None or model is None:
            st.error("⚠️ Model files missing. Please add 'vectorizer.pkl' and 'model.pkl'.")
        else:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            try:
                proba = model.predict_proba(vector_input)[0]
                confidence = proba[1] if result == 1 else proba[0]
                confidence_pct = f"{confidence * 100:.2f}%"
            except AttributeError:
                confidence_pct = "N/A"

            st.subheader("Results")
            if result == 1:
                st.error(f"🚨 **Spam Message** (Confidence: {confidence_pct})")
            else:
                st.success(f"✅ **Not Spam** (Confidence: {confidence_pct})")

            with st.expander("🔍 See how the text was preprocessed"):
                st.write("**Original Text:**")
                st.info(input_sms)
                st.write("**Transformed Text (Stemmed & Cleaned):**")
                st.code(transformed_sms if transformed_sms else "[Empty - all words were stopwords or punctuation]")

# ==========================================
# 7. TAB 3: BATCH PREDICT
# ==========================================
with tab_batch:
    st.title("📁 Batch Prediction")
    st.write("Upload a CSV file containing SMS messages to predict them in bulk.")
    st.info("The CSV file should contain a column with the text messages. You will be able to select the column name after uploading.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        if tfidf is None or model is None:
            st.error("⚠️ Model files missing. Please add 'vectorizer.pkl' and 'model.pkl'.")
        else:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')

            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            text_column = st.selectbox("Select the column containing SMS text:", df.columns)
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing..."):
                    df['Transformed'] = df[text_column].astype(str).apply(transform_text)
                    df['Prediction'] = df['Transformed'].apply(lambda x: model.predict(tfidf.transform([x]))[0])
                    df['Result'] = df['Prediction'].apply(lambda x: 'Spam' if x == 1 else 'Not Spam')
                    
                    st.success("✅ Prediction Complete!")
                    st.subheader("Results")
                    st.dataframe(df[[text_column, 'Result']])
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name='spam_predictions.csv',
                        mime='text/csv',
                    )

# ==========================================
# 8. TAB 4: ABOUT
# ==========================================
with tab_about:
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ### 📖 Overview
    This SMS Spam Classifier is an end-to-end Machine Learning application designed to detect and filter spam messages. 
    It uses Natural Language Processing (NLP) to analyze the text and a Machine Learning model to classify it as **Spam** or **Ham (Not Spam)**.

    ### 🛠️ Tech Stack
    - **Frontend:** Streamlit (with custom CSS)
    - **Backend:** Python
    - **Machine Learning:** Scikit-Learn (Stacking Classifier - Ensemble)
    - **NLP:** NLTK (Natural Language Toolkit)
    - **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)

    ### 🧠 How the Model Works
    1. **Data Cleaning:** The dataset was cleaned by removing duplicate rows (403 duplicates removed, leaving 5,169 rows) and unnecessary columns.
    2. **Text Preprocessing:** Lowercase, Tokenization, Removing special characters, Removing stopwords and punctuation, PorterStemmer.
    3. **Feature Engineering:** TF-IDF Vectorizer (max_features=3000).
    4. **Model Training:** Several models were trained and evaluated. The final deployed model is a **Stacking Classifier** that combines:
       - Support Vector Machine (SVM)
       - Bernoulli Naive Bayes (BNB)
       - Extra Trees Classifier (ETC)
       - Meta-learner: Random Forest Classifier
    5. **Model Selection:** The **Stacking Classifier** achieved:
       - **Accuracy:** 98.65%
       - **Precision:** 98.43%
       - **F1-Score:** 94.73%

    ### 👨‍💻 Author
    **Kazi Sadman**  
    *Data Science & Machine Learning Enthusiast*
    """)