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
# 2. CUSTOM CSS (Top Navbar & Hero UI)
# ==========================================
st.markdown("""
<style>
    /* 1. Hide the Sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* 2. Global Background - Dark Navy with radial glow */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1a1b2e, #0a0a0f 60%);
        color: white;
    }
    
    /* 3. Top Header Logo Styling */
    .header-logo {
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .header-logo span {
        color: #0055ff;
    }

    /* 4. Top Navigation Tabs Styling (Like the screenshot) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        background-color: transparent;
        border-bottom: 1px solid #2a2a3e;
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        color: #a0a0b0;
        font-weight: 500;
        font-size: 1rem;
        padding: 0px 10px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: white;
    }
    .stTabs [aria-selected="true"] {
        color: white !important;
        font-weight: bold !important;
        border-bottom: 2px solid #0055ff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #0055ff !important;
    }

    /* 5. Hero Section */
    .hero-container { text-align: center; padding-top: 40px; padding-bottom: 40px; }
    .hero-badge {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        background-color: rgba(0, 102, 255, 0.1); border: 1px solid #0066ff;
        color: #66b3ff; font-size: 0.8rem; font-weight: 600;
        letter-spacing: 1px; margin-bottom: 25px;
    }
    .hero-title { font-size: 4rem; font-weight: 800; line-height: 1.1; margin-bottom: 20px; color: #ffffff; }
    .gradient-text {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.2rem; color: #a0a0b0; max-width: 600px;
        margin: 0 auto 40px auto; line-height: 1.6;
    }

    /* 6. Buttons */
    div.stButton > button {
        background-color: #0055ff; color: white; border-radius: 8px; border: none;
        padding: 12px 24px; font-weight: bold; font-size: 1rem;
        transition: 0.3s; width: 100%;
    }
    div.stButton > button:hover {
        background-color: #0044cc; box-shadow: 0 4px 15px rgba(0, 85, 255, 0.5);
    }

    /* 7. Metric Cards */
    .metric-card {
        background: rgba(30, 30, 46, 0.8); padding: 1.5rem; border-radius: 12px;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #2a2a3e; border-top: 3px solid #FF4B4B;
    }
    .metric-card-green { border-top: 3px solid #00CC96; }
    .metric-card-blue { border-top: 3px solid #1E88E5; }
    .metric-card-purple { border-top: 3px solid #9C27B0; }
    .metric-card h3 { color: #FFFFFF; margin-bottom: 5px; font-size: 2rem; }
    .metric-card p { color: #A0A0A0; margin: 0; font-size: 0.9rem; }
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

# Top Logo (Left aligned)
st.markdown('<div class="header-logo">📩 Spam<span>Shield</span></div>', unsafe_allow_html=True)

# Navigation Tabs (Top aligned, just like the screenshot)
tab_dash, tab_pred, tab_batch, tab_about = st.tabs([
    "📊 Dashboard", 
    "🔍 Predict Spam", 
    "📁 Batch Predict", 
    "ℹ️ About"
])

# ==========================================
# 5. TAB 1: DASHBOARD
# ==========================================
with tab_dash:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ AI-POWERED SPAM DETECTION</div>
        <h1 class="hero-title">Detect smarter.<br><span class="gradient-text">Grow securely.</span></h1>
        <p class="hero-subtitle">
            Share high-quality messages, filter out spam instantly, and protect your inbox — all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance Metrics")
    
    # UPDATED METRICS FOR STACKING CLASSIFIER
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>98.65%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card metric-card-green"><h3>98.43%</h3><p>Precision</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card metric-card-blue"><h3>94.73%</h3><p>F1-Score</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card metric-card-purple"><h3>Stacking Clf</h3><p>Model Used</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚡ Quick Demo")
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

            st.markdown("### Results")
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

            st.write("### Data Preview")
            st.dataframe(df.head())
            
            text_column = st.selectbox("Select the column containing SMS text:", df.columns)
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing..."):
                    df['Transformed'] = df[text_column].astype(str).apply(transform_text)
                    df['Prediction'] = df['Transformed'].apply(lambda x: model.predict(tfidf.transform([x]))[0])
                    df['Result'] = df['Prediction'].apply(lambda x: 'Spam' if x == 1 else 'Not Spam')
                    
                    st.success("✅ Prediction Complete!")
                    st.write("### Results")
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