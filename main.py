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
# 2. CUSTOM CSS (Centered Top Navbar & Bold UI)
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
    
    /* 3. Center the Header Logo */
    .header-logo {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ffffff, #a0a0b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-top: 20px;
        padding-bottom: 10px;
        letter-spacing: 1px;
    }
    .header-logo span {
        background: linear-gradient(90deg, #0055ff, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 4. Center the Top Navigation Tabs */
    .stTabs {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center; /* Center tabs */
        width: 100%;
        gap: 40px;
        background-color: transparent;
        border-bottom: 1px solid #2a2a3e;
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        color: #a0a0b0;
        font-weight: 700; /* Bolder */
        font-size: 1.1rem; /* Larger */
        padding: 0px 10px;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: white;
        border-bottom: 3px solid rgba(0, 85, 255, 0.5);
    }
    .stTabs [aria-selected="true"] {
        color: #00f2fe !important; /* Bright Cyan */
        font-weight: 900 !important;
        border-bottom: 3px solid #00f2fe !important;
        text-shadow: 0px 0px 15px rgba(0, 242, 254, 0.6); /* Glowing effect */
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #00f2fe !important;
    }

    /* 5. Centered Hero Section */
    .hero-container { 
        text-align: center; 
        padding-top: 60px; 
        padding-bottom: 60px; 
        max-width: 900px;
        margin: 0 auto; /* Center the container itself */
    }
    .hero-badge {
        display: inline-block; padding: 8px 20px; border-radius: 25px;
        background-color: rgba(0, 102, 255, 0.15); border: 1px solid #0066ff;
        color: #66b3ff; font-size: 0.9rem; font-weight: 700;
        letter-spacing: 1.5px; margin-bottom: 30px;
        box-shadow: 0 0 15px rgba(0, 102, 255, 0.3);
    }
    .hero-title { 
        font-size: 4.5rem; 
        font-weight: 900; 
        line-height: 1.1; 
        margin-bottom: 25px; 
        color: #ffffff; 
    }
    .gradient-text {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 30px rgba(0, 242, 254, 0.4);
    }
    .hero-subtitle {
        font-size: 1.3rem; 
        color: #b0e0ff; /* Light cyan for color */
        max-width: 700px;
        margin: 0 auto 40px auto; /* Centered */
        line-height: 1.8;
        font-weight: 500;
    }

    /* 6. Buttons */
    div.stButton > button {
        background: linear-gradient(45deg, #0055ff, #00f2fe);
        color: white; border-radius: 10px; border: none;
        padding: 14px 28px; font-weight: bold; font-size: 1.1rem;
        transition: 0.3s; width: 100%;
        box-shadow: 0 4px 15px rgba(0, 85, 255, 0.4);
    }
    div.stButton > button:hover {
        background: linear-gradient(45deg, #0044cc, #00c0cc);
        box-shadow: 0 6px 20px rgba(0, 85, 255, 0.7);
        transform: translateY(-2px);
    }

    /* 7. Metric Cards */
    .metric-card {
        background: rgba(30, 30, 46, 0.9); padding: 2rem 1.5rem; border-radius: 15px;
        text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        border: 1px solid #2a2a3e; border-top: 4px solid #FF4B4B;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-card-green { border-top: 4px solid #00CC96; box-shadow: 0 8px 16px rgba(0, 204, 150, 0.1); }
    .metric-card-blue { border-top: 4px solid #1E88E5; box-shadow: 0 8px 16px rgba(30, 136, 229, 0.1); }
    .metric-card-purple { border-top: 4px solid #9C27B0; box-shadow: 0 8px 16px rgba(156, 39, 176, 0.1); }
    .metric-card h3 { color: #FFFFFF; margin-bottom: 10px; font-size: 2.5rem; font-weight: 800; }
    .metric-card p { color: #A0A0B0; margin: 0; font-size: 1rem; font-weight: 600; letter-spacing: 0.5px; }
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

# Navigation Tabs (Centered)
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