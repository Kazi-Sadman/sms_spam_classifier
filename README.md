
---
# 📩 SMS Spam Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)

An end-to-end Machine Learning web application that detects whether an SMS message is **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Machine Learning.


## 🚀 Live Demo

🔗 **Streamlit App:**
`https://smsspamclassifier-2ozcpwt4scvdrtrnvdytug.streamlit.app/`

## 📖 Overview
This project uses the popular SMS Spam Collection Dataset. The model is trained to classify text messages into two categories: Spam (1) and Ham (0). The application features a modern, interactive user interface built with Streamlit, allowing users to test individual messages or upload a CSV file for batch predictions.

## ✨ Features
- **📊 Dashboard:** Overview of model performance metrics (Accuracy, Precision, F1-Score).
- **🔍 Predict Spam:** Single message prediction with a confidence score and a detailed breakdown of how the text was preprocessed.
- **📁 Batch Predict:** Upload a CSV file containing multiple SMS messages to get bulk predictions and download the results.
- **ℹ️ About:** Detailed information about the dataset, NLP pipeline, and model selection.

## 🛠️ Tech Stack
- **Frontend:** Streamlit (with custom CSS for a modern UI)
- **Backend:** Python
- **Machine Learning:** Scikit-Learn, XGBoost
- **NLP:** NLTK (Natural Language Toolkit)
- **Data Handling:** Pandas, NumPy

## 🧠 NLP & Model Pipeline
1. **Data Cleaning:** Removed unnecessary columns, handled missing values, and removed 403 duplicate rows (5,169 final rows). Data was slightly imbalanced (4,516 Ham, 653 Spam).
2. **Text Preprocessing:**
   - Lowercase conversion
   - Tokenization
   - Removing special characters (non-alphanumeric)
   - Removing stopwords and punctuation
   - Stemming (PorterStemmer)
3. **Feature Engineering:** Converted text to numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency) with `max_features=3000`.
4. **Model Training & Evaluation:** Trained and evaluated multiple models (GaussianNB, MultinomialNB, BernoulliNB, SVC, KNN, Decision Tree, Random Forest, XGBoost, Voting, Stacking).
5. **Final Model Selection:** **Bernoulli Naive Bayes** was chosen for deployment.
   - **Accuracy:** 98.35%
   - **Precision:** 99.1% (Only 1 false positive)
   - **F1-Score:** 93.48%

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Kazi-Sadman/sms_spam_classifier.git
   cd sms_spam_classifier
