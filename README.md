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

This project uses the popular SMS Spam Collection Dataset. The model is trained to classify text messages into two categories: Spam (1) and Ham (0). The application features a modern, interactive user interface built with Streamlit.

## ✨ Features

- **📊 Dashboard:** Modern SaaS-style hero section with model performance metrics.
- **🔍 Predict Spam:** Single message prediction with a confidence score.
- **📁 Batch Predict:** Upload a CSV file containing multiple SMS messages to get bulk predictions.
- **ℹ️ About:** Detailed information about the dataset, NLP pipeline, and model selection.

## 🛠️ Tech Stack

- **Frontend:** Streamlit (with custom CSS for a modern, dark UI)
- **Backend:** Python
- **Machine Learning:** Scikit-Learn (Stacking Classifier)
- **NLP:** NLTK (Natural Language Toolkit)

## 🧠 NLP & Model Pipeline

1. **Data Cleaning:** Removed unnecessary columns, handled missing values, and removed 403 duplicate rows (5,169 final rows).
2. **Text Preprocessing:** Lowercase conversion, Tokenization, Removing special characters, Removing stopwords, and Stemming (PorterStemmer).
3. **Feature Engineering:** Converted text to numerical features using **TF-IDF** with `max_features=3000`.
4. **Model Training:** Trained and evaluated multiple models.
5. **Final Model Selection:** **Stacking Classifier** combining SVM, BernoulliNB, and ExtraTreesClassifier with a RandomForest meta-learner.
   - **Accuracy:** 98.65%
   - **Precision:** 98.43%
   - **F1-Score:** 94.73%

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Kazi-Sadman/sms_spam_classifier.git
   cd sms_spam_classifier
   ```
