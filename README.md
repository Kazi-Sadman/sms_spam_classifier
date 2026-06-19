# 📩 SMS Spam Detection Using Machine Learning

## 🚀 Project Overview

This project is a **Machine Learning-based SMS Spam Detection System** that classifies SMS messages as **Spam or Ham (Not Spam)** using Natural Language Processing (NLP) techniques and multiple machine learning algorithms.

The system is trained on labeled SMS datasets and uses **TF-IDF feature extraction** with classifiers such as Naive Bayes, SVM, Logistic Regression, Random Forest, and ensemble methods like **Voting and Stacking Classifiers**.

A simple **Streamlit web application** is built to demonstrate real-time spam detection.

---

## 👨‍💻 Contributors
- Kazi Sadman Zahin
- Zannatun Nur    
- Abdur Rahman Tanvir  
   

**Department of Computer Science**  
American International University–Bangladesh (AIUB)

---

## 🎯 Key Features

- SMS Spam Classification (Spam / Ham)
- NLP preprocessing:
  - Tokenization
  - Stemming
  - Stopword removal
- TF-IDF vectorization (Top 3000 features)
- Multiple ML models comparison:
  - Naive Bayes (Bernoulli & Multinomial)
  - SVM
  - Logistic Regression
  - KNN
  - Decision Tree
  - Random Forest
- Ensemble models:
  - Voting Classifier (Best performing)
  - Stacking Classifier
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Real-time prediction web app using Streamlit
- Pickle-based model saving

---

## 🧠 Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- NLTK
- TF-IDF Vectorizer
- Matplotlib, Seaborn (EDA)
- Streamlit
- Pickle

---

## 📊 Dataset

- **Source:** Kaggle SMS Spam Collection Dataset  
- **Total Samples:** ~5,500+  
- **Labels:**
  - `ham` → Normal message  
  - `spam` → Spam message  

---

## ⚙️ System Pipeline

1. Data Collection  
2. Data Cleaning  
3. Exploratory Data Analysis (EDA)  
4. Text Preprocessing:
   - Lowercasing  
   - Tokenization  
   - Stopword Removal  
   - Stemming  
5. Feature Extraction (TF-IDF)  
6. Model Training  
7. Ensemble Learning (Voting / Stacking)  
8. Evaluation  
9. Deployment using Streamlit  

---

## 🖥️ Project Structure

```bash
sms-spam-detection/
│
├── app/                  # Streamlit app
├── models/               # Saved ML model (pickle)
├── notebooks/           # Jupyter notebooks (EDA + training)
├── dataset/             # CSV dataset files
├── static/              # Images for UI/plots
├── utils/               # preprocessing functions
├── requirements.txt
├── app.py               # main Streamlit app
└── README.md

