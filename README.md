# 📩 SMS Spam Detection Using Machine Learning

An end-to-end **Machine Learning** and **Natural Language Processing (NLP)** project that automatically classifies SMS messages as **Spam** or **Ham (Not Spam)**. The project covers the complete ML pipeline—from data preprocessing and feature engineering to model evaluation and deployment using **Streamlit Community Cloud**.

---

## 🚀 Live Demo

🔗 **Streamlit App:**
`https://smsspamclassifier-2ozcpwt4scvdrtrnvdytug.streamlit.app/`

---

## 📖 Research Paper

This project is based on our research paper:

**Spam Message Detection in SMS Using Machine Learning Techniques**

📌 **Status:** prepare for submit in conference.

---

## 👥 Authors

* **Kazi Sadman Zahin**
* **Abdur Rahman Tanvir**
* **Zannatun Nur**

---

# 📌 Project Overview

Spam SMS messages are commonly used for phishing attacks, scams, fake promotions, and fraudulent activities. Detecting these messages automatically helps improve communication security and user experience.

This project uses **Machine Learning** and **Natural Language Processing (NLP)** techniques to classify SMS messages into two categories:

* 📩 Ham (Legitimate Message)
* 🚫 Spam Message

---

# ✨ Features

* 📊 Data Cleaning & Preprocessing
* 📈 Exploratory Data Analysis (EDA)
* 🧠 Natural Language Processing (NLP)
* 🔤 TF-IDF Feature Extraction
* 🤖 Multiple Machine Learning Models
* 🏆 Ensemble Learning (Voting & Stacking)
* 📉 Model Evaluation
* 🌐 Streamlit Web Application
* 💾 Model Serialization using Pickle

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Streamlit
* Pickle
* Git & GitHub

---

# 📂 Dataset

Dataset: **SMS Spam Collection Dataset**

| Description       |     Value |
| ----------------- | --------: |
| Original Messages |      5572 |
| Cleaned Messages  |      5169 |
| Classes           | Ham, Spam |

---

# 🔄 Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Text Preprocessing
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Train-Test Split
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Best Model Selection
   │
   ▼
Model Deployment
```

---

# 🧹 Data Preprocessing

The following preprocessing steps were applied:

* Remove unnecessary columns
* Remove duplicate records
* Remove null values
* Convert labels (Ham → 0, Spam → 1)
* Convert text to lowercase
* Tokenization
* Remove punctuation
* Remove special characters
* Remove stopwords
* Apply stemming

---

# 📊 Exploratory Data Analysis

EDA was performed to better understand the dataset through visualization.

Analysis included:

* Class Distribution
* Word Count
* Character Count
* Sentence Count
* Histograms
* Correlation Analysis
* WordCloud
* Most Frequent Spam Words
* Most Frequent Ham Words

---

# ⚙️ Feature Engineering

Text data was converted into numerical features using:

**TF-IDF Vectorizer**

Configuration:

* Maximum Features = **3000**

---

# 🤖 Machine Learning Models

The following algorithms were trained and compared:

* Multinomial Naive Bayes
* Bernoulli Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Voting Classifier
* Stacking Classifier

---

# 📈 Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* F1-Score
* Confusion Matrix

Since classifying a legitimate SMS as spam is undesirable, **Precision** was treated as the primary evaluation metric.

---

# 🏆 Final Model

After extensive experimentation, the **Voting Classifier** was selected as the final model because it achieved the best balance between precision and overall performance.

---


---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/SMS_SPAM_CLASSIFIER.git
```

Move to the project folder

```bash
cd SMS_SPAM_CLASSIFIER
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run main.py
```

---

# 🖥️ Application Preview

Add screenshots of your application here.

```text
screenshots/
├── home_page.png
├── prediction.png
└── spam_result.png
```

---

# 🚀 Future Improvements

* Fine-tune Transformer models (BERT)
* Bangla SMS Spam Detection
* REST API using FastAPI
* Docker Containerization
* Cloud Deployment (AWS/Azure)
* Explainable AI (XAI)
* Mobile-Friendly Interface
* Real-time SMS Classification

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

# 🙏 Acknowledgements

We sincerely thank our faculty members, mentors, and everyone who supported us throughout this project.

We also acknowledge the creators of the **SMS Spam Collection Dataset** and the open-source community for providing valuable resources.

---

# 📬 Contact

**Kazi Sadman Zahin**


* LinkedIn: https://linkedin.com/in/your-linkedin
* Email: [kazisadman897@gmail.com](mailto:your-email@example.com)

---

## ⭐ If you found this project helpful, please consider giving it a star on GitHub!

It motivates us to continue building and sharing more AI and Machine Learning projects.

```
```
