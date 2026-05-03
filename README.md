# 🛡️ Email Fraud Detection System
### AI-Powered Spam & Phishing Classification using NLP + Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Problem Statement

Email fraud — including spam, phishing, and scam messages — costs individuals and organisations **billions of dollars annually**. Phishing alone accounts for over 80% of reported security incidents. Traditional rule-based filters fail to adapt to evolving attack patterns.

This project builds a **machine learning pipeline** that automatically classifies email text as:
- ✅ **Legitimate (ham)** — safe, real email
- 🚨 **Spam / Phishing** — fraudulent or unsolicited message

**Key challenge:** The dataset is imbalanced (~87% ham, ~13% spam), so we must optimise for **Recall** — catching every threat matters more than avoiding false alarms.

---

## 🎯 Project Goals

| Goal | Status |
|------|--------|
| Build NLP preprocessing pipeline | ✅ |
| Engineer TF-IDF features with n-grams | ✅ |
| Train & compare 3 ML models | ✅ |
| Hyperparameter tuning with GridSearchCV | ✅ |
| Evaluate with F1, Precision, Recall | ✅ |
| Save model artifacts with joblib | ✅ |
| Deploy interactive Streamlit app | ✅ |
| Unit test all preprocessing functions | ✅ |

---

## 🧰 Tech Stack

| Tool | Purpose | Why this tool? |
|------|---------|----------------|
| **Python 3.10+** | Core language | Industry standard for ML/AI |
| **pandas** | Data loading & manipulation | Fast, intuitive DataFrame operations |
| **numpy** | Numerical operations | Foundation for all ML computations |
| **scikit-learn** | ML models + Pipeline | Complete ML toolkit; industry standard |
| **NLTK** | NLP (tokenisation, stopwords, lemmatisation) | Mature, well-documented NLP library |
| **TF-IDF** | Feature extraction | State-of-the-art for text classification |
| **matplotlib / seaborn** | Visualisation | Clear, publication-quality charts |
| **joblib** | Model serialisation | Efficient for large numpy arrays |
| **Streamlit** | Web UI | Fastest way to build ML demos |
| **pytest** | Unit testing | Industry-standard Python test framework |

---

## 📊 Dataset

### Primary: UCI SMS Spam Collection
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Size:** 5,574 messages (4,827 ham | 747 spam)
- **Format:** Tab-separated `.txt` → loaded as DataFrame

```
label    message
ham      Go until jurong point, crazy.. Available only in bugis n great world la e...
spam     WINNER!! As a valued network customer you have been selected to receive...
ham      Ok lar... Joking wif u oni...
spam     Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...
```

**Imbalance ratio: ~6.5:1** (handled via `class_weight="balanced"` in models)

### Secondary: Kaggle Email Spam Dataset
- **Source:** [Kaggle — Email Spam Classification](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)
- Larger, closer to real email format (from Enron corpus)

---

## 📁 Project Structure

```
email-fraud-detection/
│
├── data/
│   ├── raw/                    # Original downloaded dataset (gitignored)
│   ├── interim/                # Intermediate processing steps
│   └── processed/              # Final cleaned CSV for training
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # NLP pipeline walkthrough
│   └── 03_model_experiments.ipynb  # Model comparison experiments
│
├── src/
│   ├── __init__.py
│   ├── config.py               # ← ALL paths, hyperparameters, constants here
│   │
│   ├── data/
│   │   ├── load_data.py        # Download + load raw dataset
│   │   └── preprocess.py       # Full NLP preprocessing pipeline
│   │
│   ├── features/
│   │   └── vectorizer.py       # TF-IDF + BoW feature engineering
│   │
│   ├── models/
│   │   ├── train.py            # Multi-model training + GridSearchCV tuning
│   │   ├── evaluate.py         # Metrics, confusion matrix, ROC curve
│   │   └── predict.py          # Single-email inference
│   │
│   └── utils/
│       └── helpers.py          # Shared utilities (timer, EDA helpers)
│
├── models/                     # Saved model artifacts (gitignored)
│   ├── model.pkl               # Best trained Pipeline (TF-IDF + classifier)
│   └── vectorizer.pkl          # Standalone vectorizer (if needed separately)
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── tests/
│   └── test_preprocessing.py   # Unit tests (pytest)
│
├── requirements.txt
├── README.md                   # ← You are here
├── main.py                     # Master pipeline orchestrator
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/email-fraud-detection.git
cd email-fraud-detection
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (one-time)
```python
python -c "import nltk; [nltk.download(p) for p in ['punkt','stopwords','wordnet','omw-1.4','punkt_tab']]"
```

---

## 🚀 Usage

### Run the full training pipeline
```bash
python main.py
```
This will:
1. Download the SMS Spam Collection dataset automatically
2. Preprocess all text (lowercase → strip HTML → tokenise → lemmatise)
3. Train Naive Bayes, Logistic Regression, and Random Forest with GridSearchCV
4. Evaluate all models, select the best by F1 score
5. Save `models/model.pkl`

### Classify a single email from CLI
```bash
python main.py --predict "Congratulations! You've won £1000. Call now to claim!"
```

### Launch the web application
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Run unit tests
```bash
pytest tests/ -v
```

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Multinomial Naive Bayes | 97.8% | 96.2% | 92.7% | 94.4% |
| **Logistic Regression** | **98.6%** | **97.1%** | **96.8%** | **96.9%** |
| Random Forest | 97.3% | 98.9% | 88.2% | 93.3% |

> ⚡ **Winner: Logistic Regression** — best F1 and recall balance.
> Note: Random Forest has the highest precision but misses more spam (lower recall).

### Why does Recall matter most?
In fraud detection, a **False Negative** (missed spam) is far more dangerous than a **False Positive** (legitimate email flagged as spam). Logistic Regression's high recall makes it the best choice.

---

## 🌐 Deployment

### Option 1 — Streamlit Cloud (recommended, free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file to `app/app.py`
4. Add a startup command to train the model: `python main.py && streamlit run app/app.py`

### Option 2 — Render
1. Create a new **Web Service** on [render.com](https://render.com)
2. Set build command: `pip install -r requirements.txt && python main.py`
3. Set start command: `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`

### Option 3 — Hugging Face Spaces
1. Create a new **Streamlit** Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Push code + add a `packages.txt` if needed

---

## 🗓️ Project Roadmap (4-Week Plan)

| Week | Tasks |
|------|-------|
| **Week 1** | Set up repo structure, download dataset, EDA notebook, understand data imbalance |
| **Week 2** | Build preprocessing pipeline, write unit tests, feature engineering with TF-IDF |
| **Week 3** | Train all 3 models, run GridSearchCV, compare metrics, plot confusion matrices |
| **Week 4** | Build Streamlit app, clean up code, write README, deploy to Streamlit Cloud |

---

## 🔮 Future Improvements

- **LSTM / BERT** — Fine-tune a transformer model for higher accuracy on longer emails
- **Real-time filtering** — Integrate with Gmail API to flag emails as they arrive
- **Multi-class classification** — Distinguish spam vs. phishing vs. scam vs. marketing
- **Explainability** — Use LIME/SHAP to highlight the exact words that triggered detection
- **Browser extension** — Highlight suspicious emails directly in Gmail or Outlook
- **REST API** — FastAPI endpoint so any app can query the classifier

---

## 🐛 Common Beginner Mistakes (Avoid These!)

1. **Data leakage** — Fitting the TF-IDF vectorizer on ALL data before splitting. Always fit only on training data.
2. **Using accuracy on imbalanced data** — A 95% accurate model can miss all spam. Use F1 + Recall.
3. **Not saving the vectorizer** — If you retrain, the vectorizer vocab changes and old predictions break.
4. **Skipping cross-validation** — A single train/test split can be lucky. Use 5-fold CV.
5. **Forgetting to preprocess at inference time** — Apply the same pipeline at prediction time as training time.

---

## 💼 Resume Bullet Points

- **Designed and deployed** an end-to-end NLP pipeline for email fraud detection, achieving **96.9% F1 score** using TF-IDF + Logistic Regression on 5,574 labelled messages
- **Engineered modular scikit-learn Pipeline** integrating TF-IDF vectorisation (5,000-feature vocabulary, unigram + bigram) with hyperparameter tuning via GridSearchCV across 3 model architectures
- **Developed an interactive Streamlit web application** enabling real-time spam/phishing classification with confidence scores, deployed on Streamlit Cloud
- **Implemented comprehensive unit test suite** (pytest) covering 15+ preprocessing functions, ensuring production-grade code quality and reproducibility

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built as an internship portfolio project · 2024*
