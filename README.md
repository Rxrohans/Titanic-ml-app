# 🚢 Titanic Survival Prediction App

An interactive machine learning app that predicts whether a passenger aboard the Titanic would have survived, based on real data from the Kaggle Titanic dataset.

![streamlit-badge](https://img.shields.io/badge/built%20with-Streamlit-brightgreen)  
[🔗 Live Demo](https://titanic-ml-apps.streamlit.app/) 

---

## 📌 Features

- 🎯 Predict survival based on input features like age, sex, ticket class, etc.
- 📊 Built with a trained **Random Forest Classifier**
- 🖥️ Simple and modern **Streamlit UI**
- 💡 Educational project aligned with machine learning interview expectations

---

## 🧠 Model Details

- Algorithm: `RandomForestClassifier` (from scikit-learn)
- Accuracy: ~83% on validation data
- Features used:
  - Pclass
  - Sex
  - Age
  - SibSp
  - Parch
  - Fare
  - Embarked

---

## 🛠️ Tech Stack

- Python 3.x
- scikit-learn
- pandas, numpy
- Streamlit
- joblib

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/Rxrohans/titanic-streamlit-app.git
cd titanic-streamlit-app
pip install -r requirements.txt
streamlit run app.py
