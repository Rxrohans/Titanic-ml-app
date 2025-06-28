import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

# UI title and style
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f4f9f9;
        padding: 2rem;
        border-radius: 10px;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton button {
        background-color: #0099ff;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival")

# --- Form UI ---
with st.form("survival_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
        sex = st.selectbox("Sex", ['Male', 'Female'])
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        embarked = st.selectbox("Port of Embarkation", ['Southampton', 'Cherbourg', 'Queenstown'])

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.2, step=0.1)

    submitted = st.form_submit_button("Predict Survival")

# --- Data Processing ---
if submitted:
    sex = 0 if sex == 'Male' else 1
    embarked = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}[embarked]

    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(features)[0]

    st.markdown("---")
    if prediction == 1:
        st.success("Passenger Likely to Survive!")
    else:
        st.error("Passenger Not Likely to Survive.")

st.markdown("</div>", unsafe_allow_html=True)
