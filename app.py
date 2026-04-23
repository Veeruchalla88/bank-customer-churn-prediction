import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ---------------- UI ---------------- #
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("### 📊 Predict whether a customer will leave the bank")
st.write("Adjust customer details and click Predict to see churn risk.")

st.divider()

# ---------------- INPUTS ---------------- #

credit_score = st.slider("Credit Score", 300, 900, 600)
age = st.slider("Age", 18, 90, 40)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
products = st.slider("Number of Products", 1, 4, 1)

credit_card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])

salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# ---------------- ENCODING ---------------- #

country_Germany = 1 if country == "Germany" else 0
country_Spain = 1 if country == "Spain" else 0
gender_Male = 1 if gender == "Male" else 0

# ---------------- PREPARE INPUT ---------------- #

input_data = np.array([[credit_score, age, tenure, balance, products,
                        credit_card, active, salary,
                        country_Germany, country_Spain, gender_Male]])

# Scale input
input_data = scaler.transform(input_data)

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction[0] == 1:
        st.error(f" High Risk of Churn ({prob:.2f})")
    elif prob > 0.4:
        st.warning(f" Medium Risk of Churn ({prob:.2f})")
    else:
        st.success(f" Low Risk of Churn ({prob:.2f})")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.write("Built by Veerendra Challa ")