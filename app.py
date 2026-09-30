import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and expected feature names
@st.cache_resource 
def load_assets():
    model = joblib.load('models/credit_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# UI Layout
st.title("💳 Real-World Credit Scoring System")
st.write("Enter the applicant's details below to assess loan default risk.")

# Input fields mapping exactly to the Kaggle dataset columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=2000000, value=50000)
    emp_length = st.number_input("Employment Length (Years)", min_value=0, max_value=50, value=3)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    
with col2:
    st.subheader("Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=25.0, value=10.0)

st.divider()
st.subheader("Credit History")
col3, col4 = st.columns(2)
with col3:
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=30, value=4)
with col4:
    cb_person_default_on_file = st.selectbox("Historical Default on File?", ["Y", "N"])

# Prediction Logic
if st.button("Assess Creditworthiness", type="primary"):
    
    # 1. Create a DataFrame from user input
    input_data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_amnt / income if income > 0 else 0],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })
    
    # 2. Apply One-Hot Encoding just like we did in training
    input_encoded = pd.get_dummies(input_data)
    
    # 3. Align columns: User input might miss some categories, so we align it with the training feature names
    input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)
    
    # 4. Scale and Predict
    input_scaled = scaler.transform(input_aligned)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 5. Display Results
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Default** (Probability: {probability:.1%})")
        st.write("This applicant is flagged as risky based on historical data.")
    else:
        st.success(f"✅ **Good Credit Standing** (Risk of Default: {probability:.1%})")
        st.write("This applicant is eligible for credit.")