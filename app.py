import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")

st.title("CreditWise — Loan Approval Prediction")
st.write("Enter applicant details to predict the loan approval outcome.")

st.info(
    "This tool is an educational ML demo. "
    "Predictions should not be treated as real financial decisions."
)

st.subheader("👤 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

with col2:
    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

st.subheader("💼 Employment & Financial Information")

col1, col2 = st.columns(2)

with col1:
    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Contract", "Self-employed", "Unemployed"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=10000.0,
        step=1000.0
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        value=0.30,
        step=0.01
    )

with col2:
    employer_category = st.selectbox(
        "Employer Category",
        ["Private", "Government", "MNC", "Business", "Unemployed"]
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        max_value=900,
        value=700
    )

st.subheader("🏠 Loan Information")

col1, col2 = st.columns(2)

with col1:
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1,
        value=60,
        step=1
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Home", "Education", "Personal", "Business", "Car"]
    )

with col2:
    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

st.divider()

st.subheader("🔮 Prediction")

predict_button = st.button(
    "🔮 Predict Loan Approval",
    type="primary"
)

if predict_button:

    input_data = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category
    }])

    data_ranges = {
        "Applicant_Income": (2009, 19988),
        "Coapplicant_Income": (1, 9996),
        "Age": (21, 59),
        "Dependents": (0, 3),
        "Credit_Score": (550, 799),
        "Existing_Loans": (0, 4),
        "DTI_Ratio": (0.10, 0.60),
        "Savings": (65, 19996),
        "Collateral_Value": (36, 49954),
        "Loan_Amount": (1015, 39995),
        "Loan_Term": (12, 84)
    }

    out_of_range = []

    for feature, (lower, upper) in data_ranges.items():
        value = input_data.iloc[0][feature]

        if value < lower or value > upper:
            out_of_range.append(
                f"{feature}: {value:,.2f} "
                f"(dataset range: {lower:,.2f}–{upper:,.2f})"
            )

    if out_of_range:
        st.warning(
            "Some inputs are outside the range observed in the dataset. "
            "The prediction may be less reliable."
        )

        for item in out_of_range:
            st.write(f"- {item}")
            
    invalid_input = []

    if age < 18:
        invalid_input.append("Age cannot be below 18.")

    if applicant_income < 0:
        invalid_input.append("Applicant income cannot be negative.")

    if coapplicant_income < 0:
        invalid_input.append("Coapplicant income cannot be negative.")

    if dependents < 0:
        invalid_input.append("Dependents cannot be negative.")

    if existing_loans < 0:
        invalid_input.append("Existing loans cannot be negative.")

    if dti_ratio < 0:
        invalid_input.append("DTI ratio cannot be negative.")

    if savings < 0:
        invalid_input.append("Savings cannot be negative.")

    if collateral_value < 0:
        invalid_input.append("Collateral value cannot be negative.")

    if loan_amount <= 0:
        invalid_input.append("Loan amount must be greater than 0.")

    if loan_term <= 0:
        invalid_input.append("Loan term must be greater than 0.")

    if invalid_input:
        st.error("Please correct the following invalid inputs:")

        for item in invalid_input:
            st.write(f"- {item}")

        st.stop()

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.metric(
            "Approval Probability",
            f"{probability:.1%}"
        )
    else:
        st.error("❌ Loan Rejected")
        st.metric(
            "Approval Probability",
            f"{probability:.1%}"
        )
    st.caption(
    "This probability is the model's estimate based on patterns "
    "learned from the training data. It is not a guarantee of loan approval."
    )