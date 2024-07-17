# import libraries
import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("model.joblib")

# set title
st.title("Purchase Prediction Model")
# subheader
st.subheader("Enter Customer Information and Submit for Prediction Probability")

# age input form
age = st.number_input(
    label = "Enter customer's age",
    min_value=17,
    max_value=120,
    value=35)


# gender input form
gender = st.radio(
    label="Enter customer's gender",
    options= ["M", "F"])

# credit_score input form
credit_score = st.number_input(
    label = "Enter customer's credit score",
    min_value=10,
    max_value=1000,
    value=500)

# submit button
if st.button(label="Summit for Prediction Probability"):
    
    new_data = pd.DataFrame({"age":[age], "gender":[gender], "credit_score": [credit_score]})

    predict_proba = model.predict_proba(new_data)[0][1]
    
    st.subheader(f"Based on Customer information, Purchase Probability: {predict_proba:.0%}")


