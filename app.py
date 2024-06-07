import streamlit as st
import pandas as pd
import pickle

# Load the logistic regression model
with open('model.pkl', 'rb') as file:
    LR = pickle.load(file)

st.title("Logistic Regression Model Prediction")

st.header("Input Features")

# Input fields for the features
Pclass = st.number_input("Pclass", min_value=1, max_value=3, step=1)
SibSp = st.number_input("SibSp", min_value=0, step=1)
Parch = st.number_input("Parch", min_value=0, step=1)
Fare = st.number_input("Fare", min_value=0.0, format="%.2f")
Age = st.number_input("Age", min_value=0.0, format="%.2f")
Cabin = st.number_input("Cabin (encoded)", min_value=0, step=1)
Embarked_encoded = st.number_input("Embarked (encoded)", min_value=0.0, format="%.2f")
Sex_encoded = st.number_input("Sex (encoded)", min_value=0, max_value=1, step=1)

# Prepare the input data as a DataFrame
test_input_data = {
    'Pclass': [Pclass],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Age': [Age],
    'Cabin': [Cabin], 
    'Embarked_encoded': [Embarked_encoded],
    'Sex_encoded': [Sex_encoded]
}

test_input_df = pd.DataFrame(test_input_data)

# Predict and display the results when the button is clicked
if st.button("Predict"):
    prediction = LR.predict(test_input_df)
    prediction_proba = LR.predict_proba(test_input_df)
    
    st.subheader("Prediction")
    st.write("Prediction:", int(prediction[0]))  # 0 or 1
    
    st.subheader("Prediction Probability")
    st.write("Probability of each class:", prediction_proba[0])
    
