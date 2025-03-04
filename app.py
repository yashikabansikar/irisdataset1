import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configure the Streamlit page layout
st.set_page_config(layout="wide")

# Set the title of the web app
st.title("Iris Species Classifier")

# Load the trained model from the pickle file
loaded_model = pickle.load(open('yashika_iris_model.pkl', 'rb'))

# User input fields for flower attributes
length_sepal = st.number_input("Sepal Length (cm)")
width_sepal = st.number_input("Sepal Width (cm)")
length_petal = st.number_input("Petal Length (cm)")
width_petal = st.number_input("Petal Width (cm)")

# Organizing input features into a dictionary
input_features = {
    'sepal length (cm)': length_sepal,
    'sepal width (cm)': width_sepal,
    'petal length (cm)': length_petal,
    'petal width (cm)': width_petal
}

# Convert the dictionary into a DataFrame
features_df = pd.DataFrame(input_features, index=[0])

# Button to trigger prediction
if st.button("Classify Flower"):
    model_prediction = loaded_model.predict(features_df)
    
    # Extract the predicted species
    predicted_flower = model_prediction[0]

    # Display the classification result
    st.success(f"The model classifies this flower as **{predicted_flower}**.")
