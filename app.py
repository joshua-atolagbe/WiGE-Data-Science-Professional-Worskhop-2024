import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the function to make predictions
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                          alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                          albumin, albumin_and_globulin_ratio):
    # Encode Gender
    gender_encoded = 0 if gender=='Male' else 1 

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Total_Bilirubin': [total_bilirubin],
        'Direct_Bilirubin': [direct_bilirubin],
        'Alkaline_Phosphotase': [alkaline_phosphotase],
        'Alamine_Aminotransferase': [alamine_aminotransferase],
        'Aspartate_Aminotransferase': [aspartate_aminotransferase],
        'Total_Protiens': [total_proteins],
        'Albumin': [albumin],
        'Albumin_and_Globulin_Ratio': [albumin_and_globulin_ratio]
    })

    # Normalize the input data

    input_data_scaled = StandardScaler().fit_transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    return prediction

# Create the Streamlit app
def main():
    st.title('Liver Disease Prediction')

    # Input form
    st.sidebar.header('Input Features')
    age = st.sidebar.number_input('Age', min_value=0, max_value=150, value=25)
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    total_bilirubin = st.sidebar.number_input('Total Bilirubin', min_value=0.1, value=0.1)
    direct_bilirubin = st.sidebar.number_input('Direct Bilirubin', min_value=0.1, value=0.1)
    alkaline_phosphotase = st.sidebar.number_input('Alkaline Phosphotase', min_value=1, value=1)
    alamine_aminotransferase = st.sidebar.number_input('Alamine Aminotransferase', min_value=1, value=1)
    aspartate_aminotransferase = st.sidebar.number_input('Aspartate Aminotransferase', min_value=1, value=1)
    total_proteins = st.sidebar.number_input('Total Proteins', min_value=1, value=1)
    albumin = st.sidebar.number_input('Albumin', min_value=1, value=1)
    albumin_and_globulin_ratio = st.sidebar.number_input('Albumin and Globulin Ratio', min_value=0.1, value=0.1)

    # Predict button
    if st.sidebar.button('Predict'):
        # Make prediction
        prediction = predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                           alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                                           albumin, albumin_and_globulin_ratio)
        
        if prediction == 0:
            # Display prediction
            st.write('Patient has no liver disease!')
        else:
            st.write('Patient is diagnosed with liver disease. See doctor.')

# Run the app
if __name__ == '__main__':
    main()
