import streamlit as st
import pandas as pd
import tensorflow as tf

# Load the trained Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('insurance_charges_model.keras')
    return model

model = load_model()

st.title('Insurance Charges Prediction App')
st.write('Enter the details below to predict insurance charges.')

# Input fields for features
age = st.slider('Age', 18, 100, 30)
bmi = st.slider('BMI', 15.0, 50.0, 25.0)
children = st.slider('Number of Children', 0, 5, 1)

sex = st.radio('Sex', ['male', 'female'])
smoker = st.radio('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

# Prepare input for the model
def preprocess_input(age, bmi, children, sex, smoker, region):
    # Create a dictionary with raw input
    input_data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': sex,
        'smoker': smoker,
        'region': region
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply one-hot encoding consistent with training data
    # Create all possible dummy columns to ensure consistency
    dummy_sex = pd.get_dummies(input_df['sex'], prefix='sex', drop_first=True)
    dummy_smoker = pd.get_dummies(input_df['smoker'], prefix='smoker', drop_first=True)
    dummy_region = pd.get_dummies(input_df['region'], prefix='region', drop_first=True)
    
    # Concatenate the dummy variables with the numerical features
    processed_df = pd.concat([
        input_df[['age', 'bmi', 'children']],
        dummy_sex,
        dummy_smoker,
        dummy_region
    ], axis=1)
    
    # Ensure all expected columns from training are present, fill missing with False (for one-hot) or 0
    expected_columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 
                        'region_northwest', 'region_southeast', 'region_southwest']
    
    for col in expected_columns:
        if col not in processed_df.columns:
            if 'sex_' in col or 'smoker_' in col or 'region_' in col:
                processed_df[col] = False # For boolean dummies
            else:
                processed_df[col] = 0 # For numerical if somehow missing

    # Reorder columns to match the training data's column order
    processed_df = processed_df[expected_columns]
    
    # Convert boolean columns to int (0 or 1) if the model expects numerical input for dummies
    for col in processed_df.columns:
        if processed_df[col].dtype == 'bool':
            processed_df[col] = processed_df[col].astype(int)

    return processed_df


if st.button('Predict Charges'):
    processed_input = preprocess_input(age, bmi, children, sex, smoker, region)
    
    # Make prediction
    prediction = model.predict(processed_input)[0][0]
    
    st.success(f'Predicted Insurance Charges: ${prediction:,.2f}')
