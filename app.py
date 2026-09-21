import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# Ensure 'delivery_delay_model.sav' is in the same directory as this app.py file
try:
    logi = joblib.load('delivery_delay_model.sav')
except FileNotFoundError:
    st.error("Error: 'delivery_delay_model.sav' not found. Please ensure the model file is in the same directory.")
    st.stop()

# Get feature names from the original DataFrame for consistent input fields
# In a real application, you might hardcode these or load them from a config.
# For this example, assuming 'X.columns' represents the feature names used during training.
# Based on the kernel state, the columns are:
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

st.set_page_config(page_title="Delivery Delay Prediction")
st.title('Delivery Delay Prediction Model')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Create input fields for each feature
input_data = {}
input_data['Delivery_Distance'] = st.number_input('Delivery Distance (km)', min_value=0.0, value=25.0)
input_data['Traffic_Congestion'] = st.selectbox('Traffic Congestion Level', options=[0, 1, 2, 3, 4], format_func=lambda x: ['Low', 'Medium', 'High', 'Severe', 'Extreme'][x])
input_data['Weather_Condition'] = st.selectbox('Weather Condition', options=[0, 1, 2, 3, 4], format_func=lambda x: ['Clear', 'Rainy', 'Cloudy', 'Snowy', 'Foggy'][x])
input_data['Delivery_Slot'] = st.selectbox('Delivery Slot', options=[0, 1, 2, 3, 4], format_func=lambda x: ['Morning', 'Afternoon', 'Evening', 'Night', 'Any'][x])
input_data['Driver_Experience'] = st.number_input('Driver Experience (years)', min_value=0, value=5)
input_data['Num_Stops'] = st.number_input('Number of Stops', min_value=0, value=2)
input_data['Vehicle_Age'] = st.number_input('Vehicle Age (years)', min_value=0, value=3)
input_data['Road_Condition_Score'] = st.slider('Road Condition Score (1-10)', min_value=1, max_value=10, value=7)
input_data['Package_Weight'] = st.number_input('Package Weight (kg)', min_value=0.0, value=10.0)
input_data['Fuel_Efficiency'] = st.number_input('Fuel Efficiency (km/L)', min_value=0.0, value=15.0)
input_data['Warehouse_Processing_Time'] = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60)

# Predict button
if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Make prediction
    prediction = logi.predict(input_df)
    prediction_proba = logi.predict_proba(input_df)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('**Delivery is LIKELY to be Delayed!** ⏳')
    else:
        st.success('**Delivery is LIKELY to be On Time!** ✅')
