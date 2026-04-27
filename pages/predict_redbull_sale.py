import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Models and Training Columns ---
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('model/redbull_regression_model.pkl')
        rf_model = joblib.load('model/redbull_randomforest_model.pkl')
        st.sidebar.success('✅ โหลดโมเดลสำเร็จ!')
        return lr_model, rf_model
    except FileNotFoundError:
        st.error("Error: Model files not found. Please make sure 'redbull_regression_model.pkl' and 'redbull_randomforest_model.pkl' are in the same directory as the Streamlit app, or update the file paths.")
        st.stop()

lr_model, rf_model = load_models()


train_columns = [#สามาาถ copy มาจากไฟล์ที่ train model ได้
    'Unit_Price', 'Marketing_Spend', 'Logistics_Delay', 'Customer_Score',
    'Product_EnergyShot', 'Product_Original', 'Product_Sugarfree',
    'Region_ASIA-PACIFIC', 'Region_EUROPE-EU', 'Region_TH-CENTRAL',
    'Region_TH-EAST', 'Region_TH-NORTH', 'Region_TH-SOUTH', 'Region_USA-EAST', 'Region_USA-WEST',
    'Channel_Social Media', 'Channel_TV Ad', 'Channel_extreme sports',
    'Channel_f1 sponsorship', 'Channel_in-store promo'
]

# --- 2. Preprocessing Function ---
def preprocess_input(input_df, train_columns):
    # Ensure categorical columns are of 'object' type for get_dummies
    categorical_cols = ['Product_Variant', 'Region', 'Channel']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype('object')

    # One-Hot Encoding
    # Using the same columns as used during training is crucial
    encoded_df = pd.get_dummies(input_df, columns=categorical_cols, prefix=['Product', 'Region', 'Channel'])

    # Align columns with training data columns
    # Add missing columns with 0
    missing_cols = set(train_columns) - set(encoded_df.columns)
    for c in missing_cols:
        encoded_df[c] = 0

    # Drop extra columns (if any, though unlikely with fixed categories)
    extra_cols = set(encoded_df.columns) - set(train_columns)
    if extra_cols:
        encoded_df = encoded_df.drop(columns=list(extra_cols))

    # Ensure the order of columns is the same as during training
    final_df = encoded_df[train_columns]
    return final_df.astype(float) # Ensure all numerical for model prediction

# --- Streamlit UI ---
st.set_page_config(page_title="Red Bull Sales Prediction App", layout="centered")
st.title("📈 Red Bull Sales Prediction App")
st.subheader("ทำนายจำนวน Units_Sold โดยใช้โมเดล ML ")
# Model selection
st.info("โมเดลที่ใช้ในการทำนาย")
model_choice = st.radio(
    "กรุณาเลือกโมเดล",
    ('Linear Regression', 'Random Forest')
)

st.header("Input Features")

# Input fields for user
product_variant = st.selectbox('Product Variant', ['EnergyShot', 'Original', 'Sugarfree'])
region = st.selectbox('Region', ['TH-NORTH', 'ASIA-PACIFIC', 'TH-EAST', 'TH-CENTRAL', 'EUROPE-EU', 'TH-SOUTH', 'USA-EAST', 'USA-WEST'])
channel = st.selectbox('Channel', ['extreme sports', 'f1 sponsorship', 'TV Ad', 'in-store promo', 'Social Media'])
unit_price = st.number_input('Unit Price', min_value=20.0, max_value=60.0, value=38.0, step=0.1)
marketing_spend = st.number_input('Marketing Spend', min_value=10000.0, max_value=300000.0, value=150000.0, step=1000.0)
logistics_delay = st.slider('Logistics Delay (days)', min_value=0, max_value=90, value=45)
customer_score = st.slider('Customer Score', min_value=1, max_value=100, value=50)


if st.button('Predict Units Sold'):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Product_Variant': [product_variant],
        'Region': [region],
        'Channel': [channel],
        'Unit_Price': [unit_price],
        'Marketing_Spend': [marketing_spend],
        'Logistics_Delay': [logistics_delay],
        'Customer_Score': [customer_score]
    })

    # Preprocess the input data
    processed_input = preprocess_input(input_data, train_columns)

    # Make prediction based on selected model
    if model_choice == 'Linear Regression':
        prediction = lr_model.predict(processed_input)[0]
        st.subheader("Predicted Units Sold (Linear Regression):")
    else:
        prediction = rf_model.predict(processed_input)[0]
        st.subheader("Predicted Units Sold (Random Forest):")

    st.success(f"ประมาณการยอดขาย: {prediction:,.2f} หน่วย")

    st.markdown("---")
    st.markdown("### Input Data (for verification)")
    st.dataframe(input_data)

    st.markdown("### Preprocessed Data (for verification)")
    st.dataframe(processed_input)

