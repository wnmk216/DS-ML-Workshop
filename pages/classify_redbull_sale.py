import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Model and Encoders ---
# Make sure these files are in the same directory as your streamlit app.py
try:
    loaded_model = joblib.load('model/redbull_best_classify_model.pkl')
    loaded_encoders = joblib.load('model/redbull_encoders.pkl')
    st.sidebar.success('✅ โหลดโมเดล Decision Tree และ Encoder สำเร็จ!')
except FileNotFoundError:
    st.error("⚠️ ไม่พบไฟล์โมเดลหรือ Encoder. โปรดตรวจสอบว่าไฟล์ 'redbull_best_classify_model.pkl' และ 'redbull_encoders.pkl' อยู่ในโฟลเดอร์เดียวกับไฟล์ app.py ของคุณ")
    st.stop()

# Get feature names from the notebook context (hardcoded for the app's simplicity)
features = ['Region', 'Product_Variant', 'Channel', 'Unit_Price', 'Marketing_Spend']

# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="Red Bull High Sales Predictor", layout="centered")
st.title('🚀 Red Bull High Sales Predictor')
st.markdown("เครื่องมือนี้ช่วยคาดการณ์ว่า Product จะมียอดขายสูงหรือไม่ ขึ้นอยู่กับงบประมาณการตลาดและปัจจัยอื่นๆ")

st.subheader('ข้อมูลสำหรับทำนาย')

# --- 3. User Input Widgets ---

# Use loaded_encoders.classes_ to populate dropdowns dynamically
# Region
region_options = loaded_encoders['Region'].classes_
selected_region = st.selectbox('ภูมิภาค (Region)', region_options)

# Product Variant
product_options = loaded_encoders['Product_Variant'].classes_
selected_product = st.selectbox('ประเภทผลิตภัณฑ์ (Product Variant)', product_options)

# Channel
channel_options = loaded_encoders['Channel'].classes_
selected_channel = st.selectbox('ช่องทางการตลาด (Channel)', channel_options)

# Unit Price
unit_price = st.number_input('ราคาต่อหน่วย (Unit Price)', min_value=1.0, value=42.0, step=0.1)

# Marketing Spend
marketing_spend = st.number_input('งบประมาณการตลาด (Marketing Spend)', min_value=1000.0, value=195000.0, step=1000.0)


# --- 4. Prediction Button ---
if st.button('ทำนายโอกาสขายสูง'):
    # --- 5. Preprocess Input Data ---
    # Encode categorical features
    ch_enc = loaded_encoders['Channel'].transform([selected_channel])[0]
    pr_enc = loaded_encoders['Product_Variant'].transform([selected_product])[0]
    re_enc = loaded_encoders['Region'].transform([selected_region])[0]

    # Create DataFrame for model input (ensure column order matches training data)
    model_input_df = pd.DataFrame([{
        'Region': re_enc,
        'Product_Variant': pr_enc,
        'Channel': ch_enc,
        'Unit_Price': unit_price,
        'Marketing_Spend': marketing_spend,
    }], columns=features)

    # --- 6. Make Prediction ---
    prob = loaded_model.predict_proba(model_input_df)[0][1]  # Probability of High_Sales (class 1)
    predicted_class = loaded_model.predict(model_input_df)[0]

    # --- 7. Display Results and Recommendation ---
    st.subheader('ผลการทำนาย')

    if predicted_class == 1:
        st.success(f"**ผลลัพธ์:** มีโอกาส 'ยอดขายสูง' ({prob:.2%}) 🎉")
    else:
        st.info(f"**ผลลัพธ์:** มีโอกาส 'ยอดขายต่ำ' ({1-prob:.2%}) 📉")

    st.write(f"ความน่าจะเป็นที่จะมียอดขายสูง: **{prob:.2%}**")

    # Recommendation logic (consistent with the notebook)
    rec = '✅ แนะนำ: ควรลงทุนใน Channel/Product นี้' if prob >= 0.25 else '⚠️ พิจารณา: อาจต้องพิจารณาปัจจัยอื่นๆ หรือช่องทาง/สินค้าอื่น'
    st.markdown(f"**คำแนะนำ:** {rec}")

    st.markdown("--- ยอดขายสูงถูกกำหนดเป็น Units_Sold >= 75th percentile ของข้อมูล --- ")

