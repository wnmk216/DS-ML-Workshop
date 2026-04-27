import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. โหลดโมเดลและข้อมูลที่ใช้ในการทำนาย (Load Models and Data) ---
# ตรวจสอบให้แน่ใจว่าไฟล์โมเดลและ scaler อยู่ในไดเรกทอรีเดียวกันกับ Streamlit app
try:
    kmeans_loaded = joblib.load("model/kmeans_redbull.pkl") # โหลดโมเดล K-Means ที่ถูก train ไว้แล้ว
    scaler_loaded = joblib.load("model/scaler_redbull.pkl")   # โหลด Scaler ที่ถูก train ไว้แล้ว สำหรับการทำ Preprocessing
    st.sidebar.success("โหลดโมเดล K-Means and scaler สำเร็จ!") # แสดงข้อความเมื่อโหลดสำเร็จ
except FileNotFoundError:
    st.error("Error: kmeans_redbull.pkl or scaler_redbull.pkl not found. Please ensure they are in the same directory.") # แสดงข้อความเมื่อหาไฟล์ไม่พบ
    st.stop() # หยุดการทำงานของแอปหากโหลดโมเดลไม่ได้

# กำหนด Features ที่ใช้ในการเทรนโมเดล (ต้องตรงกับตอนเทรน)
features = ['Units_Sold', 'Marketing_Spend', 'Customer_Score', 'Logistics_Delay', 'Revenue']

# กำหนดค่า Centroids สำหรับอธิบายลักษณะของแต่ละ Cluster (คัดลอกมาจากตัวแปร `centroids` ในโน้ตบุ๊ก)
centroids_data = {
    "Units_Sold": [94217.67, 224432.27],
    "Marketing_Spend": [101039.83, 200408.08],
    "Customer_Score": [49.97, 50.60],
    "Logistics_Delay": [44.85, 44.13],
    "Revenue": [3523438.42, 8562793.71]
}
centroids = pd.DataFrame(centroids_data, index=[0, 1]) # สร้าง DataFrame ของ Centroids
centroids.index.name = "Cluster" # ตั้งชื่อ Index เป็น "Cluster"

# --- 2. ส่วนติดต่อผู้ใช้ของ Streamlit App (Streamlit App Interface) ---
st.title("Market Segmentation Predictor") # หัวข้อของแอป
st.subheader("การจัดกลุ่มข้อมูลจากข้อมูลลูกค้า.") # คำอธิบายสั้นๆ

st.info("กรุณาป้อนข้อมูลลูกค้า") # หัวข้อย่อยสำหรับส่วนกรอกข้อมูล

# สร้างช่องรับข้อมูลสำหรับ Features ต่างๆ จากผู้ใช้
units_sold = st.number_input("Units Sold", min_value=0, value=150000) # ยอดขายสินค้า
marketing_spend = st.number_input("Marketing Spend", min_value=0, value=100000) # งบประมาณการตลาด
customer_score = st.slider("Customer Score (1-99)", min_value=1, max_value=99, value=50) # คะแนนลูกค้า
logistics_delay = st.number_input("Logistics Delay (days)", min_value=0, value=30) # ความล่าช้าในการขนส่ง
unit_price = st.number_input("Unit Price", min_value=0.0, value=35.0, format="%.2f") # ราคาต่อหน่วย

# คำนวณรายได้ (Revenue) จากข้อมูลที่ผู้ใช้ป้อน
revenue = units_sold * unit_price
st.info(f"Calculated Revenue: {revenue:,.2f}") # แสดงรายได้ที่คำนวณได้

# สร้าง DataFrame สำหรับข้อมูลใหม่ที่ผู้ใช้ป้อน
new_data_input = pd.DataFrame([[units_sold, marketing_spend, customer_score, logistics_delay, revenue]],
                               columns=features)

# --- 3. ตรรกะการทำนาย (Prediction Logic) ---
if st.button("จัดกลุ่มข้อมูล"):
    if not new_data_input.isnull().values.any(): # ตรวจสอบว่ามีข้อมูลว่างเปล่าหรือไม่
        # ทำการ Scale ข้อมูลใหม่ด้วย Scaler ที่โหลดมา
        scaled_data = scaler_loaded.transform(new_data_input)

        # ทำนาย Cluster ด้วยโมเดล K-Means ที่โหลดมา
        predicted_cluster = kmeans_loaded.predict(scaled_data)[0]

        st.subheader("ผลการจัดกลุ่มลูกค้า:") # หัวข้อย่อยสำหรับผลการทำนาย
        st.warning(f"ข้อมูลใหม่นี้ถูกจัดอยู่ใน **กลุ่ม (Cluster) {predicted_cluster}**.") # แสดงผลลัพธ์ Cluster ที่ทำนายได้

        # --- 4. อธิบายลักษณะของ Cluster (Explain Cluster Characteristics) ---
        st.subheader("คุณลักษณะของข้อมูลในกลุ่ม:") # หัวข้อย่อยสำหรับลักษณะ Cluster
        st.write("การจัดกลุ่มข้อมูลนี้พิจารณาจากค่าเฉลี่ยแต่ละแอททริบิวต์ที่อยู่ในกลุ่มโดยใช้หลักการ K-means") # คำอธิบาย

        cluster_info = centroids.loc[predicted_cluster] # ดึงข้อมูล Centroid ของ Cluster ที่ทำนายได้

        st.write(f"**Cluster {predicted_cluster} - ค่าเฉลี่ยของข้อมูลในกลุ่มนี้มีดังนี้ : **") # แสดงคุณสมบัติหลักของ Cluster
        st.write(f"- **Units Sold (avg):** {cluster_info['Units_Sold']:,.0f}")
        st.write(f"- **Marketing Spend (avg):** {cluster_info['Marketing_Spend']:,.2f}")
        st.write(f"- **Customer Score (avg):** {cluster_info['Customer_Score']:,.0f}")
        st.write(f"- **Logistics Delay (avg):** {cluster_info['Logistics_Delay']:,.0f}")
        st.write(f"- **Revenue (avg):** {cluster_info['Revenue']:,.2f}")

        if predicted_cluster == 0:
            st.info("การแปลผลสำหรับ Cluster 0 -  กลุ่มลูกค้ายอดขายต่ำ / ลูกค้าใหม่")
            st.write("กลุ่มนี้มีลักษณะเด่นคือ จำนวนหน่วยที่ขายได้, งบการตลาด และรายได้ อยู่ในระดับต่ำ ส่วนคะแนนความพึงพอใจและความล่าช้าในการจัดส่งอยู่ในระดับปานกลาง")
            st.error("กลุ่มนี้น่าจะเป็น ลูกค้าที่มีส่วนร่วมน้อย หรือลูกค้าใหม่ที่เพิ่งเริ่มต้น") # คำอธิบายสำหรับ Cluster 0")
        elif predicted_cluster == 1:
            st.info(" การแปลผลสำหรับ Cluster 1 —  กลุ่มลูกค้ายอดขายสูง / ลูกค้า High-Value")
            st.write("กลุ่มนี้มีลักษณะเด่นคือ จำนวนหน่วยที่ขายได้, งบการตลาด และรายได้ อยู่ในระดับสูง ส่วนคะแนนความพึงพอใจและความล่าช้าในการจัดส่งอยู่ในระดับปานกลาง")
            st.error("กลุ่มนี้น่าจะเป็น ลูกค้าที่มีมูลค่าสูงหรือมีส่วนร่วมกับธุรกิจในระดับสูง") # คำอธิบายสำหรับ Cluster 1

    else:
        st.warning("กรุณาตรวจสอบการป้อนข้อมูลให้ครบถ้วน.") # แจ้งเตือนเมื่อข้อมูลไม่ครบถ้วน

st.markdown("--- Boot Camp: Data Science and Machine Learning --- ") # ส่วนท้ายของแอป