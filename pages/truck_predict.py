
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(layout="wide", page_title="Logistics Service Time Prediction App") # ตั้งค่าหน้าเว็บ Streamlit ให้เป็นแบบกว้างและกำหนดชื่อหน้า

st.title("🚛 Logistics Service Time Prediction & Scheduling") # แสดงหัวข้อหลักของแอปพลิเคชัน
st.write("แอปลิเคชันนี้ช่วยพยากรณ์เวลาบริการรถบรรทุก และจัดตารางการเข้าคิวเพื่อประสิทธิภาพสูงสุด") # แสดงคำอธิบายแอปพลิเคชัน

# --- 1. Load Model (with caching) ---
@st.cache_resource # ใช้ cache เพื่อให้โหลดโมเดลเพียงครั้งเดียวเมื่อแอปพลิเคชันเริ่มทำงาน
def load_model():
    try:
        # Correcting the model path as per kernel state
        model = joblib.load('model/service_time_model.pkl') # โหลดโมเดลที่บันทึกไว้
        st.success("✅ โหลดโมเดล 'service_time_model.pkl' สำเร็จแล้ว!") # แสดงข้อความแจ้งว่าโหลดโมเดลสำเร็จ
        return model
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล 'service_time_model.pkl' กรุณาตรวจสอบว่าได้รันเซลล์ที่บันทึกโมเดลแล้ว") # แสดงข้อความแจ้งเมื่อไม่พบไฟล์โมเดล
        st.stop() # หยุดการทำงานของแอปพลิเคชัน
loaded_model = load_model() # เรียกฟังก์ชันเพื่อโหลดโมเดล

# --- 2. User Input for Unseen Data (including Company Name) ---
st.header("📝 ข้อมูลรถบรรทุกที่ต้องการพยากรณ์") # แสดงหัวข้อสำหรับส่วนข้อมูลรถบรรทุก

st.subheader("อัปโหลดข้อมูลรถบรรทุก (หรือใช้ข้อมูลเริ่มต้น)")
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ที่มีข้อมูลรถบรรทุก (.csv)", type=["csv"])

df_for_editor = pd.DataFrame() # Initialize an empty DataFrame

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        # Ensure boolean columns are correctly typed for uploaded data
        for col in ['Truck_Type_4-Wheel', 'Truck_Type_6-Wheel', 'Operation_Type_Pickup', 'Weather_Rain', 'Work_Shift_Night']:
            if col in df_uploaded.columns:
                df_uploaded[col] = df_uploaded[col].astype(bool)

        # Add 'Company_Name' if missing, for display and scheduling
        if 'Company_Name' not in df_uploaded.columns:
            df_uploaded.insert(0, 'Company_Name', [f'รถบรรทุก {i+1}' for i in range(len(df_uploaded))])

        # Add a 'Select' column for user to choose which rows to predict
        if 'Select' not in df_uploaded.columns:
            df_uploaded.insert(0, 'Select', True) # Default all selected

        st.success("✅ อัปโหลดไฟล์สำเร็จ! คุณสามารถแก้ไขหรือเลือกข้อมูลในตารางด้านล่างได้")
        df_for_editor = df_uploaded
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")
        st.stop()
else:
    # Existing initial_unseen_data definition
    df_for_editor = pd.DataFrame({
        'Company_Name': ['บริษัทขนส่ง A', 'บริษัทโลจิสติกส์ B', 'บริษัทขนส่ง C', 'บริษัทขนส่ง D',
                         'บริษัทโลจิสติกส์ E', 'บริษัทขนส่ง F', 'บริษัทโลจิสติกส์ G', 'บริษัทขนส่ง H',
                         'บริษัทขนส่ง I', 'บริษัทโลจิสติกส์ J'],
        'Staff_Count': [5, 7, 3, 6, 8, 4, 9, 2, 5, 7],
        'Total_Cartons': [150, 200, 100, 250, 180, 120, 300, 90, 220, 170],
        'SKU_Count': [3, 2, 4, 3, 2, 3, 4, 1, 3, 2],
        'Truck_Type_4-Wheel': [False, False, True, False, False, True, False, True, False, False],
        'Truck_Type_6-Wheel': [True, False, False, True, False, False, False, False, True, False],
        'Operation_Type_Pickup': [True, False, False, True, False, True, False, False, True, False],
        'Weather_Rain': [False, True, False, False, True, False, False, True, False, False],
        'Work_Shift_Night': [True, False, True, False, True, False, True, False, True, False]
    })
    # Add 'Select' column to initial data
    if 'Select' not in df_for_editor.columns:
        df_for_editor.insert(0, 'Select', True) # Default all selected


st.write("คุณสามารถแก้ไขข้อมูลรถบรรทุกที่จะนำไปทำนาย หรือเพิ่ม/ลบแถวได้โดยตรงในตารางด้านล่าง และเลือกรายการที่ต้องการพยากรณ์")

# Make the DataFrame editable in Streamlit
loaded_unseen_data = st.data_editor( # สร้างตารางที่แก้ไขได้ใน Streamlit เพื่อให้ผู้ใช้ป้อนข้อมูล
    df_for_editor,
    key="unseen_data_editor",
    num_rows="dynamic", # อนุญาตให้เพิ่ม/ลบแถวได้
    hide_index=True, # ซ่อน index ของ DataFrame
    column_config={"Select": st.column_config.CheckboxColumn("เลือก", help="เลือกรายการที่ต้องการพยากรณ์")}
)

# Filter selected rows for prediction
selected_for_prediction = loaded_unseen_data[loaded_unseen_data['Select'] == True].drop(columns=['Select'])

if selected_for_prediction.empty: # ตรวจสอบว่ามีข้อมูลที่เลือกหรือไม่
    st.warning("กรุณาเลือกข้อมูลรถบรรทุกอย่างน้อยหนึ่งแถวเพื่อทำการพยากรณ์") # แสดงคำเตือนถ้าไม่มีข้อมูล
    st.stop() # หยุดการทำงานของแอปพลิเคชัน

# --- Move Start Time Input here ---
st.header("กำหนดเวลาเริ่มต้นสำหรับจัดตาราง")
col1, col2 = st.columns(2) # แบ่งหน้าจอเป็น 2 คอลัมน์
with col1:
    date_input = st.date_input("เลือกวันที่เริ่มต้นการประมวลผล", datetime.now().date(), key="scheduling_date_input") # ให้ผู้ใช้เลือกวันที่เริ่มต้น
with col2:
    time_input = st.time_input("เลือกเวลาเริ่มต้นการประมวลผล", datetime.now().time(), key="scheduling_time_input") # ให้ผู้ใช้เลือกเวลาเริ่มต้น

start_processing_time = datetime.combine(date_input, time_input)
st.write(f
