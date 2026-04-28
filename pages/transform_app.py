import streamlit as st # นำเข้าไลบรารี Streamlit สำหรับสร้าง Web Application
import pandas as pd # นำเข้าไลบรารี Pandas สำหรับจัดการข้อมูล DataFrame
import numpy as np # นำเข้าไลบรารี NumPy สำหรับการคำนวณเชิงตัวเลข
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder # นำเข้าเครื่องมือสำหรับ Preprocessing จาก scikit-learn
import warnings # นำเข้าไลบรารี warnings เพื่อจัดการคำเตือน
warnings.filterwarnings('ignore') # ตั้งค่าไม่ให้แสดงคำเตือน (เช่น DeprecationWarning)

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(layout="wide", page_title="Data Transformation Web App") # กำหนดค่าการแสดงผลหน้าเว็บ (กว้าง, ชื่อหน้าเว็บ)
st.title("🛠️ Data Transformation Web App") # แสดงชื่อหัวข้อหลักบนเว็บแอป
st.error("ใช้สำหรับชุดข้อมูลที่มีโครงสร้างเหมือน redbull_clean.csv เท่านั้น")
st.write("---") # แสดงเส้นแบ่งเพื่อจัดระเบียบหน้าเว็บ

# --- Function สำหรับแต่ละขั้นตอนการแปลงข้อมูล ---

def apply_feature_engineering(df): # ฟังก์ชันสำหรับ Feature Engineering
    if 'Unit_Price' in df.columns and 'Units_Sold' in df.columns: # ตรวจสอบว่ามีคอลัมน์ 'Unit_Price' และ 'Units_Sold' หรือไม่
        df['Revenue'] = df['Unit_Price'] * df['Units_Sold'] # สร้างคอลัมน์ใหม่ 'Revenue' จากผลคูณ
        st.success("✅ สร้าง Feature 'Revenue' สำเร็จ") # แสดงข้อความแจ้งว่าสำเร็จ
    else:
        st.warning("⚠️ ไม่พบ 'Unit_Price' หรือ 'Units_Sold' สำหรับ Feature Engineering") # แสดงข้อความเตือนถ้าไม่พบคอลัมน์
    return df # ส่ง DataFrame ที่อัปเดตแล้วกลับไป

def apply_scaling_data(df, method='StandardScaler'): # ฟังก์ชันสำหรับ Scaling Data
    # ระบุคอลัมน์ที่จะ Scaled (จากตัวอย่างใน Lab)
    scaled_cols = ['Unit_Price', 'Units_Sold', 'Marketing_Spend', 'Revenue'] # กำหนดรายการคอลัมน์ที่ต้องการ Scaling
    existing_scaled_cols = [col for col in scaled_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])] # กรองคอลัมน์ที่มีอยู่และเป็นตัวเลข

    if not existing_scaled_cols: # หากไม่มีคอลัมน์ที่สามารถ Scaling ได้
        st.warning(f"⚠️ ไม่มีคอลัมน์ตัวเลขที่ระบุใน {scaled_cols} ให้ทำการ Scaling") # แสดงข้อความเตือน
        return df # ส่ง DataFrame เดิมกลับไป

    st.subheader(f"\nApplying {method} to: {', '.join(existing_scaled_cols)}") # แสดงหัวข้อย่อยว่ากำลังใช้ method ใดกับคอลัมน์ใดบ้าง
    
    if method == 'StandardScaler': # ถ้าเลือกวิธี StandardScaler
        scaler = StandardScaler() # สร้างอ็อบเจกต์ StandardScaler
    elif method == 'MinMaxScaler': # ถ้าเลือกวิธี MinMaxScaler
        scaler = MinMaxScaler() # สร้างอ็อบเจกต์ MinMaxScaler
    else:
        st.error("❌ วิธี Scaling ไม่ถูกต้อง! เลือก 'StandardScaler' หรือ 'MinMaxScaler'") # แสดงข้อผิดพลาดหาก method ไม่ถูกต้อง
        return df # ส่ง DataFrame เดิมกลับไป

    # ตรวจสอบว่าคอลัมน์ 'Revenue' ถูกสร้างขึ้นแล้วหรือไม่ก่อน Scaling
    # ถ้ามี 'Revenue' และถูกเลือกให้ Scale ให้รวมเข้าไปด้วย
    
    try: # ลองทำการ Scaling
        df[existing_scaled_cols] = scaler.fit_transform(df[existing_scaled_cols]) # Fit และ Transform ข้อมูลในคอลัมน์ที่เลือก
        st.success(f"✅ ทำการ Scaling ด้วย {method} สำเร็จ") # แสดงข้อความแจ้งว่าสำเร็จ
    except Exception as e: # หากเกิดข้อผิดพลาด
        st.error(f"❌ เกิดข้อผิดพลาดในการ Scaling: {e}") # แสดงข้อผิดพลาด
    return df # ส่ง DataFrame ที่อัปเดตแล้วกลับไป

def apply_discretization(df): # ฟังก์ชันสำหรับ Discretization
    if 'Customer_Score' in df.columns and pd.api.types.is_numeric_dtype(df['Customer_Score']): # ตรวจสอบว่ามีคอลัมน์ 'Customer_Score' และเป็นตัวเลข
        try: # ลองทำการ Discretization
            # ตรวจสอบจำนวน unique values ของ Customer_Score
            if df['Customer_Score'].nunique() < 3: # ตรวจสอบจำนวนค่าที่ไม่ซ้ำกัน
                st.warning("⚠️ 'Customer_Score' มี unique value น้อยเกินไปสำหรับการทำ qcut 3 bins") # แสดงข้อความเตือน
                return df # ส่ง DataFrame เดิมกลับไป

            df['Customer_Score_Freq'] = pd.qcut(df['Customer_Score'], q=3, labels=['Low', 'Mid', 'High']) # ใช้ qcut แบ่งข้อมูลเป็น 3 ช่วงตามความถี่
            le_Score = LabelEncoder()
            df['Customer_Score_Freq'] = le_Score.fit_transform(df['Customer_Score_Freq'])
            df = df.drop(columns=['Customer_Score'], errors='ignore') # ลบคอลัมน์ 'Customer_Score' เดิมออก
            st.success("✅ ทำ Discretization บน 'Customer_Score' (สร้าง 'Customer_Score_Freq') สำเร็จ") # แสดงข้อความแจ้งว่าสำเร็จ
        except Exception as e: # หากเกิดข้อผิดพลาด
            st.error(f"❌ เกิดข้อผิดพลาดในการทำ Discretization: {e}") # แสดงข้อผิดพลาด
    else:
        st.warning("⚠️ ไม่พบ 'Customer_Score' หรือไม่ใช่ข้อมูลตัวเลขสำหรับ Discretization") # แสดงข้อความเตือนถ้าไม่พบคอลัมน์หรือเป็นชนิดข้อมูลที่ไม่ถูกต้อง
    return df # ส่ง DataFrame ที่อัปเดตแล้วกลับไป

def apply_encoding(df): # ฟังก์ชันสำหรับ Encoding
    categorical_cols = ['Product_Variant', 'Region', 'Channel'] # กำหนดรายการคอลัมน์ประเภท Categorical
    encoded_cols_mapping = {} # สร้าง dictionary สำหรับเก็บ mapping ของการ Encoding

    for col in categorical_cols: # วนลูปผ่านแต่ละคอลัมน์ Categorical
        if col in df.columns and pd.api.types.is_string_dtype(df[col]): # ตรวจสอบว่าคอลัมน์มีอยู่และเป็นชนิด String
            le = LabelEncoder() # สร้างอ็อบเจกต์ LabelEncoder
            df[f'{col}_enc'] = le.fit_transform(df[col]) # ทำ Label Encoding และสร้างคอลัมน์ใหม่ที่มี '_enc' ต่อท้าย
            encoded_cols_mapping[col] = {label: index for index, label in enumerate(le.classes_)} # เก็บ mapping
            df = df.drop(columns=[col]) # ลบคอลัมน์เดิมออก
            st.success(f"✅ ทำ Label Encoding บน '{col}' (สร้าง '{col}_enc') สำเร็จ") # แสดงข้อความแจ้งว่าสำเร็จ
        elif col in df.columns: # ถ้าคอลัมน์มีอยู่แต่ไม่ใช่ชนิด String
            st.warning(f"⚠️ คอลัมน์ '{col}' ไม่ใช่ข้อมูลประเภท String, ข้ามการ Encoding") # แสดงข้อความเตือน
    
    if encoded_cols_mapping: # หากมีการ Encoding เกิดขึ้น
        st.sidebar.subheader("การแมป Encoding:") # แสดงหัวข้อย่อยใน Sidebar
        for original_col, mapping in encoded_cols_mapping.items(): # วนลูปแสดง mapping
            st.sidebar.write(f"**{original_col}:**") # แสดงชื่อคอลัมน์เดิม
            st.sidebar.json(mapping) # แสดง mapping ในรูปแบบ JSON

    return df # ส่ง DataFrame ที่อัปเดตแล้วกลับไป

def apply_feature_extraction(df): # ฟังก์ชันสำหรับ Feature Extraction
    if 'Date' in df.columns: # ตรวจสอบว่ามีคอลัมน์ 'Date' หรือไม่
        try: # ลองทำการ Feature Extraction
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # แปลงคอลัมน์ 'Date' เป็น Datetime object
            df['Year'] = df['Date'].dt.year # สกัดปี
            df['Month'] = df['Date'].dt.month # สกัดเดือน
            df['Day'] = df['Date'].dt.day # สกัดวัน
            df['DayOfWeek'] = df['Date'].dt.dayofweek # สกัดวันในสัปดาห์
            st.success("✅ ทำ Feature Extraction จาก 'Date' (สร้าง Year, Month, Day, DayOfWeek) สำเร็จ") # แสดงข้อความแจ้งว่าสำเร็จ
        except Exception as e: # หากเกิดข้อผิดพลาด
            st.error(f"❌ เกิดข้อผิดพลาดในการทำ Date Feature Extraction: {e}") # แสดงข้อผิดพลาด
    else:
        st.warning("⚠️ ไม่พบ 'Date' สำหรับ Feature Extraction") # แสดงข้อความเตือนถ้าไม่พบคอลัมน์
    return df # ส่ง DataFrame ที่อัปเดตแล้วกลับไป

# --- Streamlit UI --- 

st.header("⚙️ การตั้งค่าการแปลงข้อมูล") # แสดงหัวข้อใน sidebar

uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV (ข้อมูลที่ผ่านการ Cleaning แล้ว)", type=["csv"]) # สร้าง widget สำหรับอัปโหลดไฟล์ CSV

df = None # กำหนดค่า df เป็น None เริ่มต้น
if uploaded_file is not None: # หากมีการอัปโหลดไฟล์
    df = pd.read_csv(uploaded_file) # อ่านไฟล์ CSV เข้ามาเป็น DataFrame
    st.subheader("📂 ข้อมูลที่อัปโหลด:") # แสดงหัวข้อย่อย
    st.dataframe(df.head()) # แสดง 5 แถวแรกของ DataFrame
    st.write(f"Shape: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์") # แสดงขนาดของ DataFrame

    st.sidebar.write("--- เลือกขั้นตอนการแปลงข้อมูล ---") # แสดงเส้นแบ่งใน sidebar
    enable_feature_engineering = st.sidebar.checkbox("1. Feature Engineering (สร้าง 'Revenue')", value=True) # สร้าง Checkbox สำหรับ Feature Engineering
    enable_scaling = st.sidebar.checkbox("2. Scaling Data (Standard/MinMax)", value=True) # สร้าง Checkbox สำหรับ Scaling
    scaling_method = st.sidebar.selectbox("เลือกวิธี Scaling:", ('StandardScaler', 'MinMaxScaler'), disabled=not enable_scaling) # สร้าง Selectbox สำหรับเลือกวิธี Scaling
    enable_discretization = st.sidebar.checkbox("3. Discretization ('Customer_Score')", value=True) # สร้าง Checkbox สำหรับ Discretization
    enable_encoding = st.sidebar.checkbox("4. Encoding (Product_Variant, Region, Channel)", value=True) # สร้าง Checkbox สำหรับ Encoding
    enable_feature_extraction = st.sidebar.checkbox("5. Feature Extraction ('Date')", value=True) # สร้าง Checkbox สำหรับ Feature Extraction
    
    st.sidebar.write("---") # แสดงเส้นแบ่งใน sidebar
    st.sidebar.markdown("**กดปุ่มด้านล่างเพื่อเริ่มการแปลงข้อมูล**") # แสดงข้อความเน้น
    if st.sidebar.button("▶️ เริ่มการแปลงข้อมูล"): # สร้างปุ่มเพื่อเริ่มกระบวนการแปลงข้อมูล
        st.subheader("🚀 เริ่มต้นการแปลงข้อมูล...") # แสดงหัวข้อย่อยว่ากำลังเริ่ม
        processed_df = df.copy() # สร้างสำเนาของ DataFrame เพื่อไม่ให้แก้ไขข้อมูลต้นฉบับ

        if enable_feature_engineering: # หากเลือก Feature Engineering
            processed_df = apply_feature_engineering(processed_df) # เรียกใช้ฟังก์ชัน Feature Engineering
        
        if enable_scaling: # หากเลือก Scaling
            # ต้องแน่ใจว่าคอลัมน์ Revenue มีอยู่ก่อนเรียก Scaling ถ้า Feature Engineering ถูกเลือก
            processed_df = apply_scaling_data(processed_df, scaling_method) # เรียกใช้ฟังก์ชัน Scaling

        if enable_discretization: # หากเลือก Discretization
            processed_df = apply_discretization(processed_df) # เรียกใช้ฟังก์ชัน Discretization

        if enable_encoding: # หากเลือก Encoding
            processed_df = apply_encoding(processed_df) # เรียกใช้ฟังก์ชัน Encoding

        if enable_feature_extraction: # หากเลือก Feature Extraction
            processed_df = apply_feature_extraction(processed_df) # เรียกใช้ฟังก์ชัน Feature Extraction

        st.subheader("📊 ข้อมูลหลังการแปลง (Transformed Data):") # แสดงหัวข้อย่อย
        st.dataframe(processed_df.head()) # แสดง 5 แถวแรกของ DataFrame ที่ถูกแปลงแล้ว
        st.write(f"Shape: {processed_df.shape[0]} แถว, {processed_df.shape[1]} คอลัมน์") # แสดงขนาดของ DataFrame
        
        # Download button
        csv = processed_df.to_csv(index=False).encode('utf-8') # แปลง DataFrame เป็น CSV string และเข้ารหัสเป็น utf-8
        st.download_button( # สร้างปุ่มดาวน์โหลด
            label="⬇️ ดาวน์โหลด Transformed Data เป็น CSV", # ข้อความบนปุ่ม
            data=csv, # ข้อมูลที่จะดาวน์โหลด
            file_name="redbull_transformed_data.csv", # ชื่อไฟล์เมื่อดาวน์โหลด
            mime="text/csv", # ชนิดของไฟล์
        )
        st.success("✅ การแปลงข้อมูลเสร็จสมบูรณ์!") # แสดงข้อความแจ้งว่าสำเร็จ
    else:
        st.info("⬆️ โปรดอัปโหลดไฟล์ CSV และเลือกขั้นตอนที่ต้องการ แล้วกด 'เริ่มการแปลงข้อมูล'") # แสดงข้อความแนะนำ
else:
    st.info("⬆️ โปรดอัปโหลดไฟล์ CSV เพื่อเริ่มต้นใช้งาน") # แสดงข้อความแนะนำเมื่อยังไม่ได้อัปโหลดไฟล์

