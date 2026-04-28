import streamlit as st # ไลบรารีสำหรับสร้าง Web Application
import pandas as pd # ไลบรารีสำหรับจัดการข้อมูลในรูปแบบ DataFrame
import numpy as np # ไลบรารีสำหรับคำนวณทางคณิตศาสตร์
import matplotlib.pyplot as plt # ไลบรารีสำหรับสร้างกราฟ
import seaborn as sns # ไลบรารีสำหรับสร้างกราฟที่สวยงามขึ้น
from scipy.stats.mstats import winsorize # ฟังก์ชันสำหรับจัดการ Outlier (Winsorization)
import io # ไลบรารีสำหรับจัดการ Input/Output
import warnings # ไลบรารีสำหรับจัดการคำเตือน
warnings.filterwarnings('ignore') # ไม่แสดงคำเตือน

# ตั้งค่า Streamlit page
st.set_page_config(layout="wide", page_title="Data Cleaning Workshop App")

# --- Streamlit App Title ---
st.title("🐂 Data Cleaning Workshop App") # ตั้งชื่อแอปพลิเคชัน
st.markdown("ยินดีต้อนรับสู่แอปพลิเคชัน Data Cleaning!") # ข้อความต้อนรับ
st.markdown("--- ท่านสามารถอัปโหลดไฟล์ CSV และเลือกขั้นตอนการทำความสะอาดข้อมูลได้ ---") # คำแนะนำเบื้องต้น
st.error("ใช้สำหรับชุดข้อมูล redbull_workshop_dirty.csv เท่านั้น")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"]) # ช่องสำหรับอัปโหลดไฟล์ CSV

if uploaded_file is not None: # ถ้ามีการอัปโหลดไฟล์แล้ว
    df_raw = pd.read_csv(uploaded_file) # อ่านไฟล์ CSV ที่อัปโหลด
    df = df_raw.copy() # สร้างสำเนาข้อมูลเพื่อไม่ให้แก้ไขข้อมูลต้นฉบับ
    st.success("File uploaded successfully!") # แสดงข้อความแจ้งว่าอัปโหลดสำเร็จ
    st.write("### Raw Data (First 5 Rows)") # หัวข้อแสดงข้อมูลดิบ
    st.dataframe(df_raw.head()) # แสดง 5 แถวแรกของข้อมูลดิบ

    # --- Data Cleaning Steps (as functions) ---

    def perform_data_exploration(data): # ฟังก์ชันสำหรับสำรวจข้อมูลเบื้องต้น
        st.subheader("📊 1. Data Exploration") # หัวข้อย่อย
        st.write("#### Data Shape:") # หัวข้อย่อยแสดงขนาดข้อมูล
        st.write(f"Rows: {data.shape[0]:,}, Columns: {data.shape[1]}") # แสดงจำนวนแถวและคอลัมน์
        st.write("#### Data Info:") # หัวข้อย่อยแสดงข้อมูลทั่วไปของ DataFrame
        buffer = io.StringIO()
        data.info(buf=buffer) # ดึงข้อมูล info() ไปเก็บใน buffer
        st.text(buffer.getvalue()) # แสดงผลข้อมูล info()
        st.write("#### Descriptive Statistics:") # หัวข้อย่อยแสดงสถิติเชิงพรรณนา
        st.dataframe(data.describe(include='all')) # แสดงสถิติเชิงพรรณนาสำหรับทุกคอลัมน์
        return data

    def handle_duplicate_data(data): # ฟังก์ชันสำหรับจัดการข้อมูลซ้ำ
        st.subheader("👥 2. Duplicate Data") # หัวข้อย่อย
        exact_dups = data.duplicated() # ตรวจหาแถวที่ซ้ำกัน 100%
        exact_dup_count = exact_dups.sum() # นับจำนวนแถวที่ซ้ำ
        if exact_dup_count > 0: # ถ้าพบข้อมูลซ้ำ
            st.warning(f"พบข้อมูลซ้ำ 100% จำนวน {exact_dup_count:,} แถว") # แสดงคำเตือนพร้อมจำนวน
            st.dataframe(data[exact_dups]) # แสดงตัวอย่างข้อมูลที่ซ้ำ
            data = data.drop_duplicates() # ลบข้อมูลซ้ำออกจาก DataFrame
            st.success(f"ลบข้อมูลซ้ำแล้ว: เหลือ {len(data):,} แถว") # แสดงข้อความแจ้งว่าลบสำเร็จ
        else: # ถ้าไม่พบข้อมูลซ้ำ
            st.info("ไม่พบ Exact Duplicate ในข้อมูลนี้") # แสดงข้อความแจ้ง
        return data

    def handle_inconsistent_data(data): # ฟังก์ชันสำหรับจัดการข้อมูลที่ไม่สอดคล้องกัน
        st.subheader("🔄 3. Inconsistent Data") # หัวข้อย่อย
        st.write("##### ก่อนแก้ไข Inconsistent Values (Unique values for categorical columns)") # หัวข้อย่อยแสดงค่าก่อนแก้ไข
        cat_cols = ['Region', 'Product_Variant', 'Channel'] # กำหนดคอลัมน์ประเภท Categorical
        for col in cat_cols: # วนลูปในแต่ละคอลัมน์
            unique_vals = data[col].unique() # ดึงค่า unique ของคอลัมน์
            st.write(f"**📌 {col} ({len(unique_vals)} ค่า):**") # แสดงชื่อคอลัมน์และจำนวนค่า unique
            st.write(unique_vals) # แสดงค่า unique

        st.write("##### กำลังแก้ไข Inconsistent Values...") # ข้อความแจ้งว่ากำลังแก้ไข

        # 1. Standardize Region Column
        data['Region'] = data['Region'].str.strip().str.lower() # แปลงเป็นตัวพิมพ์เล็กและลบช่องว่าง
        region_mapping = { # กำหนดการแมปค่าที่ไม่สอดคล้องกันของ Region
            'th-central': 'TH-Central', 'th central': 'TH-Central',
            'thailand central': 'TH-Central', 'thailand-central': 'TH-Central',
            'thailand': 'TH-Central',
            'usa-east': 'USA-East', 'us east': 'USA-East',
            'united states east': 'USA-East', 'u.s.a.': 'USA-East',
            'europe-eu': 'Europe-EU', 'eu': 'Europe-EU',
            'europe': 'Europe-EU', 'european union': 'Europe-EU',
            'asia-pacific': 'Asia-Pacific', 'asia-pac': 'Asia-Pacific',
            'apac': 'Asia-Pacific', 'asia pacific': 'Asia-Pacific'
        }
        data['Region'] = data['Region'].replace(region_mapping) # แทนที่ค่าตาม mapping
        data['Region'] = data['Region'].str.upper() # แปลงเป็นตัวพิมพ์ใหญ่ทั้งหมด

        # 2. Standardize Product_Variant Column
        data['Product_Variant'] = data['Product_Variant'].str.strip().str.lower() # แปลงเป็นตัวพิมพ์เล็กและลบช่องว่าง
        product_variant_mapping = { # กำหนดการแมปค่าที่ไม่สอดคล้องกันของ Product_Variant
            'original blue': 'Original Blue', 'original  blue': 'Original Blue',
            'krating daeng 250': 'Krating Daeng 250',
            'red edition': 'Red Edition',
            'sugarfree': 'Sugarfree', 'sugar free': 'Sugarfree',
            'sugarfree ': 'Sugarfree', 'sugar-free': 'Sugarfree',
            'tropical edition': 'Tropical Edition', 'tropical  edition': 'Tropical Edition',
            'tropical': 'Tropical Edition',
        }
        data['Product_Variant'] = data['Product_Variant'].replace(product_variant_mapping) # แทนที่ค่าตาม mapping

        # 3. Standardize Channel Column
        data['Channel'] = data['Channel'].str.strip().str.lower() # แปลงเป็นตัวพิมพ์เล็กและลบช่องว่าง
        channel_mapping = { # กำหนดการแมปค่าที่ไม่สอดคล้องกันของ Channel
            'social media': 'Social Media', 'social_media': 'Social Media',
            'tv ad': 'TV Ad', 'tv ads': 'TV Ad',
            'tv advertisement': 'TV Ad', 'television ad': 'TV Ad',
            'in-store promo': 'In-store Promo',
            'f1 sponsorship': 'F1 Sponsorship',
            'extreme sports': 'Extreme Sports'
        }
        data['Channel'] = data['Channel'].replace(channel_mapping) # แทนที่ค่าตาม mapping
        # Ensure consistent casing for any remaining channels not in mapping
        data['Channel'] = data['Channel'].apply(lambda x: x.title() if isinstance(x, str) else x) # ทำให้ตัวอักษรแรกเป็นตัวพิมพ์ใหญ่สำหรับคำอื่นๆ

        # Convert Date to datetime (from notebook)
        data['Date'] = pd.to_datetime(data['Date'], format='mixed') # แปลงคอลัมน์ 'Date' เป็นรูปแบบ datetime

        st.success("แก้ไข Inconsistent Values สำเร็จแล้ว!") # แสดงข้อความแจ้งว่าแก้ไขสำเร็จ
        st.write("##### หลังแก้ไข Inconsistent Values (Unique values for categorical columns)") # หัวข้อย่อยแสดงค่าหลังแก้ไข
        for col in cat_cols: # วนลูปในแต่ละคอลัมน์อีกครั้ง
            unique_vals = data[col].unique() # ดึงค่า unique หลังแก้ไข
            st.write(f"**📌 {col} ({len(unique_vals)} ค่า):**") # แสดงชื่อคอลัมน์และจำนวนค่า unique
            st.write(unique_vals) # แสดงค่า unique หลังแก้ไข
        return data

    def handle_missing_data(data): # ฟังก์ชันสำหรับจัดการข้อมูลที่หายไป (Missing Data)
        st.subheader("📭 4. Missing Data") # หัวข้อย่อย
        missing_count = data.isnull().sum() # นับจำนวนค่า Missing ในแต่ละคอลัมน์
        st.write("##### จำนวน Missing Values ก่อนแก้ไข:") # หัวข้อย่อยแสดงค่า Missing ก่อนแก้ไข
        if missing_count.sum() > 0: # ถ้ามีค่า Missing
            st.dataframe(missing_count[missing_count > 0]) # แสดงคอลัมน์ที่มีค่า Missing

            median_marketing = data['Marketing_Spend'].median() # คำนวณค่ามัธยฐานของ Marketing_Spend
            data['Marketing_Spend'] = data['Marketing_Spend'].fillna(median_marketing) # เติมค่า Missing ด้วยค่ามัธยฐาน
            st.info(f'✅ Marketing_Spend: เติมด้วย Median = {median_marketing:,.2f}') # แสดงข้อความแจ้ง

            median_score = data['Customer_Score'].median() # คำนวณค่ามัธยฐานของ Customer_Score
            data['Customer_Score'] = data['Customer_Score'].fillna(median_score) # เติมค่า Missing ด้วยค่ามัธยฐาน
            st.info(f'✅ Customer_Score: เติมด้วย Median = {median_score}') # แสดงข้อความแจ้ง

            st.success("แก้ไข Missing Values สำเร็จแล้ว!") # แสดงข้อความแจ้งว่าแก้ไขสำเร็จ
            st.write("##### จำนวน Missing Values หลังแก้ไข:") # หัวข้อย่อยแสดงค่า Missing หลังแก้ไข
            st.write(f"รวม {data.isnull().sum().sum()} ค่า (ควรเป็น 0)") # แสดงจำนวนรวมของค่า Missing (ควรเป็น 0)
        else: # ถ้าไม่มีค่า Missing
            st.info("ไม่พบ Missing Data ในข้อมูลนี้") # แสดงข้อความแจ้ง
        return data

    def handle_noisy_data(data): # ฟังก์ชันสำหรับจัดการข้อมูลผิดพลาด (Noisy Data)
        st.subheader("📢 5. Noisy Data") # หัวข้อย่อย
        st.write("##### ตรวจสอบ Business Logic ก่อนแก้ไข:") # หัวข้อย่อยแสดงการตรวจสอบ
        neg_price = data[data['Unit_Price'] <= 0] # ตรวจสอบราคาที่น้อยกว่าหรือเท่ากับ 0
        neg_units = data[data['Units_Sold'] <= 0] # ตรวจสอบจำนวนที่ขายน้อยกว่าหรือเท่ากับ 0
        neg_mkt = data[data['Marketing_Spend'] < 0] # ตรวจสอบงบการตลาดที่น้อยกว่า 0
        bad_score = data[(data['Customer_Score'] < 1) | (data['Customer_Score'] > 10)] # ตรวจสอบ Customer_Score ที่ไม่อยู่ในช่วง 1-10

        found_noisy = False # ตัวแปรสถานะว่าพบ Noisy Data หรือไม่
        if len(neg_price) > 0: # ถ้าพบราคาติดลบ
            st.warning(f"❌ Unit_Price ≤ 0  : {len(neg_price):,} แถว (ราคาต้องเป็นบวก!)") # แสดงคำเตือน
            found_noisy = True
        if len(neg_units) > 0: # ถ้าพบจำนวนที่ขายติดลบ
            st.warning(f"❌ Units_Sold ≤ 0  : {len(neg_units):,} แถว (ขายไม่ได้ติดลบ!)") # แสดงคำเตือน
            found_noisy = True
        if len(neg_mkt) > 0: # ถ้าพบงบการตลาดติดลบ
            st.warning(f"❌ Marketing < 0   : {len(neg_mkt):,} แถว (งบต้องไม่ติดลบ!)") # แสดงคำเตือน
            found_noisy = True
        if len(bad_score) > 0: # ถ้าพบ Customer_Score นอกช่วง
            st.warning(f"❌ Customer_Score ไม่ใช่ 1-10: {len(bad_score):,} แถว (คะแนนต้องอยู่ระหว่าง 1-10!)") # แสดงคำเตือน
            found_noisy = True

        if found_noisy: # ถ้าพบ Noisy Data
            initial_rows = len(data) # เก็บจำนวนแถวเริ่มต้น
            data = data[data['Unit_Price'] > 0] # กรองข้อมูลที่ราคาเป็นบวก
            data = data[data['Units_Sold'] > 0] # กรองข้อมูลที่จำนวนขายเป็นบวก
            data = data[data['Marketing_Spend'] >= 0] # กรองข้อมูลที่งบการตลาดไม่ติดลบ
            data = data[(data['Customer_Score'] >= 1) & (data['Customer_Score'] <= 10)] # กรองข้อมูลที่ Customer_Score อยู่ในช่วง 1-10
            st.success(f"แก้ไข Noisy Data สำเร็จแล้ว: ลบไป {initial_rows - len(data):,} แถว") # แสดงข้อความแจ้งว่าแก้ไขสำเร็จและจำนวนแถวที่ถูกลบ
        else: # ถ้าไม่พบ Noisy Data
            st.info("ไม่พบ Noisy Data ที่ขัดแย้งกับ Business Logic") # แสดงข้อความแจ้ง
        return data

    def perform_outlier_analysis(data): # ฟังก์ชันสำหรับตรวจจับ Outlier
        st.subheader("📐 6. Outlier Detection & Treatment") # หัวข้อย่อย
        st.markdown("##### ตรวจสอบ Outliers ด้วย Boxplot") # หัวข้อย่อยแสดงการตรวจสอบ

        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist() # ดึงชื่อคอลัมน์ที่เป็นตัวเลข
        # Customer_Score is already handled in noisy data and is expected to be 1-10, so removing it from outlier analysis
        if 'Customer_Score' in numeric_cols: # ถ้า Customer_Score อยู่ในคอลัมน์ตัวเลข
            numeric_cols.remove('Customer_Score') # ลบออกจากรายการ เพราะถูกจัดการแล้วในขั้นตอน Noisy Data

        if numeric_cols: # ถ้ามีคอลัมน์ตัวเลขให้วิเคราะห์ Outlier
            for col in numeric_cols: # วนลูปในแต่ละคอลัมน์ตัวเลข
                fig, ax = plt.subplots(figsize=(8, 2)) # สร้าง figure และ axes สำหรับ Boxplot
                sns.boxplot(x=data[col], ax=ax) # สร้าง Boxplot
                ax.set_title(f'Boxplot of {col}') # ตั้งชื่อกราฟ
                st.pyplot(fig) # แสดงกราฟใน Streamlit
                plt.close(fig) # ปิด figure เพื่อป้องกันปัญหาการแสดงผล

            st.markdown("""
            **หมายเหตุเกี่ยวกับการจัดการ Outliers:**
            ใน Workshop นี้ เราได้สังเกตว่าการใช้ `winsorize` อาจจะทำให้ Business Logic ของข้อมูลเปลี่ยนไป (เช่น `Units_Sold` ที่ถูกปรับค่าอาจไม่สะท้อนยอดขายจริง)
            ดังนั้น ในกรณีนี้ เราจะเลือก **ไม่ปรับ Outliers** ในขั้นตอนนี้ เพื่อรักษาความถูกต้องของข้อมูลตามบริบททางธุรกิจ อย่างไรก็ตาม ในสถานการณ์จริง การจัดการ Outlier ต้องพิจารณาจากบริบทและเป้าหมายการวิเคราะห์อย่างรอบคอบ.
            """) # คำอธิบายเกี่ยวกับการจัดการ Outlier
        else: # ถ้าไม่มีคอลัมน์ตัวเลข
            st.info("ไม่พบคอลัมน์ตัวเลขสำหรับวิเคราะห์ Outliers") # แสดงข้อความแจ้ง
        return data

    st.sidebar.header("เลือกขั้นตอน Data Cleaning") # หัวข้อใน Sidebar
    do_explore = st.sidebar.checkbox("1. Data Exploration", value=True) # Checkbox สำหรับ Data Exploration
    do_duplicates = st.sidebar.checkbox("2. Handle Duplicate Data", value=True) # Checkbox สำหรับ Duplicate Data
    do_inconsistent = st.sidebar.checkbox("3. Handle Inconsistent Data", value=True) # Checkbox สำหรับ Inconsistent Data
    do_missing = st.sidebar.checkbox("4. Handle Missing Data", value=True) # Checkbox สำหรับ Missing Data
    do_noisy = st.sidebar.checkbox("5. Handle Noisy Data", value=True) # Checkbox สำหรับ Noisy Data
    do_outlier = st.sidebar.checkbox("6. Outlier Detection", value=True) # Checkbox สำหรับ Outlier Detection

    st.markdown("---  ") # เส้นแบ่ง

    if st.button("Start Cleaning"): # ปุ่มสำหรับเริ่มกระบวนการ Data Cleaning
        st.write("### กำลังดำเนินการ Data Cleaning...") # ข้อความแจ้งว่ากำลังดำเนินการ
        if do_explore: # ถ้าเลือก Data Exploration
            df = perform_data_exploration(df)
        if do_duplicates: # ถ้าเลือก Duplicate Data
            df = handle_duplicate_data(df)
        if do_inconsistent: # ถ้าเลือก Inconsistent Data
            df = handle_inconsistent_data(df)
        if do_missing: # ถ้าเลือก Missing Data
            df = handle_missing_data(df)
        if do_noisy: # ถ้าเลือก Noisy Data
            df = handle_noisy_data(df)
        if do_outlier: # ถ้าเลือก Outlier Detection
            df = perform_outlier_analysis(df)

        st.markdown("---  ") # เส้นแบ่ง
        st.subheader("✅ 7. Cleaned Data Summary") # หัวข้อสรุปผล
        st.write(f"#### ก่อนทำความสะอาด: {df_raw.shape[0]:,} แถว, {df_raw.shape[1]} คอลัมน์") # แสดงขนาดข้อมูลก่อนทำความสะอาด
        st.write(f"#### หลังทำความสะอาด: {df.shape[0]:,} แถว, {df.shape[1]} คอลัมน์") # แสดงขนาดข้อมูลหลังทำความสะอาด

        st.write("### Cleaned Data (First 5 Rows)") # หัวข้อแสดงข้อมูลที่ทำความสะอาดแล้ว
        st.dataframe(df.head()) # แสดง 5 แถวแรกของข้อมูลที่ทำความสะอาดแล้ว

        # --- Download Cleaned Data ---
        csv_buffer = df.to_csv(index=False).encode('utf-8') # แปลง DataFrame เป็น CSV ในรูปแบบ byte
        st.download_button( # ปุ่มสำหรับดาวน์โหลดข้อมูล
            label="Download Cleaned Data as CSV", # ข้อความบนปุ่ม
            data=csv_buffer, # ข้อมูลที่จะให้ดาวน์โหลด
            file_name="redbull_clean.csv", # ชื่อไฟล์เมื่อดาวน์โหลด
            mime="text/csv", # ประเภทของไฟล์
            help="Click to download the cleaned dataset." # คำแนะนำเพิ่มเติม
        )
else: # ถ้ายังไม่ได้อัปโหลดไฟล์
    st.info("Please upload a CSV file to begin data cleaning.") # แสดงข้อความให้ผู้ใช้อัปโหลดไฟล์
