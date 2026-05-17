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


# --- 3. Prediction Logic ---
if st.button("ทำการพยากรณ์และจัดตารางคิว"): # สร้างปุ่มสำหรับเริ่มการพยากรณ์และจัดตาราง
    st.subheader("🔮 ผลการพยากรณ์") # แสดงหัวข้อย่อยสำหรับผลการพยากรณ์

    # Separate features for prediction (remove Company_Name)
    # Now using selected_for_prediction which already has 'Select' dropped
    X_unseen_for_prediction = selected_for_prediction.drop(columns=['Company_Name'], errors='ignore') # ลบคอลัมน์ 'Company_Name' ออกก่อนนำไปทำนาย

    # Ensure column order matches training data (from the kernel state, X.columns)
    # The X.columns from kernel state is: ['Staff_Count', 'Total_Cartons', 'SKU_Count', 'Truck_Type_4-Wheel', 'Truck_Type_6-Wheel', 'Operation_Type_Pickup', 'Weather_Rain', 'Work_Shift_Night']
    expected_model_columns = ['Staff_Count', 'Total_Cartons', 'SKU_Count', 'Truck_Type_4-Wheel', 'Truck_Type_6-Wheel', 'Operation_Type_Pickup', 'Weather_Rain', 'Work_Shift_Night']
    
    # Check if all expected columns are in X_unseen_for_prediction
    missing_cols = set(expected_model_columns) - set(X_unseen_for_prediction.columns)
    if missing_cols:
        st.error(f"❌ ข้อมูลที่อัปโหลด/แก้ไขไม่มีคอลัมน์ที่จำเป็นสำหรับการทำนาย: {', '.join(missing_cols)} กรุณาตรวจสอบรูปแบบไฟล์")
        st.stop()
    
    # Reindex to ensure correct column order and handle any extra columns from user input
    X_unseen_for_prediction = X_unseen_for_prediction[expected_model_columns]


    try:
        predictions = loaded_model.predict(X_unseen_for_prediction) # ใช้โมเดลที่โหลดมาทำนายเวลาบริการ
    except ValueError as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}. ตรวจสอบว่าคอลัมน์ในข้อมูลที่ทำนายตรงกับที่โมเดลฝึกไว้") # จัดการข้อผิดพลาดหากเกิดปัญหาในการทำนาย
        st.stop()

    prediction_results = selected_for_prediction.copy() # คัดลอกข้อมูลที่ถูกเลือก
    prediction_results['Predicted_Service_Min'] = predictions # เพิ่มคอลัมน์ 'Predicted_Service_Min' ที่มีค่าการทำนาย
    st.dataframe(prediction_results) # แสดง DataFrame ที่มีผลการทำนาย

    # --- 4. Scheduling Logic ---
    st.header("🗓️ ตารางเวลาการจัดคิวรถบรรทุก") # แสดงหัวข้อสำหรับตารางเวลา

    # User input for start time
    col1, col2 = st.columns(2) # แบ่งหน้าจอเป็น 2 คอลัมน์
    with col1:
        date_input = st.date_input("เลือกวันที่เริ่มต้นการประมวลผล", datetime.now().date()) # ให้ผู้ใช้เลือกวันที่เริ่มต้น
    with col2:
        time_input = st.time_input("เลือกเวลาเริ่มต้นการประมวลผล", datetime.now().time()) # ให้ผู้ใช้เลือกเวลาเริ่มต้น

    start_processing_time = datetime.combine(date_input, time_input) # รวมวันที่และเวลาที่เลือกเข้าด้วยกัน
    st.write(f"เวลาเริ่มต้นการประมวลผล: **{start_processing_time.strftime('%Y-%m-%d %H:%M:%S')}**") # แสดงเวลาเริ่มต้นที่ผู้ใช้เลือก

    # Create scheduling DataFrame from prediction results
    scheduling_df = prediction_results.copy() # สร้าง DataFrame สำหรับการจัดตารางจากผลการทำนาย

    # Sort by Predicted_Service_Min
    scheduling_df = scheduling_df.sort_values(by='Predicted_Service_Min').reset_index(drop=True) # เรียงลำดับรถตามเวลาบริการที่ทำนายจากน้อยไปมาก

    current_available_time = start_processing_time # กำหนดเวลาเริ่มต้นสำหรับรถคันแรก
    suggested_arrival_times = [] # รายการสำหรับเก็บเวลาที่แนะนำให้รถมาถึง
    completion_times = [] # รายการสำหรับเก็บเวลาที่รถจะบริการเสร็จ

    for index, row in scheduling_df.iterrows(): # วนลูปเพื่อคำนวณเวลาสำหรับรถแต่ละคัน
        suggested_arrival_times.append(current_available_time) # เพิ่มเวลาที่แนะนำให้รถมาถึง

        service_duration = timedelta(minutes=row['Predicted_Service_Min']) # คำนวณระยะเวลาบริการ
        current_completion_time = current_available_time + service_duration # คำนวณเวลาที่รถจะบริการเสร็จ
        completion_times.append(current_completion_time) # เพิ่มเวลาที่บริการเสร็จ

        current_available_time = current_completion_time # อัปเดตเวลาสำหรับรถคันถัดไป

    scheduling_df['Suggested_Arrival_Time'] = suggested_arrival_times # เพิ่มคอลัมน์เวลาที่แนะนำให้รถมาถึง
    scheduling_df['Completion_Time'] = completion_times # เพิ่มคอลัมน์เวลาที่บริการเสร็จ

    # Display the scheduling table
    display_cols = [ # กำหนดคอลัมน์ที่จะแสดงในตาราง
        'Company_Name', 'Staff_Count', 'Total_Cartons', 'SKU_Count',
        'Predicted_Service_Min', 'Suggested_Arrival_Time', 'Completion_Time'
    ]
    st.dataframe(scheduling_df[display_cols]) # แสดงตารางการจัดคิว

    # --- 5. Gantt Chart Visualization ---
    st.header("📊 Gantt Chart แสดงตารางคิวรถบรรทุก") # แสดงหัวข้อสำหรับ Gantt Chart

    # Prepare data for Gantt Chart
    scheduling_df['Task'] = scheduling_df['Company_Name'] + ' (เวลาบริการ: ' + scheduling_df['Predicted_Service_Min'].round(2).astype(str) + ' นาที)' # สร้างคอลัมน์ 'Task' สำหรับแสดงใน Gantt Chart

    fig_gantt = px.timeline(scheduling_df, # สร้าง Gantt Chart
                            x_start="Suggested_Arrival_Time",
                            x_end="Completion_Time",
                            y="Task",
                            color="Predicted_Service_Min",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            title="ตารางเวลาการจัดคิวรถบรรทุก (Gantt Chart)",
                            labels={
                                "Suggested_Arrival_Time": "เวลาที่ควรมาถึง",
                                "Completion_Time": "เวลาที่บริการเสร็จ",
                                "Task": "รถบรรทุก/บริษัท",
                                "Predicted_Service_Min": "เวลาบริการที่คาดการณ์ (นาที)"
                            },
                            hover_name="Company_Name")

    fig_gantt.update_yaxes(autorange="reversed") # จัดเรียงแกน Y แบบย้อนกลับ
    fig_gantt.update_layout(xaxis_title="เวลา", yaxis_title="ลำดับรถ") # ตั้งชื่อแกน X และ Y

    st.plotly_chart(fig_gantt, use_container_width=True) # แสดง Gantt Chart

st.markdown("**วิธีใช้งาน:** \n1. อัปโหลดไฟล์ CSV หรือใช้ข้อมูลเริ่มต้น \n2. ตรวจสอบ แก้ไข และเลือกข้อมูลรถบรรทุกที่ต้องการพยากรณ์ในตาราง \n3. กดปุ่ม 'ทำการพยากรณ์และจัดตารางคิว' \n4. เลือกวันที่และเวลาเริ่มต้นที่ต้องการ \n5. ดูผลลัพธ์ตารางและ Gantt Chart ที่แสดงขึ้นมา") # แสดงคำแนะนำการใช้งาน

if st.button("🏠 กลับหน้าหลัก"): # สร้างปุ่ม 'กลับหน้าหลัก'
    st.switch_page("app.py") # เปลี่ยนหน้าไปยัง 'app.py'
