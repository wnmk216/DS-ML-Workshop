import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(layout="wide", page_title="Logistics Service Time Prediction App")

st.title("🚛 Logistics Service Time Prediction & Scheduling")
st.write("แอปลิเคชันนี้ช่วยพยากรณ์เวลาบริการรถบรรทุก และจัดตารางการเข้าคิวเพื่อประสิทธิภาพสูงสุด")

# --- 1. Load Model (with caching) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('service_time_model.pkl')
        st.success("✅ โหลดโมเดล 'service_time_model.pkl' สำเร็จแล้ว!")
        return model
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล 'service_time_model.pkl' กรุณาตรวจสอบว่าได้รันเซลล์ที่บันทึกโมเดลแล้ว")
        st.stop()
loaded_model = load_model()

# --- 2. User Input for Unseen Data (including Company Name) ---
st.header("📝 ข้อมูลรถบรรทุกที่ต้องการพยากรณ์")
st.write("คุณสามารถแก้ไขข้อมูลรถบรรทุกที่จะนำไปทำนาย หรือเพิ่ม/ลบแถวได้โดยตรงในตารางด้านล่าง")

# Initial unseen data (from previous notebook state, with Company_Name)
initial_unseen_data = pd.DataFrame({
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

# Make the DataFrame editable in Streamlit
loaded_unseen_data = st.data_editor(
    initial_unseen_data,
    key="unseen_data_editor",
    num_rows="dynamic", # Allows adding/deleting rows
    hide_index=True
)

if loaded_unseen_data.empty:
    st.warning("กรุณาเพิ่มข้อมูลรถบรรทุกอย่างน้อยหนึ่งแถวเพื่อทำการพยากรณ์")
    st.stop()

# Ensure boolean columns are correctly typed after editing
for col in ['Truck_Type_4-Wheel', 'Truck_Type_6-Wheel', 'Operation_Type_Pickup', 'Weather_Rain', 'Work_Shift_Night']:
    if col in loaded_unseen_data.columns:
        loaded_unseen_data[col] = loaded_unseen_data[col].astype(bool)


# --- 3. Prediction Logic ---
if st.button("ทำการพยากรณ์และจัดตารางคิว"): # Only predict when button is clicked
    st.subheader("🔮 ผลการพยากรณ์")

    # Separate features for prediction (remove Company_Name)
    X_unseen_for_prediction = loaded_unseen_data.drop(columns=['Company_Name'], errors='ignore')

    # Ensure column order matches training data if necessary (though get_dummies handles this well)
    # It's good practice to align columns if 'X' from training is available
    # For this example, assuming the column names and order in X_unseen_for_prediction are correct
    # If not, a reindexing step would be needed: X_unseen_for_prediction = X_unseen_for_prediction[X.columns]

    try:
        predictions = loaded_model.predict(X_unseen_for_prediction)
    except ValueError as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}. ตรวจสอบว่าคอลัมน์ในข้อมูลที่ทำนายตรงกับที่โมเดลฝึกไว้")
        st.stop()

    prediction_results = loaded_unseen_data.copy()
    prediction_results['Predicted_Service_Min'] = predictions
    st.dataframe(prediction_results)

    # --- 4. Scheduling Logic ---
    st.header("🗓️ ตารางเวลาการจัดคิวรถบรรทุก")

    # User input for start time
    col1, col2 = st.columns(2)
    with col1:
        date_input = st.date_input("เลือกวันที่เริ่มต้นการประมวลผล", datetime.now().date())
    with col2:
        time_input = st.time_input("เลือกเวลาเริ่มต้นการประมวลผล", datetime.now().time())

    start_processing_time = datetime.combine(date_input, time_input)
    st.write(f"เวลาเริ่มต้นการประมวลผล: **{start_processing_time.strftime('%Y-%m-%d %H:%M:%S')}**")

    # Create scheduling DataFrame from prediction results
    scheduling_df = prediction_results.copy()

    # Sort by Predicted_Service_Min
    scheduling_df = scheduling_df.sort_values(by='Predicted_Service_Min').reset_index(drop=True)

    current_available_time = start_processing_time
    suggested_arrival_times = []
    completion_times = []

    for index, row in scheduling_df.iterrows():
        suggested_arrival_times.append(current_available_time)

        service_duration = timedelta(minutes=row['Predicted_Service_Min'])
        current_completion_time = current_available_time + service_duration
        completion_times.append(current_completion_time)

        current_available_time = current_completion_time

    scheduling_df['Suggested_Arrival_Time'] = suggested_arrival_times
    scheduling_df['Completion_Time'] = completion_times

    # Display the scheduling table
    display_cols = [
        'Company_Name', 'Staff_Count', 'Total_Cartons', 'SKU_Count',
        'Predicted_Service_Min', 'Suggested_Arrival_Time', 'Completion_Time'
    ]
    st.dataframe(scheduling_df[display_cols])

    # --- 5. Gantt Chart Visualization ---
    st.header("📊 Gantt Chart แสดงตารางคิวรถบรรทุก")

    # Prepare data for Gantt Chart
    scheduling_df['Task'] = scheduling_df['Company_Name'] + ' (เวลาบริการ: ' + scheduling_df['Predicted_Service_Min'].round(2).astype(str) + ' นาที)'

    fig_gantt = px.timeline(scheduling_df,
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

    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_layout(xaxis_title="เวลา", yaxis_title="ลำดับรถ")

    st.plotly_chart(fig_gantt, use_container_width=True)

st.markdown("**วิธีใช้งาน:** \n1. ตรวจสอบหรือแก้ไขข้อมูลรถบรรทุกที่ต้องการพยากรณ์ในตาราง \n2. กดปุ่ม 'ทำการพยากรณ์และจัดตารางคิว' \n3. เลือกวันที่และเวลาเริ่มต้นที่ต้องการ \n4. ดูผลลัพธ์ตารางและ Gantt Chart ที่แสดงขึ้นมา")

if st.button("🏠 กลับหน้าหลัก"):
    st.switch_page("app.py")
