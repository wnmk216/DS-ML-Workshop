import streamlit as st
import io
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="ระบบจำแนกความเหมาะสมในการเก็บเกี่ยวมะพร้าวน้ำหอมอัตโนมัติ",
    page_icon="🥥",
    layout="centered",
    initial_sidebar_state="auto"
)

# 1. โหลด CSS ปรับแต่งหน้าเว็บ (ไว้นอกสุด)
st.markdown("""
<style>
.stApp {
    background-color: #F8FFF7;
}

/* Header */
.main-title {
    font-size: 28px;
    color: #1B5E20;
    font-weight: bold;
    text-align: left;
    margin-bottom: 5px;
}

.sub-title {
    text-align: left;
    color: #558B2F;
    font-size: 16px;
    margin-bottom: 20px;
}

/* Card */
.block-container {
    padding-top: 2rem;
}

div[data-testid="stVerticalBlock"] {
    border-radius: 15px;
}

/* Button */
.stButton>button {
    background: #2E7D32;
    color: white;
    border-radius: 10px;
    border: none;
    font-size: 18px;
    height: 55px;
    width: 100%;
}

.stButton>button:hover {
    background: #1B5E20;
    color: white;
}

/* Upload */
[data-testid="stFileUploader"] {
    border: 2px dashed #81C784;
    border-radius: 15px;
    padding: 15px;
}

/* Success */
.stAlert {
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    font-size: 15px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# 2. จัดวาง Layout ส่วนหัว (Header)
col1, col2 = st.columns([1, 3])

with col1:
    # แสดงรูปภาพในโฟลเดอร์ images (เช็คให้ชัวร์ว่ามีไฟล์ images/coconut2.png อยู่จริง)
    # ใช้ use_container_width=True เพื่อให้รูปปรับขนาดตามคอลัมน์อัตโนมัติ
    try:
        st.image("images/coconut2.png", use_container_width=True)
    except Exception:
        # หากไม่พบไฟล์รูปภาพ จะแสดงรูปมะพร้าวสำรองเพื่อป้องกัน Error
        st.image("https://cdn-icons-png.flaticon.com/512/2909/2909784.png", use_container_width=True)

with col2:
    # แสดงชื่อระบบเคียงข้างกับรูปภาพ
    st.markdown('<p class="main-title">ระบบจำแนกความเหมาะสมในการเก็บเกี่ยวมะพร้าวน้ำหอมอัตโนมัติ</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">ตรวจสอบคุณภาพความสุกของมะพร้าวน้ำหอมด้วยเทคโนโลยี AI</p>', unsafe_allow_html=True)


# ==========================================================
# 3. โค้ดด้านล่างนี้ดึงออกมาด้านนอก เพื่อให้แสดงผลเต็มความกว้างหน้าจอปกติ
# ==========================================================

# Load the trained YOLO model
model_path = "model/best.pt"
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"ไม่สามารถโหลด Model ได้: {e} (กรุณาตรวจสอบว่ามีไฟล์ที่ {model_path} จริง)")

# Thai text for titles and labels
upload_prompt = "อัปโหลดไฟล์วิดีโอเพื่อตรวจจับมะพร้าว"
detect_button_text = "ตรวจจับมะพร้าวในวิดีโอ"
footer_text = "ระบบนี้ได้ทุนวิจัยสนับสนุนจากมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ"

st.write("---")

# Video uploader
uploaded_file = st.file_uploader(upload_prompt, type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file, format=uploaded_file.type) # Display original video

    if st.button(detect_button_text):
        with st.spinner('กำลังตรวจจับมะพร้าวในวิดีโอ... โปรดรอสักครู่'):
            # Create a temporary file to save the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video_file:
                tmp_video_file.write(uploaded_file.read())
                video_path = tmp_video_file.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("ไม่สามารถเปิดไฟล์วิดีโอได้")
                os.remove(video_path)
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                codec = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4

                # Create a temporary file for the output video
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output_video_file:
                    output_video_path = tmp_output_video_file.name

                out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))
                
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0:
                    st.warning("ไม่พบเฟรมในวิดีโอ")
                    cap.release()
                    os.remove(video_path)
                    os.remove(output_video_path)
                    st.stop()

                progress_bar = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Get the annotated frame
                    results = model.predict(frame, conf=0.25, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # บันทึกเฟรมที่ตรวจจับแล้วลงในไฟล์วิดีโอใหม่
                    out.write(annotated_frame)

                    frame_count += 1
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)

                cap.release()
                out.release()
                
                # แสดงผลวิดีโอหลังตรวจจับ
                st.success("ตรวจจับมะพร้าวในวิดีโอเสร็จสมบูรณ์!")
                st.subheader("ผลการตรวจจับ:")
                
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()
                
                st.video(video_bytes)
                
                st.download_button(
                    label="Download Result",
                    data=video_bytes,
                    file_name="result.mp4",
                    mime="video/mp4"
                )
                
                st.write(f"FPS = {fps} | Frames = {frame_count}")
                
                # ลบไฟล์ชั่วคราว
                os.remove(video_path)
                os.remove(output_video_path)

st.write("---")
st.markdown(f"<p class='footer'>{footer_text}</p>", unsafe_allow_html=True)
