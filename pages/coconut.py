import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
import tempfile
import os

# Install opencv-python to ensure it's available in the Streamlit environment
# This addresses ImportError related to cv2
if 'opencv-python' not in st.session_state:
    st.session_state['opencv-python'] = True
    st.spinner('Installing OpenCV... please wait')
    os.system('pip install opencv-python -q')

# Set page configuration
st.set_page_config(
    page_title="ระบบจำแนกความเหมาะสมในการเก็บเกี่ยวมะพร้าวน้ำหอมอัตโนมัติ",
    page_icon="🥥",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load the trained YOLOv5 model
# Make sure the path to your best.pt model is correct
model_path = "model/best.pt"
model = YOLO(model_path)

# Thai text for titles and labels
app_title = "ระบบจำแนกความเหมาะสมในการเก็บเกี่ยวมะพร้าวน้ำหอมอัตโนมัติ"
upload_prompt = "อัปโหลดไฟล์วิดีโอเพื่อตรวจจับมะพร้าว"
detect_button_text = "ตรวจจับมะพร้าวในวิดีโอ"
footer_text = "ระบบนี้ได้ทุนวิจัยสนับสนุนจากมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ"

st.title(app_title)

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
                codec = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4, ensuring compatibility

                # Create a temporary file for the output video
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output_video_file:
                    output_video_path = tmp_output_video_file.name

                out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0: # Handle cases where total_frames might be 0
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

                    # Perform inference on the frame
                    results = model.predict(source=frame, conf=0.25, verbose=False)

                    # Get the annotated frame
                    # r.plot() returns a BGR numpy array which cv2.VideoWriter expects
                    for r in results:
                        annotated_frame = r.plot()
                        out.write(annotated_frame)

                    frame_count += 1
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)

                cap.release()
                out.release()

                st.success("ตรวจจับมะพร้าวในวิดีโอเสร็จสมบูรณ์!")
                st.subheader("ผลการตรวจจับ:")
                st.video(output_video_path)

                # Clean up temporary files
                os.remove(video_path)
                os.remove(output_video_path)

st.write("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{footer_text}</p>", unsafe_allow_html=True)
