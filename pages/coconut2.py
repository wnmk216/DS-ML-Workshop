import streamlit as st
import io
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import base64

# ==========================================
# 1. การตั้งค่าหน้าเว็บ (Page Config)
# ==========================================
st.set_page_config(
    page_title="ระบบจำแนกความเหมาะสมในการเก็บเกี่ยวมะพร้าวน้ำหอมอัตโนมัติ",
    page_icon="🥥",
    layout="centered",
    initial_sidebar_state="auto"
)

# ==========================================
# 2. ฟังก์ชันแปลงภาพเป็น Base64 สำหรับพื้นหลัง
# ==========================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# พยายามโหลดภาพมะพร้าวจากเครื่อง ถ้าไม่มีจะใช้ภาพสำรองจากเว็บ
try:
    img_base64 = get_base64_of_bin_file("images/coconut2.png")
    bg_image_css = f"url('data:image/png;base64,{img_base64}')"
except Exception:
    # ภาพสำรองคุณภาพสูงจาก Unsplash ในกรณีที่ไม่พบไฟล์ในโฟลเดอร์
    bg_image_css = "url('https://images.unsplash.com/photo-1543157148-f79040713b1c?auto=format&fit=crop&q=80&w=1000')"

# ==========================================
# 3. สไตล์การตกแต่งหน้าเว็บด้วย CSS
# ==========================================
st.markdown(f"""
<style>
/* พื้นหลังของแอปพลิเคชันทั้งหมด */
.stApp {{
    background-color: #F8FFF7;
}}

/* ส่วนหัว (Header) ที่ต้องการทำภาพพื้นหลัง */
.header-bg {{
    background-image: linear-gradient(rgba(27, 94, 32, 0.8), rgba(27, 94, 32, 0.8)), {bg_image_css};
    background-size: cover;
    background-position: center;
    padding: 60px 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    margin-bottom: 30px;
}}

/* หัวข้อภาษาไทยบนภาพพื้นหลัง */
.main-title-white {{
    font-size: 30px;
    color: #FFFFFF;
    font-weight: bold;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
}}

/* คำอธิบายสั้นใต้หัวข้อ */
.sub-title-white {{
    color: #E8F5E9;
    font-size: 18px;
    margin-top: 10px;
    margin-bottom: 0;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6);
}}

/* ปรับปรุงคอนเทนเนอร์ */
.block-container {{
    padding-top: 2rem;
}}

/* ปุ่มตรวจจับสีเขียว */
.stButton>button {{
    background: #2E7D32;
    color: white;
    border-radius: 10px;
    border: none;
    font-size: 18px;
    height: 55px;
    width: 100%;
    font-weight: bold;
    transition: 0.3s;
}}

.stButton>button:hover {{
    background: #1B5E20;
    color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}}

/* ช่องอัปโหลดไฟล์ (ขอบประจุดสีเขียวอ่อน) */
[data-testid="stFileUploader"] {{
    border: 2px dashed #81C784;
    border-radius: 15px;
    padding: 15px;
    background-color: #FFFFFF;
}}

/* การ์ดความสำเร็จ */
.stAlert {{
    border-radius: 10px;
}}

/* ลายเซ็นต์ / ทุนวิจัยด้านล่างสุด */
.footer {{
    text-align: center;
    color: #757575;
    font-size: 14px;
    margin-top: 40px;
}}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. ส่วนแสดงผลบนหน้าเว็บ (UI Elements)
# ==========================================

# แสดงส่วนหัวที่มีพื้นหลังมะพร้าว (Header)
st.markdown("""
<div class="header-bg">
    <p class="main-title-white">🥥 ระบบจำแนกความเหมาะสมในการเก็บเกี่ยวมะพร้าวน้ำหอมอัตโนมัติ</p>
    <p class="sub-title-white">วิเคราะห์ระดับความแก่-อ่อนของมะพร้าวน้ำหอมผ่านวิดีโอด้วยเทคโนโลยีปัญญาประดิษฐ์</p>
</div>
""", unsafe_allow_html=True)

st.write("---")

# โหลด Model YOLOv5 / YOLOv8
model_path = "model/best.pt"
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"ไม่สามารถโหลดไฟล์โมเดลได้จากเส้นทาง '{model_path}': {e}")
    st.stop()

# คำจำกัดความภาษาไทยสำหรับปุ่มและอินพุต
upload_prompt = "กรุณาเลือกไฟล์วิดีโอเพื่อนำมาตรวจจับมะพร้าว (.mp4, .avi, .mov, .mkv)"
detect_button_text = "เริ่มการตรวจจับมะพร้าวในวิดีโอ"
footer_text = "ระบบนี้ได้ทุนวิจัยสนับสนุนจากมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ"

# ช่องสำหรับอัปโหลดวิดีโอ
uploaded_file = st.file_uploader(upload_prompt, type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # แสดงวิดีโอต้นฉบับที่อัปโหลดเข้ามา
    st.subheader("วิดีโอต้นฉบับ:")
    st.video(uploaded_file, format=uploaded_file.type)
    
    # ปุ่มสำหรับสั่งประมวลผลตรวจจับวัตถุ
    if st.button(detect_button_text):
        with st.spinner('ระบบกำลังตรวจจับและประมวลผลวิดีโอ... โปรดรอสักครู่'):
            
            # บันทึกวิดีโอที่อัปโหลดลงในไฟล์ชั่วคราว
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video_file:
                tmp_video_file.write(uploaded_file.read())
                video_path = tmp_video_file.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("ไม่สามารถเปิดไฟล์วิดีโอชั่วคราวเพื่อประมวลผลได้")
                os.remove(video_path)
            else:
                # ดึงคุณสมบัติของวิดีโอต้นฉบับ
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                codec = cv2.VideoWriter_fourcc(*'mp4v') # Codec มาตรฐานสำหรับ .mp4

                # สร้างไฟล์วิดีโอชั่วคราวสำหรับบันทึกผลลัพธ์
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output_video_file:
                    output_video_path = tmp_output_video_file.name

                out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))
                
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames == 0:
                    st.warning("ไม่พบภาพหรือเฟรมข้อมูลในไฟล์วิดีโอนี้")
                    cap.release()
                    os.remove(video_path)
                    os.remove(output_video_path)
                    st.stop()

                # หลอดแสดงสถานะความคืบหน้า (Progress Bar)
                progress_bar = st.progress(0)

                # ทำการวนลูปอ่านเฟรมของวิดีโอไปทีละเฟรม
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # ส่งเฟรมเข้าไปพยากรณ์ด้วย YOLO Model
                    results = model.predict(frame, conf=0.25, verbose=False)
                    annotated_frame = results[0].plot() # วาดกรอบสี่เหลี่ยมตรวจจับ (Bounding Box)
                    
                    # บันทึกเฟรมที่ตรวจจับแล้วลงในวิดีโอผลลัพธ์
                    out.write(annotated_frame)

                    # อัปเดตแถบเปอร์เซ็นต์ความคืบหน้า
                    frame_count += 1
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)

                # ปิดตัวแปรเชื่อมโยงไฟล์เมื่อประมวลผลเสร็จ
                cap.release()
                out.release()
                
                # แสดงผลลัพธ์วิดีโอหลังจากผ่านขั้นตอน AI ตรวจสอบแล้ว
                st.success("ตรวจจับมะพร้าวในวิดีโอเสร็จสมบูรณ์เรียบร้อยแล้ว!")
                st.subheader("วิดีโอผลการตรวจจับ:")
                
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()
                
                st.video(video_bytes)
                
                # ปุ่มดาวน์โหลดผลลัพธ์
                st.download_button(
                    label="ดาวน์โหลดวิดีโอผลลัพธ์ (Download Result)",
                    data=video_bytes,
                    file_name="detected_coconut.mp4",
                    mime="video/mp4"
                )
                
                # แสดงสถิติข้อมูลของไฟล์
                st.info(f"📊 รายละเอียดข้อมูลวิดีโอ: ความเร็วภาพ = {fps} FPS | จำนวนเฟรมทั้งหมด = {frame_count} เฟรม")
                
                # ลบไฟล์ขยะชั่วคราวออกจากระบบโฮสต์
                os.remove(video_path)
                os.remove(output_video_path)

# ==========================================
# 5. ส่วนท้ายหน้าเว็บ (Footer)
# ==========================================
st.write("---")
st.markdown(f"<p class='footer'>{footer_text}</p>", unsafe_allow_html=True)
