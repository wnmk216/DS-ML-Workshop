import streamlit as st

st.set_page_config(page_title="MyApp", layout="wide")

st.title("🏠 หน้าหลัก ")
st.write("### Boot Camp: Data Science and Machine Learning")
st.info("7 Day Intensive Hands-on Workshop")


if st.button("การทำความสะอาดข้อมูล"):
    st.switch_page("pages/cleaning_app.py")
elif st.button("การแปลงข้อมูล"):
    st.switch_page("pages/transform_app.py")

