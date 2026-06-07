import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="TCP Group: ระบบแนะนำสินค้า", page_icon="🛒")

st.title("🛒 **TCP Group:** ระบบแนะนำสินค้าจาก Market Basket Analysis")
st.markdown("--- ")

# Load the association rules
@st.cache_data
def load_rules():
    try:
        # Corrected file path: The file is in /content/, not /content/model/
        df_rules = pd.read_csv('model/Model_Association_Rules_Item.csv')
        # Convert string representations of lists/frozensets back to actual lists/sets for filtering
        df_rules['antecedents'] = df_rules['antecedents'].apply(lambda x: frozenset(eval(x)) if isinstance(x, str) and x.startswith('frozenset') else frozenset(eval(x)) if isinstance(x, str) and x.startswith('[') else frozenset([x]))
        df_rules['consequents'] = df_rules['consequents'].apply(lambda x: frozenset(eval(x)) if isinstance(x, str) and x.startswith('frozenset') else frozenset(eval(x)) if isinstance(x, str) and x.startswith('[') else frozenset([x]))
        return df_rules
    except FileNotFoundError:
        st.error("❌ เกิดข้อผิดพลาด: ไม่พบไฟล์ 'Model_Association_Rules_Item.csv' โปรดตรวจสอบให้แน่ใจว่าได้สร้างไฟล์และอยู่ในพาธที่ถูกต้อง")
        return pd.DataFrame()

df_rules = load_rules()

if not df_rules.empty:
    st.header("✨ แนะนำสินค้าที่น่าจะซื้อคู่กัน")
    st.write("เลือกสินค้าหลัก (Antecedent) เพื่อดูว่ามีสินค้าใดบ้างที่มักถูกซื้อคู่กัน (Consequent) และเหมาะสมกับการจัดโปรโมชัน")

    # Get all unique items from antecedents for selection
    all_antecedent_items = sorted(list(set(item for sublist in df_rules['antecedents'] for item in sublist)))

    selected_antecedent = st.selectbox(
        "📌 เลือกสินค้า:",
        all_antecedent_items,
        help="เลือกสินค้าเพื่อค้นหาคำแนะนำจากพฤติกรรมการซื้อของลูกค้า"
    )

    if selected_antecedent:
        # Filter rules where the selected item is in the antecedents
        recommendation_rules = df_rules[df_rules['antecedents'].apply(lambda x: selected_antecedent in x)]

        if not recommendation_rules.empty:
            st.subheader(f"✅ คำแนะนำสำหรับ: **{selected_antecedent}**")

            # Sort recommendations by lift and confidence
            recommendation_rules = recommendation_rules.sort_values(by=['lift', 'confidence'], ascending=False)

            # Display top recommendations
            for index, row in recommendation_rules.head(5).iterrows(): # Display top 5 for brevity
                antecedents_str = ", ".join(list(row['antecedents']))
                consequents_str = ", ".join(list(row['consequents']))
                st.markdown(f"- ลูกค้าที่ซื้อ **{antecedents_str}** มักจะซื้อ **{consequents_str}** ร่วมด้วย (Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

            st.markdown("--- ")
            st.subheader("📊 รายละเอียดคำแนะนำทั้งหมด")
            # Display the relevant columns for the recommendations
            st.dataframe(recommendation_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).style.format({
                'support': "{:.3f}",
                'confidence': "{:.3f}",
                'lift': "{:.3f}"
            }))
        else:
            st.info(f"💡 ไม่พบกฎความสัมพันธ์ที่แข็งแกร่งสำหรับ **{selected_antecedent}** ในฐานะสินค้าหลัก")

    st.markdown("--- ")
    st.subheader("🔍 สำรวจกฎความสัมพันธ์ทั้งหมด")
    st.write("นี่คือกฎความสัมพันธ์ทั้งหมดที่ได้จากการวิเคราะห์:")
    st.dataframe(df_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].style.format({
        'support': "{:.3f}",
        'confidence': "{:.3f}",
        'lift': "{:.3f}"
    }))

st.markdown("--- ")
st.markdown("ℹ️ หากต้องการรันแอปพลิเคชัน Streamlit นี้ ให้บันทึกโค้ดนี้เป็นไฟล์ Python (เช่น `app.py`) แล้วรันคำสั่ง `streamlit run app.py` ใน Terminal")


