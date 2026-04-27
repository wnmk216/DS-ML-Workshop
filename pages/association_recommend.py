import streamlit as st # Import library Streamlit สำหรับสร้าง Web Application
import pandas as pd # Import library Pandas สำหรับจัดการข้อมูล DataFrame

# --- 1. Load Association Rules ---
# ตรวจสอบให้แน่ใจว่า 'Model_association_rules.csv' อยู่ในไดเรกทอรีเดียวกันกับแอป Streamlit นี้
@st.cache_data # Decorator สำหรับ Cache ข้อมูล เพื่อให้ Streamlit โหลดข้อมูลเพียงครั้งเดียวเมื่อแอปเริ่มทำงาน
def load_rules():
    rules = pd.read_csv('model/Model_association_rules.csv') # โหลดไฟล์ CSV ที่บันทึกกฎความสัมพันธ์
    # แปลงคอลัมน์ 'antecedents' และ 'consequents' ที่เป็น string กลับไปเป็น frozenset
    # เพื่อให้สามารถเปรียบเทียบและใช้งานได้ง่ายขึ้นในฟังก์ชันแนะนำ
    rules['antecedents'] = rules['antecedents'].apply(lambda x: frozenset(item.strip() for item in x.split(', ')))
    rules['consequents'] = rules['consequents'].apply(lambda x: frozenset(item.strip() for item in x.split(', ')))
    return rules

df_rules_app = load_rules() # เรียกใช้ฟังก์ชันเพื่อโหลดกฎความสัมพันธ์เข้าสู่แอปพลิเคชัน

# --- 2. Extract Unique Items for Selection ---
# ดึงรายการทั้งหมดที่เป็นเอกลักษณ์จากกฎความสัมพันธ์ เพื่อใช้เป็นตัวเลือกใน Multiselect ของ Streamlit
all_items = set() # สร้าง Set เปล่าเพื่อเก็บรายการที่ไม่ซ้ำกัน
for s in df_rules_app['antecedents']: # วนลูปผ่าน Antecedents ทั้งหมด
    all_items.update(s) # เพิ่มรายการใน Antecedents เข้าไปใน Set
for s in df_rules_app['consequents']: # วนลูปผ่าน Consequents ทั้งหมด
    all_items.update(s) # เพิ่มรายการใน Consequents เข้าไปใน Set

unique_regions = sorted([item for item in all_items if item.startswith('Reg_')]) # กรองเฉพาะ Region และเรียงลำดับ
unique_products = sorted([item for item in all_items if item.startswith('Prod_')]) # กรองเฉพาะ Product และเรียงลำดับ
unique_channels = sorted([item for item in all_items if item.startswith('Chan_')]) # กรองเฉพาะ Channel และเรียงลำดับ

# --- 3. Recommendation Function ---
# ฟังก์ชันสำหรับให้คำแนะนำ โดยรับรายการที่ผู้ใช้เลือก (antecedents) และ DataFrame ของกฎความสัมพันธ์
def get_recommendations(user_selected_items: frozenset, rules_df: pd.DataFrame, top_n=5):
    potential_recommendations = [] # List สำหรับเก็บคำแนะนำที่เป็นไปได้

    for _, rule in rules_df.iterrows(): # วนลูปผ่านแต่ละกฎใน DataFrame ของกฎความสัมพันธ์
        rule_antecedents = rule['antecedents'] # ดึง Antecedents ของกฎปัจจุบัน
        rule_consequents = rule['consequents'] # ดึง Consequents ของกฎปัจจุบัน

        # ตรวจสอบว่ารายการที่ผู้ใช้เลือกมี Antecedents ของกฎทั้งหมดหรือไม่ (คือ ผู้ใช้เลือก Antecedents ของกฎนั้นๆ)
        if rule_antecedents.issubset(user_selected_items):
            # แนะนำรายการจาก Consequents ที่ผู้ใช้ยังไม่ได้เลือก
            for rec_item in rule_consequents:
                if rec_item not in user_selected_items:
                    potential_recommendations.append({ # เพิ่มรายการที่แนะนำ พร้อมค่า Confidence และ Lift
                        'item': rec_item,
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    })

    # เรียงลำดับคำแนะนำตามค่า Lift (จากมากไปน้อย) และ Confidence (จากมากไปน้อย) เพื่อให้คำแนะนำที่มีความสัมพันธ์แข็งแกร่งที่สุดขึ้นมาก่อน
    sorted_recs = sorted(potential_recommendations, key=lambda x: (x['lift'], x['confidence']), reverse=True)

    # ดึงรายการแนะนำที่ไม่ซ้ำกัน สูงสุดตามจำนวน top_n ที่กำหนด
    final_recs = [] # List สำหรับเก็บคำแนะนำสุดท้าย
    seen_items = set() # Set สำหรับตรวจสอบรายการที่ถูกแนะนำไปแล้ว เพื่อป้องกันการซ้ำซ้อน
    for rec in sorted_recs:
        if rec['item'] not in seen_items: # ถ้ายังไม่เคยแนะนำรายการนี้
            final_recs.append(rec['item']) # เพิ่มลงในรายการแนะนำสุดท้าย
            seen_items.add(rec['item']) # เพิ่มลงใน Set ของรายการที่ถูกแนะนำไปแล้ว
        if len(final_recs) >= top_n: # ถ้าได้จำนวนคำแนะนำครบตาม top_n แล้ว
            break # ออกจากลูป

    return final_recs # ส่งคืนรายการคำแนะนำ

# --- 4. Streamlit App Layout ---
st.set_page_config(layout="wide") # ตั้งค่าเลย์เอาต์ของ Streamlit ให้กว้างเต็มหน้าจอ
st.title("🛒 Behavioral Association Recommendation Engine") # ตั้งชื่อแอปพลิเคชัน
st.markdown("--- ให้ระบบแนะนำสินค้า/ช่องทาง/ภูมิภาคอื่น ๆ ที่ลูกค้ามีแนวโน้มจะสนใจ จากกฎความสัมพันธ์ ---") # เพิ่มข้อความอธิบาย

st.subheader("เลือกสิ่งที่ลูกค้ากำลังสนใจอยู่ (Antecedents):") # หัวข้อย่อยสำหรับส่วนที่ผู้ใช้เลือกข้อมูล

# Multiselect component สำหรับเลือกภูมิภาค
selected_regions = st.multiselect(
    "เลือกภูมิภาค (Region)", # ข้อความแสดงผล
    options=unique_regions, # ตัวเลือกจาก unique_regions ที่ดึงมา
    default=[] # ค่าเริ่มต้นเป็น List ว่าง
)

# Multiselect component สำหรับเลือกประเภทสินค้า
selected_products = st.multiselect(
    "เลือกประเภทสินค้า (Product Variant)",
    options=unique_products,
    default=[]
)

# Multiselect component สำหรับเลือกช่องทาง
selected_channels = st.multiselect(
    "เลือกช่องทาง (Channel)",
    options=unique_channels,
    default=[]
)

# รวมรายการที่ผู้ใช้เลือกทั้งหมดเข้าด้วยกัน และแปลงเป็น frozenset เพื่อใช้เป็น Antecedents ในฟังก์ชันแนะนำ
user_input_items = frozenset(selected_regions + selected_products + selected_channels)

# ปุ่มสำหรับเรียกใช้การแนะนำ
if st.button("💡 แนะนำ!"):
    if user_input_items: # ถ้าผู้ใช้เลือกรายการอย่างน้อยหนึ่งรายการ
        st.subheader("ผลการแนะนำ (Consequents):") # หัวข้อย่อยสำหรับผลการแนะนำ
        recommendations = get_recommendations(user_input_items, df_rules_app) # เรียกใช้ฟังก์ชันแนะนำ

        if recommendations: # ถ้ามีคำแนะนำ
            st.success("ระบบแนะนำรายการต่อไปนี้:") # แสดงข้อความสำเร็จ
            for i, rec in enumerate(recommendations): # วนลูปแสดงผลคำแนะนำ
                # แสดงผลคำแนะนำ โดยแทนที่ Prefix ให้เป็นข้อความที่อ่านง่ายขึ้น
                st.write(f"**{i+1}. {rec.replace('Reg_', 'Region: ').replace('Prod_', 'Product: ').replace('Chan_', 'Channel: ')}**")
        else:
            st.info("ไม่พบกฎการแนะนำสำหรับรายการที่คุณเลือก โปรดลองเลือกรายการอื่น ๆ") # ถ้าไม่พบคำแนะนำ
    else:
        st.warning("กรุณาเลือกอย่างน้อยหนึ่งรายการเพื่อรับคำแนะนำ") # แจ้งเตือนถ้าผู้ใช้ไม่ได้เลือกอะไรเลย

st.markdown("---") # เส้นคั่น
st.markdown("**ข้อมูลเพิ่มเติม:** \n*   `Lift` > 1: บ่งชี้ความสัมพันธ์เชิงบวก \n*   `Confidence`: ความน่าจะเป็นที่ผู้ซื้อจะซื้อ Consequent ถ้าซื้อ Antecedent") # ข้อมูลเพิ่มเติมเกี่ยวกับ Metrics