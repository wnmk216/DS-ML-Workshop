import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io # Needed for df.info() output redirection for st.text

# --- RB_COLORS (from the notebook for consistent branding) ---
RB_COLORS = ['#1565C0','#43A047','#EF6C00','#C62828','#6A1B9A',
             '#00838F','#AD1457','#558B2F','#E65100']

# --- Configuration for Streamlit Page ---
st.set_page_config(layout="wide", page_title="Red Bull Sales EDA App")

st.title("💡 EDA Web Application: Red Bull Sales Data")
st.write("""
ยินดีต้อนรับสู่แอปพลิเคชัน EDA สำหรับข้อมูลยอดขาย Red Bull!
คุณสามารถอัปโหลดไฟล์ `redbull_clean.csv` เพื่อเริ่มต้นสำรวจข้อมูลและสร้างกราฟเชิงโต้ตอบ.
""")

# --- Data Upload Section ---
uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ของคุณที่นี่ (เช่น: redbull_clean.csv)", type=["csv"])

# DataFrame variable
df = None

# --- Feature Engineering Function (from the notebook, cached for performance) ---
@st.cache_data # Cache this function to avoid re-running on every interaction
def perform_feature_engineering(data):
    # Ensure a copy to avoid SettingWithCopyWarning if data is a slice
    df_fe = data.copy()
    df_fe['Date']          = pd.to_datetime(df_fe['Date']) # แปลงคอลัมน์ 'Date' เป็นรูปแบบ datetime
    df_fe['Total_Revenue'] = df_fe['Unit_Price'] * df_fe['Units_Sold'] # สร้างคอลัมน์ 'Total_Revenue' จาก Unit_Price * Units_Sold
    df_fe['Month']         = df_fe['Date'].dt.month # ดึงเดือนจากคอลัมน์ 'Date'
    df_fe['Year']          = df_fe['Date'].dt.year # ดึงปีจากคอลัมน์ 'Date'
    df_fe['YearMonth']     = df_fe['Date'].dt.strftime('%Y-%m') # สร้างคอลัมน์ 'YearMonth' ในรูปแบบ 'YYYY-MM'
    df_fe['Delay_Group']   = pd.cut(
        df_fe['Logistics_Delay'],
        bins=[-1, 0, 2, 5, 10, 90],
        labels=['0 วัน', '1-2 วัน', '3-5 วัน', '6-10 วัน', '> 10 วัน'] # จัดกลุ่มคอลัมน์ 'Logistics_Delay' เป็น 5 ช่วง
    )
    return df_fe

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df_processed = perform_feature_engineering(df)
        st.success("อัปโหลดและประมวลผลไฟล์สำเร็จ!")

        st.write("### 📂 ภาพรวมข้อมูล")
        st.dataframe(df_processed.head())

        # Display basic info
        st.write("#### โครงสร้างข้อมูล:")
        buffer = io.StringIO()
        df_processed.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write("#### สรุปสถิติเชิงพรรณนา:")
        st.dataframe(df_processed.describe())

        # --- Sidebar for Navigation ---
        st.sidebar.header("เลือกการวิเคราะห์")
        analysis_type = st.sidebar.radio(
            "ประเภทการวิเคราะห์:",
            ["ภาพรวมข้อมูล", "Univariate Analysis", "Bivariate Analysis","Multivariate Analysis"]
        )

        if analysis_type == "ภาพรวมข้อมูล":
            st.write("คุณกำลังดูภาพรวมของชุดข้อมูลที่ประมวลผลแล้ว")

        elif analysis_type == "Univariate Analysis":
            st.header("📊 Univariate Analysis (การวิเคราะห์ตัวแปรเดี่ยว)")
            st.write("สำรวจการกระจายตัวของตัวแปรแต่ละตัว")

            # Example 1: Histogram for Customer_Score
            st.subheader("การกระจายตัวของคะแนนความพึงพอใจลูกค้า (Customer Score)")
            fig_hist_score = px.histogram(df_processed, x='Customer_Score',
                                        title='การกระจายตัวของคะแนนความพึงพอใจลูกค้า',
                                        labels={'Customer_Score': 'คะแนน (1-10)'},
                                        color_discrete_sequence=['pink']) # สร้างกราฟ Histogram สำหรับ Customer_Score
            st.plotly_chart(fig_hist_score, use_container_width=True)
            st.info("**Insight:** กราฟ Histogram นี้ช่วยให้เราเห็น 'ความพึงพอใจในการให้บริการ' หากกราฟเบ้ขวา (คะแนนสูงเยอะ) แสดงว่าพนักงานทำงานได้ดีมาก")

            # Example 2: Boxplot for Logistics_Delay
            st.subheader("การกระจายตัวของข้อมูลการจัดส่งสินค้า (Logistics Delay)")
            fig_boxplot_delay = px.box(df_processed, y='Logistics_Delay',
                                    title='การกระจายตัวของข้อมูลการจัดส่งสินค้า',
                                    labels={'Logistics_Delay': 'จำนวนวัน'}) # สร้างกราฟ Boxplot สำหรับ Logistics_Delay
            st.plotly_chart(fig_boxplot_delay, use_container_width=True)
            st.info("**Insight:** กราฟ boxplot นี้ช่วยให้เราเห็นข้อมูลการจัดส่งของบริษัท ส่วนใหญ่ใช้เวลาไม่่กี่วัน แต่มีสินค้าบางรายการที่ใช้เวลาการขนส่งนานมาก จำเป็นต้องตรวจสอบเพิ่มเติม")

            # Example 3: Bar Chart for Region (Plotly)
            st.subheader("จำนวนข้อมูลในแต่ละภูมิภาค (Region)")
            dr = df_processed['Region'].value_counts().reset_index() # นับจำนวนข้อมูลในแต่ละภูมิภาค
            dr.columns = ['Region', 'Count']
            fig_region_bar = px.bar(dr, x='Region', y='Count',
                                    title='จำนวนข้อมูลในแต่ละภูมิภาค',
                                    text='Count',
                                    template='plotly_white',
                                    color_discrete_sequence=RB_COLORS) # สร้างกราฟ Bar Chart สำหรับภูมิภาค
            fig_region_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_region_bar, use_container_width=True)
            st.info("> 💡 **Insight:** TH-NORTH และ TH-CENTRAL มี transaction สูงสุด — แสดงว่าตลาดไทยภาคเหนือและภาคกลางมีความแข็งแกร่ง ในขณะเดียวกัน ภาคใต้ ภาคตะวันออก กลับมีจำนวนข้อมูลการซื้อขายน้อยกว่ามาก ดังนั้น ในฐานะผู้บริหารควรพิจารณาหาสาเหตุเพิ่มเติม")

        elif analysis_type == "Bivariate Analysis":
            st.header("🔗 Bivariate Analysis (การวิเคราะห์ความสัมพันธ์ 2 ตัวแปร)")
            st.write("สำรวจความสัมพันธ์ระหว่างตัวแปรสองตัว")

            # Example 1: Scatter plot for Marketing Spend vs Units Sold
            st.subheader("ความสัมพันธ์ระหว่างงบการตลาดและยอดขาย (Marketing Spend vs Units Sold)")
            fig_scatter = px.scatter(df_processed, x='Marketing_Spend', y='Units_Sold',
                                     color='Product_Variant', # Color by Product_Variant
                                     size='Unit_Price',
                                     title='ความสัมพันธ์ระหว่างงบการตลาดและยอดขายแยกตามประเภทสินค้า',
                                     hover_name='Product_Variant',
                                     color_discrete_sequence=RB_COLORS) # สร้าง Scatter Plot แสดงความสัมพันธ์ Marketing_Spend กับ Units_Sold
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info("""
            **💡 Insight:** กราฟ Scatter Plot ช่วยให้เราเห็นว่าเงินที่เราจ่ายไปสำหรับสินค้าแต่ละตัว คุ้มค่าหรือไม่
            1. **ความสัมพันธ์ที่ไม่ชัดเจน** กราฟแสดงให้เห็นว่าไม่มีความสัมพันธ์เชิงเส้นตรงที่ชัดเจนระหว่างงบการตลาดกับยอดขาย
            2. **ประสิทธิภาพในการใช้จ่าย** บางจุดในกราฟอาจแสดงให้เห็นว่าบางภูมิภาคหรือสินค้าสามารถทำยอดขายได้สูงโดยใช้งบการตลาดไม่มากนัก
            3. **การประเมินความคุ้มค่า** เนื่องจากกราฟนี้เป็นแบบ Interactive และมีการแยกสีตามประเภทสินค้า ทำให้เราสามารถเจาะดูรายละเอียดได้ว่าในแต่ละสินค้า ใช้งบเท่าไหร่ และได้ยอดขายเท่าไหร่ เพื่อประเมินว่า การลงทุนนั้นคุ้มค่าหรือไม่ ในแต่ละตลาด
            **สรุป:** การเพิ่มงบการตลาดเพียงอย่างเดียวอาจไม่ใช่กลยุทธ์ที่ดีที่สุด ควรเน้นไปที่ประสิทธิภาพและกลยุทธ์การตลาดที่เหมาะสมกับสินค้าแต่ละชนิด
            """)

            # Example 2: Line chart for Monthly Sales Trend
            st.subheader("แนวโน้มยอดขายรวมของบริษัทรายเดือน")
            df_monthly = df_processed.set_index('Date').resample('ME')['Units_Sold'].sum().reset_index() # รวมยอดขายรายเดือน
            fig_line_trend = px.line(df_monthly, x='Date', y='Units_Sold',
                                     title='Trend ยอดขายรวมของบริษัทรายเดือน',
                                     markers=True,
                                     color_discrete_sequence=RB_COLORS) # สร้าง Line Chart แสดงแนวโน้มยอดขายรายเดือน
            fig_line_trend.update_traces(line_color=RB_COLORS[0]) # Use first color from RB_COLORS
            fig_line_trend.update_xaxes(dtick="M1", tickformat="%b\n%Y")
            st.plotly_chart(fig_line_trend, use_container_width=True)
            st.info("""
            **💡 Insight:** กราฟเส้นช่วยให้ฝ่ายวางแผนการผลิต (Production Planning)
            * **คาดการณ์ได้ว่าเดือนไหนความต้องการสินค้าจะพุ่งสูงขึ้น** สามารถสังเกตเห็นรูปแบบที่เกิดซ้ำกันทุกปีได้ เช่น เดือนพฤษภาคม (May) มียอดขายสูงสุดทั้งในปี 2023 และ 2024 ซึ่งอาจบ่งชี้ว่าเป็นช่วง High Season
            * **จุดสูงสุดและต่ำสุด** สามารถระบุเดือนที่มียอดขายสูงเป็นพิเศษ (Peak) และต่ำเป็นพิเศษ (Trough) ได้ เช่น เดือนกุมภาพันธ์ (Feb) มักมียอดขายค่อนข้างต่ำ ซึ่งอาจเป็น Low Season
            * **การเติบโตระหว่างปี** หากเปรียบเทียบยอดขายในเดือนเดียวกันระหว่างปี 2023 และ 2024 จะช่วยให้เห็นว่ายอดขายมีการเติบโตหรือไม่
            """)

            # Example 3:  Revenue by Region — ตลาดไหนทำรายได้สูงสุด?
            st.subheader("Revenue by Region — ตลาดไหนทำรายได้สูงสุด?")
            rev_region = (df_processed.groupby('Region')['Total_Revenue']
                .sum()
                .reset_index()
                .sort_values('Total_Revenue', ascending=True))
            print(rev_region)
            rev_region['Rev_M'] = rev_region['Total_Revenue'] / 1e6  #หารด้วย 1 ล้าน เพื่อแสดงเป็นหน่วยที่เข้าใจได้ง่าย
            rev_region['Share'] = (rev_region['Total_Revenue'] /
                        rev_region['Total_Revenue'].sum() * 100).round(1) #สร้างตัวแปรเพื่อแสดงสัดส่วนรายได้เป็นเปอร์เซ็นต์ (Percentage Share of Total Revenue) ของแต่ละภูมิภาาค
            
            #สร้างกราฟแท่งแสงสัดส่วนแต่ละภูมิภาคกับรายได้

            fig_revenue_bar = px.bar(
                rev_region, x='Rev_M', y='Region', orientation='h',
                text=rev_region.apply(lambda r: f'{r.Rev_M:.1f}M ({r.Share}%)', axis=1),
                color='Rev_M',
                color_continuous_scale='Blues',
                title='Total Revenue by Region (ล้าน)',
                template='plotly_white', height=500
            )
            fig_revenue_bar.update_traces(textposition='inside')
            fig_revenue_bar.update_coloraxes(showscale=False)
            fig_revenue_bar.update_xaxes(tickformat=".0f", nticks=10) # แกน x แสดงเป็นจำนวนเต็ม ไม่มีทศนิยม และกำหนดจำนวน tick marks
            st.plotly_chart(fig_revenue_bar, use_container_width=True)
            st.info("""
            **💡 Insight:** TH-NORTH (89.9M) และ TH-CENTRAL (91.5M) เป็น Top 2 ตลาดหลัก  
               * **รวม 4 Regions ของไทย** = 461M (~59% ของรายได้ทั้งหมด) 
            """)

            # Example 4:   Mean Customer Score ตาม Delay Group
            st.subheader("Mean Customer Score ตาม Delay Group")
            
            # # ส่วนของโปรแกรมนี้ใช้ในการคำนวณคะแนนความพึงพอใจเฉลี่ยและจำนวนครั้งของแต่ละกลุ่มความล่าช้าในการจัดส่ง และแสดงค่าสหสัมพันธ์ระหว่างความล่าช้าและคะแนนความพึงพอใจของลูกค้า
            delay_mean = (df_processed.groupby('Delay_Group', observed=True)['Customer_Score']
                            .agg(['mean', 'count'])
                            .reset_index())
            delay_mean.columns = ['Delay_Group', 'Mean_Score', 'Count']
            print(delay_mean)

            # Bar Chart: Mean Customer Score by Delay Group
            fig_delay = px.bar(delay_mean, x='Delay_Group', y='Mean_Score',
                        title='Mean Customer Score ตาม Delay Group',
                        color='Delay_Group', # Color bars based on Delay_Group
                        text='Mean_Score', # Display Mean_Score as text on bars
                        template='plotly_white',
                        hover_data={'Mean_Score': ':.2f', 'Count': ':.0f,f'}) # Specify columns and their formatting for hover_data

            fig_delay.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_delay.update_layout(yaxis_range=[5.5, 7.0]) # Set y-axis range for better visualization


            st.plotly_chart(fig_delay, use_container_width=True)
            st.info("""
            **💡 Insight:** Delay > 10 วัน เห็น pattern Score ลดลงเล็กน้อย 
                * ** กลุ่ม 0 วัน (ส่งทันที) มีคะแนนสูงสุด — ความรวดเร็วส่งผลบวกต่อ Satisfaction
            """)
        elif analysis_type == "Multivariate Analysis":
            st.header("📊 Multivariate Analysis (การวิเคราะห์หลายตัวแปร)")
            st.write("สำรวจข้อมูลสำหรับหลายตัวแปร")

            # Example 5.1 : Bubble Chart — Units Sold, Marketing, Customer Score
            st.subheader("การกระจายตัวแบบ Bubble Chart — Units Sold, Marketing, Customer Score")
            # Aggregate by Region + Product
            bubble_df = df_processed.groupby(['Region','Product_Variant']).agg(
                Avg_Units=('Units_Sold','mean'),
                Total_Rev=('Total_Revenue','sum'),
                Avg_Marketing=('Marketing_Spend','mean'),
                Avg_Score=('Customer_Score','mean')
            ).reset_index()
            bubble_df['Total_Rev_M'] = bubble_df['Total_Rev'] / 1e6

            fig_bubble = px.scatter(
                bubble_df,
                x='Avg_Marketing', y='Avg_Units',
                size='Total_Rev_M',
                color='Product_Variant',
                symbol='Product_Variant',
                color_discrete_sequence=RB_COLORS,

                hover_data={'Region':True, 'Avg_Score':':.2f',
                            'Total_Rev_M':':.1f', 'Avg_Marketing':':.0f'},

                # กำหนดชื่อเรื่องหลักของกราฟ และชื่อรอง (ด้วยแท็ก <sup>) ที่อธิบายความหมายของขนาดและสีของ Bubble
                title='Bubble Chart: Marketing Spend vs Units Sold<br>'
                    '<sup>ขนาด Bubble = Total Revenue (M) | สี = Product | Hover เพื่อดูรายละเอียด</sup>',

                labels={  # กำหนดชื่อแกนและป้ายกำกับของ Legend ให้มีความหมายที่ชัดเจนขึ้น
                    'Avg_Marketing': 'Avg Marketing Spend', #ซ้ายชื่อ Attribute ขวา ข้อความที่ปรากฎ
                    'Avg_Units': 'Avg Units Sold',
                    'Product_Variant': 'Product'
                },
                template='plotly_white', height=520
            )
          
            fig_bubble.update_traces(marker=dict(opacity=0.75, line=dict(width=1, color='white')))
            fig_bubble.update_layout(title_font_size=14, legend_title='Product')
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.info("""**💡 Insight:** Bubble Chart แสดง 3 มิติพร้อมกัน: Marketing Spend (x), Units Sold (y), Revenue (size) | Krating Daeng 250 
            * มีบาง Region ที่ Units Sold สูงมาก แม้ Marketing ไม่ได้สูงสุด 
            * กลุ่ม Bubble ใหญ่กระจายตัวแทบทุกช่วง Marketing 
            * ยืนยันว่า **Marketing Spend ไม่ได้เป็นตัวแปรหลักที่กำหนดยอดขาย**
            """)

            # Example 5.2 : 'Correlation Heatmap — Numeric Features
            st.subheader("กราฟแสดงความสัมพันธ์ระหว่างแอททริบิวต์ที่เป็นตัวเลข")

            numeric_df = df_processed.select_dtypes(include=[np.number])
            corr = numeric_df.corr() #corr คือ DataFrame ที่เก็บค่าสหสัมพันธ์ระหว่างตัวแปรต่างๆ ซึ่งเป็นตารางสี่เหลี่ยม (เช่น A-B, A-C, B-A, B-C เป็นต้น)

            fig_corr = px.imshow(corr, text_auto=True,
                            title='Correlation Heatmap — Numeric Features<br><sup>Hover เพื่อดูค่า | สีแดง = positive, สีน้ำเงิน = negative</sup>',
                            color_continuous_scale='RdBu_r',
                            template='plotly_white',
                            height=800
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            corr_flat = corr.unstack() #จะเปลี่ยน DataFrame corr ให้เป็น Series ซึ่งจะมีค่าสหสัมพันธ์ทั้งหมดออกมาเป็นแนวตั้ง และจะสร้าง MultiIndex ที่ประกอบด้วยคู่ของชื่อตัวแปรที่สหสัมพันธ์กัน (เช่น ('Units_Sold', 'Total_Revenue'))
            corr_flat = corr_flat[(corr_flat.abs() < 1.0)].abs().sort_values(ascending=False) #corr_flat เพื่อให้เหลือเฉพาะค่าสหสัมพันธ์ที่มี ค่าสัมบูรณ์  น้อยกว่า 1.0

            st.subheader("🔥 Top Correlation Pairs")
            show_df = corr_flat.drop_duplicates().head(6)
            st.dataframe(
                show_df,
                width = 'content'
            )

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {e}")
else:
    st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้น")

