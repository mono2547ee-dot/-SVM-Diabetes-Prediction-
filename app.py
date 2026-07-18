import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ตั้งค่าหน้าเว็บให้สวยงาม
st.set_page_config(
    page_title="🩺 Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# สไตล์ CSS เล็กน้อยเพื่อให้ดูทันสมัย
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #2E86C1; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🩺 ระบบทำนายความเสี่ยงโรคเบาหวาน (AI)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ประเมินสุขภาพเบื้องต้นด้วยเทคโนโลยี Machine Learning (SVM Model)</p>', unsafe_allow_html=True)

# 1. โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        return joblib.load('diabetes_svm_model.pkl')
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

# 2. Sidebar สำหรับกรอกข้อมูล
st.sidebar.markdown("### 📝 กรอกข้อมูลสุขภาพ")
st.sidebar.markdown("---")

pregnancies = st.sidebar.number_input("🤰 จำนวนครั้งที่ตั้งครรภ์", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("🩸 ระดับน้ำตาลในเลือด (Glucose)", min_value=0, max_value=400, value=100)
blood_pressure = st.sidebar.number_input("🫀 ความดันโลหิต (Blood Pressure)", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("📏 ความหนาของผิวหนัง (Skin Thickness)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("💉 ระดับอินซูลิน (Insulin)", min_value=0, max_value=1000, value=80)
bmi = st.sidebar.number_input("⚖️ ดัชนีมวลกาย (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
pedigree = st.sidebar.number_input("🧬 ประวัติเบาหวานในครอบครัว (Pedigree)", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.sidebar.number_input("🎂 อายุ (Age)", min_value=0, max_value=120, value=30)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 ทำนายผลสุขภาพ", type="primary", use_container_width=True)

# 3. การประมวลผลและแสดงผล
if predict_btn:
    # สร้าง DataFrame จากข้อมูลผู้ใช้
    input_df = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [pedigree],
        'Age': [age]
    })
    
    # ⚠️ สำคัญ: สร้าง Feature ใหม่ให้ตรงกับตอน Train โมเดลเป๊ะๆ
    input_df['AgeGroup'] = pd.cut(input_df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(float)
    input_df['Glucose_Insulin_Ratio'] = input_df['Glucose'] / (input_df['Insulin'] + 1)
    input_df['BMI_Category'] = pd.cut(input_df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    
    # เรียงคอลัมน์ให้ตรงกับที่โมเดลคาดหวัง
    input_scaled = scaler.transform(input_df[feature_names])
    
    # ทำนายผล
    prediction = model.predict(input_scaled)
    
    # แสดงผลส่วนบน (Alert)
    st.markdown("---")
    st.subheader("📊 ผลการวิเคราะห์")
    
    # 🚨 เพิ่มระบบแจ้งเตือนพิเศษสำหรับค่าน้ำตาลสูง
    if glucose >= 200:
        st.warning("⚠️ **คำเตือน:** ระดับน้ำตาลในเลือดของคุณอยู่ในเกณฑ์สูง (≥ 200 mg/dL) ซึ่งอาจบ่งชี้ถึงภาวะเบาหวาน ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติมโดยเร็ว")
    if glucose >= 300:
        st.error("🚨 **อันตราย:** ระดับน้ำตาลในเลือดสูงมาก (≥ 300 mg/dL)! กรุณาพบแพทย์ทันทีเพื่อป้องกันภาวะแทรกซ้อนร้ายแรง")

    # แสดงผลลัพธ์หลัก
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if prediction[0] == 1:
            st.markdown("### 🔴 ผลทำนาย: **มีความเสี่ยง**")
            st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=150)
        else:
            st.markdown("### 🟢 ผลทำนาย: **ปลอดภัย**")
            st.image("https://cdn-icons-png.flaticon.com/512/2913/2913520.png", width=150)

    with col2:
        st.markdown("#### 📈 ความน่าจะเป็น")
        try:
            # พยายามใช้ predict_proba (ถ้าโมเดลถูก train ด้วย probability=True)
            probabilities = model.predict_proba(input_scaled)
            prob_0 = probabilities[0][0] * 100
            prob_1 = probabilities[0][1] * 100
            
            col_a, col_b = st.columns(2)
            col_a.metric("🟢 ไม่เป็นเบาหวาน", f"{prob_0:.1f}%")
            col_b.metric("🔴 เป็นเบาหวาน", f"{prob_1:.1f}%")
            
            # Progress bar
            st.progress(prob_1 / 100)
        except Exception:
            # Fallback กรณีโมเดลไม่มี probability
            st.info("ℹ️ โมเดลนี้ให้ผลเป็นหมวดหมู่ (ไม่รองรับการแสดง % ความน่าจะเป็น)")
            if prediction[0] == 1:
                st.progress(0.8) # สมมติค่าแสดง
            else:
                st.progress(0.2)

    # แสดงข้อมูลที่กรอก
    st.markdown("---")
    st.subheader("📋 สรุปข้อมูลที่คุณกรอก")
    st.dataframe(input_df.T.style.format("{:.2f}"), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        ⚠️ <b>ข้อจำกัดความรับผิดชอบ:</b> ระบบนี้เป็นการทำนายเบื้องต้นด้วย AI เท่านั้น ไม่สามารถทดแทนการวินิจฉัยจากแพทย์ผู้เชี่ยวชาญได้ <br>
        หากมีอาการผิดปกติ หรือค่าน้ำตาลเกิน 200 mg/dL กรุณาปรึกษาแพทย์ทันที 🏥
    </div>
""", unsafe_allow_html=True)