import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

st.title("🩺 ระบบทำนายความเสี่ยงโรคเบาหวาน (SVM Model)")
st.markdown("กรุณากรอกข้อมูลสุขภาพด้านล่าง เพื่อประเมินความเสี่ยงเบื้องต้น")

# 1. โหลดโมเดล
@st.cache_resource
def load_model():
    return joblib.load('diabetes_svm_model.pkl')

try:
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
    st.stop()

# 2. Sidebar สำหรับกรอกข้อมูล
st.sidebar.header("📋 กรอกข้อมูลผู้ป่วย")

pregnancies = st.sidebar.number_input("จำนวนครั้งที่ตั้งครรภ์ (Pregnancies)", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("ระดับน้ำตาลในเลือด (Glucose)", min_value=0, max_value=300, value=100)
blood_pressure = st.sidebar.number_input("ความดันโลหิต (Blood Pressure)", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("ความหนาของผิวหนัง (Skin Thickness)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("ระดับอินซูลิน (Insulin)", min_value=0, max_value=1000, value=80)
bmi = st.sidebar.number_input("ดัชนีมวลกาย (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
pedigree = st.sidebar.number_input("ประวัติเบาหวานในครอบครัว (Pedigree)", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.sidebar.number_input("อายุ (Age)", min_value=0, max_value=120, value=30)

# 3. ปุ่มทำนาย
if st.sidebar.button("🔮 ทำนายผล", type="primary"):
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
    
    # ⚠️ สำคัญมาก: ต้องสร้าง Feature ใหม่ให้ตรงกับตอน Train โมเดลเป๊ะๆ
    input_df['AgeGroup'] = pd.cut(input_df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(float)
    input_df['Glucose_Insulin_Ratio'] = input_df['Glucose'] / (input_df['Insulin'] + 1)
    input_df['BMI_Category'] = pd.cut(input_df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    
    # เรียงคอลัมน์ให้ตรงกับที่โมเดลถูกฝึกมา
    input_df = input_df[feature_names]
    
    # ปรับสเกลข้อมูล
    input_scaled = scaler.transform(input_df)
    
    # ทำนายผล
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    # 4. แสดงผลลัพธ์
    st.markdown("---")
    st.subheader("📊 ผลการทำนาย")
    
    if prediction[0] == 1:
        st.error("⚠️ **มีความเสี่ยงเป็นโรคเบาหวาน**")
    else:
        st.success("✅ **ไม่มีความเสี่ยงเป็นโรคเบาหวาน**")
        
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ความน่าจะเป็นที่ไม่เป็นเบาหวาน", f"{probabilities[0][0]*100:.2f}%")
    with col2:
        st.metric("ความน่าจะเป็นที่เป็นเบาหวาน", f"{probabilities[0][1]*100:.2f}%")
    
    st.markdown("---")
    st.caption("⚠️ คำเตือน: ระบบนี้เป็นการทำนายเบื้องต้นจากโมเดล Machine Learning เท่านั้น ไม่สามารถทดแทนการวินิจฉัยจากแพทย์ผู้เชี่ยวชาญได้")

# (ทางเลือก) ถ้าคุณยืนยันจะ Deploy บน Heroku/Render จริงๆ ให้สร้างไฟล์ 2 ไฟล์นี้เพิ่ม:
# Procfile: web: sh setup.sh && streamlit run app.py
# setup.sh: (ดูโค้ดที่แก้ไขให้ด้านล่าง)