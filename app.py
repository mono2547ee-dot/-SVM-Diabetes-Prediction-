import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

st.title("🩺 ระบบทำนายความเสี่ยงโรคเบาหวาน (SVM Model)")
st.markdown("---")

# โหลดโมเดล
@st.cache_resource
def load_model():
    model_data = joblib.load('diabetes_svm_model.pkl')
    return model_data

try:
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
    st.stop()

# Sidebar สำหรับกรอกข้อมูล
st.sidebar.header("📋 กรอกข้อมูลสุขภาพ")

pregnancies = st.sidebar.number_input("จำนวนครั้งที่ตั้งครรภ์", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("ระดับน้ำตาลในเลือด (Glucose)", min_value=0, max_value=300, value=100)
blood_pressure = st.sidebar.number_input("ความดันโลหิต (Blood Pressure)", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("ความหนาของผิวหนัง (Skin Thickness)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("ระดับอินซูลิน (Insulin)", min_value=0, max_value=1000, value=80)
bmi = st.sidebar.number_input("ดัชนีมวลกาย (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
diabetes_pedigree = st.sidebar.number_input("ประวัติเบาหวานในครอบครัว", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.sidebar.number_input("อายุ (Age)", min_value=0, max_value=120, value=30)

# ปุ่มทำนาย
if st.sidebar.button("🔮 ทำนายผล", type="primary"):
    try:
        # สร้าง DataFrame
        input_df = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })
        
        # เพิ่ม features ที่สร้างตอน training (ถ้ามี)
        # AgeGroup
        input_df['AgeGroup'] = pd.cut(input_df['Age'], 
                                       bins=[0, 30, 45, 60, 100], 
                                       labels=[0, 1, 2, 3]).astype(float)
        # Glucose_Insulin_Ratio
        input_df['Glucose_Insulin_Ratio'] = input_df['Glucose'] / (input_df['Insulin'] + 1)
        # BMI_Category
        input_df['BMI_Category'] = pd.cut(input_df['BMI'], 
                                           bins=[0, 18.5, 25, 30, 100], 
                                           labels=[0, 1, 2, 3]).astype(float)
        
        # เลือก features และปรับสเกล
        input_scaled = scaler.transform(input_df[feature_names])
        
        # ทำนาย
        prediction = model.predict(input_scaled)
        
        # ตรวจสอบว่าโมเดล support predict_proba หรือไม่
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)
            prob_not_diabetes = probabilities[0][0] * 100
            prob_diabetes = probabilities[0][1] * 100
        else:
            # ถ้าไม่มี predict_proba ให้ใช้ decision function แทน
            decision = model.decision_function(input_scaled)
            prob_not_diabetes = 50.0  # ค่า default
            prob_diabetes = 50.0
        
        # แสดงผล
        st.markdown("---")
        st.subheader("📊 ผลการทำนาย")
        
        if prediction[0] == 1:
            st.error("⚠️ **มีความเสี่ยงเป็นโรคเบาหวาน**")
        else:
            st.success("✅ **ไม่มีความเสี่ยงเป็นโรคเบาหวาน**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"ความน่าจะเป็นที่ไม่เป็นเบาหวาน: {prob_not_diabetes:.2f}%")
        with col2:
            st.warning(f"ความน่าจะเป็นที่เป็นเบาหวาน: {prob_diabetes:.2f}%")
        
        # แสดงข้อมูลที่กรอก
        st.markdown("---")
        st.subheader("📝 ข้อมูลที่กรอก")
        st.write(input_df.T)
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        st.error("กรุณาตรวจสอบว่าโมเดลถูกฝึกด้วย probability=True หรือไม่")

# ข้อมูลเพิ่มเติม
st.markdown("---")
st.markdown("""
### ℹ️ เกี่ยวกับระบบ
- **โมเดล**: Support Vector Machine (SVM)
- **ข้อมูล**: Pima Indians Diabetes Dataset

### 📌 คำแนะนำ
ระบบนี้เป็นการทำนายเบื้องต้นเท่านั้น 
ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่ถูกต้อง
""")