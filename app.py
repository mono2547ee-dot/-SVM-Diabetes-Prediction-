import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Diabetes Prediction", page_icon="🏥", layout="wide")
st.title(" ระบบทำนายโรคเบาหวาน")

# โหลดโมเดล
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

# Sidebar
st.sidebar.header(" กรอกข้อมูลสุขภาพ")

pregnancies = st.sidebar.number_input("จำนวนครั้งที่ตั้งครรภ์", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("ระดับน้ำตาลในเลือด", min_value=0, max_value=300, value=100)
blood_pressure = st.sidebar.number_input("ความดันโลหิต", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("ความหนาของผิวหนัง", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("ระดับอินซูลิน", min_value=0, max_value=1000, value=80)
bmi = st.sidebar.number_input("ดัชนีมวลกาย (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
pedigree = st.sidebar.number_input("ประวัติเบาหวานในครอบครัว", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.sidebar.number_input("อายุ", min_value=0, max_value=120, value=30)

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
            'DiabetesPedigreeFunction': [pedigree],
            'Age': [age]
        })
        
        # เพิ่ม features (ต้องตรงกับตอน train)
        if 'AgeGroup' in feature_names:
            input_df['AgeGroup'] = pd.cut(input_df['Age'], 
                                           bins=[0, 30, 45, 60, 100], 
                                           labels=[0, 1, 2, 3]).astype(float)
        if 'Glucose_Insulin_Ratio' in feature_names:
            input_df['Glucose_Insulin_Ratio'] = input_df['Glucose'] / (input_df['Insulin'] + 1)
        if 'BMI_Category' in feature_names:
            input_df['BMI_Category'] = pd.cut(input_df['BMI'], 
                                               bins=[0, 18.5, 25, 30, 100], 
                                               labels=[0, 1, 2, 3]).astype(float)
        
        # เลือก features และปรับสเกล
        input_scaled = scaler.transform(input_df[feature_names])
        
        # ทำนาย
        prediction = model.predict(input_scaled)
        
        # แสดงผล
        st.markdown("---")
        st.subheader("📊 ผลการทำนาย")
        
        if prediction[0] == 1:
            st.error("⚠️ **มีความเสี่ยงเป็นโรคเบาหวาน**")
        else:
            st.success("✅ **ไม่มีความเสี่ยงเป็นโรคเบาหวาน**")
        
        # แสดงความน่าจะเป็น (ถ้ามี)
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"ความน่าจะเป็นที่ไม่เป็นเบาหวาน: {probabilities[0][0]*100:.2f}%")
                with col2:
                    st.warning(f"ความน่าจะเป็นที่เป็นเบาหวาน: {probabilities[0][1]*100:.2f}%")
        except:
            pass
        
        # แสดงข้อมูลที่กรอก
        st.markdown("---")
        st.subheader(" ข้อมูลที่กรอก")
        st.write(input_df.T)
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        st.error("กรุณาตรวจสอบว่าโมเดลถูกฝึกด้วย probability=True หรือไม่")

st.markdown("---")
st.markdown("**หมายเหตุ:** ระบบนี้เป็นการทำนายเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่ถูกต้อง")