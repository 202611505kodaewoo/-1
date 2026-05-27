import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------------------------
# 0. 모델 및 스케일러 로드 함수
# ----------------------------------------------------
@st.cache_resource
def load_ml_components():
    model_path = "diabetes_model_n.pkl"  
    scaler_path = "scaler.pkl"            
    
    model = None
    scaler = None
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"스케일러 로드 실패: {e}")
            
    return model, scaler

rf_model_eng, scaler = load_ml_components()

# ----------------------------------------------------
# 1. 스트림릿 웹 화면 구성
# ----------------------------------------------------
st.set_page_config(page_title="당뇨병 예측 프로그램", page_icon="🩺", layout="centered")
st.title("🩺 당뇨병 예측 데이터 입력")
st.write("아래의 정보를 입력하시면 당뇨병 여부를 예측합니다.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("임신횟수 입력", min_value=0, value=0, step=1)
    glucose = st.number_input("혈당(포도당) 입력", min_value=0.0, value=0.0, step=1.0)
    bp = st.number_input("혈압 입력", min_value=0.0, value=0.0, step=1.0)
    skin = st.number_input("피부두께 입력", min_value=0.0, value=0.0, step=1.0)

with col2:
    insulin = st.number_input("인슐린 입력", min_value=0.0, value=0.0, step=1.0)
    bmi = st.number_input("체질량지수(BMI) 입력", min_value=0.0, value=0.0, step=0.1)
    dpf = st.number_input("가족력(당뇨 백분위) 입력", min_value=0.0, value=0.0, step=0.01)
    age = st.number_input("나이 입력", min_value=0, value=0, step=1)

# 💡 2. 파생 변수 자동 계산
obesity_risk = 1 if bmi >= 25.0 else 0        
is_elderly = 1 if age >= 50 else 0            
high_glucose = 1 if glucose >= 140.0 else 0    
metabolic_risk = 1 if (bmi >= 25.0 and bp >= 130.0) else 0  

# 임시 딕셔너리로 우선 데이터를 묶어줍니다.
raw_data = {
    '임신': preg, '포도당': glucose, '혈압': bp, '피부두께': skin, 
    '인슐린': insulin, '체질량지수': bmi, '당뇨병 가계도 기능': dpf, '나이': age, 
    '비만위험': obesity_risk, '고령': is_elderly, '고혈당': high_glucose, '대사위험': metabolic_risk
}

# ⭐ [핵심 해결 방법] 스케일러가 기억하는 원본 학습 컬럼 순서 그대로 데이터를 재배열합니다.
if scaler is not None and hasattr(scaler, "feature_names_in_"):
    # 스케일러가 학습했던 컬럼 순서 리스트를 가져옵니다.
    correct_order = scaler.feature_names_in_.tolist()
    
    # 그 순서대로 값을 뽑아내어 리스트를 구성합니다.
    ordered_values = [raw_data[col] for col in correct_order]
    
    # 철저하게 정렬된 데이터프레임을 생성합니다.
    input_data = pd.DataFrame([ordered_values], columns=correct_order)
else:
    # 예외 상황용 기본 배치
    temp_columns = ['임신', '포도당', '혈압', '피부두께', '인슐린', '체질량지수', '당뇨병 가계도 기능', '나이', '비만위험', '고령', '고혈당', '대사위험']
    input_data = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age, obesity_risk, is_elderly, high_glucose, metabolic_risk]], columns=temp_columns)

st.markdown("---")
st.subheader("📊 변환된 데이터프레임 (모델 입력 데이터)")
st.dataframe(input_data)

# ----------------------------------------------------
# 3. 예측 로직 실행
# ----------------------------------------------------
if st.button("🔮 당뇨병 예측하기", use_container_width=True):
    if rf_model_eng is None or scaler is None:
        st.error("❌ 모델 또는 스케일러 파일이 올바르게 로드되지 않았습니다.")
    else:
        try:
            # 완벽히 정렬된 데이터로 스케일 변환 및 예측 진행
            scaled_array = scaler.transform(input_data)
            scaled_data = pd.DataFrame(scaled_array, columns=input_data.columns)
            
            predicted = rf_model_eng.predict(scaled_data)
            prob = rf_model_eng.predict_proba(scaled_data)
            
            st.markdown("---")
            st.subheader("🎯 예측 결과")
            
            diabetes_prob = prob[0][1] * 100
            
            if predicted[0] == 1:
                st.error(f"⚠️ **예측 결과: 당뇨 (1)**")
                st.write(f"당뇨병일 확률이 **{diabetes_prob:.1f}%**로 높게 나타났습니다. 의사와 상담을 권장합니다.")
            else:
                st.success(f"✅ **예측 결과: 정상 (0)**")
                st.write(f"당뇨병일 확률이 **{diabetes_prob:.1f}%**로 낮게 나타났습니다. 꾸준한 건강 관리를 유지하세요.")
                
        except Exception as e:
            st.error(f"❌ 예측 중 오류가 발생했습니다: {e}")
