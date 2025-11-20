import streamlit as st
import numpy as np
import joblib
import os

# ----------------------
# Config / Constants
# ----------------------
MODEL_FILENAME = "Diabetset.pkl"  # expected to be in the same folder as app.py when deployed

# Recommended ranges
RANGES = {
    "age": (10, 100),
    "glucose": (40, 300),
    "blood": (40, 140),
    "skin": (5, 80),
    "insulin": (10, 400),
    "bmi": (10, 60),
    "weight": (20.0, 250.0),
    "height": (100.0, 220.0),
}

# sensible median fallbacks if optional inputs are missing
MEDIAN_FALLBACKS = {
    "Insulin": 80.0,
    "SkinThickness": 20.0
}

# DPF mapping (human-friendly dropdown -> numeric value)
DPF_MAP = {
    "ไม่มีประวัติในครอบครัว": 0.05,
    "ญาติห่าง (เช่น ป้า/น้า/อา) เป็น": 0.5,
    "พ่อหรือแม่เป็น": 1.0,
    "พ่อแม่ + พี่น้องเป็น": 2.0,
    "หลายคนในครอบครัวเป็น": 2.5,
}

# ----------------------
# Helpers
# ----------------------
def load_model(path=MODEL_FILENAME):
    if os.path.exists(path):
        try:
            m = joblib.load(path)
            return m, None
        except Exception as e:
            return None, f"พบข้อผิดพลาดขณะโหลดโมเดล: {e}"
    else:
        return None, f"ไม่พบไฟล์โมเดล '{path}'. กรุณาอัปโหลดไฟล์ Diabetset.pkl ใน repository เดียวกับ app.py"

def clip_value(v, minv, maxv):
    return float(np.clip(v, minv, maxv))

def risk_level_from_prob(p):
    # p expected between 0 and 1
    if p < 0.30:
        return "ต่ำ", "ความเสี่ยงต่ำ แต่ควรดูแลสุขภาพอย่างสม่ำเสมอ"
    elif p < 0.60:
        return "ปานกลาง", "ความเสี่ยงปานกลาง — แนะนำตรวจสุขภาพเพิ่มเติมและปรับพฤติกรรม"
    else:
        return "สูง", "ความเสี่ยงสูง — ควรปรึกษาแพทย์และตรวจเลือด (fasting glucose / HbA1c)"

def health_advice(glucose, bmi, blood, age, insulin_provided, skin_provided):
    advice = []
    if bmi is not None:
        if bmi >= 30:
            advice.append("น้ำหนักเกิน/อ้วน: แนะนำลดน้ำหนักอย่างค่อยเป็นค่อยไป (ปรับอาหาร คุมปริมาณแคลอรี และออกกำลังกายอย่างน้อย 150 นาที/สัปดาห์)")
        elif bmi >= 25:
            advice.append("น้ำหนักเกิน: พิจารณาปรับพฤติกรรมการกินและออกกำลังกายเพื่อไม่ให้เพิ่มมากขึ้น")
        else:
            advice.append("น้ำหนักในเกณฑ์ดี: รักษาระดับความแข็งแรงและโภชนาการที่สมดุล")
    if glucose is not None:
        if glucose >= 200:
            advice.append("ค่าน้ำตาลสูงมาก — ควรรีบตรวจเลือดและปรึกษาแพทย์")
        elif glucose >= 140:
            advice.append("ค่าน้ำตาลสูง (อาจเป็น pre-diabetes) — แนะนำปรับพฤติกรรมและตรวจซ้ำ")
        else:
            advice.append("ค่าน้ำตาลในช่วงปกติ (ตามที่กรอก)")
    if blood is not None and blood >= 120:
        advice.append("ความดันค่อนข้างสูง: ควรติดตามความดันและปรับพฤติกรรม (ลดเค็ม/ออกกำลังกาย)")
    if not insulin_provided or not skin_provided:
        advice.append("ข้อมูลอินซูลิน/ความหนาผิวหนังไม่ครบ — หากต้องการการประเมินละเอียด ควรตรวจทางการแพทย์เพื่อหาค่าจริง")
    # age-specific advice
    if age is not None and age >= 60:
        advice.append("อายุมากกว่า 60 ปี — ควรตรวจสุขภาพเป็นประจำความเสี่ยงโรคระบบเมตาบอลิซึมสูงขึ้น")
    return advice

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Diabetes Risk Checker", layout="centered")

st.title("🩺 ระบบประเมินความเสี่ยงโรคเบาหวาน (Diabetes Risk Checker)")
st.write("กรอกข้อมูลตามที่ทราบ — ช่องบางช่องเป็น optional (ไม่จำเป็นต้องกรอก) ระบบจะให้คำแนะนำและแจ้งเตือนตามผลการประเมิน")

# Load model
model, model_err = load_model()
if model_err:
    st.warning(model_err)
else:
    st.success("โมเดลโหลดสำเร็จ — พร้อมใช้งาน")

with st.form("input_form"):
    st.header("ข้อมูลพื้นฐาน")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ (ปี)", min_value=int(RANGES["age"][0]), max_value=int(RANGES["age"][1]), value=40)
        pregnancies = st.number_input("จำนวนการตั้งครรภ์ (ถ้ามี)", min_value=0, max_value=20, value=0)
        weight = st.number_input("น้ำหนัก (kg)", min_value=RANGES["weight"][0], max_value=RANGES["weight"][1], value=70.0, format="%.1f")
    with col2:
        height = st.number_input("ส่วนสูง (cm)", min_value=RANGES["height"][0], max_value=RANGES["height"][1], value=170.0, format="%.1f")
        glucose = st.number_input("ระดับน้ำตาล (mg/dL)", min_value=RANGES["glucose"][0], max_value=RANGES["glucose"][1], value=100)
        blood = st.number_input("ความดันโลหิต (mmHg) (ค่าไดแอสโตลิค)", min_value=RANGES["blood"][0], max_value=RANGES["blood"][1], value=80)

    st.markdown("---")
    st.header("ข้อมูลเพิ่มเติม (Optional)")
    col3, col4 = st.columns(2)
    with col3:
        provide_skin = st.checkbox("ทราบค่าความหนาชั้นผิวหนัง (SkinThickness)?", value=False)
        if provide_skin:
            skin = st.number_input("Skin Thickness (mm)", min_value=RANGES["skin"][0], max_value=RANGES["skin"][1], value=20)
        else:
            skin = None
    with col4:
        provide_insulin = st.checkbox("ทราบค่าระดับอินซูลิน (Insulin)?", value=False)
        if provide_insulin:
            insulin = st.number_input("Insulin (μU/mL)", min_value=RANGES["insulin"][0], max_value=RANGES["insulin"][1], value=80)
        else:
            insulin = None

    st.markdown("---")
    st.header("ประวัติครอบครัว")
    dpf_label = st.selectbox("เลือกประวัติในครอบครัว (เพื่อ mapping ค่า DPF)", list(DPF_MAP.keys()))
    dpf = DPF_MAP[dpf_label]

    submit = st.form_submit_button("ทำนายความเสี่ยง")
    
if submit:
    # validation & clipping (for safety)
    glucose = clip_value(glucose, *RANGES["glucose"])
    blood = clip_value(blood, *RANGES["blood"])
    weight = clip_value(weight, *RANGES["weight"])
    height = clip_value(height, *RANGES["height"])
    bmi = round(weight / ((height/100.0)**2), 2)
    bmi = float(np.clip(bmi, *RANGES["bmi"]))

    # handle optional: use median fallbacks if missing, but note to user
    insulin_used = insulin if insulin is not None else MEDIAN_FALLBACKS["Insulin"]
    skin_used = skin if skin is not None else MEDIAN_FALLBACKS["SkinThickness"]
    insulin_provided_flag = insulin is not None
    skin_provided_flag = skin is not None

    insulin_used = clip_value(insulin_used, *RANGES["insulin"])
    skin_used = clip_value(skin_used, *RANGES["skin"])

    feature_vector = np.array([[glucose, bmi, age, blood, insulin_used, dpf, skin_used]])

    st.subheader("สรุปข้อมูลที่ใช้ในการประเมิน")
    st.write({
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "BloodPressure": blood,
        "Insulin (used)": insulin_used,
        "DPF": dpf,
        "SkinThickness (used)": skin_used,
        "Optional Provided": {
            "Insulin_provided": insulin_provided_flag,
            "Skin_provided": skin_provided_flag
        }
    })

    if model is None:
        st.error("ไม่สามารถทำนายได้เนื่องจากไม่มีโมเดล (Diabetset.pkl) ใน repository. อัปโหลดโมเดลแล้วรีเฟรชหน้าเว็บ")
    else:
        try:
            # probability if supported
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(feature_vector)[0][1]
            else:
                # fallback: use predict and convert to 0/1
                pred = model.predict(feature_vector)[0]
                prob = float(pred)
            risk_label, risk_msg = risk_level_from_prob(prob)

            st.markdown("### ผลการประเมิน")
            st.metric("ความเสี่ยง (probability)", f"{prob:.2f}", help="ความน่าจะเป็นที่โมเดลประเมินว่าจะเป็นเบาหวาน")
            if risk_label == "ต่ำ":
                st.success(f"ระดับความเสี่ยง: {risk_label} — {risk_msg}")
            elif risk_label == "ปานกลาง":
                st.warning(f"ระดับความเสี่ยง: {risk_label} — {risk_msg}")
            else:
                st.error(f"ระดับความเสี่ยง: {risk_label} — {risk_msg}")

            # Detailed health advice
            st.markdown("### คำแนะนำสุขภาพ (เบื้องต้น)")
            adv_list = health_advice(glucose, bmi, blood, age, insulin_provided_flag, skin_provided_flag)
            for a in adv_list:
                st.write("- " + a)

            # Extra targeted suggestions based on risk
            st.markdown("#### ข้อแนะนำเพิ่มเติมตามระดับความเสี่ยง")
            if prob >= 0.6:
                st.write("- ควรไปพบแพทย์เพื่อทำการตรวจระดับน้ำตาล (Fasting glucose, HbA1c) และรับคำปรึกษา")
                st.write("- หากยืนยันมีภาวะ pre-diabetes/diabetes แพทย์จะให้แนวทางการจัดการ (ยา/โภชนาการ/การออกกำลังกาย)")
            elif prob >= 0.3:
                st.write("- ควรปรับพฤติกรรม: ลดน้ำตาล/ลดแป้ง น้ำตาลผลไม้ ควบคุมปริมาณแคลอรี")
                st.write("- เริ่มออกกำลังกายแบบแอโรบิค 150 นาที/สัปดาห์ และเพิ่มการฝึกความแข็งแรง 2 วัน/สัปดาห์")
            else:
                st.write("- รักษาพฤติกรรมที่ดีต่อสุขภาพต่อไปและตรวจสุขภาพเป็นประจำ")

            # Remind about optional values
            if not insulin_provided_flag or not skin_provided_flag:
                st.info("หมายเหตุ: เนื่องจากไม่ได้กรอกค่า Insulin หรือ SkinThickness ระบบใช้ค่าประมาณในการประเมิน ผลลัพธ์อาจไม่แม่นยำเท่าการมีค่าจริง")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดขณะทำนาย: {e}")


