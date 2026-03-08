import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.express as px
from fpdf import FPDF
import tempfile

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Medical Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# DARK MODE
# -----------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    bg_color = "#0e1117"
    card_color = "rgba(255,255,255,0.05)"
    text_color = "white"
else:
    bg_color = "#f3f6fb"
    card_color = "rgba(255,255,255,0.7)"
    text_color = "black"

# -----------------------------
# CSS (GLASSMORPHISM)
# -----------------------------
st.markdown(f"""
<style>

.stApp {{
background:{bg_color};
}}

.glass {{
backdrop-filter: blur(15px);
background:{card_color};
border-radius:15px;
padding:25px;
box-shadow:0px 8px 25px rgba(0,0,0,0.15);
margin-bottom:20px;
}}

.title {{
font-size:48px;
font-weight:700;
text-align:center;
color:{text_color};
}}

.subtitle {{
text-align:center;
font-size:18px;
color:gray;
margin-bottom:25px;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
svc = pickle.load(open("svc.pkl", "rb"))

# -----------------------------
# SYMPTOMS
# -----------------------------
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91,
    'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
    'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
    'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117,
    'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
    'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
    'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

symptom_order = list(symptoms_dict.keys())

# -----------------------------
# LOAD DATASETS
# -----------------------------
base = os.path.dirname(__file__)

description = pd.read_csv(os.path.join(base,"Datasets","description.csv"))
precautions = pd.read_csv(os.path.join(base,"Datasets","precautions_df.csv"))
medications = pd.read_csv(os.path.join(base,"Datasets","medications.csv"))
diets = pd.read_csv(os.path.join(base,"Datasets","diets.csv"))
workout = pd.read_csv(os.path.join(base,"Datasets","workout_df.csv"))

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def helper(dis):

    desc = " ".join(description[description['Disease']==dis]['Description'])

    pre = precautions[precautions['Disease']==dis].iloc[:,1:].values.flatten()

    med = medications[medications['Disease']==dis]['Medication'].values

    diet = diets[diets['Disease']==dis]['Diet'].values

    work = workout[workout['disease']==dis]['workout'].values

    return desc,pre,med,diet,work

# -----------------------------
# PDF GENERATOR
# -----------------------------
def generate_pdf(disease, desc, pre, med, diet, work):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"AI Medical Report",ln=True,align="C")

    pdf.ln(5)

    pdf.set_font("Arial","B",12)
    pdf.cell(0,10,f"Predicted Disease: {disease}",ln=True)

    pdf.set_font("Arial","",11)

    pdf.multi_cell(0,8,f"\nDescription:\n{desc}")

    pdf.multi_cell(0,8,"\nPrecautions:\n" + "\n".join([str(x) for x in pre if str(x)!='nan']))

    pdf.multi_cell(0,8,"\nMedications:\n" + "\n".join([str(x) for x in med]))

    pdf.multi_cell(0,8,"\nDiet:\n" + "\n".join([str(x) for x in diet]))

    pdf.multi_cell(0,8,"\nWorkout:\n" + "\n".join([str(x) for x in work]))

    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")
    pdf.output(tmp.name)

    return tmp.name

# -----------------------------
# HEADER
# -----------------------------
col1,col2 = st.columns([3,1])

with col1:
    st.markdown('<div class="title">🩺 AI Medical Disease Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Select symptoms and let AI predict disease</div>', unsafe_allow_html=True)

with col2:
    st.image(
    "https://cdn-icons-png.flaticon.com/512/387/387561.png",
    width=120
    )

# -----------------------------
# SYMPTOM SEARCH
# -----------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

search = st.text_input("🔎 Search symptom")

filtered = [s for s in symptom_order if search.lower() in s.lower()]

selected_symptoms = st.multiselect(
"Select Symptoms",
filtered if search else symptom_order
)

predict = st.button("Predict Disease")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# PREDICTION
# -----------------------------
if predict:

    input_vector = np.zeros(len(symptom_order))

    for s in selected_symptoms:
        input_vector[symptoms_dict[s]] = 1

    pred = svc.predict([input_vector])[0]

    desc,pre,med,diet,work = helper(pred)

    st.success(f"Predicted Disease: {pred}")

    # -----------------------------
    # PROBABILITY CHART
    # -----------------------------
    probs = svc.decision_function([input_vector])[0]

    fig = px.bar(
        x=[pred],
        y=[max(probs)],
        labels={'x':'Disease','y':'Confidence'}
    )

    st.plotly_chart(fig,use_container_width=True)

    # -----------------------------
    # STRUCTURED RESULTS
    # -----------------------------
    st.markdown("## 📋 Medical Report")

    col1,col2 = st.columns(2)

    with col1:

        st.markdown("### 📄 Description")
        st.info(desc)

        st.markdown("### 🛡 Precautions")

        pre_df = pd.DataFrame(
            [p for p in pre if str(p)!='nan'],
            columns=["Precautions"]
        )

        st.table(pre_df)

    with col2:

        st.markdown("### 💊 Medications")

        med_df = pd.DataFrame(
            med,
            columns=["Medicines"]
        )

        st.table(med_df)

        st.markdown("### 🥗 Diet")

        diet_df = pd.DataFrame(
            diet,
            columns=["Recommended Diet"]
        )

        st.table(diet_df)

    st.markdown("### 🏃 Workout")

    work_df = pd.DataFrame(
        work,
        columns=["Exercises"]
    )

    st.table(work_df)

    # -----------------------------
    # PDF DOWNLOAD
    # -----------------------------
    pdf_path = generate_pdf(pred, desc, pre, med, diet, work)

    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇ Download Medical Report (PDF)",
            f,
            file_name="medical_report.pdf",
            mime="application/pdf"
        )