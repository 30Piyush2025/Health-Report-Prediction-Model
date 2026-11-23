import streamlit as st
import pandas as pd
import numpy as np
from src.hybrid_model import predict_hybrid_diagnosis
from src.preprocessor import clean_clinical_note

st.set_page_config(
    page_title="MediMind AI - Clinical Notes Diagnosis Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .medical-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .medical-sub {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .badge-icd {
        background: #e0f2fe;
        color: #0369a1;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="medical-header">🩺 MediMind AI - Clinical Notes Diagnosis Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="medical-sub">Semantic embedding analysis + Hybrid neural network diagnostic suggestions with ICD-10 mapping.</div>', unsafe_allow_html=True)

# Preset clinical case templates
PRESET_CASES = {
    "Select a pre-filled clinical case template...": "",
    "Case 1: Congestive Heart Failure (Dyspnea + Edema)": (
        "A 72-year-old female presents with progressive shortness of breath on exertion, "
        "orthopnea requiring 3 pillows, and severe bilateral lower extremity edema. "
        "History of hypertension and coronary artery disease. Labs reveal significantly elevated BNP "
        "and echocardiogram shows left ventricular hypertrophy with reduced ejection fraction."
    ),
    "Case 2: Acute Myocardial Infarction (Chest Pain)": (
        "A 61-year-old male arrives at emergency with severe substernal chest pain lasting 45 minutes, "
        "radiating to jaw and left arm. Accompanied by heavy diaphoresis and nausea. ECG demonstrates ST-segment "
        "elevations in leads II, III, and aVF with positive cardiac troponin-I."
    ),
    "Case 3: Community-Acquired Pneumonia (Fever + Cough)": (
        "A 56-year-old male presents with acute high-grade fever, shaking chills, productive cough yielding "
        "purulent rust-colored sputum, and right-sided pleuritic chest pain. Chest radiography reveals dense "
        "right lower lobe alveolar consolidation and crackles on auscultation."
    ),
    "Case 4: Acute Pancreatitis (Severe Epigastric Pain)": (
        "A 47-year-old female presents with acute severe epigastric abdominal pain boring through to the back, "
        "aggravated after meals. Associated with repeated vomiting. Serum lipase level is elevated 5 times above normal."
    ),
    "Case 5: Diabetic Foot Ulcer (Non-healing Neuropathic Wound)": (
        "A 64-year-old male with 18-year history of poorly controlled type 2 diabetes presents with a deep, "
        "painless ulcer over the plantar surface of the 1st metatarsal head. Loss of protective monofilament sensation, "
        "peripheral neuropathy, and mild localized perilesional erythema."
    )
}

selected_template = st.selectbox("📋 Load Sample Clinical Vignette:", list(PRESET_CASES.keys()))
default_text = PRESET_CASES[selected_template] if selected_template in PRESET_CASES else ""

clinical_input = st.text_area(
    "✍️ Enter Raw Doctor's Clinical Note / EHR Narrative:",
    value=default_text,
    height=160,
    placeholder="Type or paste doctor's admission note, discharge summary, or triage presentation here..."
)

# Patient demographic overrides
with st.expander("⚙️ Patient Demographics Extraction & Override (Optional)", expanded=False):
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        override_age = st.number_input("Patient Age", min_value=0, max_value=120, value=65)
    with col_demo2:
        override_gender = st.selectbox("Biological Sex", ["Female", "Male", "Auto-Extract from Note"])

st.markdown("---")

if st.button("🔍 Run Clinical Inference & Differential Diagnosis", type="primary"):
    if not clinical_input.strip():
        st.warning("⚠️ Please provide a clinical note before running diagnostic analysis.")
    else:
        gender_param = None if override_gender == "Auto-Extract from Note" else override_gender
        with st.spinner("Analyzing semantic embeddings and querying historical cases..."):
            result = predict_hybrid_diagnosis(clinical_input, manual_age=override_age, manual_gender=gender_param)

        top_dx = result['top_diagnosis']
        top_prob = result['top_probability']
        top_icd = result['top_icd10']

        # Diagnostic Banner
        st.success(f"### 🎯 Primary Clinical Impression: **{top_dx}** (ICD-10: `{top_icd}`)")
        st.write(f"Confidence / Softmax Posterior Probability: **{top_prob * 100:.1f}%**")

        tab_results1, tab_results2, tab_results3 = st.tabs(["📊 Differential Diagnosis Ranking", "📑 Evidence & Matched Findings", "🔬 Explainability & Demographics"])

        with tab_results1:
            st.subheader("Differential Diagnosis Probability Distribution")
            df_chart = pd.DataFrame([
                {
                    "Diagnosis": p['diagnosis'],
                    "ICD-10": p['icd10'],
                    "Probability": round(p['probability'] * 100, 1),
                    "Similarity Index": round(p['similarity'] * 100, 1)
                }
                for p in result['all_predictions'][:6]
            ])
            st.dataframe(df_chart, use_container_width=True)
            st.bar_chart(data=df_chart.set_index("Diagnosis")["Probability"])

        with tab_results2:
            st.subheader("Historical Evidence & Semantic Case Matches")
            for p in result['all_predictions'][:3]:
                with st.expander(f"📌 {p['diagnosis']} (Similarity: {p['similarity']*100:.1f}%)", expanded=(p['diagnosis'] == top_dx)):
                    st.write(f"**ICD-10 Code:** `{p['icd10']}`")
                    st.write(f"**Key Findings Matched:** {', '.join(p['matched_findings']) if p['matched_findings'] else 'General semantic profile'}")
                    st.caption(f"**Historical Reference Benchmark:** *\"{p['reference_snippet']}\"*")

        with tab_results3:
            st.subheader("Clinical Entity Parsing & Model Trace")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.write("**Extracted Demographics:**")
                st.write(f"- Age: **{result['patient_age']} years**")
                st.write(f"- Gender: **{result['patient_gender']}**")
            with col_t2:
                st.write("**Processed Medical Text:**")
                st.text(result['cleaned_note'])

# Disclaimer
st.info("⚠️ **Clinical Decision Support Notice:** This system is an investigational decision-support AI designed to augment clinical triage. It does not replace independent clinical judgment by licensed medical personnel.")
