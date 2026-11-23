import numpy as np

# Historical reference cases across the 10 diagnostic categories
HISTORICAL_CASES = [
    {
        'id': 'CASE-101',
        'diagnosis': 'Congestive Heart Failure',
        'icd10': 'I50.9',
        'keywords': ['shortness of breath', 'orthopnea', 'edema', 'hypertension', 'elevated bnp', 'ventricular hypertrophy', 'fluid overload'],
        'typical_age': 70,
        'typical_gender': 'Both',
        'sample_note': 'Elderly patient with progressive dyspnea on exertion, paroxysmal nocturnal dyspnea, 3+ pitting lower extremity edema, and elevated BNP.'
    },
    {
        'id': 'CASE-102',
        'diagnosis': 'Acute Myocardial Infarction',
        'icd10': 'I21.9',
        'keywords': ['chest pain', 'substernal', 'diaphoresis', 'radiation to jaw', 'elevated troponin', 'st elevation', 'angina'],
        'typical_age': 62,
        'typical_gender': 'Male',
        'sample_note': 'Crushing retrosternal chest pain radiating to left arm and neck accompanied by nausea and diaphoresis. Serum troponin significantly elevated.'
    },
    {
        'id': 'CASE-103',
        'diagnosis': 'Community-Acquired Pneumonia',
        'icd10': 'J18.9',
        'keywords': ['fever', 'productive cough', 'purulent sputum', 'pleuritic chest pain', 'crackles', 'infiltrate', 'leukocytosis'],
        'typical_age': 55,
        'typical_gender': 'Both',
        'sample_note': 'High-grade fever with chills, productive cough yielding rusty sputum, bronchial breathing, and right lower lobe consolidation on CXR.'
    },
    {
        'id': 'CASE-104',
        'diagnosis': 'Chronic Kidney Disease',
        'icd10': 'N18.9',
        'keywords': ['elevated creatinine', 'proteinuria', 'fatigue', 'electrolyte disturbance', 'hyperkalemia', 'gfr decline', 'uremia'],
        'typical_age': 65,
        'typical_gender': 'Both',
        'sample_note': 'Patient with longstanding poorly controlled diabetic nephropathy, serum creatinine 3.8 mg/dL, severe proteinuria, and eGFR of 18 mL/min.'
    },
    {
        'id': 'CASE-105',
        'diagnosis': 'Asthma Exacerbation',
        'icd10': 'J45.901',
        'keywords': ['wheezing', 'expiratory wheeze', 'bronchospasm', 'peak flow', 'inhaler', 'chest tightness', 'atopy'],
        'typical_age': 32,
        'typical_gender': 'Both',
        'sample_note': 'Young individual with acute onset of severe expiratory wheezing, tachypnea, reduced peak expiratory flow, responsive to nebulized beta-agonists.'
    },
    {
        'id': 'CASE-106',
        'diagnosis': 'Acute Pancreatitis',
        'icd10': 'K85.9',
        'keywords': ['epigastric pain', 'radiation to back', 'elevated lipase', 'elevated amylase', 'gallstones', 'nausea', 'vomiting'],
        'typical_age': 48,
        'typical_gender': 'Both',
        'sample_note': 'Severe piercing epigastric abdominal pain boring through to the back following alcohol ingestion, with serum lipase 4x upper limit of normal.'
    },
    {
        'id': 'CASE-107',
        'diagnosis': 'Diabetic Foot Ulcer',
        'icd10': 'E11.621',
        'keywords': ['neuropathy', 'foot ulcer', 'gangrene', 'peripheral vascular disease', 'loss of sensation', 'hyperglycemia', 'osteomyelitis'],
        'typical_age': 63,
        'typical_gender': 'Both',
        'sample_note': 'Non-healing plantar ulcer over the first metatarsal head in a type 2 diabetic patient with diminished monofilament sensation and peripheral neuropathy.'
    },
    {
        'id': 'CASE-108',
        'diagnosis': 'Meningitis',
        'icd10': 'G03.9',
        'keywords': ['stiff neck', 'nuchal rigidity', 'photophobia', 'altered mental status', 'kernig sign', 'brudzinski sign', 'csf pleocytosis'],
        'typical_age': 28,
        'typical_gender': 'Both',
        'sample_note': 'Acute onset fever, severe headache, prominent nuchal rigidity, positive Kernig and Brudzinski signs, and cloudy cerebrospinal fluid on lumbar puncture.'
    },
    {
        'id': 'CASE-109',
        'diagnosis': 'Alzheimer\'s Disease',
        'icd10': 'G30.9',
        'keywords': ['memory loss', 'cognitive decline', 'disorientation', 'executive dysfunction', 'aphasia', 'moca decline'],
        'typical_age': 78,
        'typical_gender': 'Both',
        'sample_note': 'Insidious progression of short-term memory impairment, spatial disorientation, impaired instrumental activities of daily living, and reduced MoCA score.'
    },
    {
        'id': 'CASE-110',
        'diagnosis': 'Surgical Site Infection',
        'icd10': 'T81.4XXA',
        'keywords': ['wound erythema', 'purulent drainage', 'incisional warmth', 'dehiscence', 'postoperative fever', 'cellulitis'],
        'typical_age': 54,
        'typical_gender': 'Both',
        'sample_note': 'Postoperative day 6 abdominal laparotomy incision exhibiting progressive peri-incisional erythema, localized induration, and foul-smelling purulent discharge.'
    }
]

def calculate_similarity_and_rank(cleaned_note: str, age: int, gender: str):
    """
    Calculates semantic overlap and hybrid clinical similarity between input note
    and reference medical cases.
    """
    lower = cleaned_note.lower()
    scored_cases = []

    for case in HISTORICAL_CASES:
        # Keyword semantic overlap score
        matches = [kw for kw in case['keywords'] if kw in lower]
        overlap_score = len(matches) / max(1, len(case['keywords']))

        # Word jaccard similarity against reference sample note
        words_in = set(lower.split())
        words_ref = set(case['sample_note'].lower().split())
        jaccard = len(words_in.intersection(words_ref)) / max(1, len(words_in.union(words_ref)))

        # Age/Demographic compatibility score
        age_diff = abs(age - case['typical_age'])
        age_affinity = max(0.0, 1.0 - (age_diff / 50.0))

        # Composite score
        composite = (overlap_score * 0.65) + (jaccard * 0.25) + (age_affinity * 0.10)

        scored_cases.append({
            'case_id': case['id'],
            'diagnosis': case['diagnosis'],
            'icd10': case['icd10'],
            'similarity_score': float(composite),
            'matched_findings': matches,
            'reference_snippet': case['sample_note']
        })

    # Sort descending
    scored_cases.sort(key=lambda x: x['similarity_score'], reverse=True)
    return scored_cases
