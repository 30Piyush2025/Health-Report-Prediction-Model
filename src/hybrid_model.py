import numpy as np
from src.similarity_matcher import calculate_similarity_and_rank, HISTORICAL_CASES
from src.preprocessor import clean_clinical_note, extract_age, extract_gender

def predict_hybrid_diagnosis(raw_note: str, manual_age=None, manual_gender=None):
    """
    Executes the Hybrid Clinical Diagnosis Pipeline:
    1. Preprocesses raw text note.
    2. Dynamically extracts or overrides patient demographics (Age, Gender).
    3. Runs semantic case retrieval and computes probability distribution across 10 diagnoses.
    """
    cleaned = clean_clinical_note(raw_note)
    extracted_age = extract_age(raw_note)
    extracted_gender = extract_gender(raw_note)

    age = manual_age if manual_age is not None else extracted_age
    gender = manual_gender if manual_gender is not None else extracted_gender

    # Run similarity engine
    ranked_cases = calculate_similarity_and_rank(cleaned, age, gender)

    # Convert raw similarity scores into normalized probability distribution (Softmax with temperature)
    scores = np.array([c['similarity_score'] for c in ranked_cases])
    temperature = 0.25
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    probabilities = exp_scores / np.sum(exp_scores)

    results = []
    for i, case in enumerate(ranked_cases):
        results.append({
            'diagnosis': case['diagnosis'],
            'icd10': case['icd10'],
            'probability': float(probabilities[i]),
            'similarity': case['similarity_score'],
            'matched_findings': case['matched_findings'],
            'reference_snippet': case['reference_snippet']
        })

    return {
        'cleaned_note': cleaned,
        'patient_age': age,
        'patient_gender': gender,
        'top_diagnosis': results[0]['diagnosis'],
        'top_probability': results[0]['probability'],
        'top_icd10': results[0]['icd10'],
        'all_predictions': results
    }
