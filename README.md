# Health Report Prediction Model

A deep learning + NLP system that reads free-text clinical notes and suggests the most likely diagnosis, while surfacing similar historical cases so the suggestion is explainable rather than a black-box label.

## Overview

Clinicians and health-tech teams often have large archives of unstructured clinical notes but no easy way to search them semantically or use them to support new diagnoses. This project turns a 5,000-record clinical notes dataset into a searchable, predictive knowledge base using **Sentence-BERT embeddings**, **cosine similarity**, and a **hybrid neural network**.

## How It Works

1. **Data Cleaning** – Raw clinical notes are lowercased, stripped of noise characters, and standardized while preserving clinically relevant tokens (units like mg/dL, %, etc.).
2. **Embedding Generation** – Each note is encoded into a 384-dimension vector using the `all-MiniLM-L6-v2` Sentence Transformer (a distilled BERT-family model), with embeddings L2-normalized for cosine similarity.
3. **Diagnosis Suggestion** – For a new note, the system computes cosine similarity against all historical embeddings and returns the top-k most similar past cases along with their recorded diagnoses and similarity scores.
4. **Dual-Score Matching** – A key feature: when two historical cases both closely match the query *and* closely resemble each other, the system reports **all three similarity scores** — query-vs-A, query-vs-B, and A-vs-B — so users can see when multiple similar cases agree on a diagnosis, increasing confidence.
5. **Demographic Enrichment** – Age and gender are extracted from note text via regex and scaled/encoded as auxiliary features.
6. **Hybrid Deep Learning Model** – A Keras model combines the text embedding branch (Dense + Dropout) with the demographic branch (Dense) via concatenation, feeding into a softmax classifier over diagnosis classes — combining semantic meaning with patient context.
7. **Evaluation & Visualization** – Model performance is assessed with precision/recall/F1 classification reports (visualized as heatmaps), and note embeddings are projected into 2D via UMAP and plotted interactively with Plotly to visually inspect diagnosis clustering.

## Outcome

- A working pipeline that converts unstructured clinical text into dense semantic vectors and retrieves clinically similar past cases in real time.
- A hybrid model that predicts diagnosis from **both** the language of the note and basic demographics (age, gender), outputting a full probability distribution across diagnosis classes rather than a single label.
- Transparent, evidence-backed predictions: instead of "trust the model," the system shows *which* past cases informed the suggestion and *how similar* they are — both to the new note and to each other.
- Clear visual diagnostics (classification heatmap, UMAP scatter plot, top-5 probability bar charts) that make it easy to see where the model performs well and where diagnosis classes overlap semantically.

## Future Uses

- **Clinical decision support tool**: integrate into an EHR system as a second-opinion assistant that flags similar historical cases for physicians reviewing new patients.
- **Triage automation**: rank incoming patient notes by urgency/similarity to known critical cases.
- **Expanded feature set**: incorporate structured data (labs, vitals, medication history) alongside notes for richer hybrid predictions.
- **Larger/domain-specific transformers**: swap in clinical-domain models (e.g., BioBERT, ClinicalBERT) for improved medical language understanding.
- **Active learning loop**: let clinicians confirm/correct predictions, feeding corrections back to continuously retrain and improve the model.
- **API/dashboard deployment**: wrap the pipeline in a Flask/FastAPI service with a web dashboard so non-technical staff can query the system directly.
- **Multi-label and rare-disease detection**: extend beyond single-diagnosis classification to flag co-occurring conditions and rare diseases underrepresented in training data.

## Tech Stack

Python, Sentence-Transformers, scikit-learn, TensorFlow/Keras, UMAP, Plotly, Matplotlib/Seaborn, Pandas, NumPy.

## Disclaimer

This project is for research and educational purposes only. It is **not** a certified medical device and should not be used for real clinical decision-making without proper validation, regulatory approval, and clinician oversight.# Health-Report-Prediction-Model
