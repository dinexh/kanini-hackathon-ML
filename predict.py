"""
AI-Powered Smart Patient Triage System — Advanced Prediction Module (v3)

UNIQUE CAPABILITIES:
  1. predict_risk()     → Risk level + confidence + probabilities
  2. Triage escalation  → Flags uncertain cases for doctor review
  3. find_similar()     → Finds similar past patients (KNN)
  4. explain_prediction() → Per-patient SHAP explanation

Accepts the same simple input format as the user's sample.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CONSCIOUSNESS_ORDER = ["Unresponsive", "Pain", "Verbal", "Alert"]
CONS_MAP = {level: i for i, level in enumerate(CONSCIOUSNESS_ORDER)}

# Cache for loaded artifacts
_cache = {}


def _load():
    """Load all model artifacts (cached after first call)."""
    if _cache:
        return _cache

    def load_pkl(name):
        with open(os.path.join(OUTPUT_DIR, name), "rb") as f:
            return pickle.load(f)

    _cache["model"] = load_pkl("best_triage_model.pkl")
    _cache["xgb_model"] = load_pkl("xgb_model.pkl")
    _cache["scaler"] = load_pkl("scaler.pkl")
    _cache["label_encoders"] = load_pkl("label_encoders.pkl")
    _cache["target_le"] = load_pkl("target_label_encoder.pkl")
    _cache["mlbs"] = load_pkl("multi_label_binarizers.pkl")
    _cache["knn"] = load_pkl("knn_similarity.pkl")
    _cache["X_train"] = load_pkl("training_data_scaled.pkl")
    _cache["y_train"] = load_pkl("training_labels.pkl")
    _cache["raw_data"] = load_pkl("raw_data.pkl")

    with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "r") as f:
        _cache["meta"] = json.load(f)

    return _cache


def _build_features(patient_data: dict) -> pd.DataFrame:
    """
    Convert raw patient input into the 47-feature engineered vector.
    Applies the same preprocessing as training.
    """
    c = _load()
    meta = c["meta"]
    feature_names = meta["feature_names"]
    all_symptoms = meta["all_symptoms"]
    all_conditions = meta["all_conditions"]

    # ── Parse raw inputs ──
    age = patient_data.get("Age", 30)

    gender_raw = patient_data.get("Gender", "Male")
    le_gender = c["label_encoders"]["Gender"]
    gender = le_gender.transform([gender_raw])[0] if gender_raw in le_gender.classes_ else 0

    bp_raw = str(patient_data.get("Blood_Pressure", "120/80"))
    bp_parts = bp_raw.split("/")
    bp_sys = float(bp_parts[0]) if len(bp_parts) >= 1 else 120
    bp_dia = float(bp_parts[1]) if len(bp_parts) >= 2 else 80

    hr = patient_data.get("Heart_Rate", 80)
    temp = patient_data.get("Temperature", 98.6)
    spo2 = patient_data.get("SpO2", 96)
    rr = patient_data.get("Respiratory_Rate", 18)
    cons_raw = patient_data.get("Consciousness_Level", "Alert")
    cons = CONS_MAP.get(cons_raw, 3)
    pain = patient_data.get("Pain_Level", 0)

    # Symptoms → binary
    sym_raw = patient_data.get("Symptoms", "None")
    sym_list = ([s.strip() for s in sym_raw.split(",")]
                if isinstance(sym_raw, str) and sym_raw.strip().lower() != "none" else [])
    sym_binary = {f"Sym_{s.replace(' ', '_')}": (1 if s in sym_list else 0) for s in all_symptoms}
    symptom_count = sum(sym_binary.values())

    # Conditions → binary
    cond_raw = patient_data.get("Pre_Existing_Conditions", "None")
    cond_list = ([c.strip() for c in cond_raw.split(",")]
                 if isinstance(cond_raw, str) and cond_raw.strip().lower() != "none" else [])
    cond_binary = {f"Cond_{c.replace(' ', '_')}": (1 if c in cond_list else 0) for c in all_conditions}
    condition_count = sum(cond_binary.values())

    # ── NEWS2 Score ──
    news2 = 0
    news2 += (3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3)
    news2 += (3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0)
    news2 += (3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3)
    temp_c = (temp - 32) * 5 / 9
    news2 += (3 if temp_c <= 35.0 else 1 if temp_c <= 36.0 else 0 if temp_c <= 38.0 else 1 if temp_c <= 39.0 else 2)
    news2 += (3 if bp_sys <= 90 else 2 if bp_sys <= 100 else 1 if bp_sys <= 110 else 0 if bp_sys <= 219 else 3)
    news2 += (3 if cons < 3 else 0)

    # ── Feature Interactions ──
    map_val = bp_dia + (bp_sys - bp_dia) / 3
    shock_idx = hr / max(bp_sys, 1)
    mod_shock = hr / max(map_val, 1)
    oxy_stress = rr / max(spo2, 1)
    fever_hypoxia = (1 if temp > 100.4 else 0) * (100 - spo2)
    age_risk = 2 if age > 65 else (2 if age < 5 else (1 if age > 50 else 0))
    age_vuln = age_risk * (pain / 10 + 0.5)
    sym_severity = symptom_count * (4 - cons) / 4
    comorbidity_burden = condition_count * (1.5 if age > 60 else 1.0)
    hemo_instab = abs(hr - 75) / 75 + abs(bp_sys - 120) / 120

    # ── Assemble feature dict ──
    row = {
        "Age": age, "Gender": gender,
        "Heart_Rate": hr, "Temperature": temp, "SpO2": spo2,
        "Respiratory_Rate": rr, "Consciousness_Level": cons, "Pain_Level": pain,
        "BP_Systolic": bp_sys, "BP_Diastolic": bp_dia,
        "Symptom_Count": symptom_count, "Condition_Count": condition_count,
        "NEWS2_Score": news2,
        "Shock_Index": shock_idx, "Modified_Shock_Index": mod_shock,
        "Oxy_Stress": oxy_stress, "Fever_Hypoxia": fever_hypoxia,
        "Age_Vulnerability": age_vuln, "Symptom_Severity": sym_severity,
        "Comorbidity_Burden": comorbidity_burden,
        "Hemodynamic_Instability": hemo_instab,
    }
    row.update(sym_binary)
    row.update(cond_binary)

    df = pd.DataFrame([{f: row.get(f, 0) for f in feature_names}], columns=feature_names)
    df_scaled = pd.DataFrame(c["scaler"].transform(df), columns=feature_names)

    return df_scaled


def predict_risk(patient_data: dict) -> dict:
    """
    Predict risk level with confidence-based triage escalation.

    If confidence < 60%, the result is flagged as "UNCERTAIN — ESCALATE TO DOCTOR".

    Returns
    -------
    dict with keys: patient_id, risk_level, confidence, probabilities,
                    needs_escalation, escalation_reason, news2_score
    """
    c = _load()
    meta = c["meta"]
    threshold = meta.get("escalation_threshold", 0.60)

    df_scaled = _build_features(patient_data)

    # Predict
    pred = c["model"].predict(df_scaled)[0]
    proba = c["model"].predict_proba(df_scaled)[0]

    risk_label = c["target_le"].inverse_transform([pred])[0]
    confidence = float(np.max(proba))

    prob_dict = {
        cls: round(float(p), 4)
        for cls, p in zip(c["target_le"].classes_, proba)
    }

    # NEWS2 score for context
    news2 = 0.0
    if "NEWS2_Score" in meta["feature_names"]:
        news2 = float(df_scaled["NEWS2_Score"].iloc[0])

    # Escalation logic
    needs_escalation = False
    escalation_reason = None

    if confidence < threshold:
        needs_escalation = True
        escalation_reason = f"Low model confidence ({confidence*100:.1f}% < {threshold*100:.0f}%)"

    # Also escalate if top 2 predictions are very close (ambiguous case)
    sorted_proba = sorted(proba, reverse=True)
    if len(sorted_proba) >= 2 and (sorted_proba[0] - sorted_proba[1]) < 0.15:
        needs_escalation = True
        escalation_reason = (f"Ambiguous prediction — top 2 classes are close: "
                             f"{sorted_proba[0]*100:.1f}% vs {sorted_proba[1]*100:.1f}%")

    return {
        "patient_id": patient_data.get("Patient_ID", "Unknown"),
        "risk_level": risk_label,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "needs_escalation": needs_escalation,
        "escalation_reason": escalation_reason,
        "news2_score": round(news2, 2),
    }


def find_similar_patients(patient_data: dict, n_similar: int = 5) -> list:
    """
    Find the most similar past patients using K-Nearest Neighbors.

    Returns a list of similar patient records with their risk levels
    and similarity scores (lower distance = more similar).
    """
    c = _load()
    df_scaled = _build_features(patient_data)

    distances, indices = c["knn"].kneighbors(df_scaled, n_neighbors=n_similar)

    raw_data = c["raw_data"]
    target_le = c["target_le"]
    y = c["y_train"]

    similar = []
    for dist, idx in zip(distances[0], indices[0]):
        record = raw_data.iloc[idx].to_dict()
        record["similarity_distance"] = round(float(dist), 4)
        record["risk_level_actual"] = target_le.inverse_transform([y.iloc[idx]])[0]
        similar.append(record)

    return similar


def explain_prediction(patient_data: dict) -> dict:
    """
    Generate a per-patient SHAP explanation showing which features
    are pushing the prediction toward High / Medium / Low risk.

    Returns the top contributing features with their SHAP values.
    """
    c = _load()
    meta = c["meta"]
    import shap

    df_scaled = _build_features(patient_data)
    xgb_model = c["xgb_model"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(df_scaled)

    # Get explanation for the predicted class
    pred = int(xgb_model.predict(df_scaled)[0])
    risk_label = c["target_le"].inverse_transform([pred])[0]

    # Handle multi-class SHAP: can be list of arrays or 3D array
    if isinstance(shap_values, list):
        sv = np.array(shap_values[pred]).flatten()
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, pred] if shap_values.shape[0] == 1 else shap_values[pred][0]
    else:
        sv = np.array(shap_values).flatten()

    feature_names = meta["feature_names"]
    contributions = sorted(
        [(f, float(v)) for f, v in zip(feature_names, sv)],
        key=lambda x: abs(x[1]), reverse=True
    )

    top_positive = [(f, round(float(v), 4)) for f, v in contributions if v > 0][:8]
    top_negative = [(f, round(float(v), 4)) for f, v in contributions if v < 0][:5]

    return {
        "predicted_class": risk_label,
        "top_risk_factors": top_positive,
        "top_protective_factors": top_negative,
        "interpretation": (
            f"The model predicted '{risk_label}' risk. "
            f"Key risk factors: {', '.join(f[0] for f in top_positive[:3])}. "
            f"Protective factors: {', '.join(f[0] for f in top_negative[:3]) or 'None significant'}."
        ),
    }


# ───────────────────────────────────────────────────────────────────
# CLI Demo
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  🏥 Smart Patient Triage — Advanced Prediction Demo")
    print("=" * 65)

    patients = {
        "HIGH-RISK": {
            "Patient_ID": "PT-10001", "Age": 72, "Gender": "Male",
            "Symptoms": "Fever, Cough, Chest Pain, Shortness of Breath, Fatigue, Nausea",
            "Blood_Pressure": "185/110", "Heart_Rate": 130, "Temperature": 103.8,
            "SpO2": 88, "Respiratory_Rate": 32, "Consciousness_Level": "Verbal",
            "Pain_Level": 9,
            "Pre_Existing_Conditions": "Diabetes, Hypertension, Heart Disease, Kidney Disease",
        },
        "LOW-RISK": {
            "Patient_ID": "PT-10002", "Age": 25, "Gender": "Female",
            "Symptoms": "Cough, Headache, Sore Throat",
            "Blood_Pressure": "118/76", "Heart_Rate": 75, "Temperature": 98.6,
            "SpO2": 98, "Respiratory_Rate": 16, "Consciousness_Level": "Alert",
            "Pain_Level": 2, "Pre_Existing_Conditions": "None",
        },
        "MEDIUM-RISK": {
            "Patient_ID": "PT-10003", "Age": 55, "Gender": "Male",
            "Symptoms": "Fever, Cough, Fatigue, Headache, Dizziness, Body Ache",
            "Blood_Pressure": "142/88", "Heart_Rate": 95, "Temperature": 101.2,
            "SpO2": 94, "Respiratory_Rate": 22, "Consciousness_Level": "Alert",
            "Pain_Level": 5, "Pre_Existing_Conditions": "Diabetes, Asthma",
        },
    }

    for label, patient in patients.items():
        result = predict_risk(patient)
        print(f"\n{'─' * 65}")
        print(f"  📋 {label} Patient  (ID: {result['patient_id']})")
        print(f"{'─' * 65}")
        print(f"  🩺 Predicted Risk Level : {result['risk_level']}")
        print(f"  📊 Confidence           : {result['confidence'] * 100:.1f}%")
        print(f"  📈 Probabilities        : {result['probabilities']}")

        if result["needs_escalation"]:
            print(f"  🚨 ESCALATION           : {result['escalation_reason']}")
        else:
            print(f"  ✅ Escalation           : Not needed")

        # SHAP explanation
        explanation = explain_prediction(patient)
        print(f"  🔍 Top risk factors     : {', '.join(f'{f}({v:+.3f})' for f, v in explanation['top_risk_factors'][:4])}")
        print(f"  🛡️  Protective factors   : {', '.join(f'{f}({v:+.3f})' for f, v in explanation['top_protective_factors'][:3]) or 'None'}")

        # Similar patients
        similar = find_similar_patients(patient, n_similar=3)
        print(f"  👥 Similar past patients:")
        for sp in similar:
            print(f"     → {sp['Patient_ID']} | Age {sp['Age']} | Risk: {sp['risk_level_actual']} | "
                  f"Distance: {sp['similarity_distance']}")

    print(f"\n{'=' * 65}")
    print("  ✅ Advanced demo complete!")
    print("=" * 65)
