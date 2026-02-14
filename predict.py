"""
AI-Powered Smart Patient Triage System — Prediction + Priority Queue (v4)

API:
  triage_patient(data)       → risk + department + priority score + queue position
  find_similar_patients(data)→ similar past patients (KNN)
  explain_prediction(data)   → SHAP explanation
  PatientQueue               → priority queue manager for a department
"""

import os, json, pickle, heapq, numpy as np, pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CONSCIOUSNESS_ORDER = ["Unresponsive", "Pain", "Verbal", "Alert"]
CONS_MAP = {level: i for i, level in enumerate(CONSCIOUSNESS_ORDER)}
_cache = {}


def _load():
    if _cache: return _cache
    def lp(n):
        with open(os.path.join(OUTPUT_DIR, n), "rb") as f: return pickle.load(f)
    _cache["risk_model"] = lp("best_risk_model.pkl")
    _cache["dept_model"] = lp("dept_model.pkl")
    _cache["xgb_risk"] = lp("xgb_risk_model.pkl")
    _cache["scaler"] = lp("scaler.pkl")
    _cache["label_encoders"] = lp("label_encoders.pkl")
    _cache["risk_le"] = lp("risk_label_encoder.pkl")
    _cache["dept_le"] = lp("dept_label_encoder.pkl")
    _cache["mlbs"] = lp("multi_label_binarizers.pkl")
    _cache["knn"] = lp("knn_similarity.pkl")
    _cache["X_train"] = lp("training_data_scaled.pkl")
    _cache["y_risk"] = lp("training_risk_labels.pkl")
    _cache["y_dept"] = lp("training_dept_labels.pkl")
    _cache["raw_data"] = lp("raw_data.pkl")
    with open(os.path.join(OUTPUT_DIR, "model_metadata.json")) as f:
        _cache["meta"] = json.load(f)
    return _cache


def _build_features(data):
    c = _load(); meta = c["meta"]; fn = meta["feature_names"]
    age = data.get("Age", 30)
    le_g = c["label_encoders"]["Gender"]
    g_raw = data.get("Gender", "Male")
    gender = le_g.transform([g_raw])[0] if g_raw in le_g.classes_ else 0
    bp_raw = str(data.get("Blood_Pressure", "120/80")).split("/")
    bp_s = float(bp_raw[0]) if len(bp_raw) >= 1 else 120
    bp_d = float(bp_raw[1]) if len(bp_raw) >= 2 else 80
    hr = data.get("Heart_Rate", 80); temp = data.get("Temperature", 98.6)
    spo2 = data.get("SpO2", 96); rr = data.get("Respiratory_Rate", 18)
    cons = CONS_MAP.get(data.get("Consciousness_Level", "Alert"), 3)
    pain = data.get("Pain_Level", 0)

    sym_raw = data.get("Symptoms", "None")
    sym_list = [s.strip() for s in sym_raw.split(",")] if isinstance(sym_raw, str) and sym_raw.strip().lower() != "none" else []
    sym_bin = {f"Sym_{s.replace(' ','_')}": (1 if s in sym_list else 0) for s in meta["all_symptoms"]}
    sym_count = sum(sym_bin.values())

    cond_raw = data.get("Pre_Existing_Conditions", "None")
    cond_list = [c.strip() for c in cond_raw.split(",")] if isinstance(cond_raw, str) and cond_raw.strip().lower() != "none" else []
    cond_bin = {f"Cond_{c.replace(' ','_')}": (1 if c in cond_list else 0) for c in meta["all_conditions"]}
    cond_count = sum(cond_bin.values())

    # NEWS2
    n2 = 0
    n2 += (3 if rr<=8 else 1 if rr<=11 else 0 if rr<=20 else 2 if rr<=24 else 3)
    n2 += (3 if spo2<=91 else 2 if spo2<=93 else 1 if spo2<=95 else 0)
    n2 += (3 if hr<=40 else 1 if hr<=50 else 0 if hr<=90 else 1 if hr<=110 else 2 if hr<=130 else 3)
    tc = (temp-32)*5/9
    n2 += (3 if tc<=35 else 1 if tc<=36 else 0 if tc<=38 else 1 if tc<=39 else 2)
    n2 += (3 if bp_s<=90 else 2 if bp_s<=100 else 1 if bp_s<=110 else 0 if bp_s<=219 else 3)
    n2 += (3 if cons<3 else 0)

    # Interactions
    map_v = bp_d+(bp_s-bp_d)/3
    row = {"Age":age,"Gender":gender,"Heart_Rate":hr,"Temperature":temp,"SpO2":spo2,
           "Respiratory_Rate":rr,"Consciousness_Level":cons,"Pain_Level":pain,
           "BP_Systolic":bp_s,"BP_Diastolic":bp_d,"Symptom_Count":sym_count,
           "Condition_Count":cond_count,"NEWS2_Score":n2,
           "Shock_Index":hr/max(bp_s,1),"Modified_Shock_Index":hr/max(map_v,1),
           "Oxy_Stress":rr/max(spo2,1),
           "Fever_Hypoxia":(1 if temp>100.4 else 0)*(100-spo2),
           "Age_Vulnerability":(2 if age>65 else 2 if age<5 else 1 if age>50 else 0)*(pain/10+0.5),
           "Symptom_Severity":sym_count*(4-cons)/4,
           "Comorbidity_Burden":cond_count*(1.5 if age>60 else 1.0),
           "Hemodynamic_Instability":abs(hr-75)/75+abs(bp_s-120)/120}
    row.update(sym_bin); row.update(cond_bin)
    df = pd.DataFrame([{f: row.get(f,0) for f in fn}], columns=fn)
    return pd.DataFrame(c["scaler"].transform(df), columns=fn)


def compute_priority_score(data, risk_level, news2_raw):
    """Compute priority score (0-100). Higher = more urgent = treated first."""
    w = {"risk": 0.40, "news2": 0.25, "vitals": 0.20, "age": 0.15}
    risk_score = {"High": 100, "Medium": 50, "Low": 10}.get(risk_level, 10)
    news2_norm = min(news2_raw / 18 * 100, 100)

    # Vitals sub-score
    vs = 0
    spo2 = data.get("SpO2", 96)
    hr = data.get("Heart_Rate", 80)
    if spo2 < 90: vs += 40
    elif spo2 < 94: vs += 20
    if hr > 120 or hr < 50: vs += 30
    elif hr > 100: vs += 15
    cons = data.get("Consciousness_Level", "Alert")
    if cons in ("Unresponsive", "Pain"): vs += 30
    elif cons == "Verbal": vs += 15
    vs = min(vs, 100)

    # Age sub-score
    age = data.get("Age", 30)
    age_s = 80 if age > 70 else (60 if age > 60 else (70 if age < 5 else (30 if age < 12 else 20)))

    priority = (w["risk"]*risk_score + w["news2"]*news2_norm +
                w["vitals"]*vs + w["age"]*age_s)
    return round(min(priority, 100), 1)


def triage_patient(data):
    """
    Full triage: predict risk level, assign department, compute priority.

    Returns dict with: patient_id, risk_level, department, priority_score,
    confidence, needs_escalation, news2_score, probabilities
    """
    c = _load(); meta = c["meta"]
    threshold = meta.get("escalation_threshold", 0.60)
    df_scaled = _build_features(data)

    # Risk prediction
    risk_pred = c["risk_model"].predict(df_scaled)[0]
    risk_proba = c["risk_model"].predict_proba(df_scaled)[0]
    risk_label = c["risk_le"].inverse_transform([risk_pred])[0]
    confidence = float(np.max(risk_proba))

    # Department prediction
    dept_pred = c["dept_model"].predict(df_scaled)[0]
    dept_proba = c["dept_model"].predict_proba(df_scaled)[0]
    dept_label = c["dept_le"].inverse_transform([dept_pred])[0]
    dept_confidence = float(np.max(dept_proba))

    # NEWS2 (unscaled)
    n2_idx = meta["feature_names"].index("NEWS2_Score") if "NEWS2_Score" in meta["feature_names"] else None
    news2_raw = 0
    if n2_idx is not None:
        news2_raw = float(df_scaled.iloc[0, n2_idx] * c["scaler"].scale_[n2_idx] + c["scaler"].mean_[n2_idx])

    # Priority score
    priority = compute_priority_score(data, risk_label, max(news2_raw, 0))

    # Escalation
    needs_esc = False; esc_reason = None
    if confidence < threshold:
        needs_esc = True; esc_reason = f"Low confidence ({confidence*100:.1f}%)"
    sorted_p = sorted(risk_proba, reverse=True)
    if len(sorted_p) >= 2 and (sorted_p[0] - sorted_p[1]) < 0.15:
        needs_esc = True; esc_reason = f"Ambiguous: {sorted_p[0]*100:.1f}% vs {sorted_p[1]*100:.1f}%"

    # ─────────────────────────────────────────────────────────────
    # GUARDRAILS: Force-correct based on clinical rules
    # ─────────────────────────────────────────────────────────────
    risk_label, dept_label = apply_clinical_guardrails(data, risk_label, dept_label, risk_proba)

    return {
        "patient_id": data.get("Patient_ID", "Unknown"),
        "risk_level": risk_label,
        "department": dept_label,
        "priority_score": priority,
        "confidence": round(confidence, 4),
        "dept_confidence": round(dept_confidence, 4),
        "needs_escalation": needs_esc,
        "escalation_reason": esc_reason,
        "news2_score": round(max(news2_raw, 0), 1),
        "risk_probabilities": {cl: round(float(p),4) for cl,p in zip(c["risk_le"].classes_, risk_proba)},
        "dept_probabilities": {cl: round(float(p),4) for cl,p in zip(c["dept_le"].classes_, dept_proba)},
    }


# ── Priority Queue ────────────────────────────────────────────────
class PatientQueue:
    """
    Priority queue for a hospital department.
    Patients are ordered by priority_score (highest = first).
    """
    def __init__(self, department):
        self.department = department
        self._heap = []  # min-heap, negate score for max-priority
        self._counter = 0

    def add_patient(self, triage_result):
        self._counter += 1
        entry = (-triage_result["priority_score"], self._counter,
                 triage_result["patient_id"], triage_result)
        heapq.heappush(self._heap, entry)

    def next_patient(self):
        if self._heap:
            neg_score, _, pid, result = heapq.heappop(self._heap)
            return result
        return None

    def peek(self):
        if self._heap:
            return self._heap[0][3]
        return None

    @property
    def size(self):
        return len(self._heap)

    def get_queue(self):
        sorted_q = sorted(self._heap)
        return [{"position": i+1, "patient_id": e[2],
                 "priority_score": -e[0], "risk_level": e[3]["risk_level"]}
                for i, e in enumerate(sorted_q)]


class HospitalQueueManager:
    """Manages queues for all departments."""
    def __init__(self):
        self.queues = {}

    def admit_patient(self, patient_data):
        result = triage_patient(patient_data)
        dept = result["department"]
        if dept not in self.queues:
            self.queues[dept] = PatientQueue(dept)
        self.queues[dept].add_patient(result)
        result["queue_position"] = self.queues[dept].size
        return result

    def get_next(self, department):
        if department in self.queues:
            return self.queues[department].next_patient()
        return None

    def view_all_queues(self):
        return {dept: q.get_queue() for dept, q in self.queues.items()}

    def summary(self):
        return {dept: q.size for dept, q in self.queues.items()}


def find_similar_patients(data, n=5):
    c = _load(); df_s = _build_features(data)
    dists, idxs = c["knn"].kneighbors(df_s, n_neighbors=n)
    raw = c["raw_data"]; rl = c["risk_le"]; yr = c["y_risk"]
    return [{"Patient_ID": raw.iloc[i]["Patient_ID"], "Age": raw.iloc[i]["Age"],
             "risk_level": rl.inverse_transform([yr.iloc[i]])[0],
             "distance": round(float(d), 4)}
            for d, i in zip(dists[0], idxs[0])]


def apply_clinical_guardrails(data, risk_label, dept_label, risk_proba):
    """
    Force-correct predictions based on strict clinical rules (Guardrails).
    Overrides ML model if specific critical criteria are met.
    """
    s = data.get("Symptoms", "")
    c = data.get("Pre_Existing_Conditions", "")
    sp = data.get("SpO2", 98)
    hr = data.get("Heart_Rate", 80)
    sys = float(str(data.get("Blood_Pressure", "120/80")).split("/")[0])
    cons = data.get("Consciousness_Level", "Alert")

    # 1. Critical Vitals -> High Risk Overrides
    if sp < 90 or hr > 130 or sys < 80 or cons in ["Unresponsive", "Pain"]:
        risk_label = "High"
    
    # 2. Condition-Specific Routing & Risk
    # Stroke -> Neurology
    if "Slurred Speech" in s or "Facial Droop" in s or "Weakness" in s:
        dept_label = "Neurology"
        risk_label = "High"
    
    # DKA -> Endocrinology (or Emergency) -> Settle on Emergency for initial triage
    if "Type 1 Diabetes" in c and ("Confusion" in s or "Vomiting" in s):
        risk_label = "High"
        # Keep Dept as predicted unless it's stupid. If ML said Gastro, change to Emergency?
        # Let's force Emergency for DKA
        dept_label = "Emergency"

    # Kidney Stones -> Nephrology
    if "Kidney Stones" in c or "Blood in Urine" in s or "Flank Pain" in s:
        dept_label = "Nephrology"
    
    # Migraine -> Neurology (Low Risk)
    if "Migraine" in c or "Light Sensitivity" in s:
        if risk_label == "High": pass # Don't downgrade if they are actually dying
        else:
            dept_label = "Neurology"

    # Anxiety -> Psychology/GenMed? 
    # If ML predicts Cardio for Anxiety, maybe override to GenMed if vitals stable?
    if "Anxiety" in c and "Chest Tightness" in s:
        if sys < 140 and hr < 110 and sp > 95:
            # Panic attack mimicking MI
            risk_label = "Medium" # Not Low because needs rule out
            dept_label = "General Medicine" # Or Psychiatry if we had it.

    return risk_label, dept_label


def text_explanation(pos, neg):
    """Generate a human-friendly explanation string."""
    factors = []
    if pos:
        factors.append(f"Risk increased by {pos[0][0].replace('Sym_','').replace('Cond_','').replace('_',' ')}")
    if neg:
        factors.append(f"offset by {neg[0][0].replace('Sym_','').replace('Cond_','').replace('_',' ')}")
    return "; ".join(factors) + "."

def explain_prediction(data):
    c = _load(); meta = c["meta"]; import shap
    df_s = _build_features(data); xm = c["xgb_risk"]
    exp = shap.TreeExplainer(xm); sv = exp.shap_values(df_s)
    pred = int(xm.predict(df_s)[0])
    rl = c["risk_le"].inverse_transform([pred])[0]
    
    # Getting SHAP values
    if isinstance(sv, list): vals = np.array(sv[pred]).flatten()
    elif sv.ndim == 3: vals = sv[0,:,pred] if sv.shape[0]==1 else sv[pred][0]
    else: vals = np.array(sv).flatten()
    
    contribs = sorted([(f,float(v)) for f,v in zip(meta["feature_names"],vals)], key=lambda x:abs(x[1]), reverse=True)
    
    # Sanitize Protective Factors (Negative SHAP)
    # Don't list "Diabetes" as protective.
    # Only list Vitals (within normal range) or "None" conditions.
    
    def is_valid_protective(feat, val):
        if "Cond_" in feat and "None" not in feat: return False # Diseases aren't protective
        if "Sym_" in feat and "None" not in feat: return False # Symptoms aren't protective (rarely)
        if "Age" in feat and data.get("Age") > 60: return False # Age > 60 isn't protective
        return True

    pos = [(f,round(v,4)) for f,v in contribs if v>0][:5]
    neg = [(f,round(v,4)) for f,v in contribs if v<0 and is_valid_protective(f,v)][:5]
    
    return {"predicted_class": rl, "top_risk_factors": pos, "top_protective_factors": neg,
            "interpretation": text_explanation(pos, neg)}


# ── CLI DEMO ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  🏥 Smart Patient Triage — Department Routing + Priority Queue")
    print("="*65)

    patients = [
        {"Patient_ID":"PT-10001","Age":72,"Gender":"Male","Symptoms":"Fever, Cough, Chest Pain, Shortness of Breath",
         "Blood_Pressure":"185/110","Heart_Rate":130,"Temperature":103.8,"SpO2":88,
         "Respiratory_Rate":32,"Consciousness_Level":"Verbal","Pain_Level":9,
         "Pre_Existing_Conditions":"Diabetes, Hypertension, Heart Disease"},
        {"Patient_ID":"PT-10002","Age":25,"Gender":"Female","Symptoms":"Cough, Headache, Sore Throat",
         "Blood_Pressure":"118/76","Heart_Rate":75,"Temperature":98.6,"SpO2":98,
         "Respiratory_Rate":16,"Consciousness_Level":"Alert","Pain_Level":2,
         "Pre_Existing_Conditions":"None"},
        {"Patient_ID":"PT-10003","Age":55,"Gender":"Male","Symptoms":"Fever, Cough, Fatigue, Dizziness",
         "Blood_Pressure":"142/88","Heart_Rate":95,"Temperature":101.2,"SpO2":94,
         "Respiratory_Rate":22,"Consciousness_Level":"Alert","Pain_Level":5,
         "Pre_Existing_Conditions":"Diabetes, Asthma"},
        {"Patient_ID":"PT-10004","Age":8,"Gender":"Female","Symptoms":"Fever, Nausea, Vomiting, Abdominal Pain",
         "Blood_Pressure":"100/65","Heart_Rate":110,"Temperature":102.5,"SpO2":97,
         "Respiratory_Rate":24,"Consciousness_Level":"Alert","Pain_Level":6,
         "Pre_Existing_Conditions":"None"},
        {"Patient_ID":"PT-10005","Age":65,"Gender":"Male","Symptoms":"Confusion, Headache, Dizziness",
         "Blood_Pressure":"160/95","Heart_Rate":88,"Temperature":98.8,"SpO2":95,
         "Respiratory_Rate":18,"Consciousness_Level":"Verbal","Pain_Level":7,
         "Pre_Existing_Conditions":"Hypertension, Stroke History"},
    ]

    # Create hospital queue manager
    hospital = HospitalQueueManager()

    print("\n📥 Admitting patients...\n")
    for p in patients:
        result = hospital.admit_patient(p)
        print(f"  📋 {result['patient_id']}")
        print(f"     Risk: {result['risk_level']} ({result['confidence']*100:.1f}%) | "
              f"Dept: {result['department']} ({result['dept_confidence']*100:.1f}%)")
        print(f"     Priority: {result['priority_score']}/100 | "
              f"NEWS2: {result['news2_score']} | "
              f"{'🚨 ESCALATE: '+result['escalation_reason'] if result['needs_escalation'] else '✅ OK'}")

    print(f"\n{'─'*65}")
    print("  🏥 Department Queues (ordered by priority)")
    print(f"{'─'*65}")

    all_queues = hospital.view_all_queues()
    for dept, queue in sorted(all_queues.items()):
        print(f"\n  🏷️  {dept} ({len(queue)} patients)")
        for entry in queue:
            print(f"     #{entry['position']}  {entry['patient_id']}  "
                  f"Priority: {entry['priority_score']}  Risk: {entry['risk_level']}")

    print(f"\n{'─'*65}")
    print("  👨‍⚕️ Doctor calls next patient from each department")
    print(f"{'─'*65}")
    for dept in sorted(all_queues.keys()):
        nxt = hospital.get_next(dept)
        if nxt:
            print(f"  {dept} → {nxt['patient_id']} (Priority: {nxt['priority_score']}, Risk: {nxt['risk_level']})")

    # SHAP explanation for first patient
    print(f"\n{'─'*65}")
    print("  🔍 SHAP Explanation for PT-10001")
    print(f"{'─'*65}")
    exp = explain_prediction(patients[0])
    print(f"  Prediction: {exp['predicted_class']}")
    print(f"  Risk factors: {', '.join(f'{f}({v:+.3f})' for f,v in exp['top_risk_factors'][:4])}")
    print(f"  Protective:   {', '.join(f'{f}({v:+.3f})' for f,v in exp['top_protective_factors'][:3]) or 'None'}")

    print(f"\n{'='*65}")
    print("  ✅ Demo complete!")
    print("="*65)
