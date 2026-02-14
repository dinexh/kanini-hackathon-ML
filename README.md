# 🏥 AI-Powered Smart Patient Triage System

An ML-based system that analyzes patient symptoms and medical history to classify risk levels, recommend departments, provide explainable insights, and support efficient prioritization.

---

## How It Works

```
Patient Data Input (Age, Vitals, Symptoms, Conditions)
       │
       ▼
┌──────────────────────────────────────────────┐
│  CTGAN Synthetic Data (SDV)                   │
│  Learns realistic feature correlations        │
│  HR↔SpO₂ (r=-0.86), HR↔RR (r=+0.84)        │
└──────────────────┬───────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌────────────────┐   ┌────────────────┐
│  RISK MODEL     │   │  DEPT MODEL    │
│  Random Forest  │   │  XGBoost       │
│  F1 = 0.7453    │   │  F1 = 0.9860   │
│  Low/Med/High   │   │  10 Departments│
└───────┬────────┘   └───────┬────────┘
        │                     │
        └──────────┬──────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  EXPLAINABILITY (SHAP)                        │
│  Shows risk factors + protective factors      │
│  Confidence score + escalation alerts         │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  PRIORITY QUEUE                               │
│  NEWS2 score + weighted priority (0-100)      │
│  Patients sorted by severity per department   │
└──────────────────────────────────────────────┘
```

---

## Pipeline (3 Scripts)

### Step 1: `generate_dataset.py` — Synthetic Data with SDV CTGAN

Creates **5,000 synthetic patients** using a hybrid CTGAN approach:

1. **1,500-row seed** with 30+ clinical archetypes (e.g., cardiac emergencies, mild fevers, borderline cases)
2. **CTGAN** (300 epochs) learns all feature correlations and risk patterns
3. Generates 5,000 patients — **CTGAN decides vitals, symptoms, conditions, and risk level**
4. **Realistic noise injection**:
   - 15% borderline/ambiguous patients (overlapping class boundaries)
   - 5% label noise (250 patients with swapped risk labels)
   - 3% random vital perturbation (150 patients)
   - 3% missing vitals (simulates incomplete records)
5. **Department** assigned via clinical symptom routing (mirrors real hospital triage)

```bash
python generate_dataset.py
# → data/patient_triage_dataset.csv (5000 rows × 14 columns)
```

**SDV-learned correlations:**
| Feature Pair | Correlation | Clinical Meaning |
|---|---|---|
| HR ↔ SpO₂ | r = -0.86 | Heart races → oxygen drops |
| HR ↔ RR | r = +0.84 | Fast heart → fast breathing |
| SpO₂ ↔ RR | r = -0.84 | Low oxygen → compensatory breathing |
| Temp ↔ HR | r = +0.74 | Fever → elevated heart rate |

**Class overlap (realistic):**
```
HR ranges:  High [83-152] | Medium [70-122] | Low [61-99]
                           ↑ overlap zones ↑
```

---

### Step 2: `train_model.py` — Dual-Model Training

Trains **two models** on the CTGAN-generated data:

| Model | Task | Algorithm | F1 Score | Accuracy |
|-------|------|-----------|----------|----------|
| **Risk Classifier** | Low / Medium / High | Random Forest (best of 5 models) | **0.7453** | 76% |
| **Dept Classifier** | 10 departments | XGBoost | **0.9860** | 99% |

**Risk model details:**
- Trains 5 models: Logistic Regression, Decision Tree, Random Forest, XGBoost, Stacking Ensemble
- Selects best by F1 score
- Cross-validation: F1 = 0.7366 ± 0.01

**Feature engineering (47 features):**
- NEWS2 clinical severity score
- Shock Index (HR / Systolic BP)
- Oxy Stress (RR × (100 − SpO₂))
- Fever-Hypoxia interaction
- Comorbidity burden count
- Multi-label encoded symptoms and conditions

```bash
python train_model.py
# → output/*.pkl (14 model artifacts)
# → output/model_metadata.json
# → output/*.png (confusion matrices, feature importance, SHAP, model comparison)
```

---

### Step 3: `predict.py` — Triage + Explainability + Queue

Uses trained models to triage new patients:

```bash
python predict.py
```

**Sample output:**
```
PT-10001 → Risk: High (90.5%) | Dept: Cardiology      | Priority: 87.1/100
PT-10002 → Risk: Low  (70.5%) | Dept: General Medicine | Priority:  7.0/100
PT-10003 → Risk: Medium (62%) | Dept: Pulmonology      | Priority: 29.9/100
PT-10005 → Risk: High (51.5%) | Dept: Neurology        | Priority: 57.6/100 🚨 ESCALATE
```

**SHAP explainability:**
```
Prediction: High
Risk factors:   NEWS2_Score(+0.755), Temperature(+0.393), Pain_Level(+0.349)
Protective:     Sym_Cough(-0.283), Respiratory_Rate(-0.155), BP_Systolic(-0.117)
```

**REST API** (`app.py` — for website integration):

```bash
python app.py   # starts server on port 5001
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/triage` | Triage a single patient |
| `POST` | `/api/triage/batch` | Triage multiple patients |
| `POST` | `/api/queue/admit` | Triage + add to department queue |
| `GET` | `/api/queue/<dept>` | View a department's queue |
| `POST` | `/api/queue/<dept>/next` | Pop highest-priority patient |
| `GET` | `/api/queue/summary` | All department queues |
| `POST` | `/api/explain` | SHAP explanation for a patient |
| `POST` | `/api/similar` | Find similar past patients (KNN) |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/metadata` | Model info + departments list |

**Example request:**

```bash
curl -X POST http://localhost:5001/api/triage \
  -H "Content-Type: application/json" \
  -d '{
    "Patient_ID": "PT-001", "Age": 72, "Gender": "Male",
    "Symptoms": "Chest Pain, Shortness of Breath",
    "Blood_Pressure": "185/110", "Heart_Rate": 130,
    "Temperature": 103.8, "SpO2": 88,
    "Respiratory_Rate": 32, "Consciousness_Level": "Verbal",
    "Pain_Level": 9, "Pre_Existing_Conditions": "Heart Disease"
  }'
```

**Response:**
```json
{
  "risk_level": "High",
  "department": "Cardiology",
  "priority_score": 87.1,
  "confidence": 0.91,
  "news2_score": 13.0,
  "needs_escalation": false,
  "risk_probabilities": {"High": 0.91, "Medium": 0.09, "Low": 0.0},
  "dept_probabilities": {"Cardiology": 0.97, "Emergency": 0.03, ...}
}
```

---

## Project Structure

```
├── generate_dataset.py    # CTGAN synthetic data + noise injection
├── train_model.py         # Dual-model training (Risk + Dept)
├── predict.py             # Prediction engine + SHAP + priority queue
├── app.py                 # Flask REST API (website integration)
├── data/
│   └── patient_triage_dataset.csv
└── output/
    ├── best_risk_model.pkl
    ├── dept_model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── model_metadata.json
    └── *.png              # Charts
```

---

## Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn sdv

# 2. Run full pipeline
python generate_dataset.py   # Generate 5000 patients via CTGAN
python train_model.py        # Train Risk + Dept models
python predict.py            # Run triage demo

# 3. Start API server (for website integration)
python app.py                # http://localhost:5001
```

---

## Input Schema

| Field | Type | Example |
|-------|------|---------|
| Patient_ID | string | PT-001 |
| Age | int | 72 |
| Gender | Male / Female | Male |
| Symptoms | comma-separated | Fever, Chest Pain |
| Blood_Pressure | string | 185/110 |
| Heart_Rate | float | 130 |
| Temperature | float (°F) | 103.8 |
| SpO2 | float (%) | 88 |
| Respiratory_Rate | float | 32 |
| Consciousness_Level | Alert/Verbal/Pain/Unresponsive | Verbal |
| Pain_Level | 0-10 | 9 |
| Pre_Existing_Conditions | comma-separated | Diabetes, Heart Disease |

## Output

| Field | Example |
|-------|---------|
| `risk_level` | High |
| `department` | Cardiology |
| `priority_score` | 87.1 / 100 |
| `confidence` | 90.5% |
| `news2_score` | 13 |
| `needs_escalation` | true (when confidence < 60%) |
| `shap_factors` | top risk/protective features |

## Departments

Cardiology · Emergency · General Medicine · Neurology · Pulmonology · Gastroenterology · Pediatrics · Nephrology · Oncology · Orthopedics