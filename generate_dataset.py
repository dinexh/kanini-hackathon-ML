"""
AI-Powered Smart Patient Triage System — Dataset Generator (v3 Advanced)

Generates 5,000 synthetic patients with realistic clinical correlations.
Symptoms and conditions are internally tracked for rule-based Risk_Level
assignment, but stored as human-readable comma-separated strings.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N_SAMPLES = 5000

# ── Demographics ──────────────────────────────────────────────────
patient_ids = [f"PT-{str(i).zfill(5)}" for i in range(1, N_SAMPLES + 1)]
ages = np.random.randint(1, 95, size=N_SAMPLES)
genders = np.random.choice(["Male", "Female"], size=N_SAMPLES)

# ── Symptoms ──────────────────────────────────────────────────────
ALL_SYMPTOMS = [
    "Fever", "Cough", "Chest Pain", "Shortness of Breath",
    "Fatigue", "Headache", "Nausea", "Vomiting", "Dizziness",
    "Sore Throat", "Body Ache", "Abdominal Pain", "Diarrhea",
    "Loss of Appetite", "Confusion", "Seizures",
]

def random_symptoms():
    n = np.random.randint(0, 6)
    if n == 0:
        return "None"
    return ", ".join(np.random.choice(ALL_SYMPTOMS, size=n, replace=False))

symptoms = [random_symptoms() for _ in range(N_SAMPLES)]

# ── Vitals ────────────────────────────────────────────────────────
bp_systolic = np.random.normal(120, 20, N_SAMPLES).clip(70, 220).astype(int)
bp_diastolic = np.random.normal(80, 12, N_SAMPLES).clip(40, 140).astype(int)
blood_pressure = [f"{s}/{d}" for s, d in zip(bp_systolic, bp_diastolic)]

heart_rate = np.random.normal(80, 15, N_SAMPLES).clip(40, 180).astype(int)
temperature = np.round(np.random.normal(98.6, 1.2, N_SAMPLES).clip(95.0, 106.0), 1)
spo2 = np.random.normal(96, 3, N_SAMPLES).clip(70, 100).astype(int)
respiratory_rate = np.random.normal(18, 4, N_SAMPLES).clip(8, 45).astype(int)

consciousness_levels = np.random.choice(
    ["Alert", "Verbal", "Pain", "Unresponsive"],
    size=N_SAMPLES, p=[0.70, 0.15, 0.10, 0.05]
)
pain_level = np.random.randint(0, 11, size=N_SAMPLES)

# ── Pre-existing Conditions ──────────────────────────────────────
ALL_CONDITIONS = [
    "Diabetes", "Hypertension", "Asthma", "Heart Disease",
    "Obesity", "Kidney Disease", "COPD", "Cancer",
    "Stroke History", "Liver Disease",
]

def random_conditions():
    n = np.random.choice([0, 1, 2, 3], p=[0.40, 0.30, 0.20, 0.10])
    if n == 0:
        return "None"
    return ", ".join(np.random.choice(ALL_CONDITIONS, size=n, replace=False))

pre_existing = [random_conditions() for _ in range(N_SAMPLES)]

# ── Risk Level Assignment ────────────────────────────────────────
def assign_risk(i):
    score = 0
    # Age
    if ages[i] > 65: score += 3
    elif ages[i] > 50: score += 2
    elif ages[i] < 5: score += 2
    # SpO2
    if spo2[i] < 90: score += 4
    elif spo2[i] < 94: score += 2
    # Heart rate
    if heart_rate[i] > 120 or heart_rate[i] < 50: score += 3
    elif heart_rate[i] > 100 or heart_rate[i] < 60: score += 1
    # Temperature
    if temperature[i] > 103.0: score += 3
    elif temperature[i] > 100.4: score += 1
    # BP
    if bp_systolic[i] > 180 or bp_systolic[i] < 90: score += 3
    elif bp_systolic[i] > 140 or bp_systolic[i] < 100: score += 1
    # Respiratory rate
    if respiratory_rate[i] > 30 or respiratory_rate[i] < 10: score += 3
    elif respiratory_rate[i] > 24: score += 1
    # Consciousness
    cons_map = {"Alert": 0, "Verbal": 2, "Pain": 4, "Unresponsive": 6}
    score += cons_map.get(consciousness_levels[i], 0)
    # Pain
    if pain_level[i] >= 8: score += 2
    elif pain_level[i] >= 5: score += 1
    # Critical symptoms
    s = symptoms[i]
    if "Chest Pain" in s: score += 3
    if "Shortness of Breath" in s: score += 3
    if "Seizures" in s: score += 4
    if "Confusion" in s: score += 2
    if "Vomiting" in s: score += 1
    # Symptom count
    if s != "None": score += len(s.split(", ")) * 0.5
    # Conditions
    c = pre_existing[i]
    if c != "None": score += len(c.split(", ")) * 1.5

    if score >= 10: return "High"
    elif score >= 5: return "Medium"
    else: return "Low"

risk_levels = [assign_risk(i) for i in range(N_SAMPLES)]

# ── Build DataFrame ──────────────────────────────────────────────
df = pd.DataFrame({
    "Patient_ID": patient_ids,
    "Age": ages, "Gender": genders,
    "Symptoms": symptoms,
    "Blood_Pressure": blood_pressure,
    "Heart_Rate": heart_rate, "Temperature": temperature,
    "SpO2": spo2, "Respiratory_Rate": respiratory_rate,
    "Consciousness_Level": consciousness_levels,
    "Pain_Level": pain_level,
    "Pre_Existing_Conditions": pre_existing,
    "Risk_Level": risk_levels,
})

# Inject ~2% missing values
for col in ["Heart_Rate", "Temperature", "SpO2", "Respiratory_Rate", "Pain_Level"]:
    mask = np.random.choice([True, False], size=N_SAMPLES, p=[0.02, 0.98])
    df.loc[mask, col] = np.nan

# Save
output_path = os.path.join(os.path.dirname(__file__), "data", "patient_triage_dataset.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Dataset generated: {output_path}")
print(f"   Shape: {df.shape}")
print(f"\n📊 Risk Level distribution:\n{df['Risk_Level'].value_counts().to_string()}")
print(f"\n📋 Sample rows:\n{df.head(3).to_string()}")
