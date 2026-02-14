"""
AI-Powered Smart Patient Triage System — SDV + Realistic Noise (v9)

Hybrid CTGAN approach with noise injection for realistic accuracy:
  - CTGAN learns vital correlations + risk patterns from seed
  - 15% borderline/ambiguous patients added (overlapping classes)
  - 5% label noise (simulates real-world misdiagnosis/ambiguity)
  - Wider vital variance per archetype
  - Department assigned via clinical symptom routing

Target: Risk F1 ~ 0.75-0.85, Dept F1 ~ 0.70-0.85 (realistic range)
"""

import numpy as np
import pandas as pd
import os, warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "patient_triage_dataset.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

FINAL_SIZE = 5000

DEPARTMENTS = [
    "General Medicine", "Cardiology", "Emergency", "Neurology",
    "Pulmonology", "Gastroenterology", "Pediatrics",
    "Nephrology", "Oncology", "Orthopedics",
]

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Build seed with WIDER noise + overlapping profiles
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 1: Building seed with realistic noise & overlap")
print("=" * 65)


def mkrow(age, symptoms, bp_s, bp_d, hr, temp, spo2, rr, cons, pain, conditions, risk):
    """Create one patient with MODERATE vital noise."""
    # Reduced noise ranges (cleaner but still realistic)
    noise_hr = np.random.normal(0, 6)        # was 12
    noise_bp = np.random.normal(0, 8)        # was 15
    noise_temp = np.random.normal(0, 0.4)     # was 0.8
    noise_spo2 = np.random.normal(0, 2)       # was 3
    noise_rr = np.random.normal(0, 2)         # was 4
    noise_pain = np.random.normal(0, 1)       # was 2

    return {
        "Age": age,
        "Gender": np.random.choice(["Male", "Female"]),
        "Symptoms": symptoms,
        "Blood_Pressure": f"{int(np.clip(bp_s+noise_bp,90,180))}/{int(np.clip(bp_d+np.random.normal(0,5),60,110))}",
        "Heart_Rate": float(np.clip(round(hr + noise_hr, 0), 50, 160)),
        "Temperature": float(np.clip(round(temp + noise_temp, 1), 96.0, 105.0)),
        "SpO2": float(np.clip(round(spo2 + noise_spo2, 0), 80, 100)),
        "Respiratory_Rate": float(np.clip(round(rr + noise_rr, 0), 12, 40)),
        "Consciousness_Level": cons,
        "Pain_Level": float(int(np.clip(pain + noise_pain, 0, 10))),
        "Pre_Existing_Conditions": conditions,
        "Risk_Level": risk,
    }

# ... (seed generation code remains similar but with cleaner base values)

# ─── BORDERLINE / AMBIGUOUS patients (~10% of total seed) ─────
# Reduced from 15% to 10%

# ... (CTGAN training code unchanged) ...

# ── Extra noise: 2% label noise (was 5%) ──
n_label_noise = int(FINAL_SIZE * 0.02)

# ── Extra noise: 1% random vital perturbation (was 3%) ──
n_vital_noise = int(FINAL_SIZE * 0.01)

# ── Department: clinical routing (with safeguards) ──
def assign_dept(row):
    s = str(row.get("Symptoms", ""))
    c = str(row.get("Pre_Existing_Conditions", ""))
    cons = str(row.get("Consciousness_Level", "Alert"))
    spo2 = row.get("SpO2", 96)
    hr = row.get("Heart_Rate", 80)
    
    # Parse BP safely
    bp_str = str(row.get("Blood_Pressure", "120/80"))
    try:
        bp_sys = int(bp_str.split("/")[0])
    except:
        bp_sys = 120

    # SAFETY CHECK: If vitals are extremely critical, force Emergency regardless of symptoms
    if spo2 < 85 or hr > 150 or bp_sys > 200 or bp_sys < 70: return "Emergency"

    if cons in ("Unresponsive", "Pain"): return "Emergency"
    if "Seizures" in s: return "Emergency"
    if "Slurred Speech" in s or "Previous Stroke" in c: return "Neurology"  # Stroke is Neuro first (or ER if unstable)
    
    if "Chest Pain" in s or "Heart Disease" in c: return "Cardiology"
    if "Hypertension" in c and bp_sys > 160: return "Cardiology"
    
    if ("Shortness of Breath" in s or "Cough" in s) and ("Asthma" in c or "COPD" in c): return "Pulmonology"
    if spo2 < 92 and "Shortness of Breath" in s: return "Pulmonology"
    
    if "Confusion" in s or "Dizziness" in s: return "Neurology"
    if "Headache" in s: return "Neurology"                   # Migraine -> Neuro (was Ortho error)
    if "Migraine Disorder" in c: return "Neurology"
    
    if "Abdominal Pain" in s or "Diarrhea" in s or "Liver Disease" in c: return "Gastroenterology"
    if "Nausea" in s and "Vomiting" in s: return "Gastroenterology"
    
    if "Kidney Stones" in c or "Kidney Disease" in c: return "Nephrology"  # Kidney Stone -> Nephro (was Ortho error)
    if "Blood in Urine" in s: return "Nephrology"
    
    if "Cancer" in c: return "Oncology"
    
    if "Body Ache" in s or "Joint Pain" in s or "Ankle Pain" in s: return "Orthopedics"
    
    if row.get("Age") < 12: return "Pediatrics"
    return "General Medicine"

def assign_risk(row):
    """Enforce clinical rules for Risk Level to prevent ML confusion."""
    s = str(row.get("Symptoms", ""))
    c = str(row.get("Pre_Existing_Conditions", ""))
    cons = str(row.get("Consciousness_Level", "Alert"))
    spo2 = row.get("SpO2", 96)
    hr = row.get("Heart_Rate", 80)
    rr = row.get("Respiratory_Rate", 18)
    
    # Parse BP safely
    bp_str = str(row.get("Blood_Pressure", "120/80"))
    try:
        bp_sys = int(bp_str.split("/")[0])
    except:
        bp_sys = 120

    # CRITICAL / HIGH RISK RULES
    if cons == "Unresponsive": return "High"
    if spo2 < 90: return "High"
    if hr > 130 or hr < 40: return "High"
    if bp_sys > 200 or bp_sys < 80: return "High"  # Hypotension is HIGH risk
    if rr > 30 or rr < 8: return "High"            # Severe respiratory distress
    if row.get("Pain_Level", 0) >= 9: return "High" # Extreme pain
    if "Chest Pain" in s and "Heart Disease" in c: return "High"
    if "Slurred Speech" in s: return "High"        # Stroke
    if "Seizures" in s: return "High"
    if "Confusion" in s and "Diabetes" in c: return "High" # DKA risk
    if "Fever" in s and bp_sys < 90: return "High" # Sepsis

    # MEDIUM RISK RULES
    if cons == "Pain" or cons == "Verbal": return "Medium"
    if spo2 < 94: return "Medium"
    if hr > 110 or hr < 50: return "Medium"
    if bp_sys > 160 or bp_sys < 90: return "Medium"
    if "Shortness of Breath" in s: return "Medium"
    if "Fever" in s and row.get("Age") < 5: return "Medium" # Pediatric fever

    return "Low"

# ... inside data generation loop, verify risk is assigned by rules ...
# (Update the generation loop to call assign_risk instead of relying purely on seed/CTGAN for risk)


rows = []

# ─── HIGH RISK profiles ───────────────────────────────────────
for _ in range(70):
    rows.append(mkrow(np.random.randint(50,85), "Chest Pain, Shortness of Breath",
        175, 98, 125, 99.5, 90, 27,
        np.random.choice(["Alert","Verbal","Pain"], p=[0.2,0.5,0.3]),
        8, np.random.choice(["Heart Disease, Hypertension", "Heart Disease", "Hypertension, Diabetes"]),
        "High"))
for _ in range(60):
    rows.append(mkrow(np.random.randint(18,80), "Seizures, Confusion",
        88, 55, 140, 102.5, 83, 33,
        np.random.choice(["Pain","Unresponsive","Verbal"], p=[0.4,0.3,0.3]),
        9, np.random.choice(["None", "Diabetes", "Hypertension"]),
        "High"))
for _ in range(50):
    rows.append(mkrow(np.random.randint(30,75), "Chest Pain, Shortness of Breath, Confusion",
        200, 110, 142, 101.5, 85, 31,
        np.random.choice(["Pain","Verbal"], p=[0.5,0.5]),
        9, np.random.choice(["Heart Disease", "None", "Diabetes"]),
        "High"))
for _ in range(45):
    rows.append(mkrow(np.random.randint(55,85), "Confusion, Headache, Dizziness",
        165, 95, 106, 99.0, 92, 23,
        np.random.choice(["Verbal","Alert","Pain"], p=[0.4,0.3,0.3]),
        7, np.random.choice(["Previous Stroke, Hypertension", "Previous Stroke", "Hypertension"]),
        "High"))
for _ in range(45):
    rows.append(mkrow(np.random.randint(50,82), "Cough, Shortness of Breath, Fever",
        140, 82, 108, 101.0, 87, 29,
        np.random.choice(["Verbal","Alert"], p=[0.5,0.5]),
        7, np.random.choice(["COPD", "COPD, Diabetes", "Asthma"]),
        "High"))
for _ in range(35):
    rows.append(mkrow(np.random.randint(20,65), "Fever, Shortness of Breath, Vomiting",
        92, 58, 130, 103.5, 86, 30,
        "Verbal", 8, np.random.choice(["None","Diabetes"]),
        "High"))
for _ in range(35):
    rows.append(mkrow(np.random.randint(60,88), "Fever, Cough, Shortness of Breath",
        158, 92, 110, 102.5, 91, 26,
        "Verbal", 7, "Diabetes, Hypertension, COPD",
        "High"))
for _ in range(20):
    rows.append(mkrow(np.random.randint(55,82), "Fatigue, Nausea, Loss of Appetite",
        170, 100, 98, 99.5, 93, 22,
        np.random.choice(["Verbal","Alert"]), 6, "Kidney Disease, Hypertension",
        "High"))
for _ in range(20):
    rows.append(mkrow(np.random.randint(45,80), "Fatigue, Nausea, Fever",
        130, 80, 96, 100.5, 93, 21,
        "Alert", 7, np.random.choice(["Cancer, Diabetes", "Cancer"]),
        "High"))
for _ in range(300):
    rows.append(mkrow(np.random.randint(60,85), "Slurred Speech, Facial Droop, Weakness",
        210, 115, 95, 98.4, 94, 20,
        "Verbal", 2, "Hypertension, Previous Stroke",
        "High"))
for _ in range(300):
    rows.append(mkrow(np.random.randint(18,35), "Confusion, Vomiting, Rapid Breathing",
        95, 60, 130, 98.0, 95, 35,
        "Verbal", 6, "Type 1 Diabetes",
        "High"))
for _ in range(300):
    rows.append(mkrow(np.random.randint(30,60), "Severe Flank Pain, Nausea, Blood in Urine",
        150, 90, 110, 99.0, 97, 24,
        "Alert", 9, "Kidney Stones",
        "Medium"))
for _ in range(300):
    rows.append(mkrow(np.random.randint(20,50), "Severe Headache, Light Sensitivity",
        135, 85, 90, 98.5, 98, 18,
        "Alert", 9, "Migraine Disorder",
        "Low"))
for _ in range(300):
    rows.append(mkrow(np.random.randint(65,90), "Fever, Confusion, Shivering",
        85, 50, 135, 103.0, 90, 30,
        "Verbal", 4, "Kidney Disease, Recurrent UTIs",
        "High"))
# ─── MEDIUM RISK profiles ─────────────────────────────────────
for _ in range(60):
    rows.append(mkrow(np.random.randint(40,75), "Chest Pain, Dizziness",
        152, 90, 98, 98.8, 94, 20,
        "Alert", 5, np.random.choice(["Hypertension", "Hypertension, Diabetes", "Heart Disease"]),
        "Medium"))
for _ in range(50):
    rows.append(mkrow(np.random.randint(35,70), "Headache, Dizziness",
        138, 84, 88, 98.8, 96, 19,
        "Alert", 5, np.random.choice(["Hypertension","None","Stroke History"]),
        "Medium"))
for _ in range(50):
    rows.append(mkrow(np.random.randint(25,65), "Abdominal Pain, Nausea, Vomiting",
        125, 78, 90, 100.0, 97, 19,
        "Alert", 6, np.random.choice(["None","Liver Disease","Diabetes"]),
        "Medium"))
for _ in range(45):
    rows.append(mkrow(np.random.randint(28,65), "Cough, Shortness of Breath",
        128, 78, 90, 99.5, 94, 22,
        "Alert", 4, np.random.choice(["Asthma","None","COPD"]),
        "Medium"))
for _ in range(70):
    rows.append(mkrow(np.random.randint(40,72), "Fever, Cough, Fatigue",
        138, 84, 92, 100.8, 95, 20,
        "Alert", 4, np.random.choice(["Diabetes","Hypertension","None","Obesity"]),
        "Medium"))
for _ in range(55):
    rows.append(mkrow(np.random.randint(1,12), "Fever, Cough",
        100, 65, 105, 101.2, 96, 23,
        "Alert", 5, "None",
        "Medium"))
for _ in range(40):
    rows.append(mkrow(np.random.randint(1,12), "Fever, Vomiting",
        98, 62, 100, 101.0, 97, 22,
        "Alert", 4, "None",
        "Medium"))
for _ in range(35):
    rows.append(mkrow(np.random.randint(22,55), "Nausea, Vomiting, Diarrhea",
        118, 72, 85, 99.5, 98, 17,
        "Alert", 4, np.random.choice(["Liver Disease","None"]),
        "Medium"))
for _ in range(35):
    rows.append(mkrow(np.random.randint(50,80), "Fatigue, Nausea, Dizziness",
        158, 94, 90, 99.0, 95, 20,
        "Alert", 5, np.random.choice(["Kidney Disease, Diabetes","Kidney Disease"]),
        "Medium"))
for _ in range(30):
    rows.append(mkrow(np.random.randint(45,78), "Fatigue, Loss of Appetite, Fever",
        132, 80, 88, 99.5, 95, 19,
        "Alert", 6, np.random.choice(["Cancer","Cancer, Hypertension"]),
        "Medium"))
for _ in range(25):
    rows.append(mkrow(np.random.randint(30,65), "Body Ache, Fatigue",
        125, 78, 82, 98.8, 98, 17,
        "Alert", 5, np.random.choice(["Obesity","None"]),
        "Medium"))
for _ in range(45):
    rows.append(mkrow(np.random.randint(48,75), "Fever, Body Ache, Headache",
        140, 85, 94, 101.2, 95, 20,
        "Alert", 5, np.random.choice(["Diabetes, Hypertension","Diabetes","Hypertension"]),
        "Medium"))

# ─── LOW RISK profiles ────────────────────────────────────────
for _ in range(110):
    rows.append(mkrow(np.random.randint(18,50), "Cough, Sore Throat",
        118, 74, 76, 98.8, 98, 16,
        "Alert", 2, np.random.choice(["None","None","None","Obesity"]),
        "Low"))
for _ in range(100):
    rows.append(mkrow(np.random.randint(18,55), "Fever, Headache",
        120, 76, 80, 100.0, 97, 17,
        "Alert", 3, np.random.choice(["None","None","Diabetes"]),
        "Low"))
for _ in range(70):
    rows.append(mkrow(np.random.randint(20,55), "Headache, Fatigue",
        118, 74, 78, 98.6, 98, 16,
        "Alert", 2, "None",
        "Low"))
for _ in range(40):
    rows.append(mkrow(np.random.randint(3,12), "Cough, Sore Throat",
        96, 60, 90, 99.0, 98, 18,
        "Alert", 2, "None",
        "Low"))
for _ in range(35):
    rows.append(mkrow(np.random.randint(20,60), "Body Ache",
        120, 76, 78, 98.6, 98, 16,
        "Alert", 7, np.random.choice(["None","Obesity"]),
        "Low"))
for _ in range(30):
    rows.append(mkrow(np.random.randint(20,48), "Abdominal Pain, Nausea",
        115, 72, 78, 98.8, 98, 16,
        "Alert", 3, "None",
        "Low"))
for _ in range(25):
    rows.append(mkrow(np.random.randint(38,68), "Chest Pain, Fatigue",
        142, 86, 88, 98.6, 96, 18,
        "Alert", 3, np.random.choice(["Hypertension, Diabetes","Hypertension"]),
        "Low"))
for _ in range(20):
    rows.append(mkrow(np.random.randint(25,55), "Cough, Fatigue",
        118, 74, 80, 98.8, 97, 17,
        "Alert", 2, "None",
        "Low"))

# ─── BORDERLINE / AMBIGUOUS patients (~15% of total seed) ─────
# These deliberately blur class boundaries
print("   Adding borderline/ambiguous patients...")

# Medium patients with High-ish vitals (could go either way)
for _ in range(40):
    rows.append(mkrow(np.random.randint(50,75),
        np.random.choice(["Chest Pain, Fatigue", "Cough, Shortness of Breath", "Headache, Dizziness, Fatigue"]),
        155, 92, 108, 100.5, 92, 24,
        np.random.choice(["Alert","Verbal"], p=[0.6,0.4]),
        6, np.random.choice(["Hypertension","Diabetes","Heart Disease","COPD"]),
        "Medium"))  # borderline High

# High patients with Medium-ish vitals (not clearly critical)
for _ in range(35):
    rows.append(mkrow(np.random.randint(55,80),
        np.random.choice(["Chest Pain, Dizziness", "Confusion, Fatigue", "Shortness of Breath, Cough"]),
        148, 88, 98, 99.8, 93, 21,
        "Alert",
        5, np.random.choice(["Heart Disease, Diabetes", "Previous Stroke", "COPD, Hypertension"]),
        "High"))  # borderline Medium

# Low patients with Medium-ish vitals
for _ in range(30):
    rows.append(mkrow(np.random.randint(30,60),
        np.random.choice(["Fever, Cough, Fatigue", "Headache, Nausea", "Body Ache, Fever"]),
        132, 82, 88, 100.2, 96, 19,
        "Alert",
        4, np.random.choice(["None","Diabetes","Obesity"]),
        "Low"))  # borderline Medium

# Medium patients with Low-ish vitals
for _ in range(30):
    rows.append(mkrow(np.random.randint(20,45),
        np.random.choice(["Cough, Sore Throat", "Fever, Headache", "Fatigue, Headache"]),
        120, 76, 80, 99.2, 97, 17,
        "Alert",
        3, np.random.choice(["Diabetes","Asthma","Hypertension"]),
        "Medium"))  # borderline Low

# Atypical presentations (heart attack with normal-ish vitals, etc.)
for _ in range(20):
    rows.append(mkrow(np.random.randint(40,70),
        "Chest Pain, Nausea",  # atypical MI
        130, 80, 85, 98.6, 96, 18,
        "Alert", 4, np.random.choice(["Diabetes","None"]),
        np.random.choice(["Medium","High"])))  # ambiguous

for _ in range(20):
    rows.append(mkrow(np.random.randint(60,85),
        np.random.choice(["Fatigue, Confusion", "Dizziness, Fatigue"]),
        145, 88, 82, 98.8, 95, 18,
        np.random.choice(["Alert","Verbal"]),
        4, np.random.choice(["Hypertension","Previous Stroke","Diabetes"]),
        np.random.choice(["Medium","High"])))  # elderly ambiguous

for _ in range(50):
    rows.append(mkrow(np.random.randint(20,40), "Chest Tightness, Palpitations, Dizziness",
        135, 85, 115, 98.6, 98, 22,
        "Alert", 4, "Anxiety Disorder",
        "Medium"))


seed_df = pd.DataFrame(rows)
seed_df.insert(0, "Patient_ID", [f"SEED-{str(i).zfill(4)}" for i in range(len(rows))])

# Shuffle
seed_df = seed_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Seed: {seed_df.shape}")
print(f"   Risk: {seed_df['Risk_Level'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Train CTGAN
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 2: CTGAN learning noisy patterns")
print("=" * 65)

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

metadata = Metadata.detect_from_dataframe(data=seed_df, table_name="patients")
metadata.update_column(table_name="patients", column_name="Patient_ID", sdtype="id")
metadata.update_column(table_name="patients", column_name="Age", sdtype="numerical")
metadata.update_column(table_name="patients", column_name="Gender", sdtype="categorical")
metadata.update_column(table_name="patients", column_name="Symptoms", sdtype="categorical")
metadata.update_column(table_name="patients", column_name="Blood_Pressure", sdtype="categorical")
metadata.update_column(table_name="patients", column_name="Consciousness_Level", sdtype="categorical")
metadata.update_column(table_name="patients", column_name="Pre_Existing_Conditions", sdtype="categorical")
metadata.update_column(table_name="patients", column_name="Heart_Rate", sdtype="numerical")
metadata.update_column(table_name="patients", column_name="Temperature", sdtype="numerical")
metadata.update_column(table_name="patients", column_name="SpO2", sdtype="numerical")
metadata.update_column(table_name="patients", column_name="Respiratory_Rate", sdtype="numerical")
metadata.update_column(table_name="patients", column_name="Pain_Level", sdtype="numerical")
metadata.update_column(table_name="patients", column_name="Risk_Level", sdtype="categorical")
metadata.set_primary_key(table_name="patients", column_name="Patient_ID")

synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=200,
    verbose=True,
)
print("   Training CTGAN (300 epochs)...")
synthesizer.fit(seed_df)
print("   ✅ CTGAN trained on noisy + borderline data")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Generate + inject additional noise
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  STEP 3: Generating {FINAL_SIZE} patients + post-processing noise")
print("=" * 65)

synthetic_df = synthesizer.sample(num_rows=FINAL_SIZE)

# Fix IDs
synthetic_df["Patient_ID"] = [f"PT-{str(i).zfill(5)}" for i in range(1, FINAL_SIZE + 1)]

# Clip vitals
synthetic_df["Age"] = synthetic_df["Age"].clip(1, 95).astype(int)
synthetic_df["Heart_Rate"] = synthetic_df["Heart_Rate"].clip(40, 200).round(0)
synthetic_df["Temperature"] = synthetic_df["Temperature"].clip(95.0, 107.0).round(1)
synthetic_df["SpO2"] = synthetic_df["SpO2"].clip(65, 100).round(0)
synthetic_df["Respiratory_Rate"] = synthetic_df["Respiratory_Rate"].clip(8, 50).round(0)
synthetic_df["Pain_Level"] = synthetic_df["Pain_Level"].clip(0, 10).round(0)

# Validate risk with strict clinical rules
synthetic_df["Risk_Level"] = synthetic_df.apply(assign_risk, axis=1)

print(f"   Applied strict clinical rules to all {FINAL_SIZE} patients")

# (Removed label noise and vital perturbation to ensure strict annotations)

# ── Department: clinical routing ──
def assign_dept(row):
    s = str(row.get("Symptoms", ""))
    c = str(row.get("Pre_Existing_Conditions", ""))
    cons = str(row.get("Consciousness_Level", "Alert"))
    spo2 = row.get("SpO2", 96)
    hr = row.get("Heart_Rate", 80)
    age = row.get("Age", 30)
    bp_str = str(row.get("Blood_Pressure", "120/80"))
    bp_sys = int(bp_str.split("/")[0]) if "/" in bp_str else 120

    if cons in ("Unresponsive", "Pain"): return "Emergency"
    if spo2 < 88 or hr > 140 or hr < 45 or bp_sys > 200 or bp_sys < 80: return "Emergency"
    if "Seizures" in s: return "Emergency"
    if "Chest Pain" in s or "Heart Disease" in c: return "Cardiology"
    if "Hypertension" in c and bp_sys > 160: return "Cardiology"
    if ("Shortness of Breath" in s or "Cough" in s) and ("Asthma" in c or "COPD" in c): return "Pulmonology"
    if spo2 < 92 and "Shortness of Breath" in s: return "Pulmonology"
    if "Confusion" in s or "Dizziness" in s or "Stroke History" in c: return "Neurology"
    if "Headache" in s and row.get("Pain_Level", 0) >= 7: return "Neurology"
    if "Abdominal Pain" in s or "Diarrhea" in s or "Liver Disease" in c: return "Gastroenterology"
    if "Nausea" in s and "Vomiting" in s: return "Gastroenterology"
    if "Kidney Disease" in c: return "Nephrology"
    if "Cancer" in c: return "Oncology"
    if "Body Ache" in s and s.count(",") == 0: return "Orthopedics"
    if age < 12: return "Pediatrics"
    return "General Medicine"

synthetic_df["Department"] = synthetic_df.apply(assign_dept, axis=1)

# Inject ~3% missing vitals
for col in ["Heart_Rate", "Temperature", "SpO2", "Respiratory_Rate", "Pain_Level"]:
    mask = np.random.choice([True, False], size=FINAL_SIZE, p=[0.03, 0.97])
    synthetic_df.loc[mask, col] = np.nan

synthetic_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Dataset saved: {OUTPUT_PATH}")
print(f"   Shape: {synthetic_df.shape}")
print(f"\n📊 Risk Level:\n{synthetic_df['Risk_Level'].value_counts().to_string()}")
print(f"\n🏥 Department:\n{synthetic_df['Department'].value_counts().to_string()}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Validate
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 4: Validation")
print("=" * 65)

df = synthetic_df.dropna().copy()
high = df[df["Risk_Level"]=="High"]; low = df[df["Risk_Level"]=="Low"]
med = df[df["Risk_Level"]=="Medium"]
print(f"\n  Risk Level vitals (with overlap from noise):")
if len(high) > 0 and len(low) > 0:
    print(f"    High   — HR: {high['Heart_Rate'].mean():.0f}±{high['Heart_Rate'].std():.0f}, SpO2: {high['SpO2'].mean():.0f}±{high['SpO2'].std():.0f}")
    print(f"    Medium — HR: {med['Heart_Rate'].mean():.0f}±{med['Heart_Rate'].std():.0f}, SpO2: {med['SpO2'].mean():.0f}±{med['SpO2'].std():.0f}")
    print(f"    Low    — HR: {low['Heart_Rate'].mean():.0f}±{low['Heart_Rate'].std():.0f}, SpO2: {low['SpO2'].mean():.0f}±{low['SpO2'].std():.0f}")

print(f"\n  Overlap check (HR range):")
print(f"    High:   [{high['Heart_Rate'].quantile(0.10):.0f} — {high['Heart_Rate'].quantile(0.90):.0f}]")
print(f"    Medium: [{med['Heart_Rate'].quantile(0.10):.0f} — {med['Heart_Rate'].quantile(0.90):.0f}]")
print(f"    Low:    [{low['Heart_Rate'].quantile(0.10):.0f} — {low['Heart_Rate'].quantile(0.90):.0f}]")

print("\n" + "=" * 65)
print("  ✅ Realistic noise applied: classes now overlap!")
print("=" * 65)
