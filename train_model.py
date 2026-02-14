"""
AI-Powered Smart Patient Triage System — Advanced Training Pipeline (v3)

WHAT MAKES THIS UNIQUE (vs. basic classifiers):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. NEWS2-inspired Clinical Severity Score (domain-knowledge feat.) │
  │ 2. Medically meaningful feature interactions (Shock Index, etc.)   │
  │ 3. Stacking Ensemble (LR + RF + XGBoost → Meta-Learner)           │
  │ 4. Confidence-based triage escalation (uncertain → doctor review)  │
  │ 5. Patient similarity scoring (find similar past cases)            │
  │ 6. SHAP explainability per-patient                                 │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "patient_triage_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Canonical value lists
ALL_SYMPTOMS = sorted([
    "Abdominal Pain", "Body Ache", "Chest Pain", "Confusion", "Cough",
    "Diarrhea", "Dizziness", "Fatigue", "Fever", "Headache",
    "Loss of Appetite", "Nausea", "Seizures", "Shortness of Breath",
    "Sore Throat", "Vomiting",
])
ALL_CONDITIONS = sorted([
    "Asthma", "COPD", "Cancer", "Diabetes", "Heart Disease",
    "Hypertension", "Kidney Disease", "Liver Disease", "Obesity",
    "Stroke History",
])
CONSCIOUSNESS_ORDER = ["Unresponsive", "Pain", "Verbal", "Alert"]
CONS_MAP = {level: i for i, level in enumerate(CONSCIOUSNESS_ORDER)}

# Confidence threshold — below this, flag for doctor review
ESCALATION_THRESHOLD = 0.60


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTIONS (what makes this model unique!)
# ═══════════════════════════════════════════════════════════════════

def compute_news2_score(df):
    """
    NEWS2 (National Early Warning Score 2) — inspired clinical severity index.

    This is a well-established triage scoring system used in UK hospitals.
    We compute a simplified version as a composite feature.
    Each vital sign gets a sub-score (0-3) based on clinical thresholds,
    and they are summed into a single severity index.

    This gives the model DOMAIN KNOWLEDGE that raw features alone can't capture.
    """
    score = pd.Series(0, index=df.index, dtype=float)

    # Respiratory Rate scoring
    rr = df["Respiratory_Rate"]
    score += np.where(rr <= 8, 3, np.where(rr <= 11, 1, np.where(rr <= 20, 0,
             np.where(rr <= 24, 2, 3))))

    # SpO2 scoring (Scale 1)
    spo2 = df["SpO2"]
    score += np.where(spo2 <= 91, 3, np.where(spo2 <= 93, 2,
             np.where(spo2 <= 95, 1, 0)))

    # Heart Rate scoring
    hr = df["Heart_Rate"]
    score += np.where(hr <= 40, 3, np.where(hr <= 50, 1, np.where(hr <= 90, 0,
             np.where(hr <= 110, 1, np.where(hr <= 130, 2, 3)))))

    # Temperature scoring (convert °F to °C for NEWS2 thresholds)
    temp_c = (df["Temperature"] - 32) * 5 / 9
    score += np.where(temp_c <= 35.0, 3, np.where(temp_c <= 36.0, 1,
             np.where(temp_c <= 38.0, 0, np.where(temp_c <= 39.0, 1, 2))))

    # Systolic BP scoring
    sbp = df["BP_Systolic"]
    score += np.where(sbp <= 90, 3, np.where(sbp <= 100, 2, np.where(sbp <= 110, 1,
             np.where(sbp <= 219, 0, 3))))

    # Consciousness scoring (AVPU)
    score += np.where(df["Consciousness_Level"] < 3, 3, 0)  # anything below Alert

    return score


def compute_feature_interactions(df):
    """
    Create medically meaningful compound features that capture
    relationships between vitals that a linear model would miss.
    """
    features = pd.DataFrame(index=df.index)

    # Shock Index = HR / Systolic BP  (>1.0 = shock, critical triage indicator)
    features["Shock_Index"] = df["Heart_Rate"] / df["BP_Systolic"].clip(lower=1)

    # Modified Shock Index = HR / Mean Arterial Pressure
    map_val = df["BP_Diastolic"] + (df["BP_Systolic"] - df["BP_Diastolic"]) / 3
    features["Modified_Shock_Index"] = df["Heart_Rate"] / map_val.clip(lower=1)

    # Oxygenation stress = Respiratory Rate / SpO2 (higher = worse)
    features["Oxy_Stress"] = df["Respiratory_Rate"] / df["SpO2"].clip(lower=1)

    # Fever-Hypoxia interaction (fever + low SpO2 = pneumonia/sepsis signal)
    features["Fever_Hypoxia"] = ((df["Temperature"] > 100.4).astype(int) *
                                  (100 - df["SpO2"]))

    # Age-vulnerability index (very young or very old with abnormal vitals)
    age_risk = np.where(df["Age"] > 65, 2, np.where(df["Age"] < 5, 2,
               np.where(df["Age"] > 50, 1, 0)))
    features["Age_Vulnerability"] = age_risk * (df["Pain_Level"] / 10 + 0.5)

    # Symptom severity score (count of symptoms × consciousness penalty)
    features["Symptom_Severity"] = (df["Symptom_Count"] *
                                     (4 - df["Consciousness_Level"]) / 4)

    # Comorbidity burden (conditions count × age factor)
    features["Comorbidity_Burden"] = (df["Condition_Count"] *
                                       np.where(df["Age"] > 60, 1.5, 1.0))

    # Hemodynamic instability = abs(HR - 75) + abs(SBP - 120) normalized
    features["Hemodynamic_Instability"] = (
        np.abs(df["Heart_Rate"] - 75) / 75 +
        np.abs(df["BP_Systolic"] - 120) / 120
    )

    return features


def parse_multi_value(series, all_values, prefix):
    """Convert comma-separated string column into binary columns."""
    def split_fn(val):
        if pd.isna(val) or val.strip().lower() == "none":
            return []
        return [v.strip() for v in val.split(",")]
    mlb = MultiLabelBinarizer(classes=all_values)
    encoded = mlb.fit_transform(series.apply(split_fn))
    cols = [f"{prefix}_{v.replace(' ', '_')}" for v in all_values]
    return pd.DataFrame(encoded, columns=cols, index=series.index), mlb


# ═══════════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 1: Loading & Advanced Preprocessing")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\n📂 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📊 Risk Level distribution:\n{df['Risk_Level'].value_counts().to_string()}")

# Drop Patient_ID
if "Patient_ID" in df.columns:
    df = df.drop("Patient_ID", axis=1)

# Parse Blood Pressure
bp_split = df["Blood_Pressure"].str.split("/", expand=True)
df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")
df = df.drop("Blood_Pressure", axis=1)

# Binarize Symptoms (and keep count)
symptom_df, symptom_mlb = parse_multi_value(df["Symptoms"], ALL_SYMPTOMS, "Sym")
df["Symptom_Count"] = symptom_df.sum(axis=1)
df = pd.concat([df.drop("Symptoms", axis=1), symptom_df], axis=1)

# Binarize Conditions (and keep count)
cond_df, cond_mlb = parse_multi_value(df["Pre_Existing_Conditions"], ALL_CONDITIONS, "Cond")
df["Condition_Count"] = cond_df.sum(axis=1)
df = pd.concat([df.drop("Pre_Existing_Conditions", axis=1), cond_df], axis=1)

# Encode Consciousness Level (ordinal)
df["Consciousness_Level"] = df["Consciousness_Level"].map(CONS_MAP)
df["Consciousness_Level"] = df["Consciousness_Level"].fillna(df["Consciousness_Level"].median())

# Encode Gender
label_encoders = {}
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"].fillna("Unknown"))
label_encoders["Gender"] = le_gender

# Handle missing values
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "Risk_Level" in numerical_cols:
    numerical_cols.remove("Risk_Level")

missing_before = df.isnull().sum().sum()
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
missing_after = df.isnull().sum().sum()
print(f"\n🔧 Missing values: {missing_before} → {missing_after}")
assert missing_after == 0, f"Still {missing_after} NaN remaining!"

# ── ADVANCED FEATURE ENGINEERING ──────────────────────────────────
print("\n🧠 Engineering advanced clinical features...")

# 1) NEWS2 Clinical Severity Score
df["NEWS2_Score"] = compute_news2_score(df)
print(f"   ✓ NEWS2 Clinical Severity Score (0-18)")

# 2) Feature Interactions
interactions = compute_feature_interactions(df)
for col in interactions.columns:
    df[col] = interactions[col]
print(f"   ✓ {len(interactions.columns)} medical feature interactions")
print(f"     (Shock_Index, Modified_Shock_Index, Oxy_Stress, Fever_Hypoxia,")
print(f"      Age_Vulnerability, Symptom_Severity, Comorbidity_Burden,")
print(f"      Hemodynamic_Instability)")

# Encode target
target_le = LabelEncoder()
df["Risk_Level"] = target_le.fit_transform(df["Risk_Level"])
print(f"   Target classes: {list(target_le.classes_)}")

# Separate features & target
X = df.drop("Risk_Level", axis=1)
y = df["Risk_Level"]
feature_names = X.columns.tolist()

print(f"\n📐 Final feature matrix: {X.shape[0]} × {X.shape[1]} features")

# Normalize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# ═══════════════════════════════════════════════════════════════════
# 2. SPLIT & CROSS-VALIDATE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 2: Splitting Dataset (80/20) + Cross-Validation")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set : {X_train.shape[0]} samples")
print(f"   Testing set  : {X_test.shape[0]} samples")

# ═══════════════════════════════════════════════════════════════════
# 3. TRAIN INDIVIDUAL MODELS + STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 3: Training Models (Individual + Stacking Ensemble)")
print("=" * 65)

# Individual models
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False,
        eval_metric="mlogloss", verbosity=0
    ),
}

results = {}

for name, model in base_models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # 5-fold cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="f1_weighted")

    results[name] = {
        "model": model, "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1, "confusion_matrix": cm,
        "y_pred": y_pred, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
    }
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── STACKING ENSEMBLE ────────────────────────────────────────────
print(f"\n🔹 Training Stacking Ensemble (LR + RF + XGBoost → Meta-Learner)...")

stacking_model = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="mlogloss", verbosity=0
        )),
    ],
    final_estimator=GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=42
    ),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1,
)

stacking_model.fit(X_train, y_train)
y_pred_stack = stacking_model.predict(X_test)

acc_s = accuracy_score(y_test, y_pred_stack)
prec_s = precision_score(y_test, y_pred_stack, average="weighted", zero_division=0)
rec_s = recall_score(y_test, y_pred_stack, average="weighted", zero_division=0)
f1_s = f1_score(y_test, y_pred_stack, average="weighted", zero_division=0)
cm_s = confusion_matrix(y_test, y_pred_stack)
cv_stack = cross_val_score(stacking_model, X_scaled, y, cv=5, scoring="f1_weighted")

results["Stacking Ensemble"] = {
    "model": stacking_model, "accuracy": acc_s, "precision": prec_s,
    "recall": rec_s, "f1_score": f1_s, "confusion_matrix": cm_s,
    "y_pred": y_pred_stack, "cv_mean": cv_stack.mean(), "cv_std": cv_stack.std(),
}
print(f"   Accuracy  : {acc_s:.4f}")
print(f"   F1-Score  : {f1_s:.4f}")
print(f"   CV F1     : {cv_stack.mean():.4f} ± {cv_stack.std():.4f}")

# ═══════════════════════════════════════════════════════════════════
# 4. EVALUATION SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 4: Model Comparison")
print("=" * 65)

comparison_df = pd.DataFrame({
    name: {
        "Accuracy": r["accuracy"], "Precision": r["precision"],
        "Recall": r["recall"], "F1-Score": r["f1_score"],
        "CV F1 (mean)": r["cv_mean"], "CV F1 (std)": r["cv_std"],
    }
    for name, r in results.items()
}).T
print(f"\n{comparison_df.to_string()}")

# Chart
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(
    kind="bar", ax=ax, colormap="viridis")
ax.set_title("Model Comparison — Smart Patient Triage (Advanced)", fontsize=14, fontweight="bold")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.legend(loc="lower right"); plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
plt.close()

# Confusion matrices
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=target_le.classes_, yticklabels=target_le.classes_, ax=ax)
    ax.set_title(name, fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 5. SELECT BEST MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 5: Best Model Selection")
print("=" * 65)

best_name = max(results, key=lambda k: results[k]["f1_score"])
best_result = results[best_name]
best_model = best_result["model"]

print(f"\n🏆 Best Model: {best_name}")
print(f"   F1-Score : {best_result['f1_score']:.4f}")
print(f"   Accuracy : {best_result['accuracy']:.4f}")
print(f"   CV F1    : {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")

print(f"\n📋 Classification Report ({best_name}):")
print(classification_report(y_test, best_result["y_pred"],
                            target_names=target_le.classes_))

# ═══════════════════════════════════════════════════════════════════
# 6. EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 6: Explainability (Feature Importance + SHAP)")
print("=" * 65)

# For SHAP, always use the XGBoost base model (TreeExplainer is fast)
xgb_model = base_models["XGBoost"]

# Feature Importance
if hasattr(xgb_model, "feature_importances_"):
    fi = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ["#e74c3c" if "NEWS2" in f or "Shock" in f or "Oxy" in f or
              "Fever_Hypoxia" in f or "Vulnerability" in f or "Severity" in f or
              "Burden" in f or "Hemodynamic" in f
              else "#2ecc71" for f in fi.index]
    fi.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Feature Importance — XGBoost\n(🔴 = engineered features, 🟢 = raw features)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"📊 Feature importance chart saved")

# SHAP
print("\n🔍 Computing SHAP values...")
try:
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(12, 10))
    if isinstance(shap_values, list):
        sv = shap_values[list(target_le.classes_).index("High")]
    else:
        sv = shap_values
    shap.summary_plot(sv, X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — XGBoost (High Risk Class)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 SHAP summary plot saved")
except Exception as e:
    print(f"   ⚠️ SHAP warning: {e}")

# ═══════════════════════════════════════════════════════════════════
# 7. PATIENT SIMILARITY MODEL (K-Nearest Neighbors)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 7: Building Patient Similarity Index")
print("=" * 65)

knn_model = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1)
knn_model.fit(X_scaled)
print(f"   ✓ KNN similarity index built on {X_scaled.shape[0]} patients")

# ═══════════════════════════════════════════════════════════════════
# 8. SAVE ALL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 8: Saving Model Artifacts")
print("=" * 65)

artifacts = {
    "best_triage_model.pkl": best_model,
    "xgb_model.pkl": xgb_model,
    "scaler.pkl": scaler,
    "label_encoders.pkl": label_encoders,
    "target_label_encoder.pkl": target_le,
    "multi_label_binarizers.pkl": {"symptoms": symptom_mlb, "conditions": cond_mlb},
    "knn_similarity.pkl": knn_model,
    "training_data_scaled.pkl": X_scaled,
    "training_labels.pkl": y,
}

for fname, obj in artifacts.items():
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"   💾 {fname}")

# Save original data for similarity lookups
df_raw = pd.read_csv(DATA_PATH)
df_raw.to_pickle(os.path.join(OUTPUT_DIR, "raw_data.pkl"))
print(f"   💾 raw_data.pkl")

# Metadata
meta = {
    "best_model": best_name,
    "feature_names": feature_names,
    "target_classes": list(target_le.classes_),
    "all_symptoms": ALL_SYMPTOMS,
    "all_conditions": ALL_CONDITIONS,
    "consciousness_levels": CONSCIOUSNESS_ORDER,
    "escalation_threshold": ESCALATION_THRESHOLD,
    "metrics": {
        "accuracy": best_result["accuracy"],
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "f1_score": best_result["f1_score"],
        "cv_f1_mean": best_result["cv_mean"],
        "cv_f1_std": best_result["cv_std"],
    },
    "unique_features": [
        "NEWS2-inspired Clinical Severity Score",
        "Shock Index & Modified Shock Index",
        "Oxygenation Stress Index",
        "Fever-Hypoxia Interaction",
        "Age Vulnerability Index",
        "Symptom Severity Score",
        "Comorbidity Burden Score",
        "Hemodynamic Instability Index",
        "Stacking Ensemble (LR + RF + XGBoost → GBM Meta-Learner)",
        "Confidence-based Triage Escalation",
        "Patient Similarity Scoring (KNN)",
    ],
    "input_schema": {
        "Patient_ID": "string (optional, not used for prediction)",
        "Age": "int (1-95)",
        "Gender": "string (Male/Female)",
        "Symptoms": "comma-separated string, e.g. 'Fever, Cough, Chest Pain'",
        "Blood_Pressure": "string 'systolic/diastolic', e.g. '120/80'",
        "Heart_Rate": "int (40-180 bpm)",
        "Temperature": "float (°F, 95.0-106.0)",
        "SpO2": "int (70-100%)",
        "Respiratory_Rate": "int (8-45 breaths/min)",
        "Consciousness_Level": "string (Alert/Verbal/Pain/Unresponsive)",
        "Pain_Level": "int (0-10)",
        "Pre_Existing_Conditions": "comma-separated string or 'None'",
    },
}
with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"   💾 model_metadata.json")

print("\n" + "=" * 65)
print("  ✅ ADVANCED TRAINING PIPELINE COMPLETE")
print("=" * 65)
print(f"\n  Best model       : {best_name}")
print(f"  F1-Score         : {best_result['f1_score']:.4f}")
print(f"  CV F1            : {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
print(f"  Total features   : {len(feature_names)} (incl. 9 engineered)")
print(f"  Escalation thr.  : {ESCALATION_THRESHOLD*100:.0f}% confidence")
print(f"  Artifacts        : {OUTPUT_DIR}/")
print("=" * 65)
