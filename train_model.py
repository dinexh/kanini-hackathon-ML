"""
AI-Powered Smart Patient Triage System — Training Pipeline (v4)

Dual-model: Risk Classifier + Department Classifier + Priority Queue
"""

import os, json, warnings, numpy as np, pandas as pd, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb, shap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "patient_triage_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_SYMPTOMS = sorted(["Abdominal Pain","Body Ache","Chest Pain","Confusion","Cough","Diarrhea","Dizziness","Fatigue","Fever","Headache","Loss of Appetite","Nausea","Seizures","Shortness of Breath","Sore Throat","Vomiting"])
ALL_CONDITIONS = sorted(["Asthma","COPD","Cancer","Diabetes","Heart Disease","Hypertension","Kidney Disease","Liver Disease","Obesity","Stroke History"])
CONSCIOUSNESS_ORDER = ["Unresponsive", "Pain", "Verbal", "Alert"]
CONS_MAP = {level: i for i, level in enumerate(CONSCIOUSNESS_ORDER)}
ESCALATION_THRESHOLD = 0.60

def compute_news2_score(df):
    score = pd.Series(0, index=df.index, dtype=float)
    rr = df["Respiratory_Rate"]
    score += np.where(rr<=8,3,np.where(rr<=11,1,np.where(rr<=20,0,np.where(rr<=24,2,3))))
    spo2 = df["SpO2"]
    score += np.where(spo2<=91,3,np.where(spo2<=93,2,np.where(spo2<=95,1,0)))
    hr = df["Heart_Rate"]
    score += np.where(hr<=40,3,np.where(hr<=50,1,np.where(hr<=90,0,np.where(hr<=110,1,np.where(hr<=130,2,3)))))
    temp_c = (df["Temperature"]-32)*5/9
    score += np.where(temp_c<=35,3,np.where(temp_c<=36,1,np.where(temp_c<=38,0,np.where(temp_c<=39,1,2))))
    sbp = df["BP_Systolic"]
    score += np.where(sbp<=90,3,np.where(sbp<=100,2,np.where(sbp<=110,1,np.where(sbp<=219,0,3))))
    score += np.where(df["Consciousness_Level"]<3,3,0)
    return score

def compute_feature_interactions(df):
    f = pd.DataFrame(index=df.index)
    f["Shock_Index"] = df["Heart_Rate"]/df["BP_Systolic"].clip(lower=1)
    map_v = df["BP_Diastolic"]+(df["BP_Systolic"]-df["BP_Diastolic"])/3
    f["Modified_Shock_Index"] = df["Heart_Rate"]/map_v.clip(lower=1)
    f["Oxy_Stress"] = df["Respiratory_Rate"]/df["SpO2"].clip(lower=1)
    f["Fever_Hypoxia"] = (df["Temperature"]>100.4).astype(int)*(100-df["SpO2"])
    age_r = np.where(df["Age"]>65,2,np.where(df["Age"]<5,2,np.where(df["Age"]>50,1,0)))
    f["Age_Vulnerability"] = age_r*(df["Pain_Level"]/10+0.5)
    f["Symptom_Severity"] = df["Symptom_Count"]*(4-df["Consciousness_Level"])/4
    f["Comorbidity_Burden"] = df["Condition_Count"]*np.where(df["Age"]>60,1.5,1.0)
    f["Hemodynamic_Instability"] = np.abs(df["Heart_Rate"]-75)/75+np.abs(df["BP_Systolic"]-120)/120
    return f

def parse_multi_value(series, all_values, prefix):
    def split_fn(val):
        if pd.isna(val) or val.strip().lower()=="none": return []
        return [v.strip() for v in val.split(",")]
    mlb = MultiLabelBinarizer(classes=all_values)
    enc = mlb.fit_transform(series.apply(split_fn))
    cols = [f"{prefix}_{v.replace(' ','_')}" for v in all_values]
    return pd.DataFrame(enc, columns=cols, index=series.index), mlb

# ── LOAD & PREPROCESS ─────────────────────────────────────────────
print("="*65+"\n  STEP 1: Loading & Preprocessing\n"+"="*65)
df = pd.read_csv(DATA_PATH)
print(f"\n📂 {df.shape[0]} rows × {df.shape[1]} cols")
print(f"\n📊 Risk:\n{df['Risk_Level'].value_counts().to_string()}")
print(f"\n🏥 Depts:\n{df['Department'].value_counts().to_string()}")

risk_labels = df["Risk_Level"].copy()
dept_labels = df["Department"].copy()
if "Patient_ID" in df.columns: df = df.drop("Patient_ID", axis=1)
bp = df["Blood_Pressure"].str.split("/", expand=True)
df["BP_Systolic"] = pd.to_numeric(bp[0], errors="coerce")
df["BP_Diastolic"] = pd.to_numeric(bp[1], errors="coerce")
df = df.drop("Blood_Pressure", axis=1)
sym_df, sym_mlb = parse_multi_value(df["Symptoms"], ALL_SYMPTOMS, "Sym")
df["Symptom_Count"] = sym_df.sum(axis=1)
df = pd.concat([df.drop("Symptoms", axis=1), sym_df], axis=1)
cond_df, cond_mlb = parse_multi_value(df["Pre_Existing_Conditions"], ALL_CONDITIONS, "Cond")
df["Condition_Count"] = cond_df.sum(axis=1)
df = pd.concat([df.drop("Pre_Existing_Conditions", axis=1), cond_df], axis=1)
df["Consciousness_Level"] = df["Consciousness_Level"].map(CONS_MAP).fillna(3)
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"].fillna("Unknown"))
label_encoders = {"Gender": le_gender}
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Risk_Level","Department"]]
for c in num_cols: df[c] = df[c].fillna(df[c].median())
df["NEWS2_Score"] = compute_news2_score(df)
ints = compute_feature_interactions(df)
for c in ints.columns: df[c] = ints[c]

risk_le = LabelEncoder(); df["Risk_Level"] = risk_le.fit_transform(risk_labels)
dept_le = LabelEncoder(); df["Department"] = dept_le.fit_transform(dept_labels)
X = df.drop(["Risk_Level","Department"], axis=1)
y_risk = df["Risk_Level"]; y_dept = df["Department"]
feature_names = X.columns.tolist()
print(f"\n📐 {X.shape[1]} features")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# ── SPLIT ─────────────────────────────────────────────────────────
X_tr,X_te,yr_tr,yr_te,yd_tr,yd_te = train_test_split(X_scaled,y_risk,y_dept,test_size=0.2,random_state=42,stratify=y_risk)
print(f"\n   Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

# ── TRAIN RISK MODELS ────────────────────────────────────────────
print("\n"+"="*65+"\n  STEP 2: Training Risk Models\n"+"="*65)
r_models = {
    "LR": LogisticRegression(max_iter=1000,random_state=42),
    "DT": DecisionTreeClassifier(max_depth=12,random_state=42),
    "RF": RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),
    "XGB": xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=42,use_label_encoder=False,eval_metric="mlogloss",verbosity=0),
}
r_res = {}
for nm,m in r_models.items():
    m.fit(X_tr,yr_tr); yp=m.predict(X_te)
    f1=f1_score(yr_te,yp,average="weighted",zero_division=0)
    cv=cross_val_score(m,X_scaled,y_risk,cv=5,scoring="f1_weighted")
    r_res[nm]={"model":m,"f1":f1,"acc":accuracy_score(yr_te,yp),"prec":precision_score(yr_te,yp,average="weighted",zero_division=0),"rec":recall_score(yr_te,yp,average="weighted",zero_division=0),"cm":confusion_matrix(yr_te,yp),"yp":yp,"cv_m":cv.mean(),"cv_s":cv.std()}
    print(f"  {nm}: F1={f1:.4f} CV={cv.mean():.4f}±{cv.std():.4f}")

print(f"\n🔹 Stacking Ensemble (Risk)...")
r_stack = StackingClassifier(estimators=[("lr",LogisticRegression(max_iter=1000,random_state=42)),("rf",RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)),("xgb",xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=42,use_label_encoder=False,eval_metric="mlogloss",verbosity=0))],final_estimator=GradientBoostingClassifier(n_estimators=100,max_depth=4,random_state=42),cv=5,stack_method="predict_proba",n_jobs=-1)
r_stack.fit(X_tr,yr_tr); yps=r_stack.predict(X_te)
f1s=f1_score(yr_te,yps,average="weighted",zero_division=0)
cvs=cross_val_score(r_stack,X_scaled,y_risk,cv=5,scoring="f1_weighted")
r_res["Stack"]={"model":r_stack,"f1":f1s,"acc":accuracy_score(yr_te,yps),"prec":precision_score(yr_te,yps,average="weighted",zero_division=0),"rec":recall_score(yr_te,yps,average="weighted",zero_division=0),"cm":confusion_matrix(yr_te,yps),"yp":yps,"cv_m":cvs.mean(),"cv_s":cvs.std()}
print(f"  Stack: F1={f1s:.4f} CV={cvs.mean():.4f}±{cvs.std():.4f}")

best_rn = max(r_res,key=lambda k:r_res[k]["f1"])
best_rm = r_res[best_rn]["model"]
print(f"\n🏆 Best Risk: {best_rn} (F1={r_res[best_rn]['f1']:.4f})")
print(classification_report(yr_te,r_res[best_rn]["yp"],target_names=risk_le.classes_))

# ── TRAIN DEPARTMENT MODEL ────────────────────────────────────────
print("="*65+"\n  STEP 3: Training Department Model\n"+"="*65)
dept_model = xgb.XGBClassifier(n_estimators=300,max_depth=8,learning_rate=0.1,random_state=42,use_label_encoder=False,eval_metric="mlogloss",verbosity=0)
dept_model.fit(X_tr,yd_tr); ydp=dept_model.predict(X_te)
df1=f1_score(yd_te,ydp,average="weighted",zero_division=0)
dacc=accuracy_score(yd_te,ydp)
dcv=cross_val_score(dept_model,X_scaled,y_dept,cv=5,scoring="f1_weighted")
print(f"  Dept XGB: F1={df1:.4f} Acc={dacc:.4f} CV={dcv.mean():.4f}±{dcv.std():.4f}")
print(classification_report(yd_te,ydp,target_names=dept_le.classes_))

# ── CHARTS ────────────────────────────────────────────────────────
print("="*65+"\n  STEP 4: Charts\n"+"="*65)
comp=pd.DataFrame({n:{"Accuracy":r["acc"],"Precision":r["prec"],"Recall":r["rec"],"F1":r["f1"]} for n,r in r_res.items()}).T
fig,ax=plt.subplots(figsize=(12,6)); comp.plot(kind="bar",ax=ax,colormap="viridis")
ax.set_title("Risk Model Comparison",fontsize=14,fontweight="bold"); ax.set_ylim(0,1.05); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"model_comparison.png"),dpi=150); plt.close()

n=len(r_res); fig,axes=plt.subplots(1,n,figsize=(5*n,4))
for ax,(nm,r) in zip(axes,r_res.items()):
    sns.heatmap(r["cm"],annot=True,fmt="d",cmap="Blues",xticklabels=risk_le.classes_,yticklabels=risk_le.classes_,ax=ax)
    ax.set_title(nm,fontsize=9)
plt.suptitle("Risk Confusion Matrices",fontsize=14,fontweight="bold"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrices.png"),dpi=150); plt.close()

fig,ax=plt.subplots(figsize=(12,10))
sns.heatmap(confusion_matrix(yd_te,ydp),annot=True,fmt="d",cmap="Greens",xticklabels=dept_le.classes_,yticklabels=dept_le.classes_,ax=ax)
ax.set_title("Department Confusion Matrix",fontsize=14,fontweight="bold"); plt.xticks(rotation=45,ha="right"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"department_confusion_matrix.png"),dpi=150); plt.close()

xgb_risk = r_models["XGB"]
fi=pd.Series(xgb_risk.feature_importances_,index=feature_names).sort_values(ascending=True)
fig,ax=plt.subplots(figsize=(10,12))
cols=["#e74c3c" if any(k in f for k in ["NEWS2","Shock","Oxy","Fever_Hypoxia","Vulnerability","Severity","Burden","Hemodynamic"]) else "#2ecc71" for f in fi.index]
fi.plot(kind="barh",ax=ax,color=cols); ax.set_title("Feature Importance",fontsize=12,fontweight="bold"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"feature_importance.png"),dpi=150); plt.close()

try:
    exp=shap.TreeExplainer(xgb_risk); sv=exp.shap_values(X_te)
    s=sv[list(risk_le.classes_).index("High")] if isinstance(sv,list) else sv
    shap.summary_plot(s,X_te,feature_names=feature_names,show=False)
    plt.title("SHAP — High Risk",fontsize=13,fontweight="bold"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"shap_summary.png"),dpi=150,bbox_inches="tight"); plt.close()
except Exception as e: print(f"⚠️ SHAP: {e}")
print("📊 All charts saved")

# ── KNN ───────────────────────────────────────────────────────────
knn = NearestNeighbors(n_neighbors=5,metric="euclidean",n_jobs=-1); knn.fit(X_scaled)

# ── SAVE ──────────────────────────────────────────────────────────
print("\n"+"="*65+"\n  STEP 5: Saving Artifacts\n"+"="*65)
arts = {"best_risk_model.pkl":best_rm,"dept_model.pkl":dept_model,"xgb_risk_model.pkl":xgb_risk,"scaler.pkl":scaler,"label_encoders.pkl":label_encoders,"risk_label_encoder.pkl":risk_le,"dept_label_encoder.pkl":dept_le,"multi_label_binarizers.pkl":{"symptoms":sym_mlb,"conditions":cond_mlb},"knn_similarity.pkl":knn,"training_data_scaled.pkl":X_scaled,"training_risk_labels.pkl":y_risk,"training_dept_labels.pkl":y_dept}
for fn,obj in arts.items():
    with open(os.path.join(OUTPUT_DIR,fn),"wb") as f: pickle.dump(obj,f)
    print(f"   💾 {fn}")
pd.read_csv(DATA_PATH).to_pickle(os.path.join(OUTPUT_DIR,"raw_data.pkl")); print("   💾 raw_data.pkl")

meta={"best_risk_model":best_rn,"feature_names":feature_names,"risk_classes":list(risk_le.classes_),"department_classes":list(dept_le.classes_),"all_symptoms":ALL_SYMPTOMS,"all_conditions":ALL_CONDITIONS,"consciousness_levels":CONSCIOUSNESS_ORDER,"escalation_threshold":ESCALATION_THRESHOLD,"risk_metrics":{"accuracy":r_res[best_rn]["acc"],"f1_score":r_res[best_rn]["f1"],"cv_f1_mean":r_res[best_rn]["cv_m"]},"dept_metrics":{"accuracy":dacc,"f1_score":df1,"cv_f1_mean":float(dcv.mean())},"priority_weights":{"risk_weight":0.40,"news2_weight":0.25,"vitals_weight":0.20,"age_weight":0.15}}
with open(os.path.join(OUTPUT_DIR,"model_metadata.json"),"w") as f: json.dump(meta,f,indent=2)
print("   💾 model_metadata.json")

print(f"\n{'='*65}\n  ✅ TRAINING COMPLETE\n{'='*65}")
print(f"  Risk : {best_rn} F1={r_res[best_rn]['f1']:.4f}")
print(f"  Dept : XGBoost F1={df1:.4f}")
print(f"  Feats: {len(feature_names)} | Depts: {list(dept_le.classes_)}")
print("="*65)