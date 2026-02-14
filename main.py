from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
from predict import (
    triage_patient, HospitalQueueManager,
    explain_prediction, find_similar_patients
)

# ── App Configuration ────────────────────────────────────────────
app = FastAPI(
    title="Smart Patient Triage API",
    description="AI-powered API for patient risk classification, department routing, and priority queue management.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared hospital queue (persists while server runs)
hospital = HospitalQueueManager()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "output", "model_metadata.json")

# ── Pydantic Models (Validation) ─────────────────────────────────

class PatientInput(BaseModel):
    Patient_ID: str = Field(..., example="PT-1001", description="Unique Patient Identifier")
    Age: int = Field(..., ge=0, le=120, example=65)
    Gender: str = Field(..., pattern="^(Male|Female)$", example="Male")
    Symptoms: str = Field(..., example="Chest Pain, Shortness of Breath")
    Blood_Pressure: str = Field(..., pattern=r"^\d{2,3}/\d{2,3}$", example="180/110")
    Heart_Rate: float = Field(..., ge=0, le=300, example=135)
    Temperature: float = Field(..., ge=90, le=115, example=99.2)
    SpO2: float = Field(..., ge=0, le=100, example=88)
    Respiratory_Rate: float = Field(..., ge=0, le=100, example=32)
    Consciousness_Level: str = Field(..., enum=["Alert", "Verbal", "Pain", "Unresponsive"], example="Pain")
    Pain_Level: int = Field(..., ge=0, le=10, example=9)
    Pre_Existing_Conditions: str = Field(default="None", example="Hypertension, Diabetes")

class TriageResult(BaseModel):
    patient_id: str
    risk_level: str
    department: str
    priority_score: float
    confidence: float
    dept_confidence: float
    needs_escalation: bool
    escalation_reason: Optional[str] = None
    news2_score: float
    risk_probabilities: Dict[str, float]
    dept_probabilities: Dict[str, float]

class QueueItem(BaseModel):
    patient_id: str
    priority_score: float
    risk_level: str
    position: int

class QueueResponse(BaseModel):
    department: str
    patients: List[QueueItem]
    size: int

class ExplanationResponse(BaseModel):
    predicted_class: str
    top_risk_factors: List[List[Any]] # [Feature, Value]
    top_protective_factors: List[List[Any]]
    interpretation: str


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "triage-ml-engine-v2"}

@app.get("/metadata", tags=["System"])
def get_metadata():
    """Get model metadata and available classes."""
    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        return {
            "risk_classes": meta.get("risk_classes", []),
            "department_classes": meta.get("department_classes", []),
            "risk_f1": meta.get("risk_metrics", {}).get("best_f1"),
            "dept_f1": meta.get("dept_metrics", {}).get("f1"),
            "features_count": meta.get("n_features"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/triage", response_model=TriageResult, tags=["Triage"])
def triage(patient: PatientInput):
    """
    Triage a single patient.
    - Validates input strict ranges (e.g. SpO2 0-100).
    - Returns ML prediction with Rule-Based Fallback.
    """
    try:
        data = patient.model_dump()
        result = triage_patient(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/triage/batch", tags=["Triage"])
def triage_batch(patients: List[PatientInput]):
    """Batch triage multiple patients."""
    try:
        results = [triage_patient(p.model_dump()) for p in patients]
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/queue/admit", tags=["Queue"])
def admit_to_queue(patient: PatientInput):
    """Admit patient to the appropriate priority queue."""
    try:
        data = patient.model_dump()
        result = hospital.admit_patient(data)
        dept = result["department"]
        return {
            "triage": result,
            "queue_size": hospital.queues[dept].size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue/{department}", response_model=QueueResponse, tags=["Queue"])
def get_department_queue(department: str):
    """Get the current priority queue for a specific department."""
    if department not in hospital.queues:
        return {"department": department, "patients": [], "size": 0}
    
    q = hospital.queues[department]
    # Reformat queue items to match Pydantic model if needed
    # HospitalQueueManager.get_queue() returns list of dicts: {'patient_id':..., 'priority_score':..., 'risk_level':..., 'position':...}
    # This matches QueueItem model.
    return {
        "department": department,
        "patients": q.get_queue(),
        "size": q.size
    }

@app.post("/queue/{department}/next", tags=["Queue"])
def pop_next_patient(department: str):
    """Get and remove the highest priority patient from the queue."""
    patient = hospital.get_next(department)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"No patients in {department} queue")
    return patient

@app.get("/queue/summary", tags=["Queue"])
def get_all_queues():
    """Get a summary of all department queues."""
    summary = {}
    for dept, q in hospital.queues.items():
        summary[dept] = {
            "size": q.size,
            "patients": q.get_queue()
        }
    return summary

@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain(patient: PatientInput):
    """Get SHAP feature importance explanation."""
    try:
        data = patient.model_dump()
        result = explain_prediction(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar", tags=["Explainability"])
def find_similar(patient: PatientInput, n: int = Query(5, ge=1, le=20)):
    """Find historically similar patients."""
    try:
        data = patient.model_dump()
        result = find_similar_patients(data, n=n)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 65)
    print("  🏥 Triage FastAPI Server Ready")
    print("  👉 Swagger UI: http://localhost:8000/docs")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8000)
