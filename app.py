"""
AI-Powered Smart Patient Triage System — REST API Server

Exposes the ML triage engine as HTTP endpoints for website integration.

Endpoints:
  POST /api/triage          — Triage a single patient
  POST /api/triage/batch    — Triage multiple patients
  POST /api/queue/admit     — Admit patient to priority queue
  GET  /api/queue/<dept>    — Get department queue
  GET  /api/queue/summary   — All queues summary
  POST /api/explain         — SHAP explanation for a patient
  POST /api/similar         — Find similar past patients
  GET  /api/health          — Health check
  GET  /api/metadata        — Model metadata + departments list
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import (
    triage_patient, HospitalQueueManager,
    explain_prediction, find_similar_patients
)
import json, os

app = Flask(__name__)
CORS(app)  # Allow all origins (configure in production)

# Shared hospital queue (persists while server runs)
hospital = HospitalQueueManager()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "output", "model_metadata.json")


# ── Health Check ──────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "triage-ml-engine"})


# ── Model Metadata ───────────────────────────────────────────────

@app.route("/api/metadata", methods=["GET"])
def metadata():
    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        return jsonify({
            "risk_classes": meta.get("risk_classes", []),
            "department_classes": meta.get("department_classes", []),
            "risk_f1": meta.get("risk_metrics", {}).get("best_f1"),
            "dept_f1": meta.get("dept_metrics", {}).get("f1"),
            "features_count": meta.get("n_features"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Single Patient Triage ────────────────────────────────────────

@app.route("/api/triage", methods=["POST"])
def triage():
    """
    POST JSON body with patient data:
    {
        "Patient_ID": "PT-001",
        "Age": 72,
        "Gender": "Male",
        "Symptoms": "Chest Pain, Shortness of Breath",
        "Blood_Pressure": "185/110",
        "Heart_Rate": 130,
        "Temperature": 103.8,
        "SpO2": 88,
        "Respiratory_Rate": 32,
        "Consciousness_Level": "Verbal",
        "Pain_Level": 9,
        "Pre_Existing_Conditions": "Heart Disease"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    try:
        result = triage_patient(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Batch Triage ─────────────────────────────────────────────────

@app.route("/api/triage/batch", methods=["POST"])
def triage_batch():
    """POST a JSON array of patient objects."""
    patients = request.get_json()
    if not patients or not isinstance(patients, list):
        return jsonify({"error": "Expected a JSON array of patients"}), 400
    try:
        results = [triage_patient(p) for p in patients]
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Queue: Admit ─────────────────────────────────────────────────

@app.route("/api/queue/admit", methods=["POST"])
def queue_admit():
    """Triage + admit patient to department queue."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    try:
        result = hospital.admit_patient(data)
        return jsonify({
            "triage": result,
            "queue_size": hospital.queues[result["department"]].size()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Queue: View Department ───────────────────────────────────────

@app.route("/api/queue/<department>", methods=["GET"])
def queue_department(department):
    if department not in hospital.queues:
        return jsonify({"department": department, "patients": [], "size": 0})
    q = hospital.queues[department]
    return jsonify({
        "department": department,
        "patients": q.get_queue(),
        "size": q.size()
    })


# ── Queue: Next Patient ─────────────────────────────────────────

@app.route("/api/queue/<department>/next", methods=["POST"])
def queue_next(department):
    """Pop the highest-priority patient from a department queue."""
    patient = hospital.get_next(department)
    if patient is None:
        return jsonify({"error": f"No patients in {department} queue"}), 404
    return jsonify(patient)


# ── Queue: Summary ───────────────────────────────────────────────

@app.route("/api/queue/summary", methods=["GET"])
def queue_summary():
    summary = {}
    for dept, q in hospital.queues.items():
        summary[dept] = {
            "size": q.size(),
            "patients": q.get_queue()
        }
    return jsonify(summary)


# ── SHAP Explanation ─────────────────────────────────────────────

@app.route("/api/explain", methods=["POST"])
def explain():
    """Get SHAP feature importance for a patient."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    try:
        result = explain_prediction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Similar Patients ─────────────────────────────────────────────

@app.route("/api/similar", methods=["POST"])
def similar():
    """Find similar past patients using KNN."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    n = request.args.get("n", 5, type=int)
    try:
        result = find_similar_patients(data, n=n)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run Server ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  🏥 Triage ML API Server")
    print("  Endpoints:")
    print("    POST /api/triage         — Triage patient")
    print("    POST /api/triage/batch   — Batch triage")
    print("    POST /api/queue/admit    — Admit to queue")
    print("    GET  /api/queue/<dept>   — View queue")
    print("    POST /api/queue/<d>/next — Next patient")
    print("    GET  /api/queue/summary  — All queues")
    print("    POST /api/explain        — SHAP explain")
    print("    POST /api/similar        — Similar patients")
    print("    GET  /api/health         — Health check")
    print("    GET  /api/metadata       — Model info")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5001, debug=True)
