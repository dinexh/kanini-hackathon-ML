"""
AI-Powered Smart Patient Triage System — REST API Server with Swagger UI

Exposes the ML triage engine as HTTP endpoints for website integration.
Serves OpenAPI/Swagger documentation at /apidocs/

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
from flasgger import Swagger
from predict import (
    triage_patient, HospitalQueueManager,
    explain_prediction, find_similar_patients
)
import json, os

app = Flask(__name__)
CORS(app)  # Allow all origins (configure in production)

# Swagger configuration
app.config['SWAGGER'] = {
    'title': 'Smart Patient Triage API',
    'uiversion': 3,
    'description': 'AI-powered API for patient risk classification, department routing, and priority queue management.',
    'version': '1.0.0',
    'specs_route': '/apidocs/'
}
swagger = Swagger(app)

# Shared hospital queue (persists while server runs)
hospital = HospitalQueueManager()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "output", "model_metadata.json")


# ── Health Check ──────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint
    ---
    responses:
      200:
        description: Service is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            service:
              type: string
              example: triage-ml-engine
    """
    return jsonify({"status": "ok", "service": "triage-ml-engine"})


# ── Model Metadata ───────────────────────────────────────────────

@app.route("/api/metadata", methods=["GET"])
def metadata():
    """
    Get model metadata and available classes
    ---
    responses:
      200:
        description: Model metadata
        schema:
          type: object
          properties:
            risk_classes:
              type: array
              items:
                type: string
            department_classes:
              type: array
              items:
                type: string
            risk_f1:
              type: number
            dept_f1:
              type: number
            features_count:
              type: integer
    """
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
    Triage a single patient
    ---
    parameters:
      - in: body
        name: patient
        required: true
        schema:
          type: object
          properties:
            Patient_ID:
              type: string
              example: PT-001
            Age:
              type: integer
              example: 72
            Gender:
              type: string
              enum: [Male, Female]
              example: Male
            Symptoms:
              type: string
              example: Chest Pain, Shortness of Breath
            Blood_Pressure:
              type: string
              example: 185/110
            Heart_Rate:
              type: number
              example: 130
            Temperature:
              type: number
              example: 103.8
            SpO2:
              type: number
              example: 88
            Respiratory_Rate:
              type: number
              example: 32
            Consciousness_Level:
              type: string
              enum: [Alert, Verbal, Pain, Unresponsive]
              example: Verbal
            Pain_Level:
              type: number
              example: 9
            Pre_Existing_Conditions:
              type: string
              example: Heart Disease
    responses:
      200:
        description: Triage result
        schema:
          type: object
          properties:
            risk_level:
              type: string
              example: High
            department:
              type: string
              example: Cardiology
            priority_score:
              type: number
              example: 87.1
            confidence:
              type: number
              example: 0.91
            news2_score:
              type: number
              example: 13.0
            needs_escalation:
              type: boolean
              example: false
            risk_probabilities:
              type: object
            dept_probabilities:
              type: object
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
    """
    Triage multiple patients
    ---
    parameters:
      - in: body
        name: patients
        required: true
        schema:
          type: array
          items:
            type: object
            properties:
              Patient_ID:
                type: string
              Age:
                type: integer
    responses:
      200:
        description: List of triage results
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
            count:
              type: integer
    """
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
    """
    Admit patient to department queue
    ---
    parameters:
      - in: body
        name: patient
        required: true
        schema:
          type: object
          properties:
            Patient_ID:
              type: string
    responses:
      200:
        description: Admission result
        schema:
          type: object
          properties:
            triage:
              type: object
            queue_size:
              type: integer
    """
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
    """
    Get queue for a department
    ---
    parameters:
      - in: path
        name: department
        required: true
        type: string
    responses:
      200:
        description: Department queue
        schema:
          type: object
          properties:
            department:
              type: string
            patients:
              type: array
              items:
                type: object
            size:
              type: integer
    """
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
    """
    Pop next patient from department queue
    ---
    parameters:
      - in: path
        name: department
        required: true
        type: string
    responses:
      200:
        description: Next patient details
      404:
        description: Queue empty
    """
    patient = hospital.get_next(department)
    if patient is None:
        return jsonify({"error": f"No patients in {department} queue"}), 404
    return jsonify(patient)


# ── Queue: Summary ───────────────────────────────────────────────

@app.route("/api/queue/summary", methods=["GET"])
def queue_summary():
    """
    Get summary of all queues
    ---
    responses:
      200:
        description: Summary of all queues
        schema:
          type: object
    """
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
    """
    Get SHAP explanation for a prediction
    ---
    parameters:
      - in: body
        name: patient
        required: true
        schema:
          type: object
          properties:
            Patient_ID:
              type: string
              example: PT-1001
            Age:
              type: integer
              example: 65
            Gender:
              type: string
              enum: [Male, Female]
              example: Male
            Symptoms:
              type: string
              example: Chest Pain, Shortness of Breath
            Blood_Pressure:
              type: string
              example: 180/110
            Heart_Rate:
              type: number
              example: 135
            Temperature:
              type: number
              example: 99.2
            SpO2:
              type: number
              example: 88
            Respiratory_Rate:
              type: number
              example: 32
            Consciousness_Level:
              type: string
              enum: [Alert, Verbal, Pain, Unresponsive]
              example: Pain
            Pain_Level:
              type: number
              example: 9
            Pre_Existing_Conditions:
              type: string
              example: Hypertension, Diabetes
    responses:
      200:
        description: Feature importance
        schema:
          type: object
          properties:
            predicted_class:
              type: string
            top_risk_factors:
              type: array
            top_protective_factors:
              type: array
            interpretation:
              type: string
    """
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
    """
    Find similar past patients
    ---
    parameters:
      - in: body
        name: patient
        required: true
        schema:
          type: object
      - in: query
        name: n
        type: integer
        default: 5
    responses:
      200:
        description: Similar patients
        schema:
          type: object
          properties:
            query_patient:
              type: object
            similar_patients:
              type: array
              items:
                type: object
    """
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
    print("=" * 65)
    print("  🏥 Triage ML API Server with Swagger UI")
    print("  👉 Swagger UI: http://localhost:5001/apidocs/")
    print("=" * 65)
    app.run(host="0.0.0.0", port=5001, debug=True)
