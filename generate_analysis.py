
import json
import pandas as pd
import numpy as np
import joblib
from predict import triage_patient, explain_prediction

def run_analysis():
    # Clinical Test Cases
    test_requests = [
        {"description": "High Risk - Possible Heart Attack", "data": {"Patient_ID": "PT-1001", "Age": 65, "Gender": "Male", "Symptoms": "Chest Pain, Shortness of Breath, Sweating", "Blood_Pressure": "180/110", "Heart_Rate": 135, "Temperature": 99.2, "SpO2": 88, "Respiratory_Rate": 32, "Consciousness_Level": "Pain", "Pain_Level": 9, "Pre_Existing_Conditions": "Hypertension, Diabetes"}},
        {"description": "Low Risk - URI / Mild Flu", "data": {"Patient_ID": "PT-1002", "Age": 24, "Gender": "Female", "Symptoms": "Cough, Sore Throat, Mild Fever", "Blood_Pressure": "118/75", "Heart_Rate": 78, "Temperature": 100.4, "SpO2": 98, "Respiratory_Rate": 16, "Consciousness_Level": "Alert", "Pain_Level": 2, "Pre_Existing_Conditions": "None"}},
        {"description": "Medium Risk - Gastroenteritis", "data": {"Patient_ID": "PT-1003", "Age": 45, "Gender": "Male", "Symptoms": "Abdominal Pain, Vomiting, Nausea", "Blood_Pressure": "130/85", "Heart_Rate": 92, "Temperature": 99.5, "SpO2": 96, "Respiratory_Rate": 20, "Consciousness_Level": "Alert", "Pain_Level": 6, "Pre_Existing_Conditions": "None"}},
        {"description": "High Risk - Septic Shock", "data": {"Patient_ID": "PT-1004", "Age": 72, "Gender": "Female", "Symptoms": "Fever, Confusion, Shivering", "Blood_Pressure": "85/50", "Heart_Rate": 145, "Temperature": 103.5, "SpO2": 89, "Respiratory_Rate": 28, "Consciousness_Level": "Verbal", "Pain_Level": 4, "Pre_Existing_Conditions": "UTI"}},
        {"description": "Low Risk - Migraine Disorder", "data": {"Patient_ID": "PT-1005", "Age": 30, "Gender": "Female", "Symptoms": "Severe Headache, Light Sensitivity, Nausea", "Blood_Pressure": "120/80", "Heart_Rate": 75, "Temperature": 98.6, "SpO2": 99, "Respiratory_Rate": 16, "Consciousness_Level": "Alert", "Pain_Level": 8, "Pre_Existing_Conditions": "Migraine Disorder"}},
        {"description": "High Risk - Stroke", "data": {"Patient_ID": "PT-1006", "Age": 68, "Gender": "Male", "Symptoms": "Slurred Speech, Facial Droop, Weakness", "Blood_Pressure": "190/100", "Heart_Rate": 90, "Temperature": 98.7, "SpO2": 96, "Respiratory_Rate": 18, "Consciousness_Level": "Alert", "Pain_Level": 0, "Pre_Existing_Conditions": "Previous Stroke"}},
        {"description": "Medium Risk - Anxiety Disorder", "data": {"Patient_ID": "PT-1007", "Age": 28, "Gender": "Male", "Symptoms": "Chest Tightness, Palpitations, Dizziness", "Blood_Pressure": "135/85", "Heart_Rate": 110, "Temperature": 98.4, "SpO2": 99, "Respiratory_Rate": 22, "Consciousness_Level": "Alert", "Pain_Level": 2, "Pre_Existing_Conditions": "Anxiety Disorder"}},
        {"description": "High Risk - DKA", "data": {"Patient_ID": "PT-1008", "Age": 22, "Gender": "Female", "Symptoms": "Confusion, Vomiting, Fruity Breath", "Blood_Pressure": "100/60", "Heart_Rate": 125, "Temperature": 99.0, "SpO2": 95, "Respiratory_Rate": 36, "Consciousness_Level": "Verbal", "Pain_Level": 5, "Pre_Existing_Conditions": "Type 1 Diabetes"}},
        {"description": "Low Risk - Kidney Stones", "data": {"Patient_ID": "PT-1009", "Age": 40, "Gender": "Male", "Symptoms": "Severe Flank Pain, Blood in Urine", "Blood_Pressure": "140/90", "Heart_Rate": 100, "Temperature": 98.9, "SpO2": 98, "Respiratory_Rate": 20, "Consciousness_Level": "Alert", "Pain_Level": 9, "Pre_Existing_Conditions": "History of Stones"}},
        {"description": "Medium Risk - Pneumonia", "data": {"Patient_ID": "PT-1010", "Age": 55, "Gender": "Female", "Symptoms": "Productive Cough, Fever, Shortness of Breath", "Blood_Pressure": "125/80", "Heart_Rate": 98, "Temperature": 101.5, "SpO2": 93, "Respiratory_Rate": 24, "Consciousness_Level": "Alert", "Pain_Level": 3, "Pre_Existing_Conditions": "Smoker"}},
        {"description": "High Risk - Severe Trauma", "data": {"Patient_ID": "PT-1011", "Age": 33, "Gender": "Male", "Symptoms": "Severe Pain, Bleeding, Deformity", "Blood_Pressure": "110/70", "Heart_Rate": 120, "Temperature": 98.2, "SpO2": 97, "Respiratory_Rate": 26, "Consciousness_Level": "Pain", "Pain_Level": 10, "Pre_Existing_Conditions": "None"}},
        {"description": "Medium Risk - Asthma Exacerbation", "data": {"Patient_ID": "PT-1012", "Age": 17, "Gender": "Male", "Symptoms": "Wheezing, Tightness", "Blood_Pressure": "128/76", "Heart_Rate": 115, "Temperature": 98.6, "SpO2": 94, "Respiratory_Rate": 28, "Consciousness_Level": "Alert", "Pain_Level": 1, "Pre_Existing_Conditions": "Asthma"}}
    ]

    results = []
    
    for case in test_requests:
        description = case.get('description', 'Test Case')
        data = case.get('data', {})
        
        print(f"  Processing: {description}...")
        
        # 1. Prediction
        try:
            pred = triage_patient(data)
        except Exception as e:
            print(f"Error predicting {description}: {e}")
            continue
            
        # 2. Explanation
        try:
            explanation = explain_prediction(data)
        except Exception as e:
            print(f"Error explaining {description}: {e}")
            explanation = {}

        # 3. Clinical Reasoning
        reasoning = []
        if data.get('SpO2', 100) < 90:
            reasoning.append(f"Hypoxia (SpO2 {data['SpO2']}%) is a primary driver for High Risk.")
        news2 = pred.get('news2_score', 0)  # Use NEWS2 from prediction
        if news2 >= 7:
            reasoning.append(f"NEWS2 Score ({news2}) implies critical condition needing urgent review.")
        elif news2 >= 5:
            reasoning.append(f"NEWS2 Score ({news2}) suggests potential deterioration.")

        # 4. Format Output
        result = {
            "case_description": description,
            "input": data,
            "prediction_summary": {
                "risk_level": pred['risk_level'],
                "department": pred['department'],
                "priority_score": pred['priority_score'],
                "model_confidence": f"{pred['confidence']:.1%}",
                "escalation_alert": pred['needs_escalation']
            },
            "ml_analysis": {
                "top_risk_drivers": explanation.get('top_risk_factors', []),
                "protective_factors": explanation.get('top_protective_factors', []),
                "clinical_severity_score_news2": news2
            },
            "clinical_reasoning": reasoning
        }
        results.append(result)

    with open('test_case_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Analysis saved to test_case_analysis.json")

if __name__ == "__main__":
    print("Analyzing test cases...")
    run_analysis()
