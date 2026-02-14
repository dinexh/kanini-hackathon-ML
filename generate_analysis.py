
import json
import pandas as pd
import numpy as np
import joblib
from predict import triage_patient, explain_prediction

def run_analysis():
    with open('test_requests.json', 'r') as f:
        test_requests = json.load(f)

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
