
from predict import triage_patient
import json

def test_case(name, data, expected_risk=None, expected_dept=None):
    print(f"\n🧪 Testing: {name}")
    try:
        res = triage_patient(data)
        print(f"   Output: Risk={res['risk_level']}, Dept={res['department']}")
        
        pass_risk = expected_risk is None or res['risk_level'] == expected_risk
        pass_dept = expected_dept is None or res['department'] == expected_dept
        
        if pass_risk and pass_dept:
            print("   ✅ PASS")
        else:
            print(f"   ❌ FAIL (Expected Risk={expected_risk}, Dept={expected_dept})")
    except Exception as e:
        print(f"   ❌ CRASH: {e}")

# 1. Stroke Guardrail
test_case("Stroke (Guardrail Trigger)", {
    "Symptoms": "Slurred Speech, Weakness",
    "Pre_Existing_Conditions": "Hypertension",
    "Age": 70
}, expected_risk="High", expected_dept="Neurology")

# 2. DKA Guardrail
test_case("DKA (Guardrail Trigger)", {
    "Symptoms": "Confusion, Vomiting",
    "Pre_Existing_Conditions": "Type 1 Diabetes",
    "Age": 20
}, expected_risk="High", expected_dept="Emergency")

# 3. Anxiety Guardrail
test_case("Anxiety (Guardrail Trigger)", {
    "Symptoms": "Chest Tightness",
    "Pre_Existing_Conditions": "Anxiety",
    "Blood_Pressure": "130/80", # Stable
    "Heart_Rate": 100, # Stable (<110)
    "SpO2": 98,
    "Age": 30
}, expected_risk="Low", expected_dept="General Medicine")

# 4. Kidney Stone Guardrail
test_case("Kidney Stone (Guardrail Trigger)", {
    "Symptoms": "Severe Flank Pain",
    "Pre_Existing_Conditions": "Kidney Stones",
    "Age": 45
}, expected_dept="Nephrology")

print("\n------------------------------------------------")
print("Verification Complete")
