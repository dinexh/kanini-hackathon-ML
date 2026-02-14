import pandas as pd

df = pd.read_csv("data/patient_triage_dataset.csv")

print("Checking for anomalies...")
print("BP > 220:", len(df[df["Blood_Pressure"].apply(lambda x: int(x.split("/")[0])) > 220]))
print("BP < 60:", len(df[df["Blood_Pressure"].apply(lambda x: int(x.split("/")[0])) < 60]))
print("HR > 200:", len(df[df["Heart_Rate"] > 200]))
print("HR < 30:", len(df[df["Heart_Rate"] < 30]))
print("SpO2 < 50:", len(df[df["SpO2"] < 50]))
print("Temp > 108:", len(df[df["Temperature"] > 108]))
print("Temp < 94:", len(df[df["Temperature"] < 94]))

print("\nSample of potential issues:")
print(df[(df["Heart_Rate"] > 180) & (df["Risk_Level"] == "Low")].head())
