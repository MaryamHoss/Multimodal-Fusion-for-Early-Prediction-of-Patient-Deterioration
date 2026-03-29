import numpy as np
import pandas as pd
import random
import os
import json

# -----------------------------
# CONFIG & HYPERPARAMETERS
# -----------------------------
NUM_PATIENTS = 1000
TIME_STEPS = 24  # 24-hour observation window
FEATURES = ["HR", "RR", "SpO2", "SBP"]
# Real-world imbalance: ~10% of patients experience clinical deterioration
CLASS_PROB = [0.9, 0.1]

# -----------------------------
# EXPANDED CLINICAL NOTES (NLP VARIETY)
# -----------------------------
# Neutral notes that appear in BOTH stable and deteriorating patients
neutral_notes = [
    "Patient resting in bed.",
    "Diet tolerated well.",
    "Family visited at bedside.",
    "Routine morning care provided.",
    "Patient requested water.",
    "Physiotherapy session completed."
]

stable_notes = [
    "Patient stable overnight.",
    "Vitals within normal limits.",
    "No signs of distress.",
    "Patient comfortable and alert.",
    "Plan to continue current management.",
    "Oxygen therapy ceased as per protocol."
]

deteriorating_notes = [
    "Increasing shortness of breath.",
    "Patient becoming confused and agitated.",
    "Oxygen requirement increasing; titrated up to 4L.",
    "Tachycardic and hypotensive trend noted.",
    "Condition worsening; Medical Emergency Team (MET) informed.",
    "Patient c/o chest pain and diaphoresis.",
    "Decreased urine output noted over last 4 hours."
]

ambiguous_notes = [
    "Patient is feeling slightly unwell.",
    "Reported mild discomfort.",
    "Patient is sleeping heavily.",
    "Nausea reported by patient.",
    "Patient appears slightly pale."
]
# -----------------------------
# DATA GENERATION FUNCTIONS
# -----------------------------

def generate_static_data():
    """Generates 'Structured' data: Age, Gender (0=M, 1=F), and Comorbidity (0=No, 1=Yes)"""
    age = np.random.randint(18, 95)
    gender = np.random.choice([0, 1])
    hx_heart_failure = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% prevalence
    return np.array([age, gender, hx_heart_failure])


def generate_series(label, T):
    t = np.linspace(0, 1, T)
    if label == 0: # Stable
        hr = 80 + np.random.normal(0, 12, T) # More noise
        rr = 18 + np.random.normal(0, 5, T)
        spo2 = 97 + np.random.normal(0, 2, T)
        sbp = 125 + np.random.normal(0, 15, T)
    else: # Deteriorating
        # The trend is now very small (only +8 HR)
        hr = 80 + (8 * t) + np.random.normal(0, 12, T)
        rr = 18 + (4 * t) + np.random.normal(0, 5, T)
        # SpO2 only drops to 94 (very close to stable 97)
        spo2 = 97 - (3 * t) + np.random.normal(0, 3, T)
        sbp = 125 - (10 * t) + np.random.normal(0, 15, T)
    
    return np.stack([hr, rr, np.clip(spo2, 85, 100), np.clip(sbp, 80, 180)], axis=1)



def generate_multimodal_notes(label, T):
    notes = []
    for t in range(T):
        rand = random.random()
        
        # 70% of notes are now 'Neutral' or 'Ambiguous' (Noise)
        if rand < 0.7:
            note = random.choice(neutral_notes + ambiguous_notes)
        elif label == 0: # Stable
            # 10% chance a stable patient gets a 'deteriorating' note (False Alarm)
            if random.random() < 0.1:
                note = random.choice(deteriorating_notes)
            else:
                note = random.choice(stable_notes)
        else: # Deteriorating
            # 20% chance a sick patient gets a 'stable' note (Missed Signal)
            if random.random() < 0.2:
                note = random.choice(stable_notes)
            else:
                # Only show deterioration in the very last 2 hours
                if t < T - 2:
                    note = random.choice(neutral_notes)
                else:
                    note = random.choice(deteriorating_notes)
        notes.append(note)
    return notes

# -----------------------------
# MAIN PIPELINE
# -----------------------------

def create_full_dataset():
    X_static = []
    X_ts = []
    X_text = []
    y = []

    for _ in range(NUM_PATIENTS):
        # 1. Class Imbalance
        label = np.random.choice([0, 1], p=CLASS_PROB)

        # 2. Static (Structured) Data
        static = generate_static_data()

        # 3. Continuous (Time-series) Data + Missingness
        ts = generate_series(label, TIME_STEPS)
        # Introduce 10% missing values
        mask = np.random.rand(*ts.shape) < 0.1
        ts[mask] = np.nan

        # 4. Text (Unstructured) Data
        notes = generate_multimodal_notes(label, TIME_STEPS)

        X_static.append(static)
        X_ts.append(ts)
        X_text.append(notes)
        y.append(label)

    return np.array(X_static), np.array(X_ts), X_text, np.array(y)


# -----------------------------
# PREPROCESSING (Handle Missingness)
# -----------------------------
def preprocess_vitals(X_ts):
    """Demonstrates handling of clinical data missingness (Forward Fill)."""
    processed_ts = []
    for i in range(X_ts.shape[0]):
        df = pd.DataFrame(X_ts[i], columns=FEATURES)
        # Forward fill: use previous known value (Last Observation Carried Forward)
        # Backward fill: for cases where first value is NaN
        df = df.ffill().bfill()
        processed_ts.append(df.values)
    return np.array(processed_ts)

#---------------------------------
# Save the dataset to disk
#---------------------------------
def save_dataset_to_disk(X_static, X_ts, X_text, y, folder="patient_data"):
    """
    Saves the multimodal dataset into a professional research structure.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 1. Save Static Data & Labels (Tabular)
    # This looks like a 'Patient Demographics' table in a hospital database
    static_df = pd.DataFrame(X_static, columns=["Age", "Gender", "Hx_Heart_Failure"])
    static_df['Patient_ID'] = range(len(static_df))
    static_df['Label'] = y
    static_df.to_csv(f"{folder}/metadata.csv", index=False)

    # 2. Save Time-Series (Vitals) as 'Long Format' 
    # This is exactly how MIMIC-IV or eICU data is stored
    vitals_records = []
    for p_idx in range(X_ts.shape[0]):
        for t_idx in range(X_ts.shape[1]):
            vitals_records.append({
                "Patient_ID": p_idx,
                "Hour": t_idx,
                "HR": X_ts[p_idx, t_idx, 0],
                "RR": X_ts[p_idx, t_idx, 1],
                "SpO2": X_ts[p_idx, t_idx, 2],
                "SBP": X_ts[p_idx, t_idx, 3]
            })
    vitals_df = pd.DataFrame(vitals_records)
    vitals_df.to_csv(f"{folder}/vitals_timeseries.csv", index=False)

    # 3. Save Clinical Notes (Unstructured)
    # Saved as a CSV with timestamps
    notes_records = []
    for p_idx in range(len(X_text)):
        for t_idx, note in enumerate(X_text[p_idx]):
            notes_records.append({
                "Patient_ID": p_idx,
                "Hour": t_idx,
                "Note_Content": note
            })
    notes_df = pd.DataFrame(notes_records)
    notes_df.to_csv(f"{folder}/clinical_notes.csv", index=False)

    print(f"Successfully saved clinical dataset to folder: '{folder}/'")

# -----------------------------
# EXECUTION
# -----------------------------
if __name__ == "__main__":
    # 1. Generate
    print("Generating synthetic clinical data...")
    X_static, X_ts, X_text, y = create_full_dataset()
    
    # 2. Save (This is your "ETL" step)
    save_dataset_to_disk(X_static, X_ts, X_text, y)

    # 3. Load Example (To show how you'd use it in your model)
    # In your model script, you would start here:
    loaded_metadata = pd.read_csv("patient_data/metadata.csv")
    print(f"\nExample from disk (Metadata):\n{loaded_metadata.head(3)}")