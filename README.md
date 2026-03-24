# Multimodal Fusion Dataset Generator

This project contains `generate_dataset.py`, a synthetic data generator for early prediction of patient deterioration.

The script creates a multimodal clinical dataset for `1000` patients across a `24`-hour observation window with:

- **Static structured features**: age, gender, history of heart failure
- **Time-series vital signs**: heart rate (`HR`), respiratory rate (`RR`), oxygen saturation (`SpO2`), systolic blood pressure (`SBP`)
- **Text notes**: neutral, stable, and deteriorating clinical note trajectories
- **Target label**: `0` (stable) or `1` (deteriorating), with class imbalance (`~10%` deteriorating)

It also simulates real-world data issues by injecting missing values in vitals and then preprocessing with forward fill + backward fill.

## How `generate_dataset.py` works

1. Randomly assigns each patient a stability label using an imbalanced class distribution.
2. Generates static demographics/comorbidity data.
3. Simulates hourly vital sign trajectories:
   - Stable patients have near-baseline noisy signals.
   - Deteriorating patients trend toward physiologic decline (e.g., rising HR/RR, falling SpO2/SBP).
4. Introduces `10%` missing values in time-series data.
5. Creates temporal clinical note sequences with realistic noise and progression.
6. Preprocesses missing vital signs using LOCF-style filling (`ffill().bfill()`).
7. Prints dataset shapes, class distribution, and a short example note sequence.

## Environment setup

A local virtual environment is expected at `.venv`.

### Activate the environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies (if needed)

```powershell
python -m pip install numpy pandas
```

## Run the generator

```powershell
python generate_dataset.py
```

Expected console output includes:

- Number of generated patients
- Shape of cleaned time-series vitals
- Shape of static features
- Class distribution
- Example of the last 3 clinical notes for one patient
